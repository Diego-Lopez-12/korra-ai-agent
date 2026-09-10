"""
File: graph.py
Author: Diego Lopez
Email: dlopez5788@sdsu.edu
Course: COMPE 475 – Microprocessors
Institution: San Diego State University
Module: 15 - Hazard Detection & Resolution
Section: 4 - Control Hazard Detection + Integrated 5-Scenario Test
Date Created: May 2026
Version: 1.0
License: Educational Use

Description:
    This file is part of Project 15 for COMPE 475, Section 4.

    Section 4 adds control hazard detection and resolution on top of the
    structural hazard, forwarding, and stalling logic from Sections 1–3.

    When an upstream worker result invalidates already-queued downstream
    work, those queued workers are flushed and the request is rerouted to
    a corrected path. This mirrors a branch misprediction recovery in a
    CPU pipeline, where speculatively queued work is discarded.
"""

# ============================================================
# IMPORTS
# ============================================================

from __future__ import annotations
import os
import sys
import time
import uuid
import logging
import threading
from collections import defaultdict, deque
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool

# ============================================================
# IMPORT WORKERS
# Running from src/react_agent with: python .\graph.py
# ============================================================

try:
    from react_agent.database_agent import graph as database_search_agent
    from react_agent.code_analysis_agent import graph as code_analysis_agent
    from react_agent.decision_routing_agent import graph as decision_routing_agent
except ModuleNotFoundError:
    from database_agent import graph as database_search_agent
    from code_analysis_agent import graph as code_analysis_agent
    from decision_routing_agent import graph as decision_routing_agent

# ============================================================
# HAZARD LOGGER
# ============================================================

try:
    import hazard_logger
except ModuleNotFoundError:
    from react_agent import hazard_logger

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2

SUPERVISOR_SYSTEM_PROMPT = """
You are Korra Supervisor, a specialized supervisor agent for COMPE 475 Project 15.

You act as a processor's Control Unit:
- The user request is the instruction
- You decode the request
- You dispatch to the correct worker or workers
- You coordinate execution in sequence
- You synthesize the final answer when multiple workers are needed

You must never do the specialized work yourself.
You must always delegate when a worker applies.

AVAILABLE WORKERS

database_search_agent:
- Use for GPA, grades, records, stored data, database operations, skill level, course history

code_analysis_agent:
- Use for Python, C, and RISC-V code analysis, explanation, bugs, complexity, improvements

decision_routing_agent:
- Use for recommendations, choices, readiness decisions, difficulty assignment, prioritization

ROUTING RULES

Single-worker:
- Database-only -> database_search_agent
- Code-only -> code_analysis_agent
- Decision-only -> decision_routing_agent

Multi-worker:
- Database + Decision:
  1) database_search_agent
  2) decision_routing_agent

- Code + Decision:
  1) code_analysis_agent
  2) decision_routing_agent

- Database + Code + Decision:
  1) database_search_agent
  2) code_analysis_agent
  3) decision_routing_agent

COORDINATION RULES

- Workers must execute in sequence
- Output from earlier workers must be available to later workers
- decision_routing_agent should use earlier worker results as evidence

SYNTHESIS RULES

- For single-worker requests, pass through the worker response
- For multi-worker requests, combine the important findings into one clean final answer

IMPORTANT

- Do not say which worker should handle the request
- Actually route the request
- Do not stop early when multiple workers are required
""".strip()


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("project15_section4")


# ============================================================
# ENV SETUP
# ============================================================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    log.error("Missing OPENAI_API_KEY")
    sys.exit("Missing OPENAI_API_KEY in your environment (.env).")


# ============================================================
# STATE
# ============================================================

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# MODEL
# ============================================================

model = ChatOpenAI(
    model=DEFAULT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    streaming=True,
)


# ============================================================
# MODULE 15 SECTION 1 - STRUCTURAL HAZARD TRACKING
# ============================================================

WORKER_BUSY: dict[str, str] = {}
WORKER_QUEUES: dict[str, deque[tuple[str, threading.Event]]] = defaultdict(deque)
HAZARD_LOCK = threading.Lock()


def _get_request_id(config: dict | None) -> str:
    try:
        configurable = config["configurable"]
        if "request_id" in configurable:
            return str(configurable["request_id"])
        if "thread_id" in configurable:
            return str(configurable["thread_id"])
    except Exception:
        pass
    return f"req_{uuid.uuid4().hex[:8]}"


def _log_structural_hazard(worker_name: str, request_id: str, blocking_request_id: str) -> None:
    hazard_logger.structural_hazard(worker_name, request_id, blocking_request_id)


def _log_resume(worker_name: str, request_id: str) -> None:
    hazard_logger.resume(worker_name, request_id)


def acquire_worker_slot(worker_name: str, request_id: str) -> None:
    wait_event: threading.Event | None = None

    with HAZARD_LOCK:
        current_owner = WORKER_BUSY.get(worker_name)

        if current_owner is None:
            WORKER_BUSY[worker_name] = request_id
            return

        if current_owner == request_id:
            return

        wait_event = threading.Event()
        WORKER_QUEUES[worker_name].append((request_id, wait_event))
        _log_structural_hazard(worker_name, request_id, current_owner)

    wait_event.wait()
    _log_resume(worker_name, request_id)


def release_worker_slot(worker_name: str, request_id: str) -> None:
    with HAZARD_LOCK:
        current_owner = WORKER_BUSY.get(worker_name)

        if current_owner != request_id:
            return

        if WORKER_QUEUES[worker_name]:
            next_request_id, next_event = WORKER_QUEUES[worker_name].popleft()
            WORKER_BUSY[worker_name] = next_request_id
            next_event.set()
        else:
            WORKER_BUSY.pop(worker_name, None)


def build_guarded_worker(worker_name: str, worker_graph):
    class WorkerState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def guarded_node(state: WorkerState, config: dict | None = None):
        request_id = _get_request_id(config)

        acquire_worker_slot(worker_name, request_id)
        try:
            # Slow these workers on purpose so demos are screenshot-friendly
            if worker_name == "database_search_agent":
                time.sleep(3)
            if worker_name == "decision_routing_agent":
                time.sleep(2)

            result = worker_graph.invoke(state, config=config)
            return result
        finally:
            release_worker_slot(worker_name, request_id)

    builder = StateGraph(WorkerState)
    builder.add_node(worker_name, guarded_node)
    builder.add_edge(START, worker_name)
    builder.add_edge(worker_name, END)

    return builder.compile(name=worker_name)


# ============================================================
# GUARDED WORKERS
# ============================================================

database_search_agent_guarded = build_guarded_worker(
    "database_search_agent",
    database_search_agent,
)

code_analysis_agent_guarded = build_guarded_worker(
    "code_analysis_agent",
    code_analysis_agent,
)

decision_routing_agent_guarded = build_guarded_worker(
    "decision_routing_agent",
    decision_routing_agent,
)


# ============================================================
# MODULE 15 SECTION 2 - DATA HAZARD FORWARDING
# ============================================================

RESULT_CACHE: dict[str, dict[str, str]] = defaultdict(dict)


def _extract_text_from_graph_result(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""
    return str(getattr(messages[-1], "content", ""))


def _log_forwarding(producer_name: str, consumer_name: str, request_id: str) -> None:
    hazard_logger.forwarding(producer_name, consumer_name, request_id)


# ============================================================
# MODULE 15 SECTION 3 - DATA HAZARD STALLING
# ============================================================

def _log_stall(worker_name: str, request_id: str) -> None:
    hazard_logger.stall(worker_name, request_id)


def _log_bubble_cycle(cycle_num: int) -> None:
    hazard_logger.bubble_cycle(cycle_num)


# ============================================================
# MODULE 15 SECTION 4 - CONTROL HAZARD TRACKING
# ============================================================

# request_id -> list of workers tentatively queued downstream
QUEUED_DOWNSTREAM: dict[str, list[str]] = defaultdict(list)


def _queue_downstream_workers(request_id: str, workers: list[str]) -> None:
    QUEUED_DOWNSTREAM[request_id] = list(workers)


def _contains_no_record(text: str) -> bool:
    lowered = text.lower()
    patterns = [
        "no record",
        "not found",
        "does not exist",
        "unable to retrieve",
        "could not find",
    ]
    return any(p in lowered for p in patterns)


def _log_control_hazard(request_id: str, reason: str) -> None:
    hazard_logger.control_hazard(request_id, reason)


def _log_flush(request_id: str, workers: list[str]) -> None:
    hazard_logger.flush(request_id, workers)


def _flush_downstream_workers(request_id: str) -> list[str]:
    flushed = list(QUEUED_DOWNSTREAM.get(request_id, []))
    if flushed:
        _log_flush(request_id, flushed)
    QUEUED_DOWNSTREAM[request_id] = []
    return flushed


# ============================================================
# GRAPH
# ============================================================

def build_graph(use_checkpointer: bool = False):
    forward_tool = create_forward_message_tool("supervisor")

    workflow = create_supervisor(
        [
            database_search_agent_guarded,
            code_analysis_agent_guarded,
            decision_routing_agent_guarded,
        ],
        model=model,
        prompt=SUPERVISOR_SYSTEM_PROMPT,
        tools=[forward_tool],
        add_handoff_messages=True,
        output_mode="last_message",
        supervisor_name="supervisor",
    )

    memory = InMemorySaver() if use_checkpointer else None

    if memory:
        return workflow.compile(checkpointer=memory)

    return workflow.compile()


# ============================================================
# GRAPH INSTANCES
# ============================================================

graph = build_graph(use_checkpointer=False)
cli_graph = build_graph(use_checkpointer=True)

log.info("Project 15 Section 4 graph initialized successfully.")


# ============================================================
# CLI HELPERS
# ============================================================

def stream_supervisor_reply(user_text: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    printed_any_token = False

    print(f"\nKorra Supervisor [thread {thread_id}]: ", end="", flush=True)

    try:
        for chunk in cli_graph.stream(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                message_chunk, _metadata = chunk

                if getattr(message_chunk, "content", None):
                    print(message_chunk.content, end="", flush=True)
                    printed_any_token = True

        if not printed_any_token:
            state = cli_graph.get_state(config)
            messages = state.values.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                print(messages[-1].content, end="", flush=True)

        print()

    except Exception as exc:
        print(f"\n[ERROR] {exc}")


def show_thread_summary(thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = cli_graph.get_state(config)
    messages = snapshot.values.get("messages", [])

    print(f"\n--- Checkpoint Summary for thread {thread_id} ---")
    print(f"Message count in saved state: {len(messages)}")

    if messages:
        last_msg = messages[-1]
        role = last_msg.__class__.__name__
        content = getattr(last_msg, "content", "")
        print(f"Last message type: {role}")
        print(f"Last message preview: {str(content)[:120]}")
    print("----------------------------------------------\n")


# ============================================================
# SECTION 1 DEMO
# ============================================================

def run_structural_hazard_demo() -> None:
    request_a = 'Save this note under hazard_A.json: {"request":"A","value":"first database request"}'
    request_b = 'Save this note under hazard_B.json: {"request":"B","value":"second database request"}'

    def worker_request(user_text: str, thread_id: str):
        config = {
            "configurable": {
                "thread_id": thread_id,
                "request_id": thread_id,
            }
        }
        try:
            result = cli_graph.invoke(
                {"messages": [HumanMessage(content=user_text)]},
                config=config,
            )
            print(f"\n--- Final result for thread {thread_id} ---")
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print(getattr(last_msg, "content", ""))
        except Exception as exc:
            print(f"\n[ERROR][thread {thread_id}] {exc}")

    print("\nStarting structural hazard demo...")
    print("Request A and Request B will both target database_search_agent.\n")

    t1 = threading.Thread(target=worker_request, args=(request_a, "A"))
    t2 = threading.Thread(target=worker_request, args=(request_b, "B"))

    t1.start()
    time.sleep(0.2)
    t2.start()

    t1.join()
    t2.join()

    print("\nStructural hazard demo complete.\n")


# ============================================================
# SECTION 2 DEMO
# ============================================================

def run_forwarding_demo() -> None:
    request_id = "FWD1"
    config = {
        "configurable": {
            "thread_id": request_id,
            "request_id": request_id,
        }
    }

    db_prompt = (
        "Read student_12345.json from the database and return only the student's "
        "GPA, completed programming courses, and skill level in a short summary."
    )

    decision_prompt_prefix = (
        "Using ONLY the forwarded database result below, decide whether this student "
        "should take COMPE 475. Give a concise recommendation and reasoning.\n\n"
        "Forwarded database result:\n"
    )

    print("\nStarting forwarding demo...")
    print("Producer: database_search_agent")
    print("Consumer: decision_routing_agent\n")

    db_result = database_search_agent_guarded.invoke(
        {"messages": [HumanMessage(content=db_prompt)]},
        config=config,
    )
    db_text = _extract_text_from_graph_result(db_result)
    RESULT_CACHE[request_id]["database_search_agent"] = db_text

    print("--- Producer output (database_search_agent) ---")
    print(db_text)

    if RESULT_CACHE[request_id].get("database_search_agent"):
        _log_forwarding("database_search_agent", "decision_routing_agent", request_id)
        print("[INFO] Supervisor bypassed. Invoking decision_routing_agent directly from cached producer output.\n")

        consumer_input = decision_prompt_prefix + RESULT_CACHE[request_id]["database_search_agent"]

        decision_result = decision_routing_agent_guarded.invoke(
            {"messages": [HumanMessage(content=consumer_input)]},
            config=config,
        )
        decision_text = _extract_text_from_graph_result(decision_result)
        RESULT_CACHE[request_id]["decision_routing_agent"] = decision_text

        print("--- Consumer output (decision_routing_agent) ---")
        print(decision_text)
    else:
        print("[ERROR] No producer output found in cache; forwarding not possible.")

    print("\nForwarding demo complete.\n")


# ============================================================
# SECTION 3 DEMO
# ============================================================

def run_stall_demo() -> None:
    request_id = "STALL1"
    config = {
        "configurable": {
            "thread_id": request_id,
            "request_id": request_id,
        }
    }

    db_prompt = (
        "Read student_12345.json from the database and return only the student's "
        "GPA, completed programming courses, and skill level in a short summary."
    )

    producer_done = threading.Event()

    print("\nStarting stall demo...")
    print("Producer: database_search_agent (slow)")
    print("Dependent consumer: code_analysis_agent\n")

    def producer():
        db_result = database_search_agent_guarded.invoke(
            {"messages": [HumanMessage(content=db_prompt)]},
            config=config,
        )
        db_text = _extract_text_from_graph_result(db_result)
        RESULT_CACHE[request_id]["database_search_agent"] = db_text
        producer_done.set()

        print("--- Producer output (database_search_agent) ---")
        print(db_text)

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    if not producer_done.is_set():
        _log_stall("code_analysis_agent", request_id)

    bubble_count = 0
    while not producer_done.is_set():
        bubble_count += 1
        _log_bubble_cycle(bubble_count)
        time.sleep(0.5)

    while bubble_count < 2:
        bubble_count += 1
        _log_bubble_cycle(bubble_count)
        time.sleep(0.1)

    _log_resume("code_analysis_agent", request_id)

    forwarded_db_text = RESULT_CACHE[request_id].get("database_search_agent", "")
    consumer_prompt = (
        "Analyze the following situation. A student's database summary is included below. "
        "Use that information to judge whether the following Python code is appropriate for "
        "their current level, and briefly explain the code concepts involved.\n\n"
        f"Forwarded database result:\n{forwarded_db_text}\n\n"
        "Python code:\n"
        "def process_numbers(nums):\n"
        "    total = 0\n"
        "    for n in nums:\n"
        "        total += n\n"
        "    return total\n"
    )

    consumer_result = code_analysis_agent_guarded.invoke(
        {"messages": [HumanMessage(content=consumer_prompt)]},
        config=config,
    )
    consumer_text = _extract_text_from_graph_result(consumer_result)
    RESULT_CACHE[request_id]["code_analysis_agent"] = consumer_text

    producer_thread.join()

    print("\n--- Consumer output (code_analysis_agent) ---")
    print(consumer_text)

    print("\nStall demo complete.\n")


# ============================================================
# SECTION 4 - CONTROL HAZARD DEMO
# ============================================================

def run_control_hazard_demo() -> None:
    request_id = "CTRL1"
    config = {
        "configurable": {
            "thread_id": request_id,
            "request_id": request_id,
        }
    }

    print("\nStarting control hazard demo...")
    print("Canonical scenario: no record found -> flush -> reroute\n")

    # Tentatively queue downstream work before DB result is known
    _queue_downstream_workers(request_id, ["code_analysis_agent", "decision_routing_agent"])
    print(f"[INFO] Tentatively queued downstream workers for request {request_id}: {QUEUED_DOWNSTREAM[request_id]}")

    db_prompt = (
        "Read missing_student_99999.json from the database and summarize the student record."
    )

    db_result = database_search_agent_guarded.invoke(
        {"messages": [HumanMessage(content=db_prompt)]},
        config=config,
    )
    db_text = _extract_text_from_graph_result(db_result)
    RESULT_CACHE[request_id]["database_search_agent"] = db_text

    print("\n--- Producer output (database_search_agent) ---")
    print(db_text)

    if _contains_no_record(db_text):
        _log_control_hazard(
            request_id,
            "database_search_agent returned no record found, invalidating queued downstream work"
        )

        flushed_workers = _flush_downstream_workers(request_id)

        print(f"[INFO] Flushed workers: {flushed_workers}")

        # Reroute to corrected path
        _log_resume("help_response_path", request_id)

        rerouted_response = (
            "Rerouted response: No student record was found in the database, so the queued "
            "Code Analysis and Decision/Routing work was discarded. The system has switched "
            "to a help/error path and is asking the user to provide a valid student record "
            "or create one before continuing."
        )

        print("\n--- Rerouted response ---")
        print(rerouted_response)
    else:
        print("[ERROR] Control hazard condition did not trigger.")

    print("\nControl hazard demo complete.\n")


# ============================================================
# SECTION 4 - COMBINED HAZARD DEMO
# ============================================================

def run_combined_hazard_demo() -> None:
    """
    Combined hazard scenario:
    - One slow DB producer creates a shared result
    - Two dependent consumers wait on that result (data hazard -> stall/bubbles)
    - When the shared DB result becomes ready, both consumers forward it
      to the same downstream decision worker
    - The shared downstream worker causes a structural hazard between the
      two consumer requests
    """
    shared_request_id = "COMBINED_SHARED"
    shared_config = {
        "configurable": {
            "thread_id": shared_request_id,
            "request_id": shared_request_id,
        }
    }

    producer_done = threading.Event()

    print("\nStarting combined hazard demo...")
    print("Two dependent requests will wait on a shared DB result, then both compete")
    print("for the same downstream decision_routing_agent.\n")

    db_prompt = (
        "Read student_12345.json from the database and return only the student's "
        "GPA, completed programming courses, and skill level in a short summary."
    )

    def shared_producer():
        db_result = database_search_agent_guarded.invoke(
            {"messages": [HumanMessage(content=db_prompt)]},
            config=shared_config,
        )
        db_text = _extract_text_from_graph_result(db_result)
        RESULT_CACHE["COMBINED_SHARED"]["database_search_agent"] = db_text
        producer_done.set()

        print("--- Shared producer output (database_search_agent) ---")
        print(db_text)

    def dependent_consumer(request_id: str):
        config = {
            "configurable": {
                "thread_id": request_id,
                "request_id": request_id,
            }
        }

        # Data hazard stall until shared DB output is ready
        if not producer_done.is_set():
            _log_stall("decision_routing_agent", request_id)

        bubble_count = 0
        while not producer_done.is_set():
            bubble_count += 1
            _log_bubble_cycle(bubble_count)
            time.sleep(0.4)

        while bubble_count < 2:
            bubble_count += 1
            _log_bubble_cycle(bubble_count)
            time.sleep(0.1)

        _log_resume("decision_routing_agent", request_id)

        forwarded_text = RESULT_CACHE["COMBINED_SHARED"].get("database_search_agent", "")
        _log_forwarding("database_search_agent", "decision_routing_agent", request_id)

        consumer_prompt = (
            "Using ONLY the forwarded database result below, decide whether the student "
            "should take COMPE 475. Give a concise recommendation and reasoning.\n\n"
            f"Forwarded database result:\n{forwarded_text}"
        )

        decision_result = decision_routing_agent_guarded.invoke(
            {"messages": [HumanMessage(content=consumer_prompt)]},
            config=config,
        )
        decision_text = _extract_text_from_graph_result(decision_result)
        RESULT_CACHE[request_id]["decision_routing_agent"] = decision_text

        print(f"\n--- Final consumer output ({request_id}) ---")
        print(decision_text)

    producer_thread = threading.Thread(target=shared_producer)
    consumer_a = threading.Thread(target=dependent_consumer, args=("COMB_A",))
    consumer_b = threading.Thread(target=dependent_consumer, args=("COMB_B",))

    producer_thread.start()
    time.sleep(0.2)
    consumer_a.start()
    time.sleep(0.1)
    consumer_b.start()

    producer_thread.join()
    consumer_a.join()
    consumer_b.join()

    print("\nCombined hazard demo complete.\n")


# ============================================================
# SCENARIO ALIASES
# ============================================================

def run_scenario_1():
    run_structural_hazard_demo()

def run_scenario_2():
    run_forwarding_demo()

def run_scenario_3():
    run_stall_demo()

def run_scenario_4():
    run_control_hazard_demo()

def run_scenario_5():
    run_combined_hazard_demo()


# ============================================================
# CLI
# ============================================================

def main() -> None:
    current_thread = "1"

    print("\nKorra CLI - Module 15 Section 4 Control Hazard + 5 Scenarios")
    print("Commands:")
    print("  /scenario1     Structural hazard demo")
    print("  /scenario2     Data forwarding demo")
    print("  /scenario3     Data stall demo")
    print("  /scenario4     Control hazard demo")
    print("  /scenario5     Combined hazard demo")
    print("  /hazardtest    Run structural hazard demo")
    print("  /forwardtest   Run data forwarding demo")
    print("  /stalltest     Run data hazard stall demo")
    print("  /controltest   Run control hazard demo")
    print("  /combinedtest  Run combined hazard demo")
    print("  /thread <id>   Switch conversation thread")
    print("  /state         Inspect saved checkpoint state")
    print("  /exit          Quit\n")
    print(f"Current thread: {current_thread}")

    while True:
        try:
            user_input = input(f"\nYou [thread {current_thread}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Supervisor Agent CLI.")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Goodbye.")
            break

        if user_input.lower() == "/scenario1":
            run_scenario_1()
            continue

        if user_input.lower() == "/scenario2":
            run_scenario_2()
            continue

        if user_input.lower() == "/scenario3":
            run_scenario_3()
            continue

        if user_input.lower() == "/scenario4":
            run_scenario_4()
            continue

        if user_input.lower() == "/scenario5":
            run_scenario_5()
            continue

        if user_input.lower() == "/hazardtest":
            run_structural_hazard_demo()
            continue

        if user_input.lower() == "/forwardtest":
            run_forwarding_demo()
            continue

        if user_input.lower() == "/stalltest":
            run_stall_demo()
            continue

        if user_input.lower() == "/controltest":
            run_control_hazard_demo()
            continue

        if user_input.lower() == "/combinedtest":
            run_combined_hazard_demo()
            continue

        if user_input.lower() == "/state":
            show_thread_summary(current_thread)
            continue

        if user_input.lower().startswith("/thread "):
            new_thread = user_input.split(maxsplit=1)[1].strip()
            if not new_thread:
                print("Please provide a thread id, e.g. /thread 2")
                continue
            current_thread = new_thread
            print(f"Switched to thread {current_thread}")
            continue

        stream_supervisor_reply(user_input, current_thread)


if __name__ == "__main__":
    main()