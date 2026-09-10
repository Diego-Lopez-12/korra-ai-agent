"""
File: graph.py
Author: Professor Denis O. Núñez
Email: donunez@sdsu.edu
Course: COMPE 475 – Microprocessors
Institution: San Diego State University
Module: 13 - Multi-Path Processing Architecture
Section: 3 - Decision/Routing Agent
Date Created: April 2026
Version: 1.0
License: Educational Use

Description:
    This file is part of Project 13 for COMPE 475, Section 3.

    Section 3 implements Korra as a specialized Decision/Routing Agent.
    This version mirrors a processor's branch/control unit by focusing on
    conditional evaluation, strategic routing, trade-off analysis, and
    reasoned recommendations across multiple possible paths.

    Korra in this section is intentionally specialized for:
        • decision-making and routing
        • conditional evaluation
        • multi-path option comparison
        • strategic recommendations and prioritization

    This Section 3 version minimizes unrelated capabilities such as:
        • database storage and retrieval
        • code analysis
        • web search and GitHub operations

    This version is intended to run with:
        • LangGraph Studio for testing
        • decision-analysis prompts
        • branch/control-style reasoning workflows

Usage Instructions:
    1. Create a .env file in the korra folder with your OpenAI API key.
    2. Start LangGraph Studio or your local backend environment.
    3. Run the graph and test the five decision-making scenarios.
    4. Capture screenshots showing the required Section 3 test cases.
"""

# ============================================================
# IMPORTS
# ============================================================

from __future__ import annotations
import os
import sys
import logging
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessageChunk,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are Korra, a specialized Decision/Routing Agent for COMPE 475 Project 13 Section 3.

Your role:
- You act like a branch/control unit in a processor.
- Your primary purpose is to evaluate conditions, compare multiple options, choose between alternative paths, and provide reasoned recommendations.
- You specialize in decision-making and routing, not database retrieval, code analysis, web search, or GitHub tasks.

Core capabilities:
- Analyze a decision problem and identify the main choice being made
- Evaluate multiple criteria and trade-offs
- Compare alternative options clearly
- Route the user toward the strongest recommendation
- Rank priorities when several factors compete
- Explain the reasoning behind the final recommendation

Decision process:
For each request, follow this structure:
1. Identify the decision to be made
2. List the key criteria or constraints
3. Compare the available options
4. Evaluate trade-offs
5. Recommend the best path
6. Briefly explain why that path is best

Decision rules:
- Always consider at least two relevant criteria when making a recommendation
- If one option is best overall, say so clearly
- If the answer depends on priorities, explain how the outcome changes based on those priorities
- If there are strong trade-offs, make them explicit
- If the user asks for prioritization, rank the options or factors clearly
- Be practical, clear, and logically structured

Response formatting rules:
- Start with "Decision:"
- Then include "Criteria:"
- Then include "Option Comparison:"
- Then include "Recommendation:"
- Then include "Why:"
- Keep responses structured, concise, and easy to follow

Safety rules:
- Do not fabricate facts
- Do not pretend there is only one correct answer when there are reasonable trade-offs
- Stay focused on decision-making and routing behavior
"""

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("korra")

# Reduce noise from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate that required API key is present
if not OPENAI_API_KEY:
    log.error("Missing required OPENAI_API_KEY in environment")
    sys.exit("Missing OPENAI_API_KEY in your environment (.env).")

# ============================================================
# STATE DEFINITION
# ============================================================

class State(TypedDict):
    """
    State schema for the Module 13 Section 3 Decision/Routing Agent.

    The add_messages merge policy preserves conversation history for each
    thread so Korra can support multi-turn decision-making, follow-up
    questions, and evolving recommendation scenarios.

    Attributes:
        messages: Full conversation history for one conversation thread.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# GRAPH INITIALIZATION
# ============================================================

def initialize_korra():
    """
    Build and compile the Korra graph for Module 13 Section 3.

    This version creates a specialized StateGraph focused on decision-making
    and routing. It mirrors a processor's branch/control unit by emphasizing
    conditional evaluation, path selection, trade-off analysis, and
    strategic recommendations.

    Graph structure:
        START -> korra -> END

    Tool nodes:
        - korra: Main chatbot node with Section 3 decision-routing prompt

    Returns:
        StateGraph: Compiled graph ready for LangGraph Studio testing
    """
    # Initialize the language model
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3)

    def korra(state: State) -> dict[str, list[BaseMessage]]:
        """
        Main chatbot node for the Section 3 Decision/Routing Agent.

        This node prepends the specialized Section 3 system prompt so the
        language model focuses on conditional logic, option comparison,
        prioritization, and recommendation-making.

        Args:
            state: Current conversation state containing message history

        Returns:
            dict: Updated state with AI response added to messages
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph_builder = StateGraph(State)

    graph_builder.add_node("korra", korra)

    graph_builder.add_edge(START, "korra")
    graph_builder.add_edge("korra", END)

    return graph_builder.compile(name="decision_routing_agent")

# Create the graph instance
graph = initialize_korra()

log.info("Korra graph initialized for Module 13 Section 3")
log.info("Tool nodes: korra (decision/routing only)")


def stream_graph_updates(user_input: str, thread_id: str) -> None:
    """
    Stream Korra's response token-by-token in the terminal.
    """
    config = {"configurable": {"thread_id": thread_id}}

    print("Korra: ", end="", flush=True)

    for chunk, metadata in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            if isinstance(chunk.content, str):
                print(chunk.content, end="", flush=True)
            elif isinstance(chunk.content, list):
                for item in chunk.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        print(item.get("text", ""), end="", flush=True)

    print()


def show_thread_history(thread_id: str) -> None:
    """
    Display saved messages for the current thread.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    print(f"\nSaved history for thread {thread_id}:")
    messages = state.values.get("messages", [])

    if not messages:
        print("(No saved messages)\n")
        return

    for msg in messages:
        role = msg.__class__.__name__
        content = getattr(msg, "content", "")
        print(f"{role}: {content}")
    print()


def main() -> None:
    """
    CLI loop for interacting with the Section 3 Decision/Routing Agent.
    """
    current_thread = "1"

    print("\nKorra CLI - Module 13 Section 3")
    print("Commands:")
    print("  /thread <id>   Switch conversation thread")
    print("  /history       Show conversation history")
    print("  quit           Exit\n")
    print(f"Current thread: {current_thread}\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if user_input.startswith("/thread "):
            new_thread = user_input.replace("/thread ", "", 1).strip()
            if new_thread:
                current_thread = new_thread
                print(f"Switched to thread {current_thread}\n")
            else:
                print("Please provide a valid thread ID.\n")
            continue

        if user_input == "/history":
            show_thread_history(current_thread)
            continue

        stream_graph_updates(user_input, current_thread)


if __name__ == "__main__":
    main()