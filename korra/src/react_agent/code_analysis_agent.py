"""
File: graph.py
Author: Professor Denis O. Núñez
Email: donunez@sdsu.edu
Course: COMPE 475 – Microprocessors
Institution: San Diego State University
Module: 13 - Multi-Path Processing Architecture
Section: 2 - Code Analysis Agent
Date Created: April 2026
Version: 1.0
License: Educational Use

Description:
    This file is part of Project 13 for COMPE 475, Section 2.

    Section 2 implements Korra as a specialized Code Analysis Agent.
    This version mirrors a processor's Arithmetic Logic Unit (ALU) by
    focusing on computational analysis, code understanding, bug detection,
    optimization suggestions, and explanation of program behavior.

    Korra in this section is intentionally specialized for:
        • Python code analysis
        • C code analysis
        • RISC-V assembly analysis
        • Code explanation and improvement suggestions

    This Section 2 version minimizes unrelated capabilities such as:
        • database storage and retrieval
        • web search
        • GitHub repository operations

    This version is intended to run with:
        • LangGraph Studio for testing
        • Direct code-input analysis workflows
        • Language-specific code reasoning prompts

Usage Instructions:
    1. Create a .env file in the korra folder with your OpenAI API key.
    2. Start LangGraph Studio or your local backend environment.
    3. Run the graph and test Python, C, and RISC-V code samples.
    4. Capture screenshots showing the required Section 2 test cases.
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
You are Korra, a specialized Code Analysis Agent for COMPE 475 Project 13 Section 2.

Your role:
- You act like an Arithmetic Logic Unit (ALU) in a processor.
- Your primary purpose is to analyze code, explain how it works, identify issues, and suggest improvements.
- You specialize in Python, C, and RISC-V assembly analysis.
- You are not a general-purpose database, web-search, or GitHub assistant in this section.

Core capabilities:
- Analyze Python code for syntax, logic, style, efficiency, and correctness
- Analyze C code for memory management, pointer usage, bugs, and best practices
- Analyze RISC-V assembly for instruction meaning, register usage, control flow, and program behavior
- Explain code clearly in plain language
- Suggest improvements and possible fixes when appropriate

Reasoning pattern:
For each request, follow this pattern:
1. Identify the language
2. Determine what the code is doing
3. Look for bugs, weaknesses, or optimization opportunities
4. Explain findings clearly
5. Suggest improvements when useful

Analysis rules:
- If the input is Python, discuss syntax, logic, loops, conditionals, edge cases, and possible optimizations
- If the input is C, discuss pointers, malloc/free usage, memory safety, best practices, and possible bugs
- If the input is RISC-V assembly, identify instructions, registers, branch behavior, arithmetic operations, and overall effect
- If the code appears correct, say so clearly and still explain what it does
- If the code has risks or weaknesses, explain them precisely
- Do not invent errors that are not supported by the code

Response formatting rules:
- Start by identifying the language
- Then explain what the code does
- Then list any issues or concerns
- Then provide improvement suggestions
- Be clear, structured, and concise

Safety rules:
- Do not fabricate code behavior
- Do not claim a bug exists unless you can justify it from the code
- Stay focused on code analysis and explanation
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
    State schema for the Module 13 Section 2 Code Analysis Agent.

    The add_messages merge policy preserves conversation history for each
    thread so Korra can support multi-turn code analysis and follow-up
    questions about previously discussed code.

    Attributes:
        messages: Full conversation history for one conversation thread.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# GRAPH INITIALIZATION
# ============================================================

def initialize_korra():
    """
    Build and compile the Korra graph for Module 13 Section 2.

    This version creates a specialized StateGraph focused on code analysis.
    It mirrors a processor's Arithmetic Logic Unit (ALU) by emphasizing
    computational reasoning, code explanation, bug identification, and
    optimization suggestions for Python, C, and RISC-V assembly.

    Graph structure:
        START -> korra -> END

    Tool nodes:
        - korra: Main chatbot node with Section 2 code-analysis prompt

    Returns:
        StateGraph: Compiled graph ready for LangGraph Studio testing
    """
    # Initialize the language model
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3)

    def korra(state: State) -> dict[str, list[BaseMessage]]:
        """
        Main chatbot node for the Section 2 Code Analysis Agent.

        This node prepends the specialized Section 2 system prompt so the
        language model focuses on analyzing Python, C, and RISC-V code.

        Args:
            state: Current conversation state containing message history

        Returns:
            dict: Updated state with AI response added to messages
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # Create the graph builder
    graph_builder = StateGraph(State)

    # Add the single specialized analysis node
    graph_builder.add_node("korra", korra)

    # Define the conversation flow
    graph_builder.add_edge(START, "korra")
    graph_builder.add_edge("korra", END)

    # Compile graph for LangGraph Studio
    return graph_builder.compile(name="code_analysis_agent")

# Create the graph instance
graph = initialize_korra()

log.info("Korra graph initialized for Module 13 Section 2")
log.info("Tool nodes: korra (code analysis only)")


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
    CLI loop for interacting with the Section 2 Code Analysis Agent.
    """
    current_thread = "1"

    print("\nKorra CLI - Module 13 Section 2")
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