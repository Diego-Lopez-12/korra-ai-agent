"""
File: graph.py
Author: Professor Denis O. Núñez
Email: donunez@sdsu.edu
Course: COMPE 475 – Microprocessors
Institution: San Diego State University
Module: 13 - Multi-Path Processing Architecture
Section: 1 - Database Search Agent
Date Created: April 2026
Version: 1.0
License: Educational Use

Description:
    This file is part of Project 13 for COMPE 475, Section 1.

    Section 1 implements Korra as a specialized Database Search Agent.
    This version mirrors a processor's memory/load-store unit by focusing
    on persistent storage, structured data retrieval, database search,
    and conversation history access.

    Korra in this section is intentionally specialized for:
        • SQLite database operations
        • Persistent data storage and retrieval
        • Structured record lookup
        • Conversation history access across sessions

    This Section 1 version minimizes unrelated capabilities such as:
        • web search
        • code analysis/file statistics
        • GitHub repository operations

    This version is intended to run with:
        • LangGraph Studio for testing
        • Persistent SQLite storage
        • Database-focused prompts and tool routing

Usage Instructions:
    1. Create a .env file in the korra folder with your OpenAI API key.
    2. Start LangGraph Studio or your local backend environment.
    3. Run the graph and test database storage/retrieval scenarios.
    4. Capture screenshots showing the required Section 1 test cases.
"""

# ============================================================
# IMPORTS
# ============================================================

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
    ToolMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from react_agent.tools.database_tool import (
    save_file_to_database,
    read_file_from_database,
    update_file_in_database,
    delete_file_from_database,
    list_files_in_database,
)

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are Korra, a specialized Database Search Agent for COMPE 475 Project 13 Section 1.

Your role:
- You act like a memory/load-store unit in a processor.
- Your primary purpose is to store, retrieve, update, delete, and search persistent information using SQLite database tools.
- You specialize in database and memory operations, not general-purpose web research, GitHub tasks, or code analysis.

Core capabilities:
- Save structured or unstructured information into persistent storage
- Read previously stored information
- Update existing stored records
- Delete records when requested
- List available stored records
- Help users organize data such as student records, course information, grades, notes, and saved conversation summaries

Reasoning pattern (ReAct):
For each request, follow this pattern:
1. THOUGHT: Briefly determine what database operation is needed
2. ACTION: Use the correct database tool
3. OBSERVATION: Review the database result carefully
4. ANSWER: Give the user a clear, natural-language response

Tool-use rules:
- Prefer database tools whenever the user asks to save, retrieve, search, list, update, or delete information
- If the user asks about stored student or course data, use database tools rather than answering from memory
- If the user asks to store conversation information for later retrieval, use the database tools
- If the needed data is not yet stored, clearly say so and suggest storing it first
- Do not use or mention web search, GitHub tools, or file-statistics capabilities

Database behavior rules:
- Treat the database as the main memory system for this agent
- Use clear filenames/record names when storing information
- Preserve data accuracy
- When useful, summarize retrieved records in a readable format
- For structured records, keep fields clearly labeled

Response formatting rules:
- Be concise, clear, and database-focused
- Tell the user what was stored, retrieved, updated, deleted, or listed
- When retrieving structured data, present the important fields clearly
- Ask for clarification only if the requested database action is ambiguous

Safety rules:
- Do not fabricate database contents
- Do not claim information was saved unless the database tool confirms it
- Do not claim information exists unless it was retrieved
- Respect privacy and handle stored information carefully
"""

# Section 1 uses only database tools so Korra behaves like a specialized
# memory/load-store unit rather than a general-purpose multi-tool agent.
DATABASE_TOOL_NAMES = {
    "save_file_to_database",
    "read_file_from_database",
    "update_file_in_database",
    "delete_file_from_database",
    "list_files_in_database",
}

DATABASE_WRITE_TOOL_NAMES = {
    "save_file_to_database",
    "update_file_in_database",
    "delete_file_from_database",
}


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

# Interrupts are useful in Studio for approving database write actions.
ENABLE_INTERRUPTS = os.getenv("ENABLE_INTERRUPTS", "true").strip().lower() == "true"

# ============================================================
# STATE DEFINITION
# ============================================================

class State(TypedDict):
    """
    State schema for the Korra graph used by the Module 11 Section 3 version.

    The add_messages merge policy preserves the conversation history for
    each thread so Korra can reason over prior user, assistant, and tool
    messages during multi-turn conversations.

    Attributes:
        messages: Full conversation history for one conversation thread.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# GRAPH INITIALIZATION
# ============================================================

def initialize_korra():
    """
    Build and compile the Korra graph for Module 13 Section 1.

    This version creates a specialized StateGraph focused only on SQLite
    database operations. It is designed to mirror a processor's load-store
    unit by emphasizing persistent data storage, retrieval, update, delete,
    and listing behavior.

    The Korra chatbot node uses a Section 1 system prompt to control:
        - Korra's database-search identity
        - ReAct-style database reasoning behavior
        - database-focused tool selection
        - clear response formatting
        - safe handling of persistent data

    Graph structure:
        START -> korra -> [route_tools] -> database_tool -> korra
                                        -> END

    Tool nodes:
        - korra: Main chatbot node with Section 1 system prompt
        - database_tool: SQLite CRUD operations

    Returns:
        StateGraph: Compiled graph ready for LangGraph Studio testing
    """
    # Initialize the language model
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3)

    # Initialize database CRUD tools.
    # Section 1 is intentionally specialized around persistent storage and retrieval.
    database_tools = [
        save_file_to_database,
        read_file_from_database,
        update_file_in_database,
        delete_file_from_database,
        list_files_in_database,
    ]

    # Bind only database tools to the model so this agent stays specialized.
    llm_with_tools = llm.bind_tools(database_tools)
    
    def is_approved(response: object) -> bool:
        """
        Convert Studio resume input into a True/False approval decision.

        Accepts either boolean values or common approval strings.
        """
        if isinstance(response, bool):
            return response

        normalized = str(response).strip().lower()
        return normalized in {"true", "yes", "y", "approve", "approved"}

    # Define korra node
    def korra(state: State) -> dict[str, list[BaseMessage]]:
        """
        Main chatbot node for the Section 1 Database Search Agent.

        This node prepends the specialized Section 1 system prompt so the
        language model focuses on storage, retrieval, persistence, and search
        operations using the SQLite database tools.

        Args:
            state: Current conversation state containing message history

        Returns:
            dict: Updated state with AI response added to messages
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    
    def database__tool(state: State) -> dict[str, list[BaseMessage]]:
        """
        Execute SQLite database operations for the Section 1 Database Search Agent.

        This is the agent's dedicated functional unit, analogous to a processor's
        load-store unit. Write operations such as save, update, and delete may
        require human approval when interrupts are enabled. Read-only operations
        such as read and list execute immediately.
        """
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]

        if ENABLE_INTERRUPTS and tool_name in DATABASE_WRITE_TOOL_NAMES:
            approval = interrupt({
                "question": "Approve this database operation?",
                "tool_name": tool_name,
                "tool_args": tool_call.get("args", {}),
                "reason": "Database write operations change persistent storage.",
                "instructions": "Resume with True to approve or False to reject."
            })

            if not is_approved(approval):
                return {
                    "messages": [
                        ToolMessage(
                            content="Database operation canceled by human reviewer.",
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        )
                    ]
                }

        tool_node = ToolNode(tools=database_tools)
        return tool_node.invoke(state)
    

    def route_tools(state: State) -> str:
        """
        Route tool calls for the Section 1 Database Search Agent.

        Since this version is intentionally specialized, only database tool
        calls are allowed to continue to execution. Any unexpected tool call
        stops the graph as a safety fallback.

        Args:
            state: Current conversation state containing message history

        Returns:
            str: "database_tool" if a database tool was called, otherwise END
        """
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            tool_name = tool_call["name"]
            log.info("Router saw tool call: %r", tool_name)

            if tool_name in DATABASE_TOOL_NAMES:
                return "database_tool"

            log.warning("Unexpected non-database tool call %r; stopping.", tool_name)
            return END

        log.info("No tool calls in last message; ending turn.")
        return END

    # Create the graph builder
    graph_builder = StateGraph(State)

    # Add nodes to the graph.
    # Section 1 uses only the main Korra node and the database tool node.
    graph_builder.add_node("korra", korra)
    graph_builder.add_node("database_tool", database__tool)

    # Define the conversation flow
    graph_builder.add_edge(START, "korra")
    graph_builder.add_conditional_edges("korra", route_tools, {
        "database_tool": "database_tool",
        END: END
    })

    # The database tool returns to Korra for the next step in the interaction
    graph_builder.add_edge("database_tool", "korra")


    # Compile graph for the LangGraph Studio.
    # Studio provides persistence automatically, so a custom checkpointer
    # should not be passed here.
    return graph_builder.compile(name="database_search_agent")

# Create the graph instance
graph = initialize_korra()

log.info("Korra graph initialized for Module 13 Section 1")
log.info("Tool nodes: database_tool (SQLite CRUD only)")


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
    CLI loop for interacting with the Section 1 Database Search Agent.
    """
    current_thread = "1"

    print("\nKorra CLI - Module 13 Section 1")
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