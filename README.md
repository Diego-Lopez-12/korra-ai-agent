# Korra — Multi-Agent Pipeline System

Korra is a **LangGraph-based multi-agent AI system** that applies computer architecture concepts to a working agentic architecture by modeling specialized functional units, control-unit coordination, pipelined execution, and hazard handling.

Korra evolved from a tool-enabled chatbot into a coordinated multi-agent system capable of routing complex requests across specialized workers, managing persistent data, and detecting and resolving execution conflicts.

## Project Resources

- [Project Presentation](docs/Korra_Project_Presentation.pdf)

## Key Features

- **Multi-Agent Architecture** — Specialized agents handle database operations, code analysis, and decision-making.
- **Supervisor Coordination** — A central supervisor routes requests and coordinates multi-agent workflows.
- **Pipeline Execution** — Complex requests can move through multiple specialized agents in sequence.
- **Hazard Detection & Resolution** — Implements software analogs of structural, data, and control hazards.
- **Persistence & Database Storage** — SQLite-backed storage and persistent conversation state.
- **Human-in-the-Loop** — Supports user intervention and approval during execution.
- **MCP Integration** — Uses the Model Context Protocol for communication between system components.
- **Web Interface** — Provides an interactive frontend for communicating with Korra.
- **Custom C Tooling** — Integrates a C-based file analysis utility with the Python agent system.
- **Docker Support** — Application components can be run in containerized environments.

## Architecture

Korra uses a supervisor-and-worker architecture inspired by the organization of a microprocessor.

### Supervisor Agent — Control Unit

The supervisor interprets incoming requests, determines which workers are required, coordinates their execution order, and synthesizes the final response.

### Database Search Agent — Memory / Load-Store Unit

Handles persistent data operations such as storing, retrieving, and updating records through an SQLite database.

### Code Analysis Agent — ALU / Compute Unit

Analyzes source code, identifies behavior and potential issues, and provides technical recommendations.

### Decision & Routing Agent — Control / Branch Unit

Uses information produced by other agents to make recommendations and higher-level decisions.

### Hazard Detection Layer

Monitors execution and resolves conflicts that occur between workers and dependent requests.

## Pipeline Execution

A complex request can require multiple agents to contribute to the final result.

For example:

```text
User Request
     |
     v
Supervisor
     |
     v
Database Search
     |
     v
Code Analysis
     |
     v
Decision / Routing
     |
     v
Final Response
```

The supervisor determines which stages are necessary for a particular request, so not every request must execute every worker.

## Hazard Handling

Korra implements several concepts inspired by hazards in pipelined processors.

### Structural Hazards

When two requests attempt to use the same worker simultaneously, the second request is queued until the worker becomes available.

```text
[STRUCTURAL HAZARD] -> Queue Request -> [RESUME]
```

### Data Forwarding

When an upstream worker has already produced data required by a downstream worker, the cached result can be forwarded directly to the consumer rather than performing an unnecessary supervisor round trip.

```text
Producer -> [FORWARDING] -> Consumer
```

### Data Stalling

If a dependent worker reaches its execution stage before the required upstream result is available, execution stalls and pipeline bubbles are inserted until the dependency is resolved.

```text
[STALL]
[BUBBLE CYCLE 1]
[BUBBLE CYCLE 2]
...
[RESUME]
```

### Control Hazards

If an upstream result invalidates tentatively scheduled downstream work, the queued work is flushed and execution is redirected to the corrected path.

```text
[CONTROL HAZARD] -> [FLUSH] -> [RESUME]
```

## Technologies

| Category | Technologies |
| --- | --- |
| Languages | Python, C, TypeScript |
| Agent Framework | LangGraph, LangChain |
| AI | OpenAI API |
| Database | SQLite |
| Communication | Model Context Protocol (MCP) |
| Frontend | React / TypeScript |
| Containerization | Docker |
| Version Control | Git / GitHub |

## Computer Architecture Concepts

Korra implements software analogs of several hardware behaviors found in modern processor architectures.

| Korra Component | Processor Concept |
| --- | --- |
| Supervisor Agent | Control Unit |
| Database Search Agent | Memory / Load-Store Unit |
| Code Analysis Agent | ALU / Compute Unit |
| Decision & Routing Agent | Control / Branch Unit |
| Multi-Agent Workflow | Instruction Pipeline |
| Worker Queuing | Structural Hazard Resolution |
| Result Forwarding | Data Forwarding / Bypassing |
| Pipeline Bubbles | Data Hazard Stalling |
| Flush & Reroute | Control Hazard / Branch Recovery |
| Human-in-the-Loop | Interrupt-like Control |
| Persistence / SQLite | Memory & Storage |

## Project Structure

```text
.
├── agent-chat-ui/          # Web-based user interface
├── docs/                   # Project documentation and presentation
├── korra/                  # LangGraph backend and agent implementation
│   ├── src/react_agent/
│   │   ├── graph.py
│   │   ├── database_agent.py
│   │   ├── code_analysis_agent.py
│   │   ├── decision_routing_agent.py
│   │   ├── hazard_logger.py
│   │   ├── prompts.py
│   │   ├── tools.py
│   │   └── ...
│   └── tests/
├── docker-compose.yml
└── README.md
```

## Example Workflow

A request such as:

> Look up a student's previous grades, analyze the provided Python code, and recommend whether they should take a computer engineering course.

can require Korra to:

1. Route the request to the **Database Search Agent** to retrieve the student's academic record.
2. Route the code to the **Code Analysis Agent** for analysis.
3. Provide the relevant results to the **Decision & Routing Agent**.
4. Combine the worker outputs into a final response for the user.

This workflow demonstrates how specialized components can cooperate under centralized coordination in a manner analogous to functional units operating within a processor.

## What I Learned

Building Korra provided hands-on experience with:

- Designing and coordinating multi-agent systems with LangGraph
- Integrating Python, C, databases, APIs, and web interfaces
- Designing system prompts for reliable agent routing
- Managing dependencies and shared resources across concurrent workflows
- Applying pipelining and hazard-resolution concepts in a working software system
- Debugging increasingly complex interactions between independent system components

Most importantly, the project helped connect theoretical computer architecture concepts to observable behavior in a working system.

## Future Improvements

Potential extensions include:

- Additional specialized agents and functional units
- Branch prediction for control-hazard handling
- Out-of-order task execution and scheduling
- Multithreaded concurrent request processing
- Performance monitoring for latency and throughput
- Multiple parallel pipelines inspired by superscalar processors

## Author

**Diego Lopez**  
Computer Engineering  
San Diego State University