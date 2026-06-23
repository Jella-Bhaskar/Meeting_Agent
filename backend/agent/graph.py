"""
=============================================================================
agent/graph.py — The LangGraph Directed Graph
=============================================================================

WHY WE NEED THIS:
This file wires the 4 nodes together into a directed graph (DAG).
LangGraph executes this graph node-by-node, passing state between them.

CONCEPT — Why LangGraph instead of just calling functions in order?
You could write: fetch() → extract() → research() → store()
But LangGraph gives you:
  ✓ Conditional routing (different paths based on state)
  ✓ Parallel branches (research multiple companies at once)
  ✓ Built-in checkpointing (resume if interrupted)
  ✓ Observability (trace every state transition)
  ✓ Retries at the node level

For this project, LangGraph is somewhat overkill (4 linear nodes) —
but it's the industry pattern you'll use when building complex agents.

CONCEPT — StateGraph vs MessageGraph:
StateGraph: nodes share a typed state dict (what we use here)
MessageGraph: nodes exchange chat messages (for chatbots)

We use StateGraph because we're tracking structured data (meetings, briefs),
not a conversation.

GRAPH STRUCTURE:
  START → fetch_calendar → extract_companies → research_companies → store_briefs → END
  
All edges are unconditional here (linear flow).
In a more complex agent, you'd add conditional edges like:
  "if no meetings found → END early"
  "if calendar auth fails → error_handler → END"
=============================================================================
"""

import uuid
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    fetch_calendar_node,
    extract_companies_node,
    research_company_node,
    store_briefs_node,
)


def build_agent_graph():
    """
    Build and compile the LangGraph StateGraph.
    
    CONCEPT — Compilation:
    LangGraph "compiles" the graph before running it, similar to how a 
    database query planner optimizes a query before executing it.
    This catches structural errors (missing nodes, invalid edges) early.
    
    Returns a compiled graph ready to invoke.
    """
    # Create graph with our state schema
    # LangGraph uses the TypedDict to validate state transitions
    graph = StateGraph(AgentState)
    
    # Register nodes (name → function)
    # The name is used in logs and traces — pick descriptive names
    graph.add_node("fetch_calendar", fetch_calendar_node)
    graph.add_node("extract_companies", extract_companies_node)
    graph.add_node("research_companies", research_company_node)
    graph.add_node("store_briefs", store_briefs_node)
    
    # Define the execution flow with edges
    # add_edge(from, to) = "after from completes, run to"
    graph.set_entry_point("fetch_calendar")           # Start here
    graph.add_edge("fetch_calendar", "extract_companies")
    graph.add_edge("extract_companies", "research_companies")
    graph.add_edge("research_companies", "store_briefs")
    graph.add_edge("store_briefs", END)               # End here
    
    # Compile validates the graph structure and returns an executable
    return graph.compile()


def run_agent() -> list:
    """
    Execute the full agent pipeline.
    
    CONCEPT — Initial State:
    We must provide all required state fields when starting the graph.
    LangGraph merges our initial state with each node's returned updates.
    
    Returns the list of MeetingBrief dicts from the final state.
    """
    compiled_graph = build_agent_graph()
    
    # Initial state — all fields required by AgentState TypedDict
    initial_state: AgentState = {
        "raw_events": [],
        "company_signals": [],
        "briefs": [],
        "run_id": str(uuid.uuid4()),
        "errors": []
    }
    
    print(f"\n🚀 Agent run starting (run_id: {initial_state['run_id'][:8]}...)")
    
    # invoke() runs the graph synchronously to completion
    # For long-running agents, use stream() to get intermediate states
    final_state = compiled_graph.invoke(initial_state)
    
    print(f"✨ Agent run complete. {len(final_state['briefs'])} briefs generated.\n")
    
    if final_state["errors"]:
        print(f"⚠️  Non-fatal errors during run:")
        for err in final_state["errors"]:
            print(f"   - {err}")
    
    return final_state["briefs"]
