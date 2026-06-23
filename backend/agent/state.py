"""
=============================================================================
agent/state.py — The Agent's Memory Blueprint
=============================================================================

WHY WE NEED THIS:
LangGraph is a framework for building stateful agents as a directed graph.
Every node (step) in the graph reads from and writes to a shared "State".

Think of State like the agent's working memory for one full run:
  - It starts with raw calendar events
  - Grows to include extracted company names
  - Eventually contains fully researched briefs

TypedDict enforces types so we catch bugs early (great for production agents).

KEY CONCEPT — Agentic State:
Unlike a normal function that takes input → returns output, an agent
accumulates state across multiple steps. Each node can add/modify fields.
This is what makes multi-step reasoning possible.
=============================================================================
"""

from typing import TypedDict, List, Optional


class MeetingEvent(TypedDict):
    """Raw meeting data from Google Calendar."""
    id: str
    title: str
    start_time: str          # ISO 8601 string e.g. "2025-06-22T10:00:00+05:30"
    end_time: str
    attendees: List[str]     # List of email addresses
    description: str         # Meeting description/body


class CompanySignal(TypedDict):
    """
    After the agent inspects a meeting, it extracts signals to identify the company.
    
    WHY SEPARATE FROM MeetingEvent?
    We want to track HOW we identified the company (from title? email? domain?)
    so we can handle edge cases (personal emails, ambiguous names) gracefully.
    """
    meeting_id: str
    company_name: Optional[str]       # "Linear", "Acme Corp", None if can't infer
    domain: Optional[str]             # "linear.app", "acme.com"
    confidence: str                   # "high", "medium", "low", "unknown"
    inference_source: str             # "attendee_email", "meeting_title", "description", "none"


class MeetingBrief(TypedDict):
    """
    The final output for each meeting — what gets displayed on the dashboard.
    
    WHY STRUCTURED OUTPUT?
    We ask Claude to return JSON matching this schema. Structured outputs let
    us reliably parse, store, and render data — unlike raw prose which breaks
    downstream parsing.
    """
    meeting_id: str
    meeting_title: str
    start_time: str
    end_time: str
    attendees: List[str]
    
    # Company intelligence
    company_name: Optional[str]
    company_overview: Optional[str]    # What they do in plain language
    recent_news: Optional[str]         # Last 60-90 days activity
    tech_signals: Optional[str]        # Stack, tools, infra clues
    pain_points: Optional[str]         # Inferred challenges
    talking_points: List[str]          # 2-3 suggested questions
    
    # Meta
    status: str                        # "researched", "fallback", "error"
    error_message: Optional[str]       # Human-readable reason if status != "researched"
    researched_at: str                 # ISO timestamp of when brief was generated


class AgentState(TypedDict):
    """
    The top-level state object that flows through every node in the graph.
    
    FLOW:
    [] raw_events → [fetch_calendar node fills this]
    [] company_signals → [extract_companies node fills this]
    [] briefs → [research_company node fills this, one per meeting]
    
    LangGraph passes this dict between nodes. Each node receives the full
    state and returns only the fields it wants to update.
    """
    raw_events: List[MeetingEvent]
    company_signals: List[CompanySignal]
    briefs: List[MeetingBrief]
    run_id: str          # Unique ID for this agent run (useful for debugging)
    errors: List[str]    # Non-fatal errors accumulated during the run
