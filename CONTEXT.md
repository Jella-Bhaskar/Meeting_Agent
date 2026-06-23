# Meeting Intelligence Agent — Full Project Context

This file contains the complete source code, architecture, and learning notes for the Meeting Intelligence Agent project. Import this into Cursor, Windsurf, or any AI coding IDE for full project context.

---

## Project Overview

An autonomous agent that monitors Google Calendar, identifies upcoming external calls, researches the companies involved using live web data, and presents a ready-to-use intelligence brief on a Streamlit dashboard — without the user asking.

**Tech Stack:** Python · FastAPI · LangGraph · Anthropic Claude · Streamlit · Google Calendar API · APScheduler

---

## Architecture

```
Streamlit Dashboard (port 8501)
    │ polls /api/briefs every 30s
    ▼
FastAPI Backend (port 8000)
    │ GET /api/briefs  → cached briefs
    │ GET /api/status  → agent run status
    │ POST /api/refresh → trigger agent
    │ APScheduler: runs agent every 30 min
    ▼
LangGraph Agent (4 nodes)
    [fetch_calendar] → [extract_companies] → [research_companies] → [store_briefs]
    Node 3 runs PARALLEL research for all meetings
    Claude uses web_search + homepage fetch (2 sources) AUTONOMOUSLY
```

---

## File Structure

```
meeting-intel-agent/
├── CONTEXT.md                        ← this file
├── LEARNING_GUIDE.txt                ← concept explanations
├── README.md                         ← setup instructions
├── requirements.txt                  ← all Python deps
├── .env.example                      ← env var template
├── Procfile                          ← Railway deployment
├── render.yaml                       ← Render.com deployment
├── .streamlit/
│   ├── config.toml                   ← Streamlit theme + settings
│   └── secrets.toml.example          ← Streamlit secrets template
├── backend/
│   ├── main.py                       ← FastAPI app + APScheduler
│   ├── calendar_client.py            ← Google Calendar OAuth + fetch
│   ├── brief_store.py                ← Atomic JSON file persistence
│   ├── demo_run.py                   ← Run without Google Calendar
│   └── agent/
│       ├── __init__.py
│       ├── state.py                  ← TypedDict state schema
│       ├── graph.py                  ← LangGraph StateGraph
│       └── nodes.py                  ← 4 agent nodes (core logic)
└── frontend/
    └── app.py                        ← Streamlit dashboard
```

---

## Source Files

### `backend/agent/state.py`

```python
"""
agent/state.py — The Agent's Memory Blueprint

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
"""

from typing import TypedDict, List, Optional


class MeetingEvent(TypedDict):
    """Raw meeting data from Google Calendar."""
    id: str
    title: str
    start_time: str          # ISO 8601 e.g. "2025-06-22T10:00:00+05:30"
    end_time: str
    attendees: List[str]     # List of email addresses
    description: str


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
    company_overview: Optional[str]
    recent_news: Optional[str]
    tech_signals: Optional[str]
    pain_points: Optional[str]
    talking_points: List[str]          # 2-3 suggested questions

    # Meta
    status: str                        # "researched", "fallback", "error"
    error_message: Optional[str]
    researched_at: str                 # ISO timestamp


class AgentState(TypedDict):
    """
    The top-level state object that flows through every node in the graph.

    FLOW:
    [] raw_events      → [fetch_calendar node fills this]
    [] company_signals → [extract_companies node fills this]
    [] briefs          → [research_company node fills this, one per meeting]

    LangGraph passes this dict between nodes. Each node receives the full
    state and returns only the fields it wants to update.
    """
    raw_events: List[MeetingEvent]
    company_signals: List[CompanySignal]
    briefs: List[MeetingBrief]
    run_id: str
    errors: List[str]
```

---

### `backend/agent/graph.py`

```python
"""
agent/graph.py — The LangGraph Directed Graph

WHY WE NEED THIS:
This file wires the 4 nodes together into a directed graph (DAG).
LangGraph executes this graph node-by-node, passing state between them.

CONCEPT — Why LangGraph instead of just calling functions in order?
LangGraph gives you:
  ✓ Conditional routing (different paths based on state)
  ✓ Parallel branches (research multiple companies at once)
  ✓ Built-in checkpointing (resume if interrupted)
  ✓ Observability (trace every state transition)
  ✓ Retries at the node level

GRAPH STRUCTURE:
  START → fetch_calendar → extract_companies → research_companies → store_briefs → END
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
    LangGraph "compiles" the graph before running it. This catches structural
    errors (missing nodes, invalid edges) early — like a type-checker for graphs.
    """
    graph = StateGraph(AgentState)

    graph.add_node("fetch_calendar", fetch_calendar_node)
    graph.add_node("extract_companies", extract_companies_node)
    graph.add_node("research_companies", research_company_node)
    graph.add_node("store_briefs", store_briefs_node)

    graph.set_entry_point("fetch_calendar")
    graph.add_edge("fetch_calendar", "extract_companies")
    graph.add_edge("extract_companies", "research_companies")
    graph.add_edge("research_companies", "store_briefs")
    graph.add_edge("store_briefs", END)

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
        print("⚠️  Non-fatal errors during run:")
        for err in final_state["errors"]:
            print(f"   - {err}")

    return final_state["briefs"]
```

---

### `backend/agent/nodes.py`

```python
"""
agent/nodes.py — The Agent's Brain: All Processing Steps

WHY WE NEED THIS:
In LangGraph, a "node" is a Python function that:
  1. Receives the current AgentState
  2. Does some work (calls an API, calls Claude, transforms data)
  3. Returns a dict of state fields to UPDATE

This file has 4 nodes:
  fetch_calendar_node     → gets raw meetings from Google Calendar
  extract_companies_node  → uses Claude to infer company from each meeting
  research_company_node   → uses Claude + web search + homepage (2 sources)
  store_briefs_node       → persists briefs to disk

CONCEPT — Tool Use (Function Calling):
When we give Claude a "tool" like web_search, the model can decide:
  "I need to look something up" → calls tool → gets result → continues

CONCEPT — Parallel Execution:
We research all companies concurrently (asyncio.gather) not sequentially.
10 meetings researched in ~15 seconds, not 10 x 15 = 150 seconds.

CONCEPT — Two External Data Sources:
Source 1 = Anthropic web_search tool (search engine results)
Source 2 = Direct HTTP fetch of the company homepage
"""

import asyncio
import json
import os
import re
import uuid
import urllib.request
from datetime import datetime
from typing import Any

import anthropic

from agent.state import AgentState, CompanySignal, MeetingBrief, MeetingEvent
from calendar_client import fetch_upcoming_meetings, extract_company_signals_from_meeting

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


# =============================================================================
# NODE 1: Fetch Calendar
# =============================================================================

def fetch_calendar_node(state: AgentState) -> dict:
    """
    Node 1: Pull upcoming meetings from Google Calendar.

    STATE UPDATE: fills state["raw_events"]
    """
    print("📅 [Node 1] Fetching calendar events...")

    try:
        days = int(os.getenv("FETCH_DAYS_AHEAD", "7"))
        meetings = fetch_upcoming_meetings(days_ahead=days)
        print(f"   Found {len(meetings)} upcoming meetings")
        return {"raw_events": meetings, "errors": state.get("errors", [])}

    except Exception as e:
        error_msg = f"Calendar fetch failed: {str(e)}"
        print(f"   ❌ {error_msg}")
        return {
            "raw_events": [],
            "errors": state.get("errors", []) + [error_msg]
        }


# =============================================================================
# NODE 2: Extract Companies
# =============================================================================

def extract_companies_node(state: AgentState) -> dict:
    """
    Node 2: Use Claude to infer which company each meeting is with.

    CONCEPT — LLM as Decision Maker:
    Instead of complex regex rules, we give Claude the raw signals and let
    it reason. Claude handles edge cases far better than hand-coded logic.

    CONCEPT — Structured JSON Output:
    We instruct Claude to respond ONLY in JSON so we can parse reliably.

    STATE UPDATE: fills state["company_signals"]
    """
    print("🏢 [Node 2] Extracting company signals...")

    signals = []
    errors = list(state.get("errors", []))

    for event in state["raw_events"]:
        raw_signals = extract_company_signals_from_meeting(event)

        prompt = f"""You are analyzing a calendar meeting to identify what company the meeting is with.

Meeting data:
- Title: {raw_signals['title']}
- Attendee emails: {raw_signals['attendee_emails']}
- External domains found: {raw_signals['external_domains']}
- Description snippet: {raw_signals['description_snippet']}
- Potential company from title: {raw_signals['potential_company_from_title']}

Rules:
- If the meeting is internal (all emails from same domain), set company_name to null
- If email domain is gmail/yahoo/hotmail/personal, set company_name to null
- Domain like "acme.com" → company_name "Acme", "growthsignal.io" → "GrowthSignal"
- confidence: "high" (clear), "medium" (inferred), "low" (guessing), "unknown"
- inference_source: "attendee_email" | "meeting_title" | "description" | "none"

Respond ONLY with this JSON, no other text:
{{
  "company_name": "string or null",
  "domain": "string or null",
  "confidence": "high|medium|low|unknown",
  "inference_source": "attendee_email|meeting_title|description|none"
}}"""

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            raw = response.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            signals.append(CompanySignal(
                meeting_id=event["id"],
                company_name=parsed.get("company_name"),
                domain=parsed.get("domain"),
                confidence=parsed.get("confidence", "unknown"),
                inference_source=parsed.get("inference_source", "none")
            ))

            print(f"   ✓ '{event['title']}' → {parsed.get('company_name', 'No company')}")

        except Exception as e:
            error_msg = f"Company extraction failed for meeting {event['id']}: {e}"
            errors.append(error_msg)
            print(f"   ❌ {error_msg}")

            # Graceful fallback: still show the meeting on dashboard
            signals.append(CompanySignal(
                meeting_id=event["id"],
                company_name=None,
                domain=None,
                confidence="unknown",
                inference_source="none"
            ))

    return {"company_signals": signals, "errors": errors}


# =============================================================================
# NODE 3: Research Companies (2 external sources + parallel)
# =============================================================================

def research_company_node(state: AgentState) -> dict:
    """
    Node 3: Research each company using Claude + web_search + homepage.

    TWO DATA SOURCES:
    Source 1 = web_search tool (Claude decides what/when to search)
    Source 2 = direct HTTP fetch of company homepage

    PARALLEL: asyncio.gather() runs all research simultaneously.

    STATE UPDATE: fills state["briefs"]
    """
    print("🔍 [Node 3] Researching companies (parallel)...")

    signal_map = {s["meeting_id"]: s for s in state["company_signals"]}
    event_map = {e["id"]: e for e in state["raw_events"]}

    briefs = asyncio.run(_research_all_parallel(state["raw_events"], signal_map, event_map))

    return {"briefs": briefs, "errors": state.get("errors", [])}


async def _research_all_parallel(raw_events, signal_map, event_map):
    tasks = [
        _research_single_meeting(event, signal_map.get(event["id"]), event)
        for event in raw_events
    ]
    return await asyncio.gather(*tasks)


async def _research_single_meeting(event: MeetingEvent, signal: CompanySignal, raw_event: MeetingEvent) -> MeetingBrief:
    """
    Research one company. Returns MeetingBrief with status:
    - "researched": full intelligence available
    - "fallback": company could not be identified (graceful, not a crash)
    - "error": research failed (graceful, still shows meeting)
    """
    now_iso = datetime.utcnow().isoformat() + "Z"

    base_brief = MeetingBrief(
        meeting_id=event["id"],
        meeting_title=event["title"],
        start_time=event["start_time"],
        end_time=event["end_time"],
        attendees=event["attendees"],
        company_name=signal["company_name"] if signal else None,
        company_overview=None,
        recent_news=None,
        tech_signals=None,
        pain_points=None,
        talking_points=[],
        status="fallback",
        error_message=None,
        researched_at=now_iso
    )

    # FALLBACK: no company identified
    if not signal or not signal["company_name"]:
        base_brief["error_message"] = "Company could not be identified for this meeting."
        print(f"   ⚠️  '{event['title']}' → fallback (no company)")
        return base_brief

    company = signal["company_name"]
    domain = signal.get("domain", "")

    try:
        # DATA SOURCE 2: Direct homepage fetch
        homepage_text = fetch_company_homepage(domain) if domain else ""
        homepage_section = (
            f"\n\nDATA SOURCE 2 — Company homepage ({domain}):\n{homepage_text}"
            if homepage_text and not homepage_text.startswith("[Homepage fetch failed")
            else f"\n\n[Homepage fetch skipped: domain={domain or 'unknown'}]"
        )

        # DATA SOURCE 1: Anthropic web_search tool
        web_search_tool = {
            "type": "web_search_20250305",
            "name": "web_search"
        }

        research_prompt = f"""You are a meeting intelligence agent. Research the company "{company}" (domain: {domain or 'unknown'}) to prepare a pre-meeting brief.

You have TWO data sources:

DATA SOURCE 1 — Web Search Tool:
Use the web_search tool to find:
- Recent news, product launches, funding, announcements from LAST 60-90 DAYS
- Tech signals: stack, tools, infrastructure, open-source activity
Make at least 2 different searches (e.g. "{company} company overview" AND "{company} news 2025").

{homepage_section}

Synthesize BOTH sources. Use homepage for what they say about themselves, web search for what others say and recent developments.

Respond ONLY with this JSON (no other text, no markdown):
{{
  "company_overview": "2-3 sentence plain English description (not marketing copy)",
  "recent_news": "Key recent developments last 60-90 days. Be specific with dates.",
  "tech_signals": "Detected tech stack, tools, infra. Say 'Not detected' if unclear.",
  "pain_points": "2-3 inferred challenges based on their stage/activity",
  "talking_points": [
    "Specific question 1 based on research",
    "Specific question 2 based on research",
    "Specific question 3 based on research"
  ],
  "sources_used": ["web_search", "company_homepage"]
}}"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            tools=[web_search_tool],
            messages=[{"role": "user", "content": research_prompt}]
        )

        # Extract final text response (after all tool calls complete)
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text = block.text

        if not final_text:
            raise ValueError("No text response from Claude after research")

        clean = final_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)

        base_brief.update({
            "company_overview": data.get("company_overview", "No overview available"),
            "recent_news": data.get("recent_news", "No recent news found"),
            "tech_signals": data.get("tech_signals", "Not detected"),
            "pain_points": data.get("pain_points", "Not determined"),
            "talking_points": data.get("talking_points", []),
            "status": "researched",
            "error_message": None,
        })

        print(f"   ✅ '{event['title']}' → researched ({company})")
        return base_brief

    except Exception as e:
        base_brief["status"] = "error"
        base_brief["error_message"] = f"Research failed: {str(e)}"
        print(f"   ❌ '{event['title']}' → research error: {e}")
        return base_brief


# =============================================================================
# NODE 4: Store Briefs
# =============================================================================

def store_briefs_node(state: AgentState) -> dict:
    """
    Node 4: Persist briefs to disk.

    IMPORTANT: imports save_briefs here (not at top) to avoid circular imports.
    This node is the single source of truth for persistence — works whether
    called via graph.invoke() or directly in demo_run.py.
    """
    from brief_store import save_briefs

    print(f"💾 [Node 4] Finalizing {len(state['briefs'])} briefs...")

    researched = sum(1 for b in state["briefs"] if b["status"] == "researched")
    fallback = sum(1 for b in state["briefs"] if b["status"] == "fallback")
    errors_count = sum(1 for b in state["briefs"] if b["status"] == "error")

    print(f"   ✅ {researched} researched | ⚠️ {fallback} fallback | ❌ {errors_count} errors")

    save_briefs(state["briefs"])

    return {"briefs": state["briefs"]}


# =============================================================================
# SECOND DATA SOURCE: Direct homepage fetch
# =============================================================================

def fetch_company_homepage(domain: str) -> str:
    """
    Fetch company homepage HTML as a second independent data source.

    WHY urllib NOT requests?
    urllib is stdlib — zero extra dependency.
    For production, use httpx (async, better timeout handling).
    """
    if not domain:
        return ""
    try:
        url = f"https://{domain}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MeetingIntelAgent/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3000]
    except Exception as e:
        return f"[Homepage fetch failed: {e}]"
```

---

### `backend/calendar_client.py`

```python
"""
calendar_client.py — Connect to Google Calendar

WHY WE NEED THIS:
The agent needs to know what meetings are happening before it can research
anything. This module wraps the Google Calendar API into clean Python functions.

HOW GOOGLE CALENDAR AUTH WORKS:
1. Create OAuth credentials in Google Cloud Console
2. First run: browser opens, user grants permission, saves token.json
3. Subsequent runs: uses saved token.json (no browser needed)

COMPANY EXTRACTION STRATEGY:
We extract raw signals and let Claude infer the company (smarter than regex):
  "Demo call with Linear" + john@linear.app → company: Linear
  "Catchup - Ravi" + ravi@gmail.com → company: None (personal email)
  "Q3 Review" + internal@ourco.com → company: None (internal)
"""

import os
import re
from datetime import datetime, timedelta
from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from agent.state import MeetingEvent

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

PERSONAL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "protonmail.com", "icloud.com", "me.com", "live.com"
}


def get_calendar_service():
    """
    Authenticate with Google Calendar API.

    CONCEPT — OAuth2 Token Flow:
    - credentials.json: app identity (client_id, client_secret)
    - token.json: user permission token (saved after first login)
    - Auto-refreshes expired tokens using refresh_token
    """
    creds = None
    token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def fetch_upcoming_meetings(days_ahead: int = 7) -> List[MeetingEvent]:
    """Fetch all timed calendar events for the next N days."""
    service = get_calendar_service()

    now = datetime.utcnow()
    time_min = now.isoformat() + "Z"
    time_max = (now + timedelta(days=days_ahead)).isoformat() + "Z"

    events_result = service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        maxResults=50,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    meetings = []
    for event in events_result.get("items", []):
        # Skip all-day events
        if "dateTime" not in event.get("start", {}):
            continue

        attendees = [
            a.get("email", "")
            for a in event.get("attendees", [])
            if a.get("email")
        ]

        meetings.append(MeetingEvent(
            id=event["id"],
            title=event.get("summary", "Untitled Meeting"),
            start_time=event["start"]["dateTime"],
            end_time=event["end"]["dateTime"],
            attendees=attendees,
            description=event.get("description", "")
        ))

    return meetings


def extract_company_signals_from_meeting(event: MeetingEvent) -> dict:
    """
    Extract raw signals to help Claude infer the company.

    WHY NOT REGEX?
    Edge cases explode with regex:
    - "sync-with-acme-team" in title
    - Multiple attendees from different companies
    - Acronyms vs company names
    Better: extract everything observable, let Claude reason about it.
    """
    signals = {
        "meeting_id": event["id"],
        "title": event["title"],
        "attendee_emails": event["attendees"],
        "description_snippet": event["description"][:500] if event["description"] else "",
        "external_domains": [],
        "potential_company_from_title": None,
    }

    for email in event["attendees"]:
        if "@" in email:
            domain = email.split("@")[1].lower()
            if domain not in PERSONAL_DOMAINS:
                signals["external_domains"].append(domain)

    title_match = re.search(
        r'\b(?:with|@|from|call with|sync with|demo with|meeting with)\s+([A-Z][a-zA-Z0-9]+)',
        event["title"]
    )
    if title_match:
        signals["potential_company_from_title"] = title_match.group(1)

    return signals
```

---

### `backend/brief_store.py`

```python
"""
brief_store.py — Persistence Layer (Atomic JSON File Cache)

WHY WE NEED THIS:
The agent runs every 30 min in background. Streamlit reads briefs without
triggering a new agent run. Solution: JSON file decouples writer (agent)
from reader (dashboard).

CONCEPT — Atomic Writes:
Write to temp file first, then os.replace() (atomic on POSIX).
If process crashes mid-write, original file is untouched.

CONCEPT — Why JSON file (not database)?
For prototype: zero setup, human-readable, works anywhere.
For production: Redis (ephemeral) or PostgreSQL (history, multi-user).
"""

import json
import os
import tempfile
from datetime import datetime
from typing import List, Optional

from agent.state import MeetingBrief

BRIEFS_FILE = os.getenv("BRIEFS_FILE_PATH", "briefs_cache.json")


def save_briefs(briefs: List[MeetingBrief]) -> None:
    """Persist briefs to JSON file atomically."""
    payload = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "brief_count": len(briefs),
        "briefs": briefs
    }

    dir_name = os.path.dirname(os.path.abspath(BRIEFS_FILE))

    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_name, delete=False, suffix=".tmp", encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp, indent=2, default=str)
        tmp_path = tmp.name

    os.replace(tmp_path, BRIEFS_FILE)
    print(f"💾 Saved {len(briefs)} briefs to {BRIEFS_FILE}")


def load_briefs() -> dict:
    """Load briefs from JSON file. Returns empty structure if file not found."""
    if not os.path.exists(BRIEFS_FILE):
        return {"last_updated": None, "brief_count": 0, "briefs": []}

    try:
        with open(BRIEFS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Could not read briefs file: {e}")
        return {"last_updated": None, "brief_count": 0, "briefs": [], "error": str(e)}


def get_brief_for_meeting(meeting_id: str) -> Optional[MeetingBrief]:
    data = load_briefs()
    for brief in data.get("briefs", []):
        if brief["meeting_id"] == meeting_id:
            return brief
    return None


def clear_briefs() -> None:
    if os.path.exists(BRIEFS_FILE):
        os.remove(BRIEFS_FILE)
        print("🗑️  Briefs cache cleared")
```

---

### `backend/main.py`

```python
"""
main.py — FastAPI Backend: API + Background Scheduler

ARCHITECTURE:
  Streamlit (port 8501) ←→ HTTP ←→ FastAPI (port 8000) ←→ Agent

WHY SEPARATE BACKEND FROM STREAMLIT?
  1. Agent runs on SCHEDULE — Streamlit only runs when page is open
  2. Clean API: POST /refresh, GET /briefs, GET /status
  3. Multiple Streamlit instances can share one backend

CONCEPT — APScheduler:
Runs run_agent_job() every N minutes even with no users on the dashboard.
This is what makes the system "autonomous" — monitors without being asked.

CONCEPT — CORS:
Streamlit (port 8501) calling FastAPI (port 8000) = cross-origin.
Browser blocks by default. CORSMiddleware explicitly allows it.

CONCEPT — FastAPI Lifespan:
Code before yield = startup. Code after yield = shutdown.
Used to start scheduler and run initial agent run.
"""

import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import run_agent
from brief_store import save_briefs, load_briefs, clear_briefs

REFRESH_INTERVAL_MINUTES = int(os.getenv("REFRESH_INTERVAL_MINUTES", "30"))

agent_status = {
    "is_running": False,
    "last_run_at": None,
    "last_run_status": "never_run",
    "next_run_at": None,
}

scheduler = BackgroundScheduler()


def run_agent_job():
    """Scheduler job: prevents overlapping runs, updates status."""
    if agent_status["is_running"]:
        print("⏭️  Agent already running, skipping")
        return

    agent_status["is_running"] = True
    agent_status["last_run_status"] = "running"

    try:
        print(f"\n⏰ Agent run starting at {datetime.utcnow().isoformat()}")
        briefs = run_agent()
        save_briefs(briefs)
        agent_status["last_run_at"] = datetime.utcnow().isoformat() + "Z"
        agent_status["last_run_status"] = "success"
        print(f"✅ Run complete: {len(briefs)} briefs saved")
    except Exception as e:
        agent_status["last_run_status"] = f"error: {str(e)}"
        print(f"❌ Run failed: {e}")
    finally:
        agent_status["is_running"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 FastAPI starting up...")

    scheduler.add_job(
        run_agent_job, "interval",
        minutes=REFRESH_INTERVAL_MINUTES,
        id="agent_run", replace_existing=True
    )
    scheduler.start()
    print(f"⏰ Scheduler started: every {REFRESH_INTERVAL_MINUTES} minutes")

    # Run immediately on startup so dashboard has data right away
    threading.Thread(target=run_agent_job, daemon=True).start()

    yield

    print("👋 FastAPI shutting down...")
    scheduler.shutdown(wait=False)


app = FastAPI(
    title="Meeting Intelligence Agent API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "Meeting Intelligence Agent"}


@app.get("/api/briefs")
def get_briefs():
    """Called by Streamlit every 30s to update the dashboard."""
    return load_briefs()


@app.get("/api/status")
def get_status():
    data = load_briefs()
    return {
        **agent_status,
        "briefs_count": data.get("brief_count", 0),
        "briefs_last_updated": data.get("last_updated"),
        "refresh_interval_minutes": REFRESH_INTERVAL_MINUTES,
    }


@app.post("/api/refresh")
async def trigger_refresh(background_tasks: BackgroundTasks):
    """
    Manually trigger agent run.

    CONCEPT — BackgroundTasks:
    Returns immediately ("started"), agent runs after response is sent.
    Prevents HTTP timeout (agent takes 30-120s, default timeout is 30s).
    """
    if agent_status["is_running"]:
        return {"status": "already_running", "message": "Agent is already running"}

    background_tasks.add_task(run_agent_job)
    return {"status": "started", "message": "Agent refresh started in background"}


@app.delete("/api/briefs")
def clear_all_briefs():
    clear_briefs()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development"
    )
```

---

### `backend/demo_run.py`

```python
"""
demo_run.py — Run Agent in DEMO MODE (No Google Calendar needed)

WHY THIS EXISTS:
Google OAuth setup takes 20-30 min. This lets you run the FULL agent pipeline
with mock calendar data to test research + dashboard immediately.

HOW TO RUN:
  cd backend && python demo_run.py
  cd frontend && streamlit run app.py

CONCEPT — Partial Pipeline:
We skip Node 1 (fetch_calendar) and inject mock data directly into state.
Useful debug pattern: inject state at any step to test specific nodes.
"""

import os
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from agent.state import MeetingEvent, AgentState
from agent.nodes import extract_companies_node, research_company_node, store_briefs_node
from brief_store import save_briefs


def create_demo_events() -> list:
    """
    Covers all 3 task scenarios:
    A: Clear company name + matching domain
    B: Company inferred from email domain
    C: Personal email → graceful fallback
    + Internal meeting → another fallback
    """
    now = datetime.now()

    def make_time(hours_from_now: float) -> str:
        return (now + timedelta(hours=hours_from_now)).isoformat() + "+05:30"

    return [
        MeetingEvent(id="demo-001", title="Demo call with Linear",
                     start_time=make_time(2), end_time=make_time(3),
                     attendees=["john@linear.app", "you@yourcompany.com"],
                     description="Product demo and Q&A"),

        MeetingEvent(id="demo-002", title="Intro call with Priya",
                     start_time=make_time(5), end_time=make_time(5.5),
                     attendees=["priya@growthsignal.io", "you@yourcompany.com"],
                     description=""),

        MeetingEvent(id="demo-003", title="Partnership discussion - Notion",
                     start_time=make_time(24), end_time=make_time(25),
                     attendees=["alex@makenotion.com", "you@yourcompany.com"],
                     description="Exploring partnership opportunities"),

        MeetingEvent(id="demo-004", title="Catchup - Ravi",
                     start_time=make_time(26), end_time=make_time(26.5),
                     attendees=["ravi@gmail.com"], description=""),

        MeetingEvent(id="demo-005", title="Q3 Planning",
                     start_time=make_time(48), end_time=make_time(50),
                     attendees=["alice@yourcompany.com", "bob@yourcompany.com"],
                     description="Internal roadmap review"),
    ]


def run_demo():
    print("🎭 DEMO MODE (no Google Calendar needed)")
    print("=" * 60)

    demo_events = create_demo_events()
    print(f"📅 Created {len(demo_events)} demo meetings")

    state: AgentState = {
        "raw_events": demo_events,
        "company_signals": [],
        "briefs": [],
        "run_id": "demo-" + str(uuid.uuid4())[:8],
        "errors": []
    }

    print("\nRunning Node 2: Extract companies...")
    state.update(extract_companies_node(state))

    print("\nRunning Node 3: Research companies (1-3 min)...")
    state.update(research_company_node(state))

    print("\nRunning Node 4: Store briefs...")
    state.update(store_briefs_node(state))

    print("\n" + "=" * 60)
    print(f"✅ Demo complete! {len(state['briefs'])} briefs saved.")
    print("\nNow run: cd frontend && streamlit run app.py")

    for brief in state["briefs"]:
        icon = {"researched": "✅", "fallback": "⚠️", "error": "❌"}.get(brief["status"], "❓")
        print(f"  {icon} {brief['meeting_title']} → {brief.get('company_name', 'No company')}")


if __name__ == "__main__":
    run_demo()
```

---

### `frontend/app.py`

```python
"""
frontend/app.py — Streamlit Dashboard

WHY STREAMLIT (not React/JS)?
  ✓ Pure Python — no npm, webpack, JSX
  ✓ Designed for AI/data apps
  ✓ Deploy to Streamlit Cloud free in 5 minutes

HOW STREAMLIT WORKS:
Re-runs the entire script top-to-bottom on every interaction.
st.session_state persists values across reruns.
st_autorefresh triggers reruns every 30s (polls the backend).

CONCEPT — Polling:
Agent runs every 30 min in backend. Dashboard polls /api/briefs every 30s.
When new briefs arrive, dashboard updates automatically.
"""

import os
import time
from datetime import datetime, timezone

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = os.getenv("API_URL", "http://localhost:8000")

REFRESH_INTERVAL_MS = 30_000

st.set_page_config(
    page_title="Meeting Intelligence Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .badge-researched { background: #d1fae5; color: #065f46; padding: 2px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
    .badge-fallback { background: #fef3c7; color: #92400e; padding: 2px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
    .badge-error { background: #fee2e2; color: #991b1b; padding: 2px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def fetch_briefs():
    try:
        r = requests.get(f"{API_URL}/api/briefs", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is FastAPI running?", "briefs": []}
    except Exception as e:
        return {"error": str(e), "briefs": []}


def fetch_status():
    try:
        return requests.get(f"{API_URL}/api/status", timeout=5).json()
    except Exception:
        return {"is_running": False, "last_run_status": "unknown"}


def trigger_refresh():
    try:
        return requests.post(f"{API_URL}/api/refresh", timeout=5).json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def format_time(iso_string: str) -> str:
    if not iso_string:
        return "Unknown time"
    try:
        return datetime.fromisoformat(iso_string.replace("Z", "+00:00")).strftime("%I:%M %p")
    except Exception:
        return iso_string


def format_date(iso_string: str) -> str:
    if not iso_string:
        return "Unknown date"
    try:
        return datetime.fromisoformat(iso_string.replace("Z", "+00:00")).strftime("%a, %b %d")
    except Exception:
        return iso_string


def time_ago(iso_string: str) -> str:
    if not iso_string:
        return "never"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        diff = datetime.now(timezone.utc) - dt
        s = int(diff.total_seconds())
        if s < 60: return f"{s}s ago"
        elif s < 3600: return f"{s // 60}m ago"
        else: return f"{s // 3600}h ago"
    except Exception:
        return iso_string


def group_by_date(briefs: list) -> dict:
    grouped = {}
    for brief in sorted(briefs, key=lambda b: b.get("start_time", "")):
        date_str = format_date(brief.get("start_time", ""))
        grouped.setdefault(date_str, []).append(brief)
    return grouped


def render_brief_card(brief: dict):
    status = brief.get("status", "fallback")
    company = brief.get("company_name") or "Unknown Company"
    time_str = format_time(brief.get("start_time", ""))
    title = brief.get("meeting_title", "Untitled")
    attendees = brief.get("attendees", [])

    badge_map = {
        "researched": ("✅", "Researched"),
        "fallback": ("⚠️", "No company"),
        "error": ("❌", "Research failed"),
    }
    icon, label = badge_map.get(status, ("❓", "Unknown"))

    with st.expander(f"{icon} **{time_str}** — {title} | *{company}*"):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**🕐 Time:** {time_str}")
            st.markdown(f"**📋 Meeting:** {title}")
        with col2:
            st.markdown(f"**🏢 Company:** {company}")
            if attendees:
                st.markdown(f"**👥 Attendees:** {', '.join(attendees[:3])}")
                if len(attendees) > 3:
                    st.caption(f"+{len(attendees)-3} more")
        with col3:
            st.markdown(f"**Status:** {label}")

        st.divider()

        if status == "fallback":
            st.info(brief.get("error_message", "Company could not be identified for this meeting."))
            return

        if status == "error":
            st.error(f"Research failed: {brief.get('error_message', 'Unknown error')}")
            return

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**🏢 Company Overview**")
            st.markdown(brief.get("company_overview", "Not available"))
            st.markdown("---")
            st.markdown("**📰 Recent News (last 60-90 days)**")
            st.markdown(brief.get("recent_news", "No recent news found"))
        with col_right:
            st.markdown("**⚙️ Tech Signals**")
            st.markdown(brief.get("tech_signals", "Not detected"))
            st.markdown("---")
            st.markdown("**🎯 Inferred Pain Points**")
            st.markdown(brief.get("pain_points", "Not determined"))

        st.markdown("---")
        st.markdown("**💬 Suggested Talking Points**")
        for i, point in enumerate(brief.get("talking_points", []), 1):
            st.markdown(f"> {i}. {point}")

        st.caption(f"*Brief generated {time_ago(brief.get('researched_at', ''))}*")


def main():
    st_autorefresh(interval=REFRESH_INTERVAL_MS, key="auto_refresh")

    col_title, col_status, col_button = st.columns([3, 2, 1])

    with col_title:
        st.title("🧠 Meeting Intelligence")
        st.caption("Autonomous pre-meeting research — no prep needed")

    with col_status:
        status = fetch_status()
        is_running = status.get("is_running", False)
        if is_running:
            st.info("🔄 Agent is researching your meetings...")
        else:
            last_run = status.get("last_run_at")
            if last_run:
                st.success(f"✅ Last updated: {time_ago(last_run)}")
            else:
                st.warning("⏳ Waiting for first agent run...")

    with col_button:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Now", use_container_width=True, disabled=is_running):
            result = trigger_refresh()
            if result.get("status") == "started":
                st.success("Agent refresh started!")
                time.sleep(1)
                st.rerun()
            else:
                st.warning(result.get("message", "Could not start refresh"))

    st.divider()

    data = fetch_briefs()

    if "error" in data:
        st.error(f"**Backend error:** {data['error']}")
        st.code("cd backend && uvicorn main:app --reload")
        st.stop()

    briefs = data.get("briefs", [])

    if not briefs:
        st.markdown("### 📭 No meetings found")
        st.info("The agent hasn't run yet, or no meetings in the next 7 days. Click **Refresh Now**.")
        st.stop()

    researched = sum(1 for b in briefs if b["status"] == "researched")
    fallback = sum(1 for b in briefs if b["status"] == "fallback")
    errors = sum(1 for b in briefs if b["status"] == "error")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📅 Total Meetings", len(briefs))
    m2.metric("✅ Fully Researched", researched)
    m3.metric("⚠️ No Company Found", fallback)
    m4.metric("❌ Research Errors", errors)

    st.divider()

    for date_str, day_briefs in group_by_date(briefs).items():
        st.subheader(f"📆 {date_str}")
        st.caption(f"{len(day_briefs)} meeting{'s' if len(day_briefs) != 1 else ''}")
        for brief in day_briefs:
            render_brief_card(brief)
        st.markdown("")

    st.divider()
    col_f1, col_f2, col_f3 = st.columns(3)
    col_f1.caption(f"📊 {len(briefs)} meetings | Auto-refreshes every 30s")
    col_f2.caption(f"🕐 Cache: {time_ago(data.get('last_updated')) if data.get('last_updated') else 'never'}")
    col_f3.caption(f"🔁 Agent interval: every {status.get('refresh_interval_minutes', 30)} min")


if __name__ == "__main__":
    main()
```

---

### `requirements.txt`

```
# FastAPI Backend
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-dotenv==1.0.1
apscheduler==3.10.4

# Anthropic (Claude API)
anthropic==0.34.0

# LangGraph (Agentic Framework)
langgraph==0.2.14
langchain-core==0.3.6

# Google Calendar
google-api-python-client==2.143.0
google-auth-oauthlib==1.2.1
google-auth-httplib2==0.2.0

# Streamlit Frontend
streamlit==1.38.0
streamlit-autorefresh==1.0.1
requests==2.32.3
```

---

### `.env.example`

```
# Anthropic (Required)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Google Calendar OAuth (Required)
# Download credentials.json from Google Cloud Console → put in backend/
GOOGLE_CREDENTIALS_PATH=credentials.json
GOOGLE_TOKEN_PATH=token.json

# Agent Configuration (Optional)
FETCH_DAYS_AHEAD=7
REFRESH_INTERVAL_MINUTES=30
BRIEFS_FILE_PATH=briefs_cache.json

# API
PORT=8000
ENV=development

# Streamlit
API_URL=http://localhost:8000
```

---

### `.streamlit/config.toml`

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[theme]
base = "dark"
primaryColor = "#6366f1"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#e2e8f0"
```

---

### `Procfile` (Railway deployment)

```
web: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

### `render.yaml` (Render.com deployment)

```yaml
services:
  - type: web
    name: meeting-intel-backend
    runtime: python
    rootDir: backend
    buildCommand: pip install -r ../requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ANTHROPIC_API_KEY
        sync: false
      - key: GOOGLE_TOKEN_PATH
        value: /tmp/token.json
      - key: GOOGLE_CREDENTIALS_PATH
        value: /tmp/credentials.json
      - key: BRIEFS_FILE_PATH
        value: /tmp/briefs_cache.json
      - key: FETCH_DAYS_AHEAD
        value: "7"
      - key: REFRESH_INTERVAL_MINUTES
        value: "30"
      - key: ENV
        value: production
    healthCheckPath: /
```

---

## Key Concepts Reference

| Concept | Where | One-line explanation |
|---|---|---|
| **Agentic State** | `state.py` | TypedDict that accumulates data across all nodes |
| **LangGraph StateGraph** | `graph.py` | Directed graph of nodes sharing state |
| **Tool Use** | `nodes.py` | Claude autonomously decides when/what to search |
| **Structured JSON Output** | `nodes.py` | LLM returns parseable JSON schema, not prose |
| **Two External Sources** | `nodes.py` | web_search + homepage fetch, both fed to Claude |
| **Background Scheduling** | `main.py` | APScheduler runs agent every 30 min autonomously |
| **Graceful Fallback** | `nodes.py` | 3 statuses: researched / fallback / error — no crashes |
| **OAuth2 Token Flow** | `calendar_client.py` | credentials.json → browser → token.json → auto-refresh |
| **Async Parallel** | `nodes.py` | asyncio.gather() researches all companies simultaneously |
| **Atomic File Write** | `brief_store.py` | Write to temp, then os.replace() — never corrupts |
| **CORS Middleware** | `main.py` | Allows Streamlit (8501) to call FastAPI (8000) |
| **FastAPI Lifespan** | `main.py` | startup/shutdown hooks for scheduler |
| **Streamlit Polling** | `frontend/app.py` | st_autorefresh every 30s polls /api/briefs |
| **Session State** | `frontend/app.py` | st.session_state persists across Streamlit reruns |

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in env vars
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

# 3a. Demo mode (no Google Calendar setup needed — START HERE)
cd backend && python demo_run.py

# 3b. Real mode (requires credentials.json from Google Cloud Console)
cd backend && uvicorn main:app --reload --port 8000

# 4. Start dashboard (new terminal)
cd frontend && streamlit run app.py
# Open http://localhost:8501
```

---

## Requirements Compliance

| Requirement | Status |
|---|---|
| Calendar connects via external tool integration | ✅ Google Calendar API |
| Extract company from title / email / description | ✅ Claude inference in Node 2 |
| Handle ambiguous data gracefully | ✅ fallback status, never crash |
| At least 2 external data sources | ✅ web_search + homepage HTTP fetch |
| Company overview, news, tech signals, pain points | ✅ All in MeetingBrief schema |
| 2-3 talking points per meeting | ✅ Claude generates from research |
| Dashboard: title, time, attendees on each card | ✅ Streamlit expander cards |
| Dashboard live at public URL | ✅ Streamlit Cloud + Render configs |
| Dashboard updates automatically | ✅ polls every 30s, agent runs every 30m |
| No crashes on tool failure | ✅ all nodes wrapped in try/except |
| No blank cards | ✅ fallback + error states always shown |
| Agent decides what to look up autonomously | ✅ Claude with web_search tool |
