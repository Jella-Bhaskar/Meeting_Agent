"""
=============================================================================
demo_run.py — Run the Agent in DEMO MODE (No Google Calendar needed)
=============================================================================

WHY THIS EXISTS:
Setting up Google OAuth takes 20-30 minutes (Cloud Console, credentials, etc.)
This demo mode lets you run the FULL agent pipeline with mock calendar data
so you can:
  1. Test the research pipeline immediately
  2. See the dashboard with real AI-generated content
  3. Learn how each component works before wiring up real calendar

WHAT THIS DOES:
Creates fake MeetingEvents (the same TypedDict the real calendar returns),
then runs the agent from Node 2 onward (extract_companies + research).
The output is identical to what the real agent produces.

HOW TO RUN:
  cd backend
  python demo_run.py

Then start Streamlit:
  cd frontend && streamlit run app.py
=============================================================================
"""

# Removed unused imports (asyncio, json) to avoid "not accessed" warnings
import os
import sys
import uuid
from datetime import datetime, timedelta

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


def auto_load_service_account_env():
    """If no GOOGLE_APPLICATION_CREDENTIALS is set, try to find a service
    account JSON in the backend directory and set the env var automatically.

    Safety: do not set the env var if the file looks like an OAuth client
    credentials (has 'web' or 'installed' keys) to avoid confusing errors.
    """
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("GOOGLE_APPLICATION_CREDENTIALS already set, skipping auto-load")
        return

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(backend_dir, "service_account.json"),
        os.path.join(backend_dir, "sa.json"),
        os.path.join(backend_dir, "credentials.json"),
        os.path.join(backend_dir, "backend_credentials.json"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            import json as _json
            with open(path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)

            # Service account JSONs have a "type" == "service_account"
            if data.get("type") == "service_account":
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
                print(f"Auto-set GOOGLE_APPLICATION_CREDENTIALS={path}")
                return

            # OAuth client JSONs (for user consent) include 'web' or 'installed' keys — skip those
            if isinstance(data, dict) and ("web" in data or "installed" in data):
                print(f"Found OAuth client JSON at {path}; not using for ADC (skipping)")
                continue

            # Heuristic: if file contains a private_key and client_email, treat as service account
            if "private_key" in data and "client_email" in data:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
                print(f"Auto-set GOOGLE_APPLICATION_CREDENTIALS={path} (heuristic)")
                return

        except Exception:
            continue

    print("No service account JSON found for Application Default Credentials.")


auto_load_service_account_env()

from agent.state import MeetingEvent, AgentState
from agent.nodes import extract_companies_node, research_company_node, store_briefs_node
from brief_store import save_briefs


def create_demo_events() -> list:
    """
    Create realistic fake calendar events for testing.
    
    These cover the 3 scenarios from the task:
    - Scenario A: Clear company name + matching domain email
    - Scenario B: Company inferred from attendee email domain
    - Scenario C: Ambiguous / personal email (graceful fallback)
    - Bonus: Internal meeting (another fallback case)
    """
    now = datetime.now()
    
    def make_time(hours_from_now: int) -> str:
        dt = now + timedelta(hours=hours_from_now)
        return dt.isoformat() + "+05:30"  # IST timezone
    
    events = [
        # Scenario A — Clear company (Linear)
        MeetingEvent(
            id="demo-001",
            title="Demo call with Accenture ",
            start_time=make_time(2),
            end_time=make_time(3),
            attendees=["john@accenture.com", "you@yourcompany.com"],
            description="Product demo and Q&A session with the Nike team"
        ),
        
        # Scenario B — Company inferred from email domain
        MeetingEvent(
            id="demo-002",
            title="Intro call with Priya",
            start_time=make_time(5),
            end_time=make_time(5.5),
            attendees=["priya@accenture.com", "you@yourcompany.com"],
            description=""
        ),
        
        # Additional researched company — Notion
        MeetingEvent(
            id="demo-003",
            title="Partnership discussion - Notion",
            start_time=make_time(24),
            end_time=make_time(25),
            attendees=["alex@notion.com", "sam@notion.com", "you@yourcompany.com"],
            description="Exploring partnership opportunities for enterprise customers"
        ),
        
        # Scenario C — Ambiguous / personal email (graceful fallback)
        MeetingEvent(
            id="demo-004",
            title="Catchup - Ravi",
            start_time=make_time(26),
            end_time=make_time(26.5),
            attendees=["ravi@gmail.com"],
            description=""
        ),
        
        # Internal meeting (another fallback)
        MeetingEvent(
            id="demo-005",
            title="Q3 Planning",
            start_time=make_time(48),
            end_time=make_time(50),
            attendees=["alice@nike.com", "bob@yourcompany.com"],
            description="Internal Q3 roadmap review"
        ),
    ]

    count = int(os.getenv("DEMO_EVENT_COUNT", "1"))
    if count < 1:
        count = 1
        
    return events[:count]


def run_demo():
    """
    Run the agent pipeline with demo data.
    
    CONCEPT — Partial Pipeline:
    We skip Node 1 (fetch_calendar) and inject mock data directly into state.
    Then run Nodes 2-4 normally. This isolates what we're testing.
    
    This is a useful debugging pattern: when a pipeline has 10 steps,
    you can inject state at any step to test just the parts you care about.
    """
    print("🎭 Running in DEMO MODE (no Google Calendar needed)")
    print("=" * 60)
    
    demo_events = create_demo_events()
    print(f"📅 Created {len(demo_events)} demo meetings:")
    for e in demo_events:
        print(f"   - {e['title']} ({e['start_time'][:16]})")
    print()
    
    # Build initial state with demo events pre-loaded
    state: AgentState = {
        "raw_events": demo_events,
        "company_signals": [],
        "briefs": [],
        "run_id": "demo-" + str(uuid.uuid4())[:8],
        "errors": []
    }
    
    # Run Nodes 2, 3, 4 manually (skipping Node 1: fetch_calendar)
    print("Running Node 2: Extract companies...")
    state.update(extract_companies_node(state))
    
    print("\nRunning Node 3: Research companies (this takes 1-3 min)...")
    state.update(research_company_node(state))
    
    print("\nRunning Node 4: Store briefs...")
    state.update(store_briefs_node(state))
    
    # Save to cache (Streamlit reads from here)
    save_briefs(state["briefs"])
    
    print("\n" + "=" * 60)
    print(f"✅ Demo complete! {len(state['briefs'])} briefs saved.")
    print()
    print("To view the dashboard:")
    print("  cd frontend && streamlit run app.py")
    print()
    print("Brief summary:")
    for brief in state["briefs"]:
        status_icon = {"researched": "✅", "fallback": "⚠️", "error": "❌"}.get(brief["status"], "❓")
        print(f"  {status_icon} {brief['meeting_title']} → {brief.get('company_name', 'No company')}")
    
    if state["errors"]:
        print(f"\nNon-fatal errors: {state['errors']}")


if __name__ == "__main__":
    run_demo()