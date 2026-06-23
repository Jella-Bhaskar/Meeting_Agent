"""
=============================================================================
calendar_client.py — Connect to Google Calendar
=============================================================================

WHY WE NEED THIS:
The agent needs to know *what meetings are happening* before it can research
anything. This module wraps the Google Calendar API into clean Python functions.

KEY CONCEPT — External Tool Integration:
Agentic systems are powerful because they can call external tools autonomously.
Here, the agent calls Google Calendar without the user doing anything.
This is what makes it "autonomous" — it monitors and acts on its own.

HOW GOOGLE CALENDAR AUTH WORKS:
1. You create OAuth credentials in Google Cloud Console
2. First run: opens browser, user grants permission, saves token.json
3. Subsequent runs: uses saved token.json (no browser needed)

EXTRACTING COMPANY FROM MEETINGS — the hard part:
  "Demo call with Linear"         → company: Linear (from title)
  "Sync with priya@acme.com"      → company: Acme (from email domain)  
  "Catchup - Ravi" ravi@gmail.com → company: None (personal email)
  "Q3 Review" internal@ourco.com  → company: None (internal meeting)

We return raw signals and let Claude make the final inference (smarter).
=============================================================================
"""

import os
import re
from datetime import datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from agent.state import MeetingEvent

# Scopes define what we're allowed to read from the user's Google account
# read-only is safer and requires less trust from the user
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# These domains mean it's an internal/personal email → skip company inference
PERSONAL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "protonmail.com", "icloud.com", "me.com", "live.com",
    "rediffmail.com", "aol.com",
    "yourcompany.com",   # ← add your own domain here
}


def get_calendar_service():
    """
    Authenticate with Google Calendar API and return a service object.
    
    CONCEPT — OAuth2 Token Flow:
    - credentials.json: your app's identity (client_id, client_secret)
    - token.json: the user's permission token (saved after first login)
    - If token is expired, the library auto-refreshes it using the refresh_token
    
    This pattern is standard for any Google API integration.
    """
    creds = None
    token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    
    # Load existing token if available
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If no valid credentials, run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the token for next run
        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())
    
    return build("calendar", "v3", credentials=creds)


def fetch_upcoming_meetings(days_ahead: int = 7) -> List[MeetingEvent]:
    """
    Fetch all calendar events for the next N days.
    
    WHY days_ahead=7?
    The task says "day or week" — fetching a week gives users more value
    and makes the agent feel proactive rather than reactive.
    
    Returns a list of MeetingEvent dicts (defined in state.py).
    """
    service = get_calendar_service()
    
    # Time range: now → N days from now, in UTC
    now = datetime.utcnow()
    time_min = now.isoformat() + "Z"
    time_max = (now + timedelta(days=days_ahead)).isoformat() + "Z"
    
    # Call Google Calendar API
    # maxResults=50 is a reasonable cap; orderBy=startTime ensures chronological order
    events_result = service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        maxResults=50,
        singleEvents=True,
        orderBy="startTime"
    ).execute()
    
    events = events_result.get("items", [])
    
    meetings = []
    for event in events:
        # Skip all-day events (no specific time = likely not an external call)
        if "dateTime" not in event.get("start", {}):
            continue
        
        # Extract attendee emails
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
    
    WHY NOT JUST DO THIS WITH REGEX?
    We could try regex for domain extraction, but edge cases explode:
    - "sync-with-acme-team" in title
    - Multiple attendees from different companies
    - Acronyms vs company names ("AI sync" vs "AI Corp")
    
    Better approach: extract everything we can observe, pass it to Claude,
    let Claude reason about it. This is the "LLM as brain" pattern.
    
    Returns a dict of raw signals (not a CompanySignal — Claude makes that).
    """
    signals = {
        "meeting_id": event["id"],
        "title": event["title"],
        "attendee_emails": event["attendees"],
        "description_snippet": event["description"][:500] if event["description"] else "",
        "external_domains": [],
        "potential_company_from_title": None,
    }
    
    # Extract non-personal domains from attendee emails
    for email in event["attendees"]:
        if "@" in email:
            domain = email.split("@")[1].lower()
            if domain not in PERSONAL_DOMAINS:
                # e.g. "linear.app" → potential company domain
                signals["external_domains"].append(domain)
    
    # Simple heuristic: look for "with <Company>" or "@ <Company>" in title
    # Claude will do smarter inference, but this helps as a hint
    title_match = re.search(
        r'\b(?:with|@|from|call with|sync with|demo with|meeting with)\s+([A-Z][a-zA-Z0-9]+)',
        event["title"]
    )
    if title_match:
        signals["potential_company_from_title"] = title_match.group(1)
    
    return signals
