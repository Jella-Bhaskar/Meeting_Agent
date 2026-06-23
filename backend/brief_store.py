"""
=============================================================================
brief_store.py — Persistence Layer (Simple JSON File Cache)
=============================================================================

WHY WE NEED THIS:
The agent runs every 30 minutes in the background. The Streamlit dashboard
needs to read briefs WITHOUT triggering a new agent run each time.

Solution: write briefs to a JSON file. Streamlit reads the file.
This decouples the agent (writer) from the dashboard (reader).

CONCEPT — Why JSON file instead of a database?
For a prototype/MVP, JSON is perfect:
  ✓ Zero setup (no DB to install/configure)
  ✓ Human-readable (easy to debug)
  ✓ Works on any deployment (Render, Railway, local)

For production, you'd replace this with:
  - Redis (fast, in-memory, good for ephemeral data)
  - PostgreSQL (if you need queries, history, multiple users)
  - CosmosDB (your Azure stack — JSON-native, scales well)

CONCEPT — Race Condition Safety:
The agent writes, Streamlit reads. What if both happen simultaneously?
We write to a temp file first, then atomically rename it to the final path.
On POSIX systems, rename() is atomic — either old or new, never corrupted.

CONCEPT — Cache Invalidation:
We store `last_updated` timestamp. The dashboard shows "last refreshed at X".
Users know whether they're seeing fresh data or stale cache.
=============================================================================
"""

import json
import os
import tempfile
from datetime import datetime
from typing import List, Optional

from agent.state import MeetingBrief

# Store briefs in a file next to the backend code
BRIEFS_FILE = os.getenv("BRIEFS_FILE_PATH", "briefs_cache.json")


def save_briefs(briefs: List[MeetingBrief]) -> None:
    """
    Persist briefs to JSON file atomically.
    
    Atomic write pattern:
    1. Write to temp file (same directory for atomic rename)
    2. Rename temp → final (atomic on POSIX)
    
    If the process crashes mid-write, the original file is untouched.
    """
    payload = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "brief_count": len(briefs),
        "briefs": briefs
    }
    
    # Write to temp file in same directory (required for atomic rename)
    dir_name = os.path.dirname(os.path.abspath(BRIEFS_FILE))
    
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=dir_name,
        delete=False,
        suffix=".tmp",
        encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp, indent=2, default=str)
        tmp_path = tmp.name
    
    # Atomic rename
    os.replace(tmp_path, BRIEFS_FILE)
    print(f"💾 Saved {len(briefs)} briefs to {BRIEFS_FILE}")


def load_briefs() -> dict:
    """
    Load briefs from JSON file.
    
    Returns dict with keys: last_updated, brief_count, briefs
    Returns empty structure if file doesn't exist yet (first run).
    
    WHY RETURN A DICT NOT JUST THE LIST?
    We want the dashboard to show "last refreshed at X" — so we need
    the metadata (last_updated) alongside the data (briefs).
    """
    if not os.path.exists(BRIEFS_FILE):
        return {
            "last_updated": None,
            "brief_count": 0,
            "briefs": []
        }
    
    try:
        with open(BRIEFS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Could not read briefs file: {e}")
        return {
            "last_updated": None,
            "brief_count": 0,
            "briefs": [],
            "error": str(e)
        }


def get_brief_for_meeting(meeting_id: str) -> Optional[MeetingBrief]:
    """Get a single brief by meeting ID. Useful for the API endpoint."""
    data = load_briefs()
    for brief in data.get("briefs", []):
        if brief["meeting_id"] == meeting_id:
            return brief
    return None


def clear_briefs() -> None:
    """Delete the briefs cache. Used for testing/resetting."""
    if os.path.exists(BRIEFS_FILE):
        os.remove(BRIEFS_FILE)
        print("🗑️  Briefs cache cleared")
