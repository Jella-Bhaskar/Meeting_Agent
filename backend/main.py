"""
=============================================================================
main.py — FastAPI Backend: API + Background Scheduler
=============================================================================

WHY WE NEED FASTAPI (even though we're using Streamlit):
Streamlit is the frontend. FastAPI is the backend API.

ARCHITECTURE:
  Streamlit (port 8501) ←→ HTTP requests ←→ FastAPI (port 8000) ←→ Agent

WHY SEPARATE BACKEND?
  1. The agent runs on a SCHEDULE (every 30 min), not when someone opens Streamlit
  2. FastAPI gives us a clean API: POST /refresh, GET /briefs, GET /status
  3. Streamlit just reads data; it doesn't run the agent itself
  4. In production, multiple Streamlit instances could share one FastAPI backend

CONCEPT — Background Task Scheduling:
APScheduler (Advanced Python Scheduler) runs the agent every N minutes
even when no one is using the dashboard. This is what makes the agent
"autonomous" — it monitors and prepares without being asked.

CONCEPT — CORS (Cross-Origin Resource Sharing):
Streamlit runs on port 8501, FastAPI on port 8000.
Browsers block cross-origin requests by default (security).
We add CORS middleware to explicitly allow Streamlit to call our API.

CONCEPT — Lifespan (startup/shutdown):
FastAPI's lifespan context manager runs code on startup and shutdown.
We use it to:
  - Run the agent once immediately on startup (so dashboard isn't empty)
  - Start the scheduler for subsequent runs
  - Cleanly shut down the scheduler on exit
=============================================================================
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import run_agent
from brief_store import save_briefs, load_briefs, clear_briefs

# How often the agent runs automatically (default: every 30 minutes)
REFRESH_INTERVAL_MINUTES = int(os.getenv("REFRESH_INTERVAL_MINUTES", "30"))

# Track agent run status for the dashboard
agent_status = {
    "is_running": False,
    "last_run_at": None,
    "last_run_status": "never_run",
    "next_run_at": None,
}

# Global scheduler instance
scheduler = BackgroundScheduler()


def run_agent_job():
    """
    The job function called by the scheduler.
    
    WHY WRAP run_agent() IN THIS FUNCTION?
    - Handles the is_running flag (prevent overlapping runs)
    - Catches all exceptions (scheduler swallows errors silently otherwise)
    - Updates agent_status for the dashboard to display
    """
    if agent_status["is_running"]:
        print("⏭️  Agent already running, skipping this scheduled run")
        return
    
    agent_status["is_running"] = True
    agent_status["last_run_status"] = "running"
    
    try:
        print(f"\n⏰ Scheduled agent run starting at {datetime.utcnow().isoformat()}")
        briefs = run_agent()
        save_briefs(briefs)
        
        agent_status["last_run_at"] = datetime.utcnow().isoformat() + "Z"
        agent_status["last_run_status"] = "success"
        print(f"✅ Scheduled run complete: {len(briefs)} briefs saved")
    
    except Exception as e:
        agent_status["last_run_status"] = f"error: {str(e)}"
        print(f"❌ Scheduled run failed: {e}")
    
    finally:
        agent_status["is_running"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    CONCEPT — Lifespan Events:
    FastAPI's lifespan replaces the old @app.on_event("startup") pattern.
    Code before `yield` runs on startup; code after `yield` runs on shutdown.
    
    We use it to start the scheduler and do an initial agent run.
    """
    # STARTUP
    print("🚀 FastAPI starting up...")
    
    # Schedule the agent to run every N minutes
    scheduler.add_job(
        run_agent_job,
        "interval",
        minutes=REFRESH_INTERVAL_MINUTES,
        id="agent_run",
        replace_existing=True
    )
    scheduler.start()
    print(f"⏰ Scheduler started: agent runs every {REFRESH_INTERVAL_MINUTES} minutes")
    
    # Run once immediately so the dashboard has data right away
    # Run in background so API is ready while agent works
    import threading
    threading.Thread(target=run_agent_job, daemon=True).start()
    
    yield  # FastAPI serves requests here
    
    # SHUTDOWN
    print("👋 FastAPI shutting down...")
    scheduler.shutdown(wait=False)


# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Meeting Intelligence Agent API",
    description="Autonomous agent that researches companies before your meetings",
    version="1.0.0",
    lifespan=lifespan
)

# Allow Streamlit (different port) to call this API
# In production, restrict origins to your actual domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production: ["https://your-streamlit-app.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    """Health check endpoint. Used by deployment platforms to verify the app is running."""
    return {"status": "ok", "service": "Meeting Intelligence Agent"}


@app.get("/api/briefs")
def get_briefs():
    """
    Get all current meeting briefs.
    Called by Streamlit every ~30 seconds to update the dashboard.
    
    Returns the entire briefs payload including metadata.
    """
    return load_briefs()


@app.get("/api/status")
def get_status():
    """
    Get the current status of the agent.
    Streamlit shows this as a status indicator in the header.
    """
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
    Manually trigger an agent run.
    Called when user clicks "Refresh Now" in the dashboard.
    
    CONCEPT — BackgroundTasks:
    FastAPI's BackgroundTasks runs the function AFTER the response is sent.
    So the API returns immediately ("refresh started") and the agent runs
    in the background. The dashboard polls /api/status to track progress.
    
    WHY NOT AWAIT THE AGENT?
    Agent runs take 30-120 seconds (web searches are slow).
    If we awaited it, the HTTP request would timeout (default: 30s).
    Background task pattern avoids the timeout.
    """
    if agent_status["is_running"]:
        return {"status": "already_running", "message": "Agent is already running"}
    
    background_tasks.add_task(run_agent_job)
    return {"status": "started", "message": "Agent refresh started in background"}


@app.delete("/api/briefs")
def clear_all_briefs():
    """Clear the briefs cache. Useful for testing."""
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
