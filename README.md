<<<<<<< HEAD
# 🧠 Meeting Intelligence Agent

Autonomous agent that monitors your Google Calendar, researches companies before external calls, and presents ready-to-use intelligence briefs — without you asking.

---

## 🎓 What You'll Learn Building This

| Concept | Where In Code | Why It Matters |
|---------|---------------|----------------|
| **Agentic State** | `agent/state.py` | How agents accumulate memory across steps |
| **LangGraph StateGraph** | `agent/graph.py` | Industry framework for multi-step agents |
| **Tool Use (Function Calling)** | `agent/nodes.py` | LLM decides autonomously when to search |
| **Structured JSON Output** | `agent/nodes.py` | Parse LLM responses reliably |
| **Background Scheduling** | `main.py` | True autonomy: agent runs without user input |
| **Graceful Fallback** | `agent/nodes.py` | Production requirement: no crashes, no blank cards |
| **External OAuth Integration** | `calendar_client.py` | Connect to Google APIs with token flow |
| **Async Parallel Execution** | `agent/nodes.py` | Research 10 companies simultaneously, not sequentially |
| **Atomic File Writes** | `brief_store.py` | Prevent data corruption during concurrent access |
| **API + Frontend Separation** | `main.py` + `frontend/app.py` | FastAPI backend, Streamlit frontend |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│            Streamlit Dashboard (port 8501)           │
│    Polls /api/briefs every 30s for live updates     │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────┐
│           FastAPI Backend (port 8000)                │
│                                                      │
│  GET /api/briefs    → returns cached briefs          │
│  GET /api/status    → agent run status              │
│  POST /api/refresh  → trigger agent run             │
│                                                      │
│  Background: APScheduler runs agent every 30 min    │
└──────────┬──────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────┐
│        LangGraph Agent (4 nodes in sequence)         │
│                                                      │
│  [fetch_calendar] → [extract_companies]              │
│        → [research_companies] → [store_briefs]      │
│                                                      │
│  Node 3 runs PARALLEL research for all meetings     │
│  Claude uses web_search tool AUTONOMOUSLY           │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Setup (Step by Step)

### Step 1: Clone and install dependencies

```bash
git clone https://github.com/your-username/meeting-intel-agent
cd meeting-intel-agent
pip install -r requirements.txt
```

### Step 2: Get your Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com/account/keys)
2. Create a new API key
3. Copy it for the next step

### Step 3: Set up Google Calendar OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Search for **"Google Calendar API"** → Enable it
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
5. Application type: **Desktop app** → Create
6. Click **Download JSON** → save as `backend/credentials.json`
7. First time you run the backend, it will open a browser for Google login
8. After login, `token.json` is created automatically

### Step 4: Configure environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   ANTHROPIC_API_KEY=sk-ant-...
#   (Google paths are already set correctly)
```

### Step 5: Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The first time it runs:
- A browser opens for Google Calendar OAuth (log in and allow access)
- The agent immediately fetches your meetings and researches companies
- This takes 1-3 minutes depending on how many meetings you have

### Step 6: Run the Streamlit dashboard

In a new terminal:

```bash
cd frontend
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — your dashboard is live!

---

## 📁 File Structure

```
meeting-intel-agent/
├── backend/
│   ├── main.py              # FastAPI app + scheduler
│   ├── calendar_client.py   # Google Calendar integration
│   ├── brief_store.py       # JSON persistence layer
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py         # TypedDict state schema (agent's memory)
│   │   ├── graph.py         # LangGraph StateGraph (wires nodes together)
│   │   └── nodes.py         # All 4 agent nodes (the actual logic)
│   └── credentials.json     # (you create this — Google OAuth)
├── frontend/
│   └── app.py               # Streamlit dashboard
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧩 Key Concepts Explained

### What is an Agent?
An agent is an LLM that can take actions (call tools, read data, write results) and loop until its goal is complete. Unlike a chatbot that responds once, an agent plans and executes multi-step tasks autonomously.

### What is LangGraph?
LangGraph is a framework for building agents as directed graphs. Each node is a Python function that reads state, does work, and returns state updates. The graph controls execution order. Advantages over raw LLM calls:
- Explicit state management
- Conditional routing (different paths based on data)
- Easy to add retries, logging, checkpointing

### What is Tool Use?
When you give Claude a `web_search` tool, it can decide mid-reasoning: "I need to look this up." It calls the tool, gets results, and continues. You don't control *when* it searches — Claude decides based on what it needs. This is fundamentally what makes agents "agentic" — they take autonomous actions.

### What is Structured Output?
We instruct Claude to return JSON matching a specific schema. This lets us:
- Parse the response into a TypedDict reliably
- Display specific fields in the UI (company_overview, recent_news, etc.)
- Validate that required fields are present

---

## 🚢 Deployment

### Backend → Railway.app (free tier)
```bash
# In backend/
# Create a Procfile:
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy:
railway login
railway init
railway up
```

Set environment variables in Railway dashboard:
- `ANTHROPIC_API_KEY`
- `GOOGLE_CREDENTIALS_PATH` (upload credentials.json as a file)

**Note:** For Google OAuth in production, use a Service Account instead of OAuth tokens. See [Google Service Account docs](https://developers.google.com/identity/protocols/oauth2/service-account).

### Frontend → Streamlit Cloud (free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `frontend/app.py`
4. Add secret: `API_URL = https://your-backend.railway.app`
5. Deploy — you get a public URL

---

## 🔧 What to Improve (Submission Note)

Given more time, I would:

1. **Google Service Account auth** — Replace OAuth token flow with a service account for headless server deployment (no browser popup needed)

2. **Redis cache** — Replace JSON file store with Redis for atomic updates and multi-instance support when scaling

3. **Webhook instead of polling** — Use Google Calendar Push Notifications to trigger the agent instantly when a new meeting is added, instead of polling every 30 min

4. **Per-meeting research status** — Stream research progress to the dashboard so users see "researching Linear..." in real-time instead of waiting

5. **CRM integration** — Pull deal history from Salesforce/HubSpot to add account context to the brief alongside public research

---

## 🐛 Troubleshooting

**"Cannot connect to backend"**
→ Make sure `uvicorn main:app --reload` is running in the `backend/` folder

**"credentials.json not found"**
→ Download OAuth credentials from Google Cloud Console (see Step 3 above)

**"Calendar fetch failed"**
→ Delete `token.json` and restart the backend — it will re-authenticate

**"Research failed" on all cards**
→ Check your `ANTHROPIC_API_KEY` in `.env` is valid and has credits

**Streamlit shows empty state after refresh**
→ Agent may still be running (takes 1-3 min). Watch backend logs.
=======
# Meeting_Agent
Meeting_Agent
>>>>>>> b155adb7bccf2610aa3227dc5b7c9fcc434d2d77
