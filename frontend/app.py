"""
frontend/app.py — Meeting Intelligence Dashboard
Improved UI: status badges, color coding, better layout, meeting timeline
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── API URL ─────────────────────────────────────────────────────────────────
try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = os.getenv("API_URL", "http://localhost:8000")

REFRESH_MS = 30_000

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Intel",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #0a0f1e; }
[data-testid="stHeader"] { background: transparent; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 12px 20px;
}

/* Expander cards */
[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    margin-bottom: 10px;
}
[data-testid="stExpander"]:hover {
    border-color: #4f46e5 !important;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-researched { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
.badge-fallback   { background: #451a03; color: #fcd34d; border: 1px solid #d97706; }
.badge-error      { background: #450a0a; color: #fca5a5; border: 1px solid #dc2626; }

/* Section headers inside cards */
.section-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 6px;
    margin-top: 16px;
}

/* Talking point items */
.tp-item {
    background: #0f172a;
    border-left: 3px solid #4f46e5;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin-bottom: 8px;
    font-size: 14px;
    color: #e2e8f0;
}

/* Meeting time badge */
.time-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8;
    margin-right: 8px;
}

/* Company name in card header */
.company-name {
    color: #818cf8;
    font-style: italic;
}

/* Info blocks */
.info-block {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.6;
    color: #cbd5e1;
}

/* Date section header */
.date-header {
    font-size: 20px;
    font-weight: 700;
    color: #e2e8f0;
    padding: 8px 0 4px 0;
    border-bottom: 1px solid #1f2937;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_briefs():
    try:
        r = requests.get(f"{API_URL}/api/briefs", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is FastAPI running on port 8000?", "briefs": []}
    except Exception as e:
        return {"error": str(e), "briefs": []}


def fetch_status():
    try:
        return requests.get(f"{API_URL}/api/status", timeout=5).json()
    except Exception:
        return {"is_running": False, "last_run_status": "unknown", "refresh_interval_minutes": 30}


def trigger_refresh():
    try:
        return requests.post(f"{API_URL}/api/refresh", timeout=5).json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Formatters ────────────────────────────────────────────────────────────────

def fmt_time(iso):
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%I:%M %p")
    except Exception:
        return iso or "—"


def fmt_date(iso):
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%A, %b %d")
    except Exception:
        return iso or "Unknown date"


def time_ago(iso):
    if not iso:
        return "never"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        s = int((datetime.now(timezone.utc) - dt).total_seconds())
        if s < 60: return f"{s}s ago"
        if s < 3600: return f"{s // 60}m ago"
        return f"{s // 3600}h ago"
    except Exception:
        return "—"


def group_by_date(briefs):
    grouped = {}
    for b in sorted(briefs, key=lambda x: x.get("start_time", "")):
        d = fmt_date(b.get("start_time", ""))
        grouped.setdefault(d, []).append(b)
    return grouped


# ── Brief card renderer ───────────────────────────────────────────────────────

def render_brief_card(brief: dict):
    status = brief.get("status", "fallback")
    company = brief.get("company_name") or "Unknown"
    title = brief.get("meeting_title", "Untitled")
    time_str = fmt_time(brief.get("start_time", ""))
    attendees = brief.get("attendees", [])

    badge_html = {
        "researched": '<span class="badge badge-researched">✅ Researched</span>',
        "fallback":   '<span class="badge badge-fallback">⚠️ No Company</span>',
        "error":      '<span class="badge badge-error">❌ Error</span>',
    }.get(status, "")

    icon = {"researched": "✅", "fallback": "⚠️", "error": "❌"}.get(status, "❓")
    label = f"{icon} {time_str}  —  {title}"
    if status == "researched":
        label += f"  |  {company}"

    with st.expander(label, expanded=(status == "researched")):

        # ── Header row ──
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown(
                f'<span class="time-pill">🕐 {time_str}</span>'
                f'<strong>{title}</strong>',
                unsafe_allow_html=True
            )
            if attendees:
                emails_str = ", ".join(f"`{e}`" for e in attendees[:4])
                if len(attendees) > 4:
                    emails_str += f" +{len(attendees)-4} more"
                st.markdown(f"👥 **Attendees:** {emails_str}")
        with col_right:
            st.markdown(badge_html, unsafe_allow_html=True)
            if status == "researched":
                st.markdown(f"🏢 **{company}**")
            st.caption(f"Brief: {time_ago(brief.get('researched_at', ''))}")

        st.divider()

        # ── Fallback state ──
        if status == "fallback":
            st.info(
                f"**Company not identified**  \n"
                f"{brief.get('error_message', 'Could not identify company for this meeting.')}"
            )
            st.caption("This is expected for personal emails (gmail, etc.) and internal meetings.")
            return

        # ── Error state ──
        if status == "error":
            st.error(f"**Research failed**  \n{brief.get('error_message', 'Unknown error')}")
            st.caption("The meeting is still displayed — only research failed.")
            return

        # ── Researched — full brief ──
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-label">🏢 Company Overview</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="info-block">{brief.get("company_overview", "Not available")}</div>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="section-label">📰 Recent News (last 60–90 days)</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="info-block">{brief.get("recent_news", "No recent news found")}</div>',
                unsafe_allow_html=True
            )

        with c2:
            st.markdown('<div class="section-label">⚙️ Tech Signals</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="info-block">{brief.get("tech_signals", "Not detected")}</div>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="section-label">🎯 Inferred Pain Points</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="info-block">{brief.get("pain_points", "Not determined")}</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-label">💬 Suggested Talking Points</div>', unsafe_allow_html=True)
        for i, pt in enumerate(brief.get("talking_points", []), 1):
            st.markdown(f'<div class="tp-item">{i}.&nbsp;&nbsp;{pt}</div>', unsafe_allow_html=True)

        if not brief.get("talking_points"):
            st.caption("No talking points generated.")


# ── Main dashboard ────────────────────────────────────────────────────────────

def main():
    # Auto-refresh every 30s
    st_autorefresh(interval=REFRESH_MS, key="auto_refresh")

    # ── Top header ──
    h1, h2, h3 = st.columns([4, 3, 1])
    with h1:
        st.markdown("# 🧠 Meeting Intelligence")
        st.caption("Autonomous pre-meeting research · Powered by Gemini 2.5 Flash")

    status = fetch_status()
    is_running = status.get("is_running", False)

    with h2:
        st.markdown("<br>", unsafe_allow_html=True)
        if is_running:
            st.info("🔄 Agent is researching your meetings…")
        else:
            last = status.get("last_run_at")
            if last:
                st.success(f"✅ Last refreshed {time_ago(last)}")
            else:
                st.warning("⏳ Agent hasn't run yet. Click Refresh Now.")

    with h3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True, disabled=is_running):
            r = trigger_refresh()
            if r.get("status") == "started":
                st.toast("Agent refresh started!", icon="🚀")
                time.sleep(1)
                st.rerun()
            else:
                st.warning(r.get("message", "Could not start"))

    st.divider()

    # ── Fetch data ──
    data = fetch_briefs()

    if "error" in data:
        st.error(f"**Backend connection error:** {data['error']}")
        st.markdown("**Start the backend:**")
        st.code("cd backend\nuvicorn main:app --reload --port 8000", language="bash")
        st.markdown("Or run demo mode:")
        st.code("cd backend\npython demo_run.py", language="bash")
        st.stop()

    briefs = data.get("briefs", [])

    if not briefs:
        st.markdown("### 📭 No meeting briefs yet")
        c1, c2 = st.columns(2)
        with c1:
            st.info(
                "**What to do:**\n"
                "1. Run `python demo_run.py` in backend/ folder\n"
                "2. Or click **Refresh** above (if using real Calendar)\n"
                "3. Briefs appear here automatically"
            )
        with c2:
            st.markdown("**Quick start:**")
            st.code("cd backend\npython demo_run.py\n\n# New terminal:\ncd frontend\nstreamlit run app.py", language="bash")
        st.stop()

    # ── Metrics ──
    r_count = sum(1 for b in briefs if b["status"] == "researched")
    f_count = sum(1 for b in briefs if b["status"] == "fallback")
    e_count = sum(1 for b in briefs if b["status"] == "error")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📅 Total Meetings", len(briefs))
    m2.metric("✅ Researched", r_count,
              delta=f"{round(r_count/len(briefs)*100)}% coverage" if briefs else None)
    m3.metric("⚠️ No Company", f_count)
    m4.metric("❌ Errors", e_count)

    st.divider()

    # ── Meeting cards by date ──
    for date_str, day_briefs in group_by_date(briefs).items():
        st.markdown(
            f'<div class="date-header">📆 {date_str}'
            f'<span style="font-size:13px;color:#6b7280;font-weight:400;margin-left:12px;">'
            f'{len(day_briefs)} meeting{"s" if len(day_briefs)!=1 else ""}</span></div>',
            unsafe_allow_html=True
        )
        for brief in day_briefs:
            render_brief_card(brief)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Footer ──
    st.divider()
    f1, f2, f3 = st.columns(3)
    f1.caption(f"📊 {len(briefs)} meetings tracked")
    f2.caption(f"🕐 Cache: {time_ago(data.get('last_updated'))}")
    f3.caption(f"🔁 Agent runs every {status.get('refresh_interval_minutes', 30)} min · Dashboard refreshes every 30s")


if __name__ == "__main__":
    main()
