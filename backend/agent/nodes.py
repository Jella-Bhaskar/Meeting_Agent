"""
=============================================================================
agent/nodes.py — The Agent's Brain (Gemini 1.5 Flash edition)
=============================================================================

FIXES IN THIS VERSION:
1. MODEL changed to gemini-1.5-flash (1500 req/day vs 20 req/day for 2.5-flash)
2. max_tokens for company extraction: 150 → 500
3. max_tokens for research: 1000 → 2500
4. parse_json_from_text replaced with safe_parse_json — tries direct json.loads
   FIRST before any regex fallback, preventing good JSON from being mangled
5. _is_meaningful threshold: len < 50 → len < 10
6. Retry with backoff on 429 quota errors
=============================================================================
"""

import asyncio
import json
import os
import re
import time
import urllib.request
from datetime import datetime

import google.generativeai as genai

from agent.state import AgentState, CompanySignal, MeetingBrief, MeetingEvent
from calendar_client import fetch_upcoming_meetings, extract_company_signals_from_meeting

# gemini-1.5-flash: 1500 req/day free tier (vs gemini-2.5-flash: only 20 req/day)
MODEL = "gemini-2.5-flash"


# =============================================================================
# Gemini client — configure INSIDE function, never at module level
# (module-level os.getenv runs before load_dotenv → key is None → auth fails)
# =============================================================================

def call_gemini(prompt: str, use_search: bool = False, max_tokens: int = 2000) -> str:
    """
    Unified Gemini API call with retry on quota errors.
    Retries up to 3 times with 30s/60s/90s backoff on 429 errors.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    adc = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if api_key:
        genai.configure(api_key=api_key)
    elif adc:
        print("Using Application Default Credentials from GOOGLE_APPLICATION_CREDENTIALS")
    else:
        raise RuntimeError(
            "No Gemini authentication configured.\n"
            "Set GEMINI_API_KEY in your .env file.\n"
            "Get key at: https://aistudio.google.com/apikey"
        )

    model = genai.GenerativeModel(
        model_name=MODEL,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.0,
        )
    )

    last_error = None
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s
                print(f"   ⏳ Quota limit hit. Waiting {wait}s before retry {attempt+1}/3...")
                time.sleep(wait)
            else:
                raise  # Non-quota error — don't retry
    raise last_error


# =============================================================================
# JSON parser — safe_parse_json tries direct parse FIRST
# The old parse_json_from_text was mangling valid JSON via regex fallbacks.
# This version: direct json.loads → strip fences → extract block → error
# =============================================================================

def safe_parse_json(text: str) -> dict:
    """
    Parse JSON from Gemini output safely.

    ROOT CAUSE OF PREVIOUS BUG:
    The old parser's Strategy 6 (regex field extraction) was triggering on
    VALID JSON because Strategy 3 (extract {} block) was failing due to a
    greedy regex edge case. The regex fallback then filled all fields with
    placeholder strings like "Information not available".

    FIX: Try json.loads directly first. Only use extraction as last resort.
    If direct parse fails, strip fences and try again. Only then extract block.
    """
    if not text:
        raise ValueError("Empty response from Gemini")

    # Strategy 1: direct parse — handles clean JSON responses
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip ```json ... ``` markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first { ... } block and parse it
    # Use a non-greedy approach to avoid matching partial blocks
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if not match:
        # Fallback: greedy match
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        # Try removing trailing commas from the block
        try:
            fixed = re.sub(r',\s*([}\]])', r'\1', match.group(0))
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from Gemini response.\nRaw text:\n{text[:500]}")


# Keep old name as alias so nothing else breaks
def parse_json_from_text(text: str) -> dict:
    return safe_parse_json(text)


# =============================================================================
# Company inference helpers (unchanged)
# =============================================================================

def infer_company_name_from_domain(domain: str) -> str:
    """Heuristic company name extraction from a single external email domain."""
    domain = domain.lower().split("/")[-1].split("?")[0].split(":")[0]
    parts = domain.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "net", "org", "app", "io"}:
        base = parts[-3]
    elif len(parts) >= 2:
        base = parts[-2]
    else:
        base = parts[0]
    return " ".join(word.capitalize() for word in base.replace("-", " ").split())


# def infer_company_from_signals(raw_signals: dict):
#     """Infer a company name from meeting signals before calling Gemini."""
#     if raw_signals.get("potential_company_from_title"):
#         company_name = raw_signals["potential_company_from_title"]
#         domain = raw_signals["external_domains"][0] if len(raw_signals["external_domains"]) == 1 else None
#         return CompanySignal(
#             meeting_id=raw_signals["meeting_id"],
#             company_name=company_name,
#             domain=domain,
#             confidence="high",
#             inference_source="meeting_title"
#         )

#     if len(raw_signals.get("external_domains", [])) == 1:
#         domain = raw_signals["external_domains"][0]
#         company_name = infer_company_name_from_domain(domain)
#         return CompanySignal(
#             meeting_id=raw_signals["meeting_id"],
#             company_name=company_name,
#             domain=domain,
#             confidence="high",
#             inference_source="attendee_email"
#         )

#     return None

def infer_company_from_signals(raw_signals: dict):
    if raw_signals.get("potential_company_from_title"):
        company_name = raw_signals["potential_company_from_title"]
        domain = raw_signals["external_domains"][0] if raw_signals["external_domains"] else None
        return CompanySignal(
            meeting_id=raw_signals["meeting_id"],
            company_name=company_name,
            domain=domain,
            confidence="high",
            inference_source="meeting_title"
        )

    # FIX: works with 1 OR more external domains — picks the first one
    if len(raw_signals.get("external_domains", [])) >= 1:
        domain = raw_signals["external_domains"][0]
        company_name = infer_company_name_from_domain(domain)
        return CompanySignal(
            meeting_id=raw_signals["meeting_id"],
            company_name=company_name,
            domain=domain,
            confidence="high",
            inference_source="attendee_email"
        )

    return None
# =============================================================================
# NODE 1: Fetch Calendar
# =============================================================================

def fetch_calendar_node(state: AgentState) -> dict:
    print("📅 [Node 1] Fetching calendar events...")
    try:
        days = int(os.getenv("FETCH_DAYS_AHEAD", "7"))
        meetings = fetch_upcoming_meetings(days_ahead=days)
        print(f"   Found {len(meetings)} upcoming meetings")
        return {"raw_events": meetings, "errors": state.get("errors", [])}
    except Exception as e:
        error_msg = f"Calendar fetch failed: {str(e)}"
        print(f"   ❌ {error_msg}")
        return {"raw_events": [], "errors": state.get("errors", []) + [error_msg]}


# =============================================================================
# NODE 2: Extract Companies
# =============================================================================

def extract_companies_node(state: AgentState) -> dict:
    """
    Uses heuristic first (fast, no API call).
    Falls back to Gemini for ambiguous cases.
    max_tokens increased to 500 to avoid truncation.
    """
    print("🏢 [Node 2] Extracting company signals (Gemini)...")

    signals = []
    errors = list(state.get("errors", []))

    for event in state["raw_events"]:
        raw_signals = extract_company_signals_from_meeting(event)
        heuristic_signal = infer_company_from_signals(raw_signals)

        if heuristic_signal:
            print(
                f"   [HEURISTIC] Inferred company: "
                f"{heuristic_signal['company_name']} "
                f"(source={heuristic_signal['inference_source']})"
            )
            signals.append(heuristic_signal)
            continue

        prompt = f"""Extract company info from this meeting. Reply with ONLY a JSON object, nothing else.

Title: {raw_signals['title']}
Emails: {raw_signals['attendee_emails']}
Domains: {raw_signals['external_domains']}
Company hint: {raw_signals['potential_company_from_title']}

Rules:
- gmail/yahoo/hotmail/outlook = personal email, set company_name to null
- internal meeting (all same domain) = null
- linear.app → "Linear", acme.com → "Acme", growthsignal.io → "GrowthSignal"
- confidence: high/medium/low/unknown
- inference_source: attendee_email/meeting_title/description/none

JSON only, no markdown, no explanation:
{{"company_name": null, "domain": null, "confidence": "unknown", "inference_source": "none"}}"""

        try:
            # max_tokens 150 → 500 to prevent truncation
            raw = call_gemini(prompt, use_search=False, max_tokens=500)
            parsed = safe_parse_json(raw)

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
            signals.append(CompanySignal(
                meeting_id=event["id"],
                company_name=None,
                domain=None,
                confidence="unknown",
                inference_source="none"
            ))

    return {"company_signals": signals, "errors": errors}


# =============================================================================
# NODE 3: Research Companies
# =============================================================================

def research_company_node(state: AgentState) -> dict:
    print("🔍 [Node 3] Researching companies (parallel)...")
    signal_map = {s["meeting_id"]: s for s in state["company_signals"]}
    event_map = {e["id"]: e for e in state["raw_events"]}
    briefs = asyncio.run(_research_all_parallel(state["raw_events"], signal_map, event_map))
    return {"briefs": briefs, "errors": state.get("errors", [])}


async def _research_all_parallel(raw_events, signal_map, event_map):
    tasks = [
        _research_single_meeting(event, signal_map.get(event["id"]))
        for event in raw_events
    ]
    return await asyncio.gather(*tasks)


async def _research_single_meeting(event: MeetingEvent, signal: CompanySignal) -> MeetingBrief:
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

    if not signal or not signal["company_name"]:
        base_brief["error_message"] = "Company could not be identified for this meeting."
        print(f"   ⚠️  '{event['title']}' → fallback (no company)")
        return base_brief

    company = signal["company_name"]
    domain = signal.get("domain", "")

    try:
        # DATA SOURCE 2: Direct homepage fetch
        homepage_text = await asyncio.to_thread(fetch_company_homepage, domain) if domain else ""
        homepage_section = (
            f"\n\nDATA SOURCE 2 — Company homepage ({domain}):\n{homepage_text}"
            if homepage_text and not homepage_text.startswith("[Homepage fetch failed")
            else f"\n\n[Homepage fetch skipped: domain={domain or 'unknown'}]"
        )

        # DATA SOURCE 1: Gemini with knowledge base
        research_prompt = f"""You are a meeting intelligence researcher. Prepare a brief for an upcoming meeting with {company}.

Based on your knowledge of {company} (domain: {domain or 'unknown'}), provide detailed information:

First-party data available:
{homepage_section}

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "company_overview": "2-3 sentences: what they do, their market, their customers",
  "recent_news": "3 bullet points of recent developments from 2024-2025: - point 1\\n- point 2\\n- point 3",
  "tech_signals": "Key tech stack, tools, frameworks, infrastructure they use",
  "pain_points": "2-3 key challenges they face based on their industry and stage",
  "talking_points": [
    "Specific insightful question 1 about their business",
    "Specific question 2 about their technology or product",
    "Specific question 3 about their strategy or growth"
  ]
}}"""

        # max_tokens 1000 → 2500 to get complete responses
        raw = await asyncio.to_thread(call_gemini, research_prompt, True, 5500)

        print(f"   [DEBUG] Raw response (first 200 chars): {repr(raw[:200])}")

        # Use safe parser — tries direct json.loads first
        data = safe_parse_json(raw)

        print(f"   [DEBUG] Parsed overview: {repr(str(data.get('company_overview',''))[:100])}")

        # _is_meaningful: only reject empty or exact placeholder strings
        # Threshold reduced from 50 → 10 chars
        def _is_meaningful(val: str) -> bool:
            if not val:
                return False
            t = val.strip()
            if len(t) < 10:
                return False
            placeholders = {
                "Information not available",
                "Recent news not available",
                "Tech signals not detected",
                "Pain points not determined",
                "No overview available",
                "No recent news found",
                "Not detected",
                "Not determined",
                "Not publicly known",
            }
            return t not in placeholders

        has_overview = _is_meaningful(data.get("company_overview", ""))
        has_news = _is_meaningful(data.get("recent_news", ""))

        print(f"   [DEBUG] has_overview={has_overview} has_news={has_news}")

        if not (has_overview or has_news):
            base_brief.update({
                "company_overview": data.get("company_overview", "Information not available"),
                "recent_news": data.get("recent_news", "Recent news not available"),
                "tech_signals": data.get("tech_signals", "Tech signals not detected"),
                "pain_points": data.get("pain_points", "Pain points not determined"),
                "talking_points": data.get("talking_points", []),
                "status": "fallback",
                "error_message": "Research returned no substantive content",
            })
            print(f"   ⚠️  '{event['title']}' → insufficient research content ({company})")
            return base_brief

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
    from brief_store import save_briefs
    print(f"💾 [Node 4] Finalizing {len(state['briefs'])} briefs...")
    researched = sum(1 for b in state["briefs"] if b["status"] == "researched")
    fallback   = sum(1 for b in state["briefs"] if b["status"] == "fallback")
    errors_count = sum(1 for b in state["briefs"] if b["status"] == "error")
    print(f"   ✅ {researched} researched | ⚠️ {fallback} fallback | ❌ {errors_count} errors")
    save_briefs(state["briefs"])
    return {"briefs": state["briefs"]}


# =============================================================================
# SECOND DATA SOURCE — Direct homepage fetch
# =============================================================================

def fetch_company_homepage(domain: str) -> str:
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
