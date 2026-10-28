"""SupplyChain Guard AI
Autonomous enterprise cascade-loss simulator, geographic threat radar and costed rerouting engine.

Inference : NVIDIA Nemotron on Nebius Token Factory (Nano plans searches via tool calling, Ultra reasons)
Grounding : Tavily web search (multi-query, cited sources)
Math      : deterministic Python (the LLM supplies structure, never the final dollar figures)
"""
import os
import re
import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pydeck as pdk
import streamlit as st
from openai import OpenAI
from tavily import TavilyClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------------------------------------------
# Configuration (all secrets come from the environment, never from source code)
# ----------------------------------------------------------------------------
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
HEAVY_MODEL = os.getenv("NEBIUS_HEAVY_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")  # cascade reasoning
FAST_MODEL = os.getenv("NEBIUS_FAST_MODEL", "nvidia/Nemotron-3-Nano-30B-A3B")      # chat / quick calls

st.set_page_config(
    page_title="SupplyChain Guard AI | Enterprise Cascade Loss Engine",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ SupplyChain Guard AI")
st.caption(
    "Autonomous Enterprise Cascade Loss Simulator, Geographic Threat Radar & Costed Rerouting Engine "
    "| NVIDIA Nemotron on Nebius Token Factory + Tavily"
)

for key, default in {
    "target_query": "",
    "auto_run": False,
    "audit": None,
    "sources": [],
    "used_fallback": False,
    "trace": [],
    "audit_subject": "",
    "audit_category": "",
    "chat_messages": [{
        "role": "assistant",
        "content": "Hello! I'm your SupplyChain Guard assistant. Run an audit, then ask me about "
                   "procurement options, risk metrics or mitigation strategy.",
    }],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ----------------------------------------------------------------------------
# LLM helpers
# ----------------------------------------------------------------------------
def get_client() -> OpenAI:
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY, timeout=120)


def strip_think(text: str) -> str:
    """Remove any <think>...</think> reasoning blocks some models emit."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    text = strip_think(text)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _num(value, default=0.0):
    try:
        if isinstance(value, str):
            value = re.sub(r"[^0-9.\-]", "", value)
        return float(value)
    except (TypeError, ValueError):
        return default


def _list(value):
    return [str(v) for v in value] if isinstance(value, list) else []


def normalize_audit(raw: dict) -> dict:
    """Coerce model output into a safe, fully-populated schema."""
    geo = raw.get("geo_coordinates") or {}
    cascade = raw.get("cascade_impact") or {}
    plan = raw.get("costed_rerouting_plan") or {}
    lat_lon_ok = (
        -90 <= _num(geo.get("bottleneck_lat"), 999) <= 90
        and -180 <= _num(geo.get("bottleneck_lon"), 999) <= 180
        and -90 <= _num(geo.get("alt_supplier_lat"), 999) <= 90
        and -180 <= _num(geo.get("alt_supplier_lon"), 999) <= 180
    )
    matrix = []
    for row in raw.get("vendor_comparison_matrix") or []:
        if isinstance(row, dict):
            matrix.append({k: str(v) for k, v in row.items()})
    return {
        "vulnerability_score": int(max(0, min(100, _num(raw.get("vulnerability_score"), 50)))),
        "threat_level": str(raw.get("threat_level", "Unknown")),
        "summary": str(raw.get("summary", "N/A")),
        "bottleneck_node": str(raw.get("bottleneck_node", "N/A")),
        "geo_coordinates": {
            "valid": lat_lon_ok,
            "bottleneck_lat": _num(geo.get("bottleneck_lat")),
            "bottleneck_lon": _num(geo.get("bottleneck_lon")),
            "alt_supplier_lat": _num(geo.get("alt_supplier_lat")),
            "alt_supplier_lon": _num(geo.get("alt_supplier_lon")),
        },
        "cascade_impact": {
            "affected_tier_1": str(cascade.get("affected_tier_1", "N/A")),
            "affected_tier_2_3": str(cascade.get("affected_tier_2_3", "N/A")),
            "impacted_product_skus": _list(cascade.get("impacted_product_skus")),
        },
        "estimated_delay_days": int(max(1, min(60, _num(raw.get("estimated_delay_days"), 14)))),
        "vendor_comparison_matrix": matrix,
        "costed_rerouting_plan": {
            "alternative_supplier": str(plan.get("alternative_supplier", "N/A")),
            "lead_time_days": int(_num(plan.get("lead_time_days"), 0)),
            "freight_premium_percent": _num(plan.get("freight_premium_percent"), 0),
            "sea_freight_cost": _num(plan.get("sea_freight_cost"), 25000),
        },
        "key_risk_factors": _list(raw.get("key_risk_factors")),
        "financial_impact_estimate": str(raw.get("financial_impact_estimate", "N/A")),
        "mitigation_steps": _list(raw.get("mitigation_steps") or raw.get("actionable_mitigation_steps")),
    }


def llm_json(prompt: str, model: str, retries: int = 2) -> dict:
    """Call the model, parse + validate JSON, retry on failure."""
    client = get_client()
    last_err = None
    for attempt in range(retries + 1):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": "Respond with a single valid JSON object and nothing else."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4000,
            )
            if attempt == 0:  # some models reject response_format; fall back on later attempts
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return normalize_audit(extract_json(resp.choices[0].message.content))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Model call failed after {retries + 1} attempts: {last_err}")


# ----------------------------------------------------------------------------
# Tavily research (multi-query, de-duplicated, cited)
# ----------------------------------------------------------------------------
def research(subject: str, category: str) -> list[dict]:
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    queries = [
        f"{subject} disruption latest news",
        f"{subject} supply chain financial impact {category}",
        f"alternative suppliers and rerouting options for {subject}",
    ]
    seen, sources = set(), []
    for q in queries:
        try:
            res = tavily.search(query=q, max_results=4)
        except Exception:  # noqa: BLE001  one failed query must not kill the audit
            continue
        for r in res.get("results", []):
            if r.get("url") and r["url"] not in seen:
                seen.add(r["url"])
                sources.append({"title": r.get("title", ""), "url": r["url"], "content": r.get("content", "")})
    return sources


SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the live web for supply chain intelligence (news, port/fab status, financial impact, alternative suppliers).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}]


def agent_research(subject: str, category: str, log, max_steps: int = 4) -> list[dict]:
    """Tool-calling planner: the Nano model decides which Tavily searches to run and when to stop."""
    client, tavily = get_client(), TavilyClient(api_key=TAVILY_API_KEY)
    messages = [
        {"role": "system", "content": (
            "You are a supply chain intelligence analyst. Use the web_search tool to gather evidence on: "
            "(1) current disruption status, (2) financial and operational impact, (3) alternative suppliers or "
            "routes. Make specific, different queries. Review results; if a gap remains, search again. "
            "When you have enough evidence, reply with exactly DONE.")},
        {"role": "user", "content": f"Investigate target '{subject}' under exposure category '{category}'."},
    ]
    sources, seen = [], set()
    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=FAST_MODEL, messages=messages, tools=SEARCH_TOOL,
            tool_choice="auto", temperature=0.2, max_tokens=800)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            log(f"🧭 Planner finished after {step} search round(s)")
            break
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls]})
        for tc in msg.tool_calls[:3]:
            try:
                query = json.loads(tc.function.arguments or "{}").get("query", "")
            except json.JSONDecodeError:
                query = ""
            log(f"🔎 Step {step + 1}: searching \"{query}\"")
            found = []
            try:
                for r in tavily.search(query=query, max_results=4).get("results", []):
                    if r.get("url") and r["url"] not in seen:
                        seen.add(r["url"])
                        item = {"title": r.get("title", ""), "url": r["url"], "content": r.get("content", "")}
                        sources.append(item)
                        found.append(item)
            except Exception as exc:  # noqa: BLE001
                log(f"   ⚠️ search failed: {exc}")
            log(f"   ↳ {len(found)} new sources")
            digest = "\n".join(f"- {s['title']}: {s['content'][:300]}" for s in found) or "No new results."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": digest})
    return sources


def build_prompt(subject: str, category: str, sources: list[dict]) -> str:
    evidence = "\n".join(f"[{i + 1}] {s['title']} ({s['url']}): {s['content'][:600]}" for i, s in enumerate(sources))
    schema = {
        "vulnerability_score": "int 0-100",
        "threat_level": "Low | Moderate | High | Critical",
        "summary": "2-4 sentence executive summary; cite evidence as [n]",
        "bottleneck_node": "specific port / fab / route",
        "geo_coordinates": {
            "bottleneck_lat": "decimal degrees", "bottleneck_lon": "decimal degrees",
            "alt_supplier_lat": "decimal degrees", "alt_supplier_lon": "decimal degrees",
        },
        "cascade_impact": {
            "affected_tier_1": "string", "affected_tier_2_3": "string",
            "impacted_product_skus": ["3-5 generic product lines"],
        },
        "estimated_delay_days": "int 1-60, realistic disruption duration based on the evidence",
        "vendor_comparison_matrix": [{
            "vendor_name": "string", "location": "string", "lead_time_days": "int",
            "unit_cost_variance": "e.g. +8%", "expedited_freight_cost": "e.g. $45,000",
            "solvency_score": "e.g. 82/100",
        }],
        "costed_rerouting_plan": {
            "alternative_supplier": "name & location", "lead_time_days": "int",
            "freight_premium_percent": "number", "sea_freight_cost": "USD estimate per shipment, number",
        },
        "key_risk_factors": ["3-5 strings"],
        "financial_impact_estimate": "short text",
        "mitigation_steps": ["3-5 actionable steps"],
    }
    return (
        "You are a Chief Procurement Officer and enterprise risk auditor.\n"
        f"Target node: '{subject}'. Exposure category: '{category}'.\n\n"
        f"GROUND-TRUTH EVIDENCE (live web search):\n{evidence or 'No evidence retrieved.'}\n\n"
        "Rules: base claims on the evidence; vendor matrix needs 3 rows (affected primary vendor first, then two "
        "alternatives); solvency scores and costs are estimates; use real coordinates; do NOT invent company revenue.\n"
        f"Return JSON matching this schema exactly:\n{json.dumps(schema, indent=2)}"
    )


# ----------------------------------------------------------------------------
# Deterministic financial engine (no LLM involvement)
# ----------------------------------------------------------------------------
def compute_finance(days, rev_per_day, penalty_per_day, sea_cost, air_cost, sea_days, air_days):
    per_day = rev_per_day + penalty_per_day
    do_nothing = days * per_day
    sea_total = sea_cost + min(days, sea_days) * per_day   # downtime until rerouted supply lands
    air_total = air_cost + min(days, air_days) * per_day
    options = {"Do nothing": do_nothing, "Reroute by sea": sea_total, "Reroute by air": air_total}
    best = min(options, key=options.get)
    return {
        "per_day": per_day, "revenue_loss": days * rev_per_day, "penalty": days * penalty_per_day,
        "do_nothing": do_nothing, "sea_total": sea_total, "air_total": air_total,
        "best": best, "best_total": options[best], "preserved": do_nothing - options[best],
    }


# ----------------------------------------------------------------------------
# Cached illustrative fallbacks so the live demo can never hard-fail on a preset
# ----------------------------------------------------------------------------
def _fb(summary, node, coords, tiers, days, vendors, plan, risks, fin, steps, score, level):
    return normalize_audit({
        "vulnerability_score": score, "threat_level": level, "summary": summary, "bottleneck_node": node,
        "geo_coordinates": dict(zip(["bottleneck_lat", "bottleneck_lon", "alt_supplier_lat", "alt_supplier_lon"], coords)),
        "cascade_impact": dict(zip(["affected_tier_1", "affected_tier_2_3", "impacted_product_skus"], tiers)),
        "estimated_delay_days": days,
        "vendor_comparison_matrix": [dict(zip(
            ["vendor_name", "location", "lead_time_days", "unit_cost_variance", "expedited_freight_cost", "solvency_score"], v))
            for v in vendors],
        "costed_rerouting_plan": plan, "key_risk_factors": risks, "financial_impact_estimate": fin,
        "mitigation_steps": steps,
    })


FALLBACKS = {
    "tsmc": _fb(
        "Concentrated leading-edge wafer capacity in Hsinchu creates a single point of failure for downstream "
        "electronics assembly. (Cached illustrative scenario.)",
        "Hsinchu Science Park, Taiwan", (24.78, 120.99, 33.31, 130.50),
        ("Direct logic-chip supply to device assemblers", "Photoresists, specialty gases and substrates",
         ["Smartphone SoC line", "Data-center accelerator line", "Automotive MCU line"]),
        21,
        [("Primary fab (affected)", "Hsinchu, Taiwan", 21, "0%", "$0", "88/100"),
         ("Alt foundry A", "Kumamoto, Japan", 9, "+9%", "$48,000", "91/100"),
         ("Alt foundry B", "Arizona, USA", 14, "+16%", "$62,000", "86/100")],
        {"alternative_supplier": "Alt foundry A, Kumamoto, Japan", "lead_time_days": 9,
         "freight_premium_percent": 9, "sea_freight_cost": 30000},
        ["Geographic concentration", "Limited advanced-node substitutes", "Long qualification cycles"],
        "Exposure scales with daily line-stoppage revenue plus contractual SLA penalties.",
        ["Trigger dual-source qualification", "Pre-book air capacity", "Build 6-week buffer on critical SKUs"],
        82, "High"),
    "red sea": _fb(
        "Red Sea transit risk forces Cape of Good Hope rerouting, adding transit time and freight cost on "
        "Asia-Europe lanes. (Cached illustrative scenario.)",
        "Bab-el-Mandeb Strait", (12.58, 43.33, 36.14, -5.35),
        ("Inbound finished goods and components delayed", "Container and bunker cost inflation",
         ["Consumer electronics", "Apparel", "Industrial parts"]),
        18,
        [("Suez-routed carrier (affected)", "Singapore to Rotterdam", 30, "0%", "$0", "79/100"),
         ("Cape-routed carrier", "Singapore to Rotterdam via Cape", 42, "+6%", "$30,000", "90/100"),
         ("Med transshipment hub", "Tangier Med, Morocco", 24, "+11%", "$41,000", "84/100")],
        {"alternative_supplier": "Tangier Med transshipment hub", "lead_time_days": 24,
         "freight_premium_percent": 11, "sea_freight_cost": 28000},
        ["Armed-conflict risk to vessels", "Insurance premium spikes", "Schedule unreliability"],
        "Cost driven by extended transit, working-capital lock-up and expedite spend.",
        ["Shift to Cape routing with buffer", "Use Med hubs for EU delivery", "Renegotiate Incoterms"],
        74, "High"),
    "lithium": _fb(
        "European battery makers rely on imported refined lithium and cathode materials, leaving cell output "
        "exposed to refinery shortfalls. (Cached illustrative scenario.)",
        "Non-EU lithium refining capacity", (-23.50, -70.40, 52.37, 4.90),
        ("Cell and module assembly slowed", "Lithium carbonate/hydroxide and precursor shortages",
         ["EV battery packs", "Grid storage modules", "Consumer cells"]),
        28,
        [("Primary refiner (affected)", "Chile / China", 35, "0%", "$0", "72/100"),
         ("EU refiner A", "Rotterdam, Netherlands", 12, "+14%", "$38,000", "88/100"),
         ("Offtake partner B", "Western Australia", 20, "+10%", "$55,000", "85/100")],
        {"alternative_supplier": "EU refiner A, Rotterdam", "lead_time_days": 12,
         "freight_premium_percent": 14, "sea_freight_cost": 26000},
        ["Refining concentration", "Price volatility", "Regulatory sourcing requirements"],
        "Exposure driven by stalled cell production and contractual delivery penalties.",
        ["Secure EU refining offtake", "Hedge lithium price", "Qualify recycled-content feedstock"],
        78, "High"),
}


def match_fallback(subject: str):
    s = subject.lower()
    for key in FALLBACKS:
        if key in s or (key == "tsmc" and "hsinchu" in s) or (key == "red sea" and "bab-el" in s):
            return FALLBACKS[key]
    return None


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Control")
    if NEBIUS_API_KEY:
        st.success("Inference: Nebius Token Factory")
    else:
        st.error("NEBIUS_API_KEY missing (preset fallbacks only)")
    if TAVILY_API_KEY:
        st.info("Grounding: Tavily Search active")
    else:
        st.error("TAVILY_API_KEY missing")
    st.code(f"Reasoning : {HEAVY_MODEL}\nFast/chat : {FAST_MODEL}", language="text")

    if st.button("🔎 Verify models on Token Factory", use_container_width=True) and NEBIUS_API_KEY:
        try:
            ids = sorted(m.id for m in get_client().models.list().data)
            nv = [i for i in ids if "nemotron" in i.lower()]
            st.write("Nemotron models available:")
            st.code("\n".join(nv) or "none found", language="text")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not list models: {exc}")

    st.markdown("---")
    st.subheader("📈 Financial Assumptions (editable)")
    st.caption("All $ figures use these inputs. They are assumptions, not model output.")
    daily_revenue_at_risk = st.number_input("Daily Facility Revenue at Risk ($)", value=250000, step=10000)
    penalty_per_day = st.number_input("Contractual SLA Penalty / Day ($)", value=15000, step=2500)
    air_mult = st.slider("Air Freight Cost Multiplier", 1.5, 4.0, 2.2, 0.1)
    sea_days = st.number_input("Sea reroute: days until supply restored", value=18, min_value=1)
    air_days = st.number_input("Air reroute: days until supply restored", value=5, min_value=1)

    st.markdown("---")
    st.header("📧 Email Dispatch")
    sender_email = st.text_input("Gmail address", placeholder="your.name@gmail.com")
    sender_app_password = st.text_input("Gmail App Password", type="password")

# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
st.subheader("🔍 Micro-Node Vulnerability & Supply Chain Target")
st.markdown("**⚡ Quick-Test Presets:**")
c1, c2, c3 = st.columns(3)
presets = [
    (c1, "🇹🇼 TSMC Fab 18 & Hsinchu Park", "TSMC Silicon Wafer Fabrication & Hsinchu Science Park Disruptions"),
    (c2, "🚢 Bab-el-Mandeb & Red Sea", "Bab-el-Mandeb Strait Shipping Congestion and Suez Route Delays"),
    (c3, "🔋 European Battery Raw Material", "European Lithium Battery Raw Material & Refinery Shortages"),
]
for col, label, query in presets:
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.target_query = query
            st.session_state.auto_run = True
            st.rerun()

target_subject = st.text_input(
    "Target Node, Route, Fab, or Critical Component",
    value=st.session_state.target_query,
    placeholder="e.g., Port of Ningbo-Zhoushan, Automotive Chip Fabrication, Rhine River Freight",
)
risk_category = st.selectbox(
    "Primary Exposure Category",
    ["Geopolitical & Maritime Bottleneck", "Tier-2/3 Supplier Insolvency",
     "Regulatory Customs Seizure", "Extreme Weather & Port Closure"],
)
run_button = st.button("🚀 Run Complete Enterprise Loss Audit", type="primary", use_container_width=True)

if run_button or st.session_state.auto_run:
    st.session_state.auto_run = False
    if not target_subject:
        st.warning("Please enter or select a supply chain target to evaluate.")
    else:
        audit, sources, used_fallback, trace = None, [], False, []
        try:
            if not (NEBIUS_API_KEY and TAVILY_API_KEY):
                raise RuntimeError("API keys not configured")
            with st.status("Agent running...", expanded=True) as status:
                def log(line):
                    trace.append(line)
                    st.write(line)

                log(f"🧭 Planner ({FAST_MODEL}) is choosing which searches to run...")
                try:
                    sources = agent_research(target_subject, risk_category, log)
                except Exception as exc:  # noqa: BLE001  tool calling unsupported/failed
                    log(f"⚠️ Tool-calling unavailable ({str(exc)[:80]}); using fixed search plan")
                    sources = []
                if len(sources) < 3:
                    log("🌐 Tavily: running fixed 3-query search plan...")
                    have = {s["url"] for s in sources}
                    sources += [s for s in research(target_subject, risk_category) if s["url"] not in have]
                log(f"✅ {len(sources)} unique sources collected")
                log(f"🧠 Reasoner ({HEAVY_MODEL}) analyzing Tier-1 to Tier-3 cascade...")
                audit = llm_json(build_prompt(target_subject, risk_category, sources), HEAVY_MODEL)
                log("📐 Python finance engine computing dollar exposure (deterministic)")
                status.update(label="Audit complete", state="complete")
        except Exception as exc:  # noqa: BLE001
            audit = match_fallback(target_subject)
            if audit:
                used_fallback = True
                st.warning(f"Live run failed ({exc}). Showing cached illustrative scenario.")
            else:
                st.error(f"Error during execution: {exc}")
        if audit:
            st.session_state.update(
                audit=audit, sources=sources, used_fallback=used_fallback,
                audit_subject=target_subject, audit_category=risk_category, trace=trace,
            )

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
audit = st.session_state.audit
if audit:
    subject, category = st.session_state.audit_subject, st.session_state.audit_category
    est_days = audit["estimated_delay_days"]
    cascade, plan = audit["cascade_impact"], audit["costed_rerouting_plan"]
    sea_cost = plan["sea_freight_cost"]
    air_cost = sea_cost * air_mult

    st.markdown("---")
    st.header("📊 Executive Financial Audit & Threat Summary")
    if st.session_state.used_fallback:
        st.caption("⚠️ Cached illustrative data (live inference unavailable).")

    base = compute_finance(est_days, daily_revenue_at_risk, penalty_per_day, sea_cost, air_cost, sea_days, air_days)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vulnerability Score", f"{audit['vulnerability_score']} / 100")
    m2.metric("Threat Level", audit["threat_level"])
    m3.metric("Estimated Delay", f"{est_days} days")
    m4.metric("Unmitigated Exposure", f"${base['do_nothing']:,.0f}")
    st.progress(audit["vulnerability_score"] / 100)

    st.markdown("---")
    st.subheader("🗺️ 3D Geographic Threat Map & Rerouting Corridor")
    geo = audit["geo_coordinates"]
    if geo["valid"]:
        b_lat, b_lon, a_lat, a_lon = geo["bottleneck_lat"], geo["bottleneck_lon"], geo["alt_supplier_lat"], geo["alt_supplier_lon"]
        points = [
            {"name": "Chokepoint: " + audit["bottleneck_node"], "lat": b_lat, "lon": b_lon, "color": [239, 68, 68]},
            {"name": "Alt supplier: " + plan["alternative_supplier"], "lat": a_lat, "lon": a_lon, "color": [34, 197, 94]},
        ]
        arcs = [{"from_lat": b_lat, "from_lon": b_lon, "to_lat": a_lat, "to_lon": a_lon}]
        st.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer("ScatterplotLayer", data=points, get_position="[lon, lat]", get_color="color",
                          get_radius=150000, pickable=True),
                pdk.Layer("ArcLayer", data=arcs, get_source_position="[from_lon, from_lat]",
                          get_target_position="[to_lon, to_lat]", get_source_color=[239, 68, 68],
                          get_target_color=[34, 197, 94], get_width=4),
            ],
            initial_view_state=pdk.ViewState(latitude=(b_lat + a_lat) / 2, longitude=(b_lon + a_lon) / 2, zoom=2, pitch=45),
            tooltip={"text": "{name}"},
        ))
    else:
        st.info("The model returned invalid coordinates, so the map is skipped for this run.")

    st.markdown("---")
    st.subheader("📝 Executive Summary")
    st.info(audit["summary"])
    if st.session_state.trace:
        with st.expander("🧭 Agent trace (what the agent did)"):
            for line in st.session_state.trace:
                st.write(line)
    if st.session_state.sources:
        with st.expander(f"🔗 Evidence: {len(st.session_state.sources)} Tavily sources"):
            for i, s in enumerate(st.session_state.sources, 1):
                st.markdown(f"**[{i}]** [{s['title'] or s['url']}]({s['url']})")

    st.markdown("---")
    st.subheader("🧮 Interactive Disruption Loss Simulator")
    sim_days = st.slider("Disruption duration (days)", 1, 60, est_days, key=f"sim_{subject}")
    fin = compute_finance(sim_days, daily_revenue_at_risk, penalty_per_day, sea_cost, air_cost, sea_days, air_days)
    s1, s2, s3 = st.columns(3)
    s1.metric("Revenue Loss", f"${fin['revenue_loss']:,.0f}")
    s2.metric("SLA Penalties", f"${fin['penalty']:,.0f}")
    s3.metric("Total Unmitigated Loss", f"${fin['do_nothing']:,.0f}",
              delta=f"{fin['do_nothing'] - base['do_nothing']:+,.0f} vs estimate", delta_color="inverse")

    st.markdown("---")
    st.subheader("✈️ Transit Mode Trade-Off (Do Nothing vs Sea vs Air)")
    t1, t2, t3 = st.columns(3)
    t1.metric("Do nothing", f"${fin['do_nothing']:,.0f}")
    t2.metric(f"Sea reroute ({sea_days}d gap)", f"${fin['sea_total']:,.0f}")
    t3.metric(f"Air reroute ({air_days}d gap)", f"${fin['air_total']:,.0f}")
    if fin["preserved"] > 0:
        st.success(f"Best option: **{fin['best']}** preserves **${fin['preserved']:,.0f}** versus doing nothing.")
    else:
        st.warning("At this duration, rerouting costs more than waiting. Recommendation: **do nothing**.")

    st.markdown("---")
    st.subheader("📊 Multi-Vendor Solvency & SLA Matrix")
    if audit["vendor_comparison_matrix"]:
        st.table(audit["vendor_comparison_matrix"])

    d1, d2 = st.columns(2)
    with d1:
        st.subheader("⚠️ Key Risk Factors")
        for f in audit["key_risk_factors"]:
            st.markdown(f"* {f}")
        st.subheader("💰 Financial & Operational Exposure")
        st.write(audit["financial_impact_estimate"])
        st.subheader("🔗 Tier Cascade Exposure")
        st.write(f"**Bottleneck:** {audit['bottleneck_node']}")
        st.write(f"**Tier-1:** {cascade['affected_tier_1']}")
        st.write(f"**Tier-2/3:** {cascade['affected_tier_2_3']}")
        for sku in cascade["impacted_product_skus"]:
            st.markdown(f"* 📦 `{sku}`")
    with d2:
        st.subheader("⚡ Costed Rerouting Brief")
        st.success(f"**Recommended Alternative:** {plan['alternative_supplier']}")
        st.write(f"**Vendor lead time:** {plan['lead_time_days']} days")
        st.write(f"**Freight premium:** +{plan['freight_premium_percent']:.0f}%")
        st.write(f"**Sea freight:** ${sea_cost:,.0f} | **Air freight:** ${air_cost:,.0f}")
        st.info(f"**Value preserved by best option:** ${max(fin['preserved'], 0):,.0f}")
        st.subheader("🛡️ Mitigation Actions")
        for step in audit["mitigation_steps"]:
            st.markdown(f"* ✅ {step}")

    # ---- Export ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export & Share Report")
    report = f"""{'=' * 72}
SUPPLYCHAIN GUARD AI: EXECUTIVE FINANCIAL RISK BRIEFING
{'=' * 72}
Target: {subject}
Category: {category}
Vulnerability: {audit['vulnerability_score']}/100 | Threat: {audit['threat_level']}
Simulated Disruption: {sim_days} days

EXECUTIVE SUMMARY
{audit['summary']}

FINANCIAL ANALYSIS (assumptions: ${daily_revenue_at_risk:,.0f}/day revenue, ${penalty_per_day:,.0f}/day SLA penalty)
- Unmitigated loss:  ${fin['do_nothing']:,.0f}
- Sea reroute total: ${fin['sea_total']:,.0f}
- Air reroute total: ${fin['air_total']:,.0f}
- Recommended:       {fin['best']} (preserves ${max(fin['preserved'], 0):,.0f})

CASCADE
- Bottleneck: {audit['bottleneck_node']}
- Tier-1: {cascade['affected_tier_1']}
- Tier-2/3: {cascade['affected_tier_2_3']}
- SKUs: {', '.join(cascade['impacted_product_skus'])}

KEY RISKS
""" + "\n".join(f"- {f}" for f in audit["key_risk_factors"]) + f"""

REROUTING
- Alternative: {plan['alternative_supplier']}
- Lead time: {plan['lead_time_days']} days | Freight premium: +{plan['freight_premium_percent']:.0f}%

MITIGATION STEPS
""" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(audit["mitigation_steps"])) + """

Generated by SupplyChain Guard AI (NVIDIA Nemotron on Nebius Token Factory + Tavily)
"""
    e1, e2 = st.columns(2)
    with e1:
        st.download_button("📥 Download Audit JSON", json.dumps(audit, indent=2),
                           "SupplyChainGuard_Audit.json", "application/json", use_container_width=True)
        st.download_button("📄 Download C-Suite Brief (.txt)", report,
                           "SupplyChainGuard_CSuite_Brief.txt", "text/plain", use_container_width=True)
    with e2:
        recipient = st.text_input("Recipient email", placeholder="recipient@example.com")
        if st.button("📧 Send Executive Report", use_container_width=True):
            sender, pwd, to = sender_email.strip(), sender_app_password.replace(" ", "").strip(), recipient.strip()
            if not sender or not pwd:
                st.error("Enter your Gmail address and App Password in the sidebar.")
            elif "@" not in to:
                st.error("Enter a valid recipient email address.")
            else:
                try:
                    msg = MIMEMultipart()
                    msg["From"], msg["To"] = f"SupplyChain Guard AI <{sender}>", to
                    msg["Subject"] = f"[SupplyChain Guard AI] Executive Risk Audit: {subject}"
                    msg.attach(MIMEText(report, "plain"))
                    ctx = ssl.create_default_context()
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20, context=ctx) as server:
                        server.login(sender, pwd)
                        server.sendmail(sender, to, msg.as_string())
                    st.success(f"Report delivered to {to}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Email failed: {exc}")

    with st.expander("📄 Raw Structured Output (JSON)"):
        st.json(audit)

# ----------------------------------------------------------------------------
# Assistant (fast Nemotron tier, aware of the current audit)
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🤖 SupplyChain AI Assistant")
st.caption(f"Powered by {FAST_MODEL} for fast, low-cost responses.")

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("Ask about risk metrics, rerouting or procurement strategy..."):
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                system = "You are a professional enterprise supply chain consultant inside SupplyChain Guard AI."
                if st.session_state.audit:
                    a = st.session_state.audit
                    system += (f" Current audit: target '{st.session_state.audit_subject}', "
                               f"score {a['vulnerability_score']}/100, summary: {a['summary']}")
                resp = get_client().chat.completions.create(
                    model=FAST_MODEL,
                    messages=[{"role": "system", "content": system}] + st.session_state.chat_messages[-10:],
                    temperature=0.4,
                    max_tokens=800,
                )
                reply = strip_think(resp.choices[0].message.content)
                st.write(reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Assistant error: {exc}")
