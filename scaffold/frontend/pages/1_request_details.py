"""Per-request telemetry drill-down over logs/app.jsonl."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

LOG_FILE = Path(__file__).resolve().parents[2] / "logs" / "app.jsonl"
MAX_REQUESTS = 50

st.set_page_config(page_title="Request Details", page_icon="🔍", layout="wide")
st.title("🔍 Request Details")


@st.cache_data
def load_events(mtime: float) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    events = []
    for line in LOG_FILE.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn line at rotation boundary
    return events


mtime = LOG_FILE.stat().st_mtime if LOG_FILE.exists() else 0.0
events = load_events(mtime)
if not events:
    st.info("No log data yet — ask a question in the chat first (logs/app.jsonl).")
    st.stop()

by_trace: dict[str, list[dict]] = {}
for ev in events:
    by_trace.setdefault(ev.get("trace_id", "-"), []).append(ev)

# One row per request_summary event, newest first.
summaries = [ev for ev in events if ev.get("message") == "request_summary"]
summaries = summaries[-MAX_REQUESTS:][::-1]
if not summaries:
    st.info("No completed requests in the log yet.")
    st.stop()

st.caption(f"Last {len(summaries)} requests (newest first) · source: logs/app.jsonl")
overview = pd.DataFrame(
    [
        {
            "time": s.get("ts", ""),
            "trace_id": s.get("trace_id", "-"),
            "kind": s.get("kind", ""),
            "flags": ", ".join(s.get("flags", [])),
            "latency_ms": s.get("latency_ms"),
            "cost_usd": s.get("cost_usd"),
            "llm_calls": s.get("llm_calls"),
            "prompt_tok": s.get("prompt_tokens"),
            "completion_tok": s.get("completion_tokens"),
        }
        for s in summaries
    ]
)
st.dataframe(overview, width="stretch", hide_index=True)

trace_ids = [s["trace_id"] for s in summaries]
picked = st.selectbox("Inspect a request (trace_id)", trace_ids)
trace_events = by_trace.get(picked, [])
summary = next(e for e in summaries if e["trace_id"] == picked)

st.divider()
st.subheader(f"Trace `{picked}` — {summary.get('kind', '?')}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total latency", f"{summary.get('latency_ms', 0):.0f} ms")
c2.metric("Cost", f"${summary.get('cost_usd', 0):.4f}")
c3.metric("LLM calls", summary.get("llm_calls", 0))
c4.metric(
    "Tokens (in + out)",
    f"{summary.get('prompt_tokens', 0)} + {summary.get('completion_tokens', 0)}",
)

stages = summary.get("stage_breakdown", {})
if stages:
    st.subheader("Stage breakdown")
    # Numeric prefix keeps pipeline order in the chart.
    STAGE_ORDER = ["intent", "retrieval", "compress", "generate"]
    named = [(k.removesuffix("_ms"), v) for k, v in stages.items()]
    named.sort(key=lambda kv: STAGE_ORDER.index(kv[0]) if kv[0] in STAGE_ORDER else 99)
    stage_df = pd.DataFrame(
        {
            "stage": [f"{i + 1}. {name}" for i, (name, _) in enumerate(named)],
            "latency_ms": [v for _, v in named],
        }
    )
    st.bar_chart(stage_df, x="stage", y="latency_ms", horizontal=True)

llm_calls = [e for e in trace_events if e.get("message") == "llm_call"]
if llm_calls:
    st.subheader("LLM calls")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "component": e.get("component"),
                    "model": e.get("model"),
                    "latency_ms": e.get("latency_ms"),
                    "prompt_tok": e.get("prompt_tokens"),
                    "completion_tok": e.get("completion_tokens"),
                    "cost_usd": e.get("cost_usd"),
                }
                for e in llm_calls
            ]
        ),
        width="stretch",
        hide_index=True,
    )

# Old log entries lack the propagated trace_id; fall back.
def _mcp_events(message: str, n: int = 6) -> tuple[list[dict], bool]:
    in_trace = [e for e in trace_events if e.get("message") == message]
    if in_trace:
        return in_trace, True
    return [e for e in events if e.get("message") == message][-n:], False


searches, correlated = _mcp_events("vector_search")
if searches:
    st.subheader("Vector searches")
    if not correlated:
        st.caption(
            "Most recent events — this trace predates trace propagation to the "
            "MCP subprocess, so these are not correlated to it."
        )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "scope": e.get("scope"),
                    "roles": ", ".join(e.get("roles", [])),
                    "results": e.get("n_raw"),
                    "latency_ms": e.get("latency_ms"),
                }
                for e in searches
            ]
        ),
        width="stretch",
        hide_index=True,
    )

audits, correlated = _mcp_events("mcp_tool_call")
if audits:
    st.subheader("MCP audit trail")
    if not correlated:
        st.caption(
            "Most recent events — this trace predates trace propagation to the "
            "MCP subprocess, so these are not correlated to it."
        )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "tool": e.get("tool"),
                    "user_id": e.get("user_id"),
                    "scope": e.get("scope"),
                    "results": e.get("n_results", e.get("n_chunks", e.get("n_docs"))),
                    "latency_ms": e.get("latency_ms"),
                }
                for e in audits
            ]
        ),
        width="stretch",
        hide_index=True,
    )

with st.expander(f"🧾 Raw events ({len(trace_events)})"):
    st.json(trace_events)
