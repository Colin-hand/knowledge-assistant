"""Streamlit UI — thin client over the FastAPI backend.

Reads users.json ONLY to offer the demo login select box (name → token);
roles never exist client-side. Run:  streamlit run frontend/app.py

Requests run in a background thread so the UI stays responsive: Stop closes
the HTTP connection, which makes the server cancel the in-flight agent
request (best-effort — uvicorn cancels the task on client disconnect).
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
USERS_FILE = Path(__file__).resolve().parents[1] / "data" / "users.json"

st.set_page_config(page_title="Internal Knowledge Assistant", page_icon="📚")


QUESTION_BANK = Path(__file__).resolve().parents[1] / "eval" / "question_bank" / "questions.json"


@st.cache_data
def demo_users() -> dict[str, dict]:
    data = json.loads(USERS_FILE.read_text())
    return {u["name"]: {"token": u["token"], "roles": u["roles"]} for u in data["users"]}


@st.cache_data
def question_bank(mtime: float) -> list[dict]:
    if not QUESTION_BANK.exists():
        return []
    return json.loads(QUESTION_BANK.read_text())


@st.cache_resource
def _executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=4)


def _post_chat(
    client: httpx.Client, token: str, prompt: str, history: list[dict], tone: str
) -> dict:
    resp = client.post(
        f"{API_URL}/chat",
        json={"query": prompt, "history": history, "tone": tone},
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    return resp.json()


def _md(text: str) -> str:
    # st.markdown treats $…$ as LaTeX math — "$99 … $79" would render as a
    # garbled formula. Escape dollars in all model/user text before display.
    return text.replace("$", "\\$")


def _abort_inflight() -> None:
    inflight = st.session_state.pop("inflight", None)
    if inflight:
        inflight["client"].close()


users = demo_users()
with st.sidebar:
    st.title("📚 Knowledge Assistant")
    selected = st.selectbox("Sign in as", list(users))
    token = users[selected]["token"]
    user_roles = set(users[selected]["roles"])
    if st.session_state.get("active_user") != selected:
        # Switching user clears history — no cross-role context bleed.
        previous = st.session_state.get("active_user")
        st.session_state["active_user"] = selected
        st.session_state["messages"] = []
        _abort_inflight()
        if previous is not None:
            st.session_state["notice"] = (
                f"Signed in as {selected} — previous chat history was cleared."
            )
    if st.button(
        "🗑️ Clear chat history",
        width="stretch",
        disabled=not st.session_state.get("messages"),
    ):
        st.session_state["messages"] = []
        _abort_inflight()
        st.session_state["notice"] = "Chat history cleared."
        st.rerun()
    st.caption("POC login: selecting a user sets the bearer token for every request.")

    TONE_OPTIONS = {
        "Business professional (default)": "professional",
        "Friendly": "friendly",
        "Concise": "concise",
    }
    tone = TONE_OPTIONS[
        st.selectbox("Response style", list(TONE_OPTIONS), help="How answers are worded — grounding and citations are unaffected.")
    ]

    mtime = QUESTION_BANK.stat().st_mtime if QUESTION_BANK.exists() else 0.0
    bank = question_bank(mtime)
    suggestions = [q["question"] for q in bank if set(q["access_roles"]) & user_roles]
    if suggestions:
        with st.expander(
            f"💡 Suggested questions ({len(suggestions)} of {len(bank)} — filtered to your access)"
        ):
            picked = st.selectbox(
                "From the question bank", suggestions, label_visibility="collapsed"
            )
            if st.button(
                "Ask this question",
                width="stretch",
                disabled=bool(st.session_state.get("inflight")),
            ):
                st.session_state["queued_prompt"] = picked
                st.rerun()
            st.caption(
                "Generated from documents your roles can read. Selecting asks it "
                "directly — Streamlit's chat box can't be pre-filled."
            )

if notice := st.session_state.pop("notice", None):
    st.toast(notice, icon="🧹")
    st.info(f"🧹 {notice}")

# Non-domain exchanges never enter the history sent to the API — they can't
# derail the intent-stage rewrite of later questions.
HISTORY_EXCLUDED_KINDS = {"out_of_domain", "greeting", "refused", "error"}


def build_history(messages: list[dict]) -> list[dict]:
    history: list[dict] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("kind") in HISTORY_EXCLUDED_KINDS:
            if history and history[-1]["role"] == "user":
                history.pop()  # also drop the user turn that triggered it
            continue
        turn = {"role": m["role"], "content": m["text"]}
        if m["role"] == "assistant" and m.get("kind"):
            turn["kind"] = m["kind"]
        history.append(turn)
    return history[-6:]


KIND_BADGE = {
    "no_result": ("ℹ️", "No accessible information"),
    "refused": ("🚫", "Refused"),
    "clarify": ("❓", "Needs clarification"),
    "error": ("⚠️", "Error"),
    "out_of_domain": ("🧭", "Out of scope"),
}


def render_answer(msg: dict) -> None:
    kind = msg.get("kind", "answered")
    if kind in KIND_BADGE:
        icon, label = KIND_BADGE[kind]
        st.markdown(f"{icon} *{label}*")
    for flag in msg.get("flags", []):
        if flag == "conflict":
            st.warning("Sources disagree on this — both values shown with their dates.")
        elif flag == "stale_source":
            st.warning("Part of this answer cites an archived document; a newer version exists.")
    st.markdown(_md(msg["text"]))
    citations = msg.get("citations", [])
    if citations:
        with st.expander(f"📎 {len(citations)} citation(s)"):
            for c in citations:
                st.markdown(
                    f"**{c['title']}** — page {c['page']} · {c['period']} · "
                    f"{c['source']} · {c['status']}"
                    + (f" · superseded by `{c['superseded_by']}`" if c.get("superseded_by") else "")
                )
                st.markdown(f"> {_md(c['quote'])}")
    meta = msg.get("meta")
    if meta:
        stages = " · ".join(
            f"{k.removesuffix('_ms')} {v:.0f}ms" for k, v in meta["stage_breakdown"].items()
        )
        st.caption(
            f"\\${meta['cost_usd']:.4f} · {meta['latency_ms']:.0f} ms · "
            f"{meta['llm_calls']} LLM calls · {meta['prompt_tokens']}+{meta['completion_tokens']} tok"
            + (f" · {stages}" if stages else "")
            + f" · trace `{meta.get('trace_id', '-')}` — details on the 🔍 Request Details page"
        )


for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_answer(msg)
        else:
            st.markdown(_md(msg["text"]))

inflight = st.session_state.get("inflight")
if inflight:
    future = inflight["future"]
    if future.done():
        st.session_state.pop("inflight", None)
        try:
            answer = future.result()
        except Exception as exc:
            if inflight.get("stopped"):
                answer = {
                    "kind": "error",
                    "text": "⏹ Generation stopped by user.",
                    "citations": [],
                    "flags": [],
                }
            else:
                answer = {
                    "kind": "error",
                    "text": f"API unreachable: {exc}",
                    "citations": [],
                    "flags": [],
                }
        st.session_state["messages"].append({"role": "assistant", **answer})
        st.rerun()
    else:
        with st.chat_message("assistant"):
            st.markdown("_Searching accessible knowledge…_")
            if st.button("⏹ Stop generation"):
                inflight["stopped"] = True
                inflight["client"].close()  # disconnect → server cancels the request
        time.sleep(0.4)
        st.rerun()

prompt = st.chat_input("Ask about internal knowledge…", disabled=bool(inflight))
if not prompt and not inflight:
    prompt = st.session_state.pop("queued_prompt", None)
if prompt:
    st.session_state["messages"].append({"role": "user", "text": prompt})
    history = build_history(st.session_state["messages"][:-1])
    client = httpx.Client(timeout=120)
    st.session_state["inflight"] = {
        "future": _executor().submit(_post_chat, client, token, prompt, history, tone),
        "client": client,
    }
    st.rerun()
