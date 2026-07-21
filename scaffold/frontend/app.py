"""Streamlit chat UI — thin client over the FastAPI backend."""

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
USERS_FILE = Path(__file__).resolve().parents[1] / "data" / "users.json"

st.set_page_config(page_title="Internal Knowledge Assistant", page_icon="📚")


@st.cache_data
def demo_users() -> dict[str, dict]:
    data = json.loads(USERS_FILE.read_text())
    return {u["name"]: {"token": u["token"], "roles": u["roles"]} for u in data["users"]}


@st.cache_resource
def _executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=4)


def _post_chat(
    client: httpx.Client, token: str, prompt: str, history: list[dict], tone: str, pid: str
) -> dict:
    resp = client.post(
        f"{API_URL}/chat",
        json={"query": prompt, "history": history, "tone": tone, "progress_id": pid},
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    return resp.json()


STAGE_LABELS = {
    "intent": "Understanding the question…",
    "retrieval": "Searching accessible knowledge…",
    "compress": "Compressing the evidence…",
    "generate": "Writing the cited answer…",
}


def _current_stage(pid: str, token: str) -> str:
    try:
        r = httpx.get(
            f"{API_URL}/progress/{pid}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=2,
        )
        return r.json().get("stage", "")
    except Exception:
        return ""


def _md(text: str) -> str:
    # Escape $ so prices don't render as LaTeX.
    return text.replace("$", "\\$")


def _meta_caption(meta: dict) -> None:
    stages = " · ".join(
        f"{k.removesuffix('_ms')} {v:.0f}ms" for k, v in meta["stage_breakdown"].items()
    )
    st.caption(
        f"\\${meta['cost_usd']:.4f} · {meta['latency_ms']:.0f} ms · "
        f"{meta['llm_calls']} LLM calls · {meta['prompt_tokens']}+{meta['completion_tokens']} tok"
        + (f" · {stages}" if stages else "")
        + f" · trace `{meta.get('trace_id', '-')}` — details on the 🔍 Request Details page"
    )


def _abort_inflight() -> None:
    inflight = st.session_state.pop("inflight", None)
    if inflight:
        inflight["client"].close()


users = demo_users()


with st.sidebar:
    st.title("📚 Knowledge Assistant")
    st.caption("Cited, access-aware answers from internal documents.")

    # --- Identity -------------------------------------------------------------
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
    st.caption("Roles: " + " · ".join(sorted(user_roles)))

    TONE_OPTIONS = {
        "Business professional (default)": "professional",
        "Friendly": "friendly",
        "Concise": "concise",
    }
    tone = TONE_OPTIONS[
        st.selectbox("Response style", list(TONE_OPTIONS), help="How answers are worded — grounding and citations are unaffected.")
    ]

    st.divider()

    # --- Housekeeping ---------------------------------------------------------
    if st.button(
        "Clear chat history",
        width="stretch",
        disabled=not st.session_state.get("messages"),
    ):
        st.session_state["messages"] = []
        _abort_inflight()
        st.session_state["notice"] = "Chat history cleared."
        st.rerun()
    st.caption("POC login: selecting a user sets the bearer token for every request.")

if notice := st.session_state.pop("notice", None):
    st.toast(notice, icon="🧹")
    st.info(f"🧹 {notice}")

# Non-domain turns are excluded from API history.
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
        elif flag == "inconsistent_source":
            st.warning("This source's own figures don't reconcile — discrepancy noted in the answer.")
        elif flag == "stale_source":
            st.warning("Part of this answer cites an archived document; a newer version exists.")
    st.markdown(_md(msg["text"]))
    citations = msg.get("citations", [])
    meta = msg.get("meta")
    if citations:
        with st.container(border=True):
            by_doc: dict[str, list[dict]] = {}
            for c in citations:
                by_doc.setdefault(c["doc_id"], []).append(c)
            links = " · ".join(
                f"[{doc_id.split('/')[-1]}](https://{cs[0]['source']}.internal/{doc_id})"
                for doc_id, cs in by_doc.items()
            )
            st.markdown(f"**Sources** — {links}")
            with st.expander("Quotes & request log", expanded=False):
                multi_doc = len(by_doc) > 1
                for doc_id, cites in by_doc.items():
                    stem = doc_id.split("/")[-1].removesuffix(".pdf")
                    for c in cites:
                        for line in c["quote"].splitlines():
                            if line.strip():
                                where = f"{stem} · p.{c['page']}" if multi_doc else f"p.{c['page']}"
                                st.caption(f"{where} — “{_md(line.strip())}”")
                if meta:
                    _meta_caption(meta)
    elif meta:
        _meta_caption(meta)


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
            stage = _current_stage(inflight.get("pid", ""), token)
            st.markdown(f"_{STAGE_LABELS.get(stage, 'Working on it…')}_")
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
    pid = uuid.uuid4().hex[:12]
    st.session_state["inflight"] = {
        "future": _executor().submit(_post_chat, client, token, prompt, history, tone, pid),
        "client": client,
        "pid": pid,
    }
    st.rerun()
