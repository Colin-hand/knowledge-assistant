"""Evaluation dashboard — reads eval/runs/ artifacts written by
`python -m knowledge_assistant.evaluation.evaluate` (retrieval or e2e mode).

Single-series charts throughout (validated series blue); every chart has a
native hover tooltip and a table view alongside.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RUNS_DIR = Path(__file__).resolve().parents[2] / "eval" / "runs"
SERIES = "#3987e5"  # validated categorical slot 1 (dark-surface step)

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 Evaluation Dashboard")


def _load(name: str):
    path = RUNS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


report = _load("report.json")
results = _load("results.json") or []
if report is None:
    st.info(
        "No evaluation run found yet. Generate the question bank, then run:\n\n"
        "`python -m knowledge_assistant.evaluation.evaluate` "
        "(add `--mode e2e --num-questions 10` for the full-app simulation)."
    )
    st.stop()

mode = report.get("mode", "retrieval")
access = report.get("access_control", {})

# ---- Access banner + headline tiles -------------------------------------------------
if access.get("passed"):
    st.success("🔒 Access control: PASSED — zero unauthorized chunks/citations across the sweep.")
else:
    st.error(
        f"🔒 Access control: FAILED — {len(access.get('failures', []))} question(s) leaked. "
        "Details at the bottom."
    )

tiles = st.columns(5)
tiles[0].metric("Mode", mode)
tiles[1].metric(
    "Questions", f"{report.get('n_questions', 0)} / {report.get('pool_size', '?')} pool"
)
if mode == "retrieval":
    tiles[2].metric("Hit-rate@k", f"{report['retrieval']['hit_rate_at_k']:.1%}")
    tiles[3].metric("MRR", f"{report['retrieval']['mrr']:.3f}")
    tiles[4].metric("Reversed relevancy", f"{report['reversed_relevancy_mean']:.3f}")
else:
    tiles[2].metric("Citation hit-rate", f"{report['citation_hit_rate']:.1%}")
    tiles[3].metric("Avg latency", f"{report['avg_latency_ms'] / 1000:.1f} s")
    tiles[4].metric("Answered", str(report.get("kind_distribution", {}).get("answered", 0)))
st.caption(
    f"Run cost ${report.get('cost_usd', 0):.4f} · duration {report.get('duration_ms', 0) / 1000:.0f} s"
    + (f" · sample seed {report['seed']}" if report.get("seed") is not None else "")
)

if not results:
    st.stop()
df = pd.DataFrame(results)
# Normalize artifacts written by older pipeline versions.
df = df.rename(columns={"pass_used": "scope"})
if "scope" not in df.columns:
    df["scope"] = "?"


def _cols(frame: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [c for c in wanted if c in frame.columns]

# ---- Retrieval mode -----------------------------------------------------------------
if mode == "retrieval":
    left, right = st.columns(2)

    with left:
        st.subheader("Hit-rate by document")
        doc_stats = (
            df.groupby("source_doc_id")
            .agg(questions=("hit", "size"), hit_rate=("hit", "mean"),
                 relevancy=("reversed_relevancy", "mean"))
            .reset_index()
            .sort_values("hit_rate")
        )
        st.bar_chart(doc_stats, x="source_doc_id", y="hit_rate", color=SERIES, horizontal=True)
        with st.expander("Table view"):
            st.dataframe(doc_stats, width="stretch", hide_index=True)

    with right:
        st.subheader("Reversed-relevancy distribution")
        bins = pd.cut(df["reversed_relevancy"], [i / 10 for i in range(11)], right=False)
        hist = (
            bins.value_counts().sort_index().rename_axis("relevancy").reset_index(name="questions")
        )
        hist["relevancy"] = hist["relevancy"].astype(str)
        st.bar_chart(hist, x="relevancy", y="questions", color=SERIES)
        st.caption("Per-query mean of per-chunk max similarity (input question vs reversed questions).")

    st.subheader("Retrieval score vs. reversed-question similarity (per chunk)")
    chunk_rows = pd.DataFrame(
        [
            {
                "retrieval score": c["retrieval_score"],
                "reversed max similarity": c["max_similarity"],
                "document": c["doc_id"],
                "chunk": c["chunk_id"],
            }
            for r in results
            for c in r.get("chunks", [])
        ]
    )
    if not chunk_rows.empty:
        st.scatter_chart(chunk_rows, x="retrieval score", y="reversed max similarity", color=SERIES)
        st.caption(
            "Bottom-right = retrieved with confidence but can't answer the question "
            "(noise); score-floor candidates live on the left edge."
        )

    st.subheader("Per-question results")
    table = df[_cols(df, ["question", "source_doc_id", "entitled_user", "scope", "hit",
                          "rank", "reversed_relevancy"])]
    st.dataframe(table, width="stretch", hide_index=True)

    with st.expander("🔎 Question drill-down (retrieved chunks + reversed questions)"):
        picked = st.selectbox("Question", df["question"].tolist())
        row = next(r for r in results if r["question"] == picked)
        scope = row.get("scope", row.get("pass_used", "?"))
        st.markdown(
            f"**Source:** `{row['source_doc_id']}` · **user:** {row['entitled_user']} · "
            f"**scope:** {scope} · **hit:** {row['hit']} (rank {row['rank']}) · "
            f"**relevancy:** {row['reversed_relevancy']:.3f}"
        )
        for c in row.get("chunks", []):
            score = c.get("retrieval_score")
            score_txt = f"{score:.3f}" if score is not None else "n/a"
            st.markdown(
                f"`{c['chunk_id']}` — score {score_txt}, max sim {c['max_similarity']:.3f}"
            )
            for rq, sim in zip(c["reversed_questions"], c["similarities"]):
                st.markdown(f"- {sim:.3f} · {rq}")

# ---- E2E mode -----------------------------------------------------------------------
else:
    left, right = st.columns(2)
    with left:
        st.subheader("Answer kinds")
        kinds = (
            pd.Series(report.get("kind_distribution", {}))
            .rename_axis("kind")
            .reset_index(name="questions")
        )
        st.bar_chart(kinds, x="kind", y="questions", color=SERIES)
    with right:
        st.subheader("Flags raised")
        flags = pd.Series(report.get("flag_counts", {}) or {"(none)": 0})
        flags = flags.rename_axis("flag").reset_index(name="answers")
        st.bar_chart(flags, x="flag", y="answers", color=SERIES)

    st.subheader("Latency & cost per question")
    perf = df[_cols(df, ["question", "latency_ms", "cost_usd", "kind", "citation_hit"])].copy()
    st.bar_chart(perf, x="question", y="latency_ms", color=SERIES, horizontal=True)
    with st.expander("Table view"):
        st.dataframe(perf, width="stretch", hide_index=True)

    with st.expander("🔎 Question drill-down (answers + citations + stages)"):
        picked = st.selectbox("Question", df["question"].tolist())
        row = next(r for r in results if r["question"] == picked)
        st.markdown(
            f"**{row['kind']}** · user {row['entitled_user']} · "
            f"cited: {', '.join(row['cited_docs']) or '—'} · flags: "
            f"{', '.join(row['flags']) or '—'}"
        )
        st.markdown(f"> {row['answer'] or '(no answer text)'}")
        if row.get("stage_breakdown"):
            st.json(row["stage_breakdown"])

# ---- Access failures detail ---------------------------------------------------------
if not access.get("passed"):
    st.subheader("🚨 Access-control failures")
    st.json(access.get("failures", []))
