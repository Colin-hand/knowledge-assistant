"""Evaluation dashboard over eval/runs/ artifacts."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RUNS_DIR = Path(__file__).resolve().parents[2] / "eval" / "runs"
SERIES = "#3987e5"  # series color (validated, dark surface)
MISS = "#d03b3b"  # missed-segment red (validated pair)
SURFACE = "#0e1117"  # chart surface; segment-gap stroke
INK_MUTED = "#c9cad1"

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
    relevancy_mean = report.get("relevancy_score_mean", report.get("reversed_relevancy_mean"))
    tiles[4].metric("Relevancy score", f"{relevancy_mean:.2f} / 5" if relevancy_mean else "—")
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
df = df.rename(columns={"pass_used": "scope", "reversed_relevancy": "relevancy_score"})
if "scope" not in df.columns:
    df["scope"] = "?"


def _cols(frame: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [c for c in wanted if c in frame.columns]

# ---- Retrieval mode -----------------------------------------------------------------
if mode == "retrieval":
    left, right = st.columns(2)

    with left:
        st.subheader("Questions hit vs missed, by document")
        doc_stats = (
            df.groupby("source_doc_id")
            .agg(questions=("hit", "size"), n_hit=("hit", "sum"))
            .reset_index()
        )
        doc_stats["n_missed"] = doc_stats["questions"] - doc_stats["n_hit"]
        doc_stats["hit_rate"] = doc_stats["n_hit"] / doc_stats["questions"]
        doc_stats["doc"] = (
            doc_stats["source_doc_id"].str.split("/").str[-1].str.replace(".pdf", "", regex=False)
        )
        # Worst documents first.
        doc_order = doc_stats.sort_values(["hit_rate", "doc"])["doc"].tolist()
        long = doc_stats.melt(
            id_vars=["doc", "source_doc_id", "questions", "hit_rate"],
            value_vars=["n_hit", "n_missed"],
            var_name="result",
            value_name="n",
        )
        long["result"] = long["result"].map({"n_hit": "hit", "n_missed": "missed"})
        long["stack_order"] = (long["result"] == "missed").astype(int)
        bars = (
            alt.Chart(long)
            .mark_bar(height=14, stroke=SURFACE, strokeWidth=2)
            .encode(
                x=alt.X("n:Q", title="questions", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("doc:N", sort=doc_order, title=None),
                color=alt.Color(
                    "result:N",
                    scale=alt.Scale(domain=["hit", "missed"], range=[SERIES, MISS]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                order=alt.Order("stack_order:Q"),
                tooltip=[
                    alt.Tooltip("source_doc_id:N", title="document"),
                    alt.Tooltip("result:N", title="result"),
                    alt.Tooltip("n:Q", title="questions"),
                    alt.Tooltip("hit_rate:Q", title="hit rate", format=".0%"),
                ],
            )
        )
        labels = (
            alt.Chart(doc_stats)
            .mark_text(align="left", dx=6, color=INK_MUTED)
            .encode(
                x=alt.X("questions:Q"),
                y=alt.Y("doc:N", sort=doc_order, title=None),
                text=alt.Text("label:N"),
            )
            .transform_calculate(label="datum.n_hit + '/' + datum.questions")
        )
        st.altair_chart(
            (bars + labels).properties(height=26 * len(doc_stats) + 20),
            width="stretch",
        )
        st.caption(
            "Bar length = questions asked per source document; red segment = "
            "questions whose source never surfaced (label: hit/total)."
        )

    with right:
        st.subheader("Relevancy score per question")
        vals = pd.to_numeric(df["relevancy_score"], errors="coerce").dropna()
        if not vals.empty:
            mean_v = float(vals.mean())
            median_v = float(vals.median())
            p80_v = float(vals.quantile(0.8))
            bars = (
                alt.Chart(pd.DataFrame({"score": vals}))
                .mark_bar(color=SERIES)
                .encode(
                    x=alt.X("score:Q", bin=alt.Bin(step=1), title="relevancy score (1–5)"),
                    y=alt.Y("count()", title="questions"),
                )
            )
            stats = pd.DataFrame(
                {
                    "value": [mean_v, median_v, p80_v],
                    "stat": [
                        f"mean {mean_v:.2f}",
                        f"median {median_v:.2f}",
                        f"p80 {p80_v:.2f}",
                    ],
                }
            )
            rules = (
                alt.Chart(stats)
                .mark_rule(strokeWidth=2, strokeDash=[6, 3])
                .encode(x=alt.X("value:Q"), color=alt.Color("stat:N", title=""))
            )
            st.altair_chart(bars + rules, width="stretch")
            st.caption(
                "Relevancy score per input question = max 1–5 judge score among "
                "its reversed questions. Headline tile = average of these scores."
            )
        else:
            st.info("No judge scores in this run — re-run the evaluation to populate them.")

    st.subheader("Per-question results")
    table = df[_cols(df, ["question", "source_doc_id", "entitled_user", "scope", "hit",
                          "rank", "relevancy_score"])]
    st.dataframe(table, width="stretch", hide_index=True)

    with st.expander("🔎 Question drill-down (retrieved chunks + reversed questions)"):
        picked = st.selectbox("Question", df["question"].tolist())
        row = next(r for r in results if r["question"] == picked)
        scope = row.get("scope", row.get("pass_used", "?"))
        st.markdown(
            f"**Source:** `{row['source_doc_id']}` · **user:** {row['entitled_user']} · "
            f"**scope:** {scope} · **hit:** {row['hit']} (rank {row['rank']}) · "
            f"**relevancy score:** {row.get('relevancy_score', row.get('reversed_relevancy', 0)):.3f}"
        )
        for c in row.get("chunks", []):
            score = c.get("retrieval_score")
            score_txt = f"{score:.3f}" if score is not None else "n/a"
            st.markdown(
                f"`{c['chunk_id']}` — score {score_txt}, max judge {c.get('max_judge', 0)}/5"
            )
            for rq, js in zip(c["reversed_questions"], c.get("judge_scores", [])):
                st.markdown(f"- {js}/5 · {rq}")

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
