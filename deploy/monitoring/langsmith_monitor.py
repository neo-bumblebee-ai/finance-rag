"""
langsmith_monitor.py — LangSmith cost + performance monitoring dashboard

Pulls run data from LangSmith and prints a formatted report:
  - Daily spend breakdown (total, per-user, per-model)
  - Latency percentiles (p50, p95)
  - Confidence score distribution
  - Low-confidence query alert (confidence < 0.6)
  - RAGAS metric trend over the last N eval runs

Usage:
    python deploy/monitoring/langsmith_monitor.py
    python deploy/monitoring/langsmith_monitor.py --days 7 --project finance-rag-production

Schedule this via cron or AWS EventBridge → Lambda for automated daily reports.
"""

from __future__ import annotations

import argparse
import os
import statistics
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()


def get_client():
    from langsmith import Client
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        raise SystemExit("LANGCHAIN_API_KEY not set. Export it or add to .env.")
    return Client(api_key=api_key)


def fetch_runs(client, project: str, days: int) -> list:
    """Fetch all 'llm' runs from the last N days."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    runs = list(client.list_runs(
        project_name=project,
        run_type="llm",
        start_time=since,
    ))
    return runs


def cost_report(runs: list) -> dict:
    """Aggregate cost by user and model."""
    total      = 0.0
    by_user    = {}
    by_model   = {}

    for run in runs:
        meta      = (run.extra or {}).get("metadata", {})
        cost      = (run.outputs or {}).get("cost_usd", 0.0) or 0.0
        user      = meta.get("user_id", "unknown")
        model     = meta.get("model", "unknown")

        total             += cost
        by_user[user]     = by_user.get(user, 0.0)   + cost
        by_model[model]   = by_model.get(model, 0.0) + cost

    return {"total": total, "by_user": by_user, "by_model": by_model}


def latency_report(runs: list) -> dict:
    """Calculate latency percentiles from run metadata."""
    latencies = []
    for run in runs:
        meta = (run.extra or {}).get("metadata", {})
        ms   = meta.get("latency_ms")
        if ms:
            latencies.append(float(ms))

    if not latencies:
        return {"p50": None, "p95": None, "count": 0}

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    return {"p50": round(p50, 1), "p95": round(p95, 1), "count": len(latencies)}


def confidence_report(runs: list) -> dict:
    """Distribution of confidence scores."""
    scores = []
    low_confidence_queries = []

    for run in runs:
        outputs = run.outputs or {}
        score   = outputs.get("confidence_score")
        if score is None:
            continue
        score = float(score)
        scores.append(score)
        if score < 0.6:
            query = (run.inputs or {}).get("query", "—")
            low_confidence_queries.append({"query": query[:80], "score": round(score, 3)})

    if not scores:
        return {"mean": None, "low_confidence_count": 0, "low_confidence_queries": []}

    return {
        "mean":                   round(statistics.mean(scores), 3),
        "median":                 round(statistics.median(scores), 3),
        "low_confidence_count":   len(low_confidence_queries),
        "low_confidence_queries": low_confidence_queries[:10],  # top 10
    }


def fetch_eval_runs(client, project: str, n: int = 5) -> list[dict]:
    """Fetch the last N RAGAS eval runs and return their metric scores."""
    runs = list(client.list_runs(
        project_name=project,
        run_type="chain",
        filter='contains(name, "ragas-eval")',
    ))
    # Sort by start_time descending, take last N
    runs.sort(key=lambda r: r.start_time or datetime.min, reverse=True)
    results = []
    for run in runs[:n]:
        outputs = run.outputs or {}
        results.append({
            "date":               (run.start_time or datetime.min).strftime("%Y-%m-%d"),
            "faithfulness":       round(outputs.get("faithfulness",      0.0), 3),
            "answer_relevancy":   round(outputs.get("answer_relevancy",  0.0), 3),
            "context_precision":  round(outputs.get("context_precision", 0.0), 3),
            "context_recall":     round(outputs.get("context_recall",    0.0), 3),
            "passed":             outputs.get("passed", "—"),
        })
    return results


def print_report(project: str, days: int, runs: list, eval_runs: list[dict]) -> None:
    cost  = cost_report(runs)
    lat   = latency_report(runs)
    conf  = confidence_report(runs)

    print(f"\n{'=' * 60}")
    print(f"  Finance RAG — LangSmith Monitor")
    print(f"  Project : {project}")
    print(f"  Period  : last {days} day(s)")
    print(f"  Runs    : {len(runs)}")
    print(f"{'=' * 60}")

    # Cost
    print(f"\n{'─' * 60}")
    print(f"  COST BREAKDOWN")
    print(f"{'─' * 60}")
    print(f"  Total spend          : ${cost['total']:.4f}")
    print(f"\n  By user:")
    for user, usd in sorted(cost["by_user"].items(), key=lambda x: -x[1]):
        print(f"    {user:<30}  ${usd:.4f}")
    print(f"\n  By model:")
    for model, usd in sorted(cost["by_model"].items(), key=lambda x: -x[1]):
        print(f"    {model:<30}  ${usd:.4f}")

    # Latency
    print(f"\n{'─' * 60}")
    print(f"  LATENCY ({lat['count']} requests)")
    print(f"{'─' * 60}")
    if lat["p50"]:
        print(f"  p50  : {lat['p50']} ms")
        print(f"  p95  : {lat['p95']} ms")
        sla_ok = "OK" if lat["p95"] <= 5000 else "BREACH"
        print(f"  p95 SLA (<=5s) : {sla_ok}")
    else:
        print("  No latency data.")

    # Confidence
    print(f"\n{'─' * 60}")
    print(f"  ANSWER CONFIDENCE")
    print(f"{'─' * 60}")
    if conf["mean"]:
        print(f"  Mean   : {conf['mean']}")
        print(f"  Median : {conf['median']}")
        print(f"  Low-confidence queries (score < 0.6) : {conf['low_confidence_count']}")
        if conf["low_confidence_queries"]:
            print(f"\n  Sample low-confidence queries:")
            for item in conf["low_confidence_queries"][:5]:
                print(f"    [{item['score']}]  {item['query']}")
    else:
        print("  No confidence data.")

    # RAGAS trend
    print(f"\n{'─' * 60}")
    print(f"  RAGAS EVAL TREND (last {len(eval_runs)} runs)")
    print(f"{'─' * 60}")
    if eval_runs:
        header = f"  {'Date':<12} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7}  Passed"
        print(header)
        print(f"  {'-' * 55}")
        for r in eval_runs:
            print(
                f"  {r['date']:<12}"
                f" {r['faithfulness']:>7.3f}"
                f" {r['answer_relevancy']:>7.3f}"
                f" {r['context_precision']:>7.3f}"
                f" {r['context_recall']:>7.3f}"
                f"  {'YES' if r['passed'] else 'NO'}"
            )
    else:
        print("  No eval runs found.")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance RAG LangSmith monitor")
    parser.add_argument("--days",    type=int, default=1,   help="Days of history to pull")
    parser.add_argument("--project", type=str,
                        default=os.environ.get("LANGCHAIN_PROJECT", "finance-rag-production"))
    args = parser.parse_args()

    client     = get_client()
    runs       = fetch_runs(client, args.project, args.days)
    eval_runs  = fetch_eval_runs(client, args.project)
    print_report(args.project, args.days, runs, eval_runs)


if __name__ == "__main__":
    main()
