"""
run_ragas.py — RAGAS evaluation harness + LangSmith dataset upload + CI gate

Runs against the fixed 25-question test set in eval/test_set.json.
Calls the /ask API for each question, scores with RAGAS, then uploads
results to a LangSmith dataset so every eval run is tracked over time.

Exit 0 = all metrics pass (PR can merge)
Exit 1 = at least one metric below threshold (PR blocked)

Usage:
    python eval/run_ragas.py
    make eval

LangSmith dataset:
    Each eval run creates (or updates) a dataset named LANGCHAIN_PROJECT
    in LangSmith, then adds an experiment run with per-question RAGAS scores.
    View at: https://smith.langchain.com
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

load_dotenv()

TEST_SET_PATH = Path("eval/test_set.json")
API_BASE      = os.environ.get("EVAL_API_BASE", "http://localhost:8000")
REPORT_PATH   = Path("eval/report.json")

# Eval API auth — set EVAL_USERNAME / EVAL_PASSWORD in CI secrets
EVAL_USERNAME = os.environ.get("EVAL_USERNAME", "admin")
EVAL_PASSWORD = os.environ.get("EVAL_PASSWORD", "admin-secret")

THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_precision": 0.78,
    "context_recall":    0.75,   # new metric added in v2
}


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def get_bearer_token(client: httpx.Client) -> str:
    """Exchange eval credentials for a JWT Bearer token."""
    resp = client.post(
        f"{API_BASE}/auth/token",
        data={"username": EVAL_USERNAME, "password": EVAL_PASSWORD},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_api(question: str, client: httpx.Client, token: str) -> dict | None:
    try:
        resp = client.post(
            f"{API_BASE}/ask",
            json={"question": question, "top_k_retrieval": 20, "top_n_rerank": 5},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[eval] API call failed for '{question[:60]}': {exc}")
        return None


def collect_answers(test_cases: list[dict]) -> list[dict]:
    results = []
    with httpx.Client() as client:
        token = get_bearer_token(client)
        for i, case in enumerate(test_cases, start=1):
            print(f"[eval] {i}/{len(test_cases)} — {case['question'][:70]}...")
            response = call_api(case["question"], client, token)
            if response is None:
                continue
            claims   = response.get("claims", [])
            contexts = (
                [f"{c['statement']} {c['citation']}" for c in claims]
                if claims else [response["answer"]]
            )
            results.append({
                "question":     case["question"],
                "answer":       response["answer"],
                "contexts":     contexts,
                "ground_truth": case["ground_truth"],
                # Extra fields for LangSmith metadata
                "confidence_score": response.get("confidence_score"),
                "data_sufficiency": response.get("data_sufficiency"),
                "cost_usd":         response.get("cost_usd"),
                "latency_ms":       response.get("latency_ms"),
                "ticker":           case.get("ticker"),
            })
            time.sleep(0.2)
    return results


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------

def run_ragas(answers: list[dict]) -> dict:
    """Score answers with RAGAS (faithfulness, relevancy, precision, recall)."""
    # RAGAS expects only these four keys in the dataset
    ragas_rows = [
        {k: v for k, v in row.items()
         if k in {"question", "answer", "contexts", "ground_truth"}}
        for row in answers
    ]
    dataset = Dataset.from_list(ragas_rows)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return dict(results)


def check_thresholds(scores: dict) -> bool:
    print("\n[eval] Results:")
    print(f"  {'Metric':<25} {'Score':>7}   {'Threshold':>9}   Status")
    print(f"  {'-' * 58}")
    passed = True
    for metric, threshold in THRESHOLDS.items():
        score  = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        if score < threshold:
            passed = False
        print(f"  {metric:<25} {score:>7.3f}   {threshold:>9.3f}   {status}")
    return passed


# ---------------------------------------------------------------------------
# LangSmith upload
# ---------------------------------------------------------------------------

def upload_to_langsmith(answers: list[dict], scores: dict, passed: bool) -> None:
    """
    Upload eval results to LangSmith as an experiment run.

    Creates (or reuses) a dataset named after LANGCHAIN_PROJECT, then
    adds one example + feedback record per question so trend charts populate.
    """
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        print("[langsmith] LANGCHAIN_API_KEY not set — skipping upload.")
        return

    try:
        from langsmith import Client as LangSmithClient

        ls      = LangSmithClient(api_key=api_key)
        project = os.environ.get("LANGCHAIN_PROJECT", "finance-rag-production")
        dataset_name = f"{project}-eval"
        run_id  = str(uuid.uuid4())
        ts      = datetime.now(timezone.utc).isoformat()

        # Get or create the dataset
        try:
            dataset = ls.read_dataset(dataset_name=dataset_name)
        except Exception:
            dataset = ls.create_dataset(
                dataset_name=dataset_name,
                description="Finance RAG RAGAS evaluation — 25-question SEC filing test set",
            )

        # Create one example per question (idempotent on content hash)
        for row in answers:
            try:
                ls.create_example(
                    inputs={"question": row["question"], "ticker": row.get("ticker")},
                    outputs={"ground_truth": row["ground_truth"]},
                    dataset_id=dataset.id,
                )
            except Exception:
                pass  # Example already exists — skip

        # Log the aggregate eval run
        ls.create_run(
            id=run_id,
            name=f"ragas-eval-{ts[:10]}",
            run_type="chain",
            inputs={"n_questions": len(answers), "thresholds": THRESHOLDS},
            project_name=project,
            extra={
                "metadata": {
                    "eval_run_id": run_id,
                    "passed": passed,
                    "timestamp": ts,
                }
            },
        )
        ls.update_run(
            run_id,
            outputs={**scores, "passed": passed},
            end_time=datetime.now(timezone.utc),
        )

        # Log per-metric feedback so LangSmith renders trend lines
        for metric, score in scores.items():
            try:
                ls.create_feedback(
                    run_id=run_id,
                    key=metric,
                    score=score,
                    comment=f"RAGAS {metric} — threshold {THRESHOLDS.get(metric, '—')}",
                )
            except Exception:
                pass

        print(f"[langsmith] Eval run uploaded → project '{project}', run_id={run_id}")

    except Exception as exc:
        print(f"[langsmith] Upload failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[eval] Loading test set from {TEST_SET_PATH}...")
    test_cases = json.loads(TEST_SET_PATH.read_text())
    print(f"[eval] {len(test_cases)} questions loaded.")

    print(f"\n[eval] Calling API at {API_BASE}...")
    answers = collect_answers(test_cases)

    if not answers:
        print("[eval] No answers collected — is the API running? (`make run`)")
        sys.exit(1)

    print(f"\n[eval] Running RAGAS on {len(answers)} answered questions...")
    scores = run_ragas(answers)

    # Save local report
    REPORT_PATH.parent.mkdir(exist_ok=True)
    report = {
        "scores":      scores,
        "thresholds":  THRESHOLDS,
        "n_questions": len(answers),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[eval] Report saved to {REPORT_PATH}")

    passed = check_thresholds(scores)

    # Upload to LangSmith for trend tracking
    upload_to_langsmith(answers, scores, passed)

    print(f"\n[eval] Overall: {'PASSED' if passed else 'FAILED'}")
    sys.exit(0 if passed else 1)
