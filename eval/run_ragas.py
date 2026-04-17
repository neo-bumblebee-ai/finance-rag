"""
run_ragas.py - RAGAS evaluation harness + CI gate

Runs against the fixed 25-question test set in eval/test_set.json.
Calls the /ask API for each question, collects answers + retrieved contexts,
then scores with RAGAS metrics.

Exit 0 = all metrics pass (PR can merge)
Exit 1 = at least one metric below threshold (PR blocked)

Usage:
    python eval/run_ragas.py
    make eval
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

load_dotenv()

TEST_SET_PATH = Path("eval/test_set.json")
API_BASE = os.environ.get("EVAL_API_BASE", "http://localhost:8000")
REPORT_PATH = Path("eval/report.json")

THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.78,
}


def call_api(question: str, client: httpx.Client) -> dict | None:
    """Call /ask and return the response dict, or None on failure."""
    try:
        resp = client.post(
            f"{API_BASE}/ask",
            json={"question": question, "top_k_retrieval": 20, "top_n_rerank": 5},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[eval] API call failed for '{question[:60]}...': {e}")
        return None


def collect_answers(test_cases: list[dict]) -> list[dict]:
    """Run each test question through the API and collect answers + contexts."""
    results = []
    with httpx.Client() as client:
        for i, case in enumerate(test_cases, start=1):
            print(f"[eval] {i}/{len(test_cases)} - {case['question'][:70]}...")
            response = call_api(case["question"], client)
            if response is None:
                continue
            # Use structured claims as RAGAS contexts (statement + citation per claim).
            # Falls back to the full answer if claims are absent (e.g. older API version).
            claims = response.get("claims", [])
            contexts = (
                [f"{c['statement']} {c['citation']}" for c in claims]
                if claims
                else [response["answer"]]
            )
            results.append({
                "question": case["question"],
                "answer": response["answer"],
                "contexts": contexts,
                "ground_truth": case["ground_truth"],
            })
            time.sleep(0.2)  # avoid hammering the local API
    return results


def run_ragas(answers: list[dict]) -> dict:
    """Run RAGAS metrics on collected answers."""
    dataset = Dataset.from_list(answers)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return dict(results)


def check_thresholds(scores: dict) -> bool:
    """Print results and return True if all metrics pass."""
    print("\n[eval] Results:")
    print(f"  {'Metric':<25} {'Score':>7}   {'Threshold':>9}   Status")
    print(f"  {'-'*55}")
    passed = True
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        if score < threshold:
            passed = False
        print(f"  {metric:<25} {score:>7.3f}   {threshold:>9.3f}   {status}")
    return passed


if __name__ == "__main__":
    print(f"[eval] Loading test set from {TEST_SET_PATH}...")
    test_cases = json.loads(TEST_SET_PATH.read_text())
    print(f"[eval] {len(test_cases)} questions loaded.")

    print(f"\n[eval] Calling API at {API_BASE}...")
    answers = collect_answers(test_cases)

    if not answers:
        print("[eval] No answers collected - is the API running? (`make run`)")
        sys.exit(1)

    print(f"\n[eval] Running RAGAS on {len(answers)} answered questions...")
    scores = run_ragas(answers)

    # Save report
    REPORT_PATH.parent.mkdir(exist_ok=True)
    report = {"scores": scores, "thresholds": THRESHOLDS, "n_questions": len(answers)}
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n[eval] Report saved to {REPORT_PATH}")

    passed = check_thresholds(scores)
    print(f"\n[eval] Overall: {'PASSED OK' if passed else 'FAILED FAIL'}")
    sys.exit(0 if passed else 1)
