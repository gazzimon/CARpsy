#!/usr/bin/env python3
"""
10_baseline_vs_golden.py

Tests whether the BASE model (no fine-tune) already produces
acceptable responses, to determine if fine-tuning is redundant.

Requires: Ollama running locally with qwen3:0.6b pulled.
  ollama pull qwen3:0.6b

Usage:
  python scripts/10_baseline_vs_golden.py [--n 20] [--model qwen3:0.6b]
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("[!] pip install requests")
    sys.exit(1)

REPO_ROOT  = Path(__file__).resolve().parent.parent
TEST_SPLIT = REPO_ROOT / "data" / "splits" / "test.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def call_ollama(model: str, system: str, user: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "top_k": 1, "num_predict": 400},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] {e}"


def extract_dtcs(text: str) -> set[str]:
    return set(re.findall(r'[PBCU]\d{4}', text.upper()))


def keyword_overlap(pred: str, gold: str) -> float:
    """Jaccard overlap on words (lowercased, ignoring stopwords)."""
    stopwords = {"a", "an", "the", "is", "it", "to", "and", "or", "of",
                 "in", "for", "on", "with", "this", "that", "be", "are",
                 "has", "have", "can", "will", "not", "no", "do", "i",
                 "my", "what", "how", "code", "fault", "your"}
    pred_words = {w for w in re.findall(r'\w+', pred.lower()) if w not in stopwords}
    gold_words = {w for w in re.findall(r'\w+', gold.lower()) if w not in stopwords}
    if not pred_words and not gold_words:
        return 1.0
    if not pred_words or not gold_words:
        return 0.0
    return len(pred_words & gold_words) / len(pred_words | gold_words)


def dtc_recall(pred: str, gold: str, user: str) -> float:
    """Did the base model mention the same DTCs present in the golden answer?"""
    gold_dtcs = extract_dtcs(gold) | extract_dtcs(user)
    if not gold_dtcs:
        return 1.0
    pred_dtcs = extract_dtcs(pred)
    return len(pred_dtcs & gold_dtcs) / len(gold_dtcs)


def length_ratio(pred: str, gold: str) -> float:
    """How close is the response length to the golden answer?"""
    pred_len = len(pred.split())
    gold_len = len(gold.split())
    if gold_len == 0:
        return 1.0
    return min(pred_len, gold_len) / max(pred_len, gold_len)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=20,          help="Number of test examples to evaluate")
    parser.add_argument("--model", type=str, default="qwen3:0.6b", help="Ollama model tag for the base model")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    if not TEST_SPLIT.exists():
        print(f"[!] Test split not found: {TEST_SPLIT}")
        print("    Run: python scripts/03_split_dataset.py")
        sys.exit(1)

    examples = load_jsonl(TEST_SPLIT)
    random.seed(args.seed)
    sample = random.sample(examples, min(args.n, len(examples)))

    print(f"\n{'='*65}")
    print(f"  Baseline evaluation: {args.model}  ({len(sample)} samples)")
    print(f"{'='*65}\n")

    results = []
    for i, ex in enumerate(sample, 1):
        msgs     = ex["messages"]
        system   = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user     = next((m["content"] for m in msgs if m["role"] == "user"),   "")
        gold     = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        pred = call_ollama(args.model, system, user)

        overlap = keyword_overlap(pred, gold)
        recall  = dtc_recall(pred, gold, user)
        lratio  = length_ratio(pred, gold)
        # composite score
        score   = 0.5 * overlap + 0.3 * recall + 0.2 * lratio

        results.append({
            "overlap": overlap,
            "dtc_recall": recall,
            "length_ratio": lratio,
            "score": score,
        })

        print(f"[{i:02d}/{len(sample)}] score={score:.2f}  overlap={overlap:.2f}  dtc_recall={recall:.2f}  len_ratio={lratio:.2f}")
        print(f"  USER:  {user[:90]}")
        print(f"  GOLD:  {gold[:120]}")
        print(f"  BASE:  {pred[:120]}")
        print()

    # aggregate
    avg = lambda key: sum(r[key] for r in results) / len(results)
    print(f"{'='*65}")
    print(f"  AGGREGATE  (n={len(results)})")
    print(f"{'='*65}")
    print(f"  Avg keyword overlap : {avg('overlap'):.2%}")
    print(f"  Avg DTC recall      : {avg('dtc_recall'):.2%}")
    print(f"  Avg length ratio    : {avg('length_ratio'):.2%}")
    print(f"  Avg composite score : {avg('score'):.2%}")
    print()

    threshold = 0.55
    if avg("score") >= threshold:
        print(f"  VERDICT: Base model scores >= {threshold:.0%} -> fine-tune may be REDUNDANT.")
        print(f"           The system prompt alone might be sufficient.")
    else:
        print(f"  VERDICT: Base model scores < {threshold:.0%} -> fine-tune adds real value.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
