#!/usr/bin/env python3
"""
05_evaluate_adapter.py  Structural evaluation of the fine-tuned LoRA adapter.

Analyzes the test split to assess:
  - DTC code identification accuracy
  - Critical severity detection rate
  - Response structure quality (length, format)
  - Manual side-by-side sample inspection

Note: This is a structural evaluation (response format).
      For real inference quality, use 06_validate_adapter.py.

Metrics:
  - Code identification rate
  - Critical response rate
  - Avg response length
  - Structure rate
"""

import sys
import json
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml

REPO_ROOT  = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
OUTPUT_DIR = REPO_ROOT / "output" / "adapter"
CONFIG_PATH = REPO_ROOT / "configs" / "lora_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def extract_code_from_user_message(user_msg: str) -> Optional[str]:
    """Extract a DTC code from the user message (e.g. 'P0420')."""
    import re
    match = re.search(r'[PBCU]\d{4}', user_msg.upper())
    return match.group(0) if match else None


def extract_code_from_assistant(assistant_msg: str) -> Optional[str]:
    """Extract a DTC code from the assistant response."""
    import re
    match = re.search(r'[PBCU]\d{4}', assistant_msg.upper())
    return match.group(0) if match else None


def contains_critical_indicator(text: str) -> bool:
    """Check whether the response flags a critical issue."""
    indicators = ["critical", "immediate", "not safe", "serious", "urgent", "[WARN]"]
    return any(ind in text.lower() for ind in indicators)


def evaluate_example(example: dict) -> dict:
    """Evaluate a single dataset example."""
    messages     = example["messages"]
    user_msg      = next((m["content"] for m in messages if m["role"] == "user"),      "")
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

    user_code      = extract_code_from_user_message(user_msg)
    assistant_code = extract_code_from_assistant(assistant_msg)

    return {
        "has_code_in_user":      bool(user_code),
        "has_code_in_assistant": bool(assistant_code),
        "code_match":            bool(user_code and assistant_code and user_code == assistant_code),
        "critical_detected":     contains_critical_indicator(assistant_msg),
        "response_length":       len(assistant_msg.split()),
        "has_structure":         (":" in assistant_msg or "\n" in assistant_msg),
    }


def generate_report(results: list[dict], total: int) -> dict:
    if not results:
        return {"error": "No results to evaluate"}

    return {
        "total_evaluated":          total,
        "code_identification_rate": sum(r["code_match"]        for r in results) / max(len(results), 1),
        "critical_response_rate":   sum(r["critical_detected"] for r in results) / max(len(results), 1),
        "avg_response_length":      sum(r["response_length"]   for r in results) / max(len(results), 1),
        "structure_rate":           sum(r["has_structure"]     for r in results) / max(len(results), 1),
        "samples_with_code_in_user":      sum(r["has_code_in_user"]      for r in results),
        "samples_with_code_in_assistant": sum(r["has_code_in_assistant"] for r in results),
    }


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print("[stats] EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total samples evaluated: {report['total_evaluated']}")
    print(f"\n  [chart] Metrics:")
    print(f"     Code identification rate:  {report['code_identification_rate']:.1%}")
    print(f"     Critical response rate:    {report['critical_response_rate']:.1%}")
    print(f"     Avg response length:       {report['avg_response_length']:.0f} words")
    print(f"     Structure rate:            {report['structure_rate']:.1%}")
    print(f"  [note] Coverage:")
    print(f"     Samples with code in user:      {report['samples_with_code_in_user']}")
    print(f"     Samples with code in assistant: {report['samples_with_code_in_assistant']}")
    print("=" * 60)


def show_samples(examples: list[dict], n: int = 3) -> None:
    print("\n" + "=" * 60)
    print("[note] SAMPLE INSPECTION")
    print("=" * 60)

    samples = random.sample(examples, min(n, len(examples)))
    for i, example in enumerate(samples, 1):
        messages      = example["messages"]
        user_msg      = next((m["content"] for m in messages if m["role"] == "user"),      "")
        assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        print(f"\n   Sample #{i} ")
        print(f"   USER:      {user_msg}")
        print(f"   ASSISTANT: {assistant_msg}")


def main():
    print("=" * 60)
    print("CARpsy  Step 5: Evaluate Dataset Structure")
    print("=" * 60)

    config = load_config()

    test_path = SPLITS_DIR / "test.jsonl"
    if not test_path.exists():
        print(f"\n  [!] Test split not found: {test_path}")
        print("  [!] Run first: python scripts/03_split_dataset.py")
        return

    print(f"\n[dir] Loading test split: {test_path}")
    test_examples = load_jsonl(test_path)
    print(f"  Total test examples: {len(test_examples)}")

    if not test_examples:
        print("  [!] No examples to evaluate")
        return

    print("\n Evaluating examples...")
    results = [evaluate_example(ex) for ex in test_examples[:1000]]

    report = generate_report(results, len(test_examples))
    print_report(report)
    show_samples(test_examples, 3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "evaluation_report.json"
    report["sample_results"] = results[:10]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to {report_path}")

    print("\n" + "=" * 60)
    print("[list] NOTE ON REAL EVALUATION")
    print("=" * 60)
    print("""
  This is a STRUCTURAL evaluation (response format check).
  For true inference quality you need:

  1. Run live inference with base model + fine-tuned adapter
     using llama-cli or the QVAC SDK

  2. Compare generated vs expected responses using:
     - BLEU score (textual similarity)
     - Severity classification accuracy
     - Human side-by-side evaluation

  3. Test in OBDient with live OBD-II data

  Run: python scripts/06_validate_adapter.py
""")
    print("[OK] Step 5 complete.")


if __name__ == "__main__":
    main()
