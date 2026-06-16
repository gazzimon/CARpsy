#!/usr/bin/env python3
"""
03_split_dataset.py  Split the dataset into train/val/test sets.

Default distribution:
  - Train:      80%
  - Validation: 10%
  - Test:       10%
"""

import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
CONFIG_PATH = REPO_ROOT / "configs" / "lora_config.yaml"

RATIOS = {
    "train": 0.80,
    "val":   0.10,
    "test":  0.10,
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_dataset(path: Path) -> list[dict]:
    """Load dataset from a JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def split_dataset(examples: list[dict], ratios: dict[str, float]) -> dict[str, list[dict]]:
    """Shuffle and split examples into train/val/test."""
    shuffled = examples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])

    return {
        "train": shuffled[:train_end],
        "val":   shuffled[train_end:val_end],
        "test":  shuffled[val_end:],
    }


def save_split(examples: list[dict], split_name: str, output_dir: Path) -> None:
    """Save a split as a JSONL file."""
    output_path = output_dir / f"{split_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"  [] {split_name}: {len(examples)} examples  {output_path}")


def main():
    print("=" * 60)
    print("CARpsy  Step 3: Split Dataset")
    print("=" * 60)

    config = load_config()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_files = list(PROCESSED_DIR.glob("*_dataset.jsonl"))
    if not dataset_files:
        print("  [!] No processed dataset found in", PROCESSED_DIR)
        print("  [!] Run first: python scripts/02_prepare_dataset.py")
        return

    dataset_path = dataset_files[0]
    print(f"\n[dir] Loading dataset: {dataset_path}")
    examples = load_dataset(dataset_path)
    print(f"  Total examples: {len(examples)}")

    if len(examples) < 100:
        print("  [WARN]  Small dataset  switching to 70/15/15 split")
        RATIOS["train"] = 0.70
        RATIOS["val"]   = 0.15
        RATIOS["test"]  = 0.15

    print("\n[split]  Splitting dataset...")
    splits = split_dataset(examples, RATIOS)

    print("\n[save] Saving splits...")
    for name, data in splits.items():
        save_split(data, name, SPLITS_DIR)

    print(f"\n[stats] Summary:")
    print(f"  Train: {len(splits['train'])} ({len(splits['train'])/len(examples)*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} ({len(splits['val'])/len(examples)*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} ({len(splits['test'])/len(examples)*100:.1f}%)")
    print(f"  Total: {len(examples)}")

    meta = {
        "dataset_source": str(dataset_path),
        "total_examples": len(examples),
        "splits": {k: len(v) for k, v in splits.items()},
        "ratios": RATIOS,
    }
    meta_path = SPLITS_DIR / "split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved to {meta_path}")

    print("\n[OK] Step 3 complete. Splits ready for fine-tuning.")


if __name__ == "__main__":
    main()
