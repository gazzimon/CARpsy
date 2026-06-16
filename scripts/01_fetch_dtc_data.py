#!/usr/bin/env python3
"""
01_fetch_dtc_data.py — Download DTC databases from public repositories.

Sources:
  - Primary:       Wal33D/dtc-database (SQLite with ~28,000 codes from 33 manufacturers)
  - Supplementary: peyo/dtc-and-vin-data (JSON)
"""

import os
import sys
import json
import sqlite3
import shutil
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import requests

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_PATH = REPO_ROOT / "configs" / "lora_config.yaml"

DTC_REPOS = {
    "wal33d": {
        "url": "https://github.com/Wal33D/dtc-database.git",
        "dir": RAW_DIR / "dtc-database",
    },
    "peyo": {
        "url": "https://github.com/peyo/dtc-and-vin-data.git",
        "dir": RAW_DIR / "dtc-and-vin-data",
    },
}


def load_config() -> dict:
    """Load configuration from lora_config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def clone_or_pull_repo(name: str, url: str, dest: Path) -> None:
    """Clone a repo if it doesn't exist, or pull latest if it does."""
    if dest.exists():
        print(f"  [{name}] Repo exists at {dest}, pulling latest...")
        os.system(f'cd /d "{dest}" && git pull 2>&1')
    else:
        print(f"  [{name}] Cloning {url}...")
        os.system(f'git clone --depth 1 "{url}" "{dest}" 2>&1')


def extract_wal33d_database(db_path: Path) -> list[dict]:
    """
    Extract DTCs from the Wal33D SQLite database.
    Typical schema includes: dtc_codes, manufacturers, vehicles.
    """
    if not db_path.exists():
        print(f"  [!] Database not found at {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"  [i] Tables found: {tables}")

    dtc_records = []

    if "dtc_codes" in tables:
        cursor.execute("PRAGMA table_info(dtc_codes);")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"  [i] Columns in dtc_codes: {columns}")

        col_map = {
            "code": next((c for c in columns if c.lower() in ("code", "dtc", "dtc_code", "fault_code")), None),
            "description": next((c for c in columns if c.lower() in ("description", "definition", "desc", "meaning")), None),
            "system": next((c for c in columns if c.lower() in ("system", "category", "type")), None),
            "manufacturer": next((c for c in columns if c.lower() in ("manufacturer", "make", "brand")), None),
        }
        print(f"  [i] Column mapping: {col_map}")

        query_cols = [v for v in col_map.values() if v]
        if query_cols:
            cursor.execute(f"SELECT {', '.join(query_cols)} FROM dtc_codes LIMIT 10")
            sample = cursor.fetchall()
            print(f"  [i] Sample rows: {sample[:3]}")

            cursor.execute(f"SELECT {', '.join(query_cols)} FROM dtc_codes")
            for row in cursor.fetchall():
                record = {}
                for i, col in enumerate(query_cols):
                    record[col] = row[i]
                dtc_records.append(record)

    conn.close()
    print(f"  [✓] Extracted {len(dtc_records)} DTC records from Wal33D")
    return dtc_records


def extract_peyo_data(data_dir: Path) -> list[dict]:
    """Extract DTCs from peyo/dtc-and-vin-data JSON files."""
    records = []

    json_files = list(data_dir.rglob("*.json"))
    dtc_files = [f for f in json_files if "dtc" in f.name.lower()]

    for file_path in dtc_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)

            print(f"  [i] {file_path.name}: {len(data) if isinstance(data, list) else 1} records")
        except Exception as e:
            print(f"  [!] Error reading {file_path}: {e}")

    print(f"  [✓] Extracted {len(records)} records from peyo/dtc-and-vin-data")
    return records


def save_raw_data(records: list[dict], filename: str) -> None:
    """Save extracted records as JSON in data/raw/."""
    output_path = RAW_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  [✓] Saved {len(records)} records to {output_path}")


def generate_summary(all_records: list[dict]) -> dict:
    """Generate a statistical summary of extracted data."""
    systems = set()
    manufacturers = set()

    for rec in all_records:
        for key in rec:
            if "system" in key.lower() and rec[key]:
                systems.add(str(rec[key]))
            if "manufacturer" in key.lower() and rec[key]:
                manufacturers.add(str(rec[key]))

    return {
        "total_records": len(all_records),
        "unique_systems": sorted(systems),
        "unique_manufacturers": sorted(manufacturers),
        "systems_count": len(systems),
        "manufacturers_count": len(manufacturers),
    }


def main():
    print("=" * 60)
    print("CARpsy — Step 1: Fetch DTC Data")
    print("=" * 60)

    config = load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []

    # ─── 1. Wal33D/dtc-database ────────────────────────────────────────────
    print("\n📦 Wal33D/dtc-database")
    repo_info = DTC_REPOS["wal33d"]
    clone_or_pull_repo("wal33d", repo_info["url"], repo_info["dir"])

    db_files = list(repo_info["dir"].rglob("*.db")) + list(repo_info["dir"].rglob("*.sqlite"))
    if db_files:
        records = extract_wal33d_database(db_files[0])
        all_records.extend(records)
        save_raw_data(records, "wal33d_dtc_raw.json")
    else:
        print("  [!] No database file found. Searching for JSON...")
        json_files = list(repo_info["dir"].rglob("*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                print(f"  [i] Loaded {jf.name}: {len(data) if isinstance(data, list) else 1} records")
            except:
                pass

    # ─── 2. peyo/dtc-and-vin-data ──────────────────────────────────────────
    print("\n📦 peyo/dtc-and-vin-data")
    repo_info = DTC_REPOS["peyo"]
    clone_or_pull_repo("peyo", repo_info["url"], repo_info["dir"])

    peyo_records = extract_peyo_data(repo_info["dir"])
    all_records.extend(peyo_records)
    save_raw_data(peyo_records, "peyo_dtc_raw.json")

    # ─── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    summary = generate_summary(all_records)
    print(f"  Total DTC records: {summary['total_records']}")
    print(f"  Systems:           {summary['systems_count']}")
    print(f"  Manufacturers:     {summary['manufacturers_count']}")

    summary_path = RAW_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved to {summary_path}")

    print("\n✅ Step 1 complete. Raw dataset ready for preparation.")


if __name__ == "__main__":
    main()
