#!/usr/bin/env python3
"""
01_fetch_dtc_data.py — Download DTC databases from public repositories.

Sources:
  - Wal33D/dtc-database    : SQLite with ~28,000 codes from 33 manufacturers
  - peyo/dtc-and-vin-data  : Supplementary JSON with DTCs + VIN data
  - OBDex (foerbsnavi)     : 9,533 generic SAE J2012 codes with causes,
                             symptoms, repair difficulty -- CC0 license
"""

import os
import sys
import json
import sqlite3
import requests
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
RAW_DIR     = REPO_ROOT / "data" / "raw"
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

# OBDex YAML files -- CC0 public domain
OBDEX_FILES = [
    "P0xxx_enriched.yaml",
    "P2xxx_enriched.yaml",
    "P3xxx_enriched.yaml",
    "B0xxx_enriched.yaml",
    "C0xxx_enriched.yaml",
    "U0xxx_enriched.yaml",
    "U3xxx_enriched.yaml",
]
OBDEX_BASE_URL = "https://raw.githubusercontent.com/foerbsnavi/OBDex/main/data/generic"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def clone_or_pull_repo(name: str, url: str, dest: Path) -> None:
    """Clone a repo if it doesn't exist, or pull latest if it does."""
    if dest.exists():
        print(f"  [{name}] Repo exists, pulling latest...")
        os.system(f'cd /d "{dest}" && git pull 2>&1')
    else:
        print(f"  [{name}] Cloning {url}...")
        os.system(f'git clone --depth 1 "{url}" "{dest}" 2>&1')


def extract_wal33d_database(db_path: Path) -> list[dict]:
    """Extract DTCs from the Wal33D SQLite database.
    Real schema: dtc_definitions(code, manufacturer, description, type, locale, is_generic, source_file)
    """
    if not db_path.exists():
        print(f"  [!] Database not found at {db_path}")
        return []

    conn   = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type=?", ("table",))
    tables = [row[0] for row in cursor.fetchall()]
    print(f"  [i] Tables: {tables}")

    # Find the DTC table (skip 'statistics')
    target = next((t for t in tables if "dtc" in t.lower() and t != "statistics"), None)
    if not target:
        print("  [!] No DTC table found")
        conn.close()
        return []

    cursor.execute(f"PRAGMA table_info({target})")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"  [i] Table '{target}' columns: {columns}")

    col_code   = next((c for c in columns if c.lower() in ("code", "dtc", "dtc_code", "fault_code")), None)
    col_desc   = next((c for c in columns if c.lower() in ("description", "definition", "desc", "meaning")), None)
    col_mfr    = next((c for c in columns if c.lower() in ("manufacturer", "make", "brand")), None)
    col_type   = next((c for c in columns if c.lower() in ("type", "system", "category")), None)
    col_locale = next((c for c in columns if c.lower() == "locale"), None)

    if not col_code or not col_desc:
        print("  [!] Cannot find required code/description columns")
        conn.close()
        return []

    if col_locale:
        cursor.execute(f"SELECT * FROM {target} WHERE {col_locale} = ?", ("en",))
    else:
        cursor.execute(f"SELECT * FROM {target}")

    col_names   = [d[0] for d in cursor.description]
    dtc_records = []
    for row in cursor.fetchall():
        r      = dict(zip(col_names, row))
        record = {
            "code":        str(r[col_code]).strip().upper(),
            "description": str(r[col_desc]).strip(),
            "source":      "wal33d",
        }
        if col_mfr and r.get(col_mfr):
            record["manufacturer"] = str(r[col_mfr]).strip()
        if col_type and r.get(col_type):
            record["system"] = str(r[col_type]).strip()
        dtc_records.append(record)

    conn.close()
    print(f"  [v] Extracted {len(dtc_records)} records from Wal33D")
    return dtc_records


def extract_peyo_data(data_dir: Path) -> list[dict]:
    """Extract DTCs from peyo/dtc-and-vin-data JSON files."""
    records    = []
    json_files = [f for f in data_dir.rglob("*.json") if "dtc" in f.name.lower()]

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # Tag source
                for item in data:
                    if isinstance(item, dict):
                        item.setdefault("source", "peyo")
                records.extend(data)
            elif isinstance(data, dict):
                data.setdefault("source", "peyo")
                records.append(data)
            print(f"  [i] {file_path.name}: {len(data) if isinstance(data, list) else 1} records")
        except Exception as e:
            print(f"  [!] Error reading {file_path}: {e}")

    print(f"  [v] Extracted {len(records)} records from peyo")
    return records


def fetch_obdex() -> list[dict]:
    """
    Download and parse OBDex enriched YAML files (CC0 license).
    OBDex YAML is a LIST of dicts, each with:
      code, category, title{en,de}, description{en,de},
      common_causes, symptoms, repair, flags, related_codes
    """
    obdex_dir = RAW_DIR / "obdex"
    obdex_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for filename in OBDEX_FILES:
        url   = f"{OBDEX_BASE_URL}/{filename}"
        local = obdex_dir / filename

        if not local.exists():
            print(f"  [download] {filename}...")
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                local.write_bytes(resp.content)
            except Exception as e:
                print(f"  [!] Failed to download {filename}: {e}")
                continue
        else:
            print(f"  [cached] {filename}")

        try:
            with open(local, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"  [!] Failed to parse {filename}: {e}")
            continue

        # OBDex is a LIST of entry dicts
        if not isinstance(data, list):
            print(f"  [!] Unexpected format in {filename}: {type(data)}")
            continue

        for entry in data:
            if not isinstance(entry, dict):
                continue

            code = str(entry.get("code", "")).strip().upper()
            if not code:
                continue

            # Bilingual fields -- prefer English
            desc    = entry.get("description", {})
            desc_en = (desc.get("en") or desc.get("de") or "") if isinstance(desc, dict) else str(desc)

            title    = entry.get("title", {})
            title_en = (title.get("en") or title.get("de") or "") if isinstance(title, dict) else str(title)

            description = desc_en or title_en
            if not description:
                continue

            # Common causes (field may be "common_causes" or "causes")
            causes_raw = entry.get("common_causes") or entry.get("causes") or []
            causes     = []
            for c in (causes_raw if isinstance(causes_raw, list) else []):
                if isinstance(c, dict):
                    n    = c.get("name", {})
                    n_en = (n.get("en") or n.get("de") or "") if isinstance(n, dict) else str(n)
                    pct  = c.get("likelihood") or c.get("probability") or ""
                    label = "{} ({}%)".format(n_en, pct) if pct else n_en
                    if label:
                        causes.append(label)
                elif isinstance(c, str) and c:
                    causes.append(c)

            # Symptoms
            syms_raw = entry.get("symptoms", [])
            symptoms = []
            for s in (syms_raw if isinstance(syms_raw, list) else []):
                if isinstance(s, dict):
                    sn   = s.get("name", {}) or s.get("description", {})
                    s_en = (sn.get("en") or sn.get("de") or "") if isinstance(sn, dict) else str(sn)
                    if s_en:
                        symptoms.append(s_en)
                elif isinstance(s, str) and s:
                    symptoms.append(s)

            repair = entry.get("repair", {})
            repair_difficulty = repair.get("difficulty", "") if isinstance(repair, dict) else ""

            all_records.append({
                "code":              code,
                "description":       description.strip(),
                "causes":            causes,
                "symptoms":          symptoms,
                "repair_difficulty": repair_difficulty,
                "related_codes":     entry.get("related_codes", []),
                "source":            "obdex",
            })

    print(f"  [v] Extracted {len(all_records)} enriched records from OBDex")
    return all_records


def save_raw_data(records: list[dict], filename: str) -> None:
    output_path = RAW_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  [v] Saved {len(records)} records -> {output_path}")


def generate_summary(all_records: list[dict]) -> dict:
    systems       = set()
    manufacturers = set()
    sources       = {}

    for rec in all_records:
        src = rec.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        for key in rec:
            if "system" in key.lower() and rec[key]:
                systems.add(str(rec[key]))
            if "manufacturer" in key.lower() and rec[key]:
                manufacturers.add(str(rec[key]))

    return {
        "total_records":        len(all_records),
        "by_source":            sources,
        "unique_systems":       sorted(systems),
        "unique_manufacturers": sorted(manufacturers),
    }


def main():
    print("=" * 60)
    print("CARpsy - Step 1: Fetch DTC Data")
    print("=" * 60)

    load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []

    # ─── 1. Wal33D/dtc-database ────────────────────────────────────────────
    print("\n[1/3] Wal33D/dtc-database (~28,000 codes, 33 manufacturers)")
    repo = DTC_REPOS["wal33d"]
    clone_or_pull_repo("wal33d", repo["url"], repo["dir"])

    db_files = list(repo["dir"].rglob("*.db")) + list(repo["dir"].rglob("*.sqlite"))
    if db_files:
        # Prefer the main data DB, not the android asset copy
        main_db = next((d for d in db_files if "android" not in str(d)), db_files[0])
        records = extract_wal33d_database(main_db)
        all_records.extend(records)
        save_raw_data(records, "wal33d_dtc_raw.json")
    else:
        print("  [!] No SQLite file found, searching JSON...")
        for jf in repo["dir"].rglob("*.json"):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                    print(f"  [i] Loaded {jf.name}: {len(data)} records")
            except Exception:
                pass

    # ─── 2. peyo/dtc-and-vin-data ──────────────────────────────────────────
    print("\n[2/3] peyo/dtc-and-vin-data")
    repo = DTC_REPOS["peyo"]
    clone_or_pull_repo("peyo", repo["url"], repo["dir"])
    peyo_records = extract_peyo_data(repo["dir"])
    all_records.extend(peyo_records)
    save_raw_data(peyo_records, "peyo_dtc_raw.json")

    # ─── 3. OBDex ──────────────────────────────────────────────────────────
    print("\n[3/3] OBDex - enriched SAE J2012 codes (CC0)")
    obdex_records = fetch_obdex()
    all_records.extend(obdex_records)
    save_raw_data(obdex_records, "obdex_dtc_raw.json")

    # ─── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = generate_summary(all_records)
    print(f"  Total records: {summary['total_records']}")
    for src, count in summary["by_source"].items():
        print(f"    [{src}] {count}")
    print(f"  Systems:       {len(summary['unique_systems'])}")
    print(f"  Manufacturers: {len(summary['unique_manufacturers'])}")

    summary_path = RAW_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved to {summary_path}")
    print("\n[OK] Step 1 complete. Raw dataset ready for preparation.")


if __name__ == "__main__":
    main()
