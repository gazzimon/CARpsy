#!/usr/bin/env python3
"""
02_prepare_dataset.py  Transform raw DTCs into chat format (system/user/assistant).

Converts raw DTC records into ChatML-style conversations for supervised
fine-tuning with QVAC Fabric (qvac-fabric-llm.cpp).

Output format: JSONL  one training example per line.
  {"messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}

Sources used:
  - wal33d: 18,805 records — code + description + manufacturer (34 makes)
  - peyo:   3,374 records  — code + description + manufacturer

Sources excluded:
  - obdex: 9,533 records — causes/symptoms fields were empty (scraping failure)
"""

import sys
import json
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml

REPO_ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR       = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CONFIG_PATH   = REPO_ROOT / "configs" / "lora_config.yaml"

# Sources excluded from training — data quality issues documented above
EXCLUDED_SOURCES = {"obdex"}

# Minimum description length to be considered useful
MIN_DESC_LENGTH = 20

# A description is "rich" (contains inline causes) when it has 3+ comma/semicolon
# separated clauses and is long enough to be a real causes list
RICH_DESC_MIN_LENGTH  = 80
RICH_DESC_MIN_CLAUSES = 3

#  System prompt (matches OBDient's runtime prompt)
SYSTEM_PROMPT = (
    "You are OBDient, an expert automotive diagnostic assistant. "
    "You receive OBD-II fault codes and vehicle data. "
    "Explain what each code means, its severity, and recommended actions. "
    "Always respond in English, clearly and concisely. "
    "Maximum 3 sentences. Prioritize safety. "
    "If something is urgent, indicate it clearly."
)

#  User question templates — single-code only (no {code2} placeholder)
USER_TEMPLATES = [
    "I'm getting code {code} on my {make} {model}. What does it mean?",
    "My check engine light is on with code {code}. Should I be worried?",
    "What is {code} and how serious is it?",
    "Code {code} appeared on my scanner. What should I check?",
    "Can you explain fault code {code} for a {make} {model}?",
    "Is code {code} critical? I drive a {make} {model}.",
    "What does DTC {code} mean in plain English?",
    "My car shows {code}. Do I need to go to a mechanic?",
    "Code {code} on {make} {model} {year}. What are the common causes?",
    "Reading code {code} on my vehicle. What repairs might be needed?",
    "Check engine  {code}. How urgent is this?",
    "Fault code {code} detected. Your advice?",
    "What's the meaning of OBD-II code {code}?",
    "Got {code} after engine light came on. Diagnosis?",
    "{make} {model} showing {code}. Is it safe to drive?",
]

MAX_TOKENS_PER_EXAMPLE = 512


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_raw_data() -> list[dict]:
    """Load DTCs from raw JSON files, skipping excluded sources."""
    all_records = []
    for json_file in sorted(RAW_DIR.glob("*_raw.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue

            # Filter by source field; fall back to filename prefix
            source_name = json_file.stem.replace("_dtc_raw", "").replace("_raw", "")
            kept = []
            for record in data:
                src = record.get("source", source_name).lower()
                if src in EXCLUDED_SOURCES:
                    continue
                kept.append(record)

            all_records.extend(kept)
            skipped = len(data) - len(kept)
            print(f"  [i] {json_file.name}: {len(kept)} kept, {skipped} excluded")
        except Exception as e:
            print(f"  [!] Error loading {json_file}: {e}")
    return all_records


def is_rich_description(text: str) -> bool:
    """Return True when the description looks like an inline causes list.

    Wal33d sometimes stores a comma/semicolon-separated list of probable
    causes in the description field instead of a plain code definition.
    We detect this so the response can format them as causes rather than
    treating the whole blob as a one-liner definition.
    """
    if len(text) < RICH_DESC_MIN_LENGTH:
        return False
    clauses = [c.strip() for c in text.replace(";", ",").split(",") if c.strip()]
    return len(clauses) >= RICH_DESC_MIN_CLAUSES


def normalize_record(record: dict) -> Optional[dict]:
    """Normalize a raw DTC record to a standard schema."""
    normalized = {}

    for key in ("code", "dtc", "dtc_code", "fault_code", "Code", "DTC"):
        if key in record and record[key]:
            normalized["code"] = str(record[key]).strip().upper()
            break

    for key in ("description", "definition", "desc", "meaning", "Description", "Definition"):
        if key in record and record[key]:
            desc = str(record[key]).strip()
            if len(desc) >= MIN_DESC_LENGTH:
                normalized["description"] = desc
            break

    if "code" not in normalized or "description" not in normalized:
        return None

    for key in ("system", "category", "type", "System"):
        if key in record and record[key]:
            normalized["system"] = str(record[key]).strip()
            break
    else:
        code = normalized["code"]
        if code.startswith(("P0", "P1", "P2", "P3")):
            normalized["system"] = "Powertrain"
        elif code.startswith("B"):
            normalized["system"] = "Body"
        elif code.startswith("C"):
            normalized["system"] = "Chassis"
        elif code.startswith("U"):
            normalized["system"] = "Network"
        else:
            normalized["system"] = "Generic"

    for key in ("manufacturer", "make", "brand", "Manufacturer"):
        if key in record and record[key]:
            make = str(record[key]).strip()
            if make.upper() not in ("GENERIC", "OTHER", ""):
                normalized["make"] = make
            break

    for key in ("model", "Model"):
        if key in record and record[key]:
            normalized["model"] = str(record[key]).strip()
            break

    for key in ("year", "Year"):
        if key in record and record[key]:
            normalized["year"] = str(record[key]).strip()
            break

    normalized["source"] = record.get("source", "unknown")
    return normalized


def determine_severity(code: str) -> str:
    """Determine DTC severity. Logic mirrors OBDient's dtcParser.ts."""
    code_upper = code.upper()
    numeric_part = ""
    for c in code_upper[1:]:
        if c.isdigit() or c.upper() in "ABCDEF":
            numeric_part += c

    try:
        numeric = int(numeric_part, 16) if numeric_part else 0
    except ValueError:
        return "warning"

    if code_upper.startswith("P"):
        if 0x0100 <= numeric <= 0x0199:
            return "critical"
        if 0x0300 <= numeric <= 0x0399:
            return "critical"
        if 0x0A00 <= numeric <= 0x0AFF:
            return "critical"
        return "warning"
    if code_upper.startswith(("B", "C", "U")):
        return "info"
    return "warning"


def generate_assistant_response(code: str, description: str, severity: str) -> str:
    """Generate the assistant response for a given DTC.

    When the description contains an inline causes list (rich description),
    the response separates the severity sentence from the causes. Otherwise
    a plain template is used.
    """
    if severity == "critical":
        severity_line = "Do not drive until diagnosed — this is a serious issue."
    elif severity == "warning":
        severity_line = "Schedule an inspection soon to prevent further damage."
    else:
        severity_line = "Monitor for related symptoms; may be intermittent."

    if is_rich_description(description):
        # Description is actually a causes list — present it as such
        clauses = [c.strip() for c in description.replace(";", ",").split(",") if c.strip()]
        top_causes = "; ".join(clauses[:3])
        if severity == "critical":
            return f"[WARN] **CRITICAL** {code}: Common causes include {top_causes}. {severity_line}"
        return f"{code}: Common causes include {top_causes}. {severity_line}"
    else:
        # Standard: definition + severity action
        if severity == "critical":
            return f"[WARN] **CRITICAL** {code}: {description}. {severity_line}"
        return f"{code}: {description}. {severity_line}"


def generate_user_question(code: str, make: str = "Generic", model: str = "Vehicle",
                           year: str = "2020") -> str:
    """Generate a varied user question from templates."""
    template = random.choice(USER_TEMPLATES)
    return template.format(
        code=code,
        make=make if make else "Generic",
        model=model if model else "Vehicle",
        year=year if year else "2020",
    )


def create_chat_example(normalized: dict) -> Optional[dict]:
    """Create a chat-format example from a normalized DTC record."""
    code        = normalized["code"]
    description = normalized["description"]
    make        = normalized.get("make", "Generic")
    model       = normalized.get("model", "Vehicle")
    year        = normalized.get("year", "2020")
    severity    = determine_severity(code)

    user_msg      = generate_user_question(code, make, model, year)
    assistant_msg = generate_assistant_response(code, description, severity)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def create_multi_code_example(codes: list[dict]) -> Optional[dict]:
    """Create a training example with multiple simultaneous DTC codes."""
    if len(codes) < 2:
        return None

    code_strs = [c["code"] for c in codes]
    user_msg = (
        f"I have multiple fault codes: {', '.join(code_strs)}. "
        f"Can you tell me what they mean and if they're related?"
    )

    assistant_lines = []
    for c in codes:
        sev = determine_severity(c["code"])
        desc = c["description"]
        if is_rich_description(desc):
            clauses = [x.strip() for x in desc.replace(";", ",").split(",") if x.strip()]
            desc = "; ".join(clauses[:2])
        prefix = "[WARN] CRITICAL — " if sev == "critical" else ""
        assistant_lines.append(f"{prefix}{c['code']}: {desc}")
    assistant_msg = "\n".join(assistant_lines)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def deduplicate(examples: list[dict]) -> tuple[list[dict], int]:
    """Remove duplicate examples based on the full assistant response."""
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        assistant_msg = next((m["content"] for m in ex["messages"] if m["role"] == "assistant"), "")
        if assistant_msg not in seen:
            seen.add(assistant_msg)
            unique.append(ex)
    return unique, len(examples) - len(unique)


def filter_by_token_limit(examples: list[dict], max_tokens: int) -> tuple[list[dict], int]:
    valid: list[dict] = []
    skipped = 0
    for ex in examples:
        total = sum(estimate_tokens(m["content"]) for m in ex["messages"])
        if total <= max_tokens:
            valid.append(ex)
        else:
            skipped += 1
    return valid, skipped


def save_dataset(examples: list[dict], filename: str) -> None:
    output_path = PROCESSED_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"  [] Saved {len(examples)} examples to {output_path}")


def main():
    print("=" * 60)
    print("CARpsy  Step 2: Prepare Dataset (clean build)")
    print("=" * 60)
    print(f"  Excluded sources: {', '.join(sorted(EXCLUDED_SOURCES))}")
    print(f"  Min description length: {MIN_DESC_LENGTH} chars")

    config = load_config()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[dir] Loading raw data...")
    raw_records = load_raw_data()
    print(f"  Total raw records loaded: {len(raw_records)}")

    print("\n[proc] Normalizing records...")
    normalized = []
    skipped_norm = 0
    for record in raw_records:
        n = normalize_record(record)
        if n:
            normalized.append(n)
        else:
            skipped_norm += 1
    print(f"  Normalized:  {len(normalized)}")
    print(f"  Skipped (missing/short code or description): {skipped_norm}")

    # Report rich descriptions
    rich_count = sum(1 for n in normalized if is_rich_description(n["description"]))
    print(f"  Rich descriptions (inline causes): {rich_count}")

    print("\n Generating single-code chat examples...")
    max_ex = config["dataset"].get("max_examples", 20000)
    single_examples = []
    for n in normalized[:max_ex]:
        example = create_chat_example(n)
        if example:
            single_examples.append(example)
    print(f"  Generated: {len(single_examples)} examples")

    print("\n[link] Generating multi-code chat examples...")
    multi_examples = []
    from collections import defaultdict
    by_make = defaultdict(list)
    for n in normalized:
        make = n.get("make", "Generic")
        by_make[make].append(n)

    for make, codes in by_make.items():
        if len(codes) < 2:
            continue
        for i in range(0, len(codes) - 1, 2):
            if i + 1 < len(codes):
                pair = [codes[i], codes[i + 1]]
                example = create_multi_code_example(pair)
                if example:
                    multi_examples.append(example)
                    if len(multi_examples) >= 2000:
                        break
        if len(multi_examples) >= 2000:
            break
    print(f"  Generated: {len(multi_examples)} multi-code examples")

    all_examples = single_examples + multi_examples

    print("\n[find] Validating dataset quality...")
    all_examples, dupes = deduplicate(all_examples)
    print(f"  Duplicates removed: {dupes}")

    all_examples, too_long = filter_by_token_limit(all_examples, MAX_TOKENS_PER_EXAMPLE)
    print(f"  Discarded (>{MAX_TOKENS_PER_EXAMPLE} tokens): {too_long}")

    random.shuffle(all_examples)

    # Quality spot-check
    print("\n[check] Quality spot-check (5 random examples):")
    for ex in random.sample(all_examples, min(5, len(all_examples))):
        user   = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        asst   = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        print(f"  Q: {user[:80]}")
        print(f"  A: {asst[:120]}")
        print()

    print(f"[stats] Total clean examples: {len(all_examples)}")
    save_dataset(all_examples, "obdient_chat_dataset.jsonl")

    total_tokens = sum(
        sum(len(m["content"].split()) for m in ex["messages"])
        for ex in all_examples
    )
    print(f"\n[chart] Statistics:")
    print(f"  Total examples:          {len(all_examples)}")
    print(f"  Rich-description examples: {rich_count}")
    print(f"  Estimated total tokens:  {total_tokens}")
    print(f"  Avg tokens per example:  {total_tokens // len(all_examples) if all_examples else 0}")

    stats = {
        "total_examples":   len(all_examples),
        "single_code":      len(single_examples),
        "multi_code":       len(multi_examples),
        "rich_descriptions": rich_count,
        "duplicates_removed": dupes,
        "too_long_removed": too_long,
        "excluded_sources": list(EXCLUDED_SOURCES),
        "min_desc_length":  MIN_DESC_LENGTH,
        "max_tokens_per_example": MAX_TOKENS_PER_EXAMPLE,
        "estimated_tokens": total_tokens,
        "system_prompt":    SYSTEM_PROMPT,
    }
    stats_path = PROCESSED_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats saved to {stats_path}")

    print("\n[OK] Step 2 complete. Dataset ready for splitting.")


if __name__ == "__main__":
    main()
