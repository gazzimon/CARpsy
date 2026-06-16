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
"""

import sys
import json
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CONFIG_PATH = REPO_ROOT / "configs" / "lora_config.yaml"

#  System prompt (matches OBDient's prompt) 
SYSTEM_PROMPT = (
    "You are OBDient, an expert automotive diagnostic assistant. "
    "You receive OBD-II fault codes and vehicle data. "
    "Explain what each code means, its severity, and recommended actions. "
    "Always respond in English, clearly and concisely. "
    "Maximum 3 sentences. Prioritize safety. "
    "If something is urgent, indicate it clearly."
)

#  User question templates (for variety) 
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
    "I have {code} and {code2}. Are they related?",
    "Reading code {code} on my vehicle. What repairs might be needed?",
    "Check engine  {code}. How urgent is this?",
    "Fault code {code} detected. Your advice?",
    "What's the meaning of OBD-II code {code}?",
    "Got {code} after engine light came on. Diagnosis?",
]

#  Assistant response templates by severity 
CRITICAL_RESPONSE = (
    "[WARN] **CRITICAL**: {code}  {description}. "
    "This is a serious issue that requires immediate attention. "
    "It is not safe to drive until this is diagnosed and repaired."
)

WARNING_RESPONSE = (
    "{code}: {description}. "
    "This should be inspected soon, but the vehicle is likely safe for short trips. "
    "Schedule a diagnostic check to prevent further damage."
)

INFO_RESPONSE = (
    "{code}: {description}. "
    "This code alone is not critical, but monitor for related symptoms. "
    "If you're not experiencing any issues, it may be historical or intermittent."
)

MAX_TOKENS_PER_EXAMPLE = 512  # matches training context_length


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_raw_data() -> list[dict]:
    """Load all DTCs from raw JSON files."""
    all_records = []
    for json_file in RAW_DIR.glob("*_raw.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                print(f"  [i] Loaded {len(data) if isinstance(data, list) else 1} records from {json_file.name}")
        except Exception as e:
            print(f"  [!] Error loading {json_file}: {e}")
    return all_records


def normalize_record(record: dict) -> Optional[dict]:
    """Normalize a raw DTC record to a standard schema with code, description, system, make."""
    normalized = {}

    for key in ("code", "dtc", "dtc_code", "fault_code", "Code", "DTC"):
        if key in record and record[key]:
            normalized["code"] = str(record[key]).strip().upper()
            break

    for key in ("description", "definition", "desc", "meaning", "Description", "Definition"):
        if key in record and record[key]:
            normalized["description"] = str(record[key]).strip()
            break

    for key in ("system", "category", "type", "System"):
        if key in record and record[key]:
            normalized["system"] = str(record[key]).strip()
            break
    else:
        if normalized.get("code", "").startswith(("P0", "P1", "P2", "P3")):
            normalized["system"] = "Powertrain"
        elif normalized.get("code", "").startswith("B"):
            normalized["system"] = "Body"
        elif normalized.get("code", "").startswith("C"):
            normalized["system"] = "Chassis"
        elif normalized.get("code", "").startswith("U"):
            normalized["system"] = "Network"
        else:
            normalized["system"] = "Generic"

    for key in ("manufacturer", "make", "brand", "Manufacturer"):
        if key in record and record[key]:
            normalized["make"] = str(record[key]).strip()
            break

    # OBDex enriched fields  pass through as-is
    for key in ("causes", "symptoms", "repair_difficulty", "related_codes", "source"):
        if key in record:
            normalized[key] = record[key]

    for key in ("model", "Model"):
        if key in record and record[key]:
            normalized["model"] = str(record[key]).strip()
            break

    for key in ("year", "Year"):
        if key in record and record[key]:
            normalized["year"] = str(record[key]).strip()
            break

    if "code" not in normalized or "description" not in normalized:
        return None

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
        if 0x0100 <= numeric <= 0x0199:  # Fuel/air metering
            return "critical"
        if 0x0300 <= numeric <= 0x0399:  # Misfire
            return "critical"
        if 0x0A00 <= numeric <= 0x0AFF:  # Control module
            return "critical"
        return "warning"
    if code_upper.startswith(("B", "C", "U")):
        return "info"
    return "warning"


def generate_assistant_response(code: str, description: str, severity: str, make: str = "",
                                causes: list = None, symptoms: list = None,
                                repair_difficulty: str = "") -> str:
    """Generate the assistant response for a given DTC.
    Uses enriched OBDex fields when available for richer, more useful responses.
    """
    context = {"code": code, "description": description}

    # Build enriched response when OBDex data is present
    if causes or symptoms:
        if severity == "critical":
            base = f"[WARN] **CRITICAL**: {code}  {description}. Do not drive until diagnosed."
        elif severity == "warning":
            base = f"{code}: {description}. Schedule an inspection soon."
        else:
            base = f"{code}: {description}. Monitor for related symptoms."

        parts = [base]
        if causes:
            top = causes[:2]  # top 2 causes to stay within token limit
            parts.append(f"Common causes: {'; '.join(top)}.")
        if symptoms:
            parts.append(f"Symptoms: {', '.join(symptoms[:2])}.")
        if repair_difficulty:
            parts.append(f"Repair difficulty: {repair_difficulty}.")
        return " ".join(parts)

    # Fallback: template-based response (Wal33D / peyo records)
    if severity == "critical":
        return CRITICAL_RESPONSE.format(**context)
    elif severity == "warning":
        return WARNING_RESPONSE.format(**context)
    else:
        return INFO_RESPONSE.format(**context)


def generate_user_question(code: str, make: str = "Generic", model: str = "Vehicle", year: str = "2020", code2: str = "") -> str:
    """Generate a varied user question from templates."""
    template = random.choice(USER_TEMPLATES)
    return template.format(
        code=code,
        make=make if make else "Generic",
        model=model if model else "Vehicle",
        year=year if year else "2020",
        code2=code2 if code2 else code,
    )


def create_chat_example(normalized: dict) -> Optional[dict]:
    """Create a chat-format example from a normalized DTC record."""
    code        = normalized["code"]
    description = normalized["description"]
    make        = normalized.get("make", "Generic")
    model       = normalized.get("model", "Vehicle")
    year        = normalized.get("year", "2020")
    severity    = determine_severity(code)

    # OBDex enriched fields (absent in Wal33D/peyo records)
    causes           = normalized.get("causes", [])
    symptoms         = normalized.get("symptoms", [])
    repair_difficulty = normalized.get("repair_difficulty", "")

    user_msg      = generate_user_question(code, make, model, year)
    assistant_msg = generate_assistant_response(code, description, severity, make,
                                                causes, symptoms, repair_difficulty)

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
        prefix = "[WARN] CRITICAL" if sev == "critical" else ""
        assistant_lines.append(f"{prefix} {c['code']}: {c['description']}")
    assistant_msg = "\n".join(assistant_lines)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def estimate_tokens(text: str) -> int:
    """Fast token estimate: 1 token  4 characters."""
    return len(text) // 4


def deduplicate(examples: list[dict]) -> tuple[list[dict], int]:
    """Remove duplicate examples based on the user message."""
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        user_msg = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
        if user_msg not in seen:
            seen.add(user_msg)
            unique.append(ex)
    return unique, len(examples) - len(unique)


def filter_by_token_limit(examples: list[dict], max_tokens: int) -> tuple[list[dict], int]:
    """Discard examples whose total length exceeds max_tokens."""
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
    """Save dataset as JSONL."""
    output_path = PROCESSED_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"  [] Saved {len(examples)} examples to {output_path}")


def main():
    print("=" * 60)
    print("CARpsy  Step 2: Prepare Dataset")
    print("=" * 60)

    config = load_config()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    #  Load raw data 
    print("\n[dir] Loading raw data...")
    raw_records = load_raw_data()
    print(f"  Total raw records: {len(raw_records)}")

    #  Normalize 
    print("\n[proc] Normalizing records...")
    normalized = []
    skipped = 0
    for record in raw_records:
        n = normalize_record(record)
        if n:
            normalized.append(n)
        else:
            skipped += 1
    print(f"  Normalized:  {len(normalized)}")
    print(f"  Skipped (missing code/description): {skipped}")

    #  Generate single-code examples 
    print("\n Generating single-code chat examples...")
    single_examples = []
    for n in normalized[:config["dataset"]["max_examples"]]:
        example = create_chat_example(n)
        if example:
            single_examples.append(example)
    print(f"  Generated: {len(single_examples)} examples")

    #  Generate multi-code examples 
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

    #  Combine 
    all_examples = single_examples + multi_examples

    #  Quality validation 
    print("\n[find] Validating dataset quality...")
    all_examples, dupes = deduplicate(all_examples)
    print(f"  Duplicates removed: {dupes}")

    all_examples, too_long = filter_by_token_limit(all_examples, MAX_TOKENS_PER_EXAMPLE)
    print(f"  Discarded (>{MAX_TOKENS_PER_EXAMPLE} tokens): {too_long}")

    random.shuffle(all_examples)

    print(f"\n[stats] Total examples: {len(all_examples)}")
    save_dataset(all_examples, "obdient_chat_dataset.jsonl")

    #  Statistics 
    total_tokens = sum(
        sum(len(m["content"].split()) for m in ex["messages"])
        for ex in all_examples
    )
    print(f"\n[chart] Statistics:")
    print(f"  Total examples:          {len(all_examples)}")
    print(f"  Estimated total tokens:  {total_tokens}")
    print(f"  Avg tokens per example:  {total_tokens // len(all_examples) if all_examples else 0}")

    stats = {
        "total_examples": len(all_examples),
        "single_code": len(single_examples),
        "multi_code": len(multi_examples),
        "duplicates_removed": dupes,
        "too_long_removed": too_long,
        "max_tokens_per_example": MAX_TOKENS_PER_EXAMPLE,
        "estimated_tokens": total_tokens,
        "system_prompt": SYSTEM_PROMPT,
    }
    stats_path = PROCESSED_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats saved to {stats_path}")

    print("\n[OK] Step 2 complete. Dataset ready for splitting.")


if __name__ == "__main__":
    main()
