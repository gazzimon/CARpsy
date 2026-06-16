#!/usr/bin/env python3
"""
02_prepare_dataset.py — Transforma DTCs crudos a formato chat (system/user/assistant).

Convierte los registros DTC extraídos en conversaciones estilo ChatGPT
para fine-tuning supervisado con QVAC Fabric.

Formato de salida: JSONL (una línea = un ejemplo de entrenamiento)
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
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

# ─── System prompt (coincide con el de OBDient) ──────────────────────────────
SYSTEM_PROMPT = (
    "You are OBDient, an expert automotive diagnostic assistant. "
    "You receive OBD-II fault codes and vehicle data. "
    "Explain what each code means, its severity, and recommended actions. "
    "Always respond in English, clearly and concisely. "
    "Maximum 3 sentences. Prioritize safety. "
    "If something is urgent, indicate it clearly."
)

# ─── Plantillas de preguntas de usuario (para generar variedad) ──────────────
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
    "Check engine — {code}. How urgent is this?",
    "Fault code {code} detected. Your advice?",
    "What's the meaning of OBD-II code {code}?",
    "Got {code} after engine light came on. Diagnosis?",
]

# ─── Respuestas del asistente por severidad ──────────────────────────────────
CRITICAL_RESPONSE = (
    "⚠️ **CRITICAL**: {code} — {description}. "
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


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_raw_data() -> list[dict]:
    """Carga todos los DTCs desde los archivos raw JSON."""
    all_records = []
    for json_file in RAW_DIR.glob("*_raw.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                print(f"  [i] Cargados {len(data) if isinstance(data, list) else 1} registros de {json_file.name}")
        except Exception as e:
            print(f"  [!] Error cargando {json_file}: {e}")
    return all_records


def normalize_record(record: dict) -> Optional[dict]:
    """Normaliza un registro DTC a un formato estándar con code, description, system, make."""
    normalized = {}

    # Mapear 'code'
    for key in ("code", "dtc", "dtc_code", "fault_code", "Code", "DTC"):
        if key in record and record[key]:
            normalized["code"] = str(record[key]).strip().upper()
            break

    # Mapear 'description'
    for key in ("description", "definition", "desc", "meaning", "Description", "Definition"):
        if key in record and record[key]:
            normalized["description"] = str(record[key]).strip()
            break

    # Mapear 'system' (P, B, C, U)
    for key in ("system", "category", "type", "System"):
        if key in record and record[key]:
            normalized["system"] = str(record[key]).strip()
            break
    else:
        # Inferir sistema del código
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

    # Mapear 'make' (fabricante)
    for key in ("manufacturer", "make", "brand", "Manufacturer"):
        if key in record and record[key]:
            normalized["make"] = str(record[key]).strip()
            break

    # Mapear 'model'
    for key in ("model", "Model"):
        if key in record and record[key]:
            normalized["model"] = str(record[key]).strip()
            break

    # Mapear 'year'
    for key in ("year", "Year"):
        if key in record and record[key]:
            normalized["year"] = str(record[key]).strip()
            break

    # Validar que tenemos al menos code y description
    if "code" not in normalized or "description" not in normalized:
        return None

    return normalized


def determine_severity(code: str) -> str:
    """Determina severidad basada en el código DTC."""
    code_upper = code.upper()
    numeric_part = ""
    for c in code_upper[1:]:
        if c.isdigit() or c.upper() in "ABCDEF":
            numeric_part += c

    try:
        numeric = int(numeric_part, 16) if numeric_part else 0
    except ValueError:
        return "warning"

    # Lógica similar a dtcParser.ts de OBDient
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


def generate_assistant_response(code: str, description: str, severity: str, make: str = "") -> str:
    """Genera la respuesta del asistente."""
    context = {"code": code, "description": description}

    if severity == "critical":
        return CRITICAL_RESPONSE.format(**context)
    elif severity == "warning":
        return WARNING_RESPONSE.format(**context)
    else:
        return INFO_RESPONSE.format(**context)


def generate_user_question(code: str, make: str = "Generic", model: str = "Vehicle", year: str = "2020", code2: str = "") -> str:
    """Genera una pregunta de usuario variada."""
    template = random.choice(USER_TEMPLATES)
    return template.format(
        code=code,
        make=make if make else "Generic",
        model=model if model else "Vehicle",
        year=year if year else "2020",
        code2=code2 if code2 else code,
    )


def create_chat_example(normalized: dict, include_code2: bool = False) -> Optional[dict]:
    """Crea un ejemplo en formato chat a partir de un registro normalizado."""
    code = normalized["code"]
    description = normalized["description"]
    make = normalized.get("make", "Generic")
    model = normalized.get("model", "Vehicle")
    year = normalized.get("year", "2020")
    severity = determine_severity(code)

    # Generar pregunta de usuario
    user_msg = generate_user_question(code, make, model, year)

    # Generar respuesta del asistente
    assistant_msg = generate_assistant_response(code, description, severity, make)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def create_multi_code_example(codes: list[dict]) -> Optional[dict]:
    """Crea un ejemplo con múltiples códigos DTC simultáneos."""
    if len(codes) < 2:
        return None

    code_strs = [c["code"] for c in codes]
    descriptions = [c["description"] for c in codes]

    user_msg = (
        f"I have multiple fault codes: {', '.join(code_strs)}. "
        f"Can you tell me what they mean and if they're related?"
    )

    assistant_lines = []
    for i, c in enumerate(codes):
        sev = determine_severity(c["code"])
        prefix = "⚠️ CRITICAL" if sev == "critical" else "•"
        assistant_lines.append(f"{prefix} {c['code']}: {c['description']}")
    assistant_msg = "\n".join(assistant_lines)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


MAX_TOKENS_PER_EXAMPLE = 512  # coincide con context_length del entrenamiento


def estimate_tokens(text: str) -> int:
    """Aproximación rápida: 1 token ≈ 4 caracteres."""
    return len(text) // 4


def deduplicate(examples: list[dict]) -> tuple[list[dict], int]:
    """Elimina ejemplos duplicados basándose en el mensaje del usuario."""
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        user_msg = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
        if user_msg not in seen:
            seen.add(user_msg)
            unique.append(ex)
    return unique, len(examples) - len(unique)


def filter_by_token_limit(examples: list[dict], max_tokens: int) -> tuple[list[dict], int]:
    """Descarta ejemplos cuya longitud total supera max_tokens."""
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
    """Guarda el dataset en formato JSONL."""
    output_path = PROCESSED_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"  [✓] Guardados {len(examples)} ejemplos en {output_path}")


def main():
    print("=" * 60)
    print("CARpsy — Step 2: Prepare Dataset")
    print("=" * 60)

    config = load_config()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Cargar datos crudos ───────────────────────────────────────────────
    print("\n📂 Loading raw data...")
    raw_records = load_raw_data()
    print(f"  Total registros crudos: {len(raw_records)}")

    # ─── Normalizar ─────────────────────────────────────────────────────────
    print("\n🔄 Normalizing records...")
    normalized = []
    skipped = 0
    for record in raw_records:
        n = normalize_record(record)
        if n:
            normalized.append(n)
        else:
            skipped += 1
    print(f"  Normalizados: {len(normalized)}")
    print(f"  Omitidos (sin code/description): {skipped}")

    # ─── Generar ejemplos individuales ──────────────────────────────────────
    print("\n💬 Generating single-code chat examples...")
    single_examples = []
    for n in normalized[:config["dataset"]["max_examples"]]:
        example = create_chat_example(n)
        if example:
            single_examples.append(example)
    print(f"  Generados: {len(single_examples)} ejemplos")

    # ─── Generar ejemplos multi-código ──────────────────────────────────────
    print("\n🔗 Generating multi-code chat examples...")
    multi_examples = []
    # Agrupar por make para crear ejemplos realistas de múltiples DTCs
    from collections import defaultdict
    by_make = defaultdict(list)
    for n in normalized:
        make = n.get("make", "Generic")
        by_make[make].append(n)

    for make, codes in by_make.items():
        if len(codes) < 2:
            continue
        # Crear pares de códigos que suelen aparecer juntos
        for i in range(0, len(codes) - 1, 2):
            if i + 1 < len(codes):
                pair = [codes[i], codes[i + 1]]
                example = create_multi_code_example(pair)
                if example:
                    multi_examples.append(example)
                    if len(multi_examples) >= 2000:  # límite
                        break
        if len(multi_examples) >= 2000:
            break
    print(f"  Generados: {len(multi_examples)} ejemplos multi-código")

    # ─── Combinar ───────────────────────────────────────────────────────────
    all_examples = single_examples + multi_examples

    # ─── Validación de calidad ──────────────────────────────────────────────
    print("\n🔍 Validando calidad del dataset...")
    all_examples, dupes = deduplicate(all_examples)
    print(f"  Duplicados eliminados: {dupes}")

    all_examples, too_long = filter_by_token_limit(all_examples, MAX_TOKENS_PER_EXAMPLE)
    print(f"  Ejemplos descartados por longitud (>{MAX_TOKENS_PER_EXAMPLE} tokens): {too_long}")

    random.shuffle(all_examples)

    print(f"\n📊 Total ejemplos generados: {len(all_examples)}")

    save_dataset(all_examples, "obdient_chat_dataset.jsonl")

    # ─── Estadísticas ───────────────────────────────────────────────────────
    total_tokens = sum(
        sum(len(m["content"].split()) for m in ex["messages"])
        for ex in all_examples
    )
    print(f"\n📈 Estadísticas:")
    print(f"  Total ejemplos: {len(all_examples)}")
    print(f"  Total tokens (aprox): {total_tokens}")
    print(f"  Promedio tokens/ejemplo: {total_tokens // len(all_examples) if all_examples else 0}")

    # Guardar metadatos
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
    print(f"  Estadísticas guardadas en {stats_path}")

    print("\n✅ Step 2 complete. Dataset listo para split.")


if __name__ == "__main__":
    main()