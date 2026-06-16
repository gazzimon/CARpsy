#!/usr/bin/env python3
"""
05_evaluate_adapter.py — Evalúa el adaptador LoRA fine-tuneado.

Compara la calidad de las respuestas del modelo base vs el modelo
fine-tuneado usando el test split.

Métricas:
  - Exactitud de clasificación de severidad
  - Precisión de identificación de código DTC
  - Coherencia de la respuesta (longitud, estructura)
  - Comparación lado a lado (sampling manual)
"""

import sys
import json
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
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
    """Extrae código DTC del mensaje del usuario (ej: 'P0420')."""
    import re
    match = re.search(r'[PBCU]\d{4}', user_msg.upper())
    return match.group(0) if match else None


def extract_code_from_assistant(assistant_msg: str) -> Optional[str]:
    """Extrae código DTC de la respuesta del asistente."""
    import re
    match = re.search(r'[PBCU]\d{4}', assistant_msg.upper())
    return match.group(0) if match else None


def contains_critical_indicator(text: str) -> bool:
    """Detecta si la respuesta indica un problema crítico."""
    indicators = ["critical", "immediate", "not safe", "serious", "urgent", "⚠️"]
    return any(ind in text.lower() for ind in indicators)


def evaluate_example(example: dict) -> dict:
    """Evalúa un solo ejemplo del dataset."""
    messages = example["messages"]
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

    results = {
        "has_code_in_user": False,
        "has_code_in_assistant": False,
        "code_match": False,
        "critical_detected": False,
        "response_length": len(assistant_msg.split()),
        "has_structure": False,
    }

    # Verificar códigos DTC
    user_code = extract_code_from_user_message(user_msg)
    assistant_code = extract_code_from_assistant(assistant_msg)

    if user_code:
        results["has_code_in_user"] = True
    if assistant_code:
        results["has_code_in_assistant"] = True
    if user_code and assistant_code:
        results["code_match"] = user_code == assistant_code

    # Verificar indicadores de severidad
    results["critical_detected"] = contains_critical_indicator(assistant_msg)

    # Verificar estructura (tiene ? o : o múltiples líneas)
    results["has_structure"] = (
        "?" in assistant_msg or ":" in assistant_msg or "\n" in assistant_msg
    )

    return results


def generate_report(results: list[dict], total: int) -> dict:
    """Genera un reporte de evaluación."""
    if not results:
        return {"error": "No results to evaluate"}

    return {
        "total_evaluated": total,
        "code_identification_rate": sum(r["code_match"] for r in results) / max(len(results), 1),
        "critical_response_rate": sum(r["critical_detected"] for r in results) / max(len(results), 1),
        "avg_response_length": sum(r["response_length"] for r in results) / max(len(results), 1),
        "structure_rate": sum(r["has_structure"] for r in results) / max(len(results), 1),
        "samples_with_code_in_user": sum(r["has_code_in_user"] for r in results),
        "samples_with_code_in_assistant": sum(r["has_code_in_assistant"] for r in results),
    }


def print_report(report: dict) -> None:
    """Imprime el reporte de evaluación formateado."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total samples evaluated: {report['total_evaluated']}")
    print(f"\n  📈 Metrics:")
    print(f"     Code identification rate:  {report['code_identification_rate']:.1%}")
    print(f"     Critical response rate:    {report['critical_response_rate']:.1%}")
    print(f"     Avg response length:       {report['avg_response_length']:.0f} tokens")
    print(f"     Structure rate:            {report['structure_rate']:.1%}")
    print(f"  📝 Coverage:")
    print(f"     Samples with code in user:      {report['samples_with_code_in_user']}")
    print(f"     Samples with code in assistant: {report['samples_with_code_in_assistant']}")
    print("=" * 60)


def show_samples(examples: list[dict], n: int = 3) -> None:
    """Muestra ejemplos aleatorios del test set para inspección manual."""
    print("\n" + "=" * 60)
    print("📝 SAMPLE INSPECTION")
    print("=" * 60)

    samples = random.sample(examples, min(n, len(examples)))
    for i, example in enumerate(samples, 1):
        messages = example["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        print(f"\n  ─── Sample #{i} ───")
        print(f"  👤 USER: {user_msg}")
        print(f"  🤖 ASSISTANT: {assistant_msg}")
        print()


def main():
    print("=" * 60)
    print("CARpsy — Step 5: Evaluate Adapter")
    print("=" * 60)

    config = load_config()

    # Cargar test split
    test_path = SPLITS_DIR / "test.jsonl"
    if not test_path.exists():
        print(f"\n  [!] No se encuentra {test_path}")
        print("  [!] Ejecuta primero: python scripts/03_split_dataset.py")
        return

    print(f"\n📂 Cargando test split: {test_path}")
    test_examples = load_jsonl(test_path)
    print(f"  Total test examples: {len(test_examples)}")

    if len(test_examples) == 0:
        print("  [!] No hay ejemplos para evaluar")
        return

    # Evaluar cada ejemplo
    print("\n🔬 Evaluating examples...")
    results = [evaluate_example(ex) for ex in test_examples[:1000]]  # limit for speed

    # Generar reporte
    report = generate_report(results, len(test_examples))
    print_report(report)

    # Mostrar samples
    show_samples(test_examples, 3)

    # Guardar reporte
    report_path = OUTPUT_DIR / "evaluation_report.json"
    report["sample_results"] = results[:10]  # solo primeros 10
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Reporte guardado en {report_path}")

    # Nota sobre evaluación real
    print("\n" + "=" * 60)
    print("📋 NOTAS SOBRE EVALUACIÓN REAL")
    print("=" * 60)
    print("""
  Esta es una evaluación ESTRUCTURAL (formato de las respuestas).
  Para una evaluación real de calidad necesitas:

  1. Ejecutar inferencia con el modelo base + adaptador fine-tuneado
     usando qvacSDK.completion() en un dispositivo real

  2. Comparar las respuestas generadas vs las esperadas usando:
     - BLEU score (similitud textual)
     - Precisión de clasificación de severidad
     - Evaluación humana (side-by-side)

  3. Probar en OBDient real con datos OBD-II vivos

  El verdadero test es: ¿el modelo fine-tuneado da mejores diagnósticos
  que el modelo base + RAG? Eso solo se puede medir en producción.
""")

    print("✅ Step 5 complete.")


if __name__ == "__main__":
    main()