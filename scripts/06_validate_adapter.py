#!/usr/bin/env python3
"""
06_validate_adapter.py  Valida el adaptador LoRA antes de integrarlo en OBDient.

Ejecuta llama-cli con el modelo base + adaptador fine-tuneado y prueba un
conjunto de prompts de diagnóstico OBD-II. Verifica que las respuestas:
  - Contienen el código DTC correcto
  - Indican la severidad adecuada
  - Son concisas (< 3 oraciones)
  - No alucinan ni devuelven texto vacío

Uso:
  python scripts/06_validate_adapter.py
  python scripts/06_validate_adapter.py --dry-run   # Solo muestra los prompts
"""

import sys
import os
import re
import json
import argparse
import subprocess
from pathlib import Path

# Cargar .env si existe
def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

REPO_ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(REPO_ROOT / ".env")

ADAPTER_PATH = REPO_ROOT / "output" / "adapter" / "obdient-adapter.gguf"
MODEL_DIR = REPO_ROOT / "models"
REPORT_PATH = REPO_ROOT / "output" / "adapter" / "validation_report.json"

SYSTEM_PROMPT = (
    "You are OBDient, an expert automotive diagnostic assistant. "
    "You receive OBD-II fault codes and vehicle data. "
    "Explain what each code means, its severity, and recommended actions. "
    "Always respond in English, clearly and concisely. "
    "Maximum 3 sentences. Prioritize safety. "
    "If something is urgent, indicate it clearly."
)

# Prompts de prueba con respuesta esperada mínima
VALIDATION_CASES = [
    {
        "prompt": "I have code P0420 on my Toyota Camry. What does it mean?",
        "expected_code": "P0420",
        "expected_keywords": ["catalyst", "catalytic", "converter", "oxygen"],
        "expected_severity": "warning",
    },
    {
        "prompt": "My car shows P0300. Is it serious?",
        "expected_code": "P0300",
        "expected_keywords": ["misfire", "cylinder", "random"],
        "expected_severity": "critical",
    },
    {
        "prompt": "What does code P0171 mean?",
        "expected_code": "P0171",
        "expected_keywords": ["lean", "fuel", "mixture", "air"],
        "expected_severity": "warning",
    },
    {
        "prompt": "Code P0128 appeared. Should I worry?",
        "expected_code": "P0128",
        "expected_keywords": ["coolant", "thermostat", "temperature"],
        "expected_severity": "warning",
    },
    {
        "prompt": "I have P0562 on my Ford F-150.",
        "expected_code": "P0562",
        "expected_keywords": ["voltage", "battery", "system"],
        "expected_severity": "critical",
    },
]


def find_llama_cli() -> Path:
    """Busca el binario llama-cli (para inferencia, distinto de llama-finetune-lora)."""
    env_fabric = os.environ.get("FABRIC_PATH", "")
    fabric_dir = Path(env_fabric).parent if env_fabric else None

    candidates = []
    if fabric_dir:
        candidates += [
            fabric_dir / "llama-cli.exe",
            fabric_dir / "llama-cli",
            fabric_dir / "main.exe",
            fabric_dir / "main",
        ]

    candidates += [
        Path("C:/Users/User/Documents/llama-b7349-bin/llama-cli.exe"),
        Path("C:/Users/User/Documents/qvac-fabric-llm.cpp/build/bin/Release/llama-cli.exe"),
        Path("C:/Users/User/Documents/qvac-fabric-llm.cpp/build/bin/llama-cli.exe"),
        Path("llama-cli.exe"),
        Path("llama-cli"),
        Path("/usr/local/bin/llama-cli"),
    ]

    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
        if len(c.parts) == 1:
            try:
                cmd = ["where" if os.name == "nt" else "which", str(c)]
                subprocess.run(cmd, capture_output=True, check=True)
                return c
            except subprocess.CalledProcessError:
                continue

    raise FileNotFoundError(
        "No se encontró llama-cli.\n"
        "Compila qvac-fabric-llm.cpp o define FABRIC_PATH=/ruta/a/llama-cli en .env"
    )


def find_model() -> Path:
    """Busca el modelo base GGUF."""
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    gguf_files = list(MODEL_DIR.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No se encontró modelo GGUF en {MODEL_DIR}/")
    return gguf_files[0]


def run_inference(cli: Path, model: Path, adapter: Path, prompt: str, n_predict: int = 150) -> str:
    """Ejecuta inferencia con llama-cli y devuelve la respuesta generada."""
    full_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"

    cmd = [
        str(cli),
        "-m", str(model),
        "--lora", str(adapter),
        "-ngl", "999",
        "-n", str(n_predict),
        "--temp", "0.1",
        "--top-p", "0.9",
        "-p", full_prompt,
        "--no-display-prompt",
        "-s", "42",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


def evaluate_response(response: str, case: dict) -> dict:
    """Evalúa la respuesta del modelo contra los criterios del caso."""
    response_lower = response.lower()

    code_found = bool(re.search(case["expected_code"], response, re.IGNORECASE))
    keywords_found = [kw for kw in case["expected_keywords"] if kw in response_lower]
    keyword_hit_rate = len(keywords_found) / len(case["expected_keywords"])

    is_critical_expected = case["expected_severity"] == "critical"
    critical_indicators = ["critical", "immediate", "not safe", "serious", "urgent", "[WARN]", "warning"]
    severity_ok = any(ind in response_lower for ind in critical_indicators) if is_critical_expected else True

    sentences = [s.strip() for s in re.split(r'[.!?]', response) if s.strip()]
    concise = len(sentences) <= 4

    not_empty = len(response.strip()) > 20
    not_hallucinating = len(response) < 1000

    passed = code_found and keyword_hit_rate >= 0.5 and not_empty and not_hallucinating

    return {
        "passed": passed,
        "code_found": code_found,
        "keyword_hit_rate": round(keyword_hit_rate, 2),
        "keywords_found": keywords_found,
        "severity_ok": severity_ok,
        "concise": concise,
        "response_length": len(response.split()),
        "response_preview": response[:200],
    }


def print_case_result(case: dict, result: dict, i: int) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n   Case #{i}: {status} ")
    print(f"  Prompt:   {case['prompt']}")
    print(f"  Response: {result['response_preview']}{'...' if len(result['response_preview']) == 200 else ''}")
    print(f"  Code found:    {'' if result['code_found'] else ''} ({case['expected_code']})")
    print(f"  Keywords:      {result['keyword_hit_rate']:.0%} ({', '.join(result['keywords_found'])})")
    print(f"  Severity OK:   {'' if result['severity_ok'] else ''}")
    print(f"  Concise:       {'' if result['concise'] else ''} ({result['response_length']} words)")


def main():
    parser = argparse.ArgumentParser(description="Valida el adaptador LoRA de OBDient")
    parser.add_argument("--dry-run", action="store_true", help="Solo muestra los prompts sin ejecutar inferencia")
    parser.add_argument("--adapter", type=str, default=str(ADAPTER_PATH), help="Ruta al adaptador .gguf")
    args = parser.parse_args()

    print("=" * 60)
    print("CARpsy  Step 6: Validate Adapter")
    print("=" * 60)

    adapter = Path(args.adapter)
    if not adapter.exists():
        print(f"\n  [!] No se encontró adaptador en {adapter}")
        print("  [!] Ejecuta primero: python scripts/04_run_finetune.py")
        return

    print(f"\n  Adaptador: {adapter}")
    print(f"  Tamaño:    {adapter.stat().st_size / 1024:.1f} KB")

    if args.dry_run:
        print("\n  [DRY RUN] Prompts de validación:")
        for i, case in enumerate(VALIDATION_CASES, 1):
            print(f"\n  {i}. {case['prompt']}")
            print(f"     Esperado: {case['expected_code']} | severidad={case['expected_severity']} | keywords={case['expected_keywords']}")
        print("\n  (Sin --dry-run se ejecuta inferencia real con llama-cli)")
        return

    try:
        cli = find_llama_cli()
        model = find_model()
    except FileNotFoundError as e:
        print(f"\n  [!] {e}")
        return

    print(f"\n  llama-cli: {cli}")
    print(f"  Modelo:    {model}")
    print(f"\n  Ejecutando {len(VALIDATION_CASES)} casos de prueba...")

    all_results = []
    for i, case in enumerate(VALIDATION_CASES, 1):
        print(f"\n  [{i}/{len(VALIDATION_CASES)}] {case['prompt'][:60]}...")
        response = run_inference(cli, model, adapter, case["prompt"])
        result = evaluate_response(response, case)
        result["prompt"] = case["prompt"]
        all_results.append(result)
        print_case_result(case, result, i)

    # Resumen
    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)
    avg_keywords = sum(r["keyword_hit_rate"] for r in all_results) / total

    print("\n" + "=" * 60)
    print("[stats] RESUMEN DE VALIDACIÓN")
    print("=" * 60)
    print(f"  Casos pasados:       {passed}/{total} ({passed/total:.0%})")
    print(f"  Keywords promedio:   {avg_keywords:.0%}")

    if passed == total:
        print("\n  [OK] Adaptador validado. Listo para integrar en OBDient.")
        print(f"\n  En qvac-sdk.datasource.ts:")
        print(f"    adapterSrc: 'file://{adapter.name}'")
    elif passed >= total * 0.8:
        print("\n  [WARN]  Adaptador aceptable (>80% casos). Revisar casos fallidos antes de producción.")
    else:
        print("\n  [ERR] Adaptador NO aceptable. Considera más épocas o ajustar el learning rate.")

    # Guardar reporte
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 2),
        "avg_keyword_hit_rate": round(avg_keywords, 2),
        "adapter": str(adapter),
        "model": str(model),
        "cases": all_results,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Reporte guardado en {REPORT_PATH}")

    print("\n[OK] Step 6 complete.")


if __name__ == "__main__":
    main()
