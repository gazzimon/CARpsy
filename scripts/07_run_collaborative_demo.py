#!/usr/bin/env python3
"""
07_run_collaborative_demo.py  Simula una red P2P de agentes especializados.

Arquitectura:
  Orchestrator recibe una consulta compuesta y la descompone en subtareas.
  Cada subtarea se delega al agente especializado correspondiente:

    CARpsy-DTC      →  diagnostica el código OBD-II
    CARpsy-Parts    →  estima precio de piezas (conocimiento embebido)
    CARpsy-Repair   →  describe el procedimiento de reparación

  Los agentes se ejecutan en paralelo (subprocess async) y el orquestador
  combina las respuestas en una respuesta final coherente.

Uso:
  python scripts/07_run_collaborative_demo.py
  python scripts/07_run_collaborative_demo.py --query "P0420 Toyota Camry 2019"
  python scripts/07_run_collaborative_demo.py --demo   # modo hackathon (guión completo)
"""

import sys
import os
import re
import json
import time
import asyncio
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


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

# Unsloth exports a MERGED model (base + LoRA fused) — used as the model directly.
# QVAC Fabric would export a small LoRA delta loaded with --lora on top of a base model.
MERGED_MODEL_PATH = REPO_ROOT / "output" / "adapter" / "qwen3-1.7b.Q4_K_M.gguf"
ADAPTER_PATH = MERGED_MODEL_PATH
MODEL_DIR = REPO_ROOT / "models"


# ─── Agentes especializados ────────────────────────────────────────────────────

@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str
    color: str  # ANSI color code


AGENTS = {
    "dtc": Agent(
        name="CARpsy-DTC",
        role="Diagnóstico de código OBD-II",
        system_prompt=(
            "You are CARpsy-DTC, an expert in OBD-II diagnostic trouble codes. "
            "Given a DTC code, explain ONLY: what the fault is, which system it affects, "
            "and the severity (critical/warning/info). Maximum 2 sentences. Be precise."
        ),
        color="\033[94m",  # azul
    ),
    "parts": Agent(
        name="CARpsy-Parts",
        role="Estimación de piezas y costo",
        system_prompt=(
            "You are CARpsy-Parts, an automotive parts pricing specialist. "
            "Given a fault description, list the most likely parts to replace with estimated costs in USD. "
            "Format: 'Part: $min-$max'. Maximum 3 parts. If unknown, say 'varies by region'."
        ),
        color="\033[92m",  # verde
    ),
    "repair": Agent(
        name="CARpsy-Repair",
        role="Procedimiento de reparación",
        system_prompt=(
            "You are CARpsy-Repair, an expert automotive technician. "
            "Given a fault code, describe the step-by-step diagnosis procedure in 3 steps maximum. "
            "Start with the quickest/cheapest check. Be practical and actionable."
        ),
        color="\033[93m",  # amarillo
    ),
}

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
RED = "\033[91m"


# ─── Utilidades ────────────────────────────────────────────────────────────────

def find_llama_cli() -> Optional[Path]:
    env_fabric = os.environ.get("FABRIC_PATH", "")
    fabric_dir = Path(env_fabric).parent if env_fabric else None

    candidates = []
    if fabric_dir:
        candidates += [fabric_dir / "llama-cli.exe", fabric_dir / "llama-cli"]

    candidates += [
        Path("C:/Users/User/Documents/llama-b7349-bin/llama-cli.exe"),
        Path("C:/Users/User/Documents/qvac-fabric-llm.cpp/build/bin/Release/llama-cli.exe"),
        Path("llama-cli.exe"),
        Path("llama-cli"),
    ]

    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
        if len(c.parts) == 1:
            try:
                subprocess.run(
                    ["where" if os.name == "nt" else "which", str(c)],
                    capture_output=True, check=True
                )
                return c
            except subprocess.CalledProcessError:
                continue
    return None


def find_model() -> Optional[Path]:
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    gguf_files = list(MODEL_DIR.glob("*.gguf"))
    return gguf_files[0] if gguf_files else None


def extract_dtc_code(query: str) -> str:
    """Extrae el código DTC de la consulta (ej. P0420, C1234)."""
    match = re.search(r'\b[PBCU][0-9]{4}\b', query.upper())
    return match.group(0) if match else query


# ─── Inferencia ────────────────────────────────────────────────────────────────

async def run_inference_async(
    cli: Path,
    model: Path,
    adapter: Path,
    agent: Agent,
    prompt: str,
    n_predict: int = 150,
) -> tuple[str, str]:
    """Ejecuta inferencia de forma asíncrona. Retorna (agent_name, response).

    Soporta dos modos:
    - Merged (Unsloth): adapter IS el modelo completo, no se necesita --lora
    - LoRA delta (QVAC Fabric): adapter se carga sobre model con --lora
    """
    full_prompt = (
        f"<|im_start|>system\n{agent.system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    is_merged = (adapter.resolve() == MERGED_MODEL_PATH.resolve())

    if is_merged:
        cmd = [
            str(cli),
            "-m", str(adapter),
            "-ngl", "999",
            "-n", str(n_predict),
            "--temp", "0.1",
            "--top-p", "0.9",
            "-p", full_prompt,
            "--no-display-prompt",
            "-s", "42",
        ]
    else:
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

    start = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
    elapsed = time.time() - start

    response = stdout.decode("utf-8", errors="replace").strip()
    # Truncar en token de fin de turno si aparece
    for stop_token in ["<|im_end|>", "<|endoftext|>", "<|user|>"]:
        if stop_token in response:
            response = response.split(stop_token)[0].strip()

    return agent.name, response, round(elapsed, 1)


# ─── Orquestador ───────────────────────────────────────────────────────────────

async def orchestrate(query: str, cli: Path, model: Path, adapter: Path, verbose: bool = True) -> dict:
    """
    Recibe una consulta compuesta y la distribuye a los 3 agentes en paralelo.
    Combina las respuestas en un diagnóstico completo.
    """
    dtc_code = extract_dtc_code(query)

    if verbose:
        print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
        print(f"{BOLD}{CYAN}  ORQUESTADOR QVAC — Consulta recibida{RESET}")
        print(f"{BOLD}{CYAN}{'─'*60}{RESET}")
        print(f"  Consulta: {query}")
        print(f"  Código detectado: {dtc_code}")
        print(f"\n  Delegando a 3 agentes especializados en paralelo...")

    # Prompts específicos para cada agente
    prompts = {
        "dtc":    f"Diagnose OBD-II code {dtc_code}.",
        "parts":  f"What parts might need replacement for fault code {dtc_code}? Estimate costs.",
        "repair": f"What is the diagnostic procedure for {dtc_code}?",
    }

    # Ejecutar los 3 agentes en paralelo
    tasks = [
        run_inference_async(cli, model, adapter, AGENTS[key], prompts[key])
        for key in ["dtc", "parts", "repair"]
    ]

    t_start = time.time()
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    t_total = time.time() - t_start

    results = {}
    for i, (key, agent) in enumerate(zip(["dtc", "parts", "repair"], AGENTS.values())):
        r = results_raw[i]
        if isinstance(r, Exception):
            results[key] = {"agent": agent.name, "response": f"[ERROR: {r}]", "elapsed": 0}
        else:
            name, response, elapsed = r
            results[key] = {"agent": name, "response": response, "elapsed": elapsed}

    if verbose:
        print(f"\n  Todos los agentes respondieron en {t_total:.1f}s\n")
        print(f"{BOLD}{'─'*60}{RESET}")

        for key, color_key in [("dtc", "dtc"), ("parts", "parts"), ("repair", "repair")]:
            agent = AGENTS[color_key]
            r = results[key]
            print(f"\n{agent.color}{BOLD}  [{agent.name}] — {agent.role}{RESET}")
            print(f"{agent.color}  {r['response']}{RESET}")
            print(f"  {RESET}⏱  {r['elapsed']}s")

        print(f"\n{BOLD}{'─'*60}{RESET}")
        print(f"{BOLD}{CYAN}  RESPUESTA COMBINADA PARA: {dtc_code}{RESET}")
        print(f"{BOLD}{'─'*60}{RESET}\n")

        print(f"  {BOLD}Diagnóstico:{RESET}  {results['dtc']['response']}")
        print(f"\n  {BOLD}Piezas/Costo:{RESET} {results['parts']['response']}")
        print(f"\n  {BOLD}Procedimiento:{RESET} {results['repair']['response']}")
        print(f"\n{CYAN}  Total: {t_total:.1f}s | Modelo local | Sin internet | Sin API keys{RESET}")
        print(f"{BOLD}{'─'*60}{RESET}\n")

    return {
        "query": query,
        "dtc_code": dtc_code,
        "total_elapsed": round(t_total, 1),
        "agents": results,
    }


# ─── Modo Demo Hackathon ────────────────────────────────────────────────────────

DEMO_SCRIPT = [
    {
        "step": 1,
        "title": "Diagnóstico Simple",
        "narration": "Un usuario escanea su auto y obtiene el código P0420.",
        "query": "P0420 Toyota Camry 2019",
    },
    {
        "step": 2,
        "title": "Consulta Compuesta con Costo",
        "narration": "El mismo usuario quiere saber si puede repararlo él mismo y cuánto le costará.",
        "query": "P0300 Ford F-150 2018 — random misfire, repair procedure and cost",
    },
    {
        "step": 3,
        "title": "Fallo Crítico — Urgencia",
        "narration": "Otro nodo de la red detecta un fallo crítico y solicita priorización.",
        "query": "P0562 low battery voltage — is it safe to drive?",
    },
]


async def run_demo(cli: Path, model: Path, adapter: Path):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  CARpsy — DEMO HACKATHON QVAC{RESET}")
    print(f"{BOLD}  Red colaborativa de agentes de diagnóstico automotriz{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"""
  Arquitectura demostrada:
  ┌─────────────────────────────────────────────────────┐
  │   CONSULTA USUARIO                                   │
  │        ↓                                             │
  │   [ORQUESTADOR QVAC]                                 │
  │    ↙        ↓        ↘                               │
  │ [DTC]   [PARTS]   [REPAIR]   ← agentes paralelos    │
  │    ↘        ↓        ↙                               │
  │   RESPUESTA COMBINADA                                │
  │                                                      │
  │   100% local · sin API · sin datos a la nube         │
  └─────────────────────────────────────────────────────┘
""")

    for item in DEMO_SCRIPT:
        print(f"\n{BOLD}{CYAN}  ── DEMO STEP {item['step']}: {item['title']} ──{RESET}")
        print(f"  {item['narration']}")
        print(f"\n  Presiona ENTER para ejecutar...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n  Demo cancelada.")
            return

        await orchestrate(item["query"], cli, model, adapter)

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  FIN DE LA DEMO{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print("""
  Puntos clave para el jurado:
  ✓ Especialización: cada agente tiene un dominio específico
  ✓ Paralelismo: los 3 agentes responden simultáneamente
  ✓ Privacidad: todo corre en el dispositivo del usuario
  ✓ Escalabilidad: cualquier nodo puede añadir agentes
  ✓ QVAC Fabric: adapter LoRA generado con llama-finetune-lora
""")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CARpsy — Demo colaborativa P2P")
    parser.add_argument("--query", type=str, help="Consulta directa (ej. 'P0420 Toyota')")
    parser.add_argument("--demo", action="store_true", help="Ejecutar guión completo de demo")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar configuración sin ejecutar inferencia")
    args = parser.parse_args()

    print("=" * 60)
    print("CARpsy  Step 7: Collaborative P2P Demo")
    print("=" * 60)

    # Verificar dependencias
    if not ADAPTER_PATH.exists():
        print(f"\n  [!] Adaptador no encontrado: {ADAPTER_PATH}")
        print("  [!] Ejecuta primero: python scripts/04_run_finetune.py")
        print("\n  [DEMO MODE] Usando respuestas simuladas para la demo...")
        # En demo sin adapter, mostrar el flujo igualmente
        if args.dry_run or args.demo:
            print("\n  Modo dry-run: el adapter no existe, se muestra el flujo de demo.")
            for item in DEMO_SCRIPT:
                print(f"\n  Step {item['step']}: {item['title']}")
                print(f"  Query: {item['query']}")
            return

    cli = find_llama_cli()
    is_merged = (ADAPTER_PATH.resolve() == MERGED_MODEL_PATH.resolve())

    if is_merged:
        model = ADAPTER_PATH  # merged model needs no separate base
    else:
        model = find_model()

    mode_label = "Unsloth merged" if is_merged else "QVAC Fabric LoRA"

    if args.dry_run:
        print(f"\n  Modelo:    {ADAPTER_PATH}")
        print(f"  Modo:      {mode_label}")
        print(f"  llama-cli: {cli or 'NO ENCONTRADO'}")
        print("\n  Consultas de demo:")
        for item in DEMO_SCRIPT:
            print(f"\n  Step {item['step']}: {item['query']}")
        return

    if not cli:
        print(f"\n  [!] llama-cli no encontrado. Define FABRIC_PATH en .env")
        return

    print(f"\n  Modelo:    {ADAPTER_PATH}")
    print(f"  Modo:      {mode_label}")
    print(f"  llama-cli: {cli}")

    if args.demo:
        asyncio.run(run_demo(cli, model, ADAPTER_PATH))
    elif args.query:
        result = asyncio.run(orchestrate(args.query, cli, model, ADAPTER_PATH))
        out_path = REPO_ROOT / "output" / "demo_result.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Resultado guardado en {out_path}")
    else:
        # Sin argumentos → demo interactiva
        asyncio.run(run_demo(cli, model, ADAPTER_PATH))


if __name__ == "__main__":
    main()
