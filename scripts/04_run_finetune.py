#!/usr/bin/env python3
"""
04_run_finetune.py — Ejecuta fine-tuning LoRA LOCAL con llama-finetune-lora.

QVAC Fabric = llama.cpp. No hay API, no hay cloud.
Este script construye el comando y ejecuta el binario local.

Requisitos:
  1. Compilar qvac-fabric-llm.cpp (llama.cpp fork)
  2. Tener el binario llama-finetune-lora disponible
  3. Tener el modelo base GGUF descargado
"""

import sys
import os
import subprocess
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SPLIT = REPO_ROOT / "data" / "splits" / "train.jsonl"
VAL_SPLIT = REPO_ROOT / "data" / "splits" / "val.jsonl"
MODEL_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "output" / "adapter"
CONFIG_PATH = REPO_ROOT / "configs" / "training_config.yaml"


def find_llama_binary() -> Path:
    """Busca el binario llama-finetune-lora en ubicaciones comunes."""
    candidates = [
        # Ya descargado en Documents
        Path("C:/Users/User/Documents/llama-b7349-bin/llama-finetune-lora.exe"),
        # Windows - compilación con MSVC
        Path("C:/Users/User/Documents/qvac-fabric-llm.cpp/build/bin/Release/llama-finetune-lora.exe"),
        Path("C:/Users/User/Documents/qvac-fabric-llm.cpp/build/bin/llama-finetune-lora.exe"),
        # En PATH
        Path("llama-finetune-lora.exe"),
        Path("llama-finetune-lora"),
        # Linux/Mac
        Path("/usr/local/bin/llama-finetune-lora"),
        Path("./llama-finetune-lora"),
    ]

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
        # Si es solo nombre (en PATH), probar con where/which
        if candidate.suffix in (".exe", "") and len(candidate.parts) == 1:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["where", str(candidate)],
                        capture_output=True, check=True
                    )
                else:
                    subprocess.run(
                        ["which", str(candidate)],
                        capture_output=True, check=True
                    )
                return candidate
            except subprocess.CalledProcessError:
                continue

    raise FileNotFoundError(
        "No se encontró llama-finetune-lora.\n\n"
        "1. Clona: git clone https://github.com/tetherto/qvac-fabric-llm.cpp.git\n"
        "2. Compila:\n"
        "   cd qvac-fabric-llm.cpp && mkdir build && cd build\n"
        "   cmake .. -DLLAMA_CUDA=OFF && cmake --build . --config Release\n"
        "3. Asegúrate que el binario esté en PATH o configura FABRIC_PATH en .env"
    )


def find_model_gguf() -> Path:
    """Busca el modelo base GGUF en models/."""
    gguf_files = list(MODEL_DIR.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(
            f"No se encontró modelo GGUF en {MODEL_DIR}/\n\n"
            "Descarga LLaMA 3.2 1B Instruct Q4_K_M desde:\n"
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF\n\n"
            "O usa el modelo que ya descargó OBDient en el teléfono."
        )
    return gguf_files[0]


def build_command(
    binary: Path,
    model: Path,
    dataset: Path,
    config: dict,
) -> list[str]:
    """Construye la línea de comandos para llama-finetune-lora."""
    args = [str(binary)]

    # Modelo y dataset
    args.extend(["-m", str(model)])
    args.extend(["-f", str(dataset)])

    # LoRA configuration
    lora = config.get("lora", {})
    args.extend(["--lora-rank", str(lora.get("rank", 8))])
    args.extend(["--lora-alpha", str(lora.get("alpha", 16))])
    modules = lora.get("target_modules", ["attn_q", "attn_k", "attn_v", "attn_o"])
    args.extend(["--lora-modules", ",".join(modules)])

    # Output adapter
    adapter_path = OUTPUT_DIR / "obdient-adapter.gguf"
    args.extend(["--output-adapter", str(adapter_path)])

    # Training parameters
    training = config.get("training", {})
    args.extend(["-c", str(training.get("context_length", 512))])
    args.extend(["-b", str(training.get("batch_size", 4))])
    args.extend(["-ub", str(training.get("ubatch_size", 512))])

    lr = training.get("learning_rate", "2e-4")
    args.extend(["--learning-rate", str(lr)])
    args.extend(["--weight-decay", str(training.get("weight_decay", "1e-2"))])

    epochs = training.get("num_epochs", 3)
    args.extend(["--epochs", str(epochs)])

    # GPU layers (999 = todas las capas en GPU si hay CUDA)
    args.extend(["-ngl", str(training.get("gpu_layers", 999))])

    # SFT con assistant-loss-only
    if config.get("training", {}).get("assistant_loss_only", True):
        args.append("--assistant-loss-only")

    # Checkpointing
    checkpoint_dir = REPO_ROOT / "output" / "checkpoints"
    args.extend([
        "--checkpoint-save-steps",
        str(training.get("checkpoint_steps", 100)),
    ])
    args.extend(["--checkpoint-save-dir", str(checkpoint_dir)])

    # Flash attention
    if not training.get("flash_attention", False):
        args.append("-fa")
        args.append("off")

    # Warmup
    warmup = training.get("warmup_steps", 0)
    if warmup > 0:
        args.extend(["--warmup-steps", str(warmup)])

    # Scheduler
    scheduler = training.get("lr_scheduler", "constant")
    args.extend(["--lr-scheduler", scheduler])

    return args


def show_command_preview(args: list[str]) -> None:
    """Muestra el comando que se va a ejecutar."""
    print("\n" + "=" * 60)
    print("🚀 COMANDO A EJECUTAR")
    print("=" * 60)
    # Formato legible
    cmd_str = " \\\n  ".join(args)
    print(f"\n  {cmd_str}")
    print()


def main():
    print("=" * 60)
    print("CARpsy — Step 4: Run Fine-Tuning (LOCAL)")
    print("=" * 60)

    # Crear directorios
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "output" / "checkpoints").mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Verificar dataset
    if not TRAIN_SPLIT.exists():
        print(f"\n  [!] No se encuentra {TRAIN_SPLIT}")
        print("  [!] Ejecuta primero: python scripts/03_split_dataset.py")
        return

    # Verificar dataset de validación
    val_dataset = VAL_SPLIT if VAL_SPLIT.exists() else TRAIN_SPLIT

    try:
        # 1. Encontrar binario
        print("\n🔍 Buscando llama-finetune-lora...")
        binary = find_llama_binary()
        print(f"  [✓] Encontrado: {binary}")

        # 2. Encontrar modelo
        print("\n📦 Buscando modelo base GGUF...")
        model = find_model_gguf()
        print(f"  [✓] Modelo: {model}")
        print(f"      Tamaño: {model.stat().st_size / 1024 / 1024:.0f} MB")

        # 3. Cargar configuración
        print("\n⚙️  Cargando configuración...")
        import yaml
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f)
        else:
            config = {"lora": {}, "training": {}}
        print(f"  [✓] Configuración cargada")

        # 4. Estadísticas del dataset
        print("\n📊 Dataset:")
        with open(TRAIN_SPLIT, "r", encoding="utf-8") as f:
            train_lines = sum(1 for _ in f)
        print(f"      Train examples: {train_lines}")
        print(f"      Batch size: {config.get('training', {}).get('batch_size', 4)}")
        print(f"      Steps por época: {train_lines // config.get('training', {}).get('batch_size', 4)}")

        # 5. Construir y mostrar comando
        args = build_command(binary, model, TRAIN_SPLIT, config)
        show_command_preview(args)

        # 6. Confirmar
        print("  Presiona ENTER para comenzar el fine-tuning (Ctrl+C para cancelar)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n  Cancelado.")
            return

        # 7. Ejecutar
        print("\n" + "=" * 60)
        print("🔥 INICIANDO FINE-TUNING")
        print("=" * 60)
        print("  (Esto puede tomar horas dependiendo de tu hardware)")
        print()

        result = subprocess.run(args, check=False)

        if result.returncode == 0:
            adapter = OUTPUT_DIR / "obdient-adapter.gguf"
            if adapter.exists():
                print(f"\n✅ FINE-TUNING COMPLETADO")
                print(f"   Adaptador: {adapter}")
                print(f"   Tamaño: {adapter.stat().st_size / 1024:.1f} KB")
                print(f"\n📋 Próximos pasos:")
                print(f"   1. Evaluar: python scripts/05_evaluate_adapter.py")
                print(f"   2. Probar con llama-cli:")
                print(f"      llama-cli -m {model} --lora {adapter} -ngl 999 \\")
                print(f"        -p \"I have code P0420 on my Toyota. What does it mean?\"")
                print(f"   3. Copiar adaptador al proyecto OBDient")
            else:
                print(f"\n  ⚠️  El proceso terminó pero no se encontró el adaptador.")
        else:
            print(f"\n  ❌ Error durante fine-tuning (código: {result.returncode})")
            print("     Revisa los mensajes de error arriba.")

    except FileNotFoundError as e:
        print(f"\n  ❌ ERROR: {e}")
        print("\n  Soluciones rápidas:")
        print("  1. Clona: git clone https://github.com/tetherto/qvac-fabric-llm.cpp.git")
        print("  2. Compila: cd qvac-fabric-llm.cpp && mkdir build && cd build")
        print("     cmake .. && cmake --build . --config Release")
        print("  3. Descarga modelo GGUF en models/")
    except Exception as e:
        print(f"\n  ❌ Error inesperado: {e}")
        raise


if __name__ == "__main__":
    main()