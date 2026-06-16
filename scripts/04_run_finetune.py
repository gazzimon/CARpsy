#!/usr/bin/env python3
"""
04_run_finetune.py  Run LoRA fine-tuning locally with llama-finetune-lora.

QVAC Fabric = qvac-fabric-llm.cpp (llama.cpp fork with LoRA training support).
No cloud API, no data leaves your machine.

Prerequisites:
  1. Compile qvac-fabric-llm.cpp    https://github.com/tetherto/qvac-fabric-llm.cpp
  2. Have the llama-finetune-lora binary available
  3. Download the base GGUF model into models/

Environment variables (set in .env):
  FABRIC_PATH   full path to llama-finetune-lora binary (optional, auto-detected)
  MODEL_PATH    full path to the base .gguf model      (optional, auto-detected)
"""

import sys
import os
import subprocess
import json
from pathlib import Path


#  Load .env without external dependencies 
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

TRAIN_SPLIT = REPO_ROOT / "data" / "splits" / "train.jsonl"
VAL_SPLIT   = REPO_ROOT / "data" / "splits" / "val.jsonl"
MODEL_DIR   = REPO_ROOT / "models"
OUTPUT_DIR  = REPO_ROOT / "output" / "adapter"
CONFIG_PATH = REPO_ROOT / "configs" / "training_config.yaml"


def find_llama_binary() -> Path:
    """Find the llama-finetune-lora binary. FABRIC_PATH env var takes priority."""
    env_path = os.environ.get("FABRIC_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists() and os.access(p, os.X_OK):
            return p
        raise FileNotFoundError(f"FABRIC_PATH={env_path} does not exist or is not executable.")

    candidates = [
        Path("C:/Users/User/Documents/llama-b7349-bin/llama-finetune-lora.exe"),
        Path("C:/Users/User/Documents/qvac-fabric-llm.cpp/build/bin/Release/llama-finetune-lora.exe"),
        Path("C:/Users/User/Documents/qvac-fabric-llm.cpp/build/bin/llama-finetune-lora.exe"),
        Path("llama-finetune-lora.exe"),
        Path("llama-finetune-lora"),
        Path("/usr/local/bin/llama-finetune-lora"),
        Path("./llama-finetune-lora"),
    ]

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
        if len(candidate.parts) == 1:
            try:
                cmd = ["where" if os.name == "nt" else "which", str(candidate)]
                subprocess.run(cmd, capture_output=True, check=True)
                return candidate
            except subprocess.CalledProcessError:
                continue

    raise FileNotFoundError(
        "llama-finetune-lora not found.\n\n"
        "1. Clone:   git clone https://github.com/tetherto/qvac-fabric-llm.cpp.git\n"
        "2. Build:   cd qvac-fabric-llm.cpp && mkdir build && cd build\n"
        "            cmake .. -DLLAMA_CUDA=OFF && cmake --build . --config Release\n"
        "3. Set:     FABRIC_PATH=/path/to/llama-finetune-lora in your .env"
    )


def find_model_gguf() -> Path:
    """Find the base GGUF model. MODEL_PATH env var takes priority."""
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"MODEL_PATH={env_path} does not exist.")

    gguf_files = list(MODEL_DIR.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(
            f"No GGUF model found in {MODEL_DIR}/\n\n"
            "Options:\n"
            "  A) Download Qwen3-1.7B-Q4_K_M (recommended for QVAC hackathon):\n"
            "     https://huggingface.co/Qwen/Qwen3-1.7B-GGUF\n"
            "     Place the .gguf file in models/\n\n"
            "  B) Or define MODEL_PATH=/path/to/model.gguf in your .env"
        )
    return gguf_files[0]


def build_command(binary: Path, model: Path, dataset: Path, config: dict, checkpoint_dir: Path) -> list[str]:
    """Build the llama-finetune-lora command line.

    This QVAC Fabric binary only accepts the flags listed in its --help output.
    No -ngl, -c, -b, -ub, -fa, or --flash-attn — those cause backend assertion failures.
    Backend selection and context length are managed internally by the binary.
    """
    args = [str(binary)]

    args.extend(["-m", str(model)])
    args.extend(["-f", str(dataset)])

    lora = config.get("lora", {})
    args.extend(["--lora-rank",    str(lora.get("rank", 8))])
    args.extend(["--lora-alpha",   str(lora.get("alpha", 16))])
    modules = lora.get("target_modules", ["attn_q", "attn_k", "attn_v", "attn_o"])
    args.extend(["--lora-modules", ",".join(modules)])

    adapter_path = OUTPUT_DIR / "carpsy-adapter.gguf"
    args.extend(["--output-adapter", str(adapter_path)])

    training = config.get("training", {})
    args.extend(["--learning-rate", str(training.get("learning_rate", "2e-4"))])
    args.extend(["--weight-decay",  str(training.get("weight_decay", "1e-2"))])
    args.extend(["--num-epochs",    str(training.get("num_epochs", 3))])

    if training.get("assistant_loss_only", True):
        args.append("--assistant-loss-only")

    args.extend(["--checkpoint-save-steps", str(training.get("checkpoint_steps", 100))])
    args.extend(["--checkpoint-save-dir",   str(checkpoint_dir)])

    # Auto-resume from latest checkpoint if one exists
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("checkpoint-*.gguf")):
        args.append("--auto-resume")
        print("  [resume] Auto-resuming from latest checkpoint")

    warmup = training.get("warmup_steps", 0)
    if warmup > 0:
        args.extend(["--warmup-steps", str(warmup)])

    args.extend(["--lr-scheduler", training.get("lr_scheduler", "cosine")])

    return args


def show_command_preview(args: list[str]) -> None:
    print("\n" + "=" * 60)
    print(" COMMAND TO EXECUTE")
    print("=" * 60)
    cmd_str = " \\\n  ".join(args)
    print(f"\n  {cmd_str}\n")


def main():
    print("=" * 60)
    print("CARpsy  Step 4: Run Fine-Tuning (LOCAL)")
    print("=" * 60)

    checkpoint_dir = REPO_ROOT / "output" / "checkpoints"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_SPLIT.exists():
        print(f"\n  [!] Training split not found: {TRAIN_SPLIT}")
        print("  [!] Run first: python scripts/03_split_dataset.py")
        return

    try:
        print("\n[find] Locating llama-finetune-lora...")
        binary = find_llama_binary()
        print(f"  [] Found: {binary}")

        print("\n[pkg] Locating base GGUF model...")
        model = find_model_gguf()
        print(f"  [] Model: {model}")
        print(f"      Size:  {model.stat().st_size / 1024 / 1024:.0f} MB")

        print("\n  Loading configuration...")
        import yaml
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f)
        else:
            config = {"lora": {}, "training": {}}
        print(f"  [] Config loaded")

        print("\n[stats] Dataset info:")
        with open(TRAIN_SPLIT, "r", encoding="utf-8") as f:
            train_lines = sum(1 for _ in f)
        print(f"      Train examples: {train_lines}")

        args = build_command(binary, model, TRAIN_SPLIT, config, checkpoint_dir)
        show_command_preview(args)

        print("  Press ENTER to start fine-tuning (Ctrl+C to cancel)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n  Cancelled.")
            return

        print("\n" + "=" * 60)
        print("[FIRE] FINE-TUNING STARTED")
        print("=" * 60)
        print("  (This may take hours depending on your hardware)\n")

        result = subprocess.run(args, check=False)

        if result.returncode == 0:
            adapter = OUTPUT_DIR / "carpsy-adapter.gguf"
            if adapter.exists():
                print(f"\n[OK] FINE-TUNING COMPLETE")
                print(f"   Adapter: {adapter}")
                print(f"   Size:    {adapter.stat().st_size / 1024:.1f} KB")
                print(f"\n[list] Next steps:")
                print(f"   1. Validate: python scripts/06_validate_adapter.py")
                print(f"   2. Test with llama-cli:")
                print(f"      llama-cli -m {model} --lora {adapter} -ngl 999 \\")
                print(f"        -p \"I have code P0420 on my Toyota. What does it mean?\"")
                print(f"   3. Copy adapter to the OBDient project")
            else:
                print(f"\n  [WARN]  Process finished but adapter file not found.")
        else:
            print(f"\n  [ERR] Fine-tuning failed (exit code: {result.returncode})")
            print("     Check the error messages above.")

    except FileNotFoundError as e:
        print(f"\n  [ERR] ERROR: {e}")
    except Exception as e:
        print(f"\n  [ERR] Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
