# CARpsy — LoRA Fine-Tuning for OBDient Automotive Diagnostics

> **Hackathon project** — Fine-tuning a small LLM with QVAC Fabric to improve OBD-II diagnostic accuracy inside the [OBDient](https://github.com/tetherto/obdient) mobile app.  
> Runs **100% locally** — no cloud API, no data leaves your machine.

---

## What is CARpsy?

CARpsy is a 5-step pipeline that:

1. Downloads ~28,000 OBD-II Diagnostic Trouble Codes (DTCs) from public databases
2. Converts them into chat-format training examples (system / user / assistant)
3. Splits the dataset into train / val / test
4. Fine-tunes a quantized GGUF model using **QVAC Fabric** (`llama-finetune-lora`)
5. Validates the resulting LoRA adapter before shipping it to OBDient

The generated `.gguf` LoRA adapter can be loaded directly by OBDient's QVAC SDK datasource.

---

## Recommended Model — Qwen3-1.7B

The QVAC Fabric hackathon prioritises **Qwen3 and Gemma3** architectures, as these have verified LoRA fine-tuning support in `qvac-fabric-llm.cpp`.

| Model | Size on disk | VRAM (training) | Notes |
|-------|-------------|-----------------|-------|
| **Qwen3-1.7B Q4_K_M** ✅ | ~1.1 GB | ~4–6 GB | **Recommended** — QVAC-native, mobile-friendly, multilingual |
| Qwen3-0.6B Q4_K_M | ~0.4 GB | ~2–3 GB | Fastest; lower quality |
| Qwen3-4B Q4_K_M | ~2.5 GB | ~8–10 GB | Best quality; needs 10 GB+ VRAM |
| LLaMA 3.2 1B Q4_K_M | ~0.7 GB | ~4–5 GB | Works but not a QVAC-native arch |

**Download Qwen3-1.7B-Q4_K_M:**
```
https://huggingface.co/Qwen/Qwen3-1.7B-GGUF
```
Place the `.gguf` file in `models/`.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM       | 8 GB    | 16 GB       |
| GPU       | CPU-only (slow) | 6 GB+ VRAM (NVIDIA/AMD/Apple) |
| Disk      | 5 GB    | 15 GB       |
| OS        | Windows / Linux / macOS | Linux + CUDA |

Training Qwen3-1.7B with LoRA rank 8 requires ~4–6 GB RAM (CPU) or VRAM (GPU).

---

## Prerequisites

### 1. Install Python 3.11+

**Windows** (recommended — install from python.org, NOT the Microsoft Store stub):
```
https://www.python.org/downloads/
```
Make sure to check **"Add Python to PATH"** during installation.

Verify:
```bash
python --version   # should print Python 3.11.x or 3.12.x
pip --version
```

### 2. Compile QVAC Fabric (llama.cpp fork)

```bash
git clone https://github.com/tetherto/qvac-fabric-llm.cpp.git
cd qvac-fabric-llm.cpp
mkdir build && cd build

# CPU only
cmake .. -DLLAMA_CUDA=OFF
# NVIDIA GPU
cmake .. -DLLAMA_CUDA=ON
# AMD GPU (ROCm)
cmake .. -DLLAMA_HIPBLAS=ON

cmake --build . --config Release
```

Verify the binary exists:
```bash
# Windows
.\build\bin\Release\llama-finetune-lora.exe --help
# Linux / macOS
./build/bin/llama-finetune-lora --help
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure paths (optional)

Copy `.env.example` to `.env` and set your paths:
```bash
cp .env.example .env
```

---

## Pipeline — Step by Step

```bash
# Step 1 — Download ~28,000 DTC codes from GitHub repos
python scripts/01_fetch_dtc_data.py

# Step 2 — Convert to chat-format JSONL + validate quality
python scripts/02_prepare_dataset.py

# Step 3 — Split 80 / 10 / 10 (train / val / test)
python scripts/03_split_dataset.py

# Step 4 — Run LoRA fine-tuning locally
python scripts/04_run_finetune.py

# Step 5 — Structural evaluation of the dataset
python scripts/05_evaluate_adapter.py

# Step 6 — Live inference validation of the adapter
python scripts/06_validate_adapter.py
```

---

## What Step 4 runs internally

```bash
llama-finetune-lora \
  -m models/qwen3-1.7b-q4_k_m.gguf \
  -f data/splits/train.jsonl \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-modules "attn_q,attn_k,attn_v,attn_o" \
  --assistant-loss-only \
  --output-adapter output/adapter/carpsy-adapter.gguf \
  -ngl 999 \
  -c 512 -b 4 -ub 512 \
  --learning-rate 2e-4 \
  --lr-scheduler cosine \
  --checkpoint-save-steps 100 \
  --checkpoint-save-dir output/checkpoints
```

If training is interrupted, Step 4 will **automatically resume** from the latest checkpoint.

---

## Dataset Format

Training examples use the standard ChatML format (JSONL, one example per line):

```json
{"messages": [
  {"role": "system",    "content": "You are OBDient, an expert automotive diagnostic assistant..."},
  {"role": "user",      "content": "I'm getting code P0420 on my Toyota Camry. What does it mean?"},
  {"role": "assistant", "content": "P0420: Catalyst System Efficiency Below Threshold. This should be inspected soon..."}
]}
```

The `--assistant-loss-only` flag ensures the model learns only from assistant tokens, ignoring the system prompt and user question during backpropagation.

---

## Integration with OBDient

Once `carpsy-adapter.gguf` is generated and validated, integrate it in the OBDient QVAC datasource:

```typescript
// qvac-sdk.datasource.ts
const modelId = await loadModel({
  modelSrc:    QWEN3_1_7B_Q4_K_M,
  adapterSrc: 'file://carpsy-adapter.gguf',
});
```

---

## Project Structure

```
CARpsy/
├── data/
│   ├── raw/              # Raw DTC records (gitignored)
│   ├── processed/        # Chat-format JSONL dataset (gitignored)
│   └── splits/           # train / val / test splits (gitignored)
├── models/               # Base GGUF model — download separately (gitignored)
├── scripts/
│   ├── 01_fetch_dtc_data.py      # Download Wal33D & peyo DTC databases
│   ├── 02_prepare_dataset.py     # Convert to chat format + deduplicate
│   ├── 03_split_dataset.py       # 80/10/10 split
│   ├── 04_run_finetune.py        # Execute llama-finetune-lora (local)
│   ├── 05_evaluate_adapter.py    # Structural dataset evaluation
│   └── 06_validate_adapter.py    # Live inference adapter validation
├── configs/
│   ├── lora_config.yaml          # LoRA + dataset configuration
│   └── training_config.yaml      # llama-finetune-lora CLI arguments
├── output/
│   ├── adapter/                  # carpsy-adapter.gguf (generated)
│   └── checkpoints/              # Training checkpoints (auto-resume)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Data Sources

| Source | Format | Records |
|--------|--------|---------|
| [Wal33D/dtc-database](https://github.com/Wal33D/dtc-database) | SQLite | ~28,000 codes, 33 manufacturers |
| [peyo/dtc-and-vin-data](https://github.com/peyo/dtc-and-vin-data) | JSON | DTCs + VIN data |
| [xinings/DTC-Database](https://github.com/xinings/DTC-Database) | JSON/XML | ~6,665 codes |

---

## QVAC Fabric Reference

- **Repository:** https://github.com/tetherto/qvac-fabric-llm.cpp
- **Fine-tuning docs:** `examples/training/README.md`
- **Supported GPU backends:** CUDA (NVIDIA), Vulkan (AMD/Intel), Metal (Apple)
- **Verified LoRA architectures:** Qwen3, Gemma3, BitNet
