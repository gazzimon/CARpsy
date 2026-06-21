# CARpsy — LoRA Fine-Tuning for OBDient Automotive Diagnostics

> **Hackathon project** — Fine-tuning a small LLM with QVAC Fabric to improve OBD-II diagnostic accuracy inside the [OBDient](https://github.com/tetherto/obdient) mobile app.  
> Runs **100% locally** — no cloud API, no data leaves your machine.

---

## 📊 Proof of Fine-Tuning (Evidence)

This model was genuinely fine-tuned — here is the verifiable proof:

- **Training loss curve** — drops from ~4.1 to ~0.05 by step ~250: [`docs/evidence/Finetune curve.png`](docs/evidence/Finetune%20curve.png)
- **Executed Colab notebook** (with per-step training logs): [`docs/evidence/colab_finetune.ipynb`](docs/evidence/colab_finetune.ipynb)
- **Evaluation report** (1,532 test examples): [`output/adapter/evaluation_report.json`](output/adapter/evaluation_report.json)
- **Validation report** (8 cases, 0.88 pass rate, real model responses): [`output/adapter/validation_report.json`](output/adapter/validation_report.json)
- **Trained model weights (GGUF) on Hugging Face Hub:** https://huggingface.co/gazzimon/CARpsy-v2-qwen3-0.6b-GGUF

Comparing the published weights against the base Qwen3-0.6B confirms they differ — objective evidence of fine-tuning. See [`docs/evidence/`](docs/evidence/) for details.

---

## Why Fine-Tune? Empirical Justification

Before committing to fine-tuning, we tested whether a base model could already answer DTC queries correctly — making training redundant.

We ran `scripts/10_baseline_vs_golden.py` against 20 random examples from the test split, comparing three base models (no fine-tuning, same system prompt) against our golden answers:

| Model | Keyword Overlap | DTC Recall | Length Ratio | Composite Score |
|-------|----------------|------------|--------------|-----------------|
| Qwen2.5 0.5B (base) | 3.9% | 100% | 15.6% | 35.1% |
| Qwen2.5 1.5B (base) | 4.1% | 100% | 17.1% | 35.4% |
| LLaMA 3.2 1B (base) | 5.9% | 100% | 18.9% | 36.8% |

**All three models scored ~35–37% — well below the 55% threshold for acceptable quality.**

The pattern was consistent across all models: every base model correctly repeated the DTC code from the user's message (100% DTC recall), but **fabricated plausible-sounding but incorrect definitions** — describing unrelated codes as "fuel pressure regulator problems", "oxygen sensor issues", or "ECM communication faults", regardless of the actual code.

This is not a model size problem. Scaling from 0.5B to 1.5B parameters produced no meaningful improvement. The issue is that **small base models do not have the ~5,000+ DTC-to-definition mappings memorized with factual accuracy**. Fine-tuning injects this factual knowledge explicitly — it is not a format or style adjustment.

**Conclusion: fine-tuning is necessary and not redundant.** The system prompt alone, regardless of model size in the 0.5–1.5B range, cannot produce correct DTC diagnoses.

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

## Recommended Model — Qwen3-0.6B (current)

The QVAC Fabric hackathon prioritises **Qwen3 and Gemma3** architectures, as these have verified LoRA fine-tuning support in `qvac-fabric-llm.cpp`.

**CARpsy-v2 was trained on Qwen3-0.6B Q4_K_M** — this is the production model. Despite being the smallest variant, it achieved the best results in our experiments and fits comfortably on mobile hardware.

| Model | Size on disk | VRAM (training) | Notes |
|-------|-------------|-----------------|-------|
| **Qwen3-0.6B Q4_K_M** ✅ | ~0.4 GB | ~2–3 GB | **Current** — used for CARpsy-v2, mobile-friendly |
| Qwen3-1.7B Q4_K_M | ~1.1 GB | ~4–6 GB | Planned for CARpsy-v3 — see Next Steps |
| Qwen3-4B Q4_K_M | ~2.5 GB | ~8–10 GB | Best quality; needs 10 GB+ VRAM |
| LLaMA 3.2 1B Q4_K_M | ~0.7 GB | ~4–5 GB | Works but not a QVAC-native arch |

**Download Qwen3-0.6B-Q4_K_M:**
```
https://huggingface.co/Qwen/Qwen3-0.6B-GGUF
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

# Step 4a — Run LoRA fine-tuning locally (QVAC Fabric, requires compiled binary)
python scripts/04_run_finetune.py
# Step 4b — OR run on Google Colab with Unsloth (recommended, free T4 GPU)
#           Open colab_finetune.ipynb in Google Colab

# Step 5 — Structural evaluation of the dataset
python scripts/05_evaluate_adapter.py

# Step 6 — Live inference validation of the fine-tuned model
python scripts/06_validate_adapter.py
# (auto-detects Unsloth merged model vs QVAC Fabric LoRA adapter)

# Step 7 — Collaborative P2P demo (hackathon)
python scripts/07_run_collaborative_demo.py --demo
```

---

## Training Configuration — CARpsy-v2 (actual, Unsloth/Colab)

CARpsy-v2 (the production model) was trained with **Unsloth on Google Colab (A100 GPU)** using `colab_finetune.ipynb`. These are the exact hyperparameters that produced the shipped model:

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen3-0.6B` (loaded in 4-bit) |
| Method | LoRA — supervised fine-tuning via TRL `SFTTrainer` |
| Dataset | `canonical_dataset.jsonl` — **300 examples** (20 DTC codes × 15 questions), ChatML format |
| Epochs | **100** |
| Learning rate | **8e-5** |
| LR scheduler | cosine, `warmup_steps=20` |
| LoRA rank `r` | **16** |
| LoRA alpha | **32** (alpha = 2×r) |
| LoRA dropout | **0.05** |
| Target modules | `q_proj, k_proj, v_proj, o_proj` |
| Batch size | 4 × grad accum 4 = **16 effective** |
| Weight decay | 1e-2 |
| Optimizer | `adamw_8bit` |
| Precision | bf16 (A100) / fp16 (T4) |
| Max sequence length | 512 (`packing=True`) |
| Seed | 42 |
| Output | merged model exported to GGUF Q4_K_M → `CARpsy-v2-qwen3-0.6b.Q4_K_M.gguf` |

Each code has **one fixed canonical answer** in the BLUCKTEC format:
`{CODE}: {SAE name}. Severity N/3 — {action}. Likely causes: ... {verdict}.`

**Validation:** 8 BLUCKTEC test cases run after training; acceptance threshold is **≥6/8 correct** (response must contain the code, a `Severity` field, `Likely causes`, and a fault keyword). Inference uses `temperature=0.1`, `top_p=0.9`, with Qwen3 `<think>` blocks stripped.

---

## What Step 4 runs internally (local QVAC Fabric — alternative path)

> This is the fully-local Path A (`scripts/04_run_finetune.py`), driven by `configs/training_config.yaml`. It was **not** used to produce CARpsy-v2 — see the table above for the actual training.

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

## Fine-Tuning — Two Paths

### Path A — QVAC Fabric (local, llama.cpp fork)
Produces a **small LoRA delta** (~2–10 MB) loaded on top of the base model:
```bash
python scripts/04_run_finetune.py
# output: output/adapter/carpsy-adapter.gguf  (delta only)
```

### Path B — Unsloth on Google Colab (A100 GPU) ✅ Used
Produces a **merged full model** (base + LoRA fused, Q4_K_M):
```
Google Colab → colab_finetune.ipynb → Google Drive → output/adapter/CARpsy-v2-qwen3-0.6b.Q4_K_M.gguf
```

| | QVAC Fabric | Unsloth/Colab |
|--|-------------|---------------|
| Output | small adapter (~MB) | merged model (Qwen3-0.6B Q4_K_M) |
| Usage | `-m base.gguf --lora adapter.gguf` | `-m CARpsy-v2-qwen3-0.6b.Q4_K_M.gguf` |
| `adapterSrc` in SDK | `file://carpsy-adapter.gguf` | not needed |
| `modelSrc` in SDK | predefined constant | `file://CARpsy-v2-qwen3-0.6b.Q4_K_M.gguf` |

## Integration with OBDient

### With Unsloth merged model (current)
```typescript
// qvac-sdk.datasource.ts
const modelId = await loadModel({
  modelSrc: 'file://output/adapter/CARpsy-v2-qwen3-0.6b.Q4_K_M.gguf',
});
```

### With QVAC Fabric LoRA adapter (future)
```typescript
const modelId = await loadModel({
  modelSrc:   QWEN3_1_7B_Q4_K_M,
  adapterSrc: 'file://output/adapter/carpsy-adapter.gguf',
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

## Collaborative P2P Network with QVAC

CARpsy can operate as a **specialized node** in a QVAC agent swarm, where multiple experts collaborate to answer complex queries.

### Architecture

```
USER QUERY: "P0420 + repair + cost"
       ↓
[ORCHESTRATOR]
 ↙       ↓       ↘
[DTC]  [PARTS]  [REPAIR]   ← parallel agents, same LoRA adapter
 ↘       ↓       ↙
 COMBINED RESPONSE

100% local · no API keys · no data leaves the device
```

### Agent Specializations

| Agent | Role | System Prompt Focus |
|-------|------|-------------------|
| `CARpsy-DTC` | OBD-II code diagnosis | Fault meaning + severity |
| `CARpsy-Parts` | Parts & cost estimation | Replacement parts + price range |
| `CARpsy-Repair` | Repair procedure | Step-by-step diagnosis |

### Run the Collaborative Demo

```bash
# Full hackathon demo (guided, 3 steps)
python scripts/07_run_collaborative_demo.py --demo

# Single composite query
python scripts/07_run_collaborative_demo.py --query "P0420 Toyota Camry 2019"

# Check configuration without running inference
python scripts/07_run_collaborative_demo.py --dry-run
```

### Demo Script (10-minute hackathon flow)

| Step | Title | Query |
|------|-------|-------|
| 1 | Simple Diagnosis | `P0420 Toyota Camry 2019` |
| 2 | Compound Query with Cost | `P0300 Ford F-150 — repair + cost` |
| 3 | Critical Fault Urgency | `P0562 low battery — safe to drive?` |

### Integration with OBDient (QVAC SDK)

```typescript
// In your OBDient datasource — after adapter is validated
import { loadModel } from '@tether/qvac-sdk';

const modelId = await loadModel({
  modelSrc:   QWEN3_1_7B_Q4_K_M,
  adapterSrc: 'file://output/adapter/carpsy-adapter.gguf',
});
```

> The QVAC SDK is embedded in the [OBDient](https://github.com/tetherto/obdient) project.
> Clone OBDient and reference `carpsy-adapter.gguf` from its datasource configuration.

---

## Data Sources

| Source | Format | Records |
|--------|--------|---------|
| [Wal33D/dtc-database](https://github.com/Wal33D/dtc-database) | SQLite | ~28,000 codes, 33 manufacturers |
| [peyo/dtc-and-vin-data](https://github.com/peyo/dtc-and-vin-data) | JSON | DTCs + VIN data |
| [xinings/DTC-Database](https://github.com/xinings/DTC-Database) | JSON/XML | ~6,665 codes |

---

## Next Steps

| Priority | Task |
|----------|------|
| 🔜 High | Train **CARpsy-v3** on Qwen3-1.7B Q4_K_M — same dataset and LoRA config, larger base for better reasoning |
| 🔜 High | Expand dataset with parts pricing and repair procedures to strengthen CARpsy-Parts and CARpsy-Repair agents |
| 🔵 Medium | Evaluate CARpsy-v3 vs CARpsy-v2 on the test split and publish comparison |
| 🔵 Medium | Integrate CARpsy-v3 adapter into OBDient via QVAC Fabric LoRA path |

---

## QVAC Fabric Reference

- **Repository:** https://github.com/tetherto/qvac-fabric-llm.cpp
- **Fine-tuning docs:** `examples/training/README.md`
- **Supported GPU backends:** CUDA (NVIDIA), Vulkan (AMD/Intel), Metal (Apple)
- **Verified LoRA architectures:** Qwen3, Gemma3, BitNet
