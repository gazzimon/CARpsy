# CARpsy — Fine-Tuning de LLaMA 3.2 1B para Diagnóstico Automotriz

**Fine-tuning LoRA del modelo LLaMA 3.2 1B (Q4) usando llama.cpp para mejorar OBDient.**

Este repositorio prepara datasets, ejecuta fine-tuning con LoRA/QLoRA y genera adaptadores GGUF que mejoran la precisión del diagnóstico automotriz. **Todo corre 100% local en tu PC.**

## ¿Qué es QVAC Fabric?

QVAC Fabric es un fork de [llama.cpp](https://github.com/ggerganov/llama.cpp) con soporte de fine-tuning LoRA. No es un servicio cloud — es un **binario compilado** que ejecutas en tu máquina.

- **Repo oficial:** https://github.com/tetherto/qvac-fabric-llm.cpp
- **Documentación de fine-tuning:** `examples/training/README.md`
- **No necesita API key. No sube datos a ningún servidor.**

## Requisitos de hardware

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| RAM | 16 GB | 32 GB |
| GPU | CPU (lento) | GPU con 8GB+ VRAM |
| Disco | 10 GB libres | 20 GB libres |
| OS | Windows/Linux/macOS | Linux con CUDA |

Para LLaMA 3.2 1B con LoRA (rank 8) se necesitan ~5-8 GB de RAM en entrenamiento.

## Estructura

```
CARpsy/
├── data/
│   ├── raw/              # Datos crudos (Wal33D/dtc-database, etc.)
│   ├── processed/        # Dataset transformado a formato chat JSONL
│   └── splits/           # Train/val/test splits (JSONL)
├── models/               # Modelos base GGUF (descargar aparte)
├── scripts/
│   ├── 01_fetch_dtc_data.py      # Descarga Wal33D/dtc-database
│   ├── 02_prepare_dataset.py     # Transforma DTCs a formato chat
│   ├── 03_split_dataset.py       # Divide en train/val/test
│   ├── 04_run_finetune.py        # EJECUTA llama-finetune-lora (LOCAL)
│   └── 05_evaluate_adapter.py    # Evalúa el adaptador generado
├── configs/
│   └── training_config.yaml      # Args para llama-finetune-lora
├── output/
│   └── adapter/                  # Adaptador LoRA generado (.gguf)
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### 1. Compilar llama.cpp (QVAC Fabric)

```bash
# Clonar
git clone https://github.com/tetherto/qvac-fabric-llm.cpp.git
cd qvac-fabric-llm.cpp

# Compilar (Windows - MSVC)
mkdir build && cd build
cmake .. -DLLAMA_CUDA=ON   # Si tienes GPU NVIDIA
cmake .. -DLLAMA_CUDA=OFF  # Solo CPU
cmake --build . --config Release

# Verificar que existe el binario
./build/bin/Release/llama-finetune-lora.exe --help  # Windows
./build/bin/llama-finetune-lora --help               # Linux/Mac
```

### 2. Descargar modelo base GGUF

Descargar LLaMA 3.2 1B Instruct (Q4_K_M) — el mismo modelo que usa OBDient:

```bash
# Opción A: Desde Hugging Face
# https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF

# Opción B: Usar el modelo que ya descargó OBDient en el teléfono
# (extraerlo del dispositivo Android)
```

Colocar el `.gguf` en `models/llama-3.2-1b-q4.gguf`

### 3. Preparar dataset

```bash
pip install -r requirements.txt
python scripts/01_fetch_dtc_data.py    # Descarga ~28,000 DTCs
python scripts/02_prepare_dataset.py   # Genera dataset en formato chat
python scripts/03_split_dataset.py     # Divide 80/10/10
```

### 4. Ejecutar fine-tuning (local)

```bash
python scripts/04_run_finetune.py
```

Esto ejecuta internamente:
```bash
llama-finetune-lora ^
  -m models/llama-3.2-1b-q4.gguf ^
  -f data/splits/train.jsonl ^
  --lora-rank 8 ^
  --lora-alpha 16 ^
  --lora-modules "attn_q,attn_k,attn_v,attn_o" ^
  --assistant-loss-only ^
  --output-adapter output/adapter/obdient-adapter.gguf ^
  -ngl 999 ^
  -c 512 -b 4 -ub 512 ^
  --learning-rate 2e-4 ^
  --checkpoint-save-steps 100 ^
  --checkpoint-save-dir output/checkpoints
```

### 5. Probar el adaptador

```bash
# Usar el modelo fine-tuneado con llama-cli
llama-cli -m models/llama-3.2-1b-q4.gguf ^
  --lora output/adapter/obdient-adapter.gguf ^
  -ngl 999 -p "I have code P0420 on my Toyota. What does it mean?"
```

## Integración con OBDient

Una vez generado el adaptador `obdient-adapter.gguf`, se integra en el datasource QVAC de OBDient:

```typescript
// En qvac-sdk.datasource.ts
const modelId = await loadModel({
  modelSrc: LLAMA_3_2_1B_INST_Q4_0,
  adapterSrc: 'file://obdient-adapter.gguf',  // ← adaptador fine-tuneado
});
```

## Formato del Dataset

El dataset usa formato **JSONL** con mensajes role/content:

```json
{"messages": [
  {"role": "system", "content": "You are OBDient, an expert automotive diagnostic assistant..."},
  {"role": "user", "content": "I'm getting code P0420 on my Toyota Corolla. What does it mean?"},
  {"role": "assistant", "content": "P0420: Catalyst System Efficiency Below Threshold..."}
]}
```

El flag `--assistant-loss-only` hace que el modelo solo aprenda de los tokens del asistente, ignorando el system prompt y la pregunta del usuario durante el entrenamiento.

## Fuentes de Datos

- **Wal33D/dtc-database** — ~28,000 códigos DTC de 33 marcas (SQLite)
- **peyo/dtc-and-vin-data** — DTCs + VINs en JSON (complemento)
- **xinings/DTC-Database** — ~6,665 códigos en XML/JSON