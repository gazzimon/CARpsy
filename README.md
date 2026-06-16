# CARpsy — QVAC Fine-Tuning for OBDient

**Fine-tuning del modelo LLaMA 3.2 1B para diagnóstico automotriz usando QVAC Fabric + QLoRA.**

Este repositorio contiene las herramientas para preparar datasets, hacer fine-tuning y generar adaptadores LoRA que mejoran la precisión del asistente OBDient en diagnóstico de códigos DTC.

## Estructura

```
CARpsy/
├── data/
│   ├── raw/              # Datos crudos (Wal33D/dtc-database, etc.)
│   ├── processed/        # Datos transformados a formato chat JSONL
│   └── splits/           # Train/val/test splits
├── scripts/
│   ├── 01_fetch_dtc_data.py      # Descarga Wal33D/dtc-database
│   ├── 02_prepare_dataset.py     # Transforma DTCs a formato chat
│   ├── 03_split_dataset.py       # Divide en train/val/test
│   ├── 04_upload_to_qvac.py      # Sube dataset a QVAC Fabric
│   └── 05_evaluate_adapter.py    # Evalúa el adaptador fine-tuneado
├── configs/
│   └── lora_config.yaml          # Configuración de QLoRA
├── output/
│   └── adapter/                  # Adaptador LoRA generado (.safetensors)
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10+
- Acceso a [QVAC Fabric](https://qvac.tether.io/dev/fabric/)
- GPU recomendada (o CPU para QLoRA con rank bajo)

## Uso

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar datos DTC
python scripts/01_fetch_dtc_data.py

# 3. Preparar dataset en formato chat
python scripts/02_prepare_dataset.py

# 4. Dividir en train/val/test
python scripts/03_split_dataset.py

# 5. Subir dataset a QVAC Fabric y lanzar fine-tuning
python scripts/04_upload_to_qvac.py

# 6. Evaluar el adaptador resultante
python scripts/05_evaluate_adapter.py
```

## Flujo de Fine-Tuning con QVAC Fabric

1. **Preparación**: Convertir DTCs de Wal33D/dtc-database a conversaciones (system/user/assistant)
2. **Fine-Tuning**: Usar QVAC Fabric con QLoRA (rank 8-16, target: q_proj, v_proj)
3. **Exportación**: Obtener adaptador .safetensors (~10-50MB)
4. **Integración**: Cargar adaptador en OBDient via `adapterSrc` en `loadModel()`

## Fuentes de Datos

- **Wal33D/dtc-database** — ~28,000 códigos DTC de 33 marcas (fuente principal)
- **peyo/dtc-and-vin-data** — DTCs + VINs en JSON (complemento)
- **OBD Knowledge Base** — Conocimiento existente en OBDient (`src/data/knowledge/`)