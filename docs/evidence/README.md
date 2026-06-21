# Evidencia de Fine-Tuning — CARpsy

Esta carpeta contiene la prueba de que el modelo fue realmente afinado (no es el modelo base sin tocar).

## Archivos

| Archivo | Qué demuestra |
|---------|---------------|
| `Finetune curve.png` | Curva de training loss: baja de ~4.1 a ~0.05 hacia el step 250 y se estabiliza. Prueba directa de aprendizaje real. |
| `colab_finetune.ipynb` | Notebook ejecutado con outputs (logs de entrenamiento, métricas por step) exportado desde la sesión real de Colab. |
| `../../output/adapter/evaluation_report.json` | Evaluación sobre 1.532 ejemplos del test set: code_identification_rate, structure_rate, etc. |
| `../../output/adapter/validation_report.json` | 8 casos de validación con las respuestas reales del modelo afinado (pass_rate 0.88). |

## Modelo entrenado (GGUF)

Pesos publicados en Hugging Face Hub:

- **https://huggingface.co/gazzimon/CARpsy-v2-qwen3-0.6b-GGUF**

Comparar estos pesos contra el modelo base (Qwen3-0.6B) confirma que difieren — evidencia objetiva del fine-tuning.

## Cómo reproducir

1. Dataset y splits versionados en [`data/splits/`](../../data/splits/) (train 12.395 / val 1.549 / test 1.550).
2. Config de entrenamiento en [`configs/`](../../configs/) (LoRA r=8, alpha=16).
3. Notebook de entrenamiento: [`colab_finetune.ipynb`](../../colab_finetune.ipynb).
