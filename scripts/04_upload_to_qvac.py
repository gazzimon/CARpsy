#!/usr/bin/env python3
"""
04_upload_to_qvac.py — Sube el dataset a QVAC Fabric y lanza fine-tuning.

NOTA: Este script es un template. QVAC Fabric puede requerir una API
específica o CLI. Ajustar según la documentación oficial de Fabric.

Flujo esperado:
  1. Autenticación con QVAC Fabric
  2. Subida del dataset (train.jsonl + val.jsonl)
  3. Configuración del job de fine-tuning (desde lora_config.yaml)
  4. Lanzamiento del job
  5. Monitoreo del progreso
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
CONFIG_PATH = REPO_ROOT / "configs" / "lora_config.yaml"
OUTPUT_DIR = REPO_ROOT / "output" / "adapter"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    """Carga un archivo JSONL."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ─── QVAC Fabric API (template) ──────────────────────────────────────────────
# NOTA: Reemplazar con la API real de QVAC Fabric

class QVACFabricClient:
    """
    Cliente template para QVAC Fabric.
    Reemplazar los métodos con la API/CLI real de Fabric.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.qvac.tether.io/v1"  # URL hipotética
        self.job_id = None

    def authenticate(self) -> bool:
        """Autenticación contra QVAC Fabric."""
        print("  [i] Autenticando con QVAC Fabric...")
        if not self.api_key:
            print("  [!] No API key configurada. Usando modo simulación.")
            return False
        # TODO: Implementar autenticación real
        # response = requests.post(f"{self.base_url}/auth", ...)
        print("  [✓] Autenticación simulada exitosa")
        return True

    def upload_dataset(self, train_path: Path, val_path: Path) -> Optional[str]:
        """Sube el dataset a QVAC Fabric."""
        print(f"\n  📤 Subiendo dataset...")
        print(f"     Train: {train_path}")
        print(f"     Val:   {val_path}")

        train_data = load_jsonl(train_path)
        val_data = load_jsonl(val_path)

        print(f"     Train examples: {len(train_data)}")
        print(f"     Val examples:   {len(val_data)}")

        # TODO: Implementar subida real
        # with open(train_path) as f:
        #     response = requests.post(f"{self.base_url}/datasets", files={...})
        # dataset_id = response.json()["id"]

        dataset_id = f"obdient-ds-{int(time.time())}"
        print(f"  [✓] Dataset subido (simulado). ID: {dataset_id}")
        return dataset_id

    def create_finetune_job(self, dataset_id: str, config: dict) -> Optional[str]:
        """Crea y lanza un job de fine-tuning."""
        print(f"\n  🚀 Creando job de fine-tuning...")
        print(f"     Dataset ID: {dataset_id}")
        print(f"     Modelo base: {config['model']['base']}")
        print(f"     LoRA rank: {config['lora']['rank']}")
        print(f"     Learning rate: {config['training']['learning_rate']}")
        print(f"     Épocas: {config['training']['num_epochs']}")

        # TODO: Implementar creación de job real
        # payload = {
        #     "model": config["model"]["base"],
        #     "dataset_id": dataset_id,
        #     "lora_config": config["lora"],
        #     "training_config": config["training"],
        # }
        # response = requests.post(f"{self.base_url}/finetune", json=payload)

        self.job_id = f"ft-{int(time.time())}"
        print(f"  [✓] Job creado (simulado). Job ID: {self.job_id}")
        return self.job_id

    def wait_for_completion(self, poll_interval: int = 30) -> bool:
        """Espera a que el job de fine-tuning termine."""
        print(f"\n  ⏳ Monitoreando job {self.job_id}...")
        print(f"     (Modo simulación: esperando 5s)")

        # TODO: Implementar polling real
        # while True:
        #     response = requests.get(f"{self.base_url}/finetune/{self.job_id}")
        #     status = response.json()["status"]
        #     if status == "completed": return True
        #     if status == "failed": return False
        #     time.sleep(poll_interval)

        time.sleep(5)
        print(f"  [✓] Fine-tuning completado (simulado)")
        return True

    def download_adapter(self, output_dir: Path) -> Optional[Path]:
        """Descarga el adaptador LoRA fine-tuneado."""
        print(f"\n  📥 Descargando adaptador...")

        # TODO: Implementar descarga real
        # response = requests.get(f"{self.base_url}/finetune/{self.job_id}/adapter")
        # with open(output_dir / "adapter.safetensors", "wb") as f:
        #     f.write(response.content)

        output_path = output_dir / "adapter.safetensors"
        print(f"  [✓] Adaptador descargado (simulado) en: {output_path}")
        return output_path


def main():
    print("=" * 60)
    print("CARpsy — Step 4: Upload to QVAC Fabric & Fine-Tune")
    print("=" * 60)

    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verificar que existen los splits
    train_path = SPLITS_DIR / "train.jsonl"
    val_path = SPLITS_DIR / "val.jsonl"

    if not train_path.exists():
        print(f"\n  [!] No se encuentra {train_path}")
        print("  [!] Ejecuta primero: python scripts/03_split_dataset.py")
        return

    # Cargar API key (desde variable de entorno o .env)
    import os
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    api_key = os.getenv("QVAC_FABRIC_API_KEY")
    if not api_key:
        print("\n  ⚠️  QVAC_FABRIC_API_KEY no configurada")
        print("     Crear archivo .env con: QVAC_FABRIC_API_KEY=tu_key")
        print("     Continuando en modo simulación...\n")

    # Inicializar cliente QVAC Fabric
    client = QVACFabricClient(api_key)

    # ─── 1. Autenticar ─────────────────────────────────────────────────────
    print("\n🔐 Step 1: Authentication")
    client.authenticate()

    # ─── 2. Subir dataset ──────────────────────────────────────────────────
    print("\n📦 Step 2: Upload Dataset")
    dataset_id = client.upload_dataset(train_path, val_path)
    if not dataset_id:
        print("  [!] Error subiendo dataset")
        return

    # ─── 3. Crear y lanzar fine-tuning ─────────────────────────────────────
    print("\n⚙️ Step 3: Create Fine-Tuning Job")
    job_id = client.create_finetune_job(dataset_id, config)
    if not job_id:
        print("  [!] Error creando job")
        return

    # ─── 4. Esperar completado ─────────────────────────────────────────────
    print("\n⏳ Step 4: Wait for Completion")
    success = client.wait_for_completion()
    if not success:
        print("  [!] Fine-tuning falló")
        return

    # ─── 5. Descargar adaptador ────────────────────────────────────────────
    print("\n💾 Step 5: Download Adapter")
    adapter_path = client.download_adapter(OUTPUT_DIR)

    if adapter_path:
        print(f"\n✅ Fine-tuning completo. Adaptador en: {adapter_path}")
        print(f"   Tamaño: {adapter_path.stat().st_size / 1024:.1f} KB")
        print(f"\n📋 Próximos pasos:")
        print(f"   1. Evaluar: python scripts/05_evaluate_adapter.py")
        print(f"   2. Integrar en OBDient vía adapterSrc en loadModel()")
    else:
        print("\n  [!] No se pudo descargar el adaptador")

    print("\n✅ Step 4 complete.")


if __name__ == "__main__":
    main()