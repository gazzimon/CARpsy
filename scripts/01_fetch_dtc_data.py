#!/usr/bin/env python3
"""
01_fetch_dtc_data.py — Descarga Wal33D/dtc-database y otras fuentes DTC.

Este script clona o actualiza el repositorio Wal33D/dtc-database en data/raw/
y extrae la información relevante para el fine-tuning.

Fuentes:
  - Principal: Wal33D/dtc-database (SQLite con ~28,000 códigos de 33 marcas)
  - Complementaria: peyo/dtc-and-vin-data (JSON)
"""

import os
import sys
import json
import sqlite3
import shutil
from pathlib import Path
from typing import Optional

# Asegurar que podemos importar desde el directorio padre
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import requests

# ─── Rutas ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_PATH = REPO_ROOT / "configs" / "lora_config.yaml"

# Repositorios a descargar
DTC_REPOS = {
    "wal33d": {
        "url": "https://github.com/Wal33D/dtc-database.git",
        "dir": RAW_DIR / "dtc-database",
    },
    "peyo": {
        "url": "https://github.com/peyo/dtc-and-vin-data.git",
        "dir": RAW_DIR / "dtc-and-vin-data",
    },
}


def load_config() -> dict:
    """Carga la configuración desde lora_config.yaml"""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def clone_or_pull_repo(name: str, url: str, dest: Path) -> None:
    """Clona un repo si no existe, o hace pull si ya existe."""
    if dest.exists():
        print(f"  [{name}] Repo exists at {dest}, pulling latest...")
        os.system(f'cd /d "{dest}" && git pull 2>&1')
    else:
        print(f"  [{name}] Cloning {url}...")
        os.system(f'git clone --depth 1 "{url}" "{dest}" 2>&1')


def extract_wal33d_database(db_path: Path) -> list[dict]:
    """
    Extrae los DTCs de la base de datos SQLite de Wal33D.
    La estructura típica tiene tablas como: dtc_codes, manufacturers, vehicles.
    """
    if not db_path.exists():
        print(f"  [!] Database not found at {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Explorar esquema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"  [i] Tablas encontradas: {tables}")

    dtc_records = []

    if "dtc_codes" in tables:
        # Obtener columnas
        cursor.execute("PRAGMA table_info(dtc_codes);")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"  [i] Columnas de dtc_codes: {columns}")

        # Determinar qué columnas están disponibles
        col_map = {
            "code": next((c for c in columns if c.lower() in ("code", "dtc", "dtc_code", "fault_code")), None),
            "description": next((c for c in columns if c.lower() in ("description", "definition", "desc", "meaning")), None),
            "system": next((c for c in columns if c.lower() in ("system", "category", "type")), None),
            "manufacturer": next((c for c in columns if c.lower() in ("manufacturer", "make", "brand")), None),
        }
        print(f"  [i] Mapeo de columnas: {col_map}")

        # Consultar todos los registros
        query_cols = [v for v in col_map.values() if v]
        if query_cols:
            cursor.execute(f"SELECT {', '.join(query_cols)} FROM dtc_codes LIMIT 10")
            sample = cursor.fetchall()
            print(f"  [i] Muestra de datos: {sample[:3]}")

            # Traer todos
            cursor.execute(f"SELECT {', '.join(query_cols)} FROM dtc_codes")
            for row in cursor.fetchall():
                record = {}
                for i, col in enumerate(query_cols):
                    record[col] = row[i]
                dtc_records.append(record)

    conn.close()
    print(f"  [✓] Extraídos {len(dtc_records)} registros DTC de Wal33D")
    return dtc_records


def extract_peyo_data(data_dir: Path) -> list[dict]:
    """Extrae DTCs de los archivos JSON de peyo/dtc-and-vin-data."""
    records = []

    # Buscar archivos JSON relacionados con DTC
    json_files = list(data_dir.rglob("*.json"))
    dtc_files = [f for f in json_files if "dtc" in f.name.lower()]

    for file_path in dtc_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Puede ser lista o dict
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)

            print(f"  [i] {file_path.name}: {len(data) if isinstance(data, list) else 1} registros")
        except Exception as e:
            print(f"  [!] Error leyendo {file_path}: {e}")

    print(f"  [✓] Extraídos {len(records)} registros de peyo/dtc-and-vin-data")
    return records


def save_raw_data(records: list[dict], filename: str) -> None:
    """Guarda los datos extraídos como JSON en data/raw/."""
    output_path = RAW_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  [✓] Guardados {len(records)} registros en {output_path}")


def generate_summary(all_records: list[dict]) -> dict:
    """Genera un resumen estadístico de los datos extraídos."""
    systems = set()
    manufacturers = set()

    for rec in all_records:
        for key in rec:
            if "system" in key.lower() and rec[key]:
                systems.add(str(rec[key]))
            if "manufacturer" in key.lower() and rec[key]:
                manufacturers.add(str(rec[key]))

    return {
        "total_records": len(all_records),
        "unique_systems": sorted(systems),
        "unique_manufacturers": sorted(manufacturers),
        "systems_count": len(systems),
        "manufacturers_count": len(manufacturers),
    }


def main():
    print("=" * 60)
    print("CARpsy — Step 1: Fetch DTC Data")
    print("=" * 60)

    config = load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []

    # ─── 1. Wal33D/dtc-database ────────────────────────────────────────────
    print("\n📦 Wal33D/dtc-database")
    repo_info = DTC_REPOS["wal33d"]
    clone_or_pull_repo("wal33d", repo_info["url"], repo_info["dir"])

    # Buscar archivos .db o .sqlite
    db_files = list(repo_info["dir"].rglob("*.db")) + list(repo_info["dir"].rglob("*.sqlite"))
    if db_files:
        records = extract_wal33d_database(db_files[0])
        all_records.extend(records)
        save_raw_data(records, "wal33d_dtc_raw.json")
    else:
        print("  [!] No se encontró archivo de base de datos. Buscando JSON...")
        json_files = list(repo_info["dir"].rglob("*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                print(f"  [i] Cargado {jf.name}: {len(data) if isinstance(data, list) else 1} registros")
            except:
                pass

    # ─── 2. peyo/dtc-and-vin-data ──────────────────────────────────────────
    print("\n📦 peyo/dtc-and-vin-data")
    repo_info = DTC_REPOS["peyo"]
    clone_or_pull_repo("peyo", repo_info["url"], repo_info["dir"])

    peyo_records = extract_peyo_data(repo_info["dir"])
    all_records.extend(peyo_records)
    save_raw_data(peyo_records, "peyo_dtc_raw.json")

    # ─── Resumen ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    summary = generate_summary(all_records)
    print(f"  Total registros DTC: {summary['total_records']}")
    print(f"  Sistemas: {summary['systems_count']}")
    print(f"  Fabricantes: {summary['manufacturers_count']}")

    # Guardar resumen
    summary_path = RAW_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Resumen guardado en {summary_path}")

    print("\n✅ Step 1 complete. Dataset raw listo para preparación.")


if __name__ == "__main__":
    main()