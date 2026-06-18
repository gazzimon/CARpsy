#!/usr/bin/env python3
"""
09_build_canonical_dataset.py — Dataset canónico de alta calidad para CARpsy.

Diseño:
  - 20 códigos DTC más frecuentes en talleres reales
  - 1 respuesta canónica FIJA por código (estilo BLUCKTEC 440)
  - 15 variantes de pregunta por código
  - Formato uniforme: código | nombre SAE | severidad | acción | causas | veredicto

Formato de respuesta canónica:
  {CODE}: {NOMBRE OFICIAL SAE}.
  Severity {N}/3 — {INSTRUCCIÓN DE ACCIÓN}.
  Likely causes: {CAUSA 1}, {CAUSA 2}.
  {SAFE_TO_DRIVE_VERDICT}.

Uso:
  python scripts/09_build_canonical_dataset.py
  python scripts/09_build_canonical_dataset.py --preview   # muestra 3 ejemplos sin guardar
"""

import json
import argparse
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_PATH   = PROCESSED_DIR / "canonical_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are CARpsy, an expert automotive diagnostic assistant specialized in OBD-II fault codes. "
    "When given a DTC code, identify it precisely, state its severity, list the most likely causes, "
    "and advise whether the vehicle is safe to drive. "
    "Always respond in English. Be concise and direct. Prioritize safety above all."
)

# ─── Base de conocimiento canónica ────────────────────────────────────────────
# Cada entrada: código → respuesta canónica FIJA (nunca generada, nunca aleatoria)
# Severity: 1/3 = info/emissions  |  2/3 = warning/performance  |  3/3 = critical/safety

CANONICAL = {
    "P0420": (
        "P0420: Catalyst System Efficiency Below Threshold (Bank 1). "
        "Severity 2/3 — Schedule repair soon; ignoring risks catalytic converter failure and increased emissions. "
        "Likely causes: faulty catalytic converter, defective downstream O2 sensor, exhaust leak before sensor. "
        "Safe to drive short distances to the shop."
    ),
    "P0300": (
        "P0300: Random or Multiple Cylinder Misfire Detected. "
        "Severity 3/3 — Do not drive; misfires destroy the catalytic converter within minutes and risk engine fire. "
        "Likely causes: worn spark plugs, failed ignition coils, clogged fuel injectors, vacuum leak. "
        "Do not drive — stop the engine immediately if the check engine light is flashing."
    ),
    "P0301": (
        "P0301: Cylinder 1 Misfire Detected. "
        "Severity 3/3 — Repair immediately; sustained misfiring causes catalytic converter and engine damage. "
        "Likely causes: faulty spark plug (cylinder 1), bad ignition coil (cylinder 1), clogged fuel injector. "
        "Do not drive."
    ),
    "P0171": (
        "P0171: System Too Lean (Bank 1) — air-fuel mixture has too much air or too little fuel. "
        "Severity 2/3 — Schedule repair; prolonged lean condition causes engine wear and converter damage. "
        "Likely causes: vacuum or intake air leak, dirty MAF sensor, weak fuel pump. "
        "Safe to drive short distances — avoid extended highway driving."
    ),
    "P0128": (
        "P0128: Coolant Temperature Below Thermostat Regulating Temperature — engine runs too cold. "
        "Severity 2/3 — Schedule repair; chronic cold running causes sludge buildup and increased engine wear. "
        "Likely causes: thermostat stuck open, faulty coolant temperature sensor (ECT). "
        "Safe to drive short distances."
    ),
    "P0562": (
        "P0562: System Voltage Low — PCM detected battery voltage below calibrated threshold. "
        "Severity 3/3 — Do not drive; engine can stall without warning and electrical systems may fail. "
        "Likely causes: failing alternator, worn-out battery, loose or corroded battery cables. "
        "Do not drive — test battery and alternator before starting the engine again."
    ),
    "P0455": (
        "P0455: EVAP System Large Leak Detected — fuel vapors escaping to atmosphere. "
        "Severity 1/3 — Address at next service; no immediate drivability risk but will fail emissions test. "
        "Likely causes: loose or faulty gas cap, cracked EVAP hose, faulty purge valve. "
        "Safe to drive — tighten the gas cap first and clear the code."
    ),
    "P0442": (
        "P0442: EVAP System Small Leak Detected. "
        "Severity 1/3 — Address at next service; safe to drive but vehicle will fail emissions test. "
        "Likely causes: faulty or loose gas cap, small crack in EVAP hose, faulty purge valve. "
        "Safe to drive — start diagnosis by replacing the gas cap."
    ),
    "P0087": (
        "P0087: Fuel Rail Pressure Too Low — PCM detects fuel pressure below minimum for combustion. "
        "Severity 3/3 — Do not drive; engine will stall without warning when pressure drops. "
        "Likely causes: clogged fuel filter, failing fuel pump, faulty fuel pressure regulator. "
        "Do not drive — replace the fuel filter first as it is the most common cause."
    ),
    "P0335": (
        "P0335: Crankshaft Position Sensor 'A' Circuit Malfunction — no or erratic signal from crank sensor. "
        "Severity 3/3 — Do not drive; engine may stall without warning and will not restart. "
        "Likely causes: faulty crankshaft position sensor, damaged reluctor wheel, damaged wiring. "
        "Do not drive."
    ),
    "P0340": (
        "P0340: Camshaft Position Sensor 'A' Circuit Malfunction (Bank 1) — no or erratic cam signal. "
        "Severity 3/3 — Do not drive; engine may stall and will have difficulty restarting. "
        "Likely causes: faulty camshaft position sensor, broken sensor wiring, stretched timing chain. "
        "Do not drive."
    ),
    "P0016": (
        "P0016: Crankshaft/Camshaft Position Correlation Error (Bank 1 Sensor A) — cam and crank out of sync. "
        "Severity 3/3 — Do not drive; a jumped timing chain causes catastrophic engine damage (bent valves). "
        "Likely causes: stretched or jumped timing chain, worn tensioner, low oil pressure, oil sludge in VVT. "
        "Do not drive — check oil level immediately and do not restart the engine."
    ),
    "P0011": (
        "P0011: Camshaft Position Timing Over-Advanced (Bank 1). "
        "Severity 2/3 — Do not drive; can progress rapidly to timing chain damage and engine failure. "
        "Likely causes: low or dirty engine oil (most common), stuck VVT solenoid, timing chain stretch. "
        "Do not drive — change the engine oil and filter immediately as the first step."
    ),
    "P0700": (
        "P0700: Transmission Control System Malfunction — TCM has detected a fault and stored additional codes. "
        "Severity 2/3 — Do not drive; P0700 is a gateway code, scan for P07xx codes to find the real fault. "
        "Likely causes: secondary transmission fault, low or dirty transmission fluid, faulty shift solenoid. "
        "Do not drive until all related codes are diagnosed."
    ),
    "P0325": (
        "P0325: Knock Sensor 1 Circuit Malfunction (Bank 1) — knock sensor signal absent or out of range. "
        "Severity 2/3 — Schedule repair; ECM retards ignition timing causing performance and fuel economy loss. "
        "Likely causes: faulty knock sensor, loose sensor mounting bolt, damaged sensor wiring. "
        "Safe to drive short distances — avoid high-load driving until repaired."
    ),
    "P0101": (
        "P0101: MAF Sensor Circuit Range/Performance — mass air flow readings are out of expected range. "
        "Severity 2/3 — Schedule repair promptly; reduced power and hesitation create a driving hazard. "
        "Likely causes: dirty or contaminated MAF sensor, vacuum leak after the MAF, defective MAF sensor. "
        "Not recommended to drive — clean the MAF sensor with MAF-specific spray as a first step."
    ),
    "P0401": (
        "P0401: EGR Flow Insufficient Detected — exhaust gas recirculation flow is below specification. "
        "Severity 2/3 — Schedule repair; engine may knock under load and NOx emissions increase. "
        "Likely causes: clogged or stuck EGR valve, carbon-blocked EGR passages, faulty EGR solenoid. "
        "Safe to drive short distances — remove and clean the EGR valve with carbon cleaner."
    ),
    "P0507": (
        "P0507: Idle Control System RPM High — engine idle speed is higher than the target. "
        "Severity 2/3 — Schedule repair; high idle wastes fuel and stresses the drivetrain. "
        "Likely causes: faulty IAC valve, vacuum leak at throttle body, dirty throttle body. "
        "Safe to drive — clean the throttle body as the first diagnostic step."
    ),
    "P0302": (
        "P0302: Cylinder 2 Misfire Detected. "
        "Severity 3/3 — Repair immediately; catalytic converter can be destroyed within minutes of sustained misfiring. "
        "Likely causes: faulty spark plug (cylinder 2), bad ignition coil (cylinder 2), clogged fuel injector. "
        "Do not drive — swap cylinder 2 coil with a known-good cylinder to isolate the fault."
    ),
    "P0520": (
        "P0520: Engine Oil Pressure Sensor/Switch Circuit Malfunction. "
        "Severity 3/3 — Do not drive; verify actual oil pressure immediately — low oil pressure causes catastrophic engine damage. "
        "Likely causes: faulty oil pressure sensor, low actual engine oil pressure, damaged sensor wiring. "
        "Do not drive — check oil level first; if normal, test actual pressure with a mechanical gauge."
    ),
}

# ─── Variantes de pregunta (15 por código) ────────────────────────────────────
# El código siempre aparece explícitamente en la pregunta para anclar el aprendizaje

QUESTION_TEMPLATES = [
    "I have code {code} on my {make} {model}. What does it mean?",
    "My check engine light is on with code {code}. Is it serious?",
    "What is DTC {code} and how serious is it?",
    "Code {code} appeared on my scanner. What should I do?",
    "Can you explain fault code {code}?",
    "Is code {code} critical? I drive a {make} {model}.",
    "What does {code} mean in plain English?",
    "My {make} {model} shows {code}. Do I need to go to a mechanic right away?",
    "Code {code} on my {make} {model} {year}. What are the common causes?",
    "Just scanned my car and got {code}. Your diagnosis?",
    "Check engine showing {code}. How urgent is this?",
    "Got {code} after the engine light came on. What does it mean?",
    "{make} {model} is showing {code}. Is it safe to drive?",
    "Mechanic said I have {code}. What should I know?",
    "My scanner reads {code}. What parts should I inspect?",
]

VEHICLES = [
    ("Toyota",      "Camry",         "2019"),
    ("Ford",        "F-150",         "2018"),
    ("Honda",       "Civic",         "2020"),
    ("Chevrolet",   "Silverado",     "2017"),
    ("Toyota",      "RAV4",          "2021"),
    ("Nissan",      "Altima",        "2016"),
    ("Ford",        "Escape",        "2019"),
    ("Honda",       "CR-V",          "2020"),
    ("Hyundai",     "Elantra",       "2018"),
    ("Jeep",        "Grand Cherokee","2015"),
    ("Dodge",       "Ram 1500",      "2019"),
    ("BMW",         "320i",          "2018"),
    ("Volkswagen",  "Jetta",         "2017"),
    ("Subaru",      "Outback",       "2020"),
    ("Kia",         "Sportage",      "2019"),
]


def generate_examples(code: str, canonical_response: str) -> list[dict]:
    examples = []
    for i, template in enumerate(QUESTION_TEMPLATES):
        make, model, year = VEHICLES[i % len(VEHICLES)]
        user_msg = template.format(code=code, make=make, model=model, year=year)
        examples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": canonical_response},
            ]
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Genera dataset canónico CARpsy")
    parser.add_argument("--preview", action="store_true", help="Muestra 3 ejemplos sin guardar")
    args = parser.parse_args()

    print("=" * 60)
    print("CARpsy  Step 9: Build Canonical Dataset")
    print("=" * 60)
    print(f"  Códigos:              {len(CANONICAL)}")
    print(f"  Variantes por código: {len(QUESTION_TEMPLATES)}")
    print(f"  Total ejemplos:       {len(CANONICAL) * len(QUESTION_TEMPLATES)}")

    all_examples = []
    for code, response in CANONICAL.items():
        examples = generate_examples(code, response)
        all_examples.extend(examples)
        print(f"  {code}: {len(examples)} ejemplos")

    if args.preview:
        print("\n" + "=" * 60)
        print("PREVIEW — primeros 3 ejemplos")
        print("=" * 60)
        for ex in all_examples[:3]:
            msgs = ex["messages"]
            print(f"\n  USER:      {msgs[1]['content']}")
            print(f"  ASSISTANT: {msgs[2]['content']}")
        print("\n  (Sin --preview se guarda el dataset completo)")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n  Guardado en: {OUTPUT_PATH}")
    print(f"  Total líneas: {len(all_examples)}")
    print("\n  Próximo paso:")
    print("  Sube canonical_dataset.jsonl a Colab y úsalo como dataset de entrenamiento.")
    print("  Hiperparámetros recomendados para re-entrenamiento:")
    print("    - epochs:        3")
    print("    - learning_rate: 5e-5")
    print("    - lora_r:        16")
    print("    - lora_alpha:    32")
    print("\n[OK] Step 9 complete.")


if __name__ == "__main__":
    main()
