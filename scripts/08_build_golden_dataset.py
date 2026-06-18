#!/usr/bin/env python3
"""
08_build_golden_dataset.py — Genera ejemplos "golden" de alta calidad para los
códigos DTC más comunes, basados en fuentes verificadas de la web.

Los golden examples tienen:
  - Descripciones correctas y verificadas
  - Causas probables específicas
  - Nivel de severidad real
  - Si es seguro conducir
  - Estimación de costo de reparación
  - Múltiples variantes de pregunta por código

Fuentes usadas:
  - fixdapp.com (top 10 más comunes + costos)
  - carparts.com/blog (definiciones + causas)
  - mechanicbase.com (causas + pasos diagnóstico)
  - obd-codes.com (definición oficial SAE)

Uso:
  python scripts/08_build_golden_dataset.py
  python scripts/08_build_golden_dataset.py --merge   # fusiona con el dataset existente
"""

import json
import random
import argparse
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLITS_DIR    = REPO_ROOT / "data" / "splits"
OUTPUT_PATH   = PROCESSED_DIR / "golden_examples.jsonl"

SYSTEM_PROMPT = (
    "You are OBDient, an expert automotive diagnostic assistant. "
    "You receive OBD-II fault codes and vehicle data. "
    "Explain what each code means, its severity, and recommended actions. "
    "Always respond in English, clearly and concisely. "
    "Maximum 3 sentences. Prioritize safety. "
    "If something is urgent, indicate it clearly."
)

# ─── Base de conocimiento verificada ──────────────────────────────────────────
# Estructura: code → {definition, causes, severity, safe_to_drive, cost, fix}
# severity: "critical" | "warning" | "info"
# Sources: fixdapp.com, carparts.com, mechanicbase.com, obd-codes.com

DTC_KNOWLEDGE = {
    "P0420": {
        "definition": "Catalyst System Efficiency Below Threshold (Bank 1)",
        "causes": ["faulty catalytic converter", "defective downstream O2 sensor", "exhaust leak", "damaged spark plugs", "high fuel pressure"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive short distances, but a clogged catalyst can destroy the engine.",
        "parts": ["catalytic converter", "downstream O2 sensor"],
        "fix": "Diagnose O2 sensor readings first; replace catalytic converter if confirmed faulty.",
    },
    "P0430": {
        "definition": "Catalyst System Efficiency Below Threshold (Bank 2)",
        "causes": ["faulty catalytic converter on Bank 2", "damaged O2 sensor (Bank 2)", "exhaust leak", "faulty fuel injectors"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Vehicle runs, but increased emissions and potential engine damage if converter is blocked.",
        "parts": ["catalytic converter (Bank 2)", "O2 sensor (Bank 2)"],
        "fix": "Test O2 sensor output and inspect exhaust for leaks before replacing the catalytic converter.",
    },
    "P0300": {
        "definition": "Random or Multiple Cylinder Misfire Detected",
        "causes": ["worn spark plugs", "faulty ignition coils", "clogged fuel injectors", "vacuum leaks", "failed head gasket", "low fuel pressure"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — severe misfires can damage the catalytic converter in minutes and cause engine fire.",
        "parts": ["spark plugs (all cylinders)", "ignition coils", "fuel injectors"],
        "fix": "Check spark plugs and ignition coils first. A flashing check engine light means stop driving immediately.",
    },
    "P0301": {
        "definition": "Cylinder 1 Misfire Detected",
        "causes": ["faulty spark plug on cylinder 1", "bad ignition coil on cylinder 1", "clogged fuel injector", "low compression on cylinder 1"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — continued misfiring damages the catalytic converter and can cause engine failure.",
        "parts": ["spark plug (cylinder 1)", "ignition coil (cylinder 1)", "fuel injector (cylinder 1)"],
        "fix": "Swap the cylinder 1 spark plug and ignition coil with a known-good cylinder to isolate the fault.",
    },
    "P0302": {
        "definition": "Cylinder 2 Misfire Detected",
        "causes": ["faulty spark plug on cylinder 2", "bad ignition coil on cylinder 2", "clogged fuel injector", "low compression on cylinder 2"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — continued misfiring can destroy the catalytic converter.",
        "parts": ["spark plug (cylinder 2)", "ignition coil (cylinder 2)", "fuel injector (cylinder 2)"],
        "fix": "Swap cylinder 2 spark plug and coil with a known-good cylinder to isolate the fault.",
    },
    "P0303": {
        "definition": "Cylinder 3 Misfire Detected",
        "causes": ["worn spark plug on cylinder 3", "ignition coil failure on cylinder 3", "faulty injector", "spark plug wire issue"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Stop driving — misfires overheat and damage the catalytic converter.",
        "parts": ["spark plug (cylinder 3)", "ignition coil (cylinder 3)", "fuel injector (cylinder 3)"],
        "fix": "Replace the cylinder 3 spark plug as a first step; swap ignition coil to verify.",
    },
    "P0171": {
        "definition": "System Too Lean (Bank 1) — air-fuel mixture has too much air or too little fuel on Bank 1",
        "causes": ["vacuum or intake air leak", "dirty or faulty MAF sensor", "faulty PCV valve", "weak fuel pump or clogged filter", "bad O2 sensor", "exhaust leak before upstream O2 sensor"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Short drives to the shop are okay; prolonged lean conditions cause engine wear and catalytic converter damage.",
        "parts": ["MAF sensor", "PCV valve", "vacuum hoses", "upstream O2 sensor", "fuel filter"],
        "fix": "Inspect all vacuum hoses for cracks, clean the MAF sensor, and test fuel pressure.",
    },
    "P0174": {
        "definition": "System Too Lean (Bank 2) — air-fuel mixture has too much air or too little fuel on Bank 2",
        "causes": ["vacuum leaks at intake manifold gaskets", "malfunctioning MAF sensor", "clogged fuel filter", "faulty O2 sensor on Bank 2", "exhaust leak before O2 sensor"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — prolonged lean condition causes misfires, overheating, and catalytic converter damage.",
        "parts": ["intake manifold gasket", "MAF sensor", "O2 sensor (Bank 2)", "fuel filter"],
        "fix": "Check for vacuum leaks using smoke testing; clean or replace MAF sensor; verify fuel delivery.",
    },
    "P0128": {
        "definition": "Coolant Temperature Below Thermostat Regulating Temperature — engine runs too cold",
        "causes": ["thermostat stuck open or opening too early", "faulty coolant temperature sensor (ECT)", "wiring issue to ECT sensor", "low coolant level with air pockets"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive short distances; chronic cold running causes sludge buildup and increased engine wear.",
        "parts": ["thermostat", "coolant temperature sensor (ECT)"],
        "fix": "Replace the thermostat first — it resolves the majority of P0128 cases.",
    },
    "P0562": {
        "definition": "System Voltage Low — PCM detected battery/system voltage below the calibrated threshold",
        "causes": ["failing alternator or voltage regulator", "worn-out battery", "loose or corroded battery cables", "parasitic battery drain", "blown fuse in charging circuit"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — the engine can stall without warning, and electrical systems may fail unpredictably.",
        "parts": ["battery", "alternator", "voltage regulator", "battery cables"],
        "fix": "Test battery voltage (should be 12.6V+ off, 13.5–14.5V running); test alternator output.",
    },
    "P0455": {
        "definition": "Evaporative Emission Control System (EVAP) Large Leak Detected",
        "causes": ["loose, missing, or faulty gas cap (most common)", "cracked EVAP hose", "faulty purge or vent valve", "damaged charcoal canister", "cracked fuel tank"],
        "severity": "info",
        "safe_to_drive": True,
        "safe_note": "Safe to drive, but excess fuel vapors escape into the atmosphere; may fail emissions test.",
        "parts": ["gas cap", "EVAP purge valve", "EVAP vent valve", "charcoal canister", "EVAP hoses"],
        "fix": "First, tighten or replace the gas cap and clear the code. If it returns, smoke-test the EVAP system.",
    },
    "P0449": {
        "definition": "Evaporative Emission System Vent Valve/Solenoid Circuit Malfunction",
        "causes": ["faulty EVAP vent valve solenoid", "damaged wiring to solenoid", "loose or defective fuel cap", "torn EVAP hoses", "bad carbon canister"],
        "severity": "info",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; no performance impact, but will fail emissions testing.",
        "parts": ["EVAP vent valve solenoid", "gas cap", "carbon canister", "EVAP hoses"],
        "fix": "Check the gas cap first. If the code persists, test the vent valve solenoid with a multimeter.",
    },
    "P0101": {
        "definition": "Mass Air Flow (MAF) Sensor Circuit Range/Performance — MAF sensor readings are out of expected range",
        "causes": ["dirty or contaminated MAF sensor wire", "defective MAF sensor", "vacuum leaks", "faulty throttle position sensor", "PCM malfunction"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Reduced power and hesitation make this a driving hazard; schedule repair promptly.",
        "parts": ["MAF sensor", "air filter", "throttle position sensor"],
        "fix": "Clean the MAF sensor with MAF-specific spray cleaner as a first step; replace if cleaning fails.",
    },
    "P0113": {
        "definition": "Intake Air Temperature (IAT) Sensor Circuit High Input — IAT sensor signal is too high",
        "causes": ["faulty IAT sensor", "damaged or shorted IAT sensor wiring", "IAT sensor connector corrosion", "PCM fault (rare)"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive short distances; incorrect air temperature readings affect fuel economy and performance.",
        "parts": ["IAT sensor", "IAT sensor wiring harness"],
        "fix": "Inspect IAT sensor wiring for damage; replace the sensor if voltage reads above 4.9V at key-on.",
    },
    "P0118": {
        "definition": "Engine Coolant Temperature (ECT) Sensor Circuit High Input — ECT sensor reads abnormally high",
        "causes": ["faulty ECT sensor", "open circuit in ECT sensor wiring", "corroded ECT sensor connector", "low coolant level causing sensor exposure to air"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Do not drive — a faulty coolant sensor can mask real overheating and cause engine damage.",
        "parts": ["ECT sensor", "ECT sensor wiring harness"],
        "fix": "Check coolant level first; inspect sensor wiring for open circuits; replace the ECT sensor.",
    },
    "P0335": {
        "definition": "Crankshaft Position Sensor 'A' Circuit Malfunction — no or erratic signal from crankshaft position sensor",
        "causes": ["faulty crankshaft position sensor", "damaged reluctor wheel (tone ring)", "damaged sensor wiring or connector", "PCM fault"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — the engine may stall without warning and will not restart.",
        "parts": ["crankshaft position sensor", "reluctor wheel"],
        "fix": "Replace the crankshaft position sensor; inspect the reluctor wheel for missing teeth.",
    },
    "P0340": {
        "definition": "Camshaft Position Sensor 'A' Circuit Malfunction (Bank 1) — no or erratic signal from camshaft sensor",
        "causes": ["faulty camshaft position sensor", "damaged reluctor ring on camshaft", "broken or shorted sensor wiring", "timing chain stretched or jumped"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — engine may stall and will have difficulty restarting.",
        "parts": ["camshaft position sensor", "timing chain kit"],
        "fix": "Replace the camshaft position sensor; verify timing marks align correctly.",
    },
    "P0700": {
        "definition": "Transmission Control System Malfunction — the TCM has detected a fault and stored additional transmission codes",
        "causes": ["secondary transmission fault codes present (check for P07xx codes)", "low or dirty transmission fluid", "faulty solenoid", "wiring issue in TCM circuit", "failing TCM"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — P0700 is a gateway code; the real fault may cause sudden loss of drive.",
        "parts": ["transmission fluid", "transmission solenoid pack", "TCM"],
        "fix": "Scan for additional P07xx or P08xx codes stored alongside P0700 — those reveal the actual fault.",
    },
    "P0741": {
        "definition": "Torque Converter Clutch Circuit Performance or Stuck Off — TCC is not engaging properly",
        "causes": ["faulty torque converter clutch solenoid", "low or contaminated transmission fluid", "worn torque converter", "internal valve body issue", "TCM fault"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Avoid highway driving — TCC failure increases transmission heat and can cause complete transmission failure.",
        "parts": ["TCC solenoid", "torque converter", "transmission fluid", "valve body"],
        "fix": "Check transmission fluid level and condition first; test TCC solenoid resistance.",
    },
    "P0325": {
        "definition": "Knock Sensor 1 Circuit Malfunction (Bank 1) — knock sensor signal is absent or out of range",
        "causes": ["faulty knock sensor", "damaged knock sensor wiring or connector", "loose knock sensor mounting bolt", "engine internal knocking (detonation)", "PCM fault"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe short distances, but the ECM will retard ignition timing causing performance and fuel economy loss.",
        "parts": ["knock sensor", "knock sensor wiring harness"],
        "fix": "Check sensor mounting torque and connector; replace knock sensor if wiring checks out.",
    },
    "P0016": {
        "definition": "Crankshaft/Camshaft Position Correlation (Bank 1 Sensor A) — camshaft and crankshaft positions are out of sync",
        "causes": ["stretched or jumped timing chain", "worn timing chain tensioner or guides", "low oil pressure causing VVT issues", "faulty cam or crank sensor", "oil sludge blocking VVT actuator"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — a jumped timing chain can cause catastrophic engine damage (bent valves).",
        "parts": ["timing chain kit", "timing chain tensioner", "timing chain guides", "VVT actuator"],
        "fix": "Check oil level and condition immediately; do not start the engine until timing is verified.",
    },
    "P0011": {
        "definition": "'A' Camshaft Position Timing Over-Advanced or System Performance (Bank 1)",
        "causes": ["low or dirty engine oil (most common)", "stuck open VVT solenoid", "faulty camshaft position actuator", "timing chain stretch", "PCM fault"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — can progress to timing chain damage and engine failure.",
        "parts": ["engine oil + filter", "VVT solenoid (Bank 1)", "camshaft actuator"],
        "fix": "Change the engine oil and filter first — dirty oil is the #1 cause of P0011.",
    },
    "P0021": {
        "definition": "'A' Camshaft Position Timing Over-Advanced or System Performance (Bank 2)",
        "causes": ["low or dirty engine oil", "stuck open VVT solenoid on Bank 2", "faulty camshaft actuator", "timing chain stretch on Bank 2"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — timing chain damage can follow quickly.",
        "parts": ["engine oil + filter", "VVT solenoid (Bank 2)", "camshaft actuator (Bank 2)"],
        "fix": "Change engine oil and filter immediately; inspect the Bank 2 VVT solenoid.",
    },
    "P0442": {
        "definition": "Evaporative Emission Control System (EVAP) Small Leak Detected",
        "causes": ["faulty or loose gas cap", "small crack in EVAP hose", "faulty purge valve", "micro-leak in fuel tank or charcoal canister"],
        "severity": "info",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; will fail emissions test. Start with the gas cap.",
        "parts": ["gas cap", "EVAP hoses", "purge valve"],
        "fix": "Tighten or replace the gas cap, clear the code, and drive two complete drive cycles.",
    },
    "P0141": {
        "definition": "O2 Sensor Heater Circuit Malfunction (Bank 1, Sensor 2) — downstream oxygen sensor heater is not working",
        "causes": ["faulty O2 sensor", "blown heater circuit fuse", "damaged O2 sensor wiring", "PCM output driver failure"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; but fuel economy suffers and catalytic converter health cannot be monitored.",
        "parts": ["downstream O2 sensor (Bank 1, Sensor 2)", "O2 sensor heater fuse"],
        "fix": "Check the O2 sensor heater fuse first; replace the sensor if fuse and wiring are intact.",
    },
    "P0131": {
        "definition": "O2 Sensor Circuit Low Voltage (Bank 1, Sensor 1) — upstream oxygen sensor output is too low",
        "causes": ["faulty upstream O2 sensor", "exhaust leak before the sensor", "damaged sensor wiring", "lean fuel mixture causing continuous low signal"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe short distances; a faulty upstream O2 sensor causes poor fuel economy and may damage the catalytic converter.",
        "parts": ["upstream O2 sensor (Bank 1, Sensor 1)"],
        "fix": "Check for exhaust leaks near the sensor first; replace the upstream O2 sensor if no leaks found.",
    },
    "P0507": {
        "definition": "Idle Control System RPM High — engine idle speed is higher than expected",
        "causes": ["faulty idle air control (IAC) valve", "vacuum leak at throttle body or intake manifold", "dirty throttle body", "faulty throttle position sensor"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; but high idle wastes fuel and puts stress on the drivetrain.",
        "parts": ["IAC valve", "throttle body", "throttle position sensor"],
        "fix": "Clean the throttle body with throttle body cleaner; check for vacuum leaks around the intake.",
    },
    "P0505": {
        "definition": "Idle Air Control System Malfunction — IAC system cannot maintain the target idle speed",
        "causes": ["faulty idle air control valve", "dirty or clogged IAC passages", "vacuum leaks", "faulty throttle position sensor", "carbon buildup on throttle plate"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; engine may stall at stops in severe cases.",
        "parts": ["IAC valve", "throttle body"],
        "fix": "Clean the throttle body and IAC valve passages; replace IAC valve if cleaning fails.",
    },
    "P0401": {
        "definition": "Exhaust Gas Recirculation (EGR) Flow Insufficient Detected",
        "causes": ["clogged or stuck EGR valve", "blocked EGR passages with carbon deposits", "faulty EGR solenoid", "vacuum line to EGR valve cracked or disconnected"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive short distances; NOx emissions increase and engine may knock under load.",
        "parts": ["EGR valve", "EGR solenoid", "EGR vacuum line"],
        "fix": "Remove and clean the EGR valve with carbon cleaner; replace if passages are permanently blocked.",
    },
    "P0404": {
        "definition": "Exhaust Gas Recirculation Circuit Range/Performance",
        "causes": ["sticking EGR valve (carbon buildup)", "faulty EGR position sensor", "damaged EGR wiring", "vacuum leak at EGR vacuum line"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; increased NOx emissions and possible rough idle.",
        "parts": ["EGR valve", "EGR position sensor"],
        "fix": "Inspect and clean the EGR valve; test the EGR position sensor voltage.",
    },
}

# ─── Templates de preguntas ────────────────────────────────────────────────────

SINGLE_CODE_TEMPLATES = [
    "I'm getting code {code} on my {make} {model}. What does it mean?",
    "My check engine light is on with code {code}. Is it serious?",
    "What is {code} and how serious is it?",
    "Code {code} appeared on my scanner. What should I do?",
    "Can you explain fault code {code} for a {make} {model}?",
    "Is code {code} critical? I drive a {make} {model}.",
    "What does DTC {code} mean in plain English?",
    "My {make} shows {code}. Do I need to go to a mechanic right away?",
    "Code {code} on my {make} {model} {year}. What are the common causes?",
    "Reading code {code} on my vehicle. What repairs might be needed?",
    "Check engine showing {code}. How urgent is this?",
    "Just scanned my car and got {code}. Your diagnosis?",
    "Got {code} after engine light came on. What does it mean?",
    "{make} {model} showing {code}. Is it safe to drive?",
    "Mechanic said I have {code}. What should I know before I go in?",
]

PARTS_TEMPLATES = [
    "I have {code} on my {make} {model}. What parts will my mechanic need?",
    "Code {code} came up. What components are likely to be replaced?",
    "Got {code} on my car. What should the mechanic inspect and quote?",
    "My {make} has {code}. What parts should I ask about when getting a quote?",
]

DRIVE_TEMPLATES = [
    "My car shows {code}. Is it safe to drive to work tomorrow?",
    "Got {code} on the highway. Can I keep driving?",
    "{code} just appeared. Should I pull over or can I drive home?",
    "Code {code} on my {make}. Is it okay to drive short distances?",
]

CAUSE_TEMPLATES = [
    "What causes {code} on a {make} {model}?",
    "Why would my {make} trigger {code}?",
    "What are the most common causes of {code}?",
    "My mechanic mentioned {code}. What usually causes this?",
]

VEHICLES = [
    ("Toyota", "Camry", "2019"),
    ("Ford", "F-150", "2018"),
    ("Honda", "Civic", "2020"),
    ("Chevrolet", "Silverado", "2017"),
    ("Toyota", "RAV4", "2021"),
    ("Nissan", "Altima", "2016"),
    ("Ford", "Escape", "2019"),
    ("Honda", "CR-V", "2020"),
    ("Hyundai", "Elantra", "2018"),
    ("Jeep", "Grand Cherokee", "2015"),
    ("Dodge", "Ram 1500", "2019"),
    ("BMW", "320i", "2018"),
    ("Volkswagen", "Jetta", "2017"),
    ("Subaru", "Outback", "2020"),
    ("Kia", "Sportage", "2019"),
]


# ─── Generadores de respuesta ──────────────────────────────────────────────────

def build_diagnostic_response(code: str, info: dict) -> str:
    """Construye una respuesta diagnóstica completa y verificada."""
    severity   = info["severity"]
    safe       = info["safe_to_drive"]
    definition = info["definition"]
    causes     = info["causes"]
    safe_note  = info["safe_note"]

    top_causes = random.sample(causes, min(2, len(causes)))
    causes_str = " or ".join(top_causes)

    if severity == "critical":
        prefix = f"[WARN] **CRITICAL** {code}: {definition}."
        action = f"Do not drive — {causes_str} are common causes. {safe_note}"
    elif severity == "warning":
        prefix = f"{code}: {definition}."
        drive_str = "Safe to drive short distances" if safe else "Avoid driving"
        action = f"Common causes: {causes_str}. {drive_str} — schedule an inspection soon."
    else:
        prefix = f"{code}: {definition}."
        action = f"Monitor for symptoms; common causes include {causes_str}. Safe to drive, but address at next service."

    return f"{prefix} {action}"


def build_parts_response(code: str, info: dict) -> str:
    """Respuesta orientada al mecánico: qué piezas inspeccionar y cotizar."""
    definition = info["definition"]
    parts = info["parts"]
    fix = info["fix"]
    parts_str = ", ".join(parts[:3])
    return (
        f"{code}: {definition}. "
        f"Parts to inspect and quote: {parts_str}. "
        f"Recommended diagnostic first step: {fix}"
    )


def build_drive_response(code: str, info: dict) -> str:
    safe = info["safe_to_drive"]
    definition = info["definition"]
    safe_note = info["safe_note"]
    severity = info["severity"]

    if severity == "critical":
        verdict = "[WARN] **CRITICAL** — do not drive."
    elif safe:
        verdict = "Safe to drive short distances to a repair shop."
    else:
        verdict = "Not recommended to drive."

    return f"{code}: {definition}. {verdict} {safe_note}"


def build_cause_response(code: str, info: dict) -> str:
    definition = info["definition"]
    causes = info["causes"]
    top = causes[:3]
    return (
        f"{code}: {definition}. "
        f"Most common causes: {top[0]}, {top[1] if len(top) > 1 else 'wiring issues'}, "
        f"and {top[2] if len(top) > 2 else 'PCM fault'}. "
        f"Start diagnosis with the simplest and cheapest option first."
    )


# ─── Generador de ejemplos ────────────────────────────────────────────────────

def generate_examples_for_code(code: str, info: dict, n_variants: int = 8) -> list[dict]:
    examples = []
    vehicles = random.sample(VEHICLES, min(n_variants, len(VEHICLES)))

    for i, (make, model, year) in enumerate(vehicles[:n_variants]):
        # Rotar entre tipos de pregunta
        bucket = i % 4

        if bucket == 0:
            # Diagnóstico general
            template = random.choice(SINGLE_CODE_TEMPLATES)
            user_msg = template.format(code=code, make=make, model=model, year=year)
            assistant_msg = build_diagnostic_response(code, info)

        elif bucket == 1:
            # Piezas para presupuesto del mecánico
            template = random.choice(PARTS_TEMPLATES)
            user_msg = template.format(code=code, make=make, model=model, year=year)
            assistant_msg = build_parts_response(code, info)

        elif bucket == 2:
            # Seguridad de conducir
            template = random.choice(DRIVE_TEMPLATES)
            user_msg = template.format(code=code, make=make, model=model, year=year)
            assistant_msg = build_drive_response(code, info)

        else:
            # Causas
            template = random.choice(CAUSE_TEMPLATES)
            user_msg = template.format(code=code, make=make, model=model, year=year)
            assistant_msg = build_cause_response(code, info)

        examples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        })

    return examples


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genera golden examples verificados para DTC comunes")
    parser.add_argument("--merge", action="store_true", help="Fusionar con el dataset existente y re-splitear")
    parser.add_argument("--variants", type=int, default=10, help="Variantes por código (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print("CARpsy  Step 8: Build Golden Dataset")
    print("=" * 60)
    print(f"  Códigos en la base de conocimiento: {len(DTC_KNOWLEDGE)}")
    print(f"  Variantes por código:               {args.variants}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    all_examples = []
    for code, info in DTC_KNOWLEDGE.items():
        examples = generate_examples_for_code(code, info, n_variants=args.variants)
        all_examples.extend(examples)
        print(f"  {code}: {len(examples)} ejemplos generados")

    print(f"\n  Total golden examples: {len(all_examples)}")

    # Guardar
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Guardado en: {OUTPUT_PATH}")

    if args.merge:
        print("\n[merge] Fusionando con dataset existente...")

        base_path = PROCESSED_DIR / "obdient_chat_dataset.jsonl"
        if not base_path.exists():
            print(f"  [!] No se encontró {base_path}")
            print("  [!] Ejecuta primero: python scripts/02_prepare_dataset.py")
            return

        with open(base_path, encoding="utf-8") as f:
            base_examples = [json.loads(l) for l in f if l.strip()]

        # Los golden examples van primero para asegurar que se incluyen en train
        merged = all_examples + base_examples

        # Deduplicar por respuesta del asistente
        seen: set[str] = set()
        deduped = []
        for ex in merged:
            key = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            if key not in seen:
                seen.add(key)
                deduped.append(ex)

        removed = len(merged) - len(deduped)
        print(f"  Base examples:   {len(base_examples)}")
        print(f"  Golden examples: {len(all_examples)}")
        print(f"  Duplicados:      {removed}")
        print(f"  Total merged:    {len(deduped)}")

        random.shuffle(deduped)

        merged_path = PROCESSED_DIR / "obdient_chat_dataset.jsonl"
        with open(merged_path, "w", encoding="utf-8") as f:
            for ex in deduped:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Dataset fusionado guardado en: {merged_path}")

        # Re-splitear
        print("\n[split] Re-spliteando dataset...")
        import subprocess
        result = subprocess.run(
            ["python", "scripts/03_split_dataset.py"],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Verificar que los códigos problemáticos están en train
        print("\n[verify] Verificando códigos críticos en train.jsonl...")
        train_path = SPLITS_DIR / "train.jsonl"
        with open(train_path, encoding="utf-8") as f:
            train_text = f.read()

        for code in ["P0420", "P0300", "P0171", "P0128", "P0562", "P0430", "P0455"]:
            count = train_text.count(code)
            status = "OK" if count >= 3 else "WARN — pocas ocurrencias"
            print(f"  {code}: {count:3d} ocurrencias en train — {status}")

    print("\n[OK] Step 8 complete.")
    if not args.merge:
        print("  Ejecuta con --merge para fusionar con el dataset principal:")
        print("  python scripts/08_build_golden_dataset.py --merge")


if __name__ == "__main__":
    main()
