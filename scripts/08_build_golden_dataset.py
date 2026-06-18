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

    # ── Fuel system ───────────────────────────────────────────────────────────
    "P0087": {
        "definition": "Fuel Rail/System Pressure Too Low — PCM detects fuel pressure drops below minimum for proper combustion",
        "causes": ["clogged fuel filter", "failing fuel pump", "faulty fuel pressure regulator", "kinked or restricted fuel supply line", "faulty fuel rail pressure sensor"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — engine will stall without warning when fuel pressure drops too low.",
        "parts": ["fuel filter", "fuel pump", "fuel pressure regulator", "fuel rail pressure sensor"],
        "fix": "Test fuel pressure with a gauge at the rail; replace the fuel filter first as it is the cheapest and most common cause.",
    },
    "P0088": {
        "definition": "Fuel Rail/System Pressure Too High — fuel pressure exceeds the upper threshold",
        "causes": ["faulty fuel pressure regulator stuck closed", "clogged fuel return line", "faulty fuel rail pressure sensor"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Avoid driving — excessive fuel pressure causes rich running, engine flooding, and potential fuel system damage.",
        "parts": ["fuel pressure regulator", "fuel return line", "fuel rail pressure sensor"],
        "fix": "Check fuel return line for blockages; test and replace the fuel pressure regulator.",
    },
    "P0172": {
        "definition": "System Too Rich (Bank 1) — air-fuel mixture has too much fuel or too little air on Bank 1",
        "causes": ["faulty upstream O2 sensor", "leaking fuel injectors", "faulty MAP sensor reading low vacuum", "coolant temperature sensor failure causing over-fueling", "clogged air filter"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive short distances; running rich damages the catalytic converter over time.",
        "parts": ["upstream O2 sensor", "fuel injectors", "MAP sensor", "air filter"],
        "fix": "Check the air filter first; inspect fuel injectors for leaks; verify O2 sensor output voltage.",
    },
    "P0175": {
        "definition": "System Too Rich (Bank 2) — air-fuel mixture has too much fuel or too little air on Bank 2",
        "causes": ["faulty O2 sensor on Bank 2", "leaking fuel injectors on Bank 2", "faulty MAP sensor", "failing coolant temperature sensor"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive short distances; prolonged rich condition degrades the catalytic converter.",
        "parts": ["O2 sensor (Bank 2)", "fuel injectors (Bank 2)", "MAP sensor"],
        "fix": "Inspect Bank 2 O2 sensor output; check for leaking fuel injectors on Bank 2 cylinders.",
    },

    # ── MAF / MAP sensors ─────────────────────────────────────────────────────
    "P0102": {
        "definition": "Mass Air Flow (MAF) Sensor Circuit Low Input — MAF signal is below the expected range",
        "causes": ["dirty or contaminated MAF sensor", "air intake leak after the MAF sensor", "damaged MAF sensor wiring", "clogged air filter", "faulty PCM (rare)"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; rich running can damage the catalytic converter over time — schedule repair soon.",
        "parts": ["MAF sensor", "air filter", "MAF sensor wiring harness"],
        "fix": "Inspect the air filter for contamination; clean the MAF sensor with MAF spray cleaner.",
    },
    "P0103": {
        "definition": "Mass Air Flow (MAF) Sensor Circuit High Input — MAF signal is above the expected range",
        "causes": ["faulty MAF sensor", "short circuit in MAF sensor wiring", "PCM fault"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive short distances; lean or rich running affects performance and may damage catalyst.",
        "parts": ["MAF sensor", "MAF sensor wiring harness"],
        "fix": "Inspect MAF sensor wiring for shorts; replace the MAF sensor if wiring checks out.",
    },
    "P0107": {
        "definition": "Manifold Absolute Pressure (MAP) Sensor Circuit Low Input — MAP signal voltage is too low",
        "causes": ["faulty MAP sensor", "damaged MAP sensor wiring or connector", "vacuum hose to MAP sensor disconnected or cracked", "PCM fault"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — incorrect MAP readings cause poor fuel control, rough running, and possible stalling.",
        "parts": ["MAP sensor", "vacuum hose to MAP sensor", "MAP sensor wiring harness"],
        "fix": "Inspect the vacuum hose connected to the MAP sensor for cracks; replace the MAP sensor if hose is intact.",
    },
    "P0108": {
        "definition": "Manifold Absolute Pressure (MAP) Sensor Circuit High Input — MAP signal voltage is too high",
        "causes": ["faulty MAP sensor", "short to voltage in MAP sensor wiring", "vacuum leak causing incorrect pressure reading"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — incorrect MAP signal causes incorrect fueling and may cause stalling.",
        "parts": ["MAP sensor", "MAP sensor wiring harness"],
        "fix": "Check for vacuum leaks around the intake; replace the MAP sensor if wiring is undamaged.",
    },

    # ── Throttle position ─────────────────────────────────────────────────────
    "P0121": {
        "definition": "Throttle/Pedal Position Sensor A Circuit Range/Performance — TPS signal is out of expected range",
        "causes": ["faulty throttle position sensor", "carbon buildup on throttle body", "TPS wiring connector corrosion", "faulty throttle body"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — erratic throttle response is a safety hazard.",
        "parts": ["throttle position sensor", "throttle body"],
        "fix": "Clean the throttle body; if the code persists, replace the throttle position sensor.",
    },
    "P0122": {
        "definition": "Throttle/Pedal Position Sensor A Circuit Low Input — TPS signal voltage is too low",
        "causes": ["faulty TPS sensor", "open circuit or short to ground in TPS wiring", "corroded TPS connector"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Do not drive — vehicle may not accelerate correctly or may go into limp mode.",
        "parts": ["throttle position sensor", "TPS wiring harness"],
        "fix": "Inspect TPS connector for corrosion; replace the TPS if voltage at idle is below 0.5V.",
    },
    "P0123": {
        "definition": "Throttle/Pedal Position Sensor A Circuit High Input — TPS signal voltage is too high",
        "causes": ["faulty TPS sensor", "short to voltage in TPS wiring", "faulty PCM"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Do not drive — the engine may not respond correctly to throttle input.",
        "parts": ["throttle position sensor", "TPS wiring harness"],
        "fix": "Inspect TPS wiring for shorts to voltage; replace the TPS if voltage at idle exceeds 4.9V.",
    },

    # ── O2 sensors ────────────────────────────────────────────────────────────
    "P0130": {
        "definition": "O2 Sensor Circuit Malfunction (Bank 1, Sensor 1) — upstream oxygen sensor signal is out of range or absent",
        "causes": ["faulty upstream O2 sensor", "exhaust leak before the sensor", "damaged sensor wiring", "sensor contaminated by coolant or oil"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive short distances; poor O2 readings cause incorrect fueling and catalytic converter degradation.",
        "parts": ["upstream O2 sensor (Bank 1, Sensor 1)"],
        "fix": "Check for exhaust leaks near the sensor; replace the upstream O2 sensor if wiring is intact.",
    },
    "P0133": {
        "definition": "O2 Sensor Circuit Slow Response (Bank 1, Sensor 1) — upstream O2 sensor is sluggish, not switching fast enough",
        "causes": ["aged or contaminated upstream O2 sensor", "exhaust leak before the sensor", "O2 sensor poisoned by silicone or coolant"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; sluggish O2 response causes reduced fuel economy and possible catalytic converter damage.",
        "parts": ["upstream O2 sensor (Bank 1, Sensor 1)"],
        "fix": "Replace the upstream O2 sensor — slow response is a sign of aging and cleaning will not help.",
    },
    "P0135": {
        "definition": "O2 Sensor Heater Circuit Malfunction (Bank 1, Sensor 1) — upstream O2 sensor heater is not working",
        "causes": ["faulty O2 sensor heater element", "blown heater circuit fuse", "damaged sensor wiring", "PCM output driver failure"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; cold O2 sensor takes longer to enter closed-loop, reducing fuel efficiency.",
        "parts": ["upstream O2 sensor (Bank 1, Sensor 1)", "O2 sensor heater fuse"],
        "fix": "Check the O2 sensor heater fuse first; replace the sensor if fuse and wiring are intact.",
    },
    "P0136": {
        "definition": "O2 Sensor Circuit Malfunction (Bank 1, Sensor 2) — downstream O2 sensor signal is out of range",
        "causes": ["faulty downstream O2 sensor", "catalytic converter damage causing abnormal exhaust", "damaged sensor wiring"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; catalytic converter efficiency cannot be monitored without a working downstream sensor.",
        "parts": ["downstream O2 sensor (Bank 1, Sensor 2)"],
        "fix": "Inspect sensor wiring; replace the downstream O2 sensor if wiring is undamaged.",
    },
    "P0155": {
        "definition": "O2 Sensor Heater Circuit Malfunction (Bank 2, Sensor 1) — Bank 2 upstream O2 sensor heater is not working",
        "causes": ["faulty Bank 2 upstream O2 sensor heater", "blown fuse", "damaged wiring", "PCM driver fault"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; fuel economy and cold-start emissions are affected.",
        "parts": ["upstream O2 sensor (Bank 2, Sensor 1)", "O2 heater fuse"],
        "fix": "Check the heater circuit fuse first; replace the Bank 2 upstream O2 sensor if needed.",
    },

    # ── Misfire per-cylinder (4–8) ────────────────────────────────────────────
    "P0304": {
        "definition": "Cylinder 4 Misfire Detected",
        "causes": ["worn spark plug on cylinder 4", "faulty ignition coil on cylinder 4", "clogged fuel injector", "low compression on cylinder 4"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — misfires overheat and damage the catalytic converter rapidly.",
        "parts": ["spark plug (cylinder 4)", "ignition coil (cylinder 4)", "fuel injector (cylinder 4)"],
        "fix": "Swap the cylinder 4 spark plug and ignition coil with a known-good cylinder to isolate the fault.",
    },
    "P0305": {
        "definition": "Cylinder 5 Misfire Detected",
        "causes": ["worn spark plug on cylinder 5", "faulty ignition coil on cylinder 5", "clogged fuel injector", "low compression on cylinder 5"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — catalytic converter damage can occur within minutes of sustained misfiring.",
        "parts": ["spark plug (cylinder 5)", "ignition coil (cylinder 5)", "fuel injector (cylinder 5)"],
        "fix": "Swap cylinder 5 spark plug and coil with a known-good cylinder to isolate the fault.",
    },
    "P0306": {
        "definition": "Cylinder 6 Misfire Detected",
        "causes": ["worn spark plug on cylinder 6", "faulty ignition coil on cylinder 6", "clogged fuel injector", "low compression on cylinder 6"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — stop immediately if the check engine light is flashing.",
        "parts": ["spark plug (cylinder 6)", "ignition coil (cylinder 6)", "fuel injector (cylinder 6)"],
        "fix": "Replace the cylinder 6 spark plug as the first diagnostic step.",
    },
    "P0307": {
        "definition": "Cylinder 7 Misfire Detected",
        "causes": ["worn spark plug on cylinder 7", "faulty ignition coil on cylinder 7", "clogged fuel injector", "low compression on cylinder 7"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — misfires cause rapid catalytic converter damage and potential engine fire.",
        "parts": ["spark plug (cylinder 7)", "ignition coil (cylinder 7)", "fuel injector (cylinder 7)"],
        "fix": "Swap cylinder 7 ignition coil with a known-good one to confirm the fault.",
    },
    "P0308": {
        "definition": "Cylinder 8 Misfire Detected",
        "causes": ["worn spark plug on cylinder 8", "faulty ignition coil on cylinder 8", "clogged fuel injector", "low compression on cylinder 8"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — catalytic converter and engine damage can result from continued operation.",
        "parts": ["spark plug (cylinder 8)", "ignition coil (cylinder 8)", "fuel injector (cylinder 8)"],
        "fix": "Replace all spark plugs if vehicle has high mileage; isolate by swapping coils between cylinders.",
    },

    # ── Fuel injectors ────────────────────────────────────────────────────────
    "P0201": {
        "definition": "Injector Circuit Malfunction — Cylinder 1 fuel injector circuit open or shorted",
        "causes": ["faulty cylinder 1 fuel injector", "open or shorted injector wiring", "damaged injector connector", "PCM driver failure"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — a non-firing injector causes a cylinder misfire and rapid catalytic converter damage.",
        "parts": ["fuel injector (cylinder 1)", "injector wiring harness"],
        "fix": "Measure injector resistance (should be 12–17 Ohms); replace if open or shorted.",
    },
    "P0202": {
        "definition": "Injector Circuit Malfunction — Cylinder 2 fuel injector circuit open or shorted",
        "causes": ["faulty cylinder 2 fuel injector", "open or shorted injector wiring", "damaged connector", "PCM fault"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — cylinder 2 misfire damages the catalytic converter.",
        "parts": ["fuel injector (cylinder 2)", "injector wiring harness"],
        "fix": "Measure cylinder 2 injector resistance; swap with a known-good injector to confirm.",
    },
    "P0204": {
        "definition": "Injector Circuit Malfunction — Cylinder 4 fuel injector circuit open or shorted",
        "causes": ["faulty cylinder 4 injector", "damaged wiring to cylinder 4 injector", "PCM output failure"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — cylinder 4 is disabled, causing misfire and catalyst damage.",
        "parts": ["fuel injector (cylinder 4)", "injector wiring harness"],
        "fix": "Measure injector resistance on cylinder 4; check wiring continuity to PCM pin.",
    },

    # ── EVAP additional ───────────────────────────────────────────────────────
    "P0440": {
        "definition": "Evaporative Emission Control System Malfunction — general EVAP system fault detected",
        "causes": ["loose or faulty gas cap", "faulty EVAP purge control valve", "cracked EVAP hoses", "damaged charcoal canister", "fuel tank leak"],
        "severity": "info",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; fuel vapors escape into atmosphere and vehicle may fail emissions test.",
        "parts": ["gas cap", "EVAP purge valve", "charcoal canister", "EVAP hoses"],
        "fix": "Tighten or replace the gas cap first and clear the code; if it returns, smoke-test the EVAP system.",
    },
    "P0441": {
        "definition": "Evaporative Emission Control System Incorrect Purge Flow — EVAP purge flow is out of specification",
        "causes": ["faulty EVAP purge valve stuck open or closed", "disconnected or cracked purge hose", "faulty EVAP vent valve", "PCM fault"],
        "severity": "info",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; may cause rough idle if purge valve is stuck open.",
        "parts": ["EVAP purge valve", "purge hose", "EVAP vent valve"],
        "fix": "Test the EVAP purge valve by applying vacuum; replace if it does not hold.",
    },
    "P0446": {
        "definition": "Evaporative Emission Control System Vent Control Circuit Malfunction",
        "causes": ["faulty EVAP vent solenoid", "blocked or damaged vent tube", "faulty gas cap", "PCM fault"],
        "severity": "info",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; no performance impact but will fail emissions testing.",
        "parts": ["EVAP vent solenoid", "vent tube", "gas cap"],
        "fix": "Test the vent solenoid with a multimeter; replace if it does not open when commanded.",
    },

    # ── Vehicle speed / idle ──────────────────────────────────────────────────
    "P0500": {
        "definition": "Vehicle Speed Sensor (VSS) Malfunction — no or erratic signal from the vehicle speed sensor",
        "causes": ["faulty vehicle speed sensor", "damaged VSS wiring or connector", "broken tone ring on output shaft", "ABS wheel speed sensor wiring (on some vehicles)", "PCM fault"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — speedometer and cruise control may not work; transmission may shift incorrectly.",
        "parts": ["vehicle speed sensor", "VSS wiring harness", "tone ring"],
        "fix": "Locate the VSS on the transmission; inspect wiring and connector before replacing the sensor.",
    },
    "P0520": {
        "definition": "Engine Oil Pressure Sensor/Switch Circuit Malfunction",
        "causes": ["faulty oil pressure sensor/switch", "damaged sensor wiring", "low actual engine oil pressure (serious)", "PCM fault"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — verify actual oil pressure immediately; low oil pressure causes catastrophic engine damage.",
        "parts": ["oil pressure sensor", "oil pressure sensor wiring"],
        "fix": "Check oil level first. If oil level is normal, test actual oil pressure with a mechanical gauge before replacing the sensor.",
    },
    "P0521": {
        "definition": "Engine Oil Pressure Sensor/Switch Range/Performance — oil pressure reading is out of expected range",
        "causes": ["faulty oil pressure sensor giving erratic readings", "oil viscosity incorrect for conditions", "engine oil pump wear", "damaged wiring"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive until actual oil pressure is verified — do not assume it is just a bad sensor.",
        "parts": ["oil pressure sensor", "engine oil + filter"],
        "fix": "Verify actual oil pressure with a mechanical gauge; if pressure is normal, replace the sensor.",
    },

    # ── PCM / ECM ─────────────────────────────────────────────────────────────
    "P0600": {
        "definition": "Serial Communication Link Malfunction — PCM cannot communicate on the CAN bus",
        "causes": ["damaged CAN bus wiring", "faulty module on the network pulling the bus down", "PCM internal fault", "corroded connectors on the data link"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — multiple vehicle systems may be unresponsive or behaving incorrectly.",
        "parts": ["CAN bus wiring harness", "PCM/ECM"],
        "fix": "Check CAN bus wiring for shorts or opens; use a scan tool to identify which module is disrupting the network.",
    },
    "P0606": {
        "definition": "ECM/PCM Processor Fault — internal PCM processor has failed self-test",
        "causes": ["internal PCM hardware failure", "corrupted PCM software", "power or ground supply issue to PCM"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — the engine control module is not functioning correctly.",
        "parts": ["PCM/ECM", "PCM power relay", "PCM ground wiring"],
        "fix": "Verify PCM power and ground connections first; if intact, the PCM likely requires replacement or reprogramming.",
    },

    # ── Alternator / charging ─────────────────────────────────────────────────
    "P0622": {
        "definition": "Alternator Field Terminal Circuit Malfunction — PCM cannot control the alternator field",
        "causes": ["faulty alternator", "damaged alternator field wiring", "faulty voltage regulator (internal to alternator)", "PCM output fault"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — battery will drain quickly without alternator charging; vehicle will stall.",
        "parts": ["alternator", "alternator field wiring"],
        "fix": "Test alternator output voltage (should be 13.5–14.5V at idle); replace alternator if output is absent.",
    },

    # ── Transmission additional ───────────────────────────────────────────────
    "P0715": {
        "definition": "Transmission Input/Turbine Speed Sensor Circuit Malfunction — no signal from the turbine speed sensor",
        "causes": ["faulty turbine speed sensor", "damaged sensor wiring or connector", "debris on sensor tip", "low transmission fluid"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — transmission may shift erratically or not at all.",
        "parts": ["transmission input speed sensor", "sensor wiring harness"],
        "fix": "Inspect sensor and connector for debris or damage; replace the sensor if wiring is undamaged.",
    },
    "P0720": {
        "definition": "Output Shaft Speed Sensor Circuit Malfunction — no or erratic signal from the output speed sensor",
        "causes": ["faulty output shaft speed sensor", "damaged sensor wiring", "debris on tone ring", "low transmission fluid"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — incorrect speed data causes erratic shifting and potential drivetrain damage.",
        "parts": ["output shaft speed sensor", "sensor wiring harness"],
        "fix": "Inspect the sensor tip for metal debris; clean and reinstall before replacing the sensor.",
    },
    "P0730": {
        "definition": "Incorrect Gear Ratio — transmission is not achieving the expected gear ratio",
        "causes": ["worn clutch packs or bands", "faulty shift solenoid", "low or degraded transmission fluid", "internal valve body issue"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Avoid driving — slipping gears increase transmission heat and can cause complete failure.",
        "parts": ["transmission fluid", "shift solenoids", "valve body"],
        "fix": "Check and replace transmission fluid first; scan for additional P07xx codes to identify the specific gear ratio fault.",
    },
    "P0750": {
        "definition": "Shift Solenoid A Malfunction — Shift Solenoid A is not operating correctly",
        "causes": ["faulty shift solenoid A", "low or dirty transmission fluid", "blocked solenoid body passages", "wiring fault to solenoid A"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — transmission may not shift properly, causing overheating.",
        "parts": ["shift solenoid A", "transmission fluid + filter", "valve body"],
        "fix": "Change transmission fluid and filter first; test solenoid A resistance (typically 11–26 Ohms).",
    },
    "P0755": {
        "definition": "Shift Solenoid B Malfunction — Shift Solenoid B is not operating correctly",
        "causes": ["faulty shift solenoid B", "dirty transmission fluid", "wiring fault to solenoid B", "internal valve body issue"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — incorrect shifting increases transmission wear.",
        "parts": ["shift solenoid B", "transmission fluid + filter"],
        "fix": "Change transmission fluid; test solenoid B resistance; replace solenoid if resistance is out of spec.",
    },
    "P0760": {
        "definition": "Shift Solenoid C Malfunction — Shift Solenoid C is not operating correctly",
        "causes": ["faulty shift solenoid C", "dirty transmission fluid blocking solenoid", "wiring fault"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — transmission shifting issues can escalate to complete failure.",
        "parts": ["shift solenoid C", "transmission fluid + filter"],
        "fix": "Change transmission fluid and filter; test solenoid C resistance.",
    },

    # ── Cooling system additional ─────────────────────────────────────────────
    "P0116": {
        "definition": "Engine Coolant Temperature Sensor Circuit Range/Performance — ECT sensor reading is implausible",
        "causes": ["faulty coolant temperature sensor", "air pockets in the cooling system", "low coolant level", "stuck-open thermostat causing slow warm-up"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — an inaccurate coolant reading can mask real overheating.",
        "parts": ["coolant temperature sensor", "thermostat"],
        "fix": "Check coolant level and bleed air pockets; replace the ECT sensor if level is correct.",
    },
    "P0117": {
        "definition": "Engine Coolant Temperature Sensor Circuit Low Input — ECT sensor voltage is too low (reads very hot)",
        "causes": ["faulty ECT sensor", "short to ground in ECT sensor wiring", "corroded ECT connector"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Do not drive — the sensor may indicate overheating when the engine is not, or miss actual overheating.",
        "parts": ["coolant temperature sensor", "ECT wiring harness"],
        "fix": "Inspect the ECT connector for corrosion; replace the ECT sensor.",
    },

    # ── Secondary air / turbo ─────────────────────────────────────────────────
    "P0410": {
        "definition": "Secondary Air Injection System Malfunction — the AIR pump system is not functioning correctly",
        "causes": ["faulty secondary air pump", "clogged or stuck AIR pump check valve", "faulty AIR pump relay", "vacuum hose to AIR system cracked"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive; increased cold-start emissions and will fail emissions test.",
        "parts": ["secondary air pump", "AIR pump check valve", "AIR pump relay"],
        "fix": "Listen for the secondary air pump operating at cold start; replace the pump if it does not run.",
    },
    "P0299": {
        "definition": "Turbocharger/Supercharger A Underboost Condition — boost pressure is below target",
        "causes": ["boost leak in intercooler pipe or hose", "faulty wastegate stuck open", "worn turbocharger", "faulty boost pressure sensor"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — significant power loss; avoid highway driving until diagnosed.",
        "parts": ["intercooler hoses", "wastegate actuator", "turbocharger", "boost pressure sensor"],
        "fix": "Pressurize the intake system with a boost leak tester; most underboost faults are caused by boost leaks.",
    },
    "P0234": {
        "definition": "Turbocharger/Supercharger A Overboost Condition — boost pressure exceeds the maximum limit",
        "causes": ["stuck closed wastegate", "faulty boost control solenoid", "faulty boost pressure sensor reading low"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — excessive boost pressure can damage the engine in seconds.",
        "parts": ["wastegate actuator", "boost control solenoid", "boost pressure sensor"],
        "fix": "Inspect the wastegate actuator for freedom of movement; test the boost control solenoid.",
    },

    # ── Knock / timing ────────────────────────────────────────────────────────
    "P0326": {
        "definition": "Knock Sensor 1 Circuit Range/Performance (Bank 1) — knock sensor signal is erratic or implausible",
        "causes": ["faulty knock sensor", "engine knocking due to incorrect fuel or carbon buildup", "loose knock sensor mounting", "damaged wiring"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe short distances but ECM retards timing causing power loss and potential engine knock damage.",
        "parts": ["knock sensor", "knock sensor wiring"],
        "fix": "Check for actual engine knock; tighten the knock sensor to spec torque; replace if signal remains erratic.",
    },
    "P0012": {
        "definition": "Intake Camshaft Position Timing Over-Retarded (Bank 1) — camshaft timing is too far retarded",
        "causes": ["low or dirty engine oil", "stuck VVT solenoid in retard position", "worn timing chain", "faulty camshaft actuator"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — retarded timing causes power loss and can lead to timing chain damage.",
        "parts": ["engine oil + filter", "VVT solenoid (Bank 1)", "timing chain kit"],
        "fix": "Change the engine oil and filter immediately — dirty oil is the most common cause.",
    },
    "P0014": {
        "definition": "Exhaust Camshaft Position Timing Over-Advanced (Bank 1) — exhaust cam is too far advanced",
        "causes": ["low or dirty engine oil blocking VVT actuator", "stuck VVT solenoid", "worn timing chain"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — can escalate to timing chain damage.",
        "parts": ["engine oil + filter", "VVT solenoid (exhaust, Bank 1)"],
        "fix": "Change engine oil and filter first; inspect the exhaust VVT solenoid.",
    },
    "P0017": {
        "definition": "Crankshaft/Camshaft Position Correlation (Bank 1 Sensor B) — camshaft B and crankshaft are out of sync",
        "causes": ["stretched timing chain", "worn tensioner or guides", "oil sludge in VVT actuator", "faulty cam or crank sensor"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — timing chain failure causes catastrophic valve train damage.",
        "parts": ["timing chain kit", "timing chain tensioner", "VVT actuator"],
        "fix": "Check oil level and condition immediately; do not run the engine until timing is verified.",
    },

    # ── Cooling fan ───────────────────────────────────────────────────────────
    "P0480": {
        "definition": "Cooling Fan 1 Control Circuit Malfunction — PCM cannot control the primary cooling fan",
        "causes": ["faulty cooling fan relay", "failed cooling fan motor", "damaged fan control wiring", "PCM output fault"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Do not drive in traffic or at low speeds — without a working fan, engine can overheat quickly.",
        "parts": ["cooling fan relay", "cooling fan motor", "fan control wiring"],
        "fix": "Test the cooling fan relay by swapping with a known-good relay; test fan motor ground and power supply.",
    },

    # ── Transmission range / 4WD ──────────────────────────────────────────────
    "P0705": {
        "definition": "Transmission Range Sensor Circuit Malfunction (PRNDL Input) — TCM cannot determine which gear range is selected",
        "causes": ["faulty transmission range sensor (TR sensor)", "misadjusted gear shifter cable", "damaged TR sensor wiring", "dirty or corroded connector"],
        "severity": "warning",
        "safe_to_drive": False,
        "safe_note": "Not recommended — transmission may not shift correctly or engage the wrong gear.",
        "parts": ["transmission range sensor", "gear selector cable"],
        "fix": "Inspect and adjust the shifter cable; replace the transmission range sensor if adjustment does not resolve.",
    },

    # ── Fuel pump circuit ─────────────────────────────────────────────────────
    "P0230": {
        "definition": "Fuel Pump Primary Circuit Malfunction — PCM detects a fault in the fuel pump control circuit",
        "causes": ["blown fuel pump fuse", "faulty fuel pump relay", "damaged fuel pump wiring", "faulty fuel pump driver module", "failed fuel pump"],
        "severity": "critical",
        "safe_to_drive": False,
        "safe_note": "Do not drive — without fuel delivery the engine will stall immediately.",
        "parts": ["fuel pump fuse", "fuel pump relay", "fuel pump", "fuel pump wiring harness"],
        "fix": "Check the fuel pump fuse and relay first — they are the cheapest and fastest components to test.",
    },

    # ── A/C ───────────────────────────────────────────────────────────────────
    "P0533": {
        "definition": "A/C Refrigerant Pressure Sensor Circuit High Input — refrigerant pressure reading is too high",
        "causes": ["faulty A/C pressure sensor", "overcharged A/C system", "condenser blockage causing high head pressure", "damaged sensor wiring"],
        "severity": "warning",
        "safe_to_drive": True,
        "safe_note": "Safe to drive with A/C off; do not use A/C until the system is inspected — overcharge can damage the compressor.",
        "parts": ["A/C pressure sensor", "A/C refrigerant (recharge if needed)"],
        "fix": "Measure actual system pressure with an A/C manifold gauge set; replace sensor if pressure is within spec.",
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
