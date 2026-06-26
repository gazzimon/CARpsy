# ADR-0004: Learning loop sobre SHIMI

- **Estado:** Propuesto
- **Fecha:** 2026-06-24
- **Deciders:** Gustavo (gazzimon)
- **Relacionados:** ADR-0001 (nodos reificados `CausalHypothesis`)
- **Repos afectados:** `gazzimon/OBDient` (esquema, router, UI), `gazzimon/CARpsy` (sin cambios de modelo)

---

## Contexto y problema

Hoy SHIMI es un grafo de conocimiento estático: se ingesta desde JSON-LD a
SQLite (Drizzle) y el router multi-agente determinístico recupera hipótesis
causales (`CausalHypothesis`, ADR-0001) vía el closure transitivo de
`subClassOf` y el `faultClass` derivado de las posiciones del DTC. El grafo no
mejora con el uso: dos talleres que ven el mismo síntoma diez veces no
acumulan ninguna ventaja sobre el primer diagnóstico.

La tesis de producto de OBDient es offline-first con inferencia local. Un lazo
de aprendizaje que dependa de la nube, de reentrenar el modelo, o que rompa el
determinismo del router contradiría esa tesis. Necesitamos un mecanismo que:

1. Aprenda de cada sesión de diagnóstico **sin reentrenar** y **sin red**.
2. **No comprometa el determinismo** ni la reproducibilidad del router.
3. **No corrompa** la ontología curada a mano.
4. Tenga **riesgo de regresión cero** en el arranque (cold-start).
5. Habilite, a futuro, agregación federada respetando privacidad.

## Drivers de decisión

- Offline-first y costo computacional nulo en el dispositivo.
- Auditabilidad: cada decisión debe poder reconstruirse bit a bit.
- Provenance estricta entre conocimiento curado y aprendido.
- Reutilización de patrones ya identificados como adoptables (AuditLogger con
  clock inyectado, función pura `decideDiagnosticAction()` testeable, receipts
  de proof-of-inference).

## Decisión

Adoptamos un **lazo de aprendizaje de dos caminos** sobre SHIMI, con
actualización estadística **Beta-Binomial conjugada** y la ontología actuando
como prior (pseudo-counts).

### Separación de caminos

- **Camino de inferencia (solo lectura, determinista):** ELM327 → firma de caso
  → retrieval SHIMI → re-ranker → CARpsy → salida rankeada. Opera sobre un
  *snapshot inmutable* de pesos. Nunca escribe.
- **Camino de aprendizaje (post-hoc, escritura):** captura de resultado →
  evidencia append-only → actualización de stats → (causa novedosa → arista
  aprendida) → snapshot. Corre **después** de la respuesta, en otra transacción.

Esta separación es lo que preserva el determinismo: la mutación nunca ocurre
dentro de la ruta de decisión.

### Modelo estadístico: Beta-Binomial conjugado

Cada par `(firma_de_caso, hipótesis)` mantiene dos contadores: `α`
(confirmaciones) y `β` (refutaciones). El score es la media posterior con
corrección conservadora (límite inferior del intervalo creíble), para que pocas
evidencias no disparen una hipótesis a la cima:

```
posterior_mean = α / (α + β)
lower_bound    = posterior_mean − k · sqrt( (α·β) / ((α+β)² · (α+β+1)) )   // k ≈ 1.0
```

La ontología entra como **pseudo-counts** (`α₀`, `β₀`) por arista. Antes de
cualquier evidencia, el ranking es idéntico al actual (puro ontología). La
evidencia solo refina. **Riesgo de regresión = cero.**

### Firma de caso

Clave determinista para agregar evidencia entre casos "parecidos". Hash de
features con los continuos bucketizados:

```typescript
function caseSignature(ctx: DiagnosticContext): string {
  const parts = [
    [...ctx.dtcSet].sort().join('|'),
    ctx.faultClass,
    `${ctx.make}:${ctx.model}:${bucket(ctx.year, 3)}:${ctx.engine}`,
    `coolant:${bucket(ctx.freezeFrame.coolantTemp, 20)}`,
    `load:${bucket(ctx.freezeFrame.engineLoad, 25)}`,
  ];
  return sha256(parts.join('§')); // determinista, sin PII ni VIN
}
```

Buckets gruesos al inicio (más generalización, aprendizaje rápido); se refinan
solo con volumen.

### Re-ranker como función pura

Patrón `decideDiagnosticAction()`: función pura, determinista, testeable con
tablas. Desempate estable por `id` para garantizar orden total reproducible.

```typescript
export function rankHypotheses(input: RankInput): RankedHypothesis[] {
  return input.candidates
    .map(h => {
      const s = input.stats.get(`${input.signature}|${h.id}`);
      const α = h.priorAlpha + (s?.alpha ?? 0);
      const β = h.priorBeta  + (s?.beta  ?? 0);
      return { ...h, score: lowerBound(α, β, 1.0), alpha: α, beta: β };
    })
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
}
```

### Esquema (Drizzle)

```typescript
export const causalHypothesis = sqliteTable('causal_hypothesis', {
  id: text('id').primaryKey(),
  faultClass: text('fault_class').notNull(),
  effectDtc: text('effect_dtc').notNull(),
  source: text('source', { enum: ['ontology', 'learned'] }).notNull().default('ontology'),
  priorAlpha: real('prior_alpha').notNull().default(1),
  priorBeta:  real('prior_beta').notNull().default(1),
});

export const outcomeEvidence = sqliteTable('outcome_evidence', {
  id: text('id').primaryKey(),
  sessionId: text('session_id').notNull(),
  hypothesisId: text('hypothesis_id').notNull(),
  caseSignature: text('case_signature').notNull(),
  outcome: text('outcome', { enum: ['confirmed', 'refuted', 'partial'] }).notNull(),
  source: text('source', { enum: ['mechanic', 'user', 'auto'] }).notNull(),
  observedAt: integer('observed_at').notNull(),   // clock inyectado, no Date.now()
  signature: text('signature'),                   // ECDSA opcional
});

export const contextStats = sqliteTable('context_stats', {
  caseSignature: text('case_signature').notNull(),
  hypothesisId: text('hypothesis_id').notNull(),
  alpha: real('alpha').notNull().default(0),
  beta:  real('beta').notNull().default(0),
  evidenceCount: integer('evidence_count').notNull().default(0),
}, (t) => ({ pk: primaryKey({ columns: [t.caseSignature, t.hypothesisId] }) }));

export const diagnosticSession = sqliteTable('diagnostic_session', {
  id: text('id').primaryKey(),
  vehicleId: text('vehicle_id').notNull(),
  dtcSet: text('dtc_set', { mode: 'json' }).notNull(),
  freezeFrame: text('freeze_frame', { mode: 'json' }),
  routerVersion: text('router_version').notNull(),
  knowledgeVersion: text('knowledge_version').notNull(),
  createdAt: integer('created_at').notNull(),
});
```

`contextStats` es un posterior **materializado**: reconstruible al 100% desde
`outcomeEvidence`, que es la única fuente de verdad append-only.

### El lazo, paso a paso

**Inferencia (read-only):**
1. Leer DTCs + freeze-frame del ELM327.
2. Calcular `caseSignature`.
3. SHIMI recupera candidatos (closure `subClassOf` → `faultClass` → `CausalHypothesis`).
4. `rankHypotheses()` re-ordena con el snapshot de `contextStats`.
5. CARpsy genera narrativa del top-k.
6. Loguear la sesión con su `knowledgeVersion` (receipt de proof-of-inference).

**Aprendizaje (post-hoc, write):**
7. El usuario/mecánico marca la causa real (confirmada) y las descartadas (refutadas).
8. Append a `outcomeEvidence` con `observedAt` del clock inyectado.
9. Update transaccional de `contextStats` (`α++`/`β++`).
10. Causa novedosa → arista `learned` provisional (prior bajo, cola de curación).
11. Periódicamente: snapshot → bump de `knowledgeVersion`.

### Garantía de determinismo

El router decide con `rankHypotheses(candidatos, snapshot, firma)` — función
pura sobre un snapshot inmutable. Toda mutación vive en el camino de
aprendizaje, fuera de la ruta de decisión. Mismo input + mismo
`knowledgeVersion` ⇒ mismo output. Como cada sesión loguea su versión, cualquier
diagnóstico histórico se reconstruye exactamente.

## Plan de implementación por fases

- **Fase 0 — solo capturar:** UI de "¿cuál fue la causa real?" + tabla
  `outcomeEvidence` append-only. Cero cambio de comportamiento. Acumula dataset
  desde el día uno y de-riskea todo lo demás.
- **Fase 1 — re-ranking conservador:** materializar `contextStats`, enchufar
  `rankHypotheses` con pseudo-counts ontológicos y scoring por límite inferior.
- **Fase 2 — aristas aprendidas:** causas novedosas como hipótesis provisionales
  con cola de curación antes de graduarlas a `source:'ontology'`.
- **Fase 3 — priors por vehículo:** `vehicleProfile` ajusta priors por VIN.
  Habilita la narrativa "el diagnóstico que crece con tu auto".
- **Fase 4 (roadmap) — agregación federada:** exportar deltas de counts (sin PII,
  sin VIN) y agregarlos sobre la flota. Candidato natural para ejecutar el conteo
  federado dentro de un enclave Acurast TEE.

## Consecuencias

### Positivas

- OBDient mejora con el uso sin red ni reentrenamiento, reforzando la tesis offline.
- Determinismo y reproducibilidad preservados; auditabilidad total vía log append-only.
- Cold-start sin regresión: degrada con gracia a comportamiento puramente ontológico.
- Absorbe tres patrones ya priorizados: `AuditLogger` (= `outcomeEvidence`),
  `decideDiagnosticAction()` puro (= `rankHypotheses`), proof-of-inference receipt
  (= `knowledgeVersion` por sesión).

### Negativas / costos

- Nuevas tablas y migración Drizzle; superficie de esquema mayor.
- Requiere UI de captura de resultado y disciplina del usuario para que el lazo
  tenga señal de calidad.
- La bucketización de la firma es un hiperparámetro: gruesa generaliza pero
  pierde precisión; fina es precisa pero tarda en acumular evidencia.

### Riesgos y mitigaciones

- **Sesgo por feedback escaso o ruidoso** → scoring por límite inferior + priors
  ontológicos dominan hasta que hay volumen.
- **Contaminación de la ontología** → separación estricta `source` +
  cola de curación para aristas aprendidas.
- **Drift entre `contextStats` y la evidencia** → `contextStats` siempre
  reconstruible desde `outcomeEvidence`; agregar job de verificación.

## Alternativas consideradas

- **Fine-tuning online de CARpsy:** descartado. Rompe reproducibilidad, es caro
  on-device y mezcla la señal estadística con la generativa.
- **Pesos aprendidos sin separar provenance:** descartado. Corrompe la ontología
  curada y vuelve imposible auditar el origen de una hipótesis.
- **Aprendizaje dentro de la ruta de decisión:** descartado. Rompe el
  determinismo del router, que es un requisito en contexto automotriz.