# Informe de revisión — `nu_oscillations_exp.ipynb`

_Generado: 2026-04-20 · 65 celdas (0–64) · Ruta: `USC-FPII/nu_oscillations_exp.ipynb`_

**Nota operativa:** Auto-fix de metadata no aplicado (permisos `NotebookEdit` denegados en el subagente).

---

## 1. Brevedad (slide-friendly)

**Resumen:** 3 celdas exceden ~20 líneas; 2 en el límite.

- **Cell #25 (md, slide) — Alto** — **Exercise: LBL appearance probability** (35 líneas). Split: enunciado + 3 `fragment` (Q1/Q2/Q3).
- **Cell #36 (md, slide) — Alto** — **Exercise: JUNO reactor spectrum** (30 líneas). Misma estructura.
- **Cell #38 (md, "-") — Medio** — DUNE Physics/Beam/Status (22 líneas). Mover **Status** a `fragment`.
- **Cell #46 (md, "-") — Medio** — HK construction (18 líneas). Separar **J-PARC beam** en `fragment`.
- **Cell #60 (md, slide) — Medio** — IceCube DeepCore/Upgrade/Astrophysical (~28 líneas). 3 subslides.

---

## 2. Estilo telegráfico

**Resumen:** 4 celdas con prosa o sin bold labels.

- **Cell #2 (md, slide) — Medio** — Frase introductoria de prosa. Reemplazar por `**Experimental programme:**`.
- **Cell #9 (md, "-") — Alto** — 3 frases seguidas sobre NOvA νμ→νμ y octant ambiguity. Convertir a bullets.
- **Cell #26 (md, slide) — Medio** — Dos frases largas sobre NuFit. Reemplazar por `**Global fits:** ... → [NuFit](http://www.nu-fit.org)`.
- **Cell #40 (md, "-") — Bajo** — Frase suelta sobre ProtoDUNE; redundante con #39.
- **Cell #56 (md, slide) — Medio** — Frase de prosa sola. Convertir a bullet o mover a `notes`.

---

## 3. Inglés

**Resumen:** 3-4 errores menores.

- **Cell #3 — Medio** — `"Long Base Line"` → `"Long-Baseline"` (término estándar).
- **Cell #14 — Bajo** — Caption tabla: `ICHEP 2021` → `ICHEP 2022` (inconsistente con título y filename).
- **Cell #18 — Bajo** — `δ-CP` → `$\delta_{\rm CP}$` (consistencia).
- **Cell #25 — Bajo** — `hierarchy--δ_CP` doble guión → em-dash.

---

## 4. Metadata de slideshow

**Estado:** 65 celdas. **1 celda sin metadata** (fix requerido). 10 celdas con valores dudosos.

### 4a. Celda sin metadata (auto-fix pendiente)

- **Cell #55 (md) — Alto** — `"metadata": {}` vacío (SVG timeline). Valor sugerido: `fragment` (continúa el Summary de #54).

Fix JSON:
```json
"metadata": {
  "slideshow": {"slide_type": "fragment"}
}
```

### 4b. Valores existentes dudosos

| Cell | Actual | Sugerido | Motivo |
|------|--------|----------|--------|
| 2 | slide | fragment | Continúa título H1 de #1 |
| 4 | slide | subslide | H3 `### List of LBL` |
| 7 | slide | subslide | H3 `### NOvA` |
| 16 | slide | subslide | H4 `#### NOvA 10-yr` |
| 18 | slide | subslide | H3 `### T2K` |
| 22 | slide | subslide | H3 `### T2K + NOvA joint` |
| 34 | slide | subslide | H4 `#### JUNO first results` |
| 50 | slide | subslide | H4 `#### HK sensitivity` |
| 58 | slide | subslide | H4 `#### KM3NeT/ORCA` |
| 59 | slide | subslide | H4 `#### KM3NeT/ARCA` |
| 61–64 | slide / "-" | skip | References |

### 4c. Resumen

| Estado | Celdas |
|--------|--------|
| `slide_type` explícito | 36 |
| `slide_type: "-"` (válido) | 28 |
| Sin metadata (fix) | **1** (#55) |
| **Total** | **65** |

---

## 5. Formato de ejercicios

**Resumen:** 2 ejercicios con encabezado no uniforme y longitud excesiva.

| Cell | Encabezado actual | Recomendado |
|------|-------------------|-------------|
| #25 | `**Exercise: LBL appearance probability...**` | `### Exercise 1: LBL appearance probability and CP asymmetry` |
| #36 | `**Exercise: JUNO and the reactor spectrum...**` | `### Exercise 2: JUNO reactor spectrum interference` |

Ambos con `slide_type: subslide` y split Q1/Q2/Q3 en `fragment`.

---

## Recuento global

| Prioridad | Issues |
|-----------|--------|
| Alto | 3 (cells #25, #36, #55) |
| Medio | 9 |
| Bajo | 4 |

### Acciones pendientes (requieren permisos)

1. **Cell #55** — Añadir `slideshow.slide_type = "fragment"` al `metadata: {}` vacío.
2. Resto de sugerencias son cambios de contenido del autor.
