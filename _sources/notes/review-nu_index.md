# Informe de revision — `nu_index.ipynb`

_Fecha: 2026-04-20 · 26 celdas · ruta: `USC-FPII/nu_index.ipynb`_

---

## 1. Brevedad (slide-friendly)

**Celdas afectadas: 5**

- **Cell #11** (md, `slide`): 21 líneas renderizadas. Combina la tabla de doblets leptónicos, los números cuánticos del neutrino SM y la ecuación de masa. **Alto** — muy probable desbordamiento de pantalla en RISE.
  - Sugerencia: partir en dos celdas: (a) doblets + quantum numbers, (b) ausencia de $\nu_R$ + término de masa.

- **Cell #15** (md, `slide`): 3 bullets de 2–3 líneas cada uno sobre PMNS, masa absoluta y Dirac/Majorana. 16–18 líneas renderizadas. **Medio** — riesgo de overflow en slides con fuente estándar.
  - Sugerencia: cada bullet → una celda `fragment`, o comprimir a frases más cortas.

- **Cell #21** (md, `slide`): diagrama ASCII de 22 líneas (incluyendo el bloque de código). **Medio** — el diagrama es ancho y alto a la vez; puede no caber verticalmente.
  - Sugerencia: cambiar a `subslide`, reducir anchura del ASCII o usar una imagen.

- **Cell #24** (md, `slide`): 11 líneas para 5 conferencias. Cada entrada es una frase completa con ciudad, país, fechas. **Bajo** — justo en el límite.
  - Sugerencia: convertir a tabla de 3 columnas (Conferencia | Lugar | Año).

- **Cell #8** (md, `slide`): 13 líneas. Último párrafo en prosa (línea final sobre CKM/PMNS) añade una línea no estructurada al final de un bloque de bullets. **Bajo** — no desborda pero contamina el estilo.

---

## 2. Estilo telegráfico

**Celdas afectadas: 7**

- **Cell #3** (md, `slide`): `slide_type=slide`. Dos frases de prosa continua sin bullets ni bold labels.
  - `These lectures are about some selected topics…` → **Alto** para un NB de estilo telegráfico.
  - Sugerencia: `**Topics:** neutrino in SM · oscillations · double-beta decay` (1 bullet, telegraphic).

- **Cell #6** (md, `slide`): bullets inconsistentes: los dos primeros empiezan con "They" (capitalizado), el tercero con "are" (minúscula). **Bajo.**
  - Sugerencia: `- Unique: only fundamental fermion that could be its own antiparticle.`

- **Cell #7** (md, `slide`): bullets de 2 líneas cada uno. El tercero y cuarto son especialmente largos:
  - `"we do not know their nature: neutrinos are the only fundamental fermions that could be their own antiparticle. They can be purely neutral! Neutrinos can be Majorana fermions."` → 3 frases en un solo bullet. **Medio.**
  - Sugerencia: separar en 2 bullets o comprimir a `- Unknown nature: could be Majorana (purely neutral)`.

- **Cell #8** (md, `slide`): párrafo final de prosa: `"The CKM and PMNS matrices are both rooted in Higgs physics: they arise from the Yukawa couplings of quarks and leptons to the Higgs field across three generations."` — 2 líneas continuas sin bullet. **Medio.**
  - Sugerencia: eliminar o convertir en nota (`<!-- -->` o celda `notes`).

- **Cell #15** (md, `slide`): tres bullets de 2–3 líneas, sin bold labels. Tendencia a prosa larga. **Medio.**
  - Sugerencia: añadir bold labels al inicio de cada bullet, e.g. `**PMNS matrix:**`, `**Absolute mass:**`, `**Dirac vs Majorana:**`.

- **Cell #16** (md, `subslide`): caption debajo de la imagen: `"The vastly different scale of neutrino masses compared with other fermions may be an indication of a new energy scale."` — frase de prosa tras un figura. **Bajo.**
  - Sugerencia: convertir en bullet italicizado o eliminar (la figura habla por sí sola).

- **Cell #17** (md, `fragment`): empieza con `"But the mass of the neutrino, the flavour structure and their coupling to the Higgs, raise further questions:"` — conector de prosa largo antes de los bullets. **Medio.**
  - Sugerencia: reemplazar por bold label: `**Further open questions:**`

---

## 3. Inglés

**Celdas afectadas: 3**

- **Cell #6** (md, `slide`): inconsistencia de capitalización en lista.
- **Cell #7** (md, `slide`): bullets sin capitalización consistente.
- **Cell #17** (md, `fragment`): British English (`flavour`) coherente, pero introductoria larga. **Bajo.**

No errores de terminología técnica graves.

---

## 4. Metadata de slideshow

**Estado inicial:**
- 26 celdas totales
- 24 celdas con `slide_type` definido y no vacío
- 2 celdas con `slide_type: ""` (vacío): Cell #0 y Cell #25

**Auto-fix requerido (no aplicado — permisos denegados):**

| Celda | Tipo | slide_type actual | Valor inferido | Regla |
|-------|------|-------------------|----------------|-------|
| #0 | code | `""` | `skip` | celda de setup (`import orbit_nb`) |
| #25 | code | `""` | `skip` | celda vacía al final |

**Celdas a revisar manualmente:**
- **Cell #13** (md, `slide`): considerar `fragment`.
- **Cell #18** (md, `slide`): 1 línea, considerar `fragment`.
- **Cell #21** (md, `slide`): `### Conceptual map`, considerar `subslide`.

---

## 5. Formato de ejercicios

**No se detectaron ejercicios.** `nu_index` es índice/introducción.

---

## Recuento global

| Prioridad | N issues |
|-----------|----------|
| **Alto** | 2 |
| **Medio** | 6 |
| **Bajo** | 10 |

**Total: 26 celdas · 2 issues altos · 6 medios · 10 bajos**

---

## Resumen ejecutivo

**Modificaciones pendientes de aplicación manual:**
1. Cell #0 (`import orbit_nb`): `"slide_type": ""` → `"skip"`
2. Cell #25 (empty code): `"slide_type": ""` → `"skip"`

**Problemas principales:**

| Prioridad | Celda | Descripción |
|-----------|-------|-------------|
| Alto | #3 | Prosa continua; reformular en bullets telegráficos |
| Alto | #11 | 21 líneas (SM lepton content + mass term); partir en dos |
| Medio | #7 | Bullets de 2–3 líneas; comprimir o separar en fragments |
| Medio | #8 | Párrafo final de prosa tras bullets; eliminar o convertir a `notes` |
| Medio | #15 | Bullets prose-heavy (PMNS/Majorana); añadir bold labels |
| Medio | #17 | Conector `"But the mass of…"`; reemplazar por bold label |
| Bajo | #6, #7 | Inconsistencia de capitalización en bullets |
| Bajo | #13, #18, #21 | `slide_type` presente pero posiblemente inadecuado |
| Bajo | #16 | Caption de figura en prosa; convertir o eliminar |
| Bajo | #24 | Lista de conferencias verbosa; considerar tabla |
