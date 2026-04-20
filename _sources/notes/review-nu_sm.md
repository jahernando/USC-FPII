# Review Report — `nu_sm.ipynb`

_Generated: 2026-04-20 · 68 cells · path: `USC-FPII/nu_sm.ipynb`_

---

## Criterion 4 — Slideshow metadata (auto-fix)

**Result: NO changes needed.** All 68 cells already have `slideshow.slide_type` defined.

- Cells with metadata already present: **68**
- Cells added by auto-fix: **0**
- Cells with values that warrant manual review: **9**

---

## Criterion 1 — Brevedad (slide-friendly)

**Summary:** 10 cells exceed the ~15-20 line guideline.

| Cell | Type | slide_type | Issue | Priority |
|------|------|-----------|-------|----------|
| #4 | md | slide | 22 lines. H2 + H3 fused with image + prose + Bohr quote. Split. | Alto |
| #20 | md | slide | 14 lines dense prose (delayed-coincidence). Split or move to notes. | Alto |
| #24 | md | slide | 17 lines. H2 `## Neutrinos & chirality` fused with `### Parity violation`. Split. | Alto |
| #27 | md | subslide | 22 lines of bullets (Goldhaber method). Split after 3rd bullet. | Medio |
| #29 | md | slide | 22 lines: large table + 2 LaTeX displays. Move Observations to fragment. | Medio |
| #34 | md | slide | 18 lines. H2 `## The three neutrino families` mixed with chirality-helicity equations + prose. Split. | Alto |
| #50 | md | slide | 16 lines dense prose on first NC event. Split. | Medio |
| #57 | md | slide | 16 lines. `### Invisible width` mixes derivation, result, observation, note. Split. | Medio |
| #60 | md | slide | 17 lines: large table + 3 observation bullets. Split Observations → fragment. | Medio |
| #64 | md | subslide | 15 lines — recommended readings table. Borderline. | Bajo |

---

## Criterion 2 — Estilo telegráfico

**Summary:** 7 cells with prose-heavy passages.

| Cell | Issue | Suggestion | Priority |
|------|-------|-----------|----------|
| #3 | Three sentences of prose as objective | Two bullets | Medio |
| #17 | Wordy connector ("indeed") | Remove / shorten | Bajo |
| #19 | Trailing prose sentence after image | Convert to bullet | Bajo |
| #20 | Compound sentences after numbered list | Two bullets | Medio |
| #50 | Prose paragraph on NC event | Bold-label: `**Process:**`, `**Event (Jul 1973):**` | Medio |
| #57 | Last paragraph (4 compound sentences) | 3 bullets: peak / width / LEP scan | Medio |
| #34 | "This property was embedded in V-A... Glashow/Weinberg/Salam" | `- **GWS theory:** Glashow (1961), Weinberg (1967), Salam (1968)` | Bajo |

---

## Criterion 3 — Inglés

**Summary:** 6 issues.

| Cell | Issue | Correction | Priority |
|------|-------|-----------|----------|
| #25 | Trailing period in heading `### The helicity of the neutrino.` | Remove period | Bajo |
| #10 | Double space before `$\beta$` | Remove extra space | Bajo |
| #32 | `99% of the time` (pion → muon BR) | `~99.99%` | Bajo |
| #37 | Ref `[20]` points to Neddermeyer & Anderson, not Pontecorvo | Check reference numbering | **Alto** |
| #46 | Sentence fragment: `BR(mu→eγ) 𝒪(10⁻⁵²) due to U_PMNS` | Add verb: `~ 𝒪(10⁻⁵²) within SM (GIM via U_PMNS)` | Medio |
| #46 | Empty link text `[](https://arxiv.org/...)` | Add link text | Medio |

---

## Criterion 4 — Metadata de slideshow

**Final state:** 68 cells, all with `slide_type` defined. No auto-fix applied.

**Dubious values (manual review):**

| Cell | Current | Recommended | Reason |
|------|---------|-------------|--------|
| #5 | slide | subslide | Exercise mid-section |
| #13 | slide | subslide | Exercise mid-section |
| #16 | slide | subslide | Exercise mid-section |
| #21 | slide | subslide | Exercise mid-section |
| #33 | slide | subslide | Exercise mid-section |
| #45 | slide | subslide | Exercise mid-section |
| #59 | slide | subslide | Exercise mid-section |
| #64 | subslide | slide | H2 `## Recommended readings` should open new slide |
| #65 | subslide | skip | References page |
| #66 | slide | skip | References page |
| #67 | slide | skip | References page |

---

## Criterion 5 — Formato de ejercicios

**Summary:** 7 full exercises. 2 structural inconsistencies.

| Cell | Header | Structure | Issues |
|------|--------|-----------|--------|
| #5 | **Exercise: Simulating β-decay electron spectrum** | Statement + Questions in #6 | Split — inconsistent |
| #13 | **Exercise: Fermi constant from muon lifetime** | Statement + Questions in #14 | Split — inconsistent |
| #16 | **Exercise: Neutrino mean free path** | Same cell | OK |
| #21 | **Exercise: Reactor ν flux and event rate** | Same cell | OK |
| #33 | **Exercise: Helicity suppression in π and K decays** | Same cell | OK |
| #45 | **Exercise: Quantitative test of lepton universality** | Same cell | OK |
| #59 | **Exercise: Z lineshape and N_ν** | Same cell | OK |

**Action:** Merge #5+#6 and #13+#14.

---

## Global summary

**Total:** 68 cells · **Alto:** 5 · **Medio:** 12 · **Bajo:** 10

### Top-priority actions

1. **Cell #37** — Fix wrong reference (`[20]` cited as Pontecorvo, but is Neddermeyer & Anderson).
2. **Cells #4, #24, #34** — Split fused H2+H3 cells.
3. **Cells #5+#6, #13+#14** — Merge exercise statement with its Questions block.
4. **Cells #20, #50, #57** — Split long prose cells (14–16 lines) or convert to bullets.
5. **Cell #46** — Fix broken empty link and BR(μ→eγ) sentence fragment.

**Auto-fix applied:** None (permiso `NotebookEdit` no disponible; todas las cells ya tenían `slide_type`).
