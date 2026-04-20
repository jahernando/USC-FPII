# Review — nu_mass_nature.ipynb

_Generated: 2026-04-20 · 104 cells (2 code + 102 markdown)_

---

## Metadata auto-fix status

**BLOCKED — auto-fix could not be applied.**

The Edit tool is rejected for `.ipynb` files (tool returns "Use NotebookEdit"), and NotebookEdit does not expose a metadata field — it only replaces cell source. The complete list of cells requiring `slide_type: fragment` is provided in Section 4 below so the fix can be applied manually or via a script.

**Manual fix script** (run once in terminal):

```python
import json, pathlib

NB = pathlib.Path("USC-FPII/nu_mass_nature.ipynb")
nb = json.loads(NB.read_text())

FRAGMENT_IDS = {
    "ae1fb101",  # cell 9  — KATRIN MAC-E method
    "c3adebbd",  # cell 17 — Tension cosmological cap
    "cb611238",  # cell 22 — sterile ν_R continuation
    "4872f780",  # cell 26 — Dirac/Majorana nature question
    "7cc34ca7",  # cell 28 — Weinberg op. continuation
    "23ccf7de",  # cell 33 — Majorana PMNS phases
    "76836614",  # cell 39 — LNV helicity suppression
    "28bba958",  # cell 49 — Th232/U238 decay chains
    "e2c033a4",  # cell 50 — muon flux image
    "c9a6f9a0",  # cell 55 — EXO-200 images
    "d296fe8d",  # cell 60 — KamLAND-Zen spectra + result
    "7c45fd00",  # cell 62 — KamLAND-Zen 800 result
    "3a831343",  # cell 67 — LEGEND spectrum cuts image
    "20dff6e4",  # cell 68 — LEGEND-200 spectrum images
    "d9291831",  # cell 69 — LEGEND-200 published results
    "973eb82d",  # cell 72 — CUORE assembly images
    "acfe260c",  # cell 76 — NEXT-100 xsection image
    "945449de",  # cell 77 — NEXT topology images
    "45141e1d",  # cell 78 — NEXT sensitivity plots
    "25f474e7",  # cell 81 — NEXT-100 detector images
    "3f89d19a",  # cell 82 — NEXT-100 events images
    "5040bcd7",  # cell 83 — First NEXT-100 results
    "65a38d95",  # cell 87 — next-gen sensitivities table
    "40d03de2",  # cell 89 — ESPP ton-scale table
    "04a74731",  # cell 90 — Theory/R&D priorities
    "5dcf320d",  # cell 91 — bb0nu timeline image
    "a78db253",  # cell 94 — Conclusions continuation
}

for cell in nb["cells"]:
    if cell.get("id") in FRAGMENT_IDS:
        cell.setdefault("metadata", {})
        cell["metadata"]["slideshow"] = {"slide_type": "fragment"}

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("Done —", len(FRAGMENT_IDS), "cells patched.")
```

---

## Section 1 — Brevity (slide-friendly)

**Summary: 6 cells with length or density issues (2 High, 4 Medium).**

- **Cell #51** (md, slide — `### Sensitivity regimes`): 18 rendered lines including two display equations + figure + long caption sentence. Will overflow a standard slide. **Priority: Alto** — split into (a) the two-regime equations and (b) the figure + caption.

- **Cell #85** (md, slide — `### Summary: current limits and future prospects`): 18+ lines — long table (5 cols × 4 rows) plus 5 prose bullet points. **Priority: Alto** — move the 5 "Key observations" bullets to a `fragment` cell.

- **Cell #56** (md, slide — `**EXO-200 results:**`): 14 lines including a centred figure. Borderline; acceptable but the prose sentence at the end could be a separate fragment. **Priority: Medio**.

- **Cell #65** (md, slide — `### GERDA (2015–2020)`): 16 lines including two sub-figures and a result sentence. Likely overflows. **Priority: Medio** — move result sentence to a fragment.

- **Cell #99** (md, slide — `### Other long-lived particle experiments`): 15 lines including a 5-row table + 5-bullet complementarity block + 2-line conclusion sentence. Tight but borderline. **Priority: Medio**.

- **Cell #100–101** (md, slide — References): 16 lines each. References pages are typically `notes` or `skip` in slide presentations, not standalone slides displayed to students. **Priority: Bajo** — consider setting `slide_type: notes` or `skip`.

---

## Section 2 — Telegraphic Style

**Summary: 5 cells with prose issues (1 High, 4 Medium).**

- **Cell #16** (md, slide — `#### DESI 2024`): Source line 480 contains a run-on sentence spanning 2 lines without a period: `"The Dark Energy Spectroscopic Instrument (DESI) measured BAO from >6 million extragalactic objects (galaxies, quasars, Lyα forest) across 0.1 < z < 4.2 in its first year of observations"`. This is a continuous prose sentence. **Priority: Alto** → Replace with bullet format: `* >6 million objects (galaxies, QSOs, Lyα) across 0.1 < z < 4.2 — first year`.

- **Cell #26** (md, MISSING slide_type — `"Neutrinos have mass..."`): 2 lines of narrative prose with no bullet structure: `"Neutrinos have mass — as demonstrated by oscillations — and they are the only fundamental fermions that could be Majorana particles. / *What, then, is the nature of neutrinos: Dirac or Majorana?*"`. **Priority: Medio** — acceptable as a rhetorical bridging cell but should be a `fragment` under cell #25. Keep as-is if intentional; else refactor to bold-label + one-liner.

- **Cell #52** (md, slide — `### Detector requirements`): Last sentence is prose: `"Different isotopes and detection techniques offer complementary trade-offs between these three requirements, as we shall see in the next section."` This is a connector phrase typical of written text. **Priority: Medio** → Delete or replace with `→ see next section`.

- **Cell #60** (md, MISSING — KamLAND-Zen result): Last sentence is a long 2-line prose statement: `"Despite the modest energy resolution, the large mass and the absence of events in the RoI made this the world-leading limit at the time."` **Priority: Medio** → Convert to two bullets: `* Modest ΔE (~10% FWHM) compensated by large mass` and `* World-leading limit at the time`.

- **Cell #85** (md, slide — Summary table): 5 "Key observations" bullets are written as full sentences with subordinate clauses (e.g., `"the discovery gain of CUPID comes from scintillation particle ID, reducing the α background by two orders of magnitude"`). **Priority: Medio** → Compress to short telegraphic labels.

---

## Section 3 — English

**Summary: 6 issues found (0 High, 4 Medium, 2 Bajo).**

- **Cell #57** (md, slide): Heading `"#### nEXO (very likely out of the race!)"` — editorial comment in a heading is informal and student-facing. Also potentially factually contentious. **Priority: Medio** → Rename to `#### nEXO (5-tonne LXe TPC, SNOLAB)` and move the comment to the cell body if needed.

- **Cell #80** (md, slide — `### NEXT-100 (2024–)`): Source line 1841: `"first xenon operation at 4 bar in October 2024 [[NEXT-7]](https://arxiv.org/abs/2511.01710)"` — reference [NEXT-7] is cited here but in the References (cell #102) it points to arXiv:2505.17848 which is the NEXT-100 detector paper, while arXiv:2511.01710 is [NEXT-8] (Kr calibration). **Priority: Medio** — fix reference label to `[[NEXT-7]](https://arxiv.org/abs/2505.17848)` in cell #80 (the commissioning context is more naturally reference [NEXT-7]).

- **Cell #81** (md, MISSING — NEXT-100 images): Image caption `"NEXT100 traking and energy planes"` — typo: `"traking"` → `"tracking"`. **Priority: Medio**.

- **Cell #14** (md, slide — `### Future experiments / ### Project 8`): Two consecutive `###` headings in one cell (`### Future experiments\n### Project 8`). This renders as two headings on the same slide without visual separation, and structurally conflates section and subsection. **Priority: Medio** → Remove the `### Future experiments` heading and start directly with `### Project 8`, or promote Project 8 to its own slide.

- **Cell #15** (md, slide): Mixes `### Cosmological bounds on neutrino mass` and `#### CMB + BAO: Planck 2018` in a single cell — two heading levels rendered together. **Priority: Bajo** — structurally works but is inconsistent with the pattern used elsewhere (e.g., DESI 2024 gets its own slide cell #16).

- **Cell #53** (md, slide): Section heading `"## The Experimental Search for ββ0ν Decays"` — no number (other sections are `## 1.`, `## 2.`, `## 3.`). The heading immediately continues with `"### Liquid xenon experiments"` in the same cell. **Priority: Bajo** → Add section number `## 4.` and split the `###` subsection to a separate cell.

---

## Section 4 — Slideshow Metadata

**Summary: 27 cells without `slide_type` (all should be `fragment`). 77 cells already have metadata.**

All 27 missing cells follow content slides (not section titles) and contain continuation text or image panels — all should be `fragment`.

| Cell # | Cell ID | First-line identifier | Recommended `slide_type` |
|--------|---------|----------------------|--------------------------|
| 9 | ae1fb101 | Detection method — MAC-E filter | fragment |
| 17 | c3adebbd | **Tension!** cosmological cap | fragment |
| 22 | cb611238 | sterile ν_R field, hierarchy problem | fragment |
| 26 | 4872f780 | Neutrinos have mass… Dirac or Majorana? | fragment |
| 28 | 7cc34ca7 | Weinberg op. only 5-dim op., Feynman diagrams | fragment |
| 33 | 23ccf7de | **Majorana case**: PMNS Majorana phases | fragment |
| 39 | 76836614 | LNV helicity-suppressed probabilities table | fragment |
| 49 | 28bba958 | Th232 / U238 decay chain images | fragment |
| 50 | e2c033a4 | Muon flux vs depth underground labs | fragment |
| 55 | c9a6f9a0 | EXO-200 detection technique / calibration images | fragment |
| 60 | d296fe8d | KamLAND-Zen spectra + result sentence | fragment |
| 62 | 7c45fd00 | KamLAND-Zen 800 exclusion + combined result | fragment |
| 67 | 3a831343 | LEGEND energy spectrum cuts (image) | fragment |
| 68 | 20dff6e4 | LEGEND-200 spectrum two-panel images | fragment |
| 69 | d9291831 | LEGEND-200 published results (PRL 2026) | fragment |
| 72 | 973eb82d | CUORE assembly images | fragment |
| 76 | acfe260c | NEXT-100 cross-section (image) | fragment |
| 77 | 945449de | NEXT topology / blobs discrimination | fragment |
| 78 | 45141e1d | NEXT T_bb0nu vs Exposure / BkgIndex plots | fragment |
| 81 | 25f474e7 | NEXT-100 detector and tracking planes (images) | fragment |
| 82 | 3f89d19a | NEXT-100 S2 / Kr events (images) | fragment |
| 83 | 5040bcd7 | First NEXT-100 results (Nov 2025) | fragment |
| 87 | 65a38d95 | Next-gen sensitivities table (4 experiments) | fragment |
| 89 | 40d03de2 | ESPP ton-scale experiments table | fragment |
| 90 | 04a74731 | Theory / R&D priorities + bottom-line | fragment |
| 91 | 5dcf320d | ββ0ν timeline SVG (image) | fragment |
| 94 | a78db253 | Conclusions continuation: m_ν cosmic role | fragment |

**Counts:**
- Cells already with `slide_type`: 77
- Cells missing `slide_type` (to add): 27 (all → `fragment`)
- Values to review manually: 0

---

## Section 5 — Exercise Format Consistency

**Summary: 5 exercises identified. 3 issues (2 Medium, 1 Bajo).**

The 5 exercises and their cells:

| # | Cell | Heading pattern | slide_type |
|---|------|-----------------|------------|
| E1 | 13 | `**Exercise: KATRIN — endpoint spectrum and kinematic neutrino mass**` | slide |
| E2 | 36 | `**Exercise: The seesaw mechanism and the PMNS matrix**` | slide |
| E3 | 45 | `**Exercise: Effective Majorana mass spectrum**` | slide |
| E4 | 46 | `**Exercise: Half-life limits and NME uncertainty**` | slide |
| E5 | 92 | `**Exercise: Experimental sensitivity as a function of exposure**` | slide |

**Consistent features:** all use `**Exercise: <title>**` as first line, all have `slide_type: slide`, all include numbered questions below. Good overall consistency.

**Issues:**

- **E1 vs E2–E5 internal structure:** E1 has three questions without a `**Questions:**` label header — the list begins immediately after the formula. E2–E5 all use `**Questions:**` (or `**Part A / Part B**`) as a sub-label before the numbered list. **Priority: Medio** → Add `**Questions:**` label to E1, after the display equation block.

- **E2 question length:** E2 (cell #36) has 5 questions and two labelled parts (Part A / Part B), making it significantly longer (15+ lines) than the other exercises (~10 lines each). **Priority: Medio** → Consider splitting into two exercises or moving Part B to a `fragment` cell.

- **Numbering:** There are no exercise numbers in the headings (no "Exercise 1", "Exercise 2", etc.) — just descriptive titles. This is internally consistent but differs from the numbered exercise pattern used in other NB chapters. **Priority: Bajo** — flag for course-level standardisation.

---

## Global Summary

| Criterion | Cells affected | High | Medium | Low |
|-----------|---------------|------|--------|-----|
| 1. Brevity | 6 | 2 | 2 | 2 |
| 2. Telegraphic style | 5 | 1 | 4 | 0 |
| 3. English | 6 | 0 | 4 | 2 |
| 4. Slide metadata | 27 | 0* | 0 | 27 |
| 5. Exercises | 3 | 0 | 2 | 1 |
| **Total** | **35 distinct cells** | **3** | **12** | **5 (+27 metadata)** |

\* The 27 metadata issues are not "blocking" in the sense of corrupting content, but they do cause slides to be presented as `—` (untyped) and may render unexpectedly in RISE. Treat as **Medium** for course use.

**Top-priority actions:**
1. Run the Python script above to add `slide_type: fragment` to all 27 missing cells.
2. Split cell #51 (`### Sensitivity regimes`, 18 lines) and cell #85 (`### Summary: current limits`, 18+ lines).
3. Fix the prose sentence in cell #16 (DESI 2024 description).
4. Rename the heading of cell #57 (nEXO) and fix the `### Future experiments` double-heading in cell #14.
5. Fix the `"traking"` typo in cell #81.
