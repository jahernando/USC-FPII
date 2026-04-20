# Review — nu_oscillations.ipynb

_Reviewer: lecture-nb-reviewer agent · Date: 2026-04-20_
_Path: `USC-FPII/nu_oscillations.ipynb` · 113 cells (0–112)_

---

## Summary

| Criterion | Issues found | High | Medium | Low |
|-----------|-------------|------|--------|-----|
| 1. Brevity | 18 cells | 5 | 9 | 4 |
| 2. Telegraphic style | 14 cells | 2 | 8 | 4 |
| 3. English | 14 issues | 3 | 7 | 4 |
| 4. Slideshow metadata | 0 missing | — | 5 (wrong type) | — |
| 5. Exercise format | 6 exercises | 0 | 3 | 3 |

**Global totals: 113 cells · 10 high · 22 medium · 15 low**

---

## 1. Brevity (slide-friendly)

**Findings:** 18 cells exceed recommended length or are candidates for splitting. 5 are flagged High (likely to overflow a slide). The code cell #0 (CSS) is intentionally long but is `skip`, which is correct.

### High (blocks slide rendering)

- **Cell #41** (md, slide) — `"In the two-family case, the free Hamiltonian…"` — **26 rendered lines** with 4 display equations. Split after the `U H U^T` product into a new subslide or fragment.
- **Cell #43** (md, slide) — `"The matter part of the Hamiltonian…"` — **23 lines**, 3 display equations. Split after defining x into a new cell.
- **Cell #47** (md, slide) — `"Resonance condition. When increasing the density…"` — **25 lines**, 4 display equations. Split resonance condition (lines 1–12) from the inequality constraint (lines 13–25).
- **Cell #96** (md, skip) — derivation "Manipulating the amplitude" — **32 lines** with 6 equations. Marked `skip`; OK for students but consider moving the entire derivation to a notes file.
- **Cell #103** (md, slide) — `"Let's consider the case of atmospheric neutrinos with accelerator…"` — **22 lines**, 5 equations. Split MINOS ν_μ survival derivation into two cells.

### Medium (tight fit, recommend trimming)

- **Cell #17** (md, slide) — E/L regimes discussion — **19 lines**. Three indented cases with sub-equations. Consider one case per fragment.
- **Cell #21** (md, slide) — Exercise: Oscillation length — **17 lines**. Two questions with inline math. Acceptable as exercise but tight.
- **Cell #22** (md, slide) — Exercise: Exclusion region — **16 lines** + 3-part question 2 inline. Split Q1 from Q2/Q3.
- **Cell #38** (md, slide) — Matter potential V_e estimate — **20 lines** with Sun/Earth values. Consider two cells.
- **Cell #46** (md, slide) — Matter domination scenarios — **19 lines**. Split vacuum limit from matter domination.
- **Cell #50** (md, slide) — MSW adiabatic P(νe→νe) — **16 lines**. Just within limit but dense.
- **Cell #55** (md, slide) — Exercise: MSW resonance — **16 lines**. Two multi-part questions. Split Q1/Q2.
- **Cell #95** (md, slide) — Three-family amplitude derivation — **30 lines** (at the limit). Consider splitting at the probability formula line.
- **Cell #101** (md, slide) — φ_solar/φ_atm ratio approximation — **21 lines**. Split the two cases into separate fragments.

### Low (cosmetic / borderline)

- **Cell #7** (md, slide) — oscillation diagram + 3 equations — **18 lines**. Borderline; image helps.
- **Cell #14** (md, slide) — 3 regimes + first-max equation — **18 lines**.
- **Cell #97** (md, slide) — simplified three-family formula — **16 lines** mostly equations; OK.
- **Cell #110** (md, slide) — References — **45 lines**. Use `skip` or split into two slides; or reduce to a single-line bibliography pointer.

---

## 2. Telegraphic style

**Findings:** 14 cells contain prose sentences where bullets or bold-label format would be more slide-appropriate. The NB mixes telegraphic cells (good) with paragraph prose cells (should be revised).

### High (breaks slide flow with long prose)

- **Cell #4** (md, slide) — `## Introduction` — 5 full prose sentences with no bullets. Convert to a 4-bullet timeline:
  ```
  - 1960s: theoretical introduction of mixing/oscillations
  - 1970s–1994: **Solar Neutrino Problem** — Davis deficit
  - 1997: **SK first evidence** of atmospheric oscillations
  - Today: solid 3-family picture; some anomalies remain
  ```
- **Cell #40** (md, slide) — `"Neutrino propagation in matter can be described by quantum mechanics, via a modified Dirac equation or QFT, with the same result. We present here the propagation…"` — Two-sentence prose intro followed by math. Convert to: `**Method:** Schrödinger equation (same result as QFT approach)`.

### Medium

- **Cell #5** (md, slide) — `"And again Pontecorvo associated neutrino mixing and oscillations [25] in 1967."` — Conversational connector "And again". Replace with `- 1967: Pontecorvo linked mixing and oscillations [25]`.
- **Cell #35** (md, slide) — `"From a combined result of three phases of SNO [17], the total flux of 8B solar neutrino is…"` — Passive/wordy. Suggest: `- SNO combined (3 phases) [17]: $\Phi_{^8\!B} = (5.25\pm0.16^{+0.11}_{-0.14})\times10^6$ cm⁻²s⁻¹ — consistent with SSM`.
- **Cell #37** (md, slide) — `"Neutrinos can *coherently scatter* via NC in matter, but only νe can CC! (Mikheyev, Smirnov effect [20]) The interaction will be related to the *density of electrons*…"` — Two loose sentences. Add bold labels: `**NC:** all flavours scatter coherently; **CC:** νe only (Wolfenstein–MSW [20])`.
- **Cell #46** (md, slide) — `"Therefore νe ↔ ν₂ and νμ ↔ ν₁. Neutrinos of a given flavour have unique mass states and they do not oscillate!"` — Connector "Therefore". Rewrite as bullets: `- νe ↔ ν₂, νμ ↔ ν₁ (unique mass states)` / `- No oscillation in matter-dominated limit`.
- **Cell #49** (md, slide) — `"When the matter density varies along propagation, the solutions are more complicated… But if the density variation is slow (adiabatic regime)…"` — Long compound sentence with "But if". Rewrite as two bullets: `**Variable density:** numerical solutions in general` / `**Adiabatic limit:** density varies slowly → stay in same mass eigenstate`.
- **Cell #50** (md, slide) — `"If the energy of the neutrino is below the resonance, in the vacuum dominated, and the detector size (Earth) is large compared with the oscillation length, the probability is averaged"` — 35-word sentence. Break into: `**Below resonance** (vacuum-dominated, L ≫ L_osc): averaged probability`.
- **Cell #104** (md, slide) — `"Notice that as P(νμ→νe) is suppressed, then it is sensitive to second-order effects,"` — Sentence cut off (comma at end). Complete or remove. Suggest: `**Note:** P(νμ→νe) suppressed → probe of sub-leading effects (θ₁₃, δCP)`.
- **Cell #2** (md, slide) — `"Present the theory and the main experimental evidence for neutrino oscillations."` — Single full sentence as entire cell body. Convert to bold label: `**Goal:** theory of oscillations + main experimental evidence`.

### Low

- **Cell #6** (md, slide) — `"About the derivation of the probability formula: The basic ingredients: … Different derivations with the same result:"` — Nearly telegraphic but "About the derivation" heading is not a bold label. Change to `**Derivation approaches:**`.
- **Cell #99** (md, slide) — `"There are at least three massive neutrinos, we define two mass squared differences"` — Run-on. Split: `**Two independent Δm²:** (solar + atmospheric)`.
- **Cell #74** (md, slide) — `"T2K exploits the fact that the neutrino spectrum is narrower (but less intense) off-axis by 2.5°"` — Wordy. Suggest: `- Off-axis beam (2.5°): narrower, less intense spectrum`.
- **Cell #63** (md, slide) — `"The solution is MSW adiabatic flavour transitions in solar matter, the so-called large mixing angle (LMA), with parameters:"` — One sentence OK but "the so-called" is informal. Suggest: `**Solution:** MSW adiabatic transitions in solar matter → **LMA solution**`.

---

## 3. English (spelling, grammar, terminology)

**Findings:** 14 issues found. 3 are High (wrong standard terminology or grammar errors that could confuse students). Most are typos or inconsistencies.

### High

- **Cell #35** (md, slide) — `"In 2001, SNO reported the initial result of CC measurement [15], was evidence of non-νe flux."` — Fragment: missing subject, "was evidence" needs "which was" or rephrase. Suggest: `"SNO (2001) [15]: CC result showed non-νe component in solar flux"`.
- **Cell #250** (source line, md) — `"Coherence of mass eigen-states over macroscopic distances"` → `eigenstates` (one word, no hyphen). Standard particle physics terminology. Also appears at line 284: `"eigen-states of the free hamiltonian"` → `eigenstates of the free Hamiltonian`. Affects cells #6 and #7.
- **Cell #1160** (source line, md, cell #45) — `"**Questions:** Check that they correspond to the eigen-values of H"` → `eigenvalues`. Standard notation.

### Medium

- **Cell #5** (md, slide) — `"And again Pontecorvo associated neutrino mixing and oscillations [25] in 1967."` — "And again" is informal/informal connector. Rewrite: `"Pontecorvo (1967) associated neutrino mixing with oscillations [25]."` or bullet form.
- **Cell #25** (md, slide) — `"During 1964-2005 J. Bahcall et al elaborated the Standard Solar Model, SSM."` — `elaborated` is a false cognate from Spanish (`elaboró`). Correct: `developed` or `constructed`. Also missing comma: "Bahcall et al., SSM".
- **Cell #26** (md, slide) — `"The detector techniques are sensitive to different energy ranges. Galium:"` — `Galium` → `Gallium` (standard English spelling).
- **Cell #37** (md, slide) — `"Neutrinos can *coherently scatter* via NC in matter, but only νe can CC!"` — "can CC" is not standard phrasing. Suggest: `only νe scatter via CC` or `only νe have CC interactions`.
- **Cell #40** (md, slide) — `"Where H₀ is the hamiltonian and U is the mix matrix in vacuum"` — `hamiltonian` → `Hamiltonian` (proper noun); `mix matrix` → `mixing matrix`.
- **Cell #56** (md, slide) — `"### [KamLand Experiment]"` — heading uses `KamLand`; correct is `KamLAND` (all experiments inside use correct capitalisation). Fix heading.
- **Cell #74** (md, slide) — `"off-axis by 2.5$^o$ degrees"` — `$^o$ degrees` is redundant ("degrees degrees"). Use either `$2.5°$` or `$2.5^\circ$`. Also: `JPARC` → `J-PARC`.
- **Cell #82** (md, slide) — `"## SBL Reactor Experiments. The  $\theta_{13}$ angle"` — double space before `$\theta_{13}$`. Minor but visible in source.
- **Cell #89** (md, slide) — `"### DoubleCHOOZ"` — inconsistent: roadmap (cell #3, line 180) calls it `Double Chooz`. Standardise to `Double CHOOZ` (official experiment name).
- **Cell #999** (source line, md, cell #40) — `"Where H₀ is the hamiltonian and U is the mix matrix in vacuum, and Ve affects only νe."` — capital `Where` mid-cell (should be lowercase since it continues a math expression block).

### Low

- **Cell #4** (md, slide) — `"neutrino oscillation with three neutrino flavours"` — British `flavours` vs. American `flavors`. NB mixes both; pick one convention and apply throughout (current dominant usage in the NB is `flavour` — acceptable for a European course).
- **Cell #74** (md, slide) — `"T2K exploits the fact that the neutrino spectrum is narrower (but less intense) off-axis"` — `exploits` sounds slightly informal in academic English. Neutral: `T2K uses the off-axis configuration (2.5°):`.
- **Cell #104** (md, slide) — cut-off sentence: `"Notice that as P(νμ→νe) is suppressed, then it is sensitive to second-order effects,"` — trailing comma, sentence incomplete. Fix or remove.

---

## 4. Slideshow metadata

**Findings:** All 113 cells already have `slide_type` defined. **No auto-fix was needed.**

| Metric | Count |
|--------|-------|
| Cells with `slide_type` defined | 113 |
| Cells where `slide_type` was added | 0 |
| Cells with questionable `slide_type` (suggest manual review) | 5 |

### Cells with questionable slide_type (present but potentially incorrect)

Rule: exercise cells should be `subslide`; continuation math cells of a derivation should be `fragment` or `subslide`.

- **Cell #21** (md, `slide`) — `**Exercise: Oscillation length…**` — exercises should be `subslide` (new slide independent of section). Suggest changing to `subslide`.
- **Cell #22** (md, `slide`) — `**Exercise: Exclusion region…**` — same rationale. Suggest `subslide`.
- **Cell #55** (md, `slide`) — `**Exercise: The MSW resonance…**` — suggest `subslide`.
- **Cell #62** (md, `slide`) — `**Exercise: KamLAND…**` — suggest `subslide`.
- **Cell #72** (md, `slide`) — `**Exercise: SuperKamiokande…**` — suggest `subslide`.
- **Cell #92** (md, `slide`) — `**Exercise: Daya Bay…**` — suggest `subslide`.
- **Cell #16** (code, `slide`) — `import oscillations` (bare import) — a standalone import as `slide` will show an empty code slide. Suggest `skip` or `fragment`.

---

## 5. Exercise format

**Findings:** 6 exercise cells found; 1 additional `Question`-style cell. Overall consistency is good. All use `**Exercise: <Title>**` format. Issues are minor.

### Exercise cells identified

| Cell | Header pattern | slide_type | Length | Notes |
|------|---------------|-----------|--------|-------|
| #21 | `**Exercise: Oscillation length and experimental regimes**` | slide | 17 lines | 3 questions; good structure |
| #22 | `**Exercise: Exclusion region in the (Δm², sin²2θ) plane**` | slide | 16 lines | 3 sub-questions; good |
| #55 | `**Exercise: The MSW resonance in the Sun**` | slide | 16 lines | 2 questions; OK |
| #62 | `**Exercise: KamLAND and the measurement of θ₁₂**` | slide | 13 lines | 2 questions; OK |
| #72 | `**Exercise: SuperKamiokande and the Up/Down asymmetry**` | slide | 14 lines | 2 questions; OK |
| #92 | `**Exercise: Daya Bay and the reactor determination of θ₁₃**` | slide | 13 lines | 2 questions; OK |

### Issues

- **Consistency (Medium):** All 6 exercises use `**Exercise: Title**` format — consistent. However, all are `slide_type=slide` instead of `subslide`. This means they are embedded in the flow of the surrounding section rather than starting a fresh slide stack. Recommend changing all 6 to `subslide` for cleaner navigation in RISE.

- **Numbering (Low):** Exercises are not numbered (no "Exercise 1", "Exercise 2"…). If the NB is used standalone, numbered exercises aid reference. Consider adding `**Exercise 1:**`, `**Exercise 2:**`, etc.

- **Question cells (Low):** There are also two bare `**Question:**` cells (cells ~#15 and ~#45, lines 473 and 551/1160) with a different format from the Exercise cells. Decide: convert to `**Exercise N:**` or keep as inline quick questions. Currently inconsistent with the 6 main exercises.

- **Cell #53** (md, subslide) — `**Question:**` — this has `subslide` correctly, unlike the 6 exercise cells which have `slide`. It is an isolated estimation question without a title label. Add a brief title: `**Question: estimate θ₁₂ from Borexino data**`.

---

## Global issues / other notes

- **Image filename typo:** Cell #7 references `./imgs/nu_oscilations_diagram.png` (one `l` in `oscilations`). If the file is actually named that way on disk, it works, but the filename itself is misspelled. Consider renaming to `nu_oscillations_diagram.png` and updating the reference.

- **Cell #1 type mismatch:** The cell at JSON line 141 is `cell_type: markdown` but its slide_type is `slide` with no markdown heading — its source is `import orbit_nb`. This is actually `cell_type: code` looking at context; confirm that the metadata matches the cell type in the JSON.

- **Trailing content cell #104:** Sentence `"Notice that as P(νμ→νe) is suppressed, then it is sensitive to second-order effects,"` ends with a comma and no closing thought. This is likely an incomplete draft sentence. Either complete it (`"…to subleading effects: θ₁₃ and δ_CP"`) or remove.

- **Reference numbering:** References jump from [1] directly to [5], then [6,7,8,13,14,15,16,17,20,21,22,23,24,25,26b,27,28,29,33,38]. References [2,3,4,9,10,11,12,18,19,26,30,31,32,34–37] are absent. This is not an error (references may be used in other NB chapters) but a reader might notice the gaps.

- **Cell #109** (`## Next: Current and Future Experiments`) is a placeholder with no content. Mark `slide_type: skip` or add a brief bullet list of upcoming topics.

---

## Priority matrix (all issues)

| Priority | Count | Description |
|----------|-------|-------------|
| **High** | 10 | 5 overlong cells (overflow), 2 prose-heavy cells (no bullets), 3 grammar/terminology errors |
| **Medium** | 22 | 9 moderately long cells, 8 telegraphic style issues, 7 English errors, 5 wrong slide_type for exercises |
| **Low** | 15 | 4 borderline-length cells, 4 minor style, 4 low-severity English, 3 exercise format cosmetics |
