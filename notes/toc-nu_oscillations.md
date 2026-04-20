# TOC — nu_oscillations

_Generated: 2026-04-20 · 113 cells · path: `USC-FPII/nu_oscillations.ipynb`_

---

## Table of contents

- 1. Introduction
  - 1.1 Roadmap
  - 1.2 The origin of neutrino oscillations
- 2. Two-family oscillation probability
  - 2.1 Oscillation probability: two-family case (derivation slides)
  - 2.2 Oscillation length and regimes
  - 2.3 Interactive oscillation: two families (code widget)
  - 2.4 Exercise: Oscillation length and experimental regimes
  - 2.5 Exercise: Exclusion region in the (Δm², sin²2θ) plane
- 3. Solar Neutrinos
  - 3.1 Davis experiment and the Solar Neutrino Problem
  - 3.2 Solar Model (Bahcall SSM)
  - 3.3 (Super)Kamiokande experiment
  - 3.4 SNO experiment
  - 3.5 Solar parameter fits
  - 3.6 Neutrino oscillations in matter (MSW effect)
  - 3.7 Matter Hamiltonian and effective parameters
  - 3.8 Resonance condition and adiabatic limit
  - 3.9 Borexino results
  - 3.10 Question: estimating θ₁₂ from Borexino data (code)
  - 3.11 Exercise: The MSW resonance in the Sun
  - 3.12 KamLAND experiment
  - 3.13 Exercise: KamLAND and measurement of θ₁₂
  - 3.14 Solution of Solar Neutrino Problem
- 4. Atmospheric Neutrinos
  - 4.1 SuperKamiokande atmospheric results
  - 4.2 Exercise: SuperKamiokande and the Up/Down asymmetry
  - 4.3 Confirmation of atmospheric oscillations (LBL experiments)
  - 4.4 T2K
  - 4.5 MINOS
- 5. SBL Reactor Experiments — the θ₁₃ angle
  - 5.1 Daya Bay Experiment
  - 5.2 RENO Experiment
  - 5.3 Double CHOOZ
  - 5.4 θ₁₃ summary
  - 5.5 Exercise: Daya Bay and the reactor determination of θ₁₃
- 6. Oscillations with 3 neutrinos
  - 6.1 PMNS matrix
  - 6.2 Three-family oscillation probability (derivation)
  - 6.3 CP, T and CPT symmetry
  - 6.4 Mass ordering
  - 6.5 Oscillation probability in experimental scenarios
    - 6.5.1 Atmospheric/accelerator regime (MINOS)
    - 6.5.2 Solar/reactor regime (KamLAND)
    - 6.5.3 SBL reactor regime (Daya Bay)
  - 6.6 Interactive oscillation: three families (PMNS)
- 7. Next: Current and Future Experiments
- 8. References

---

## Cell listing

| # | Type | Slide | Identifier | Notes |
|---|------|-------|------------|-------|
| 0 | code | skip | CSS / Orbit theme setup (HTML style cell) | long (~116 lines); OK as skip |
| 1 | md | slide | title import `orbit_nb` | source: `import orbit_nb` — code, not md |
| 2 | md | slide | **(objective slide)** Present theory and experimental evidence | prose sentence, not telegraphic |
| 3 | md | slide | ### Roadmap (1–6) | OK |
| 4 | md | slide | ## Introduction | prose-heavy (5 sentences); should be bullets; long (10 lines) |
| 5 | md | slide | ## The origin of neutrino oscillations | prose; "And again Pontecorvo…" informal |
| 6 | md | slide | About the derivation of the probability formula | OK |
| 7 | md | slide | Oscillation diagram + flavour-to-mass mixing equation | long (~18 lines) |
| 8 | md | slide | Amplitude formula A(α→β) | OK |
| 9 | md | slide | Two-family case: mixing matrix + time evolution | OK |
| 10 | md | subslide | Manipulating amplitude → sinusoidal form | OK |
| 11 | md | subslide | Amplitude (subslide continuation) | OK |
| 12 | md | slide | **Oscillation probability** sin²2θ sin²(ΔmL/4E) | OK |
| 13 | md | slide | Oscillation length L_osc | OK |
| 14 | md | slide | Three detection regimes: E/L vs Δm² | long (~18 lines); prose intro sentence |
| 15 | md | slide | Units: phase = 1.27 Δm²L/E | OK |
| 16 | code | slide | `import oscillations` (single import) | slide_type=slide unusual for a bare import |
| 17 | md | slide | E/L ≪ Δm² and E/L ≫ Δm² regimes (averages) | long (~19 lines); candidate to split |
| 18 | md | slide | dm2_ranges table (image) | OK |
| 19 | md | slide | ### Interactive oscillation: two families | OK |
| 20 | code | fragment | `oscillations.plot_2fam_interactive()` | OK |
| 21 | md | slide | **Exercise: Oscillation length and experimental regimes** | long (~17 lines); exercise slide_type should be subslide |
| 22 | md | slide | **Exercise: Exclusion region (Δm², sin²2θ) plane** | long (~16 lines); slide_type=slide should be subslide |
| 23 | md | slide | ## Solar Neutrinos (Davis image) | OK |
| 24 | md | slide | Davis experiment history bullets | OK |
| 25 | md | slide | ### Solar Model + solar_chains image | OK |
| 26 | md | slide | Solar flux image | OK |
| 27 | md | slide | Solar experiments table image | OK |
| 28 | md | slide | Solar deficit results image + measurements | OK |
| 29 | md | slide | ### (Super)Kamiokande experiment | OK |
| 30 | md | slide | SK detector description bullets | OK |
| 31 | md | slide | SK solar flux result | OK |
| 32 | md | slide | ### SNO experiment (photo) | OK |
| 33 | md | slide | SNO event + detection channels | OK |
| 34 | md | slide | SNO results image | OK |
| 35 | md | slide | SNO CC/NC combined results | long (~9 lines); "From a combined result…" prose; "was evidence of" grammar issue |
| 36 | md | slide | Solar parameter fits (dm2_tan2theta image) | OK |
| 37 | md | slide | ### Neutrino oscillations in matter (MSW) | prose-heavy; "Mikheyev, Smirnov effect" reference attribution oddly placed |
| 38 | md | slide | Matter potential V_e: numerical estimate | long (~20 lines); OK content |
| 39 | md | slide | V_e in terms of n_e + coherence length | OK |
| 40 | md | slide | Schrödinger equation for matter propagation | prose intro; "via a modified Dirac equation or QFT" — informal |
| 41 | md | slide | Free Hamiltonian in 2-family + mixing matrix | long (~26 lines) — candidate to split |
| 42 | md | slide | Time evolution equation + vacuum probability | OK |
| 43 | md | slide | Matter Hamiltonian total: effective parameters | long (~23 lines) — candidate to split |
| 44 | md | subslide | Effective Hamiltonian in mass basis + mixing angle in matter | OK |
| 45 | md | slide | Effective propagation probability + Question | OK |
| 46 | md | slide | Different scenarios: vacuum limit, matter domination | long (~19 lines) |
| 47 | md | slide | Resonance condition | long (~25 lines) — too long; split recommended |
| 48 | code | fragment | `oscillations.plot_msw_parameters()` | OK |
| 49 | md | slide | Adiabatic propagation (varying density) | long (~12 lines); "But if the density…" connector prose |
| 50 | md | slide | MSW adiabatic result: P(νe→νe) = sin²θ₀ | long (~16 lines) |
| 51 | md | slide | solar_matter image | OK |
| 52 | md | slide | Borexino results | OK |
| 53 | md | subslide | **Question:** estimate θ₁₂ from Borexino figure | OK |
| 54 | code | fragment | code: compute theta12 from Borexino data | OK |
| 55 | md | slide | **Exercise: The MSW resonance in the Sun** | long (~16 lines); slide_type=slide should be subslide |
| 56 | md | slide | ### KamLAND Experiment (map image) | "KamLand" typo (capital D missing) |
| 57 | md | slide | KamLAND drawing image | OK |
| 58 | md | slide | KamLAND photo image | OK |
| 59 | md | slide | KamLAND detector description bullets | OK |
| 60 | md | slide | KamLAND oscillation plot | OK |
| 61 | md | slide | KamLAND dm2_tan2theta12 fit image | OK |
| 62 | md | slide | **Exercise: KamLAND and measurement of θ₁₂** | long (~13 lines); slide_type=slide should be subslide |
| 63 | md | slide | ### Solution of Solar Neutrino Problem | OK |
| 64 | md | slide | ## Atmospheric neutrinos | OK |
| 65 | md | slide | Atmospheric neutrino production + flux bullets | OK |
| 66 | md | slide | SK atmospheric event image | OK |
| 67 | md | slide | SK lepton direction, event types bullets | OK |
| 68 | md | slide | SK first result: Up/Down asymmetry | OK |
| 69 | md | slide | SK L/E oscillation pattern result | OK |
| 70 | md | slide | SK best parameters (dm2, theta) | OK |
| 71 | md | slide | SK zenith distributions image | OK |
| 72 | md | slide | **Exercise: SuperKamiokande and Up/Down asymmetry** | long (~14 lines); slide_type=slide should be subslide |
| 73 | md | slide | ### Confirmation of Atmospheric oscillations (LBL image) | OK |
| 74 | md | slide | ### T2K description | "off-axis by 2.5$^o$ degrees" redundant unit; typo "JPARC" (should be J-PARC) |
| 75 | md | slide | T2K map image | OK |
| 76 | md | slide | T2K beam off-axis image | OK |
| 77 | md | slide | T2K numu disappearance result | OK |
| 78 | md | slide | ### MINOS | OK |
| 79 | md | slide | MINOS description bullets | OK |
| 80 | md | slide | MINOS numu disappearance result | OK |
| 81 | md | slide | MINOS second result plot | OK |
| 82 | md | slide | ## SBL Reactor Experiments. The θ₁₃ angle | double space in "The  $\\theta_{13}$" |
| 83 | md | slide | ### Daya Bay Experiment | OK |
| 84 | md | slide | Daya Bay detector description | OK |
| 85 | md | slide | Daya Bay first result | OK |
| 86 | md | slide | Daya Bay 2018 result + fit | OK |
| 87 | md | slide | ### RENO Experiment | OK |
| 88 | md | slide | RENO results | OK |
| 89 | md | slide | ### DoubleCHOOZ | inconsistent capitalization (Double CHOOZ vs DoubleCHOOZ) |
| 90 | md | slide | DoubleCHOOZ far/near ratio plot | OK |
| 91 | md | slide | θ₁₃ summary (Neutrino 2024) | OK |
| 92 | md | slide | **Exercise: Daya Bay and reactor determination of θ₁₃** | long (~13 lines); slide_type=slide should be subslide |
| 93 | md | slide | ## Oscillations with 3 neutrinos + diagram | OK |
| 94 | md | slide | PMNS matrix image | OK |
| 95 | md | slide | Three-family oscillation probability derivation (From amplitude) | long (~30 lines) — matches limit exactly |
| 96 | md | skip | Derivation: manipulating amplitude step-by-step | very long (~32 lines); skip is appropriate |
| 97 | md | slide | Simplified probability formula (with i>j) | long (~16 lines) |
| 98 | md | slide | ### CP, T and CPT symmetry transformations | OK |
| 99 | md | slide | Mass ordering: two measured Δm² values | long (~16 lines) |
| 100 | md | slide | Mass ordering: NH vs IH diagram | OK |
| 101 | md | slide | Ratio φ_solar/φ_atm approximation | long (~21 lines) — candidate to split |
| 102 | md | slide | ### Oscillation probability in experimental scenarios | OK |
| 103 | md | slide | MINOS approximation: P(νμ→νμ) | long (~22 lines) — split recommended |
| 104 | md | slide | MINOS: P(νμ→νe) and P(νμ→ντ) | incomplete last bullet "Notice that as… is sensitive to second-order effects," — sentence cut off |
| 105 | md | slide | KamLAND approximation: P(ν̄e→ν̄e) | long (~15 lines) |
| 106 | md | slide | Daya Bay SBL approximation: P(ν̄e→ν̄e) | OK |
| 107 | md | slide | ### Interactive oscillation: three families (PMNS) | long (~11 lines) — OK |
| 108 | code | fragment | `oscillations.plot_3fam_interactive()` | OK |
| 109 | md | slide | ## Next: Current and Future Experiments | placeholder slide only |
| 110 | md | slide | ## References | long (~45 lines); ref numbering gap (1,5,6,7,8,13–17,20–29,33,38) — not sequential |
| 111 | code | skip | empty cell | OK |
