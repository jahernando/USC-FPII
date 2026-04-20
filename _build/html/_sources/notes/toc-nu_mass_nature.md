# TOC — nu_mass_nature

_Generated: 2026-04-20 · 104 cells · path: `USC-FPII/nu_mass_nature.ipynb`_

---

## Table of Contents

- Preamble: Neutrino Mass and Nature (title + objective + roadmap)
- 1. Experimental Limits on the Neutrino Mass
  - 1.1 Neutrino mass from oscillations: recap
  - 1.2 Kinematic neutrino mass measurements
  - 1.3 KATRIN experiment
    - Detection method (MAC-E)
    - First results (KNM1, 2019)
    - Combined result (KNM1–KNM5)
    - Evolution of results
    - Exercise: KATRIN endpoint spectrum and kinematic neutrino mass
  - 1.4 Future experiments
    - Project 8
  - 1.5 Cosmological bounds on neutrino mass
    - CMB + BAO: Planck 2018
    - DESI 2024: Baryon Acoustic Oscillations
    - Tension / model dependence
  - 1.6 Summary: neutrino mass constraints
- 2. How to Provide Mass to the Neutrinos
  - 2.1 Dirac neutrino masses
  - 2.2 Dirac masses in the SM
    - Sterile right-chiral field and hierarchy problem
  - 2.3 Majorana fermions
    - The Majorana condition
    - Constructing the Majorana spinor
    - Nature of neutrinos: Dirac or Majorana?
  - 2.4 The Weinberg operator
    - Weinberg op. continuation and Feynman diagrams
  - 2.5 Mixing and the U_PMNS Matrix
  - 2.6 Dirac masses: three families
  - 2.7 The U_PMNS matrix (Pontecorvo–Maki–Nakagawa–Sakata)
  - 2.8 Majorana masses: three families
    - Majorana PMNS phases
  - 2.9 The seesaw mechanism
    - Type I seesaw
    - Exercise: The seesaw mechanism and the PMNS matrix
- 3. How to Determine the Nature of the Neutrino
  - 3.1 Signature of Dirac and Majorana neutrinos
    - LNV helicity suppression
  - 3.2 Double beta decay without neutrinos
  - 3.3 ββ2ν: the allowed double beta decay
  - 3.4 Phase factor and nuclear matrix elements
  - 3.5 The effective Majorana mass
  - 3.6 Interplay: m_νe, m_ββ, and Σm_ν
  - 3.7 Exercise: Effective Majorana mass spectrum
  - 3.8 Exercise: Half-life limits and NME uncertainty
  - 3.9 Experimental signature of ββ0ν
  - 3.10 Background sources and mitigation
    - Th232/U238 decay chains (images)
    - Muon flux vs depth (image)
  - 3.11 Sensitivity regimes
  - 3.12 Detector requirements
- 4. The Experimental Search for ββ0ν Decays
  - 4.1 Liquid xenon experiments
    - EXO-200 (2011–2015)
      - Detection and calibration (images)
      - EXO-200 results
    - nEXO
    - XLZD
    - KamLAND-Zen (2011–2018)
      - KamLAND-Zen spectra + result
    - KamLAND-Zen 800 (2019–2024)
      - KamLAND-Zen 800 exclusion plot + result
    - KamLAND2-Zen
  - 4.2 Germanium semiconductor experiments
    - GERDA (2015–2020)
    - LEGEND-200
      - LEGEND energy spectrum (image)
      - LEGEND-200 spectrum images
      - LEGEND-200 published results (PRL 2026)
    - LEGEND-1000
  - 4.3 Cryogenic bolometers
    - CUORE (2017–2024)
      - CUORE assembly images
    - CUPID — CUORE upgrade
  - 4.4 High-pressure gaseous xenon experiments
    - NEXT experiment (2017–)
      - NEXT-100 cross-section (image)
      - NEXT topology/blobs discrimination (images)
      - NEXT T_bb0nu sensitivity plots
    - NEXT-White detector
    - NEXT-100 (2024–)
      - NEXT-100 detector images
      - NEXT-100 S2/Kr events (images)
      - First NEXT-100 results (Nov 2025)
    - NEXT-HD
  - 4.5 Summary: current limits and future prospects
  - 4.6 Next-generation ββ0ν experiments
    - Next-gen sensitivities table
  - 4.7 The 2026 ESPP Briefing Book on ββ0ν
    - ESPP ton-scale experiments table
    - Theory and R&D priorities
    - ββ0ν timeline (SVG)
  - 4.8 Exercise: Experimental sensitivity as a function of exposure
- 5. Conclusions
  - Conclusions continuation (m_ν cosmic role)
- Appendix: About heavy neutral leptons
  - Why heavy neutrinos?
  - HNL searches at the LHC
  - SHiP — CERN SPS Beam Dump Facility
  - Other long-lived particle experiments
- References (cells 100–102)
- End divider

---

## Cell Listing

| # | Type | Slide | Identifier | Notes |
|---|------|-------|------------|-------|
| 0 | code | skip | CSS/style setup cell | — |
| 1 | md | slide | # Neutrino Mass and Nature | H1 title |
| 2 | code | skip | import numpy, matplotlib; print timestamp | — |
| 3 | md | slide | **Objective**: oscillations prove m_ν≠0, three questions | — |
| 4 | md | slide | ### Roadmap — 6-section outline | — |
| 5 | md | slide | ## 1. Experimental Limits on the Neutrino Mass | H2 section |
| 6 | md | slide | ### Neutrino mass from oscillations: recap | — |
| 7 | md | slide | ### Kinematic neutrino mass measurements | — |
| 8 | md | slide | ### KATRIN experiment | — |
| 9 | md | **MISSING** | Detection method — MAC-E filter description | → fragment |
| 10 | md | slide | ### KATRIN: first results (2019) | — |
| 11 | md | slide | ### KATRIN: combined result (KNM1–KNM5) | — |
| 12 | md | slide | ### KATRIN: evolution of results | — |
| 13 | md | slide | **Exercise: KATRIN — endpoint spectrum and kinematic neutrino mass** | exercise |
| 14 | md | slide | ### Future experiments / ### Project 8 | double heading: two ### in one cell |
| 15 | md | slide | ### Cosmological bounds on neutrino mass / #### CMB + BAO | mixed heading levels in one cell |
| 16 | md | slide | #### DESI 2024: Baryon Acoustic Oscillations | prose-heavy (2 lines) |
| 17 | md | **MISSING** | **Tension!** — cosmological cap discussion | → fragment |
| 18 | md | slide | ### Summary: neutrino mass constraints | — |
| 19 | md | slide | ## 2. How to Provide Mass to the Neutrinos | H2 section |
| 20 | md | slide | ### Dirac neutrino masses | — |
| 21 | md | slide | ### Dirac masses in the SM | — |
| 22 | md | **MISSING** | sterile ν_R, Yukawa continuation, hierarchy problem | → fragment |
| 23 | md | slide | ### Majorana fermions | — |
| 24 | md | slide | #### The Majorana condition | — |
| 25 | md | slide | #### Constructing the Majorana spinor | — |
| 26 | md | **MISSING** | Neutrinos have mass… Dirac or Majorana? (2-line prose) | → fragment; prose |
| 27 | md | slide | ### The Weinberg operator | — |
| 28 | md | **MISSING** | Weinberg op. only 5-dim operator + Feynman diagram images | → fragment |
| 29 | md | slide | ### Mixing and the U_PMNS Matrix | — |
| 30 | md | slide | ### Dirac masses: three families | — |
| 31 | md | slide | ### The U_PMNS matrix (Pontecorvo–Maki–Nakagawa–Sakata) [MNS, Pon] | — |
| 32 | md | slide | ### Majorana masses: three families | — |
| 33 | md | **MISSING** | **Majorana case**: PMNS Majorana phases η₁,η₂ | → fragment |
| 34 | md | slide | ### The seesaw mechanism | — |
| 35 | md | slide | #### Type I seesaw | — |
| 36 | md | slide | **Exercise: The seesaw mechanism and the PMNS matrix** | exercise |
| 37 | md | slide | ## 3. How to Determine the Nature of the Neutrino | H2 section |
| 38 | md | slide | ### Signature of Dirac and Majorana neutrinos | — |
| 39 | md | **MISSING** | LNV helicity suppression — probability table | → fragment |
| 40 | md | slide | ### Double beta decay without neutrinos | — |
| 41 | md | slide | ### ββ2ν: the allowed double beta decay | — |
| 42 | md | slide | ### Phase factor and nuclear matrix elements | — |
| 43 | md | slide | ### The effective Majorana mass | — |
| 44 | md | slide | ### Interplay: m_νe, m_ββ, and Σm_ν | — |
| 45 | md | slide | **Exercise: Effective Majorana mass spectrum** | exercise |
| 46 | md | slide | **Exercise: Half-life limits and NME uncertainty** | exercise |
| 47 | md | slide | ### Experimental signature of ββ0ν | — |
| 48 | md | slide | ### Background sources and mitigation | — |
| 49 | md | **MISSING** | Th232/U238 decay chain images | → fragment |
| 50 | md | **MISSING** | Muon flux vs depth underground labs (image) | → fragment |
| 51 | md | slide | ### Sensitivity regimes | long (18 lines) |
| 52 | md | slide | ### Detector requirements | prose-heavy ending |
| 53 | md | slide | ## The Experimental Search for ββ0ν Decays / ### Liquid xenon | missing section number; double heading |
| 54 | md | slide | ### EXO-200 (2011–2015) | — |
| 55 | md | **MISSING** | EXO-200 detection technique / calibration (images) | → fragment |
| 56 | md | slide | **EXO-200 results:** full table + result line | long (14 lines), prose at end |
| 57 | md | slide | #### nEXO (very likely out of the race!) | editorial comment in heading |
| 58 | md | slide | #### XLZD | — |
| 59 | md | slide | ### KamLAND-Zen (2011–2018) | — |
| 60 | md | **MISSING** | KamLAND-Zen spectra + result sentence | → fragment; prose sentence |
| 61 | md | slide | ### KamLAND-Zen 800 (2019–2024) | — |
| 62 | md | **MISSING** | KamLAND-Zen 800 exclusion plot + combined result | → fragment |
| 63 | md | slide | #### KamLAND2-Zen | — |
| 64 | md | slide | ### Germanium semiconductor experiments | — |
| 65 | md | slide | ### GERDA (2015–2020) | long (16 lines) |
| 66 | md | slide | ### LEGEND-200 | — |
| 67 | md | **MISSING** | LEGEND energy spectrum cuts (image) | → fragment |
| 68 | md | **MISSING** | LEGEND-200 spectrum images (two plots) | → fragment |
| 69 | md | **MISSING** | LEGEND-200 published results (PRL 2026) | → fragment |
| 70 | md | slide | #### LEGEND-1000 | — |
| 71 | md | slide | ### CUORE (2017–2024) | — |
| 72 | md | **MISSING** | CUORE assembly images | → fragment |
| 73 | md | slide | #### CUPID — CUORE upgrade | — |
| 74 | md | slide | ### High-pressure gaseous xenon experiments | prose-only (1 sentence) |
| 75 | md | slide | ### NEXT experiment (2017–) | — |
| 76 | md | **MISSING** | NEXT-100 cross-section image | → fragment |
| 77 | md | **MISSING** | NEXT topology/blobs discrimination (images) | → fragment |
| 78 | md | **MISSING** | NEXT T_bb0nu vs Exposure / BkgIndex plots | → fragment |
| 79 | md | slide | #### NEXT-White detector | — |
| 80 | md | slide | ### NEXT-100 (2024–) | typo: "autoreload" ref in NEXT-7 |
| 81 | md | **MISSING** | NEXT-100 detector and tracking planes (images) | → fragment; typo in caption |
| 82 | md | **MISSING** | NEXT-100 S2/Kr events (images) | → fragment |
| 83 | md | **MISSING** | First NEXT-100 results (Nov 2025) — two bullet points | → fragment; long bullet lines |
| 84 | md | slide | #### NEXT-HD | — |
| 85 | md | slide | ### Summary: current limits and future prospects | long (18+ lines); prose-heavy |
| 86 | md | slide | ### Next-generation ββ0ν experiments (image) | — |
| 87 | md | **MISSING** | Next-gen sensitivities table (4 experiments) | → fragment |
| 88 | md | slide | ### The 2026 ESPP Briefing Book on ββ0ν | — |
| 89 | md | **MISSING** | ESPP ton-scale experiments table | → fragment |
| 90 | md | **MISSING** | Theory/R&D priorities + bottom line | → fragment |
| 91 | md | **MISSING** | bb0nu timeline SVG image | → fragment |
| 92 | md | slide | **Exercise: Experimental sensitivity as a function of exposure** | exercise |
| 93 | md | slide | ## Conclusions | — |
| 94 | md | **MISSING** | Conclusions continuation: m_ν cosmic role | → fragment |
| 95 | md | slide | ## Appendix: About heavy neutral leptons | H2 appendix |
| 96 | md | slide | ### Why heavy neutrinos? | — |
| 97 | md | slide | ### HNL searches at the LHC | — |
| 98 | md | slide | ### SHiP — CERN SPS Beam Dump Facility | — |
| 99 | md | slide | ### Other long-lived particle experiments | long (15 lines) |
| 100 | md | slide | ## References (Majorana, Weinberg, MNS, Pon, [1]–[7b]) | long (16 lines) |
| 101 | md | slide | References continued ([9]–[L1K26]) | long (16 lines) |
| 102 | md | slide | NEXT references ([NEXT-1] – [FY86]) | — |
| 103 | md | slide | --- (end divider) | — |
