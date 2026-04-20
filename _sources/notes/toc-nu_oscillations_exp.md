# TOC — nu_oscillations_exp

_65 celdas · `USC-FPII/nu_oscillations_exp.ipynb`_

## Table of contents

- 1. Current and Future Neutrino Oscillation Experiments
- 2. Long-Baseline Experiments: δ_CP and mass hierarchy
  - 2.1 List of LBL experiments
  - 2.2 LBL appearance probability (Cervera formula)
  - 2.3 NOvA (detector, results 2021, conferences, 10-yr analysis 2025)
  - 2.4 T2K (appearance vs E, PID, Neutrino 2024)
  - 2.5 T2K + NOvA joint analysis (Oct 2025, Nature)
  - 2.6 Exercise 1: LBL appearance and CP asymmetry
- 3. Neutrino Mixing — Global fits
  - 3.1 NuFit-6.0 (2024)
  - 3.2 Next-generation experiments overview
- 4. SBL: JUNO (2025–)
  - 4.1 Detector + goals + status
  - 4.2 Prospects
  - 4.3 First results (Nov 2025)
  - 4.4 Exercise 2: JUNO reactor spectrum interference
- 5. LBL: DUNE (~2031–)
  - 5.1 Map + appearance probability
  - 5.2 Physics + beam + detector + status
  - 5.3 ProtoDUNE-HD and VD
  - 5.4 ICARUS at LNGS
  - 5.5 DUNE sensitivities
- 6. HyperKamiokande (2028–)
  - 6.1 Detector layout + construction
  - 6.2 Appearance spectra
  - 6.3 δ_CP sensitivity + coverage
  - 6.4 Accelerator sensitivity (May 2025)
  - 6.5 Parameter resolution table
- 7. Summary and conclusions
  - 7.1 Oscillation timeline
- 8. Appendix: Neutrino telescopes
  - 8.1 KM3NeT: ORCA + ARCA
  - 8.2 IceCube: DeepCore, Upgrade, astrophysical
- 9. References [1]–[45]

## Cell listing

| # | Type | Slide | Identifier | Notes |
|---|------|-------|------------|-------|
| 0 | code | skip | `import orbit_nb` | setup |
| 1 | md | slide | # Current and Future Neutrino Oscillation Experiments | H1 |
| 2 | md | slide | Intro bullets LBL/Reactor/Telescopes | prose line; consider fragment |
| 3 | md | slide | ## Long Base Line Experiments | rename → Long-Baseline |
| 4 | md | slide | ### List of LBL (image) | suggest subslide |
| 5 | md | - | LBL appearance probability Cervera formula | — |
| 6 | md | - | Structure of amplitude bullets | — |
| 7 | md | slide | ### NOvA — beam + off-axis + detector | suggest subslide |
| 8 | md | - | NOvA map + photo + event | — |
| 9 | md | - | NOvA νμ disappearance results | prose-heavy |
| 10 | md | - | CP violation goal + formula | — |
| 11 | md | - | NOvA bi-probability diagram | — |
| 12 | md | - | NOvA numu/nue 2021 | — |
| 13 | md | - | NOvA δ limits 2021 | — |
| 14 | md | - | NOvA TAUP 2019 + ICHEP 2022 | caption: "ICHEP 2021" → fix |
| 15 | md | - | NOvA Neutrino 2024 | — |
| 16 | md | slide | #### NOvA 10-yr analysis (Sep 2025) | suggest subslide |
| 17 | md | subslide | NOvA 10-yr results | — |
| 18 | md | slide | ### T2K — P(νe)/P(ν̄e) | suggest subslide; δ-CP notation |
| 19 | md | - | T2K PID 2019 | — |
| 20 | md | - | T2K appearance + θ₂₃ vs δ_CP 2020 | — |
| 21 | md | - | T2K ellipses + CP posterior 2024 | — |
| 22 | md | slide | ### T2K + NOvA joint (Oct 2025, Nature) | suggest subslide |
| 23 | md | subslide | T2K+NOvA bi-probability | — |
| 24 | md | subslide | T2K+NOvA allowed regions | — |
| 25 | md | slide | **Exercise: LBL appearance + CP asymmetry** | 35 lines; split; rename `### Exercise 1` |
| 26 | md | slide | ## Neutrino Mixing Parameters — Global fits | prose; add bold label |
| 27 | md | - | NuFit-6.0 table + description | — |
| 28 | md | - | NuFit 2024 fit table | — |
| 29 | md | slide | ### Next-generation experiments | — |
| 30 | md | slide | ### SBL: JUNO (2025–) | — |
| 31 | md | - | JUNO map + drawing + image | — |
| 32 | code | fragment | `import oscillations` / plot_juno_spectrum | — |
| 33 | md | - | JUNO prospects: spectrum + P_ee | — |
| 34 | md | slide | #### JUNO first results (Nov 2025) | suggest subslide |
| 35 | md | subslide | JUNO spectrum + tension 2025 | — |
| 36 | md | slide | **Exercise: JUNO reactor spectrum** | 30 lines; split; rename `### Exercise 2` |
| 37 | md | slide | ### LBL: DUNE — map + appearance | — |
| 38 | md | - | DUNE physics + beam + status 2025-26 | 22 lines; split Status |
| 39 | md | - | ProtoDUNE-HD and VD | — |
| 40 | md | - | ICARUS at LNGS | prose tail line |
| 41 | md | - | DUNE Phase I FD1/FD2 | — |
| 42 | md | - | DUNE MH + δ_CP sensitivity | — |
| 43 | md | - | DUNE sensitivity vs modules | — |
| 44 | md | - | DUNE reach vs exposure | — |
| 45 | md | slide | ### HyperKamiokande (2028–) | — |
| 46 | md | - | HK construction + detector + J-PARC | 18 lines; split |
| 47 | md | - | HK appearance probability vs E | — |
| 48 | md | - | HK nue/barnue spectrum | — |
| 49 | md | - | HK δ_CP sensitivity 2018 | — |
| 50 | md | slide | #### HK accelerator sensitivity (May 2025) | suggest subslide |
| 51 | md | subslide | HK expected nue events | caption prose-heavy |
| 52 | md | subslide | HK CPV significance + δ_CP res 2025 | — |
| 53 | md | subslide | HK parameter resolution table | — |
| 54 | md | slide | ## Summary and conclusions | — |
| 55 | md | **MISSING** | Oscillation timeline SVG | **no metadata — needs `fragment`** |
| 56 | md | slide | ## Appendix: Neutrino telescopes | prose; add bold label |
| 57 | md | slide | ### KM3NeT: ORCA and ARCA | — |
| 58 | md | slide | #### KM3NeT/ORCA | suggest subslide |
| 59 | md | slide | #### KM3NeT/ARCA | suggest subslide |
| 60 | md | slide | ### IceCube: DeepCore, Upgrade, atmospheric | 28 lines; 3 subslides |
| 61 | md | slide | ## References [1]–[9] | suggest skip |
| 62 | md | - | References [10]–[19] | suggest skip |
| 63 | md | - | References [20]–[29] | suggest skip |
| 64 | md | - | References [30]–[45] | suggest skip |
