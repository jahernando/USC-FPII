#!/usr/bin/env python
# coding: utf-8

# # Neutrino Mass: Scale and Absolute Measurements

# In[ ]:


import time
print(' Last version ', time.asctime() )


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import numpy as np
import matplotlib.pyplot as plt


# **Objective:**
# 
# Neutrino oscillations measure *mass-squared differences*, not the absolute mass scale. This notebook covers the complementary measurements that constrain the absolute neutrino mass:
# 
# - Kinematic measurements (KATRIN, Project 8)
# - Cosmological bounds (Planck, DESI)
# - The interplay between different observables and mass ordering
# 

# ## Neutrino Mass from Oscillations: Recap
# 
# Oscillation experiments measure mass-squared differences:
# 
# $$\Delta m^2_{21} = m_2^2 - m_1^2 = (7.41^{+0.21}_{-0.20})\times 10^{-5}\,\mathrm{eV}^2$$
# 
# $$|\Delta m^2_{31}| = (2.511^{+0.028}_{-0.027})\times 10^{-3}\,\mathrm{eV}^2 \quad (\text{NuFit-6.0, 2024})$$
# 
# These fix the *splittings* but not the absolute scale $m_1$. Two orderings are allowed:
# 
# | Ordering | $m_1$ | $m_2$ | $m_3$ | $\sum m_i$ (min) |
# |---|---|---|---|---|
# | **Normal (NH)** | lightest | $m_1+7\times10^{-3}$ eV | $m_1+50\times10^{-3}$ eV | $\geq 59$ meV |
# | **Inverted (IH)** | heaviest | $m_3+50\times10^{-3}$ eV | $m_3+51\times10^{-3}$ eV | $\geq 100$ meV |
# 
# **Current status on ordering**: NuFit-6.0 prefers NH at $\sim2.5\sigma$; JUNO first results (2025) favour NH at $\sim2\sigma$ from reactor data alone. A definitive measurement requires JUNO full statistics, DUNE, or HyperK.

# ## Kinematic Neutrino Mass Measurements
# 
# Beta-decay endpoint spectroscopy measures the *effective electron-neutrino mass*:
# 
# $$m^{\mathrm{eff}}_{\nu_e} = \sqrt{\sum_i |U_{ei}|^2\, m_i^2}$$
# 
# This is independent of the Majorana/Dirac nature and of CP phases. Near the endpoint $Q_\beta$, the spectrum is distorted by a factor $\propto \sqrt{(Q_\beta-T)^2 - m^{\mathrm{eff}2}_{\nu_e}}$.

# ### KATRIN: Kinematic Endpoint Measurement
# 
# The **KATRIN** (KArlsruhe TRItium Neutrino) experiment at KIT (Karlsruhe, Germany) measures the tritium $\beta$-decay spectrum with a large MAC-E filter spectrometer.

# ### KATRIN Experiment
# 
# | | |
# | :--: | :--: |
# | <img src="./imgs/Katrin_spectrum.png" width=400 align="center"> | <img src="./imgs/Katrin_E0m2plot.png" width=370 align="center"> |
# | Tritium $\beta$-decay spectrum near endpoint | $m^2_{\nu_e}$ fit result |
# 
# The [KATRIN experiment](http://www.katrin.kit.edu) at KIT (Karlsruhe, Germany) measures the tritium $\beta$-decay spectrum with a large
# magnetic-adiabatic collimation and electrostatic (MAC-E) filter spectrometer.
# 
# **Design sensitivity**: $m^{eff}_{\nu_e} < 0.2$ eV (90% CL) after full $10^3$ day run.
# 
# **Published results**:
#  * 2019 [arXiv:1909.06048](https://arxiv.org/abs/1909.06048): $m^{eff}_{\nu_e} < 1.1$ eV (90% CL)
#  * 2021 [arXiv:2105.08533](https://arxiv.org/abs/2105.08533): $m^{eff}_{\nu_e} < 0.8$ eV (90% CL)
#  * 2024 [arXiv:2406.13516](https://arxiv.org/abs/2406.13516): $m^{eff}_{\nu_e} < \mathbf{0.45}$ **eV** (90% CL) — *Science 2025*
# 
# The 2024 result uses a combined analysis of all data runs (KNM1–KNM5, ~300 days of data)
# with improved systematic treatment, giving:
# 
# $$
# m^2_{\nu_e} = (0.14 \pm 0.15)\;\mathrm{eV}^2 \quad \Rightarrow \quad m^{eff}_{\nu_e} < 0.45\;\mathrm{eV} \;\text{at } 90\%\;\text{CL}
# $$

# ### Interplay with $\beta\beta0\nu$ Searches
# 
# KATRIN constrains $m^{eff}_{\nu_e}$, while $\beta\beta0\nu$ experiments constrain $m_{\beta\beta}$.
# They are related but different:
# 
# | Quantity | Formula | Probes |
# | :-- | :-- | :-- |
# | $m^{eff}_{\nu_e}$ | $\sqrt{\sum_i \|U_{ei}\|^2 m_i^2}$ | absolute mass scale (Dirac or Majorana) |
# | $m_{\beta\beta}$ | $\|\sum_i U^2_{ei} m_i\|$ | Majorana nature + CP phases |
# 
# The KATRIN limit $m^{eff}_{\nu_e} < 0.45$ eV constrains the parameter space of $m_{\beta\beta}$:
# 
#  * In the **quasi-degenerate regime** ($m_0 \gg \Delta m_{ij}$): $m_{\beta\beta} \lesssim m^{eff}_{\nu_e} < 0.45$ eV
#  * For **IH**, the minimum $m_{\beta\beta} \sim 15\text{--}50$ meV is well below the KATRIN reach,
#    so $\beta\beta0\nu$ experiments are the decisive probe.
#  * Future KATRIN sensitivity ($< 0.2$ eV) will further constrain the degenerate region.

# **Exercise: KATRIN — endpoint spectrum and kinematic neutrino mass**
# 
# KATRIN measures the tritium $\beta$-decay spectrum near the endpoint $Q_\beta = 18\,574$ eV.
# Near the endpoint, the differential rate behaves as:
# 
# $$
# \frac{dN}{dT} \propto (Q_\beta - T) \sqrt{(Q_\beta - T)^2 - m^2_{\nu_e}} \;\Theta(Q_\beta - T - m_{\nu_e})
# $$
# 
# The effective mass $m^{eff}_{\nu_e} = \sqrt{\sum_i |U_{ei}|^2 m_i^2}$ sets the kinematic endpoint.
# 
# **Questions:**
# 
# 1. Visualise the spectrum and Kurie plot for $m_\nu = 0$, 0.2, 0.45, 1 eV.
#    Where does the difference become visible?
# 2. Why is measuring a sub-eV neutrino mass so challenging?
#    Estimate the fraction of the spectrum affected for $m_\nu = 0.45$ eV.
# 3. KATRIN gives $m^{eff}_{\nu_e} < 0.45$ eV. Can we directly compare this with
#    $m_{\beta\beta}$ from KamLAND-Zen? What are the key differences?

# In[ ]:


import majorana
majorana.exercise_katrin_endpoint()


# ### Project 8
# 
# [Project 8](https://www.project8.org) uses **Cyclotron Radiation Emission Spectroscopy (CRES)** to measure tritium $\beta$-decay with single-electron sensitivity.
# 
# - Demonstrated 18 eV resolution on a single electron in a magnetic trap
# - Target: $m_{\nu_e} < 40$ meV with an atomic tritium source
# - Complementary to KATRIN: different systematic uncertainties
# - Timeline: atomic source R&D ~2026; physics run ~2030+

# ## Cosmological Bounds on Neutrino Mass
# 
# Massive neutrinos suppress small-scale structure formation (free-streaming). Cosmological observations constrain $\Sigma m_\nu = m_1 + m_2 + m_3$.
# 
# ### CMB + BAO: Planck 2018
# 
# The Planck Collaboration (2018) combining CMB temperature/polarisation and lensing with BAO data (BOSS, 6dFGS, SDSS) obtains:
# 
# $$\Sigma m_\nu < 0.12 \text{ eV} \quad (95\%\,\mathrm{CL})$$
# 
# This is the most stringent laboratory-independent constraint on the neutrino mass scale. It disfavours the IH quasi-degenerate region ($\Sigma m_\nu \gtrsim 0.10$ eV) but is still compatible with both orderings in the non-degenerate limit.
# 
# ### DESI 2024: Baryon Acoustic Oscillations
# 
# The **Dark Energy Spectroscopic Instrument (DESI)** first-year BAO data (2024) combined with CMB (Planck + ACT) and supernova data yields:
# 
# $$\Sigma m_\nu < 0.072 \text{ eV} \quad (95\%\,\mathrm{CL}) \quad [\text{DESI+CMB+SN}]$$
# 
# This result is in **tension** with the minimum mass required by oscillations in the IH ($\Sigma m_\nu^{\min}(\mathrm{IH}) \approx 100$ meV), and is close to the NH minimum ($\approx 59$ meV). It would be the first hint of a neutrino mass detection if confirmed, or a sign of new physics in the dark energy sector.
# 
# **Caution**: cosmological bounds are model-dependent (assume $\Lambda$CDM + $\nu$). Extensions (dynamical dark energy, modified gravity) can shift the bound significantly.

# ## Summary: Neutrino Mass Constraints
# 
# | Observable | Quantity | Best bound | Method |
# |---|---|---|---|
# | $\Delta m^2_{21}$ | mass splitting | $7.41\times10^{-5}$ eV² | Oscillations |
# | $|\Delta m^2_{31}|$ | mass splitting | $2.51\times10^{-3}$ eV² | Oscillations |
# | $m^{\mathrm{eff}}_{\nu_e}$ | endpoint mass | $< 0.45$ eV (90% CL) | KATRIN 2024 |
# | $\Sigma m_\nu$ | sum of masses | $< 0.12$ eV (95% CL) | Planck 2018 |
# | $\Sigma m_\nu$ | sum of masses | $< 0.072$ eV (95% CL) | DESI 2024 |
# | $m_{\beta\beta}$ | Majorana eff. mass | $< 28\text{--}122$ meV (90% CL) | KamLAND-Zen 800 |
# 
# The three approaches are complementary: oscillations give the splittings, kinematic and cosmological measurements constrain the absolute scale, and $\beta\beta0\nu$ probes the Majorana nature.

# ## Conclusions
# 
# - Neutrino oscillations firmly establish that at least two neutrino masses are non-zero, with $\Sigma m_\nu^{\min}(\mathrm{NH}) \approx 59$ meV.
# - KATRIN constrains $m^{\mathrm{eff}}_{\nu_e} < 0.45$ eV; its design goal of $< 0.2$ eV is within reach.
# - Cosmological observations (Planck, DESI) push $\Sigma m_\nu < 0.072\text{--}0.12$ eV, approaching the minimum allowed by oscillations.
# - The mass ordering (NH vs IH) is one of the key remaining unknowns; JUNO and DUNE will determine it definitively in the coming decade.
# - The origin of neutrino mass — Dirac or Majorana? — is addressed in **[Are neutrinos Majorana particles?](nu_majorana.ipynb)**.

# ## References
# 
# [K1] M. Aker et al. (KATRIN), Nature Phys. 18 (2022) 160, [arXiv:2105.08533](https://arxiv.org/abs/2105.08533).
# 
# [K2] M. Aker et al. (KATRIN), Phys. Rev. Lett. 133 (2024) 251801, [arXiv:2406.13516](https://arxiv.org/abs/2406.13516).
# 
# [P18] N. Aghanim et al. (Planck), A&A 641 (2020) A6, [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
# 
# [D24] DESI Collaboration, JCAP 02 (2025) 021, [arXiv:2404.03002](https://arxiv.org/abs/2404.03002).
# 
# [P8] A. Ashtari Esfahani et al. (Project 8), Phys. Rev. Lett. 131 (2023) 102502, [arXiv:2212.05048](https://arxiv.org/abs/2212.05048).
# 
# [NF6] I. Esteban et al. (NuFit-6.0), JHEP 12 (2024) 216, [arXiv:2410.05380](https://arxiv.org/abs/2410.05380).
