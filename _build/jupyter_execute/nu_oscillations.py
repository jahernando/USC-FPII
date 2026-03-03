#!/usr/bin/env python
# coding: utf-8

# # Neutrino Oscillations: Theory and Foundations

# In[1]:


import time
print(' Last version ', time.asctime() )


# In[2]:


# general imports
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
 
# numpy and matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats     as stats
import scipy.constants as units

plt.style.context('seaborn-colorblind');


# **Objective:**
# 
# Show the evidences of neutrino oscillations.
# 
# Indicate the open questions and the future.

# ## Introduction
# 
# Neutrino mixing and oscillations were theoretically introduced in the 60 of XX century.
# 
# Davis' experiment persistently reported since 70's less solar neutrino flux that predicted by Bahcall's standard solar model SSM. That was known as the **Solar Neutrino Problem**.
# 
# In 1997 **Super-Kamiokande reported the first evidence of atmospheric neutrino oscillations**.
# 
# Since then neutrino oscillation has been observed in accelerator and reactor neutrinos.
# 
# The current picture of neutrino oscillation with three neutrino flavor and masses is solid, but some inconsistencies exist.
# 
# Neutrino oscillations are **the only evidence of Physics Beyond de SM**.

# ## The origin of neutrino oscillations
# 
# Pontecorvo was the first to introduce the concept of oscillations [1] in 1952.
# 
# Z. Maki, M. Nakagawa and S. Sakata introduced the concept of mixing between mass and flavour states [24] in 1962.
# 
# And again Pontecorvo associated neutrino mixing and oscillations [25] in 1967.
# 

# ### Oscillation probability: two-family case

# About the derivation of the probability formula:
# 
# The basic ingredients:
# 
# - Uncertainty in momentum at production and detection
# 
# - Coherence of mass eigen-states over macroscopic distances
# 
# Different derivations with the same result:
# 
# - quantum mechanics with neutrinos as plane waves
# 
# - quantum mechanics with neutrinos as wave packets
# 
# - quantum field theory
# 
# See P Hernandez's [lectures](https://arxiv.org/abs/1708.01046)

# 
# | |
# | :--: |
# | <img src="./imgs/nu_oscilations_diagram.png" width=600 align="center">|
# 
# Neutrinos of a given flavour, $\nu_\alpha$ are produced via CC interactions.
#     
# They are a combination of neutrinos of given mass, related via a unitary matrix $U$
# 
# $$
# | \nu_{\alpha} \rangle = \sum_{i} U_{\alpha i}^* \, | \nu_i \rangle 
# $$
# 
# That propagate in time, $t$, as eigen-states of the free hamiltonian, $E_i = \sqrt{p^2 + m^2_i}$
# 
# $$
# | \nu  (t) \rangle = \sum_{i} U_{\alpha i}^* \, e^{-i E_i t} \, | \nu_i \rangle
# $$   
# 

# 
# 
# A neutrino, $\nu_\beta$, can now interact via CC with an amplitude:
# 
# $$
# \mathcal{A}_{\alpha \beta}(t) = \langle \nu_\beta \, | \, \nu(t)  \rangle
# $$
#    
# 
# And probability
# 
# $$
# \mathcal{P}(\nu_\alpha \to \nu_\beta, t) = \left| \mathcal{A}_{\alpha  \beta}(t) \right|^2 =
# \left| \sum_{i} U_{\beta i} U^*_{\alpha i} e^{-i E_i  t} \right |^2 
# $$
# 
# 
# 
# 

# Let's consider a 2 families case:
# 
# $$
# \begin{pmatrix} \nu_\alpha \\ \nu_\beta \end{pmatrix}  = 
# \begin{pmatrix} \cos \theta & \sin \theta \\ -\sin \theta & \cos \theta \end{pmatrix}
# \begin{pmatrix} \nu_1 \\ \nu_2 \end{pmatrix}  
# $$
# 
# The neutrino time evolution is:
# 
# $$
# | \nu (t) \rangle = \cos \theta e^{-iE_1 t}| \nu_1 \rangle + \sin \theta e^{-iE_2 t}| \nu_2 \rangle 
# $$
# 
# The Amplitude to detect a $\nu_\beta$ neutrino at time $t$ is (except for a global phase):
# 
# $$
# \mathcal{A}_{\alpha \beta} = \langle \nu_\beta | \nu(t) \rangle
#  = \cos \theta \sin \theta \, (e^{-i E_2 t} - e^{-i E_1 t}) = \sin 2 \theta \sin  \frac{E_2 - E_1}{2} t
# $$
# 
# If we approximate $E_i \simeq p + \frac{m^2_i}{2 p}$ and with $t \to L$
# 
# $$
# \frac{E_2 - E_1}{2} t = \frac{m^2_2 - m^2_1}{4 E} t \to \frac{\Delta m^2_{21} L}{4 E}
# $$
# 
# 

# 
# Manipulating:
# 
# $$
# \mathcal{A}_{\alpha \beta}(t) =
# \cos \theta \sin \theta \, e^{-i \frac{E_2 + E_1}{2}}(e^{-i \frac{E_2 - E_1}{2} t} - e^{-i \frac{E_1 - E_2}{2}t})
# $$
# 
# $$
# \mathcal{A}_{\alpha \beta}(t) = - i e ^{-i \frac{E_2 + E_1}{2}}\sin 2 \theta \sin \frac{E_2 - E_1}{2} t
# $$
# 
# 
# 
# 

# The oscillation probability is:
# 
# $$
# \mathcal{P}(\nu_\alpha \to \nu_\beta) = \sin^2 2 \theta \sin^2 \frac{\Delta m^2_{21} L}{4 E} 
# $$
# 
# Where $\Delta m^2_{21} = m^2_2 - m^2_1$.
# 
# This probability depends on two nature parameters:
# 
#   - mixing angle, $\theta$. If $\theta = 0$ there is no oscillations. It controls the amplitude.
# 
#   - $\Delta m^2_{21}$. If neutrinos are mass degenerated, there is no oscillation. It controls the phase.
#  
# 

# The oscillation depends on the relation between $\Delta m^2$ and $E/L$.
# 
# The oscillation length.
# 
# $$
# L_{osc} = \pi \frac{2 E}{\Delta m^2}
# $$
# 

# As the observation is at fixed $L$ the dependence of the probability with $E$ of the initial $\nu$ flux is relevant.
# 
# We consider 3 cases:
# 
#   - At the first maximum of the oscillation.
# 
#   $$
#   \frac{\Delta m^2 L}{4E_{\mathrm{max}}} = \frac{\pi}{2} \to E_{\mathrm{max}} = \frac{\Delta m^2 L}{2\pi}
#   $$
# 
#   We observe the oscillation behavior as a function of $E$. 

# We can write the phase and the probability in the relevant units:
# 
# $$
# \frac{\Delta m^2 L }{4 E} = 1.27 \frac{\Delta m^2}{\mathrm{eV^2}} \frac{L}{\mathrm{km}} \frac{\mathrm{GeV}}{E}
# $$
# 
# $$
# \mathcal{P}(\nu_\alpha \to \nu_\beta) = \sin^2(2 \theta) \, \sin^2 \left(1.27 \frac{\Delta m^2}{\mathrm{eV^2}} \frac{L}{\mathrm{km}} \frac{\mathrm{GeV}}{E} \right)
# $$
# 
# **Question:** Draw the oscillation probability for $\Delta m^2 = 2.5 \times 10^{-3} \, \mathrm{eV}^{2}$ and $\theta = \pi/4$ as a function of $L/E$

# In[3]:


import oscillations

oscillations.plot_posc_intro()


#   - $E/L \ll \Delta m^2$ the oscillation is fast. We observe the average. 
#   
#   $$
#   \mathcal{P}(\nu_\alpha \to \nu_\beta) \simeq \frac{1}{2} \sin^2 2\theta
#   $$  
# 
#   In this case the probability does not depend on $\Delta m^2$ only on $\theta$. 
# 
#   It corresponds to the case where neutrinos are decoupled! It does not show the characteristic signature of neutrino oscillations.
# 
# 
#   - $E/L \gg \Delta m^2$ Oscillation is too small in the detector. 
#   
#   The probability is suppressed:
# 
#   $$
#   \mathcal{P}(\nu_\alpha \to \nu_\beta) = \left(\frac{\Delta m^2 L}{4E} \right)^2 \, \sin^2 2\theta 
#   $$
# 
#   And we can not separate the two phisical parameters $\Delta m^2$ and $\theta$.
# 

# 
# | |
# | :--: |
# | <img src="./imgs/dm2_ranges.png" width=600 align="center">|
# 
# Table with the energy, distance and $\Delta m^2$ ranges accessible by experiments.
# 
# Nature has been kind to us with solar and atmospheric neutrinos!
# 
# With man-made neutrinos we can have Long and Short Base Line (LBL, SBL) experiments.
# 
# **Question:** Discuss the implications that the probability is invariance respect: $\pm \Delta m^2_{21}$ and $\theta \to \pi/2 - \theta$.

# ### Interactive oscillation: two families
# 
# Desliza los controles para explorar cómo cambia
# $\mathcal{P}(\nu_\alpha \to \nu_\beta)$ con el ángulo de mezcla $\theta$, la diferencia de masas $\Delta m^2$
# y la distancia de propagación $L$.
# La gráfica izquierda fija $L$ y varía $E$; la derecha muestra $P$ vs $L/E$.

# In[4]:


import oscillations

oscillations.plot_2fam_interactive()


# **Exercise: Oscillation length and experimental regimes**
# 
# The two-family oscillation probability depends only on the ratio $\Delta m^2 L/E$ through the phase:
# $$
# \phi = 1.267\,\frac{\Delta m^2\,[\mathrm{eV}^2]\; L\,[\mathrm{km}]}{E\,[\mathrm{GeV}]}
# $$
# The oscillation length $L_{\rm osc} = (\pi/1.267)\, E/\Delta m^2$ sets the natural scale.
# 
# **Questions:**
# 
# 1. Compute $L_{\rm osc}$ for the atmospheric ($\Delta m^2_{31} = 2.5\times10^{-3}$ eV$^2$) and solar ($\Delta m^2_{21} = 7.5\times10^{-5}$ eV$^2$) scales at $E = 1$ GeV and at $E = 3$ MeV. Which scale is probed by SuperKamiokande ($L\sim 10$--$10\,000$ km, $E\sim 1$ GeV) and which by KamLAND ($L = 180$ km, $E\sim 3$ MeV)?
# 
# 2. Three detection regimes exist depending on $L$ relative to $L_{\rm osc}$:
#    (a) $L \ll L_{\rm osc}$: no visible oscillation, $P \approx 1$;
#    (b) $L \sim L_{\rm osc}$: oscillation resolved;
#    (c) $L \gg L_{\rm osc}$: rapid oscillations average to $\langle P \rangle = 1 - \frac{1}{2}\sin^2 2\theta$.
#    Using `oscillations.posc_2fam`, plot $P$ vs $L/E$ over $10^0$--$10^6$ km/GeV for solar parameters ($\Delta m^2_{21}$, $\theta_{12}$) and identify each regime.
# 
# 3. The table in the notebook lists energy, distance and target $\Delta m^2$ for each experiment. Verify numerically that SK-atm, KamLAND and Daya Bay are each designed so that $\phi \simeq \pi/2$ (first oscillation maximum) at their respective peak energies.

# In[5]:


import oscillations

oscillations.exercise_osc_regimes()


# **Exercise: Exclusion region in the ($\Delta m^2$,  $\sin^2 2\theta$) plane**
# 
# Consider an academic $\bar\nu_e$ disappearance experiment: reactor source at $L = 250$ m, effective detection energy $E_{\rm eff} = 3$ MeV. No oscillation signal is observed. The result is an upper limit on the disappearance probability:
# $$
# P_{\rm osc}(\theta,\Delta m^2) = \sin^2 2\theta \;\sin^2\!\left(1.267\,\frac{\Delta m^2\,[\mathrm{eV}^2]\; L\,[\mathrm{km}]}{E\,[\mathrm{GeV}]}\right) \;\leq\; P_{\rm lim} = 0.05 \quad (90\%\ \mathrm{CL})
# $$
# The **excluded region** is the set $\{(\Delta m^2,\theta)\,:\, P_{\rm osc} > P_{\rm lim}\}$.
# 
# **Questions:**
# 
# 1. **Two asymptotic regimes.** Analyze the shape of the exclusion boundary $P_{\rm osc} = P_{\rm lim}$:
#    (a) *Large $\Delta m^2$* ($\phi \gg 1$): using $\langle\sin^2\phi\rangle = 1/2$, show the boundary is a **vertical line** at $\sin^2 2\theta = 2\,P_{\rm lim}$. What is the minimum mixing angle excluded for any $\Delta m^2$?
#    (b) *Small $\Delta m^2$* ($\phi \ll 1$): using $\sin\phi \approx \phi$, show the boundary satisfies $\sin^2 2\theta \propto (\Delta m^2)^{-2}$, i.e. **slope $-2$** in the log--log plane. Derive the minimum $\Delta m^2$ detectable at $\sin^2 2\theta = 1$.
# 
# 2. **Sensitivity reach as a function of $L/E$.** The minimum detectable $\Delta m^2$ at maximum mixing scales as $\Delta m^2_{\rm min} \approx \sqrt{P_{\rm lim}}/(1.267\,L/E)$. Compute $\Delta m^2_{\rm min}$ for Bugey ($L=15$ m, $E=3$ MeV), CHOOZ ($L=1$ km, $E=3$ MeV) and KamLAND ($L=180$ km, $E=3$ MeV). Identify which $\Delta m^2$ scale each experiment targeted.
# 
# 3. **Numerical exclusion contour.** Write a function `exclusion_contour(L_km, E_GeV, P_lim)` that returns the boundary curve $\sin^2 2\theta_{\rm max}(\Delta m^2)$ using `oscillations.posc_2fam`. Plot the excluded region (shaded) in the $(\sin^2 2\theta, \Delta m^2)$ plane on a log--log scale for the academic experiment, CHOOZ and KamLAND. Identify the asymptotic regimes and verify the theoretical slopes.

# In[6]:


import oscillations

oscillations.exercise_exclusion_contour()


# ## Solar Neutrinos
# 
# | |
# | :--: |
# | <img src="./imgs/davis_experiment.png" width=300 align="center">|

# 
#  - In 1946 Pontecorvo proposed the detection of $\nu_e$ via $\nu_e + ^{37}\mathrm{Cl} \to ^{37} \mathrm{Ar} + e$, with 814 keV threshold
#  
#  - In 1962 J. Bahcall calculated the Solar Model [5]
#  
#  - In 1964 R. Davis at Homestake experiment [6]
#  
#     - 615 tons of $\mathrm{Cl}_2\mathrm{CH}_{14}$, 1 $\nu_e$ per day
#     
#     - deep underground (1600 m Homestake mine)
#     
#     - Filtered the tank and chemical processing, $\tau_{1/2} = 34.8$ d
#     
#  - Solar Neutrino Problem 
#  
#  - Nobel prize Davis and Koshiba in 2002.
# 
# 

# ### Solar Model
# 
# | |
# | :--: |
# | <img src="./imgs/solar_chains.png" width=600 align="center">|
# 
# - During 1964-2005 J. Bahcall et al elaborated the Standard Solar Model, SSM.
# 
# - They predict a neutrino flux for different reaction detections: $pp, \mathrm{Be}, \mathrm{B}$
# 

# 
# | |
# | :--: |
# | <img src="./imgs/solar_flux.png" width=600 align="center">|
# 
# - The detector techniques are sensitive to different energy ranges. Galium: $\nu_e + ^{71}\mathrm{Ga} \to ^{71}\mathrm{Ge} + e$, (threshold at 233 keV)
# 

# | |
# | :--: |
# | <img src="./imgs/solar_experiments.png" width=800 align="center">|
# 

# | |
# | :--: |
# | <img src="./imgs/solar_deficit_results.png" width=600 align="center"> |
# 
# - Homestake reported (1979-1994) 1/3 of the solar flux (SNU, 1 capture in $10^{36}$ atoms) [7], to be compared with SSM latest predictions [8]
# 
# - Different solar deficit depending on the energy range. [9, 10, 11, 12]
# 
# - These experiments have not directionality neither time information!

# ### [(Super)Kamiokande experiment](http://www-sk.icrr.u-tokyo.ac.jp/index-e.html)
# 
# 
# | | |
# | :--: | :--: |
# |<img src="./imgs/SK_drawing.png" width=300 align="center"> | <img src="./imgs/SK_photo.png" width=300 align="center">|
# 

# - Kamiokande-I, II (1987-1995) was 3 kton water detector, in Kamioka mine. 
# 
# - SuperKamiokande (1996-), 50 kton
# 
# - 50 cm PMTs to detect Cherenkov light. SuperKamiokande 11000 PMTs
# 
# - Neutrino elastic scattering (ES), $\nu_x \, e^- \to \nu_x \, e$ (CC and NC).

# 
# | |
# | :--: |
# | <img src="./imgs/SK_solar_flux.png" width=500 align="center">|
# 
# - Correlation with the Solar direction. SK is a neutrino telescope!
# 
# - Less solar (B) flux than predicted: $(2.34 \pm 0.04) \times 10^6 \, \mathrm{cm}^{-2}\mathrm{s}^{-1}$ [[13]](https://arxiv.org/abs/1606.07538), with SSM $(5.46 \pm 0.66) \times 10^6 \, \mathrm{cm}^{-2}\mathrm{s}^{-1}$ [14]
# 
# 
# 

# ### [SNO experiment](https://sno.phy.queensu.ca)
# 
# 
# | |
# | :--: |
# | <img src="./imgs/SNO_photo.png" width=500 align="center">|
# 
#  - Sudbury Neutrino Observatory 1999-2006, Canada.
#  - 1 kton ultra-pure heavy water $\mathrm{D}_2\mathrm{O}$
#  - Spherical acrylic vessel 12 m diameter. 9500 PMTs
#  - Shield Water and 2000 m underground 
#  

# 
# | |
# | :--: |
# | <img src="./imgs/SNO_event.png" width=500 align="center">|
# 
#  - Measure Cherenkov light
#  - ES (NC + CC) $\nu_x \, e \to \nu_x\, e$ (contributions: $\nu_\tau, \nu_\mu = 0.15 \nu_e$)
#  - $\nu_e$ CC in $D_2O$: $\nu_e \, d \to e^- \, \, p \,\, p$ ($E_{thr} > 5$ MeV)
#  - $\nu_x$ NC: $\nu_x \, d \to \nu_x \, p \, n$, then $n \, H^2 \to H^3 \, \gamma$ ($E_{thr} > 2.2$ MeV)

# 
# | |
# | :--: |
# | <img src="./imgs/SNO_results.png" width=600 align="center">|
# 
# SNO results [[16]](https://arxiv.org/abs/nucl-ex/0502021)
# 

# 
# - In 2001, SNO reported the initial result of CC measurement [15], was an evidence of non-$\nu_e$ flux. In 2004 solar neutrino NC [16]
# 
# - From a combined result of three phases of SNO [17], the total flux of 8B solar neutrino is 
# $(5.25 \pm 0.16^{+0.11}_{-0.14})$ $\mathrm{cm}^{-2}\mathrm{s}^{-1}$, consistent with the SSM.
# 
# - Several solutions where possible combining all Solar experiment. 
# 
# - The solution of solar neutrino problem was the so-called large mixing angle (LMA) with parameters $\Delta m^2_{21} = 7.5 \times 10^{-5} \, \mathrm{eV}^2$ and $\sin^2 \theta \simeq 0.3$.
# 

# 
# | |
# | :--: |
# | <img src="./imgs/solar_dm2_tan2theta.png" width=500 align="center">|

# ### Neutrino oscillations in matter
# 
# | |
# | :--: |
# | <img src="./imgs/matter_effects_diagram.png" width=500 align="center">|
# 
# Neutrinos can *coherently scatter* via NC in matter, but only $\nu_e$ can CC! (Mikheyev, Smirnov effect [20])
# 
# The interaction will be related with the *density of the electrons* in the media, $n_e$, and $G_F$.
# 
# We model this interaction with a potential $V_e = \sqrt{2} G_F N_e$ 
# 
# 

# Where:
#     
# $$
# V_e \simeq 8 \times 10^{-14} \, f_e \, \rho  \;\; \left(\mathrm{eV \, cm}^3/\mathrm{g}\right)
# $$
# 
# with $f_e$ the fraction of electrons over nucleons
# 
# $$
# f_e = \frac{n_e}{n_p + n_n}
# $$
# 
# 
# 
# In the Sun, for $E_\nu \sim 1$ MeV, $f_e \sim 1/2$, $\rho_{\mathrm{Sun}} \sim 100 \, \mathrm{g/cm}^3$. In the Earth,  $\rho_{\mathrm{Earth}} \sim 10 \, \mathrm{g/cm}^3$,
# 
# 
# $$
# V^{\mathrm{Sun}}_e \sim 10 ^{-12} \; \mathrm{eV}, \;\;\;
# V^{\mathrm{Earth}}_e \sim 10^{-13} \, \mathrm{eV}
# $$

# In terms of the electron density, $n_e$:
# 
# $$
# V_e = 1.256 \times 10^{-37} n_e \, \left(\mathrm{eV} \, \mathrm{cm}^{3}\right)
# $$
# 
# We can associate a typical *length* to this effect:
# 
# $$
# L = \frac{2\pi}{V_e} = 9.8 \times 10^{32}  \left(\frac{\mathrm{cm}^{3}}{n_e}\right) \, \mathrm{cm}
# $$

# The neutrinos propagation in matter can be described by quantum mechanics, via a modified Dirac equation or QFT, with the same result. 
# 
# We present here the propagation in quantum mechanics, using the Schrödinger equation.
# 
# 
# The Hamiltonian of the neutrino propagation is:
# 
# $$
# \mathcal{H} = U \mathcal{H_0}  U^T+ V_e
# $$
# 
# 
# Where $\mathcal{H_0}$ is the hamiltonian and $U$ is the mix matrix in vacuum, and $V_e$ affects only to $\nu_e$.
# 

# In two family and $H_0$, the free hamiltonian can be re-written removing terms which are proportional to the identity matrix, which results in a global phase.
# 
# $$
# \mathcal{H}_0 = \begin{pmatrix} E_1 & 0 \\ 0 & E_2 \end{pmatrix} \to 
#  p + \frac{1}{2p}  \begin{pmatrix} m^2_1 & 0 \\ 0 & m^2_2 \end{pmatrix} \to
#  \frac{1}{2E}\begin{pmatrix} m^2_1 & 0 \\ 0 & m^2_2 \end{pmatrix} \to 
#  \frac{1}{4E} \begin{pmatrix}  - \Delta m^2_{21}& 0 \\ 0 & \Delta m^2_{21} \end{pmatrix} 
# $$
# 
# With $\Delta m^2_{21} = m^2_2 - m^2_1$
# 
# Introducing the mixing matrix:
# 
# $$
# U = \begin{pmatrix} \cos \theta_0 & \sin \theta_0 \\ - \sin \theta_0 & \cos \theta_0 \end{pmatrix}
# $$
# 
# 
# It ends:
# 
# $$
# U  \mathcal{H}_0 U^T = \frac{\Delta m^2_{21}}{4E}\begin{pmatrix} 
# -\cos 2\theta_0 & \sin 2 \theta_0 \\
# \sin 2\theta_0 &   \cos 2 \theta_0 \\
# \end{pmatrix}
# $$
# 
# 
#     

# From here we obtain the time evolution:
# 
# $$
# i \frac{\partial \mathcal{A}_{\alpha}}{\partial t} = U \mathcal{H} U^T \mathcal{A}_{\alpha} 
# $$
# 
# where $\alpha$ indicates neutrinos in flavour basis.
# 
# And the oscillation probability formula:
# 
# The effective propagation probability:
# 
# $$
# \mathcal{P}(\nu_\alpha \to \nu_\beta) = \sin^2 2 \theta_0 \, \sin^2 \frac{\Delta m^2_{21} L}{ 4 E}
# $$
# 
# 

# The part of the hamiltonian in matter, we can also re-write it as:
# 
# $$
# V_e = \begin{pmatrix} V_e & 0 \\ 0 & 0 \end{pmatrix} \to
#  \begin{pmatrix} V_e/2 & 0 \\ 0 & -V_e/2 \end{pmatrix} 
# $$
# 
# The total hamiltonian is:
# 
# $$
# \mathcal{H} = U \mathcal{H}_0 U^T + V_e  = \frac{\Delta m^2_{21}}{4E}
# \begin{pmatrix} 
# - (\cos 2\theta_0 - x) &  \sin 2 \theta_0 \\
#  \sin 2\theta_0 &   (\cos 2 \theta_0  - x)\\
# \end{pmatrix}
# $$
# 
# with: 
# 
# $$
# x = \frac{2 E V_e} {\Delta m^2_{21}}
# $$
# 
# 

# We can re-write $H$, 
# 
# $$
# U  \mathcal{H} U^T = \frac{\Delta m^2_m}{4E}\begin{pmatrix} 
# -\cos 2\theta_m & \sin 2 \theta_m \\
# \sin 2\theta_m &   \cos 2 \theta_m \\
# \end{pmatrix}
# $$
# 
# we encounter that neutrino propagation in matter has the same behavior, oscillates, but with the effective parameters:
# 
# $$
# \tan 2 \theta_m = \frac{\sin 2 \theta_0}{\cos 2 \theta_0 - x}
# $$
# 
# $$
# \Delta m^{2}_{m}= \Delta m^{2}_{21} \sqrt{(\cos 2 \theta_0 - x)^2 + \sin^2 2 \theta_0}
# $$
# 

# The effective propagation probability:
# 
# $$
# \mathcal{P}(\nu_\alpha \to \nu_\beta) = \sin^2 2 \theta_m \, \sin^2 \frac{\Delta m^2_{m} L}{ 4 E}
# $$
# 
# **Questions:** Check that they correspond to the eigen-values of $\mathcal{H}$ and the rotation angle between eigen and flavour states.
# 

# Different scenarios:
# 
#  *  Vacuum limit,  $x \ll \cos 2 \theta_0, \; N_e = 0$, we recover: $\theta_0, \, \Delta m^2_{21}$
# 
# 
#  *  Matter domination. In the case $x >> \cos 2 \theta_0$:
# 
# $$
# \tan 2 \theta_m \to 0^-, \; \theta_m = \pi/2
# $$
# 
# With $\theta_m = \pi/2$ we have:
#    
# $$
# U(\theta_m) = \begin{pmatrix} 0 & 1  \\ -1 & 0 \end{pmatrix}
# $$
# 
# Therefore $\nu_e \leftrightarrow \nu_2$ and  $\nu_\mu \leftrightarrow \nu_1$. 
# 
# Neutrinos of a given flavour has a unique mass states and they do not oscillate!
# 

# 
#  *  Resonance condition. When increasing the density for a $E \sim 0.5$ MeV, a resonance condition may happen:
# 
# $$
# \cos 2 \theta_0 = \frac{2 E}{\Delta m^2_{21}} V_e 
# $$
# 
# 
# In this case the effective:
# 
# $$
# \Delta m^2_m = \Delta m^2_{21}  \sin 2 \theta_0, \; \tan 2 \theta_m  = \infty, \theta_m = \pi/4 
# $$
# 
# That condition requires:
# $$
# \Delta m^2_{21} \cos 2 \theta_0 > 0
# $$
# 
# That is:
# 
# $$
# |\theta_0| < \pi/4 \Rightarrow \Delta m^2_{21} > 0
# $$
# 

# In[7]:


import oscillations

oscillations.plot_msw_parameters()


# When the matter density varies along propagation, the solutions are more complicated (numerical solutions).
# 
# But if the variation of density is slow, and the initial scenario is matter dominated, as for neutrinos in the core of the sun, $\nu_2$, they are in the same state.
# 
# In the Sun:
# 
# $$
# N_e(t) = N_e(0) e^{-r/R}
# $$
# 
# where $r$ is the radius respect the origin

# 
# Above the resonance, neutrinos in the Sun, are $\nu_2$, escape as $\nu_2$ and interacts in the Earth.
# 
# 
# Therefore the probability that a $\nu_2$ interacts as $\nu_e$, when it is back in vacuum:
# 
# $$
# \mathcal{P}(\nu_e \to \nu_e) = \sin^2 \theta_0
# $$
# 
# If the energy of the neutrino is bellow the resonance, in the vacuum dominated, and the detector size (Earth), large compared with the oscillations, the probability is averaged:
# 
# $$
# \mathcal{P}(\nu_e \to \nu_e) = 1 - \frac{1}{2}\sin^2 2 \theta_0
# $$
# 
# 

# 
# | |
# | :--: |
# | <img src="./imgs/solar_matter.png" width=500 align="center">|

# 
# | |
# | :--: |
# | <img src="./imgs/borexino_results.png" width=500 align="center">|
# 
# - [Borexino](http://borex.lngs.infn.it) at Gran Sasso, Italy, is an ultra-pure 200 t LS detector, 0.18 MeV threshold and 5% energy resolution
# [[29]](https://arxiv.org/abs/0808.2868)
# 
# - Survival probability depends on the neutrino energy range
# 

# In[8]:


import oscillations

oscillations.exercise_solar_lma_check()


# **Exercise: The MSW resonance in the Sun**
# 
# The effective mixing angle in matter is enhanced at the resonance energy:
# $$
# E_{\rm res} = \frac{\Delta m^2 \cos 2\theta_0}{2\,V_e}, \qquad V_e \simeq 7.63\times10^{-14}\,Y_e\,\rho\;[\mathrm{g/cm}^3]\;\mathrm{eV}
# $$
# At resonance ($E = E_{\rm res}$), $\theta_m = 45°$ regardless of the vacuum angle. For neutrinos produced above the resonance and propagating adiabatically, the exit state is $|\nu_2\rangle$ and the survival probability simplifies to $P(\nu_e\to\nu_e) \simeq \sin^2\theta_{12}$.
# 
# **Questions:**
# 
# 1. Using the solar core density $\rho \simeq 150$ g/cm$^3$ and $Y_e \simeq 0.5$, compute $V_e$ in eV and the resonance energy $E_{\rm res}$ for the solar parameters ($\Delta m^2_{21}$, $\theta_{12}$). Do the $^8$B solar neutrinos ($E\sim 5$--15 MeV) sit above or below the resonance?
# 
# 2. The adiabatic MSW prediction for $^8$B neutrinos is $P_{ee} \simeq \sin^2\theta_{12} \approx 0.31$. Compare with the SNO CC/NC ratio ($P_{ee} \simeq 0.34\pm0.04$) and the Borexino measurement at $\sim 7$ MeV. Is the agreement consistent with the adiabatic limit?
# 
# 3. Using `oscillations.posc_matter_2fam`, plot $P(\nu_e\to\nu_e)$ vs $E$ from 0.1 to 20 MeV at $L = 1.496\times10^8$ km (Sun--Earth distance) for $\rho = 0$ (vacuum) and $\rho = 100$ g/cm$^3$ (average solar density). Identify the resonance transition and the approach to the adiabatic limit at high energy.

# In[9]:


import oscillations

oscillations.exercise_msw_resonance()


# ### [KamLand Experiment](https://www.awa.tohoku.ac.jp/kamlande/)
# 
# If $\Delta m^2_{21} \simeq 7.5 \times 10^{-5}$ eV$^2$, oscillation can be observed with reactors neutrinos at LBS, $\mathcal{O}(100)$ km and E $\mathcal{O}(1)$ MeV.
# 
# <img src="./imgs/KamLAND_map.png" width=400 align="center">
# 
# 

# 
# <img src="./imgs/KamLAND_drawing.png" width=500 align="center">
# 

# <img src="./imgs/KamLAND_photo.png" width=500 align="center">

# - 1 k ton ultra-pure liquid-scintillator in 13 m spherical balloon.
# 
# - flux $\bar{\nu}_e$ from 55 reactors in Japan and South Korea, $L \sim 180$ km
# 
# - Detection: e+ scintillation and annihilation, and $n$ capture in H, 2.2 MeV $\gamma$ delayed 210 $\mu$s
# 
# - In Kamioka mine. First results in 2002 confirming $\bar{\nu}_e$ disappearance [[21]](https://arxiv.org/abs/hep-ex/0212021) 
# 
#     - ratio observed/expected with no oscillations: $0.611 \pm 0.085 \pm 0.041$
# 

# 
# 
# <img src="./imgs/KamLAND_oscillation.png" width=500 align="center">
# 
# - Confirmation of oscillation pattern in 2004 [[22]](https://arxiv.org/abs/hep-ex/0406035) and [[23]](https://arxiv.org/abs/1303.4667)

# 
# <img src="./imgs/KamLAND_dm2_tan2theta12.png" width=500 align="center">
# 
# - Confirmation of oscillation pattern in 2004 [[22]](https://arxiv.org/abs/hep-ex/0406035) [[23]](https://arxiv.org/abs/1303.4667)

# **Exercise: KamLAND and the measurement of $\theta_{12}$**
# 
# KamLAND measures $\bar\nu_e$ disappearance from Japanese reactors at $\langle L\rangle \simeq 180$ km with $E_{\bar\nu}\sim 1$--8 MeV. In the two-family approximation (atmospheric phases averaged out), the survival probability is:
# $$
# P(\bar\nu_e \to \bar\nu_e) \approx 1 - \sin^2 2\theta_{12} \sin^2\!\left(1.267\,\frac{\Delta m^2_{21} L}{E}\right) - \frac{1}{2}\sin^2 2\theta_{13}
# $$
# 
# **Questions:**
# 
# 1. Compute the oscillation phase $\phi_{21} = 1.267\,\Delta m^2_{21} L/E$ at $L = 180$ km for $E = 3$, 5 and 7 MeV. At which energy is $\phi_{21} = \pi/2$ (first oscillation minimum)? Compare with the KamLAND oscillation figure above.
# 
# 2. The KamLAND measured ratio (corrected for $\theta_{13}$) is $R \approx 0.498\pm0.044$. Using $\langle\sin^2\phi_{21}\rangle = 1/2$ (energy-averaged), extract $\sin^2 2\theta_{12}$ and compare with NuFit-6.0.
# 
# 3. Using `oscillations.posc_2fam`, plot $P$ vs $L/E$ for KamLAND parameters ($\Delta m^2_{21}$, $\theta_{12}$) over $L/E \in [10^1, 10^4]$ km/GeV. Mark the KamLAND operating point ($L/E \approx 180\,\mathrm{km}/3\,\mathrm{MeV} = 6\times10^4$ km/GeV). Is the experiment in the resolved-oscillation or averaged regime?

# In[10]:


import oscillations

oscillations.exercise_kamland()


# ### Solution of Solar Neutrino Problem 
# 
# The MSW adiabatic flavour transitions in the solar matter, the so-called large mixing angle (LMA) with parameters:
# 
# $$
# \Delta m^2_{21} = 7.5 \times 10^{-5} \;\, \mathrm{eV}^2,  \;\; \sin^2 \theta \simeq 0.3
# $$
# 
# Confirmed total SSM $\nu$ flux with SNO+ NC data.
# 
# Confirmed oscillation with reactor neutrinos LBL KamLAND experiment.

# ## Atmospheric neutrinos
# 
# - Neutrinos are produced in the cascades generated by cosmic rays impinging on the atmosphere
# 
# - For every $\pi$ there are 2 $\nu_\mu$ and 1 $\nu_e$ ($\nu$ and $\bar{\nu}$)
# 
# - Range of Energy is very large: 0.1-100 GeV
# 
# - Range of distance, L, is also large: 10 - 100 km
# 
# - Flux depends on Energy
# 
# - SuperKamiokande can detect $\nu_e, \nu_\mu$ via inverse decay.

# ### SuperKamiokande

# <img src="./imgs/SK_mue_atm_event.png" width=800 align="center">

# - Lepton direction indicates $\nu$ direction
# 
# - There are fully contained, stopping, upward, through-going muons different E ranges
# 
# - In 1998 SK observed a deficit of up-going muons [[24]](https://arxiv.org/abs/hep-ex/9807003)

# <img src="./imgs/SK_atm_first_result.png" width=500 align="center">
# 
# Up/Down assymmetry:
# $$
# \mathcal{A}_{UD}= 0.296 \pm 0.048 \pm 0.001,
# $$
# 

# 
# <img src="./imgs/SK_atm_LE_result.png" width=500 align="center">
# 
# 
# Evidence of oscillation pattern [[25]](https://arxiv.org/abs/hep-ex/0404034)
# 

# 
# <img src="./imgs/SK_atm_LE_dm2theta.png" width=500 align="center">
# 
# 
# Best parameters [[25]](https://arxiv.org/abs/hep-ex/0404034)
# 
# $$
# \Delta m^2 \simeq 2.5 \times 10^{-3} \;\; \mathrm{eV}^2 , \;\; \theta \simeq \pi/4
# $$

# <img src="./imgs/SK_zenith_distributions.png" width=800 align="center">

# **Exercise: SuperKamiokande and the Up/Down asymmetry**
# 
# Atmospheric neutrinos from cosmic-ray showers are produced at height $h \simeq 20$ km. The path length to SK depends on the zenith angle $\theta_z$:
# $$
# L(\theta_z) = \sqrt{(R_E\cos\theta_z)^2 + 2R_E h} \;-\; R_E\cos\theta_z
# $$
# with $R_E = 6371$ km. Downward ($\theta_z = 0°$): $L \sim h$. Upward ($\theta_z = 180°$): $L \sim 2R_E \approx 12\,756$ km.
# 
# **Questions:**
# 
# 1. Compute $L$ for $\theta_z = 0°$, $90°$ and $180°$. For $E_\nu = 1$ GeV and $\Delta m^2_{32} = 2.5\times10^{-3}$ eV$^2$, compute $\phi_{32}$ for each direction. Which zenith range has $\phi_{32}\sim\pi/2$?
# 
# 2. The SK Up/Down asymmetry for multi-GeV $\mu$-like events is $\mathcal{A}_{UD} = (U-D)/(U+D) = -0.296\pm0.048$. In the two-family approximation, $\mathcal{A}_{UD} \approx -\sin^2(2\theta_{23})\,\langle\sin^2\phi\rangle$ with $\langle\sin^2\phi\rangle \approx 0.5$ for the upward sample. Extract $\sin^2(2\theta_{23})$ and compare with NuFit-6.0.
# 
# 3. Plot $L(\theta_z)$ and $P(\nu_\mu\to\nu_\mu)$ vs $\cos\theta_z\in[-1,+1]$ for $E = 1$ GeV using `oscillations.posc_2fam` with atmospheric parameters. Overlay the no-oscillation line and identify the zenith region of maximum sensitivity.

# In[11]:


import oscillations

oscillations.exercise_sk_asymmetry()


# ### Confirmation of Atmospheric oscillations
# 
# <img src="./imgs/LBS_experiments.png" width=800 align="center">
# 
# 
# The atmospheric $\Delta m^2 \simeq 2.5 \times 10^{-3}$ eV$^2$ is accessible with accelerator $\nu_\mu$ neutrinos of $\mathcal{O}(1)$ GeV at $\mathcal{O}(1000)$ km.

# ### T2K
# 
# - [T2K](https://t2k-experiment.org) 0.6 GeV $\nu_\mu (\bar{\nu}_\mu)$ from JPARC to SK at 290 km.
# - T2K exploits the fact that the neutrino spectrum is narrower (but less intense) off-axis by 2.5$^o$ degrees.
# - A near and a far detector to estimate flux and control systematic errors.
# - Far Detector is SuperKamiokande

# 
# <img src="./imgs/T2K_map.png" width=800 align="center">

# <img src="./imgs/T2K_beam_offaxis.png" width=300 align="center">
# 
# Neutrino flux for different off-axis angles.

# 
# | |
# | :--: |
# | <img src="./imgs/T2K_numu_dis.png" width=800 align="center">|
# 
# - Measure $\nu_\mu$ disappearance (relevant for atmospheric oscillations)
# - Measure $\nu_e$ appearance (see later)
# - Beam with $\bar{\nu}_\mu$ and $\nu_\mu$ (relevant for CP and matter effects, see later)

# ### MINOS
# 
# | | | |
# | :--: | :--: | :--: |
# |<img src="./imgs/minos_map.png" width=400 align="center">|<img src="./imgs/minos_photo.png" width=400 align="center">| <img src="./imgs/minos_event.png" width=400 align="center">|

# 
# - [MINOS](https://www-numi.fnal.gov) experiment at Soudan Mine, 730 km from FermiLab NuMI beam peak at 3 GeV 
# 
# - 5.4 ktons iron-scintillation tracking planes and calorimeter (486 planes, 30 m long, 8 m high)
# 
# - similar near detector 0.9 ton
# 
# - magnetized iron-tracking, separation $\mu^+/\mu^-$
# 
# 

# 
# | |
# | :--: |
# | <img src="./imgs/MINOS_numu_atmconf.png" width=500 align="center">|
# 
# $$
# |\Delta m^2| = (2.41\pm 0.09 ) \times 10^{−3} \mathrm{eV}^2, \;\; \sin^2 2θ = 0.950 \pm 0.035
# $$
# 
# MINOS (2013) [[26]](https://arxiv.org/abs/1304.6335)
# 

# 
# | |
# | :--: |
# | <img src="./imgs/MINOS_numu_atmconf2.png" width=500 align="center">|
# 
# oscillation probability from MINOS+ (2014)

# ## Oscillations with 3 neutrinos
# 
# | | 
# | :--: |
# | <img src="./imgs/nuosc_3families_diagram.png" width=400 align="center">|
# 
# 
# 
# The neutrino oscillation with $n$ families is ruled by a complex *unitary* matrix 
# 
# $$
# | \nu_{\alpha} \rangle = \sum_i U^*_{\alpha i} | \nu_i \rangle
# $$
# 
# With *vacuum* plane wave propagation the oscillation amplitude:
# 
# $$
# \mathcal{A}_{\alpha \beta}  = \langle \nu_{\beta} | \nu_{\alpha} (t) \rangle 
#  = \sum_i U_{\beta i} U^*_{\alpha i} \, e^{-i E_i t} 
# $$

# | |
# | :--: |
# | <img src="./imgs/U_PMNS.png" width=800 align="center">|
# 
# Where $s_{13} = \sin \theta_{13}, \; c_{13} = \cos \theta_{13}$. The CP phase is $\delta$.
# 
# Solar oscillations amplitude is ruled by $\theta_{12}$ while atmospheric by $\theta_{23}$.
# 
# *Notice*: If neutrinos are Majorana there are two phases more (see next chapter)
# 

# | |
# | :--: |
# |<img src="./imgs/U_PMNS_alphai.png" width=800 align="center">|

# From the amplitude:
# 
# $$
# \mathcal{A}_{\alpha \beta}  = \delta_{\alpha \beta} -2i \sum_i U_{\beta i} U^*_{\alpha i} e^{-i \frac{\Delta E_{ip}}{2}} \sin \frac{\Delta E_{ip} t}{2}   
# $$
# 
# We obtain the oscillation probability:
# 
# $$
# \mathcal{P}(\nu_\alpha \to \nu_\beta)  = \delta_{\alpha \beta} - 4  \sum_i |U_{\alpha i}|^2   \delta_{\alpha \beta} \sin^2 \frac{\Delta E_{ip}t}{2} \\
# 	 + 4 \sum_{i,j} \mathrm{Re}\left[ U_{\beta i} U^*_{\alpha i} U^*_{\beta j} U_{\alpha j} \right] \cos \frac{\Delta E_{ij}t}{2} \sin \frac{\Delta E_{ip}t}{2}
# 	\sin \frac{\Delta E_{jp} t}{2} \\
# 	 + 4 \sum_{i,j} \mathrm{Im}\left[ U_{\beta i} U^*_{\alpha i} U^*_{\beta j} U_{\alpha j} \right] \sin \frac{\Delta E_{ij}t}{2} \sin \frac{\Delta E_{ip}t}{2}
# 	\sin \frac{\Delta E_{jp} t}{2}.
# $$
# 
# With:
# 
# $$
# \Delta E_{ip} \equiv E_i - E_p \simeq \frac{m^2_i-m^2_p}{2E} = \frac{\Delta m^2_{ip}}{2E}
# $$
# 
# and $p$-index arbitrary (i.e 1)
# 

# Manipulating the amplitude:
# 
# $$
#  \mathcal{A}_{\alpha \beta} = e^{-i E_p t}\sum_{i} U_{\beta i} U^*_{\alpha i} \, e^{-i (E_i-E_p)}
# $$
# 
# $$
# \mathcal{A}_{\alpha \beta} =  \sum_i U_{\beta i} U^*_{\alpha i} (e^{-i \Delta E_{ip} t} + 1 - 1)   
# = \sum_{i} U_{\beta i} U^*_{\alpha i} + \sum_i U_{\beta i} U^*_{\alpha i} e^{-i\frac{\Delta E_{ip t}}{2}}
# \left(e^{-i\frac{\Delta E_{ip t}}{2}} - e^{+i\frac{\Delta E_{ip t}}{2}}  \right)
# $$
# 
# 
# $$
# \mathcal{A}_{\alpha \beta}  = \delta_{\alpha \beta} -2i \sum_i U_{\beta i} U^*_{\alpha i} e^{-i \frac{\Delta E_{ip}}{2}} \sin \frac{\Delta E_{ip} t}{2}   
# $$
# 
# 
# 
# Computing the probability:
# 
# $$
# \mathcal{P}(\nu_\alpha \to \nu_ \beta) =  
# \left( \delta_{\alpha \beta} -2i \sum_i U_{\beta i} U^*_{\alpha i} e^{-i \frac{\Delta E_{ip}}{2}} \sin \frac{\Delta E_{ip} t}{2} \right)  
# \left( \delta_{\alpha \beta} +2i \sum_j U^*_{\beta j} U_{\alpha j} e^{i \frac{\Delta E_{jp}}{2}} \sin \frac{\Delta E_{jp} t}{2}   \right)
# $$
# 

# with $i > j$:
# 
# $$
# \mathcal{P}(\nu_\alpha \to \nu_\beta)  = \delta_{\alpha \beta} - 4 \sum_i |U_{\alpha i}|^2 (\delta_{\alpha \beta} - |U_{\beta i}|^2 )   \sin^2 \frac{\Delta E_{ip}t}{2} \\
# 	 + 8 \sum_{i>j} \mathrm{Re}\left[ U_{\beta i} U^*_{\alpha i} U^*_{\beta j} U_{\alpha j} \right] \cos \frac{\Delta E_{ij}t}{2} \sin \frac{\Delta E_{ip}t}{2}
# 	\sin \frac{\Delta E_{jp} t}{2} \\
# 	 + 8 \sum_{i>j} \mathrm{Im}\left[ U_{\beta i} U^*_{\alpha i} U^*_{\beta j} U_{\alpha j} \right] \sin \frac{\Delta E_{ij}t}{2} \sin \frac{\Delta E_{ip}t}{2}
# 	\sin \frac{\Delta E_{jp} t}{2}.
#  $$

# ### CP, T and CPT symmetry

# We can study CP, T, CP transformations:
# 
# $$
# \mathrm{CP} \; : \;  \mathcal{P}(\nu_\alpha \to \nu_\beta) \Rightarrow \mathcal{P}(\bar{\nu}_\alpha \to \bar{\nu}_\beta) \\ 
#  \mathrm{T} \; : \;  \mathcal{P}(\nu_\alpha \to \nu_\beta) \Rightarrow \mathcal{P}( \nu_\beta \to \nu_\alpha) \\ 
#  \mathrm{CPT} \; : \;  \mathcal{P}(\nu_\alpha \to \nu_\beta) \Rightarrow \mathcal{P}( \bar{\nu}_\beta \to \bar{\nu}_\alpha) \\ 
#  $$

# ### Mass ordering

# In Nature, we have measured two mass squared differences:
# 
# $$
#  \Delta m^2_A \simeq 2.5 \times 10^{-3} \; \mathrm{eV}^2 \\
#  \Delta m^2_\odot \simeq 7.5 \times 10^{-5} \; \mathrm{eV}^2 
# $$
# 
# There are at least three massive neutrinos, we define two mass squared differences:
# 
# $$
#  \Delta m^2_{21} = \Delta m^2_\odot \\
#  \Delta m^2_{31} = \Delta m^2_{32}= \pm \Delta m^2_A 
# $$

# 
# | |
# | :--: |
# | <img src="./imgs/hierarchy.png" width=400 align="center">|
# 
# That correspond to the *normal* (NH) and *inverted* (IH) mass hierarchies.
# 
# $$
# \mathrm{NH:} \; m_1 < m_2 < m_3, \;\; \mathrm{IH:} \; m_3 < m_1 < m_2
# $$
# 
# 

# The ratio:
# 
# $$
# \frac{\Delta m^2_\odot}{\Delta m^2_A} \simeq 3 \times 10^{-2}
# $$
# 
# With:
# 
# $$
# \phi_\odot  = \frac{ \Delta m^2_\odot L }{ 4E}, \;\; 
# \phi_A  = \frac{ \Delta m^2_A L }{ 4E}, \;\; 
# $$
# 
# 
# We have:
# 
# $$
# \phi_\odot \sim \mathcal{O}(\pi)  \;\; \Rightarrow \phi_{A} >> \phi_\odot \\
#  \phi_A \sim \mathcal{O}(\pi)  \;\; \Rightarrow \phi_\odot << \phi_A .
# $$

# The oscillation probability in terms of $\Delta m^2_\odot, \, \Delta m^2_A$:
# 
# $$
# \mathcal{P}(\nu_\alpha \to \nu_\beta) = \delta_{\alpha \beta} - 4 \, |U_{\alpha 3}|^2 (\delta_{\alpha \beta} - |U_{\beta 3}|^2 )   \sin^2 \frac{\Delta m^2_A L}{4 E} \\
# 	 - 4 \, |U_{\alpha 2}|^2 (\delta_{\alpha \beta} - |U_{\beta 2}|^2 )   \sin^2 \frac{\Delta m^2_\odot L}{4 E} \\
# 	 \pm 8 \, \mathrm{Re}\left[ U_{\beta 3} U^*_{\alpha 3} U^*_{\beta 2} U_{\alpha 2} \right] \cos \frac{\Delta m^2_A L}{4 E} \sin \frac{\Delta m^2_A L}{4 E}
# 	\sin \frac{\Delta m^2_\odot L}{4 E}  \\
# 	 + 8 \, \mathrm{Im}\left[ U_{\beta 3} U^*_{\alpha 3} U^*_{\beta 2} U_{\alpha 2} \right] \sin \frac{\Delta m^2_A L}{4 E} \sin \frac{\Delta m^2_A L}{4 E}
# 	\sin \frac{\Delta m^2_\odot L}{4 E}.
# $$

# ### Oscillation probability in experimental scenarios

# Let's consider the case of atmospheric neutrinos with accelerator. 
# MINOS experiment with $L \sim 750$ km and $E  \sim 1-5$ GeV:
# 
# $$
# \phi_A \sim \mathcal{O}(\pi) \;\; \Rightarrow \phi_\odot \sim 0
# $$
# 
# For the survival probability $\nu_\mu \to \nu_\mu$, using the standard U-PMNS parameterization, we get:
# 
# $$
# \mathcal{P}(\nu_\mu \to \nu_\mu) \simeq 1 - 4 |U_{\mu 3}|^2 (1-|U_{\mu 3}|^2) \sin^2 \frac{\Delta m^2_A L}{4 E}
# $$
# with $U_{\mu3} = s_{23}c_{13}$:
# 
# $$
# \mathcal{P}(\nu_\mu \to \nu_\mu) = 1 - 4 s^2_{23} c^2_{13}(1-s^2_{23}c^2_{13}) \sin^2 \frac{\Delta m^2_A L}{4 E}
# $$
# Now consider the case $s_{13} \sim 0, \, c_{13} \sim 1$:
# 
# $$
# \mathcal{P}(\nu_\mu \to \nu_\mu) \simeq 1 - \sin^2 2 \theta_{23} \sin^2 \frac{\Delta m^2_A L}{4 E}
# $$
# And we recuperate the two-families oscillation formula.

# 
# We can compute now the probability to oscillate to other flavors at MINOS, with the previous approximations:
# 
# $$
# \mathcal{P}(\nu_\mu \to \nu_e)  \simeq 4 s^2_{23} c^2_{13} s^2_{13} \sin^2 \frac{\Delta m^2_A L}{4 E} \sim 0 \\
# \mathcal{P}(\nu_\mu \to \nu_\tau)  \simeq 4 s^2_{23} c^2_{13} c^2_{23} c^2_{13} \sin^2 \frac{\Delta m^2_A L}{4 E} \simeq \sin^2 2 \theta_{23} \sin^2 \frac{\Delta m^2_A L}{4E} \\
# $$
# 
# Notice that as $\mathcal{P}(\nu_\mu \to \nu_e)$ is suppressed, then it is sensitive to second-order effects, 
# 
# 

# For the atmospheric case with reactor neutrinos.
# KamLAND experiment with $L=180$ km and $E \sim 1-5$ MeV. 
# 
# $$
# \phi_\odot \sim \mathcal{O}(\pi) \Rightarrow \phi_A >> \phi_\odot
# $$
# 
# The atmospheric oscillation is then averaged $\sin^2 \phi_A = 1/2, \sin \phi_A =  \cos \phi_A = 0$, we have:
# 
# $$
# \mathcal{P}(\bar{\nu}_e \to \bar{\nu}_e) \simeq 1 - 4 |U_{e2}|^2 (1-|U_{e2}|^2) \sin^2 \frac{\Delta m^2_\odot L}{4E} - 2 |U_{e3}|^2 (1-|U_{e3}|^2)
# $$
# 
# With $U_{e2} = s_{12} c_{13}$ and $U_{e3} = s_{13} e^{-i\delta}$, and taking $s_{13} \sim 0, \, c_{13} \sim 1$:
# 
# $$
# \mathcal{P}(\bar{\nu}_e \to \bar{\nu}_e) \simeq 1 - 4 s^2_{12} c^2_{12} \sin^2 \frac{\Delta m^2_\odot L}{4E} = 1 - \sin^2 2 \theta_{12} \sin^2 \frac{\Delta m^2_\odot L}{4E}
# $$

# Now consider the case of reactor neutrinos in the atmospheric regime:
# DayaBay experiment with $L=1$ km and $E \sim 1-5$ MeV.
# 
# $$
# \phi_A \sim \mathcal{O}(\pi) \Rightarrow \phi_\odot \sim 0.
# $$
# 
# Now:
# 
# $$
# \mathcal{P}(\bar{\nu}_e \to \bar{\nu}_e) \simeq 1 - 4 |U_{e3}|^2 (1-|U_{e3}|^2) \sin^2 \frac{\Delta m^2_A L}{4 E}
# $$
# 
# with $U_{e3} = s_{13}$, we get:
# 
# $$
# \mathcal{P}(\bar{\nu}_e \to \bar{\nu}_e) \simeq 1 - 4 s^2_{13} (1-s^2_{13}) \sin^2 \frac{\Delta m^2_A L}{4 E} = 1 - \sin^2 2 \theta_{13} \sin^2 \frac{\Delta m^2_A L}{4 E}
# $$
# 
# The amplitude of the oscillation corresponds to $\sin^2 2 \theta_{13}$.

# ### Interactive oscillation: three families (PMNS)
# 
# Los parámetros por defecto corresponden al ajuste global **NuFit-6.0** (2024) [[33]](https://arxiv.org/abs/2410.05380), Ordenación Normal (NH), con $\Delta m^2_{21}$ y $\sin^2\theta_{12}$ actualizados con los primeros resultados de JUNO [[38]](https://arxiv.org/abs/2511.14593):
# 
# $$
# \theta_{12} = 33.7°, \quad \theta_{13} = 8.6°, \quad \theta_{23} = 48.3°, \quad \delta_{\rm CP} = 212°
# $$
# $$
# \Delta m^2_{21} = 7.48 \times 10^{-5}\;{\rm eV}^2, \quad \Delta m^2_{31} = 2.52 \times 10^{-3}\;{\rm eV}^2
# $$
# 
# Desliza los controles para explorar la dependencia en energía y distancia, o para comparar diferentes valores de los parámetros.

# In[12]:


import oscillations

oscillations.plot_3fam_interactive()


# ---
# ## Next: Experimental Evidence
# 
# The experimental results — solar, atmospheric, reactor (θ₁₃), long-baseline (δ_CP, mass ordering) and the new generation of experiments — are covered in the companion notebook **[Neutrino Oscillations: Experimental Evidence](nu_oscillations_exp.ipynb)**.
