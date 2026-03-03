#!/usr/bin/env python
# coding: utf-8

# # Neutrinos and the construction of the Standard Model

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


# *Objective:*
# 
# Show how the neutrinos experiments defined the SM construction.
# 
# How parity violation and Neutral Current were fundamental to the acceptance of the SM.
# 

# ## Pauli postulated the existence of the neutrino
# 
# The $\beta$ decay in nuclei was a mystery in the 1920s.
# 
# $$
# ^A_Z X \to ^A_{Z+1}X' + e \;\; (?)
# $$
# 
# | |
# |:--:|
# |<img src="./imgs/bspectrum_1935.png" width=500 align="center">|
# ||
# 
# The spectrum was continuous. 
# 
# If only an electron, $\beta$, was emitted, the released energy would be monochromatic.
# 
# N. Bohr: *"At the present stage of atomic theory, however, we may say that we have no argument... for upholding the energy principle in the case of $\beta$-ray disintegrations."*
# 
# 

# **Exercise: Simulating the $\beta$-decay electron spectrum**
# 
# In a **2-body** decay ($A \to B + e$, without a neutrino), energy-momentum conservation fixes the electron energy uniquely: $T_e = Q$ (monochromatic).
# 
# In a **3-body** decay ($A \to B + e + \bar{\nu}$), the electron energy is distributed continuously:
# $$
# \frac{dN}{dT_e} \propto p_e\, E_e\, (Q - T_e)^2 \times F(Z, T_e)
# $$
# where $F(Z, T_e)$ is the Fermi function (Coulomb correction). For the endpoint region, a finite neutrino mass $m_\nu$ modifies the spectrum:
# $$
# (Q - T_e)^2 \;\to\; (Q - T_e)\sqrt{(Q - T_e)^2 - m_\nu^2}
# $$
# 
# **Questions:**
# 1. Simulate both spectra for $^{210}$Bi ($Q = 1.161$ MeV) and compare with the historical figure above.
# 2. How does the endpoint shape change for $m_\nu = 50$ keV? Is it detectable?
# 3. KATRIN uses tritium ($Q = 18.6$ keV): why is tritium better suited than bismuth for a neutrino mass search?
# 

# In[ ]:


from sm import beta_spectrum
beta_spectrum()


# [Pauli](https://en.wikipedia.org/wiki/Wolfgang_Pauli) postulated the existence of a light neutral particle that escapes undetected in $\beta$ decays! 
# 
# This is his famous letter, sent to the Gauverein meeting in Tubingen.
# 
# | |
# | :--: |
# |<img src="./imgs/Pauli_letter_neutrino.jpg" width=800 align="center">|
# ||
# 
# 

# *I have hit upon a desperate remedy to save... the law of conservation of energy. Namely, the possibility that there could exist in the nuclei electrically neutral particles, that I wish to call neutrons, which have spin 1/2*
# 
# *The continuous beta spectrum would then become understandable by the assumption that in beta decay a neutron is emitted in addition to the electron such that the sum of the energies of the neutron and the electron is constant.*
# 
# Later Pauli will comment: *“I have done a terrible thing. I postulated a particle that can not be detected!”*
# 
# 

# The neutron was discovered by Chadwick (1932) [[1]](https://www.nature.com/articles/129312a0)
# 
# Later [Fermi](https://en.wikipedia.org/wiki/Enrico_Fermi) named Pauli's particle "neutrino" at the Solvay Conference in Brussels in 1933.
# 

# Fermi constructed the theory of  $\beta$-decay [2] in 1934, explaining it in terms of a 4-fermion interaction $n \to p + e + \bar{\nu}_e$ with strength $G_F$ .
# 
# 
# *Theory of β rays emission of radioactive substances, built on the hypothesis that the electron emitted by the nuclei do not exist before the decay. On the contrary they are created together with a neutrino.*
# 
# | |
# | :--: |
# |<img src="./imgs/fermi_currents_SM.png" width=300 align="center">|
# ||
# 
# $$
# \frac{G_F}{\sqrt{2}} \, (\bar{\Psi}_n \gamma_\mu \Psi_p) \, (\bar{\Psi}_\nu \gamma^\mu \Psi_e)
# $$
# 

# 
# The strength of the interaction is controlled by Fermi constant $G_F$. 
# 
# In the modern view, $G_F$ is expressed in term of the weak constant, $g$, and the $W$ mass, $m_W$:
# 
# $$
# \frac{G_F}{\sqrt{2}} = \frac{g^2}{ 8 m_W^2}
# $$
# 
# $$
# G_F = 1.1663788(7) \times 10^{-5} \; \mathrm{GeV}^{-2}
# $$
# 
# Fermi argued that the weak interaction is weak because it is short range, not because the coupling itself is small. 
# 
# This interaction would also predict the scattering of neutrinos off matter, via the inverse process $\bar{\nu}_e + p \to n + e^+$

# ### Unitarity violation in Fermi theory and the need for the $W$ boson
# 
# The Fermi theory is an effective theory valid only at low energies. The amplitude for $\nu_\mu + e^- \to \nu_e + \mu^-$ grows with energy:
# $$
# \mathcal{M} \sim G_F\, s, \qquad s = E_{\rm CM}^2
# $$
# Unitarity of the S-matrix requires $|a_J| \leq 1$ for each partial wave, which is violated above:
# $$
# \sqrt{s_{\rm max}} \sim \frac{1}{\sqrt{G_F}} \approx 300 \; \mathrm{GeV}
# $$
# The cure is the $W$ boson propagator, which tames the high-energy behaviour:
# $$
# \frac{G_F}{\sqrt{2}} \quad\xrightarrow{\text{full theory}}\quad \frac{g^2}{8} \cdot \frac{1}{q^2 - m_W^2} \quad\xrightarrow{q^2 \ll m_W^2}\quad \frac{g^2}{8 m_W^2} = \frac{G_F}{\sqrt{2}}
# $$
# The $W^\pm$ and $Z^0$ bosons were discovered at CERN in 1983 by the UA1 and UA2 collaborations at the $Sp\bar{p}S$. Their masses, $m_W \approx 80.4$ GeV and $m_Z \approx 91.2$ GeV, were in spectacular agreement with the GWS prediction.
# 

# **Exercise: The Fermi constant from the muon lifetime**
# 
# The muon decays via $\mu^- \to e^- + \bar{\nu}_e + \nu_\mu$. In the V−A theory, the total decay width at leading order is:
# $$
# \Gamma(\mu \to e\,\nu_\mu\,\bar{\nu}_e) = \frac{G_F^2\, m_\mu^5}{192\pi^3}
# $$
# This is the **primary experimental determination of $G_F$**: measuring $\tau_\mu$ and inverting for $G_F$.
# 
# Once $G_F$ is known, the W mass follows from:
# $$
# \frac{G_F}{\sqrt{2}} = \frac{g^2}{8 m_W^2} \quad \Rightarrow \quad m_W = \left(\frac{\pi\alpha}{\sqrt{2}\,G_F\sin^2\theta_W}\right)^{1/2}
# $$
# 
# **Questions:**
# 1. Derive $G_F$ from the measured lifetime $\tau_\mu = 2.1970 \times 10^{-6}$ s. Compare with the PDG value.
# 2. Extract the weak coupling $g$ and compare $\alpha_W = g^2/4\pi$ with $\alpha_{em}$.
# 3. Verify $\sin^2\theta_W = 1 - m_W^2/m_Z^2$ using the PDG masses.
# 

# In[ ]:


from sm import fermi_constant_from_muon
fermi_constant_from_muon()


# In 1934 [Bethe](https://en.wikipedia.org/wiki/Hans_Bethe) and Peierls were able to estimate the cross section for this process [[3]](https://www.nature.com/articles/133532a0), finding it smaller than $10^{-44} \; \mathrm{cm^2}$ for a neutrino energy of 2 MeV.
# 
# $$
# \sigma(\bar{\nu}_e + p \to n + e^+) \simeq 10^{-47} \mathrm{(E/MeV)^2 m^2}
# $$
# 
# Bethe: *"it was absolutely impossible to observe processes of this kind".*
# 
# 
# **Question:** What is the mean free path for a 1 MeV $\nu$ in water? and in lead?
# 
# **Question:** What is the energy threshold for the inverse $\beta$ decay?

# **Exercise: Neutrino mean free path**
# 
# The mean free path in a medium with target number density $n$ is:
# $$
# \lambda = \frac{1}{n \cdot \sigma(E_\nu)}
# $$
# Using the IBD cross section $\sigma \approx 9.52\times10^{-44}(E_\nu/\mathrm{MeV})^2\ \mathrm{cm}^2$:
# 
# **Questions:**
# 1. Compute and plot $\lambda(E_\nu)$ in water, lead and iron for $0.1 \leq E_\nu \leq 10^4$ MeV.
# 2. At what energy does $\lambda$ in water become comparable to the Earth's radius ($R_\oplus = 6\,371$ km)?
# 3. Can a 1 MeV solar neutrino escape the Sun ($R_\odot = 696\,000$ km, mostly hydrogen)?
# 

# In[ ]:


from sm import neutrino_mean_free_path
neutrino_mean_free_path()


# ## The discovery of the neutrino. The experiment of Cowan and Reines
# 
# B. Pontecorvo [[>]](https://www.youtube.com/watch?v=yXrHnsBgQSw&t=9s) who suggested that indeed one could use the large neutrino fluxes becoming available [4] in nuclear reactors.
# 
# F. Reines and C. L. Cowan devised a method to detect antineutrinos coming from a nuclear reactor. 
# 
# Using the Savanna River reactor, SC, with 0.7 GW, with produced a flux $\phi \simeq 10^{17}$ $\nu\mathrm{/(m^2 \, s)}$ 
# 
# **Question:** the Savannah River reactor had 0.7 GW power, each fission releases 196 MeV, and 6 neutrinos that takes 9 MeV/fission. Compute the neutrino flux at 10 m below the reactor, and the rate of interaction on the free protons (H) in a 100 kg water detector.
# 

# 
# | |
# | :--: |
# | <img src="./imgs/cowan_reines_detector.png" width=500 align="center">|
# 
# 
#   - two modules 100 kg water blocks sandwiched between two liquid scintillator chambers
#   - looking the inverse $\beta$ interaction
#   - main backgrounds: neutrons spallation, cosmic rays, and natural radioactivity
#   - counting experiment: using on/off of the reactor
#   

# | |
# | :--: |
# | <img src="./imgs/cowan_reines_method.png" width=500 align="center">|
# 
# It exploits simultaneous emission of a neutron and a positron in inverse beta decays to significantly reduce backgrounds.

# The detection exploits two time-correlated signals that form a nearly background-free signature:
# 
# 1. **Prompt signal** ($t = 0$): the positron annihilates immediately with an electron, producing **two coincident 511 keV photons** detected in the liquid scintillator layers on either side of the water target.
# 
# 2. **Delayed signal** ($\Delta t \sim 5$–$10\ \mu$s later): the neutron thermalises and is captured on cadmium nuclei dissolved in the water ($\mathrm{CdCl_2}$), emitting a **cascade of $\gamma$-rays of total energy $\sim 9$ MeV**:
# $$
# e^+ + e^- \to 2\gamma\;(511\;\mathrm{keV}) \quad \xrightarrow{\ \Delta t \sim 5\text{--}10\;\mu\mathrm{s}\ } \quad n + {}^{108}\mathrm{Cd} \to {}^{109}\mathrm{Cd}^* \to \gamma\;(9\;\mathrm{MeV})
# $$
# 
# The requirement of **delayed coincidence** in the same detector module, combined with the characteristic time interval, reduces accidental backgrounds by many orders of magnitude. Switching the reactor on/off provided a direct background subtraction.
# 

# **Exercise: Reactor neutrino flux and expected event rate**
# 
# **Questions:**
# 1. Given $P = 0.7$ GW, $E_{\rm fission} = 196$ MeV, and 6 $\bar{\nu}_e$ per fission, compute the total fission rate $\dot{N}_f$ and the flux $\phi$ at $d = 11$ m.
# 2. How many free protons $N_p$ are in 200 kg of water?
# 3. Using $\langle E_\nu \rangle \approx 3.5$ MeV, compute the expected event rate. Compare with $2.9 \pm 0.2$ events/hour after applying a detection efficiency $\varepsilon \approx 10\%$.
# 

# In[ ]:


from sm import reactor_flux
reactor_flux()


# They observed $2.9\pm0.2$ events/hour not explained by background.
# 
# 
# | |
# | :--: |
# | <img src="./imgs/cowan_reines_osciloscope.png" width=500 align="center">|
# 

# In 1956 they were able to detect neutrinos [[5]](https://www.nature.com/articles/178446a0) and soon wrote a telegram to Pauli. 
# 
# | |
# | :--: |
# | <img src="./imgs/cowan_reines_telegram.png" width=500 align="center">|
# 
# 
# [Reines](https://en.wikipedia.org/wiki/Frederick_Reines) received the Nobel Prize in Physics in 1995

# ### Parity violation: the Wu experiment (1957)
# 
# Lee and Yang [14] showed in 1956 that parity had **never been tested** in weak interactions. C. S. Wu et al. immediately devised a decisive experiment: aligning the spins of $^{60}$Co nuclei in a strong magnetic field at $T \sim 10$ mK, and observing the angular distribution of the emitted electrons:
# $$
# ^{60}\mathrm{Co} \to\ ^{60}\mathrm{Ni}^* + e^- + \bar{\nu}_e
# $$
# **If parity were conserved**, flipping the magnetic field (which reverses the spin polarisation) should yield the same electron flux in all directions. Instead, electrons were emitted **preferentially opposite to the nuclear spin**:
# $$
# W(\theta) \propto 1 - \frac{v_e}{c}\,\langle J \rangle\cos\theta
# $$
# where $\theta$ is the angle between the electron momentum and the polarisation axis. The asymmetry was **maximal** — parity is completely violated in weak decays.
# 
# **Combined with Goldhaber's result** ($h_\nu = -1$), this established that:
# - The weak interaction couples **only to left-handed fermions** and right-handed antifermions.
# - Charge conjugation **C** is also maximally violated: the mirror image of a $\nu_L$ is a $\nu_R$, which does not interact weakly.
# - This motivated the V−A theory (Feynman–Gell-Mann, Sudarshan–Marshak, 1958).
# 

# ### The V−A structure of the weak current
# 
# The five Lorentz-invariant bilinear covariants and their transformation properties under parity:
# 
# | Type | Operator | Under $P$ | Helicity selected |
# |------|----------|-----------|-------------------|
# | Scalar $S$ | $\bar{\psi}\psi$ | $+$ | both |
# | Pseudo-scalar $P$ | $\bar{\psi}\gamma^5\psi$ | $-$ | both |
# | Vector $V$ | $\bar{\psi}\gamma^\mu\psi$ | $+$ | both equally |
# | Axial-vector $A$ | $\bar{\psi}\gamma^\mu\gamma^5\psi$ | $-$ | both equally |
# | **V−A** | $\bar{\psi}\gamma^\mu(1-\gamma^5)\psi$ | **max. violation** | $h = -1$ only |
# 
# Sudarshan and Marshak, and independently Feynman and Gell-Mann (1958), proposed the **V−A** current:
# $$
# J^\mu_{\rm CC} = \bar{\psi}_\nu \gamma^\mu (1 - \gamma^5) \psi_\ell = 2\,\bar{\psi}_{\nu_L} \gamma^\mu \psi_{\ell_L}
# $$
# 
# The factor $(1 - \gamma^5)/2 \equiv P_L$ is the **left-handed chirality projector**. For massless particles, chirality equals helicity, so V−A couples **only to left-handed fermions** (and right-handed antifermions).
# 
# The Fermi Lagrangian in V−A form is:
# $$
# \mathcal{L}_F = -\frac{G_F}{\sqrt{2}} \left(\bar{\psi}_p \gamma^\mu (1-\gamma^5) \psi_n\right)\left(\bar{\psi}_e \gamma_\mu (1-\gamma^5) \psi_\nu\right) + \mathrm{h.c.}
# $$
# 

# ### The neutrino in the electroweak Standard Model
# 
# Glashow (1961), Weinberg (1967) and Salam (1968) embedded the V−A structure into a gauge theory with symmetry group $SU(2)_L \times U(1)_Y$.
# 
# Leptons are organised as **left-handed doublets** and **right-handed singlets** under $SU(2)_L$:
# 
# $$
# L_i = \begin{pmatrix} \nu_{iL} \\ \ell_{iL} \end{pmatrix}, \quad i = e,\mu,\tau
# \qquad\ell_{iR} = e_R, \mu_R, \tau_R
# $$
# 
# **Key point: there is no right-handed neutrino $\nu_R$ in the SM field content.**
# 
# The charged-current (CC) and neutral-current (NC) interactions arise directly from the $SU(2)_L$ covariant derivative:
# 
# $$
# \mathcal{L}_{CC} = \frac{g}{\sqrt{2}} \sum_{i} \bar{\nu}_{iL}\, \gamma^\mu\, \ell_{iL}\, W^+_\mu + \mathrm{h.c.}
# $$
# 
# $$
# \mathcal{L}_{NC} = \frac{g}{2\cos\theta_W} \sum_{i} \bar{\nu}_{iL}\, \gamma^\mu\, \nu_{iL}\, Z_\mu
# $$
# 
# Since $\nu_R$ is absent, there is **no Dirac mass term** $m\,\bar{\nu}_L\nu_R$ in the SM. The neutrino is predicted to be **massless** — a prediction later contradicted by oscillation experiments.
# 
# $$
# \begin{array}{ll}
# \text{Symmetry:} & SU(3)_c \times SU(2)_L \times U(1)_Y \\
# \text{Neutrino content:} & \nu_L \text{ only (left-handed doublet, no } \nu_R\text{)}\\
# \text{Interactions:} & \text{CC via } W^\pm \text{ and NC via } Z \text{ (purely left-handed)}\\
# \text{Mass:} & m_\nu = 0 \quad\text{(not protected by symmetry — accidental)}
# \end{array}
# $$
# 

# ## The helicity of the neutrino. 
# 
# | |
# | :--: |
# | <img src="./imgs/goldhaber_decays.png" width=300 align="center">|
# 
# 
# 
# [Goldhaber](https://en.wikipedia.org/wiki/Maurice_Goldhaber) at Brookhaven measured the neutrino helicity in 1958 via an ingenious experiment [6].
# 
# Measuring the gamma polarization of the electron capture in S wave in $\mathrm{^{152}Eu}$:
# 
# $$
# \mathrm{Eu}  + e^- \to \mathrm{^*Sm} + \nu_e \to \mathrm{Sm} + \gamma + \nu_e
# $$
# 
# The gamma has the same polarization as the neutrino helicity. 
# 
# 

# | |
# | :--: |
# | <img src="./imgs/goldhaber_experiment.png" width=300 align="center">|

# 
# - Only neutrinos going upwards are selected.
# 
#    - the recoil of $\mathrm{Sm}^*$ is transfered to the gamma.
# 
#    - Only gammas which are back-to-back to neutrinos have enough energy to produce resonance absortion.
# 
# - the polarization of the gamma is selected via the $B$ field orientation (up/down) in the ferro-magnetic support.
# 
#    - Only electrons with oposite spin to the $B$ suffer E.C. 
#    
#    - This select electrons with spin up/down.
# 
# - Switch  the polarity of $B$ and detect number of gammas in NaI (that is helicity of the neutrino).
# 
# - Measured neutrino helicity:
# 
# $$
# -1 \pm 0.3
# $$

# **Exercise: Helicity suppression in pion and kaon leptonic decays**
# 
# For a two-body decay $\pi^+ \to \ell^+ + \nu_\ell$, the helicity of the $\nu_\ell$ is fixed by momentum conservation to be $h = +1$ (right-handed). But the SM couples only left-handed neutrinos. The amplitude is therefore suppressed by the lepton mass:
# 
# $$
# \frac{\Gamma(\pi^+ \to e^+ \nu_e)}{\Gamma(\pi^+ \to \mu^+ \nu_\mu)} = \frac{m_e^2}{m_\mu^2} \cdot \frac{(m_\pi^2 - m_e^2)^2}{(m_\pi^2 - m_\mu^2)^2}
# $$
# 
# **Questions:**
# 1. Compute the ratio from PDG masses. Compare with the measured value $1.2327 \times 10^{-4}$.
# 2. Repeat the calculation for the kaon: $K^+ \to \ell^+ \nu_\ell$. PDG value: $2.434 \times 10^{-5}$.
# 3. Plot $\Gamma(P \to \ell\nu) \propto m_\ell^2(m_P^2 - m_\ell^2)^2$ as a function of $m_\ell$ and locate $m_e$ and $m_\mu$.
# 

# In[ ]:


from sm import pion_helicity_suppression
pion_helicity_suppression()


# ### Neutrinos maximally violate C and P in the SM
# 
# | |
# | :--: |
# | <img src="./imgs/SM_pidecay_CP.png" width=500 align="center">|
# 
# Pion decays to leptons are helicity suppressed. Pions decay to muons 99% of the time.
# 
# $$
# \frac{m_e^2}{m_\mu^2} = 0.22 \times 10^{-4}, \;\; \frac{\Gamma(\pi^+ \to e^+ \, \nu_e)}{ \Gamma(\pi^+ \to \mu^+ \, \nu_\mu)} = 1.2 \times 10^{-4}
# $$
# 
# **Questions:** Apply the same argument to kaon semileptonic decays.
# 
# 

# ## The three neutrino families
# 
# In the SM helicity and chirality are related.
# 
# $$
# P_{R, L } = \frac{1}{2}(1 \pm \gamma^5) = \frac{1}{2} \left(1 \pm \frac{{\bf s} \cdot {\bf p}}{|p|} \right) + \mathcal{O}\left(\frac{m}{E}\right)
# $$
# 
# The helicity operator, that measures the projection of the spin ${\bf s}$ along the momentum ${\bf p}$ is:
# 
# $$
# \Sigma = \frac{{\bf s} \cdot {\bf p}}{|p|}
# $$
# 
# 
# If the mass is zero, helicity and chirality are the same and are good quantum numbers.
# 
# Landau [13], Lee and Yang [14] and Salam [15] in 1957 proposed that neutrinos can be described with a left-handed Weyl spinor. 
# 
# This property was embedded in the V-A theory of weak interactions and ultimately in the SM by S. L. Glashow [16] in 1961, S. Weinberg [17] in 1967, and A. Salam [18] in 1968.
# 
# **Question:** Write the helicity states as left and right states.
# 

# ### The discovery of the muon and the muon neutrino
# 
# - The muon was discovered in 1937 by J. C. Street and E. C. Stevenson [19] and by S.
# H. Neddermeyer and C. D. Anderson [20]. 
#    - Initially they thought it was the $\pi$ postulated by Yukawa.
#    
#    
# - In 1949 Brown et al [[21]](https://www.nature.com/articles/163047a0), using Kodak emulsions showed $\pi, \, \mu$ decays, and the need of missing particles!
#    - It took a decade to understand it was a heavy version of e, $\mu$, "similar" β decay
# 

# 
# | |
# | :--: |
# | <img src="./imgs/mu_emulsion.png" width=800 align="center">|
# 
# 
# - It can enter Fermi interactions with a neutrino. But is the same neutrino of the $\beta$ decay?
# 
# - Rabi: *How order this?* Why there are three families?

# ### The discovery of the $\nu_\mu$
# 
# - Following a suggestion by Pontecorvo [20] 
# 
# - L. M. Lederman, M. Schwartz and J. Steinberger et al. created the first accelerator neutrino beam, from pion decays from a boosted proton beam hitting a target. They received the Nobel Prize in 1988.
# 
# - 15 GeV p in Be target, 13.5 m iron filter 
# 
# $$
# \pi^- \to \mu^- + \bar{\nu}_\mu, \; \pi^+ \to \mu^+ + \nu_\mu, 
# $$
# 
# 

# 
# | |
# | :--: |
# | <img src="./imgs/brookhaven_munu_experiment.png" width=800 align="center">|
# 
# 

# 
# - Spark chambers were a novel technology. Photographs synchronized with the beam pulse.
# 
# - 10 modules of 9 spark chambers $1.1 \times 1.1$ m$^2$, 10 tons.
# 
# - Detector surrounded by anti-coincidence scintillation planes.
# 
# - Search for $\nu + n \to p + \mu^-, \;\; \nu + n \to p + e^-$ 
# 
# - 10 triggers per hour (mostly empty photographs).
# 
# - 34 events with penetrating tracks originated in the chambers.
# 
# - Neutrinos produced in pion decays associated with a muon do not lead to electrons in scatterings off matter [22]    

# ### The third family: the tau and the tau neutrino
# 
# | |
# | :--: |
# | <img src="./imgs/tau_discovery.png" width=600 align="center">|
# 
# 
# - Perl et al., in 1975 at SPEAR in SLAC, discovered the tau lepton $\tau$ near the production threshold [[23]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.35.1489)
# 
#    - They discovered $e, \mu$ events at a given energy threshold. 
#    
#    - They are the product of leptonic decays of a new charged lepton $\tau$ with $m_\tau \simeq 1800$ MeV

# 
# | |
# | :--: |
# | <img src="./imgs/nutau_discovery.png" width=500 align="center">|
# 
# - Niwa et al., in 2000 at Fermilab, using the emulsion technique, produced $\nu_\tau$ from $D_s$ semileptonic decays [[24]](https://arxiv.org/abs/hep-ex/0012035) 
# 
#    - They observed the kink from $\tau \to \mu \, \nu_{\tau} \, \bar{\nu}_\mu$

# ### Lepton Universality
# 
# 
# - Lepton Universality implies that the coupling to each lepton doublet is the same, $g/\sqrt{2}$.
# 
# 
# | |
# | :--: |
# | <img src="./imgs/lepton_universality.png" width=1000 align="center">|
# 
# 

# **Question:** Compare the decay width, $\tau \to \nu_\tau \, \mu \, \bar{\nu}_\mu$ and $\tau \to \nu_\tau \, e \, \bar{\nu}_e$
# 
# $$
# \Gamma(\tau \to \nu_\tau \, \mu \, \bar{\nu}_\mu) \propto \frac{g_\tau^2}{m_W^2} \frac{g_\mu^2}{m_W^2} m_\tau^5, \;\;\;
# \Gamma(\tau \to \nu_\tau \, e \, \bar{\nu}_e) \propto \frac{g_\tau^2}{m_W^2} \frac{g_e^2}{m_W^2} m_\tau^5
# $$
# 
# $$
# \frac{\mathrm{BR}(\tau \to \nu_\tau \, \mu \, \bar{\nu}_\mu)}{\mathrm{BR}(\tau \to \nu_\tau \, e \, \bar{\nu}_e)} = \frac{17.36 \pm 0.05}{17.84 \pm 0.05} = 0.974 \pm 0.004
# $$
# 
# including phase space factor:
# 
# $$
# \frac{g_\mu}{g_e} = 1.001 \pm 0.002 
# $$
# 
# 
# 

# **Question:** Compare the decay width, $\mu \to \nu_\mu \, e \, \bar{\nu}_e$ and $\tau \to \nu_\tau \, e \, \bar{\nu}_e$
# 
# $$
# \frac{\Gamma(\mu \to \nu_\mu \, e \, \bar{\nu}_e)}{\Gamma(\tau \to \nu_\tau \, e \, \bar{\nu}_e)}
# =\frac{g_\mu^2}{g_\tau^2} \frac{m_\mu^5}{m_\tau^5} \frac{\rho_\mu}{\rho_\tau} 
# $$
# 
# where $\rho_{\mu(_\tau)}$ are the phase-space factors.
# 
# $$
# \frac{g_\mu^2}{g_\tau^2} = \frac{1}{\tau_\mu} \frac{\tau_\tau}{\mathrm{BR}(\tau \to \nu_\tau \, e \, \bar{\nu}_e)} \frac{m_\tau^5}{m_\mu^5} \frac{\rho_\tau}{\rho_\mu}
# $$
# 
# where $\tau_{\mu(_\tau)}$ are the lifetimes of $\mu, \, (\tau)$.
# 
# $$
# \frac{g_\mu}{g_\tau} = 1.001 \pm 0.003
# $$
# 

# **Exercise: Quantitative test of lepton universality**
# 
# Lepton universality predicts $g_e = g_\mu = g_\tau \equiv g$. The ratios of leptonic partial widths allow a direct test.
# 
# From the V−A three-body decay formula including phase space:
# $$
# \Gamma(\ell \to \nu_\ell \ell' \bar{\nu}_{\ell'}) \propto G_F^2(\ell)\, G_F^2(\ell')\, m_\ell^5\, f\!\left(\frac{m_{\ell'}^2}{m_\ell^2}\right), \quad
# f(x) = 1 - 8x + 8x^3 - x^4 - 12x^2\ln x
# $$
# 
# **Questions:**
# 1. Compute $g_\mu/g_e$ from the ratio $\mathrm{BR}(\tau\to\mu\nu\nu)/\mathrm{BR}(\tau\to e\nu\nu)$ using PDG values.
# 2. Compute $g_\tau/g_\mu$ comparing the partial widths of $\mu$ and $\tau$ leptonic decays.
# 3. Are both ratios consistent with universality at the per-mille level? Compute the deviation in units of $\sigma$.
# 

# In[ ]:


from sm import lepton_universality
lepton_universality()


# ### Lepton Number Conservation (Violation)
# 
# 
# - Lepton Number Conservation implies that with every lepton produced there is its anti-neutrino with the same flavour.
# 
#     - Total and flavour lepton numbers are preserved.
#     
#     - This is a phenomenological result and an accidental symmetry of the SM.
#     
# | |
# | :--: |
# | <img src="./imgs/lfv_muenu.png" width=400 align="center"> |
#     
#     
# - Long history of searches for Lepton Flavour Violation processes with null results
# 
#     - Strongest constraint $\mathrm{BR}(\mu \to e \, \gamma) < 4.2 \times 10^{-13}$ [](https://arxiv.org/abs/1605.05081) by MEG experiment [25]
#     
#     - $\mathrm{BR}(\mu \to e \, \gamma) \; \mathcal{O}(10^{-52})$ due to $U_{PMNS}$ matrix.

# LFV searches summary [[26]](http://pdg.lbl.gov/2019/reviews/rpp2019-rev-conservation-laws.pdf)
# 
# | |
# | :--: |
# | <img src="./imgs/LFV_limits.png" width=1000 align="center">|
# 

# ## Neutral currents
# 
# ### Gargamelle
# 
# | |
# | :--: |
# | <img src="./imgs/gargamelle_nc.png" width=500 align="center">|
# 
# Neutral Currents were detected at the Gargamelle experiment at CERN in 1973 [27].
# 

# 
# - Gargamelle was a tank of 15 tons of freon, a gigantic bubble chamber.
# 
# - $\nu_\mu$ interactions were recorded in photographs.
# 
# - NC were discovered in $\nu_\mu$ interactions without $\mu$ (penetrating track).
# 
# $$
# \left.\frac{\rm NC}{\rm CC}\right|_{\nu} = 0.21 \pm 0.03, \;\; \left.\frac{\rm NC}{\rm CC}\right|_{\bar{\nu}} = 0.45 \pm 0.09
# $$

# ### The first NC event: a recoil electron
# 
# The most convincing initial evidence for NC was not a hadronic event but a single **recoil electron** from the elastic process:
# $$
# \bar{\nu}_\mu + e^- \to \bar{\nu}_\mu + e^-
# $$
# This purely leptonic NC process has no CC counterpart (there is no $\bar{\nu}_\mu \to \mu^+$ conversion here). The event, observed on 19 July 1973, showed a single electron track of $\sim 400$ MeV with no muon — a nearly background-free signature.
# 
# The NC/CC ratios from hadronic events provided the first determination of $\sin^2\theta_W$:
# $$
# \left.\frac{\rm NC}{\rm CC}\right|_\nu = \frac{1}{2} - \sin^2\theta_W + \frac{5}{9}\sin^4\theta_W \approx 0.21 \pm 0.03
# $$
# $$
# \Rightarrow \quad \sin^2\theta_W \approx 0.23
# $$
# This value, consistent with the GWS electroweak unification, was the **first experimental confirmation of the SM electroweak structure**.
# 

# The NC  lagrangian is:
# 
# $$
# \mathcal{L}_{NC} = - \frac{g}{2 \cos \theta_W} \sum_{f} \bar{\psi}_f \gamma^\mu (g_V^f - g_A^f \gamma^5) \psi_f Z_\mu
# $$
# 
# where $i$ runs over the three generations, and
# 
# $$
# g_V^f = T_3^f - 2 Q_f \sin^2 \theta_W, \;\; g_A^f = T_3^f
# $$
# 
# Depends on the weak-isospin component $T_3$ and the charge $Q$ in electron units.
# 
# For neutrinos $g_V^\nu = g_A^\nu = 1/2$

# ### Only three neutrino families
# 
# Lepton Electron Positron (LEP) was a CERN $e^+e^-$ collider (1989–2000)
# 
# With 4 large general purpose detectors: ALEPH, DELPHI, OPAL, L3.
# 
# One of their main results was the measurement of the properties of the Z boson, in particular its width $\Gamma_Z$. LEP collected 7 M Z events.
# 
# The LEP detectors were equipped with large areas of Silicon Microstrip detectors with excellent position resolution $\mathcal{O}(10) \; \mu\mathrm{m}$.
# 
# 

# 
# | |
# | :--: |
# | <img src="./imgs/ALEPH_event.png" width=800 align="center">|
# 

# 
# The LEP measurements (in MeV) [[28]](https://arxiv.org/abs/hep-ex/0509008):
# 
# | |
# | :--: |
# |<img src="./imgs/LEP_Zwidth_nus.png" width=600 align="center">|
# 

# The $e^+ \, e^- \to f \, \bar{f}$ cross section is enhanced in the $Z$ pole:
# 
# $$
# \sigma(e^+ \, e^- \to f \, \bar{f}) = \frac{12 \pi s}{m_Z^2} \frac{\Gamma_{ee} \Gamma_{f\bar{f}}}{(s - m_Z^2)^2 + m_Z^2 \Gamma_Z^2}
# $$
# 
# where:
# 
# $$
# \Gamma_{f \bar{f}} = \frac{G_F m_Z^3}{6 \sqrt{2} \pi} \left((g_V^f)^2 + (g_A^f)^2 \right)
# $$
# 
# **Question:** Consider the case $\sqrt{s} = m_Z$, and compute the width of the resonance at half of the maximum.
# 
# **Question:** Compute the decay width with $Z \to f \bar{f}$.

# 
# **Question:** Calculate $\Gamma_{\nu \bar{\nu}}$
# 
# That is:
# 
# $$
# \frac{G_F m_Z^3}{ 6 \sqrt{2} \pi} \simeq 334 \; \mathrm{MeV} \to \Gamma_{\nu\bar{\nu}} \simeq 167 \; \mathrm{MeV}
# $$

# ### Invisible width and model-independent extraction of $N_\nu$
# 
# The total Z width decomposes as:
# $$
# \Gamma_Z = \Gamma_{\rm had} + 3\,\Gamma_{\ell\bar{\ell}} + N_\nu\,\Gamma_{\nu\bar{\nu}}
# $$
# The **invisible width** is extracted from measurements without any assumption about the SM couplings:
# $$
# \Gamma_{\rm inv} = \Gamma_Z - \Gamma_{\rm had} - 3\,\Gamma_{\ell\bar{\ell}} = 499 \pm 1 \; \mathrm{MeV}
# $$
# The number of light neutrino families then follows assuming SM NC couplings for $\nu$:
# $$
# N_\nu = \frac{\Gamma_{\rm inv}}{\Gamma^{\rm SM}_{\nu\bar{\nu}}} = \frac{499 \pm 1}{166.9} = 2.984 \pm 0.008
# $$
# 
# The key observable is the **peak hadronic cross section**: adding a fourth light neutrino increases $\Gamma_Z$, which *reduces the peak height* by $\sim 2$ nb and *broadens the resonance*. LEP scanned $\sqrt{s}$ through the Z peak at five centre-of-mass energies with $\sim 7$ million Z decays, achieving sub-permille precision on $\Gamma_Z$.
# 
# **Note**: the result assumes all invisible width comes from SM neutrinos. Any light weakly-interacting particle coupling to the Z (e.g. light dark matter) would contribute to $\Gamma_{\rm inv}$ and bias $N_\nu$ high.
# 

# 
# $$
# \Gamma_Z = 2495 \pm 2, \; \Gamma_{\ell\bar{\ell}} =  83 \pm 0.1, \; \Gamma_{\rm had} = 1746 \pm 2, \; \Gamma_{\rm inv} = 499 \pm 1 
# $$
# 
# $$
# N_\nu = \frac{\Gamma_{\mathrm{inv}}}{\Gamma^{\rm SM}_{\bar{\nu}\nu}} = 2.984 \pm 0.008
# $$
# 
# In fact the cross section is distorted due to initial and final state radiation (QED calculable effects).
# 
# **Question:** Verify the lepton universality in NC with LEP the decay with $Z \to \ell^+\ell^-$.
# 
# 

# **Exercise: Z boson lineshape and number of neutrino families**
# 
# The peak cross section for $e^+e^- \to \text{hadrons}$ at the Z pole depends on $N_\nu$ because each light neutrino family adds $\Gamma_{\nu\bar{\nu}} \approx 167$ MeV to the total width, **reducing the peak height** and **broadening the resonance**:
# 
# $$
# \sigma_{\rm had}(\sqrt{s}) = \frac{12\pi s}{m_Z^2} \frac{\Gamma_{ee}\,\Gamma_{\rm had}}{(s - m_Z^2)^2 + m_Z^2\Gamma_Z^2}
# $$
# 
# **Questions:**
# 1. Compute the SM partial widths $\Gamma_{\nu\bar\nu}$, $\Gamma_{\ell\bar{\ell}}$, $\Gamma_{\rm had}$ and the total width $\Gamma_Z$ for $N_\nu = 2, 3, 4$.
# 2. Plot $\sigma_{\rm had}(\sqrt{s})$ for the three cases and compare the peak values.
# 3. Fit the simulated LEP data below to extract $N_\nu$ as a free parameter using `scipy.optimize.curve_fit`.
# 

# In[ ]:


from sm import z_lineshape
z_lineshape()


# ### NC couplings of SM fermions
# 
# Using $\sin^2\theta_W \simeq 0.2312$, the vector and axial couplings $g_V^f = T_3^f - 2Q_f\sin^2\theta_W$, $g_A^f = T_3^f$ for each SM fermion are:
# 
# | Fermion | $Q$ | $T_3$ | $g_V$ | $g_A$ | $(g_V^2 + g_A^2)$ |
# |---------|-----|--------|--------|--------|-------------------|
# | $\nu_e,\nu_\mu,\nu_\tau$ | $0$ | $+1/2$ | $+0.500$ | $+0.500$ | $0.500$ |
# | $e, \mu, \tau$ | $-1$ | $-1/2$ | $-0.038$ | $-0.500$ | $0.252$ |
# | $u, c$ | $+2/3$ | $+1/2$ | $+0.192$ | $+0.500$ | $0.287$ |
# | $d, s, b$ | $-1/3$ | $-1/2$ | $-0.346$ | $-0.500$ | $0.370$ |
# 
# *Observations:*
# 
# - Neutrinos couple **maximally** with $g_V = g_A = 1/2$ — purely left-handed, as required by the absence of $\nu_R$.
# - Charged leptons are **nearly purely axial** ($g_V^e \approx 0$) when $\sin^2\theta_W \approx 1/4$, making the forward–backward asymmetry $A_{FB} \propto g_V g_A$ a precise probe of $\sin^2\theta_W$.
# - The colour factor $N_c = 3$ for quarks gives $\Gamma(Z \to q\bar{q}) = 3\,\Gamma(Z\to\ell\bar\ell)$.
# 
# **Question:** Show that $\sin^2\theta_W = 1/4$ makes $g_V^e = 0$ exactly. What does this imply for the forward–backward asymmetry in $e^+e^- \to \mu^+\mu^-$ at the Z pole?
# 

# ## Summary
# 
# Each experiment presented in this notebook added a key piece to the construction of the Standard Model:
# 
# | Experiment | Discovery | Consequence for the SM |
# | :-- | :-- | :-- |
# | Pauli / Fermi (1930–34) | Neutrino postulated; β decay theory | Weak interaction as a 4-fermion current |
# | Cowan & Reines (1956) | First neutrino detection | Confirmed the neutrino existence |
# | Goldhaber (1958) | Neutrino helicity $= -1$ | Neutrinos are left-handed; V−A structure |
# | Lederman et al. (1962) | $\nu_\mu \neq \nu_e$ | Lepton flavour number is conserved |
# | Gargamelle (1973) | Neutral currents observed | Confirmed the $SU(2)_L \times U(1)_Y$ electroweak structure |
# | LEP (1989–2000) | $N_\nu = 2.984 \pm 0.008$ | Exactly three neutrino families in the SM |
# 
# The neutrino in the Standard Model is therefore:
# 
# - **massless**, left-handed ($\nu_L$), with negative helicity
# - **neutral**: no electric charge, no colour
# - interacts only via **CC** ($W^\pm$) and **NC** ($Z$)
# - comes in **three flavours**: $\nu_e,\, \nu_\mu,\, \nu_\tau$, each conserved separately
# - obeys **lepton universality**: identical weak couplings for all three families
# 
# But Nature tells a different story — neutrinos *do* have mass and *do* change flavour.  
# That is the subject of the next notebooks.

# ***
# 
# ## References
# 
# [1] J. Chadwick, [Nature 129 (1932) 312](https://www.nature.com/articles/129312a0)
# 
# [2] E. Fermi, Nuovo Cim. 11 (1934) 1.
# 
# [3] H. Bethe and R. Peierls, [Nature 133 (1934) 532](https://www.nature.com/articles/133532a0).
# 
# [4] See, B. Pontecorvo, Cambridge Monogr. Part. Phys. Nucl. Phys. Cosmol., 1 (1991) 25.
# 
# [5] F. Reines and C. L. Cowan, [Nature 178 (1956) 446](https://www.nature.com/articles/178446a0); C. L. Cowan et al., Science 124 (1956) 103.
# 
# [6] M. Goldhaber, L. Grodzins, A.W. Sunyar, [Phys. Rev. 109 (1958) 1015](https://link.aps.org/doi/10.1103/PhysRev.109.1015)
# 
# [7] E. Fermi, Ricerca Scientifica 2 (1933) 12; E. Fermi, [Z. Phys. 88 (1934) 161](https://link.springer.com/article/10.1007/BF01351864).
# 
# [8] F. Perrin, Comptes Rendues 197 (1933) 1625.
# 
# [9] M. Aker et al., KATRIN collaboration, Phys. Rev. Lett. 123, 221802 (2019), Nature Physics 19, 160-166 (2022), [arXiv:2105.08533](https://arxiv.org/abs/2105.08533)
# 
# [10] K. Assamagan et al., [Phys. Rev. D53 (1996) 6065](https://link.aps.org/doi/10.1103/PhysRevD.53.6065)

# [11] R. Barate et al., ALEPH collaboration, [Eur. Phys. J. C2 (1998) 395](https://link.springer.com/article/10.1007/s100529800850)
# 
# [12] N. Aghanim et al., Planck Collaboration, [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)
# 
# [13] L. Landau, [Nucl. Phys. 3 (1957) 127](https://doi.org/10.1016/0029-5582(57)90061-5).
# 
# [14] T. D. Lee and C. N. Yang, [Phys. Rev. 105 (1957) 1671](https://link.aps.org/doi/10.1103/PhysRev.105.1671).
# 
# [15] A. Salam, [Nuovo Cim. 5 (1957) 299](https://link.springer.com/article/10.1007/BF02812841).
# 
# [16] S. L. Glashow, [Nucl. Phys. 22 (1961) 579](https://doi.org/10.1016/0029-5582(61)90469-2).
# 
# [17] S. Weinberg, [Phys. Rev. Lett. 19 (1967) 1264](https://link.aps.org/doi/10.1103/PhysRevLett.19.1264).
# 
# [18] A. Salam, Proc. of the 8th Nobel Symposium on "Elementary Particle Theory, Relativistic Groups
# and Analyticity", Stockholm, Sweden, 1968, edited by N. Svartholm, p. 367.
# 
# [19] J. C. Street and E. C. Stevenson, [Phys. Rev. 52 (1937) 1003](https://link.aps.org/doi/10.1103/PhysRev.52.1003).

# [20] S. H. Neddermeyer and C. D. Anderson, [Phys. Rev. 51 (1937) 884](https://link.aps.org/doi/10.1103/PhysRev.51.884).
# 
# [21] R. Brown et al., [Nature 163 (1949) 47](https://www.nature.com/articles/163047a0)
# 
# [22] G. Danby et al., [Phys. Rev. Lett. 9 (1962) 36](https://link.aps.org/doi/10.1103/PhysRevLett.9.36).
# 
# [23] M. L. Perl et al., [Phys. Rev. Lett. 35, 1489 (1975)](https://link.aps.org/doi/10.1103/PhysRevLett.35.1489)
# 
# [24] K. Kodama et al., DONUT Collaboration, Phys. Lett. B504, 218 (2001), [arXiv:hep-ex/0012035](https://arxiv.org/abs/hep-ex/0012035)
# 
# [25] A. M. Baldini et al. (MEG), Eur. Phys. J. C76, 8, 434 (2016), [arXiv:1605.05081](https://arxiv.org/abs/1605.05081), MEG II experiment [arXiv:2310.12614](https://arxiv.org/abs/2310.12614)
# 
# [26] A. Pich, M. Ramsey-Musolf, Particle Data Group, [Tests of Conservation Laws (2023)](https://pdg.lbl.gov/2023/reviews/rpp2023-rev-conservation-laws.pdf)
# 
# [27] F.J. Hasert et al., Gargamelle Collaboration, [Phys. Lett. B 46 (1973) 138](https://doi.org/10.1016/0370-2693(73)90499-1)
# 
# [28] LEP Electroweak Working Group, [arXiv:hep-ex/0509008](https://arxiv.org/abs/hep-ex/0509008)
