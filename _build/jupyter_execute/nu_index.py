#!/usr/bin/env python
# coding: utf-8

# # The misteries of the neutrinos

# In[1]:


import time
print(' Last version ', time.asctime() )


# *About*
# 
# These lectures are about some aspects of Neutrino Physics. 
# 
# They cover the relevance of the neutrino in the construction of the Standard Model, neutrino oscillations and double-beta decay searches.
# 

# ## Introduction
# 
# Neutrinos
# 
# - are the closest thing to nothing we have discovered.
# 
# - played an important role on the construction of the Standard Model (SM)
# 
# - have provided the **only evidence of Physics Beyond the SM** (BSM)
# 
# - are unique, neutrinos are the only fundamental fermion that can be their own antiparticle.
# 
# 
# 

# Two mayor discoveries and on one confirmation in the last decades:
# 
# - Higgs discovery ([ATLAS](https://arxiv.org/abs/1207.7214), [CMS](https://arxiv.org/abs/1207.7235)) [pdg-review](http://pdg.lbl.gov/2022/reviews/rpp2024-rev-higgs-boson.pdf)
# 
# - Confirmation of the Cabibbo-Kobayashi-Maskawa Unitary Matrix (BaBar, Belle, LHCb, ...), [pdg-review](http://pdg.lbl.gov/2024/reviews/rpp2024-rev-ckm-matrix.pdf)
# 
# - Discovery of the neutrino oscillations ([Super-Kamiokande](https://arxiv.org/abs/hep-ex/9807003), [SNO+](https://arxiv.org/abs/1109.0763)) [pdg-review](http://pdg.lbl.gov/2024/reviews/rpp2024-rev-neutrino-mixing.pdf)

# Neutrino oscillation simplified summary:
# 
# | |
# | :--: |
# | <img src="./imgs/nu_oscillation_dm2_tan2theta.png" width = 400> |
# ||

# 
# Fermions in the SM:
# 
# | |
# | :--: |
# | <img src="./imgs/teo_su3_su2l_u1y_fermions.png"> |
# ||
# 

# 
# Leptons in the SM:
# 
# $$
# L_e= \begin{pmatrix} \nu_{eL} \\ e_L \end{pmatrix}, \, 
# L_\mu = \begin{pmatrix} \nu_{\mu L} \\ \mu_L  \end{pmatrix}, \,
# L_\tau = \begin{pmatrix} \nu_{\tau L} \\ \tau_L  \end{pmatrix}; \;
# e_R, \mu_R, \tau_R 
# $$
# 
# Neutrinos in the SM:
# 
# - they only interact weakly!
# 
# $$s = 1/2, \, m = 0, \; q = 0, \;T_3 = 1/2, \; Y = -1/2$$
# 
# - they are $\nu_L$, left-chirality neutrinos, and $\bar{\nu}_R$, right-chirality anti-neutrinos. 
# 
# - neutrinos are massless! There is no $\nu_R$ to intriduce the mass in the legrangian:
# 
# $$-\mathcal{L}_{Dirac} = - m \, \bar{\nu} \nu = - m (\bar{\nu_L}  \nu_R + \bar{\nu_R} \nu_L)$$

# 
# | |
# |:--:|
# |<img src="./imgs/nu_cc_nc_feynman_rules.png" width = 600>|
# ||
# 
# 
# 
# Neutrino interacts only via this interactions lagrangian, CC ($W^\pm$) or NC ($Z$), :
# 
# $$
# \mathcal{L}_{CC} = - \frac{g}{\sqrt{2}} \sum_{\alpha = e, \mu, \tau} \bar{\nu}_{\alpha L} \gamma^\mu l_{\alpha L} W^+_\mu + \mathrm{h.c.}, \;\;\;
# \mathcal{L}_{NC} = - \frac{g}{2 \cos \theta_W} \bar{\nu}_{\alpha L} \gamma^\mu \nu_{\alpha L} Z_\mu
# $$
# 
# where $g$ is the weak constant and $\theta_W$ the Weiberg angle. 
# 
# The V-A structure is explicit in the interaction lagrangian.
# 
# 

# 
# - The lepton Number in the SM is (accidently) preserved. 
# 
# - Futhermore, Flavour Lepton Number is also accidentally preserved: electronic, muonic and tauonic.

# ### Neutrinos in Nature
# 
# Neutrinos in Nature:
# 
# - they have no color, no electric charge, and a tiny mass!
# 
# $$s = 1/2, \, m \le \mathcal{O}(eV), \; q = 0; \; T_3 = 1/2, \; Y = -1/2$$
# 
# - the $\nu_L$, left-chirality, are called 'neutrinos' and $\bar{\nu}_R$, right-chirality, 'antineutrinos'
# 
# - there are three light neutrino flavours, $\nu_e, \nu_\mu, \, \nu_\tau$.

# 
# - Neutrinos oscillates between different flavours.
# 
# - Neutrinos do no preserve flavour leptonic number, and may preserve leptonic number
# 
# - There is a mixing neutrino matrix, $U_{PMNS}$, Pontecorbo-Maki-Nakagawa-Sakata, with three angles and one CP phase (if neutrinos are Dirac).
# 
# - Their masses are unknown. But there values are very small compared with the rest of the leptons $m_\nu \ll m_e$. 
# 
# - We need to complete the SM to include neutrinos masses: either Dirac (extended SM) or Majorana (BSM).

# ### About Dirac neutrinos
# 
# 
# A Dirac spinor fulfills the Dirac equation and it can be decomposed into left and right chirality parts:
# 
# $$
# \Psi =\Psi_L + \Psi_R 
# $$
# And the parity transformation reverts them:
# 
# $$
# \mathcal{P}: \;\; \Psi_L \leftrightarrow \Psi_R
# $$
# Therefore, a parity conserving theory introduces both in equal footing.

# 
# In the chiral representation, a Dirac spinor is composed of two Weyl spinors $\psi_L, \psi_R$:
# 
# $$
# \Psi = \begin{pmatrix} \psi_L \\ \psi_R \end{pmatrix}
# $$
# 
# They admit a phase redefinition, without physical consequences
# 
# $$
# \Psi' = e^{i\theta} \Psi
# $$
# 
# This global simmetry is associated to a conserved charge, electric charge, lepton number, etc.
# 
# A mass term in the lagrangian has the form:
# 
# $$
#  - m \bar{\Psi} \Psi  =  - m \left(\bar{\Psi}_L \Psi_R + \bar{\Psi}_R \Psi_L \right)
# $$
# 
# 

# 
# As the left-chiral fields are in a duplet of $SU(2)_L$, in order to preserve the original gauge invariant, we introduce an interaction with the Higgs doublet. 
# 
# Let's consider first one family, for charged leptons, the mass term
# 
# $$
# - \mathcal{L}_{\mathrm{mass}, e} = \lambda_e \bar{L}_e \Phi \, e_R + \mathrm{h.c.}
# $$
# 
# where $\lambda_e$ is an adimensional constant, the electron Yukawa coupling.
# 
# And
# 
# $$
# L_e = \begin{pmatrix} \nu_{eL} \\ e_L \end{pmatrix}, \; 
# \Phi = \begin{pmatrix} \phi^+ \\ \phi^0 \end{pmatrix}
# $$
# 
# 
# 

# 
# After spontaneous symmetry breaking (SSB), in the unitary gauge:
# 
# $$
# \langle \Phi \rangle = \begin{pmatrix} 0 \\ \frac{v}{\sqrt{2}}\end{pmatrix}
# $$
# where $v$ is the Higgs expectation value.
# 
# We get:
# 
# $$
# \frac{\lambda_e v}{\sqrt{2}} ( \bar{e}_L e_R + \bar{e}_R e_L ) = m_e \, \bar{e} e
# $$
# 
# where:
# 
# $$
# m_e = \lambda_e \frac{v}{\sqrt{2}}
# $$

# 
# We can generate a mass for a Dirac neutrino, using the Higgs charge conjugate, in the unitary gauge:
# 
# $$
# \tilde{\Phi} = i \sigma^2 \Phi^* \rightarrow
# \langle \tilde{\Phi} \rangle = \begin{pmatrix} \frac{v}{\sqrt{2}}  \\ 0 
# \end{pmatrix}
# $$
# 
# The possible mass terms of the lagrangian is:
# 
# $$
# -\mathcal{L}_{\mathrm{mass}, \nu} = \lambda_{\nu} \bar{L}_e \tilde{\Phi} \, \nu_{eR} + h.c.
# $$
# 
# that will result on a mass term:
# 
# $$
# \lambda_\nu \frac{v}{\sqrt{2}} (\bar{\nu_L} \nu_R + \bar{\nu_R} \nu_L) = m_\nu \, \bar{\nu}_e \nu_e
# $$
# 
# 
# 

# ### About Majorana neutrinos
# 
# A Majorana spinor is a solution of the Dirac equation. 
# 
# In nature all fermions are Dirac. The only one that can be Majorana is the neutrino.
# 
# A Majorana spinor is its own charge conjugate:
# 
# $$
# \Psi_M = \Psi + \Psi^c
# $$
# 
# Where, the charge conjugation operation $\mathcal{C}$ is:
# 
# $$
# \Psi^c = -i \gamma^2 \Psi^*
# $$
# 
# If $\Psi$ is a left (right) chiral spinor $\Psi_L$, the charge conjugate, $\Psi_L^c$ is a right (left) chiral one:
# 
# 

# 
# A Majorana spinor can be constructed from a single left (right) Weyl spinor $\psi_L$, using its charge conjugate: 
# 
# $$
# \psi^c_L \to i \sigma^2 \psi^*_L
# $$
# 
# That transforms as a right-handed Weyl spinor. The Majorana spinor is:
# 
# $$
# \Psi_M = \begin{pmatrix} \psi_L  \\ i \sigma^2 \psi^*_L \end{pmatrix}
# $$
# 
# That fulfills the condition:
# 
# $$
# \Psi^c_M = \Psi_M
# $$
# And it is own antiparticle

# 
# There is no global phase symmetry for a Majorana spinor.
# 
# $$
# \psi'_L = e^{i\theta} \psi_L, \;\;\; (\psi'_L)^c = e^{-i \theta} i \sigma^2 \psi^*_L = e^{-i\theta} \psi^c_L
# $$
# 
# $$
# \Psi'_M = \begin{pmatrix} e^{i\theta }\; \psi_L  \\ e^{-i \theta} \; i \sigma^2 \psi^*_L \end{pmatrix}
# $$
# 
# And therefore there are no charge conserved. The Majorana spinor is a truly neutral particle!
# 

# The mass term for left-chiral neutrinos will be:
# 
# $$
# -\frac{m}{2} \left( \overline{\nu_L }\nu^c_L + \overline{\nu^c_L} \nu_L \right)
# $$
# 
# As $\nu_L$ are part of duplet $SU(2)_L$ in the SM, in order to preserve the gauge symmetry, the simplest term to generate the mass is to include a *double* interaction with a scalar Higgs. 
# 
# $$
# \mathcal{L}_W = - \frac{\alpha}{\Lambda} \,  \bar{L} \tilde{\Phi} \, \tilde{\Phi}^T L^c + \mathrm{h.c.} 
# $$
# 
# where $\alpha$ is a dimensionless coupling and $\Lambda$ an energy scale.
# 
# This is known as the Weinberg operator. It is the only 5-dim operation that extends the SM.    

# After the SSB. It translates to:
# 
# $$
#  \mathcal{L}_W = - \frac{\alpha v^2}{2\Lambda} (\overline{\nu_L} \nu^c_L + \overline{\nu^c_L} \nu_L)
# $$
# 
# 
# Now the neutrino smalleness of the neutrino mass cal be explained via the energy scale $\Lambda$, the coupling $\alpha$ or both.

# ### Some urgent questions
# 
#  - Is there a CP phase in the neutrino sector?
# 
#  - Is the neutrino its own antiparticle. Is Dirac or Majorana? 
# 
#  - What is the neutrino mass? Which is the neutrino mass hierarchy?
# 
#  - Why the neutrino mass is so small? Is there a new energy scale, $\Lambda$, related with neutrino masses?

# The different scale of neutrino masses compared with other fermions may be an indication of a new scale of energy.
# 
# | |
# | :--: |
# | <img src="./imgs/nu_mass_scale.png" width = 800>|
