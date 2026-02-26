#!/usr/bin/env python
# coding: utf-8

# # Are neutrinos Majorana particles?
# 

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


# *About*
# 
# * A tour to Dirac, Weyl and Majorana fermions.
# 
# * How to give masses to the neutrinos? Dirac and Majorana cases. The $U_{PMNS}$ neutrino mixing matrix.
# 
# * How to detect Majorana neutrinos? The $\beta\beta0\nu$ experiments, status and future.
# 

# ## Dirac, Weyl and Mayorana fermions

# 
# ### Dirac fermions
# 
# In the SM the charged leptons are Dirac fermions. 
# 
# They have defined masses, charges, i.e electrical charge and leptonic number. 
# 
# Charged leptons come in pair of particle and anti-particle. Anti-particles have the same mass but opposed charges.
# 
# The Dirac ecuation describes the dinamic of free half-spin fermions:
# 
# $$
# i \gamma^\mu \partial_\mu \Psi - m \Psi = 0
# $$
# 
# with mass $m$.

# The Dirac equation admits the Fourier decomposition:
# 
# $$
# \Psi(x) = \sum_{s} \int_p \left[ u_s(p) a_s(p) e^{-i p x} + v_s(p) b^\dagger_s(p) e^{+ipx} \right]
# $$
# 
# where $s = 1, 2$ for the two spin components and $u_s(p)$, and $v_s(p)$ are the Dirac spinors, composed of 4 complex elements, which depends on $\mathrm{p}$. 
# 
# The $u_s(p)$ spinor is associated to the particle, the positive energy solutions of the Dirac equation, and the $v_s(p)$ spinor is associated to the anti-particles.

# ### Weyl fermions.
# 
# The Weyl fields are eigen-states of chirality, that is, eigen-states of the $\gamma^5$ matrix.
# 
# $$
# \gamma^5 \Psi_{R/L} = \pm \Psi_{R/L}
# $$
# 
# The chiral states can be obtained using the chiral projectors:
# 
# $$
# P_R = \frac{1}{2} (1 + \gamma^5), \;\;\; P_L = \frac{1}{2} (1 - \gamma^5) \;\;\;
# $$
# 
# We decompose A Dirac field into left and right chiral fields:
# 
# $$
# \Psi = P_L \Psi + P_R \Psi = \Psi_L + \Psi_R
# $$ 
# 

# In order to deal with chirality states, we use the Chiral-Weyl representation of the Dirac matrices:
# 
# $$
# \gamma^0 = \begin{pmatrix} 0 & I \\ I & 0\end{pmatrix},
# \gamma^k = \begin{pmatrix} 0 & \sigma^k \\ -\sigma^k & 0\end{pmatrix},
# \gamma^5  = \begin{pmatrix} -I & 0 \\ 0 & I\end{pmatrix},
# C = i \gamma^2 \gamma^0 = \begin{pmatrix} i \sigma^2 & 0 \\ 0 & -i \sigma^2 \end{pmatrix}
# $$
# 

# And the Dirac spinor can be decomposed into two bi-spinor:
# 
# $$
# \Psi = \begin{pmatrix} \psi_L  \\ \psi_R \end{pmatrix}
# $$
# 
# The Weyl decomposition corresponds to the irreductible representation of the Lorentz group for Dirac spinors.
# 
# The mass term of the Dirac lagrangian mixes the two Weyl fields:
# 
# $$
# m \bar{\Psi} \Psi = m \, \bar{\psi}_L \psi_R + m \, \bar{\psi}_R \psi_L
# $$
# 
# Therefore a field to have a massive particle, needs a left and a right chiral fields.

# In the Weyl representation, the Dirac equation splits into two coupled equations:
# 
# $$
# i \gamma^\mu \partial_\mu \Psi_L - m \Psi_R = 0, \\
# i \gamma^\mu \partial_\mu \Psi_R - m \Psi_L = 0, 
# $$
# 
# We define a **Weyl spinor** if $m = 0$, in that case, both chiral fields are independent.
# 

# *Neutrinos in the SM were Weyl fermions*. There were the only fermions that were in fact not Dirac fermions!
# 
# As in the CC and NC only left-handed chiral neutrinos (and right-handle anti-neutrinos) participate.
# 
# Only two fields were necesary in the SM: 
# 
# $$
# \nu_L, \;\;\; \bar{\nu}_R
# $$

# Contrary to Dirac fermions, Weyl fermions have defined chirality and helicity.
# 
# In the case of Weyl fermions as there is no mass term, Chirality is a Lorentz and a preserved quantity. It can be used for any observer any time as a good quantity of a particle.
# 
# Helicity (the projection of the spin along the momentum direction) is a preserved quantity (as the total angular momentum conmutes the Hamiltonian, and also does the helicity $h \equiv S \cdot \hat{p}$), but it is not a Lorentz invariant as an observed who moves with a larger momentum in the same direccion that the particle sees the inverse helicity than other observer that moves with lower momentum.
# 
# But in the case of Weyl particles, as they have no mass, helicity is preserved and a Lorentz invariant quatity. It is the same for all the observers and any time.
# 

# We can resume the situation of helicity and quirality:
# 
# For Dirac fermions
# 
# | | - |  + |
# | :--:  | :--: | :--: |
# | ------ | ------------ | ------------ |
# | L | $\sim 1$ | $\mathcal{O}(m/E)$|
# | R | $\mathcal{O}(m/E)$| $\sim 1$ |
# 
# For Weyl fermions
# 
# | | + |  - |
# | :--:  | :--: | :--: |
# | L | 1 | 0 |
# | R | 0| 1 |

# As neutrinos were Weyl fermions in the SM they had, there are only two fields: $\nu_L, \; \bar{\nu}_R$, and there is a triple redundancy:
# 
# 
# Leptonic number = helicity = quirality
# 
# 

# ### Majorana fermions
# 
# Pure neutral fermions do not need to be Dirac, in fact a Dirac spinor could be redundant to describe a neutral fermion!
# 
# [Etore Majorana](https://en.wikipedia.org/wiki/Ettore_Majorana) in the 30's of the XX century proposed that neutral fermions can be *symmetric*, that is, they can be its own antiparticle.
# 
# | |
# | :--: |
# | <img src="./imgs/numass_majorana_abstract.png" width = 300> |
# 
# *"We show that it is possible to achieve complete formal symmetrization in the electron and
# positron quantum theory by means of a new quantization process. The meaning of Dirac
# equations is somewhat modified and it is no more necessary to speak of negative-energy
# states; nor to assume, for any other type of particles, especially neutral ones, the existence
# of antiparticles, corresponding to the `holes' of negative energy."*

# Majorana introduced a representation of the Dirac matrices:
# 
# $$
# \gamma^0 = \begin{pmatrix} 0 & \sigma^2 \\ \sigma^2 & 0\end{pmatrix},
# \gamma^1 = \begin{pmatrix} i\sigma^3 & 0 \\ 0 & i\sigma^3\end{pmatrix},
# \gamma^2 = \begin{pmatrix} 0 & -\sigma^2 \\ \sigma^2 & 0\end{pmatrix},
# $$
# $$
# \gamma^3 = \begin{pmatrix} i\sigma^1 & 0 \\ 0 & i\sigma^1 \end{pmatrix}, 
# \gamma^5 = \begin{pmatrix} \sigma^2 & 0 \\ 0 & -\sigma^2 \end{pmatrix}, 
# C        = \begin{pmatrix} 0 &  \sigma^2 \\ \sigma^2 & 0 \end{pmatrix}
# $$
# 
# 
# called Majorana representation, where the gamma matrices are pure imaginary.
# 

# Therefore the Dirac equation is real:
# 
# $$
# i \gamma^\mu \partial_\mu \Psi = m \Psi
# $$
# 
# And admits a real solution, $\Psi(x) = \Psi^*(x)$
# 
# That means that a particle is its antiparticle. That implies that they have no charge, they are neutral.
# 
# In Majorana words there is a *symetrization*, no need to speak of *negative-states* for neutral particles.

# The Majorana condition, in any representation of the Dirac matrices translates to:
# 
# $$
# \Psi = \Psi^c 
# $$
# 
# That is, the particle and antiparticle solutions are the same.
# 
# Where:
# 
# $$
# \Psi^c \equiv C \bar{\Psi}^T = C \gamma^0 \, \Psi^* 
# $$
# 
# 

# The carge conjugation matrix, $C$, can be obtained either by:
# 
# - Imposing the covariance of the Dirac equation with respect charge conjugation
# 
# - Changing from the Majorana representation $\tilde{\gamma}^\mu$ to another representation, $\gamma^\mu$. The two representatations are connected by a unitary matrix $U$. The $\tilde{\Psi}$ indicates that is expressed in the Majorana representation.
# 
# $$
# \gamma^\mu = U \tilde{\gamma}^\mu U^\dagger, \\
# \Psi= U \tilde{\Psi}, \;\; \Psi^c = U \tilde{\Psi}^* = U (U^\dagger \Psi)^* = U U^T \Psi^* \to C \gamma^0 = U U^T
# $$
# 
# 
# 
# 

# In the Weyl-chiral representation, a spinor:
# 
# $$
# \Psi = \begin{pmatrix} \psi_L \\ \psi_R \end{pmatrix}
# $$
# 
# transforms under C:
# 
# $$
# \Psi^c = C \gamma^0 \Psi^* = i \gamma^2 \Psi^* = \begin{pmatrix} 0 & i \sigma^2 \\ -i \sigma^2 & 0 \end{pmatrix} \begin{pmatrix} \psi^*_L \\ \psi^*_R \end{pmatrix} = \begin{pmatrix} i \sigma^2 \psi^*_R \\ -i \sigma^2 \psi^*_L \end{pmatrix}
# $$

# Therefore the charge conjugate of a left-chiral spinor is a right-chiral spinor:
# 
# $$
# \Psi_L = \begin{pmatrix} \psi_L \\ 0 \end{pmatrix} \to
# (\Psi_L)^c = \begin{pmatrix} 0 \\ -i \sigma^2 \psi^*_L \end{pmatrix}
# $$
# 
# 
# Explicitly:
# 
# $$
# -i \sigma^2 = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
# $$
# 

# We can be construct a Majorana spinor starting from any $\Psi$ spinor:
# 
# $$
# \Psi^M = \Psi + \Psi^c
# $$
# 
# Now:
# 
# $$
# \Psi^M = (\Psi^M)^c
# $$
# 

# Therefore the Majorana spinor formed with from a Weyl $\psi_L$ bi-spinor is:
# 
# $$
# \Psi^M = \Psi + \Psi^c 
# $$
# $$
# \Psi^M =  \begin{pmatrix} \psi_L \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ -i \sigma^2 \psi^*_L \end{pmatrix}  = 
# \begin{pmatrix} \psi_L \\ -i \sigma^2 \psi^*_L \end{pmatrix} 
# $$
# 
# That implies that a Majorana spinor has left and right chiral components.
# 

# The last equation shows that a Majorana fermion has:
# 
#  - left and right chiral components 
#  
#  - a mass, otherwise we obtain back the Weyl equation.

# In addition, a Majorana spinor has not the ambiguity of a global phase.
# 
# If we transform $\psi_L \to e^{i\phi} \psi_L$, that implies:
# 
# $$
# \Psi^M \to \begin{pmatrix} e^{i\phi} \, \psi_L \\ e^{-i\phi} (-i \sigma^2 \psi^*_L) \end{pmatrix}
# $$
# 
# There is no $U(1)$ global symmetry that can be associated to a Majorana fermion.
# Therefore a Majorana fermion has no associated charge. 
# 
# It is a pure neutral fermion: It has not electrical charge, no lepton number.
# 
# In Majorana's words: Majorana solutions can only describe pure neutral fermions!

# The Lagrangian of a Majorana fermion is:
# 
# $$
# \mathcal{L}_M =  \frac{1}{2} \bar{\Psi} i \gamma^\mu \partial_\mu \Psi - \frac{1}{2} m \bar{\Psi} \Psi
# $$
# 
# With the fact $1/2$, The Euler equation is the Majorana equation. If we express $\Psi$ as a function of $\psi_L$,
# we obtain the Euler equation deriving with respect $\psi_L$. 
# 
# The $1/2$ factor reflects the fact that the Majorana fermion is constructed from a bi-spinor, $\psi_L$, even if we express it as a four-spinor, $\Psi$.

# We can express the Majorana mass term as a function of the Weyl spinor $\psi_L$.
# 
# $$
# -\mathcal{L}_m = \frac{1}{2} m (\overline{\psi_L} \, (\psi_L)^c + \overline{(\psi_L)^c} \, \psi_L)
# $$
# 
# Notice that again the mass term mixes the left and chiral componentes. The right component is nos associated to $(\psi_L)^c$.

# The Majorana field, in 4 componentes, admits the Fourier decomposition:
# 
# $$
# \Psi^M = \sum_s \int_p  \left[ u_s(p) a_s(p) e^{-i p \cdot x} + v_s(p) a^\dagger_s(p) e^{i p \cdot x} \right]
# $$
# 
# Notice that with the last expression:
# 
# $$
# \Psi^M = (\Psi^M)^c
# $$
# 
# Charge conjugation switches the left and right terms of the Fourier decomposition.

# As a Majorana fermion has mass we have lost the one to ane relation, and we are back to the situation:
# 
# 
# For Dirac and Majorana:
# 
# | | - |  + |
# | :--:  | :--: | :--: |
# | -- -- | ------------ | ------------ |
# | L | $\sim 1$ | $\mathcal{O}(m/E)$|
# | R | $\mathcal{O}(m/E)$| $\sim 1$ |
# 
# When we call a neutrino, if Majorana is negative helicity, what we usually call 'antinuetrino' has positive helicity
# 

# ## Relation of neutrino mixing and masses. 

# ### Dirac case
# 
# The mass for a Dirac fermion enters in the free lagrangian as:
# 
# $$
# -\mathcal{L}_m = m \bar{\Psi} \Psi
# $$
# 
# If we decompose the field in left and right chiral components, $\Psi = \Psi_L + \Psi_R$, that implies:
# 
# $$
# -\mathcal{L}_m = m \bar{\Psi}_L \Psi_R + m \bar{\Psi}_R \Psi_L
# $$
# 
# Therefore, the mass terms connects, or needs, both quiralities.
# 

# ### Dirac masses in the SM: one family
# 
# Nevertheless the SM splits the fields by quirality and consider them different. In addition, the SM postulates that bare fermions have no bare masses.
# 
# The SM locates the left chiral fields into douplets and the right ones into singlets of $SU(2)_L$.
# 
# To give masses to the weak bosons, the SM introduces the Higgs field as a doublet of 2 complex fields, in a mexican hat potential, that after SSB, it reduces to a single scalar field, $H(x)$, an in the process it providing masses to the $W^\pm, Z$.
# 
# The Higgs field adter SSB converto into two parts:
# 
# $$
# \Phi(x) = \begin{pmatrix} \phi^+(x) \\ \phi^0(x) \end{pmatrix} \rightarrow \frac{1}{\sqrt{2}} \begin{pmatrix} 0 \\ v \end{pmatrix}  + \begin{pmatrix} 0 \\ H(x) \end{pmatrix}
# $$
# 
# One which is *universal*, associated to the vacuum energy of the field, will give rise to the femions masses and the other, the Higgs field itself, to the Higgs Physics.
# 
# 

# The SM also provides masses to the leptons. To do so, it couples the Higgs doublet with the left-handed lepton doublet and add then with the scalar right field, therefore the interation term is an scalar under the SM symmetry groups.
# 
# For one lepton family, where $l = e, \mu, \tau$:
# 
# $$
# L \equiv \begin{pmatrix} \nu_L \\  l_L \end{pmatrix}, \;\; l_R
# $$
# 
# The term in the lagrangian:
# 
# $$
# -\mathcal{L}_H = \lambda^l \, (\bar{L}  \Phi) \, l_R + \mathrm{h.c.}   
# $$
# 
# Where $\lambda^l$ is the Yukawa constant (which has no dimensions)

# This term docuples into the Higgs field, $H(x)$ interaction with the charged lepton; and the charged lepton mass.
# The first gives the Higgs Physics with leptons.
# 
# as after SSB we obtain, the term associated with the vacuum value of the Higgs is:
# 
# $$
# \left( \bar{L} \Phi \right) = \frac{v}{\sqrt{2}} \, l_L
# $$
# 
# We obtain back the mass term:
# 
# $$
# -\mathcal{L}_m = \frac{\lambda^l \, v}{\sqrt{2}} \, (\bar{l}_L l_R + \bar{l}_R l_L)
# $$
# 
# where:
# 
# $$
# m_l = \lambda^l \frac{v}{\sqrt{2}}
# $$
# 
# is the charged lepton mass.
# 
# The mass is defined via the Yukawa constant $\lambda^l$ and the vacuum Higgs energy $v$. Each charged lepton has its own Yukawa coupling.

# In order to introduce mass to the neutrinos, we have to introduce first the *sterile* right-chiral field $\nu_R$ as a singlet.
# 
# And introduce a new interaction term in the lagrangian of the leptons with the Higgs, using the Higgs conjugate, and after SSB.
# 
# $$
# \tilde{\Phi} = - i \sigma^2 \Phi^* \rightarrow  \tilde{\Phi} = 
# \frac{1}{\sqrt{2}} \begin{pmatrix} v \\ 0 \end{pmatrix} + \begin{pmatrix} H(x) \\ 0 \end{pmatrix}
# $$
# 
# With a new Yukawa coupling associated to the $\nu$
# 
# $$
# -\mathcal{L}_H = \lambda^{\nu} (\bar{L} \tilde{\Phi}) \, \nu_R + \mathrm{h.c}
# $$
# 
# This introduces the questions about the need of a $\nu_R$ field which only purpose in Nature is provide the mass to the neutrino.

# Notice that now, the interaction term after SSB gives:
# 
# $$
# \left(\bar{L} \tilde{\Phi}\right) \to \frac{v}{\sqrt{2}}\bar{\nu}_L
# $$
# 
# 
# This term generates the Higgs interaction with the neutrino (not discussed here) and the neutrino mass:
# 
# $$
# -\mathcal{L}_m = \lambda^\nu \frac{v}{\sqrt{2}} \bar{\nu}_L \nu_R + \mathrm{h.c.}
# $$
# 
# where now
# 
# $$
# m_\nu = \lambda^\nu \frac{v}{\sqrt{2}}
# $$
# 
# And the uncorfortable question arises:  *Why neutrinos have a mass at least 6 order of magniture less than its charged lepton parners*?
# 
# $$
# \lambda^\nu \ll \lambda^l
# $$

# ### Dirac masses in the SM. Several families. 
# 
# In the SM there are three flavour families: $e, \mu, \tau$, and a general Yukawa interaction could imply mixing between them.
# 
# Consider the three families lepton flavour states, $l'_i,\nu'_i, \; i=1,2,3$, the left-chiral components doublets $L'_i$ and the right-chiral singlets $l'_R$, including $\nu'_{i, R}$:
# 
# $$
# L'_i = \begin{pmatrix} \nu'_{iL} \\ l'_{iL} \end{pmatrix}, \;\;\; l'_{iR}, \nu'_{iR}; \;\;\; i = e, \mu, \tau.
# $$
# 
# We can consider in a general way the coupling of the fields between *different flavors*.
# 
# $$
# -\mathcal{L}_H = \sum_{i,j=1}^3 \lambda^l_{ij} (\bar{L}'_j \Phi) l'_{iR} + \lambda^\nu_{ij} (\bar{L}'_j \tilde{\Phi}) \nu'_{iR} + \mathrm{h.c.}
# $$
# 

# If we introduce the notation:
# 
# $$
# {\bf l}_{L/R} = \begin{pmatrix} e \\ \mu \\ \tau \end{pmatrix}_{L/R}, \;\;
# {\bf \nu}_{L/R} = \begin{pmatrix} \nu_e \\ \nu_\mu \\ \nu_\tau \end{pmatrix}_{L/R}, 
# $$
# 
# After SSB, we can write the mass lagrangian term in a compact form:
# 
# $$
# -\mathcal{L}_m = \frac{v}{\sqrt{2}} \left( {\bf \lambda}_l \bar{{\bf l}}'_L {\bf l'}_R + \lambda_\nu \bar{{\bf \nu}}'_L {\bf \nu}'_R + \mathrm{h.c.} \right)
# $$
# 
# where $\lambda_l, \, \lambda_\nu$ are $3\times3$ complex matrices.

# Furthermore, the $\frac{v}{\sqrt{2}}\lambda_\alpha$ matrix as is a complex matrix -thanks to a theorem- can be diagonalized with real values in the diagonal, $M_\alpha$, via a two unitary transformations $V_\alpha, \,U_\alpha$, with $\alpha = l, \, \nu$:
# 
# $$
# \frac{v}{\sqrt{2}} \lambda_\alpha = U^\dagger_\alpha \, M_\alpha V_\alpha
# $$
# 
# The $U_\alpha, V_\beta$ relate the chiral weak states to the mass states via:
# 
# $$
#   l_L = U_l \, l'_L, \,\; \nu_L = U_\nu \, \nu'_L \\
#   l_R = V_l \, l'_R, \,\; \nu_R = V_\nu \, \nu'_R
# $$
# 
# And we finally obtain the lagrangian mass term associated to each particle of the fields:
# 
# $$
# -\mathcal{L}_m =   M_l \, \bar{l}_L l_R + M_\nu \, \bar{\nu}_L \nu_R + \mathrm{h.c.} 
# $$
# 
# Where now the states $l, \nu$ are mass eigen-states, and the matrices $M_l, M_\nu$ are $3\times3$ diagonal matrices with the masses of the leptons and neutrinos in the diagonal.
# 
# 
# 
# 
# 

# ### How the mix affect the NC and the CC?
# 
# Does the mix of mass and weak states affect the CC and NC?
# 
# First, it does not affect the NC. Notice that the NC part of the lagrangian is not altered!
# 
# $$
# - \mathcal{L}_{NC} =  \frac{g}{2 \cos \theta_W} \bar{\nu}'_{L} \gamma^\mu \nu'_{L} Z_\mu = 
# \frac{g}{2 \cos \theta_W} \bar{\nu}_{L} \gamma^\mu \, (U^\dagger_\nu U_\nu) \, \nu_{L} Z_\mu
# $$
# 
# And $(U^\dagger_\nu U_\nu) = I$ as $U_\nu$ is a unitary matrix.

# But it does affect the CC lagrangian term:
# 
# $$
# - \mathcal{L}_{CC} = 
# \frac{g}{\sqrt{2}} \bar{l}'_{L} \gamma^\mu \nu'_L W^+_\mu + \mathrm{h.c.} = 
# \frac{g}{\sqrt{2}} \bar{l}_{L}  \gamma^\mu (U^\dagger_l U_\nu) \, \nu_L W^+_\mu + \mathrm{h.c.}
# $$
# 
# The CC are affected by the mixing matrix $U$ is:
# 
# $$
# U = U^\dagger_l \, U_\nu
# $$
# 
# This matrix is known as *Pontecorbo-Maki-Nakawaga-Sakata* matrix, $U_{PMNS}$, and it is reponsible of the neutrino-oscillations.
# 

# As only the CC are affected by the mixing, in an effective way we can asociate the mix between the flavour and mass states in leptons only two the neutrinos, via the $U$ (PMNS) mixing matrix.
# 
# $$
# \nu' = U \, \nu 
# $$
# 

# In the quark sector a similar mechanims happens. It generates the $V_{CKM}$, the *Cabibbo-Kowayashi-Maskawa* matrix.
# 
# In that case, the matrix is associated to the fields of the bottoms quarks:
# 
# $$
# {\bf b}' = V \, {\bf b}
# $$
# 
# The $V_{CKM}$ is reponsible of a reach complex physics in the hadronic sectors, in particular, the complex face of the matrix, generates CP violation physics in the hadronic sector.
# 
# Both physics are know as *Flavour Physics*.

# ### Parameters of the $U_{PMNS}$
# 
# This a unitary $3\times3$ matrix has $9$ parameters, $3$ mixing angles, and $6$ phases. 
# 
# We can use the global symmetries of the Dirac fields to re-absorbe some phases of ${\bf U}$:
# 
# $$
# l_\alpha \to e^{i\phi_\alpha} l_\alpha, \;\;\; \nu_k \to e^{i \phi_k} \nu_k
# $$
# 
# The CC lagrangian is:
# 
# $$
# \sum_{\alpha, k} \bar{l}_{\alpha L} e^{-i\phi_\alpha} U_{\alpha k} \gamma^\mu e^{i\phi_k} \nu_{k L} \, W^+_\mu + \mathrm{h.c.}
# $$
# 
# We can take a global factor:
# 
# $$
# e^{-i(\phi_e-\phi_1)} \sum_{k,\alpha} \bar{l}_{\alpha L} \gamma^\mu \, ( e^{-i(\phi_\alpha-\phi_e)} U_{\alpha k } \gamma^\mu e^{i(\phi_k-\phi_1)} ) \, \nu_{k L} W^+_\mu + \mathrm{h.c.}
# $$
# 
# In total we can reabsorb $5$ phases. Therefore there is only one phase left: $\delta_{CP}$, the CP violation phase.
# 

# The usual parameterization of $U_{PMNS}$ in terms of 3 angles and one CP-phase:
# 
# | |
# | :--: |
# |<img src="./imgs/Umatrix_dirac.png" width = 600>| 

# ## Majorana mass
# 
# The term in the lagrangian associated to a Majorana mass, given that the Majorana spinor is constructed from a $\nu_L$ field, is:
# 
# $$
# -\mathcal{L}_m = \frac{1}{2} m ( \overline{\nu_L} (\nu_L)^c +  \overline{(\nu_L)^c} \nu_L ) 
# $$
# 
# 
# But in the SM to obtain an scalar term with the field $\nu_L$, we need to introduce the coupling between the L douplet ant the $\tilde{\Phi}$ Higgs conjugate. After SSB:
# 
# $$
# \bar{L} \tilde{\Phi} \to \frac{v}{\sqrt{2}} \overline{\nu_L}  
# $$
# 

# The lagrangian Majorana mass term is:
# 
# $$
# -\mathcal{L}_m = \frac{\alpha}{\Lambda} (\bar{L} \tilde{\Phi}) \, (\tilde{\Phi}^T L^c ) + \mathrm{h.c.}
# $$
# 
# Where $\alpha$ is a un-dimensional coupling (similar to Yukawa) and $\Lambda$ is an unknown *large* energy scale
# 
# This term couples 4 fields, two from fermions and two from a scalar. It has 5-dim in $[E]^5$ 
# 
# $$
#     [L] = [E]^{3/2}, \;\;\; [\Phi] = [E]
# $$
# 
# This term is known as the Weinberg operator.
# 

# We do not know the energy scale $\Lambda$, but if we take $\alpha \sim 1$, and $v = 246$ GeV, and $m_\nu \sim 1$ eV, we get:
# 
# $$
# \Lambda \sim 10^{14} \;\; \mathrm{GeV}
# $$
# 
# The fact the Majorana neutrinos adquire mass via a different mechanism that Dirac fermions do, can explain why neutrinos has a smaller mass then their parners charged leptons.

# We should remember that the SM lagrangian has only 4-dim, renormalizable, terms. 
# 
# We can always interpret the SM as the effective low energy theory of a complete theory realized at a large energy scale $\Lambda$.
# 
# The effects of the larger theory are present at low energy via n-dim operators:
# 
# $$
# \mathcal{L}_{eff} = \mathcal{L}_{SM} + \sum_{d = 5}^n \delta \mathcal{L}_{d}
# $$
# 
# The Weinberg operator [Wein] is the only 5-dim operator in this expansion. It is the first effect of this general theory at "low" energy.

# The Weinberg operator can be represented as the following Feynman diagram:
# 
# 
# | | |
# | :--: | :--: |
# |<img src="./imgs/majo_weinberg_feynman.png" width = 300>| <img src="./imgs/majo_seesaw_feynman.png" width = 300>| 
# | Weinberg operator diargam| A realization of the a see-saw model |
# 
# The operator can be realized or interpreted as the interchange of a heavy particle, in the right diagram, via a $N$ massive extra sterile right-chiral majorana neutrino. 
# 
# In these models, known as see-saw, the larger the mass of the intercharged particle, the smaller is the mass of the SM neutrino.
# 

# ### Majorana mass, several families
# 
# In the Majorana case, we extend the SM with the Weinberg 5-dim operator, to the 3 families:
# 
# $$
# - \mathcal{L}_m =  \sum_{\beta, \beta'} \frac{\alpha^\nu_{\beta\beta'}}{\Lambda} \,  (\bar{L'_\beta} \tilde{\Phi}) \, (\tilde{\Phi}^T L'^c_{\beta'}) + \mathrm{h.c.} 
# $$
# 
# where $\alpha$ is a complex coupling and $\Lambda$ an energy scale.
# 
# After SSB:
# 
# $$
# -\mathcal{L}_m = \frac{v^2}{2 \Lambda} \left(\alpha \, \bar{\nu}'_L {\nu'_L}^c + \mathrm{hc} \right)
# $$
# 
# Now, the  $3\times3$,  $\frac{v}{\sqrt{2} \Lambda} \alpha_\nu$ matrix is symmetric and complex, and it can be diagonalized to a mass matrix $M^M_\nu$ with only a unitary $U_\nu$ matrix.
# 
# $$
# \frac{v^2}{2\Lambda}\alpha_\nu = U^\dagger_\nu M^M_\nu U^*_\nu
# $$

# The mixing matrix $U$ have now 3 angles 3 phases. Because there are no global phases that can be extracted from the Majorana neutrinos fields, and therefore we can only reabsorb the 3 phases associated to the charged lepton Dirac fields. Finally, there are 3 phases left: $\delta_{CP}$, and two Majorana phases: $\eta_1, \eta_2$.
# 
# The Majorana mixing matrix is parametized as:
#     
# | | 
# | :--: |
# |<img src="./imgs/Umatrix_majorana.png" width = 600>|
#     
# With $\theta_{ij} \in [0, \pi/2]$ and $\delta, \eta_{1,2} \in [0, 2\pi)$
# 
# *Question*: Show that the Majorana phases $\eta_{1,2}$ have no effect in neutrino oscillations.

# ## Are Neutrinos Majorana?
# 
# 
# | |
# | :--: |
# | <img src="./imgs/pidecay_majorana_dirac.png" width = 600> |
# 
# How we can experimentally distinguish a Majorana neutrino?
# 
# If neutrino is Majorana, a neutrino created in a $\pi^+$ decay whould be able to interact and to produce either $\mu^+$ or $\mu^-$

# In the SM neutrinos are Weyl spinor, in that case we have a triple degenerancy: Lepton number, quirality and helicity!
# 
#  * A neutrino is a left-chiral field and has negative helicity.
# 
#  + An antineutrino is a right-chiral field and has positive helicity.
# 
# Either or the three quantities gives the same information.
# 
# |  |  |  |  |
# | :--: | :--: | :--: | :--: |
# | | L | Q | H|
# |$\nu_L$ | +1 | L |  -1 |  
# |$\bar{\nu}_R$ | -1 | R |  +1 |  

# With mass, we can simply say that the SM neutrino has negative helicity and anti-neutrino positive helicity, with the 'correct' of 'wrong' helicity.
# 
# A Dirac neutrino is clasified by its lepton number. A Majorana neutrino can have both helicities. 
# 
# 
# A Dirac neutrino is created in a CC from a lepton, and is destroyed in a CC creating back a lepton. If Dirac, it can not create an antilepton to preserve lepton number. If Majorana, there is a $\mathcal{O}((m/E)^2)$ probability that creates the antilepton!
# 
# We have the curse of:
# 
# $$
# \mathcal{O}((m/E)^2)
# $$
# 

# 
# |  |  |  |  | |
# | :--: | :--: | :--: | :--: |:--: |
# | --  --| -- L -- |-- Q -- | -- H -1 --| -- H +1 -- |
# |$\nu_L$ | - | L |  $\sim 1$ | $\mathcal{O}(\frac{m}{E})$ |
# |$\nu_R$ | - | R |  $\mathcal{O}(\frac{m}{E})$ | $\sim 1$ |  
# 
# 
# The suppression factor in the transition rate is squared supressed:
# 
#  * For a $m_\nu \sim 1$ eV and $E \sim 1$ MeV, $\to 10^{-12}$
# 
#  * For a $m_\nu \sim 1$ eV and $E \sim 1$ Gev, $\to 10^{-18}$
# 
# Experimentally almost impossible!
# 
# Majora and Dirac are *experimentally* almost identical due to the small mass of the neutrino.
# 

# ### Double beta decay without neutrinos
# 
# | |
# | :--: |
# | <img src="./imgs/bb0nu_Feynman.png" width = 400> |
# 
# 
# The only experimentally accesible experiment is the hypothetical vary rare double beta decay *without neutrinos*.
# 
# 

# The three mass neutrinos participate in the process, each have with a suppresed coeficient:
# 
# $$
# U^2_{ei} \frac{m_i}{E}
# $$
# 
# As the *electron neutrino* is decomposed in 3 massive neutrinos and each one enter as *neutrino* in one $\beta$ vertex and as *anti-neutrino* in the other.
# 
# The transition amplitude is proportional to
# 
# $$
# \mathcal{A} \propto  m_{\beta\beta} \equiv \sum_i U^2_{ei} \, m_i
# $$
# 
# the Majorana mass, $m_{\beta\beta}$.
# 
# Note that the Majorana mass depends on the first row of the neutrino $U$ Majorana mixing matrix and the neutrino masses.
# 

# The $\beta\beta2 \nu$ (with neutrinos) was proposed [1] by [M. Goeppert-Mayer](https://en.wikipedia.org/wiki/Maria_Goeppert-Mayer) in 1935.
# 
# $$
# (A, Z) \to (A, Z+ 2) + 2 e^- + 2 \bar{\nu}_e + Q_{\beta\beta}
# $$
# 
# This is a second order decay, that happens in 35 isotopes, where single $\beta$ is kinematically forbiden.
# 
# In particular in: $^{48}\mathrm{Ca}$, $^{76}\mathrm{Ge}$,  $^{82}\mathrm{Se}$, $^{100}\mathrm{Mo}$, $^{130}\mathrm{Te}$ and $^{136}\mathrm{Xe}$. 
# 
# The half-life is quite large $\mathcal{O}(10^{19})$ yr

# | |
# | :--: |
# |<img src="./imgs/bb2nu_isotopes_T12.png" width = 500 > |
# 

# 
# In 1939 W. Furry proposed [2] the hypothetical decay $\beta\beta 0\nu$ if $\nu$ are Majorana
# 
# $$
# (A, Z) \to (A, Z+ 2) + 2 e^-  + Q_{\beta\beta}
# $$
# 
# The half-life time of this decay is:
# 
# $$
# \left(T^{0\nu}_{1/2}\right)^{-1} = G^{0\nu} \, \left| M^{0\nu} \right|^2 \, m^2_{\beta\beta}
# $$
# 
# Where $G^{0\nu}$ is the phase factor, $\left| M^{0\nu} \right|^2$, the nuclear matrix elemment (NME) squared, and
# 
# $$
# m_{\beta\beta} =  \sum_i U^2_{ei} \, m_i
# $$
# 
# is the effective majorana mass which depends of the elements of the mixing matrix $U_{ei}$ and the neutrino masses $m_i$.
# 

# About the phase factor:
# 
# |  |  |
# | :--: | :--: |
# |<img src="./imgs/bb0nu_table_G0nu_Qbb.png" width = 350>| <img src="./imgs/bb0nu_isotope_table_Qbb.png" width = 350>|
# 
# Notice:
# $$
# G^{0\nu}/m^2_e \sim \, \mathcal{O}(10^{-26}) \;\; 1/ (\mathrm{y \, eV}^2)
# $$
# 
# The best isotope will be the one with the largest phase space and largest Q, and with a larger *natural* abundance.
# 
# There is no *best isotope*. The current experiments use as a targer, mostly, $^{136}\mathrm{Xe}$ and $^{76}\mathrm{Ge}$.

# About the majorana mass:
# 
# The majorana mass, $m_{\beta\beta}$ depends on the $U_{\alpha i}$ matrix elements, the mass of the lightest neutrino, $m_0$, and $\Delta m^2_{32}, \; \Delta m^2_{21}$
# 
# In NH:
#     
# $$
# m_{\beta\beta} = \left| m_0 c^2_{12} c^2_{13} + \sqrt{m^2_0 + \Delta m^2_{21}} \, s^2_{12} c^2_{13} e^{2i (\eta_2 -\eta_1)} + \sqrt{m^2_0 + \Delta m^2_{32} + \Delta m^2_{21}} \, s^2_{13} e^{-2i(\delta_{CP} + \eta_1)}\right|
# $$
# 
# In IH:
# 
# $$
# m_{\beta\beta} = \left| m_0  s^2_{13} + \sqrt{|m^2_0 - \Delta m^2_{32}|} \, s^2_{12} c^2_{13} e^{2i (\eta_2 + \delta_{CP})} + \sqrt{|m^2_0 - \Delta m^2_{32} - \Delta m^2_{21}|} \, c^2_{12} c^2_{13} e^{2i(\eta_1 + \delta_{CP})}\right|
# $$
# 
# where $m_0$ is the mass of the lightest neutrino.

# In the limit $m_0 \to 0$
# 
# in NH:
# 
# $$
# m_{\beta\beta} \simeq \left| \sqrt{\Delta m^2_{21}} \, s^2_{12} c^2_{13} + \sqrt{\Delta m^2_{32}} \, s^2_{13} e^{-2i(\delta_{CP} + \eta_2 )} \right| \simeq 1.1 - 4.2 \; \mathrm{meV}
# $$
# 
# In IH:
# 
# $$
# m_{\beta\beta} \simeq \ \sqrt{|\Delta m^2_{32}|} \, c^2_{13} \left| s^2_{12} + \, c^2_{12} e^{-2i(\eta_2 - \eta_1)}\right| \ge \sqrt{|\Delta m^2_{32}|} \, c^2_{13} \cos^2 2\theta_{12} \simeq 15 - 50 \; \mathrm{meV}
# $$
# 

# | |
# | :--: |
# | <img src="./imgs/mbb_vs_mlight.png" width = 400> |
# 
# The majorana mass vs the lightest neutrino for Normal (red) or Inverted (blue) Hierarchy.
# 
# Values of $U_{\alpha i}$ from NuFit group 2019.
# 
# The objective of the next generation of $\beta\beta0\nu$ experiments is to cover the IH region.

# 
# | |
# | :--: |
# | <img src="./imgs/bb0nu_NME.png" width = 400> |
# 
# The matrix element has a large uncertainty theoretical error (see [3]).
# 
# It translates in a large uncertainty on $T^{0\nu}_{1/2}$ for $m_{\beta\beta} = 1$ meV (bottom plot). $M^{0\nu} \sim \, \mathcal{O}(1)$

# *Question:* Compute $T^{0\nu}_{1/2}$ for $^{136}\mathrm{Xe}$ for $m_{\beta\beta} = 50$ meV.

# In[1]:


G0nu, M0nu, mbb = 5.5e-26, 3, 50e-3
T0nu = 1./(G0nu * M0nu**2 * mbb**2)
print('life-time {:1.2e} y'.format(T0nu))


# ### $\beta\beta$ experimental signature
# 
# 
# | |
# | :--: |
# |<img src="./imgs/bb2nu_bb0nu_spectrum.png" width = 400> |
# 
# 
# - Signature: A mono-energetic spectrum of the two electrons which energy is $Q_{\beta\beta}$
# 

# Given the fact the the $T^{0\nu}_{1/2}$ is very large, $\mathcal{O}(10^{25})$ y, we approximate, the number of expected $\beta\beta2\nu$ events:
# 
# $$
# N_{\beta\beta} = \ln(2) \frac{N_A}{W} \frac{a \epsilon M t}{T^{0\nu}_{1/2}}
# $$
# 
# where $N_A$ is Avogradro's number, $W$ is the molar mass, $a$ the isotopic abundance, $\epsilon$ the detection efficiency in the RoI, $t$ the time of exposure, $M$ the target mass and $T^{0\nu}_{1/2}$ the half-live.
# 
# The number of events depends on the *exposure*: $M \, t$ (ton y)

# *question:* Compute the number of events expected in 100 kg y  of $^{136}\mathrm{Xe}$ at 90% abundance vs the $T^{0\nu}_{1/2}$ 

# In[4]:


NA, acc, eff, W = 6.02e23, 0.9, 1., 136.
T0nu, M, t = 1e26, 1e5, 1
nbb = np.log(2.) * NA * acc * eff * M * t / (W * T0nu)
print('Nbb = {:6.2f} events'.format(nbb))


# The signal is identified:
# 
#   - energy resolutiion, Region of Interest (RoI), $\Delta E = \mathrm{FWHM} = 2{\sqrt {2\ln 2}}\;\sigma \simeq 2.355 \;\sigma$
#   
#   - other discrimination variables: i.e tracks (2 $\beta$), pulse shape, ...

# The background comes from different sources (mostly gamma interactions in the detector):
#  
#   - cosmogenic muons (spalation). Detector installed underground and  veto-system. 
#   
#     * Reduction: $\mathcal{O}(10^6)$
#  
#   - natural radioactivity (U, Th radio-active chains). Ultra radio-pure materials and handeling.
#   
#     * Activities: $\mathcal{O}(10^{-6})$ Bq/kg, tipical activities 1-100 Bq/kg.
#   
#   - $^{220-222}\mathrm{Rn}$ natural radioactivity. Abatement systems, degasing of materials
#   
#   - neutrons from the rock (activation). Detector inner shielding.

# 
# | | |
# | :--: | :--: |
# | <img src="./imgs/Th232_chain.png" width = 300 align='left'> | <img src="./imgs/U238_chain.png" width = 300 align='right'> !

# 
# For $^{136}\mathrm{Xe}$, the isotopes $^{208}\mathrm{Tl}$ y $^{214}\mathrm{Bi}$ are the main contamination, due to the  $\gamma$ similar in energy to $Q_{\beta\beta}$.

# | |
# | :--: |
# | <img src="./imgs/muflux_undergroundlabs.png" width = 400 align='center'> |
# 
# Water equivalent meters of the main underground laboratories.

# The background events depends in the background-index, $b$, in counts/(ton yr keV).
# 
# And the RoI (FWHM, $\Delta E$) in keV
# 
# $$
# N_{bkg} \propto b \, \Delta E \, M \, t
# $$
# 
# While the signal
# 
# $$
# N_{sig} \propto M \, t 
# $$

# 
# The sensitivity, $S^{0\nu}$, of an experiment, that is the $Z$ number of sigmas, of the $\beta\beta0\nu$ events above the fluctuation of number of background events has two domains:
# 
#  i) background free experiment
#  
# $$
#  \mathcal{S}(T^{0\nu}_{1/2}) \propto \epsilon \, M \, t
# $$
#  
#  ii) expected background index, $b$
#  
# $$
# \mathcal{S}(T^{0\nu}_{1/2}) \propto \frac{N_{sig}}{\sqrt{N_{bkg}}} = \sqrt{\frac{M \, t}{b \Delta E}}
# $$
#  
#  *Question*: What is the sensitivity to $m_{\beta\beta}$? An increase of a factor 100 is mass is a factor 10 in $T^{0\nu}_{\beta\beta}$ and $\sqrt{10}$ in $m_{\beta\beta}$!

# 
# | |
# | :--: |
# | <img src="./imgs/Xe_mbb_vs_exposure_bkginRoI.png" width = 400 align='center'> |
# 
# Possible limits on $m_{\beta\beta}$ of a perfect Xe experiment for different bkg-index in RoI [[4]](https://arxiv.org/abs/1502.00581).
# 
# To cover the IH allowed region (grey area) we aim for a 1 ton detector, 1 counts/(RoI ton yr). 
# 
#   - Next Generation Experiment holy grail!

# *question*: Show the dependence of $S(T^{0\nu}_{1/2})$ as a function of $b$ index and exposure.

# The main ingredientes for a $\beta\beta$ detector are:
#     
#    - large exposure of the target isotope, $M \, t$
#     
#    - excelellent energy resolution, small $\Delta E$
#     
#    - ultra radiopure detector and  extra handles to reject the bacground, very small $b$
#     
#     
# For the next generation experiments we aim:
# 
#    - $M \, t \; \mathcal{O}(1)$ ton yr, $\Delta E \; \mathcal{O}(1-0.1)$ %, and $b \; \mathcal{O}(1)$ c/(ton RoI yr) 

# ## Search for Majorana Neutrinos
# 
# 
# We present the results of the three main experiments:
# 
#   * EXO
#   
#   * KamLAND-Zen
#   
#   * GERDA/LEGEND
# 

# ### EXO-200  (2011-2015)
# 
# 
# - Located at [WIPP](https://www.wipp.energy.gov) MN-USA
# 
# - 175 kg enriched 80% $^{136}\mathrm{Xe}$
# 
# - Symmetric Liquid Xenon TPC with charge readout (ionization) in the anodes and light collection (scintillation) with APDs
# 
# - Inside a cryostat, shielded by a lead castle and protected with a muon veto
# 

# 
# | | |
# | :--: | :--: |
# | <img src="./imgs/EXO200_detector.png" width = 350 align='center'>  | <img src="./imgs/EXO200_photo.png" width = 450 align='center'> |
# | Exo200 drawing| Exo200 detector |

# | | |
# | :--: | :--: |
# | <img src="./imgs/EXO200_detector_operation.png" width = 300 align='center'> | <img src="./imgs/EXO200_calibration.png" width = 300 align='center'> |
# | Exo technique | Calibration |
# 

# Principle of operation: 
# 
#   - interaction in $^{136}\mathrm{Xe}$ produce scintillation light (detected by the APDs) and ionization electrons.
#    
#   - electrons drift in the E field of the TPC (~300 V/cm)
#   
#   - they are collected in the wires at the anode
#   
# Calibration using $^{228}\mathrm{Th}$ with a peak at 2.6 MeV. Scintillation and ionization light are correlated
# 
# Signal is mostly a single point deposition (SS) while background ($\gamma$) can produce several depositions (MS)

# Main characteristics of EXO analysis:
# 
# - Total exposure $234.1$ kg yr 
# 
# - $Q_{\beta\beta} = 2458$ keV, $^{136}\mathrm{Xe} \to ^{136}\mathrm{Ba} + 2 e^-$
# 
# - Energy resolution: 3% FWHM 
#     
#     - 1.23 % $\sigma$ energy resolution phase-I and  1.15 % $\sigma$ phase-II
# 
# - bkg-index $1.7\times 10^{-3}$ c/(kg keV yr) phase-I and $1.9\times 10^{-3}$ c/(kg keV yr), phase-II
# 
#     - separation of Multi-Site (mostly background) and Single-Site (mostly signal) events 
# 

# 
# | |
# | :--: |
# | <img src="./imgs/EXO200_energy_spectrum.png" width = 700 align='center'> |
# 
# 
# With 324.1 kg y exposure, EXO-200 established a limit $\mathcal{L}(T^{0\nu}_{1/2}) \gt 3.5 \times 10^{25}$ y at 90% CL, with an expected sensitivity $\mathcal{S}(T^{0\nu}_{1/2}) \gt 5 \times 10^{25}$, complete dataset (2019) [[5]](https://arxiv.org/abs/1906.02723)
# 
# Its translates to a range in $m_{\beta\beta}$ 93 - 286 meV (using [3])

# ### KamLAND-Zen (2011-2018)
# 
# 
# - Located the Kamioka mine, re-use of KamLAND detector
# 
# - Inner transparent balloon 3 m diameter with LS and 300 kg of $^{136}\mathrm{Xe}$
# 
# - Base mass 300 kg Xenon
# 
# - Total exposure 504 kg yr
# 
# - $Q_{\beta\beta} = 2458$ keV, $^{136}\mathrm{Xe} \to ^{136}\mathrm{Ba} + 2 e^-$
# 
# - calibration with $^{228}\mathrm{Th}$
# 
# - Poor energy resolution 11 % FWHM
# 
# - bkg-index $1.6 \times 10^{-4}$ counts/(kg yr keV) dominated by muon spallation and $\beta\beta2\nu$
# 
#     - in phase-I LS was contaminated with $^{110m}\mathrm{Ag}$ and required 18 months of LS purification.
# 
# 

# | | 
# | :--: |
# | <img src="./imgs/KamLAND-zen_detector.png" width = 400 align='center'> |
# 
# $^{136}\mathrm{Xe}$ is disolved in LS inside a transparent thin 3 meter balloon.
# 
# Energy is measured from the scintillating light and the position with the light distribution in PMTs.
# 
# The energy resolution is *poor* 11% FWHM
# 
# Target mass is large, 300 kg

# 
# | |
# | :--: |
# | <img src="./imgs/KamLAND-zen_event_distribution.png" width = 400 align='center'> |
# 
# Position of evens in RoI in $z$ and $r^2$, radius squared, in the balloon.
# 
# Most of the events are close or originating from the balloon wall (black line).
# 
# Only events inside a 1 m sphere (dashed line) are accepted.

# 
# | | | 
# | :--: | :--: |
# | <img src="./imgs/KamLAND-zen_energy_spectrum.png" width = 350 align='center'>  | <img src="./imgs/KamLAND-zen_energy_spectrum_zoom.png" width = 350 align='center'>  |
# 
# Energy spectum KamLAND-zen (2016) [[6]](https://arxiv.org/abs/1605.02889)
# 
# With  504 kg yr, KamLAND-zen established a limit $\mathcal{L}(T^{0\nu}_{1/2})  \ge 10.7 \times 10^{25}$ yr at 90% CL, with a sensitivity  $\mathcal{S}(T^{0\nu}_{1/2}) > 5.6 \times 10^{25}$ yr
# 
# It translates to a range in $m_{\beta\beta}$ 61 - 165 meV
# 

# | |
# | :--: |
# | <img src="./imgs/KamLAND-zen_mbb_vs_mlightest.png" width = 350 align='center'> |
# 
# Limits on $m_{\beta\beta}$ vs $m_{light}$ imposed by KamLAND-Zen [6] vs the mass of the lightest neutrino.

# ### KamLAND-Zen 800 (2019-):
# 
#   * 745 kg $^{136}\mathrm{Xe}$
#   
#   * 2 times large balloon and better radiopure materials
#   * reduction $^{12}\mathrm{C}$, $^{136}\mathrm{Xe}$ spallation by data analysis
#   * current exposure: 970 kg yr  
#   * Expected: Improve KamLand-Zeon 400 by a factor 4 in 5 y, ($m_{\beta\beta} < 20-80$ meV)
# 

# 
# | | | 
# | :--: | :--: |
# | <img src="./imgs/KamLANDZEN800_Edistribution.png" width = 400 align='center'>  | <img src="./imgs/KamLANDZEN800_bkg.png" width = 300 align='center'>  |
# | spectrum | bkg estimation in RoI|
# 
# [KamLAND-Zen 800 2022](https://arxiv.org/abs/2203.02139)
# 
# 

# | |
# | :--: | 
# | <img src="./imgs/KamLANDZEN800_mbb.png" width = 300 align='center'>  |
# 
# * with 2.0970 t yr exposure, (combined with previous KamLANDZEN results) they reach a limit $\mathcal{\tau}^{\beta\beta0\nu}_{1/2} > 3.8 \times 10^{26}$ yr at 90% C.L, and
# $m_{\beta\beta} \;\; 28-122$ meV.
# 
# [8] [arXiv:2406.11438](https://arxiv.org/abs/2406.11438)

# ### Gerda (2015-2020)
# 
# - 37 High Purity Ge diodes (87% $^{76}\mathrm{Ge}$): 
# 
#    - 15.6 kg Coaxial detectors (CD). Phase-I
#    
#    - 20 kg of Broad Energy Germanium (BEG). Phase-II 35 kg 
# 
# - Located in strings with very low mass and radio-pure electronics
# 
# - inside a cryogenic bath of 63 $\mathrm{m}^3$ of Liquid Argon. Detector are instrumented with a courtain of light-fibers with SiPM readout
# 
# - in a water tank 590 $\mathrm{m}^3$ instrumented with PMTs
# 
# - detector at LNGS, Italy
# 

# | | |
# | :--: | :--: |
# |<img src="./imgs/GERDA_detector.png" width = 250 align='left'> | <img src="./imgs/GERDA_detector_strings.png" width = 250 align='right'> |
# | Gerda drawing | Gerda veto+Ge detectors|

# 
# 
# - base mass 35 kg
# 
# - $Q_{\beta\beta} = 2039$ keV, $^{76}\mathrm{Ge} \to ^{76}\mathrm{Se} + 2e^-$
# 
# - weekly calibration with $^{228}\mathrm{Th}$
# 
# - **Excellent energy resolution** 0.2 % FWHM
# 
# - bkg-index $5.6^{+3.4}_{-2.4}\times 10^{-4}$ c/(kg y keV) for BEGe detectors and $5.7^{+4.1}_{-2.6}\times 10^{-4}$ c/(kg y keV) CD. 
# 
#    - **Almost background free experiment**.

# 
# | | | |
# | :--: | :--: | :--: |
# | <img src="./imgs/GERDA_Gesensors_photo.png" width = 200 align='left'>  | <img src="./imgs/GERDA_sensor_operation.png" width = 250 align='right'> |  <img src="./imgs/GERDA_detection_scheme.png" width = 200 align='center'> |
# | sesor | operation | discrimination |
# 

# | |
# | :--: |
# | <img src="./imgs/GERDA_energy_spectrum_phaseII.png" width = 600 align='center'> |
#  Energy spectum of phase-II GERDA (2019) [[7]](https://arxiv.org/abs/1909.02726) |
# 
# With  82.4 kg yr, GERDA established a limit $\mathcal{L}(T^{0\nu}_{1/2}) \gt 9 \times 10^{25}$ yr at 90% CL, with a sensitivity  $\mathcal{S}(T^{0\nu}_{1/2}) > 11 \times 10^{25}$ yr
# 
# With 103.7 kg y, final GERDA result (2020), [[7b]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.252502), impossed a limit of $T^{0\nu}_{1/2}) > 1.8 \times 10^{26}$ y.
# 
# Its translates to a range in $m_{\beta\beta}$ 104 - 228 meV

# | |
# | :--: |
# | <img src="./imgs/GERDA_mbb_vs_mlight_exclusion.png" width = 700 align='center'> |
# 
# GERDA (2019) [[6]](https://arxiv.org/abs/1909.02726) $m_{\beta\beta}$ limit vs the mass of the lightest neutrino, the sum of neutrino masses and the effective neutrino electron mass.

# #### LEGEND200
# 
# LEGEND 200 (kg) is the upgrade of GERDA (and the fussion of the MAJORANA collaboration)
# 
# There are plans for the upgrate to 1000 kg ton, but funding pending!
# 
# A factor 5 in background level compared with GERDA
# 
# First data taking with 143 kg in LNGS during 2023, re-starting data taking in 2025
# 
# Capability of different cuts to reduce the background:
# 
# | | |
# | :--: | :--: |
# | <img src="./imgs/LEGEND_scheme.png" width = 300 align='center'> |<img src="./imgs/LEGEND_string.png" width = 200 align='center'> |
# | LEGEND scheme | picture of the Ge strings |
# 
# 
# Neutrino 2024

#  *  200 kg of enriched Ge (×5 yr), in GERDA cryostat
#  *  Taking physics data since March 2023 with 142 kg of enrGe
#  *  b  $2\times  10^{−4}$ cts / (keV kg yr)
#  *  Sensitibity $\tau_{1/2}^{\beta\beta0\nu} > 10^{27}$ yr
# 

# Capability of different cuts to reduce the background:
# 
# | |
# | :--: |
# | <img src="./imgs/LEGEND_cuts_drawing.png" width = 500 align='center'> |
# | <img src="./imgs/LEGEND_pulseshape.png" width = 500 align='center'> |
# 
# 
# Neutrino 2024

# | |
# | :--: |
# | <img src="./imgs/LEGEND_Espectrum_cuts.png" width = 500 align='center'> |
# | Blind analysis - E espectrum after different cuts| 
# 
# Neutrino 2024

# | | |
# | :--: | :--: |
# | <img src="./imgs/LEGEND_Espectrum_fit.png" width = 500 align='center'> | <img src="./imgs/LEGEND_blind24.png" width = 300 align='center'> |
# | Energy spectrum fit | Zoom in the RoI (blind analysis)|
# 
# 
# Neutrino 2024

# | | 
# | :--: | 
# | <img src="./imgs/LEGEND_box24.png" width = 300 align='center'> |
# 
# Preliminary results:
#  * Exposure: 48.3 kg yr (golden)
#  * 1 event in RoI (1.5 $\sigma$), $ b= 5.3 \times 10^{-4}$ cnts/(keV kg yr)
#  * Combined result with GERDA, Majorana previous results: $\tau^{\beta\beta0\nu}_{1/2} = 1.9 \times 10^{26}$ yr, while the median sensitivity is $2.8 \times 10^{26}$ yr
#  

# ### NEXT Experiment (2017-)
# 
# - 100 kg $^{136}\mathrm{Xe}$ High Pressure Gas 
# 
#    - Excellent resolution < 1 % FWHM at Qbb (2.48 MeV)
# 
#    - NEXT-White ~10 kg NEXT-White prototype (2017-2022)
# 
#    - NEXT-100, ~100 kg (2024-)
# 
# - Detectors located at [Canfranc Laboratory](https://lsc-canfranc.es), Spain

# | | 
# | :--: |
# | <img src="./imgs/NEXT_soft.png" width = 400 align='center'> | 

# Concept:
#     
#   1. Scintillation emission (S1), $t_0$ time
#   
#   2. Secondary electrons drift towards the anode (1 mm/ 1 $\mu$s)
#   
#   3. EL light emission, (S2), detector at energy plane (energy) and in tracking plane (position)
#     
# 

# Detector:
# 
# - An asymmetric Time Projection Chamber (600 V/cm)
# 
#    - Cilinder (1 m length, 1 m diameter)
# 
#    - Cathode: PMTs plane (light detector, energy measurement)
#    
#    - Anode: SiPMs in a fine grid (1 x 1 cm$^2$), light collection near source, tracking reconstruction
#       
# - Electro-Luminiscence region at the Anode
# 
#    - $\times 10$ E field, to convert secondary electrons to light, linear production
# 
#    - Light is proportional to secondary electrons
#    
#    - Number of secondary electrons are proportional to energy interaction
# 

# | | 
# | :--: |
# | <img src="./imgs/NEXT100_xsection.png" width = 400 align='center'> |

# | | | 
# | :--: | :--: |
# | <img src="./imgs/NEXT_topology_tracks.png" width = 400 align='center'> |  <img src="./imgs/NEXT_blobs_discrimination.png" width = 250 align='center'> |
# 

# | | |
# | :--: | :--: |
# |<img src="./imgs/NEXT_Tbb0nu_vs_Expo.png"     width = 300 align='left'>  | <img src="./imgs/NEXT_Tbb0nu_vs_BkgIndex.png" width = 300 align='right'> |
# 
# 
# Background index $= 4 \times 10^4$ counts/keV kg y and Exposure = 275  kg y. [NEXT-1]

# | | | |
# | :--: | :--: | :--: |
# |<img src="./imgs/NEW_xsection.png"            width = 500 align='center'> |<img src="./imgs/NEW_foto_energy_plane.png"   width = 480 align='left'> |<img src="./imgs/NEW_foto_tracking_plane.png" width = 480 align='right'>| 
# 
# [NEXT-2]

# | | | 
# | :--: | :--: |
# | <img src="./imgs/NEW_foto.png" width = 400 align='center'>  |  <img src="./imgs/NEW_in_platform.png" width = 300 align='center'> |
# | NEXT-White detector | Hall-A LSC NEXT installation|

# | | | 
# | :--: | :--: |
# | <img src="./imgs/NEW_LR_events.png" width = 350 align='center'> |  <img src="./imgs/NEW_blobs_de.png" width = 350 align='center'> |
# | events in NEXT-White | blob energy distributions in NEXT-White |
# 
# 
# Reconstructed double and single electrons using NEW calibration data [NEXT-3] [NEXT-4].

# ### NEXT100 (2024-
# 
# Is a 100 kg Xe HPTPC operating at the LSC.
# 
# A demonstrator for a larger 1 ton detector, estimate background level, improved topology signal, energy resolution at a large scale detector
# 
# | | 
# | :--: |
# | <img src="./imgs/NEXT100_detector.png" width = 350 align='center'> |  
# | NEXT100 detector |
# 

# ## NEXT generation of $\beta\beta0\nu$ experiments
# 
# There is a large list of proposal and next-generation experiments for $\beta\beta0\nu$
# 
# They use different techniques.
# 
# They target is to cover the IH allowed $m_{\beta\beta}$ >15 meV
# 
# reach $\mathcal{O}(10)$ meV with different isotopes and techniques.
# 
# In general: ton detectors, with BI < 1 c / (ton FWHM y)
# 

# 
# | |
# | :--: |
# | <img src="./imgs/bb0nu_future_experiments.png" width = 600 align='center'> |
# 
# from [GG21-Guilliani]

# **Liquid Scintillator** detectors:
# 
#  - KamLAND-zen 800 kg (2019-), ton (2025?) $^{136}\mathrm{Xe}$, at Kamioka, Japan
#  
#  - SNO+ (2019-), 1330 kg $^{130}\mathrm{Te}$ at SNOLab, Canada
#  
# **Time Proyection Chambers** :
# 
#  - nEXO (202?), 5 tons, Liquid $^{136}\mathrm{Xe}$
#  
#  - NEXT-White (2018-2021), NEXT-100 (2024-), -ton (2028-?), 100 kg (ton) High Preasure $^{136}\mathrm{Xe}$ at Canfranc, Spain

# 
# **Germanio**:
# 
#  - LEGEND-200, ton (GERDA and Majorana collaborations) $^{67}\mathrm{Ge}$ at LNGS
# 
#  - LEGEND-100 $^{67}\mathrm{Ge}$ at SNOLab.
# 
# 
# **Bolometers**:
# 
#  - CUORE, 206 kg $^{130}\mathrm{Te}$ bolometers, $\Delta E = 7.7 \pm 0.5$ keV FWHM [11] at LNGS, ITay
#  
#  - CUPID (CUPID-0 wiht $^{82}\mathrm{Se}$) with bolometers and scintillators [12]
#  
#  - AMoRE ($^{100}\mathrm{Mo}$, $^{60}\mathrm{Ca}$) with bolometers and scintillators, at Yangyang, S. Korea [13].
# 
# Tracking detectors:
# 
#  - NEMO (different isotopes) [14]. 
# 
# 
# Note: The dates with $?$ indicate proposed experiments not yet approved or funded.

# ### KamLAND2-Zen 
# 
#   * Planned
# 
#   * large source (> ton of $^{136}\mathrm{Xe}$), more ambitious in the future
# 
#   * Reduce $^{214}\mathrm{Bi}$ (tagging events) and $\beta\beta2\nu$ bkg (improve energy resolution),
# 
#   * brighter ($5 \times $), $ 2 \times $ better energy resolution $\sigma_E = 2.5$ \% at $Q_{\beta\beta}$.
# 
#       * Improvement in PMT QE (20%->30%)
#       
#   * 5y Sensitivity: $3 \, 10^{27}$ y, $m_{\beta\beta} < 14-37$ meV
#   

# | |
# | :--: |
# | <img src="./imgs/KamLAND2_fig.png" width = 200 align='center'> |
#   
#   

# |            | $\Delta E$ (keV) |  --- BI (c/(keV kg y)--- | --- Exposure (kg y) ---| ------- $T_{1/2}$ y ------| --- $m_{\beta\beta}$ eV --- |
# | :-- | :--: | :--: | :--: | :--: | :--: |
# | **KamLAND-Zen 400** |  172 (7%)|   $1.6 \times 10^{-4}$       |          350    |  $1.07 \, 10^{26}$ |  60-160 |
# | KamLAND-Zen 800     | 7%       |          | $745 \times 5$  |                  |  20-80  |
# | KamLAND2-Zen ton    | 2.5 %    |  | $1000 \times 10$|                  |  14-37  |
# 
# 

# ### Legend-1000
# 
# | |
# | :--: |
# | <img src="./imgs/Legend_sim_exp.png" width = 450 align='center'> |
# 
# A simulated experiments of Legend-1000 [GS21-Schönert]

# | |
# | :--: |
# | <img src="./imgs/Legend_fig.png" width = 400 align='center'> |
# 
#    
# 

# |            | $\Delta E$ (keV) |  --- BI (c/(keV kg y)--- | --- Exposure (kg y) ---| ------- $T_{1/2}$ y ------| --- $m_{\beta\beta}$ eV --- |
# | :-- | :--: | :--: | :--: | :--: | :--: |
# | **Gerda** [[7b]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.252502)  (2020)    | 3  | $5.2 \, 10^{-4}$ |          103.7       |  $1.8 \, 10^{26}$             |  79-180     |
# | Legend-200  | 3  | $2 \, 10^{-4}$   | $200 \times 5$  | $20^{27}$        | 34-78 |
# | [Legend-1000](https://arxiv.org/abs/2107.11462) | 3  | $10^{-5}$        | $1000 \times 10$| $1.3 \, 20^{28}$ | 9-21  |
# 
# 

# ### nEXO
# 
#   * Most likely it will not see the light!
#   
#   * Installation at SNOLab
#   * Based on EXO-200:
#       * Liquid Xe TPC, measurement of scintilation and charge
# 
#   * 5 tons of $^{136}\mathrm{Xe}$. 
# 
#       * cilindrical TPC $\phi = 1.3$ m, in a cryostat and Vacuum. 
# 
#   * Inprovement in ligh sensors (SiPMs) and light colection, $\sigma_E = 0.8$ % at $Q_{\beta\beta}$.
#   
#   * Improvement in radiopurity (electroformed Cu) and smaller fidutial internal region.
#   

# | |
# | :--: |
# | <img src="./imgs/nEXO_fig.png" width = 500 align='center'> |
# 
# 
# 

# |            | $\sigma_E$ (keV) |  --- BI (c/(keV kg y)--- | --- Exposure (kg y) ---| ------- $T_{1/2}$ y ------| --- $m_{\beta\beta}$ eV --- |
# | :-- | :--: | :--: | :--: | :--: | :--: |
# | **EXO-200** [[5]](https://arxiv.org/abs/1906.02723)    | 29  |  |    234.1   |  $3.5 \, 10^{25}$             |  93-286     |
# | nEXO  | 19  | $7.5 \, 10^{-5}$ (eff)   | $3281 \times 10$  | $1.35 \, 20^{28}$        | 5-15 |
# 
# 

# ### NEXT-1ton

# | | |
# | :--: | :--: |
# | <img src="./imgs/NEXT1ton_design.png" width = 350 align='center'> | <img src="./imgs/NEXT1ton_reach_vs_expon.png" width = 250 align='center'> |
# | NEX-ton drawing | NEXT-ton sensitibity|
# 
# [NEXT-5]

# ## Conclusions
# 
# The fact that neutrinos can be its one particle is one of the crucial questions still open in Particle Physics.
# 
# If neutrinos are Majorana this implies NP at a large scale.
# 
# The search of Majorana neutrinos is experimentally very challenging. The only experimental way is to observe a $\beta\beta0\nu$ decay.
# 
# Several best limit experiments KamLAND-Zen, $\tau^{2\beta0\nu}_{1/2}(^{126}\mathrm{Xe}) > 1.23 \times 10^{26}$ y at 90 % C.L. and Gerda $T^{2\beta0\nu}_{1/2}(^{76}\mathrm{Ge}) > 1.8 \times 10^{26}$ y at 90 % C.L.
# 
# Several next generation experiment is construction: CUPID, NEXT, KamLAND2-Zen, LEGEND, they will cover the IH $m_{\beta\beta}$ allowed region in the next decades.

# ----
# 
# ## References
# ---
# 
# [Majo] E. Majorana, Nuovo Cimento 14 (1937) 171-184
# 
# [Wein] S. Weinberg, "Baryon and lepton-nonconserving processes", Phys. Rev. Lettr. 43 (1979) 1566.
# 
# [1] M. Goeppert-Mayer, Phys. Rev. 48 (1935) 512.
# 
# [2] W. Furry, Phys. Rev. 56 (1939) 1184.
# 
# [3]  Engel J, Menendez J. Rept. Prog. Phys. 80:046301 (2017)
# 
# [4] "Phenomenology of neutrinoless double beta decay", J.J. Gómez-Cademas, J. Martín-Albo, [arXiv:1502.00581v2](https://arxiv.org/abs/1502.00581).
# 
# [5] G. Anton et al. (EXO-200 Collaboration), Phys. Rev. Lett. 123, 161802 (2019), [arXiv:1906.02723](https://arxiv.org/abs/1906.02723).
# 
# [6] A. Gando et al. (KamLAND-Zen), Phys. Rev. Lett. 117, 8, 082503 (2016), [Addendum: Phys.
# Rev. Lett.117,no.10,109903(2016)], [arXiv:1605.02889](https://arxiv.org/abs/1605.02889).
# 
# [7] M. Agostini et al. (GERDA), Science 365, 1445 (2019), [arXiv:1909.02726]((https://arxiv.org/abs/1902.02726)).
# 
# [7b] M. Agostini et al. (GERDA Collaboration), [Phys. Rev. Lett. 125, 252502 (2020)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.252502) 
# 
# [8] KamLAND-Zen 800 PRL (2023) [arXiv:2203.02139](https://arxiv.org/abs/2203.02139), Complete Data Set [arXiv:2406.11438](https://arxiv.org/abs/2406.11438)

# [8] KamLAND-Zen 800 PRL (2023) [arXiv:2203.02139](https://arxiv.org/abs/2203.02139), Complete Data Set [ariXiv:2406.11438](https://arxiv.org/abs/2406.11438)

# [10] S. Andringa et al. (SNO+), Adv. High Energy Phys. 2016, 6194250 (2016),
# [arXiv:1508.05759](https://arxiv.org/abs/1508.05759).
# 
# [11] C. Alduino et al. (CUORE), Phys. Rev. Lett. 120, 13, 132501 (2018), [arXiv:1710.07988](https://arxiv.org/abs/1710.07988)
# 
# [12] O. Azzolini et al. (CUPID), Phys. Rev. Lett. 123, 3, 032501 (2019), [arXiv:1906.05001](https://arxiv.org/abs/1906.05001)
# 
# [13] V. Alenkov et al., Eur. Phys. J. C79, 9, 791 (2019), [arXiv:1903.09483](https://arxiv.org/abs/1903.09483)
# 
# 
# [14] R. Arnold et al. (NEMO-3), Phys. Rev. D92, 7, 072011 (2015), [arXiv:1506.05825](https://arxiv.org/abs/1506.05825). R. Arnold et al. (NEMO-3), Phys. Rev. Lett. 119, 4, 041801 (2017), [arXiv:1705.08847](https://arxiv.org/abs/1906.05001).
# 
# [15] S. I. Alvis et al. (Majorana) (2019), [arXiv:1902.02299](https://arxiv.org/abs/1902.02299).
# 
# [16] J. Dolinski et al, (status and prospects) (2019) [arXiv:1902.04097](https://arxiv.org/abs/1902.04097).

# [NEXT-1] "Sensitivity of NEXT-100 to neutrinoless double beta decay", J. Martín-Albo et al., NEXT collaboration, JHEP05 (2016) 159, [arXiv:1411.09246](https://arxiv.org/abs/1511.09246).
# 
# [NEXT-2] "The Next White (NEW) detector", F. Monrabal et al., NEXT collaboration, JHEP10 (2019) 230, [arXiv:1804.02409](https://arxiv.org/abs/1804.02409)
# 
# [NEXT-3] "Energy calibration of the NEXT-White detector with 1% resolution near Qββ of 136Xe", J. Renner et al. NEXT Collaboration, JINST 13, 10, P10020 (2018), [arXiv:1905.13110](https://arxiv.org/abs/1905.13110).
#     
# [NEXT-4] "Boosting background suppression in the NEXT experiment through Richardson-Lucy deconvolution", A. Simon, NEXT Collaboration, [arXiv:2102.11931](https://arxiv.org/abs/2102.11931)
# 
# [NEXT-5] "Sensitivity of a tonne-scale NEXT detector for neutrinoless double beta decay searches.", C. Adams, NEXT Collaboration, [arXiv:2005.06467](https://arxiv.org/abs/2005.06467)
# 

# ### Conferences
# 
# [Nu20-Detwiler] Neutrino 2020, J. Detwiler, ["*Future neutrinoless $\beta\beta$ experiments*"](https://indico.fnal.gov/event/43209/contributions/187827/attachments/130703/159511/20200701_Nu20_FutureNDBD_Detwiler.pdf)
# 
# [GS21-Guiliani] Gran Sasso 2021,  Workshop on Future of Double Beta Decay, A. Guiliani, ["*Survey on other next generation Double Beta experiments*"](https://agenda.infn.it/event/27143/contributions/142991/attachments/85091/112800/NA-EU-workshop-Giuliani.pdf)
# 
# [GS21-Schönert] Gran Sasso 2021, Workshop on Future of Double Beta Decay, S. Schönert, ["*LEGEND-1000*"](https://agenda.infn.it/event/27143/contributions/142986/attachments/85142/112907/LEGEND-1000_for_upload.pdf)

# ---------
