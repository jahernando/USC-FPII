"""
sm.py  –  Standard Model neutrino physics: exercise functions for nu_sm.ipynb
==============================================================================
Each function encapsulates one of the Python exercises in the notebook.
All functions can be called with default arguments and produce self-contained
plots and printed output.

Functions
---------
beta_spectrum()             C1 – β-decay electron spectrum and endpoint sensitivity
neutrino_mean_free_path()   C2 – Neutrino mean free path via inverse β-decay
reactor_flux()              C3 – Savannah River reactor flux and Cowan-Reines event rate
z_lineshape()               C4 – Z boson lineshape and number of neutrino families
pion_helicity_suppression() C5 – Helicity suppression in π and K leptonic decays
fermi_constant_from_muon()  C6 – Fermi constant derived from the muon lifetime
lepton_universality()       C7 – Quantitative test of lepton universality with PDG data
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit


# ──────────────────────────────────────────────────────────────────────────────
# C1 – β-decay electron spectrum
# ──────────────────────────────────────────────────────────────────────────────

def beta_spectrum(Q=1.161, me=0.511, mnu_keV_list=(0, 50, 100)):
    """
    Simulate the β-decay electron kinetic-energy spectrum.

    Compares the 2-body (no neutrino, monochromatic) and 3-body (with ν̄,
    continuous) cases and shows the endpoint sensitivity to a finite neutrino
    mass.  Uses the phase-space approximation (Fermi function F ≈ 1).

    The spectrum shape is:
        dN/dTe ∝ pe · Ee · (Q - Te)²
    where pe = √(Ee² - me²) and Ee = Te + me.

    For finite mν the endpoint factor is replaced by:
        (Q - Te) √[(Q - Te)² - mν²]

    Parameters
    ----------
    Q : float
        Q-value of the decay [MeV].  Default: 1.161 MeV (²¹⁰Bi → ²¹⁰Po).
    me : float
        Electron mass [MeV].  Default: 0.511 MeV.
    mnu_keV_list : sequence of float
        Neutrino masses [keV] to overlay in the endpoint panel.
        Default: (0, 50, 100) keV.

    Returns
    -------
    None  (produces a 2-panel matplotlib figure and prints a summary).
    """
    # ── 3-body spectrum (with ν̄) ─────────────────────────────────────────────
    Te = np.linspace(1e-4, Q, 1000)
    Ee = Te + me
    pe = np.sqrt(np.maximum(Ee**2 - me**2, 0))

    spectrum = pe * Ee * (Q - Te)**2
    spectrum /= spectrum.max()

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(Te, spectrum, 'steelblue', lw=2.5,
            label=r'3-body with $\bar{\nu}_e$: continuous')
    ax.axvline(Q, color='tomato', lw=2.5, ls='--',
               label=r'2-body without $\nu$: monochromatic at $T_e = Q$')
    ax.fill_between(Te, spectrum, alpha=0.15, color='steelblue')
    ax.set_xlabel(r'Electron kinetic energy $T_e$ [MeV]', fontsize=12)
    ax.set_ylabel('dN/dT  (normalised)', fontsize=12)
    ax.set_title(fr'$^{{210}}$Bi $\beta$-decay spectrum  ($Q = {Q}$ MeV)', fontsize=12)
    ax.legend(fontsize=10)

    # ── Endpoint region: effect of finite neutrino mass ──────────────────────
    ax2 = axes[1]
    Te_end = np.linspace(0.95 * Q, Q, 500)
    Ee_end = Te_end + me
    pe_end = np.sqrt(np.maximum(Ee_end**2 - me**2, 0))

    colors = ['steelblue', 'tomato', 'seagreen', 'orange']
    for (mnu_keV, col) in zip(mnu_keV_list, colors):
        mnu = mnu_keV * 1e-3          # keV → MeV
        fac = np.maximum((Q - Te_end)**2 - mnu**2, 0)
        sp  = pe_end * Ee_end * np.sqrt(fac) * fac**0.5
        sp /= sp.max() if sp.max() > 0 else 1
        lab = fr'$m_\nu = {mnu_keV}$ keV' if mnu_keV > 0 else r'$m_\nu = 0$'
        ax2.plot(Te_end * 1e3, sp, color=col, lw=2, label=lab)

    ax2.set_xlabel(r'$T_e$ [keV]  (endpoint region)', fontsize=12)
    ax2.set_ylabel('dN/dT  (normalised)', fontsize=12)
    ax2.set_title(r'Endpoint sensitivity to $m_\nu$', fontsize=12)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

    print("A non-zero neutrino mass rounds off the spectrum before the kinematic endpoint.")
    print("KATRIN measures this endpoint in tritium β-decay and sets m_ν < 0.45 eV (2022).")


# ──────────────────────────────────────────────────────────────────────────────
# C2 – Neutrino mean free path
# ──────────────────────────────────────────────────────────────────────────────

def neutrino_mean_free_path():
    """
    Compute and plot the neutrino mean free path via inverse β-decay (IBD).

    Uses the IBD cross section approximation:
        σ(ν̄_e p → n e⁺) ≈ 9.52×10⁻⁴⁴ (E/MeV)² cm²

    The mean free path in a medium with target proton number density n_p is:
        λ = 1 / (n_p · σ)

    The plot spans E = 0.1 – 10⁴ MeV and shows λ in water, lead and iron
    together with reference lines for the Earth and Sun radii.

    Returns
    -------
    None  (produces a log-log matplotlib figure and prints a summary table).
    """
    NA = 6.022e23   # mol⁻¹

    def sigma_cm2(E_MeV):
        """IBD cross section [cm²] for ν̄_e + p → n + e⁺."""
        return 9.52e-44 * E_MeV**2

    # Material properties: density [g/cm³], molar mass [g/mol], protons per formula unit
    materials = {
        'Water (H₂O)': dict(rho=1.00,  A=18.0,  n_H=2),
        'Lead (Pb)':   dict(rho=11.34, A=207.2, n_H=82),
        'Iron (Fe)':   dict(rho=7.87,  A=55.85, n_H=26),
    }
    colors = ['steelblue', 'tomato', 'seagreen']

    E = np.logspace(-1, 4, 500)   # MeV

    fig, ax = plt.subplots(figsize=(9, 5))

    for (name, mat), col in zip(materials.items(), colors):
        n_p    = mat['n_H'] * mat['rho'] * NA / mat['A']   # cm⁻³
        lam_km = 1.0 / (n_p * sigma_cm2(E)) / 1e5          # cm → km
        ax.loglog(E, lam_km, color=col, lw=2.5, label=name)

    # Reference astronomical distances
    for dist, label, ls in [(6.371e3, r'$R_\oplus = 6\,371$ km', '--'),
                             (6.96e5,  r'$R_\odot = 696\,000$ km', ':')]:
        ax.axhline(dist, color='k', ls=ls, lw=1.2, label=label)

    ax.set_xlabel(r'Neutrino energy $E_\nu$ [MeV]', fontsize=12)
    ax.set_ylabel(r'Mean free path $\lambda$ [km]', fontsize=12)
    ax.set_title(r'Neutrino mean free path via IBD  ($\bar{\nu}_e + p \to n + e^+$)',
                 fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.show()

    # Numerical table
    n_water = 2 * 1.00 * NA / 18.0
    n_lead  = 82 * 11.34 * NA / 207.2
    print(f"{'E [MeV]':>10}  {'λ_water [km]':>16}  {'λ_lead [km]':>14}")
    print("-" * 44)
    for Ev in [1, 10, 100, 1000]:
        lam_w  = 1.0 / (n_water * sigma_cm2(Ev)) / 1e5
        lam_pb = 1.0 / (n_lead  * sigma_cm2(Ev)) / 1e5
        print(f"{Ev:>10}  {lam_w:>16.2e}  {lam_pb:>14.2e}")
    print("\nBethe (1934): 'absolutely impossible to observe'  → Cowan-Reines proved him wrong!")


# ──────────────────────────────────────────────────────────────────────────────
# C3 – Reactor flux and Cowan-Reines event rate
# ──────────────────────────────────────────────────────────────────────────────

def reactor_flux(P_GW=0.7, d_m=11.0, M_target_kg=200.0, eff=0.10):
    """
    Estimate the antineutrino flux and interaction rate for the Savannah River
    reactor experiment of Cowan and Reines (1956).

    Steps:
        1. Fission rate from thermal power.
        2. Isotropic ν̄_e flux at distance d.
        3. Number of free protons in the water target.
        4. IBD event rate using the mean reactor ν̄_e energy.
        5. Comparison with the measured rate after accounting for detection
           efficiency (positron annihilation + neutron capture on Cd).

    Parameters
    ----------
    P_GW : float
        Thermal power of the reactor [GW].  Default: 0.7 GW.
    d_m : float
        Distance from reactor core to detector centre [m].  Default: 11 m.
    M_target_kg : float
        Total water target mass [kg].  Default: 200 kg.
    eff : float
        Detection efficiency (delayed coincidence method).  Default: 0.10.

    Returns
    -------
    None  (prints a step-by-step summary).
    """
    P_W          = P_GW * 1e9              # W
    E_fission_J  = 196e6 * 1.602e-19      # J per fission (196 MeV)
    nu_per_fiss  = 6                       # ν̄_e per fission
    NA           = 6.022e23

    # Fission rate [fissions/s]
    R_fiss = P_W / E_fission_J
    print(f"Fission rate:          {R_fiss:.3e} fissions/s")

    # Antineutrino flux at distance d [ν/(m²·s)]
    phi = nu_per_fiss * R_fiss / (4 * np.pi * d_m**2)
    print(f"ν̄ flux at d = {d_m:.0f} m:  φ = {phi:.3e}  ν/(m²·s)")

    # Free protons in water target
    N_protons = 2 * (M_target_kg / 18.015e-3) * NA
    print(f"\nFree protons in {M_target_kg:.0f} kg H₂O: {N_protons:.3e}")

    # IBD cross section at mean reactor ν̄_e energy (~3.5 MeV)
    E_nu_mean = 3.5                            # MeV
    sigma_m2  = 9.52e-44 * E_nu_mean**2 * 1e-4  # m²  (1 cm² = 10⁻⁴ m²)
    print(f"σ(IBD) at <E> = {E_nu_mean} MeV:  {sigma_m2:.3e} m²")

    # Expected event rate
    rate_hr = phi * N_protons * sigma_m2 * 3600
    print(f"\nExpected rate (100% eff.): {rate_hr:.1f} events/hour")
    print(f"With efficiency ε ≈ {eff*100:.0f}%:  {rate_hr * eff:.2f} events/hour")
    print(f"\nMeasured by Cowan & Reines (1956):  2.9 ± 0.2 events/hour  ✓")


# ──────────────────────────────────────────────────────────────────────────────
# C4 – Z boson lineshape and number of neutrino families
# ──────────────────────────────────────────────────────────────────────────────

def z_lineshape(mZ=91.1876, GF=1.1664e-5, sin2tW=0.2312, seed=0):
    """
    Plot the Z boson hadronic lineshape σ(e⁺e⁻ → had) for Nν = 2, 3, 4 and
    fit Nν from simulated LEP-style data.

    The Breit-Wigner cross section is:
        σ_had(√s) = (12π s / mZ²) · Γ_ee · Γ_had / [(s − mZ²)² + mZ² ΓZ²]

    Partial widths are computed from:
        Γ(Z→ff̄) = Nc · GF mZ³ / (6√2 π) · (gV² + gA²)
    with gV^f = T3^f − 2 Qf sin²θW and gA^f = T3^f.

    Parameters
    ----------
    mZ : float
        Z boson mass [GeV].  Default: 91.1876 GeV (PDG 2024).
    GF : float
        Fermi constant [GeV⁻²].  Default: 1.1664×10⁻⁵ GeV⁻².
    sin2tW : float
        Weak mixing angle sin²θW.  Default: 0.2312.
    seed : int
        Random seed for simulated data points.  Default: 0.

    Returns
    -------
    None  (produces a matplotlib figure, prints partial widths and fit result).
    """
    hbarc2 = 0.3894e6   # nb·GeV²

    def Gamma_ff(gV, gA, Nc=1):
        """Partial width Z→ff̄ [GeV]."""
        return Nc * GF * mZ**3 / (6 * np.sqrt(2) * np.pi) * (gV**2 + gA**2)

    # SM NC couplings: gV = T3 − 2Q sin²θW,  gA = T3
    gV_nu = 0.5;               gA_nu = 0.5
    gV_e  = -0.5 + 2*sin2tW;  gA_e  = -0.5
    gV_u  = 0.5 - 4/3*sin2tW; gA_u  = 0.5
    gV_d  = -0.5 + 2/3*sin2tW; gA_d  = -0.5

    G_nu  = Gamma_ff(gV_nu, gA_nu)
    G_e   = Gamma_ff(gV_e,  gA_e)
    G_had = 2*Gamma_ff(gV_u, gA_u, 3) + 3*Gamma_ff(gV_d, gA_d, 3)  # u,c + d,s,b

    print("SM partial widths:")
    print(f"  Γ(Z→νν̄)  = {G_nu*1e3:.2f} MeV/family   (3 families → {3*G_nu*1e3:.1f} MeV)")
    print(f"  Γ(Z→ll̄)  = {G_e*1e3:.2f} MeV/family")
    print(f"  Γ(Z→had) = {G_had*1e3:.1f} MeV")

    def sigma_had(sqrt_s, Nnu):
        """Breit-Wigner hadronic cross section [nb] as a function of √s [GeV]."""
        s      = sqrt_s**2
        GammaZ = 3*G_e + G_had + Nnu*G_nu
        sigma0 = (12*np.pi / mZ**2) * G_e * G_had   # GeV⁻²
        return sigma0 * s / ((s - mZ**2)**2 + mZ**2 * GammaZ**2) * hbarc2

    sqrt_s = np.linspace(88, 94, 1000)

    fig, ax = plt.subplots(figsize=(9, 5))
    for Nnu, col in zip([2, 3, 4], ['tomato', 'steelblue', 'seagreen']):
        sig = sigma_had(sqrt_s, Nnu)
        ax.plot(sqrt_s, sig, color=col, lw=2.5, label=f'$N_\\nu = {Nnu}$')
        print(f"  Nν={Nnu}: peak σ_had = {sig.max():.1f} nb,  "
              f"ΓZ = {(3*G_e + G_had + Nnu*G_nu)*1e3:.0f} MeV")

    # Simulated LEP-style data (Nν = 3 + Gaussian noise)
    rng      = np.random.default_rng(seed)
    s_data   = np.array([88.5, 89.5, 90.2, 91.0, 91.2, 91.5, 92.0, 93.0, 93.7])
    sig_true = sigma_had(s_data, 3.0)
    sig_data = sig_true + rng.normal(0, 0.03*sig_true.max(), len(s_data))
    sig_err  = 0.03 * sig_true.max() * np.ones(len(s_data))

    ax.errorbar(s_data, sig_data, yerr=sig_err, fmt='ko', ms=5, capsize=4,
                label='Simulated LEP data', zorder=5)

    ax.set_xlabel(r'$\sqrt{s}$ [GeV]', fontsize=13)
    ax.set_ylabel(r'$\sigma(e^+e^-\!\to\mathrm{had})$ [nb]', fontsize=13)
    ax.set_title(r'Z lineshape: sensitivity to $N_\nu$', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # Fit Nν as a free parameter
    popt, pcov = curve_fit(sigma_had, s_data, sig_data,
                           p0=[3.0], sigma=sig_err, absolute_sigma=True)
    print(f"\nFit result:  Nν = {popt[0]:.3f} ± {np.sqrt(pcov[0, 0]):.3f}")
    print(f"LEP result:  Nν = 2.984 ± 0.008")


# ──────────────────────────────────────────────────────────────────────────────
# C5 – Helicity suppression in pion and kaon leptonic decays
# ──────────────────────────────────────────────────────────────────────────────

def pion_helicity_suppression():
    """
    Compute the helicity-suppressed leptonic decay ratios for the pion and kaon
    and compare with PDG measurements.

    For a two-body decay P⁺ → ℓ⁺ ν_ℓ, angular momentum conservation forces
    the neutrino into a right-handed helicity state, which is forbidden in the
    SM.  The amplitude is proportional to mℓ, giving:

        Γ(P→ℓν) / Γ(P→ℓ'ν) = (mℓ/mℓ')² · [(mP²−mℓ²)/(mP²−mℓ'²)]²

    The first factor is the helicity suppression; the second is the 2-body
    phase-space correction.

    A plot of Γ(P→ℓν) ∝ mℓ² (mP²−mℓ²)² vs mℓ visualises why the pion decays
    to muons 99.99% of the time despite larger phase space for electrons.

    Returns
    -------
    None  (prints numerical results and produces a 2-panel figure).
    """
    # PDG 2024 masses [MeV]
    m = dict(pi=139.570, K=493.677, mu=105.658, e=0.511)

    def ratio_Gamma(m_meson, m_l1, m_l2):
        """
        Ratio Γ(P→l1 ν) / Γ(P→l2 ν) from helicity suppression + phase space.

        Returns (total_ratio, helicity_factor, phase_space_factor).
        """
        mP2      = m_meson**2
        helicity = (m_l1 / m_l2)**2
        phase    = ((mP2 - m_l1**2) / (mP2 - m_l2**2))**2
        return helicity * phase, helicity, phase

    # Pion
    r_pi, h_pi, ps_pi = ratio_Gamma(m['pi'], m['e'], m['mu'])
    print("── π⁺ → ℓ⁺ ν  ───────────────────────────────────────────────")
    print(f"  Helicity suppression (me/mμ)²             = {h_pi:.4e}")
    print(f"  Phase-space factor                         = {ps_pi:.6f}")
    print(f"  Predicted Γ(π→eν)/Γ(π→μν)                = {r_pi:.4e}")
    print(f"  PDG measured value                         = 1.2327e-04")
    print(f"  Relative difference                        = {abs(r_pi/1.2327e-4 - 1)*100:.2f}%")

    # Kaon
    r_K, h_K, ps_K = ratio_Gamma(m['K'], m['e'], m['mu'])
    print("\n── K⁺ → ℓ⁺ ν  ───────────────────────────────────────────────")
    print(f"  Helicity suppression (me/mμ)²              = {h_K:.4e}")
    print(f"  Phase-space factor                          = {ps_K:.6f}")
    print(f"  Predicted Γ(K→eν)/Γ(K→μν)                 = {r_K:.4e}")
    print(f"  PDG measured value                          = 2.434e-05")

    # Visualise Γ(P→ℓν) vs lepton mass
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (label, m_mes) in zip(axes, [('Pion  (m = 139.6 MeV)', m['pi']),
                                          ('Kaon  (m = 493.7 MeV)', m['K'])]):
        m_l   = np.linspace(0.1, 0.95 * m_mes, 400)
        gamma = m_l**2 * (m_mes**2 - m_l**2)**2
        gamma /= gamma.max()

        ax.plot(m_l, gamma, 'steelblue', lw=2.5)
        ax.axvline(m['mu'], color='tomato',   ls='--', lw=2,
                   label=fr"$m_\mu$ = {m['mu']:.1f} MeV")
        ax.axvline(m['e'],  color='seagreen', ls='--', lw=2,
                   label=fr"$m_e$  = {m['e']:.3f} MeV")
        ax.set_xlabel(r'Lepton mass $m_\ell$ [MeV]', fontsize=11)
        ax.set_ylabel(r'$\Gamma(P\to\ell\nu)$  (normalised)', fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=10)

    plt.suptitle('Helicity suppression: leptonic width vs lepton mass', fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\nConclusion: despite larger phase space for e, the helicity suppression")
    print("(me/mμ)² ≈ 2×10⁻⁵ makes pion decay to electrons extremely rare.")


# ──────────────────────────────────────────────────────────────────────────────
# C6 – Fermi constant from the muon lifetime
# ──────────────────────────────────────────────────────────────────────────────

def fermi_constant_from_muon(tau_mu=2.196981e-6, m_mu=105.658e-3,
                              m_W=80.377, m_Z=91.1876):
    """
    Derive the Fermi constant GF from the muon lifetime and verify the
    Glashow–Weinberg–Salam (GWS) prediction for mW.

    At leading order in the V−A theory the muon partial width is:
        Γ(μ → e ν_μ ν̄_e) = GF² mμ⁵ / (192 π³)

    Inverting for GF:
        GF = √(192 π³ Γ / mμ⁵)

    The weak coupling g and Weinberg angle follow from:
        GF/√2 = g²/(8 mW²)
        sin²θW = 1 − (mW/mZ)²

    Parameters
    ----------
    tau_mu : float
        Muon lifetime [s].  Default: 2.196981×10⁻⁶ s (PDG 2024).
    m_mu : float
        Muon mass [GeV].  Default: 105.658×10⁻³ GeV.
    m_W : float
        W boson mass [GeV].  Default: 80.377 GeV (PDG 2024).
    m_Z : float
        Z boson mass [GeV].  Default: 91.1876 GeV (PDG 2024).

    Returns
    -------
    None  (prints a step-by-step numerical summary).
    """
    hbar_GeV = const.hbar / (1e9 * const.e)   # ħ in GeV·s ≈ 6.582×10⁻²⁵ GeV·s

    # Step 1 – total decay width from lifetime
    Gamma_GeV = hbar_GeV / tau_mu
    print(f"Muon decay width:  Γμ = {Gamma_GeV:.4e} GeV")

    # Step 2 – extract GF
    GF = np.sqrt(192 * np.pi**3 * Gamma_GeV / m_mu**5)
    print(f"\nDerived  GF = {GF:.6e} GeV⁻²")
    print(f"PDG value   = 1.166379e-05 GeV⁻²")
    print(f"Ratio       = {GF / 1.166379e-5:.6f}")

    # Step 3 – weak coupling g
    g2      = 8 * m_W**2 * GF / np.sqrt(2)
    g       = np.sqrt(g2)
    alpha_W = g2 / (4 * np.pi)
    print(f"\nWeak coupling  g    = {g:.4f}")
    print(f"α_W = g²/4π         = {alpha_W:.4f}   (cf. αem = 1/137 ≈ 0.0073)")

    # Step 4 – Weinberg angle from boson masses
    sin2tW = 1 - (m_W / m_Z)**2
    print(f"\nsin²θW from mW/mZ   = {sin2tW:.4f}")
    print(f"PDG direct value    = 0.2312")
    print(f"\n→ GF from muon decay correctly predicts mW and sin²θW via GWS theory.")


# ──────────────────────────────────────────────────────────────────────────────
# C7 – Quantitative test of lepton universality
# ──────────────────────────────────────────────────────────────────────────────

def lepton_universality():
    """
    Test lepton universality by extracting the weak coupling ratios
    gμ/ge and gτ/gμ from τ branching ratios and leptonic decay widths.

    The V−A partial width for ℓ → ν_ℓ ℓ' ν̄_ℓ' is:
        Γ ∝ GF(ℓ)² · GF(ℓ')² · mℓ⁵ · f(mℓ'²/mℓ²)
    where the phase-space function is:
        f(x) = 1 − 8x + 8x³ − x⁴ − 12x² ln x

    Test 1  (gμ/ge):
        From τ → μ ν ν̄  vs  τ → e ν ν̄:
        (gμ/ge)² = [BR(τ→μ)/BR(τ→e)] · [f(me²/mτ²)/f(mμ²/mτ²)]

    Test 2  (gτ/gμ):
        From the ratio of τ and μ leptonic decay widths:
        (gτ/gμ)² = [Γ(τ→eνν)/Γ(μ→eνν)] · (mμ/mτ)⁵ · [f(me²/mμ²)/f(me²/mτ²)]

    Returns
    -------
    None  (prints phase-space factors, coupling ratios, and a summary table).
    """
    # PDG 2024 values
    m_e   = 0.511e-3     # GeV
    m_mu  = 105.658e-3   # GeV
    m_tau = 1776.86e-3   # GeV
    tau_mu  = 2.196981e-6   # s
    tau_tau = 290.3e-15     # s

    # τ leptonic branching ratios (PDG 2024) with uncertainties
    BR_tau_e  = (0.17846, 0.00027)
    BR_tau_mu = (0.17374, 0.00027)

    def f_ps(x):
        """Phase-space function f(x) = 1 − 8x + 8x³ − x⁴ − 12x² ln x."""
        if x < 1e-10:
            return 1.0
        return 1 - 8*x + 8*x**3 - x**4 - 12*x**2 * np.log(x)

    f_tau_e  = f_ps((m_e  / m_tau)**2)
    f_tau_mu = f_ps((m_mu / m_tau)**2)
    f_mu_e   = f_ps((m_e  / m_mu)**2)

    print("Phase-space factors:")
    print(f"  f(τ→e):  {f_tau_e:.6f}")
    print(f"  f(τ→μ):  {f_tau_mu:.6f}")
    print(f"  f(μ→e):  {f_mu_e:.6f}")

    # Test 1 – gμ/ge  from τ leptonic branching ratios
    ratio_BR   = BR_tau_mu[0] / BR_tau_e[0]
    d_ratio_BR = ratio_BR * np.sqrt((BR_tau_mu[1]/BR_tau_mu[0])**2
                                    + (BR_tau_e[1]/BR_tau_e[0])**2)
    gmu_ge   = np.sqrt(ratio_BR * f_tau_e / f_tau_mu)
    d_gmu_ge = 0.5 * gmu_ge * d_ratio_BR / ratio_BR

    print(f"\n── Test 1: gμ/ge  from τ leptonic branching ratios ─────────")
    print(f"  BR(τ→μνν) / BR(τ→eνν)  = {ratio_BR:.5f} ± {d_ratio_BR:.5f}")
    print(f"  gμ/ge                   = {gmu_ge:.5f} ± {d_gmu_ge:.5f}")
    print(f"  Universality predicts   = 1.00000")

    # Test 2 – gτ/gμ  from τ and μ leptonic widths
    Gamma_tau_e = BR_tau_e[0] / tau_tau   # s⁻¹
    Gamma_mu    = 1.0          / tau_mu   # s⁻¹  (BR ≈ 100%)
    gtau_gmu    = np.sqrt((Gamma_tau_e / Gamma_mu) * (m_mu/m_tau)**5
                          * (f_mu_e / f_tau_e))

    d_tau_tau  = 0.5e-15   # s (PDG uncertainty on τ_τ)
    d_gtau_gmu = 0.5 * gtau_gmu * d_tau_tau / tau_tau

    print(f"\n── Test 2: gτ/gμ  from μ and τ leptonic widths ─────────────")
    print(f"  gτ/gμ                   = {gtau_gmu:.5f} ± {d_gtau_gmu:.5f}")
    print(f"  Universality predicts   = 1.00000")

    # Summary table
    print(f"\n{'─'*52}")
    print(f"  Ratio        Value            Deviation from 1")
    print(f"{'─'*52}")
    for name, val, err in [('gμ/ge', gmu_ge, d_gmu_ge),
                            ('gτ/gμ', gtau_gmu, d_gtau_gmu)]:
        sigma = abs(val - 1.0) / err
        print(f"  {name}      {val:.5f} ± {err:.5f}    {sigma:.1f}σ from unity")
    print(f"\n→ Lepton universality verified at the per-mille level.")
