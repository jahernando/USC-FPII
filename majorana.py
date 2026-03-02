"""
majorana.py — Física de neutrinos de Majorana y decaimiento doble beta sin neutrinos.

Exporta funciones de física y ejercicios interactivos para explorar:
  - La masa efectiva de Majorana m_ββ en función de la masa más ligera
  - Conversión vida media ↔ m_ββ con incertidumbre en NME
  - Sensibilidad experimental en función de la exposición
  - El espectro de tritio y el método cinético de KATRIN
  - Widget interactivo de fases de Majorana

Funciones de física:
    mbb_value(m0_eV, alpha21, alpha31, ordering, bf) — m_ββ para fases dadas
    mbb_band(m0_eV, ordering, bf, n_phases)          — banda NH/IH variando fases
    halflife_to_mbb(T_yr, isotope, g_A)              — T^{0ν}_{1/2} → m_ββ (meV)
    mbb_to_halflife(mbb_meV, isotope, g_A)           — m_ββ → T^{0ν}_{1/2}
    sensitivity_T(Mt_kgyr, BI, dE_keV, eta, eps, n_sigma) — T^{0ν}_{1/2} de sensibilidad
    meff_katrin(T_eV, Q_eV, m_nu_eV)                 — espectro beta cerca del endpoint

Ejercicios:
    exercise_mbb_spectrum(bf)        — banda m_ββ vs m_min para NH e IH
    exercise_halflife_mbb()          — T^{0ν}_{1/2} → m_ββ con incertidumbre NME
    exercise_sensitivity()           — sensibilidad vs exposición para distintos BI
    exercise_katrin_endpoint()       — espectro de tritio cerca del endpoint (Kurie)
    exercise_mbb_phases_interactive(bf) — widget interactivo con fases de Majorana
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

# ── Constantes físicas ─────────────────────────────────────────────────────────
M_E_MEV  = 0.511          # masa del electrón (MeV)
M_E_EV   = 0.511e6        # masa del electrón (eV)
NA       = 6.022e23       # número de Avogadro

# ── Parámetros de oscilación: NuFit-6.0 NH (importado de oscillations.py) ─────
BF_NUFIT60_NH = dict(
    th12   = 33.68,   th13   =  8.55,   th23   = 48.30,
    delta  = 212.0,
    dm2_21 = 7.48e-5,   # eV²  (actualizado con JUNO 2025)
    dm2_31 = 2.524e-3,  # eV²  NH
)

# ── Datos nucleares por isótopo ────────────────────────────────────────────────
# G^{0ν}: factor de espacio de fases [yr^{-1}], Kotila & Iachello (2012)
# NME   : elemento de matriz nuclear M' (sin g_A), rango de modelos nucleares
#         (mínimo, máximo, central) de la revisión Dolinski et al. (2019)
# Q_keV : energía de Q [keV]
# eta   : fracción de enriquecimiento típica en experimentos actuales
ISOTOPES = {
    '76Ge':  dict(A=76,  Q_keV=2039, eta=0.874, G0nu=2.363e-15,
                  NME=4.5, NME_min=2.8, NME_max=6.1),
    '82Se':  dict(A=82,  Q_keV=2996, eta=0.922, G0nu=10.16e-15,
                  NME=3.8, NME_min=2.7, NME_max=5.5),
    '130Te': dict(A=130, Q_keV=2528, eta=0.346, G0nu=14.22e-15,
                  NME=3.9, NME_min=2.6, NME_max=5.7),
    '136Xe': dict(A=136, Q_keV=2458, eta=0.888, G0nu=14.58e-15,
                  NME=3.0, NME_min=1.6, NME_max=4.4),
    '150Nd': dict(A=150, Q_keV=3371, eta=0.056, G0nu=63.03e-15,
                  NME=2.5, NME_min=1.5, NME_max=3.5),
}

# Resultados experimentales actuales (90 % CL) ─────────────────────────────────
EXPT_LIMITS = {
    'KamLAND-Zen 800': dict(isotope='136Xe', T_yr=3.8e26, mbb_lo=28,  mbb_hi=122),
    'LEGEND-200 comb.': dict(isotope='76Ge',  T_yr=2.8e26, mbb_lo=None, mbb_hi=None),
    'CUORE':            dict(isotope='130Te', T_yr=3.5e25, mbb_lo=70,  mbb_hi=250),
}


# ── Funciones de física ────────────────────────────────────────────────────────

def _masses_NH(m0_eV, dm2_21, dm2_31):
    """Masas (eV) para la ordenación normal con m₁ = m₀."""
    m1 = m0_eV
    m2 = np.sqrt(m1**2 + dm2_21)
    m3 = np.sqrt(m1**2 + dm2_31)
    return m1, m2, m3


def _masses_IH(m0_eV, dm2_21, dm2_31):
    """Masas (eV) para la ordenación invertida con m₃ = m₀."""
    m3 = m0_eV
    m1 = np.sqrt(m3**2 + abs(dm2_31))
    m2 = np.sqrt(m1**2 + dm2_21)
    return m1, m2, m3


def mbb_value(m0_eV, alpha21_rad, alpha31_rad, ordering='NH', bf=None):
    """
    Masa efectiva de Majorana |m_ββ| (eV) para fases de Majorana dadas.

    Parámetros
    ----------
    m0_eV       : float — masa del neutrino más ligero (eV)
    alpha21_rad : float — fase de Majorana α₂₁ (rad), ∈ [0, 2π]
    alpha31_rad : float — fase de Majorana α₃₁ (rad), ∈ [0, 2π]
    ordering    : 'NH' o 'IH'
    bf          : dict, opcional — parámetros de oscilación

    Devuelve
    --------
    float — |m_ββ| en eV
    """
    if bf is None:
        bf = BF_NUFIT60_NH
    th12 = np.radians(bf['th12']); th13 = np.radians(bf['th13'])
    Ue1 = np.cos(th12) * np.cos(th13)
    Ue2 = np.sin(th12) * np.cos(th13)
    Ue3 = np.sin(th13)

    if ordering == 'NH':
        m1, m2, m3 = _masses_NH(m0_eV, bf['dm2_21'], bf['dm2_31'])
    else:
        m1, m2, m3 = _masses_IH(m0_eV, bf['dm2_21'], bf['dm2_31'])

    term = (Ue1**2 * m1
            + Ue2**2 * m2 * np.exp(1j * alpha21_rad)
            + Ue3**2 * m3 * np.exp(1j * alpha31_rad))
    return np.abs(term)


def mbb_band(m0_eV, ordering='NH', bf=None, n_phases=2000):
    """
    Banda [mín, máx] de |m_ββ| (eV) variando las fases de Majorana α₂₁, α₃₁ ∈ [0, 2π].

    Parámetros
    ----------
    m0_eV    : array-like — masa del neutrino más ligero (eV)
    ordering : 'NH' o 'IH'
    bf       : dict, opcional
    n_phases : int — número de muestras de fases (mayor = más preciso)

    Devuelve
    --------
    mbb_min, mbb_max : ndarray — en eV
    """
    if bf is None:
        bf = BF_NUFIT60_NH
    m0 = np.atleast_1d(np.asarray(m0_eV, dtype=float))
    th12 = np.radians(bf['th12']); th13 = np.radians(bf['th13'])
    Ue1 = np.cos(th12) * np.cos(th13)
    Ue2 = np.sin(th12) * np.cos(th13)
    Ue3 = np.sin(th13)

    rng = np.random.default_rng(42)
    alpha21 = rng.uniform(0, 2 * np.pi, n_phases)
    alpha31 = rng.uniform(0, 2 * np.pi, n_phases)

    mbb_min = np.full_like(m0, np.inf)
    mbb_max = np.zeros_like(m0)

    for a21, a31 in zip(alpha21, alpha31):
        vals = np.abs(mbb_value(m0, a21, a31, ordering, bf))
        mbb_min = np.minimum(mbb_min, vals)
        mbb_max = np.maximum(mbb_max, vals)

    return mbb_min, mbb_max


def halflife_to_mbb(T_yr, isotope, g_A=1.27):
    """
    Convierte la vida media T^{0ν}_{1/2} en la masa efectiva de Majorana |m_ββ|.

    Usa: 1/T = G^{0ν} × (g_A² M')² × (m_ββ/m_e)²

    Parámetros
    ----------
    T_yr    : float o array — vida media (años)
    isotope : str — clave de ISOTOPES, e.g. '136Xe'
    g_A     : float — constante axial (defecto 1.27)

    Devuelve
    --------
    (mbb_central, mbb_min, mbb_max) en meV  [usando rango de NME]
    """
    d = ISOTOPES[isotope]
    G = d['G0nu']
    me_eV = M_E_EV

    def _mbb(NME):
        M_eff = g_A**2 * NME
        # m_ββ (eV) = m_e × sqrt(1 / (T × G × M_eff²))
        return me_eV * np.sqrt(1.0 / (np.asarray(T_yr) * G * M_eff**2)) * 1e3  # meV

    return _mbb(d['NME']), _mbb(d['NME_max']), _mbb(d['NME_min'])


def mbb_to_halflife(mbb_meV, isotope, g_A=1.27):
    """
    Convierte |m_ββ| en la vida media T^{0ν}_{1/2} (años).

    Parámetros
    ----------
    mbb_meV : float o array — masa efectiva de Majorana (meV)
    isotope : str
    g_A     : float

    Devuelve
    --------
    (T_central, T_min, T_max) en años
    """
    d = ISOTOPES[isotope]
    G = d['G0nu']
    mbb_eV = np.asarray(mbb_meV) * 1e-3

    def _T(NME):
        M_eff = g_A**2 * NME
        return 1.0 / (G * M_eff**2 * (mbb_eV / M_E_EV)**2)

    return _T(d['NME']), _T(d['NME_max']), _T(d['NME_min'])


def sensitivity_T(Mt_kgyr, BI_cts_keV_kg_yr, dE_keV,
                  eta=0.90, eps=0.85, n_sigma=1.64, A=136):
    """
    Sensibilidad a T^{0ν}_{1/2} (años) para un experimento de cero fondo o
    dominado por fondo.

    En el límite de cero fondo (B ≪ 1):
        T_sens ≃ ln2 × (N_A × η / A) × ε × Mt / n_σ

    En el límite de fondo dominante (B ≫ 1):
        T_sens ≃ ln2 × (N_A × η / A) × ε × sqrt(Mt / (BI × ΔE)) / n_σ

    Parámetros
    ----------
    Mt_kgyr            : array — exposición (kg·yr)
    BI_cts_keV_kg_yr   : float — índice de fondo [cts/(keV·kg·yr)]
    dE_keV             : float — resolución energética FWHM (keV)
    eta                : float — fracción enriquecida (∼0.90)
    eps                : float — eficiencia de señal (∼0.85)
    n_sigma            : float — nivel de CL (1.64 → 90 % CL)
    A                  : int   — masa atómica del isótopo

    Devuelve
    --------
    T_ZB, T_BG : ndarray — sensibilidad en límite cero fondo y fondo dominante (yr)
    """
    Mt = np.asarray(Mt_kgyr, dtype=float)
    factor = np.log(2) * NA * eta / A * eps / n_sigma  # mol/g × kg = 1e3 mol
    factor *= 1e3   # kg → g para N_A/A
    T_ZB = factor * Mt
    T_BG = factor * np.sqrt(Mt / (BI_cts_keV_kg_yr * dE_keV))
    return T_ZB, T_BG


def meff_katrin(T_eV, Q_eV, m_nu_eV):
    """
    Espectro diferencial de la desintegración beta de tritio cerca del endpoint.

    Aproximación relativista en la región final del espectro (|T - Q| ≪ Q):
        dN/dT ∝ (Q - T) × sqrt((Q - T)² - m_ν²) × Θ(Q - T - m_ν)

    Parámetros
    ----------
    T_eV     : array — energía cinética del electrón (eV)
    Q_eV     : float — valor Q del tritio (18 574 eV)
    m_nu_eV  : float — masa del neutrino (eV)

    Devuelve
    --------
    dNdT : ndarray — espectro diferencial (u.a.)
    """
    eps = Q_eV - np.asarray(T_eV, dtype=float)          # Q − T
    arg = eps**2 - m_nu_eV**2
    dNdT = np.where(arg > 0, eps * np.sqrt(arg), 0.0)
    return dNdT


# ── Ejercicios ─────────────────────────────────────────────────────────────────

def exercise_mbb_spectrum(bf=None):
    """
    Ejercicio: Banda de m_ββ vs masa mínima para NH e IH.

    Q1: ¿Cuál es el valor mínimo de m_ββ en NH cuando m₁ → 0?
        ¿Y en IH cuando m₃ → 0?
    Q2: ¿Para qué valor de m_min se solapan las bandas NH e IH?
        (Régimen cuasi-degenerado)
    Q3: ¿Qué cota impone KATRIN (m_eff < 0.45 eV) en el espacio de parámetros?
    Q4: ¿Puede el límite de KamLAND-Zen 800 (m_ββ < 28-122 meV) excluir ya la IH?

    Parámetros
    ----------
    bf : dict, opcional — parámetros de oscilación NuFit-6.0 NH por defecto.
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    m0 = np.logspace(-4, np.log10(0.5), 600)   # 0.1 meV → 500 meV

    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {'NH': 'C0', 'IH': 'C1'}
    labels = {'NH': 'Normal ordering (NH)', 'IH': 'Inverted ordering (IH)'}

    for ordering in ('IH', 'NH'):
        mmin, mmax = mbb_band(m0, ordering, bf)
        c = colors[ordering]
        ax.fill_between(m0 * 1e3, mmin * 1e3, mmax * 1e3,
                        alpha=0.35, color=c, label=labels[ordering])
        ax.plot(m0 * 1e3, mmax * 1e3, color=c, lw=1.5)
        ax.plot(m0 * 1e3, mmin * 1e3, color=c, lw=1.5, ls='--')

    # Límites experimentales actuales
    ax.axhline(122, color='darkgreen', lw=1.8, ls='-.',
               label=r'KamLAND-Zen 800 upper (122 meV)')
    ax.axhline(28,  color='darkgreen', lw=1.8, ls=':',
               label=r'KamLAND-Zen 800 lower (28 meV, opt. NME)')

    # Cota cosmológica: Σm < 0.12 eV → m_min < ~20 meV (NH) / ~14 meV (IH)
    ax.axvline(20, color='gray', lw=1.2, ls='--', alpha=0.7,
               label=r'Cosmological bound $\Sigma m_\nu < 0.12$ eV (NH)')

    # Futura sensibilidad de nEXO / LEGEND-1000
    ax.axhline(10, color='purple', lw=1.2, ls=':', alpha=0.8,
               label=r'Next-gen target ($m_{\beta\beta} \sim 10$ meV)')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$m_{\rm min}$ (meV)', fontsize=13)
    ax.set_ylabel(r'$|m_{\beta\beta}|$ (meV)', fontsize=13)
    ax.set_xlim(0.1, 500); ax.set_ylim(0.2, 500)
    ax.set_title(r'Effective Majorana mass $|m_{\beta\beta}|$ vs lightest neutrino mass',
                 fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout(); plt.show()

    # Respuestas numéricas
    print("── Q1: minimum m_ββ in the massless limit ──")
    mmin_NH, _ = mbb_band(np.array([1e-5]), 'NH', bf)
    mmin_IH, _ = mbb_band(np.array([1e-5]), 'IH', bf)
    print(f"  NH (m₁ → 0):  m_ββ_min ≈ {mmin_NH[0]*1e3:.1f} meV")
    print(f"  IH (m₃ → 0):  m_ββ_min ≈ {mmin_IH[0]*1e3:.1f} meV")
    print("── Q2: quasi-degenerate regime ──")
    print("  NH and IH bands merge when m_min ≳ sqrt(Δm²₃₁) ≈ 50 meV")
    print(f"  i.e. m_min ≳ {np.sqrt(bf['dm2_31'])*1e3:.0f} meV")


def exercise_halflife_mbb():
    """
    Ejercicio: Conversión vida media ↔ masa efectiva de Majorana con NME.

    Q1: Para los límites actuales de T^{0ν}_{1/2}, ¿qué límite en m_ββ se obtiene
        con el NME central, mínimo y máximo?
    Q2: ¿Por qué los experimentos de ¹³⁶Xe dan límites en m_ββ similares
        a los de ⁷⁶Ge pese a tener vidas medias comparables?
    Q3: Calcula la vida media que necesitarías para alcanzar m_ββ = 20 meV
        en cada isótopo.
    """
    # Límites de T^{0ν}_{1/2} actuales (90 % CL)
    current_T = {
        'KamLAND-Zen 800 (136Xe)': ('136Xe', 3.80e26),
        'LEGEND-200 comb.  (76Ge)': ('76Ge',  2.80e26),
        'CUORE         (130Te)':    ('130Te', 3.50e25),
        'EXO-200       (136Xe)':    ('136Xe', 3.50e25),
    }

    print("── Q1: Current T½ limits → m_ββ (meV) ──")
    print(f"{'Experiment':<30} {'T½ (yr)':>12}  {'m_ββ central':>14} "
          f"{'m_ββ min':>10} {'m_ββ max':>10}  [NME range]")
    print("─" * 85)
    for name, (iso, T) in current_T.items():
        cen, lo, hi = halflife_to_mbb(T, iso)
        print(f"{name:<30} {T:>12.2e}  {cen:>12.1f} meV"
              f"  {lo:>8.1f}   {hi:>8.1f}")

    # Gráfico: m_ββ vs T para distintos isótopos
    T_range = np.logspace(24, 28, 300)
    fig, ax = plt.subplots(figsize=(9, 6))
    styles = {'136Xe': ('C0', '-'), '76Ge': ('C1', '--'), '130Te': ('C2', '-.'), '82Se': ('gray', ':')}
    for iso, (color, ls) in styles.items():
        cen, lo, hi = halflife_to_mbb(T_range, iso)
        ax.plot(T_range, cen, color=color, ls=ls, lw=2, label=iso)
        ax.fill_between(T_range, lo, hi, color=color, alpha=0.15)

    # Límites actuales
    for exp, (iso, T) in current_T.items():
        ax.axvline(T, color='black', lw=0.8, ls=':', alpha=0.5)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$T^{0\nu}_{1/2}$ (yr)', fontsize=13)
    ax.set_ylabel(r'$|m_{\beta\beta}|$ (meV)', fontsize=13)
    ax.set_xlim(1e24, 1e28); ax.set_ylim(1, 1000)
    ax.axhline(100, color='green',  lw=1.2, ls='--', alpha=0.7, label='100 meV')
    ax.axhline(20,  color='purple', lw=1.2, ls='--', alpha=0.7, label='20 meV (IH reach)')
    ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.25)
    ax.set_title(r'$|m_{\beta\beta}|$ as a function of $T^{0\nu}_{1/2}$ — NME uncertainty band', fontsize=11)
    plt.tight_layout(); plt.show()

    # Q3
    target_meV = 20.0
    print(f"\n── Q3: Required T½ to reach m_ββ = {target_meV} meV ──")
    for iso in ('136Xe', '76Ge', '130Te', '82Se'):
        Tc, Tlo, Thi = mbb_to_halflife(target_meV, iso)
        print(f"  {iso:>6}: T½ = {Tc:.2e} yr  "
              f"(range {Thi:.1e} – {Tlo:.1e} yr due to NME)")


def exercise_sensitivity():
    """
    Ejercicio: Sensibilidad experimental a T^{0ν}_{1/2} en función de la exposición.

    Q1: ¿En qué régimen opera KamLAND-Zen 800 (BI ≈ 1.6×10⁻⁴)?
        ¿Y GERDA (BI ≈ 2×10⁻⁴) con ΔE = 3 keV?
    Q2: ¿Cómo escala la sensibilidad con Mt en el límite de cero fondo?
        ¿Y en el límite de fondo dominante?
    Q3: ¿Cuánta exposición necesita un experimento de ⁷⁶Ge con BI = 10⁻⁵ para
        alcanzar la región IH (T½ ~ 10²⁷ yr)?
    Q4: Estima la sensibilidad en m_ββ (meV) de cada escenario.
    """
    Mt = np.logspace(0, 4, 400)  # 1 – 10 000 kg·yr

    # Parámetros de experimentos representativos
    expts = [
        dict(label='KamLAND-Zen 800 (136Xe)',  iso='136Xe', BI=1.6e-4, dE=172, A=136,
             color='C0', ls='-'),
        dict(label='LEGEND-200 (76Ge)',         iso='76Ge',  BI=5e-4,  dE=3,   A=76,
             color='C1', ls='--'),
        dict(label='Future Xe (BI=1e-5)',       iso='136Xe', BI=1e-5,  dE=50,  A=136,
             color='C2', ls='-.'),
        dict(label='Future Ge (BI=1e-5)',       iso='76Ge',  BI=1e-5,  dE=3,   A=76,
             color='C3', ls=':'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for e in expts:
        T_ZB, T_BG = sensitivity_T(Mt, e['BI'], e['dE'], A=e['A'])
        T_eff = np.minimum(T_ZB, T_BG)   # realistic: take the worse of both regimes
        ax1.plot(Mt, T_BG,  color=e['color'], ls=e['ls'], lw=2,
                 label=f"{e['label']} (BG-dominated)")
        ax1.plot(Mt, T_ZB,  color=e['color'], ls=e['ls'], lw=1, alpha=0.4)

        # Convert T_eff → m_ββ
        mbb_c, _, _ = halflife_to_mbb(T_BG, e['iso'])
        ax2.plot(Mt, mbb_c, color=e['color'], ls=e['ls'], lw=2, label=e['label'])

    # Referencias
    for ax in (ax1, ax2):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'Exposure $M \cdot t$ (kg·yr)', fontsize=12)
        ax.grid(True, which='both', alpha=0.25)

    ax1.axhline(1e27, color='red', lw=1.5, ls='--', alpha=0.8,
                label=r'IH target $T^{0\nu}_{1/2} = 10^{27}$ yr')
    ax1.set_ylabel(r'$T^{0\nu}_{1/2}$ sensitivity (yr)', fontsize=12)
    ax1.set_ylim(1e24, 1e29)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.set_title('Half-life sensitivity (BG-dominated regime)', fontsize=11)

    ax2.axhline(15,  color='red',    lw=1.5, ls='--', alpha=0.8,
                label=r'IH min $m_{\beta\beta} \approx 15$ meV')
    ax2.axhline(100, color='orange', lw=1.2, ls=':', alpha=0.8,
                label=r'100 meV')
    ax2.set_ylabel(r'$|m_{\beta\beta}|$ sensitivity (meV)', fontsize=12)
    ax2.set_ylim(1, 1000)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_title(r'$m_{\beta\beta}$ sensitivity (central NME)', fontsize=11)

    plt.tight_layout(); plt.show()

    print("── Q2: Scaling of sensitivity ──")
    print("  Zero-background regime:      T½_sens ∝ Mt       (sensitivity grows linearly)")
    print("  Background-dominated regime: T½_sens ∝ √(Mt)    (sensitivity grows as √Mt)")
    print("  → Low background is crucial: reducing BI by ×10 is equivalent to ×100 exposure")

    print("\n── Q3: Required exposure to reach T½ = 10²⁷ yr ──")
    for e in expts:
        # In BG regime: Mt = (T_target × n_sigma × sqrt(BI × dE) / (factor))^2
        # Solve numerically
        Mt_vals = np.logspace(0, 6, 10000)
        _, T_BG_arr = sensitivity_T(Mt_vals, e['BI'], e['dE'], A=e['A'])
        idx = np.searchsorted(T_BG_arr, 1e27)
        if idx < len(Mt_vals):
            print(f"  {e['label']}: Mt ≈ {Mt_vals[idx]:.0f} kg·yr")
        else:
            print(f"  {e['label']}: Mt > 10⁶ kg·yr (unreachable in BG regime)")


def exercise_katrin_endpoint():
    """
    Ejercicio: Espectro de tritio cerca del endpoint y método cinético (KATRIN).

    Q1: ¿A qué energía cinética T se detecta el efecto de una masa m_ν = 0.45 eV?
        ¿Y de m_ν = 0.2 eV (sensibilidad futura)?
    Q2: ¿Por qué es tan difícil medir masas pequeñas de neutrinos en la
        desintegración beta? Estima la fracción del espectro afectada.
    Q3: Dibuja el gráfico de Kurie K(T) = √(dN/dT) para m_ν = 0, 0.45, 1 eV.
        ¿Qué diferencia observas?
    Q4: ¿Cuál es la relación entre m^{eff}_{ν_e} (KATRIN) y m_ββ (β β₀ν)?
    """
    Q = 18574.0  # endpoint del tritio (eV)
    m_vals = [0.0, 0.20, 0.45, 1.0]   # eV
    colors = ['C0', 'C2', 'C1', 'C3']
    dE_plot = 20.0   # rango de energía a representar (eV debajo del Q)

    T = np.linspace(Q - dE_plot, Q + 1, 2000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for m, col in zip(m_vals, colors):
        spec = meff_katrin(T, Q, m)
        spec_norm = spec / spec.max() if spec.max() > 0 else spec
        label = rf'$m_\nu = {m:.2f}$ eV'
        ax1.plot(T - Q, spec_norm, lw=2, color=col, label=label)

        # Kurie plot: K(T) = sqrt(dN/dT)
        kurie = np.sqrt(spec_norm)
        ax2.plot(T - Q, kurie, lw=2, color=col, label=label)

    ax1.set_xlabel(r'$T - Q_\beta$ (eV)', fontsize=12)
    ax1.set_ylabel('Spectrum (a.u.)', fontsize=12)
    ax1.set_xlim(-dE_plot, 3); ax1.set_ylim(-0.02, 1.05)
    ax1.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)
    ax1.set_title(r'$\beta$-decay spectrum near endpoint ($Q_\beta = 18\,574$ eV)', fontsize=10)

    ax2.set_xlabel(r'$T - Q_\beta$ (eV)', fontsize=12)
    ax2.set_ylabel(r'Kurie function $K(T) = \sqrt{dN/dT}$ (a.u.)', fontsize=12)
    ax2.set_xlim(-dE_plot, 3); ax2.set_ylim(-0.02, 1.05)
    ax2.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
    ax2.set_title('Kurie plot — linear for massless neutrinos', fontsize=10)

    plt.tight_layout(); plt.show()

    print("── Q1: Endpoint shift and affected fraction of spectrum ──")
    for m in [0.45, 0.2]:
        dT = m  # shift in eV
        frac = (m / Q)**3 / 3   # approx fraction of events in last m_ν range
        print(f"  m_ν = {m:.2f} eV: endpoint shifts by {dT:.2f} eV, "
              f"fraction affected ≈ {frac*100:.2e} %")

    print("\n── Q4: m_eff_{νe} vs m_ββ ──")
    print("  KATRIN:   m^{eff}_{νe} = sqrt(Σ |Uei|² mᵢ²)  — incoherent sum")
    print("  ββ₀ν exp: m_ββ = |Σ Uei² mᵢ e^{iα}|          — coherent sum with Majorana phases")
    print("  Key difference: m_ββ can cancel (Majorana phases), m^{eff}_{νe} cannot.")
    print(f"  Current KATRIN limit: m^{{eff}}_{{νe}} < 0.45 eV → m_ββ ≲ 0.45 eV (quasi-deg.)")


def exercise_mbb_phases_interactive(bf=None):
    """
    Widget interactivo: m_ββ en función de las fases de Majorana.

    Muestra m_ββ vs m_min para NH e IH, con las fases α₂₁, α₃₁
    controladas con deslizadores. Ilustra la cancelación de amplitudes.

    Parámetros
    ----------
    bf : dict, opcional — parámetros de oscilación NuFit-6.0 por defecto.
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    m0 = np.logspace(-4, np.log10(0.5), 400)

    def _plot(alpha21_deg, alpha31_deg, show_band):
        a21 = np.radians(alpha21_deg)
        a31 = np.radians(alpha31_deg)

        fig, ax = plt.subplots(figsize=(8, 6))

        for ordering, color, label in [('NH', 'C0', 'NH'), ('IH', 'C1', 'IH')]:
            if show_band:
                mn, mx = mbb_band(m0, ordering, bf, n_phases=800)
                ax.fill_between(m0 * 1e3, mn * 1e3, mx * 1e3,
                                alpha=0.20, color=color)

            vals = np.array([mbb_value(m, a21, a31, ordering, bf) for m in m0])
            ax.plot(m0 * 1e3, vals * 1e3, color=color, lw=2.5,
                    label=rf'{label}: $\alpha_{{21}}={alpha21_deg:.0f}°,\;\alpha_{{31}}={alpha31_deg:.0f}°$')

        ax.axhline(122, color='darkgreen', lw=1.5, ls='-.', alpha=0.8,
                   label=r'KamLAND-Zen 800 (122 meV)')
        ax.axhline(28,  color='darkgreen', lw=1.5, ls=':',  alpha=0.8,
                   label=r'KamLAND-Zen 800 (28 meV)')
        ax.axhline(15,  color='purple',    lw=1.2, ls='--', alpha=0.7,
                   label=r'IH min ≈ 15 meV')

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$m_{\rm min}$ (meV)', fontsize=12)
        ax.set_ylabel(r'$|m_{\beta\beta}|$ (meV)', fontsize=12)
        ax.set_xlim(0.1, 500); ax.set_ylim(0.2, 500)
        ax.set_title(r'$|m_{\beta\beta}|$ for fixed Majorana phases', fontsize=11)
        ax.legend(fontsize=9, loc='upper left'); ax.grid(True, which='both', alpha=0.25)
        plt.tight_layout(); plt.show()

    interact(
        _plot,
        alpha21_deg=widgets.IntSlider(value=0,   min=0, max=360, step=10,
                                      description=r'α₂₁ (°)',
                                      style={'description_width': '60px'},
                                      layout=widgets.Layout(width='500px')),
        alpha31_deg=widgets.IntSlider(value=0,   min=0, max=360, step=10,
                                      description=r'α₃₁ (°)',
                                      style={'description_width': '60px'},
                                      layout=widgets.Layout(width='500px')),
        show_band=widgets.Checkbox(value=True, description='Show full phase band',
                                   layout=widgets.Layout(width='250px')),
    )
