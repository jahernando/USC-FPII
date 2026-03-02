
"""
oscillations.py — Utilidades de probabilidad de oscilación de neutrinos.

Exporta funciones de física y widgets interactivos para explorar
oscilaciones de 2 y 3 familias con parámetros deslizables.

Funciones principales:
    plot_2fam_interactive()  — widget interactivo 2 familias
    plot_3fam_interactive()  — widget interactivo 3 familias (PMNS completa)

Funciones de física (reutilizables):
    posc_2fam(E_GeV, L_km, theta_deg, dm2_eV2)
    pmns_matrix(th12, th13, th23, delta)
    osc_prob_3fam(alpha, beta, L_km, E_GeV, U, dm2_21, dm2_31)
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

# ── Mejores valores ajustados: NuFit 6.0 (2024), Ordenación Normal (NH) ───────
# Referencia: I. Esteban et al., JHEP 12 (2024) 216, arXiv:2410.05380
# Δm²₂₁ y θ₁₂ actualizados con primeros resultados JUNO (arXiv:2511.21650)
BF_NUFIT60_NH = dict(
    th12   = 33.68,      # ángulo solar        (grados)  sin²θ₁₂ = 0.3083
    th13   =  8.55,      # ángulo reactor      (grados)  sin²θ₁₃ = 0.02215
    th23   = 48.30,      # ángulo atmosférico  (grados)  sin²θ₂₃ = 0.558
    delta  = 212.0,      # fase CP             (grados)
    dm2_21 = 7.48e-5,    # Δm²₂₁  (eV²)  — actualizado con JUNO 2025
    dm2_31 = 2.524e-3,   # Δm²₃₁  (eV²) — NH: positivo
)

# Alias para compatibilidad con código anterior
BF_NUFIT53_NH = BF_NUFIT60_NH
_FLABELS = [r'$\nu_e$', r'$\nu_\mu$', r'$\nu_\tau$']
_FCOLORS = ['C0', 'C1', 'C2']


# ── Funciones de física ────────────────────────────────────────────────────────

def posc_2fam(E_GeV, L_km, theta_deg, dm2_eV2):
    """
    Probabilidad de oscilación P(να→νβ) para mezcla de 2 familias.

    Parámetros
    ----------
    E_GeV    : array-like — energía del neutrino (GeV)
    L_km     : float      — distancia de propagación (km)
    theta_deg: float      — ángulo de mezcla (grados)
    dm2_eV2  : float      — diferencia de masas cuadradas Δm² (eV²)

    Devuelve
    --------
    ndarray — P(E) = sin²(2θ) sin²(Δm² L / 4E)
    """
    phi = 1.267 * dm2_eV2 * L_km / np.asarray(E_GeV, dtype=float)  # Δm² L/4E
    return np.sin(np.radians(2.0 * theta_deg))**2 * np.sin(phi)**2


def pmns_matrix(th12, th13, th23, delta):
    """
    Matriz de mezcla PMNS en la parametrización estándar PDG.

    Parámetros
    ----------
    th12, th13, th23 : float — ángulos de mezcla (radianes)
    delta            : float — fase de violación CP (radianes)

    Devuelve
    --------
    ndarray (3×3) compleja — matriz unitaria U_PMNS
    """
    c12, s12 = np.cos(th12), np.sin(th12)
    c13, s13 = np.cos(th13), np.sin(th13)
    c23, s23 = np.cos(th23), np.sin(th23)
    emid = np.exp(-1j * delta)
    eid  = np.exp( 1j * delta)
    return np.array([
        [ c12*c13,                    s12*c13,                    s13*emid ],
        [-s12*c23 - c12*s23*s13*eid,  c12*c23 - s12*s23*s13*eid, s23*c13  ],
        [ s12*s23 - c12*c23*s13*eid, -c12*s23 - s12*c23*s13*eid, c23*c13  ],
    ])


def osc_prob_3fam(alpha, beta, L_km, E_GeV, U, dm2_21, dm2_31):
    """
    Probabilidad de oscilación P(να→νβ) para 3 familias, vectorizada en E.

    Usa la fórmula exacta con índice de referencia p = 0:

        P = δ_αβ - 4 Σ_{i>j} Re[X_ij] sin²(Δm²_ij L/4E)
                 - 2 Σ_{i>j} Im[X_ij] sin(Δm²_ij L/2E)

    con X_ij = U_βi U*_αi U*_βj U_αj.

    Parámetros
    ----------
    alpha, beta : int       — índices de sabor inicial y final (0=e, 1=μ, 2=τ)
    L_km        : float     — distancia de propagación (km)
    E_GeV       : array-like — energía (GeV)
    U           : ndarray   — matriz PMNS 3×3 (compleja)
    dm2_21      : float     — Δm²₂₁ (eV²)
    dm2_31      : float     — Δm²₃₁ (eV²)

    Devuelve
    --------
    ndarray — P(E) acotada en [0, 1]
    """
    dm2 = np.array([0., dm2_21, dm2_31])
    E   = np.asarray(E_GeV, dtype=float)
    P   = np.full_like(E, float(alpha == beta))
    for i, j in [(1, 0), (2, 0), (2, 1)]:
        dij = dm2[i] - dm2[j]
        phi = 1.267 * dij * L_km / E             # Δm²_ij L / 4E
        Xij = U[beta, i] * U[alpha, i].conj() * U[beta, j].conj() * U[alpha, j]
        P  -= 4.0 * Xij.real * np.sin(phi)**2
        P  -= 2.0 * Xij.imag * np.sin(2.0 * phi)
    return np.clip(P, 0., 1.)


# ── Widgets interactivos ───────────────────────────────────────────────────────

def plot_2fam_interactive():
    """
    Widget interactivo de oscilación de 2 familias.

    Deslizadores: θ (°), log₁₀(Δm²/eV²), log₁₀(L/km).
    Gráfica izquierda: P vs E con L fija.
    Gráfica derecha:   P vs L/E con la longitud de oscilación indicada.
    """
    def _plot(theta_deg=45.0, log_dm2=-2.6, log_L=3.0):
        dm2  = 10**log_dm2
        L    = 10**log_L
        E    = np.logspace(-2, 3, 2000)
        prob = posc_2fam(E, L, theta_deg, dm2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        # P vs E (L fija)
        ax1.plot(E, prob, lw=2, color='steelblue')
        ax1.set_xscale('log')
        ax1.set_xlabel(r'$E$ (GeV)', fontsize=12)
        ax1.set_ylabel(r'$\mathcal{P}(\nu_\alpha \to \nu_\beta)$', fontsize=12)
        ax1.set_ylim(0, 1.05)
        ax1.set_title(fr'$L = {L:.0f}$ km', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # P vs L/E
        LoE   = np.logspace(0, 6, 2000)
        prob2 = np.sin(np.radians(2 * theta_deg))**2 * np.sin(1.267 * dm2 * LoE)**2
        Losc  = np.pi / (1.267 * dm2)
        ax2.plot(LoE, prob2, lw=2, color='darkorange')
        ax2.set_xscale('log')
        ax2.set_xlabel(r'$L/E$ (km/GeV)', fontsize=12)
        ax2.set_ylabel(r'$\mathcal{P}(\nu_\alpha \to \nu_\beta)$', fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.set_title(fr'$L_\mathrm{{osc}} = {Losc:.1f}$ km/GeV', fontsize=12)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            fr'2 familias: $\theta = {theta_deg:.1f}°$,  '
            fr'$\Delta m^2 = {dm2:.2e}$ eV²',
            fontsize=13
        )
        plt.tight_layout()
        plt.show()

    interact(
        _plot,
        theta_deg = widgets.FloatSlider(
            value=45.0, min=0.5, max=89.5, step=0.5,
            description='θ (°)', style={'description_width': 'initial'},
            continuous_update=False),
        log_dm2   = widgets.FloatSlider(
            value=-2.6, min=-6.0, max=-1.0, step=0.05,
            description='log₁₀(Δm²/eV²)', style={'description_width': 'initial'},
            continuous_update=False),
        log_L     = widgets.FloatSlider(
            value=3.0, min=0.0, max=5.0, step=0.1,
            description='log₁₀(L/km)', style={'description_width': 'initial'},
            continuous_update=False),
    )


def plot_3fam_interactive(bf=None):
    """
    Widget interactivo de oscilación de 3 familias (PMNS completa).

    Parámetros
    ----------
    bf : dict, opcional
        Valores iniciales de los parámetros. Por defecto BF_NUFIT53_NH
        (NuFIT 5.3, 2023, Ordenación Normal).
        Claves esperadas: th12, th13, th23, delta (grados), dm2_21, dm2_31 (eV²).

    Controles: sabor inicial, log₁₀(L/km), θ₁₂, θ₁₃, θ₂₃, δ_CP,
               log₁₀(Δm²₂₁), log₁₀(|Δm²₃₁|).
    Gráfica izquierda: P vs E para los tres sabores finales.
    Gráfica derecha:   matriz de probabilidades a E = 1 GeV (heatmap).
    """
    if bf is None:
        bf = BF_NUFIT53_NH

    def _plot(alpha_i=1, log_L=3.0,
              th12_deg=bf['th12'], th13_deg=bf['th13'],
              th23_deg=bf['th23'], delta_deg=bf['delta'],
              log_dm2_21=np.log10(bf['dm2_21']),
              log_dm2_31=np.log10(bf['dm2_31'])):
        L    = 10**log_L
        dm21 = 10**log_dm2_21
        dm31 = 10**log_dm2_31
        U    = pmns_matrix(np.radians(th12_deg), np.radians(th13_deg),
                           np.radians(th23_deg), np.radians(delta_deg))
        E    = np.logspace(-2, 3, 3000)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # P vs E para los tres sabores finales
        for beta in range(3):
            P   = osc_prob_3fam(alpha_i, beta, L, E, U, dm21, dm31)
            lbl = fr'{_FLABELS[alpha_i]} → {_FLABELS[beta]}'
            ax1.plot(E, P, lw=2, color=_FCOLORS[beta], label=lbl)
        ax1.axhline(1/3, ls=':', color='gray', alpha=0.5, label='1/3 (promedio)')
        ax1.set_xscale('log')
        ax1.set_xlabel(r'$E$ (GeV)', fontsize=12)
        ax1.set_ylabel(r'$\mathcal{P}(\nu_\alpha \to \nu_\beta)$', fontsize=12)
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(fr'$L = {L:.0f}$ km', fontsize=12)

        # Matriz de probabilidades a E_ref
        E_ref = 1.0  # GeV
        Pmat  = np.array([[osc_prob_3fam(a, b, L, E_ref, U, dm21, dm31)
                           for b in range(3)] for a in range(3)])
        im = ax2.imshow(Pmat, vmin=0, vmax=1, cmap='RdYlGn', aspect='equal')
        plt.colorbar(im, ax=ax2, label='Probabilidad')
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(_FLABELS, fontsize=12)
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(_FLABELS, fontsize=12)
        ax2.set_xlabel('Sabor final', fontsize=11)
        ax2.set_ylabel('Sabor inicial', fontsize=11)
        for a in range(3):
            for b in range(3):
                color = 'black' if 0.2 < Pmat[a, b] < 0.8 else 'white'
                ax2.text(b, a, f'{Pmat[a, b]:.3f}', ha='center', va='center',
                         fontsize=10, color=color)
        ax2.set_title(fr'Matriz $\mathcal{{P}}$ a $E = {E_ref:.0f}$ GeV', fontsize=12)

        fig.suptitle(
            fr'3 familias: $\theta_{{12}}={th12_deg:.1f}°$, '
            fr'$\theta_{{13}}={th13_deg:.1f}°$, '
            fr'$\theta_{{23}}={th23_deg:.1f}°$, '
            fr'$\delta_\mathrm{{CP}}={delta_deg:.0f}°$ — '
            fr'$\Delta m^2_{{21}}={dm21:.2e}$, '
            fr'$\Delta m^2_{{31}}={dm31:.2e}$ eV²',
            fontsize=10
        )
        plt.tight_layout()
        plt.show()

    interact(
        _plot,
        alpha_i    = widgets.Dropdown(
            options=[('νe', 0), ('νμ', 1), ('ντ', 2)], value=1,
            description='ν inicial:', style={'description_width': 'initial'}),
        log_L      = widgets.FloatSlider(
            value=3.0, min=1.0, max=5.0, step=0.1,
            description='log₁₀(L/km)', style={'description_width': 'initial'},
            continuous_update=False),
        th12_deg   = widgets.FloatSlider(
            value=bf['th12'], min=0.0, max=90.0, step=0.5,
            description='θ₁₂ (°)', style={'description_width': 'initial'},
            continuous_update=False),
        th13_deg   = widgets.FloatSlider(
            value=bf['th13'], min=0.0, max=30.0, step=0.1,
            description='θ₁₃ (°)', style={'description_width': 'initial'},
            continuous_update=False),
        th23_deg   = widgets.FloatSlider(
            value=bf['th23'], min=0.0, max=90.0, step=0.5,
            description='θ₂₃ (°)', style={'description_width': 'initial'},
            continuous_update=False),
        delta_deg  = widgets.FloatSlider(
            value=bf['delta'], min=0.0, max=360.0, step=5.0,
            description='δ_CP (°)', style={'description_width': 'initial'},
            continuous_update=False),
        log_dm2_21 = widgets.FloatSlider(
            value=np.log10(bf['dm2_21']), min=-6.0, max=-3.0, step=0.05,
            description='log₁₀(Δm²₂₁/eV²)', style={'description_width': 'initial'},
            continuous_update=False),
        log_dm2_31 = widgets.FloatSlider(
            value=np.log10(bf['dm2_31']), min=-4.0, max=-1.0, step=0.05,
            description='log₁₀(|Δm²₃₁|/eV²)', style={'description_width': 'initial'},
            continuous_update=False),
    )

# ── Efectos de materia (MSW) ──────────────────────────────────────────────────

def posc_matter_2fam(E_MeV, L_km, theta_deg, dm2_eV2, rho_gcc=0.0, Ye=0.5):
    """
    Probabilidad de oscilación P(νe→νe) en materia de densidad constante,
    caso de 2 familias (fórmula analítica MSW).

    Parámetros
    ----------
    E_MeV    : array-like — energía del neutrino (MeV)
    L_km     : float      — distancia de propagación (km)
    theta_deg: float      — ángulo de mezcla en vacío (grados)
    dm2_eV2  : float      — Δm² en vacío (eV²), positivo
    rho_gcc  : float      — densidad del medio (g/cm³), 0 = vacío
    Ye       : float      — fracción de electrones (≈0.5 corteza, ≈0.46 manto)

    Devuelve
    --------
    ndarray — P(E) = sin²(2θ_m) sin²(Δm²_m L / 4E)
    """
    E     = np.asarray(E_MeV, dtype=float) * 1e-3   # → GeV
    th0   = np.radians(theta_deg)
    cos2  = np.cos(2.0 * th0)
    sin2  = np.sin(2.0 * th0)
    # Potencial de CC en eV: Ve = √2 G_F Ne ≃ 7.63e-14 * Ye * rho  (eV)
    Ve    = 7.63e-14 * Ye * rho_gcc        # eV
    Ve_GeV = Ve * 1e-9                     # GeV
    # Δm²/2E en GeV (factor 1/2)
    Delta = dm2_eV2 * 1e-18 / (2.0 * E)   # GeV  (dm2 en eV², E en GeV)
    A     = Ve_GeV                         # GeV
    # Δm²_m / 2E
    Delta_m = np.sqrt((Delta * cos2 - A)**2 + (Delta * sin2)**2)
    # sin²(2θ_m)
    sin2_2thm = (Delta * sin2)**2 / Delta_m**2
    # fase en materia: Δm²_m * L / 4E = Delta_m * L_km * 1e3 / 2  (en nat. units)
    phi_m  = 1.267 * 2.0 * Delta_m * 1e18 * L_km / (np.asarray(E_MeV, float))
    return sin2_2thm * np.sin(phi_m)**2


# ── Espectro de antineutrinos de reactor (análogo JUNO) ───────────────────────

def _reactor_spectrum(E_MeV):
    """Espectro aproximado de ν̄_e de reactor (flujo × sección eficaz IBD)."""
    # Aproximación analítica: exponencial × umbral IBD
    E = np.asarray(E_MeV, dtype=float)
    flux = np.where(E > 1.806, np.exp(0.87 - 0.16 * E - 0.091 * E**2), 0.0)
    sigma_ibd = np.where(E > 1.806, 9.52e-44 * (E - 1.293)**2, 0.0)  # cm²
    return flux * sigma_ibd


def plot_juno_spectrum(bf=None):
    """
    Widget interactivo: espectro de ν̄_e de reactor a L = 52.5 km (JUNO).

    Muestra el espectro visible con oscilación (3 familias), resaltando el
    patrón de interferencia entre Δm²₂₁ y Δm²₃₁ que JUNO usa para la
    ordenación de masas.  Deslizadores: θ₁₂, θ₁₃, Δm²₂₁, Δm²₃₁,
    y selector de ordenación (NH / IH).
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    def _plot(th12_deg=bf['th12'], th13_deg=bf['th13'],
              log_dm2_21=np.log10(bf['dm2_21']),
              log_dm2_31=np.log10(bf['dm2_31']),
              ordering='NH'):
        L     = 52.5   # km — distancia media JUNO
        dm21  = 10**log_dm2_21
        dm31  = 10**log_dm2_31 if ordering == 'NH' else -10**log_dm2_31
        U     = pmns_matrix(np.radians(th12_deg), np.radians(th13_deg),
                            np.radians(bf['th23']),  np.radians(bf['delta']))
        E_MeV = np.linspace(1.9, 8.5, 2000)
        E_GeV = E_MeV * 1e-3

        P_ee  = osc_prob_3fam(0, 0, L, E_GeV, U, dm21, dm31)
        spec0 = _reactor_spectrum(E_MeV)
        spec  = spec0 * P_ee

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.fill_between(E_MeV, spec0 / spec0.max(), alpha=0.25,
                         color='gray', label='Sin oscilación')
        ax1.plot(E_MeV, spec / spec.max(), lw=1.5, color='C0',
                 label='Con oscilación (3ν)')
        ax1.set_ylabel('Espectro (u.a.)', fontsize=11)
        ax1.set_ylim(0, 1.15)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(
            fr'Espectro reactor en JUNO  ($L={L}$ km, ordenación {ordering})',
            fontsize=12)

        ax2.plot(E_MeV, P_ee, lw=1.5, color='C1')
        ax2.set_xlabel(r'$E_{\bar\nu}$ (MeV)', fontsize=12)
        ax2.set_ylabel(r'$P(\bar\nu_e \to \bar\nu_e)$', fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(1, ls='--', color='gray', alpha=0.5)

        plt.tight_layout()
        plt.show()

    interact(
        _plot,
        th12_deg   = widgets.FloatSlider(
            value=bf['th12'], min=25.0, max=42.0, step=0.5,
            description='θ₁₂ (°)', style={'description_width': 'initial'},
            continuous_update=False),
        th13_deg   = widgets.FloatSlider(
            value=bf['th13'], min=4.0, max=14.0, step=0.1,
            description='θ₁₃ (°)', style={'description_width': 'initial'},
            continuous_update=False),
        log_dm2_21 = widgets.FloatSlider(
            value=np.log10(bf['dm2_21']), min=-5.5, max=-4.0, step=0.02,
            description='log₁₀(Δm²₂₁/eV²)', style={'description_width': 'initial'},
            continuous_update=False),
        log_dm2_31 = widgets.FloatSlider(
            value=np.log10(bf['dm2_31']), min=-3.5, max=-2.5, step=0.02,
            description='log₁₀(|Δm²₃₁|/eV²)', style={'description_width': 'initial'},
            continuous_update=False),
        ordering   = widgets.RadioButtons(
            options=['NH', 'IH'], value='NH',
            description='Ordenación:', style={'description_width': 'initial'}),
    )


# ── Funciones de ejercicios ───────────────────────────────────────────────────

def plot_posc_intro():
    """
    Dos gráficas introductorias de P de oscilación a 2 familias:
      - izquierda: P vs L/E (L fijo en unidades de E = 1 GeV)
      - derecha:   P vs E   (con L = 1000 km)
    Parámetros de ejemplo: θ = π/4, Δm² = 2.5×10⁻³ eV² (escala atmosférica).
    """
    theta0 = np.pi / 4.0
    dm2_32 = 2.5e-3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ls       = np.logspace(1, 4, 1000)
    prob_LoE = np.sin(2 * theta0)**2 * np.sin(1.27 * dm2_32 * ls)**2
    ax1.plot(ls, prob_LoE, lw=2, color='steelblue')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$L/E$ (km/GeV),  $E = 1$ GeV', fontsize=12)
    ax1.set_ylabel('probability', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_title(fr'$\theta = \pi/4$,  $\Delta m^2 = {dm2_32}$ eV²', fontsize=12)
    ax1.grid(True, alpha=0.3)

    L_km   = 1000.0
    es     = np.logspace(-1, 2, 1000)
    prob_E = posc_2fam(es, L_km, np.degrees(theta0), dm2_32)
    ax2.plot(es, prob_E, lw=2, color='darkorange')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$E$ (GeV),  $L = 1000$ km', fontsize=12)
    ax2.set_ylabel('probability', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.set_title(fr'$\theta = \pi/4$,  $\Delta m^2 = {dm2_32}$ eV²', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def exercise_osc_regimes(bf=None):
    """
    Ejercicio: longitud de oscilación y regímenes experimentales.

    Q1: Calcula L_osc para las escalas atmosférica y solar.
    Q2: Dibuja P vs L/E en los tres regímenes para parámetros solares.
    Q3: Verifica φ ≈ π/2 para SK-atm, KamLAND y Daya Bay.

    Parámetros
    ----------
    bf : dict, opcional — valores iniciales (por defecto BF_NUFIT60_NH).
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    def _L_osc(E_GeV, dm2_eV2):
        return np.pi / 1.267 * E_GeV / dm2_eV2

    print("Oscillation lengths at E = 1 GeV:")
    print(f"  Atmospheric (dm2_31=2.5e-3 eV2): L_osc = {_L_osc(1, 2.5e-3):.1f} km")
    print(f"  Solar       (dm2_21=7.5e-5 eV2): L_osc = {_L_osc(1, 7.5e-5):.0f} km")
    print("Oscillation lengths at E = 3 MeV:")
    print(f"  Solar (dm2_21=7.5e-5 eV2): L_osc = {_L_osc(3e-3, 7.5e-5):.1f} km")

    LoE = np.logspace(0, 6, 5000)
    P   = posc_2fam(1.0 / LoE, 1.0, bf['th12'], bf['dm2_21'])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(LoE, P, lw=1.5, color='steelblue')
    ax.set_xscale('log')
    ax.set_xlabel(r'$L/E$ (km/GeV)', fontsize=12)
    ax.set_ylabel(r'$P(\nu_\alpha \to \nu_\alpha)$', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(1 - 0.5 * np.sin(np.radians(2 * bf['th12']))**2, ls='--', color='red',
               alpha=0.7, label=r'Average $\langle P\rangle$')
    ax.set_title(r'Three regimes: $L\ll L_{\rm osc}$,  $L\sim L_{\rm osc}$,  $L\gg L_{\rm osc}$')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    print("\nOscillation phases at each experiment's peak:")
    expts = [
        ('SK-atm',   1000.,  1.0,  2.5e-3),
        ('KamLAND',  180.,   3e-3, 7.5e-5),
        ('Daya Bay', 1.648,  4e-3, 2.52e-3),
    ]
    for name, L, E, dm2 in expts:
        phi = 1.267 * dm2 * L / E
        print(f"  {name}: L={L} km, E={E*1e3:.0f} MeV -> phi = {phi:.2f} rad  ({phi/np.pi:.2f} pi)")


def exercise_exclusion_contour():
    """
    Ejercicio: región de exclusión en el plano (Δm², sin²2θ).

    Q1: Análisis analítico de los regímenes asintóticos (Δm² grande/pequeño).
    Q2: Alcance de sensibilidad Δm²_min en función de L/E.
    Q3: Contorno de exclusión numérico para experimentos académico, CHOOZ y KamLAND.
    """
    P_lim = 0.05
    L_km  = 0.250   # 250 m
    E_GeV = 3e-3    # 3 MeV

    print(f"Large dm2 asymptote: sin^2(2theta) = 2 * P_lim = {2*P_lim:.2f}")
    dm2_min = np.sqrt(P_lim) / (1.267 * L_km / E_GeV)
    print(f"Min detectable dm2 at sin^2(2theta)=1: dm2_min = {dm2_min:.4f} eV^2")

    print("\nSensitivity reach (dm2_min at full mixing):")
    for name, L, E in [('Bugey', 0.015, 3e-3), ('CHOOZ', 1.0, 3e-3), ('KamLAND', 180., 3e-3)]:
        d = np.sqrt(P_lim) / (1.267 * L / E)
        print(f"  {name}: L={L*1e3:.0f} m -> dm2_min = {d:.4f} eV^2")

    def _contour(L_km, E_GeV, P_lim=0.05):
        dm2_arr = np.logspace(-3, 2, 800)
        phi_arr = 1.267 * dm2_arr * L_km / E_GeV
        sin2phi = np.sin(phi_arr)**2
        with np.errstate(divide='ignore', invalid='ignore'):
            s22t = np.where(sin2phi > 1e-20, np.minimum(P_lim / sin2phi, 1.0), 1.0)
        return dm2_arr, s22t

    fig, ax = plt.subplots(figsize=(8, 6))
    configs = [
        (0.250, 3e-3, 0.05, 'Academic (L=250 m)', 'C0', '-'),
        (1.000, 3e-3, 0.05, 'CHOOZ   (L=1 km)',   'C1', '--'),
        (180.0, 3e-3, 0.05, 'KamLAND (L=180 km)', 'C2', ':'),
    ]
    for L, E, Plim, label, color, ls in configs:
        dm2, s22t = _contour(L, E, Plim)
        ax.fill_betweenx(dm2, s22t, 1.0, alpha=0.15, color=color)
        ax.plot(s22t, dm2, lw=2, color=color, ls=ls, label=label)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\sin^2 2\theta$', fontsize=13)
    ax.set_ylabel(r'$\Delta m^2$ (eV$^2$)', fontsize=13)
    ax.set_xlim(1e-3, 1.2); ax.set_ylim(1e-3, 1e2)
    ax.set_title('Exclusion regions in the oscillation parameter plane', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_msw_parameters():
    """
    Gráficas del Δm²_eff y ángulo de mezcla efectivo θ_eff vs energía
    en presencia de materia, para condiciones típicas del núcleo solar.

    Muestra la resonancia MSW (máximo de θ_eff) y el límite de dominación
    de materia donde la mezcla efectiva se suprime.
    """
    V0e   = 1.256e-36   # eV/cm³
    ne    = 5e25        # e/cm³  (núcleo solar típico)
    Ve    = V0e * ne
    dm2_0 = 7.5e-5      # eV²
    th0   = 0.1845 * np.pi

    print(f"Potencial de coherencia V_e = {Ve:.3e} eV")
    print(f"Δm²₀ / 2E₀  (E₀=1 MeV) = {dm2_0 / (2e6):.3e}")

    E  = np.linspace(0, 2e6, 500)   # eV
    x  = 2 * Ve * E / dm2_0
    s2, c2 = np.sin(2 * th0), np.cos(2 * th0)
    dm2_eff = dm2_0 * np.sqrt((c2 - x)**2 + s2**2)
    theff   = np.arctan(s2 / (c2 - x)) / 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(E / 1e6, dm2_eff, lw=2, color='steelblue')
    ax1.set_xlabel("E (MeV)", fontsize=12)
    ax1.set_ylabel(r"$\Delta m^2_{\rm eff}$ (eV²)", fontsize=12)
    ax1.set_title("Masa efectiva en materia"); ax1.grid(True, alpha=0.3)

    ax2.plot(E / 1e6, theff / np.pi, lw=2, color='darkorange')
    ax2.set_xlabel("E (MeV)", fontsize=12)
    ax2.set_ylabel(r"$\theta_{\rm eff}/\pi$", fontsize=12)
    ax2.set_title("Ángulo de mezcla efectivo"); ax2.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()


def exercise_msw_resonance(bf=None):
    """
    Ejercicio: La resonancia MSW en el Sol.

    Q1: Calcula V_e en el núcleo solar y la energía de resonancia E_res.
    Q2: Compara la predicción adiabática P_ee = sin²θ₁₂ con SNO/Borexino.
    Q3: Dibuja P(νe→νe) vs E en vacío y en materia con posc_matter_2fam.

    Parámetros
    ----------
    bf : dict, opcional — valores iniciales (por defecto BF_NUFIT60_NH).
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    L_sun = 1.496e8   # km  (1 UA)
    rho   = 150.0; Ye = 0.5

    Ve_eV     = 7.63e-14 * Ye * rho
    th0       = np.radians(bf['th12'])
    E_res_MeV = bf['dm2_21'] * np.cos(2 * th0) / (2 * Ve_eV) * 1e6
    print(f"Ve (solar core)       = {Ve_eV:.3e} eV")
    print(f"Resonance energy E_res = {E_res_MeV:.2f} MeV")
    print(f"8B range (5-15 MeV) is {'above' if 10 > E_res_MeV else 'below'} the resonance")
    print(f"Adiabatic prediction: P_ee = sin²(θ₁₂) = {np.sin(th0)**2:.3f}")

    E_MeV = np.linspace(0.1, 20, 1000)
    P_vac = posc_matter_2fam(E_MeV, L_sun, bf['th12'], bf['dm2_21'], rho_gcc=0)
    P_mat = posc_matter_2fam(E_MeV, L_sun, bf['th12'], bf['dm2_21'], rho_gcc=100)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(E_MeV, P_vac, lw=1.5, label='Vacuum', color='steelblue')
    ax.plot(E_MeV, P_mat, lw=2,   label=r'Matter ($\rho=100$ g/cm$^3$)', color='darkorange')
    ax.axhline(np.sin(th0)**2, ls='--', color='green', alpha=0.7,
               label=fr'Adiabatic limit $\sin^2\theta_{{12}}={np.sin(th0)**2:.2f}$')
    ax.axvline(E_res_MeV, ls=':', color='red', alpha=0.7,
               label=fr'$E_\mathrm{{res}}={E_res_MeV:.1f}$ MeV')
    ax.set_xlabel(r'$E_\nu$ (MeV)', fontsize=12)
    ax.set_ylabel(r'$P(\nu_e\to\nu_e)$', fontsize=12)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_title('MSW effect: solar neutrino survival probability')
    plt.tight_layout(); plt.show()


def exercise_kamland(bf=None):
    """
    Ejercicio: KamLAND y la medida de θ₁₂.

    Q1: Fases de oscilación φ₂₁ a distintas energías (3, 5, 7 MeV).
    Q2: Extrae sin²(2θ₁₂) de la ratio observada/esperada medida.
    Q3: Dibuja P vs L/E marcando el punto de operación de KamLAND.

    Parámetros
    ----------
    bf : dict, opcional — valores iniciales (por defecto BF_NUFIT60_NH).
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    L = 180.0   # km

    print("Q1 — Oscillation phases at KamLAND (L = 180 km):")
    for E_MeV in [3, 5, 7]:
        phi = 1.267 * bf['dm2_21'] * L / (E_MeV * 1e-3)
        print(f"  E = {E_MeV} MeV -> phi21 = {phi:.2f} rad  ({phi/np.pi:.2f} pi)")
    E_min = 1.267 * bf['dm2_21'] * L / (np.pi / 2) * 1e3
    print(f"  First minimum (phi=pi/2) at E = {E_min:.1f} MeV")

    R = 0.498
    sin2_2th13 = np.sin(np.radians(2 * bf['th13']))**2
    sin2_2th12 = 2 * (1 - R - 0.5 * sin2_2th13)
    print(f"\nQ2 — Extracted sin^2(2θ₁₂) = {sin2_2th12:.3f}")
    print(f"     NuFit-6.0 sin^2(2θ₁₂) = {np.sin(np.radians(2*bf['th12']))**2:.3f}")

    LoE    = np.logspace(1, 5, 3000)
    P      = posc_2fam(1.0 / LoE, 1.0, bf['th12'], bf['dm2_21'])
    LoE_KL = L / 3e-3

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(LoE, P, lw=2, color='steelblue')
    ax.axvline(LoE_KL, ls='--', color='darkorange',
               label=f'KamLAND (L/E = {LoE_KL:.0f} km/GeV)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$L/E$ (km/GeV)', fontsize=12)
    ax.set_ylabel(r'$P(\bar\nu_e\to\bar\nu_e)$', fontsize=12)
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('KamLAND: solar oscillation')
    plt.tight_layout(); plt.show()


def exercise_sk_asymmetry(bf=None):
    """
    Ejercicio: SuperKamiokande y la asimetría Up/Down atmosférica.

    Q1: Longitud de camino L(θ_z) y fase φ₃₂ en distintas direcciones cenitales.
    Q2: Extrae sin²(2θ₂₃) de la asimetría A_UD medida por SK.
    Q3: Dibuja L(θ_z) y P(νμ→νμ) vs cos θ_z.

    Parámetros
    ----------
    bf : dict, opcional — valores iniciales (por defecto BF_NUFIT60_NH).
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    R_E = 6371.0; h = 20.0; E = 1.0   # km, km, GeV

    def _path_length(theta_z_deg):
        cz = np.cos(np.radians(theta_z_deg))
        return np.sqrt((R_E * cz)**2 + 2 * R_E * h) - R_E * cz

    print("Q1 — Path lengths and phases (E = 1 GeV):")
    for tz in [0, 90, 180]:
        L   = _path_length(tz)
        phi = 1.267 * bf['dm2_31'] * L / E
        print(f"  theta_z = {tz}°: L = {L:.0f} km, phi32 = {phi:.2f} rad")

    A_UD       = -0.296
    sin2_2th23 = abs(A_UD) / 0.5
    print(f"\nQ2 — Extracted sin^2(2θ₂₃) = {sin2_2th23:.3f}")
    print(f"     NuFit-6.0 sin^2(2θ₂₃)  = {np.sin(np.radians(2*bf['th23']))**2:.3f}")

    cos_tz = np.linspace(-1, 1, 500)
    tz_arr = np.degrees(np.arccos(cos_tz))
    L_arr  = _path_length(tz_arr)
    P_arr  = posc_2fam(E, L_arr, bf['th23'], bf['dm2_31'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(cos_tz, L_arr / 1e3, lw=2, color='steelblue')
    ax1.set_xlabel(r'$\cos\theta_z$', fontsize=12)
    ax1.set_ylabel(r'$L$ ($10^3$ km)', fontsize=12)
    ax1.set_title('Path length vs zenith angle'); ax1.grid(True, alpha=0.3)

    ax2.plot(cos_tz, P_arr, lw=2, color='darkorange', label=r'$P(\nu_\mu\to\nu_\mu)$')
    ax2.axhline(1, ls='--', color='gray', alpha=0.6, label='No oscillation')
    ax2.set_xlabel(r'$\cos\theta_z$', fontsize=12)
    ax2.set_ylabel(r'$P(\nu_\mu\to\nu_\mu)$', fontsize=12)
    ax2.set_ylim(0, 1.05); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_title(fr'SK atmospheric: $E = {E}$ GeV')
    plt.tight_layout(); plt.show()


def exercise_dayabay(bf=None):
    """
    Ejercicio: Daya Bay y la medida de θ₁₃.

    Q1: Fase φ₃₁ en el detector lejano (L = 1.648 km) a E = 4 MeV.
    Q2: Extrae sin²(2θ₁₃) de la ratio far/near medida (R = 0.944).
    Q3: Dibuja P(ν̄e→ν̄e) vs E comparando 3 familias y aprox. 2 familias.

    Parámetros
    ----------
    bf : dict, opcional — valores iniciales (por defecto BF_NUFIT60_NH).
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    L = 1.648   # km (detector lejano Daya Bay)

    E_MeV = 4.0
    phi31 = 1.267 * bf['dm2_31'] * L / (E_MeV * 1e-3)
    E_max = 1.267 * bf['dm2_31'] * L / (np.pi / 2) * 1e3
    print(f"Q1 — phi31 at E = {E_MeV} MeV: {phi31:.3f} rad ({phi31/np.pi:.3f} pi)")
    print(f"     First maximum at E = {E_max:.1f} MeV")

    R      = 0.944
    E_eff  = 4e-3   # GeV
    sin2ph = np.sin(1.267 * bf['dm2_31'] * L / E_eff)**2
    sin2_2th13 = (1 - R) / sin2ph
    th13_deg   = 0.5 * np.degrees(np.arcsin(np.sqrt(sin2_2th13)))
    print(f"\nQ2 — Extracted sin^2(2θ₁₃) = {sin2_2th13:.4f},  θ₁₃ = {th13_deg:.2f}°")
    print(f"     NuFit-6.0: sin^2(2θ₁₃) = {np.sin(np.radians(2*bf['th13']))**2:.4f},  θ₁₃ = {bf['th13']:.2f}°")

    E_GeV = np.linspace(1e-3, 10e-3, 1000)
    U     = pmns_matrix(np.radians(bf['th12']), np.radians(bf['th13']),
                        np.radians(bf['th23']), np.radians(bf['delta']))
    P_3f  = osc_prob_3fam(0, 0, L, E_GeV, U, bf['dm2_21'], bf['dm2_31'])
    P_2f  = 1 - np.sin(np.radians(2 * bf['th13']))**2 * np.sin(1.267 * bf['dm2_31'] * L / E_GeV)**2

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(E_GeV * 1e3, P_3f, lw=2,        label='3-family (full)',    color='C0')
    ax.plot(E_GeV * 1e3, P_2f, lw=2, ls='--', label='2-family approx.', color='C1')
    ax.set_xlabel(r'$E_{\bar\nu}$ (MeV)', fontsize=12)
    ax.set_ylabel(r'$P(\bar\nu_e\to\bar\nu_e)$', fontsize=12)
    ax.set_ylim(0.8, 1.02); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(fr'Daya Bay: $L = {L}$ km')
    plt.tight_layout(); plt.show()


def exercise_lbl_biprobability(bf=None):
    """
    Ejercicio: Probabilidad de aparición en LBL y asimetría CP.

    Q1: P(νμ→νe) y A_CP en T2K para δ_CP = ±90°.
    Q2: A_CP vs δ_CP en T2K (L=295 km) y NOvA (L=810 km).
    Q3: Diagrama bi-probabilidad (elipse de Cabibbo) para NH e IH.

    Parámetros
    ----------
    bf : dict, opcional — valores iniciales (por defecto BF_NUFIT60_NH).
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    def _Pnue_pair(L_km, E_GeV, delta_deg, dm2_31=None):
        """Devuelve (P(νμ→νe), P(ν̄μ→ν̄e)) para δ_CP dado."""
        if dm2_31 is None:
            dm2_31 = bf['dm2_31']
        U_nu  = pmns_matrix(np.radians(bf['th12']), np.radians(bf['th13']),
                            np.radians(bf['th23']),  np.radians(delta_deg))
        U_bar = pmns_matrix(np.radians(bf['th12']), np.radians(bf['th13']),
                            np.radians(bf['th23']), -np.radians(delta_deg))
        return (osc_prob_3fam(1, 0, L_km, E_GeV, U_nu,  bf['dm2_21'], dm2_31),
                osc_prob_3fam(1, 0, L_km, E_GeV, U_bar, bf['dm2_21'], dm2_31))

    print("Q1 — T2K (L = 295 km, E = 0.6 GeV):")
    for delta in [-90, +90]:
        Pnu, Pnubar = _Pnue_pair(295, 0.6, delta)
        Acp = (Pnu - Pnubar) / (Pnu + Pnubar + 1e-15)
        print(f"  delta = {delta:+4d}°: P(ν) = {Pnu:.4f}  P(ν̄) = {Pnubar:.4f}  A_CP = {Acp:+.3f}")

    deltas = np.linspace(0, 360, 500)

    fig, ax = plt.subplots(figsize=(9, 4))
    for L, E, label, color in [(295, 0.6, 'T2K', 'C0'), (810, 2.0, 'NOvA', 'C1')]:
        Acp_arr = []
        for d in deltas:
            Pnu, Pnubar = _Pnue_pair(L, E, d)
            Acp_arr.append((Pnu - Pnubar) / (Pnu + Pnubar + 1e-15))
        ax.plot(deltas, Acp_arr, lw=2, color=color, label=label)
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel(r'$\delta_{\rm CP}$ (°)', fontsize=12)
    ax.set_ylabel(r'$\mathcal{A}_{\rm CP}$', fontsize=12)
    ax.set_title('CP asymmetry vs δ_CP'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(7, 6))
    for dm31_sign, ls, olabel in [(+1, '-', 'NH'), (-1, '--', 'IH')]:
        for L, E, color, exp in [(295, 0.6, 'C0', 'T2K'), (810, 2.0, 'C1', 'NOvA')]:
            pairs = [_Pnue_pair(L, E, d, dm31_sign * abs(bf['dm2_31'])) for d in deltas]
            Pnu_arr, Pnubar_arr = zip(*pairs)
            ax.plot(Pnu_arr, Pnubar_arr, color=color, ls=ls, lw=1.5,
                    label=f'{exp} ({olabel})' if ls == '-' else f'_{exp} ({olabel})')
    Pnu_bf, Pnubar_bf = _Pnue_pair(295, 0.6, -90)
    ax.plot(Pnu_bf, Pnubar_bf, 'k*', ms=14, zorder=5, label=r'T2K $\delta=-90°$')
    ax.set_xlabel(r'$P(\nu_\mu\to\nu_e)$', fontsize=12)
    ax.set_ylabel(r'$P(\bar\nu_\mu\to\bar\nu_e)$', fontsize=12)
    ax.set_title('Bi-probability (Cabibbo) diagram')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def exercise_juno_smearing(bf=None):
    """
    Ejercicio: JUNO y el patrón de interferencia en el espectro de reactor.

    Q1: Fases φ₂₁, φ₃₁, φ₃₂ a E = 5 MeV para NH e IH.
    Q2: P_ee ideal vs P_ee convolucionada con la resolución energética de JUNO
        (σ_E/E = 3%/√(E/MeV)).

    Parámetros
    ----------
    bf : dict, opcional — valores iniciales (por defecto BF_NUFIT60_NH).
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    L     = 52.5   # km
    E_ref = 5e-3   # GeV
    dm32_NH = bf['dm2_31'] - bf['dm2_21']
    dm32_IH = -abs(bf['dm2_31']) + bf['dm2_21']

    print("Q1 — Phases at E = 5 MeV, L = 52.5 km:")
    for name, dm31, dm32 in [('NH', bf['dm2_31'], dm32_NH), ('IH', -abs(bf['dm2_31']), dm32_IH)]:
        phi21 = 1.267 * bf['dm2_21'] * L / E_ref
        phi31 = abs(1.267 * dm31 * L / E_ref)
        phi32 = abs(1.267 * dm32 * L / E_ref)
        print(f"  {name}: phi21 = {phi21:.1f}  phi31 = {phi31:.1f}  phi32 = {phi32:.1f}  (rad)")
    ncycles = (1.267 * bf['dm2_31'] * L / 2e-3 - 1.267 * bf['dm2_31'] * L / 8e-3) / (2 * np.pi)
    print(f"  Fast oscillation cycles in [2, 8] MeV: {ncycles:.1f}")

    E_MeV     = np.linspace(2.0, 8.0, 2000)
    E_GeV_arr = E_MeV * 1e-3
    U   = pmns_matrix(np.radians(bf['th12']), np.radians(bf['th13']),
                      np.radians(bf['th23']), np.radians(bf['delta']))
    Pee = osc_prob_3fam(0, 0, L, E_GeV_arr, U, bf['dm2_21'], bf['dm2_31'])

    sigma_MeV = 0.03 * np.sqrt(E_MeV)
    Pee_smear = np.zeros_like(Pee)
    for i, (E0, sig) in enumerate(zip(E_MeV, sigma_MeV)):
        kernel = np.exp(-0.5 * ((E_MeV - E0) / sig)**2)
        kernel /= kernel.sum()
        Pee_smear[i] = np.dot(kernel, Pee)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(E_MeV, Pee,       lw=1,   color='steelblue', alpha=0.6, label='Ideal $P_{ee}$')
    ax.plot(E_MeV, Pee_smear, lw=2.5, color='darkorange',
            label=r'Smeared ($\sigma/E = 3\%/\sqrt{E}$)')
    ax.set_xlabel(r'$E_{\bar\nu}$ (MeV)', fontsize=12)
    ax.set_ylabel(r'$P(\bar\nu_e\to\bar\nu_e)$', fontsize=12)
    ax.set_ylim(0.2, 1.0); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_title(fr'JUNO: interference pattern at $L = {L}$ km (NH)')
    plt.tight_layout(); plt.show()


def exercise_neutrino_mass_scale(bf=None):
    """
    Ejercicio: Escala absoluta de masas y ordenación de masas.

    Q1: Suma mínima de masas para NH (m₁→0) e IH (m₃→0).
    Q2: Banda de |⟨m_ββ⟩| vs m_min para NH e IH (desintegración ββ₀ν).
    Q3: m_β (masa cinemática KATRIN) vs m_min para NH e IH.

    Parámetros
    ----------
    bf : dict, opcional — valores iniciales (por defecto BF_NUFIT60_NH).
    """
    if bf is None:
        bf = BF_NUFIT60_NH

    th12 = np.radians(bf['th12']); th13 = np.radians(bf['th13'])
    Ue_sq = np.array([
        np.cos(th12)**2 * np.cos(th13)**2,
        np.sin(th12)**2 * np.cos(th13)**2,
        np.sin(th13)**2,
    ])
    dm21 = bf['dm2_21']; dm31 = bf['dm2_31']

    m1_NH = 0.0
    m2_NH, m3_NH = np.sqrt(m1_NH**2 + dm21), np.sqrt(m1_NH**2 + dm31)
    dm31_IH = -abs(dm31)
    m3_IH   = 0.0
    m1_IH   = np.sqrt(m3_IH**2 - dm31_IH)
    m2_IH   = np.sqrt(m1_IH**2 + dm21)
    print(f"Q1 — NH (m₁=0): m₂={m2_NH*1e3:.1f}, m₃={m3_NH*1e3:.1f} meV -> Sum={sum([m1_NH,m2_NH,m3_NH])*1e3:.1f} meV")
    print(f"     IH (m₃=0): m₁={m1_IH*1e3:.1f}, m₂={m2_IH*1e3:.1f} meV -> Sum={sum([m1_IH,m2_IH,m3_IH])*1e3:.1f} meV")
    print("     Cosmological bound: Sum < 120 meV")

    m_min = np.linspace(0, 0.3, 500)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for dm31_val, color, label in [(+abs(dm31), 'C0', 'NH'), (-abs(dm31), 'C1', 'IH')]:
        if dm31_val > 0:
            m = np.array([m_min, np.sqrt(m_min**2 + dm21), np.sqrt(m_min**2 + dm31_val)])
        else:
            m3 = m_min; m1 = np.sqrt(m3**2 - dm31_val); m2 = np.sqrt(m1**2 + dm21)
            m = np.array([m1, m2, m3])
        mbb_max = np.sum(Ue_sq[:, None] * m, axis=0)
        mbb_min = np.abs(Ue_sq[0]*m[0] - Ue_sq[1]*m[1] - Ue_sq[2]*m[2])
        ax1.fill_between(m_min * 1e3, mbb_min * 1e3, mbb_max * 1e3, alpha=0.35, color=color)
        ax1.plot(m_min * 1e3, mbb_max * 1e3, color=color, lw=2, label=label)
        m_beta = np.sqrt(np.sum(Ue_sq[:, None] * m**2, axis=0))
        ax2.plot(m_min * 1e3, m_beta * 1e3, color=color, lw=2, label=label)

    ax1.axhline(36, ls='--', color='red', lw=1.5, label='KamLAND-Zen (36 meV)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel(r'$m_{\rm min}$ (meV)', fontsize=12)
    ax1.set_ylabel(r'$|\langle m_{\beta\beta}\rangle|$ (meV)', fontsize=12)
    ax1.set_xlim(1, 300); ax1.set_ylim(0.5, 300)
    ax1.legend(fontsize=10); ax1.grid(True, which='both', alpha=0.3)
    ax1.set_title(r'$\beta\beta_{0\nu}$ effective mass')

    ax2.axhline(450, ls='--', color='purple', lw=1.5, label='KATRIN bound (450 meV)')
    ax2.axhline(200, ls=':',  color='purple', lw=1.5, label='Future sensitivity (200 meV)')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$m_{\rm min}$ (meV)', fontsize=12)
    ax2.set_ylabel(r'$m_\beta$ (meV)', fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
    ax2.set_title(r'KATRIN kinematic mass $m_\beta$')
    plt.tight_layout(); plt.show()
