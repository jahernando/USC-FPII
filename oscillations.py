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
