"""
Neutrino cross-section landscape plot.
Reproduces the classic ν̄_e e⁻ cross-section vs energy figure
with updated experiments (2025).

Physics:
  ν̄_e + e⁻ → ν̄_e + e⁻  (elastic, NC t-channel Z + CC s-channel W)
  ν̄_e + e⁻ → W⁻ → all   (Glashow resonance at E_ν = M_W²/(2m_e) ≈ 6.3 PeV)

References:
  - Glashow, Phys. Rev. 118, 316 (1960)
  - Formaggio & Zeller, Rev. Mod. Phys. 84, 1307 (2012)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Physical constants ──
GF       = 1.1664e-5   # GeV⁻²  (Fermi constant)
me       = 0.511e-3    # GeV     (electron mass)
MW       = 80.377      # GeV     (W boson mass)
GammaW   = 2.085       # GeV     (W total width)
MZ       = 91.1876     # GeV     (Z boson mass)
GammaZ   = 2.4952      # GeV     (Z total width)
sin2tw   = 0.23122     # sin²θ_W
hbarc2   = 0.3894e-27  # GeV² → cm²  conversion (ℏc)²
cm2_to_mb = 1e27       # cm² → mb


def sigma_nue_bar_e(E_nu_eV):
    """
    Total cross section for ν̄_e + e⁻ → ν̄_e + e⁻
    including NC (t-channel Z) and CC (s-channel W with Glashow resonance).

    Parameters
    ----------
    E_nu_eV : array
        Neutrino energy in eV

    Returns
    -------
    sigma : array
        Cross section in mb (millibarns)
    """
    E_nu = E_nu_eV * 1e-9  # eV → GeV
    s = 2 * me * E_nu      # center-of-mass energy squared

    # ── Low-energy coefficients (4-Fermi limit) ──
    gL = -0.5 + sin2tw          # ≈ -0.269
    gR = sin2tw                 # ≈ 0.231

    # For ν̄_e e⁻: CC adds +1 to gL coefficient
    # dσ/dy = G_F² s / π × [(gL+1)² (1-y)² + gR²]
    # Integrating: σ = G_F² s / π × [(gL+1)²/3 + gR²]
    cL = gL + 1.0  # ≈ 0.731 (CC + NC left)
    cR = gR        # ≈ 0.231 (NC right)

    # ── NC contribution (t-channel Z, no resonance) ──
    sigma_nc = GF**2 * s / np.pi * (gL**2 / 3 + gR**2)

    # ── CC contribution with Glashow resonance (s-channel W⁻) ──
    # At low energy: σ_CC ~ G_F² s / π × (1/3) (from the (1-y)² integration)
    # Near resonance: Breit-Wigner
    #   σ_BW = (2J+1)/((2s1+1)(2s2+1)) × 4π/k² × Γ_i Γ_f / ((√s - M)² + Γ²/4)
    # For W⁻ (J=1), initial ν̄_e e⁻ (s1=s2=1/2):
    #   prefactor = 3/4
    # Γ(W→eν) ≈ Γ_W / 9 (3 lepton + 6 quark channels ≈ 9)
    # k² = s/4 (massless limit)

    Br_enu = 0.1086  # Br(W → eν_e)
    Gamma_enu = Br_enu * GammaW

    # Breit-Wigner for s-channel W → e ν_e (elastic)
    # Using relativistic BW: σ = 3π/(s) × Γ_enu² × s / ((s-MW²)² + MW² ΓW²)
    # = 3π Γ_enu² / ((s-MW²)² + MW² ΓW²)
    sigma_cc_res = (3.0 * np.pi * Gamma_enu**2 * s /
                    ((s - MW**2)**2 + MW**2 * GammaW**2))

    # Combined (interference is small for the total, we add)
    sigma_total = sigma_nc + sigma_cc_res

    # Convert GeV⁻² → cm² → mb
    sigma_mb = sigma_total * hbarc2 * cm2_to_mb

    return sigma_mb


def sigma_glashow_total(E_nu_eV):
    """
    Total Glashow resonance cross section:
    ν̄_e + e⁻ → W⁻ → anything (all W decay channels)
    """
    E_nu = E_nu_eV * 1e-9
    s = 2 * me * E_nu

    # σ = 3π Γ_enu Γ_W / ((s-MW²)² + MW² ΓW²) × s
    Br_enu = 0.1086
    Gamma_enu = Br_enu * GammaW

    sigma_total = (3.0 * np.pi * Gamma_enu * GammaW * s /
                   ((s - MW**2)**2 + MW**2 * GammaW**2))

    sigma_mb = sigma_total * hbarc2 * cm2_to_mb
    return sigma_mb


# ── Experiment categories (updated 2025) ──
experiments = {
    'Big Bang': {
        'energy_range': (1e-4, 1e-1),   # eV (CνB, ~0.2 meV thermal)
        'sigma_range':  (1e-32, 1e-26),  # mb
        'color': '#d8c8e8',              # lavender
        'experiments': ['PTOLEMY'],
    },
    'Solar': {
        'energy_range': (5e4, 2e7),
        'sigma_range':  (1e-29, 1e-24),
        'color': '#fff59d',
        'experiments': ['Borexino', 'Super-K', 'SNO+',
                        'JUNO', 'Hyper-K', 'DUNE'],
    },
    'Terrestrial': {
        'energy_range': (5e5, 5e7),
        'sigma_range':  (1e-27, 1e-13),
        'color': '#ffe0b2',
        'experiments': ['KamLAND', 'Borexino', 'SNO+', 'JUNO'],
    },
    'Reactor': {
        'energy_range': (1e6, 1e7),
        'sigma_range':  (1e-28, 1e-11),
        'color': '#c8e6c9',
        'experiments': ['JUNO', 'RENO', 'TAO',
                        'PROSPECT-II', 'DANSS',
                        'STEREO'],
    },
    'SuperNova': {
        'energy_range': (2e6, 1e8),
        'sigma_range':  (1e-24, 1e-5),
        'color': '#ffcdd2',
        'experiments': ['Super-K', 'JUNO', 'Hyper-K',
                        'IceCube', 'DUNE', 'KM3NeT'],
    },
    'Accelerator': {
        'energy_range': (1e8, 5e11),
        'sigma_range':  (1e-23, 1e-15),
        'color': '#f8bbd0',
        'experiments': ['T2K', 'NOvA', 'SBND', 'ICARUS',
                        'DUNE', 'T2HK', 'MINERvA',
                        'FASERν', 'SND@LHC'],
    },
    'Atmospheric': {
        'energy_range': (1e8, 1e14),
        'sigma_range':  (1e-23, 1e-16),
        'color': '#bbdefb',
        'experiments': ['Super-K', 'IceCube/DeepCore',
                        'KM3NeT/ORCA', 'Hyper-K',
                        'DUNE', 'INO'],
    },
    'Cosmic': {
        'energy_range': (1e13, 1e19),
        'sigma_range':  (1e-23, 1e-16),
        'color': '#90a4ae',
        'experiments': ['IceCube', 'IceCube-Gen2',
                        'KM3NeT/ARCA', 'Baikal-GVD',
                        'P-ONE', 'PUEO',
                        'RNO-G', 'GRAND',
                        'Trinity'],
    },
}


def draw_box(ax, energy_range, sigma_range, color, alpha=0.35):
    """Draw a rectangular region for an experiment category."""
    e_lo, e_hi = energy_range
    s_lo, s_hi = sigma_range

    rect = plt.Rectangle(
        (e_lo, s_lo), e_hi - e_lo, s_hi - s_lo,
        facecolor=color, edgecolor='grey',
        alpha=alpha, linewidth=0.8, zorder=1
    )
    ax.add_patch(rect)


def make_plot(savename='nu_xsection_landscape.png', dpi=200):
    """Generate the neutrino cross-section landscape figure."""

    # ── Compute cross sections ──
    E = np.logspace(-1, 19, 5000)  # eV

    sigma_elastic = sigma_nue_bar_e(E)
    sigma_glashow = sigma_glashow_total(E)

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw experiment category boxes
    for name, info in experiments.items():
        draw_box(ax, info['energy_range'], info['sigma_range'], info['color'])

        e_mid = np.sqrt(info['energy_range'][0] * info['energy_range'][1])
        s_top = info['sigma_range'][1]
        s_bot = info['sigma_range'][0]

        # Category label — bold, large, above the box
        ax.text(e_mid, s_top * 3, name,
                fontsize=14, fontweight='bold', ha='center', va='bottom',
                color='#222222', zorder=5)

        # Experiment list — inside the box, centered
        exp_text = '\n'.join(info['experiments'])
        s_center = np.sqrt(s_top * s_bot)  # geometric center in log scale
        ax.text(e_mid, s_center, exp_text,
                fontsize=8, ha='center', va='center',
                color='#444444', linespacing=1.4, zorder=5)

    # ── Cross-section curves ──
    ax.loglog(E, sigma_elastic, 'k-', linewidth=1.8, zorder=4,
              label=r'$\bar{\nu}_e\, e^- \to \bar{\nu}_e\, e^-$')
    ax.loglog(E, sigma_glashow, 'r--', linewidth=1.2, zorder=4, alpha=0.7,
              label=r'$\bar{\nu}_e\, e^- \to W^- \to$ all (Glashow)')

    # ── Resonance annotation ──
    E_res = MW**2 / (2 * me) * 1e9  # eV
    sigma_peak = sigma_glashow_total(np.array([E_res]))[0]
    ax.annotate(f'Glashow resonance\n$E_\\nu = M_W^2/(2m_e)$\n$\\approx 6.3$ PeV',
                xy=(E_res, sigma_peak),
                xytext=(E_res * 0.01, sigma_peak * 0.5),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                color='red', zorder=6)

    # ── Axes ──
    ax.set_xlim(1e-1, 1e19)
    ax.set_ylim(1e-32, 1e0)
    ax.set_xlabel('Neutrino Energy (eV)', fontsize=14)
    ax.set_ylabel(r'Cross Section (mb)', fontsize=14)
    ax.set_title(r'Neutrino interaction landscape: $\bar{\nu}_e\, e^-$ cross section',
                 fontsize=15, fontweight='bold', pad=15)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, which='major', alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(savename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved: {savename}')
    plt.close(fig)
    return savename


if __name__ == '__main__':
    make_plot()
