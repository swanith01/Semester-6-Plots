#!/usr/bin/env python
# =============================================================================
# run_lae_diagnostics.py
#
# Standalone, notebook-free port of CELL 3b for ONE seed.
# Produces three diagnostics from the SiMPLE-Gen LAE catalogue:
#
#   1. LAE overlay on x_HI            (3 representative redshifts)
#   2. Lyman-alpha luminosity function (3 redshift bins)
#   3. Observed L_Lya lightcone        (y-z panel, binned)   <- new
#
# Usage:
#     python run_lae_diagnostics.py --seed 1
#
# Rebuilds InputParameters + loads the cached lightcone for the xHI
# background and lightcone geometry, then loads the LAE catalogue from
# SiMPLE-Gen/SiMPLEGen/data/seed_N/lightcone_lae/.
# =============================================================================

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import py21cmfast as p21c
from astropy.cosmology import FlatLambdaCDM

# =============================================================================
# CONFIG  (mirrors CELL 1 / CELL 2 / CELL 3b)
# =============================================================================

CACHE_SUBDIR  = "kSZ2_halo_project/cache"
PLOT_SUBDIR   = "kSZ2_halo_project/plots/LAE_diagnostics"
SIMPLEGEN_DATA = "/user1/swanith/SiMPLE-Gen/SiMPLEGen/data"

# Simulation parameters (CELL 1)
Z_MIN, Z_MAX, Z_STEP_FACTOR = 5.0, 20.0, 1.02
HII_DIM, BOX_LEN            = 32, 400.0
N_THREADS                   = 32
SAMPLER_MIN_MASS            = 1e8
SAMPLER_BUFFER_FACTOR       = 2.0
Z_HEAT_MAX                  = 20.0

# LAE diagnostic settings (CELL 3b)
Z_TARGETS = [5.6245, 6.6095, 7.0752]
DZ_HALO   = 0.10
REW_CUT   = 10.0      # Angstrom
LLYA_CUT  = 1e42      # erg/s
DZ_LF_BIN = 0.30

cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086, Ob0=0.0489)


def build_inputs(seed):
    """Rebuild the per-seed InputParameters (identical to the worker)."""
    node_z = np.array(p21c.get_logspaced_redshifts(
        min_redshift=Z_MIN, max_redshift=Z_MAX, z_step_factor=Z_STEP_FACTOR))
    return p21c.InputParameters(
        node_redshifts=node_z, random_seed=seed,
        simulation_options=p21c.SimulationOptions(
            HII_DIM=HII_DIM, BOX_LEN=BOX_LEN, N_THREADS=N_THREADS,
            Z_HEAT_MAX=Z_HEAT_MAX, SAMPLER_MIN_MASS=SAMPLER_MIN_MASS,
            SAMPLER_BUFFER_FACTOR=SAMPLER_BUFFER_FACTOR),
        matter_options=p21c.MatterOptions(
            KEEP_3D_VELOCITIES=True,
            USE_INTERPOLATION_TABLES='hmf-interpolation'),
        astro_options=p21c.AstroOptions(INHOMO_RECO=True, USE_TS_FLUCT=True),
    )


def load_lae(seed):
    """Load the SiMPLE-Gen LAE catalogue for one seed."""
    d = os.path.join(SIMPLEGEN_DATA, f"seed_{seed}", "lightcone_lae")
    out = {k: np.load(os.path.join(d, f"{k}.npy"))
           for k in ["LLya", "REW", "damping", "halomass",
                     "coords", "redshifts"]}
    out["LLya_obs"] = out["LLya"] * out["damping"]
    out["is_LAE"]   = ((out["REW"] >= REW_CUT) &
                       (out["LLya_obs"] >= LLYA_CUT) &
                       (out["damping"] > 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--root", type=str, default=os.getcwd())
    args = ap.parse_args()

    seed = args.seed
    root = os.path.abspath(args.root)

    plot_dir = os.path.join(root, PLOT_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    lc_path = os.path.join(root, CACHE_SUBDIR, f"seed_{seed}", "lightcone.h5")

    print("=" * 70)
    print(f"LAE diagnostics  --  seed {seed}")
    print("=" * 70)
    print(f"  lightcone : {lc_path}")
    print(f"  LAE data  : {SIMPLEGEN_DATA}/seed_{seed}/lightcone_lae/")
    print(f"  plots     : {plot_dir}")

    if not os.path.exists(lc_path):
        print(f"\n  FATAL: no lightcone.h5 for seed {seed}")
        sys.exit(3)

    # --- load lightcone + LAE catalogue --------------------------------
    inputs    = build_inputs(seed)
    lightcone = p21c.LightCone.from_file(lc_path, safe=False)
    xHI_lc    = np.asarray(lightcone.lightcones['neutral_fraction'])
    z_lc      = np.asarray(lightcone.lightcone_redshifts, dtype=np.float64)
    n_lc      = len(z_lc)

    data = load_lae(seed)
    n_lae = int(data["is_LAE"].sum())
    print(f"  loaded {len(data['LLya']):,} halos, "
          f"{n_lae:,} pass LAE cut "
          f"(REW>={REW_CUT}A, L_obs>={LLYA_CUT:.0e})")

    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm',
        'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 13,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
    })

    # ===================================================================
    # PART 1 — LAE overlay on x_HI
    # ===================================================================
    print("\n[1/3] LAE overlay on x_HI ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                             constrained_layout=True)
    for ax, z_target in zip(axes, Z_TARGETS):
        z_idx  = int(np.argmin(np.abs(z_lc - z_target)))
        xHI_sl = xHI_lc[:, :, z_idx].T

        im = ax.imshow(xHI_sl, origin='lower', extent=[0, BOX_LEN, 0, BOX_LEN],
                       cmap='Blues', vmin=0.0, vmax=1.0,
                       interpolation='bilinear')
        plt.colorbar(im, ax=ax, label=r'$x_{\rm HI}$',
                     fraction=0.046, pad=0.02)

        zmask = np.abs(data["redshifts"] - z_target) <= DZ_HALO
        if zmask.sum() == 0:
            ax.set_title(f"$z={z_target:.3f}$  no halos in window")
            continue
        coords_z = data["coords"][zmask]
        is_lae_z = data["is_LAE"][zmask]
        c_non = coords_z[~is_lae_z]
        c_lae = coords_z[is_lae_z]

        ax.scatter(c_non[:, 0], c_non[:, 1], s=3, color='0.4',
                   alpha=0.3, edgecolors='none',
                   label=f'Non-LAE ({(~is_lae_z).sum():,})', zorder=2)
        ax.scatter(c_lae[:, 0], c_lae[:, 1], s=14, color='red',
                   alpha=0.9, edgecolors='none',
                   label=f'LAE ({is_lae_z.sum():,})', zorder=3)
        ax.set_xlim(0, BOX_LEN); ax.set_ylim(0, BOX_LEN)
        ax.set_xlabel('x [cMpc]'); ax.set_ylabel('y [cMpc]')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.75)
        ax.set_title(f"$z = {z_target:.3f}$  "
                     f"($\\bar x_{{\\rm HI}}={xHI_sl.mean():.2f}$)")

    fig.suptitle(f"Seed {seed} — LAEs (red) over $x_{{\\rm HI}}$",
                 fontsize=14, fontweight='bold')
    f1 = os.path.join(plot_dir, f"LAE_overlay_seed{seed}.png")
    fig.savefig(f1, dpi=200); fig.savefig(f1.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  saved {f1}")

    # ===================================================================
    # PART 2 — Lyman-alpha luminosity function
    # ===================================================================
    print("\n[2/3] Lyman-alpha luminosity function ...")
    logL_edges = np.arange(41.5, 44.0, 0.2)
    logL_cen   = 0.5 * (logL_edges[1:] + logL_edges[:-1])
    dlogL      = logL_edges[1] - logL_edges[0]

    fig, axes = plt.subplots(1, len(Z_TARGETS),
                             figsize=(6 * len(Z_TARGETS), 5.5),
                             constrained_layout=True, sharey=True)
    for ax, z_target in zip(axes, Z_TARGETS):
        zlo, zhi = z_target - DZ_LF_BIN / 2, z_target + DZ_LF_BIN / 2
        chi_lo = cosmo.comoving_distance(zlo).to_value('Mpc')
        chi_hi = cosmo.comoving_distance(zhi).to_value('Mpc')
        vol    = (BOX_LEN ** 2) * (chi_hi - chi_lo)

        sel = ((data["redshifts"] >= zlo) & (data["redshifts"] < zhi)
               & data["is_LAE"])
        n_sel = int(sel.sum())
        if n_sel == 0:
            ax.text(0.5, 0.5, f"no LAEs in\n[{zlo:.2f}, {zhi:.2f})",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"$z \\in [{zlo:.2f}, {zhi:.2f})$  N=0")
            ax.set_xlabel(r'$\log_{10}(L_{\rm Ly\alpha,obs})$')
            continue

        logL = np.log10(data["LLya_obs"][sel])
        counts, _ = np.histogram(logL, bins=logL_edges)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi     = counts / (vol * dlogL)
            phi_err = np.sqrt(counts) / (vol * dlogL)
        good = counts > 0
        ax.errorbar(logL_cen[good], phi[good], yerr=phi_err[good],
                    fmt='o-', color='C3', capsize=2, lw=1.4, ms=5,
                    label=f'N = {n_sel:,}')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\log_{10}(L_{\rm Ly\alpha,obs}\,/\,\mathrm{erg\,s^{-1}})$')
        if ax is axes[0]:
            ax.set_ylabel(r'$\phi$  [cMpc$^{-3}$ dex$^{-1}$]')
        ax.set_title(f"$z \\in [{zlo:.2f}, {zhi:.2f})$")
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(f"Seed {seed} — Lyman-$\\alpha$ luminosity function",
                 fontsize=14, fontweight='bold')
    f2 = os.path.join(plot_dir, f"LF_seed{seed}.png")
    fig.savefig(f2, dpi=200); fig.savefig(f2.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  saved {f2}")

    # ===================================================================
    # PART 3 — observed L_Lya lightcone (y-z panel)
    #
    # Bin every halo onto the (HII_DIM, n_lc) lightcone grid using its
    # (y, redshift). Two panels: summed observed L_Lya per pixel, and
    # LAE count per pixel. Striped (one slab per node) by construction.
    # ===================================================================
    print("\n[3/3] observed L_Lya lightcone panel ...")
    cell_size = BOX_LEN / HII_DIM

    # y index from coords[:,1]; z index from nearest LC redshift
    yi = np.clip((data["coords"][:, 1] / cell_size).astype(int),
                 0, HII_DIM - 1)
    zi = np.clip(np.searchsorted(
        0.5 * (z_lc[1:] + z_lc[:-1]), data["redshifts"]),
        0, n_lc - 1)

    L_grid = np.zeros((HII_DIM, n_lc), dtype=np.float64)
    N_grid = np.zeros((HII_DIM, n_lc), dtype=np.float64)
    lae    = data["is_LAE"]
    np.add.at(L_grid, (yi[lae], zi[lae]), data["LLya_obs"][lae])
    np.add.at(N_grid, (yi[lae], zi[lae]), 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             constrained_layout=True)

    with np.errstate(divide='ignore'):
        logL = np.log10(np.where(L_grid > 0, L_grid, np.nan))
    im0 = axes[0].imshow(logL, aspect='auto', origin='lower',
                         extent=[z_lc[0], z_lc[-1], 0, BOX_LEN],
                         cmap='inferno', interpolation='nearest')
    plt.colorbar(im0, ax=axes[0], pad=0.01,
                 label=r'$\log_{10}\sum L_{\rm Ly\alpha,obs}$ [erg s$^{-1}$]')
    axes[0].set_xlabel(r'Redshift  $z$')
    axes[0].set_ylabel(r'y  [cMpc]')
    axes[0].set_title(r'Summed observed $L_{\rm Ly\alpha}$ per pixel')

    im1 = axes[1].imshow(np.log10(N_grid + 1), aspect='auto', origin='lower',
                         extent=[z_lc[0], z_lc[-1], 0, BOX_LEN],
                         cmap='inferno', interpolation='nearest')
    plt.colorbar(im1, ax=axes[1], pad=0.01,
                 label=r'$\log_{10}(N_{\rm LAE}+1)$ per pixel')
    axes[1].set_xlabel(r'Redshift  $z$')
    axes[1].set_ylabel(r'y  [cMpc]')
    axes[1].set_title(r'LAE count per pixel')

    fig.suptitle(f"Seed {seed} — LAE lightcone (y-z slice)  "
                 f"[{n_lae:,} LAEs]", fontsize=15, fontweight='bold')
    f3 = os.path.join(plot_dir, f"LAE_lightcone_seed{seed}.png")
    fig.savefig(f3, dpi=200); fig.savefig(f3.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  saved {f3}")

    print("\n" + "=" * 70)
    print(f"LAE diagnostics complete  --  seed {seed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
