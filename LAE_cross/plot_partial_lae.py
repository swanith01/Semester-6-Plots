#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_partial_lae.py
-------------------
Plot the LAE lightcone from the per-slice CHECKPOINTS of a run that has not
yet reached the assembly step (e.g. killed by walltime). Read-only: touches
nothing the pipeline writes. Assembles whatever snap_*.npz exist, applies the
same LAE cut as cell_3b.py Part 3, and makes the y-z panel.

Usage:
    python plot_partial_lae.py --seed 2 \
        --snap-dir /user1/swanith/SiMPLE-Gen/SiMPLEGen/data/seed_2/lightcone_lae/snapshots \
        --out      /user1/swanith/kSZ2_halo_project/plots/LAE_partial_seed2.png
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless on a compute node
import matplotlib.pyplot as plt

# ── LAE cut (identical to cell_3b.py) ──────────────────────────────────────
REW_CUT  = 10.0      # Angstrom
LLYA_CUT = 1e42      # erg/s

# ── geometry (matches CELL 1 / run_lightcone.py) ───────────────────────────
BOX     = 400.0
HII_DIM = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--snap-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(args.snap_dir, "snap_*.npz")))
    if not fs:
        raise SystemExit(f"no checkpoints found in {args.snap_dir}")
    print(f"found {len(fs)} checkpoints")

    # ── assemble from checkpoints (mirrors run_lightcone.py end block) ─────
    all_LLya, all_REW, all_damp, all_coords, all_z = [], [], [], [], []
    for f in fs:
        d = np.load(f)
        if bool(d["empty"]):
            continue
        n = len(d["halomass"])
        all_LLya.append(d["LLya"])
        all_REW.append(d["REW"])
        all_damp.append(d["damping"])
        all_coords.append(d["coords"])
        all_z.append(np.full(n, float(d["z_slice"]), dtype=np.float32))

    LLya   = np.concatenate(all_LLya)
    REW    = np.concatenate(all_REW)
    damp   = np.concatenate(all_damp)
    coords = np.vstack(all_coords)
    zred   = np.concatenate(all_z)

    LLya_obs = LLya * damp
    is_LAE   = (REW >= REW_CUT) & (LLya_obs >= LLYA_CUT) & (damp > 0)

    print(f"  total halos      : {len(zred):,}")
    print(f"  LAEs (after cut) : {int(is_LAE.sum()):,}")
    print(f"  z range          : {zred.min():.4f} – {zred.max():.4f}")
    print(f"  unique z (slices): {len(np.unique(zred))}")

    # ── z-axis from the distinct slice redshifts (the fix's continuum) ─────
    z_slices = np.unique(zred)
    n_lc     = len(z_slices)
    cell     = BOX / HII_DIM

    yi = np.clip((coords[:, 1] / cell).astype(int), 0, HII_DIM - 1)
    # map each halo's z_slice to its column index
    zi = np.searchsorted(z_slices, zred)
    zi = np.clip(zi, 0, n_lc - 1)

    L_grid = np.zeros((HII_DIM, n_lc))
    N_grid = np.zeros((HII_DIM, n_lc))
    lae    = is_LAE
    np.add.at(L_grid, (yi[lae], zi[lae]), LLya_obs[lae])
    np.add.at(N_grid, (yi[lae], zi[lae]), 1.0)

    with np.errstate(divide="ignore"):
        logL = np.log10(np.where(L_grid > 0, L_grid, np.nan))
        logN = np.where(N_grid > 0, np.log10(N_grid), np.nan)

    # ── plot (same style as cell_3b.py Part 3) ─────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

    im0 = axes[0].imshow(logL, aspect="auto", origin="lower",
                         extent=[z_slices[0], z_slices[-1], 0, BOX],
                         cmap="inferno", interpolation="nearest")
    plt.colorbar(im0, ax=axes[0], pad=0.01,
                 label=r"$\log_{10}\sum L_{\rm Ly\alpha,obs}$ [erg s$^{-1}$]")
    axes[0].set_xlabel(r"Redshift $z$"); axes[0].set_ylabel(r"y [cMpc]")
    axes[0].set_title(r"Summed observed $L_{\rm Ly\alpha}$ per pixel")

    im1 = axes[1].imshow(logN, aspect="auto", origin="lower",
                         extent=[z_slices[0], z_slices[-1], 0, BOX],
                         cmap="inferno", interpolation="nearest")
    plt.colorbar(im1, ax=axes[1], pad=0.01,
                 label=r"$\log_{10} N_{\rm LAE}$ per pixel")
    axes[1].set_xlabel(r"Redshift $z$"); axes[1].set_ylabel(r"y [cMpc]")
    axes[1].set_title(r"LAE count per pixel")

    fig.suptitle(f"Seed {args.seed} — PARTIAL LAE lightcone "
                 f"({n_lc} slices, z={z_slices[0]:.2f}-{z_slices[-1]:.2f}, "
                 f"{int(lae.sum()):,} LAEs)",
                 fontsize=15, fontweight="bold")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    fig.savefig(args.out.replace(".png", ".pdf"))
    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()
