#!/usr/bin/env python3
"""
type_b_grids.py
===============
Computes Type B lightcones (with physical values) for all 5 seeds.
Run on the cluster after stitch_lightcones.py has finished.

Produces per seed:
    lc_halos_mass.npz  — (64, 512) average halo mass per cell  [Msun]
    lc_lae_lum.npz     — (64, 512) average Lya luminosity      [erg/s]
    lc_lbg_muv.npz     — (64, 512) average MUV magnitude

These are small files (~500KB each) — rsync to desktop for proportional
marker plotting.

Usage:
    python type_b_grids.py              # all seeds
    python type_b_grids.py --seed 1    # single seed
"""

import numpy as np
import os
import sys
import argparse
import logging
import json
import traceback
from datetime import datetime
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# ══════════════════════════════════════════════════════════════════════════════
# 0. Parameters  (must match stitch_lightcones.py)
# ══════════════════════════════════════════════════════════════════════════════

COSMO      = FlatLambdaCDM(H0=67.77, Om0=0.3086, Ob0=0.0489)
H_LITTLE   = 0.6777

BOX_LEN    = 400.0
NGRID      = 64
CELL       = BOX_LEN / NGRID        # 6.25 cMpc

ZMIN       = 5.12
ZMAX       = 19.89
N_LC_PIX   = 512
ANGLE_DEG  = 10.0
X_SLICE    = 32

MASS_CUT   = 1e10                   # Msun — halo mass cut
MUV_CUT    = -17.0                  # LBG brightness cut

ALL_SEEDS  = [1, 2, 3, 4, 5]

ROOT_HALOS = "/user1/swanith/lightcone_halos_64/catalogues"
ROOT_LAE   = "/user1/jahaan/swanith/obs_prop"
ROOT_LBG   = "/user1/jahaan/swanith/int_prop"
OUT_ROOT   = "/user1/swanith/kSZ2_halo_project/26Jun2026_400Mpc_halo_LAE_LBG"

z_arr      = np.linspace(ZMIN, ZMAX, N_LC_PIX)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Logging
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(seed):
    log_dir  = os.path.join(OUT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir,
                f'typeB_seed{seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger   = logging.getLogger(f'typeB_seed{seed}')
    logger.setLevel(logging.DEBUG)
    fh  = logging.FileHandler(log_file)
    ch  = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%H:%M:%S')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

# ══════════════════════════════════════════════════════════════════════════════
# 2. Helpers  (same as stitch_lightcones.py)
# ══════════════════════════════════════════════════════════════════════════════

def comoving_distance_cMpc(z):
    return COSMO.comoving_distance(z).to(u.Mpc).value

def comoving_pixel(z, z0=ZMIN):
    d = comoving_distance_cMpc(z) - comoving_distance_cMpc(z0)
    return int(d / CELL) % NGRID

def periodic(n, ngrid=NGRID):
    return int(n) % ngrid

def rotate_index(i, j, k, angle_deg=ANGLE_DEG, ngrid=NGRID):
    a  = np.deg2rad(angle_deg)
    ir = np.cos(a)*i - np.sin(a)*j
    jr = np.sin(a)*i + np.cos(a)*j
    return periodic(ir), periodic(jr), periodic(k)

def get_slab(box, y_cell):
    """Extract rotated (NGRID, NGRID) slab from 3D box at LoS position y_cell."""
    slab = np.zeros((NGRID, NGRID), dtype=np.float64)
    for ix in range(NGRID):
        for iy in range(NGRID):
            rx, ry, rz = rotate_index(ix, iy, y_cell)
            slab[ix, iy] = box[rx, ry, rz]
    return slab

def get_snapshot_redshifts(seed, logger):
    coeval_dir = os.path.join(ROOT_HALOS, f'seed_{seed}')
    dirs  = [d for d in os.listdir(coeval_dir) if d.startswith('coeval_z')]
    zvals = []
    for d in dirs:
        try:
            z = float(d.replace('coeval_z', ''))
            if ZMIN - 0.5 <= z <= ZMAX + 0.5:
                zvals.append(z)
        except ValueError:
            pass
    zvals = np.array(sorted(zvals))
    logger.info(f"  Found {len(zvals)} snapshots: z={zvals[0]:.4f}→{zvals[-1]:.4f}")
    return zvals

def find_nearest(z, snap_z):
    return np.argmin(np.abs(snap_z - z))

# ══════════════════════════════════════════════════════════════════════════════
# 3. Type B grid builders
# ══════════════════════════════════════════════════════════════════════════════

def build_halo_mass_grid(seed, z, logger):
    """
    Build (NGRID,NGRID,NGRID) grids:
      occ  — occupation count
      mval — average halo mass per cell  [Msun]
    """
    cpath = os.path.join(ROOT_HALOS, f'seed_{seed}', f'coords_z{z:.4f}.npy')
    mpath = os.path.join(ROOT_HALOS, f'seed_{seed}', f'masses_z{z:.4f}.npy')
    if not os.path.exists(cpath):
        raise FileNotFoundError(f"Missing: {cpath}")

    coords = np.load(cpath, mmap_mode='r')   # (N,3) cMpc
    masses = np.load(mpath, mmap_mode='r')   # (N,)  Msun

    sel    = masses > MASS_CUT
    coords = coords[sel]
    masses = masses[sel]

    occ  = np.zeros((NGRID, NGRID, NGRID), dtype=np.float32)
    mval = np.zeros((NGRID, NGRID, NGRID), dtype=np.float64)

    ix_arr = (coords[:, 0] / CELL).astype(int) % NGRID
    iy_arr = (coords[:, 1] / CELL).astype(int) % NGRID
    iz_arr = (coords[:, 2] / CELL).astype(int) % NGRID

    for ix, iy, iz, m in zip(ix_arr, iy_arr, iz_arr, masses):
        occ [ix, iy, iz] += 1.0
        mval[ix, iy, iz] += m

    mask = occ > 0
    mval[mask] /= occ[mask]
    return occ, mval


def build_lae_lum_grid(seed, z, logger):
    """
    Build (NGRID,NGRID,NGRID) grids:
      occ  — LAE occupation count
      lgrid — average Lya luminosity per cell  [erg/s]
    ids are 0-indexed into full coords array.
    lum is aligned 1-to-1 with ids.
    """
    idpath  = os.path.join(ROOT_LAE, 'halo_ids_obs',
                           f'halo_ids_obs_z{z:.4f}_64_400_s{seed}.npy')
    lumpath = os.path.join(ROOT_LAE, 'lya_lum_obs',
                           f'lya_lum_obs_z{z:.4f}_64_400_s{seed}.npy')
    cpath   = os.path.join(ROOT_HALOS, f'seed_{seed}', f'coords_z{z:.4f}.npy')

    if not os.path.exists(idpath):
        logger.warning(f"  LAE ids missing z={z:.4f} s{seed} → empty grid")
        return (np.zeros((NGRID,NGRID,NGRID), dtype=np.float32),
                np.zeros((NGRID,NGRID,NGRID), dtype=np.float64))

    ids    = np.load(idpath,  mmap_mode='r')
    lum    = np.load(lumpath, mmap_mode='r')
    coords = np.load(cpath,   mmap_mode='r')

    lae_coords = coords[ids]            # 0-indexed into full array
    occ   = np.zeros((NGRID,NGRID,NGRID), dtype=np.float32)
    lgrid = np.zeros((NGRID,NGRID,NGRID), dtype=np.float64)

    ix_arr = (lae_coords[:, 0] / CELL).astype(int) % NGRID
    iy_arr = (lae_coords[:, 1] / CELL).astype(int) % NGRID
    iz_arr = (lae_coords[:, 2] / CELL).astype(int) % NGRID

    for ix, iy, iz, l in zip(ix_arr, iy_arr, iz_arr, lum):
        occ  [ix, iy, iz] += 1.0
        lgrid[ix, iy, iz] += l

    mask = occ > 0
    lgrid[mask] /= occ[mask]
    return occ, lgrid


def build_lbg_muv_grid(seed, z, logger):
    """
    Build (NGRID,NGRID,NGRID) grids:
      occ   — LBG occupation count (after MUV cut)
      mgrid — average MUV per cell
    ids are 0-indexed into full coords array.
    MUV is aligned 1-to-1 with ids.
    """
    idpath  = os.path.join(ROOT_LBG, 'halo_ids_lbg',
                           f'halo_ids_lbg_z{z:.4f}_64_400_s{seed}.npy')
    muvpath = os.path.join(ROOT_LBG, 'MUV_lbg',
                           f'MUV_lbg_z{z:.4f}_64_400_s{seed}.npy')
    cpath   = os.path.join(ROOT_HALOS, f'seed_{seed}', f'coords_z{z:.4f}.npy')

    if not os.path.exists(idpath):
        logger.warning(f"  LBG ids missing z={z:.4f} s{seed} → empty grid")
        return (np.zeros((NGRID,NGRID,NGRID), dtype=np.float32),
                np.zeros((NGRID,NGRID,NGRID), dtype=np.float64))

    ids    = np.load(idpath,  mmap_mode='r')
    muv    = np.load(muvpath, mmap_mode='r')
    coords = np.load(cpath,   mmap_mode='r')

    bright     = muv < MUV_CUT
    lbg_coords = coords[ids[bright]]
    muv_sel    = muv[bright]

    occ   = np.zeros((NGRID,NGRID,NGRID), dtype=np.float32)
    mgrid = np.zeros((NGRID,NGRID,NGRID), dtype=np.float64)

    ix_arr = (lbg_coords[:, 0] / CELL).astype(int) % NGRID
    iy_arr = (lbg_coords[:, 1] / CELL).astype(int) % NGRID
    iz_arr = (lbg_coords[:, 2] / CELL).astype(int) % NGRID

    for ix, iy, iz, m in zip(ix_arr, iy_arr, iz_arr, muv_sel):
        occ  [ix, iy, iz] += 1.0
        mgrid[ix, iy, iz] += m

    mask = occ > 0
    mgrid[mask] /= occ[mask]
    return occ, mgrid

# ══════════════════════════════════════════════════════════════════════════════
# 4. Stitch Type B lightcone
# ══════════════════════════════════════════════════════════════════════════════

def stitch_type_b(snap_z, grids_occ, grids_val, logger):
    """
    Stitch Type B grids into two 2D lightcones:
      lc_occ : (NGRID, N_LC_PIX) — occupation  (same as Type A)
      lc_val : (NGRID, N_LC_PIX) — physical value (mass/lum/MUV)

    Uses nearest-snapshot (discrete tracers).
    Returns (lc_occ, lc_val).
    """
    snap_list = list(range(len(snap_z)))
    lc_occ = np.zeros((NGRID, N_LC_PIX), dtype=np.float32)
    lc_val = np.zeros((NGRID, N_LC_PIX), dtype=np.float64)

    for n, z in enumerate(z_arr):
        y_cell  = comoving_pixel(z)
        nearest = snap_list[find_nearest(z, snap_z)]

        slab_occ = get_slab(grids_occ[nearest], y_cell)
        slab_val = get_slab(grids_val[nearest], y_cell)

        lc_occ[:, n] = slab_occ[:, X_SLICE]
        lc_val[:, n] = slab_val[:, X_SLICE]

        if n % 100 == 0:
            logger.debug(f"  pixel {n}/{N_LC_PIX}  z={z:.3f}")

    return lc_occ, lc_val

# ══════════════════════════════════════════════════════════════════════════════
# 5. Per-seed pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_seed(seed):
    logger   = setup_logger(seed)
    seed_dir = os.path.join(OUT_ROOT, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Type B  SEED {seed}  started {datetime.now().isoformat()}")
    logger.info(f"{'='*60}")

    snap_z = get_snapshot_redshifts(seed, logger)

    # ── Halos + mass ──────────────────────────────────────────────────────────
    logger.info("Building halo+mass grids...")
    occ_h, val_h = {}, {}
    for i, z in enumerate(snap_z):
        try:
            occ_h[i], val_h[i] = build_halo_mass_grid(seed, z, logger)
        except Exception as e:
            logger.error(f"  Halo z={z:.4f}: {e}")
            occ_h[i] = np.zeros((NGRID,NGRID,NGRID), dtype=np.float32)
            val_h[i] = np.zeros((NGRID,NGRID,NGRID), dtype=np.float64)
        if i % 10 == 0:
            logger.info(f"  Halo grid {i+1}/{len(snap_z)}  z={z:.4f}")

    logger.info("Stitching halo+mass lightcone...")
    lc_h_occ, lc_h_mass = stitch_type_b(snap_z, occ_h, val_h, logger)
    out = os.path.join(seed_dir, 'lc_halos_mass.npz')
    np.savez_compressed(out, lc_occ=lc_h_occ, lc_val=lc_h_mass,
                        field='halos_mass', seed=seed,
                        zmin=ZMIN, zmax=ZMAX, ngrid=NGRID, box_len=BOX_LEN,
                        z_arr=z_arr)
    logger.info(f"Saved: {out}  "
                f"mass range: {lc_h_mass[lc_h_occ>0].min():.2e}"
                f"–{lc_h_mass[lc_h_occ>0].max():.2e} Msun")

    # ── LAEs + Lya luminosity ─────────────────────────────────────────────────
    logger.info("Building LAE+lum grids...")
    occ_l, val_l = {}, {}
    for i, z in enumerate(snap_z):
        try:
            occ_l[i], val_l[i] = build_lae_lum_grid(seed, z, logger)
        except Exception as e:
            logger.error(f"  LAE z={z:.4f}: {e}")
            occ_l[i] = np.zeros((NGRID,NGRID,NGRID), dtype=np.float32)
            val_l[i] = np.zeros((NGRID,NGRID,NGRID), dtype=np.float64)
        if i % 10 == 0:
            logger.info(f"  LAE grid {i+1}/{len(snap_z)}  z={z:.4f}")

    logger.info("Stitching LAE+lum lightcone...")
    lc_l_occ, lc_l_lum = stitch_type_b(snap_z, occ_l, val_l, logger)
    out = os.path.join(seed_dir, 'lc_lae_lum.npz')
    valid = lc_l_occ > 0
    np.savez_compressed(out, lc_occ=lc_l_occ, lc_val=lc_l_lum,
                        field='lae_lum', seed=seed,
                        zmin=ZMIN, zmax=ZMAX, ngrid=NGRID, box_len=BOX_LEN,
                        z_arr=z_arr)
    if valid.any():
        logger.info(f"Saved: {out}  "
                    f"log10(L) range: {np.log10(lc_l_lum[valid].min()):.2f}"
                    f"–{np.log10(lc_l_lum[valid].max()):.2f}")
    else:
        logger.info(f"Saved: {out}  (no LAEs in lightcone)")

    # ── LBGs + MUV ────────────────────────────────────────────────────────────
    logger.info("Building LBG+MUV grids...")
    occ_b, val_b = {}, {}
    for i, z in enumerate(snap_z):
        try:
            occ_b[i], val_b[i] = build_lbg_muv_grid(seed, z, logger)
        except Exception as e:
            logger.error(f"  LBG z={z:.4f}: {e}")
            occ_b[i] = np.zeros((NGRID,NGRID,NGRID), dtype=np.float32)
            val_b[i] = np.zeros((NGRID,NGRID,NGRID), dtype=np.float64)
        if i % 10 == 0:
            logger.info(f"  LBG grid {i+1}/{len(snap_z)}  z={z:.4f}")

    logger.info("Stitching LBG+MUV lightcone...")
    lc_b_occ, lc_b_muv = stitch_type_b(snap_z, occ_b, val_b, logger)
    out = os.path.join(seed_dir, 'lc_lbg_muv.npz')
    valid = lc_b_occ > 0
    np.savez_compressed(out, lc_occ=lc_b_occ, lc_val=lc_b_muv,
                        field='lbg_muv', seed=seed,
                        zmin=ZMIN, zmax=ZMAX, ngrid=NGRID, box_len=BOX_LEN,
                        z_arr=z_arr)
    if valid.any():
        logger.info(f"Saved: {out}  "
                    f"MUV range: {lc_b_muv[valid].min():.2f}"
                    f"–{lc_b_muv[valid].max():.2f}")
    else:
        logger.info(f"Saved: {out}  (no LBGs in lightcone)")

    logger.info(f"{'='*60}")
    logger.info(f"SEED {seed} Type B complete  {datetime.now().isoformat()}")
    logger.info(f"{'='*60}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, nargs='+', default=ALL_SEEDS)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Type B lightcone grids  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Seeds: {args.seed}")
    print(f"{'='*60}\n")

    for seed in args.seed:
        try:
            process_seed(seed)
        except Exception as e:
            print(f"FAILED seed {seed}: {e}\n{traceback.format_exc()}")

    print("\nAll done.")
