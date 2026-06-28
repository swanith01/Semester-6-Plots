"""
stitch_lightcones.py
====================
400 Mpc / 64^3 lightcone stitching for 5 seeds.
Fields: xHI, density, vz, halos, LAEs, LBGs.

Designed for swarm/pride cluster (PBS).
- Full checkpointing: each (seed, field) saved independently
- Logging: one log file per seed
- Error files: errors logged and skipped gracefully
- Resume-safe: skips already-completed (seed, field) pairs

Usage:
    python stitch_lightcones.py              # all seeds
    python stitch_lightcones.py --seed 3     # single seed
    python stitch_lightcones.py --seed 1 3 5 # multiple seeds
"""

import numpy as np
import os
import sys
import logging
import argparse
import traceback
import json
from datetime import datetime
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# ══════════════════════════════════════════════════════════════════════════════
# 0. Parameters
# ══════════════════════════════════════════════════════════════════════════════

# ── Cosmology ─────────────────────────────────────────────────────────────────
COSMO     = FlatLambdaCDM(H0=67.77, Om0=0.3086, Ob0=0.0489)
H_LITTLE  = 0.6777          # dimensionless h

# ── Box ───────────────────────────────────────────────────────────────────────
BOX_LEN   = 400.0           # cMpc (NOT cMpc/h)
NGRID     = 64
CELL      = BOX_LEN / NGRID # 6.25 cMpc

# ── Lightcone ─────────────────────────────────────────────────────────────────
ZMIN      = 5.12
ZMAX      = 19.89
N_LC_PIX  = 512             # pixels along LoS (enough for 400 Mpc range)
ANGLE_DEG = 10.0            # rotation to reduce box repetition
X_SLICE   = 32              # fixed transverse index for 2D slice

# ── Selection cuts ────────────────────────────────────────────────────────────
MASS_CUT_IDX = 10**9.5      # Msun — ids index into this mass-cut subset
MUV_CUT      = -17.0        # LBG brightness cut

# ── Seeds ─────────────────────────────────────────────────────────────────────
ALL_SEEDS = [1, 2, 3, 4, 5]

# ── Data paths ────────────────────────────────────────────────────────────────
ROOT_COEVAL = "/user1/swanith/lightcone_halos_64/catalogues"
ROOT_HALOS  = "/user1/swanith/lightcone_halos_64/catalogues"
ROOT_LAE    = "/user1/jahaan/swanith/obs_prop"
ROOT_LBG    = "/user1/jahaan/swanith/int_prop"

# ── Output ────────────────────────────────────────────────────────────────────
OUT_ROOT    = "/user1/swanith/kSZ2_halo_project/26Jun2026_400Mpc_halo_LAE_LBG"

# ── Fields to stitch ─────────────────────────────────────────────────────────
# (name, type) where type = 'continuous' or 'discrete'
FIELDS = [
    ('xH',     'continuous'),
    ('density', 'continuous'),
    ('vz',      'continuous'),
    ('halos',   'discrete'),
    ('lae',     'discrete'),
    ('lbg',     'discrete'),
]

# ══════════════════════════════════════════════════════════════════════════════
# 1. Setup logging
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(seed, out_root):
    log_dir = os.path.join(out_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'seed_{seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = logging.getLogger(f'seed_{seed}')
    logger.setLevel(logging.DEBUG)
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_file}")
    return logger

# ══════════════════════════════════════════════════════════════════════════════
# 2. Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def checkpoint_path(seed, field, out_root):
    return os.path.join(out_root, 'checkpoints', f'seed_{seed}_{field}.done')

def is_done(seed, field, out_root):
    return os.path.exists(checkpoint_path(seed, field, out_root))

def mark_done(seed, field, out_root, meta=None):
    os.makedirs(os.path.join(out_root, 'checkpoints'), exist_ok=True)
    cp = checkpoint_path(seed, field, out_root)
    with open(cp, 'w') as f:
        json.dump({'seed': seed, 'field': field,
                   'time': datetime.now().isoformat(),
                   'meta': meta or {}}, f, indent=2)

def output_path(seed, field, out_root):
    seed_dir = os.path.join(out_root, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)
    return os.path.join(seed_dir, f'lc_{field}.npz')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Snapshot discovery
# ══════════════════════════════════════════════════════════════════════════════

def get_snapshot_redshifts(seed, logger):
    """
    Read all available coeval redshifts for a seed from the directory names.
    Returns sorted numpy array of redshift floats.
    """
    coeval_dir = os.path.join(ROOT_COEVAL, f'seed_{seed}')
    dirs = [d for d in os.listdir(coeval_dir) if d.startswith('coeval_z')]
    zvals = []
    for d in dirs:
        try:
            z = float(d.replace('coeval_z', ''))
            # Only keep snapshots within our z range (with margin)
            if ZMIN - 0.5 <= z <= ZMAX + 0.5:
                zvals.append(z)
        except ValueError:
            logger.warning(f"Could not parse redshift from dir: {d}")
    zvals = np.array(sorted(zvals))
    logger.info(f"Found {len(zvals)} snapshots: z={zvals[0]:.4f} → z={zvals[-1]:.4f}")
    return zvals

def get_halo_redshifts(seed, logger):
    """Get redshifts for which halo catalogs exist."""
    halo_dir = os.path.join(ROOT_HALOS, f'seed_{seed}')
    files = [f for f in os.listdir(halo_dir) if f.startswith('coords_z')]
    zvals = []
    for f in files:
        try:
            z = float(f.replace('coords_z', '').replace('.npy', ''))
            if ZMIN - 0.5 <= z <= ZMAX + 0.5:
                zvals.append(z)
        except ValueError:
            pass
    return np.array(sorted(zvals))

# ══════════════════════════════════════════════════════════════════════════════
# 4. Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def comoving_distance_cMpc(z):
    """Comoving distance in cMpc (not cMpc/h)."""
    return COSMO.comoving_distance(z).to(u.Mpc).value

def comoving_pixel(z, z0=ZMIN):
    """Cell index along LoS corresponding to redshift z."""
    d = comoving_distance_cMpc(z) - comoving_distance_cMpc(z0)
    return int(d / CELL) % NGRID

def periodic(n, ngrid=NGRID):
    return int(n) % ngrid

def rotate_index(i, j, k, angle_deg=ANGLE_DEG, ngrid=NGRID):
    a = np.deg2rad(angle_deg)
    ir = np.cos(a)*i - np.sin(a)*j
    jr = np.sin(a)*i + np.cos(a)*j
    return periodic(ir, ngrid), periodic(jr, ngrid), periodic(k, ngrid)

def get_slab(box, y_cell):
    """
    Extract a rotated 2D slab from a 3D box at LoS position y_cell.
    Returns array of shape (NGRID, NGRID) — the transverse plane.
    Uses dtype=float64 to avoid float32 overflow for large values (e.g. luminosities).
    """
    slab = np.zeros((NGRID, NGRID), dtype=np.float64)
    for ix in range(NGRID):
        for iy in range(NGRID):
            rx, ry, rz = rotate_index(ix, iy, y_cell)
            slab[ix, iy] = box[rx, ry, rz]
    return slab

def find_nearest_snapshot(z, snap_z):
    """Return index of nearest snapshot redshift."""
    return np.argmin(np.abs(snap_z - z))

# ══════════════════════════════════════════════════════════════════════════════
# 5. Box loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_field_box(seed, z, field_name, logger):
    """Load a continuous field box (xHI, density, vz)."""
    field_map = {
        'xH':      'neutral_fraction.npy',
        'density': 'density.npy',
        'vz':      'velocity_z.npy',
    }
    fname = field_map[field_name]
    path  = os.path.join(ROOT_COEVAL, f'seed_{seed}', f'coeval_z{z:.4f}', fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    box = np.load(path, mmap_mode='r')
    if field_name == 'vz':
        box = np.array(box) / (1 + z) * 3.086e19
    return box

def load_halo_grid(seed, z, logger):
    """
    Load halo catalog and bin onto (NGRID,NGRID,NGRID) occupation grid.
    Returns grid of shape (NGRID, NGRID, NGRID).
    """
    cpath = os.path.join(ROOT_HALOS, f'seed_{seed}', f'coords_z{z:.4f}.npy')
    mpath = os.path.join(ROOT_HALOS, f'seed_{seed}', f'masses_z{z:.4f}.npy')
    if not os.path.exists(cpath) or not os.path.exists(mpath):
        raise FileNotFoundError(f"Missing halo files at z={z:.4f}")
    coords  = np.load(cpath, mmap_mode='r')   # (N,3) cMpc, 0-indexed
    masses  = np.load(mpath, mmap_mode='r')   # (N,) Msun
    sel     = masses > 1e10                   # halo mass cut
    coords  = coords[sel]
    grid    = np.zeros((NGRID, NGRID, NGRID), dtype=np.float32)
    ix_arr  = (coords[:, 0] / CELL).astype(int) % NGRID
    iy_arr  = (coords[:, 1] / CELL).astype(int) % NGRID
    iz_arr  = (coords[:, 2] / CELL).astype(int) % NGRID
    for ix, iy, iz in zip(ix_arr, iy_arr, iz_arr):
        grid[ix, iy, iz] += 1.0
    return grid

def load_lae_grid(seed, z, logger):
    """
    Load LAE catalog and bin onto (NGRID,NGRID,NGRID) grid.
    ids index into the mass > 10^9.5 Msun subset of the halo array.
    """
    idpath  = os.path.join(ROOT_LAE, 'halo_ids_obs',
                           f'halo_ids_obs_z{z:.4f}_64_400_s{seed}.npy')
    cpath   = os.path.join(ROOT_HALOS, f'seed_{seed}', f'coords_z{z:.4f}.npy')
    mpath   = os.path.join(ROOT_HALOS, f'seed_{seed}', f'masses_z{z:.4f}.npy')
    if not os.path.exists(idpath):
        logger.warning(f"  LAE ids missing at z={z:.4f} seed={seed}, using empty grid")
        return np.zeros((NGRID, NGRID, NGRID), dtype=np.float32)
    ids     = np.load(idpath, mmap_mode='r')   # indices into mass-cut array
    coords  = np.load(cpath,  mmap_mode='r')
    masses  = np.load(mpath,  mmap_mode='r')
    # Build mass-cut subset
    lae_coords  = coords[ids]                  # 0-indexed into full array
    grid        = np.zeros((NGRID, NGRID, NGRID), dtype=np.float32)
    ix_arr  = (lae_coords[:, 0] / CELL).astype(int) % NGRID
    iy_arr  = (lae_coords[:, 1] / CELL).astype(int) % NGRID
    iz_arr  = (lae_coords[:, 2] / CELL).astype(int) % NGRID
    for ix, iy, iz in zip(ix_arr, iy_arr, iz_arr):
        grid[ix, iy, iz] += 1.0
    return grid

def load_lbg_grid(seed, z, logger):
    """
    Load LBG catalog (MUV < MUV_CUT) and bin onto grid.
    ids index into the mass > 10^9.5 Msun subset.
    MUV is aligned 1-to-1 with ids.
    """
    idpath  = os.path.join(ROOT_LBG, 'halo_ids_lbg',
                           f'halo_ids_lbg_z{z:.4f}_64_400_s{seed}.npy')
    muvpath = os.path.join(ROOT_LBG, 'MUV_lbg',
                           f'MUV_lbg_z{z:.4f}_64_400_s{seed}.npy')
    cpath   = os.path.join(ROOT_HALOS, f'seed_{seed}', f'coords_z{z:.4f}.npy')
    mpath   = os.path.join(ROOT_HALOS, f'seed_{seed}', f'masses_z{z:.4f}.npy')
    if not os.path.exists(idpath) or not os.path.exists(muvpath):
        logger.warning(f"  LBG files missing at z={z:.4f} seed={seed}, using empty grid")
        return np.zeros((NGRID, NGRID, NGRID), dtype=np.float32)
    ids     = np.load(idpath,  mmap_mode='r')
    muv     = np.load(muvpath, mmap_mode='r')
    coords  = np.load(cpath,   mmap_mode='r')
    masses  = np.load(mpath,   mmap_mode='r')
    # Build mass-cut subset
    # Apply MUV cut (muv aligned with ids)
    bright      = muv < MUV_CUT
    lbg_coords  = coords[ids[bright]]              # 0-indexed into full array
    grid        = np.zeros((NGRID, NGRID, NGRID), dtype=np.float32)
    ix_arr  = (lbg_coords[:, 0] / CELL).astype(int) % NGRID
    iy_arr  = (lbg_coords[:, 1] / CELL).astype(int) % NGRID
    iz_arr  = (lbg_coords[:, 2] / CELL).astype(int) % NGRID
    for ix, iy, iz in zip(ix_arr, iy_arr, iz_arr):
        grid[ix, iy, iz] += 1.0
    return grid

# ══════════════════════════════════════════════════════════════════════════════
# 6. Stitching
# ══════════════════════════════════════════════════════════════════════════════

def stitch_continuous(seed, field_name, snap_z, z_arr, logger):
    """
    Stitch a continuous field (xHI, density, vz) into a 2D lightcone.
    Interpolates across snapshots for each LoS pixel.
    Returns (NGRID, N_LC_PIX) array.
    """
    lc = np.zeros((NGRID, N_LC_PIX), dtype=np.float32)

    # Load all boxes into memory (they're only 64^3 float32 = 1MB each)
    logger.info(f"  Loading {len(snap_z)} boxes for {field_name}...")
    boxes = {}
    for z in snap_z:
        try:
            boxes[z] = load_field_box(seed, z, field_name, logger)
        except FileNotFoundError as e:
            logger.error(f"  SKIP snapshot z={z:.4f}: {e}")

    loaded_z = np.array(sorted(boxes.keys()))
    logger.info(f"  Loaded {len(loaded_z)} / {len(snap_z)} boxes")

    for n, z in enumerate(z_arr):
        y_cell = comoving_pixel(z)
        # Collect slabs from all loaded snapshots
        slabs = []
        for sz in loaded_z:
            slabs.append(get_slab(boxes[sz], y_cell))
        slabs = np.stack(slabs, axis=-1)   # (NGRID, NGRID, N_snaps)

        # Interpolate in z for each transverse cell
        interp = interp1d(loaded_z, slabs, axis=-1,
                          bounds_error=False, fill_value='extrapolate')
        lc[:, n] = interp(z)[:, X_SLICE]  # take X_SLICE row

        if n % 50 == 0:
            logger.debug(f"  LoS pixel {n}/{N_LC_PIX}  z={z:.3f}")

    return lc


def stitch_discrete(seed, field_name, snap_z, z_arr, logger):
    """
    Stitch a discrete field (halos, LAEs, LBGs) into a 2D lightcone.
    Uses nearest snapshot (no interpolation — occupation numbers are integers).
    Returns (NGRID, N_LC_PIX) array.
    """
    lc = np.zeros((NGRID, N_LC_PIX), dtype=np.float32)

    loader_map = {
        'halos': load_halo_grid,
        'lae':   load_lae_grid,
        'lbg':   load_lbg_grid,
    }
    loader = loader_map[field_name]

    # Load all grids
    logger.info(f"  Loading {len(snap_z)} grids for {field_name}...")
    grids = {}
    for z in snap_z:
        try:
            grids[z] = loader(seed, z, logger)
        except FileNotFoundError as e:
            logger.error(f"  SKIP snapshot z={z:.4f}: {e}")
        except Exception as e:
            logger.error(f"  ERROR at z={z:.4f}: {e}\n{traceback.format_exc()}")

    loaded_z = np.array(sorted(grids.keys()))
    logger.info(f"  Loaded {len(loaded_z)} / {len(snap_z)} grids")

    for n, z in enumerate(z_arr):
        y_cell  = comoving_pixel(z)
        nearest = loaded_z[find_nearest_snapshot(z, loaded_z)]
        slab    = get_slab(grids[nearest], y_cell)
        lc[:, n] = slab[:, X_SLICE]

        if n % 50 == 0:
            logger.debug(f"  LoS pixel {n}/{N_LC_PIX}  z={z:.3f}  nearest_snap={nearest:.4f}")

    return lc

# ══════════════════════════════════════════════════════════════════════════════
# 7. Per-seed pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_seed(seed, out_root):
    logger = setup_logger(seed, out_root)
    logger.info(f"{'='*60}")
    logger.info(f"SEED {seed}  started at {datetime.now().isoformat()}")
    logger.info(f"{'='*60}")

    # ── Discover snapshots ────────────────────────────────────────────────────
    try:
        snap_z      = get_snapshot_redshifts(seed, logger)
        halo_snap_z = get_halo_redshifts(seed, logger)
    except Exception as e:
        logger.error(f"Failed to discover snapshots: {e}\n{traceback.format_exc()}")
        return

    # ── LoS redshift array ────────────────────────────────────────────────────
    z_arr = np.linspace(ZMIN, ZMAX, N_LC_PIX)

    # ── Process each field ────────────────────────────────────────────────────
    errors = []
    for field_name, field_type in FIELDS:
        logger.info(f"\n── Field: {field_name} ({field_type}) ──")

        # Skip if already done
        if is_done(seed, field_name, out_root):
            logger.info(f"  SKIP: checkpoint exists for seed={seed} field={field_name}")
            continue

        out_file = output_path(seed, field_name, out_root)
        t_start  = datetime.now()

        try:
            # Choose correct snapshot z list
            use_snap_z = halo_snap_z if field_name in ('halos','lae','lbg') else snap_z

            if field_type == 'continuous':
                lc = stitch_continuous(seed, field_name, use_snap_z, z_arr, logger)
            else:
                lc = stitch_discrete(seed, field_name, use_snap_z, z_arr, logger)

            # Save
            np.savez_compressed(out_file, lc=lc,
                                z_arr=z_arr,
                                seed=seed,
                                field=field_name,
                                zmin=ZMIN, zmax=ZMAX,
                                ngrid=NGRID, box_len=BOX_LEN)
            t_elapsed = (datetime.now() - t_start).total_seconds()
            logger.info(f"  Saved: {out_file}  shape={lc.shape}  "
                        f"min={lc.min():.4f}  max={lc.max():.4f}  "
                        f"time={t_elapsed:.1f}s")

            mark_done(seed, field_name, out_root,
                      meta={'shape': list(lc.shape),
                            'min': float(lc.min()),
                            'max': float(lc.max()),
                            'elapsed_s': t_elapsed})

        except Exception as e:
            msg = f"FAILED seed={seed} field={field_name}: {e}\n{traceback.format_exc()}"
            logger.error(msg)
            errors.append(msg)
            # Write error file
            err_dir = os.path.join(out_root, 'logs')
            with open(os.path.join(err_dir, f'error_seed{seed}_{field_name}.txt'), 'w') as f:
                f.write(msg)
            # Continue to next field — don't crash the whole seed
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"SEED {seed} finished. Errors: {len(errors)}")
    for e in errors:
        logger.error(e)
    logger.info(f"{'='*60}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, nargs='+', default=ALL_SEEDS,
                        help='Seed(s) to process (default: all 1-5)')
    parser.add_argument('--out', type=str, default=OUT_ROOT,
                        help='Output root directory')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'checkpoints'), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"400 Mpc Lightcone Stitching — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Seeds: {args.seed}")
    print(f"Output: {args.out}")
    print(f"{'='*60}\n")

    for seed in args.seed:
        process_seed(seed, args.out)

    print("\nAll done.")
