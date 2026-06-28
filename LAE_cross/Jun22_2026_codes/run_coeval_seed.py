#!/usr/bin/env python3
# =============================================================================
# run_coeval_seed.py
# kSZ2 x LAE Project  —  22 Jun 2026  (updated 27 Jun 2026)
# =============================================================================
# Runs py21cmfast v4 coeval boxes for a single seed.
# Saves per-redshift-node .npy files needed by Jahaan's LAE pipeline.
#
# RESOLUTION NOTE:
#   Original: BOX_LEN=400, DIM=1024, HII_DIM=512 → crashed (PerturbHaloField
#             out-of-bounds bug triggered by DIM != HII_DIM in py21cmfast v4)
#   Updated:  BOX_LEN=200, DIM=512,  HII_DIM=512 → same physical resolution
#             (0.39 cMpc/cell), DIM=HII_DIM avoids the crash, half the volume.
#
# OUTPUT STRUCTURE:
#   ~/kSZ2_LAE_project_22Jun2026/seed_{N}/coeval_z{z:.6f}/
#       hires_density.npy        (512, 512, 512)  float32  [dimensionless overdensity]
#       hires_vz.npy             (512, 512, 512)  float32  [Mpc/s, comoving LoS velocity]
#       neutral_fraction.npy     (512, 512, 512)  float32  [xHI, 0-1]
#       kinetic_temperature.npy  (512, 512, 512)  float32  [K]
#       halo_coords.npy          (N_halos, 3)     float32  [cMpc]
#       halo_masses.npy          (N_halos,)       float32  [M_sun]
#
# USAGE:
#   python run_coeval_seed.py --seed 1
#
# PBS: submit via submit_all_seeds.sh
# =============================================================================

import os
import sys
import time
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run py21cmfast coeval boxes for one seed.")
parser.add_argument("--seed", type=int, required=True, help="Random seed (1-5)")
args = parser.parse_args()
SEED = args.seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.expanduser("~/kSZ2_LAE_project_22Jun2026")
SEED_DIR     = os.path.join(PROJECT_ROOT, f"seed_{SEED}")
CACHE_DIR    = os.path.join(SEED_DIR, "cache")

for d in [PROJECT_ROOT, SEED_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# py21cmfast setup
# ---------------------------------------------------------------------------

import py21cmfast as p21c
p21c.config["direc"] = CACHE_DIR
p21c.config["HALO_CATALOG_MEM_FACTOR"] = 3.0

print(f"\n{'='*70}")
print(f"kSZ2 x LAE  —  py21cmfast v{p21c.__version__}  —  seed {SEED}")
print(f"{'='*70}")
print(f"  Project root : {PROJECT_ROOT}")
print(f"  Seed dir     : {SEED_DIR}")
print(f"  Cache dir    : {CACHE_DIR}")

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

# Redshift nodes — log-spaced, same z_step_factor as existing pipeline
Z_MIN         = 5.0
Z_MAX         = 20.0
Z_STEP_FACTOR = 1.02

node_redshifts = p21c.get_logspaced_redshifts(
    min_redshift  = Z_MIN,
    max_redshift  = Z_MAX,
    z_step_factor = Z_STEP_FACTOR,
)
node_redshifts = np.array(node_redshifts)

print(f"\n  z range      : {Z_MIN} -> {Z_MAX}  (step factor {Z_STEP_FACTOR})")
print(f"  Nodes        : {len(node_redshifts)}  "
      f"[{node_redshifts.min():.4f}, {node_redshifts.max():.4f}]")

# Build InputParameters
inputs = p21c.InputParameters(
    node_redshifts = node_redshifts,
    random_seed    = SEED,

    simulation_options = p21c.SimulationOptions(
        BOX_LEN               = 200.0,       # cMpc  (halved from 400; same resolution)
        HII_DIM               = 512,         # ionisation/temperature grid
        DIM                   = 512,         # density/velocity/halo grid (= HII_DIM avoids crash)
        N_THREADS             = int(os.environ.get("OMP_NUM_THREADS", 64)),
        Z_HEAT_MAX            = 20.0,
        SAMPLER_MIN_MASS      = 1e8,         # M_sun
        SAMPLER_BUFFER_FACTOR = 2.0,
    ),

    matter_options = p21c.MatterOptions(
        KEEP_3D_VELOCITIES       = True,      # needed for hires_vz
        USE_INTERPOLATION_TABLES = "hmf-interpolation",
    ),

    astro_options = p21c.AstroOptions(
        INHOMO_RECO  = True,
        USE_TS_FLUCT = True,
    ),
)

print(f"\n  BOX_LEN      : {inputs.simulation_options.BOX_LEN:.1f} cMpc")
print(f"  HII_DIM      : {inputs.simulation_options.HII_DIM}  "
      f"({inputs.simulation_options.BOX_LEN/inputs.simulation_options.HII_DIM:.3f} cMpc/cell)")
print(f"  DIM          : 512  "
      f"({inputs.simulation_options.BOX_LEN/512:.3f} cMpc/cell  — same as 400Mpc/1024)")
print(f"  N_THREADS    : {inputs.simulation_options.N_THREADS}")
print(f"  KEEP_3D_VEL  : {inputs.matter_options.KEEP_3D_VELOCITIES}")
print(f"  INHOMO_RECO  : {inputs.astro_options.INHOMO_RECO}")
print(f"  USE_TS_FLUCT : {inputs.astro_options.USE_TS_FLUCT}")
print(f"  SAMPLE_METHOD: {inputs.matter_options.SAMPLE_METHOD}")

# ---------------------------------------------------------------------------
# Initial conditions  (computed once, reused for all redshifts)
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print(f"  Computing initial conditions ...")
t0_ic = time.time()

cache    = p21c.OutputCache(CACHE_DIR)
init_box = p21c.compute_initial_conditions(inputs=inputs, cache=cache, write=True)

print(f"  ✓ Initial conditions done  ({(time.time()-t0_ic)/60:.1f} min)")

# ---------------------------------------------------------------------------
# Helper: save one coeval node
# ---------------------------------------------------------------------------

def save_coeval(coeval, halo_catalog, out_dir):
    """
    Extract and save all fields needed by Jahaan's pipeline.
    Purges large arrays immediately after saving to keep RAM flat.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- hires density (DIM grid) ----------------------------------------
    hires_den = coeval.get("hires_density").astype(np.float32)
    np.save(os.path.join(out_dir, "hires_density.npy"), hires_den)
    del hires_den

    # ---- hires LoS velocity (DIM grid, z-axis) ----------------------------
    # py21cmfast stores the z-component as hires_vz when KEEP_3D_VELOCITIES=True
    # Units: internal code units (Mpc/s comoving) — same convention as your
    # existing pipeline.  Conversion note kept in README.
    hires_vz = coeval.get("hires_vz").astype(np.float32)
    np.save(os.path.join(out_dir, "hires_vz.npy"), hires_vz)
    del hires_vz

    # ---- neutral fraction (HII_DIM grid) -----------------------------------
    xHI = coeval.get("neutral_fraction").astype(np.float32)
    np.save(os.path.join(out_dir, "neutral_fraction.npy"), xHI)
    del xHI

    # ---- kinetic temperature (HII_DIM grid) --------------------------------
    Tk = coeval.get("kinetic_temperature").astype(np.float32)
    np.save(os.path.join(out_dir, "kinetic_temperature.npy"), Tk)
    del Tk

    # ---- halo catalogue ----------------------------------------------------
    if halo_catalog is not None:
        coords = halo_catalog.get("halo_coords").astype(np.float32)  # (N,3) cMpc
        masses = halo_catalog.get("halo_masses").astype(np.float32)  # (N,)  M_sun
        np.save(os.path.join(out_dir, "halo_coords.npy"), coords)
        np.save(os.path.join(out_dir, "halo_masses.npy"), masses)
        n_halos = len(masses)
        del coords, masses
    else:
        n_halos = 0

    return n_halos


# ---------------------------------------------------------------------------
# Main coeval loop
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print(f"  Running {len(node_redshifts)} coeval nodes  (seed {SEED})")
print(f"{'='*70}\n")

t0_total      = time.time()
prev_halocat  = None   # for the Meraxes-style backward stepping in determine_halo_catalog

# generate_coeval steps from high-z to low-z internally
for coeval, in_outputs in p21c.generate_coeval(
    inputs = inputs,
    cache  = cache,
):
    z   = coeval.redshift
    out_dir = os.path.join(SEED_DIR, f"coeval_z{z:.6f}")

    t_node = time.time()

    # ---- halo catalog for this redshift ------------------------------------
    # determine_halo_catalog goes high-z to low-z, using the previous
    # (higher-z) catalog as the descendant condition.
    # On the first call prev_halocat is None → samples from density grid.
    try:
        halo_catalog = p21c.determine_halo_catalog(
            redshift            = z,
            initial_conditions  = init_box,
            descendant_halos    = prev_halocat,
            inputs              = inputs,
        )
        prev_halocat = halo_catalog
    except Exception as e:
        print(f"  ✗ Halo catalog failed at z={z:.4f}: {e}")
        halo_catalog = None

    # ---- save all fields ---------------------------------------------------
    n_halos = save_coeval(coeval, halo_catalog, out_dir)

    elapsed = (time.time() - t0_total) / 60
    node_t  = time.time() - t_node
    print(f"  z={z:7.4f}  halos={n_halos:>10,}  "
          f"node_time={node_t:.1f}s  elapsed={elapsed:.1f}min  -> {out_dir}")
    sys.stdout.flush()

    # ---- purge coeval from memory (1024^3 fields are ~4 GB each) ----------
    coeval.purge()

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

total_min = (time.time() - t0_total) / 60
print(f"\n{'='*70}")
print(f"  ✓  SEED {SEED} COMPLETE  —  {total_min:.1f} min total")
print(f"  Output: {SEED_DIR}")
print(f"{'='*70}\n")
