#!/usr/bin/env python
# =============================================================================
# run_seed_lae.py
#
# Standalone, notebook-free driver for ONE seed of the SiMPLE-Gen LAE pipeline.
# This is a port of CELL 2b: it takes an existing cached lightcone.h5 for the
# given seed and runs SiMPLE-Gen's run_one_seed.py helper on it.
#
# Cell 2 is assumed to have already run (lightcone.h5 cached for every seed).
# If the lightcone is missing this script exits non-zero so PBS flags it.
#
# Usage:
#     python run_seed_lae.py --seed 3
#
# Designed to be launched once per seed by a PBS job array (run_lae.pbs).
# =============================================================================

import os
import sys
import time
import argparse
import subprocess

# =============================================================================
# CONFIG  --  these mirror CELL 1 / CELL 2 / CELL 2b constants.
# Edit here if your layout differs.
# =============================================================================

# Root of this project on the cluster. By default we assume this script is run
# from the project root (where kSZ2_halo_project/ lives). Override with --root.
DEFAULT_PROJECT_ROOT = os.getcwd()

# Cache dir relative to project root  (CELL 1: cache_dir = ".../cache")
# IN_CACHE  : where the already-cached input lightcone.h5 lives (read-only).
# OUT_CACHE : where THIS run writes its log / bookkeeping. Override with
#             --out-cache so a pride run does not write into swarm's cache.
IN_CACHE_SUBDIR  = "kSZ2_halo_project/cache_64"
OUT_CACHE_SUBDIR = "kSZ2_halo_project/cache_64"  # default: same as input

# SiMPLE-Gen location and helper (CELL 2b)
SIMPLEGEN_DIR = "/user1/swanith/SiMPLE-Gen"
HELPER        = os.path.join(SIMPLEGEN_DIR, "run_one_seed.py")

# Simulation parameters  (CELL 1 + CELL 2b)
Z_MIN          = 5.0
Z_MAX          = 20.0
Z_STEP_FACTOR  = 1.02
HII_DIM        = 64
BOX_LEN        = 400.0
N_THREADS      = 32
SAMPLER_MIN_MASS      = 1e8
SAMPLER_BUFFER_FACTOR = 2.0
Z_HEAT_MAX            = 20.0

# LAE env vars  (CELL 2b)
SIMPLEGEN_MH_CUT = "9.5"


def main():
    ap = argparse.ArgumentParser(
        description="Run the SiMPLE-Gen LAE pipeline for a single seed.")
    ap.add_argument("--seed", type=int, required=True,
                    help="Random seed (1-5).")
    ap.add_argument("--root", type=str, default=DEFAULT_PROJECT_ROOT,
                    help="Project root containing kSZ2_halo_project/. "
                         "Defaults to the current working directory.")
    ap.add_argument("--out-cache", type=str, default=None,
                    help="Cache subdir (relative to --root) for THIS run's "
                         "outputs/log. Defaults to the input cache. Set this "
                         "to e.g. kSZ2_halo_project/cache_27May2026_pride to "
                         "keep a pride run separate from swarm's cache.")
    args = ap.parse_args()

    seed = args.seed
    root = os.path.abspath(args.root)

    in_cache_dir  = os.path.join(root, IN_CACHE_SUBDIR)
    out_cache_sub = args.out_cache if args.out_cache else OUT_CACHE_SUBDIR
    out_cache_dir = os.path.join(root, out_cache_sub)

    in_seed_dir  = os.path.join(in_cache_dir,  f"seed_{seed}")
    out_seed_dir = os.path.join(out_cache_dir, f"seed_{seed}")
    os.makedirs(out_seed_dir, exist_ok=True)

    lc_path   = os.path.join(in_seed_dir,  "lightcone.h5")
    log_path  = os.path.join(out_seed_dir, "simplegen.log")

    print("=" * 70, flush=True)
    print(f"run_seed_lae.py  --  seed {seed}", flush=True)
    print("=" * 70, flush=True)
    print(f"  project root : {root}", flush=True)
    print(f"  in  cache    : {in_cache_dir}", flush=True)
    print(f"  out cache    : {out_cache_dir}", flush=True)
    print(f"  lightcone    : {lc_path}", flush=True)
    print(f"  helper       : {HELPER}", flush=True)
    print(f"  log          : {log_path}", flush=True)
    print(f"  BOX_LEN      : {BOX_LEN}", flush=True)
    print(f"  HII_DIM      : {HII_DIM}", flush=True)
    print(f"  log10 MH cut : {SIMPLEGEN_MH_CUT}", flush=True)

    # --- sanity checks --------------------------------------------------
    if not os.path.exists(HELPER):
        print(f"\n  FATAL: SiMPLE-Gen helper not found at {HELPER}",
              flush=True)
        sys.exit(2)

    if not os.path.exists(lc_path):
        print(f"\n  FATAL: no lightcone.h5 for seed {seed} at {lc_path}\n"
              f"  Cell 2 must run (and cache) before this LAE stage.",
              flush=True)
        sys.exit(3)

    # --- build the command  (verbatim from CELL 2b) ---------------------
    env = os.environ.copy()
    env["SIMPLEGEN_SEED"]    = str(seed)
    env["SIMPLEGEN_BOX_LEN"] = str(float(BOX_LEN))
    env["SIMPLEGEN_HII_DIM"] = str(int(HII_DIM))
    env["SIMPLEGEN_HALO_DIR"] = f"/user1/swanith/lightcone_halos_64/catalogues/seed_{seed}"
    env["SIMPLEGEN_MH_CUT"]  = SIMPLEGEN_MH_CUT

    cmd = [
        sys.executable, "-u", HELPER,
        "--lc-path",               lc_path,
        "--z-min",                 str(Z_MIN),
        "--z-max",                 str(Z_MAX),
        "--z-step-factor",         str(Z_STEP_FACTOR),
        "--hii-dim",               str(int(HII_DIM)),
        "--box-len",               str(float(BOX_LEN)),
        "--n-threads",             str(int(N_THREADS)),
        "--sampler-min-mass",      str(float(SAMPLER_MIN_MASS)),
        "--sampler-buffer-factor", str(float(SAMPLER_BUFFER_FACTOR)),
        "--z-heat-max",            str(float(Z_HEAT_MAX)),
    ]

    print(f"\n  command:\n    {' '.join(cmd)}\n", flush=True)

    # --- run ------------------------------------------------------------
    # Stream the helper's output to BOTH the per-seed simplegen.log and to
    # this process's stdout (which PBS captures), so progress is visible
    # live in the PBS .o file and also archived next to the cache.
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, cwd=SIMPLEGEN_DIR,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        proc.wait()

    dt = (time.time() - t0) / 60.0
    rc = proc.returncode

    print(f"\n{'='*70}", flush=True)
    if rc == 0:
        print(f"  seed {seed}  DONE  in {dt:.1f} min", flush=True)
    else:
        print(f"  seed {seed}  FAILED  (rc={rc})  after {dt:.1f} min",
              flush=True)
        print(f"  see log: {log_path}", flush=True)
    print(f"{'='*70}", flush=True)

    sys.exit(rc)


if __name__ == "__main__":
    main()
