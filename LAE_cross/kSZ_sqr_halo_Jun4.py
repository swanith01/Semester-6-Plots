# %%
# %%
# %%
# =============================================================================
# CELL 1: Imports, Setup, and Configuration
# kSZ²–Halo Cross-Correlation Pipeline  (py21cmFAST v4.1.0)
# =============================================================================

import os
import glob
import time
import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

# =============================================================================
# CACHE
# =============================================================================

cache_dir     = "kSZ2_halo_project/cache"
cache_dir_abs = os.path.abspath(cache_dir)
os.makedirs(cache_dir, exist_ok=True)

import py21cmfast as p21c
p21c.config['direc'] = cache_dir_abs

print(f"✓ py21cmfast version : {p21c.__version__}")
print(f"✓ Cache              : {cache_dir_abs}")

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

project_dir = "kSZ2_halo_project"
plot_dir    = f"{project_dir}/plots"
halo_dir    = "lightcone_halos/catalogues"

for d in [project_dir, plot_dir, cache_dir]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# MULTI-SEED SETUP
# =============================================================================

RANDOM_SEEDS = list(range(1, 6))   # seeds 1, 2, 3, 4, 5
N_SEEDS      = len(RANDOM_SEEDS)

# =============================================================================
# SIMULATION PARAMETERS  (v4.1.0 verified)
#
# SimulationOptions : box geometry, threads, sampler mass/buffer
# MatterOptions     : KEEP_3D_VELOCITIES, interpolation tables
#                     SAMPLE_METHOD='MASS-LIMITED' by default → halo sampler on
# AstroOptions      : spin temperature, inhomogeneous recombinations
#
# NOTE: random_seed here is a TEMPLATE value only. Per-seed InputParameters
#       objects are constructed inside CELL 2's worker, one per entry in
#       RANDOM_SEEDS, inheriting all the seed-independent options below.
# =============================================================================

z_min         = 5.0
z_max         = 20.0
z_step_factor = 1.02

node_redshifts_custom = np.array(
    p21c.get_logspaced_redshifts(
        min_redshift  = z_min,
        max_redshift  = z_max,
        z_step_factor = z_step_factor,
    )
)

inputs = p21c.InputParameters(
    node_redshifts     = node_redshifts_custom,
    random_seed        = RANDOM_SEEDS[0],   # template; per-seed inputs built in CELL 2

    simulation_options = p21c.SimulationOptions(
        HII_DIM               = 32,
        BOX_LEN               = 400.0,
        N_THREADS             = 32,
        Z_HEAT_MAX            = 20.0,
        SAMPLER_MIN_MASS      = 1e8,
        SAMPLER_BUFFER_FACTOR = 2.0,
    ),

    matter_options = p21c.MatterOptions(
        KEEP_3D_VELOCITIES       = True,
        USE_INTERPOLATION_TABLES = 'hmf-interpolation',
    ),

    astro_options = p21c.AstroOptions(
        INHOMO_RECO  = True,
        USE_TS_FLUCT = True,
    ),
)

# =============================================================================
# COSMOLOGY
# =============================================================================

cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086)

# =============================================================================
# PLOT SETTINGS
# =============================================================================

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset' : 'cm',
    'font.size'        : 28,
    'axes.labelsize'   : 24,
    'axes.titlesize'   : 24,
    'xtick.labelsize'  : 24,
    'ytick.labelsize'  : 24,
    'legend.fontsize'  : 16,
    'xtick.direction'  : 'in',
    'ytick.direction'  : 'in',
    'xtick.top'        : True,
    'ytick.right'      : True,
    'xtick.major.size' : 6,
    'ytick.major.size' : 6,
    'xtick.minor.size' : 3,
    'ytick.minor.size' : 3,
    'axes.linewidth'   : 1.0,
    'lines.linewidth'  : 1.8,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
})
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

# =============================================================================
# SUMMARY
# =============================================================================

halo_mass_files = sorted(glob.glob(f"{halo_dir}/masses_z*.npy"))

print(f"\n{'='*70}")
print(f"kSZ2-HALO PIPELINE READY  (py21cmfast v{p21c.__version__})")
print(f"{'='*70}")
print(f"  BOX_LEN          : {inputs.simulation_options.BOX_LEN:.1f} Mpc")
print(f"  HII_DIM          : {inputs.simulation_options.HII_DIM}")
print(f"  cell size        : {inputs.simulation_options.cell_size:.3f} Mpc")
print(f"  z range          : {z_min} -> {z_max}  ({len(inputs.node_redshifts)} nodes)")
print(f"  KEEP_3D_VEL      : {inputs.matter_options.KEEP_3D_VELOCITIES}")
print(f"  SAMPLE_METHOD    : {inputs.matter_options.SAMPLE_METHOD}")
#print(f"  has_discrete_halos: {inputs.matter_options.has_discrete_halos}")
print(f"  USE_TS_FLUCT     : {inputs.astro_options.USE_TS_FLUCT}")
print(f"  INHOMO_RECO      : {inputs.astro_options.INHOMO_RECO}")
print(f"  N seeds          : {N_SEEDS}  ({RANDOM_SEEDS})")
print(f"  halo catalogues  : {len(halo_mass_files)} snapshots in {halo_dir}/")
print(f"  plots            : {os.path.abspath(plot_dir)}")
print(f"{'='*70}")

# %%
# os.remove(f"{cache_dir}/lightcone.h5")          # line 472
# os.remove(f"{cache_dir}/field_arrays.npz")      # line 480
# os.remove("kSZ2_halo_project/cache/halo_arrays.npz")   # line 506

# %%
# import os
# import glob

# # Define your base search directory
# search_base = "kSZ2_halo_project"

# # Patterns to look for (including common variations)
# patterns = [
#     "**/lightcone.h5",
#     "**/field_arrays.npz",
#     "**/halo_arrays.npz",
#     "**/*lightcone*",
#     "**/*halo*",
#     "**/*field*"
# ]

# print(f"{'='*70}")
# print(f"SEARCHING FOR FILES IN: {os.path.abspath(search_base)}")
# print(f"{'='*70}")

# found_any = False
# for pattern in patterns:
#     # recursive=True allows searching through subfolders like seed_1, seed_2, etc.
#     matches = glob.glob(os.path.join(search_base, pattern), recursive=True)
    
#     if matches:
#         found_any = True
        
#         print(f"\nPattern match: [{pattern}]")
#         for match in sorted(matches):
#             file_size = os.path.getsize(match) / (1024**2) # Convert to MB
#             print(f"  → {match:60} ({file_size:.2f} MB)")

# if not found_any:
#     print("\n[!] No files matching those signatures were found.")
#     print(f"    Check if '{search_base}' is the correct relative path.")

# print(f"\n{'='*70}")

# %%
# import os
# import glob

# # Your project root
# base_dir = "/user1/swanith/kSZ2_halo_project"

# # Specific file names to target
# target_files = [
#     "lightcone.h5",
#     "field_arrays.npz",
#     "halo_arrays.npz"
# ]

# print(f"{'='*70}")
# print(f"CLEANING DATA FILES IN: {base_dir}")
# print(f"{'='*70}")

# count = 0
# # We search recursively inside 'cache' to avoid touching 'plots'
# # but specifically look for the file names in our target list
# for root, dirs, files in os.walk(os.path.join(base_dir, "cache")):
#     for filename in files:
#         if filename in target_files:
#             file_path = os.path.join(root, filename)
#             try:
#                 # Get size for logging before deleting
#                 size = os.path.getsize(file_path) / (1024**2)
#                 os.remove(file_path)
#                 print(f"  [DELETED] {filename:<18} | {file_path} ({size:.2f} MB)")
#                 count += 1
#             except Exception as e:
#                 print(f"  [ERROR]   Could not delete {file_path}: {e}")

# print(f"{'='*70}")
# print(f"DONE: Removed {count} files.")
# print(f"{'='*70}")

# %%
# %%
# %%
# =============================================================================
# CELL 2 (FINAL): Lightcone + Field + Halo Array Construction — ALL SEEDS
# Pro caching (HDF5 + npz per seed) + concurrent seeds (ProcessPool/spawn)
# Following 21cmFAST's make_lightcone_slices recipe EXACTLY
# =============================================================================

import os
import glob
import time
import h5py
import numpy as np
from astropy.units import pixel
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# IMPORT THE WORKER FROM THE SEPARATE MODULE
from halo_lightcone_worker import run_or_load_seed_halo

print("\n" + "="*70)
print("CELL 2 — LIGHTCONE + FIELD + HALO ARRAYS (ALL SEEDS, PARALLELISED)")
print("="*70)

# =============================================================================
# PATHS & SHARED SETUP
# =============================================================================

HALO_OUT      = halo_dir
BOX_LEN       = float(inputs.simulation_options.BOX_LEN)
HII_DIM       = int(inputs.simulation_options.HII_DIM)
cell_size_mpc = BOX_LEN / HII_DIM
MASS_CUT      = 10.0**8.5   # M_sun

os.makedirs(HALO_OUT, exist_ok=True)

# =============================================================================
# CORE / WORKER ALLOCATION
# =============================================================================

N_TOTAL_CORES              = int(os.environ.get('PBS_NCPUS',
                                                os.cpu_count() or 32))
DESIRED_THREADS_PER_WORKER = 32        # Change this if you want more/less 12 for nproc=64; 8 for 32 cores
N_WORKERS                  = max(1, N_TOTAL_CORES // DESIRED_THREADS_PER_WORKER)
N_WORKERS                  = min(N_WORKERS, N_SEEDS)

print(f"\n  Available cores    : {N_TOTAL_CORES}")
print(f"  Workers            : {N_WORKERS} (concurrent seeds)")
print(f"  Threads per worker : {DESIRED_THREADS_PER_WORKER}")
print(f"  Total threads used : {N_WORKERS * DESIRED_THREADS_PER_WORKER}"
      f"/{N_TOTAL_CORES}")
print(f"  Seeds              : {RANDOM_SEEDS}")
print(f"  Mass cut           : 10^{np.log10(MASS_CUT):.1f} M_sun")
print(f"  Cell size          : {cell_size_mpc:.2f} cMpc")

# =============================================================================
# DISPATCH — concurrent seeds via ProcessPoolExecutor with spawn context
# (spawn is critical: py21cmfast's CFFI / C globals don't survive fork)
# =============================================================================

seed_metadata = {}   # {seed: {'seed_cache_dir', 'halo_out_seed_dir',
                     #         'sim_time', 'status', 'n_lc'}}
scan_start = time.time()
mp_ctx     = mp.get_context("spawn")

print(f"\n{'='*70}")
print(f"DISPATCHING {N_SEEDS} SEEDS — {N_WORKERS} concurrent workers")
print(f"{'='*70}", flush=True)

with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as ex:
    futures = {}
    for seed in RANDOM_SEEDS:
        seed_cache_dir    = os.path.join(cache_dir, f"seed_{seed}")
        halo_out_seed_dir = os.path.join(HALO_OUT, f"seed_{seed}")

        args = (
            seed, seed_cache_dir, halo_out_seed_dir,
            z_min, z_max, z_step_factor,
            HII_DIM, BOX_LEN,
            DESIRED_THREADS_PER_WORKER,
            float(inputs.simulation_options.Z_HEAT_MAX),
            float(inputs.simulation_options.SAMPLER_MIN_MASS),
            float(inputs.simulation_options.SAMPLER_BUFFER_FACTOR),
            MASS_CUT,
        )
        fut = ex.submit(run_or_load_seed_halo, args)
        futures[fut] = seed

    completed_count = 0
    for fut in as_completed(futures):
        seed = futures[fut]
        try:
            (seed_done, seed_cache_dir, halo_out_seed_dir,
             n_lc_seed, sim_time, status) = fut.result()
        except Exception as e:
            seed_done         = seed
            seed_cache_dir    = None
            halo_out_seed_dir = None
            n_lc_seed         = 0
            sim_time          = 0.0
            status            = f"crashed: {e}"

        completed_count += 1
        seed_metadata[seed_done] = {
            'seed_cache_dir'   : seed_cache_dir,
            'halo_out_seed_dir': halo_out_seed_dir,
            'sim_time'         : sim_time,
            'status'           : status,
            'n_lc'             : n_lc_seed,
        }

        if 'crashed' in status:
            msg = f"✗ {status}"
        elif sim_time > 0:
            msg = f"✓ {status}  (lightcone sim: {sim_time/60:.2f} min)"
        else:
            msg = f"✓ {status}"

        elapsed = (time.time() - scan_start) / 60
        print(f"  [{completed_count:2d}/{N_SEEDS}] seed {seed_done:3d}: "
              f"{msg}   (elapsed: {elapsed:.1f} min)", flush=True)

print(f"\n{'='*70}")
print(f"✓ ALL WORKERS RETURNED  —  {time.time()-scan_start:.1f} s total")
print(f"{'='*70}")

# =============================================================================
# Load every cached lightcone + field + halo array into the main process
# =============================================================================

print(f"\n=== LOADING CACHED ARRAYS INTO MAIN PROCESS ===", flush=True)

lightcones          = {}   # {seed: LightCone}
density_lc_all      = {}
neutral_frac_lc_all = {}
los_velocity_lc_all = {}
brightness_lc_all   = {}
kinetic_temp_lc_all = {}
halo_mass_lc_all    = {}
halo_count_lc_all   = {}

for seed in RANDOM_SEEDS:
    meta           = seed_metadata.get(seed, {})
    seed_cache_dir = meta.get('seed_cache_dir')

    if not seed_cache_dir or not os.path.exists(seed_cache_dir):
        print(f"  ✗ seed {seed}: no cache dir — skipping")
        continue

    lightcone_cache = os.path.join(seed_cache_dir, "lightcone.h5")
    fields_cache    = os.path.join(seed_cache_dir, "field_arrays.npz")
    halos_cache     = os.path.join(seed_cache_dir, "halo_arrays.npz")

    try:
        lc = p21c.LightCone.from_file(lightcone_cache, safe=False)
        lightcones[seed] = lc

        d = np.load(fields_cache)
        density_lc_all[seed]      = d['density_lc']
        neutral_frac_lc_all[seed] = d['neutral_frac_lc']
        los_velocity_lc_all[seed] = d['los_velocity_lc']
        brightness_lc_all[seed]   = d['brightness_lc']
        kinetic_temp_lc_all[seed] = d['kinetic_temp_lc']

        d = np.load(halos_cache)
        halo_mass_lc_all [seed] = d['halo_mass_lc']
        halo_count_lc_all[seed] = d['halo_count_lc']

        n_filled = int(np.isfinite(halo_mass_lc_all[seed]).sum())
        print(f"  ✓ seed {seed:3d}  fields={density_lc_all[seed].shape}  "
              f"halos filled={n_filled:,} pixels")

    except Exception as e:
        print(f"  ✗ seed {seed}: load failed — {e}")

# =============================================================================
# Seed-independent geometry arrays (take from first loaded seed)
# =============================================================================

if len(lightcones) == 0:
    raise RuntimeError("No seeds loaded successfully — cannot proceed.")

first_seed   = next(iter(lightcones))
first_lc     = lightcones[first_seed]
z_lc         = np.array(first_lc.lightcone_redshifts, dtype=np.float32)
n_lc         = len(z_lc)
lc_distances = np.array(first_lc.lightcone_distances.to_value('Mpc'),
                        dtype=np.float32)

c_Mpc_s     = 299792.458 / 3.08567758e19
vel_rms_kms = los_velocity_lc_all[first_seed].std() / c_Mpc_s * 299792.458

print(f"\n  z_lc        : {n_lc} slices  [{z_lc.min():.2f}, {z_lc.max():.2f}]")
print(f"  lc_distances: [{lc_distances.min():.1f}, {lc_distances.max():.1f}] Mpc")
print(f"  v_los rms (seed {first_seed}): "
      f"{los_velocity_lc_all[first_seed].std():.3e} Mpc/s  "
      f"(~{vel_rms_kms:.0f} km/s)  ✓")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print(f"CELL 2 COMPLETE — {len(lightcones)}/{N_SEEDS} SEEDS LOADED")
print("="*70)
print(f"  z_lc                : {n_lc} slices  "
      f"[{z_lc.min():.2f}, {z_lc.max():.2f}]")
print(f"  density_lc_all      : {len(density_lc_all)} seeds × "
      f"{density_lc_all[first_seed].shape}")
print(f"  neutral_frac_lc_all : {len(neutral_frac_lc_all)} seeds × "
      f"{neutral_frac_lc_all[first_seed].shape}")
print(f"  los_velocity_lc_all : {len(los_velocity_lc_all)} seeds × "
      f"{los_velocity_lc_all[first_seed].shape}  [Mpc/s]  ✓")
print(f"  brightness_lc_all   : {len(brightness_lc_all)} seeds × "
      f"{brightness_lc_all[first_seed].shape}")
print(f"  kinetic_temp_lc_all : {len(kinetic_temp_lc_all)} seeds × "
      f"{kinetic_temp_lc_all[first_seed].shape}")
print(f"  halo_mass_lc_all    : {len(halo_mass_lc_all)} seeds × "
      f"{halo_mass_lc_all[first_seed].shape}")
print(f"  halo_count_lc_all   : {len(halo_count_lc_all)} seeds × "
      f"{halo_count_lc_all[first_seed].shape}")
print("="*70)

# %%
# %%
# =============================================================================
# CELL 2b: SiMPLE-Gen LAE pipeline — all 5 seeds via per-seed subprocesses
# =============================================================================

import os
import subprocess
import time
import numpy as np

SIMPLEGEN_DIR = "/user1/swanith/SiMPLE-Gen"   # location of run_one_seed.py
HELPER        = os.path.join(SIMPLEGEN_DIR, "run_one_seed.py")

# uses cache_dir, RANDOM_SEEDS, inputs, z_min, z_max, z_step_factor from cell 1
# uses MASS_CUT from cell 2

print("=" * 70)
print(f"CELL 2b — LAE pipeline for {N_SEEDS} seeds")
print("=" * 70)
print(f"  helper    : {HELPER}")
print(f"  BOX_LEN   : {inputs.simulation_options.BOX_LEN}")
print(f"  HII_DIM   : {inputs.simulation_options.HII_DIM}")
print(f"  log10 MH  : {np.log10(MASS_CUT):.2f}")
print(f"  seeds     : [1]")

t_all = time.time()

for seed in [1]:
    lc_path = os.path.join(cache_dir, f"seed_{seed}", "lightcone.h5")
    log_path = os.path.join(cache_dir, f"seed_{seed}", "simplegen.log")

    if not os.path.exists(lc_path):
        print(f"  ✗ seed {seed}: no lightcone.h5 at {lc_path} — skipping")
        continue

    print(f"\n── seed {seed} ──  lc: {lc_path}")
    t0 = time.time()

    env = os.environ.copy()
    env["SIMPLEGEN_SEED"]    = str(seed)
    env["SIMPLEGEN_BOX_LEN"] = str(float(inputs.simulation_options.BOX_LEN))
    env["SIMPLEGEN_HII_DIM"] = str(int(inputs.simulation_options.HII_DIM))
    env["SIMPLEGEN_MH_CUT"]  = "9.5"

    cmd = [
        "python", "-u",HELPER,
        "--lc-path",               lc_path,
        "--z-min",                 str(z_min),
        "--z-max",                 str(z_max),
        "--z-step-factor",         str(z_step_factor),
        "--hii-dim",               str(int(inputs.simulation_options.HII_DIM)),
        "--box-len",               str(float(inputs.simulation_options.BOX_LEN)),
        "--n-threads",             str(int(inputs.simulation_options.N_THREADS)),
        "--sampler-min-mass",      str(float(inputs.simulation_options.SAMPLER_MIN_MASS)),
        "--sampler-buffer-factor", str(float(inputs.simulation_options.SAMPLER_BUFFER_FACTOR)),
        "--z-heat-max",            str(float(inputs.simulation_options.Z_HEAT_MAX)),
    ]

    with open(log_path, "w") as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

    dt = (time.time() - t0) / 60
    status = "✓" if result.returncode == 0 else f"✗ (rc={result.returncode})"
    print(f"  {status} seed {seed} finished in {dt:.1f} min  log: {log_path}")

    if result.returncode != 0:
        print(f"     last 20 lines of log:")
        with open(log_path) as f:
            tail = f.readlines()[-20:]
        print("".join(f"       {ln}" for ln in tail))

print(f"\n{'='*70}")
print(f"✓ Cell 2b complete in {(time.time()-t_all)/60:.1f} min total")
print(f"{'='*70}")
print(f"  Outputs at: {SIMPLEGEN_DIR}/SiMPLEGen/data/seed_<N>/lightcone_lae/")

# %%
# import os
# import time

# # Check halo arrays cache (npz)
# halos_cache = "kSZ2_halo_project/cache/halo_arrays.npz"
# print(f"halo_arrays.npz: {time.ctime(os.path.getmtime(halos_cache))}")

# # Check raw catalogues
# halo_dir = "lightcone_halos/catalogues"
# files = sorted(os.listdir(halo_dir))
# masses_files = [f for f in files if f.startswith('masses')]

# if masses_files:
#     first_file = os.path.join(halo_dir, masses_files[0])
#     last_file = os.path.join(halo_dir, masses_files[-1])
#     print(f"First catalogue: {masses_files[0]} - {time.ctime(os.path.getmtime(first_file))}")
#     print(f"Last catalogue:  {masses_files[-1]} - {time.ctime(os.path.getmtime(last_file))}")
#     print(f"Total catalogues: {len(masses_files)}")

# %%
# result = p21c.run_lightcone(
#     inputs     = inputs,
#     lightconer = lightconer,
#     write      = True,
# )

# print(f"Type: {type(result)}")
# print(f"Length (if tuple): {len(result) if isinstance(result, tuple) else 'N/A'}")
# if isinstance(result, tuple):
#     for i, item in enumerate(result):
#         print(f"  [{i}] {type(item)}")
# else:
#     print(f"Result: {result}")
#     print(f"Has 'save' method: {hasattr(result, 'save')}")

# %%
# lc_file = "/user1/swanith/.conda/envs/p21c_v4/lib/python3.11/site-packages/py21cmfast/lightcones.py"

# with open(lc_file) as f:
#     lines = f.readlines()

# print("="*70)
# print("FINDING THE METHOD CONTAINING LINE 341")
# print("="*70)

# # Search backwards from line 341 to find the method definition
# for i in range(340, max(0, 320), -1):
#     if 'def ' in lines[i]:
#         # Found the method, print from here
#         for j in range(i, min(i + 30, len(lines))):
#             print(f"{j:4d}: {lines[j]}", end='')
#         break

# %%
# import os
# cache_dir = "kSZ2_halo_project/cache"
# lightcone_cache = f"{cache_dir}/lightcone.h5"

# if os.path.exists(lightcone_cache):
#     os.remove(lightcone_cache)
#     print(f"Deleted: {lightcone_cache}")
# else:
#     print(f"File not found: {lightcone_cache}")

# # Also delete the field arrays cache to be safe
# fields_cache = f"{cache_dir}/field_arrays.npz"
# if os.path.exists(fields_cache):
#     os.remove(fields_cache)
#     print(f"Deleted: {fields_cache}")

# %%
# import os
# halos_cache = "kSZ2_halo_project/cache/halo_arrays.npz"
# if os.path.exists(halos_cache):
#     os.remove(halos_cache)
#     print(f"Deleted: {halos_cache}")

# %%
# import inspect, py21cmfast as p21c

# # all public methods on the lightconer
# print([m for m in dir(p21c.RectilinearLightconer) if not m.startswith('_')])

# # source of the pixel-distance method — this is the key stitching logic
# print(inspect.getsource(p21c.RectilinearLightconer.get_lc_distances_in_pixels))
# import inspect
# print(inspect.getsource(p21c.RectilinearLightconer.make_lightcone_slices))

# import inspect
# print(inspect.getsource(p21c.RectilinearLightconer.coeval_subselect))

# %%
# import os
# os.remove("kSZ2_halo_project/cache/halo_arrays.npz")

# %%
# %%
# =============================================================================
# CELL 3: DIAGNOSTIC PLOTS — using raw halo catalogues
# Single-seed diagnostic; cosmic-variance plots come later (seed-averaged cells)
#
# PATCHED for the slice-centric (Option A) worker:
#   - CELL 2 now saves the FULL-BOX halo catalogue per node
#     (masses_z*.npy / coords_z*.npy), NOT the per-slab subset.
#   - CELL 3 therefore rebuilds the `lightconer` and re-derives the exact
#     one-cell slab cut the worker used, via slab_for_lc_idx().
#   - HMF volume fixed to BOX_LEN**2 * cell_size_mpc (one coeval cell thick).
#   - Plot 2 subsamples to keep matplotlib fast.
# =============================================================================

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.units import pixel

# =============================================================================
# DIAGNOSTIC SEED — pick one seed from RANDOM_SEEDS for these plots
# =============================================================================

DIAG_SEED = RANDOM_SEEDS[0]   # change to any seed in RANDOM_SEEDS

print("\n" + "="*70)
print(f"CELL 3 — DIAGNOSTIC PLOTS  (seed {DIAG_SEED})")
print("="*70)

# Bind single-seed aliases to the per-seed dicts built in CELL 2
halo_mass_lc    = halo_mass_lc_all   [DIAG_SEED]
halo_count_lc   = halo_count_lc_all  [DIAG_SEED]
density_lc      = density_lc_all     [DIAG_SEED]
neutral_frac_lc = neutral_frac_lc_all[DIAG_SEED]
los_velocity_lc = los_velocity_lc_all[DIAG_SEED]
kinetic_temp_lc = kinetic_temp_lc_all[DIAG_SEED]
lightcone       = lightcones         [DIAG_SEED]

PLOT_DIR = plot_dir
HALO_OUT = os.path.join(halo_dir, f"seed_{DIAG_SEED}")   # per-seed catalogues
os.makedirs(PLOT_DIR, exist_ok=True)

# =============================================================================
# REBUILD LIGHTCONER + SLAB HELPER  (replicates CELL 2's exact slab mapping)
#
# The worker saves the full-box catalogue, not the slab. To make CELL 3's
# plots show exactly the halos CELL 2 binned into halo_arrays.npz, we rebuild
# the `lightconer` (cheap — only needs `inputs`) and re-apply the identical
# one-cell slab filter.  `lightcone.h5` stores the lightcone DATA but not the
# `lightconer` geometry object, so it must be reconstructed here.
# =============================================================================

cell_size_mpc = BOX_LEN / HII_DIM

lightconer = p21c.RectilinearLightconer.between_redshifts(
    min_redshift = min(inputs.node_redshifts) + 0.1,
    max_redshift = max(inputs.node_redshifts) - 0.1,
    quantities   = ("brightness_temp", "density", "neutral_fraction",
                    "kinetic_temperature", "velocity_z"),
    resolution   = inputs.simulation_options.cell_size,
)
lcpix = lightconer.get_lc_distances_in_pixels(
    inputs.simulation_options.cell_size)


def slab_for_lc_idx(z_idx, m_box, c_box):
    """Reproduce CELL 2's one-cell slab filter for a given LC index.

    Parameters
    ----------
    z_idx : int
        Lightcone slice index.
    m_box, c_box : ndarray
        Full-box halo masses / coords as saved by the worker
        (masses_z*.npy / coords_z*.npy).

    Returns
    -------
    (m_slab, c_slab) : the halos CELL 2 binned at this z_idx.
    """
    lcidx  = int((lcpix.max() - lcpix[z_idx] + 1*pixel).to_value(pixel))
    z_cell = (-lcidx + lightconer.index_offset) % HII_DIM
    z_lo   = z_cell * cell_size_mpc
    z_hi   = z_lo + cell_size_mpc
    slab   = (c_box[:, 2] >= z_lo) & (c_box[:, 2] < z_hi)
    return m_box[slab], c_box[slab]


# Find slices with halos
halo_exists = np.array([np.any(np.isfinite(halo_mass_lc[:,:,i]))
                        for i in range(n_lc)])
idx_with_halos = np.where(halo_exists)[0]

# Pick ~6 slices evenly spaced
if len(idx_with_halos) > 6:
    idx_pick = idx_with_halos[np.linspace(0, len(idx_with_halos)-1, 6,
                                          dtype=int)]
else:
    idx_pick = idx_with_halos

z_pick = z_lc[idx_pick]
print(f"\nFound {len(idx_with_halos)} slices with halos, "
      f"plotting {len(idx_pick)}")

# Get matching node redshifts for each LC index
node_redshifts_sorted = sorted(inputs.node_redshifts)
lc_distances = np.array(lightcone.lightcone_distances.to_value('Mpc'),
                        dtype=np.float32)


def find_node_for_lc_idx(z_idx):
    """Find which node redshift corresponds to this LC index."""
    target_dc = lc_distances[z_idx]
    closest_z_node = min(
        node_redshifts_sorted,
        key=lambda z: abs(cosmo.comoving_distance(z).to_value('Mpc')
                          - target_dc))
    return closest_z_node


# Plot 2 subsampling cap (matplotlib scatter slows badly past ~1e5 points)
PLOT2_MAX_SCATTER = 50_000

# =============================================================================
# PLOT 1: HEAVIEST 2000 HALOS OVER DENSITY (slab-filtered raw positions)
# =============================================================================

print("\n[1/3] Heaviest 2000 halos over density...")
print("-"*50)

with mpl.rc_context(plt.rcParams):
    ncols = 3
    nrows = int(np.ceil(len(idx_pick) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5.5*nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, z_idx in zip(axes, idx_pick):
        z_node = z_lc[z_idx]

        dens_slice = density_lc[:, :, z_idx].T
        im = ax.imshow(1 + dens_slice, origin='lower',
                       extent=[0, BOX_LEN, 0, BOX_LEN],
                       cmap='Greys',
                       norm=mcolors.LogNorm(vmin=0.1, vmax=10),
                       interpolation='bilinear')
        plt.colorbar(im, ax=ax, label=r'$1+\delta$',
                     fraction=0.046, pad=0.15)

        # Load FULL-BOX catalogue, then re-derive the slab
        z_node_match = find_node_for_lc_idx(z_idx)
        tag = f"z{z_node_match:.4f}"
        masses_path = os.path.join(HALO_OUT, f"masses_{tag}.npy")
        coords_path = os.path.join(HALO_OUT, f"coords_{tag}.npy")

        n_halos = 0
        if os.path.exists(masses_path):
            m_box = np.load(masses_path)
            c_box = np.load(coords_path)
            m_slab, c_slab = slab_for_lc_idx(z_idx, m_box, c_box)
            n_halos = len(m_slab)

            if n_halos > 0:
                n_top   = min(2000, n_halos)
                top_idx = np.argsort(m_slab)[-n_top:]
                sc = ax.scatter(c_slab[top_idx, 0], c_slab[top_idx, 1],
                                c=np.log10(m_slab[top_idx]),
                                s=10, cmap='plasma', alpha=0.8,
                                vmin=8.5, vmax=10.5, edgecolors='none')
                plt.colorbar(sc, ax=ax, label=r'$\log_{10}(M_\odot)$',
                             fraction=0.046, pad=0.04)

        ax.set_xlim(0, BOX_LEN)
        ax.set_ylim(0, BOX_LEN)
        ax.set_xlabel("x  [cMpc]")
        ax.set_ylabel("y  [cMpc]")
        ax.set_title(f"$z = {z_node:.2f}$  ({n_halos:,} halos)", fontsize=12)

    for ax in axes[len(idx_pick):]:
        ax.set_visible(False)

    fig.suptitle(f"Heaviest 2000 halos per slice + Density field  "
                 f"(seed {DIAG_SEED})", fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(PLOT_DIR,
                "02_heaviest_2000_halos_over_density.png"),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR,
                "02_heaviest_2000_halos_over_density.pdf"),
                bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 02_heaviest_2000_halos_over_density.png / .pdf")

# =============================================================================
# PLOT 2: ALL HALOS IN SLAB OVER DENSITY (subsampled for plotting speed)
# =============================================================================

print("\n[2/3] All halos in slab over density...")
print("-"*50)

with mpl.rc_context(plt.rcParams):
    ncols = 3
    nrows = int(np.ceil(len(idx_pick) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5.5*nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    rng = np.random.default_rng(0)

    for ax, z_idx in zip(axes, idx_pick):
        z_node = z_lc[z_idx]

        dens_slice = density_lc[:, :, z_idx].T
        im = ax.imshow(1 + dens_slice, origin='lower',
                       extent=[0, BOX_LEN, 0, BOX_LEN],
                       cmap='Greys',
                       norm=mcolors.LogNorm(vmin=0.1, vmax=10),
                       interpolation='bilinear')
        plt.colorbar(im, ax=ax, label=r'$1+\delta$',
                     fraction=0.046, pad=0.15)

        z_node_match = find_node_for_lc_idx(z_idx)
        tag = f"z{z_node_match:.4f}"
        masses_path = os.path.join(HALO_OUT, f"masses_{tag}.npy")
        coords_path = os.path.join(HALO_OUT, f"coords_{tag}.npy")

        n_halos = 0
        if os.path.exists(masses_path):
            m_box = np.load(masses_path)
            c_box = np.load(coords_path)
            m_slab, c_slab = slab_for_lc_idx(z_idx, m_box, c_box)
            n_halos = len(m_slab)

            if n_halos > 0:
                # subsample for plotting; n_halos in title stays the true count
                if n_halos > PLOT2_MAX_SCATTER:
                    sub = rng.choice(n_halos, PLOT2_MAX_SCATTER,
                                     replace=False)
                    m_plot, c_plot = m_slab[sub], c_slab[sub]
                else:
                    m_plot, c_plot = m_slab, c_slab

                sc = ax.scatter(c_plot[:, 0], c_plot[:, 1],
                                c=np.log10(m_plot),
                                s=3, cmap='plasma', alpha=0.6,
                                vmin=8.5, vmax=10.5, edgecolors='none')
                plt.colorbar(sc, ax=ax, label=r'$\log_{10}(M_\odot)$',
                             fraction=0.046, pad=0.04)

        ax.set_xlim(0, BOX_LEN)
        ax.set_ylim(0, BOX_LEN)
        ax.set_xlabel("x  [cMpc]")
        ax.set_ylabel("y  [cMpc]")
        ax.set_title(f"$z = {z_node:.2f}$  ({n_halos:,} halos)", fontsize=12)

    for ax in axes[len(idx_pick):]:
        ax.set_visible(False)

    fig.suptitle(f"All halos in slab + Density field  (seed {DIAG_SEED})",
                 fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(PLOT_DIR,
                "02b_all_halos_in_slab_over_density.png"),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR,
                "02b_all_halos_in_slab_over_density.pdf"),
                bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 02b_all_halos_in_slab_over_density.png / .pdf")

# =============================================================================
# PLOT 3: LIGHTCONE FIELDS
# =============================================================================

print("\n[3/3] All lightcone fields...")
print("-"*50)

fields_to_plot = [
    ("neutral_frac_lc", r"Neutral Fraction $x_\mathrm{HI}$", "plasma"),
    ("density_lc",      r"Matter Density $\delta$",          "viridis"),
    ("los_velocity_lc", "LOS Velocity [Mpc/s]",              "RdBu_r"),
    ("kinetic_temp_lc", r"Kinetic Temperature $T_k$ [K]",    "inferno"),
]

with mpl.rc_context(plt.rcParams):
    n_fields = len(fields_to_plot)
    fig, axes = plt.subplots(n_fields, 1, figsize=(14, 4*n_fields),
                             constrained_layout=True)
    if n_fields == 1:
        axes = [axes]

    for ax, (field_var, label, cmap) in zip(axes, fields_to_plot):
        field_data  = eval(field_var)
        mid_x       = HII_DIM // 2
        field_slice = field_data[mid_x, :, :]

        im = ax.imshow(field_slice, aspect='auto', origin='lower',
                       cmap=cmap, extent=[z_lc[0], z_lc[-1], 0, BOX_LEN],
                       interpolation='bilinear')
        plt.colorbar(im, ax=ax, label=label, pad=0.01)
        ax.set_xlabel("Redshift  $z$", fontsize=13)
        ax.set_ylabel("y  [cMpc]", fontsize=13)
        ax.set_title(label, fontsize=12)

    fig.suptitle(f"Lightcone Fields (y-z slice at mid-x)  (seed {DIAG_SEED})",
                 fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(PLOT_DIR, "03_all_lightcone_fields.png"),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, "03_all_lightcone_fields.pdf"),
                bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 03_all_lightcone_fields.png / .pdf")

print("\n" + "="*70)
print(f"CELL 3 COMPLETE  (seed {DIAG_SEED})")
print("="*70)

# =============================================================================
# HMF REDSHIFT EVOLUTION with CAMB + Lightcone Overlay (Fixed Volume)
# =============================================================================
import camb
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d

print("\n[X/4] HMF redshift evolution...")
print("-"*50)

H0     = 67.77
h      = H0 / 100.0
ombh2  = 0.04897 * h**2
omch2  = (0.3086 - 0.04897) * h**2
ns     = 0.9665
sigma8 = 0.8102
Om0    = 0.3086

cosmo_astropy = FlatLambdaCDM(H0=H0, Om0=Om0)
rho_mean_0    = Om0 * cosmo_astropy.critical_density0.to(
                    'M_sun/Mpc^3').value

# CAMB setup
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0, tau=0.054)
pars.InitPower.set_params(ns=ns, As=2.1e-9)
pars.set_matter_power(redshifts=[0.0], kmax=1000.0)

results = camb.get_results(pars)
kh, _, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1e4,
                                              npoints=500)
pk = pk[0]


def sigma_R_raw(R, kh, pk):
    def integrand(lnk):
        k  = np.exp(lnk)
        kR = k * R
        W  = 3 * (np.sin(kR) - kR * np.cos(kR)) / kR**3
        pk_interp = np.interp(k / h, kh, pk)
        pk_mpc    = pk_interp / h**3
        return k**3 * pk_mpc * W**2 / (2 * np.pi**2)
    val, _ = quad(integrand, np.log(1e-4), np.log(1e4), limit=200)
    return np.sqrt(val)


sigma8_raw = sigma_R_raw(8.0 / h, kh, pk)
norm       = sigma8 / sigma8_raw

R_grid     = np.logspace(-2, 3, 200)
sigma_grid = np.array([norm * sigma_R_raw(R, kh, pk) for R in R_grid])
sigma_interp = interp1d(np.log(R_grid), np.log(sigma_grid),
                        kind='cubic', fill_value='extrapolate')


def sigma_M(M_msun, z):
    R   = (3 * M_msun / (4 * np.pi * rho_mean_0)) ** (1/3)
    s0  = np.exp(sigma_interp(np.log(R)))
    Omz = cosmo_astropy.Om(z)
    gz  = (5/2)*Omz / (Omz**(4/7) - (1-Omz)
                       + (1+Omz/2)*(1+(1-Omz)/70))
    g0  = (5/2)*Om0 / (Om0**(4/7) - (1-Om0)
                       + (1+Om0/2)*(1+(1-Om0)/70))
    Dz  = gz / (g0 * (1 + z))
    return s0 * Dz


def dlnsigma_dlnM(M_msun, z, dlogM=0.01):
    M1 = M_msun * 10**(dlogM)
    M2 = M_msun * 10**(-dlogM)
    return (np.log(sigma_M(M1, z)) - np.log(sigma_M(M2, z))) \
           / (2*dlogM*np.log(10))


def f_sheth_tormen(nu, a=0.707, p=0.3, A=0.3222):
    nu2 = a * nu**2
    return A * np.sqrt(2*nu2/np.pi) * (1 + nu2**(-p)) * np.exp(-nu2/2)


def hmf_theory(M_msun, z, delta_c=1.686):
    s        = sigma_M(M_msun, z)
    nu       = delta_c / s
    dlnsdlnM = dlnsigma_dlnM(M_msun, z)
    f        = f_sheth_tormen(nu)
    return (rho_mean_0 / M_msun) * f * np.abs(dlnsdlnM)


# Plot HMF evolution with lightcone overlay
M_bins  = np.logspace(8.5, 12, 30)
M_cents = 0.5 * (M_bins[:-1] + M_bins[1:])
dlnM    = np.diff(np.log(M_bins))

z_sample = np.linspace(5, 20, 8)
colors   = plt.cm.plasma(np.linspace(0.1, 0.9, len(z_sample)))

with mpl.rc_context(plt.rcParams):
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    # Theory HMF
    for z, color in zip(z_sample, colors):
        hmf_st = np.array([hmf_theory(M, z) for M in M_cents])
        ax.plot(M_cents, hmf_st, lw=2.5, color=color,
                label=f'$z={z:.1f}$ (ST)')

    # Overlay lightcone HMF from our simulation
    halo_exists = np.array([np.any(np.isfinite(halo_mass_lc[:,:,i]))
                            for i in range(n_lc)])
    idx_with_halos = np.where(halo_exists)[0]

    if len(idx_with_halos) > 0:
        if len(idx_with_halos) > 6:
            idx_pick = idx_with_halos[np.linspace(
                0, len(idx_with_halos)-1, 6, dtype=int)]
        else:
            idx_pick = idx_with_halos

        colors_lc = plt.cm.Spectral(np.linspace(0.1, 0.9, len(idx_pick)))

        # slab is exactly one coeval cell thick (matches CELL 2 worker)
        V_slice = BOX_LEN**2 * cell_size_mpc

        for z_idx, color_lc in zip(idx_pick, colors_lc):
            z_node = z_lc[z_idx]

            # Load full-box catalogue, re-derive slab
            z_node_match = find_node_for_lc_idx(z_idx)
            tag          = f"z{z_node_match:.4f}"
            masses_path  = os.path.join(HALO_OUT, f"masses_{tag}.npy")
            coords_path  = os.path.join(HALO_OUT, f"coords_{tag}.npy")

            if os.path.exists(masses_path) and os.path.exists(coords_path):
                m_box       = np.load(masses_path)
                c_box       = np.load(coords_path)
                m_all, _    = slab_for_lc_idx(z_idx, m_box, c_box)
            else:
                continue

            if len(m_all) > 0:
                counts, _ = np.histogram(m_all, bins=M_bins)
                hmf_sim   = counts / (V_slice * dlnM)
                good      = counts >= 5

                ax.scatter(M_cents[good], hmf_sim[good], s=50,
                           color=color_lc, marker='o', alpha=0.6,
                           edgecolors='black', linewidth=0.5,
                           label=f'$z={z_node:.2f}$ (LC sim)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$M\,[M_\odot]$', fontsize=14)
    ax.set_ylabel(r'$dn/d\ln M$  [cMpc$^{-3}$]', fontsize=14)
    ax.set_title(rf"HMF Evolution: Sheth-Tormen (lines) vs Lightcone "
                 rf"(points)  |  CAMB P(k)  (seed {DIAG_SEED})",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, framealpha=0.9, loc='best')

    plt.savefig(os.path.join(plot_dir, "05_hmf_redshift_evolution.png"),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "05_hmf_redshift_evolution.pdf"),
                bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 05_hmf_redshift_evolution.png / .pdf")

    # =============================================================================
# PLOT 4: HALO LIGHTCONE — y-z slice (mass + count)
# =============================================================================

print("\n[4/4] Halo lightcone y-z slice...")
print("-"*50)

# x-slice index (mid-box, matches the field plots)
mid_x        = HII_DIM // 2
x_slice_cMpc = mid_x * cell_size_mpc

with mpl.rc_context(plt.rcParams):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

    # ── top: average halo mass per pixel ────────────────────────────────
    mass_slice = halo_mass_lc[mid_x, :, :]          # (HII_DIM, n_lc)
    with np.errstate(invalid='ignore'):
        logm = np.log10(mass_slice)
    im0 = axes[0].imshow(
        logm, aspect='auto', origin='lower',
        extent=[z_lc[0], z_lc[-1], 0, BOX_LEN],
        cmap='inferno', interpolation='nearest')
    plt.colorbar(im0, ax=axes[0],
                 label=r'$\log_{10}\langle M_\mathrm{halo}\rangle\ [M_\odot]$',
                 pad=0.01)
    axes[0].set_xlabel(r"Redshift  $z$", fontsize=13)
    axes[0].set_ylabel(r"y  [cMpc]", fontsize=13)
    axes[0].set_title(
        rf"Average halo mass per pixel  "
        rf"(mass cut $> 10^{{{np.log10(MASS_CUT):.1f}}}\ M_\odot$)",
        fontsize=12)

    # ── bottom: halo count per pixel ────────────────────────────────────
    count_slice = halo_count_lc[mid_x, :, :]
    im1 = axes[1].imshow(
        np.log10(count_slice + 1), aspect='auto', origin='lower',
        extent=[z_lc[0], z_lc[-1], 0, BOX_LEN],
        cmap='inferno', interpolation='nearest')
    plt.colorbar(im1, ax=axes[1],
                 label=r'$\log_{10}(N_\mathrm{halo}+1)$ per pixel',
                 pad=0.01)
    axes[1].set_xlabel(r"Redshift  $z$", fontsize=13)
    axes[1].set_ylabel(r"y  [cMpc]", fontsize=13)
    axes[1].set_title("Halo count per pixel", fontsize=12)

    fig.suptitle(
        rf"Halo Lightcone — y-z slice at x = {x_slice_cMpc:.1f} cMpc  "
        rf"(seed {DIAG_SEED})", fontsize=16, fontweight='bold')

    plt.savefig(os.path.join(PLOT_DIR, "04_halo_lightcone_slice.png"),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, "04_halo_lightcone_slice.pdf"),
                bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 04_halo_lightcone_slice.png / .pdf")

# %%
# %%
# =============================================================================
# CELL 3b: LAE diagnostics — overlay on xHI + luminosity function + LAE lightcone
# =============================================================================

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.cosmology import FlatLambdaCDM

# ── config ────────────────────────────────────────────────────────────────
SIMPLEGEN_DIR  = "/user1/swanith/SiMPLE-Gen/SiMPLEGen/data"
OBS_LF_DIR     = "/user1/swanith/kSZ2_halo_project/obs_lf"
PLOT_DIR       = os.path.join(plot_dir, "LAE_diagnostics")
os.makedirs(PLOT_DIR, exist_ok=True)

# match Jahaan's model redshifts exactly
Z_TARGETS = [5.756, 5.946, 6.149, 6.368, 6.604, 6.860, 7.139, 7.444, 7.780, 8.150]
DZ_HALO   = 0.10
REW_CUT   = 10.0
LLYA_CUT  = 1e42
DZ_LF_BIN = 0.30

cosmo_lf = FlatLambdaCDM(H0=67.77, Om0=0.3086, Ob0=0.0489)
BOX      = float(inputs.simulation_options.BOX_LEN)

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 13,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})

# only plot seeds with completed data
RANDOM_SEEDS_DONE = [s for s in RANDOM_SEEDS
                     if os.path.exists(os.path.join(
                         SIMPLEGEN_DIR, f"seed_{s}",
                         "lightcone_lae", "redshifts.npy"))]
print(f"  plotting seeds: {RANDOM_SEEDS_DONE}")


def load_seed(seed):
    d = os.path.join(SIMPLEGEN_DIR, f"seed_{seed}", "lightcone_lae")
    try:
        out = {k: np.load(os.path.join(d, f"{k}.npy"))
               for k in ["LLya", "REW", "damping", "halomass",
                         "coords", "redshifts"]}
        out["LLya_obs"] = out["LLya"] * out["damping"]
        out["is_LAE"]   = ((out["REW"] >= REW_CUT) &
                           (out["LLya_obs"] >= LLYA_CUT) &
                           (out["damping"] > 0))
        return out
    except FileNotFoundError as e:
        print(f"  ✗ seed {seed}: {e}")
        return None


def load_jahaan_lf(z_target, dz=0.05):
    """Load Jahaan's model LF file nearest to z_target."""
    files = glob.glob(os.path.join(OBS_LF_DIR, "lya_lum_obs_z*.npy"))
    best, best_dz = None, 1e9
    for f in files:
        z_str = os.path.basename(f).split('_z')[1].split('_')[0]
        z_f   = float(z_str)
        if abs(z_f - z_target) < best_dz:
            best_dz = abs(z_f - z_target)
            best    = f
    if best is None or best_dz > dz:
        return None, None
    lums = np.load(best)
    return lums, float(os.path.basename(best).split('_z')[1].split('_')[0])


# =============================================================================
# Part 1 — LAE overlay on xHI (LAEs only, no non-LAEs)
# 3 panels per seed from Z_TARGETS[:3]
# =============================================================================
print("=" * 70)
print("CELL 3b — LAE overlay on xHI")
print("=" * 70)

lightcones = {}
for seed in RANDOM_SEEDS_DONE:
    lc_path = os.path.join(cache_dir, f"seed_{seed}", "lightcone.h5")
    if os.path.exists(lc_path):
        lightcones[seed] = p21c.LightCone.from_file(lc_path, safe=False)
        print(f"  ✓ seed {seed} lightcone loaded")

for seed in RANDOM_SEEDS_DONE:
    data = load_seed(seed)
    if data is None or seed not in lightcones:
        continue

    lc        = lightcones[seed]
    xHI_seed  = lc.lightcones['neutral_fraction']
    z_lc_seed = np.array(lc.lightcone_redshifts)

    z_panel = [Z_TARGETS[0], Z_TARGETS[4], Z_TARGETS[-1]]  # z~5.76, 6.60, 8.15
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)

    for ax, z_target in zip(axes, z_panel):
        z_idx  = int(np.argmin(np.abs(z_lc_seed - z_target)))
        xHI_sl = xHI_seed[:, :, z_idx].T

        im = ax.imshow(xHI_sl, origin='lower', extent=[0, BOX, 0, BOX],
                       cmap='Blues', vmin=0.0, vmax=1.0,
                       interpolation='bilinear')
        plt.colorbar(im, ax=ax, label=r'$x_{\rm HI}$',
                     fraction=0.046, pad=0.02)

        z_mask   = np.abs(data["redshifts"] - z_target) <= DZ_HALO
        is_lae_z = data["is_LAE"][z_mask]
        c_lae    = data["coords"][z_mask][is_lae_z]

        if len(c_lae) > 0:
            ax.scatter(c_lae[:, 0], c_lae[:, 1], s=8, color='red',
                       alpha=0.8, edgecolors='none',
                       label=f'LAE ({is_lae_z.sum():,})', zorder=3)

        ax.set_xlim(0, BOX); ax.set_ylim(0, BOX)
        ax.set_xlabel('x [cMpc]'); ax.set_ylabel('y [cMpc]')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=9,
                  framealpha=0.75, edgecolor='white')
        ax.set_title(f"$z = {z_target:.3f}$  (|Δz| ≤ {DZ_HALO})\n"
                     f"$\\bar x_{{\\rm HI}}={xHI_sl.mean():.2f}$")

    fig.suptitle(f"Seed {seed} — LAEs (red) over $x_{{\\rm HI}}$  "
                 f"(REW ≥ {REW_CUT} Å, $L_{{\\rm Ly\\alpha}} \\geq 10^{{42}}$ erg/s)",
                 fontsize=14, fontweight='bold')
    fpng = os.path.join(PLOT_DIR, f"LAE_overlay_seed{seed}.png")
    fig.savefig(fpng); fig.savefig(fpng.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  ✓ seed {seed}: {fpng}")


# =============================================================================
# Part 2 — Lyman-α luminosity function vs Jahaan's model
# =============================================================================
print("\n" + "=" * 70)
print("CELL 3b — Lyman-α LF vs Jahaan's model")
print("=" * 70)

logL_edges = np.arange(41.5, 44.0, 0.2)
logL_cen   = 0.5 * (logL_edges[1:] + logL_edges[:-1])
dlogL      = logL_edges[1] - logL_edges[0]
lf_colors  = plt.cm.viridis(np.linspace(0.05, 0.95, len(Z_TARGETS)))

for seed in RANDOM_SEEDS_DONE:
    data = load_seed(seed)
    if data is None:
        continue

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)

    for z_target, color in zip(Z_TARGETS, lf_colors):
        zlo = z_target - DZ_LF_BIN / 2
        zhi = z_target + DZ_LF_BIN / 2
        chi_lo = cosmo_lf.comoving_distance(zlo).to_value('Mpc')
        chi_hi = cosmo_lf.comoving_distance(zhi).to_value('Mpc')
        vol    = (BOX ** 2) * (chi_hi - chi_lo)

        sel   = ((data["redshifts"] >= zlo) & (data["redshifts"] < zhi)
                 & data["is_LAE"])
        n_sel = int(sel.sum())
        if n_sel == 0:
            continue

        logL_obs  = np.log10(data["LLya_obs"][sel])
        counts, _ = np.histogram(logL_obs, bins=logL_edges)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi     = counts / (vol * dlogL)
            phi_err = np.sqrt(counts) / (vol * dlogL)
        good = counts > 0

        ax.errorbar(logL_cen[good], phi[good], yerr=phi_err[good],
                    fmt='o-', color=color, capsize=2, lw=1.6, ms=5,
                    label=f'$z={z_target:.2f}$ (N={n_sel:,})')

        # ── overplot Jahaan's model LF ────────────────────────────
        lums, z_jah = load_jahaan_lf(z_target)
        if lums is not None:
            counts_j, _ = np.histogram(np.log10(lums), bins=logL_edges)
            # Jahaan's box: 320 cMpc, same dlogL
            vol_j = 320.0**3
            with np.errstate(divide='ignore', invalid='ignore'):
                phi_j = counts_j / (vol_j * dlogL)
            good_j = counts_j > 0
            ax.plot(logL_cen[good_j], phi_j[good_j],
                    's--', color=color, lw=1.2, ms=4, alpha=0.6,
                    label=f'Jahaan $z={z_jah:.2f}$')

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}(L_{\rm Ly\alpha,obs}\,/\,\mathrm{erg\,s^{-1}})$')
    ax.set_ylabel(r'$\phi$  [cMpc$^{-3}$ dex$^{-1}$]')
    ax.set_title(f"Seed {seed} — Lyman-$\\alpha$ luminosity function",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)

    fpng = os.path.join(PLOT_DIR, f"LF_seed{seed}.png")
    fig.savefig(fpng); fig.savefig(fpng.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  ✓ seed {seed}: {fpng}")


# =============================================================================
# Part 3 — LAE lightcone (y-z slice), Brewer Blues colormap
# x-axis starts at z=5
# =============================================================================
HII_DIM = int(inputs.simulation_options.HII_DIM)
print("\n" + "=" * 70)
print("CELL 3b — LAE lightcone (y-z slice)")
print("=" * 70)

# Brewer Blues: white → dark blue
blues = plt.colormaps['YlGnBu']

for seed in RANDOM_SEEDS_DONE:
    data = load_seed(seed)
    if data is None or seed not in lightcones:
        continue

    lc        = lightcones[seed]
    z_lc_seed = np.array(lc.lightcone_redshifts, dtype=np.float64)
    n_lc      = len(z_lc_seed)
    cell_size = BOX / HII_DIM

    yi = np.clip((data["coords"][:, 1] / cell_size).astype(int),
                 0, HII_DIM - 1)
    zi = np.clip(np.searchsorted(
        0.5 * (z_lc_seed[1:] + z_lc_seed[:-1]), data["redshifts"]),
        0, n_lc - 1)

    L_grid = np.zeros((HII_DIM, n_lc))
    N_grid = np.zeros((HII_DIM, n_lc))
    lae    = data["is_LAE"]
    np.add.at(L_grid, (yi[lae], zi[lae]), data["LLya_obs"][lae])
    np.add.at(N_grid, (yi[lae], zi[lae]), 1.0)

    with np.errstate(divide='ignore'):
        logL = np.log10(np.where(L_grid > 0, L_grid, np.nan))
        logN = np.where(N_grid > 0, np.log10(N_grid), np.nan)

    # x-axis: only show z>=5
    z_start = 5.0
    zi_start = np.argmin(np.abs(z_lc_seed - z_start))

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

    im0 = axes[0].imshow(logL[:, zi_start:], aspect='auto', origin='lower',
                         extent=[5.0, z_lc_seed[-1], 0, BOX],
                         cmap=blues, interpolation='nearest')
    plt.colorbar(im0, ax=axes[0], pad=0.01,
                 label=r'$\log_{10}\sum L_{\rm Ly\alpha,obs}$ [erg s$^{-1}$]')
    axes[0].set_xlabel(r'Redshift  $z$')
    axes[0].set_ylabel(r'y  [cMpc]')
    axes[0].set_xlim(5.0, z_lc_seed[-1])
    axes[0].set_xticks([5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20])
    axes[0].set_title(r'Summed observed $L_{\rm Ly\alpha}$ per pixel')

    im1 = axes[1].imshow(logN[:, zi_start:], aspect='auto', origin='lower',
                         extent=[5.0, z_lc_seed[-1], 0, BOX],
                         cmap=blues, interpolation='nearest')
    plt.colorbar(im1, ax=axes[1], pad=0.01,
                 label=r'$\log_{10} N_{\rm LAE}$ per pixel')
    axes[1].set_xlabel(r'Redshift  $z$')
    axes[1].set_ylabel(r'y  [cMpc]')
    axes[1].set_xlim(5.0, z_lc_seed[-1])
    axes[1].set_xticks([5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20])
    axes[1].set_title(r'LAE count per pixel')

    fig.suptitle(f"Seed {seed} — LAE lightcone (y-z slice)  "
                 f"[{int(lae.sum()):,} LAEs]",
                 fontsize=15, fontweight='bold')

    fpng = os.path.join(PLOT_DIR, f"LAE_lightcone_seed{seed}.png")
    fig.savefig(fpng); fig.savefig(fpng.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  ✓ seed {seed}: {fpng}")

print("\n" + "=" * 70)
print(f"✓ CELL 3b complete  →  {PLOT_DIR}")
print("=" * 70)

# %%
# %%
# =============================================================================
# CELL 4: Reionization History + Optical Depth Analysis  — ALL SEEDS
# v4.1.0; mean ± std across seeds (cosmic-variance band)
# Inherited from Cell 1: inputs, cosmo, plot_dir, RANDOM_SEEDS, N_SEEDS
# Inherited from Cell 2: lightcones (dict)
# Produces:  tau_results_all  (dict keyed by seed; used by Cells 5+)
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

print("\n" + "="*70)
print(f"CELL 4 — REIONIZATION HISTORY + OPTICAL DEPTH  ({N_SEEDS} seeds)")
print("="*70)

# =============================================================================
# PHYSICAL CONSTANTS  (seed-independent)
# =============================================================================

h_little       = 0.6766
Omega_b        = 0.04897468161869667
rho_crit_p_cm3 = 1.88e-29 * h_little**2 / (1.67e-24)
n_H0_cm3       = Omega_b * rho_crit_p_cm3
sigma_T_cm2    = 6.65e-25
cm_per_Mpc     = 3.086e24
n_e0_Mpc3      = n_H0_cm3   * cm_per_Mpc**3
sigma_T_Mpc2   = sigma_T_cm2 / cm_per_Mpc**2
prefactor      = n_e0_Mpc3 * sigma_T_Mpc2

print(f"\nPhysical constants:")
print(f"  n_H0      = {n_H0_cm3:.6e} cm^-3")
print(f"  sigma_T   = {sigma_T_cm2:.6e} cm^2")
print(f"  Prefactor = {prefactor:.6e} Mpc^-1")

# =============================================================================
# PART A: REIONIZATION HISTORIES — per seed + mean across seeds
# =============================================================================

all_z_nodes   = {}
all_x_e_nodes = {}
all_xHI_nodes = {}

for seed, lc in lightcones.items():
    z_nodes_s = np.array(lc.node_redshifts, dtype=float)
    gq        = lc.global_quantities

    # v4.1.0 key is 'neutral_fraction' (renamed from 'xH_box' in v3)
    if 'neutral_fraction' in gq:
        xHI_nodes_s = np.array(gq['neutral_fraction'], dtype=float)
    elif 'xH_box' in gq:
        xHI_nodes_s = np.array(gq['xH_box'], dtype=float)
    else:
        raise ValueError(
            f"No neutral fraction in global_quantities for seed {seed}. "
            f"Keys: {list(gq.keys())}"
        )

    # sort low-z → high-z
    sort_idx               = np.argsort(z_nodes_s)
    all_z_nodes  [seed]    = z_nodes_s  [sort_idx]
    all_xHI_nodes[seed]    = xHI_nodes_s[sort_idx]
    all_x_e_nodes[seed]    = 1.0 - all_xHI_nodes[seed]

# Common redshift grid for averaging — intersection of all seeds' ranges
z_min_common = max(all_z_nodes[s].min() for s in lightcones)
z_max_common = min(all_z_nodes[s].max() for s in lightcones)
z_common_xe  = np.linspace(z_min_common, z_max_common, 500)

xe_interp = np.array([
    np.interp(z_common_xe, all_z_nodes[s], all_x_e_nodes[s])
    for s in lightcones.keys()
])
xHI_interp = np.array([
    np.interp(z_common_xe, all_z_nodes[s], all_xHI_nodes[s])
    for s in lightcones.keys()
])

xe_mean  = np.mean(xe_interp,  axis=0)
xe_std   = np.std (xe_interp,  axis=0)
xHI_mean = np.mean(xHI_interp, axis=0)
xHI_std  = np.std (xHI_interp, axis=0)

# Mean reionization midpoint
z_xe_half_mean = float(np.interp(0.5, xe_mean[::-1], z_common_xe[::-1]))
print(f"\nMean z(x_e = 0.5) across {N_SEEDS} seeds: z = {z_xe_half_mean:.2f}")
for seed in lightcones.keys():
    z_half = float(np.interp(0.5, all_x_e_nodes[seed], all_z_nodes[seed]))
    print(f"  Seed {seed:3d}: z(x_e=0.5) = {z_half:.2f}")

# Colormap for per-seed lines (used in all three plots)
cmap_seeds = plt.cm.plasma(np.linspace(0.1, 0.9, len(lightcones)))

# =============================================================================
# PLOT 4a: x_e vs z  (per-seed thin + mean ± 1σ)
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

for i, seed in enumerate(lightcones.keys()):
    ax.plot(all_z_nodes[seed], all_x_e_nodes[seed],
            color=cmap_seeds[i], lw=1.0, alpha=0.4)

ax.fill_between(z_common_xe, xe_mean - xe_std, xe_mean + xe_std,
                color='darkblue', alpha=0.2, label=r'$\pm 1\sigma$')
ax.plot(z_common_xe, xe_mean,
        color='darkblue', lw=2.5, label=f'Mean ({N_SEEDS} seeds)')

ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.7)
ax.axvline(z_xe_half_mean, color='gray', ls='--', lw=1, alpha=0.7)
ax.text(z_xe_half_mean + 0.1, 0.52,
        fr'$z(x_e=0.5)={z_xe_half_mean:.2f}$', fontsize=13,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'Ionisation Fraction $x_e$')
ax.set_ylim(-0.05, 1.05)
ax.invert_xaxis()
ax.legend(loc='best')

fig.savefig(f"{plot_dir}/xe_history.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{plot_dir}/xe_history.pdf", bbox_inches='tight')
plt.close(fig)
print(f"\n✓ Saved: xe_history.png / .pdf")

# =============================================================================
# PLOT 4b: x_HI vs z  (per-seed thin + mean ± 1σ)
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

for i, seed in enumerate(lightcones.keys()):
    ax.plot(all_z_nodes[seed], all_xHI_nodes[seed],
            color=cmap_seeds[i], lw=1.0, alpha=0.4)

ax.fill_between(z_common_xe, xHI_mean - xHI_std, xHI_mean + xHI_std,
                color='darkred', alpha=0.2, label=r'$\pm 1\sigma$')
ax.plot(z_common_xe, xHI_mean,
        color='darkred', lw=2.5, label=f'Mean ({N_SEEDS} seeds)')

ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'Neutral Fraction $x_{\rm HI}$')
ax.set_ylim(-0.05, 1.05)
ax.invert_xaxis()
ax.legend(loc='best')

fig.savefig(f"{plot_dir}/xHI_history.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{plot_dir}/xHI_history.pdf", bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved: xHI_history.png / .pdf")

# =============================================================================
# PART B: OPTICAL DEPTH τ(<z) — per seed, then averaged
# =============================================================================

print("\n--- Optical Depth Calculation ---")
tau_results_all = {}

for seed, lc in lightcones.items():
    red_axis = np.array(lc.lightcone_redshifts, dtype=float)
    pos_axis = np.array(lc.lightcone_distances, dtype=float)

    # sort low-z → high-z if needed
    if red_axis[0] > red_axis[-1]:
        red_axis = red_axis[::-1]
        pos_axis = pos_axis[::-1]

    z_nodes_s   = all_z_nodes  [seed]
    x_e_nodes_s = all_x_e_nodes[seed]

    x_e_interp = np.interp(red_axis, z_nodes_s, x_e_nodes_s)

    ds_Mpc  = np.abs(np.diff(pos_axis))
    z_mid   = 0.5 * (red_axis[:-1] + red_axis[1:])
    x_e_mid = 0.5 * (x_e_interp[:-1] + x_e_interp[1:])

    dtau      = prefactor * x_e_mid * (1.0 + z_mid)**2 * ds_Mpc
    tau       = np.cumsum(dtau)
    tau_total = float(tau[-1])

    tau_results_all[seed] = {
        'red_axis' : red_axis,
        'z_mid'    : z_mid,
        'x_e_mid'  : x_e_mid,
        'ds_Mpc'   : ds_Mpc,
        'dtau'     : dtau,
        'tau'      : tau,
        'tau_total': tau_total,
    }
    print(f"  Seed {seed:3d}: tau_total = {tau_total:.6f}")

tau_totals = np.array([tau_results_all[s]['tau_total'] for s in lightcones])
print(f"\n  Mean tau = {tau_totals.mean():.6f} ± {tau_totals.std():.6f}")

# Stack τ(<z) across seeds — handle potential grid mismatch safely
ref_seed  = next(iter(lightcones))
ref_z_mid = tau_results_all[ref_seed]['z_mid']

grids_match = all(
    tau_results_all[s]['z_mid'].shape == ref_z_mid.shape
    and np.allclose(tau_results_all[s]['z_mid'], ref_z_mid)
    for s in lightcones
)

if grids_match:
    z_common_tau = ref_z_mid
    tau_matrix   = np.array([tau_results_all[s]['tau'] for s in lightcones])
else:
    z_min_t = max(tau_results_all[s]['z_mid'].min() for s in lightcones)
    z_max_t = min(tau_results_all[s]['z_mid'].max() for s in lightcones)
    z_common_tau = np.linspace(z_min_t, z_max_t, 500)
    tau_matrix   = np.array([
        np.interp(z_common_tau,
                  tau_results_all[s]['z_mid'],
                  tau_results_all[s]['tau'])
        for s in lightcones
    ])

tau_mean = np.mean(tau_matrix, axis=0)
tau_std  = np.std (tau_matrix, axis=0)

# =============================================================================
# PLOT 4c: Cumulative optical depth τ(<z)  (per-seed thin + mean ± 1σ)
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

for i, seed in enumerate(lightcones.keys()):
    z_mid_seed = tau_results_all[seed]['z_mid']
    tau_seed   = tau_results_all[seed]['tau']
    ax.plot(z_mid_seed, tau_seed,
            color=cmap_seeds[i], lw=1.0, alpha=0.4)

ax.fill_between(z_common_tau, tau_mean - tau_std, tau_mean + tau_std,
                color='darkgreen', alpha=0.2, label=r'$\pm 1\sigma$')
ax.plot(z_common_tau, tau_mean, color='darkgreen', lw=2.5,
        label=fr'Mean ({N_SEEDS} seeds), '
              fr'$\tau_\mathrm{{total}}={tau_totals.mean():.4f}\pm{tau_totals.std():.4f}$')

ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'Cumulative Optical Depth $\tau(<z)$')
ax.invert_xaxis()
ax.legend(loc='best')
ax.text(0.05, 0.95,
        f'$\\tau_{{\\rm total}} = {tau_totals.mean():.6f} \\pm '
        f'{tau_totals.std():.6f}$\n'
        f'N seeds = {N_SEEDS}',
        transform=ax.transAxes, fontsize=13,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.savefig(f"{plot_dir}/tau_history.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{plot_dir}/tau_history.pdf", bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved: tau_history.png / .pdf")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print(f"CELL 3 COMPLETE")
print(f"{'='*70}")
print(f"  z(x_e = 0.5) mean : {z_xe_half_mean:.2f}")
print(f"  tau_total    mean : {tau_totals.mean():.6f} ± {tau_totals.std():.6f}")
print(f"  tau_results_all   : ready for kSZ integrand (Cell 5)")
print(f"{'='*70}")

# %%
# %%
# =============================================================================
# CELL 5: kSZ Integrand with Visibility Function  — ALL SEEDS
#
# kSZ integrand:
#   (1 + delta_b) * x_e * (v_los / c) * exp[-tau(z)]
#
# No parallelization needed — per-seed work is just NumPy on small 3D arrays.
#
# Inherited from Cell 1 : inputs, cache_dir, plot_dir, RANDOM_SEEDS, N_SEEDS
# Inherited from Cell 2 : lightcones (dict), density_lc_all,
#                         neutral_frac_lc_all, los_velocity_lc_all
# Inherited from Cell 4 : tau_results_all
# Produces              : kSZ_integrand_all  (dict keyed by seed)
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

print("\n" + "="*70)
print(f"CELL 5 — kSZ INTEGRAND WITH VISIBILITY FUNCTION  ({N_SEEDS} seeds)")
print("="*70)

# =============================================================================
# CONSTANTS
# =============================================================================

c_Mpc_s = 299792.458 / 3.08567758e19   # speed of light in Mpc/s
z_obs   = 5.0                           # observer redshift

print(f"  c          = {c_Mpc_s:.6e} Mpc/s")
print(f"  z_obs      = {z_obs}")

# =============================================================================
# PLOT SEED — used for the diagnostic mid-slice figure only
# =============================================================================

PLOT_SEED = RANDOM_SEEDS[0]   # change to any seed in RANDOM_SEEDS

# =============================================================================
# CACHE DIR
# =============================================================================

integrand_dir = f"{cache_dir}/kSZ_integrands"
os.makedirs(integrand_dir, exist_ok=True)

# =============================================================================
# PER-SEED LOOP — load from cache or compute
# =============================================================================

kSZ_integrand_all = {}

for seed in RANDOM_SEEDS:
    if seed not in lightcones:
        print(f"  ✗ seed {seed}: lightcone not loaded — skipping")
        continue

    integrand_cache = f"{integrand_dir}/kSZ_integrand_seed{seed}.npy"

    if os.path.exists(integrand_cache):
        kSZ_integrand_all[seed] = np.load(integrand_cache)
        rms = np.sqrt(np.mean(kSZ_integrand_all[seed]**2))
        print(f"  ✓ seed {seed:3d}: cached    "
              f"shape={kSZ_integrand_all[seed].shape}  rms={rms:.4e}")
        continue

    # ── redshift axis (per seed) ──────────────────────────────────────────
    lc            = lightcones[seed]
    red_axis_full = np.array(lc.lightcone_redshifts, dtype=np.float64)
    if red_axis_full[0] > red_axis_full[-1]:
        red_axis_full = red_axis_full[::-1]

    ind_z = np.where(
        red_axis_full <= float(inputs.simulation_options.Z_HEAT_MAX))[0]

    # ── 3D fields from CELL 2 dicts ──────────────────────────────────────
    # los_velocity_lc is in Mpc/s (confirmed in CELL 2)
    density_1plus = (1.0 + density_lc_all     [seed][:, :, ind_z]).astype(np.float64)
    x_e_3D        = (1.0 - neutral_frac_lc_all[seed][:, :, ind_z]).astype(np.float64)
    v_los_Mpc_s   =        los_velocity_lc_all[seed][:, :, ind_z] .astype(np.float64)

    # ── interpolate tau(z) onto LC redshift grid (per-seed tau) ──────────
    tr        = tau_results_all[seed]
    red_axis  = np.array(tr['red_axis'], dtype=np.float64)
    tau_array = np.array(tr['tau'],      dtype=np.float64)
    z_mid_arr = np.array(tr['z_mid'],    dtype=np.float64)

    tau_extended = np.concatenate([[0.0],        tau_array])
    z_extended   = np.concatenate([[red_axis[0]], z_mid_arr])

    tau_at_lc     = np.interp(red_axis_full[ind_z], z_extended, tau_extended)
    visibility_3D = np.exp(-tau_at_lc)[None, None, :]

    # ── integrand ────────────────────────────────────────────────────────
    kSZ_integrand_seed = (density_1plus
                          * x_e_3D
                          * (v_los_Mpc_s / c_Mpc_s)
                          * visibility_3D)

    np.save(integrand_cache, kSZ_integrand_seed)
    kSZ_integrand_all[seed] = kSZ_integrand_seed

    rms = np.sqrt(np.mean(kSZ_integrand_seed**2))
    print(f"  ✓ seed {seed:3d}: computed  "
          f"shape={kSZ_integrand_seed.shape}  rms={rms:.4e}  → cached")

# =============================================================================
# DIAGNOSTIC PLOT — mid-slice of the integrand cube  (PLOT_SEED only)
# =============================================================================

if PLOT_SEED not in kSZ_integrand_all:
    raise RuntimeError(f"PLOT_SEED={PLOT_SEED} has no integrand — cannot plot.")

print(f"\n  Diagnostic plot (seed {PLOT_SEED})")

kSZ_integrand = kSZ_integrand_all[PLOT_SEED]   # local alias for plot code
lc_plot       = lightcones[PLOT_SEED]

red_axis_full = np.array(lc_plot.lightcone_redshifts, dtype=np.float64)
if red_axis_full[0] > red_axis_full[-1]:
    red_axis_full = red_axis_full[::-1]
ind_z = np.where(
    red_axis_full <= float(inputs.simulation_options.Z_HEAT_MAX))[0]

lc_distances = np.array(lc_plot.lightcone_distances, dtype=np.float64)
if lc_distances[0] > lc_distances[-1]:
    lc_distances = lc_distances[::-1]

slice_2D = kSZ_integrand[:, :, kSZ_integrand.shape[2] // 2]
x_extent = float(lc_distances[ind_z].max())
y_extent = float(inputs.simulation_options.BOX_LEN)
vmax     = float(np.percentile(np.abs(kSZ_integrand), 99))

fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)

im = ax.imshow(
    slice_2D.T,
    extent=[0, x_extent, 0, y_extent],
    aspect='auto',
    cmap='seismic',
    origin='lower',
    vmin=-vmax,
    vmax=vmax,
)
plt.colorbar(im, ax=ax, label=r'kSZ Integrand  $(1+\delta)x_e v_{\rm los}/c\,e^{-\tau}$')
ax.set_xlabel('Comoving Distance [Mpc]')
ax.set_ylabel('Comoving Distance [Mpc]')

# twin redshift axis on top
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
xticks     = ax.get_xticks()
z_at_ticks = np.interp(xticks, lc_distances, red_axis_full)
ax2.set_xticks(xticks)
ax2.set_xticklabels([f'{z:.1f}' for z in z_at_ticks])
ax2.set_xlabel(r'Redshift $z$')

fig.suptitle(
    rf'kSZ Integrand: $(1+\delta_b)\,x_e\,v_{{\rm los}}/c\;e^{{-\tau(z)}}$  '
    rf'(seed {PLOT_SEED})',
    fontweight='bold')

fname = f"kSZ_integrand_seed{PLOT_SEED}"
fig.savefig(f"{plot_dir}/{fname}.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{plot_dir}/{fname}.pdf", bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: {fname}.png / .pdf")

# =============================================================================
# SUMMARY
# =============================================================================

rms_vals = np.array([
    np.sqrt(np.mean(kSZ_integrand_all[s]**2)) for s in kSZ_integrand_all
])

print(f"\n{'='*70}")
print(f"CELL 5 COMPLETE")
print(f"{'='*70}")
print(f"  kSZ_integrand_all : {len(kSZ_integrand_all)}/{N_SEEDS} seeds  "
      f"shape={kSZ_integrand_all[PLOT_SEED].shape}")
print(f"  rms across seeds  : mean={rms_vals.mean():.4e} ± {rms_vals.std():.4e}")
print(f"  ready for Cell 6 — per-seed LoS integration → kSZ maps")
print(f"{'='*70}")

# %%
# %%
# =============================================================================
# CELL 6: Line-of-Sight Integrated kSZ Map  — ALL SEEDS
#
# kSZ(z_obs) = ∫ dχ [ n_e0 σ_T (1/a²) (1+δ_b) x_e (v_los/c) exp(-τ) ]
#
# No parallelization — per-seed work is one np.sum over HII_DIM² × ~few-hundred.
#
# Inherited from Cell 1 : inputs, cache_dir, plot_dir, RANDOM_SEEDS, N_SEEDS
# Inherited from Cell 4 : tau_results_all
# Inherited from Cell 5 : kSZ_integrand_all
# Produces              : kSZ_map_all  (dict keyed by seed)
# =============================================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt

print("\n" + "="*70)
print(f"CELL 6 — LINE-OF-SIGHT kSZ MAP INTEGRATION  ({N_SEEDS} seeds)")
print("="*70)

# =============================================================================
# PHYSICAL CONSTANTS (CGS)
# =============================================================================

c_cm_s        = 3.0e10
sigma_T_cm2   = 6.6525e-25
n_e0_cm3      = 2.06e-7
Mpc_to_cm     = 3.0857e24
prefactor_cgs = n_e0_cm3 * sigma_T_cm2 * c_cm_s   # s⁻¹

z_obs = 5.0

print(f"  prefactor = {prefactor_cgs:.4e} s⁻¹")
print(f"  z_obs     = {z_obs:.1f}")

# =============================================================================
# PLOT SEED — used for the diagnostic map figure only
# =============================================================================

PLOT_SEED = RANDOM_SEEDS[0]   # change to any seed in RANDOM_SEEDS

# =============================================================================
# CACHE DIR
# =============================================================================

kSZ_maps_dir = f"{cache_dir}/kSZ_maps"
os.makedirs(kSZ_maps_dir, exist_ok=True)

# =============================================================================
# PER-SEED LOOP — load from cache or compute
# =============================================================================

kSZ_map_all = {}

for seed in RANDOM_SEEDS:
    if seed not in kSZ_integrand_all:
        print(f"  ✗ seed {seed}: no kSZ integrand — skipping")
        continue

    map_cache = f"{kSZ_maps_dir}/kSZ_map_z{z_obs:.1f}_seed{seed}.npy"

    if os.path.exists(map_cache):
        kSZ_map_all[seed] = np.load(map_cache)
        rms = np.sqrt(np.mean(kSZ_map_all[seed]**2))
        print(f"  ✓ seed {seed:3d}: cached    "
              f"shape={kSZ_map_all[seed].shape}  RMS={rms:.4e}")
        continue

    # ── axes from this seed's tau_results (full lightcone) ───────────────
    tr        = tau_results_all[seed]
    red_axis  = np.array(tr['red_axis'], dtype=np.float64)
    ds_Mpc    = np.array(tr['ds_Mpc'],   dtype=np.float64)
    z_mid     = np.array(tr['z_mid'],    dtype=np.float64)
    ds_cm     = ds_Mpc * Mpc_to_cm

    # scale factor at midpoints (same length as z_mid / ds_Mpc)
    a        = 1.0 / (1.0 + red_axis)
    a_sq_mid = 0.5 * (a[:-1]**2 + a[1:]**2)

    # ── integration range: z_mid >= z_obs ────────────────────────────────
    # Note: kSZ_integrand has shape (HII_DIM, HII_DIM, N_ind_z)
    # where N_ind_z = number of lightcone slices up to Z_HEAT_MAX.
    # z_mid has N_full-1 elements covering the full lightcone.
    # We need to select only the slices that:
    #   (a) are within kSZ_integrand's range  AND
    #   (b) satisfy z_mid >= z_obs
    kSZ_integrand_seed = kSZ_integrand_all[seed]
    n_integrand_slices = kSZ_integrand_seed.shape[2]   # N_ind_z - 1 after midpoint

    # midpoint integrand
    kSZ_int_mid = 0.5 * (kSZ_integrand_seed[:, :, :-1]
                         + kSZ_integrand_seed[:, :,  1:])
    # kSZ_int_mid shape: (HII_DIM, HII_DIM, N_ind_z - 1)

    # z_mid is on the full lightcone grid — take the first N slices
    n_mid     = kSZ_int_mid.shape[2]
    z_mid_int = z_mid[:n_mid]
    ds_cm_int = ds_cm[:n_mid]
    a_sq_int  = a_sq_mid[:n_mid]

    # integration range mask
    idx_integrate = np.where(z_mid_int >= z_obs)[0]

    # ── full LoS integrand with prefactor ────────────────────────────────
    ds_cm_sel = ds_cm_int[idx_integrate]
    a_sq_sel  = a_sq_int [idx_integrate]

    kSZ_int_full = ((prefactor_cgs / a_sq_sel[None, None, :])
                    * kSZ_int_mid[:, :, idx_integrate]
                    * (ds_cm_sel / c_cm_s)[None, None, :])

    t0      = time.time()
    kSZ_map = np.sum(kSZ_int_full, axis=2)
    dt      = time.time() - t0

    np.save(map_cache, kSZ_map)
    kSZ_map_all[seed] = kSZ_map

    rms = np.sqrt(np.mean(kSZ_map**2))
    print(f"  ✓ seed {seed:3d}: computed  shape={kSZ_map.shape}  RMS={rms:.4e}  "
          f"z={z_mid_int[idx_integrate].max():.2f}→"
          f"{z_mid_int[idx_integrate].min():.2f} "
          f"({len(idx_integrate)} slices, {dt:.2f}s)  → cached")

# =============================================================================
# DIAGNOSTIC PLOT — kSZ + kSZ² maps side by side  (PLOT_SEED only)
# =============================================================================

if PLOT_SEED not in kSZ_map_all:
    raise RuntimeError(f"PLOT_SEED={PLOT_SEED} has no kSZ map — cannot plot.")

print(f"\n  Diagnostic plot (seed {PLOT_SEED})")

kSZ_map       = kSZ_map_all[PLOT_SEED]
kSZ2_map      = kSZ_map**2
kSZ2_centered = kSZ2_map - kSZ2_map.mean()

BOX_LEN = float(inputs.simulation_options.BOX_LEN)
extent  = [0, BOX_LEN, 0, BOX_LEN]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

vmax = float(np.percentile(np.abs(kSZ_map), 99))
im0  = axes[0].imshow(kSZ_map.T, origin='lower', extent=extent,
                      cmap='seismic', vmin=-vmax, vmax=vmax)
plt.colorbar(im0, ax=axes[0],
             label=r'$\Delta T_{\rm kSZ}$  [dimensionless]',
             fraction=0.046, pad=0.04)
axes[0].set_xlabel('x  [cMpc]')
axes[0].set_ylabel('y  [cMpc]')
axes[0].set_title(
    rf'kSZ map (seed {PLOT_SEED})  '
    r'$\int(1+\delta)\,x_e\,v_{\rm los}/c\;e^{-\tau}\,d\chi$',
    fontweight='bold')

v2  = float(np.percentile(np.abs(kSZ2_centered), 99))
im1 = axes[1].imshow(kSZ2_centered.T, origin='lower', extent=extent,
                     cmap='seismic', vmin=-v2, vmax=v2)
plt.colorbar(im1, ax=axes[1],
             label=r'$(\Delta T_{\rm kSZ})^2 - \langle(\Delta T)^2\rangle$',
             fraction=0.046, pad=0.04)
axes[1].set_xlabel('x  [cMpc]')
axes[1].set_ylabel('y  [cMpc]')
axes[1].set_title(rf' (mean-subtracted, seed {PLOT_SEED})',
                  fontweight='bold')

fname = f"kSZ_map_seed{PLOT_SEED}"
fig.savefig(f"{plot_dir}/{fname}.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{plot_dir}/{fname}.pdf", bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: {fname}.png / .pdf")

# =============================================================================
# SUMMARY
# =============================================================================

rms_vals  = np.array([
    np.sqrt(np.mean(kSZ_map_all[s]**2)) for s in kSZ_map_all
])
rms2_vals = np.array([
    np.sqrt(np.mean((kSZ_map_all[s]**2)**2)) for s in kSZ_map_all
])

print(f"\n{'='*70}")
print(f"CELL 6 COMPLETE")
print(f"{'='*70}")
print(f"  kSZ_map_all   : {len(kSZ_map_all)}/{N_SEEDS} seeds  "
      f"shape={kSZ_map_all[PLOT_SEED].shape}")
print(f"  kSZ  RMS  across seeds : mean={rms_vals.mean():.4e} ± "
      f"{rms_vals.std():.4e}")
print(f"  kSZ² RMS  across seeds : mean={rms2_vals.mean():.4e} ± "
      f"{rms2_vals.std():.4e}")
print(f"  ready for Cell 7 — kSZ² × halo cross-correlation (parallelised)")
print(f"{'='*70}")

# %%
# %%
# =============================================================================
# CELL 6b: Individual Slide-Ready Maps
#   - kSZ map  @ z_obs = 5
#   - kSZ² map @ z_obs = 5  (mean-subtracted)
#   - Halo-over-density maps at several redshifts (boosted contrast)
#
# Each map saved as its own PNG/PDF — no panels.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print("\n" + "="*70)
print("CELL 6b — INDIVIDUAL SLIDE MAPS")
print("="*70)

# -----------------------------------------------------------------------------
# Rebind single-seed aliases from per-seed dicts (Cell 2 outputs).
# Makes 6b independent of Cell 3 having been re-run in this session.
# -----------------------------------------------------------------------------
DIAG_SEED    = RANDOM_SEEDS[0]
halo_mass_lc = halo_mass_lc_all[DIAG_SEED]
density_lc   = density_lc_all  [DIAG_SEED]
lightcone    = lightcones      [DIAG_SEED]
HALO_OUT     = os.path.join(halo_dir, f"seed_{DIAG_SEED}")

# Rebuild find_node_for_lc_idx in case Cell 3 wasn't run this session
node_redshifts_sorted = sorted(inputs.node_redshifts)
lc_distances = np.array(lightcone.lightcone_distances.to_value('Mpc'),
                        dtype=np.float32)

def find_node_for_lc_idx(z_idx):
    target_dc = lc_distances[z_idx]
    return min(node_redshifts_sorted,
               key=lambda z: abs(cosmo.comoving_distance(z).to_value('Mpc')
                                 - target_dc))

# -----------------------------------------------------------------------------
# Output dir
# -----------------------------------------------------------------------------
slide_dir = f"{plot_dir}/slides_maps"
os.makedirs(slide_dir, exist_ok=True)

res_str = f"{HII_DIM}$^3$ cells, {BOX_LEN:.0f} cMpc box"
extent  = [0, BOX_LEN, 0, BOX_LEN]

def _save(fig, fname):
    fig.savefig(f"{slide_dir}/{fname}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{slide_dir}/{fname}.pdf",            bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}.png / .pdf")

# =============================================================================
# 1) kSZ map @ z_obs = 5
# =============================================================================
kSZ_map       = kSZ_map_all[PLOT_SEED]
kSZ2_map      = kSZ_map ** 2
kSZ2_centered = kSZ2_map - kSZ2_map.mean()

fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
vmax = float(np.percentile(np.abs(kSZ_map), 99))
im = ax.imshow(kSZ_map.T, origin='lower', extent=extent,
               cmap='seismic', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax,
             label=r'$\Delta T_{\rm kSZ}/T_{\rm kSZ}$',
             fraction=0.046, pad=0.04)
ax.set_xlabel('x  [cMpc]')
ax.set_ylabel('y  [cMpc]')
ax.set_title(f'$z_{{\\rm obs}}=5$ ')
_save(fig, "slide_kSZ_map_z5")

# =============================================================================
# 2) kSZ² map @ z_obs = 5  (mean-subtracted)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
v2 = float(np.percentile(np.abs(kSZ2_centered), 99))
im = ax.imshow(kSZ2_centered.T, origin='lower', extent=extent,
               cmap='seismic', vmin=-v2, vmax=v2)
plt.colorbar(im, ax=ax,
             label=r'$(\Delta T_{\rm kSZ}/T_{\rm kSZ})^2 - \langle\cdot\rangle$',
             fraction=0.046, pad=0.04)
ax.set_xlabel('x  [cMpc]')
ax.set_ylabel('y  [cMpc]')
ax.set_title(f'$z_{{\\rm obs}}=5$  ({res_str})')
_save(fig, "slide_kSZ2_map_z5")

# =============================================================================
# 3) Halo maps over density — several redshifts, individual figures
#    ALL halos in slab, unified mass colorscale across all z.
# =============================================================================
halo_exists    = np.array([np.any(np.isfinite(halo_mass_lc[:, :, i]))
                           for i in range(n_lc)])
idx_with_halos = np.where(halo_exists)[0]

if len(idx_with_halos) > 6:
    idx_pick = idx_with_halos[np.linspace(0, len(idx_with_halos)-1, 6, dtype=int)]
else:
    idx_pick = idx_with_halos

# Tighter density range → more pixels saturate to black/white → punchier
DENS_VMIN, DENS_VMAX = 0.5, 2.5

# Unified halo mass range across all redshifts (log10 M_sun)
HALO_LOGM_MIN, HALO_LOGM_MAX = 8.5, 10.5

for z_idx in idx_pick:
    z_node = float(z_lc[z_idx])
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    # Density background
    dens_slice = density_lc[:, :, z_idx].T
    im_d = ax.imshow(1 + dens_slice, origin='lower', extent=extent,
                     cmap='Greys',
                     norm=mcolors.LogNorm(vmin=DENS_VMIN, vmax=DENS_VMAX),
                     interpolation='bilinear')
    cb_d = plt.colorbar(im_d, ax=ax, label=r'$1+\delta$',
                        fraction=0.046, pad=0.04)

    # ALL halos overlay (unified mass scale across z)
    z_node_match = find_node_for_lc_idx(z_idx)
    tag          = f"z{z_node_match:.4f}"
    masses_path  = os.path.join(HALO_OUT, f"masses_{tag}.npy")
    coords_path  = os.path.join(HALO_OUT, f"coords_{tag}.npy")

    n_halos = 0
    if os.path.exists(masses_path):
        m_slab  = np.load(masses_path)
        c_slab  = np.load(coords_path)
        n_halos = len(m_slab)
        if n_halos > 0:
            sc = ax.scatter(c_slab[:, 0], c_slab[:, 1],
                            c=np.log10(m_slab),
                            s=4, cmap='plasma', alpha=0.6,
                            vmin=HALO_LOGM_MIN, vmax=HALO_LOGM_MAX,
                            edgecolors='none')
            cb_h = plt.colorbar(sc, ax=ax,
                                label=r'$\log_{10}(M_{\rm halo}/M_\odot)$',
                                fraction=0.046, pad=0.04)

    ax.set_xlim(0, BOX_LEN)
    ax.set_ylim(0, BOX_LEN)
    ax.set_xlabel('x  [cMpc]')
    ax.set_ylabel('y  [cMpc]')
    ax.set_title(f'$z={z_node:.2f}$  ')

    _save(fig, f"slide_halo_density_z{z_node:.2f}")

print(f"\n  Saved {2 + len(idx_pick)} maps to:")
print(f"    {os.path.abspath(slide_dir)}")
print("="*70)

# %%
# =============================================================================
# INVALIDATE STALE CROSS-CORRELATION CACHE
# Old caches were built from halo_count_lc (number overdensity).
# We now want mass overdensity, so force recomputation.
# =============================================================================

import glob
stale = glob.glob(f"{cache_dir}/seed_*/kSZ2_halo_cross_seed*.npy")
for f in stale:
    os.remove(f)
print(f"  Cleared {len(stale)} stale cross-correlation cache files")

# %%
# %%
# =============================================================================
# CELL 7: kSZ²–Halo Cross-Correlation Power Spectra — ALL SEEDS (PARALLELISED)
#
# Inherited from Cell 1 : inputs, cache_dir, RANDOM_SEEDS, N_SEEDS
# Inherited from Cell 2 : lightcones, halo_count_lc_all,
#                         N_TOTAL_CORES, DESIRED_THREADS_PER_WORKER
# Inherited from Cell 6 : kSZ_map_all
# Produces              : halo_cross_results_all  (dict keyed by seed)
# =============================================================================

import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# IMPORT THE WORKER FROM THE SEPARATE MODULE
from halo_cross_corr_worker import compute_cross_corr_for_seed_halo

print("\n" + "="*70)
print(f"CELL 7 — kSZ²–HALO CROSS-CORRELATION  ({N_SEEDS} seeds, PARALLELISED)")
print("="*70)

# =============================================================================
# MAP PROPERTIES (seed-independent)
# =============================================================================

npix_side    = int(inputs.simulation_options.HII_DIM)
box_size_Mpc = float(inputs.simulation_options.BOX_LEN)
pix_size_Mpc = box_size_Mpc / npix_side
pix_area     = pix_size_Mpc**2

print(f"\n  Map      : {npix_side}² pixels")
print(f"  Box      : {box_size_Mpc:.1f} Mpc")
print(f"  Pixel    : {pix_size_Mpc:.3f} Mpc")

# =============================================================================
# k-SPACE GRID (seed-independent)
# =============================================================================

dk        = 2 * np.pi / (npix_side * pix_size_Mpc)
kx        = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
ky        = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
kgrid     = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
k_bins    = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), 35)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

print(f"  dk       : {dk:.6f} Mpc⁻¹")
print(f"  k range  : [{kgrid.min():.4f}, {kgrid.max():.4f}] Mpc⁻¹")
print(f"  N bins   : {len(k_centers)}")

# =============================================================================
# SAFETY NET — load halo_count_lc_all / kSZ_map_all from cache if missing
# =============================================================================

z_obs = 5.0   # matches CELL 6

if ('halo_count_lc_all' not in dir()
        and 'halo_count_lc_all' not in globals()) \
   or len(halo_count_lc_all) == 0:
    print("\n  halo_count_lc_all not in memory → loading from CELL 2 cache")
    halo_count_lc_all = {}
    for seed in RANDOM_SEEDS:
        halos_cache = os.path.join(cache_dir, f"seed_{seed}", "halo_arrays.npz")
        if os.path.exists(halos_cache):
            halo_count_lc_all[seed] = np.load(halos_cache)['halo_count_lc']

if ('kSZ_map_all' not in dir()
        and 'kSZ_map_all' not in globals()) \
   or len(kSZ_map_all) == 0:
    print("  kSZ_map_all not in memory → loading from CELL 6 cache")
    kSZ_map_all = {}
    for seed in RANDOM_SEEDS:
        map_cache = f"{cache_dir}/kSZ_maps/kSZ_map_z{z_obs:.1f}_seed{seed}.npy"
        if os.path.exists(map_cache):
            kSZ_map_all[seed] = np.load(map_cache)

# lc_redshifts is seed-independent — take from first available lightcone
ref_seed         = next(iter(lightcones))
lc_redshifts_arr = np.array(lightcones[ref_seed].lightcone_redshifts,
                            dtype=np.float64)

# =============================================================================
# BUILD MASS-WEIGHTED HALO FIELD  (replaces count field as the tracer)
#
# halo_mass_lc  = mean halo mass per cell  (NaN where no halos)
# halo_count_lc = halo count per cell
# total mass per cell = mean × count, with NaN → 0 in empty cells
# =============================================================================

if ('halo_mass_lc_all' not in dir()
        and 'halo_mass_lc_all' not in globals()) \
   or len(halo_mass_lc_all) == 0:
    print("\n  halo_mass_lc_all not in memory → loading from CELL 2 cache")
    halo_mass_lc_all = {}
    for seed in RANDOM_SEEDS:
        halos_cache = os.path.join(cache_dir, f"seed_{seed}", "halo_arrays.npz")
        if os.path.exists(halos_cache):
            halo_mass_lc_all[seed] = np.load(halos_cache)['halo_mass_lc']

halo_mass_field_all = {}
for seed in RANDOM_SEEDS:
    if seed not in halo_mass_lc_all or seed not in halo_count_lc_all:
        continue
    mean_mass = np.nan_to_num(halo_mass_lc_all[seed], nan=0.0)   # M_sun
    count     = halo_count_lc_all[seed]                          # number
    halo_mass_field_all[seed] = (mean_mass * count).astype(np.float32)  # M_sun

print(f"\n  Built mass-weighted halo field for "
      f"{len(halo_mass_field_all)} seeds  "
      f"(total mass per cell, M_sun)")

# =============================================================================
# BUILD WORKER ARGUMENTS
# =============================================================================

worker_args = []
for seed in RANDOM_SEEDS:
    if seed not in halo_mass_field_all or seed not in kSZ_map_all:
        print(f"  ✗ seed {seed}: missing inputs — skipping")
        continue

    seed_cache_dir = os.path.join(cache_dir, f"seed_{seed}")
    os.makedirs(seed_cache_dir, exist_ok=True)

    args = (
        seed,
        halo_mass_field_all[seed], kSZ_map_all[seed],   # ← was halo_count_lc_all[seed]
        lc_redshifts_arr,
        seed_cache_dir,
        npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
        dk, kgrid, k_bins, k_centers,
    )
    worker_args.append(args)

# =============================================================================
# CORE / WORKER ALLOCATION
# =============================================================================

N_WORKERS = max(1, N_TOTAL_CORES // DESIRED_THREADS_PER_WORKER)
N_WORKERS = min(N_WORKERS, len(worker_args))

print(f"\n  Available cores    : {N_TOTAL_CORES}")
print(f"  Workers            : {N_WORKERS} (concurrent seeds)")
print(f"  Threads per worker : {DESIRED_THREADS_PER_WORKER}")

# =============================================================================
# DISPATCH — concurrent seeds via ProcessPoolExecutor with spawn context
# =============================================================================

print(f"\n{'='*70}")
print(f"DISPATCHING {len(worker_args)} SEEDS — {N_WORKERS} concurrent workers")
print(f"{'='*70}", flush=True)

mp_ctx                 = mp.get_context("spawn")
halo_cross_results_all = {}
scan_start             = time.time()

with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as ex:
    futures = {ex.submit(compute_cross_corr_for_seed_halo, args): args[0]
               for args in worker_args}

    completed = 0
    for fut in as_completed(futures):
        seed = futures[fut]
        try:
            seed_done, hcr_dict, status_msg = fut.result()
            completed += 1
            halo_cross_results_all[seed_done] = hcr_dict
            elapsed = (time.time() - scan_start) / 60
            print(f"  [{completed:2d}/{len(worker_args)}] seed {seed_done:3d}: "
                  f"✓ {status_msg}   (elapsed: {elapsed:.1f} min)",
                  flush=True)
        except Exception as e:
            completed += 1
            elapsed = (time.time() - scan_start) / 60
            print(f"  [{completed:2d}/{len(worker_args)}] seed {seed:3d}: "
                  f"✗ crashed: {e}   (elapsed: {elapsed:.1f} min)",
                  flush=True)

print(f"\n{'='*70}")
print(f"✓ ALL WORKERS RETURNED  —  {time.time()-scan_start:.1f} s total")
print(f"{'='*70}")

# =============================================================================
# ERROR-BUDGET SUMMARY (averaged across seeds × redshifts × k-bins)
# =============================================================================

print(f"\n=== ERROR BUDGET SUMMARY (averaged across seeds) ===")

all_sample_err = []
all_cosmic_err = []
all_total_err  = []

for seed, hcr in halo_cross_results_all.items():
    for z_halo, res in hcr.items():
        C     = res['C_cross_1d']
        valid = ~np.isnan(C) & (C != 0)
        if np.any(valid):
            all_sample_err.extend(
                (res['C_cross_err_sample'][valid]
                 / np.abs(C[valid])).tolist())
            all_cosmic_err.extend(
                (res['C_cross_err_cosmic'][valid]
                 / np.abs(C[valid])).tolist())
            all_total_err.extend(
                (res['C_cross_err_total'][valid]
                 / np.abs(C[valid])).tolist())

if len(all_sample_err) > 0:
    print(f"  Sample variance (mean): "
          f"{np.nanmean(all_sample_err)*100:.1f}%")
    print(f"  Cosmic variance (mean): "
          f"{np.nanmean(all_cosmic_err)*100:.1f}%")
    print(f"  Total  variance (mean): "
          f"{np.nanmean(all_total_err)*100:.1f}%")

# =============================================================================
# SUMMARY
# =============================================================================

n_redshifts = [len(halo_cross_results_all[s]) for s in halo_cross_results_all]

print(f"\n{'='*70}")
print(f"CELL 7 COMPLETE")
print(f"{'='*70}")
print(f"  halo_cross_results_all : {len(halo_cross_results_all)}/{N_SEEDS} seeds")
print(f"  Redshifts per seed     : "
      f"min={min(n_redshifts)}, max={max(n_redshifts)}, mean={np.mean(n_redshifts):.0f}")
print(f"  ready for Cell 8 — k→ℓ conversion and seed-averaged D_ℓ plots")
print(f"{'='*70}")

# %%
# %%
# =============================================================================
# CELL 8: Visualize kSZ²–Halo Cross-Correlation — ALL SEEDS (mean ± seed-std)
#
# Inherited from Cell 1 : inputs, cosmo, plot_dir, RANDOM_SEEDS, N_SEEDS
# Inherited from Cell 2 : lightcones (for reionisation, via Cell 4)
# Inherited from Cell 4 : xe_mean, z_common_xe (seed-averaged reionisation)
# Inherited from Cell 7 : halo_cross_results_all
# =============================================================================

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

print("\n" + "="*70)
print(f"CELL 8 — kSZ²–HALO CROSS-CORRELATION VISUALISATION  ({N_SEEDS} seeds)")
print("="*70)

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

plot_dir_save = f"{plot_dir}/plot_final_cell"
os.makedirs(plot_dir_save, exist_ok=True)
print(f"  Output : {os.path.abspath(plot_dir_save)}")

# =============================================================================
# LOAD halo_cross_results_all FROM CACHE IF NEEDED
# =============================================================================

if ('halo_cross_results_all' not in dir()
        and 'halo_cross_results_all' not in globals()) \
   or len(halo_cross_results_all) == 0:
    print("  halo_cross_results_all not in memory → loading from CELL 7 cache")
    halo_cross_results_all = {}
    for seed in RANDOM_SEEDS:
        cc_cache = os.path.join(cache_dir, f"seed_{seed}",
                                f"kSZ2_halo_cross_seed{seed}.npy")
        if os.path.exists(cc_cache):
            halo_cross_results_all[seed] = np.load(cc_cache,
                                                   allow_pickle=True).item()
            print(f"    ✓ seed {seed}: {len(halo_cross_results_all[seed])} z")
else:
    print(f"  Using in-memory results "
          f"({len(halo_cross_results_all)} seeds)")

# =============================================================================
# k → ℓ CONVERSION (per seed)
# =============================================================================

print("\n  Converting k → ℓ for each seed ...")

h_little           = 0.6766
cross_corr_ell_all = {}

for seed, hcr in halo_cross_results_all.items():
    ccr_ell = {}
    for z_obs in sorted(hcr.keys()):
        res              = hcr[z_obs]
        D_A_Mpc          = float(cosmo.angular_diameter_distance(z_obs).value)
        chi_comoving_Mpc = float(cosmo.comoving_distance(z_obs).value)
        k_centers        = res['k_centers']

        ell_from_k       = k_centers * chi_comoving_Mpc / h_little
        scale            = h_little**2 / D_A_Mpc**2

        C_cross_ell      = res['C_cross_1d']         * scale
        C_cross_ell_err  = res['C_cross_err_total']  * scale

        pref             = ell_from_k * (ell_from_k + 1) / (2 * np.pi)
        D_cross_ell      = pref * C_cross_ell
        D_cross_ell_err  = pref * C_cross_ell_err

        ccr_ell[z_obs] = {
            'ell_from_k'     : ell_from_k,
            'D_cross_ell'    : D_cross_ell,
            'D_cross_ell_err': D_cross_ell_err,
            'r_cross'        : res['r_cross'],
        }
    cross_corr_ell_all[seed] = ccr_ell

print(f"  ✓ Converted {len(cross_corr_ell_all)} seeds")

# =============================================================================
# SEED-AVERAGING AT EACH REDSHIFT
# total error = sqrt( seed-to-seed std² + mean measurement error² )
# =============================================================================

print("\n  Averaging across seeds ...")

# Union of all z keys (typically all seeds share the same z grid)
all_z = sorted(set().union(*[s.keys()
                              for s in cross_corr_ell_all.values()]))

cross_corr_ell_averaged = {}

for z_obs in all_z:
    D_seeds     = []
    D_err_seeds = []
    r_seeds     = []
    ell_ref     = None

    for seed, ell_res in cross_corr_ell_all.items():
        if z_obs not in ell_res:
            continue
        res = ell_res[z_obs]
        if ell_ref is None:
            ell_ref = res['ell_from_k']
        D_seeds    .append(res['D_cross_ell'])
        D_err_seeds.append(res['D_cross_ell_err'])
        r_seeds    .append(res['r_cross'])

    if len(D_seeds) == 0:
        continue

    D_matrix     = np.array(D_seeds)
    D_err_matrix = np.array(D_err_seeds)
    r_matrix     = np.array(r_seeds)

    D_mean          = np.nanmean(D_matrix,     axis=0)
    if D_matrix.shape[0] >= 2:
        D_std_seeds = np.nanstd (D_matrix, ddof=1, axis=0)
    else:
        D_std_seeds = np.zeros_like(D_mean)
    D_err_meas_mean = np.nanmean(D_err_matrix, axis=0)
    D_err_total     = np.sqrt(D_std_seeds**2 + D_err_meas_mean**2)

    r_mean = np.nanmean(r_matrix, axis=0)

    cross_corr_ell_averaged[z_obs] = {
        'ell_from_k'     : ell_ref,
        'D_mean'         : D_mean,
        'D_std_seeds'    : D_std_seeds,
        'D_err_meas_mean': D_err_meas_mean,
        'D_err_total'    : D_err_total,
        'r_mean'         : r_mean,
        'n_seeds'        : len(D_seeds),
    }

print(f"  ✓ Averaged {len(cross_corr_ell_averaged)} redshifts")

# =============================================================================
# REIONISATION HISTORY (from CELL 4 seed-averaged outputs)
# =============================================================================

def z_at_xe(xe_val):
    # xe_mean / z_common_xe are seed-averaged from CELL 4
    # x_e decreases as z increases → reverse for np.interp
    return float(np.interp(xe_val, xe_mean[::-1], z_common_xe[::-1]))

# =============================================================================
# SHARED PLOT SETTINGS
# =============================================================================

all_z_arr   = np.array(sorted(cross_corr_ell_averaged.keys()))
cmap        = mpl.cm.rainbow
norm        = mpl.colors.Normalize(vmin=all_z_arr.min(), vmax=all_z_arr.max())

ell_targets = [500, 1000, 3000]
colors_ell  = ['darkblue', 'darkgreen', 'darkred']

# =============================================================================
# PLOT 1: Rainbow D_ℓ vs ℓ  (seed-averaged, ±1σ_total band)
# =============================================================================

print("\n=== PLOT 1: Rainbow D_ℓ vs ℓ (seed-averaged) ===")

fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)

for z_obs in all_z_arr[::2]:
    res = cross_corr_ell_averaged[z_obs]
    ell = res['ell_from_k']
    D   = res['D_mean']
    E   = res['D_err_total']
    valid = np.isfinite(D) & np.isfinite(E) & (ell > 10)
    if valid.sum() > 5:
        color = cmap(norm(z_obs))
        ax.plot(ell[valid], D[valid], color=color, lw=1.5, alpha=0.85)
        ax.fill_between(ell[valid],
                        D[valid] - E[valid],
                        D[valid] + E[valid],
                        color=color, alpha=0.15)

ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel(r'Multipole $\ell$')
ax.set_ylabel(r'$D_\ell^{\,\mathrm{kSZ}^2 \times \delta_{M_h}}$ ')#ax.set_ylabel(r'$D_\ell^{\rm kSZ^2 \times h}$  [dimensionless]')
ax.set_title(rf'kSZ$^2$–Halo Cross-Power $D_\ell$ vs Redshift  '
             rf'({N_SEEDS} seeds, mean ± total)',
             fontweight='bold')
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
plt.colorbar(sm, ax=ax, pad=0.02).set_label(r'Redshift $z$')

ax.text(0.02, 0.02,
        f'{N_SEEDS} seeds\n'
        r'$\sigma_{\rm tot} = \sqrt{\sigma_{\rm seeds}^2 + \sigma_{\rm meas}^2}$',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fname = "kSZ2_halo_Dl_vs_ell_rainbow"
fig.savefig(f"{plot_dir_save}/{fname}.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{plot_dir_save}/{fname}.pdf", bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: {fname}.png / .pdf")

# =============================================================================
# PLOTS 2a–2f: D_ℓ vs z at fixed ℓ  (one plot per ℓ × {with errors, no errors})
# =============================================================================

print("\n=== PLOTS 2a–2f: D_ℓ vs z at fixed ℓ "
      "(6 plots: 3 ℓ × {with errors, no errors}) ===")

def plot_Dl_vs_z(ell_target, color, with_errors):
    """
    Make one D_ℓ vs z plot at a fixed multipole, with or without error bars.
    Both PNG and PDF are saved.
    """
    z_plot, D_plot, err_plot = [], [], []
    for z_obs in all_z_arr:
        res = cross_corr_ell_averaged[z_obs]
        ell = res['ell_from_k']
        D   = res['D_mean']
        E   = res['D_err_total']
        idx = int(np.argmin(np.abs(ell - ell_target)))
        if np.isfinite(D[idx]) and np.isfinite(E[idx]):
            z_plot  .append(z_obs)
            D_plot  .append(D[idx])
            err_plot.append(E[idx])

    if len(z_plot) <= 2:
        print(f"    ✗ ℓ={ell_target}: too few valid z — skipping")
        return None

    z_plot   = np.array(z_plot)
    D_plot   = np.array(D_plot)
    err_plot = np.array(err_plot)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

    if with_errors:
        ax.errorbar(z_plot, D_plot, yerr=err_plot,
                    color=color, lw=2.5, marker='o', markersize=6,
                    capsize=4, alpha=0.85,
                    label=rf'$\ell={ell_target}$  (mean ± $\sigma_{{\rm tot}}$)')
        ax.fill_between(z_plot, D_plot - err_plot, D_plot + err_plot,
                        color=color, alpha=0.2)
        err_tag      = "with errors"
        fname_suffix = "with_errors"
    else:
        ax.plot(z_plot, D_plot,
                color=color, lw=2.5, marker='o', markersize=6, alpha=0.9,
                label=rf'$\ell={ell_target}$  (mean, {N_SEEDS} seeds)')
        err_tag      = "no errors"
        fname_suffix = "no_errors"

    ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'$D_\ell^{\,\mathrm{kSZ}^2 \times \delta_{M_h}}$')#ax.set_ylabel(r'$D_\ell^{\rm kSZ^2 \times h}$  [dimensionless]')
    ax.set_yscale('symlog', linthresh=1e-12)
    ax.set_title(rf'kSZ$^2$–Halo Cross-Power Evolution  '
                 rf'($\ell={ell_target}$, {err_tag}, {N_SEEDS} seeds)',
                 fontweight='bold')
    ax.invert_xaxis()
    ax.legend(loc='best')

    fname = f"kSZ2_halo_Dl_vs_z_ell{ell_target}_{fname_suffix}"
    fig.savefig(f"{plot_dir_save}/{fname}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{plot_dir_save}/{fname}.pdf", bbox_inches='tight')
    plt.close(fig)
    return fname

for ell_target, color in zip(ell_targets, colors_ell):
    for with_errors in (True, False):
        fname = plot_Dl_vs_z(ell_target, color, with_errors)
        if fname is not None:
            print(f"  ✓ Saved: {fname}.png / .pdf")

# =============================================================================
# PLOT 3: r vs z at fixed ℓ  (3 ℓ overlaid, seed-averaged)
# =============================================================================

print("\n=== PLOT 3: r vs z at fixed ℓ (seed-averaged) ===")

fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

for ell_target, color in zip(ell_targets, colors_ell):
    z_plot, r_plot = [], []
    for z_obs in all_z_arr:
        res = cross_corr_ell_averaged[z_obs]
        ell = res['ell_from_k']
        r   = res['r_mean']
        idx = int(np.argmin(np.abs(ell - ell_target)))
        if np.isfinite(r[idx]) and np.abs(r[idx]) < 1.5:
            z_plot.append(z_obs)
            r_plot.append(r[idx])
    if len(z_plot) > 2:
        ax.plot(np.array(z_plot), np.array(r_plot),
                color=color, lw=2.5, marker='o', markersize=5,
                alpha=0.85, label=rf'$\ell={ell_target}$')

# reionisation phase markers (seed-averaged from CELL 4)
for xe_val, ls in zip([0.2, 0.5, 0.9], [':', '--', ':']):
    z_m = z_at_xe(xe_val)
    ax.axvline(z_m, color='gray', ls=ls, lw=1, alpha=0.6)
    ax.text(z_m + 0.05, 1.05, fr'$x_e={xe_val}$', fontsize=10, color='gray')

ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'Correlation Coefficient $r$')
ax.set_title(rf'kSZ$^2$–Halo Correlation Coefficient  '
             rf'({N_SEEDS} seeds, mean)',
             fontweight='bold')
ax.invert_xaxis()
ax.legend(loc='best')

fname = "kSZ2_halo_r_vs_z"
fig.savefig(f"{plot_dir_save}/{fname}.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{plot_dir_save}/{fname}.pdf", bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: {fname}.png / .pdf")

# =============================================================================
# PLOT 4: r vs ℓ rainbow (seed-averaged)
# =============================================================================

print("\n=== PLOT 4: r vs ℓ rainbow (seed-averaged) ===")

fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)

for z_obs in all_z_arr:
    res   = cross_corr_ell_averaged[z_obs]
    ell   = res['ell_from_k']
    r     = res['r_mean']
    valid = np.isfinite(r) & (ell > 10) & (np.abs(r) < 1.5)
    if valid.sum() > 5:
        ax.plot(ell[valid], r[valid],
                color=cmap(norm(z_obs)), lw=1.5, alpha=0.75)

ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
ax.set_xscale('log')
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel(r'Multipole $\ell$')
ax.set_ylabel(r'Correlation Coefficient $r$')
ax.set_title(rf'kSZ$^2$–Halo Correlation Coefficient vs $\ell$  '
             rf'({N_SEEDS} seeds, mean)',
             fontweight='bold')
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
plt.colorbar(sm, ax=ax, pad=0.02).set_label(r'Redshift $z$')

fname = "kSZ2_halo_r_vs_ell_rainbow"
fig.savefig(f"{plot_dir_save}/{fname}.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{plot_dir_save}/{fname}.pdf", bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Saved: {fname}.png / .pdf")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print(f"CELL 7 COMPLETE  ({N_SEEDS} seeds averaged)")
print(f"{'='*70}")
print(f"  Plots saved to: {os.path.abspath(plot_dir_save)}")
print(f"\n  D_ℓ vs ℓ rainbow:")
print(f"    kSZ2_halo_Dl_vs_ell_rainbow.png / .pdf")
print(f"\n  D_ℓ vs z (6 plots — one per ℓ × {{with, no}} errors):")
for ell_target in ell_targets:
    for sfx in ("with_errors", "no_errors"):
        print(f"    kSZ2_halo_Dl_vs_z_ell{ell_target}_{sfx}.png / .pdf")
print(f"\n  Correlation coefficient:")
print(f"    kSZ2_halo_r_vs_z.png / .pdf")
print(f"    kSZ2_halo_r_vs_ell_rainbow.png / .pdf")
print(f"{'='*70}")


