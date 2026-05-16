# =============================================================================
# halo_lightcone_worker.py
#
# Per-seed worker for kSZ²-halo CELL 2: lightcone + field arrays + halo
# catalogues + per-slab raw halo catalogues.
#
# Spawned by ProcessPoolExecutor with mp.get_context("spawn") — must be a
# top-level module function so spawn workers can import it.
#
# Logic is a verbatim port of the original (single-seed) CELL 2 STEP 1/2/3
# wrapped in run_or_load_seed_halo(args).  Physics, slab filter, binning,
# transposes, and savez keys are unchanged.
# =============================================================================

import os
import time
import numpy as np
import py21cmfast as p21c
from astropy.cosmology import FlatLambdaCDM
from astropy.units import pixel


def run_or_load_seed_halo(args):
    """
    Run (or load from cache) the full CELL 2 pipeline for one seed.

    Caches
    ------
    seed_cache_dir/lightcone.h5
    seed_cache_dir/field_arrays.npz
    seed_cache_dir/halo_arrays.npz
    halo_out_seed_dir/masses_z*.npy, coords_z*.npy

    Returns
    -------
    (seed, seed_cache_dir, halo_out_seed_dir, n_lc, sim_time, status_str)
    """
    (seed, seed_cache_dir, halo_out_seed_dir,
     z_min, z_max, z_step_factor,
     HII_DIM, BOX_LEN, N_THREADS, Z_HEAT_MAX,
     SAMPLER_MIN_MASS, SAMPLER_BUFFER_FACTOR,
     MASS_CUT) = args

    os.makedirs(seed_cache_dir,    exist_ok=True)
    os.makedirs(halo_out_seed_dir, exist_ok=True)

    # Each spawn worker is a fresh interpreter — safe to set per-seed cache dir
    p21c.config['direc'] = os.path.abspath(seed_cache_dir)

    lightcone_cache = os.path.join(seed_cache_dir, "lightcone.h5")
    fields_cache    = os.path.join(seed_cache_dir, "field_arrays.npz")
    halos_cache     = os.path.join(seed_cache_dir, "halo_arrays.npz")

    sim_time     = 0.0
    status_parts = []

    # ── Per-seed InputParameters ─────────────────────────────────────────
    node_redshifts_custom = np.array(
        p21c.get_logspaced_redshifts(
            min_redshift  = z_min,
            max_redshift  = z_max,
            z_step_factor = z_step_factor,
        )
    )
    inputs = p21c.InputParameters(
        node_redshifts     = node_redshifts_custom,
        random_seed        = seed,
        simulation_options = p21c.SimulationOptions(
            HII_DIM               = HII_DIM,
            BOX_LEN               = BOX_LEN,
            N_THREADS             = N_THREADS,
            Z_HEAT_MAX            = Z_HEAT_MAX,
            SAMPLER_MIN_MASS      = SAMPLER_MIN_MASS,
            SAMPLER_BUFFER_FACTOR = SAMPLER_BUFFER_FACTOR,
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

    cell_size_mpc = BOX_LEN / HII_DIM
    cosmo         = FlatLambdaCDM(H0=67.77, Om0=0.3086)

    lightconer = p21c.RectilinearLightconer.between_redshifts(
        min_redshift = min(inputs.node_redshifts) + 0.1,
        max_redshift = max(inputs.node_redshifts) - 0.1,
        quantities   = (
            "brightness_temp",
            "density",
            "neutral_fraction",
            "kinetic_temperature",
            "velocity_z",
        ),
        resolution   = inputs.simulation_options.cell_size,
    )

    # =====================================================================
    # STEP 1: LIGHTCONE
    # =====================================================================
    if os.path.exists(lightcone_cache):
        lightcone = p21c.LightCone.from_file(lightcone_cache, safe=False)
        status_parts.append("lc:cached")
    else:
        t0 = time.time()
        lightcone = p21c.run_lightcone(
            inputs     = inputs,
            lightconer = lightconer,
            write      = True,
        )
        lightcone.save(lightcone_cache)
        sim_time = time.time() - t0
        status_parts.append("lc:computed")

    z_lc         = np.array(lightcone.lightcone_redshifts,
                            dtype=np.float32)
    n_lc         = len(z_lc)
    lc_distances = np.array(lightcone.lightcone_distances.to_value('Mpc'),
                            dtype=np.float32)

    # =====================================================================
    # STEP 2: FIELD ARRAYS
    # =====================================================================
    if os.path.exists(fields_cache):
        status_parts.append("fields:cached")
    else:
        density_lc      = np.array(lightcone.lightcones['density'],
                                    dtype=np.float32)
        neutral_frac_lc = np.array(lightcone.lightcones['neutral_fraction'],
                                    dtype=np.float32)
        brightness_lc   = np.array(lightcone.lightcones['brightness_temp'],
                                    dtype=np.float32)
        los_velocity_lc = np.array(lightcone.lightcones['velocity_z'],
                                    dtype=np.float32)
        kinetic_temp_lc = np.array(lightcone.lightcones['kinetic_temperature'],
                                    dtype=np.float32)

        np.savez_compressed(
            fields_cache,
            density_lc      = density_lc,
            neutral_frac_lc = neutral_frac_lc,
            los_velocity_lc = los_velocity_lc,
            brightness_lc   = brightness_lc,
            kinetic_temp_lc = kinetic_temp_lc,
            z_lc            = z_lc,
        )
        status_parts.append("fields:computed")

    # =====================================================================
    # STEP 3: HALO CATALOGUES + LIGHTCONE ARRAYS (with slab filter)
    # =====================================================================
    if os.path.exists(halos_cache):
        status_parts.append("halos:cached")
    else:
        init_box              = p21c.compute_initial_conditions(inputs=inputs)
        node_redshifts_sorted = sorted(inputs.node_redshifts)

        halo_mass_lc  = np.full((HII_DIM, HII_DIM, n_lc), np.nan,
                                dtype=np.float32)
        halo_count_lc = np.zeros((HII_DIM, HII_DIM, n_lc),
                                dtype=np.float32)

        lcpix = lightconer.get_lc_distances_in_pixels(
            inputs.simulation_options.cell_size)

        for i, z_node in enumerate(node_redshifts_sorted):
            halo_cat = p21c.determine_halo_catalog(
                redshift           = z_node,
                initial_conditions = init_box,
                inputs             = inputs,
            )
            pt_halo_cat = p21c.perturb_halo_catalog(
                initial_conditions = init_box,
                inputs             = inputs,
                halo_catalog       = halo_cat,
            )

            dc_node = cosmo.comoving_distance(z_node).to_value('Mpc')
            z_idx   = np.argmin(np.abs(lc_distances - dc_node))

            masses_to_use = pt_halo_cat.get('halo_masses')
            coords_to_use = pt_halo_cat.get('halo_coords')

            if masses_to_use is None or len(masses_to_use) == 0:
                print(f"  [seed {seed:3d}] z={z_node:.3f}  no halos",
                      flush=True)
                continue

            cut   = masses_to_use > MASS_CUT
            m_cut = masses_to_use[cut]
            c_cut = coords_to_use[cut]

            if len(m_cut) == 0:
                print(f"  [seed {seed:3d}] z={z_node:.3f}  "
                      f"0 halos above mass cut", flush=True)
                continue

            # ──────────────────────────────────────────────────────────────
            # SLAB FILTER: Map LC slice to coeval z-layer (proper mapping)
            # ──────────────────────────────────────────────────────────────
            lcidx  = int((lcpix.max() - lcpix[z_idx] + 1*pixel)
                         .to_value(pixel))
            z_cell = (-lcidx + lightconer.index_offset) % HII_DIM
            z_lo   = z_cell * cell_size_mpc
            z_hi   = z_lo + cell_size_mpc

            slab_mask = (c_cut[:, 2] >= z_lo) & (c_cut[:, 2] < z_hi)
            m_slab    = m_cut[slab_mask]
            c_slab    = c_cut[slab_mask]

            if len(m_slab) == 0:
                print(f"  [seed {seed:3d}] z={z_node:.3f}  "
                      f"0 halos in slab [{z_lo:.2f}, {z_hi:.2f}] cMpc",
                      flush=True)
                continue

            # ─── Save raw slab halos for HMF analysis ───
            tag = f"z{z_node:.4f}"
            np.save(os.path.join(halo_out_seed_dir, f"masses_{tag}.npy"),
                    m_slab)
            np.save(os.path.join(halo_out_seed_dir, f"coords_{tag}.npy"),
                    c_slab)

            # Bin halos at this slice
            mass_map, _, _ = np.histogram2d(
                c_slab[:,0], c_slab[:,1],
                bins=HII_DIM, range=[[0,BOX_LEN],[0,BOX_LEN]],
                weights=m_slab)
            count_map, _, _ = np.histogram2d(
                c_slab[:,0], c_slab[:,1],
                bins=HII_DIM, range=[[0,BOX_LEN],[0,BOX_LEN]])

            with np.errstate(invalid='ignore', divide='ignore'):
                avg_map = np.where(count_map > 0,
                                   mass_map / count_map, np.nan)

            halo_mass_lc [:,:,z_idx] = avg_map.T.astype(np.float32)
            halo_count_lc[:,:,z_idx] = count_map.T.astype(np.float32)

            print(f"  [seed {seed:3d}] z={z_node:.3f}  {cut.sum():,} total  "
                  f"→  {len(m_slab):,} in slab  "
                  f"→  LC idx {z_idx}  z_cell {z_cell}", flush=True)

        np.savez_compressed(
            halos_cache,
            halo_mass_lc  = halo_mass_lc,
            halo_count_lc = halo_count_lc,
            z_lc          = z_lc,
        )
        status_parts.append("halos:computed")

    return (seed, seed_cache_dir, halo_out_seed_dir,
            n_lc, sim_time, " | ".join(status_parts))
