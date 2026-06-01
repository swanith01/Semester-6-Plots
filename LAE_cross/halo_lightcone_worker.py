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
        n_nodes               = len(node_redshifts_sorted)

        halo_mass_lc  = np.full((HII_DIM, HII_DIM, n_lc), np.nan,
                                dtype=np.float32)
        halo_count_lc = np.zeros((HII_DIM, HII_DIM, n_lc),
                                dtype=np.float32)

        lcpix = lightconer.get_lc_distances_in_pixels(
            inputs.simulation_options.cell_size)

        # ──────────────────────────────────────────────────────────────────
        # OPTION A: SLICE-CENTRIC HALO LIGHTCONE
        #
        # The old loop iterated over node redshifts and wrote one LC index
        # per node (argmin), leaving every other LC slice as NaN — hence the
        # striped halo lightcone.  Here we instead iterate over EVERY LC
        # slice and assign it to its nearest node redshift.  This partitions
        # all n_lc slices across the nodes with no gaps and no double-writes,
        # so the halo array fills continuously like the field lightcones.
        #
        # Per-node perturbed halo catalogues are computed lazily and cached
        # in `node_cat_cache` so each node is evaluated at most once, and
        # only if it actually owns at least one LC slice.
        # ──────────────────────────────────────────────────────────────────

        # slice -> owning node: nearest node in comoving distance
        node_dc = np.array(
            [cosmo.comoving_distance(zn).to_value('Mpc')
             for zn in node_redshifts_sorted],
            dtype=np.float64)
        # owner[z_idx] = index into node_redshifts_sorted
        owner = np.argmin(
            np.abs(lc_distances[:, None] - node_dc[None, :]), axis=1)

        # ──────────────────────────────────────────────────────────────────
        # MEMORY-BOUNDED NODE-GROUPED LOOP
        #
        # The full-box catalogue for one node is large (~10^8 halos →
        # ~GBs).  An earlier lazy-cache version kept every node's catalogue
        # alive in a dict, which accumulated until the OOM killer terminated
        # the worker (~14 min in).
        #
        # Fix: loop over NODES (outer), and for each node process ALL the LC
        # slices it owns (inner), then explicitly `del` the catalogue before
        # moving to the next node.  At most ONE node catalogue is alive at a
        # time — peak memory is bounded by the single largest node, not the
        # sum over all nodes.  No physics / output change.
        # ──────────────────────────────────────────────────────────────────
        import gc

        for node_idx in range(n_nodes):
            # which LC slices does this node own?
            slices_here = np.where(owner == node_idx)[0]
            if len(slices_here) == 0:
                continue

            z_node = node_redshifts_sorted[node_idx]

            # ── compute the node's full-box mass-cut catalogue ──────────────
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

            masses_to_use = pt_halo_cat.get('halo_masses')
            coords_to_use = pt_halo_cat.get('halo_coords')

            if masses_to_use is None or len(masses_to_use) == 0:
                print(f"  [seed {seed:3d}] z={z_node:.3f}  no halos",
                      flush=True)
                del halo_cat, pt_halo_cat
                gc.collect()
                continue

            cut   = masses_to_use > MASS_CUT
            m_cut = masses_to_use[cut]
            c_cut = coords_to_use[cut]
            # drop references to the un-cut arrays + p21c objects ASAP
            del masses_to_use, coords_to_use, cut, halo_cat, pt_halo_cat

            if len(m_cut) == 0:
                print(f"  [seed {seed:3d}] z={z_node:.3f}  "
                      f"0 halos above mass cut", flush=True)
                del m_cut, c_cut
                gc.collect()
                continue

            # ─── Save raw mass-cut halos for HMF analysis (once per node) ───
            tag = f"z{z_node:.4f}"
            np.save(os.path.join(halo_out_seed_dir, f"masses_{tag}.npy"),
                    m_cut)
            np.save(os.path.join(halo_out_seed_dir, f"coords_{tag}.npy"),
                    c_cut)
	    # ─── Save coeval field boxes for SiMPLEGen ───────────────────────
            coeval_out = os.path.join(halo_out_seed_dir, f"coeval_{tag}")
            if not os.path.exists(coeval_out):
                os.makedirs(coeval_out, exist_ok=True)
                coeval = p21c.run_coeval(
                    inputs         = inputs,
                    out_redshifts  = z_node,
                    initial_conditions = init_box,
                    write          = False,
                )
                box = coeval[0]
                np.save(f"{coeval_out}/density.npy",
                        box.density.astype(np.float32))
                np.save(f"{coeval_out}/neutral_fraction.npy",
                        box.neutral_fraction.astype(np.float32))
                np.save(f"{coeval_out}/kinetic_temp.npy",
                        box.kinetic_temperature.astype(np.float32))
                np.save(f"{coeval_out}/velocity_z.npy",
                        box.velocity_z.astype(np.float32))
                del coeval, box
                gc.collect()
                print(f"  [seed {seed:3d}] z={z_node:.3f}  coeval boxes saved",
                      flush=True)
            # ─────────────────────────────────────────────────────────────────
            # ── process every LC slice owned by this node ───────────────────
            for z_idx in slices_here:
                z_idx = int(z_idx)

                # SLAB FILTER: map this LC slice to its coeval z-layer
                lcidx  = int((lcpix.max() - lcpix[z_idx] + 1*pixel)
                             .to_value(pixel))
                z_cell = (-lcidx + lightconer.index_offset) % HII_DIM
                z_lo   = z_cell * cell_size_mpc
                z_hi   = z_lo + cell_size_mpc

                slab_mask = (c_cut[:, 2] >= z_lo) & (c_cut[:, 2] < z_hi)
                m_slab    = m_cut[slab_mask]
                c_slab    = c_cut[slab_mask]

                if len(m_slab) == 0:
                    # genuinely empty slab (common at high z) — leave NaN/0
                    continue

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

                print(f"  [seed {seed:3d}] LC idx {z_idx:4d}  "
                      f"z={z_node:.3f} (node {node_idx})  "
                      f"{len(m_cut):,} in box  →  {len(m_slab):,} in slab  "
                      f"z_cell {z_cell}", flush=True)

            # ── free this node's catalogue before the next node ────────────
            del m_cut, c_cut
            gc.collect()

        n_filled_seed = int(np.isfinite(halo_mass_lc).any(axis=(0,1)).sum())
        print(f"  [seed {seed:3d}] halo lightcone: "
              f"{n_filled_seed}/{n_lc} LC slices populated", flush=True)

        np.savez_compressed(
            halos_cache,
            halo_mass_lc  = halo_mass_lc,
            halo_count_lc = halo_count_lc,
            z_lc          = z_lc,
        )
        status_parts.append("halos:computed")

    return (seed, seed_cache_dir, halo_out_seed_dir,
            n_lc, sim_time, " | ".join(status_parts))
