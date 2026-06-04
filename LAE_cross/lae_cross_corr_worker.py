# =============================================================================
# lae_cross_corr_worker.py
#
# Per-seed worker for kSZ²-LAE CELL 7b: cross-correlation power spectra at
# every LC slice that has LAEs (z < ~7.5).
#
# Spawned by ProcessPoolExecutor with mp.get_context("spawn") — must be a
# top-level module function so spawn workers can import it.
#
# Tracer field: LAE observed Lyman-alpha luminosity per pixel (LLya_obs),
# gridded from the SiMPLEGen catalogue into (HII_DIM, HII_DIM, n_lc).
# Physics, FFT conventions, mode counting, and result-dict keys are identical
# to halo_cross_corr_worker.py.
# =============================================================================

import os
import numpy as np


def _build_lae_lightcone(coords, lya_obs, redshifts, z_lc, HII_DIM, BOX_LEN):
    """
    Grid LAE catalogue into a (HII_DIM, HII_DIM, n_lc) luminosity lightcone.

    Each LC slice gets the summed LLya_obs of all LAEs whose redshift falls
    in the half-open interval [z_lo, z_hi) centred on that slice.

    Parameters
    ----------
    coords    : (N_LAE, 3) float32  comoving coords in cMpc
    lya_obs   : (N_LAE,)  float32  observed Lya luminosity [erg/s]
    redshifts : (N_LAE,)  float32  LAE redshifts
    z_lc      : (n_lc,)   float64  lightcone redshift array
    HII_DIM   : int
    BOX_LEN   : float  cMpc

    Returns
    -------
    lae_lc : (HII_DIM, HII_DIM, n_lc) float32
    """
    n_lc   = len(z_lc)
    lae_lc = np.zeros((HII_DIM, HII_DIM, n_lc), dtype=np.float64)

    cell_size = BOX_LEN / HII_DIM

    # bin edges between LC slices
    z_edges      = np.empty(n_lc + 1)
    z_edges[1:-1] = 0.5 * (z_lc[:-1] + z_lc[1:])
    z_edges[0]   = z_lc[0]  - 0.5 * (z_lc[1]  - z_lc[0])
    z_edges[-1]  = z_lc[-1] + 0.5 * (z_lc[-1] - z_lc[-2])

    # pixel indices
    xi = np.clip((coords[:, 0] / cell_size).astype(np.int32), 0, HII_DIM - 1)
    yi = np.clip((coords[:, 1] / cell_size).astype(np.int32), 0, HII_DIM - 1)

    # assign each LAE to its LC slice
    zi = np.searchsorted(z_edges[1:], redshifts, side='right')
    zi = np.clip(zi, 0, n_lc - 1)

    # accumulate luminosity per pixel per slice
    for k in range(n_lc):
        mask = zi == k
        if not np.any(mask):
            continue
        np.add.at(lae_lc[:, :, k], (xi[mask], yi[mask]), lya_obs[mask])

    return lae_lc


def compute_cross_corr_for_seed_lae(args):
    """
    Run (or load from cache) the kSZ²-LAE cross-correlation for one seed.

    Cache
    -----
    seed_cache_dir/kSZ2_lae_cross_seed{seed}.npy

    Returns
    -------
    (seed, lae_cross_results_dict, status_str)
    """
    (seed, lae_data_dir, kSZ_map, lc_redshifts,
     seed_cache_dir,
     npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
     dk, kgrid, k_bins, k_centers,
     REW_CUT, LLYA_CUT) = args

    os.makedirs(seed_cache_dir, exist_ok=True)
    cc_cache = os.path.join(seed_cache_dir,
                            f"kSZ2_lae_cross_seed{seed}.npy")

    if os.path.exists(cc_cache):
        lae_cross_results = np.load(cc_cache, allow_pickle=True).item()
        return (seed, lae_cross_results,
                f"cached ({len(lae_cross_results)} z)")

    # ── load LAE catalogue ────────────────────────────────────────────────
    coords    = np.load(os.path.join(lae_data_dir, "coords.npy"))
    redshifts = np.load(os.path.join(lae_data_dir, "redshifts.npy"))
    lya       = np.load(os.path.join(lae_data_dir, "LLya.npy"))
    damping   = np.load(os.path.join(lae_data_dir, "damping.npy"))
    rew       = np.load(os.path.join(lae_data_dir, "REW.npy"))

    lya_obs = lya * damping
    is_lae  = ((rew >= REW_CUT) & (lya_obs >= LLYA_CUT) & (damping > 0))

    coords_lae    = coords   [is_lae]
    lya_obs_lae   = lya_obs  [is_lae]
    redshifts_lae = redshifts[is_lae]

    # ── grid LAEs into lightcone ──────────────────────────────────────────
    lae_lc = _build_lae_lightcone(
        coords_lae, lya_obs_lae, redshifts_lae,
        lc_redshifts, npix_side, box_size_Mpc
    )

    # ── kSZ² FFT ──────────────────────────────────────────────────────────
    kSZ2_map          = kSZ_map ** 2
    kSZ2_map_centered = kSZ2_map - kSZ2_map.mean()
    fft_kSZ2_shifted  = np.fft.fftshift(np.fft.fft2(kSZ2_map_centered))
    auto_kSZ2_ps2d    = np.abs(fft_kSZ2_shifted)**2 * pix_area / npix_side**2
    kSZ2_rms          = float(np.sqrt(np.mean(kSZ2_map**2)))

    # ── find LC slices with LAEs ──────────────────────────────────────────
    lae_exists = np.array([lae_lc[:, :, i].sum() > 0
                           for i in range(lae_lc.shape[2])])
    lc_indices_with_lae = np.where(lae_exists)[0]

    lae_cross_results = {}
    area_2D           = box_size_Mpc ** 2

    for idx_closest in lc_indices_with_lae:
        z_lae = float(lc_redshifts[idx_closest])

        # ── LAE luminosity overdensity ─────────────────────────────────
        lae_slice = lae_lc[:, :, idx_closest].astype(np.float64)
	lae_slice /= 1e42   # normalise to units of 1e42 erg/s to avoid overflow
        lae_mean  = lae_slice.mean()
        if lae_mean <= 0:
            continue
        delta_lae = (lae_slice - lae_mean) / lae_mean

        # ── FFTs ──────────────────────────────────────────────────────────
        fft_lae_shifted = np.fft.fftshift(np.fft.fft2(delta_lae))
        auto_lae_ps2d   = (np.abs(fft_lae_shifted)**2
                           * pix_area / npix_side**2)
        cross_ps2d      = (np.real(np.conj(fft_kSZ2_shifted) * fft_lae_shifted)
                           * pix_area / npix_side**2)

        # ── k-binning ─────────────────────────────────────────────────────
        C_cross_1d         = np.full(len(k_centers), np.nan)
        C_cross_err_sample = np.full(len(k_centers), np.nan)
        C_cross_err_cosmic = np.full(len(k_centers), np.nan)
        C_cross_err_total  = np.full(len(k_centers), np.nan)
        P_kSZ2_1d          = np.full(len(k_centers), np.nan)
        P_lae_1d           = np.full(len(k_centers), np.nan)
        n_modes            = np.zeros(len(k_centers))

        for j in range(len(k_centers)):
            mask  = (kgrid >= k_bins[j]) & (kgrid < k_bins[j + 1])
            n_pix = int(mask.sum())
            if n_pix > 0:
                cv                    = cross_ps2d[mask]
                C_cross_1d[j]         = np.mean(cv)
                C_cross_err_sample[j] = np.std(cv) / np.sqrt(n_pix)
                P_kSZ2_1d[j]          = np.mean(auto_kSZ2_ps2d[mask])
                P_lae_1d[j]           = np.mean(auto_lae_ps2d[mask])

                n_modes[j] = (k_centers[j]
                              * (k_bins[j + 1] - k_bins[j])
                              * area_2D / (2 * np.pi)) / 2

                if n_modes[j] > 0:
                    C_cross_err_cosmic[j] = (
                        np.sqrt(P_kSZ2_1d[j] * P_lae_1d[j]
                                + C_cross_1d[j]**2)
                        / np.sqrt(n_modes[j]))
                C_cross_err_total[j] = np.sqrt(
                    C_cross_err_sample[j]**2
                    + C_cross_err_cosmic[j]**2)

        with np.errstate(divide='ignore', invalid='ignore'):
            r_cross = C_cross_1d / np.sqrt(P_kSZ2_1d * P_lae_1d)

        lae_cross_results[z_lae] = {
            'k_centers'          : k_centers,
            'C_cross_1d'         : C_cross_1d,
            'C_cross_err_sample' : C_cross_err_sample,
            'C_cross_err_cosmic' : C_cross_err_cosmic,
            'C_cross_err_total'  : C_cross_err_total,
            'P_kSZ2_1d'          : P_kSZ2_1d,
            'P_lae_1d'           : P_lae_1d,
            'r_cross'            : r_cross,
            'n_modes'            : n_modes,
            'z_actual'           : z_lae,
            'idx_closest'        : int(idx_closest),
            'lae_mean'           : float(lae_mean),
            'lae_rms'            : float(np.sqrt(np.mean(delta_lae**2))),
            'kSZ2_rms'           : kSZ2_rms,
        }

    np.save(cc_cache, lae_cross_results)
    return (seed, lae_cross_results,
            f"computed ({len(lae_cross_results)} z)")
