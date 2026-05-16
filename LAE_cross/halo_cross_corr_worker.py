# =============================================================================
# halo_cross_corr_worker.py
#
# Per-seed worker for kSZ²-halo CELL 7: cross-correlation power spectra at
# every LC slice that has halos.
#
# Spawned by ProcessPoolExecutor with mp.get_context("spawn") — must be a
# top-level module function so spawn workers can import it.
#
# Logic is a verbatim port of the original (single-seed) CELL 7 inner loop:
# kSZ² FFT → per-z halo FFT → cross 2D PS → k-binning → error budget.
# Physics, FFT conventions, mode counting, and result-dict keys are unchanged
# EXCEPT: n_modes now uses 2D annulus geometry (was 3D shell — bug fix).
# =============================================================================

import os
import numpy as np


def compute_cross_corr_for_seed_halo(args):
    """
    Run (or load from cache) the kSZ²-halo cross-correlation pipeline for one seed.

    Cache
    -----
    seed_cache_dir/kSZ2_halo_cross_seed{seed}.npy   (pickled dict)

    Returns
    -------
    (seed, halo_cross_results_dict, status_str)
    """
    (seed, halo_count_lc, kSZ_map, lc_redshifts,
     seed_cache_dir,
     npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
     dk, kgrid, k_bins, k_centers) = args

    os.makedirs(seed_cache_dir, exist_ok=True)
    cc_cache = os.path.join(seed_cache_dir,
                            f"kSZ2_halo_cross_seed{seed}.npy")

    if os.path.exists(cc_cache):
        halo_cross_results = np.load(cc_cache, allow_pickle=True).item()
        return (seed, halo_cross_results,
                f"cached ({len(halo_cross_results)} z)")

    # ── kSZ² FFT ──────────────────────────────────────────────────────────
    kSZ2_map          = kSZ_map**2
    kSZ2_map_centered = kSZ2_map - kSZ2_map.mean()
    fft_kSZ2_shifted  = np.fft.fftshift(np.fft.fft2(kSZ2_map_centered))
    auto_kSZ2_ps2d    = np.abs(fft_kSZ2_shifted)**2 * pix_area / npix_side**2

    kSZ2_rms = float(np.sqrt(np.mean(kSZ2_map**2)))

    # ── Find LC slices that actually have halos ───────────────────────────
    halo_exists = np.array([halo_count_lc[:,:,i].sum() > 0
                            for i in range(halo_count_lc.shape[2])])
    lc_indices_with_halos = np.where(halo_exists)[0]

    halo_cross_results = {}

    # 2D box area (used in n_modes — geometric correction vs original 3D shell)
    area_2D = box_size_Mpc**2

    for i, idx_closest in enumerate(lc_indices_with_halos):
        z_halo   = float(lc_redshifts[idx_closest])
        z_actual = z_halo

        # ── halo field slice (overdensity built from passed-in field) ─────
        # NOTE: parameter is named halo_count_lc for historical reasons, but
        # Cell 7 now passes halo_mass_field_all (total halo mass per cell).
        # The overdensity construction below is field-agnostic.
        halo_slice = halo_count_lc[:, :, idx_closest].astype(np.float64)

        halo_mean = halo_slice.mean()
        if halo_mean <= 0:
            continue

        delta_h = (halo_slice - halo_mean) / halo_mean

        # ── FFTs ──────────────────────────────────────────────────────────
        fft_halo_shifted = np.fft.fftshift(np.fft.fft2(delta_h))
        auto_halo_ps2d   = (np.abs(fft_halo_shifted)**2
                            * pix_area / npix_side**2)
        cross_ps2d       = (np.real(np.conj(fft_kSZ2_shifted) * fft_halo_shifted)
                            * pix_area / npix_side**2)

        # ── k-binning ─────────────────────────────────────────────────────
        C_cross_1d         = np.full(len(k_centers), np.nan)
        C_cross_err_sample = np.full(len(k_centers), np.nan)
        C_cross_err_cosmic = np.full(len(k_centers), np.nan)
        C_cross_err_total  = np.full(len(k_centers), np.nan)
        P_kSZ2_1d          = np.full(len(k_centers), np.nan)
        P_halo_1d          = np.full(len(k_centers), np.nan)
        n_modes            = np.zeros(len(k_centers))

        for j in range(len(k_centers)):
            mask  = (kgrid >= k_bins[j]) & (kgrid < k_bins[j + 1])
            n_pix = int(mask.sum())
            if n_pix > 0:
                cv                    = cross_ps2d[mask]
                C_cross_1d[j]         = np.mean(cv)
                C_cross_err_sample[j] = np.std(cv) / np.sqrt(n_pix)
                P_kSZ2_1d[j]          = np.mean(auto_kSZ2_ps2d[mask])
                P_halo_1d[j]          = np.mean(auto_halo_ps2d[mask])

                # ── 2D annulus mode count (independent modes only) ────────
                # N_modes = (2π k Δk · A) / (2π)² = k Δk · A / (2π)
                # Factor of 1/2: hermitian symmetry of real-input FFT.
                n_modes[j] = (k_centers[j]
                              * (k_bins[j + 1] - k_bins[j])
                              * area_2D / (2 * np.pi)) / 2

                if n_modes[j] > 0:
                    C_cross_err_cosmic[j] = (
                        np.sqrt(P_kSZ2_1d[j] * P_halo_1d[j]
                                + C_cross_1d[j]**2)
                        / np.sqrt(n_modes[j]))
                C_cross_err_total[j] = np.sqrt(
                    C_cross_err_sample[j]**2
                    + C_cross_err_cosmic[j]**2)

        with np.errstate(divide='ignore', invalid='ignore'):
            r_cross = C_cross_1d / np.sqrt(P_kSZ2_1d * P_halo_1d)

        halo_cross_results[z_halo] = {
            'k_centers'          : k_centers,
            'C_cross_1d'         : C_cross_1d,
            'C_cross_err_sample' : C_cross_err_sample,
            'C_cross_err_cosmic' : C_cross_err_cosmic,
            'C_cross_err_total'  : C_cross_err_total,
            'P_kSZ2_1d'          : P_kSZ2_1d,
            'P_halo_1d'          : P_halo_1d,
            'r_cross'            : r_cross,
            'n_modes'            : n_modes,
            'z_actual'           : z_actual,
            'idx_closest'        : int(idx_closest),
            'halo_mean'          : float(halo_mean),
            'halo_rms'           : float(np.sqrt(np.mean(delta_h**2))),
            'kSZ2_rms'           : kSZ2_rms,
        }

    np.save(cc_cache, halo_cross_results)
    return (seed, halo_cross_results,
            f"computed ({len(halo_cross_results)} z)")
