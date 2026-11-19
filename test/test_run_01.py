import os
import re
import glob
import argparse
import warnings
import mne
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.linalg as linalg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("ERROR")


# =====================================================================
# CONFIGURATION & CONSTANT VARIABLES
# =====================================================================
EPOCX_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
EI_CHANNELS = ["AF3", "AF4", "F3", "F4", "FC5", "FC6"]
POSTERIOR_CHANNELS = ["O1", "O2", "P7", "P8"]
EOG_PROXY_CHANNELS = ["AF3", "AF4"]

BANDS = {
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
}

L_FREQ = 1.0
H_FREQ = 40.0
ASR_CUTOFF_K = 25.0
MAX_ICA_EOG_COMPS = 2
ICA_METHOD = "infomax"

KEYSTROKE_OFFSET_S = 2.0
HOLD_TRANSITION_S = 3.0
SWALLOW_WINDOW_S = 5.0
POST_SWALLOW_SETTLE_S = 30.0
POST_SWALLOW_BIN_S = 30.0
PRE_EVENT_ERP_S = 1.0
CLEAN_CWT_BASE_DUR_S = 5.0

DRINK_KEYWORDS = ("eau", "water", "the_vert", "drink", "tea", "lait", "coke", "milo")
SURVEY_ALIASES = {
    "chocolate": "chocolat", "chocolat": "chocolat",
    "biscuit": "biscuit", "cracker": "biscuit",
    "green tea": "the_vert", "the vert": "the_vert", "the_vert": "the_vert", "tea": "the_vert",
    "water": "eau", "eau": "eau",
}


def print_config_summary():
    """Prints hardware constants, ROI channels, and preprocessing settings."""
    print("=" * 78)
    print("[Config] Pipeline Configuration (14-Ch Emotiv EPOC X | Native CMS/DRL, No CAR)")
    print(f"  • Channels (14)   : {', '.join(EPOCX_CHANNELS)}")
    print(f"  • Frontal ROI     : {', '.join(EI_CHANNELS)}")
    print(f"  • Passband & ASR  : {L_FREQ}–{H_FREQ} Hz (Notch {NOTCH_FREQ} Hz) | ASR k={ASR_CUTOFF_K}")
    print("=" * 78)


# =====================================================================
# EDF LOADING & EXPERIMENTAL PHASE MARKERS
# =====================================================================
def _list_edfs(folder):
    return [f for f in sorted(glob.glob(os.path.join(folder, "*.edf"))) if not f.lower().endswith(".md.edf")]


def _is_post_swallow_file(path):
    return re.search(r"post[\s_-]*swallow", os.path.basename(path).lower()) is not None


def _edf_timestamp_key(path):
    name = os.path.basename(path)
    m = re.search(r"(\d{4}\.\d{2}\.\d{2}T\d{2}\.\d{2}\.\d{2})", name)
    return (0, m.group(1), name.lower()) if m else (1, "", name.lower())


def _discover_post_swallow_edfs(edf_path):
    abs_primary = os.path.abspath(edf_path)
    parent_dir = os.path.dirname(abs_primary)
    if os.path.basename(parent_dir).lower() == "rest" or _is_post_swallow_file(abs_primary):
        return []
    candidates = [
        p for p in _list_edfs(parent_dir)
        if os.path.abspath(p) != abs_primary and _is_post_swallow_file(p)
    ]
    return sorted(candidates, key=_edf_timestamp_key)


def _load_single_edf(edf_path):
    """Loads a single EDF file, picks EPOC X channels, scales to Volts, and assigns 10-20 montage."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.rename_channels({ch: ch.replace("EEG.", "").strip() for ch in raw.ch_names})
    raw.pick([ch for ch in EPOCX_CHANNELS if ch in raw.ch_names])
    scale = 1e-6 if np.ptp(raw.get_data()) > 0.1 else 1.0
    raw.apply_function(lambda x: x * scale, picks="eeg")
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return raw


def _combine_edf_recordings(edf_path, extra_post_swallow_edf_path=None):
    """Loads primary EDF and chronologically stitches any continuation post-swallow EDF(s)."""
    if extra_post_swallow_edf_path is None:
        ext_candidates = _discover_post_swallow_edfs(edf_path)
    elif extra_post_swallow_edf_path is False:
        ext_candidates = []
    elif isinstance(extra_post_swallow_edf_path, (str, os.PathLike)):
        ext_candidates = [os.fspath(extra_post_swallow_edf_path)]
    else:
        ext_candidates = [os.fspath(p) for p in extra_post_swallow_edf_path if p]

    abs_primary = os.path.abspath(edf_path)
    seen = {abs_primary}
    unique_ext_paths = []
    for p in ext_candidates:
        abs_p = os.path.abspath(p)
        if os.path.exists(p) and abs_p not in seen:
            seen.add(abs_p)
            unique_ext_paths.append(p)
    unique_ext_paths.sort(key=_edf_timestamp_key)

    raw_primary = _load_single_edf(edf_path)
    primary_dur = raw_primary.times[-1]
    sfreq = raw_primary.info["sfreq"]
    cum_samples = raw_primary.n_times

    raws = [raw_primary]
    stitched_segments = []

    for ext_path in unique_ext_paths:
        raw_ext = _load_single_edf(ext_path)
        if raw_ext.info["sfreq"] != sfreq:
            raw_ext.resample(sfreq, verbose=False)

        common_chs = [ch for ch in raws[0].ch_names if ch in raw_ext.ch_names]
        if len(common_chs) < len(raws[0].ch_names):
            for r in raws:
                r.pick(common_chs)
        if raw_ext.ch_names != common_chs:
            raw_ext.pick(common_chs)

        stitched_segments.append((ext_path, raw_ext.times[-1], cum_samples / sfreq))
        cum_samples += raw_ext.n_times
        raws.append(raw_ext)

    raw = mne.concatenate_raws(raws, verbose=False) if len(raws) > 1 else raws[0]
    return raw, primary_dur, stitched_segments


def load_edf_and_markers(
    edf_path,
    marker_csv_path=None,
    extra_post_swallow_edf_path=None,
    verbose_on=True,
):
    """
    Loads an EmotivPRO EDF file, stitches any '*_post_swallow*.edf' continuation file(s),
    and parses intervalMarker CSVs. Selects the LATER marker when multiple Post-Swallow
    markers exist.
    """
    raw, primary_dur, stitched_segments = _combine_edf_recordings(edf_path, extra_post_swallow_edf_path)

    t_max = raw.times[-1]
    phase_windows = {}
    path_hint = f"{os.path.basename(os.path.dirname(edf_path))}_{os.path.basename(edf_path)}".lower()
    is_drink_by_name = any(k in path_hint for k in DRINK_KEYWORDS)
    is_drink_trial = is_drink_by_name

    marker_frames = []
    if marker_csv_path and os.path.exists(marker_csv_path):
        primary_df = pd.read_csv(marker_csv_path)
        if not primary_df.empty and "latency" in primary_df.columns:
            marker_frames.append(primary_df)

    for ext_path, _, offset_s in stitched_segments:
        ext_csv = f"{os.path.splitext(ext_path)[0]}_intervalMarker.csv"
        if os.path.exists(ext_csv):
            ext_df = pd.read_csv(ext_csv)
            if not ext_df.empty and "latency" in ext_df.columns:
                ext_df = ext_df.copy()
                ext_df["latency"] = ext_df["latency"] + offset_s
                marker_frames.append(ext_df)

    markers_df = pd.concat(marker_frames, ignore_index=True) if marker_frames else pd.DataFrame()
    df_post = pd.DataFrame()
    t_post = None

    if not markers_df.empty and "latency" in markers_df.columns:
        markers_df["type_clean"] = markers_df["type"].astype(str).str.strip().str.lower()
        markers_df["adj_latency"] = (markers_df["latency"] - KEYSTROKE_OFFSET_S).clip(lower=0.5)
        markers_df = markers_df.sort_values("adj_latency").reset_index(drop=True)

        is_post = markers_df["type_clean"].str.contains("post", na=False)
        if "marker_value" in markers_df.columns:
            is_post = is_post | (markers_df["marker_value"] == 5)

        df_non_post, df_post = markers_df[~is_post], markers_df[is_post]

        def _pick_time(df_sub, patterns):
            mask = df_sub["type_clean"].str.contains("|".join(patterns), na=False)
            matches = df_sub.loc[mask, "adj_latency"]
            return float(matches.iloc[0]) if not matches.empty else None

        t_reach = _pick_time(df_non_post, ["reach", "take", "open"])
        t_hold = _pick_time(df_non_post, ["hold", "touch", "feed", "pour", "drink"])
        t_chew = _pick_time(df_non_post, ["chew", "masticat"])
        t_swallow = _pick_time(df_non_post, ["swallow", "gulp"])
        # Select the LATER (last) marker if multiple Post-Swallow markers exist
        t_post = float(df_post["adj_latency"].iloc[-1]) if not df_post.empty else None

        # Auto-repair 4-marker drink trials where keys were mislabeled (e.g. Reach, Hold, Hold, Chew)
        if is_drink_by_name and t_swallow is None and len(markers_df) == 4:
            t_reach, t_hold, t_swallow, t_post = [float(x) for x in markers_df["adj_latency"].iloc[:4]]
            t_chew = None

        if t_swallow is None and not df_non_post.empty:
            t_swallow = float(df_non_post["adj_latency"].iloc[-1])
        is_drink_trial = (t_chew is None) or is_drink_by_name
        t_chew = None if is_drink_trial else t_chew
        t_after_hold = t_swallow if is_drink_trial else t_chew
        if t_reach is None and t_hold is not None:
            t_reach = max(0.5, t_hold - 3.0)

        first_candidates = [t for t in (t_reach, t_hold, t_chew, t_swallow) if t is not None]
        if first_candidates:
            phase_windows["2_Pre_Touch_Rest"] = (0.0, first_candidates[0])

        if t_reach is not None and t_hold is not None and t_hold > t_reach:
            phase_windows["2a_Reach_Phase"] = (t_reach, t_hold)

        if t_hold is not None and t_after_hold is not None and t_after_hold > t_hold:
            suffix = "Drink" if is_drink_trial else "Food"
            t_core = min(t_hold + HOLD_TRANSITION_S, t_after_hold - 1.0)
            t_core = t_core if t_core > t_hold else t_hold + 0.5 * (t_after_hold - t_hold)
            phase_windows["2b_Hold_Transition"] = (t_hold, t_core)
            phase_windows[f"2c_Hold_{suffix}_Core"] = (t_core, t_after_hold)
            phase_windows[f"2b_Hold_{suffix}_Full"] = (t_hold, t_after_hold)

        if t_chew is not None and t_swallow is not None and t_swallow > t_chew:
            phase_windows["3_Chew_Phase"] = (t_chew, t_swallow)

        if t_swallow is not None:
            if t_post is not None and t_post > t_swallow:
                t_sw_end = min(t_max, t_post)
            elif stitched_segments and primary_dur > t_swallow:
                t_sw_end = min(primary_dur, t_swallow + SWALLOW_WINDOW_S)
            else:
                t_sw_end = min(t_max, t_swallow + SWALLOW_WINDOW_S)
            t_clean = min(t_sw_end + POST_SWALLOW_SETTLE_S, max(t_sw_end, t_max - 10.0))
            phase_windows["3b_Swallow_Phase"] = (t_swallow, t_sw_end)
            phase_windows["4a_Post_Swallow_Settling"] = (t_sw_end, t_clean)
            phase_windows["4b_Post_Swallow_Clean_Recovery"] = (t_clean, t_max)
            phase_windows["4_Post_Swallow_Full"] = (t_sw_end, t_max)

    if verbose_on and (not markers_df.empty or stitched_segments):
        print(f"\n[Load] {os.path.basename(edf_path)} (Dur: {t_max:.1f}s | Drink: {is_drink_trial})")
        if stitched_segments:
            ext_names = ", ".join(os.path.basename(p) for p, _, _ in stitched_segments)
            print(f"  • Combined {len(stitched_segments)} post-swallow file(s) onto primary ({primary_dur:.1f}s -> {t_max:.1f}s): {ext_names}")
        if len(df_post) > 1:
            print(f"  • Multi-Post-Swallow detected ({len(df_post)} markers); selected later at {t_post:.2f}s (adj).")
        for k, (s, e) in phase_windows.items():
            print(f"    - {k:30s}: {s:6.2f}s -> {e:6.2f}s ({e - s:6.2f}s)")

    return raw, phase_windows, is_drink_trial


# =====================================================================
# PREPROCESSING, CONSERVATIVE ASR & ICA DENOISING (NO CAR)
# =====================================================================
def _apply_conservative_subspace_asr(raw_target, raw_calib, cutoff_k=ASR_CUTOFF_K):
    """Applies conservative subspace Artifact Subspace Reconstruction against calibration data."""
    data_calib = raw_calib.get_data()
    n_ch, _ = data_calib.shape
    sfreq = raw_target.info["sfreq"]

    win_samples = int(sfreq)
    n_wins = max(1, data_calib.shape[1] // win_samples)
    calib_trimmed = data_calib[:, :n_wins * win_samples].reshape(n_ch, n_wins, win_samples)

    cov_calib = np.cov(data_calib) + 1e-12 * np.eye(n_ch)
    _, eigvecs_c = linalg.eigh(cov_calib)

    proj_calib = np.tensordot(eigvecs_c.T, calib_trimmed, axes=([1], [0]))
    proj_rms = np.sqrt(np.mean(proj_calib ** 2, axis=-1))
    thresh_eig = (np.mean(proj_rms, axis=1) + cutoff_k * np.std(proj_rms, axis=1)) ** 2

    data_out = raw_target.get_data().copy()
    step, wlen = max(1, int(0.25 * sfreq)), max(2, int(0.50 * sfreq))
    n_modified, total_wins = 0, 0

    for start in range(0, data_out.shape[1] - wlen, step):
        seg = data_out[:, start:start + wlen]
        total_wins += 1
        eigvals_s, eigvecs_s = linalg.eigh(np.cov(seg) + 1e-12 * np.eye(n_ch))
        proj_thresh = np.sum((eigvecs_c.T @ eigvecs_s) ** 2 * thresh_eig[:, None], axis=0)
        exceed_mask = eigvals_s > proj_thresh
        if np.any(exceed_mask):
            n_modified += 1
            scales = np.where(exceed_mask, np.sqrt(proj_thresh / np.maximum(eigvals_s, 1e-30)), 1.0)
            data_out[:, start:start + wlen] = (eigvecs_s @ np.diag(scales) @ eigvecs_s.T) @ seg

    raw_cleaned = raw_target.copy()
    raw_cleaned._data = data_out
    return raw_cleaned, (n_modified / max(1, total_wins)) * 100.0


def preprocess_and_denoise(raw, raw_rest_calib=None, verbose_on=True):
    """Applies 1-40 Hz Bandpass, Conservative ASR (k=25), and ICA without CAR."""
    raw_work = raw.copy()
    raw_work.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design="firwin", verbose=False)

    # Boundary annotations protect FIR filtering across splice points; strip them afterwards
    # so downstream ASR, ICA, Welch PSD, and Epochs treat the combined recording as single-shot.
    if len(raw_work.annotations) > 0:
        boundary_idx = [
            i for i, desc in enumerate(raw_work.annotations.description)
            if desc in ("BAD boundary", "EDGE boundary")
        ]
        if boundary_idx:
            raw_work.annotations.delete(boundary_idx)

    raw_asr, asr_pct = _apply_conservative_subspace_asr(
        raw_work, raw_rest_calib or raw_work, cutoff_k=ASR_CUTOFF_K
    )

    n_ch = len(raw_asr.ch_names)
    ica = mne.preprocessing.ICA(
        n_components=min(12, n_ch - 1),
        method=ICA_METHOD,
        random_state=42,
        max_iter="auto",
    )
    ica.fit(raw_asr, verbose=False)

    eog_picks = [ch for ch in EOG_PROXY_CHANNELS if ch in raw_asr.ch_names]
    eog_inds, eog_scores = (
        ica.find_bads_eog(raw_asr, ch_name=eog_picks, threshold=2.5, verbose=False)
        if eog_picks else ([], None)
    )
    if len(eog_inds) > 0:
        max_scores = np.max(np.abs(eog_scores), axis=0)
        sorted_eog = sorted([int(i) for i in eog_inds], key=lambda idx: max_scores[idx], reverse=True)
        ica.exclude = sorted(sorted_eog[:MAX_ICA_EOG_COMPS])

    raw_clean = ica.apply(raw_asr, verbose=False)

    if verbose_on:
        print(f"[Denoise] ASR modified: {asr_pct:.1f}% | ICA pruned: {ica.exclude} | Ref: Native CMS/DRL (No CAR)")

    return raw_clean


# =====================================================================
# NEURAL METRIC CALCULATION (PSD, FAA, DELTA-FAA, Z-SCORE, EI)
# =====================================================================
def calculate_neural_metrics(psds_uv2, freqs, ch_names, duration_s,
                             baseline_ref=None, zscore_ref_stats=None,
                             phase_label="", verbose_on=True, return_channel_maps=False):
    """Computes band powers, Alpha/Beta ratios, Engagement Index, FAA, deltas, and Z-scores."""
    band_arrays = {}
    for b, (f0, f1) in BANDS.items():
        mask = (freqs >= f0) & (freqs <= f1)
        band_arrays[b] = np.maximum(integrate.trapezoid(psds_uv2[:, mask], freqs[mask], axis=-1), 1e-12)
    band_powers = {b: dict(zip(ch_names, arr)) for b, arr in band_arrays.items()}

    fc_chs = [c for c in EI_CHANNELS if c in ch_names]
    post_chs = [c for c in POSTERIOR_CHANNELS if c in ch_names]

    theta_fc = float(np.mean([band_powers["Theta"][c] for c in fc_chs]))
    alpha_fc = float(np.mean([band_powers["Alpha"][c] for c in fc_chs]))
    beta_fc = float(np.mean([band_powers["Beta"][c] for c in fc_chs]))
    alpha_post = float(np.mean([band_powers["Alpha"][c] for c in post_chs]))
    beta_post = float(np.mean([band_powers["Beta"][c] for c in post_chs]))

    ab_ratio_global = float(np.mean(band_arrays["Alpha"]) / max(np.mean(band_arrays["Beta"]), 1e-12))
    ab_ratio_fc = alpha_fc / max(beta_fc, 1e-12)
    ab_ratio_post = alpha_post / max(beta_post, 1e-12)
    engagement_index = beta_fc / max(alpha_fc + theta_fc, 1e-12)

    faa_f4_f3 = float(np.log(band_powers["Alpha"]["F4"]) - np.log(band_powers["Alpha"]["F3"]))
    faa_af4_af3 = float(np.log(band_powers["Alpha"]["AF4"]) - np.log(band_powers["Alpha"]["AF3"]))

    ref = baseline_ref or {
        "Alpha_Beta_Ratio_FC": ab_ratio_fc,
        "Alpha_Beta_Ratio_Post": ab_ratio_post,
        "Engagement_Index": engagement_index,
        "FAA_F4_F3": faa_f4_f3,
        "FAA_AF4_AF3": faa_af4_af3,
    }
    ab_mu, ab_sd = (zscore_ref_stats or {}).get("ab_fc", (ab_ratio_fc, 1.0))
    ei_mu, ei_sd = (zscore_ref_stats or {}).get("ei", (engagement_index, 1.0))

    metrics = {
        "Phase": phase_label,
        "Duration_s": round(float(duration_s), 2),
        "Theta_FC_uV2": theta_fc,
        "Alpha_FC_uV2": alpha_fc,
        "Beta_FC_uV2": beta_fc,
        "Alpha_Beta_Ratio_Global": ab_ratio_global,
        "Alpha_Beta_Ratio_FC": ab_ratio_fc,
        "Alpha_Beta_Ratio_Post": ab_ratio_post,
        "Engagement_Index": engagement_index,
        "FAA_F4_F3": faa_f4_f3,
        "FAA_AF4_AF3": faa_af4_af3,
        "Delta_Alpha_Beta_FC": ab_ratio_fc - ref["Alpha_Beta_Ratio_FC"],
        "Delta_Alpha_Beta_Post": ab_ratio_post - ref.get("Alpha_Beta_Ratio_Post", ab_ratio_post),
        "Delta_Engagement_Index": engagement_index - ref["Engagement_Index"],
        "Delta_FAA_F4_F3": faa_f4_f3 - ref["FAA_F4_F3"],
        "Delta_FAA_AF4_AF3": faa_af4_af3 - ref["FAA_AF4_AF3"],
        "Z_Alpha_Beta_FC": (ab_ratio_fc - ab_mu) / max(ab_sd, 1e-6) if zscore_ref_stats else np.nan,
        "Z_Engagement_Index": (engagement_index - ei_mu) / max(ei_sd, 1e-6) if zscore_ref_stats else np.nan,
    }

    if verbose_on and phase_label:
        print(f"  [Metrics | {phase_label:30s}] EI: {engagement_index:.4f} | A/B(FC): {ab_ratio_fc:.4f} | "
              f"FAA(F4/F3): {faa_f4_f3:+.4f} (Δ: {metrics['Delta_FAA_F4_F3']:+.4f})")

    if not return_channel_maps:
        return metrics

    ch_map = {
        "ch_names": list(ch_names),
        "Theta": band_arrays["Theta"].copy(),
        "Alpha": band_arrays["Alpha"].copy(),
        "Beta": band_arrays["Beta"].copy(),
        "AB_Ratio": band_arrays["Alpha"] / np.maximum(band_arrays["Beta"], 1e-12),
    }
    return metrics, ch_map


# =====================================================================
# STATIONARY WELCH PSD & TOPOMAP EXTRACTION
# =====================================================================
def _extract_welch_psd_segment(raw_seg):
    """Computes Welch PSD in µV²/Hz for a cropped MNE Raw segment."""
    sfreq = raw_seg.info["sfreq"]
    n_per_seg = int(min(2.0 * sfreq, len(raw_seg.times)))
    spectrum = raw_seg.compute_psd(
        method="welch", fmin=L_FREQ, fmax=H_FREQ,
        n_fft=n_per_seg, n_per_seg=n_per_seg, n_overlap=n_per_seg // 2,
        window="hann", verbose=False,
    )
    psds, freqs = spectrum.get_data(return_freqs=True)
    return psds * 1e12, freqs


def compute_stationary_welch(raw_rest_clean, raw_task_clean, phase_windows, verbose_on=True):
    """Computes Welch PSD metrics across rest, task phases, and 30s post-swallow windows."""
    rest_dur = raw_rest_clean.times[-1]

    z_ab_vals, z_ei_vals = [], []
    for w_s in np.arange(0.0, max(10.0, rest_dur - 10.0), 10.0):
        sub_r = raw_rest_clean.copy().crop(tmin=w_s, tmax=min(rest_dur, w_s + 10.0))
        p_uv2, f_hz = _extract_welch_psd_segment(sub_r)
        m_r = calculate_neural_metrics(p_uv2, f_hz, sub_r.ch_names, 10.0, verbose_on=False)
        z_ab_vals.append(m_r["Alpha_Beta_Ratio_FC"])
        z_ei_vals.append(m_r["Engagement_Index"])

    zscore_ref = {
        "ab_fc": (float(np.mean(z_ab_vals)), float(np.std(z_ab_vals) + 1e-6)),
        "ei":    (float(np.mean(z_ei_vals)), float(np.std(z_ei_vals) + 1e-6)),
    }

    topomap_data = {"info": raw_task_clean.info.copy(), "phases": {}, "windows_30s": []}

    psds_base, freqs_base = _extract_welch_psd_segment(raw_rest_clean)
    base_metrics, base_map = calculate_neural_metrics(
        psds_base, freqs_base, raw_rest_clean.ch_names, rest_dur,
        zscore_ref_stats=zscore_ref, phase_label="1_Baseline_Rest_EDF",
        verbose_on=verbose_on, return_channel_maps=True,
    )
    topomap_data["phases"]["1_Baseline_Rest_EDF"] = base_map
    stationary_rows = [base_metrics]
    pre_touch_ref = None

    active_windows = phase_windows or {"2_Full_Task_Unmarked": (0.0, raw_task_clean.times[-1])}
    for phase_name, (t_start, t_end) in active_windows.items():
        if t_end - t_start < 0.5:
            continue
        seg = raw_task_clean.copy().crop(tmin=t_start, tmax=t_end)
        psds_seg, freqs_seg = _extract_welch_psd_segment(seg)
        m_seg, seg_map = calculate_neural_metrics(
            psds_seg, freqs_seg, seg.ch_names, t_end - t_start,
            baseline_ref=pre_touch_ref, zscore_ref_stats=zscore_ref,
            phase_label=phase_name, verbose_on=verbose_on, return_channel_maps=True,
        )
        topomap_data["phases"][phase_name] = seg_map
        if phase_name == "2_Pre_Touch_Rest":
            pre_touch_ref = m_seg
        stationary_rows.append(m_seg)

    df_stationary = pd.DataFrame(stationary_rows)

    post_windows_rows = []
    t_post_start, t_post_end = active_windows.get(
        "4_Post_Swallow_Full", active_windows.get("2_Full_Task_Unmarked", (None, None))
    )
    if t_post_start is not None:
        win_start, win_idx = t_post_start, 1
        while win_start + POST_SWALLOW_BIN_S <= t_post_end + 0.5:
            win_end = min(t_post_end, win_start + POST_SWALLOW_BIN_S)
            if win_end - win_start < 5.0:
                break
            seg_30s = raw_task_clean.copy().crop(tmin=win_start, tmax=win_end)
            p_30s, f_30s = _extract_welch_psd_segment(seg_30s)
            m_30s, map_30s = calculate_neural_metrics(
                p_30s, f_30s, seg_30s.ch_names, win_end - win_start,
                baseline_ref=pre_touch_ref, zscore_ref_stats=zscore_ref,
                phase_label="", verbose_on=False, return_channel_maps=True,
            )
            m_30s.update({
                "Window_Index": win_idx,
                "Start_Time_s": round(win_start, 2),
                "End_Time_s": round(win_end, 2),
                "Time_Post_Swallow_s": round((win_start - t_post_start) + (POST_SWALLOW_BIN_S / 2.0), 1),
                "Is_Clean_Recovery_Win": (win_idx >= 2),
            })
            post_windows_rows.append(m_30s)

            map_30s.update({
                "Window_Index": win_idx,
                "Rel_Start_s": round(win_start - t_post_start, 0),
                "Rel_End_s": round(win_end - t_post_start, 0),
                "FAA_F4_F3": m_30s["FAA_F4_F3"],
                "AB_FC": m_30s["Alpha_Beta_Ratio_FC"],
            })
            topomap_data["windows_30s"].append(map_30s)
            win_start += POST_SWALLOW_BIN_S
            win_idx += 1

    return df_stationary, pd.DataFrame(post_windows_rows), pre_touch_ref, topomap_data


# =====================================================================
# TRANSIENT ERP, GFP & CLEAN-ANCHORED MORLET CWT
# =====================================================================
def compute_transient_erp_cwt(raw_task_clean, phase_windows, pre_touch_ref=None):
    """Computes single-trial ERP, GFP, and clean-anchored Morlet CWT across key task phases."""
    if not phase_windows:
        return {}

    sfreq = raw_task_clean.info["sfreq"]
    freqs = np.arange(2.0, 40.5, 0.5)
    n_cycles = freqs / 2.0

    _, t_pre_end = phase_windows.get("2_Pre_Touch_Rest", (0.0, 6.0))
    c_end = max(2.0, t_pre_end - 1.0)
    c_start = max(0.0, c_end - CLEAN_CWT_BASE_DUR_S)
    base_epochs = mne.make_fixed_length_epochs(
        raw_task_clean.copy().crop(tmin=c_start, tmax=c_end),
        duration=c_end - c_start, preload=True, verbose=False,
    )
    power_base = base_epochs.compute_tfr(
        method="morlet", freqs=freqs, n_cycles=n_cycles,
        use_fft=True, return_itc=False, average=True, verbose=False,
    )
    clean_base_spectrum = np.mean(power_base.data, axis=-1, keepdims=True)

    hold_key = "2b_Hold_Drink_Full" if "2b_Hold_Drink_Full" in phase_windows else "2b_Hold_Food_Full"
    targets = [
        ("Reach",   phase_windows.get("2a_Reach_Phase")),
        ("Hold",    phase_windows.get(hold_key)),
        ("Swallow", phase_windows.get("3b_Swallow_Phase")),
    ]

    cwt_results = {}
    raw_erp_lp = raw_task_clean.copy().filter(l_freq=None, h_freq=20.0, verbose=False)

    for label, win_tuple in targets:
        if not win_tuple or (win_tuple[1] - win_tuple[0]) <= 0.5:
            continue
        onset_s, end_s = win_tuple
        post_dur_s = round(end_s - onset_s, 3)
        events = np.array([[int(round(onset_s * sfreq)), 0, 1]])

        evoked = mne.Epochs(
            raw_erp_lp, events=events, event_id={label: 1},
            tmin=-PRE_EVENT_ERP_S, tmax=post_dur_s,
            baseline=(-PRE_EVENT_ERP_S, 0.0), preload=True, verbose=False,
        ).average()
        times = evoked.times
        erp_fc_uv = evoked.copy().pick(EI_CHANNELS).get_data() * 1e6
        gfp_uv = np.std(evoked.get_data() * 1e6, axis=0)

        power = mne.Epochs(
            raw_task_clean, events=events, event_id={label: 1},
            tmin=-PRE_EVENT_ERP_S, tmax=post_dur_s,
            baseline=None, preload=True, verbose=False,
        ).compute_tfr(
            method="morlet", freqs=freqs, n_cycles=n_cycles,
            use_fft=True, return_itc=False, average=True, verbose=False,
        )

        power_db_matrix = 10.0 * np.log10(np.maximum(power.data, 1e-30) / np.maximum(clean_base_spectrum, 1e-30))
        cwt_uv2 = power.data * 1e12
        ch_names = power.ch_names
        fc_idx = [ch_names.index(c) for c in EI_CHANNELS if c in ch_names]

        inst_band_ch = {}
        for b, (f0, f1) in BANDS.items():
            f_mask = (freqs >= f0) & (freqs <= f1)
            inst_band_ch[b] = integrate.trapezoid(cwt_uv2[:, f_mask, :], freqs[f_mask], axis=1)

        inst_bands = {b: np.mean(arr[fc_idx, :], axis=0) for b, arr in inst_band_ch.items()}
        ei_t = inst_bands["Beta"] / np.maximum(inst_bands["Alpha"] + inst_bands["Theta"], 1e-12)
        ab_t = inst_bands["Alpha"] / np.maximum(inst_bands["Beta"], 1e-12)

        a_ch = {
            c: np.maximum(inst_band_ch["Alpha"][ch_names.index(c), :], 1e-12)
            for c in ("F3", "F4", "AF3", "AF4")
        }

        df_inst = pd.DataFrame({
            "Time_s": times,
            "GFP_uV": gfp_uv,
            "Theta_FC_uV2": inst_bands["Theta"],
            "Alpha_FC_uV2": inst_bands["Alpha"],
            "Beta_FC_uV2": inst_bands["Beta"],
            "Alpha_Beta_Ratio_FC": ab_t,
            "Engagement_Index": ei_t,
            "FAA_F4_F3": np.log(a_ch["F4"]) - np.log(a_ch["F3"]),
            "FAA_AF4_AF3": np.log(a_ch["AF4"]) - np.log(a_ch["AF3"]),
        })

        use_core = (label == "Hold" and post_dur_s > HOLD_TRANSITION_S + 1.0)
        eval_mask = (times >= HOLD_TRANSITION_S) if use_core else (times >= 0.0)
        mask_desc = (
            f"Core Hold [+{HOLD_TRANSITION_S}s to +{post_dur_s:.1f}s]"
            if use_core else f"Active [0.0s to +{post_dur_s:.1f}s]"
        )

        sub_eval = df_inst[eval_mask]
        pk_idx = sub_eval["GFP_uV"].idxmax()
        mean_faa = float(sub_eval["FAA_F4_F3"].mean())
        delta_faa = mean_faa - (pre_touch_ref["FAA_F4_F3"] if pre_touch_ref else 0.0)

        cwt_results[label] = {
            "df_inst": df_inst,
            "times": times,
            "freqs": freqs,
            "erp_fc_uv": erp_fc_uv,
            "gfp_uv": gfp_uv,
            "fc_tfr_db": np.mean(power_db_matrix[fc_idx, :, :], axis=0),
            "peak_gfp_uv": float(sub_eval.loc[pk_idx, "GFP_uV"]),
            "peak_latency_s": float(sub_eval.loc[pk_idx, "Time_s"]),
            "eval_window_desc": mask_desc,
            "mean_ei": float(sub_eval["Engagement_Index"].mean()),
            "peak_ei": float(sub_eval["Engagement_Index"].max()),
            "mean_ab": float(sub_eval["Alpha_Beta_Ratio_FC"].mean()),
            "peak_ab": float(sub_eval["Alpha_Beta_Ratio_FC"].max()),
            "mean_faa_f4_f3": mean_faa,
            "delta_faa_f4_f3": delta_faa,
        }

    return cwt_results


# =====================================================================
# EXPORT CSVS, DIAGNOSTIC PLOTS & POST-SWALLOW TOPOMAPS
# =====================================================================
def _compute_robust_vlim(vecs):
    """Computes 5th-95th percentile color limits with a minimum span fallback."""
    all_vals = np.concatenate(vecs)
    vmin, vmax = float(np.percentile(all_vals, 5)), float(np.percentile(all_vals, 95))
    return (vmin, vmax) if (vmax - vmin > 1e-4) else (vmin - 0.1, vmax + 0.1)


def _plot_post_swallow_topomaps(participant_id, condition_name, topomap_data, output_dir):
    """Plots Post-Swallow Phase Topomaps and 30s Window Progression Topomaps."""
    if not topomap_data or "info" not in topomap_data:
        return

    info = topomap_data["info"]
    ch_names = info["ch_names"]
    phases_map = topomap_data.get("phases", {})
    wins_30s = topomap_data.get("windows_30s", [])

    # 1. Stationary Post-Swallow Phase Comparison Topomaps (4x4 Grid)
    base_key = "2_Pre_Touch_Rest" if "2_Pre_Touch_Rest" in phases_map else "1_Baseline_Rest_EDF"
    fallback_post = phases_map.get("4_Post_Swallow_Full", phases_map.get("2_Full_Task_Unmarked"))
    p_base = phases_map.get(base_key)
    p_settle = phases_map.get("4a_Post_Swallow_Settling", fallback_post)
    p_clean = phases_map.get("4b_Post_Swallow_Clean_Recovery", fallback_post)

    if p_base is not None and p_clean is not None:
        row_specs = [
            ("Theta", "Theta (4–8 Hz)\nlog10(µV²)", "YlOrRd", True),
            ("Alpha", "Alpha (8–13 Hz)\nlog10(µV²)", "YlGnBu", True),
            ("Beta", "Beta (13–30 Hz)\nlog10(µV²)", "PuRd", True),
            ("AB_Ratio", "Alpha / Beta Ratio\n(Calmness)", "viridis", False),
        ]
        col_titles = [
            "Pre-Touch Rest\n(Baseline Anchor)",
            "Post-Swallow Settling\n(0–30s Oral Buffer)",
            "Post-Swallow Clean Recovery\n(30s–End Aftertaste)",
            "Relative Shift (Clean vs. Rest)\n(Δ dB / Δ Ratio)",
        ]

        fig, axes = plt.subplots(4, 4, figsize=(15.5, 13.5))
        fig.suptitle(
            f"[{participant_id} — {condition_name}] Post-Swallow Cortical Topomaps (14-Ch Native CMS/DRL)",
            fontsize=14, fontweight="bold", y=0.98,
        )

        for r_idx, (m_key, m_label, cmap_name, is_log) in enumerate(row_specs):
            raw_vecs = [p_base[m_key], p_settle[m_key], p_clean[m_key]]
            vecs = [np.log10(np.maximum(v, 1e-12)) if is_log else v for v in raw_vecs]
            diff_vec = (10.0 * (vecs[2] - vecs[0])) if is_log else (vecs[2] - vecs[0])

            vlim_abs = _compute_robust_vlim(vecs)
            max_d = max(float(np.max(np.abs(diff_vec))), 0.25)

            for c_idx, vec in enumerate(vecs):
                im, _ = mne.viz.plot_topomap(
                    vec, info, axes=axes[r_idx, c_idx], names=ch_names,
                    cmap=cmap_name, vlim=vlim_abs, contours=4, show=False,
                )
                if r_idx == 0:
                    axes[r_idx, c_idx].set_title(col_titles[c_idx], fontsize=10, fontweight="bold", pad=10)
            axes[r_idx, 0].text(
                -0.28, 0.5, m_label, transform=axes[r_idx, 0].transAxes,
                fontsize=10, fontweight="bold", va="center", ha="right", rotation=90,
            )
            plt.colorbar(im, ax=axes[r_idx, 2], fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

            im_d, _ = mne.viz.plot_topomap(
                diff_vec, info, axes=axes[r_idx, 3], names=ch_names,
                cmap="RdBu_r", vlim=(-max_d, max_d), contours=4, show=False,
            )
            if r_idx == 0:
                axes[r_idx, 3].set_title(col_titles[3], fontsize=10, fontweight="bold", pad=10)
            plt.colorbar(im_d, ax=axes[r_idx, 3], fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

        plt.tight_layout(rect=[0.04, 0.01, 0.99, 0.95])
        plt.savefig(os.path.join(output_dir, "post_swallow_phase_topomaps.png"), dpi=300)
        plt.close(fig)

    # 2. Continuous 30s Post-Swallow Windows Topomap Progression
    if wins_30s:
        n_wins = len(wins_30s)
        fig_w, axes_w = plt.subplots(3, n_wins, figsize=(max(12.0, 2.8 * n_wins), 9.0), squeeze=False)
        fig_w.suptitle(
            f"[{participant_id} — {condition_name}] Post-Swallow 30s Window Topomaps (Win 1=Settling | Win 2+=Recovery)",
            fontsize=13, fontweight="bold", y=0.98,
        )

        win_specs = [
            ("Alpha", "Alpha (8–13 Hz)\nlog10(µV²)", "YlGnBu", True),
            ("Beta", "Beta (13–30 Hz)\nlog10(µV²)", "PuRd", True),
            ("AB_Ratio", "Alpha / Beta Ratio\n(Calmness)", "viridis", False),
        ]

        for r_idx, (m_key, m_label, cmap_name, is_log) in enumerate(win_specs):
            vecs = [np.log10(np.maximum(w[m_key], 1e-12)) if is_log else w[m_key] for w in wins_30s]
            vlim_w = _compute_robust_vlim(vecs)

            for c_idx, w in enumerate(wins_30s):
                im_w, _ = mne.viz.plot_topomap(
                    vecs[c_idx], info, axes=axes_w[r_idx, c_idx], names=ch_names,
                    cmap=cmap_name, vlim=vlim_w, contours=4, show=False,
                )
                if r_idx == 0:
                    tag = "Settle" if w["Window_Index"] == 1 else "Clean"
                    axes_w[r_idx, c_idx].set_title(
                        f"W{w['Window_Index']} ({int(w['Rel_Start_s'])}–{int(w['Rel_End_s'])}s) [{tag}]\n"
                        f"FAA:{w['FAA_F4_F3']:+.2f} | A/B:{w['AB_FC']:.2f}",
                        fontsize=8.5, fontweight="bold", pad=8,
                    )
            axes_w[r_idx, 0].text(
                -0.25, 0.5, m_label, transform=axes_w[r_idx, 0].transAxes,
                fontsize=9.5, fontweight="bold", va="center", ha="right", rotation=90,
            )
            plt.colorbar(im_w, ax=axes_w[r_idx, -1], fraction=0.046, pad=0.04).ax.tick_params(labelsize=7.5)

        plt.tight_layout(rect=[0.04, 0.01, 0.99, 0.93])
        plt.savefig(os.path.join(output_dir, "post_swallow_30s_windows_topomaps.png"), dpi=300)
        plt.close(fig_w)


def display_and_export_results(participant_id, condition_name, df_stationary,
                               df_post_30s, cwt_results, topomap_data=None, output_dir="."):
    """Exports CSV tables, post-swallow recovery trajectories, CWT/ERP panels, and topomaps."""
    os.makedirs(output_dir, exist_ok=True)

    df_stationary.to_csv(os.path.join(output_dir, "stationary_phases_welch.csv"), index=False)
    if not df_post_30s.empty:
        df_post_30s.to_csv(os.path.join(output_dir, "post_swallow_30s_timeseries.csv"), index=False)
    for label, res in cwt_results.items():
        res["df_inst"].to_csv(os.path.join(output_dir, f"{label.lower()}_cwt_erp_instantaneous.csv"), index=False)

    _plot_post_swallow_topomaps(participant_id, condition_name, topomap_data, output_dir)

    if not df_post_30s.empty:
        base_row = df_stationary.iloc[0]
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), sharex=True)
        x_t = df_post_30s["Time_Post_Swallow_s"]
        for ax in axes:
            ax.axvspan(0, POST_SWALLOW_SETTLE_S, color="#ffebee", alpha=0.45, label="0–30s Oral Settling Buffer (Win 1)")
            ax.grid(True, alpha=0.3)

        axes[0].plot(x_t, df_post_30s["Alpha_Beta_Ratio_FC"], marker="o", color="#2b5c8f", linewidth=2.2, label="Task Window (30s)")
        axes[0].axhline(base_row["Alpha_Beta_Ratio_FC"], color="#d95f02", linestyle="--", label=f"Baseline Rest EDF ({base_row['Alpha_Beta_Ratio_FC']:.2f})")
        axes[0].set_ylabel("Alpha / Beta Ratio (FC)")
        axes[0].set_title(f"[{participant_id} - {condition_name}] Post-Swallow Recovery Trajectory (30s Windows)")
        axes[0].legend(loc="upper left")

        axes[1].plot(x_t, df_post_30s["Engagement_Index"], marker="s", color="#388e3c", linewidth=2.2, label="Engagement Index: β / (α + θ)")
        axes[1].axhline(base_row["Engagement_Index"], color="#d95f02", linestyle="--", label=f"Baseline Rest EDF ({base_row['Engagement_Index']:.2f})")
        axes[1].set_ylabel("Engagement Index")
        axes[1].legend(loc="upper right")

        axes[2].plot(x_t, df_post_30s["Delta_FAA_F4_F3"], marker="^", color="#7b1fa2", linewidth=2.2, label="Relative ΔFAA (F4/F3 vs. Pre-Touch Rest)")
        axes[2].axhline(0.0, color="black", linestyle=":", alpha=0.7, label="Pre-Touch Rest Anchor (0.0)")
        axes[2].set_ylabel("ΔFAA (ln F4 − ln F3)")
        axes[2].set_xlabel("Time Elapsed After Post-Swallow Marker Onset (Seconds)")
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "post_swallow_30s_trajectory.png"), dpi=300)
        plt.close(fig)

    alpha_lo, alpha_hi = BANDS["Alpha"]
    for label, res in cwt_results.items():
        times, freqs, df_i = res["times"], res["freqs"], res["df_inst"]
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
        for idx, ch in enumerate(EI_CHANNELS):
            axes[0].plot(times, res["erp_fc_uv"][idx], alpha=0.78, linewidth=1.2, label=ch)
        axes[0].axvline(0.0, color="red", linestyle="--", linewidth=1.5, label=f"{label} Onset (t=0)")
        axes[0].set_ylabel("ERP (µV)")
        axes[0].set_title(f"[{participant_id} - {condition_name}] {label} Phase — Single-Trial ERP (Frontal/Central ROI)")
        axes[0].legend(loc="upper right", ncol=4, fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, res["gfp_uv"], color="#d95f02", linewidth=2.0, label="Global Field Power (14 Ch)")
        axes[1].fill_between(times, 0, res["gfp_uv"], color="#d95f02", alpha=0.15)
        axes[1].axvline(0.0, color="red", linestyle="--", linewidth=1.5)
        axes[1].scatter(
            [res["peak_latency_s"]], [res["peak_gfp_uv"]], color="black", zorder=5,
            label=f"Peak GFP: {res['peak_gfp_uv']:.1f} µV @ +{res['peak_latency_s']:.2f}s",
        )
        axes[1].set_ylabel("GFP (µV)")
        axes[1].set_title(f"{label} Phase — Global Cortical Synchrony (GFP Spike Detection)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        im = axes[2].pcolormesh(times, freqs, res["fc_tfr_db"], cmap="RdBu_r", shading="gouraud", vmin=-6, vmax=6)
        axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1.5)
        axes[2].axhline(alpha_lo, color="gray", linestyle=":", alpha=0.6)
        axes[2].axhline(alpha_hi, color="gray", linestyle=":", alpha=0.6)
        axes[2].set_ylabel("Frequency (Hz)")
        axes[2].set_title(f"{label} Phase — Morlet CWT Spectrogram (dB vs. Clean Pre-Touch Rest Anchor)")
        fig.colorbar(im, ax=axes[2], orientation="vertical", pad=0.01).set_label("Power (dB)")

        ax4_t = axes[3].twinx()
        p1, = axes[3].plot(times, df_i["Engagement_Index"], color="#2b5c8f", linewidth=2.0, label="Instantaneous Engagement Index")
        p2, = ax4_t.plot(times, df_i["Alpha_Beta_Ratio_FC"], color="#388e3c", linewidth=1.8, linestyle="-.", label="Instantaneous Alpha/Beta Ratio")
        axes[3].axvline(0.0, color="red", linestyle="--", linewidth=1.5)
        axes[3].set_xlabel(f"Time Relative to Adjusted {label} Marker Onset (Seconds)")
        axes[3].set_ylabel("Engagement Index", color="#2b5c8f")
        ax4_t.set_ylabel("Alpha / Beta Ratio", color="#388e3c")
        axes[3].set_title(f"{label} Phase — Instantaneous Neural Metric Spikes (CWT)")
        axes[3].legend(handles=[p1, p2], loc="upper right")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label.lower()}_cwt_erp_analysis.png"), dpi=300)
        plt.close(fig)


# =====================================================================
# SURVEY PARSER & DATASET RUNNERS
# =====================================================================
def parse_participant_survey(part_dir):
    """Parses participant subjective tasting survey ratings into a condition-keyed dict."""
    survey_files = glob.glob(os.path.join(part_dir, "*survey*.txt"))
    if not survey_files:
        return {}

    survey_data, current_cond = {}, None
    with open(survey_files[0], "r", encoding="utf-8", errors="ignore") as f:
        for line in (l.strip() for l in f if l.strip()):
            key = line.lower().rstrip(":")
            if key in SURVEY_ALIASES:
                current_cond = SURVEY_ALIASES[key]
                survey_data.setdefault(current_cond, {})
            elif current_cond and "-" in line:
                metric, val_str = line.split("-", 1)
                m_num = re.search(r"(\d+(?:\.\d+)?)", val_str)
                if m_num:
                    survey_data[current_cond][metric.strip().capitalize()] = float(m_num.group(1))
    return survey_data


def run_single_trial_pipeline(rest_edf_path, task_edf_path, task_csv_path,
                              participant_id="participant1", condition_name="food1",
                              output_dir=".", verbose_on=True,
                              extra_post_swallow_edf_path=None, raw_rest_clean=None):
    """Runs the complete single-trial pipeline from EDF ingestion through plots and CSV exports."""
    if verbose_on and raw_rest_clean is None:
        print_config_summary()

    if raw_rest_clean is None:
        raw_rest, _, _ = load_edf_and_markers(rest_edf_path, verbose_on=verbose_on)
        raw_rest_clean = preprocess_and_denoise(raw_rest, verbose_on=verbose_on)

    raw_task, phase_windows, _ = load_edf_and_markers(
        task_edf_path, task_csv_path,
        extra_post_swallow_edf_path=extra_post_swallow_edf_path,
        verbose_on=verbose_on,
    )
    raw_task_clean = preprocess_and_denoise(
        raw_task, raw_rest_calib=raw_rest_clean, verbose_on=verbose_on
    )

    df_stationary, df_post_30s, pre_touch_ref, topomap_data = compute_stationary_welch(
        raw_rest_clean, raw_task_clean, phase_windows, verbose_on=verbose_on
    )
    cwt_results = compute_transient_erp_cwt(
        raw_task_clean, phase_windows, pre_touch_ref=pre_touch_ref
    )
    display_and_export_results(
        participant_id, condition_name, df_stationary, df_post_30s,
        cwt_results, topomap_data=topomap_data, output_dir=output_dir,
    )
    return df_stationary, df_post_30s, cwt_results, topomap_data


def run_dataset_folder_pipeline(dataset_root="ensemble_de_donnees", trial=False, verbose_on=True, results_root=None):
    """Runs the pipeline across all participant folders and generates cohort summary CSVs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates = [
        os.path.abspath(dataset_root),
        os.path.abspath(os.path.join(project_root, dataset_root)),
        os.getcwd(),
    ]
    resolved_data = next((p for p in candidates if os.path.isdir(p) and (
        glob.glob(os.path.join(p, "*nouveau*")) or os.path.isdir(os.path.join(p, "testament"))
    )), os.path.abspath(dataset_root))
    results_root = os.path.abspath(results_root) if results_root else os.path.join(project_root, "results")

    scan_dir = os.path.join(resolved_data, "testament") if trial else resolved_data
    all_dirs = [
        d for d in sorted(glob.glob(os.path.join(scan_dir, "*")))
        if os.path.isdir(d) and os.path.basename(d).lower() != "testament"
    ]
    nouveau_dirs = [d for d in all_dirs if "nouveau" in os.path.basename(d).lower()]
    participant_dirs = nouveau_dirs or all_dirs

    if verbose_on:
        print_config_summary()
        print("\n" + "=" * 78)
        print(f"[Dataset Runner] Mode: {'TRIAL (testament)' if trial else 'FULL DATASET (nouveau folders)'}")
        print(f"  • Dataset Root: {resolved_data}")
        print(f"  • Participants: {[os.path.basename(d) for d in participant_dirs]}")
        print("=" * 78)

    results_all, cohort_stationary_list, cohort_survey_rows = {}, [], []

    for part_dir in participant_dirs:
        participant_id = os.path.basename(part_dir)
        survey_scores = parse_participant_survey(part_dir)
        rest_edfs = _list_edfs(os.path.join(part_dir, "rest"))
        if not rest_edfs:
            continue

        raw_rest, _, _ = load_edf_and_markers(rest_edfs[0], verbose_on=verbose_on)
        raw_rest_clean = preprocess_and_denoise(raw_rest, verbose_on=verbose_on)

        for cond_dir in sorted(glob.glob(os.path.join(part_dir, "*"))):
            cond_name = os.path.basename(cond_dir)
            if not os.path.isdir(cond_dir) or cond_name.lower() == "rest":
                continue

            cond_edfs = _list_edfs(cond_dir)
            if not cond_edfs:
                continue

            primary_edfs = sorted(
                [f for f in cond_edfs if not _is_post_swallow_file(f)],
                key=_edf_timestamp_key,
            )
            post_ext_edfs = sorted(
                [f for f in cond_edfs if _is_post_swallow_file(f)],
                key=_edf_timestamp_key,
            )
            task_edf_path = (primary_edfs or cond_edfs)[0]
            extra_post_edfs = post_ext_edfs if primary_edfs else []

            cond_csvs = [c for c in sorted(glob.glob(os.path.join(cond_dir, "*.csv"))) if not _is_post_swallow_file(c)]
            task_csv_path = cond_csvs[0] if cond_csvs else None

            out_subdir = os.path.join(results_root, participant_id, cond_name)
            df_stat, df_post30, cwt_res, topo_data = run_single_trial_pipeline(
                rest_edf_path=rest_edfs[0], task_edf_path=task_edf_path, task_csv_path=task_csv_path,
                participant_id=participant_id, condition_name=cond_name, output_dir=out_subdir,
                verbose_on=verbose_on, extra_post_swallow_edf_path=extra_post_edfs, raw_rest_clean=raw_rest_clean,
            )
            results_all[f"{participant_id}_{cond_name}"] = (df_stat, df_post30, cwt_res, topo_data)

            df_copy = df_stat.copy()
            df_copy.insert(0, "Condition", cond_name)
            df_copy.insert(0, "Participant", participant_id)
            cohort_stationary_list.append(df_copy)

            phase_dict = df_stat.set_index("Phase").to_dict(orient="index")
            r_pre = phase_dict.get("2_Pre_Touch_Rest", {})
            r_hold = phase_dict.get("2c_Hold_Food_Core", phase_dict.get("2c_Hold_Drink_Core", {}))
            r_chew = phase_dict.get("3_Chew_Phase", {})
            r_swal = phase_dict.get("3b_Swallow_Phase", {})
            r_settle = phase_dict.get("4a_Post_Swallow_Settling", {})
            r_clean = phase_dict.get("4b_Post_Swallow_Clean_Recovery", phase_dict.get("4_Post_Swallow_Full", {}))

            clean_wins = df_post30[df_post30["Is_Clean_Recovery_Win"]] if not df_post30.empty else pd.DataFrame()
            s_cond = survey_scores.get(cond_name.lower(), {})

            cohort_survey_rows.append({
                "Participant": participant_id,
                "Condition": cond_name,
                "Survey_Delicious": s_cond.get("Delicious", np.nan),
                "Survey_Happy": s_cond.get("Happy", np.nan),
                "Survey_Calm": s_cond.get("Calm", np.nan),
                "PreTouch_AB_FC": r_pre.get("Alpha_Beta_Ratio_FC", np.nan),
                "PreTouch_FAA_F4_F3": r_pre.get("FAA_F4_F3", np.nan),
                "HoldCore_FAA_F4_F3": r_hold.get("FAA_F4_F3", np.nan),
                "HoldCore_Delta_FAA": r_hold.get("Delta_FAA_F4_F3", np.nan),
                "Chew_FAA_F4_F3": r_chew.get("FAA_F4_F3", np.nan),
                "Swallow_FAA_F4_F3": r_swal.get("FAA_F4_F3", np.nan),
                "PostSettle_AB_FC": r_settle.get("Alpha_Beta_Ratio_FC", np.nan),
                "PostSettle_FAA_F4_F3": r_settle.get("FAA_F4_F3", np.nan),
                "PostClean_AB_FC": r_clean.get("Alpha_Beta_Ratio_FC", np.nan),
                "PostClean_Delta_AB_FC": r_clean.get("Delta_Alpha_Beta_FC", np.nan),
                "PostClean_AB_Post": r_clean.get("Alpha_Beta_Ratio_Post", np.nan),
                "PostClean_EI": r_clean.get("Engagement_Index", np.nan),
                "PostClean_FAA_F4_F3": r_clean.get("FAA_F4_F3", np.nan),
                "PostClean_Delta_FAA_F4_F3": r_clean.get("Delta_FAA_F4_F3", np.nan),
                "PostClean_FAA_AF4_AF3": r_clean.get("FAA_AF4_AF3", np.nan),
                "PostClean_Delta_FAA_AF4_AF3": r_clean.get("Delta_FAA_AF4_AF3", np.nan),
                "PostCleanWin_Max_FAA": float(clean_wins["FAA_F4_F3"].max()) if not clean_wins.empty else np.nan,
                "PostCleanWin_Max_AB_FC": float(clean_wins["Alpha_Beta_Ratio_FC"].max()) if not clean_wins.empty else np.nan,
            })

    if cohort_stationary_list:
        os.makedirs(results_root, exist_ok=True)
        pd.concat(cohort_stationary_list, ignore_index=True).to_csv(
            os.path.join(results_root, "cohort_stationary_summary.csv"), index=False
        )

    if cohort_survey_rows:
        df_cmp = pd.DataFrame(cohort_survey_rows)
        eau_faa = df_cmp[df_cmp["Condition"] == "eau"].set_index("Participant")["PostClean_FAA_F4_F3"].to_dict()
        eau_ab = df_cmp[df_cmp["Condition"] == "eau"].set_index("Participant")["PostClean_AB_FC"].to_dict()
        df_cmp["PostClean_FAA_vs_Water"] = df_cmp["PostClean_FAA_F4_F3"] - df_cmp["Participant"].map(eau_faa)
        df_cmp["PostClean_AB_vs_Water"] = df_cmp["PostClean_AB_FC"] - df_cmp["Participant"].map(eau_ab)
        df_cmp.to_csv(os.path.join(results_root, "cohort_survey_eeg_comparison.csv"), index=False)

        if verbose_on:
            print("\n" + "=" * 95)
            print("[COHORT SUMMARY] SURVEY SCORES VS. POST-SWALLOW CLEAN RECOVERY NEURAL METRICS")
            print("=" * 95)
            cols = [
                "Participant", "Condition", "Survey_Delicious", "Survey_Happy", "Survey_Calm",
                "PostClean_FAA_F4_F3", "PostClean_Delta_FAA_F4_F3", "PostClean_FAA_vs_Water",
                "PostClean_AB_FC", "PostClean_Delta_AB_FC", "PostClean_AB_vs_Water", "PostClean_EI",
            ]
            print(df_cmp[cols].round(4).to_string(index=False))
            print("=" * 95)

    return results_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Dataset Pipeline with ASR, ICA, Welch PSD, CWT & Post-Swallow Topomaps")
    parser.add_argument("--trial", action="store_true", default=False)
    parser.add_argument("--dataset-root", type=str, default="ensemble_de_donnees")
    parser.add_argument("--output-dir", type=str, default=None)
    args, _ = parser.parse_known_args()
    run_dataset_folder_pipeline(
        dataset_root=args.dataset_root,
        trial=args.trial,
        verbose_on=True,
        results_root=args.output_dir,
    )
