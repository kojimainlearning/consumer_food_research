import os
import glob
import warnings
import mne
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.linalg as linalg
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")


# =====================================================================
# STEP 1: CONFIGURATION AND CONSTANT VARIABLES
# =====================================================================
def step1_get_config(verbose_on=True):
    """
    Step 1: Defines hardware constants, frequency bands, ROI channels,
    timing buffers, and denoising thresholds for the Emotiv EPOC X pipeline.
    """
    config = {
        # Emotiv EPOC X 14-Channel 10-20 Montage
        "EPOCX_CHANNELS": [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
        ],
        # Frontal & Fronto-Central ROI (Excludes T7/T8 temporalis muscle sensors)
        "EI_CHANNELS": ["AF3", "AF4", "F3", "F4", "FC5", "FC6"],
        "EOG_PROXY_CHANNELS": ["AF3", "AF4"],

        # Frequency Bands (Hz)
        "BANDS": {
            "Theta": (4.0, 8.0),
            "Alpha": (8.0, 13.0),
            "Beta":  (13.0, 30.0)
        },

        # Filtering & Denoising Parameters
        "L_FREQ": 1.0,                 # High-pass cutoff (Hz)
        "H_FREQ": 40.0,                # Low-pass cutoff (Hz)
        "NOTCH_FREQ": 50.0,            # Mains power line notch filter (50 Hz Malaysia/EU)
        "ASR_CUTOFF_K": 25.0,          # Conservative ASR threshold (k=20-40 preserves brain waves)
        "MAX_ICA_EOG_COMPS": 2,        # Max ocular ICA components to prune on 14-ch montage
        "APPLY_POST_ICA_CAR": False,   # False = retain native CMS/DRL reference (prevents EMG bleed)

        # Experimental Timing & Transition Buffers (Seconds)
        "KEYSTROKE_OFFSET_S": 2.0,     # Shift manual keystrokes earlier by 2.0s
        "HOLD_TRANSITION_S": 3.0,      # First 3.0s of Hold = arm-lowering/sipping transition
        "SWALLOW_WINDOW_S": 5.0,       # Fixed 5.0s duration for acute Swallow phase
        "POST_SWALLOW_SETTLE_S": 30.0, # First 30.0s (Window 1) = oral clearing buffer for solids
        "POST_SWALLOW_BIN_S": 30.0,    # 30.0s continuous window size for Post-Swallow recovery
        "PRE_EVENT_ERP_S": 1.0,        # 1.0s pre-onset window for ERP waveform display
        "CLEAN_CWT_BASE_DUR_S": 5.0    # 5.0s clean segment inside Pre_Touch_Rest for CWT dB anchor
    }

    if verbose_on:
        print("=" * 72)
        print("[Step 1] Pipeline Configuration Initialized")
        print(f"  • EPOC X Channels (14) : {', '.join(config['EPOCX_CHANNELS'])}")
        print(f"  • Frontal/Central ROI  : {', '.join(config['EI_CHANNELS'])}")
        print(f"  • Passband & ASR       : {config['L_FREQ']}–{config['H_FREQ']} Hz "
              f"(Notch: {config['NOTCH_FREQ']} Hz) | ASR Cutoff k = {config['ASR_CUTOFF_K']}")
        print(f"  • Timing Buffers       : Keystroke Shift = -{config['KEYSTROKE_OFFSET_S']}s | "
              f"Hold Transition Trim = {config['HOLD_TRANSITION_S']}s | "
              f"Swallow = {config['SWALLOW_WINDOW_S']}s | Post-Swallow Bin = {config['POST_SWALLOW_BIN_S']}s")
        print("=" * 72)

    return config


# =====================================================================
# STEP 2: LOAD EDF FILE & PARSE EXPERIMENTAL PHASE MARKERS
# =====================================================================
def step2_load_edf_and_markers(
    edf_path=None,
    marker_csv_path=None,
    config=None,
    trial=False,
    verbose_on=True,
    dataset_root="ensemble_de_donnees"
):
    """
    Step 2: Loads an EmotivPRO EDF file, isolates the 14 EEG channels, fixes
    microvolt scaling, and parses the intervalMarker CSV (supporting both
    Solid Food trials with 'Chew' and Liquid Drink trials without 'Chew').

    Parameters
    ----------
    edf_path : str, optional
        Path to EDF file. If None or a directory, executes the dataset pipeline across folders.
    marker_csv_path : str, optional
        Path to intervalMarker CSV file.
    config : dict, optional
        Pipeline configuration dictionary.
    trial : bool, default False
        When True, only runs the dataset in 'testament' folder in 'ensemble_de_donnees'.
        When False, runs all datasets in 'ensemble_de_donnees' except the folder called 'testament'.
    verbose_on : bool, default True
        Whether to print verbose output.
    dataset_root : str, default 'ensemble_de_donnees'
        Path to root dataset folder.
    """
    # Allow passing trial via first positional argument: step2_load_edf_and_markers(True)
    if isinstance(edf_path, bool):
        trial = edf_path
        edf_path = None

    # If edf_path is omitted or a directory, route directly to batch pipeline runner
    if edf_path is None or (isinstance(edf_path, str) and os.path.isdir(edf_path)):
        root_dir = edf_path if (isinstance(edf_path, str) and os.path.isdir(edf_path)) else dataset_root
        return run_dataset_folder_pipeline(dataset_root=root_dir, trial=trial, verbose_on=verbose_on)

    if config is None:
        config = step1_get_config(verbose_on=False)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Strip Emotiv prefixes (e.g., 'EEG.AF3' -> 'AF3') and pick the 14 EEG channels
    rename_map = {ch: ch.replace("EEG.", "").strip() for ch in raw.ch_names}
    raw.rename_channels(rename_map)

    available_eeg = [ch for ch in config["EPOCX_CHANNELS"] if ch in raw.ch_names]
    raw.pick(available_eeg)

    # Scale check: EmotivPRO often exports raw microvolts (>0.1) instead of Volts
    if np.ptp(raw.get_data()) > 0.1:
        raw.apply_function(lambda x: x * 1e-6, picks="eeg")

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    sfreq = raw.info["sfreq"]
    t_max = raw.times[-1]

    if verbose_on:
        print(f"\n[Step 2] Loaded EDF: {os.path.basename(edf_path)}")
        print(f"  • Channels: {len(raw.ch_names)} | Sampling Rate: {sfreq:.1f} Hz | Duration: {t_max:.2f}s")

    phase_windows = {}
    markers_df = None
    is_drink_trial = False

    if marker_csv_path and os.path.exists(marker_csv_path):
        markers_df = pd.read_csv(marker_csv_path)
        if not markers_df.empty and "latency" in markers_df.columns:
            markers_df["type_clean"] = markers_df["type"].astype(str).str.strip().str.lower()
            markers_df["adj_latency"] = (
                markers_df["latency"] - config["KEYSTROKE_OFFSET_S"]
            ).clip(lower=0.5)
            markers_df = markers_df.sort_values("adj_latency").reset_index(drop=True)

            def find_marker_time(keywords):
                for kw in keywords:
                    m = markers_df[markers_df["type_clean"].str.contains(kw, na=False)]
                    if not m.empty:
                        return float(m["adj_latency"].iloc[0])
                return None

            t_reach = find_marker_time(["reach", "take", "open"])
            t_hold = find_marker_time(["hold", "touch", "feed", "pour", "drink"])
            t_chew = find_marker_time(["chew", "masticat"])
            t_swallow = find_marker_time(["swallow", "gulp"])

            # Fallback if positional markers used
            if t_swallow is None and len(markers_df) >= 1:
                t_swallow = float(markers_df["adj_latency"].iloc[-1])

            is_drink_trial = (t_chew is None)
            t_after_hold = t_swallow if is_drink_trial else t_chew

            # If reach missing but hold present
            if t_reach is None and t_hold is not None:
                t_reach = max(0.5, t_hold - 3.0)

            # Ensure valid ordering: reach < hold < after_hold
            has_reach = (t_reach is not None)
            has_hold = (t_hold is not None and (t_after_hold is None or t_hold < t_after_hold))

            t_first_event = t_reach if has_reach else (t_hold if has_hold else (t_chew if t_chew is not None else t_swallow))
            if t_first_event is not None and t_first_event > 0.0:
                phase_windows["2_Pre_Touch_Rest"] = (0.0, t_first_event)

            ann_onsets, ann_durs, ann_descs = [], [], []

            if has_reach and has_hold and t_hold > t_reach:
                phase_windows["2a_Reach_Phase"] = (t_reach, t_hold)
                ann_onsets.append(t_reach)
                ann_durs.append(t_hold - t_reach)
                ann_descs.append("Reach")

            if has_hold and t_after_hold is not None and t_after_hold > t_hold:
                hold_label_full = "2b_Hold_Drink_Full" if is_drink_trial else "2b_Hold_Food_Full"
                hold_label_core = "2c_Hold_Drink_Core" if is_drink_trial else "2c_Hold_Food_Core"

                t_hold_core_start = min(t_hold + config["HOLD_TRANSITION_S"], t_after_hold - 1.0)
                if t_hold_core_start <= t_hold:
                    t_hold_core_start = t_hold + 0.5 * (t_after_hold - t_hold)

                phase_windows["2b_Hold_Transition"] = (t_hold, t_hold_core_start)
                phase_windows[hold_label_core] = (t_hold_core_start, t_after_hold)
                phase_windows[hold_label_full] = (t_hold, t_after_hold)

                ann_onsets.extend([t_hold, t_hold_core_start])
                ann_durs.extend([t_hold_core_start - t_hold, t_after_hold - t_hold_core_start])
                ann_descs.extend(["Hold_Transition", "Hold_Core"])

            if not is_drink_trial and t_chew is not None and t_swallow is not None and t_swallow > t_chew:
                phase_windows["3_Chew_Phase"] = (t_chew, t_swallow)
                ann_onsets.append(t_chew)
                ann_durs.append(t_swallow - t_chew)
                ann_descs.append("Chew")

            if t_swallow is not None:
                t_swallow_end = min(t_max, t_swallow + config["SWALLOW_WINDOW_S"])
                t_post_clean_start = min(t_swallow_end + config["POST_SWALLOW_SETTLE_S"], t_max - 10.0)

                phase_windows["3b_Swallow_Phase"] = (t_swallow, t_swallow_end)
                phase_windows["4a_Post_Swallow_Settling"] = (t_swallow_end, t_post_clean_start)
                phase_windows["4b_Post_Swallow_Clean_Recovery"] = (t_post_clean_start, t_max)
                phase_windows["4_Post_Swallow_Full"] = (t_swallow_end, t_max)

                ann_onsets.extend([t_swallow, t_swallow_end])
                ann_durs.extend([t_swallow_end - t_swallow, t_max - t_swallow_end])
                ann_descs.extend(["Swallow", "Post_Swallow"])

            valid_anns = [(o, d, desc) for o, d, desc in zip(ann_onsets, ann_durs, ann_descs) if d > 0]
            if valid_anns:
                raw.set_annotations(mne.Annotations(
                    onset=[v[0] for v in valid_anns],
                    duration=[v[1] for v in valid_anns],
                    description=[v[2] for v in valid_anns]
                ))

            if verbose_on:
                trial_type_str = "LIQUID DRINK (No Chew Phase)" if is_drink_trial else "SOLID FOOD (With Chew Phase)"
                print(f"  • Detected Trial Type: {trial_type_str}")
                print(f"  • Adjusted Phase Windows (-{config['KEYSTROKE_OFFSET_S']}s keystroke offset applied):")
                for p_name, (p_start, p_end) in phase_windows.items():
                    print(f"      - {p_name:30s}: {p_start:6.2f}s -> {p_end:6.2f}s (Duration: {p_end - p_start:6.2f}s)")
    elif verbose_on:
        print("  • [Notice] No valid marker CSV provided or file is empty.")

    return raw, phase_windows, is_drink_trial


# =====================================================================
# STEP 3: PREPROCESSING, CONSERVATIVE ASR, AND FASTICA DENOISING
# =====================================================================
def _apply_conservative_subspace_asr(raw_target, raw_calib, cutoff_k=25.0):
    """
    Built-in sliding-window Artifact Subspace Reconstruction (ASR) limiter.
    Uses the clean baseline covariance matrix to attenuate transient high-variance
    subspace spikes exceeding k * std without distorting low-amplitude brain signals.
    """
    data_calib = raw_calib.get_data()
    n_ch, _ = data_calib.shape
    sfreq = raw_target.info["sfreq"]

    # Compute baseline channel-wise RMS distribution across 1s windows
    win_samples = int(sfreq)
    n_wins = max(1, data_calib.shape[1] // win_samples)
    calib_trimmed = data_calib[:, :n_wins * win_samples].reshape(n_ch, n_wins, win_samples)
    rms_per_win = np.sqrt(np.mean(calib_trimmed ** 2, axis=-1))

    # Eigendecomposition of clean calibration covariance
    cov_calib = np.cov(data_calib) + 1e-12 * np.eye(n_ch)
    eigvals_c, eigvecs_c = linalg.eigh(cov_calib)

    # Project calibration windows into eigenspace to estimate threshold T_i = mu_i + k * sigma_i
    proj_calib = np.tensordot(eigvecs_c.T, calib_trimmed, axes=([1], [0]))
    proj_rms = np.sqrt(np.mean(proj_calib ** 2, axis=-1))
    thresh_eig = (np.mean(proj_rms, axis=1) + cutoff_k * np.std(proj_rms, axis=1)) ** 2

    # Sliding window (0.5s window, 0.25s step) reconstruction on target data
    data_out = raw_target.get_data().copy()
    step = max(1, int(0.25 * sfreq))
    wlen = max(2, int(0.50 * sfreq))
    n_modified = 0
    total_wins = 0

    for start in range(0, data_out.shape[1] - wlen, step):
        seg = data_out[:, start:start + wlen]
        total_wins += 1
        cov_seg = np.cov(seg) + 1e-12 * np.eye(n_ch)
        eigvals_s, eigvecs_s = linalg.eigh(cov_seg)

        # Project thresholds onto current segment eigenvectors
        proj_thresh = np.sum((eigvecs_c.T @ eigvecs_s) ** 2 * thresh_eig[:, None], axis=0)
        exceed_mask = eigvals_s > proj_thresh

        if np.any(exceed_mask):
            n_modified += 1
            # Soft-attenuate exceeding subspace components back to threshold ceiling
            scale_factors = np.ones(n_ch)
            scale_factors[exceed_mask] = np.sqrt(proj_thresh[exceed_mask] / np.maximum(eigvals_s[exceed_mask], 1e-30))
            S_mat = eigvecs_s @ np.diag(scale_factors) @ eigvecs_s.T
            data_out[:, start:start + wlen] = S_mat @ seg

    raw_cleaned = raw_target.copy()
    raw_cleaned._data = data_out
    pct_mod = (n_modified / max(1, total_wins)) * 100.0
    return raw_cleaned, pct_mod


def step3_preprocess_and_denoise(raw, raw_rest_calib=None, config=None, verbose_on=True):
    """
    Step 3: Applies 50 Hz Notch filter, 1-40 Hz FIR Bandpass filter,
    Conservative ASR (k=25) calibrated on resting data, and FastICA restricted
    to clear ocular/muscle components (WITHOUT spherical spline interpolation
    on the sparse 14-channel montage).
    """
    if config is None:
        config = step1_get_config(verbose_on=False)

    raw_work = raw.copy()

    # 1. Notch Filter (50 Hz power-line hum) & Bandpass Filter (1.0 - 40.0 Hz)
    raw_work.notch_filter(freqs=config["NOTCH_FREQ"], verbose=False)
    raw_work.filter(l_freq=config["L_FREQ"], h_freq=config["H_FREQ"], fir_design="firwin", verbose=False)

    # 2. Conservative ASR Subspace Spike Suppression (k = 25)
    calib_source = raw_rest_calib if raw_rest_calib is not None else raw_work
    raw_asr, asr_pct = _apply_conservative_subspace_asr(
        raw_work, calib_source, cutoff_k=config["ASR_CUTOFF_K"]
    )

    # 3. Conservative FastICA (Prune at most 1-2 clear ocular blink components on AF3/AF4)
    n_ch = len(raw_asr.ch_names)
    ica = mne.preprocessing.ICA(
        n_components=n_ch - 1,
        method="fastica",
        random_state=42,
        max_iter="auto"
    )
    ica.fit(raw_asr, verbose=False)

    eog_picks = [ch for ch in config["EOG_PROXY_CHANNELS"] if ch in raw_asr.ch_names]
    excluded_comps = []
    if eog_picks:
        eog_inds, eog_scores = ica.find_bads_eog(
            raw_asr, ch_name=eog_picks, threshold=2.5, verbose=False
        )
        if len(eog_inds) > 0:
            # Keep at most MAX_ICA_EOG_COMPS highest-scoring blink components
            max_scores = np.max(np.abs(eog_scores), axis=0)
            sorted_eog = sorted([int(i) for i in eog_inds], key=lambda idx: max_scores[idx], reverse=True)
            excluded_comps.extend(sorted_eog[:config["MAX_ICA_EOG_COMPS"]])

    ica.exclude = sorted(list(set(excluded_comps)))
    raw_clean = ica.apply(raw_asr.copy(), verbose=False)

    # 4. Optional Post-ICA Common Average Reference (CAR)
    if config.get("APPLY_POST_ICA_CAR", False):
        raw_clean.set_eeg_reference("average", projection=False, verbose=False)
        ref_str = "Post-ICA CAR"
    else:
        ref_str = "Native CMS/DRL (Zero Chewing Leakage)"

    if verbose_on:
        print("[Step 3] Preprocessing, ASR & Conservative FastICA Complete")
        print(f"  • Bad-Channel Spline Interpolation : Disabled (Protects 14-Ch Spatial Independence)")
        print(f"  • Conservative ASR (k={config['ASR_CUTOFF_K']:.0f})          : {asr_pct:.1f}% transient windows attenuated")
        print(f"  • FastICA Ocular Components Pruned : {ica.exclude if ica.exclude else 'None'}")
        print(f"  • EEG Reference Scheme             : {ref_str}")

    return raw_clean


# =====================================================================
# STEP 6: METRIC CALCULATION (PSD, FAA, DELTA-FAA, Z-SCORE, EI)
# (Defined before Steps 4 & 5 so both functions can call it directly)
# =====================================================================
def step6_calculate_neural_metrics(psds_uv2, freqs, ch_names, duration_s, config=None,
                                   baseline_ref=None, zscore_ref_stats=None,
                                   phase_label="", verbose_on=True):
    """
    Step 6: Integrates Theta, Alpha, and Beta band powers (uV^2) and computes:
      - Frontal/Central & Global Alpha/Beta Ratio (Calmness / Relaxation)
      - NASA Engagement Index: Beta / (Alpha + Theta)
      - Frontal Alpha Asymmetry (FAA): ln(Right Alpha) - ln(Left Alpha) on F4/F3 & AF4/AF3
      - Relative Delta Metrics (Phase - 2_Pre_Touch_Rest) & Within-Subject Z-Scores
    """
    if config is None:
        config = step1_get_config(verbose_on=False)

    band_powers = {}
    for band_name, (f_low, f_high) in config["BANDS"].items():
        idx = (freqs >= f_low) & (freqs <= f_high)
        power_per_ch = integrate.trapezoid(psds_uv2[:, idx], freqs[idx], axis=-1)
        band_powers[band_name] = dict(zip(ch_names, np.maximum(power_per_ch, 1e-12)))

    # Global Mean Band Powers
    alpha_global = float(np.mean(list(band_powers["Alpha"].values())))
    beta_global = float(np.mean(list(band_powers["Beta"].values())))
    ab_ratio_global = alpha_global / max(beta_global, 1e-12)

    # Frontal & Fronto-Central ROI (AF3, AF4, F3, F4, FC5, FC6)
    fc_chs = [c for c in config["EI_CHANNELS"] if c in ch_names]
    theta_fc = float(np.mean([band_powers["Theta"][c] for c in fc_chs]))
    alpha_fc = float(np.mean([band_powers["Alpha"][c] for c in fc_chs]))
    beta_fc = float(np.mean([band_powers["Beta"][c] for c in fc_chs]))

    ab_ratio_fc = alpha_fc / max(beta_fc, 1e-12)
    engagement_index = beta_fc / max(alpha_fc + theta_fc, 1e-12)

    # Frontal Alpha Asymmetry (FAA): ln(Right) - ln(Left)
    faa_f4_f3 = float(np.log(band_powers["Alpha"]["F4"]) - np.log(band_powers["Alpha"]["F3"]))
    faa_af4_af3 = float(np.log(band_powers["Alpha"]["AF4"]) - np.log(band_powers["Alpha"]["AF3"]))

    metrics = {
        "Phase": phase_label,
        "Duration_s": round(float(duration_s), 2),
        "Theta_FC_uV2": theta_fc,
        "Alpha_FC_uV2": alpha_fc,
        "Beta_FC_uV2": beta_fc,
        "Alpha_Beta_Ratio_Global": ab_ratio_global,
        "Alpha_Beta_Ratio_FC": ab_ratio_fc,
        "Engagement_Index": engagement_index,
        "FAA_F4_F3": faa_f4_f3,
        "FAA_AF4_AF3": faa_af4_af3,
        "Delta_Alpha_Beta_FC": 0.0,
        "Delta_Engagement_Index": 0.0,
        "Delta_FAA_F4_F3": 0.0,
        "Delta_FAA_AF4_AF3": 0.0,
        "Z_Alpha_Beta_FC": np.nan,
        "Z_Engagement_Index": np.nan
    }

    # Relative Delta Shift against within-trial 2_Pre_Touch_Rest anchor
    if baseline_ref is not None:
        metrics["Delta_Alpha_Beta_FC"] = ab_ratio_fc - baseline_ref["Alpha_Beta_Ratio_FC"]
        metrics["Delta_Engagement_Index"] = engagement_index - baseline_ref["Engagement_Index"]
        metrics["Delta_FAA_F4_F3"] = faa_f4_f3 - baseline_ref["FAA_F4_F3"]
        metrics["Delta_FAA_AF4_AF3"] = faa_af4_af3 - baseline_ref["FAA_AF4_AF3"]

    # Z-Score Standardization against Baseline Rest distribution (mu, sigma)
    if zscore_ref_stats is not None:
        ab_mu, ab_sd = zscore_ref_stats.get("ab_fc", (ab_ratio_fc, 1.0))
        ei_mu, ei_sd = zscore_ref_stats.get("ei", (engagement_index, 1.0))
        metrics["Z_Alpha_Beta_FC"] = (ab_ratio_fc - ab_mu) / max(ab_sd, 1e-6)
        metrics["Z_Engagement_Index"] = (engagement_index - ei_mu) / max(ei_sd, 1e-6)

    if verbose_on and phase_label:
        print(f"  [Step 6 Metrics | {phase_label:30s}] "
              f"EI: {engagement_index:.4f} | A/B (FC): {ab_ratio_fc:.4f} | "
              f"FAA (F4/F3): {faa_f4_f3:+.4f} (Δ: {metrics['Delta_FAA_F4_F3']:+.4f})")

    return metrics


# =====================================================================
# STEP 4: WELCH'S PSD ON REST PROFILE & STATIONARY TASK PHASES
# =====================================================================
def _extract_welch_psd_segment(raw_seg, config):
    sfreq = raw_seg.info["sfreq"]
    n_samples = len(raw_seg.times)
    n_per_seg = int(min(2.0 * sfreq, n_samples))
    n_overlap = int(n_per_seg // 2)

    spectrum = raw_seg.compute_psd(
        method="welch",
        fmin=config["L_FREQ"],
        fmax=config["H_FREQ"],
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        window="hann",
        verbose=False
    )
    psds, freqs = spectrum.get_data(return_freqs=True)
    return psds * 1e12, freqs  # Convert V^2/Hz -> uV^2/Hz


def step4_compute_stationary_welch(raw_rest_clean, raw_task_clean, phase_windows, config=None, verbose_on=True):
    """
    Step 4: Computes Welch's PSD on the 2-minute Baseline Rest EDF, each stationary
    task phase (including Core Hold [3s-end] and Clean Post-Swallow Recovery [30s-end]),
    and slices the Post-Swallow phase into continuous 30-second windows.
    """
    if config is None:
        config = step1_get_config(verbose_on=False)

    if verbose_on:
        print("\n[Step 4] Computing Welch's PSD on Rest Profile, Stationary Phases & 30s Windows...")

    # 1. Estimate Baseline Rest distribution across 10s sub-epochs for Z-score standardization
    rest_dur = raw_rest_clean.times[-1]
    z_ab_vals, z_ei_vals = [], []
    for w_s in np.arange(0.0, max(10.0, rest_dur - 10.0), 10.0):
        sub_r = raw_rest_clean.copy().crop(tmin=w_s, tmax=min(rest_dur, w_s + 10.0))
        p_uv2, f_hz = _extract_welch_psd_segment(sub_r, config)
        m_r = step6_calculate_neural_metrics(
            p_uv2, f_hz, sub_r.ch_names, 10.0, config, verbose_on=False
        )
        z_ab_vals.append(m_r["Alpha_Beta_Ratio_FC"])
        z_ei_vals.append(m_r["Engagement_Index"])

    zscore_ref_stats = {
        "ab_fc": (float(np.mean(z_ab_vals)), float(np.std(z_ab_vals) + 1e-6)),
        "ei":    (float(np.mean(z_ei_vals)), float(np.std(z_ei_vals) + 1e-6))
    }

    # 2. Full Baseline Rest EDF Metrics
    psds_base, freqs_base = _extract_welch_psd_segment(raw_rest_clean, config)
    base_metrics = step6_calculate_neural_metrics(
        psds_base, freqs_base, raw_rest_clean.ch_names, rest_dur, config,
        zscore_ref_stats=zscore_ref_stats, phase_label="1_Baseline_Rest_EDF", verbose_on=verbose_on
    )

    stationary_rows = [base_metrics]
    pre_touch_ref = None

    # 3. Task Phase Metrics (anchored to 2_Pre_Touch_Rest for relative Delta calculations)
    if not phase_windows:
        full_seg = raw_task_clean.copy()
        p_full, f_full = _extract_welch_psd_segment(full_seg, config)
        m_full = step6_calculate_neural_metrics(
            p_full, f_full, full_seg.ch_names, full_seg.times[-1], config,
            baseline_ref=None, zscore_ref_stats=zscore_ref_stats,
            phase_label="2_Full_Task_Unmarked", verbose_on=verbose_on
        )
        stationary_rows.append(m_full)

    for phase_name, (t_start, t_end) in phase_windows.items():
        if t_end - t_start < 0.5:
            continue
        seg = raw_task_clean.copy().crop(tmin=t_start, tmax=t_end)
        psds_seg, freqs_seg = _extract_welch_psd_segment(seg, config)

        m_seg = step6_calculate_neural_metrics(
            psds_seg, freqs_seg, seg.ch_names, t_end - t_start, config,
            baseline_ref=pre_touch_ref, zscore_ref_stats=zscore_ref_stats,
            phase_label=phase_name, verbose_on=verbose_on
        )
        if phase_name == "2_Pre_Touch_Rest":
            pre_touch_ref = m_seg
            m_seg["Delta_Alpha_Beta_FC"] = 0.0
            m_seg["Delta_Engagement_Index"] = 0.0
            m_seg["Delta_FAA_F4_F3"] = 0.0
            m_seg["Delta_FAA_AF4_AF3"] = 0.0

        stationary_rows.append(m_seg)

    df_stationary = pd.DataFrame(stationary_rows)

    # 4. Continuous 30-Second Windows across Post-Swallow Phase (or full task if unmarked)
    post_windows_rows = []
    if "4_Post_Swallow_Full" in phase_windows:
        t_post_start, t_post_end = phase_windows["4_Post_Swallow_Full"]
    elif not phase_windows:
        t_post_start, t_post_end = 0.0, raw_task_clean.times[-1]
    else:
        t_post_start, t_post_end = None, None

    if t_post_start is not None:
        bin_s = config["POST_SWALLOW_BIN_S"]
        win_start = t_post_start
        win_idx = 1

        while win_start + bin_s <= t_post_end + 0.5:
            win_end = min(t_post_end, win_start + bin_s)
            if win_end - win_start < 5.0:
                break
            seg_30s = raw_task_clean.copy().crop(tmin=win_start, tmax=win_end)
            p_30s, f_30s = _extract_welch_psd_segment(seg_30s, config)

            m_30s = step6_calculate_neural_metrics(
                p_30s, f_30s, seg_30s.ch_names, win_end - win_start, config,
                baseline_ref=pre_touch_ref, zscore_ref_stats=zscore_ref_stats,
                phase_label="", verbose_on=False
            )
            m_30s["Window_Index"] = win_idx
            m_30s["Start_Time_s"] = round(win_start, 2)
            m_30s["End_Time_s"] = round(win_end, 2)
            m_30s["Time_Post_Swallow_s"] = round((win_start - t_post_start) + (bin_s / 2.0), 1)
            m_30s["Is_Clean_Recovery_Win"] = (win_idx >= 2)  # Window 1 = oral clearing buffer
            post_windows_rows.append(m_30s)

            win_start += bin_s
            win_idx += 1

    df_post_30s = pd.DataFrame(post_windows_rows)
    if verbose_on and not df_post_30s.empty:
        print(f"  • Extracted {len(df_post_30s)} continuous {config['POST_SWALLOW_BIN_S']}s windows "
              f"(Window 1 = Oral Settling Buffer; Windows 2–{len(df_post_30s)} = Clean Recovery).")

    return df_stationary, df_post_30s, pre_touch_ref


# =====================================================================
# STEP 5: ERP EXTRACTION, GFP & CLEAN-ANCHORED MORLET CWT
# =====================================================================
def step5_compute_transient_erp_cwt(raw_task_clean, phase_windows, pre_touch_ref=None,
                                    config=None, verbose_on=True):
    """
    Step 5: Performs Single-Trial ERP (1-20 Hz), Global Field Power (GFP), and
    Morlet Wavelet CWT (2-40 Hz) on transient phases ('Reach', 'Hold', 'Swallow').
    CRITICAL UPGRADE: Anchors CWT decibel (dB) spectrograms to a clean 5-second
    window inside 2_Pre_Touch_Rest instead of [-1.0s, 0.0s] to prevent movement bleed!
    """
    if config is None:
        config = step1_get_config(verbose_on=False)

    if not phase_windows:
        if verbose_on:
            print("\n[Step 5] Unmarked task (no phase markers found). Skipping transient ERP/CWT analysis.")
        return {}

    sfreq = raw_task_clean.info["sfreq"]
    freqs = np.arange(2.0, 40.5, 0.5)
    n_cycles = freqs / 2.0

    # 1. Compute Clean Reference CWT Spectrum from 2_Pre_Touch_Rest (e.g. last 5s before Reach)
    t_pre_start, t_pre_end = phase_windows.get("2_Pre_Touch_Rest", (0.0, 6.0))
    c_end = max(2.0, t_pre_end - 1.0)
    c_start = max(0.0, c_end - config["CLEAN_CWT_BASE_DUR_S"])
    raw_cwt_base = raw_task_clean.copy().crop(tmin=c_start, tmax=c_end)
    base_epochs = mne.make_fixed_length_epochs(
        raw_cwt_base, duration=c_end - c_start, preload=True, verbose=False
    )
    power_base = base_epochs.compute_tfr(
        method="morlet", freqs=freqs, n_cycles=n_cycles,
        use_fft=True, return_itc=False, average=True, verbose=False
    )
    # Shape: (n_channels, n_freqs, 1)
    clean_base_spectrum = np.mean(power_base.data, axis=-1, keepdims=True)

    # Identify transient phases to analyze
    hold_key = "2b_Hold_Drink_Full" if "2b_Hold_Drink_Full" in phase_windows else "2b_Hold_Food_Full"
    transient_targets = [
        ("Reach",   phase_windows.get("2a_Reach_Phase")),
        ("Hold",    phase_windows.get(hold_key)),
        ("Swallow", phase_windows.get("3b_Swallow_Phase"))
    ]

    cwt_results = {}
    raw_erp_lp = raw_task_clean.copy().filter(l_freq=None, h_freq=20.0, verbose=False)

    for label, win_tuple in transient_targets:
        if win_tuple is None:
            continue
        onset_s, end_s = win_tuple
        post_dur_s = round(end_s - onset_s, 3)
        if post_dur_s <= 0.5:
            continue

        if verbose_on:
            print(f"\n[Step 5] Running ERP, GFP & Clean-Anchored Morlet CWT on '{label}' "
                  f"(Onset: {onset_s:.2f}s, Window: [-{config['PRE_EVENT_ERP_S']}s, +{post_dur_s:.2f}s])...")

        onset_sample = int(round(onset_s * sfreq))
        events = np.array([[onset_sample, 0, 1]])
        event_id = {label: 1}

        # A. Single-Trial ERP & Global Field Power (GFP)
        epochs_erp = mne.Epochs(
            raw_erp_lp, events=events, event_id=event_id,
            tmin=-config["PRE_EVENT_ERP_S"], tmax=post_dur_s,
            baseline=(-config["PRE_EVENT_ERP_S"], 0.0), preload=True, verbose=False
        )
        evoked = epochs_erp.average()
        times = evoked.times
        erp_fc_uv = evoked.copy().pick(config["EI_CHANNELS"]).get_data() * 1e6
        gfp_uv = np.std(evoked.get_data() * 1e6, axis=0)

        # B. Morlet CWT Time-Frequency Power
        epochs_tfr = mne.Epochs(
            raw_task_clean, events=events, event_id=event_id,
            tmin=-config["PRE_EVENT_ERP_S"], tmax=post_dur_s,
            baseline=None, preload=True, verbose=False
        )
        power = epochs_tfr.compute_tfr(
            method="morlet", freqs=freqs, n_cycles=n_cycles,
            use_fft=True, return_itc=False, average=True, verbose=False
        )

        # Normalize CWT Spectrogram in dB against Clean Pre-Touch Rest Anchor!
        power_db_matrix = 10.0 * np.log10(
            np.maximum(power.data, 1e-30) / np.maximum(clean_base_spectrum, 1e-30)
        )

        # Integrate instantaneous band powers (uV^2) across time t
        cwt_uv2 = power.data * 1e12
        ch_names = power.ch_names
        fc_idx = [ch_names.index(c) for c in config["EI_CHANNELS"] if c in ch_names]

        inst_bands = {}
        for b_name, (f_low, f_high) in config["BANDS"].items():
            f_mask = (freqs >= f_low) & (freqs <= f_high)
            b_ch_t = integrate.trapezoid(cwt_uv2[:, f_mask, :], freqs[f_mask], axis=1)
            inst_bands[b_name] = np.mean(b_ch_t[fc_idx, :], axis=0)

        ei_t = inst_bands["Beta"] / np.maximum(inst_bands["Alpha"] + inst_bands["Theta"], 1e-12)
        ab_t = inst_bands["Alpha"] / np.maximum(inst_bands["Beta"], 1e-12)

        f_alpha = (freqs >= 8.0) & (freqs <= 13.0)
        a_f4 = np.maximum(integrate.trapezoid(cwt_uv2[ch_names.index("F4"), f_alpha, :], freqs[f_alpha], axis=0), 1e-12)
        a_f3 = np.maximum(integrate.trapezoid(cwt_uv2[ch_names.index("F3"), f_alpha, :], freqs[f_alpha], axis=0), 1e-12)
        a_af4 = np.maximum(integrate.trapezoid(cwt_uv2[ch_names.index("AF4"), f_alpha, :], freqs[f_alpha], axis=0), 1e-12)
        a_af3 = np.maximum(integrate.trapezoid(cwt_uv2[ch_names.index("AF3"), f_alpha, :], freqs[f_alpha], axis=0), 1e-12)

        faa_f4_f3_t = np.log(a_f4) - np.log(a_f3)
        faa_af4_af3_t = np.log(a_af4) - np.log(a_af3)

        df_inst = pd.DataFrame({
            "Time_s": times,
            "GFP_uV": gfp_uv,
            "Theta_FC_uV2": inst_bands["Theta"],
            "Alpha_FC_uV2": inst_bands["Alpha"],
            "Beta_FC_uV2": inst_bands["Beta"],
            "Alpha_Beta_Ratio_FC": ab_t,
            "Engagement_Index": ei_t,
            "FAA_F4_F3": faa_f4_f3_t,
            "FAA_AF4_AF3": faa_af4_af3_t
        })

        # For Hold, evaluate both Full [0s, end] and Settled Core [3.0s, end]
        if label == "Hold" and post_dur_s > config["HOLD_TRANSITION_S"] + 1.0:
            eval_mask = times >= config["HOLD_TRANSITION_S"]
            mask_desc = f"Core Hold [+{config['HOLD_TRANSITION_S']}s to +{post_dur_s:.1f}s]"
        else:
            eval_mask = times >= 0.0
            mask_desc = f"Active [0.0s to +{post_dur_s:.1f}s]"

        sub_eval = df_inst[eval_mask]
        pk_idx = sub_eval["GFP_uV"].idxmax()
        pk_gfp = float(sub_eval.loc[pk_idx, "GFP_uV"])
        pk_lat = float(sub_eval.loc[pk_idx, "Time_s"])

        pre_f4 = pre_touch_ref["FAA_F4_F3"] if pre_touch_ref else 0.0
        mean_faa = float(sub_eval["FAA_F4_F3"].mean())
        delta_faa = mean_faa - pre_f4

        if verbose_on:
            print(f"  • [{label} | {mask_desc}] Mean GFP: {sub_eval['GFP_uV'].mean():.2f} uV "
                  f"(Peak: {pk_gfp:.2f} uV @ +{pk_lat:.2f}s) | "
                  f"Mean EI: {sub_eval['Engagement_Index'].mean():.4f} | "
                  f"Mean FAA (F4/F3): {mean_faa:+.4f} (Δ: {delta_faa:+.4f})")

        cwt_results[label] = {
            "df_inst": df_inst,
            "times": times,
            "freqs": freqs,
            "erp_fc_uv": erp_fc_uv,
            "gfp_uv": gfp_uv,
            "fc_tfr_db": np.mean(power_db_matrix[fc_idx, :, :], axis=0),
            "peak_gfp_uv": pk_gfp,
            "peak_latency_s": pk_lat,
            "eval_window_desc": mask_desc,
            "mean_ei": float(sub_eval["Engagement_Index"].mean()),
            "peak_ei": float(sub_eval["Engagement_Index"].max()),
            "mean_ab": float(sub_eval["Alpha_Beta_Ratio_FC"].mean()),
            "peak_ab": float(sub_eval["Alpha_Beta_Ratio_FC"].max()),
            "mean_faa_f4_f3": mean_faa,
            "delta_faa_f4_f3": delta_faa
        }

    return cwt_results


# =====================================================================
# STEP 7: FINAL RESULT DISPLAY, CSV EXPORT & DIAGNOSTIC PLOTS
# =====================================================================
def step7_display_and_export_results(participant_id, condition_name, df_stationary,
                                     df_post_30s, cwt_results, output_dir=".",
                                     config=None, verbose_on=True):
    """
    Step 7: Prints the consolidated summary tables (Stationary Welch's PSD,
    Clean-Anchored CWT Transient Spikes, and Post-Swallow 30s Windows), exports
    all CSV files, and saves high-resolution diagnostic figures.
    """
    if config is None:
        config = step1_get_config(verbose_on=False)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Save CSV outputs
    df_stationary.to_csv(os.path.join(output_dir, "stationary_phases_welch.csv"), index=False)
    if not df_post_30s.empty:
        df_post_30s.to_csv(os.path.join(output_dir, "post_swallow_30s_timeseries.csv"), index=False)
    for label, res in cwt_results.items():
        res["df_inst"].to_csv(
            os.path.join(output_dir, f"{label.lower()}_cwt_erp_instantaneous.csv"), index=False
        )

    if verbose_on:
        print("\n" + "=" * 78)
        print(f"[Step 7] FINAL SUMMARY: {participant_id} | Condition: {condition_name}")
        print("=" * 78)

        print("\n--- A. Stationary Phases (Welch's PSD + Relative Δ vs. Pre-Touch Rest) ---")
        disp_cols = [
            "Phase", "Duration_s", "Alpha_FC_uV2", "Beta_FC_uV2",
            "Alpha_Beta_Ratio_FC", "Delta_Alpha_Beta_FC", "Engagement_Index",
            "FAA_F4_F3", "Delta_FAA_F4_F3", "FAA_AF4_AF3", "Delta_FAA_AF4_AF3"
        ]
        print(df_stationary[disp_cols].round(4).to_string(index=False))

        if cwt_results:
            print("\n--- B. Transient Phases (Clean-Anchored Morlet CWT & Settled Core Metrics) ---")
            cwt_summary = []
            for label, res in cwt_results.items():
                cwt_summary.append({
                    "Phase": label,
                    "Eval_Window": res["eval_window_desc"],
                    "Mean_EI (Peak)": f"{res['mean_ei']:.4f} ({res['peak_ei']:.4f})",
                    "Mean_A/B (Peak)": f"{res['mean_ab']:.4f} ({res['peak_ab']:.4f})",
                    "Mean_FAA_F4/F3 (Δ)": f"{res['mean_faa_f4_f3']:+.4f} ({res['delta_faa_f4_f3']:+.4f})",
                    "Peak_GFP_Latency": f"{res['peak_gfp_uv']:.2f} uV @ +{res['peak_latency_s']:.2f}s"
                })
            print(pd.DataFrame(cwt_summary).to_string(index=False))

        if not df_post_30s.empty:
            print("\n--- C. Post-Swallow 30-Second Windows (Win 1 = Settling Buffer | Win 2+ = Clean Recovery) ---")
            p_cols = [
                "Window_Index", "Start_Time_s", "End_Time_s", "Is_Clean_Recovery_Win",
                "Alpha_Beta_Ratio_FC", "Engagement_Index", "FAA_F4_F3", "Delta_FAA_F4_F3"
            ]
            print(df_post_30s[p_cols].round(4).to_string(index=False))

    # 2. Plot Post-Swallow 30s Recovery Trajectory (with Window 1 Settling Buffer shaded)
    if not df_post_30s.empty:
        base_row = df_stationary.iloc[0]
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), sharex=True)
        x_t = df_post_30s["Time_Post_Swallow_s"]

        for ax in axes:
            ax.axvspan(0, config["POST_SWALLOW_SETTLE_S"], color="#ffebee", alpha=0.45,
                       label="0–30s Oral Settling Buffer (Win 1)")

        axes[0].plot(x_t, df_post_30s["Alpha_Beta_Ratio_FC"], marker="o", color="#2b5c8f",
                     linewidth=2.2, label="Task Window (30s)")
        axes[0].axhline(base_row["Alpha_Beta_Ratio_FC"], color="#d95f02", linestyle="--",
                        label=f"Baseline Rest EDF ({base_row['Alpha_Beta_Ratio_FC']:.2f})")
        axes[0].set_ylabel("Alpha / Beta Ratio (FC)")
        axes[0].set_title(f"[{participant_id} - {condition_name}] Post-Swallow Recovery Trajectory (30s Windows)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x_t, df_post_30s["Engagement_Index"], marker="s", color="#388e3c",
                     linewidth=2.2, label="Engagement Index: β / (α + θ)")
        axes[1].axhline(base_row["Engagement_Index"], color="#d95f02", linestyle="--",
                        label=f"Baseline Rest EDF ({base_row['Engagement_Index']:.2f})")
        axes[1].set_ylabel("Engagement Index")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x_t, df_post_30s["Delta_FAA_F4_F3"], marker="^", color="#7b1fa2",
                     linewidth=2.2, label="Relative ΔFAA (F4/F3 vs. Pre-Touch Rest)")
        axes[2].axhline(0.0, color="black", linestyle=":", alpha=0.7, label="Pre-Touch Rest Anchor (0.0)")
        axes[2].set_ylabel("ΔFAA (ln F4 − ln F3)")
        axes[2].set_xlabel("Time Elapsed After Swallow Phase (Seconds)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "post_swallow_30s_trajectory.png"), dpi=300)
        plt.close()

    # 3. Plot 4-Panel ERP + GFP + Clean-Anchored CWT Spectrogram for Each Transient Phase
    for label, res in cwt_results.items():
        times = res["times"]
        freqs = res["freqs"]
        df_i = res["df_inst"]

        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
        for idx, ch in enumerate(config["EI_CHANNELS"]):
            axes[0].plot(times, res["erp_fc_uv"][idx], alpha=0.78, linewidth=1.2, label=ch)
        axes[0].axvline(0.0, color="red", linestyle="--", linewidth=1.5, label=f"{label} Onset (t=0)")
        if label == "Hold":
            axes[0].axvline(config["HOLD_TRANSITION_S"], color="#2e7d32", linestyle="--",
                            linewidth=1.5, label=f"Core Hold Start (+{config['HOLD_TRANSITION_S']}s)")
        axes[0].set_ylabel("ERP (µV)")
        axes[0].set_title(f"[{participant_id} - {condition_name}] {label} Phase — Single-Trial ERP (Frontal/Central ROI)")
        axes[0].legend(loc="upper right", ncol=4, fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, res["gfp_uv"], color="#d95f02", linewidth=2.0, label="Global Field Power (14 Ch)")
        axes[1].fill_between(times, 0, res["gfp_uv"], color="#d95f02", alpha=0.15)
        axes[1].axvline(0.0, color="red", linestyle="--", linewidth=1.5)
        if label == "Hold":
            axes[1].axvline(config["HOLD_TRANSITION_S"], color="#2e7d32", linestyle="--", linewidth=1.5)
        axes[1].scatter([res["peak_latency_s"]], [res["peak_gfp_uv"]], color="black", zorder=5,
                        label=f"Peak GFP: {res['peak_gfp_uv']:.1f} µV @ +{res['peak_latency_s']:.2f}s")
        axes[1].set_ylabel("GFP (µV)")
        axes[1].set_title(f"{label} Phase — Global Cortical Synchrony (GFP Spike Detection)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        im = axes[2].pcolormesh(times, freqs, res["fc_tfr_db"], cmap="RdBu_r",
                                shading="gouraud", vmin=-6, vmax=6)
        axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1.5)
        if label == "Hold":
            axes[2].axvline(config["HOLD_TRANSITION_S"], color="#2e7d32", linestyle="--", linewidth=1.5)
        axes[2].axhline(8.0, color="gray", linestyle=":", alpha=0.6)
        axes[2].axhline(13.0, color="gray", linestyle=":", alpha=0.6)
        axes[2].set_ylabel("Frequency (Hz)")
        axes[2].set_title(f"{label} Phase — Morlet CWT Spectrogram (dB vs. Clean Pre-Touch Rest Anchor)")
        cbar = fig.colorbar(im, ax=axes[2], orientation="vertical", pad=0.01)
        cbar.set_label("Power (dB)")

        ax4_t = axes[3].twinx()
        p1, = axes[3].plot(times, df_i["Engagement_Index"], color="#2b5c8f",
                           linewidth=2.0, label="Instantaneous Engagement Index")
        p2, = ax4_t.plot(times, df_i["Alpha_Beta_Ratio_FC"], color="#388e3c",
                         linewidth=1.8, linestyle="-.", label="Instantaneous Alpha/Beta Ratio")
        axes[3].axvline(0.0, color="red", linestyle="--", linewidth=1.5)
        if label == "Hold":
            axes[3].axvline(config["HOLD_TRANSITION_S"], color="#2e7d32", linestyle="--", linewidth=1.5)
        axes[3].set_xlabel(f"Time Relative to Adjusted {label} Marker Onset (Seconds)")
        axes[3].set_ylabel("Engagement Index", color="#2b5c8f")
        ax4_t.set_ylabel("Alpha / Beta Ratio", color="#388e3c")
        axes[3].set_title(f"{label} Phase — Instantaneous Neural Metric Spikes (CWT)")
        axes[3].legend(handles=[p1, p2], loc="upper right")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label.lower()}_cwt_erp_analysis.png"), dpi=300)
        plt.close()


# =====================================================================
# PLUG-AND-PLAY RUNNERS (SINGLE CONDITION & FULL DATASET FOLDER TREE)
# =====================================================================
def run_single_trial_pipeline(rest_edf_path, task_edf_path, task_csv_path,
                              participant_id="participant1", condition_name="food1",
                              output_dir=".", verbose_on=True):
    """Runs Steps 1-7 on a single (Rest EDF + Food/Drink Task EDF & Marker CSV) pair."""
    config = step1_get_config(verbose_on=verbose_on)

    raw_rest, _, _ = step2_load_edf_and_markers(rest_edf_path, None, config, verbose_on=verbose_on)
    raw_task, phase_windows, _ = step2_load_edf_and_markers(
        task_edf_path, task_csv_path, config, verbose_on=verbose_on
    )

    raw_rest_clean = step3_preprocess_and_denoise(raw_rest, None, config, verbose_on=verbose_on)
    raw_task_clean = step3_preprocess_and_denoise(
        raw_task, raw_rest_calib=raw_rest_clean, config=config, verbose_on=verbose_on
    )

    df_stationary, df_post_30s, pre_touch_ref = step4_compute_stationary_welch(
        raw_rest_clean, raw_task_clean, phase_windows, config, verbose_on=verbose_on
    )

    cwt_results = step5_compute_transient_erp_cwt(
        raw_task_clean, phase_windows, pre_touch_ref=pre_touch_ref,
        config=config, verbose_on=verbose_on
    )

    step7_display_and_export_results(
        participant_id, condition_name, df_stationary, df_post_30s,
        cwt_results, output_dir=output_dir, config=config, verbose_on=verbose_on
    )
    return df_stationary, df_post_30s, cwt_results


def run_dataset_folder_pipeline(dataset_root="ensemble_de_donnees", trial=False, verbose_on=True):
    """
    Pipeline runner:
    - When trial=True:
        Only runs datasets in 'testament' folder inside dataset_root (ensemble_de_donnees/testament/*).
    - When trial=False:
        Runs all datasets in dataset_root EXCEPT the 'testament' folder (e.g., ensemble_de_donnees/nouveau_lkx/*).
    """
    if trial:
        scan_root = os.path.join(dataset_root, "testament")
        participant_dirs = [
            d for d in sorted(glob.glob(os.path.join(scan_root, "*")))
            if os.path.isdir(d)
        ]
        if verbose_on:
            print("\n" + "=" * 78)
            print("[Dataset Runner] TRIAL MODE ACTIVE (trial=True)")
            print(f"  • Target directory  : {scan_root}")
            print(f"  • Participants found: {[os.path.basename(d) for d in participant_dirs]}")
            print("=" * 78)
    else:
        participant_dirs = [
            d for d in sorted(glob.glob(os.path.join(dataset_root, "*")))
            if os.path.isdir(d) and os.path.basename(d).lower() != "testament"
        ]
        if verbose_on:
            print("\n" + "=" * 78)
            print("[Dataset Runner] FULL DATASET MODE (trial=False, excluding 'testament')")
            print(f"  • Target directory  : {dataset_root}")
            print(f"  • Participants found: {[os.path.basename(d) for d in participant_dirs]}")
            print("=" * 78)

    results_all = {}
    for part_dir in participant_dirs:
        participant_id = os.path.basename(part_dir)
        rest_dir = os.path.join(part_dir, "rest")
        rest_edfs = [f for f in glob.glob(os.path.join(rest_dir, "*.edf")) if not f.endswith(".md.edf")]
        if not rest_edfs:
            if verbose_on:
                print(f"Warning: No valid rest EDF found in {rest_dir}, skipping {participant_id}.")
            continue
        rest_edf_path = rest_edfs[0]

        for cond_dir in sorted(glob.glob(os.path.join(part_dir, "*"))):
            cond_name = os.path.basename(cond_dir)
            if not os.path.isdir(cond_dir) or cond_name.lower() == "rest":
                continue
            task_edfs = [f for f in glob.glob(os.path.join(cond_dir, "*.edf")) if not f.endswith(".md.edf")]
            task_csvs = glob.glob(os.path.join(cond_dir, "*.csv"))
            if not task_edfs:
                continue
            task_edf_path = task_edfs[0]
            task_csv_path = task_csvs[0] if task_csvs else None

            out_subdir = os.path.join("results", participant_id, cond_name)
            res = run_single_trial_pipeline(
                rest_edf_path=rest_edf_path,
                task_edf_path=task_edf_path,
                task_csv_path=task_csv_path,
                participant_id=participant_id,
                condition_name=cond_name,
                output_dir=out_subdir,
                verbose_on=verbose_on
            )
            results_all[f"{participant_id}_{cond_name}"] = res

    return results_all


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process EEG Dataset Pipeline with ASR, ICA, Welch PSD, and CWT")
    parser.add_argument("--trial", action="store_true", default=False,
                        help="If set, only runs datasets in the 'testament' folder. Otherwise runs all folders except 'testament'.")
    parser.add_argument("--dataset-root", type=str, default="ensemble_de_donnees",
                        help="Root dataset directory path (default: 'ensemble_de_donnees').")
    args, _ = parser.parse_known_args()

    # Step 2 with trial parameter automatically runs the pipeline across selected dataset folders
    step2_load_edf_and_markers(trial=args.trial, dataset_root=args.dataset_root, verbose_on=True)