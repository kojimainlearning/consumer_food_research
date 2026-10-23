# %% [markdown]
# # Part 1: Configuration, Constants & Dataset Directory Discovery

# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import mne

def get_pipeline_config(verbose_on: bool = True) -> dict:
    """
    Step 1A: Returns a centralized dictionary of hardware channels,
    frequency bands, and experimental timing parameters.
    """
    config = {
        # Emotiv EPOC X 14-channel 10-20 layout
        "EPOCX_CHANNELS": [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
        ],
        # Frontal & Fronto-Central ROI for NASA Engagement Index
        "EI_CHANNELS": ["AF3", "AF4", "F3", "F4", "FC5", "FC6"],
        
        # Filtering & Denoising Parameters
        "L_FREQ": 1.0,              # High-pass cutoff (Hz) for DC drift removal
        "H_FREQ": 40.0,             # Low-pass cutoff (Hz) for high-freq EMG suppression
        "NOTCH_FREQ": 50.0,         # Powerline notch filter (50 Hz)
        "ERP_LOWPASS_HZ": 20.0,     # Smoothing low-pass filter for ERP waveforms
        
        # Experimental Timing Parameters
        "KEYSTROKE_OFFSET_S": 2.0,  # Shift manual keystroke markers 2.0s earlier
        "SWALLOW_WINDOW_S": 5.0,    # Fixed duration for the Swallow phase
        "POST_SWALLOW_WIN_S": 30.0, # 30s window size for Post-Swallow trajectory
        "POST_SWALLOW_STEP_S": 30.0,# Step size (set to 10.0 for overlapping sliding windows)
        "PRE_BASELINE_S": 1.0,      # Pre-event baseline (-1.0s to 0.0s) for ERP & CWT
        
        # EEG Frequency Bands (Hz)
        "BANDS": {
            "Theta": (4.0, 8.0),
            "Alpha": (8.0, 13.0),
            "Beta":  (13.0, 30.0)
        },
        
        # Morlet CWT Frequency Grid (2.0 Hz to 40.0 Hz in 0.5 Hz steps)
        "CWT_FREQS": np.arange(2.0, 40.5, 0.5)
    }
    
    if verbose_on:
        print("=" * 65)
        print("[Step 1] Pipeline Configuration Initialized")
        print(f"  • EPOC X Channels ({len(config['EPOCX_CHANNELS'])}): {', '.join(config['EPOCX_CHANNELS'])}")
        print(f"  • Engagement Index ROI: {', '.join(config['EI_CHANNELS'])}")
        print(f"  • Filter Passband: {config['L_FREQ']}–{config['H_FREQ']} Hz (Notch: {config['NOTCH_FREQ']} Hz)")
        print(f"  • Marker Shift: -{config['KEYSTROKE_OFFSET_S']}s | Swallow Window: {config['SWALLOW_WINDOW_S']}s | Post-Swallow Bin: {config['POST_SWALLOW_WIN_S']}s")
        print("=" * 65)
        
    return config


def discover_dataset_files(dataset_root: str = "ensemble_de_donnees", verbose_on: bool = True) -> dict:
    """
    Step 1B: Scans the hierarchical dataset directory:
      dataset / participantX / {rest, food1, food2, ...}
    Automatically pairs the main .edf file (ignoring .md.edf) and the Marker .csv file.
    """
    root_path = Path(dataset_root)
    dataset_map = {}
    
    if not root_path.exists():
        if verbose_on:
            print(f"[Step 1 Warning] Directory '{dataset_root}' not found. Ready for manual file paths.")
        return dataset_map

    for part_dir in sorted(root_path.iterdir()):
        if not part_dir.is_dir():
            continue
        p_id = part_dir.name
        dataset_map[p_id] = {"rest": None, "foods": {}}
        
        for cond_dir in sorted(part_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            
            # Match main .edf (strictly exclude .md.edf hardware metadata files)
            edf_candidates = [
                f for f in cond_dir.glob("*.edf")
                if not f.name.lower().endswith(".md.edf")
            ]
            # Match either _InternalMarker.csv or _intervalMarker.csv
            csv_candidates = [
                f for f in cond_dir.glob("*.csv")
                if "marker" in f.name.lower()
            ]
            
            file_bundle = {
                "edf": str(edf_candidates[0]) if edf_candidates else None,
                "csv": str(csv_candidates[0]) if csv_candidates else None
            }
            
            if cond_dir.name.lower() == "rest":
                dataset_map[p_id]["rest"] = file_bundle
            else:
                dataset_map[p_id]["foods"][cond_dir.name] = file_bundle

    if verbose_on:
        print(f"\n[Step 1] Discovered {len(dataset_map)} Participant(s) in '{dataset_root}':")
        for p_id, info in dataset_map.items():
            rest_status = "Found" if (info["rest"] and info["rest"]["edf"]) else "Missing"
            food_list = list(info["foods"].keys())
            print(f"  • {p_id}: Rest EDF = [{rest_status}] | Food Trials ({len(food_list)}) = {food_list}")
            
    return dataset_map


# %% [markdown]
# # Part 2: Load EDF Recording & Parse Marker CSV

# %%
def load_emotiv_edf_and_markers(
    edf_path: str,
    csv_path: str = None,
    config: dict = None,
    is_rest_file: bool = False,
    verbose_on: bool = True
) -> tuple:
    """
    Step 2: Loads the EmotivPRO EDF file, isolates the 14 EPOC X channels,
    normalizes voltage units to Volts, applies the 10-20 montage, and calculates
    the keystroke-adjusted phase windows from the marker CSV file.
    """
    if config is None:
        config = get_pipeline_config(verbose_on=False)
        
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Strip any 'EEG.' prefixes exported by EmotivPRO
    rename_map = {ch: ch.replace("EEG.", "").strip() for ch in raw.ch_names}
    raw.rename_channels(rename_map)
    
    # Select the 14 EPOC X EEG channels
    available_eeg = [ch for ch in config["EPOCX_CHANNELS"] if ch in raw.ch_names]
    raw.pick(available_eeg)
    
    # Ensure signal is scaled in Volts (if peak-to-peak > 0.1, raw file is in microvolts)
    if np.ptp(raw.get_data()) > 0.1:
        raw.apply_function(lambda x: x * 1e-6, picks="eeg")
        
    # Attach standard 10-20 electrode coordinates
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")
    
    sfreq = raw.info["sfreq"]
    total_duration_s = raw.times[-1]
    phase_windows = {}
    
    if not is_rest_file and csv_path and os.path.exists(csv_path):
        df_markers = pd.read_csv(csv_path)
        if not df_markers.empty and "type" in df_markers.columns:
            df_markers["type"] = df_markers["type"].astype(str).str.strip()
            offset = config["KEYSTROKE_OFFSET_S"]
            df_markers["adj_latency"] = (df_markers["latency"] - offset).clip(lower=0.0)
            
            # Safely extract marker timings (Touch is optional)
            touch_series = df_markers.loc[df_markers["type"].str.lower() == "touch", "adj_latency"]
            chew_series = df_markers.loc[df_markers["type"].str.lower() == "chew", "adj_latency"]
            swallow_series = df_markers.loc[df_markers["type"].str.lower() == "swallow", "adj_latency"]
            
            t_touch = float(touch_series.iloc[0]) if not touch_series.empty else None
            t_chew = float(chew_series.iloc[0]) if not chew_series.empty else None
            t_swallow = float(swallow_series.iloc[0]) if not swallow_series.empty else None
            
            # Build phase windows dynamically
            if t_touch is not None:
                phase_windows["Pre_Touch_Rest"] = (0.0, t_touch)
                if t_chew is not None and t_chew > t_touch:
                    phase_windows["Touch"] = (t_touch, t_chew)
            elif t_chew is not None:
                # If Touch marker was omitted, pre-chew baseline is Pre_Touch_Rest
                phase_windows["Pre_Touch_Rest"] = (0.0, t_chew)
                
            if t_chew is not None and t_swallow is not None and t_swallow > t_chew:
                phase_windows["Chew"] = (t_chew, t_swallow)
                
            if t_swallow is not None:
                t_swallow_end = min(t_swallow + config["SWALLOW_WINDOW_S"], total_duration_s)
                phase_windows["Swallow"] = (t_swallow, t_swallow_end)
                if total_duration_s > t_swallow_end:
                    phase_windows["Post_Swallow"] = (t_swallow_end, total_duration_s)
            
            # Embed MNE Annotations onto the Raw object for available phases
            annot_onsets, annot_durations, annot_descriptions = [], [], []
            for p_name, (t0, t1) in phase_windows.items():
                if t1 > t0:
                    annot_onsets.append(t0)
                    annot_durations.append(t1 - t0)
                    annot_descriptions.append(p_name)
                    
            if annot_onsets:
                raw.set_annotations(
                    mne.Annotations(onset=annot_onsets, duration=annot_durations, description=annot_descriptions)
                )
            
    if verbose_on:
        print(f"\n[Step 2] Loaded EDF: {Path(edf_path).name}")
        print(f"  • Channels: {len(raw.ch_names)} | Sampling Rate: {sfreq} Hz | Duration: {total_duration_s:.2f}s")
        if phase_windows:
            print(f"  • Adjusted Phase Windows (-{config['KEYSTROKE_OFFSET_S']}s offset applied):")
            for p_name, (t0, t1) in phase_windows.items():
                print(f"      - {p_name:<15}: {t0:7.2f}s  -> {t1:7.2f}s  (Duration: {t1 - t0:6.2f}s)")
        elif not is_rest_file:
            print("  • [Notice] No valid food phase markers found. Running continuous analysis without ERP segmentation.")
                
    return raw, phase_windows


# %% [markdown]
# # Part 3: Bandpass Filtering, Bad-Channel Interpolation & ICA Denoising

# %%
def preprocess_and_denoise_eeg(
    raw: mne.io.BaseRaw,
    config: dict = None,
    verbose_on: bool = True
) -> mne.io.BaseRaw:
    """
    Step 3: Multi-stage artifact removal pipeline to preserve physiological EEG
    signals while suppressing ocular blinks and jaw/temporalis EMG noise:
      1. 50 Hz Notch filter (powerline noise removal)
      2. 1.0 - 40.0 Hz Zero-Phase FIR Bandpass filter
      3. Automated flat/noisy channel Z-score detection & spherical spline interpolation
      4. Common Average Reference (CAR)
      5. FastICA with automated EOG (AF3/AF4) and Muscle (EMG) component exclusion
    """
    if config is None:
        config = get_pipeline_config(verbose_on=False)
        
    raw_proc = raw.copy()
    
    # 1. Notch Filter at 50 Hz (if Nyquist allows)
    if raw_proc.info["sfreq"] / 2.0 > config["NOTCH_FREQ"]:
        raw_proc.notch_filter(freqs=config["NOTCH_FREQ"], verbose=False)
        
    # 2. Bandpass Filter (1 - 40 Hz)
    raw_proc.filter(
        l_freq=config["L_FREQ"],
        h_freq=config["H_FREQ"],
        fir_design="firwin",
        verbose=False
    )
    
    # 3. Automated Bad-Channel Detection & Interpolation (Robust Z-score > 4.0)
    data_stds = np.std(raw_proc.get_data(), axis=1)
    median_std = np.median(data_stds)
    mad_std = np.median(np.abs(data_stds - median_std)) + 1e-12
    robust_z = np.abs(data_stds - median_std) / (1.4826 * mad_std)
    
    bad_channels = [
        raw_proc.ch_names[i] for i, z in enumerate(robust_z)
        if z > 4.0 or data_stds[i] < 1e-9
    ]
    if bad_channels and len(bad_channels) < len(raw_proc.ch_names) // 2:
        raw_proc.info["bads"] = bad_channels
        raw_proc.interpolate_bads(reset_bads=True, verbose=False)
        
    # 4. Common Average Reference (CAR)
    raw_proc.set_eeg_reference("average", projection=False, verbose=False)
    
    # 5. FastICA for Blink (EOG) and Jaw/Facial Muscle (EMG) Removal
    n_comp = max(len(raw_proc.ch_names) - len(bad_channels) - 1, 2)
    ica = mne.preprocessing.ICA(
        n_components=n_comp,
        method="fastica",
        random_state=42,
        max_iter="auto"
    )
    ica.fit(raw_proc, verbose=False)
    
    excluded_Setting = []
    # Identify ocular artifacts via frontal polar electrodes AF3 & AF4
    eog_proxies = [ch for ch in ["AF3", "AF4"] if ch in raw_proc.ch_names]
    if eog_proxies:
        eog_inds, _ = ica.find_bads_eog(raw_proc, ch_name=eog_proxies, threshold=2.5, verbose=False)
        excluded_Setting.extend(eog_inds)
        
    # Identify high-frequency temporalis/masseter muscle artifacts via spectral slope
    try:
        muscle_inds, _ = ica.find_bads_muscle(raw_proc, threshold=0.6, verbose=False)
        excluded_Setting.extend(muscle_inds)
    except Exception:
        pass
        
    ica.exclude = sorted(list(set(excluded_Setting)))
    raw_clean = ica.apply(raw_proc.copy(), verbose=False)
    
    if verbose_on:
        print("\n[Step 3] Preprocessing & ICA Denoising Complete")
        print(f"  • Interpolated Bad Channels: {bad_channels if bad_channels else 'None'}")
        print(f"  • FastICA Components Excluded (EOG + EMG): {ica.exclude if ica.exclude else 'None'}")
        
    return raw_clean


# %% [markdown]
# # Part 6 (Core Engine): Metric Calculation (FAA, Alpha/Beta Ratio, Engagement Index)
# *(Defined before Parts 4 & 5 so both Welch's PSD and Morlet CWT can call it directly)*

# %%
def calculate_neural_metrics(
    band_powers: dict,
    config: dict = None,
    label: str = "Segment",
    verbose_on: bool = True
) -> dict:
    """
    Step 6: Centralized mathematical calculator for:
      - Frontal Alpha Asymmetry (FAA): ln(Right Alpha) - ln(Left Alpha) on F4/F3 and AF4/AF3
      - Alpha / Beta Ratio: Global (14 ch) and Frontal-Central ROI
      - NASA Engagement Index: Beta / (Alpha + Theta) averaged across AF3, AF4, F3, F4, FC5, FC6
    Supports both scalar band powers (from Welch's PSD) and 1D time-series arrays (from Morlet CWT).
    """
    if config is None:
        config = get_pipeline_config(verbose_on=False)
        
    avail_channels = list(band_powers["Alpha"].keys())
    ei_channels = [ch for ch in config["EI_CHANNELS"] if ch in avail_channels]
    eps = 1e-12
    
    # 1. Global Mean Band Powers (across all 14 channels)
    theta_global = np.mean([band_powers["Theta"][ch] for ch in avail_channels], axis=0)
    alpha_global = np.mean([band_powers["Alpha"][ch] for ch in avail_channels], axis=0)
    beta_global  = np.mean([band_powers["Beta"][ch]  for ch in avail_channels], axis=0)
    
    # 2. Frontal & Fronto-Central Mean Band Powers (AF3, AF4, F3, F4, FC5, FC6)
    theta_fc = np.mean([band_powers["Theta"][ch] for ch in ei_channels], axis=0)
    alpha_fc = np.mean([band_powers["Alpha"][ch] for ch in ei_channels], axis=0)
    beta_fc  = np.mean([band_powers["Beta"][ch]  for ch in ei_channels], axis=0)
    
    # 3. Ratios & NASA Engagement Index
    alpha_beta_global = alpha_global / (beta_global + eps)
    alpha_beta_fc     = alpha_fc / (beta_fc + eps)
    engagement_index  = beta_fc / (alpha_fc + theta_fc + eps)
    
    # 4. Frontal Alpha Asymmetry (FAA)
    faa_f4_f3 = (
        np.log(np.maximum(band_powers["Alpha"]["F4"], eps)) -
        np.log(np.maximum(band_powers["Alpha"]["F3"], eps))
    )
    faa_af4_af3 = (
        np.log(np.maximum(band_powers["Alpha"]["AF4"], eps)) -
        np.log(np.maximum(band_powers["Alpha"]["AF3"], eps))
    )
    
    metrics = {
        "Theta_FC_uV2": theta_fc,
        "Alpha_FC_uV2": alpha_fc,
        "Beta_FC_uV2": beta_fc,
        "Alpha_Beta_Ratio_Global": alpha_beta_global,
        "Alpha_Beta_Ratio_FC": alpha_beta_fc,
        "FAA_F4_F3": faa_f4_f3,
        "FAA_AF4_AF3": faa_af4_af3,
        "Engagement_Index": engagement_index
    }
    
    if verbose_on:
        if np.ndim(engagement_index) == 0:
            print(
                f"  [Step 6 Metrics | {label:<22}] "
                f"EI: {engagement_index:6.4f} | "
                f"Alpha/Beta (FC): {alpha_beta_fc:6.4f} | "
                f"FAA (F4/F3): {faa_f4_f3:+7.4f}"
            )
        else:
            print(
                f"  [Step 6 CWT Metrics | {label:<18}] "
                f"Peak EI: {np.max(engagement_index):6.4f} (Mean: {np.mean(engagement_index):6.4f}) | "
                f"Peak Alpha/Beta: {np.max(alpha_beta_fc):6.4f} | "
                f"Peak FAA: {np.max(faa_f4_f3):+7.4f}"
            )
            
    return metrics


# %% [markdown]
# # Part 4: Welch's PSD on Rest Profile, Chew & Post-Swallow Phase

# %%
def extract_welch_band_powers(raw_segment: mne.io.BaseRaw, config: dict) -> dict:
    """
    Helper for Step 4: Computes Welch's PSD (2.0s Hanning window, 50% overlap)
    and integrates absolute band power (uV^2) per channel using the trapezoidal rule.
    """
    sfreq = raw_segment.info["sfreq"]
    n_per_seg = int(min(2.0 * sfreq, len(raw_segment.times)))
    n_overlap = int(n_per_seg // 2)
    
    spectrum = raw_segment.compute_psd(
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
    psds_uv2 = psds * 1e12  # Convert V^2/Hz to uV^2/Hz
    ch_names = raw_segment.ch_names
    
    band_powers = {}
    for band_name, (f_low, f_high) in config["BANDS"].items():
        idx = (freqs >= f_low) & (freqs <= f_high)
        power_per_ch = integrate.trapezoid(psds_uv2[:, idx], freqs[idx], axis=-1)
        band_powers[band_name] = dict(zip(ch_names, power_per_ch))
        
    return band_powers


def analyze_welch_rest_and_post_swallow(
    raw_rest_clean: mne.io.BaseRaw,
    raw_task_clean: mne.io.BaseRaw,
    phase_windows: dict,
    config: dict = None,
    verbose_on: bool = True
) -> tuple:
    """
    Step 4: Applies Welch's PSD to:
      1. Baseline Rest EDF profile
      2. Pre-Touch Rest, Chew, and Full Post-Swallow phases
      3. Continuous 30-second time windows across the Post-Swallow recovery phase
    """
    if config is None:
        config = get_pipeline_config(verbose_on=False)
        
    if verbose_on:
        print("\n[Step 4] Computing Welch's PSD on Rest Profile & Task Segments...")
        
    # A. Stationary Phase Comparisons
    stationary_segments = {
        "1_Baseline_Rest_EDF": raw_rest_clean,
    }
    
    if not phase_windows:
        # No markers available: process the whole task recording
        stationary_segments["2_Full_Task_Unmarked"] = raw_task_clean
    else:
        if "Pre_Touch_Rest" in phase_windows:
            t0, t1 = phase_windows["Pre_Touch_Rest"]
            if t1 > t0:
                stationary_segments["2_Pre_Touch_Rest"] = raw_task_clean.copy().crop(tmin=t0, tmax=t1)
        if "Touch" in phase_windows:
            t0, t1 = phase_windows["Touch"]
            if t1 > t0:
                stationary_segments["2b_Touch_Phase"] = raw_task_clean.copy().crop(tmin=t0, tmax=t1)
        if "Chew" in phase_windows:
            t0, t1 = phase_windows["Chew"]
            if t1 > t0:
                stationary_segments["3_Chew_Phase"] = raw_task_clean.copy().crop(tmin=t0, tmax=t1)
        if "Post_Swallow" in phase_windows:
            t0, t1 = phase_windows["Post_Swallow"]
            if t1 > t0:
                stationary_segments["4_Post_Swallow_Full"] = raw_task_clean.copy().crop(tmin=t0, tmax=t1)
    
    phase_rows = []
    for phase_label, seg in stationary_segments.items():
        bp = extract_welch_band_powers(seg, config)
        m = calculate_neural_metrics(bp, config=config, label=phase_label, verbose_on=verbose_on)
        m["Phase"] = phase_label
        m["Duration_s"] = round(seg.times[-1] - seg.times[0], 2)
        phase_rows.append(m)
        
    df_stationary_phases = pd.DataFrame(phase_rows).set_index("Phase")
    
    # B. Continuous 30-Second Windows
    # If Post_Swallow marker exists, run across post-swallow. Otherwise run across the full task.
    if "Post_Swallow" in phase_windows:
        t_win_start, t_win_end = phase_windows["Post_Swallow"]
        is_post_swallow = True
    else:
        t_win_start, t_win_end = 0.0, raw_task_clean.times[-1]
        is_post_swallow = False
        
    win_size = config["POST_SWALLOW_WIN_S"]
    step_size = config["POST_SWALLOW_STEP_S"]
    
    post_ts_rows = []
    win_start = t_win_start
    win_idx = 1
    
    while win_start + win_size <= t_win_end:
        win_end = win_start + win_size
        seg_30s = raw_task_clean.copy().crop(tmin=win_start, tmax=win_end)
        bp_30s = extract_welch_band_powers(seg_30s, config)
        
        lbl = f"Post-Swallow Win #{win_idx}" if is_post_swallow else f"Task Win #{win_idx}"
        m_30s = calculate_neural_metrics(
            bp_30s, config=config, label=lbl, verbose_on=False
        )
        m_30s["Window_Index"] = win_idx
        m_30s["Start_Time_s"] = round(win_start, 2)
        m_30s["End_Time_s"] = round(win_end, 2)
        m_30s["Time_Post_Swallow_s"] = round((win_start - t_win_start) + (win_size / 2.0), 1)
        post_ts_rows.append(m_30s)
        
        win_start += step_size
        win_idx += 1
        
    df_post_swallow_30s = pd.DataFrame(post_ts_rows)
    
    if verbose_on:
        mode_desc = "in Post-Swallow phase" if is_post_swallow else "across full task recording (unmarked)"
        print(f"  • Extracted {len(df_post_swallow_30s)} continuous {win_size}s windows {mode_desc}.")
        
    return df_stationary_phases, df_post_swallow_30s


# %% [markdown]
# # Part 5: ERP Extraction, GFP Computation & Morlet CWT (`Touch` & `Swallow`)

# %%
def analyze_transient_erp_and_cwt(
    raw_task_clean: mne.io.BaseRaw,
    phase_windows: dict,
    phase_name: str = "Touch",
    config: dict = None,
    verbose_on: bool = True
) -> dict:
    """
    Step 5: Extracts [-1.0s, +duration] around the adjusted marker onset for transient
    phases ('Touch' or 'Swallow') and computes:
      1. Single-Trial Event-Related Potential (ERP, 1-20 Hz smoothed)
      2. Global Field Power (GFP) & Peak Cortical Spike Latency
      3. Morlet Continuous Wavelet Transform (CWT, 2-40 Hz, dynamic cycles = f/2)
      4. Instantaneous FAA, Alpha/Beta Ratio, and Engagement Index time-series via Step 6
    """
    if config is None:
        config = get_pipeline_config(verbose_on=False)
        
    if not phase_windows or phase_name not in phase_windows:
        if verbose_on:
            print(f"\n[Step 5 Notice] '{phase_name}' phase not present in markers. Skipping ERP & CWT.")
        return None
        
    t_onset, t_end = phase_windows[phase_name]
    post_duration_s = t_end - t_onset
    pre_base_s = config["PRE_BASELINE_S"]
    sfreq = raw_task_clean.info["sfreq"]
    
    if verbose_on:
        print(f"\n[Step 5] Running ERP, GFP & Morlet CWT on '{phase_name}' Phase "
              f"(Onset: {t_onset:.2f}s, Window: [-{pre_base_s:.1f}s, +{post_duration_s:.2f}s])...")
        
    onset_sample = int(round(t_onset * sfreq))
    events = np.array([[onset_sample, 0, 1]])
    event_id = {phase_name: 1}
    
    # 1. ERP & Global Field Power (GFP) Extraction
    raw_erp = raw_task_clean.copy().filter(l_freq=None, h_freq=config["ERP_LOWPASS_HZ"], verbose=False)
    epochs_erp = mne.Epochs(
        raw_erp,
        events=events,
        event_id=event_id,
        tmin=-pre_base_s,
        tmax=post_duration_s,
        baseline=(-pre_base_s, 0.0),
        preload=True,
        verbose=False
    )
    evoked = epochs_erp.average()
    times = evoked.times
    
    erp_fc_uv = evoked.copy().pick(config["EI_CHANNELS"]).get_data() * 1e6
    erp_all_uv = evoked.get_data() * 1e6
    gfp_uv = np.std(erp_all_uv, axis=0)
    
    post_mask = times >= 0.0
    peak_idx = int(np.argmax(gfp_uv[post_mask]))
    peak_latency_s = float(times[post_mask][peak_idx])
    peak_gfp_uv = float(gfp_uv[post_mask][peak_idx])
    
    # 2. Morlet Continuous Wavelet Transform (CWT)
    epochs_tfr = mne.Epochs(
        raw_task_clean,
        events=events,
        event_id=event_id,
        tmin=-pre_base_s,
        tmax=post_duration_s,
        baseline=None,
        preload=True,
        verbose=False
    )
    freqs = config["CWT_FREQS"]
    n_cycles = freqs / 2.0
    
    power = epochs_tfr.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=True,
        verbose=False
    )
    
    # Integrate instantaneous band powers P_c(t) in uV^2 across Theta, Alpha, Beta
    cwt_uv2 = power.data * 1e12  # Shape: (n_channels, n_freqs, n_times)
    ch_names = power.ch_names
    
    inst_band_powers = {b: {} for b in config["BANDS"]}
    for band_name, (f_low, f_high) in config["BANDS"].items():
        f_mask = (freqs >= f_low) & (freqs <= f_high)
        band_ch_t = integrate.trapezoid(cwt_uv2[:, f_mask, :], freqs[f_mask], axis=1)
        for ch_idx, ch in enumerate(ch_names):
            inst_band_powers[band_name][ch] = band_ch_t[ch_idx, :]
            
    # 3. Call Step 6 to Compute Instantaneous Neural Metrics
    inst_metrics = calculate_neural_metrics(
        inst_band_powers, config=config, label=f"{phase_name} (CWT)", verbose_on=verbose_on
    )
    
    # 4. Baseline-Normalized Spectrogram (dB relative to [-1.0s, 0.0s])
    power_db = power.copy().apply_baseline(baseline=(-pre_base_s, 0.0), mode="logratio", verbose=False)
    power_db.data *= 10.0  # Convert log10 ratio to dB
    fc_indices = [ch_names.index(c) for c in config["EI_CHANNELS"] if c in ch_names]
    fc_tfr_db = np.mean(power_db.data[fc_indices, :, :], axis=0)
    
    df_instantaneous = pd.DataFrame({
        "Time_s": times,
        "GFP_uV": gfp_uv,
        "Theta_FC_uV2": inst_metrics["Theta_FC_uV2"],
        "Alpha_FC_uV2": inst_metrics["Alpha_FC_uV2"],
        "Beta_FC_uV2": inst_metrics["Beta_FC_uV2"],
        "Alpha_Beta_Ratio_FC": inst_metrics["Alpha_Beta_Ratio_FC"],
        "Engagement_Index": inst_metrics["Engagement_Index"],
        "FAA_F4_F3": inst_metrics["FAA_F4_F3"],
        "FAA_AF4_AF3": inst_metrics["FAA_AF4_AF3"]
    })
    
    if verbose_on:
        print(f"  • [{phase_name} Peak Cortical Response] GFP Peak = {peak_gfp_uv:.2f} uV at t = +{peak_latency_s:.3f}s")
        
    return {
        "phase_name": phase_name,
        "times": times,
        "freqs": freqs,
        "erp_fc_uv": erp_fc_uv,
        "gfp_uv": gfp_uv,
        "peak_latency_s": peak_latency_s,
        "peak_gfp_uv": peak_gfp_uv,
        "fc_tfr_db": fc_tfr_db,
        "df_instantaneous": df_instantaneous
    }


# %% [markdown]
# # Part 7: Final Result Display & Visualization

# %%
def display_and_export_results(
    participant_id: str,
    food_name: str,
    df_stationary: pd.DataFrame,
    df_post_30s: pd.DataFrame,
    cwt_touch_res: dict,
    cwt_swallow_res: dict,
    config: dict = None,
    output_dir: str = "results",
    verbose_on: bool = True
) -> None:
    """
    Step 7: Prints formatted summary tables, exports CSV results, and renders:
      - Figure 1: Post-Swallow 30-Second Continuous Trajectory vs. Rest Baseline
      - Figure 2 & 3: 4-Panel ERP + GFP + Morlet CWT + Instantaneous Metrics for Touch & Swallow
    """
    if config is None:
        config = get_pipeline_config(verbose_on=False)
        
    out_path = Path(output_dir) / participant_id / food_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Combine Stationary Phases + Transient CWT Mean/Peak Summary
    transient_summary = []
    for res in [cwt_touch_res, cwt_swallow_res]:
        if res is None:
            continue
        df_inst_post = res["df_instantaneous"][res["df_instantaneous"]["Time_s"] >= 0.0]
        transient_summary.append({
            "Phase": f"{res['phase_name']}_CWT_Mean (Peak)",
            "Duration_s": round(float(df_inst_post["Time_s"].iloc[-1]), 2),
            "Engagement_Index": f"{df_inst_post['Engagement_Index'].mean():.4f} ({df_inst_post['Engagement_Index'].max():.4f})",
            "Alpha_Beta_Ratio_FC": f"{df_inst_post['Alpha_Beta_Ratio_FC'].mean():.4f} ({df_inst_post['Alpha_Beta_Ratio_FC'].max():.4f})",
            "FAA_F4_F3": f"{df_inst_post['FAA_F4_F3'].mean():+.4f} ({df_inst_post['FAA_F4_F3'].max():+.4f})",
            "Peak_GFP_uV_at_Latency": f"{res['peak_gfp_uv']:.2f} uV @ +{res['peak_latency_s']:.2f}s"
        })
    df_transient_summary = pd.DataFrame(transient_summary)
    if not df_transient_summary.empty:
        df_transient_summary = df_transient_summary.set_index("Phase")
    
    if verbose_on:
        print("\n" + "=" * 75)
        print(f"[Step 7] FINAL SUMMARY: {participant_id} | Condition: {food_name}")
        print("=" * 75)
        print("\n--- A. Stationary Phases (Welch's PSD) ---")
        cols_show = ["Duration_s", "Theta_FC_uV2", "Alpha_FC_uV2", "Beta_FC_uV2",
                     "Alpha_Beta_Ratio_FC", "Engagement_Index", "FAA_F4_F3", "FAA_AF4_AF3"]
        print(df_stationary[cols_show].round(4).to_string())
        
        print("\n--- B. Transient Phases (ERP + Morlet CWT Mean & Peak Spike) ---")
        if not df_transient_summary.empty:
            print(df_transient_summary.to_string())
        else:
            print("  (No transient marker phases available for ERP / CWT analysis)")
        
        print("\n--- C. Continuous 30-Second Windows ---")
        if not df_post_30s.empty:
            cols_ts = ["Window_Index", "Start_Time_s", "End_Time_s",
                       "Alpha_Beta_Ratio_FC", "Engagement_Index", "FAA_F4_F3"]
            print(df_post_30s[cols_ts].round(4).to_string(index=False))
        else:
            print("  (Duration insufficient for 30s continuous windows)")
        
    # Export CSVs
    df_stationary.to_csv(out_path / "stationary_phases_welch.csv")
    if not df_post_30s.empty:
        df_post_30s.to_csv(out_path / "post_swallow_30s_timeseries.csv", index=False)
    if cwt_touch_res is not None:
        cwt_touch_res["df_instantaneous"].to_csv(out_path / "touch_cwt_erp_instantaneous.csv", index=False)
    if cwt_swallow_res is not None:
        cwt_swallow_res["df_instantaneous"].to_csv(out_path / "swallow_cwt_erp_instantaneous.csv", index=False)
    
    # -----------------------------------------------------------------
    # Plot 1: Continuous 30-Second Recovery / Task Trajectory
    # -----------------------------------------------------------------
    if not df_post_30s.empty:
        baseline_row = df_stationary.loc["1_Baseline_Rest_EDF"]
        x_time = df_post_30s["Time_Post_Swallow_s"]
        
        has_post_swallow = "4_Post_Swallow_Full" in df_stationary.index
        traj_title = "Post-Swallow Recovery Trajectory" if has_post_swallow else "Continuous Task Trajectory (Unmarked)"
        x_lbl = "Time Elapsed After Swallow Phase (Seconds)" if has_post_swallow else "Task Window Center Time (Seconds)"
        
        fig1, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
        axes[0].plot(x_time, df_post_30s["Alpha_Beta_Ratio_FC"], marker="o", color="#2b5c8f", lw=2, label="Task Window (30s)")
        axes[0].axhline(baseline_row["Alpha_Beta_Ratio_FC"], color="#d95f02", ls="--", label="Baseline (Rest EDF)")
        axes[0].set_ylabel("Alpha / Beta Ratio (FC)")
        axes[0].set_title(f"[{participant_id} - {food_name}] {traj_title} (30s Windows)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(x_time, df_post_30s["Engagement_Index"], marker="s", color="#388e3c", lw=2, label="Engagement Index: $\\beta / (\\alpha + \\theta)$")
        axes[1].axhline(baseline_row["Engagement_Index"], color="#d95f02", ls="--", label="Baseline (Rest EDF)")
        axes[1].set_ylabel("Engagement Index")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(x_time, df_post_30s["FAA_F4_F3"], marker="^", color="#7b1fa2", lw=2, label="FAA ($\\ln F4 - \\ln F3$)")
        axes[2].axhline(baseline_row["FAA_F4_F3"], color="#d95f02", ls="--", label="Baseline (Rest EDF)")
        axes[2].axhline(0.0, color="gray", ls=":", alpha=0.6)
        axes[2].set_ylabel("FAA (F4/F3)")
        axes[2].set_xlabel(x_lbl)
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig1.savefig(out_path / "post_swallow_30s_trajectory.png", dpi=300)
        plt.show()
        plt.close(fig1)
    
    # -----------------------------------------------------------------
    # Plot 2 & 3: 4-Panel ERP + GFP + Morlet CWT + Instantaneous Metrics
    # -----------------------------------------------------------------
    for res in [cwt_touch_res, cwt_swallow_res]:
        if res is None:
            continue
        p_name = res["phase_name"]
        t_arr = res["times"]
        freqs = res["freqs"]
        df_inst = res["df_instantaneous"]
        
        fig, ax = plt.subplots(4, 1, figsize=(11, 11.5), sharex=True)
        
        # Panel 1: ERP traces
        for idx, ch in enumerate(config["EI_CHANNELS"]):
            ax[0].plot(t_arr, res["erp_fc_uv"][idx], alpha=0.75, lw=1.1, label=ch)
        ax[0].axvline(0.0, color="red", ls="--", lw=1.5, label=f"{p_name} Onset (t=0)")
        ax[0].set_ylabel("ERP ($\mu$V)")
        ax[0].set_title(f"[{participant_id} - {food_name}] {p_name} Phase — Single-Trial ERP (Frontal/Central ROI)")
        ax[0].legend(loc="upper right", ncol=4, fontsize=8)
        ax[0].grid(True, alpha=0.3)
        
        # Panel 2: GFP
        ax[1].plot(t_arr, res["gfp_uv"], color="#d95f02", lw=2.0, label="Global Field Power (14 Ch)")
        ax[1].fill_between(t_arr, 0, res["gfp_uv"], color="#d95f02", alpha=0.15)
        ax[1].axvline(0.0, color="red", ls="--", lw=1.5)
        ax[1].scatter([res["peak_latency_s"]], [res["peak_gfp_uv"]], color="black", zorder=5,
                      label=f"Peak GFP: {res['peak_gfp_uv']:.1f} $\mu$V @ +{res['peak_latency_s']:.2f}s")
        ax[1].set_ylabel("GFP ($\mu$V)")
        ax[1].set_title(f"{p_name} Phase — Global Cortical Synchrony (GFP Spike Detection)")
        ax[1].legend(loc="upper right")
        ax[1].grid(True, alpha=0.3)
        
        # Panel 3: Morlet CWT Spectrogram (dB)
        im = ax[2].pcolormesh(t_arr, freqs, res["fc_tfr_db"], cmap="RdBu_r", shading="gouraud", vmin=-6, vmax=6)
        ax[2].axvline(0.0, color="black", ls="--", lw=1.5)
        ax[2].axhline(8.0, color="gray", ls=":", alpha=0.6)
        ax[2].axhline(13.0, color="gray", ls=":", alpha=0.6)
        ax[2].set_ylabel("Frequency (Hz)")
        ax[2].set_title(f"{p_name} Phase — Morlet CWT Time-Frequency Spectrogram (dB vs. -1s Baseline)")
        cbar = fig.colorbar(im, ax=ax[2], orientation="vertical", pad=0.01)
        cbar.set_label("dB")
        
        # Panel 4: Instantaneous EI & Alpha/Beta Ratio
        ax4_twin = ax[3].twinx()
        l1, = ax[3].plot(t_arr, df_inst["Engagement_Index"], color="#2b5c8f", lw=2.0, label="Instantaneous Engagement Index")
        l2, = ax4_twin.plot(t_arr, df_inst["Alpha_Beta_Ratio_FC"], color="#388e3c", lw=1.8, ls="-.", label="Instantaneous Alpha/Beta Ratio")
        ax[3].axvline(0.0, color="red", ls="--", lw=1.5)
        ax[3].set_xlabel(f"Time Relative to Adjusted {p_name} Marker Onset (Seconds)")
        ax[3].set_ylabel("Engagement Index", color="#2b5c8f")
        ax4_twin.set_ylabel("Alpha / Beta Ratio", color="#388e3c")
        ax[3].set_title(f"{p_name} Phase — Instantaneous Neural Metric Spikes (CWT)")
        ax[3].legend(handles=[l1, l2], loc="upper right")
        ax[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(out_path / f"{p_name.lower()}_cwt_erp_analysis.png", dpi=300)
        plt.show()
        plt.close(fig)


# %% [markdown]
# # Master Runner: Plug-and-Play Execution (Batch Folder or Single Trial)

# %%
def run_participant_food_pipeline(
    participant_id: str,
    food_name: str,
    rest_edf_path: str,
    food_edf_path: str,
    food_csv_path: str,
    verbose_on: bool = True
) -> dict:
    """
    Executes Steps 1 through 7 end-to-end for one participant and one food recording.
    """
    # Step 1: Configuration
    config = get_pipeline_config(verbose_on=verbose_on)
    
    # Step 2: Load Rest EDF and Food Task EDF + Marker CSV
    raw_rest, _ = load_emotiv_edf_and_markers(
        rest_edf_path, is_rest_file=True, config=config, verbose_on=verbose_on
    )
    raw_task, phase_windows = load_emotiv_edf_and_markers(
        food_edf_path, csv_path=food_csv_path, is_rest_file=False, config=config, verbose_on=verbose_on
    )
    
    # Step 3: Filter & Denoise both recordings with FastICA
    raw_rest_clean = preprocess_and_denoise_eeg(raw_rest, config=config, verbose_on=verbose_on)
    raw_task_clean = preprocess_and_denoise_eeg(raw_task, config=config, verbose_on=verbose_on)
    
    # Step 4 (& Step 6): Welch's PSD on Rest, Chew, and Post-Swallow (Full + 30s windows)
    df_stationary, df_post_30s = analyze_welch_rest_and_post_swallow(
        raw_rest_clean, raw_task_clean, phase_windows, config=config, verbose_on=verbose_on
    )
    
    # Step 5 (& Step 6): ERP, GFP & Morlet CWT on Touch and Swallow
    cwt_touch = analyze_transient_erp_and_cwt(
        raw_task_clean, phase_windows, phase_name="Touch", config=config, verbose_on=verbose_on
    )
    cwt_swallow = analyze_transient_erp_and_cwt(
        raw_task_clean, phase_windows, phase_name="Swallow", config=config, verbose_on=verbose_on
    )
    
    # Step 7: Display Tables, Plots, and Export CSVs
    display_and_export_results(
        participant_id=participant_id,
        food_name=food_name,
        df_stationary=df_stationary,
        df_post_30s=df_post_30s,
        cwt_touch_res=cwt_touch,
        cwt_swallow_res=cwt_swallow,
        config=config,
        verbose_on=verbose_on
    )
    
    return {
        "df_stationary": df_stationary,
        "df_post_30s": df_post_30s,
        "cwt_touch": cwt_touch,
        "cwt_swallow": cwt_swallow
    }


if __name__ == "__main__":
    # Automatic Batch Execution Across the 'dataset/' Folder Structure
    dataset_index = discover_dataset_files(dataset_root="ensemble_de_donnees", verbose_on=True)
    
    for participant_id, data_paths in dataset_index.items():
        if not data_paths["rest"] or not data_paths["rest"]["edf"]:
            continue
        rest_edf = data_paths["rest"]["edf"]
        
        for food_name, food_files in data_paths["foods"].items():
            if food_files["edf"]:
                run_participant_food_pipeline(
                    participant_id=participant_id,
                    food_name=food_name,
                    rest_edf_path=rest_edf,
                    food_edf_path=food_files["edf"],
                    food_csv_path=food_files["csv"],
                    verbose_on=True
                )