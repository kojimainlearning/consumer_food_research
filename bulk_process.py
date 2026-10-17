import mne
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

# --- Configuration & Parameters ---
DATA_FOLDER = "ensemble_de_donnees"
RESULTS_CSV = "eeg_results_summary.csv"

SAMPLING_FREQ = 256
EMOTIV_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
LOW_CUTOFF = 1.0
HIGH_CUTOFF = 45.0
EPOCH_LENGTH = 4.0
EPOCH_OVERLAP = 2.0

def load_eeg_data(file_path, verbose=False):
    file_path = file_path if file_path != None or "" else "ensemble_de_donnees/eau/lkq_eau.edf"
    raw_eeg = mne.io.read_raw_edf(file_path, preload=True, verbose=verbose)

    new_names = {}
    for ch in raw.ch_names:
        # Take the second word if it starts with 'EEG', otherwise keep it as is
        if "EEG" in ch:
            parts = ch.split()
            if len(parts) > 1:
                new_names[ch] = parts[1]
    
    if new_names:
        raw.rename_channels(new_names, verbose=False)

    eeg_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    raw_eeg.pick(eeg_channels, verbose=verbose)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw_eeg.set_montage(montage, verbose=verbose)

    return raw_eeg

def bandpass_filter(raw_eeg, l_freq=LOW_CUTOFF, h_freq=HIGH_CUTOFF, verbose=False):
    raw_filtered = raw_eeg.copy().filter(
        l_freq=l_freq, 
        h_freq=h_freq, 
        fir_design="firwin", 
        verbose=verbose)
    return raw_filtered

def noise_removal_using_ica(raw_filtered, ica_components, verbose=False):
    ica = mne.preprocessing.ICA(n_components=ica_components, random_state=42, verbose=verbose, max_iter=1000)
    ica.fit(raw_filtered)

    eog_indices, _ = ica.find_bads_eog(raw_filtered, ch_name=['F7', 'F8'], verbose=verbose
    muscle_indices, _ = ica.find_bads_muscle(raw_filtered, verbose=verbose)
    ica.exclude = list(set(eog_indices + muscle_indices))

    raw_clean = ica.apply(raw_filtered.copy(), verbose=verbose)

    return raw_clean


def process_eeg_data(raw):
    """
    Preprocess the EEG data: filtering, epoching, and artifact rejection.
    """
    # Band-pass filter the data
    raw_filtered = raw.copy().filter(l_freq=LOW_CUTOFF, h_freq=HIGH_CUTOFF, verbose=False)
    
    # ICA for artifact removal
    ica = mne.preprocessing.ICA(n_components=14, random_state=97, max_iter=1000)
    ica.fit(raw_filtered, verbose=False)
    eog_indices, _ = ica.find_bads_eog(raw_filtered, ch_name=["F7", "F8"], verbose=False)
    muscle_indices, _ = ica.find_bads_muscle(raw_filtered, verbose=False)
    ica.exclude = list(set(eog_indices + muscle_indices))
    
    raw_clean = ica.apply(raw_filtered.copy(), verbose=False)
    
    # Epoching
    events = mne.make_fixed_length_events(raw_clean, id=1, duration=EPOCH_LENGTH, overlap=EPOCH_OVERLAP)
    epochs = mne.Epochs(raw_clean, events, tmin=0, tmax=EPOCH_LENGTH, baseline=None, preload=True, verbose=False)

    # Artifact rejection
    reject_criteria = 130e-6  
    max_bad_channels_allowed = 3

    clean_epochs_list = []

    for i in range(len(epochs)):
        epoch = epochs[i].get_data()[0]
        ptp_amplitude = np.ptp(epoch, axis=1) 
        
        bad_ch_idx = np.where(ptp_amplitude > reject_criteria)[0]
        bad_channels = [epochs.ch_names[idx] for idx in bad_ch_idx]
        
        if len(bad_channels) == 0:
            clean_epochs_list.append(epochs[i].copy())
            
        elif len(bad_channels) <= max_bad_channels_allowed:
            temp_epoch = epochs[i].copy()
            temp_epoch.info["bads"] = bad_channels
            temp_epoch.interpolate_bads(reset_bads=True, verbose=False)
            
            clean_epochs_list.append(temp_epoch)
        
        
    if len(clean_epochs_list) == 0:
        return None, None, 0  # No clean epochs left after artifact rejection
    
    clean_epochs = mne.concatenate_epochs(clean_epochs_list, verbose=False)

    # Feature Extraction
    spectrum = clean_epochs.compute_psd(method="welch", fmin=1, fmax=45, n_fft=256, n_overlap=128)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    def get_band_power(psds, freqs, band):
        band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.mean(psds[:, :, band_idx], axis=2)
    
    theta_power = get_band_power(psds, freqs, (4, 8))
    alpha_power = get_band_power(psds, freqs, (8, 13))
    beta_power = get_band_power(psds, freqs, (13, 30))
    
    # Calculate FAA and Relaxation Index
    ch_names = clean_epochs.ch_names
    f3_idx, f4_idx = ch_names.index("F3"), ch_names.index("F4")
    faa_per_epoch = np.log(alpha_power[:, f4_idx] + 1e-12) - np.log(alpha_power[:, f3_idx] + 1e-12)
    
    o1_idx, o2_idx = ch_names.index("O1"), ch_names.index("O2")
    calmness_per_epoch = np.mean(alpha_power[:, [o1_idx, o2_idx]], axis=1) / (np.mean(beta_power[:, [o1_idx, o2_idx]], axis=1) + 1e-12)
    
    final_faa = np.mean(faa_per_epoch)
    final_calmness = np.mean(calmness_per_epoch)
    num_clean_epochs = len(clean_epochs)
    
    return final_faa, final_calmness, num_clean_epochs

def main():
    # Find all EDF files in the data folder and its subdirectories
    files = glob.glob(os.path.join(DATA_FOLDER, "**", "*.edf"), recursive=True)
    
    if not files:
        print(f"No .edf files found inside '{DATA_FOLDER}'. Please check the folder path.")
        return
    
    results = []
    print(f"Found {len(files)} EDF files to process. Starting pipeline...\n")
    
    for file_path in tqdm(files, desc="Processing EDF files", unit="file"):
        filename = os.path.basename(file_path)
        name_without_ext = filename.replace(".edf", "")
                
        # Split by underscore, but only separate the name
        parts = name_without_ext.split("_")
        if len(parts) >= 2:
            participant_id = parts[0]
            food_type = "_".join(parts[1:])
        else:
            participant_id = "Unknown"
            food_type = name_without_ext
            
        print(f"Processing: {filename} (Participant: {participant_id}, Food: {food_type})...")
        
        try:
            raw = load_eeg_data(file_path)
            faa, calmness, n_epochs = process_eeg_data(raw)
            
            if faa is not None:
                print(f"  -> Success! FAA: {faa:.4f}, Calmness: {calmness:.4f}, Clean Epochs: {n_epochs}")
                results.append({
                    "Participant": participant_id,
                    "Food": food_type,
                    "FAA_Happiness": faa,
                    "Calmness_Ratio": calmness,
                    "Clean_Epochs": n_epochs
                })
            else:
                print(f"  -> FAILED: No clean epochs survived for this file.")
                results.append({
                    "Participant": participant_id,
                    "Food": food_type,
                    "FAA_Happiness": None,
                    "Calmness_Ratio": None,
                    "Clean_Epochs": 0
                })
                
        except Exception as e:
            print(f"  -> ERROR processing this file: {e}")
            
    # Save all results to a CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nPipeline complete! All results saved to '{RESULTS_CSV}'.")
    
if __name__ == "__main__":
    main()