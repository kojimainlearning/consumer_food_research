import mne
import numpy as np
import pandas as pd
import os
import glob

# --- Configuration & Parameters ---
DATA_FOLDER = 'eeg_data'  # Folder where your CSV/EDF files are stored
RESULTS_CSV = 'eeg_results_summary.csv'

SAMPLING_FREQ = 256
EMOTIV_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
LOW_CUTOFF = 1.0
HIGH_CUTOFF = 45.0
EPOCH_LENGTH = 4.0
EPOCH_OVERLAP = 2.0


def load_eeg_data(file_path):
    """
    Load EEG data from a CSV or EDF file.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
        # Assuming the first column is time and the rest are EEG channels
        eeg_data = data[EMOTIV_CHANNELS].values.T  # Transpose to shape (n_channels, n_samples)
        info = mne.create_info(ch_names=EMOTIV_CHANNELS, sfreq=SAMPLING_FREQ, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)
    elif file_path.endswith('.edf'):
        raw = mne.io.read_raw_edf(file_path, preload=True)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
    else:
        raise ValueError("Unsupported file format. Please use CSV or EDF.")
    
    return raw

def process_eeg_data(raw):
    """
    Preprocess the EEG data: filtering, epoching, and artifact rejection.
    """
    # Filter the data
    raw_filtered = raw.copy().filter(l_freq=LOW_CUTOFF, h_freq=HIGH_CUTOFF, verbose=False)

    # Create epochs
    events = mne.make_fixed_length_events(raw, id=1, duration=EPOCH_LENGTH, overlap=EPOCH_OVERLAP)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=EPOCH_LENGTH, baseline=None, preload=True)

    # Artifact rejection
    reject_criteria = 10e-6  # 100 µV
    epochs.drop_bad(reject=reject_criteria)

    return epochs