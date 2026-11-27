# Decoding Comfort Food with Consumer Emotional Responses Using EEG

**Author:** Lin Sheng Jie

## Overview

This project investigates consumer emotional and cognitive responses to comfort foods and beverages using 14-channel electroencephalography (EEG) recorded with the Emotiv EPOC X headset alongside self-reported sensory ratings. It provides an automated preprocessing, spectral/time-frequency feature extraction and cohort visualisation workflow implemented in [complete_process.ipynb](complete_process.ipynb).

---

## Required Libraries

All Python dependencies are listed in [requirements.txt](requirements.txt):

- **EEG & Signal Processing:** `mne`, `scipy`, `numpy`
- **Data Handling:** `pandas`
- **Visualisation:** `matplotlib`
- **Notebook Execution:** `jupyter`, `ipykernel`
- **Statistical Calculation:** `statsmodels`

Install them with:

```bash
pip install -r requirements.txt
```

---

## Included Dataset (`ensemble_de_donnees/`)

The repository includes EmotivPRO EEG recordings (`.edf`), interval event markers (`_intervalMarker.csv`), session metadata (`.json`), motion/device streams (`.md.edf`) and self-report surveys (`*_survey.txt`):

| Subset | Conditions Recorded | Surveys |
| :--- | :--- | :--- |
| **Main Cohort (`n = 8`)** | `rest` (baseline), `eau` (Water), `the_vert` (Green Tea), `biscuit` (Biscuit), `chocolat` (Chocolate) | 0–10 ratings for **Delicious**, **Happy**, and **Calm** (`the_vert`, `biscuit`, `chocolat`) |
| **Pilot Test (`ensemble_de_donnees/testament/`, `n = 2`)** |`rest`, `biscuit`, `chocolat`, `the_vert` | None |

Split recordings caused by Bluetooth headset disconnections are automatically discovered, chronologically stitched and treated as continuous single-shot trials after boundary-aware filtering.

---

## Experiment & Processing Framework

1. **Acquisition & Montage:** 14-channel Emotiv EPOC X (`AF3`, `F7`, `F3`, `FC5`, `T7`, `P7`, `O1`, `O2`, `P8`, `T8`, `FC6`, `F4`, `F8`, `AF4`; 128 Hz) using native reference.
2. **Phase Segmentation:** Keystroke markers (corrected by `-2.0 s` reaction offset) divide each tasting trial into `Pre-Touch Rest` (baseline), `Reach`, `Hold` (3 s transition + core), `Chew` (food trials), `Swallow`, `Post-Swallow Settling` (0–30 s oral motion buffer), and `Post-Swallow Clean Recovery` (30 s–end aftertaste), plus consecutive 30-second post-swallow bins.
3. **Denoising Pipeline:** Bandpass FIR filter -> ASR -> Infomax ICA
4. **Feature Extraction:**
   - **Frontal Alpha Asymmetry (FAA)** $\ln(\alpha_{F4}) - \ln(\alpha_{F3})$ (and `AF4`/`AF3`)
   - **Relaxation Index:** Frontal-central Alpha/Beta ratio ($\alpha_{\text{FC}} / \beta_{\text{FC}}$)
   - **Engagement Index (EI):** $\beta_{\text{FC}} / (\alpha_{\text{FC}} + \theta_{\text{FC}})$
   - **Baseline-Referenced Deltas ($\Delta$):** Difference between each task/recovery phase and `Pre-Touch Rest`
   - **Transient Dynamics & Topomaps:** Single-trial ERPs, GFP, Morlet CWT spectrograms and 14-channel topomaps.

---

## How to Run `complete_process.ipynb`

1. Install dependencies from `requirements.txt` and select the corresponding Python/Jupyter kernel.
2. Open [complete_process.ipynb](complete_process.ipynb).
3. **Step-by-step walkthrough:** Run the cells from the top to trace each processing stage inline on the reference trial.
4. **Full cohort analysis & visualisations:** Run the `Run Full Dataset Pipeline & Inspect Cohort Summary` cell followed by the four `Group-Averaged Cohort Visualisations` cells to process all 8 participants and export cohort CSV tables and summary plots.

---

## Summary of Final Results (`n = 8` Participants, 32 Trials)

The table below summarises cohort-averaged self-reported survey ratings (`0–10`) and **Post-Swallow Clean Recovery** baseline-referenced EEG deltas (relative to `Pre-Touch Rest`, $\text{Mean} \pm \text{SD}$):

| Condition | Delicious (0–10) | Happy (0–10) | Calm (0–10) | $\Delta$ FAA (`F4/F3`) | $\Delta$ Relaxation Index | $\Delta$ Engagement Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Water (`eau`)** | — | — | — | $-0.0069 \pm 0.1212$ | $+0.1427 \pm 0.2260$ | $-0.0865 \pm 0.1325$ |
| **Green Tea (`the_vert`)** | $6.06 \pm 1.52$ | $5.50 \pm 1.93$ | $6.56 \pm 0.82$ | $+0.1079 \pm 0.1937$ | $+0.0662 \pm 0.2951$ | $-0.0801 \pm 0.2983$ |
| **Biscuit (`biscuit`)** | $6.00 \pm 1.20$ | $5.38 \pm 0.74$ | $6.25 \pm 0.89$ | $+0.0412 \pm 0.3949$ | $+0.3705 \pm 0.5549$ | $-0.0930 \pm 0.1175$ |
| **Chocolate (`chocolat`)** | $8.06 \pm 0.78$ | $7.63 \pm 1.06$ | $5.63 \pm 1.30$ | $-0.0383 \pm 0.0855$ | $+0.2670 \pm 0.5402$ | $+0.0806 \pm 0.5734$ |

### Key Takeaways

- **Hedonic Pleasure vs. Arousal (Chocolate):** Chocolate scored highest in **Delicious** ($8.06 \pm 0.78$) and **Happy** ($7.63 \pm 1.06$) but lowest in **Calm** ($5.63 \pm 1.30$), and was the only condition with a positive mean post-swallow $\Delta$ Engagement Index ($+0.0806$), reflecting sustained post-consumption cognitive/hedonic engagement.
- **Calmness & Approach Motivation (Green Tea):** Green Tea achieved the highest **Calm** rating ($6.56 \pm 0.82$) and the highest mean positive $\Delta$ FAA ($+0.1079 \pm 0.1937$), with its relaxation and FAA responses emerging after the initial 90 s post-swallow period.
- **Post-Swallow Cortical Relaxation (Solid Comfort Foods):** Biscuit ($+0.3705 \pm 0.5549$) and Chocolate ($+0.2670 \pm 0.5402$) produced the largest mean increases in frontal Relaxation Index compared to Water ($+0.1427$) and Green Tea ($+0.0662$), accompanied by stronger posterior (`O1`/`O2`/`P7`/`P8`) Alpha activity in cohort topomaps.
