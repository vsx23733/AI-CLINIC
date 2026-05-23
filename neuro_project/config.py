"""Centralised configuration: paths, EEG channels, bands, window/baseline,
and the lists of subjects/videos. Imported by every other module.
No side effects on import (no I/O)."""

import os

# ---------------------------------------------------------------- paths
PROJECT_ROOT = r"E:\COURS ECOLE\COURS PGE 4\AI CLINIC"
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "deap-dataset",
                            "data_preprocessed_python")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "notebooks", "cache_full_v2")
SPEC_CACHE_DIR = os.path.join(PROJECT_ROOT, "notebooks", "cache_spec")
ARTIFACTS_DIR  = os.path.join(PROJECT_ROOT, "neuro_project", "artifacts")
MODELS_DIR     = os.path.join(ARTIFACTS_DIR, "models")

for _d in (ARTIFACTS_DIR, MODELS_DIR, SPEC_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- signal
SAMPLE_RATE      = 128           # Hz, DEAP preprocessed
WINDOW_SIZE      = 256           # 2 s
STEP_SIZE        = 16            # 0.125 s
BASELINE_SEC     = 3             # DEAP pre-stimulus to drop
BASELINE_SAMPLES = BASELINE_SEC * SAMPLE_RATE          # 384

# Welch (feature extraction)
WELCH_NPERSEG = 128              # < WINDOW_SIZE -> real averaging

# STFT (spectrograms for DL)
STFT_NPERSEG  = 64
STFT_NOVERLAP = 32
STFT_FREQ_LO  = 4.0              # DEAP is filtered 4-45 Hz
STFT_FREQ_HI  = 45.0

# ---------------------------------------------------------------- EEG / peripheral
EEG_CHANNELS = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]   # 14 ch
BAND_NAMES   = ['Theta', 'Alpha', 'LowBeta', 'HighBeta', 'Gamma']
BANDS        = [(4, 8), (8, 12), (12, 16), (16, 25), (25, 45)]

# DEAP peripheral channels (0-based)
PERIPH = {
    'hEOG': 32, 'vEOG': 33,
    'zEMG': 34, 'tEMG': 35,
    'GSR':  36, 'Resp': 37,
    'BVP':  38, 'Temp': 39,
}

# ---------------------------------------------------------------- meta
SUBJECT_LIST   = [f"{i:02d}" for i in range(1, 33)]   # '01'..'32'
N_VIDEOS       = 40
LABEL_NAMES    = ['Valence', 'Arousal', 'Dominance', 'Liking']

# ---------------------------------------------------------------- feature names
PERIPH_FEATURE_NAMES = [
    'zEMG_mean', 'zEMG_std',
    'tEMG_mean', 'tEMG_std',
    'GSR_mean', 'GSR_std', 'GSR_range', 'GSR_slope',
    'Resp_mean', 'Resp_std',
    'BVP_mean', 'BVP_std', 'BVP_range',
    'Temp_mean', 'Temp_slope',
]
EEG_FEATURE_NAMES = [f"ch{ch}_{band}" for ch in EEG_CHANNELS for band in BAND_NAMES]
ALL_FEATURE_NAMES = EEG_FEATURE_NAMES + PERIPH_FEATURE_NAMES
N_FEATURES = len(ALL_FEATURE_NAMES)            # 85


def normalise_subject_id(sid) -> str:
    """Accept '01', 1, '1' -> always returns the zero-padded 2-char string."""
    if isinstance(sid, int):
        sid = str(sid)
    return sid.zfill(2)
