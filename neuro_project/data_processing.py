import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm
import random
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# np.trapz est déprécié en numpy >= 2 ; np.trapezoid est le nouveau nom.
trapz = getattr(np, "trapezoid", np.trapz)

sns.set_style('whitegrid')

DATA_PATH = r"E:\COURS ECOLE\COURS PGE 4\AI CLINIC\data\deap-dataset\data_preprocessed_python"
CACHE_DIR = r"E:\COURS ECOLE\COURS PGE 4\AI CLINIC\notebooks\cache_full_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

EEG_CHANNELS = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]
BAND_NAMES = ['Theta', 'Alpha', 'LowBeta', 'HighBeta', 'Gamma']
BANDS = [(4, 8), (8, 12), (12, 16), (16, 25), (25, 45)]

SAMPLE_RATE   = 128
WINDOW_SIZE   = 256        # 2 s
STEP_SIZE     = 128        # 1.0 s (50% overlap w/ WINDOW_SIZE=256, was 16 / 94% overlap)
BASELINE_SEC  = 3
BASELINE_SAMPLES = BASELINE_SEC * SAMPLE_RATE   # 384 : pré-stimulus DEAP, exclu
WELCH_NPERSEG = 128        # < WINDOW_SIZE -> Welch moyenne vraiment les segments

# Canaux périphériques DEAP (indices 32..39, 0-based)
PERIPH = {
    'hEOG': 32, 'vEOG': 33,
    'zEMG': 34, 'tEMG': 35,
    'GSR':  36, 'Resp': 37,
    'BVP':  38, 'Temp': 39,
}

SUBJECT_LIST = [f'{i:02d}' for i in range(1, 33)]

def bandpower(signal, bands, sf=SAMPLE_RATE):
    """Puissance de `signal` dans chaque bande via Welch (PSD moyennee) + integration trapeze."""
    nperseg = min(WELCH_NPERSEG, len(signal))
    freqs, psd = welch(signal, sf, nperseg=nperseg)
    out = np.empty(len(bands))
    for i, (lo, hi) in enumerate(bands):
        idx = (freqs >= lo) & (freqs <= hi)
        out[i] = trapz(psd[idx], freqs[idx])
    return out


def peripheral_features(periph_window):
    """periph_window: (8, window_size) -> 15 features (zEMG,tEMG,GSR,Resp,BVP,Temp)."""
    # 0=hEOG 1=vEOG 2=zEMG 3=tEMG 4=GSR 5=Resp 6=BVP 7=Temp
    feats = []
    z = periph_window[2]
    feats += [np.mean(np.abs(z)), np.std(z)]                                  # zEMG
    t = periph_window[3]
    feats += [np.mean(np.abs(t)), np.std(t)]                                  # tEMG
    g = periph_window[4]
    feats += [np.mean(g), np.std(g), np.max(g) - np.min(g),
              np.polyfit(np.arange(len(g)), g, 1)[0]]                          # GSR
    r = periph_window[5]
    feats += [np.mean(r), np.std(r)]                                          # Resp
    b = periph_window[6]
    feats += [np.mean(b), np.std(b), np.max(b) - np.min(b)]                   # BVP
    tp = periph_window[7]
    feats += [np.mean(tp), np.polyfit(np.arange(len(tp)), tp, 1)[0]]          # Temp
    return np.array(feats)


PERIPH_FEATURE_NAMES = [
    'zEMG_mean', 'zEMG_std',
    'tEMG_mean', 'tEMG_std',
    'GSR_mean', 'GSR_std', 'GSR_range', 'GSR_slope',
    'Resp_mean', 'Resp_std',
    'BVP_mean', 'BVP_std', 'BVP_range',
    'Temp_mean', 'Temp_slope',
]
EEG_FEATURE_NAMES = [f'ch{ch}_{band}' for ch in EEG_CHANNELS for band in BAND_NAMES]
ALL_FEATURE_NAMES = EEG_FEATURE_NAMES + PERIPH_FEATURE_NAMES

def _feature_vector(eeg_trial, periph_trial, start, size):
    """85-dim feature vector (70 EEG bandpowers + 15 peripheral stats) over
    [start, start+size) of one trial. Used both for stimulus windows and for
    the pre-stimulus baseline (same shape, so they can be subtracted)."""
    eeg_feats = []
    for ch in EEG_CHANNELS:
        eeg_feats.extend(bandpower(eeg_trial[ch, start:start + size], BANDS))
    periph_feats = peripheral_features(periph_trial[:, start:start + size])
    return np.concatenate([eeg_feats, periph_feats])


def extract_subject_features(subject_id, force=False):
    """Features (EEG+periph) d'un sujet, cache .npz. Returns X(n,85), y(n,2), video_ids(n,).
    Each window's features are baseline-corrected: raw_feats - features(pre-stimulus
    baseline), to remove per-trial/per-subject offset drift (absolute bandpower/EMG
    levels vary a lot between people and sessions; the baseline never carried any
    signal downstream before this, it was just dropped)."""
    cache_file = os.path.join(CACHE_DIR, f's{subject_id}_full_v4.npz')
    if os.path.exists(cache_file) and not force:
        d = np.load(cache_file)
        return d['X'], d['y'], d['video_ids']

    with open(os.path.join(DATA_PATH, f's{subject_id}.dat'), 'rb') as f:
        sub = pickle.load(f, encoding='latin1')
    data, labels = sub['data'], sub['labels']      # (40,40,8064), (40,4)

    X_list, y_list, vid_list = [], [], []

    for trial in range(40):
        eeg    = data[trial, :32]
        periph = data[trial, 32:40]
        va     = labels[trial, :2]                 # [Valence, Arousal]

        baseline_feats = _feature_vector(eeg, periph, 0, BASELINE_SAMPLES)

        start = BASELINE_SAMPLES                   # <-- on saute la baseline pre-stimulus
        while start + WINDOW_SIZE <= eeg.shape[1]:
            feats = _feature_vector(eeg, periph, start, WINDOW_SIZE) - baseline_feats
            X_list.append(feats)
            y_list.append(va)
            vid_list.append(trial)
            start += STEP_SIZE

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    vids = np.array(vid_list, dtype=np.int16)
    np.savez_compressed(cache_file, X=X, y=y, video_ids=vids)
    return X, y, vids


def load_subject(subject_id, cache_dir=CACHE_DIR):
    path = os.path.join(cache_dir, f"s{subject_id}_full_v4.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache absent : {path}")
    d = np.load(path)
    return d["X"], d["y"], d["video_ids"]



if __name__ == "__main__":
    # Demo : tirage d'un sujet au hasard + window-level GroupKFold pour verification.
    # Cet bloc ne s'execute que si on lance "python data_processing.py" directement,
    # pas lors d'un "import" depuis un autre module du package.

    random_subject_id = f"{random.randint(1, 32):02d}"
    X, y, vids = load_subject(random_subject_id, CACHE_DIR)

    kfold_spliter = GroupKFold(n_splits=5)
    results = {}
    for i, (train_index, test_index) in enumerate(kfold_spliter.split(X, y, groups=vids)):
        custom_scaler = StandardScaler().fit(X[train_index])
        scaled_train_X = custom_scaler.transform(X[train_index])
        scaled_test_X = custom_scaler.transform(X[test_index])

        regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
        regressor.fit(scaled_train_X, y[train_index])

        y_pred = regressor.predict(scaled_test_X)
        y_true = y[test_index]

        mae_score = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
        r2 = r2_score(y_true, y_pred, multioutput='raw_values')
        rmse_score = root_mean_squared_error(y_true, y_pred, multioutput='raw_values')

        results[i] = {
            "mae":  [mae_score[0],  mae_score[1]],
            "r2":   [r2[0],         r2[1]],
            "rmse": [rmse_score[0], rmse_score[1]],
        }

    V_metrics = {"mae": 0.0, "r2": 0.0, "rmse": 0.0}
    A_metrics = {"mae": 0.0, "r2": 0.0, "rmse": 0.0}
    for v in results.values():
        for k in ("mae", "r2", "rmse"):
            V_metrics[k] += v[k][0]
            A_metrics[k] += v[k][1]
    V_metrics = {k: v / len(results) for k, v in V_metrics.items()}
    A_metrics = {k: v / len(results) for k, v in A_metrics.items()}

    print(f"Sujet {random_subject_id} - moyenne sur {len(results)} folds")
    print(f"  Valence : {V_metrics}")
    print(f"  Arousal : {A_metrics}")

