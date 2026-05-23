"""Spectrogram extractor — input for the CNN baseline (stage 5 DL).

Per window: STFT on each of the 14 EEG channels -> stack -> log-power -> crop
to [4, 45] Hz. Per subject: cached as cache_spec/sXX_spec.npz with keys
X (n, 14, F, T), y (n, 2), video_ids (n,).
"""
from __future__ import annotations

import os
import pickle
from typing import Tuple

import numpy as np
from scipy.signal import spectrogram

from .config import (
    BASELINE_SAMPLES, DATA_PATH, EEG_CHANNELS, SAMPLE_RATE, SPEC_CACHE_DIR,
    STEP_SIZE, STFT_FREQ_HI, STFT_FREQ_LO, STFT_NOVERLAP, STFT_NPERSEG,
    WINDOW_SIZE, normalise_subject_id,
)


# ---------------------------------------------------------------- single window
def window_to_spectrogram(
    signal_1d: np.ndarray,
    fs: int = SAMPLE_RATE,
    nperseg: int = STFT_NPERSEG,
    noverlap: int = STFT_NOVERLAP,
    f_lo: float = STFT_FREQ_LO,
    f_hi: float = STFT_FREQ_HI,
) -> Tuple[np.ndarray, np.ndarray]:
    """One 1D signal -> (f_kept, log_power[F, T]).

    Returned spectrogram is log(power + 1e-10), cropped to [f_lo, f_hi]."""
    f, _, Sxx = spectrogram(signal_1d, fs=fs, nperseg=nperseg, noverlap=noverlap)
    log_Sxx = np.log(Sxx + 1e-10).astype(np.float32)
    mask = (f >= f_lo) & (f <= f_hi)
    return f[mask], log_Sxx[mask, :]


def window_to_multichannel_spec(
    eeg_window: np.ndarray,
    channels: list[int] = EEG_CHANNELS,
) -> np.ndarray:
    """eeg_window: (n_channels_total, window_size).
    Returns (len(channels), F, T) float32, where F/T are the cropped sizes."""
    spec_list = []
    for ch in channels:
        _, S = window_to_spectrogram(eeg_window[ch])
        spec_list.append(S)
    return np.stack(spec_list, axis=0).astype(np.float32)


# ---------------------------------------------------------------- per-subject
def extract_subject_spectrograms(
    subject_id: str, data_path: str = DATA_PATH,
    cache_dir: str = SPEC_CACHE_DIR, force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For one subject: returns (X[n,14,F,T], y[n,2], vids[n]).
    Cached as `sXX_spec.npz`. Reads DEAP `.dat` file (pickle latin-1)."""
    sid = normalise_subject_id(subject_id)
    cache_file = os.path.join(cache_dir, f"s{sid}_spec.npz")
    if os.path.exists(cache_file) and not force:
        d = np.load(cache_file)
        return d["X"], d["y"], d["video_ids"]

    path = os.path.join(data_path, f"s{sid}.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"DEAP file missing : {path}")
    with open(path, "rb") as f:
        sub = pickle.load(f, encoding="latin1")
    data, labels = sub["data"], sub["labels"]                     # (40,40,8064), (40,4)

    # probe one window to learn F, T
    probe = window_to_multichannel_spec(data[0, :, :WINDOW_SIZE])
    n_ch, F, T = probe.shape

    X_list, y_list, vid_list = [], [], []
    for trial in range(40):
        eeg = data[trial, :32]
        va  = labels[trial, :2]
        start = BASELINE_SAMPLES
        while start + WINDOW_SIZE <= eeg.shape[1]:
            X_list.append(window_to_multichannel_spec(eeg[:, start:start + WINDOW_SIZE]))
            y_list.append(va)
            vid_list.append(trial)
            start += STEP_SIZE

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    vids = np.array(vid_list, dtype=np.int16)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_file, X=X, y=y, video_ids=vids)
    return X, y, vids


def load_subject_spectrograms(
    subject_id: str, cache_dir: str = SPEC_CACHE_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sid = normalise_subject_id(subject_id)
    p = os.path.join(cache_dir, f"s{sid}_spec.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Spectrogram cache absent: {p}. "
            "Run extract_subject_spectrograms(sid) first.")
    d = np.load(p)
    return d["X"], d["y"], d["video_ids"]
