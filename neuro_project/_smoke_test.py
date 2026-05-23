"""End-to-end smoke test of the package on synthetic data.

Does NOT touch DEAP. Validates imports, shapes, and the conceptual data flow:
  splits -> classical CV -> liking surface -> gap report -> spectrogram brick
  -> CNN forward (if torch available).
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

# allow `python neuro_project/_smoke_test.py` from project root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from neuro_project import config            # noqa: E402
from neuro_project.splits import (          # noqa: E402
    aggregate_by_video, assert_no_group_leak, video_leaveoneout, window_groupkfold,
)
from neuro_project.models_classical import (  # noqa: E402
    LOVO_FACTORIES, MODEL_FACTORIES, _aggregate_window_to_video, _metrics,
)
from neuro_project.liking_model import (    # noqa: E402
    fit_liking_model, optimal_zone, response_surface,
)
from neuro_project.gap_analysis import compute_gap, format_report   # noqa: E402
from neuro_project.spectrograms import (    # noqa: E402
    window_to_multichannel_spec, window_to_spectrogram,
)
from neuro_project.persistence import (     # noqa: E402
    load_df, load_obj, save_df, save_obj,
)


def section(name): print(f"\n=== {name} ===")


# ---------------------------------------------------------------- synthetic
rng = np.random.default_rng(0)
N_VIDS = 40
N_WIN_PER_VID = 50                              # 2000 windows / "subject"
n = N_VIDS * N_WIN_PER_VID
X_feat = rng.standard_normal((n, config.N_FEATURES)).astype(np.float32)
vids   = np.repeat(np.arange(N_VIDS), N_WIN_PER_VID).astype(np.int16)
# y constant per video (DEAP-like), plus a sliver of noise
y_per_vid = rng.uniform(1, 9, size=(N_VIDS, 2)).astype(np.float32)
y_feat = np.repeat(y_per_vid, N_WIN_PER_VID, axis=0)
y_feat += rng.normal(scale=0.05, size=y_feat.shape).astype(np.float32)

section("imports OK")
print("config.N_FEATURES =", config.N_FEATURES)
print("X", X_feat.shape, "y", y_feat.shape, "vids", vids.shape)


# ---------------------------------------------------------------- splits
section("splits — GroupKFold (4a)")
for i, (tr, te) in enumerate(window_groupkfold(X_feat, y_feat, vids, n_splits=5)):
    assert_no_group_leak(tr, te, vids)
    print(f"  fold {i}: train={len(tr)} test={len(te)} "
          f"test_vids={sorted(set(vids[te]))[:3]}...")
print("OK: 5 folds, no group leak")


section("splits — aggregate_by_video + LOVO (4b)")
Xv, yv, groups = aggregate_by_video(X_feat, y_feat, vids)
print(f"  Xv={Xv.shape} yv={yv.shape} groups={groups.shape}")
n_folds = sum(1 for _ in video_leaveoneout(Xv, yv, groups))
assert n_folds == N_VIDS, f"expected {N_VIDS} LOVO folds, got {n_folds}"
print(f"OK: {n_folds} LOVO folds")


# ---------------------------------------------------------------- classical models
section("models_classical — metrics + factories")
m = _metrics(y_feat[:1000], y_feat[:1000] + rng.normal(scale=0.1, size=(1000, 2)))
print(f"  metrics keys: {sorted(m.keys())}")
print(f"  mae_V={m['mae_V']:.3f}  r2_V={m['r2_V']:.3f}")

# Each factory builds and a quick fit on a tiny sample
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(Xv)
for name, fac in {**MODEL_FACTORIES, **LOVO_FACTORIES}.items():
    mdl = fac()
    mdl.fit(sc.transform(Xv), yv)
    out = mdl.predict(sc.transform(Xv[:3]))
    assert out.shape == (3, 2), f"{name} bad output shape {out.shape}"
    print(f"  {name:14s} -> predict OK shape {out.shape}")


# ---------------------------------------------------------------- liking + surface
section("liking_model — fit + surface + optimal zone")
# Build a fake labels frame: Liking peaks near high V, mid A
df_labels = pd.DataFrame({
    "subject": np.repeat([f"{i:02d}" for i in range(1, 11)], N_VIDS),
    "video":   np.tile(np.arange(N_VIDS), 10),
    "Valence": rng.uniform(1, 9, N_VIDS * 10),
    "Arousal": rng.uniform(1, 9, N_VIDS * 10),
})
df_labels["Liking"] = (
    9 - ((df_labels["Valence"] - 7.0) ** 2 + (df_labels["Arousal"] - 5.5) ** 2) / 5
    + rng.normal(scale=0.3, size=len(df_labels))
).clip(1, 9)

lm = fit_liking_model(df_labels, kind="kernel_ridge")
surf = response_surface(lm, step=0.25)
zone = optimal_zone(surf, top_pct=0.1)
print(f"  surface shape: {surf.shape}")
print(f"  (V*, A*) = ({zone.V_star:.2f}, {zone.A_star:.2f})  "
      f"Liking* = {zone.Liking_star:.2f}  threshold = {zone.threshold:.2f}")
# Surface argmax should land near (V=7, A=5.5) given our synthetic g
assert abs(zone.V_star - 7.0) < 1.5 and abs(zone.A_star - 5.5) < 1.5, "argmax far from expected"
print("OK: argmax close to designed peak")


# ---------------------------------------------------------------- gap analysis
section("gap_analysis — compute_gap + format_report")
# pretend the M1 model returned per-window predictions for one ad
V_hat = rng.uniform(3, 5, size=200).astype(np.float32)   # too low valence
A_hat = rng.uniform(4, 6, size=200).astype(np.float32)
report = compute_gap(V_hat, A_hat, lm, zone, surf.V_grid, surf.A_grid, weak_k=3)
print(f"  V_hat={report.V_hat:.2f} A_hat={report.A_hat:.2f} "
      f"dV={report.dV:+.2f} dA={report.dA:+.2f} dist={report.distance:.2f}")
assert report.dV > 0, "expected positive dV (we built V_hat lower than V*)"
print(format_report(report, ad_description="30s car ad, fast cuts, techno 140 BPM"))


# ---------------------------------------------------------------- spectrograms
section("spectrograms — single window + multichannel")
# sinus 10 Hz, 2 s @ 128 Hz -> peak at 10 Hz expected
t = np.arange(config.WINDOW_SIZE) / config.SAMPLE_RATE
sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)
f_kept, S = window_to_spectrogram(sig)
peak_freq = f_kept[np.argmax(S.sum(axis=1))]
print(f"  F kept: {len(f_kept)} (4-45 Hz)   T: {S.shape[1]}   peak at {peak_freq:.1f} Hz")
assert 8.0 <= peak_freq <= 12.0, f"peak should be ~10 Hz, got {peak_freq}"

# multichannel: build a fake (32, 256) EEG window
fake_eeg = rng.standard_normal((32, config.WINDOW_SIZE)).astype(np.float32)
spec = window_to_multichannel_spec(fake_eeg)
print(f"  multichannel spec shape: {spec.shape}  dtype={spec.dtype}")
assert spec.ndim == 3 and spec.shape[0] == len(config.EEG_CHANNELS)
print("OK: spectrogram brick works")


# ---------------------------------------------------------------- DL (optional)
section("models_dl — torch (optional)")
try:
    import torch  # noqa: F401
    from neuro_project.models_dl import build_cnn, predict_cnn
    F, T = spec.shape[1], spec.shape[2]
    X_dl = rng.standard_normal((8, len(config.EEG_CHANNELS), F, T)).astype(np.float32)
    model = build_cnn(n_eeg_channels=len(config.EEG_CHANNELS))
    out = predict_cnn(model, X_dl)
    print(f"  CNN out shape: {out.shape}  (expected (8, 2))")
    assert out.shape == (8, 2)
    print("OK: torch CNN forward")
except ImportError:
    print("  torch not installed -> CNN skipped (not a failure)")


# ---------------------------------------------------------------- persistence
section("persistence — round-trip")
with tempfile.TemporaryDirectory() as td:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    save_df(df, "demo", dirpath=td)
    assert load_df("demo", dirpath=td).equals(df)
    save_obj({"hello": np.arange(5)}, "demo_obj", dirpath=td)
    assert (load_obj("demo_obj", dirpath=td)["hello"] == np.arange(5)).all()
    assert load_obj("missing", dirpath=td) is None
print("OK: df + joblib + None-safe load")


print("\nALL CHECKS PASSED")
