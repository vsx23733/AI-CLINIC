# neuro_project

EEG + peripheral physiological signals → emotion (Valence, Arousal) → Liking → targeted advice on advertising.

Python package for the **AI CLINIC / Neuromarketing** project (PGE 4). The pipeline goes from the raw DEAP `.dat` file all the way to a structured text report, ready to be consumed by an LLM in a later stage.

---

## 1. Objective in one sentence

From a viewer's **physiological response**, predict their `[Valence, Arousal]`, compare it to the **optimal zone `(V*, A*)`** that maximizes **Liking** learned from DEAP, and produce a **quantified gap report**. This report is the natural input for a downstream LLM advisor (out of scope here).

---

## 2. Pipeline (diagram)

```
┌────────────────────────────────────────────────────────────────────┐
│ 1. RAW DATA                                                          │
│   DEAP/sXX.dat : data (40, 40, 8064) + labels (40, 4)                │
│   labels = [Valence, Arousal, Dominance, Liking] on 1..9             │
│   File : (external)                                                  │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 2. DATA PROCESSING                                                   │
│   load(.dat) → drop 3 s baseline → windowing 2 s / 0.125 s           │
│   EEG features    : 14 ch × 5 bands (Welch)              = 70        │
│   Periph features : EMG/GSR/Resp/BVP/Temp                = 15        │
│   Total = 85                                                         │
│   File : data_processing.py                                          │
└────────────────────────────┬───────────────────────────────────────┘
                              │ persisted as .npz
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 3. CACHE                                                             │
│   cache_full_v2/sXX_full_v2.npz   (85 features)                      │
│   cache_spec/sXX_spec.npz         (spectrograms 14×F×T)              │
└────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
┌──────────────────┐ ┌─────────────────┐ ┌──────────────────────────┐
│ 4a SPLIT         │ │ 4b SPLIT        │ │ 4c SPLIT                 │
│ GroupKFold(5)    │ │ LeaveOneVideo   │ │ train/val (DL)           │
│ window-level     │ │ Out (40 folds)  │ │                          │
│ File: splits     │ │ File: splits    │ │ File: splits / models_dl │
└────────┬─────────┘ └────────┬────────┘ └────────────┬─────────────┘
         ▼                    ▼                        ▼
┌────────────────────────────────────────────────────────────────────┐
│ 5. MODELS                                                            │
│   M1 classical : RF, GradientBoost, Ridge, Dummy (sklearn)           │
│   M1 deep      : CNN over spectrograms (PyTorch)                     │
│   Files : models_classical.py · models_dl.py · spectrograms.py       │
└────────────────────────────┬───────────────────────────────────────┘
                              │ persisted
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 6. ARTIFACTS                                                         │
│   artifacts/df_results.pkl/.csv (window-level metrics)               │
│   artifacts/df_lovo.pkl/.csv    (video-level metrics)                │
│   artifacts/models/*.joblib · *.pt                                   │
│   File : persistence.py                                              │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 7. PREDICTORS                                                        │
│   M1 : physiology → [V, A]    (models above)                         │
│   M2 : (V, A) → Liking        (KernelRidge / RF / Poly2)             │
│   File : liking_model.py                                             │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 8. RESPONSE SURFACE + OPTIMAL ZONE                                   │
│   g(V, A) evaluated on a 1..9 × 1..9 grid → Liking heatmap           │
│   (V*, A*) = argmax     +  zone mask (top 10 % by default)           │
│   File : liking_model.py                                             │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 9. GAP ANALYSIS                                                      │
│   measured (V̂, Â) vs (V*, A*) → ΔV, ΔA, distance, in_zone            │
│   Weak segments per time window                                      │
│   Structured text report (LLM-ready)                                 │
│   File : gap_analysis.py                                             │
└────────────────────────────┬───────────────────────────────────────┘
                              │   ═══ boundary: the video is NOT in DEAP ═══
                              │   beyond, the actual ad must come from an external source
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 10. PROMPT BUILDER                                  [implemented]   │
│   Input 1 : output of format_report(report)  ← stage 9               │
│   Input 2 : ad metadata (text provided by the user)                  │
│             genre, tempo, palette, segments, target audience         │
│   Output  : a single structured natural-language prompt (system+user)│
│   Suggested file : advisor/prompt_builder.py                         │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 11. LLM ADVISOR (pretrained, text-in / text-out)    [implemented]   │
│   Model : Claude / GPT / Llama / Mistral (your choice)               │
│   Reads the prompt → produces targeted creative advice :             │
│     "raise tempo in mid-section, warmer palette,                     │
│      add surprise beat at 0:35 to lift arousal"                      │
│   Constraints :                                                      │
│     - the LLM does NOT see the video (text-only)                     │
│     - relies solely on the numeric report + description              │
│   Suggested file : advisor/llm_advisor.py                            │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 12. ACTIONABLE RECOMMENDATIONS                      [implemented]   │
│   Structured advice per segment + global :                           │
│     - segment 0:28-0:41 : boost arousal (+fast cut, music swell),    │
│       keep positive valence (warm colors)                            │
│     - global : audience match OK, 5 s too long                       │
│   Output : JSON + readable text (plug into a UI or an export).       │
│   Suggested file : advisor/recommendations.py                        │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 13. CLOSED LOOP (re-measurement)                    [implemented]   │
│   Re-edited ad → re-run the subject (or a panel) measuring EEG       │
│   → stages 2 → 9 → 11 on the new version                             │
│   → compare gap(v_old) vs gap(v_new) : did Liking improve?           │
│   Acts as objective A/B evaluation of the LLM recommendation.        │
│   Suggested file : advisor/closed_loop.py                            │
└────────────────────────────────────────────────────────────────────┘
```

### Reading the stage 9 → stage 10 boundary

The visual separator after stage 9 marks two important things:

1. **Dataset boundary**: DEAP does not contain the video itself (only physio + ratings). So everything *above* the line can be learned from DEAP; everything *below* requires an external source (the actual ad or its text description).
2. **Package scope boundary**: `neuro_project/` intentionally stops at stage 9 (`format_report` is designed to be the natural input of the prompt builder). Stages 10–13 are left open because they depend on product choices (which LLM, which UI, which metadata format, which re-measurement protocol).

### Stage → input → output mapping

| Stage | Input | Output | File (existing or proposed) |
|:-:|---|---|---|
| 1  | DEAP `.dat` | `(data, labels)` in memory | external |
| 2  | `(data, labels)` | features `(n, 85)` + `y` + `vids` | `data_processing.py` |
| 3  | features | `.npz` cache | `data_processing.py` / `spectrograms.py` |
| 4a | cache | GroupKFold folds | `splits.py` |
| 4b | cache | LOVO folds | `splits.py` |
| 4c | spectrogram cache | train/val DL | `models_dl.py` |
| 5  | folds + features | trained models | `models_classical.py` · `models_dl.py` |
| 6  | models + metrics | `.pkl/.csv/.joblib/.pt` | `persistence.py` |
| 7  | persisted models | M1 + M2 loaded | `models_classical.py` · `liking_model.py` |
| 8  | M2 | `ResponseSurface` + `OptimalZone` | `liking_model.py` |
| 9  | M1.predict + zone | `GapReport` + text | `gap_analysis.py` |
| 10 | text report + ad metadata | LLM prompt | `advisor/prompt_builder.py` |
| 11 | prompt | advice text (JSON) | `advisor/llm_advisor.py` (Ollama) |
| 12 | advice text | structured recommendations (JSON) | `advisor/recommendations.py` |
| 13 | re-edited ad + re-measured EEG | comparison gap_old vs gap_new | `advisor/closed_loop.py` |

---

## 3. File tree

```
neuro_project/
├── README.md              ← this document
├── __init__.py            ← exposes config
├── config.py              ← constants & paths (no I/O)
├── persistence.py         ← save/load DataFrames, joblib, torch
├── data_processing.py     ← bandpower + peripheral features (stage 2)
├── splits.py              ← GroupKFold, LOVO, per-video aggregation
├── models_classical.py    ← RF/GB/Ridge/Dummy + CV runners + persistence
├── spectrograms.py        ← STFT → log-power → (14, F, T)
├── models_dl.py           ← PyTorch CNN + training loop
├── liking_model.py        ← M2 (V,A)→Liking + surface + optimal zone
├── gap_analysis.py        ← GapReport + text formatting
├── main.py                ← CLI orchestrator (interactive menu + subcommands)
├── _smoke_test.py         ← end-to-end test on synthetic data
├── advisor/               ← stages 10–13
│   ├── __init__.py
│   ├── prompt_builder.py  ← AdMetadata + build_prompt(report, ad_metadata)
│   ├── llm_advisor.py     ← Ollama client (stdlib urllib, no deps) + advise()
│   ├── recommendations.py ← parse + validate LLM JSON → RecommendationSet
│   ├── closed_loop.py     ← compare_gaps(old, new) → ImprovementSummary
│   └── _smoke_test.py     ← fixture test ; live Ollama if reachable
└── artifacts/             ← (created on demand)
    ├── *.pkl  /  *.csv
    └── models/  *.joblib  *.pt
```

---

## 4. Detailed reference — logic and math, function by function

### 4.1 `config.py`

Pure constants. Defines `SAMPLE_RATE=128`, `WINDOW_SIZE=256` (2 s), `STEP_SIZE=16` (0.125 s), `BASELINE_SAMPLES=384` (3 s pre-stimulus), the 14 EEG channels, the 5 bands (Theta–Gamma), the 8 peripheral channels, and the subject list.

`normalise_subject_id(sid)`: accepts `'01'`, `1`, `'1'` → returns `'01'`. Centralizes zero-padding.

---

### 4.2 `data_processing.py` — extracting the 85 features

#### `bandpower(signal, bands, sf)`
Estimates **band power** via the **Welch** method followed by trapezoidal integration.

**Welch**: split the signal `x` into segments of length `nperseg` with overlap, apply a Hann window to each, take the FFT of each segment, then average the periodograms:

```
P(f) = (1 / (M · U)) · Σ_{m=1..M} | FFT(w · x_m) |²(f)
```

where `M` = number of segments, `w` = Hann window, `U` = normalisation factor. Averaging **reduces the variance** of the estimator (vs a single raw periodogram).

**Bandpower** over `[f_lo, f_hi]`:

```
BP = ∫_{f_lo}^{f_hi} P(f) df  ≈  Σ_k (P(f_k) + P(f_{k+1})) / 2  ·  Δf      (trapezoid)
```

Implementation: `np.trapezoid(psd[mask], freqs[mask])`. With `nperseg=128 < WINDOW_SIZE=256`, Welch effectively averages ~3 segments.

#### `peripheral_features(periph_window)`
15 hand-crafted statistics over the 8 peripheral channels (hEOG/vEOG are ignored — they are ocular artefacts, not emotional):
- **EMG (zEMG, tEMG)**: `mean(|x|)` (contraction energy), `std`.
- **GSR**: `mean`, `std`, `range = max−min`, `slope = np.polyfit(t, x, 1)[0]` (linear trend = tonic drift).
- **Resp**: `mean`, `std`.
- **BVP**: `mean`, `std`, `range` (pulse amplitude).
- **Temp**: `mean`, `slope`.

#### `extract_subject_features(subject_id, force=False)`
- Loads `sXX.dat` (pickle, latin-1 encoding) → `data (40,40,8064)`, `labels (40,4)`.
- For each trial: `start = 384` (skips the DEAP pre-stimulus baseline), `while start + WINDOW_SIZE <= 8064` with `step = 16`.
- Per window: concatenates 70 EEG bandpowers + 15 peripheral stats → vector of 85.
- `y = labels[:, :2]` = `[Valence, Arousal]`.
- `video_ids` stores the trial index (0..39) → serves as the **group** for CV.
- Compressed cache at `cache_full_v2/sXX_full_v2.npz`.

#### `load_subject(subject_id, cache_dir)`
Pure cache read → `(X, y, vids)`. No computation.

---

### 4.3 `splits.py` — cross-validation

#### `window_groupkfold(X, y, vids, n_splits=5)`
Wraps `sklearn.model_selection.GroupKFold`. Partitions the **groups** (videos) into `n_splits` blocks. At each fold, one block becomes test, the rest train. **All windows of a video stay together** → no leakage between neighbouring windows with 94 % overlap.

GroupKFold is **deterministic** (no shuffle / random_state) — the algorithm picks the partition. Constraint: `n_splits ≤ |unique(groups)|`.

#### `aggregate_by_video(X, y, vids, agg='mean')`
Collapses `(n_windows, 85)` → `(40, 85)` by averaging features across all windows of a video. Mathematically:

```
X_v[i, j] = (1 / |W_i|) · Σ_{k ∈ W_i} X[k, j]
```

where `W_i` = set of window indices belonging to video `i`. Target `y_v = first()` (constant per video). This is the step that makes "video-level" regression possible: 40 independent points per subject, the real problem size.

#### `video_leaveoneout(Xv, yv, groups)`
`LeaveOneGroupOut` with `groups = video_id` ⇒ 40 folds: 39 videos in train, 1 in test. Each video is tested exactly once.

#### `assert_no_group_leak(train_idx, test_idx, groups)`
Verifies `set(groups[train_idx]) ∩ set(groups[test_idx]) == ∅`. Development-only sanity check.

---

### 4.4 `models_classical.py` — classical ML models

#### `MODEL_FACTORIES`
- **RandomForestRegressor** (n=200) — natively multi-output: one `fit` for `(V, A)`; importance via impurity decrease (Gini/MSE).
- **MultiOutputRegressor(GradientBoostingRegressor)** — tree boosting; the wrapper duplicates the model for V and A.

#### `LOVO_FACTORIES`
- **DummyRegressor(strategy='mean')**: predicts the mean of `y_train`. **R² = 0 anchor**. This is the reference against which we judge whether a model has learned anything at all.
- **Ridge(alpha=10)**: `min_w ||y − Xw||² + α ||w||²`. Closed-form solution: `w = (XᵀX + αI)⁻¹ Xᵀy`. L2 regularisation → stable in high-dim / small-n.
- **RandomForest(n=300)**.

#### `_metrics(y_true, y_pred)`
Computes MAE, RMSE, R² **separately for V and A** (`multioutput='raw_values'`):

```
MAE  = (1/n) Σ |y_i − ŷ_i|
RMSE = √( (1/n) Σ (y_i − ŷ_i)² )
R²   = 1 − Σ (y_i − ŷ_i)² / Σ (y_i − ȳ)²
```

R² < 0 ⇒ the model predicts worse than the mean. That's the honest signal on DEAP at the window level, due to the fact that an entire video shares the same target.

#### `_aggregate_window_to_video(...)`
For each test video, ad-level prediction = `median` of the window predictions (robust to outlier windows). Then compared to the video labels (`first`).

#### `run_window_cv(subject_id, factory, n_splits=5)`
For one subject: loops over the 5 GroupKFold folds. At each fold:
1. `StandardScaler` fitted **on train only**: `z = (x − μ_train) / σ_train`. No leakage.
2. `model = factory().fit(scaled_train, y_train)`.
3. `y_pred = model.predict(scaled_test)`.
4. Metrics.

Returns the **mean of metrics across the 5 folds** (stable estimate).

#### `run_lovo(subject_id)`
For one subject: for each LOVO model, 40 LeaveOneVideoOut folds on the aggregated `(40, 85)` dataset. Predictions collected into `preds[te]` (1 row / fold). Metrics computed over the 40 hold-out videos.

#### `fit_and_save_final(subject_id, model_name, level)`
Re-fits **on the whole subject** (no split, for final inference), and saves `{scaler, model, features, level}` via joblib.

#### `predict_va(subject_id, X, model_name, level)`
Reloads the bundle, applies `scaler.transform`, then `model.predict` → `(n, 2)`.

---

### 4.5 `spectrograms.py` — features for the CNN

#### `window_to_spectrogram(signal_1d)`
1. **STFT power** via `scipy.signal.spectrogram`:
   ```
   S_xx(f, t) = | STFT{x · w}(f, t) |²
   ```
   `nperseg=64`, `noverlap=32` → `F = nperseg/2 + 1 = 33` frequency bins, `T = 7` time bins on a 2 s window.
2. **Log compression**: `log(S + 1e-10)`. Rationale: EEG power varies over several orders of magnitude (huge delta vs tiny gamma); log compresses the dynamic range → better stationarity for a CNN. `+eps` avoids `log(0) = -∞`.
3. **Frequency crop**: `mask = (f >= 4) & (f <= 45)`. DEAP is already filtered 4–45 Hz, drop the useless bins → `F ≈ 21` final bins.

Output: `(F≈21, T=7)` float32.

#### `window_to_multichannel_spec(eeg_window, channels)`
Applies `window_to_spectrogram` to the 14 EEG channels and stacks on axis 0 → `(14, F, T)`. The 14 channels become the "channels" of the CNN, like R/G/B for an image.

#### `extract_subject_spectrograms(subject_id)`
Same skeleton as `extract_subject_features` but replacing the Welch extraction with the multichannel STFT extraction. Output `(n_windows, 14, F, T)`, cached at `cache_spec/sXX_spec.npz`. ~190 MB / subject in float32.

---

### 4.6 `models_dl.py` — PyTorch CNN

#### Architecture `build_cnn(n_eeg_channels=14, n_outputs=2)`

```
Input  (B, 14, F, T)
  │
  ├─ Conv2d(14 → 32, k=3×3, pad=1)          # 2D cross-correlation
  ├─ BatchNorm2d(32)                         # mini-batch normalisation
  ├─ ReLU                                    # max(0, x)
  ├─ MaxPool2d(k=2×2, ceil_mode=True)        # spatial downsample
  ├─ Conv2d(32 → 64, k=3×3, pad=1)
  ├─ BatchNorm2d(64)
  ├─ ReLU
  └─ AdaptiveAvgPool2d(1×1)                  # → (B, 64, 1, 1)
  │
  └─ Linear(64 → 2)                          # regression head
Output (B, 2)  → [V̂, Â]
```

**2D convolution** mathematically:
```
y[b, c_out, i, j] = Σ_{c_in, di, dj} K[c_out, c_in, di, dj] · x[b, c_in, i+di, j+dj] + bias
```
**BatchNorm**: `y = γ · (x − μ_B) / √(σ_B² + ε) + β`, where `μ_B`, `σ_B²` are the mini-batch stats (in train mode); running stats used at eval.

#### `train_cnn(X_train, y_train, X_val, y_val, cfg)`
- **Loss**: `SmoothL1Loss` (Huber). Quadratic near 0, linear beyond → less sensitive to outliers than MSE.
  ```
  L(y, ŷ) = { 0.5 (y − ŷ)²        if |y − ŷ| < 1
            { |y − ŷ| − 0.5       otherwise
  ```
- **Optimizer**: `AdamW(lr=1e-3, weight_decay=1e-4)`. AdamW = Adam + decoupled L2 (regularisation applied directly to weights, not via the gradient).
- **Loop**: `epochs=15` by default, shuffled mini-batches via `np.random.default_rng(42)`. **Early stopping** on val loss with `patience=3`. Best state restored at the end.

#### `train_subject_cnn_cv(X, y, vids, n_splits=5)`
Window-level GroupKFold for the CNN. Same as `run_window_cv` but with DL training instead of sklearn.

#### `save_cnn / load_cnn`
`save_torch(model.state_dict())` then rebuild (`build_cnn` + `load_state_dict`). Allows inference without retraining.

---

### 4.7 `liking_model.py` — Liking surface

#### `load_deap_labels(subject_ids)`
Loads only the DEAP **labels** (not the signals). Returns a long DataFrame `(subject, video, V, A, Dominance, Liking)`. 32 × 40 = 1280 rows max.

#### `fit_liking_model(df_labels, kind='kernel_ridge')`
Learns `g : (V, A) → Liking` on these 1280 (V, A, Liking) triples.

- **KernelRidge** (default): Ridge in the feature space induced by an RBF kernel.
  ```
  min_α  || y − K α ||² + λ αᵀ K α
  K(x, x') = exp(−γ ||x − x'||²)
  ```
  Solution: `α = (K + λ I)⁻¹ y`, prediction `ŷ(x) = Σ_i α_i K(x_i, x)`. Smooth non-linear surface → suitable for visualizing an optimal "zone".
- **RandomForestRegressor**: non-parametric, captures non-monotonic effects.
- **Poly2**: Ridge over degree-2 polynomial features → interpretable quadratic surface:
  ```
  L̂ = β₀ + β₁ V + β₂ A + β₃ V² + β₄ VA + β₅ A²
  ```

#### `response_surface(model, v_range=(1,9), a_range=(1,9), step=0.1)`
Evaluates `g` on a **regular grid**:
```
V_grid = [v_min, v_min+step, ..., v_max]      (n_V = 81)
A_grid = [a_min, a_min+step, ..., a_max]      (n_A = 81)
L[i, j] = g(V_grid[i], A_grid[j])             (n_V × n_A)
```
Returns a serializable `ResponseSurface`.

#### `optimal_zone(surface, top_pct=0.10)`
1. **Global argmax**:
   ```
   (i*, j*) = argmax_{i, j}  L[i, j]
   V* = V_grid[i*]   A* = A_grid[j*]   L* = L[i*, j*]
   ```
2. **Zone mask**: threshold = `1 − top_pct` quantile of `L` → keeps the top 10 % most-liked cells:
   ```
   thr = Q_{0.90}(L)
   zone_mask = (L >= thr)
   ```
   Allows advice within a **neighbourhood**, not just on a single pixel.

#### `plot_surface(...)`
Matplotlib heatmap + mask contour + red star on `(V*, A*)`.

---

### 4.8 `gap_analysis.py` — stage 9

#### `per_window_liking(V_hat, A_hat, liking_model, window_size_s, step_s)`
For each window i, computes:
```
L_pred_i = g(V̂_i, Â_i)
t0_i = i · step_s     t1_i = t0_i + window_size_s
```
Returns a list of `WindowSegment(start_s, end_s, V_hat, A_hat, Liking_pred)`.

#### `weak_segments(segments, k=5)`
Sorts by ascending `Liking_pred` → top-k **least-liked** segments. That's where the ad needs intervention.

#### `compute_gap(V_hat, A_hat, liking_model, zone, V_grid, A_grid, ...)`
Ad-level summary:
```
V̂ = median(V̂_windows)           Â = median(Â_windows)
L̂ = g(V̂, Â)
ΔV = V* − V̂                     ΔA = A* − Â
distance = √(ΔV² + ΔA²)
in_zone  = zone_mask[ closest_index(V̂), closest_index(Â) ]
```
Median chosen for robustness against outlier windows. Returns a serializable `GapReport` + the list of weak segments.

#### `format_report(report, ad_description=None)`
Serialises everything as **structured text**, ready to paste into an LLM prompt. Sections: PREDICTED RESPONSE, TARGET ZONE, GAP, WEAK SEGMENTS, AD DESCRIPTION. **This is the stage 10 boundary**: since the video is not in DEAP, `ad_description` must come from the external user.

---

### 4.9 `persistence.py`

- `save_df / load_df`: pickle (preserves dtypes) **and** a human-readable CSV side by side.
- `save_obj / load_obj`: `joblib` with `compress=3` (efficient for forests).
- `save_torch / load_torch`: `torch.save(state_dict)`, lazy torch import.
- All `load_*` variants return `None` if the file does not exist → no need for caller-side `try/except`.

---

## 5. CLI orchestrator (`main.py`)

A single entry point exposes the whole pipeline.

```bash
# Interactive menu (default — no argument)
python -m neuro_project.main

# Scripted subcommands
python -m neuro_project.main status                              # inventory
python -m neuro_project.main extract --subjects 01,02-04         # stage 2
python -m neuro_project.main extract-spec --subjects 01          # stage 5 DL prep
python -m neuro_project.main train-window --model RandomForest   # stages 4a + 5
python -m neuro_project.main train-lovo                          # stages 4b + 5
python -m neuro_project.main train-cnn --subject 01 --epochs 10  # stage 5 DL
python -m neuro_project.main save-models --subject 01 --level window
python -m neuro_project.main liking --kind kernel_ridge          # stages 7-8
python -m neuro_project.main predict --subject 01 --model RandomForest \
                                     --description "30s car ad, fast cuts"
python -m neuro_project.main advise --report gap_report \
                                    --genre automotive --duration-s 30 \
                                    --ollama-model gemma4:latest
python -m neuro_project.main loop --old gap_report_v1 --new gap_report_v2
python -m neuro_project.main smoke                               # both smoke tests
python -m neuro_project.main --help                              # full subcommand list
```

The interactive menu calls the same `cmd_*` functions and prompts for each input with defaults. Subjects accept ranges and lists: `01,02-04` → `['01','02','03','04']`; empty = all 32 subjects.

## 6. Quickstart (programmatic)

```python
# 1. Window-level CV over all subjects, all classical models
from neuro_project.models_classical import batch_window_cv
from neuro_project.persistence import save_df

df = batch_window_cv()
save_df(df, "df_results")

# 2. LOVO video-level
from neuro_project.models_classical import batch_lovo
save_df(batch_lovo(), "df_lovo")

# 3. Liking surface + optimal zone
from neuro_project.liking_model import (load_deap_labels, fit_liking_model,
                                        response_surface, optimal_zone)
labels = load_deap_labels()
lm   = fit_liking_model(labels, kind="kernel_ridge")
surf = response_surface(lm)
zone = optimal_zone(surf)

# 4. Gap for a measured ad (V_hat, A_hat from M1, per window)
from neuro_project.gap_analysis import compute_gap, format_report
report = compute_gap(V_hat, A_hat, lm, zone, surf.V_grid, surf.A_grid)
print(format_report(report, ad_description="30s car ad, fast cuts, techno 140 BPM"))

# 5. Deep learning CNN (first run ~30s/subject, cached afterwards)
from neuro_project.spectrograms import extract_subject_spectrograms
from neuro_project.models_dl import TrainConfig, train_cnn

X, y, vids = extract_subject_spectrograms("01")
model, res = train_cnn(X[:1500], y[:1500], X[1500:], y[1500:],
                       cfg=TrainConfig(epochs=15))

# 6. Advisor (stages 10-13) — Ollama-powered creative advice
#    Pre-req: `ollama serve` running locally, model pulled (e.g. `ollama pull llama3.1`)
from neuro_project.advisor.prompt_builder import AdMetadata, AdSegment
from neuro_project.advisor.llm_advisor   import advise, OllamaConfig
from neuro_project.advisor.recommendations import parse_advice, format_recommendations
from neuro_project.advisor.closed_loop  import compare_gaps, format_summary

ad = AdMetadata(
    title="EcoBoost Drive 30s", duration_s=30.0,
    genre="automotive", target_audience="25-45 urban, eco-curious",
    tempo="moderate -> fast finale", palette="cool blues + warm sunset payoff",
    segments=[AdSegment(0, 8, "commute"),
              AdSegment(8, 22, "open road"),
              AdSegment(22, 30, "logo")],
)

# stage 10+11 : prompt + LLM call → raw JSON text
raw = advise(report, ad, OllamaConfig(model="llama3.1", temperature=0.3))

# stage 12 : parse + validate + render
rs = parse_advice(raw)
print(format_recommendations(rs))

# stage 13 : after re-editing the ad and re-measuring → new report
summary = compare_gaps(report_old=report, report_new=report_v2)
print(format_summary(summary))
```

## 7. Sanity check that everything works

```bash
# core pipeline (stages 2-9)
python neuro_project/_smoke_test.py
# advisor (stages 10-13) — Ollama live call is optional & skipped if absent
python neuro_project/advisor/_smoke_test.py
```
Both should end with `ALL CHECKS PASSED`. The advisor test mocks the LLM with a fixture JSON, so it runs even without Ollama; if Ollama is reachable and the model is pulled, an additional live call is performed.

---

## 8. Known limitations / not implemented

- **Negative R² at window-level** on DEAP: this is structural (1 label per 60 s video; ~465 windows share the same target). The honest reading → use the video-level (LOVO) metric.
- **40 videos per subject** = small for continuous regression. The standard DEAP protocol of high/low classification (threshold 5) works better with these features.
- **Stages 10–13 (LLM advisor + closed loop)** now implemented over **Ollama** (local LLM). You need an Ollama server running (`ollama serve`) with a model pulled (`ollama pull llama3.1` or any tag). Switch model via `OllamaConfig(model="...")`. JSON output is enforced server-side (`format='json'`); the parser tolerates minor noise and accumulates `parse_warnings`.
- **DEAP videos not bundled** with the dataset (YouTube links only). The stage 9 → 10 boundary is precisely this: the ad must be supplied **from outside** DEAP (text, or a multimodal LLM in production).
