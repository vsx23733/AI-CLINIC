# neuro_project

EEG + signaux physiologiques → émotion (Valence, Arousal) → Liking → conseils ciblés sur la publicité.

Package Python pour le projet **AI CLINIC / Neuromarketing** (PGE 4). Le pipeline va du fichier brut DEAP `.dat` jusqu'à un rapport texte structuré, prêt à être consommé par un LLM dans une étape ultérieure.

---

## 1. Objectif en une phrase

À partir de la **réponse physiologique** d'un spectateur, prédire son `[Valence, Arousal]`, le comparer à la **zone optimale `(V*, A*)`** qui maximise le **Liking** appris sur DEAP, et fournir un **rapport quantifié du gap**. Ce rapport est l'entrée naturelle d'un advisor LLM (hors-scope ici).

---

## 2. Pipeline (diagramme)

```
┌────────────────────────────────────────────────────────────────────┐
│ 1. RAW DATA                                                          │
│   DEAP/sXX.dat : data (40, 40, 8064) + labels (40, 4)                │
│   labels = [Valence, Arousal, Dominance, Liking] sur 1..9            │
│   Fichier : (externe)                                                │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 2. DATA PROCESSING                                                   │
│   load(.dat) → exclusion baseline 3 s → fenêtrage 2 s / 0.125 s     │
│   Features EEG : 14 ch × 5 bandes (Welch)                = 70       │
│   Features périph : EMG/GSR/Resp/BVP/Temp                = 15       │
│   Total = 85                                                         │
│   Fichier : data_processing.py                                       │
└────────────────────────────┬───────────────────────────────────────┘
                              │ persistance .npz
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 3. CACHE                                                             │
│   cache_full_v2/sXX_full_v2.npz   (features 85)                      │
│   cache_spec/sXX_spec.npz         (spectrogrammes 14×F×T)            │
└────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
┌──────────────────┐ ┌─────────────────┐ ┌──────────────────────────┐
│ 4a SPLIT         │ │ 4b SPLIT        │ │ 4c SPLIT                 │
│ GroupKFold(5)    │ │ LeaveOneVideo   │ │ train/val (DL)           │
│ window-level     │ │ Out (40 folds)  │ │                          │
│ Fichier: splits  │ │ Fichier: splits │ │ Fichier: splits / models_dl
└────────┬─────────┘ └────────┬────────┘ └────────────┬─────────────┘
         ▼                    ▼                        ▼
┌────────────────────────────────────────────────────────────────────┐
│ 5. MODÈLES                                                           │
│   M1 classique : RF, GradientBoost, Ridge, Dummy (sklearn)           │
│   M1 deep      : CNN sur spectrogrammes (PyTorch)                    │
│   Fichiers : models_classical.py · models_dl.py · spectrograms.py    │
└────────────────────────────┬───────────────────────────────────────┘
                              │ persistance
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 6. ARTIFACTS                                                         │
│   artifacts/df_results.pkl/.csv (window-level metrics)               │
│   artifacts/df_lovo.pkl/.csv    (video-level metrics)                │
│   artifacts/models/*.joblib · *.pt                                   │
│   Fichier : persistence.py                                           │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 7. PRÉDICTEURS                                                       │
│   M1 : physiologie → [V, A]   (modèles ci-dessus)                    │
│   M2 : (V, A) → Liking        (KernelRidge / RF / Poly2)             │
│   Fichier : liking_model.py                                          │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 8. SURFACE DE RÉPONSE + ZONE OPTIMALE                                │
│   g(V, A) évalué sur grille 1..9 × 1..9 → heatmap Liking             │
│   (V*, A*) = argmax     +  masque zone (top 10 % par défaut)         │
│   Fichier : liking_model.py                                          │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 9. GAP ANALYSIS                                                      │
│   (V̂, Â) mesurés vs (V*, A*) → ΔV, ΔA, distance, in_zone             │
│   Segments faibles par fenêtre temporelle                            │
│   Rapport texte structuré (prêt pour LLM)                            │
│   Fichier : gap_analysis.py                                          │
└────────────────────────────┬───────────────────────────────────────┘
                              │   ═══ frontière : la vidéo n'est PAS dans DEAP ═══
                              │   au-delà, l'ad réel doit venir d'une source externe
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 10. PROMPT BUILDER                                  [à implémenter] │
│   Entrée 1 : sortie de format_report(report)  ← étape 9             │
│   Entrée 2 : métadonnées de l'ad (texte fourni par l'utilisateur)   │
│              genre, tempo, palette, segments, audience cible        │
│   Sortie   : 1 prompt natural-language structuré (system + user)    │
│   Fichier proposé : advisor/prompt_builder.py                       │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 11. LLM ADVISOR (pretrained, text-in / text-out)    [à implémenter] │
│   Modèle : Claude / GPT / Llama / Mistral (au choix)                │
│   Lit le prompt → produit des conseils créatifs ciblés :            │
│     "raise tempo in mid-section, warmer palette,                    │
│      add surprise beat at 0:35 to lift arousal"                     │
│   Contraintes :                                                     │
│     - le LLM NE VOIT PAS la vidéo (text-only)                       │
│     - se base uniquement sur le report numérique + la description   │
│   Fichier proposé : advisor/llm_advisor.py                          │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 12. ACTIONABLE RECOMMENDATIONS                      [à implémenter] │
│   Conseils structurés par segment + global :                        │
│     - segment 0:28-0:41 : booster arousal (+coupe rapide, montée    │
│       musicale), garder valence positive (couleurs chaudes)         │
│     - global : audience match OK, durée trop longue de 5 s          │
│   Sortie : JSON + texte lisible (à brancher sur l'UI ou un export). │
│   Fichier proposé : advisor/recommendations.py                      │
└────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 13. CLOSED LOOP (re-mesure)                         [à implémenter] │
│   Ad réédité → on relance le sujet (ou un panel) en mesurant l'EEG  │
│   → étapes 2 → 9 → 11 sur la nouvelle version                       │
│   → compare gap(v_old) vs gap(v_new) : amélioration de Liking ?     │
│   Sert d'évaluation A/B objective de la recommandation LLM.         │
│   Fichier proposé : advisor/closed_loop.py                          │
└────────────────────────────────────────────────────────────────────┘
```

### Lecture de la frontière étape 9 → étape 10

La séparation visuelle après l'étape 9 marque deux choses importantes :

1. **Limite du dataset** : DEAP ne contient pas la vidéo elle-même (uniquement physio + ratings). Donc tout ce qui est *au-dessus* de la ligne peut être appris à partir de DEAP ; tout ce qui est *en-dessous* nécessite une source externe (l'ad réel ou sa description texte).
2. **Limite de scope du package** : `neuro_project/` s'arrête volontairement à l'étape 9 (le `format_report` est conçu pour être l'input naturel du prompt builder). Les étapes 10–13 sont laissées libres car elles dépendent de choix produit (quel LLM, quelle UI, quel format de métadonnée, quel protocole de re-mesure).

### Mapping étape → entrée → sortie

| Étape | Entrée | Sortie | Fichier (existant ou proposé) |
|:-:|---|---|---|
| 1  | DEAP `.dat` | `(data, labels)` en mémoire | externe |
| 2  | `(data, labels)` | features `(n, 85)` + `y` + `vids` | `data_processing.py` |
| 3  | features | `.npz` cache | `data_processing.py` / `spectrograms.py` |
| 4a | cache | folds GroupKFold | `splits.py` |
| 4b | cache | folds LOVO | `splits.py` |
| 4c | cache spectrogrammes | train/val DL | `models_dl.py` |
| 5  | folds + features | modèles entraînés | `models_classical.py` · `models_dl.py` |
| 6  | modèles + métriques | `.pkl/.csv/.joblib/.pt` | `persistence.py` |
| 7  | modèles persistés | M1 + M2 chargés | `models_classical.py` · `liking_model.py` |
| 8  | M2 | `ResponseSurface` + `OptimalZone` | `liking_model.py` |
| 9  | M1.predict + zone | `GapReport` + texte | `gap_analysis.py` |
| 10 | report texte + métadonnée ad | prompt LLM | *à créer* `advisor/prompt_builder.py` |
| 11 | prompt | conseils texte | *à créer* `advisor/llm_advisor.py` |
| 12 | conseils texte | recommandations structurées (JSON) | *à créer* `advisor/recommendations.py` |
| 13 | ad rééditée + EEG re-mesuré | comparaison gap_old vs gap_new | *à créer* `advisor/closed_loop.py` |

---

## 3. Arborescence des fichiers

```
neuro_project/
├── README.md              ← ce document
├── __init__.py            ← exporte config
├── config.py              ← constantes & chemins (aucun I/O)
├── persistence.py         ← save/load DataFrames, joblib, torch
├── data_processing.py     ← features bandpower + périph (étape 2)
├── splits.py              ← GroupKFold, LOVO, agrégation par vidéo
├── models_classical.py    ← RF/GB/Ridge/Dummy + runners CV + persistance
├── spectrograms.py        ← STFT → log-puissance → (14, F, T)
├── models_dl.py           ← CNN PyTorch + boucle d'entraînement
├── liking_model.py        ← M2 (V,A)→Liking + surface + zone optimale
├── gap_analysis.py        ← GapReport + format texte
├── _smoke_test.py         ← test bout-en-bout sur données synthétiques
└── artifacts/             ← (créé à la volée)
    ├── *.pkl  /  *.csv
    └── models/  *.joblib  *.pt
```

---

## 4. Référence détaillée — logique et maths par fonction

### 4.1 `config.py`

Constantes pures. Définit `SAMPLE_RATE=128`, `WINDOW_SIZE=256` (2 s), `STEP_SIZE=16` (0.125 s), `BASELINE_SAMPLES=384` (3 s pré-stimulus), les 14 canaux EEG, les 5 bandes (Theta–Gamma), les 8 canaux périphériques, et la liste des sujets.

`normalise_subject_id(sid)` : accepte `'01'`, `1`, `'1'` → renvoie `'01'`. Centralise le zéro-padding.

---

### 4.2 `data_processing.py` — extraction des 85 features

#### `bandpower(signal, bands, sf)`
Estime la **puissance de bande** par la méthode de **Welch** puis intégration au trapèze.

**Welch** : on découpe le signal `x` en segments de longueur `nperseg` avec recouvrement, on fenêtre par Hann, on prend la FFT de chaque segment, on moyenne les périodogrammes :

```
P(f) = (1 / (M · U)) · Σ_{m=1..M} | FFT(w · x_m) |²(f)
```

où `M` = nombre de segments, `w` = fenêtre de Hann, `U` = facteur de normalisation. Le moyennage **réduit la variance** de l'estimateur (contre un seul périodogramme brut).

**Bandpower** dans `[f_lo, f_hi]` :

```
BP = ∫_{f_lo}^{f_hi} P(f) df  ≈  Σ_k (P(f_k) + P(f_{k+1})) / 2  ·  Δf      (trapèze)
```

Implémentation : `np.trapezoid(psd[mask], freqs[mask])`. Avec `nperseg=128 < WINDOW_SIZE=256`, Welch moyenne effectivement ~3 segments.

#### `peripheral_features(periph_window)`
15 statistiques manuelles sur les 8 canaux périphériques (on ignore hEOG/vEOG car oculaires, non émotionnels) :
- **EMG (zEMG, tEMG)** : `mean(|x|)` (énergie de contraction), `std`.
- **GSR** : `mean`, `std`, `range = max−min`, `slope = np.polyfit(t, x, 1)[0]` (pente linéaire = tendance tonique).
- **Resp** : `mean`, `std`.
- **BVP** : `mean`, `std`, `range` (amplitude pouls).
- **Temp** : `mean`, `slope`.

#### `extract_subject_features(subject_id, force=False)`
- Charge `sXX.dat` (pickle, encoding latin-1) → `data (40,40,8064)`, `labels (40,4)`.
- Pour chaque essai : `start = 384` (exclut la baseline pré-stimulus DEAP), boucle `while start + WINDOW_SIZE <= 8064` avec `step = 16`.
- Par fenêtre : concatène 70 bandpowers EEG + 15 stats périph → vecteur 85.
- `y = labels[:, :2]` = `[Valence, Arousal]`.
- `video_ids` mémorise l'essai (0..39) → sert de **groupe** pour la CV.
- Cache compressé `cache_full_v2/sXX_full_v2.npz`.

#### `load_subject(subject_id, cache_dir)`
Lecture pure du cache → `(X, y, vids)`. Pas de calcul.

---

### 4.3 `splits.py` — validation croisée

#### `window_groupkfold(X, y, vids, n_splits=5)`
Encapsule `sklearn.model_selection.GroupKFold`. Partitionne les **groupes** (vidéos) en `n_splits` blocs. À chaque fold, un bloc devient le test, le reste le train. **Toutes les fenêtres d'une vidéo restent ensemble** → pas de fuite entre fenêtres voisines à 94 % de recouvrement.

GroupKFold est **déterministe** (pas de shuffle / random_state) — c'est l'algorithme qui choisit la partition. Contrainte : `n_splits ≤ |unique(groups)|`.

#### `aggregate_by_video(X, y, vids, agg='mean')`
Réduit `(n_windows, 85)` → `(40, 85)` en moyennant les features de toutes les fenêtres d'une vidéo. Mathématiquement :

```
X_v[i, j] = (1 / |W_i|) · Σ_{k ∈ W_i} X[k, j]
```

où `W_i` = ensemble des indices de fenêtre appartenant à la vidéo `i`. Cible `y_v = first()` (constante par vidéo). C'est l'étape qui rend la régression « vidéo-level » possible : 40 points indépendants par sujet, taille réelle du problème.

#### `video_leaveoneout(Xv, yv, groups)`
`LeaveOneGroupOut` avec `groups = vidéo_id` ⇒ 40 folds : 39 vidéos en train, 1 en test. Chaque vidéo est testée exactement une fois.

#### `assert_no_group_leak(train_idx, test_idx, groups)`
Vérifie `set(groups[train_idx]) ∩ set(groups[test_idx]) == ∅`. Sanity check de développement.

---

### 4.4 `models_classical.py` — modèles ML classiques

#### `MODEL_FACTORIES`
- **RandomForestRegressor** (n=200) — natif multi-sortie : un seul `fit` pour `(V, A)` ; importance par réduction d'impureté (Gini/MSE).
- **MultiOutputRegressor(GradientBoostingRegressor)** — boosting d'arbres ; le wrapper duplique le modèle pour V et A.

#### `LOVO_FACTORIES`
- **DummyRegressor(strategy='mean')** : prédit la moyenne de `y_train`. **Ancre R² = 0**. C'est la référence vis-à-vis de laquelle on juge si un modèle a appris quoi que ce soit.
- **Ridge(alpha=10)** : `min_w ||y − Xw||² + α ||w||²`. Solution fermée : `w = (XᵀX + αI)⁻¹ Xᵀy`. Régularisation L2 → stable en haute dimension / petit n.
- **RandomForest(n=300)**.

#### `_metrics(y_true, y_pred)`
Calcule MAE, RMSE, R² **séparément pour V et A** (`multioutput='raw_values'`) :

```
MAE  = (1/n) Σ |y_i − ŷ_i|
RMSE = √( (1/n) Σ (y_i − ŷ_i)² )
R²   = 1 − Σ (y_i − ŷ_i)² / Σ (y_i − ȳ)²
```

R² < 0 ⇒ le modèle prédit moins bien que la moyenne. C'est le signal honnête sur DEAP au niveau fenêtre, dû au fait qu'une vidéo entière partage la même cible.

#### `_aggregate_window_to_video(...)`
Pour chaque vidéo de test, la prédiction d'ad-level = `médiane` des prédictions de fenêtre (robuste aux fenêtres aberrantes). On recompare alors aux labels vidéo (`first`).

#### `run_window_cv(subject_id, factory, n_splits=5)`
Pour un sujet : boucle sur les 5 folds GroupKFold. À chaque fold :
1. `StandardScaler` fitté **sur le train uniquement** : `z = (x − μ_train) / σ_train`. Pas de fuite.
2. `model = factory().fit(scaled_train, y_train)`.
3. `y_pred = model.predict(scaled_test)`.
4. Métriques.

Retourne la **moyenne des métriques sur les 5 folds** (estimation stable).

#### `run_lovo(subject_id)`
Pour un sujet : pour chaque modèle LOVO, 40 folds LeaveOneVideoOut sur le dataset agrégé `(40, 85)`. Prédiction collectée à `preds[te]` (1 ligne / fold). Métriques calculées sur les 40 vidéos hold-out.

#### `fit_and_save_final(subject_id, model_name, level)`
Réentraîne **sur toutes les données** du sujet (sans split, pour l'inférence finale), sauvegarde `{scaler, model, features, level}` via joblib.

#### `predict_va(subject_id, X, model_name, level)`
Recharge le bundle, applique `scaler.transform`, puis `model.predict` → `(n, 2)`.

---

### 4.5 `spectrograms.py` — features pour le CNN

#### `window_to_spectrogram(signal_1d)`
1. **STFT puissance** via `scipy.signal.spectrogram` :
   ```
   S_xx(f, t) = | STFT{x · w}(f, t) |²
   ```
   `nperseg=64`, `noverlap=32` → `F = nperseg/2 + 1 = 33` bins fréquentiels, `T = 7` bins temporels sur une fenêtre 2 s.
2. **Log-compression** : `log(S + 1e-10)`. Justification : la puissance EEG varie sur plusieurs ordres de grandeur (delta énorme vs gamma minuscule) ; le log compresse la dynamique → meilleure stationnarité pour un CNN. `+eps` évite `log(0) = -∞`.
3. **Crop fréquentiel** : `mask = (f >= 4) & (f <= 45)`. DEAP est déjà filtré 4–45 Hz, on jette les bins inutiles → `F ≈ 21` bins finaux.

Sortie : `(F≈21, T=7)` float32.

#### `window_to_multichannel_spec(eeg_window, channels)`
Applique `window_to_spectrogram` aux 14 canaux EEG et empile sur l'axe 0 → `(14, F, T)`. Les 14 canaux deviennent les « channels » du CNN, comme R/G/B d'une image.

#### `extract_subject_spectrograms(subject_id)`
Même squelette que `extract_subject_features` mais en remplaçant l'extraction Welch par l'extraction STFT multichannel. Sortie `(n_windows, 14, F, T)`, mise en cache `cache_spec/sXX_spec.npz`. ~190 Mo / sujet en float32.

---

### 4.6 `models_dl.py` — CNN PyTorch

#### Architecture `build_cnn(n_eeg_channels=14, n_outputs=2)`

```
Input  (B, 14, F, T)
  │
  ├─ Conv2d(14 → 32, k=3×3, pad=1)          # cross-corrélation 2D
  ├─ BatchNorm2d(32)                         # normalise par mini-batch
  ├─ ReLU                                    # max(0, x)
  ├─ MaxPool2d(k=2×2, ceil_mode=True)        # downsample spatial
  ├─ Conv2d(32 → 64, k=3×3, pad=1)
  ├─ BatchNorm2d(64)
  ├─ ReLU
  └─ AdaptiveAvgPool2d(1×1)                  # → (B, 64, 1, 1)
  │
  └─ Linear(64 → 2)                          # tête de régression
Output (B, 2)  → [V̂, Â]
```

**Convolution 2D** mathématiquement :
```
y[b, c_out, i, j] = Σ_{c_in, di, dj} K[c_out, c_in, di, dj] · x[b, c_in, i+di, j+dj] + bias
```
**BatchNorm** : `y = γ · (x − μ_B) / √(σ_B² + ε) + β`, où `μ_B`, `σ_B²` sont les stats du mini-batch (en train) ; running stats utilisés en eval.

#### `train_cnn(X_train, y_train, X_val, y_val, cfg)`
- **Loss** : `SmoothL1Loss` (Huber). Quadratique près de 0, linéaire au-delà → moins sensible aux outliers que MSE.
  ```
  L(y, ŷ) = { 0.5 (y − ŷ)²        si |y − ŷ| < 1
            { |y − ŷ| − 0.5       sinon
  ```
- **Optimiseur** : `AdamW(lr=1e-3, weight_decay=1e-4)`. AdamW = Adam + L2 décorrélé (régularisation appliquée directement aux poids, pas via le gradient).
- **Boucle** : `epochs=15` par défaut, mini-batches shuffles via `np.random.default_rng(42)`. **Early stopping** sur la val loss avec `patience=3`. Restauration du best state à la fin.

#### `train_subject_cnn_cv(X, y, vids, n_splits=5)`
Window-level GroupKFold pour le CNN. Identique à `run_window_cv` mais avec entraînement DL au lieu de sklearn.

#### `save_cnn / load_cnn`
`save_torch(model.state_dict())` puis reconstruction (`build_cnn` + `load_state_dict`). Permet l'inférence sans re-entraîner.

---

### 4.7 `liking_model.py` — surface de Liking

#### `load_deap_labels(subject_ids)`
Charge uniquement les **labels** DEAP (pas les signaux). Renvoie un long DataFrame `(subject, video, V, A, Dominance, Liking)`. 32 × 40 = 1280 lignes maximum.

#### `fit_liking_model(df_labels, kind='kernel_ridge')`
Apprend `g : (V, A) → Liking` sur ces 1280 (V, A, Liking).

- **KernelRidge** (défaut) : Ridge dans un espace de features induit par un noyau RBF.
  ```
  min_α  || y − K α ||² + λ αᵀ K α
  K(x, x') = exp(−γ ||x − x'||²)
  ```
  Solution : `α = (K + λ I)⁻¹ y`, prédiction `ŷ(x) = Σ_i α_i K(x_i, x)`. Surface lisse non linéaire → adapté pour visualiser une « zone » optimale.
- **RandomForestRegressor** : non paramétrique, capte des effets non monotones.
- **Poly2** : Ridge sur les features polynomiales de degré 2 → surface quadratique interprétable :
  ```
  L̂ = β₀ + β₁ V + β₂ A + β₃ V² + β₄ VA + β₅ A²
  ```

#### `response_surface(model, v_range=(1,9), a_range=(1,9), step=0.1)`
Évalue `g` sur une **grille régulière** :
```
V_grid = [v_min, v_min+step, ..., v_max]      (n_V = 81)
A_grid = [a_min, a_min+step, ..., a_max]      (n_A = 81)
L[i, j] = g(V_grid[i], A_grid[j])             (n_V × n_A)
```
Retourne un `ResponseSurface` sérialisable.

#### `optimal_zone(surface, top_pct=0.10)`
1. **Argmax global** :
   ```
   (i*, j*) = argmax_{i, j}  L[i, j]
   V* = V_grid[i*]   A* = A_grid[j*]   L* = L[i*, j*]
   ```
2. **Masque de zone** : seuil = quantile `1 − top_pct` de `L` → garde les 10 % de cellules les plus likées :
   ```
   thr = Q_{0.90}(L)
   zone_mask = (L >= thr)
   ```
   Permet de conseiller dans un **voisinage**, pas juste sur un pixel.

#### `plot_surface(...)`
Heatmap matplotlib + contour du masque + étoile rouge sur `(V*, A*)`.

---

### 4.8 `gap_analysis.py` — étape 9

#### `per_window_liking(V_hat, A_hat, liking_model, window_size_s, step_s)`
Pour chaque fenêtre i, calcule :
```
L_pred_i = g(V̂_i, Â_i)
t0_i = i · step_s     t1_i = t0_i + window_size_s
```
Renvoie une liste de `WindowSegment(start_s, end_s, V_hat, A_hat, Liking_pred)`.

#### `weak_segments(segments, k=5)`
Tri par `Liking_pred` croissant → top-k des segments les **moins liked**. C'est là qu'il faut intervenir dans l'ad.

#### `compute_gap(V_hat, A_hat, liking_model, zone, V_grid, A_grid, ...)`
Résumé ad-level :
```
V̂ = median(V̂_windows)           Â = median(Â_windows)
L̂ = g(V̂, Â)
ΔV = V* − V̂                     ΔA = A* − Â
distance = √(ΔV² + ΔA²)
in_zone  = zone_mask[ closest_index(V̂), closest_index(Â) ]
```
Médiane choisie pour la robustesse aux fenêtres aberrantes. Retourne un `GapReport` sérialisable + liste des segments faibles.

#### `format_report(report, ad_description=None)`
Sérialise tout en **texte structuré**, prêt à être collé dans un prompt LLM. Sections : PREDICTED RESPONSE, TARGET ZONE, GAP, WEAK SEGMENTS, AD DESCRIPTION. **C'est la frontière de la stage 10** : la vidéo n'étant pas dans DEAP, l'`ad_description` doit venir de l'utilisateur externe.

---

### 4.9 `persistence.py`

- `save_df / load_df` : pickle (préserve dtypes) **et** CSV (lisible humainement) côte à côte.
- `save_obj / load_obj` : `joblib` avec `compress=3` (efficace pour les forêts).
- `save_torch / load_torch` : `torch.save(state_dict)`, import torch paresseux.
- Toutes les versions `load_*` renvoient `None` si le fichier n'existe pas → pas besoin de `try/except` côté appelant.

---

## 5. Quickstart

```python
# 1. Window-level CV sur tous les sujets, tous les modèles classiques
from neuro_project.models_classical import batch_window_cv
from neuro_project.persistence import save_df

df = batch_window_cv()
save_df(df, "df_results")

# 2. LOVO video-level
from neuro_project.models_classical import batch_lovo
save_df(batch_lovo(), "df_lovo")

# 3. Liking surface + zone optimale
from neuro_project.liking_model import (load_deap_labels, fit_liking_model,
                                        response_surface, optimal_zone)
labels = load_deap_labels()
lm   = fit_liking_model(labels, kind="kernel_ridge")
surf = response_surface(lm)
zone = optimal_zone(surf)

# 4. Gap pour un ad mesuré (V_hat, A_hat de M1, par fenêtre)
from neuro_project.gap_analysis import compute_gap, format_report
report = compute_gap(V_hat, A_hat, lm, zone, surf.V_grid, surf.A_grid)
print(format_report(report, ad_description="30s car ad, fast cuts, techno 140 BPM"))

# 5. CNN deep learning (1re fois ~30s/sujet, ensuite cache)
from neuro_project.spectrograms import extract_subject_spectrograms
from neuro_project.models_dl import TrainConfig, train_cnn

X, y, vids = extract_subject_spectrograms("01")
model, res = train_cnn(X[:1500], y[:1500], X[1500:], y[1500:],
                       cfg=TrainConfig(epochs=15))
```

## 6. Vérification que tout marche

```bash
python neuro_project/_smoke_test.py
```
→ doit afficher `ALL CHECKS PASSED`. Ne nécessite ni DEAP ni torch (CNN sauté si torch absent).

---

## 7. Limites connues / non implémenté

- **R² négatif au niveau fenêtre** sur DEAP : c'est structurel (1 label par vidéo de 60 s ; ~465 fenêtres partagent la même cible). Lecture honnête → utiliser le niveau vidéo (LOVO).
- **40 vidéos par sujet** = peu pour la régression continue. La classification high/low (seuil 5) est le protocole DEAP standard, qui fonctionne mieux avec ces features.
- **Étape 10 LLM Advisor** non implémentée — `format_report` produit déjà le bloc texte attendu. Plug into your favourite LLM API.
- **Vidéos DEAP non incluses** dans le dataset (liens YouTube uniquement). La frontière entre étape 9 et 10 est précisément cela : l'ad doit être fournie **en dehors** de DEAP (texte, ou multimodal LLM en production).
