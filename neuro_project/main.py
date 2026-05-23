"""main.py — CLI orchestrator for the neuro_project pipeline.

Two ways to use it:

    1. Interactive menu (default)
        python -m neuro_project.main
        python neuro_project/main.py

    2. Scripted subcommands
        python -m neuro_project.main extract        --subjects 01,02
        python -m neuro_project.main extract-spec   --subjects 01
        python -m neuro_project.main train-window   --model RandomForest
        python -m neuro_project.main train-lovo
        python -m neuro_project.main train-cnn      --subject 01 --epochs 10
        python -m neuro_project.main save-models    --subject 01
        python -m neuro_project.main liking         --kind kernel_ridge
        python -m neuro_project.main predict        --subject 01 --model RandomForest
        python -m neuro_project.main advise         --subject 01 --description "30s ..."
        python -m neuro_project.main loop           --old artifacts/report_v1.pkl --new artifacts/report_v2.pkl
        python -m neuro_project.main status
        python -m neuro_project.main smoke
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import traceback
from typing import Iterable, List, Optional

# ---------------------------------------------------------------- console / paths
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                # allow `python main.py`

from neuro_project import config                                              # noqa: E402
from neuro_project.persistence import (                                       # noqa: E402
    list_artifacts, load_df, load_obj, save_df, save_obj,
)


# ================================================================ helpers
BANNER = r"""
============================================================
   neuro_project  ·  EEG → Emotion → Liking → LLM advisor
============================================================
"""

def hr() -> None:
    print("-" * 60)

def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  > {prompt}{suffix}: ").strip()
    return val or default

def ask_int(prompt: str, default: int) -> int:
    while True:
        v = ask(prompt, str(default))
        try:
            return int(v)
        except ValueError:
            print("    (please enter an integer)")

def ask_float(prompt: str, default: float) -> float:
    while True:
        v = ask(prompt, str(default))
        try:
            return float(v)
        except ValueError:
            print("    (please enter a number)")

def ask_yn(prompt: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    v = ask(f"{prompt} ({d})").lower()
    if not v:
        return default
    return v in ("y", "yes", "o", "oui", "1", "true")

def parse_subjects(arg: Optional[str]) -> List[str]:
    """'01,02-04' → ['01','02','03','04']. None → all 32 subjects."""
    if not arg:
        return list(config.SUBJECT_LIST)
    out: List[str] = []
    for token in arg.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            for i in range(int(a), int(b) + 1):
                out.append(f"{i:02d}")
        else:
            out.append(config.normalise_subject_id(token))
    return out


# ================================================================ pipeline commands
# ---------------------------------------------------------------- stage 2 / 5DL prep
def cmd_extract(subjects: Iterable[str]) -> None:
    """Stage 2 — Welch bandpowers + peripheral stats → cache_full_v2/*.npz."""
    from neuro_project.data_processing import extract_subject_features
    subs = list(subjects)
    print(f"Stage 2 : extracting features for {len(subs)} subject(s).")
    print(f"          → {config.CACHE_DIR}")
    try:
        from tqdm.auto import tqdm
        it = tqdm(subs)
    except ImportError:
        it = subs
    for sid in it:
        try:
            X, y, vids = extract_subject_features(sid)
            print(f"  s{sid} : X={X.shape}  y={y.shape}  vids={len(set(vids.tolist()))}")
        except FileNotFoundError as e:
            print(f"  s{sid} : SKIPPED ({e})")
    print("Done.")


def cmd_extract_spec(subjects: Iterable[str]) -> None:
    """Stage 5 DL prep — spectrograms (14, F, T) → cache_spec/*.npz."""
    from neuro_project.spectrograms import extract_subject_spectrograms
    subs = list(subjects)
    print(f"Stage 5 (DL prep) : extracting spectrograms for {len(subs)} subject(s).")
    print(f"          → {config.SPEC_CACHE_DIR}")
    try:
        from tqdm.auto import tqdm
        it = tqdm(subs)
    except ImportError:
        it = subs
    for sid in it:
        try:
            X, y, vids = extract_subject_spectrograms(sid)
            print(f"  s{sid} : X={X.shape}")
        except FileNotFoundError as e:
            print(f"  s{sid} : SKIPPED ({e})")
    print("Done.")


# ---------------------------------------------------------------- stage 4a + 5 classical
def cmd_train_window(subjects: Iterable[str], model: Optional[str] = None,
                     save_name: str = "df_results") -> None:
    """Stages 4a + 5 (classical) — window-level GroupKFold CV. Persists DataFrame."""
    from neuro_project.models_classical import MODEL_FACTORIES, batch_window_cv
    factories = MODEL_FACTORIES if not model else {model: MODEL_FACTORIES[model]}
    print(f"Stage 4a+5 : window-level CV, models = {list(factories)}, subjects = {len(list(subjects))}")
    df = batch_window_cv(subject_ids=list(subjects), factories=factories)
    print(df.groupby("model")[[c for c in df.columns if c.startswith(("win_", "vid_"))]]
            .mean().round(3))
    p = save_df(df, save_name)
    print(f"Persisted: {p}")


def cmd_train_lovo(subjects: Iterable[str], save_name: str = "df_lovo") -> None:
    """Stages 4b + 5 (classical) — LeaveOneVideoOut at video level."""
    from neuro_project.models_classical import batch_lovo
    print(f"Stage 4b+5 : LOVO video-level, subjects = {len(list(subjects))}")
    df = batch_lovo(subject_ids=list(subjects))
    cols = ["mae_V", "mae_A", "r2_V", "r2_A"]
    print(df.groupby("model")[cols].mean().round(3))
    p = save_df(df, save_name)
    print(f"Persisted: {p}")


# ---------------------------------------------------------------- stage 5 DL
def cmd_train_cnn(subject: str, epochs: int = 10, batch_size: int = 64,
                  val_split: float = 0.2) -> None:
    """Stage 5 (DL) — train a CNN on one subject's spectrograms."""
    import numpy as np
    try:
        from neuro_project.models_dl import TrainConfig, save_cnn, train_cnn
        from neuro_project.spectrograms import (extract_subject_spectrograms,
                                                load_subject_spectrograms)
    except ImportError as e:
        print(f"PyTorch not available: {e}")
        return
    sid = config.normalise_subject_id(subject)
    try:
        X, y, vids = load_subject_spectrograms(sid)
    except FileNotFoundError:
        print(f"Spectrogram cache missing for s{sid}. Extracting now.")
        X, y, vids = extract_subject_spectrograms(sid)
    n = len(X)
    n_val = int(n * val_split)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    cfg = TrainConfig(epochs=epochs, batch_size=batch_size)
    print(f"CNN training : subject s{sid}, train={len(tr_idx)} val={len(val_idx)}, "
          f"epochs={epochs}, batch={batch_size}")
    model, res = train_cnn(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], cfg=cfg)
    print(f"Best val loss = {res.best_val:.4f} at epoch {res.best_epoch}")
    tag = f"cnn_s{sid}"
    save_cnn(model, tag)
    print(f"Saved: {tag}.pt in {config.MODELS_DIR}")


# ---------------------------------------------------------------- stage 6 — final fit
def cmd_save_models(subjects: Iterable[str], level: str = "window",
                    model: Optional[str] = None) -> None:
    """Stage 6 — fit final classical models on the WHOLE subject and joblib-dump."""
    from neuro_project.models_classical import (LOVO_FACTORIES, MODEL_FACTORIES,
                                                fit_and_save_final)
    facs = MODEL_FACTORIES if level == "window" else LOVO_FACTORIES
    names = [model] if model else list(facs)
    subs = list(subjects)
    print(f"Stage 6 : fitting {len(names)} model(s) × {len(subs)} subject(s) at level={level}")
    for sid in subs:
        for name in names:
            try:
                tag = fit_and_save_final(sid, name, level=level)
                print(f"  s{sid} / {name} → {tag}.joblib")
            except Exception as e:                                            # noqa: BLE001
                print(f"  s{sid} / {name} : FAILED ({e})")


# ---------------------------------------------------------------- stages 7-8
def cmd_liking(kind: str = "kernel_ridge", save: bool = True) -> None:
    """Stages 7-8 — fit Liking model + compute response surface + optimal zone."""
    from neuro_project.liking_model import (fit_liking_model, load_deap_labels,
                                            optimal_zone, response_surface)
    print("Stage 7-8 : Liking model + optimal zone")
    labels = load_deap_labels()
    if labels.empty:
        print("  No DEAP labels found (data path missing). Aborting.")
        return
    print(f"  {len(labels)} (subject, video) label rows")
    lm = fit_liking_model(labels, kind=kind)
    surf = response_surface(lm)
    zone = optimal_zone(surf)
    print(f"  Surface shape : {surf.shape}")
    print(f"  Argmax        : (V*, A*) = ({zone.V_star:.2f}, {zone.A_star:.2f})  "
          f"Liking* = {zone.Liking_star:.2f}  threshold = {zone.threshold:.2f}")
    if save:
        save_obj({"model": lm, "kind": kind}, "liking_model")
        save_obj({"V_grid": surf.V_grid, "A_grid": surf.A_grid,
                  "Liking": surf.Liking, "zone": zone}, "liking_zone")
        print(f"  Persisted   : liking_model.joblib + liking_zone.joblib")
    return lm, surf, zone


# ---------------------------------------------------------------- stage 9
def cmd_predict(subject: str, model_name: str = "RandomForest",
                level: str = "window",
                report_name: str = "gap_report",
                ad_description: Optional[str] = None) -> None:
    """Stage 9 — load saved model, predict V/A, compare to optimal zone, save report."""
    import numpy as np
    from neuro_project.data_processing import load_subject
    from neuro_project.gap_analysis import compute_gap, format_report
    from neuro_project.models_classical import predict_va
    from neuro_project.splits import aggregate_by_video

    sid = config.normalise_subject_id(subject)
    print(f"Stage 9 : predict & gap for s{sid} ({model_name}, level={level})")

    X, y, vids = load_subject(sid)
    if level == "video":
        X, _, _ = aggregate_by_video(X, y, vids)
    try:
        y_pred = predict_va(sid, X, model_name=model_name, level=level)
    except FileNotFoundError as e:
        print(f"  {e}")
        if ask_yn("Fit + save the model now ?", default=True):
            from neuro_project.models_classical import fit_and_save_final
            fit_and_save_final(sid, model_name, level=level)
            y_pred = predict_va(sid, X, model_name=model_name, level=level)
        else:
            return
    print(f"  predictions shape : {y_pred.shape}")

    # need the Liking model + zone
    bundle_lm   = load_obj("liking_model")
    bundle_zone = load_obj("liking_zone")
    if bundle_lm is None or bundle_zone is None:
        print("  Liking model / zone not persisted yet. Fitting now...")
        out = cmd_liking()
        if out is None:
            return
        lm, surf, zone = out
        V_grid, A_grid = surf.V_grid, surf.A_grid
    else:
        lm   = bundle_lm["model"]
        zone = bundle_zone["zone"]
        V_grid, A_grid = bundle_zone["V_grid"], bundle_zone["A_grid"]

    report = compute_gap(y_pred[:, 0], y_pred[:, 1], lm, zone, V_grid, A_grid)
    text = format_report(report, ad_description=ad_description)
    print(); print(text); print()

    # save report (pickle for advise / loop, .txt for reading)
    p_pkl = os.path.join(config.ARTIFACTS_DIR, report_name + ".pkl")
    with open(p_pkl, "wb") as f:
        pickle.dump(report, f)
    p_txt = os.path.join(config.ARTIFACTS_DIR, report_name + ".txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Persisted: {p_pkl}\n           {p_txt}")
    return report


def _load_report(name_or_path: str):
    """Load a pickled GapReport from a name (looked up under artifacts/) or a path."""
    candidates = [
        name_or_path,
        name_or_path + ".pkl",
        os.path.join(config.ARTIFACTS_DIR, name_or_path),
        os.path.join(config.ARTIFACTS_DIR, name_or_path + ".pkl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"GapReport not found from '{name_or_path}'")


# ---------------------------------------------------------------- stages 10-11-12
def cmd_advise(report_name: str = "gap_report",
               description: str = "", title: str = "",
               duration_s: float = 30.0, genre: str = "",
               target_audience: str = "", tempo: str = "",
               palette: str = "", model_tag: Optional[str] = None,
               temperature: float = 0.4,
               timeout_s: float = 600.0,
               keep_alive: str = "10m",
               preload: bool = True,
               think: Optional[bool] = False,
               num_predict: int = 4096,
               save_name: str = "advice") -> None:
    """Stages 10-11-12 — build prompt, call Ollama, parse, save."""
    import time
    from neuro_project.advisor.llm_advisor   import OllamaClient, OllamaConfig, advise
    from neuro_project.advisor.prompt_builder import AdMetadata
    from neuro_project.advisor.recommendations import (format_recommendations,
                                                        parse_advice,
                                                        save_recommendations)
    print("Stages 10-11-12 : LLM advisor over Ollama")
    report = _load_report(report_name)

    ad = AdMetadata(
        title=title, duration_s=duration_s, genre=genre,
        target_audience=target_audience, tempo=tempo,
        palette=palette, free_text=description,
    )
    cfg_kwargs: dict = {
        "temperature": temperature,
        "timeout_s":   timeout_s,
        "keep_alive":  keep_alive,
        "think":       think,
        "num_predict": num_predict,
    }
    if model_tag:
        cfg_kwargs["model"] = model_tag
    cfg = OllamaConfig(**cfg_kwargs)
    print(f"  Ollama model = {cfg.model}  temperature = {cfg.temperature}  "
          f"timeout = {cfg.timeout_s:.0f}s  keep_alive = {cfg.keep_alive}")
    print(f"  think        = {cfg.think}  num_predict = {cfg.num_predict}")

    if preload:
        print("  Pre-loading model into RAM (first call only)...", end="", flush=True)
        t0 = time.time()
        try:
            OllamaClient(cfg).preload()
            print(f" done in {time.time() - t0:.1f}s")
        except Exception as e:                                                # noqa: BLE001
            print(f" SKIPPED ({e})")

    print("  Generating advice (may take a few minutes on CPU)...", flush=True)
    t0 = time.time()
    try:
        raw = advise(report, ad, cfg)
    except Exception as e:                                                    # noqa: BLE001
        print(f"  Ollama call failed after {time.time() - t0:.1f}s : {e}")
        print("  Hint: increase --timeout-s, try a smaller model, "
              "or run `ollama run <model>` once to pre-warm it.")
        return
    print(f"  Generated in {time.time() - t0:.1f}s ({len(raw)} chars)")

    rs = parse_advice(raw)
    print(); print(format_recommendations(rs)); print()
    p = save_recommendations(rs, os.path.join(config.ARTIFACTS_DIR, save_name + ".json"))
    print(f"Persisted: {p} (+ .txt)")


# ---------------------------------------------------------------- stage 13
def cmd_loop(old_name: str, new_name: str) -> None:
    """Stage 13 — compare two GapReports (before/after re-edit)."""
    from neuro_project.advisor.closed_loop import compare_gaps, format_summary
    old = _load_report(old_name)
    new = _load_report(new_name)
    summary = compare_gaps(old, new)
    print(); print(format_summary(summary)); print()
    p = os.path.join(config.ARTIFACTS_DIR, "loop_summary.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(format_summary(summary))
    print(f"Persisted: {p}")


# ---------------------------------------------------------------- meta
def cmd_status() -> None:
    """Inventory: caches, persisted models, results."""
    import glob
    print("\n--- Caches ---")
    for label, d, pat in [("features", config.CACHE_DIR, "*.npz"),
                          ("spectrograms", config.SPEC_CACHE_DIR, "*.npz")]:
        files = glob.glob(os.path.join(d, pat))
        size_mb = sum(os.path.getsize(f) for f in files) / 1e6
        print(f"  {label:14s}: {len(files):3d} files  ({size_mb:.1f} MB)  → {d}")
    print("\n--- Artifacts ---")
    for f in list_artifacts():
        print(f"  {f}")
    if not list_artifacts():
        print("  (empty)")
    print("\n--- Ollama (advisor) ---")
    try:
        from neuro_project.advisor.llm_advisor import OllamaClient, list_local_models
        c = OllamaClient()
        if c.ping():
            print(f"  reachable at {c.cfg.host}")
            models = list_local_models()
            print(f"  local models: {models if models else '(none pulled)'}")
        else:
            print(f"  unreachable at {c.cfg.host} (run `ollama serve`)")
    except Exception as e:                                                    # noqa: BLE001
        print(f"  check failed: {e}")


def cmd_smoke() -> None:
    """Run both smoke tests."""
    import subprocess
    py = sys.executable
    for path in (os.path.join(HERE, "_smoke_test.py"),
                 os.path.join(HERE, "advisor", "_smoke_test.py")):
        print(f"\n>>> {path}")
        rc = subprocess.call([py, path])
        print(f"<<< exit code {rc}")


# ================================================================ interactive menu
MENU = """
[1]  Extract features (stage 2)              [7]  Liking model + optimal zone (7-8)
[2]  Extract spectrograms (stage 5 DL prep)  [8]  Predict gap for a subject (9)
[3]  Train window-level GroupKFold (4a+5)    [9]  Advise via Ollama (10-11-12)
[4]  Train LOVO video-level (4b+5)           [10] Closed-loop comparison (13)
[5]  Train CNN deep learning (5 DL)          [11] Pipeline status
[6]  Save final models (6)                   [12] Run smoke tests
                                              [0]  Exit
"""

def interactive() -> None:
    print(BANNER)
    while True:
        print(MENU)
        choice = ask("Choice", "0")
        try:
            if   choice == "0": break
            elif choice == "1":
                arg = ask("Subjects (e.g. 01,02-04, or empty=all)", "")
                cmd_extract(parse_subjects(arg or None))
            elif choice == "2":
                arg = ask("Subjects", "01")
                cmd_extract_spec(parse_subjects(arg))
            elif choice == "3":
                arg = ask("Subjects", "01")
                model = ask("Model (RandomForest/GradientBoost, empty=both)", "")
                cmd_train_window(parse_subjects(arg), model=model or None)
            elif choice == "4":
                arg = ask("Subjects", "01")
                cmd_train_lovo(parse_subjects(arg))
            elif choice == "5":
                sid = ask("Subject id", "01")
                ep  = ask_int("Epochs", 10)
                bs  = ask_int("Batch size", 64)
                cmd_train_cnn(sid, epochs=ep, batch_size=bs)
            elif choice == "6":
                arg = ask("Subjects", "01")
                level = ask("Level (window/video)", "window")
                model = ask("Model (empty = all)", "")
                cmd_save_models(parse_subjects(arg), level=level, model=model or None)
            elif choice == "7":
                kind = ask("Kind (kernel_ridge/rf/poly2)", "kernel_ridge")
                cmd_liking(kind=kind)
            elif choice == "8":
                sid = ask("Subject id", "01")
                model = ask("Model name", "RandomForest")
                level = ask("Level (window/video)", "window")
                desc  = ask("Ad description (optional)", "")
                cmd_predict(sid, model_name=model, level=level,
                            ad_description=desc or None)
            elif choice == "9":
                report = ask("Report name (under artifacts/, no ext.)", "gap_report")
                print("  Ad metadata (skip with Enter):")
                title  = ask("  title", "")
                dur    = ask_float("  duration_s", 30.0)
                genre  = ask("  genre", "")
                aud    = ask("  target audience", "")
                tempo  = ask("  tempo", "")
                pal    = ask("  palette", "")
                desc   = ask("  free text", "")
                model  = ask("Ollama model tag (empty = default)", "")
                temp   = ask_float("Temperature", 0.4)
                cmd_advise(report_name=report, title=title, duration_s=dur,
                           genre=genre, target_audience=aud, tempo=tempo,
                           palette=pal, description=desc,
                           model_tag=model or None, temperature=temp)
            elif choice == "10":
                old = ask("Old report (name or path)", "gap_report_v1")
                new = ask("New report (name or path)", "gap_report_v2")
                cmd_loop(old, new)
            elif choice == "11":
                cmd_status()
            elif choice == "12":
                cmd_smoke()
            else:
                print("  Unknown choice.")
        except KeyboardInterrupt:
            print("\n  (interrupted)")
        except Exception as e:                                                # noqa: BLE001
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
        hr()
    print("Bye.")


# ================================================================ argparse main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neuro_project",
                                description="EEG → Emotion → Liking → LLM advisor")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("extract", help="Stage 2 : feature extraction")
    sp.add_argument("--subjects", default=None)

    sp = sub.add_parser("extract-spec", help="Stage 5 DL prep : spectrograms")
    sp.add_argument("--subjects", default=None)

    sp = sub.add_parser("train-window", help="Stage 4a + 5 classical : window-level CV")
    sp.add_argument("--subjects", default=None)
    sp.add_argument("--model", default=None, choices=["RandomForest", "GradientBoost"])

    sp = sub.add_parser("train-lovo", help="Stage 4b + 5 classical : LOVO video-level")
    sp.add_argument("--subjects", default=None)

    sp = sub.add_parser("train-cnn", help="Stage 5 DL : train CNN on spectrograms")
    sp.add_argument("--subject", required=True)
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--batch-size", type=int, default=64)

    sp = sub.add_parser("save-models", help="Stage 6 : fit final models on all data")
    sp.add_argument("--subjects", default=None)
    sp.add_argument("--level", default="window", choices=["window", "video"])
    sp.add_argument("--model", default=None)

    sp = sub.add_parser("liking", help="Stages 7-8 : Liking model + optimal zone")
    sp.add_argument("--kind", default="kernel_ridge", choices=["kernel_ridge", "rf", "poly2"])

    sp = sub.add_parser("predict", help="Stage 9 : predict V/A + gap report")
    sp.add_argument("--subject", required=True)
    sp.add_argument("--model",   default="RandomForest")
    sp.add_argument("--level",   default="window", choices=["window", "video"])
    sp.add_argument("--description", default=None)
    sp.add_argument("--report-name", default="gap_report")

    sp = sub.add_parser("advise", help="Stages 10-11-12 : Ollama advisor")
    sp.add_argument("--report",     default="gap_report")
    sp.add_argument("--description", default="")
    sp.add_argument("--title",      default="")
    sp.add_argument("--duration-s", type=float, default=30.0)
    sp.add_argument("--genre",      default="")
    sp.add_argument("--audience",   default="")
    sp.add_argument("--tempo",      default="")
    sp.add_argument("--palette",    default="")
    sp.add_argument("--ollama-model", default=None)
    sp.add_argument("--temperature", type=float, default=0.4)
    sp.add_argument("--timeout-s",   type=float, default=600.0,
                    help="Request timeout in seconds (default 600).")
    sp.add_argument("--keep-alive",  default="10m",
                    help="Keep model in RAM after the call (Ollama keep_alive).")
    sp.add_argument("--no-preload",  action="store_true",
                    help="Skip the warm-up call.")
    sp.add_argument("--think",       action="store_true",
                    help="Enable the model's reasoning trace (slower). "
                         "Default OFF — structured JSON tasks don't need it.")
    sp.add_argument("--num-predict", type=int, default=4096,
                    help="Max output tokens (default 4096).")

    sp = sub.add_parser("loop", help="Stage 13 : closed-loop comparison")
    sp.add_argument("--old", required=True)
    sp.add_argument("--new", required=True)

    sub.add_parser("status", help="Inventory : caches, models, Ollama")
    sub.add_parser("smoke",  help="Run all smoke tests")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        interactive()
        return 0

    c = args.cmd
    if   c == "extract":       cmd_extract(parse_subjects(args.subjects))
    elif c == "extract-spec":  cmd_extract_spec(parse_subjects(args.subjects))
    elif c == "train-window":  cmd_train_window(parse_subjects(args.subjects), model=args.model)
    elif c == "train-lovo":    cmd_train_lovo(parse_subjects(args.subjects))
    elif c == "train-cnn":     cmd_train_cnn(args.subject, epochs=args.epochs,
                                             batch_size=args.batch_size)
    elif c == "save-models":   cmd_save_models(parse_subjects(args.subjects),
                                               level=args.level, model=args.model)
    elif c == "liking":        cmd_liking(kind=args.kind)
    elif c == "predict":       cmd_predict(args.subject, model_name=args.model,
                                           level=args.level,
                                           ad_description=args.description,
                                           report_name=args.report_name)
    elif c == "advise":        cmd_advise(report_name=args.report,
                                          description=args.description,
                                          title=args.title, duration_s=args.duration_s,
                                          genre=args.genre, target_audience=args.audience,
                                          tempo=args.tempo, palette=args.palette,
                                          model_tag=args.ollama_model,
                                          temperature=args.temperature,
                                          timeout_s=args.timeout_s,
                                          keep_alive=args.keep_alive,
                                          preload=not args.no_preload,
                                          think=args.think,
                                          num_predict=args.num_predict)
    elif c == "loop":          cmd_loop(args.old, args.new)
    elif c == "status":        cmd_status()
    elif c == "smoke":         cmd_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
