"""Deep-learning baseline — small CNN over EEG spectrograms (stage 5 DL).

The torch import is lazy: just importing this module does NOT require torch.
torch is needed only when calling build_cnn / train_cnn / predict_cnn.

Model: 2-block convnet on (14, F, T) -> regression head -> (V, A).
- conv1: 14 channels -> 32, kernel 3x3, BN, ReLU, MaxPool 2x2
- conv2: 32 -> 64, kernel 3x3, BN, ReLU, AdaptiveAvgPool -> (64, 1, 1)
- fc:    64 -> 2 (linear)
Loss: SmoothL1 (Huber) on V and A. Optimiser: AdamW.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .persistence import load_torch, save_torch
from .splits import window_groupkfold


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:                     # pragma: no cover
        raise ImportError(
            "PyTorch is required for DL models. Install with: "
            "pip install torch") from e
    return __import__("torch")


# ---------------------------------------------------------------- model
def build_cnn(n_eeg_channels: int = 14, n_outputs: int = 2):
    """Returns an nn.Module ready to consume (B, 14, F, T) float tensors."""
    torch = _require_torch()
    nn = torch.nn

    class EEGSpecCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(n_eeg_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(64, n_outputs)

        def forward(self, x):                     # x : (B, C, F, T)
            z = self.features(x).flatten(1)       # (B, 64)
            return self.head(z)                   # (B, n_outputs)

    return EEGSpecCNN()


# ---------------------------------------------------------------- training
@dataclass
class TrainConfig:
    epochs:        int   = 15
    batch_size:    int   = 64
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    device:        str   = "auto"                 # 'auto' | 'cpu' | 'cuda'
    log_every:     int   = 50
    early_stop_patience: int = 3


@dataclass
class TrainResult:
    train_losses: List[float] = field(default_factory=list)
    val_losses:   List[float] = field(default_factory=list)
    best_val:     float = math.inf
    best_epoch:   int = -1


def _resolve_device(device: str):
    torch = _require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _iter_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        yield X[sel], y[sel]


def train_cnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    cfg: Optional[TrainConfig] = None,
) -> Tuple[object, TrainResult]:
    """Standard supervised training loop. Inputs are numpy arrays:
    X: (n, 14, F, T) float32   y: (n, 2) float32"""
    torch = _require_torch()
    cfg = cfg or TrainConfig()
    device = _resolve_device(cfg.device)
    rng = np.random.default_rng(42)

    model = build_cnn(n_eeg_channels=X_train.shape[1]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.SmoothL1Loss()

    res = TrainResult()
    bad_epochs = 0

    for ep in range(cfg.epochs):
        # ----- train
        model.train()
        ep_loss, n = 0.0, 0
        for xb, yb in _iter_minibatches(X_train, y_train, cfg.batch_size, True, rng):
            xb_t = torch.as_tensor(xb, device=device)
            yb_t = torch.as_tensor(yb, device=device)
            optim.zero_grad(set_to_none=True)
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            loss.backward()
            optim.step()
            ep_loss += float(loss.item()) * len(xb); n += len(xb)
        train_loss = ep_loss / max(n, 1)

        # ----- val
        model.eval()
        with torch.no_grad():
            vloss, m = 0.0, 0
            for xb, yb in _iter_minibatches(X_val, y_val, cfg.batch_size, False, rng):
                xb_t = torch.as_tensor(xb, device=device)
                yb_t = torch.as_tensor(yb, device=device)
                vloss += float(loss_fn(model(xb_t), yb_t).item()) * len(xb); m += len(xb)
            val_loss = vloss / max(m, 1)

        res.train_losses.append(train_loss)
        res.val_losses.append(val_loss)
        if val_loss < res.best_val:
            res.best_val = val_loss; res.best_epoch = ep; bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                break

    # restore best
    model.load_state_dict(best_state)
    return model, res


# ---------------------------------------------------------------- CV helper
def train_subject_cnn_cv(
    X: np.ndarray, y: np.ndarray, vids: np.ndarray,
    n_splits: int = 5, cfg: Optional[TrainConfig] = None,
) -> List[TrainResult]:
    """Window-level GroupKFold CV for the CNN. Returns one TrainResult per fold."""
    results: List[TrainResult] = []
    for tr, te in window_groupkfold(X, y, vids, n_splits=n_splits):
        _, res = train_cnn(X[tr], y[tr], X[te], y[te], cfg=cfg)
        results.append(res)
    return results


# ---------------------------------------------------------------- inference / persist
def predict_cnn(model, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    torch = _require_torch()
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], device=device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def save_cnn(model, tag: str) -> str:
    return save_torch(model.state_dict(), tag)


def load_cnn(tag: str, n_eeg_channels: int = 14):
    """Rebuild architecture + load weights. Returns model or None."""
    state = load_torch(tag)
    if state is None:
        return None
    model = build_cnn(n_eeg_channels=n_eeg_channels)
    model.load_state_dict(state)
    return model
