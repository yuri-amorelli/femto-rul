"""Training loop for the RNN.

Explicit and un-magical on purpose — this is the part of PyTorch worth being
able to write from memory: zero_grad -> forward -> loss -> backward -> step.
Includes early stopping on a validation split and gradient clipping (RNNs can
produce exploding gradients).

Experiment tracking stays outside this module: callers can pass an
`on_epoch_end(epoch, train_loss, val_loss)` callback (e.g. an MLflow logger),
so the training loop has no dependency on any tracking library.
"""
from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 100, lr: float = 1e-3, patience: int = 10,
                clip: float = 5.0, device: torch.device | None = None,
                on_epoch_end: Callable[[int, float, float], None] | None = None) -> dict:
    device = device or get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # SmoothL1 (Huber) is less sensitive than MSE to the large RUL values early
    # in a bearing's life, where the target is a flat plateau.
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = {"train": [], "val": []}
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"epoch {epoch:3d} | train {train_loss:10.2f} | val {val_loss:10.2f}")
        if on_epoch_end is not None:
            on_epoch_end(epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val": best_val, "epochs_run": epochs_run}
