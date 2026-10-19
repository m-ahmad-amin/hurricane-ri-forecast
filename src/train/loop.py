from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import summarize_predictions


def _pos_weight(loader: DataLoader) -> torch.Tensor:
    labels = []
    for batch in loader:
        labels.append(batch["label"].numpy())
    y = np.concatenate(labels)
    n_pos = max(float((y == 1).sum()), 1.0)
    n_neg = max(float((y == 0).sum()), 1.0)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses = []
    probs = []
    labels = []
    indices = []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            tabular = batch["tabular"].to(device)
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            index = batch["index"].numpy()

            logit = model(tabular, image)
            loss = criterion(logit, label)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            prob = torch.sigmoid(logit).detach().cpu().numpy()
            losses.append(float(loss.detach().cpu()))
            probs.append(prob)
            labels.append(label.detach().cpu().numpy())
            indices.append(index)

    return (
        float(np.mean(losses)),
        np.concatenate(probs),
        np.concatenate(labels),
        np.concatenate(indices),
    )


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: Path,
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = _pos_weight(loaders["train"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_val = float("inf")
    best_state = None
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, _, _, _ = run_epoch(
            model, loaders["train"], criterion, device, optimizer=optimizer
        )
        metrics = {"epoch": epoch, "train_loss": train_loss}
        if "val" in loaders:
            val_loss, val_prob, val_true, _ = run_epoch(
                model, loaders["val"], criterion, device, optimizer=None
            )
            summary = summarize_predictions(val_true, val_prob)
            metrics["val_loss"] = val_loss
            metrics["val_brier"] = summary["brier"]
            metrics["val_bss"] = summary["brier_skill_vs_climatology"]
            score = val_loss
        else:
            score = train_loss

        history.append(metrics)
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f}"
            + (f" val_loss={metrics.get('val_loss', float('nan')):.4f}" if "val_loss" in metrics else "")
        )

        if score < best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "epoch": epoch, "metrics": metrics}, checkpoint_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "checkpoint": str(checkpoint_path), "best_val": best_val}


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    probs = []
    labels = []
    indices = []
    for batch in tqdm(loader, desc="predict", leave=False):
        tabular = batch["tabular"].to(device)
        image = batch["image"].to(device)
        logit = model(tabular, image)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(batch["label"].numpy())
        indices.append(batch["index"].numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(indices)
