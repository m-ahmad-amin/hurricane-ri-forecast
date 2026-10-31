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
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses = []
    probs = []
    labels = []
    indices = []
    use_amp = scaler is not None and device.type == "cuda"
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            tabular = batch["tabular"].to(device, non_blocking=True)
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            index = batch["index"].numpy()

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(amp_device, enabled=use_amp):
                logit = model(tabular, image)
                loss = criterion(logit, label)

            if train_mode:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            prob = torch.sigmoid(logit.detach()).float().cpu().numpy()
            losses.append(float(loss.detach().float().cpu()))
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
    patience: int = 8,
    min_epochs: int = 5,
    use_pos_weight: bool = True,
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if use_pos_weight:
        pos_weight = _pos_weight(loaders["train"]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        pos_weight = None
        criterion = nn.BCEWithLogitsLoss()
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history = []
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, _, _, _ = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if "val" in loaders:
            val_loss, val_prob, val_true, _ = run_epoch(
                model, loaders["val"], criterion, device, optimizer=None, scaler=None
            )
            summary = summarize_predictions(val_true, val_prob)
            metrics["val_loss"] = val_loss
            metrics["val_brier"] = summary["brier"]
            metrics["val_bss"] = summary["brier_skill_vs_climatology"]
            # lower is better: prefer Brier, fall back to loss
            score = float(summary["brier"])
        else:
            score = train_loss

        history.append(metrics)
        scheduler.step()
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f}"
            + (
                f" val_loss={metrics.get('val_loss', float('nan')):.4f}"
                f" val_brier={metrics.get('val_brier', float('nan')):.4f}"
                f" val_bss={metrics.get('val_bss', float('nan')):.4f}"
                if "val_loss" in metrics
                else ""
            )
        )

        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "epoch": epoch,
                    "metrics": metrics,
                    "best_score": best_score,
                },
                checkpoint_path,
            )
        else:
            stale += 1

        if epoch >= min_epochs and stale >= patience:
            print(f"early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "checkpoint": str(checkpoint_path),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "use_pos_weight": bool(use_pos_weight),
        "pos_weight": None if pos_weight is None else float(pos_weight.detach().cpu().item()),
    }


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
    for batch in tqdm(loader, desc="predict", leave=False, mininterval=2.0):
        tabular = batch["tabular"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            logit = model(tabular, image)
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        labels.append(batch["label"].numpy())
        indices.append(batch["index"].numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(indices)
