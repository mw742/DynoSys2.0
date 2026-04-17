#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, pickle, copy
from pathlib import Path

# Force Python stdlib dataclasses before pandas imports it from PyTorch site-packages
import sys, sysconfig, importlib.util
_stdlib_dc = Path(sysconfig.get_paths()["stdlib"]) / "dataclasses.py"
_spec = importlib.util.spec_from_file_location("dataclasses", str(_stdlib_dc))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules["dataclasses"] = _mod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from bodygraph_utils import RESULTS, set_seed, pheno_task, save_json

PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", pheno_task(PHENO))
SEED = int(os.environ.get("SEED", "2026"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
LAYERS = int(os.environ.get("LAYERS", "1"))
DROPOUT = float(os.environ.get("DROPOUT", "0.20"))
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
PATIENCE = int(os.environ.get("PATIENCE", "8"))
MIN_EPOCHS = int(os.environ.get("MIN_EPOCHS", "15"))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "5"))
MIN_DELTA_CONT = float(os.environ.get("MIN_DELTA_CONT", "0.005"))
MIN_DELTA_BIN = float(os.environ.get("MIN_DELTA_BIN", "0.001"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "5.0"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INDIR = RESULTS / PHENO / "sequence_inputs"
OUTDIR = RESULTS / PHENO / "bodygraph_lstm_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)

class SeqDataset(Dataset):
    def __init__(self, pack):
        self.x = pack["X_feat"]
        self.y = pack["y"]
        self.m = pack["mask"]
        self.t = pack["time_num"]
        self.iid = pack["iid"]

    def __len__(self):
        return len(self.iid)

    def __getitem__(self, i):
        return {
            "x": self.x[i],
            "y": self.y[i],
            "m": self.m[i],
            "t": self.t[i],
            "iid": self.iid[i],
        }

def collate(batch):
    maxlen = max(len(b["y"]) for b in batch)
    feat_dim = batch[0]["x"].shape[1]
    x = np.zeros((len(batch), maxlen, feat_dim), dtype=np.float32)
    y = np.zeros((len(batch), maxlen), dtype=np.float32)
    m = np.zeros((len(batch), maxlen), dtype=np.float32)
    t = np.zeros((len(batch), maxlen), dtype=np.float32)
    lengths = np.zeros(len(batch), dtype=np.int64)
    iids = []

    for i, b in enumerate(batch):
        L = len(b["y"])
        x[i, :L, :] = b["x"]
        y[i, :L] = b["y"]
        m[i, :L] = b["m"]
        t[i, :L] = b["t"]
        lengths[i] = L
        iids.append(b["iid"])

    return {
        "x": torch.tensor(x),
        "y": torch.tensor(y),
        "m": torch.tensor(m),
        "t": torch.tensor(t),
        "lengths": torch.tensor(lengths),
        "iid": iids,
    }

class BodyLSTM(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=1, dropout=0.2, task_type="continuous"):
        super().__init__()
        self.task_type = task_type
        self.lstm = nn.LSTM(
            input_dim + 1,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=(dropout if layers > 1 else 0.0),
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, t):
        xt = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        h, _ = self.lstm(xt)
        out = self.head(h).squeeze(-1)
        return out

def masked_metrics(y_true, y_pred, mask, task_type):
    yt = y_true[mask > 0].astype(float)
    yp = y_pred[mask > 0].astype(float)

    if len(yt) == 0:
        return {}

    out = {"n_obs": int(len(yt))}

    if task_type == "continuous":
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        mae = np.mean(np.abs(yt - yp))
        denom = np.sum((yt - yt.mean()) ** 2)
        r2 = np.nan if denom <= 0 else (1.0 - np.sum((yt - yp) ** 2) / denom)
        corr = np.nan if len(yt) < 3 else np.corrcoef(yt, yp)[0, 1]
        out.update({
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "corr": float(corr),
        })
    else:
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
        p = 1.0 / (1.0 + np.exp(-yp))
        out["event_rate"] = float(yt.mean())
        out["brier"] = float(brier_score_loss(yt.astype(int), p))
        out["auroc"] = float(roc_auc_score(yt.astype(int), p)) if len(np.unique(yt)) > 1 else np.nan
        out["auprc"] = float(average_precision_score(yt.astype(int), p)) if len(np.unique(yt)) > 1 else np.nan

    return out

def evaluate(model, loader):
    model.eval()
    ys, ps, ms, recs = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            m = batch["m"].to(DEVICE)
            t = batch["t"].to(DEVICE)

            pred = model(x, t)

            ys.append(y.cpu().numpy())
            ps.append(pred.cpu().numpy())
            ms.append(m.cpu().numpy())

            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            m_np = m.cpu().numpy()

            for i, iid in enumerate(batch["iid"]):
                L = int(batch["lengths"][i].item())
                for k in range(L):
                    recs.append({
                        "IID": iid,
                        "step": k,
                        "y_true": float(y_np[i, k]),
                        "y_pred_raw": float(pred_np[i, k]),
                        "mask": float(m_np[i, k]),
                    })

    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    m = np.concatenate(ms, axis=0)

    return masked_metrics(y, p, m, TASK_TYPE), pd.DataFrame(recs)

def get_selection_rule(task_type: str):
    if task_type == "continuous":
        return {
            "metric_name": "rmse",
            "mode": "min",
            "min_delta": MIN_DELTA_CONT,
        }
    else:
        return {
            "metric_name": "auprc",
            "mode": "max",
            "min_delta": MIN_DELTA_BIN,
        }

def metric_value(metrics: dict, rule: dict):
    name = rule["metric_name"]
    val = metrics.get(name, np.nan)

    if rule["metric_name"] == "auprc" and (val is None or np.isnan(val)):
        for fallback in ["auroc", "neg_brier"]:
            if fallback == "neg_brier" and "brier" in metrics and not np.isnan(metrics["brier"]):
                return -float(metrics["brier"]), "neg_brier"
            if fallback in metrics and not np.isnan(metrics[fallback]):
                return float(metrics[fallback]), fallback
        return np.nan, "auprc"

    return float(val), name

def is_improved(current: float, best: float, mode: str, min_delta: float):
    if np.isnan(current):
        return False
    if np.isnan(best):
        return True
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)

def main():
    set_seed(SEED)

    with open(INDIR / "train_sequences.pkl", "rb") as f:
        train_pack = pickle.load(f)

    val_path = INDIR / "val_sequences.pkl"
    test_path = INDIR / "test_sequences.pkl"

    if val_path.exists():
        with open(val_path, "rb") as f:
            val_pack = pickle.load(f)
        selection_pack = val_pack
        selection_name = "validation"
    else:
        with open(test_path, "rb") as f:
            val_pack = pickle.load(f)
        selection_pack = val_pack
        selection_name = "test_fallback"
        print("[WARN] val_sequences.pkl not found; using test set for early stopping fallback.")

    with open(test_path, "rb") as f:
        test_pack = pickle.load(f)

    train_ds = SeqDataset(train_pack)
    sel_ds = SeqDataset(selection_pack)
    test_ds = SeqDataset(test_pack)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    sel_loader = DataLoader(sel_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    input_dim = train_pack["X_feat"][0].shape[1]
    model = BodyLSTM(
        input_dim=input_dim,
        hidden=HIDDEN,
        layers=LAYERS,
        dropout=DROPOUT,
        task_type=TASK_TYPE,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss(reduction="none") if TASK_TYPE == "continuous" else nn.BCEWithLogitsLoss(reduction="none")

    rule = get_selection_rule(TASK_TYPE)
    best_metric = np.nan
    best_metric_name = rule["metric_name"]
    best_state = None
    best_epoch = None
    bad_epochs = 0
    stop_reason = "max_epochs_reached"
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            m = batch["m"].to(DEVICE)
            t = batch["t"].to(DEVICE)

            pred = model(x, t)
            loss_mat = criterion(pred, y)
            loss = (loss_mat * m).sum() / torch.clamp(m.sum(), min=1.0)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}")

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()

            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if len(losses) else np.nan
        sel_metrics, _ = evaluate(model, sel_loader)
        current_metric, used_metric_name = metric_value(sel_metrics, rule)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_set": selection_name,
            "selection_metric_requested": rule["metric_name"],
            "selection_metric_used": used_metric_name,
            "selection_metric_value": current_metric,
            **sel_metrics,
        }
        history.append(history_row)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.5f} "
            f"{used_metric_name}={current_metric:.5f}"
        )

        can_start_early_stop = epoch >= max(MIN_EPOCHS, WARMUP_EPOCHS)

        improved = is_improved(
            current=current_metric,
            best=best_metric,
            mode=rule["mode"],
            min_delta=rule["min_delta"],
        )

        if improved:
            best_metric = current_metric
            best_metric_name = used_metric_name
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            if epoch > WARMUP_EPOCHS:
                bad_epochs += 1

        if can_start_early_stop and bad_epochs >= PATIENCE:
            stop_reason = f"no_{rule['metric_name']}_improvement_for_{PATIENCE}_epochs"
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("[WARN] No best_state captured; keeping final model state.")

    test_metrics, pred_df = evaluate(model, test_loader)

    torch.save(model.state_dict(), OUTDIR / "bodygraph_lstm_model.pt")
    pred_df.to_csv(OUTDIR / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(OUTDIR / "training_history.csv", index=False)

    training_summary = {
        "pheno": PHENO,
        "task_type": TASK_TYPE,
        "model": "BodyLSTM",
        "input_dim": input_dim,
        "hidden": HIDDEN,
        "layers": LAYERS,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "selection_set": selection_name,
        "selection_metric_requested": rule["metric_name"],
        "selection_metric_used": best_metric_name,
        "selection_metric_mode": rule["mode"],
        "best_selection_metric_value": None if np.isnan(best_metric) else float(best_metric),
        "min_delta": rule["min_delta"],
        "patience": PATIENCE,
        "min_epochs": MIN_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "grad_clip": GRAD_CLIP,
        "device": DEVICE,
        "stop_reason": stop_reason,
    }
    save_json(training_summary, OUTDIR / "training_summary.json")

    summary = {
        **training_summary,
        **test_metrics,
    }
    save_json(summary, OUTDIR / "summary.json")

    print("[INFO] LSTM outputs written to:", OUTDIR)

if __name__ == "__main__":
    main()
