#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path

import numpy as np
import pandas as pd

from bodygraph_utils import RESULTS, pheno_task, save_json

PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", pheno_task(PHENO))
INDIR = RESULTS / PHENO / "bodygraph_gtrnn"
OUTDIR = RESULTS / PHENO / "graph_state_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    energy_path = INDIR / "test_energy_states.csv"
    pred_path = INDIR / "test_predictions.csv"
    if not energy_path.exists() or not pred_path.exists():
        raise FileNotFoundError("Run train_graph_temporal.py first")

    e = pd.read_csv(energy_path)
    p = pd.read_csv(pred_path)

    df = e.merge(p[["IID", "step", "y_true", "y_pred_raw", "mask"]], on=["IID", "step"], how="left")
    df = df[df["mask"] > 0].copy()

    # simple state summaries
    df["energy_delta"] = df.groupby("IID")["energy_graph"].diff()
    subj = df.groupby("IID").agg(
        mean_energy=("energy_graph", "mean"),
        sd_energy=("energy_graph", "std"),
        mean_energy_delta=("energy_delta", "mean"),
        final_energy=("energy_graph", "last"),
        final_pred=("y_pred_raw", "last"),
        final_true=("y_true", "last"),
        n_steps=("step", "count"),
    ).reset_index()

    if TASK_TYPE == "binary":
        subj["group"] = np.where(subj["final_true"] >= 0.5, "event_or_case", "no_event")
    else:
        q = subj["final_true"].quantile(0.75)
        subj["group"] = np.where(subj["final_true"] >= q, "high_symptom", "lower_symptom")

    group = subj.groupby("group").agg(
        n_subjects=("IID", "count"),
        mean_energy=("mean_energy", "mean"),
        mean_final_energy=("final_energy", "mean"),
        mean_sd_energy=("sd_energy", "mean"),
        mean_energy_delta=("mean_energy_delta", "mean"),
    ).reset_index()

    # crude collapse flag: large positive rise in graph energy
    thr = subj["mean_energy_delta"].mean() + subj["mean_energy_delta"].std()
    subj["collapse_like_flag"] = (subj["mean_energy_delta"] > thr).astype(int)

    subj.to_csv(OUTDIR / "subject_state_summary.csv", index=False)
    group.to_csv(OUTDIR / "group_state_summary.csv", index=False)

    save_json({
        "pheno": PHENO,
        "task_type": TASK_TYPE,
        "n_subjects": int(subj.shape[0]),
        "energy_definition": "mean squared graph-propagated node activation at each step",
        "collapse_like_flag_rule": "subject mean_energy_delta > global mean + 1 SD",
        "mean_collapse_like_rate": float(subj["collapse_like_flag"].mean()),
    }, OUTDIR / "state_analysis_summary.json")

    print("[INFO] Graph state analysis written to:", OUTDIR)

if __name__ == "__main__":
    main()
