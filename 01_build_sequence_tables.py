#!/usr/bin/env python3
from __future__ import annotations
import os, pickle
from pathlib import Path
import pandas as pd

from bodygraph_utils import (
    RESULTS, set_seed, pheno_task, load_panel, split_subjects, feature_columns,
    build_node_groups, build_subject_sequences, transform_subject_sequences,
    make_adjacency, save_json
)

PHENO = os.environ.get("PHENO", "externalizing")
TASK_TYPE = os.environ.get("TASK_TYPE", pheno_task(PHENO))
INPUT_TABLE = os.environ.get("INPUT_TABLE", "")
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.20"))
MAX_TIMEPOINTS = int(os.environ.get("MAX_TIMEPOINTS", "5"))
DROP_TIME = os.environ.get("DROP_TIME", "0") == "1"
MISSINGNESS_THRESHOLD = float(os.environ.get("MISSINGNESS_THRESHOLD", "0.70"))
SEED = int(os.environ.get("SEED", "2026"))

OUTDIR = RESULTS / PHENO / "sequence_inputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    set_seed(SEED)
    df = load_panel(PHENO, INPUT_TABLE if INPUT_TABLE else None)
    train_ids, test_ids = split_subjects(df, test_size=TEST_SIZE, seed=SEED)
    train_df = df[df["IID"].isin(train_ids)].copy()
    test_df = df[df["IID"].isin(test_ids)].copy()

    feat_cols = feature_columns(train_df, PHENO, TASK_TYPE,
                                missingness_threshold=MISSINGNESS_THRESHOLD,
                                drop_time=DROP_TIME)
    groups = build_node_groups(feat_cols)

    train_pack = build_subject_sequences(train_df, PHENO, TASK_TYPE, feat_cols, groups,
                                         node_fit_stats=None, max_timepoints=MAX_TIMEPOINTS)
    test_pack = transform_subject_sequences(test_df, PHENO, TASK_TYPE,
                                            train_pack["feature_stats"], groups,
                                            train_pack["node_stats"], max_timepoints=MAX_TIMEPOINTS)
    adjacency = make_adjacency(train_pack["node_names"])

    with open(OUTDIR / "train_sequences.pkl", "wb") as f:
        pickle.dump(train_pack, f)
    with open(OUTDIR / "test_sequences.pkl", "wb") as f:
        pickle.dump(test_pack, f)
    with open(OUTDIR / "adjacency.pkl", "wb") as f:
        pickle.dump(adjacency, f)

    pd.DataFrame({
        "feature": feat_cols,
        "node_group": [next((k for k,v in groups.items() if c in v), "other") for c in feat_cols]
    }).to_csv(OUTDIR / "feature_to_node_mapping.csv", index=False)

    save_json({
        "pheno": PHENO,
        "task_type": TASK_TYPE,
        "input_table": str(INPUT_TABLE) if INPUT_TABLE else "default_weighted_unified_table",
        "n_train_subjects": len(train_pack["iid"]),
        "n_test_subjects": len(test_pack["iid"]),
        "n_features": len(feat_cols),
        "n_nodes": len(train_pack["node_names"]),
        "node_names": train_pack["node_names"],
        "max_timepoints": MAX_TIMEPOINTS,
        "drop_time": DROP_TIME,
        "missingness_threshold": MISSINGNESS_THRESHOLD,
    }, OUTDIR / "sequence_build_summary.json")

    print(f"[INFO] Sequence inputs written to: {OUTDIR}")

if __name__ == "__main__":
    main()
