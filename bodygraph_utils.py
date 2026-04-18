#!/usr/bin/env python3
from __future__ import annotations
import json, os, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path("/gpfs/group/ehlers/mwei/")
RESULTS_V2 = ROOT / "results"

PHENO_TO_TASK = {
    "externalizing": "continuous",
    "internalizing": "continuous",
    "alcohol": "binary",
    "nicotine": "binary",
    "cannabis": "binary",
    "any": "binary",
}

ID_COLS = ["FID", "IID", "src_subject_id", "eventname", "time_num", "time_year"]
NON_FEATURE_PREFIXES = ("Unnamed:",)

def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def pheno_task(pheno: str) -> str:
    if pheno not in PHENO_TO_TASK:
        raise ValueError(f"Unknown phenotype: {pheno}")
    return PHENO_TO_TASK[pheno]

def default_input_table(pheno: str) -> Path:
    return ROOT / "results" / pheno / "unified_inputs" / f"{pheno}_unified_weighted.tsv"

def target_col(pheno: str, task_type: str) -> str:
    return f"{pheno}_t" if task_type == "continuous" else "binary_event"

def exclude_cols(pheno: str, task_type: str) -> set:
    if task_type == "continuous":
        return {
            "FID", "IID", "src_subject_id", "eventname", "time_year",
            f"{pheno}_t", f"{pheno}_lag", f"delta_{pheno}",
        }
    return {
        "FID", "IID", "src_subject_id", "eventname", "time_year",
        "time", "event", "binary_event", "observed_by_anchor",
        "baseline_age_year", "baseline_age_month",
        "anchor_offset_month", "anchor_age_month",
    }

def load_panel(pheno: str, input_table: str | None = None) -> pd.DataFrame:
    path = Path(input_table) if input_table else default_input_table(pheno)
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    df = pd.read_csv(path, sep="\t").copy()
    df["IID"] = df["IID"].astype(str)
    if "time_num" not in df.columns:
        raise ValueError("Input table must contain time_num")
    return df.sort_values(["IID", "time_num"]).reset_index(drop=True)

def feature_columns(df: pd.DataFrame, pheno: str, task_type: str,
                    missingness_threshold: float = 0.70,
                    drop_time: bool = False) -> List[str]:
    exc = exclude_cols(pheno, task_type)
    cols = []
    for c in df.columns:
        if c in exc:
            continue
        if any(str(c).startswith(p) for p in NON_FEATURE_PREFIXES):
            continue
        if drop_time and c == "time_num":
            continue
        cols.append(c)
    num_ok = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() < (1.0 - missingness_threshold):
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        num_ok.append(c)
    return num_ok

def split_subjects(df: pd.DataFrame, test_size: float = 0.2, seed: int = 2026) -> Tuple[set, set]:
    subj = df["IID"].astype(str).drop_duplicates().values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(subj))
    tr, te = next(splitter.split(idx, groups=subj))
    return set(subj[tr]), set(subj[te])

def build_node_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    groups = {
        "foundation": [],
        "covariates": [],
        "environment": [],
        "family": [],
        "physical": [],
        "technology": [],
        "neurocognition": [],
        "brain": [],
        "outcome_history": [],
        "time": [],
        "other": [],
    }
    for c in feature_cols:
        lc = c.lower()
        if c == "time_num":
            groups["time"].append(c)
        elif c.startswith("PC") or lc.startswith("site_") or c in {"sex", "age"}:
            groups["covariates"].append(c)
        elif "prs" in lc or c in {"CNT", "CNT2"}:
            groups["foundation"].append(c)
        elif lc.startswith("environment_"):
            groups["environment"].append(c)
        elif lc.startswith("general_family_context_"):
            groups["family"].append(c)
        elif lc.startswith("physical_health_"):
            groups["physical"].append(c)
        elif lc.startswith("technology_"):
            groups["technology"].append(c)
        elif lc.startswith("neurorecognition_"):
            groups["neurocognition"].append(c)
        elif lc.startswith("brain_") or lc.startswith("smri_") or lc.startswith("dmri_") or lc.startswith("fmri_"):
            groups["brain"].append(c)
        elif "lag" in lc or "delta_" in lc:
            groups["outcome_history"].append(c)
        else:
            groups["other"].append(c)
    return groups

def aggregate_to_nodes(df: pd.DataFrame, groups: Dict[str, List[str]], fit_stats=None):
    node_names = list(groups.keys())
    if fit_stats is None:
        fit_stats = {}
        for node, cols in groups.items():
            if len(cols) == 0:
                fit_stats[node] = {"imputer": None, "scaler": None, "cols": cols}
                continue
            x = df[cols].apply(pd.to_numeric, errors="coerce")
            imp = SimpleImputer(strategy="median")
            xs = imp.fit_transform(x)
            scaler = StandardScaler()
            xs = scaler.fit_transform(xs)
            fit_stats[node] = {"imputer": imp, "scaler": scaler, "cols": cols}
        fitted = True
    else:
        fitted = False

    out = []
    for node in node_names:
        cols = fit_stats[node]["cols"]
        if len(cols) == 0:
            out.append(np.zeros((df.shape[0], 1), dtype=np.float32))
            continue
        x = df[cols].apply(pd.to_numeric, errors="coerce")
        imp = fit_stats[node]["imputer"]
        scaler = fit_stats[node]["scaler"]
        xs = imp.transform(x)
        xs = scaler.transform(xs)
        out.append(xs.mean(axis=1, keepdims=True).astype(np.float32))
    arr = np.concatenate(out, axis=1)
    return arr, node_names, fit_stats if fitted else fit_stats

def make_adjacency(node_names: List[str]) -> np.ndarray:
    # Simple interpretable prior graph
    edges = {
        ("foundation", "brain"), ("foundation", "outcome_history"), ("foundation", "environment"),
        ("covariates", "brain"), ("covariates", "outcome_history"),
        ("environment", "brain"), ("environment", "outcome_history"),
        ("family", "brain"), ("family", "outcome_history"),
        ("physical", "brain"), ("physical", "outcome_history"),
        ("technology", "brain"), ("technology", "outcome_history"),
        ("neurocognition", "brain"), ("neurocognition", "outcome_history"),
        ("time", "outcome_history"), ("brain", "outcome_history"),
    }
    n = len(node_names)
    idx = {n:i for i,n in enumerate(node_names)}
    A = np.eye(n, dtype=np.float32)
    for a,b in edges:
        if a in idx and b in idx:
            A[idx[a], idx[b]] = 1.0
            A[idx[b], idx[a]] = 1.0
    # normalize
    d = A.sum(axis=1, keepdims=True)
    return A / np.clip(d, 1.0, None)

def build_subject_sequences(df: pd.DataFrame, pheno: str, task_type: str,
                            feature_cols: List[str], groups: Dict[str, List[str]],
                            node_fit_stats=None, max_timepoints: int | None = None):
    target = target_col(pheno, task_type)
    if target not in df.columns:
        raise ValueError(f"Target {target} missing from panel")
    subjects = []
    node_names_out = None
    X_list, Xnode_list, y_list, mask_list, time_list, iid_list = [], [], [], [], [], []
    # feature-level imputer/scaler from all rows in this df
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    feat_imputer = SimpleImputer(strategy="median")
    feat_scaler = StandardScaler()
    X_imp = feat_imputer.fit_transform(X_df)
    X_std = feat_scaler.fit_transform(X_imp)
    X_std = pd.DataFrame(X_std, columns=feature_cols, index=df.index)

    node_arr, node_names, node_fit_stats = aggregate_to_nodes(df, groups, fit_stats=node_fit_stats)
    node_df = pd.DataFrame(node_arr, columns=node_names, index=df.index)
    node_names_out = node_names

    for iid, g in df.groupby("IID", sort=False):
        g = g.sort_values("time_num")
        y = pd.to_numeric(g[target], errors="coerce").values.astype(np.float32)
        valid = np.isfinite(y)
        if task_type == "binary":
            # binary rows can have nan if censored before anchor
            if valid.sum() == 0:
                continue
        else:
            if valid.sum() == 0:
                continue
        x = X_std.loc[g.index, feature_cols].values.astype(np.float32)
        xn = node_df.loc[g.index, node_names].values.astype(np.float32)
        t = pd.to_numeric(g["time_num"], errors="coerce").fillna(0).values.astype(np.float32)
        m = valid.astype(np.float32)
        y = np.nan_to_num(y, nan=0.0)
        if max_timepoints is not None:
            x = x[:max_timepoints]
            xn = xn[:max_timepoints]
            y = y[:max_timepoints]
            m = m[:max_timepoints]
            t = t[:max_timepoints]
        X_list.append(x); Xnode_list.append(xn); y_list.append(y); mask_list.append(m); time_list.append(t); iid_list.append(iid)

    feat_stats = {"imputer": feat_imputer, "scaler": feat_scaler, "feature_cols": feature_cols}
    return {
        "X_feat": X_list,
        "X_node": Xnode_list,
        "y": y_list,
        "mask": mask_list,
        "time_num": time_list,
        "iid": iid_list,
        "feature_stats": feat_stats,
        "node_stats": node_fit_stats,
        "node_names": node_names_out,
    }

def transform_subject_sequences(df: pd.DataFrame, pheno: str, task_type: str,
                                feat_stats, groups: Dict[str, List[str]], node_fit_stats,
                                max_timepoints: int | None = None):
    target = target_col(pheno, task_type)
    X_df = df[feat_stats["feature_cols"]].apply(pd.to_numeric, errors="coerce")
    X_imp = feat_stats["imputer"].transform(X_df)
    X_std = feat_stats["scaler"].transform(X_imp)
    X_std = pd.DataFrame(X_std, columns=feat_stats["feature_cols"], index=df.index)

    node_arr, node_names, _ = aggregate_to_nodes(df, groups, fit_stats=node_fit_stats)
    node_df = pd.DataFrame(node_arr, columns=node_names, index=df.index)

    X_list, Xnode_list, y_list, mask_list, time_list, iid_list = [], [], [], [], [], []
    for iid, g in df.groupby("IID", sort=False):
        g = g.sort_values("time_num")
        y = pd.to_numeric(g[target], errors="coerce").values.astype(np.float32)
        valid = np.isfinite(y)
        if task_type == "binary":
            if valid.sum() == 0:
                continue
        else:
            if valid.sum() == 0:
                continue
        x = X_std.loc[g.index, feat_stats["feature_cols"]].values.astype(np.float32)
        xn = node_df.loc[g.index, node_names].values.astype(np.float32)
        t = pd.to_numeric(g["time_num"], errors="coerce").fillna(0).values.astype(np.float32)
        m = valid.astype(np.float32)
        y = np.nan_to_num(y, nan=0.0)
        if max_timepoints is not None:
            x = x[:max_timepoints]
            xn = xn[:max_timepoints]
            y = y[:max_timepoints]
            m = m[:max_timepoints]
            t = t[:max_timepoints]
        X_list.append(x); Xnode_list.append(xn); y_list.append(y); mask_list.append(m); time_list.append(t); iid_list.append(iid)
    return {
        "X_feat": X_list,
        "X_node": Xnode_list,
        "y": y_list,
        "mask": mask_list,
        "time_num": time_list,
        "iid": iid_list,
        "feature_stats": feat_stats,
        "node_stats": node_fit_stats,
        "node_names": node_names,
    }

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

