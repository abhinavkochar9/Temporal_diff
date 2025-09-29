#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-style report generator for LymphFit/PostCorrect results.

Reads the CSVs produced by batch_generate_reports.py and builds a clean
report bundle with:
  - Tables: patient-level deviations, exercise aggregates, top/bottom lists
  - Figures: distribution boxplots, group means bars
  - Per-patient LLM prompt files (no LLM calls here)

USAGE
-----
Run after your batch script has generated metrics:
    python paper_reports.py \
        --batch-root outputs/batch_reports \
        --out-root  outputs/paper_reports

Optional flags:
    --fps 25               # if you want captions to reflect a specific FPS
    --topk 3               # how many top/bottom patients to list

This script is READ-ONLY with respect to your metrics. It does not
recompute deviations; it only summarizes and produces report assets.
"""

import os
import argparse
import json
import math
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# Joint ordering for heatmaps (rows)
HEATMAP_JOINTS = [
    "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow",    "Right_Elbow",
    "Left_Wrist",    "Right_Wrist",
]

# ----------------------
# Helpers & formatting
# ----------------------

def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _fmt_mean_sd(series: pd.Series, digits: int = 2) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "NA"
    return f"{s.mean():.{digits}f}±{s.std(ddof=1):.{digits}f}"


def _combined_quality_score(row: pd.Series, w_ang=0.5, w_pos=0.3, w_temp=0.2) -> float:
    """Lower is better. Combines angle, positional (normalized), and temporal components.
    - Angle: mean of elbow + shoulder mae (deg)
    - Positional: pos_norm (shoulder-span–normalized units)
    - Temporal: mean of DTW (unitless) and scaled |lag_s| (×10 to match magnitude)
    """
    ang = np.nanmean([row.get("elbow_angle_mae_deg"), row.get("shoulder_angle_mae_deg")])

    # Prefer normalized value if available; else convert percent→norm
    pos = row.get("pos_norm")
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        pct = row.get("pos_err_pct_mean")
        pos = (float(pct) / 100.0) if (pct is not None and not pd.isna(pct)) else np.nan

    dtw = row.get("dtw_distance")
    lag = row.get("peak_lag_s")
    lag = abs(lag) if isinstance(lag, (int, float)) and not math.isnan(lag) else np.nan
    temp = np.nanmean([dtw, lag * 10.0])

    parts = [
        (w_ang * ang) if not np.isnan(ang) else np.nan,
        (w_pos * pos) if not np.isnan(pos) else np.nan,
        (w_temp * temp) if not np.isnan(temp) else np.nan,
    ]
    if all(np.isnan(parts)):
        return np.nan
    return float(np.nansum(parts))


# ----------------------
# Core generator
# ----------------------

def build_report(batch_root: str, out_root: str, fps: float = 25.0, topk: int = 3) -> None:
    # Inputs from the batch step
    quality_root = os.path.join(batch_root, "quality")
    quality_pat_dir = os.path.join(quality_root, "patient_summaries")
    quality_master_fp = os.path.join(quality_root, "all_patients_quality_master.csv")

    # Outputs for paper-style pack
    tables_dir = os.path.join(out_root, "tables")
    figs_dir = os.path.join(out_root, "figures")
    prompts_dir = os.path.join(out_root, "per_patient_prompts")
    _ensure_dirs(out_root, tables_dir, figs_dir, prompts_dir)

    # Load quality dataframe
    if os.path.exists(quality_master_fp):
        dfq = pd.read_csv(quality_master_fp)
    else:
        if not os.path.isdir(quality_pat_dir):
            raise FileNotFoundError(
                f"No quality data found. Expected either {quality_master_fp} or {quality_pat_dir}/<PATIENT>.csv"
            )
        files = [os.path.join(quality_pat_dir, f) for f in os.listdir(quality_pat_dir) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError(
                f"No per-patient quality CSVs in {quality_pat_dir}. Run the batch metrics first."
            )
        dfq = pd.concat((pd.read_csv(fp) for fp in files), ignore_index=True)

    # Basic checks & normalizations
    required_cols = [
        "patient", "exercise",
        "elbow_angle_mae_deg", "shoulder_angle_mae_deg", "pos_err_pct_mean",
        "dtw_distance", "peak_lag_s", "sym_dev_abs_mean", "jerk_rms"
    ]
    missing = [c for c in required_cols if c not in dfq.columns]
    if missing:
        raise ValueError(f"Quality CSVs are missing columns: {missing}")

    dfq["exercise"] = dfq["exercise"].astype(str)

    # Create normalized positional magnitude column (shoulder-span units)
    if "pos_norm" not in dfq.columns:
        dfq["pos_norm"] = pd.to_numeric(dfq.get("pos_err_pct_mean"), errors="coerce") / 100.0

    # ------------- TABLES -------------
    # Patient-level averages across exercises (Table 1 analogue)
    agg_cols = [
        "elbow_angle_mae_deg", "shoulder_angle_mae_deg", "pos_norm",
        "dtw_distance", "peak_lag_s", "sym_dev_abs_mean", "jerk_rms"
    ]
    tbl_patient = dfq.groupby("patient", as_index=False)[agg_cols].mean()
    tbl_patient.rename(columns={
        "elbow_angle_mae_deg": "Elbow (°)",
        "shoulder_angle_mae_deg": "Shoulder (°)",
        "pos_norm": "Pos (norm units)",
        "dtw_distance": "DTW",
        "peak_lag_s": "Lag (s)",
        "sym_dev_abs_mean": "SymΔ",
        "jerk_rms": "Jerk"
    }, inplace=True)
    tbl_patient_fp = os.path.join(tables_dir, "patient_deviation_table.csv")
    tbl_patient.to_csv(tbl_patient_fp, index=False)

    # Exercise-level aggregates across all patients
    tbl_ex = dfq.groupby("exercise", as_index=False)[agg_cols].mean().sort_values("exercise")
    tbl_ex_fp = os.path.join(tables_dir, "exercise_aggregates.csv")
    tbl_ex.to_csv(tbl_ex_fp, index=False)

    # Top/Bottom-K patients by combined score (lower is better)
    dfq_rank = dfq.copy()
    dfq_rank["comb_score"] = dfq_rank.apply(_combined_quality_score, axis=1)
    patient_rank = dfq_rank.groupby("patient", as_index=False)["comb_score"].mean().dropna()
    patient_rank.sort_values("comb_score", inplace=True)
    topk_df = patient_rank.head(topk).assign(rank="Top")
    bottomk_df = patient_rank.tail(topk).assign(rank="Bottom")
    tbl_tb = pd.concat([topk_df, bottomk_df], ignore_index=True)
    tbl_tb_fp = os.path.join(tables_dir, f"top_bottom_patients_k{topk}.csv")
    tbl_tb.to_csv(tbl_tb_fp, index=False)

    # ------------- FIGURES -------------
    # Fig 1: distribution boxplots for angles and positional error
    try:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].boxplot([dfq["elbow_angle_mae_deg"].dropna()])
        axes[0].set_title("Elbow angle error (°)")
        axes[1].boxplot([dfq["shoulder_angle_mae_deg"].dropna()])
        axes[1].set_title("Shoulder angle error (°)")
        axes[2].boxplot([dfq["pos_norm"].dropna()])
        axes[2].set_title("Positional error (normalized)")
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xticks([])
        fig.tight_layout()
        fig1_fp = os.path.join(figs_dir, "fig1_boxplots.png")
        fig.savefig(fig1_fp, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not write fig1_boxplots: {e}")

    # Fig 2: group means with error bars vs Expert baseline (0)
    try:
        metrics = ["elbow_angle_mae_deg", "shoulder_angle_mae_deg", "pos_norm"]
        means = [pd.to_numeric(dfq[m], errors="coerce").dropna().mean() for m in metrics]
        stds  = [pd.to_numeric(dfq[m], errors="coerce").dropna().std(ddof=1) for m in metrics]
        labels = ["Elbow°", "Shoulder°", "Pos%"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_ylabel("Group mean (±SD)")
        ax.set_title("Group deviations vs Expert baseline")
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        fig2_fp = os.path.join(figs_dir, "fig2_group_vs_expert_bar.png")
        fig.savefig(fig2_fp, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not write fig2_group_vs_expert_bar: {e}")

    # ------------- PER-PATIENT PROMPTS -------------
    # Create one LLM-ready text file per patient with concise metrics by exercise
    for pid, dfp in dfq.groupby("patient"):
        lines: List[str] = []
        lines.append(f"Patient: {pid}")
        lines.append("Context: Metrics compare patient vs Expert baseline (lower is better). Do not fabricate numbers.")
        lines.append(f"FPS used in analysis: {fps}")
        lines.append("Exercises included: " + ", ".join(sorted(dfp["exercise"].unique())))
        lines.append("")
        lines.append("Per-exercise metrics:")
        dfp_sorted = dfp.sort_values("exercise").copy()
        dfp_sorted["comb_score"] = dfp_sorted.apply(_combined_quality_score, axis=1)
        for _, r in dfp_sorted.iterrows():
            lines.append(
                (
                    f"- {r['exercise']}: "
                    f"Elbow°={r['elbow_angle_mae_deg']:.2f}, "
                    f"Shoulder°={r['shoulder_angle_mae_deg']:.2f}, "
                    f"Pos(norm)={r['pos_norm']:.3f}, "
                    f"DTW={r['dtw_distance']:.2f}, "
                    f"Lag(s)={(r['peak_lag_s'] if pd.notna(r['peak_lag_s']) else np.nan):.2f}, "
                    f"SymΔ={r['sym_dev_abs_mean']:.3f}, "
                    f"Jerk={r['jerk_rms']:.2f}, "
                    f"Score={r['comb_score']:.2f}"
                )
            )
        # Identify best/worst exercises by combined score
        best_row = dfp_sorted.loc[dfp_sorted['comb_score'].idxmin()]
        worst_row = dfp_sorted.loc[dfp_sorted['comb_score'].idxmax()]
        lines += [
            "",
            f"Best exercise: {best_row['exercise']} (score {best_row['comb_score']:.2f})",
            f"Worst exercise: {worst_row['exercise']} (score {worst_row['comb_score']:.2f})",
            "",
            "Instructions for LLM (use metrics above; do NOT invent baselines):",
            "- Write a concise clinical-style summary of deviations across angles, positions, and timing.",
            "- Mention worst and best exercises and quantify the errors.",
            "- Suggest 2–3 corrective cues (e.g., raise right elbow ~10°, keep wrist level with shoulder).",
        ]
        with open(os.path.join(prompts_dir, f"{pid}.txt"), 'w') as f:
            f.write("\n".join(lines))

    # ------------- CAPTIONS (OPTIONAL UTILITY) -------------
    captions = {
        "fig1_boxplots": {
            "elbow": _fmt_mean_sd(dfq["elbow_angle_mae_deg"]),
            "shoulder": _fmt_mean_sd(dfq["shoulder_angle_mae_deg"]),
            "pos": _fmt_mean_sd(dfq["pos_norm"]),
        },
        "fig2_group_vs_expert_bar": {
            "note": "Bars show mean±SD across all patient×exercise entries; Expert is baseline (0)."
        },
        "summary": {
            "n_patients": int(dfq["patient"].nunique()),
            "n_exercises": int(dfq["exercise"].nunique()),
            "fps": fps,
        }
    }
    with open(os.path.join(out_root, "captions.json"), 'w') as f:
        json.dump(captions, f, indent=2)

    print("\n[REPORT PACK]")
    print(f"- Patient table:        {tbl_patient_fp}")
    print(f"- Exercise aggregates:  {tbl_ex_fp}")
    print(f"- Top/Bottom:           {tbl_tb_fp}")
    print(f"- Figures dir:          {figs_dir}")
    print(f"- Prompts dir:          {prompts_dir}")
    print(f"- Captions:             {os.path.join(out_root, 'captions.json')}")


# ----------------------
# Minimal loaders & geometry for heatmaps
# ----------------------

def _to_numeric_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.interpolate(limit_direction="both")

def _load_joint_csv(base_dir: str, patient: str, exercise: str, joint: str) -> Optional[pd.DataFrame]:
    path = os.path.join(base_dir, patient, exercise, f"{joint}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or len(df.columns) < 2:
            return None
        out = df.iloc[:, :2].copy()
        out.columns = ["X", "Y"]
        out["X"] = _to_numeric_series(out["X"]) 
        out["Y"] = _to_numeric_series(out["Y"]) 
        return out.reset_index(drop=True)
    except Exception:
        return None

def _make_xy(jmap: Dict[str, pd.DataFrame], name: str) -> Optional[np.ndarray]:
    d = jmap.get(name)
    if d is None or d.empty:
        return None
    return np.stack([d["X"].to_numpy(float), d["Y"].to_numpy(float)], axis=1)

def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a

def _angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    un = np.linalg.norm(u, axis=1)
    vn = np.linalg.norm(v, axis=1)
    denom = np.clip(un * vn, 1e-9, None)
    cos = np.clip(np.sum(u * v, axis=1) / denom, -1.0, 1.0)
    return np.arccos(cos)

def _shoulder_span(jmap: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
    L = _make_xy(jmap, "Left_Shoulder"); R = _make_xy(jmap, "Right_Shoulder")
    if L is None or R is None:
        return None
    return np.linalg.norm(R - L, axis=1)

def _torso_center(jmap: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
    L = _make_xy(jmap, "Left_Shoulder"); R = _make_xy(jmap, "Right_Shoulder")
    if L is None or R is None:
        return None
    return 0.5 * (L + R)

def _elbow_angle_series(jmap: Dict[str, pd.DataFrame], side: str) -> Optional[np.ndarray]:
    S = _make_xy(jmap, f"{side}_Shoulder")
    E = _make_xy(jmap, f"{side}_Elbow")
    W = _make_xy(jmap, f"{side}_Wrist")
    if S is None or E is None or W is None:
        return None
    u = _vec(E, S)  # Elbow->Shoulder
    v = _vec(E, W)  # Elbow->Wrist
    return _angle_between(u, v)

def _shoulder_angle_series(jmap: Dict[str, pd.DataFrame], side: str) -> Optional[np.ndarray]:
    S  = _make_xy(jmap, f"{side}_Shoulder")
    SO = _make_xy(jmap, f"{'Left' if side=='Right' else 'Right'}_Shoulder")
    E  = _make_xy(jmap, f"{side}_Elbow")
    if S is None or SO is None or E is None:
        return None
    u = _vec(S, SO)  # Shoulder->Opposite shoulder
    v = _vec(S, E)   # Shoulder->Elbow
    return _angle_between(u, v)

def _positional_error_series(p_map: Dict[str, pd.DataFrame], e_map: Dict[str, pd.DataFrame], joint: str) -> Optional[np.ndarray]:
    # Returns normalized deviation (shoulder-span units), patient - expert
    tc_p = _torso_center(p_map); tc_e = _torso_center(e_map)
    sp_p = _shoulder_span(p_map); sp_e = _shoulder_span(e_map)
    if tc_p is None or tc_e is None or sp_p is None or sp_e is None:
        return None
    P = _make_xy(p_map, joint); E = _make_xy(e_map, joint)
    if P is None or E is None:
        return None
    dp = np.linalg.norm(P - tc_p, axis=1) / np.clip(sp_p, 1e-9, None)
    de = np.linalg.norm(E - tc_e, axis=1) / np.clip(sp_e, 1e-9, None)
    n = min(len(dp), len(de))
    if n == 0:
        return None
    return (dp[:n] - de[:n])  # normalized (no percent)

def _angle_error_series(p_map: Dict[str, pd.DataFrame], e_map: Dict[str, pd.DataFrame], joint: str) -> Optional[np.ndarray]:
    # Supports shoulder/elbow only; wrists -> None
    if "Shoulder" in joint:
        side = "Left" if joint.startswith("Left") else "Right"
        ap = _shoulder_angle_series(p_map, side)
        ae = _shoulder_angle_series(e_map, side)
    elif "Elbow" in joint:
        side = "Left" if joint.startswith("Left") else "Right"
        ap = _elbow_angle_series(p_map, side)
        ae = _elbow_angle_series(e_map, side)
    else:
        return None
    if ap is None or ae is None:
        return None
    n = min(len(ap), len(ae))
    if n == 0:
        return None
    return np.degrees(ap[:n] - ae[:n])  # deg difference (patient - expert)

def _resample_to_bins(arr: np.ndarray, bins: int = 100) -> np.ndarray:
    if arr is None or len(arr) == 0:
        return np.array([])
    x_old = np.linspace(0.0, 1.0, num=len(arr))
    x_new = np.linspace(0.0, 1.0, num=bins)
    return np.interp(x_new, x_old, arr)


def build_heatmaps(
    patient_dir: str,
    expert_dir: str,
    out_root: str,
    bins: int = 100,
    patients: Optional[List[str]] = None,
    exercises: Optional[List[str]] = None,
) -> None:
    fig_dir = os.path.join(out_root, "figures", "heatmaps")
    os.makedirs(fig_dir, exist_ok=True)

    def _list_dirs(root: str) -> List[str]:
        return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]) if os.path.isdir(root) else []

    all_patients = _list_dirs(patient_dir)
    if not all_patients:
        print(f"[HEATMAP] No patients in {patient_dir}")
        return

    targets_patients = patients if patients else all_patients

    for pid in targets_patients:
        ex_root = os.path.join(patient_dir, pid)
        all_ex = _list_dirs(ex_root)
        if not all_ex:
            print(f"[HEATMAP] Skip {pid}: no exercises")
            continue
        targets_ex = exercises if exercises else all_ex

        for ex in targets_ex:
            # Load joint CSVs
            p_map: Dict[str, Optional[pd.DataFrame]] = {}
            e_map: Dict[str, Optional[pd.DataFrame]] = {}
            for j in HEATMAP_JOINTS:
                p_map[j] = _load_joint_csv(patient_dir, pid, ex, j)
                e_map[j] = _load_joint_csv(expert_dir,  pid, ex, j)

            if not any((p_map[k] is not None and e_map[k] is not None) for k in HEATMAP_JOINTS):
                print(f"[HEATMAP] {pid}/{ex}: missing data; skipping")
                continue

            # Positional error matrix (normalized units)
            pos_rows = []
            for j in HEATMAP_JOINTS:
                s = _positional_error_series(p_map, e_map, j)
                if s is None or len(s) == 0:
                    pos_rows.append(np.full(bins, np.nan))
                else:
                    pos_rows.append(_resample_to_bins(s, bins=bins))
            pos_mat = np.vstack(pos_rows)

            # Angle error matrix (deg) for shoulder/elbow; wrists NaN
            ang_rows = []
            for j in HEATMAP_JOINTS:
                s = _angle_error_series(p_map, e_map, j)
                if s is None or len(s) == 0:
                    ang_rows.append(np.full(bins, np.nan))
                else:
                    ang_rows.append(_resample_to_bins(s, bins=bins))
            ang_mat = np.vstack(ang_rows)

            # Plot helpers
            def _plot_heat(mat: np.ndarray, title: str, fname: str, cbar_label: str):
                fig, ax = plt.subplots(figsize=(10, 3.2))
                im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap='coolwarm')
                ax.set_yticks(range(len(HEATMAP_JOINTS)))
                ax.set_yticklabels(HEATMAP_JOINTS)
                ax.set_xticks([0, bins//4, bins//2, 3*bins//4, bins-1])
                ax.set_xticklabels(["0%","25%","50%","75%","100%"])
                ax.set_xlabel("Normalized time")
                ax.set_title(title)
                cbar = fig.colorbar(im, ax=ax)
                cbar.ax.set_ylabel(cbar_label)
                fig.tight_layout()
                fp = os.path.join(fig_dir, fname)
                fig.savefig(fp, dpi=200)
                plt.close(fig)
                print(f"[HEATMAP] Wrote {fp}")

            _plot_heat(pos_mat, f"Positional error (normalized): {pid} / {ex}", f"{pid}__{ex}__pos.png", "Normalized deviation (shoulder-span units)")
            _plot_heat(ang_mat, f"Angle error (deg): {pid} / {ex}", f"{pid}__{ex}__angle.png", "Degrees")

# ----------------------
# CLI
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Build paper-style report assets from batch metrics.")
    parser.add_argument("--batch-root", default=os.path.join("outputs", "batch_reports"), help="Root of batch metrics outputs")
    parser.add_argument("--out-root", default=os.path.join("outputs", "paper_reports"), help="Destination directory for the report bundle")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS to reference in captions/prompts")
    parser.add_argument("--topk", type=int, default=3, help="Top/Bottom-K patients in the ranking table")
    parser.add_argument("--heatmaps", action="store_true", help="Also generate per-patient×exercise heatmaps")
    parser.add_argument("--patient-dir", default="Patient_Temporal", help="Root folder for patient CSVs")
    parser.add_argument("--expert-dir", default="Expert_Temporal", help="Root folder for expert CSVs")
    parser.add_argument("--heatmap-bins", type=int, default=100, help="Number of normalized time bins per heatmap")
    parser.add_argument("--heatmap-patients", default="ALL", help="Comma-separated patient IDs or ALL")
    parser.add_argument("--heatmap-exercises", default="ALL", help="Comma-separated exercise names or ALL")
    args = parser.parse_args()

    build_report(args.batch_root, args.out_root, fps=args.fps, topk=args.topk)

    if args.heatmaps:
        pats = None if str(args.heatmap_patients).upper() == "ALL" else [p.strip() for p in str(args.heatmap_patients).split(',') if p.strip()]
        exes = None if str(args.heatmap_exercises).upper() == "ALL" else [e.strip() for e in str(args.heatmap_exercises).split(',') if e.strip()]
        build_heatmaps(
            patient_dir=args.patient_dir,
            expert_dir=args.expert_dir,
            out_root=args.out_root,
            bins=args.heatmap_bins,
            patients=pats,
            exercises=exes,
        )


if __name__ == "__main__":
    main()
