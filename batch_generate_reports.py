#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch generator for patient deviation reports with stepwise summaries.

This script extends the original batch generator by computing per-exercise and per-joint
deviation metrics (MAE, RMSE, Corr) as before, plus per-exercise quality metrics
(angular, positional, temporal, symmetry, smoothness).  Additionally, it now
calculates a stepwise summary for each patient and exercise by dividing each
exercise into steps based on the number of step images (two repetitions per
exercise are assumed).  For each step in each rep, the script computes the
per-joint mean absolute error and writes these results into per-patient CSV
files under the ``stepwise_summaries`` output directory.

Outputs:

  - outputs/batch_reports/all_patients_master.csv: per-joint metrics for every
    patient/exercise.
  - outputs/batch_reports/patient_summaries/<PATIENT>.csv: per-joint metrics per
    patient.
  - outputs/batch_reports/overall_patient_summary.csv: mean per-joint metrics
    across all exercises and joints for each patient.
  - outputs/batch_reports/quality/patient_summaries/<PATIENT>.csv: per-exercise
    quality metrics (angular, positional, temporal, symmetry, smoothness).
  - outputs/batch_reports/quality/all_patients_quality_master.csv: all quality
    metrics concatenated.
  - outputs/batch_reports/quality/overall_patient_quality_summary.csv: mean
    quality metrics per patient.
  - outputs/batch_reports/stepwise_summaries/<PATIENT>.csv: per-step MAE per
    joint for each exercise and repetition.

The stepwise summary relies on the presence of step images under directories
``Step_Images/Expert/<EXERCISE>`` or ``Step_Images/<EXERCISE>`` in the same
folder as this script (or its parent).  The number of step images defines
the number of steps.  Each exercise is assumed to have two repetitions; the
time series is split equally into two halves, and each half is further divided
into steps corresponding to the number of images.  The per-step MAE is
computed over these segments for each joint.

"""

import os
import json
import math
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# =========================
# ====== CONFIG ===========
# =========================
PATIENT_DIR = "Patient_Temporal"
EXPERT_DIR = "Expert_Temporal"
JOINT_ORDER = [
    "Right_Shoulder", "Left_Shoulder",
    "Right_Elbow",    "Left_Elbow",
    "Right_Wrist",    "Left_Wrist"
]

# Frame rate for temporal metrics (DTW & lag) and jerk
FPS = 25.0  # adjust if your videos use a different fps

# Output folders
OUT_ROOT = os.path.join("outputs", "batch_reports")
OUT_PATIENT_SUM = os.path.join(OUT_ROOT, "patient_summaries")
os.makedirs(OUT_PATIENT_SUM, exist_ok=True)

# Extra quality metrics outputs (per exercise)
OUT_QUALITY_ROOT = os.path.join(OUT_ROOT, "quality")
OUT_QUALITY_PATIENT = os.path.join(OUT_QUALITY_ROOT, "patient_summaries")
os.makedirs(OUT_QUALITY_PATIENT, exist_ok=True)

# Stepwise summary output directory
OUT_STEPWISE_PATIENT = os.path.join(OUT_ROOT, "stepwise_summaries")
os.makedirs(OUT_STEPWISE_PATIENT, exist_ok=True)

# Behavior
SYNC_LENGTH = True          # trim to common length for metrics alignment
ENABLE_DESPIKE = True       # apply same aggressive despike as app to patient data

# Despike params (mirroring your app defaults / sliders)
DESPIKE_PARAMS = dict(
    win=35,             # local window (odd)
    k_neighbors=10,     # neighbors each side
    z_thresh=1.5,       # MAD multiplier
    jump_abs=None,      # absolute jump threshold (None disables)
    jump_pct_range=25.0,# % of local range
    passes=2,           # repeat passes
    post_ma=3           # small moving average (odd); set None/0 to disable
)

# =========================
# ====== HELPERS ==========
# =========================

def list_dirs(root: str) -> List[str]:
    if not (root and os.path.isdir(root)):
        return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


def _to_numeric_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.interpolate(limit_direction="both")


def load_joint_csv(base_dir: str, patient_id: str, exercise: str, joint: str) -> Optional[pd.DataFrame]:
    """
    Return DataFrame with columns ['X','Y'] or None if missing/invalid.
    """
    path = os.path.join(base_dir, patient_id, exercise, f"{joint}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or len(df.columns) < 2:
            return None
        df2 = df.iloc[:, :2].copy()
        df2.columns = ["X", "Y"]
        df2["X"] = _to_numeric_series(df2["X"])
        df2["Y"] = _to_numeric_series(df2["Y"])
        return df2.reset_index(drop=True)
    except Exception:
        return None


def _neighbor_mean(x: np.ndarray, i: int, k: int) -> float:
    n = len(x)
    left = max(0, i - k)
    right = min(n, i + k + 1)
    idx = list(range(left, i)) + list(range(i+1, right))
    if not idx:
        return float(x[i])
    return float(np.nanmean(x[idx]))


def _local_stats(x: np.ndarray, i: int, half_win: int) -> Tuple[float, float, float]:
    n = len(x)
    l = max(0, i - half_win)
    r = min(n, i + half_win + 1)
    seg = x[l:r]
    med = float(np.nanmedian(seg))
    mad = float(np.nanmedian(np.abs(seg - med)) + 1e-9)
    low = np.nanpercentile(seg, 5) if np.isfinite(np.nanpercentile(seg, 5)) else med
    high = np.nanpercentile(seg, 95) if np.isfinite(np.nanpercentile(seg, 95)) else med
    loc_range = max(1e-9, float(high - low))
    return med, mad, loc_range


def _despike_series(
    s: pd.Series,
    win: int = 35,
    k_neighbors: int = 10,
    z_thresh: float = 1.5,
    jump_abs: Optional[float] = None,
    jump_pct_range: float = 25.0,
    passes: int = 2,
    post_ma: Optional[int] = 3
) -> pd.Series:
    if s is None or s.empty:
        return s
    x = pd.to_numeric(s, errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return pd.Series(x, index=s.index)

    win = max(3, int(win))
    if win % 2 == 0: win += 1
    half = win // 2

    out = x.copy()
    for _ in range(max(1, passes)):
        x = out.copy()
        for i in range(1, n - 1):
            nb_mean = _neighbor_mean(x, i, k_neighbors)
            med, mad, loc_rng = _local_stats(x, i, half)

            thr_mad = z_thresh * mad
            thr_jump = 0.0
            if jump_abs is not None:
                thr_jump = max(thr_jump, float(jump_abs))
            if jump_pct_range is not None:
                thr_jump = max(thr_jump, float(jump_pct_range) / 100.0 * loc_rng)

            dev_nb = abs(x[i] - nb_mean)
            jump_prev = abs(x[i] - x[i-1])
            jump_next = abs(x[i] - x[i+1])

            if (dev_nb > thr_mad) or ((jump_prev > thr_jump) and (jump_next > thr_jump)):
                out[i] = nb_mean

    if post_ma and post_ma >= 3:
        k = int(post_ma)
        if k % 2 == 0: k += 1
        pad = k // 2
        padded = np.pad(out, (pad, pad), mode="reflect")
        kernel = np.ones(k) / k
        out = np.convolve(padded, kernel, mode="valid")

    return pd.Series(out, index=s.index)


def clean_patient_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    p = df.copy()
    p["X"] = _despike_series(p["X"], **DESPIKE_PARAMS)
    p["Y"] = _despike_series(p["Y"], **DESPIKE_PARAMS)
    return p.interpolate(limit_direction="both").reset_index(drop=True)


def _has_any_data(jmap: Dict[str, Optional[pd.DataFrame]]) -> bool:
    """Return True if any DataFrame in the map is non-empty."""
    for v in jmap.values():
        if v is not None and not v.empty:
            return True
    return False


# ---------- Geometry & signal helpers for extended metrics ----------

def _make_xy(jmap: Dict[str, pd.DataFrame], name: str) -> Optional[np.ndarray]:
    d = jmap.get(name)
    if d is None or d.empty:
        return None
    return np.stack([d["X"].to_numpy(float), d["Y"].to_numpy(float)], axis=1)


def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a  # from a -> b


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
    S = _make_xy(jmap, f"{side}_Shoulder")
    SO = _make_xy(jmap, f"{'Left' if side=='Right' else 'Right'}_Shoulder")
    E = _make_xy(jmap, f"{side}_Elbow")
    if S is None or SO is None or E is None:
        return None
    u = _vec(S, SO)  # Shoulder->Opposite shoulder (defines shoulder line)
    v = _vec(S, E)   # Shoulder->Elbow
    return _angle_between(u, v)


def _positional_error_pct(p_map: Dict[str, pd.DataFrame], e_map: Dict[str, pd.DataFrame]) -> Optional[float]:
    tc_p = _torso_center(p_map); tc_e = _torso_center(e_map)
    sp_p = _shoulder_span(p_map); sp_e = _shoulder_span(e_map)
    if tc_p is None or tc_e is None or sp_p is None or sp_e is None:
        return None
    vals = []
    for j in JOINT_ORDER:
        P = _make_xy(p_map, j); E = _make_xy(e_map, j)
        if P is None or E is None:
            continue
        dp = np.linalg.norm(P - tc_p, axis=1) / np.clip(sp_p, 1e-9, None)
        de = np.linalg.norm(E - tc_e, axis=1) / np.clip(sp_e, 1e-9, None)
        n = min(len(dp), len(de))
        if n == 0:
            continue
        vals.append(np.abs(dp[:n] - de[:n]))
    if not vals:
        return None
    mae = float(np.mean(np.concatenate(vals)))
    return 100.0 * mae


def _global_motion_signal(jmap: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
    tc = _torso_center(jmap); sp = _shoulder_span(jmap)
    if tc is None or sp is None:
        return None
    parts = []
    for j in JOINT_ORDER:
        P = _make_xy(jmap, j)
        if P is None:
            return None
        parts.append(np.linalg.norm(P - tc, axis=1) / np.clip(sp, 1e-9, None))
    return np.mean(np.stack(parts, axis=1), axis=1)


def _crosscorr_lag_seconds(x: np.ndarray, y: np.ndarray, fps: float) -> float:
    n = min(len(x), len(y))
    if n == 0:
        return float('nan')
    x = x[:n] - np.mean(x[:n]); y = y[:n] - np.mean(y[:n])
    c = np.correlate(x, y, mode='full')
    lags = np.arange(-n+1, n)
    best = int(lags[np.argmax(c)])
    # Lag is positive if patient occurs after expert
    return float(-best) / float(fps)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from fastdtw import fastdtw
        dist, _ = fastdtw(a, b, dist=lambda x,y: abs(x-y))
        return float(dist)
    except Exception:
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return float('nan')
        D = np.full((n+1, m+1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(a[i-1] - b[j-1])
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        return float(D[n, m])


def _deg(rad: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if rad is None:
        return None
    return np.degrees(rad)


def _mae_arr(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float('nan')
    n = min(len(a), len(b))
    if n == 0:
        return float('nan')
    return float(np.mean(np.abs(a[:n] - b[:n])))


def _jerk_rms(jmap: Dict[str, pd.DataFrame], fps: float) -> Optional[float]:
    dt = 1.0 / float(fps)
    mags = []
    for j in JOINT_ORDER:
        J = jmap.get(j)
        if J is None or J.empty:
            return None
        xy = np.stack([J['X'].to_numpy(float), J['Y'].to_numpy(float)], axis=1)
        r = np.linalg.norm(xy, axis=1)
        if len(r) < 5:
            return None
        jerk = np.diff(r, n=3) / (dt ** 3)
        mags.append(jerk)
    minlen = min(len(m) for m in mags)
    if minlen <= 0:
        return None
    J = np.stack([m[:minlen] for m in mags], axis=1)
    return float(np.sqrt(np.mean(J ** 2)))


def _symmetry_deviation_abs_mean(p_map: Dict[str, pd.DataFrame], e_map: Dict[str, pd.DataFrame]) -> Optional[float]:
    tc_p = _torso_center(p_map); sp_p = _shoulder_span(p_map)
    tc_e = _torso_center(e_map); sp_e = _shoulder_span(e_map)
    if tc_p is None or sp_p is None or tc_e is None or sp_e is None:
        return None
    diffs = []
    for part in ["Shoulder","Elbow","Wrist"]:
        PL = _make_xy(p_map, f"Left_{part}");  PR = _make_xy(p_map, f"Right_{part}")
        EL = _make_xy(e_map, f"Left_{part}");  ER = _make_xy(e_map, f"Right_{part}")
        if PL is None or PR is None or EL is None or ER is None:
            continue
        ml = np.linalg.norm(PL - tc_p, axis=1) / np.clip(sp_p, 1e-9, None)
        mr = np.linalg.norm(PR - tc_p, axis=1) / np.clip(sp_p, 1e-9, None)
        dl = np.linalg.norm(EL - tc_e, axis=1) / np.clip(sp_e, 1e-9, None)
        dr = np.linalg.norm(ER - tc_e, axis=1) / np.clip(sp_e, 1e-9, None)
        n = min(len(ml), len(mr), len(dl), len(dr))
        if n == 0:
            continue
        sym_p = ml[:n] - mr[:n]
        sym_e = dl[:n] - dr[:n]
        diffs.append(np.abs(sym_p - sym_e))
    if not diffs:
        return None
    return float(np.mean(np.concatenate(diffs)))


def quick_metrics(a: pd.Series, b: pd.Series) -> Tuple[float, float, float]:
    n = min(len(a), len(b))
    if n == 0:
        return (np.nan, np.nan, np.nan)
    a = a.iloc[:n].astype(float)
    b = b.iloc[:n].astype(float)
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    corr = float(np.corrcoef(a, b)[0, 1]) if (np.std(a) > 0 and np.std(b) > 0) else np.nan
    return (mae, rmse, corr)


# ---------- Step image helpers for stepwise summary ----------

def _canonicalize(name: str) -> str:
    """Return a lowercase alphanumeric version of the string (remove non-alphanumeric)."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _natural_key(s: str) -> List:
    """Key for natural sorting (numeric parts as integers)."""
    _num_re = re.compile(r'(\d+)')
    s = os.path.basename(str(s))
    return [int(t) if t.isdigit() else t.lower() for t in _num_re.split(s)]


def _list_step_images_in_dir(ex_dir: str) -> List[str]:
    """List image files directly under the given exercise directory."""
    if not ex_dir or not os.path.isdir(ex_dir):
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [os.path.join(ex_dir, f) for f in os.listdir(ex_dir)
             if os.path.isfile(os.path.join(ex_dir, f)) and os.path.splitext(f)[1].lower() in exts]
    files.sort(key=_natural_key)
    return files


def _resolve_step_images_for_exercise(base_dir: str, exercise_name: str) -> List[str]:
    """Resolve step image paths for a given exercise.

    Search common locations relative to base_dir for step images of the exercise.
    Returns a list of image file paths. If none found, returns empty list.
    The search order tries several name variants and directories (Expert, then generic).
    """
    if not base_dir:
        return []
    variants = []
    raw = exercise_name
    variants.extend([
        raw,
        raw.replace("_", " "),
        raw.replace(" ", "_"),
        raw.replace(" ", ""),
        _canonicalize(raw),
        raw.title().replace(" ", ""),
    ])
    roots = [base_dir, os.getcwd(), os.path.dirname(base_dir)]
    subdirs = [os.path.join("Step_Images", "Expert"), os.path.join("Step_Images")]
    for root in roots:
        for sub in subdirs:
            for v in variants:
                ex_dir = os.path.join(root, sub, v)
                step_paths = _list_step_images_in_dir(ex_dir)
                if step_paths:
                    return step_paths
    return []


# =========================
# ===== MAIN LOGIC ========
# =========================

def main():
    if not os.path.isdir(PATIENT_DIR):
        raise FileNotFoundError(f"Patient data directory not found: {PATIENT_DIR}")
    if not os.path.isdir(EXPERT_DIR):
        print(f"[WARN] Expert data directory not found: {EXPERT_DIR} — rows may be NaN where expert data is missing.")

    patients = list_dirs(PATIENT_DIR)
    if not patients:
        raise RuntimeError(f"No patient folders in {PATIENT_DIR}")

    master_rows = []

    for patient in patients:
        exercises = list_dirs(os.path.join(PATIENT_DIR, patient))
        if not exercises:
            print(f"[INFO] skipping {patient}: no exercises found.")
            continue

        per_patient_rows = []
        per_exercise_quality_rows = []
        per_patient_stepwise_rows = []  # store per-step metrics for this patient

        for ex in exercises:
            # Load all joints
            patient_dfs: Dict[str, Optional[pd.DataFrame]] = {}
            expert_dfs:  Dict[str, Optional[pd.DataFrame]] = {}

            for joint in JOINT_ORDER:
                p_df = load_joint_csv(PATIENT_DIR, patient, ex, joint)
                e_df = load_joint_csv(EXPERT_DIR,  patient, ex, joint)

                if ENABLE_DESPIKE and p_df is not None:
                    p_df = clean_patient_df(p_df)

                if SYNC_LENGTH and (p_df is not None) and (e_df is not None):
                    n = min(len(p_df), len(e_df))
                    p_df = p_df.iloc[:n].reset_index(drop=True)
                    e_df = e_df.iloc[:n].reset_index(drop=True)

                patient_dfs[joint] = p_df
                expert_dfs[joint]  = e_df

            # ----- Extended quality metrics (per exercise) -----
            if _has_any_data(patient_dfs) and _has_any_data(expert_dfs):
                # Spatial: angles (deg) averaged across sides
                el_L_p = _elbow_angle_series(patient_dfs, "Left")
                el_R_p = _elbow_angle_series(patient_dfs, "Right")
                el_L_e = _elbow_angle_series(expert_dfs,  "Left")
                el_R_e = _elbow_angle_series(expert_dfs,  "Right")
                elbow_mae_deg = np.nanmean([
                    _mae_arr(_deg(el_L_p), _deg(el_L_e)),
                    _mae_arr(_deg(el_R_p), _deg(el_R_e))
                ])

                sh_L_p = _shoulder_angle_series(patient_dfs, "Left")
                sh_R_p = _shoulder_angle_series(patient_dfs, "Right")
                sh_L_e = _shoulder_angle_series(expert_dfs,  "Left")
                sh_R_e = _shoulder_angle_series(expert_dfs,  "Right")
                shoulder_mae_deg = np.nanmean([
                    _mae_arr(_deg(sh_L_p), _deg(sh_L_e)),
                    _mae_arr(_deg(sh_R_p), _deg(sh_R_e))
                ])

                pos_err_pct = _positional_error_pct(patient_dfs, expert_dfs)

                # Temporal: DTW and peak lag
                sig_p = _global_motion_signal(patient_dfs)
                sig_e = _global_motion_signal(expert_dfs)
                if sig_p is None or sig_e is None:
                    dtw_dist = float('nan'); lag_s = float('nan')
                else:
                    dtw_dist = _dtw_distance(sig_p, sig_e)
                    lag_s = _crosscorr_lag_seconds(sig_p, sig_e, FPS)

                # Symmetry and Smoothness
                sym_dev = _symmetry_deviation_abs_mean(patient_dfs, expert_dfs)
                jerk = _jerk_rms(patient_dfs, FPS)
            else:
                elbow_mae_deg = shoulder_mae_deg = pos_err_pct = float('nan')
                dtw_dist = lag_s = sym_dev = jerk = float('nan')

            per_exercise_quality_rows.append(dict(
                patient=patient,
                exercise=ex,
                elbow_angle_mae_deg=(round(float(elbow_mae_deg),4) if not np.isnan(elbow_mae_deg) else np.nan),
                shoulder_angle_mae_deg=(round(float(shoulder_mae_deg),4) if not np.isnan(shoulder_mae_deg) else np.nan),
                pos_err_pct_mean=(round(float(pos_err_pct),4) if (pos_err_pct is not None and not np.isnan(pos_err_pct)) else np.nan),
                dtw_distance=(round(float(dtw_dist),6) if not (isinstance(dtw_dist,float) and math.isnan(dtw_dist)) else np.nan),
                peak_lag_s=(round(float(lag_s),4) if not (isinstance(lag_s,float) and math.isnan(lag_s)) else np.nan),
                sym_dev_abs_mean=(round(float(sym_dev),6) if (sym_dev is not None and not np.isnan(sym_dev)) else np.nan),
                jerk_rms=(round(float(jerk),6) if (jerk is not None and not np.isnan(jerk)) else np.nan),
            ))

            # ----- Compute metrics per joint -----
            for joint in JOINT_ORDER:
                p_df = patient_dfs[joint]
                e_df = expert_dfs[joint]

                if p_df is None or e_df is None or len(p_df) == 0 or len(e_df) == 0:
                    row = dict(
                        patient=patient, exercise=ex, joint=joint,
                        mae_x=np.nan, rmse_x=np.nan, corr_x=np.nan,
                        mae_y=np.nan, rmse_y=np.nan, corr_y=np.nan
                    )
                else:
                    mae_x, rmse_x, corr_x = quick_metrics(p_df["X"], e_df["X"])
                    mae_y, rmse_y, corr_y = quick_metrics(p_df["Y"], e_df["Y"])
                    row = dict(
                        patient=patient, exercise=ex, joint=joint,
                        mae_x=round(mae_x, 6), rmse_x=round(rmse_x, 6), corr_x=(round(corr_x, 6) if not np.isnan(corr_x) else np.nan),
                        mae_y=round(mae_y, 6), rmse_y=round(rmse_y, 6), corr_y=(round(corr_y, 6) if not np.isnan(corr_y) else np.nan),
                    )

                master_rows.append(row)
                per_patient_rows.append(row)

            # ----- Compute stepwise metrics -----
            # Determine number of steps based on step images
            base_dir = os.path.dirname(os.path.abspath(__file__))
            step_paths = _resolve_step_images_for_exercise(base_dir, ex)
            step_count = len(step_paths)
            if step_count > 0 and _has_any_data(patient_dfs) and _has_any_data(expert_dfs):
                # Determine length of time series for segmentation
                # Use first joint with data as reference
                ref_joint = None
                for j in JOINT_ORDER:
                    if patient_dfs.get(j) is not None and expert_dfs.get(j) is not None:
                        ref_joint = j
                        break
                if ref_joint is not None:
                    p_len = len(patient_dfs[ref_joint]) if patient_dfs[ref_joint] is not None else 0
                    e_len = len(expert_dfs[ref_joint]) if expert_dfs[ref_joint] is not None else 0
                    L = min(p_len, e_len) if SYNC_LENGTH else max(p_len, e_len)
                    if L >= 2 * step_count and step_count > 0:
                        rep_len = L // 2
                        # Names for steps derived from image file names
                        step_names = [os.path.splitext(os.path.basename(p))[0] for p in step_paths]
                        for rep_i, rep_start in enumerate([0, rep_len], start=1):
                            for idx, step_name in enumerate(step_names, start=1):
                                a_start = rep_start + int(math.floor((idx - 1) * rep_len / step_count))
                                a_end = rep_start + int(math.floor(idx * rep_len / step_count))
                                b_start = a_start
                                b_end = a_end
                                step_metrics: Dict[str, float] = {}
                                for joint in JOINT_ORDER:
                                    p_df = patient_dfs[joint]
                                    e_df = expert_dfs[joint]
                                    if p_df is None or e_df is None:
                                        step_metrics[joint] = np.nan
                                    else:
                                        # Ensure indexes within bounds
                                        a0 = max(0, min(len(p_df) - 1, a_start))
                                        b0 = max(a0 + 1, min(len(p_df), a_end))
                                        a1 = max(0, min(len(e_df) - 1, b_start))
                                        b1 = max(a1 + 1, min(len(e_df), b_end))
                                        n = min(b0 - a0, b1 - a1)
                                        if n <= 0:
                                            step_metrics[joint] = np.nan
                                        else:
                                            p_x = p_df["X"].iloc[a0:a0+n].to_numpy(dtype=float)
                                            p_y = p_df["Y"].iloc[a0:a0+n].to_numpy(dtype=float)
                                            e_x = e_df["X"].iloc[a1:a1+n].to_numpy(dtype=float)
                                            e_y = e_df["Y"].iloc[a1:a1+n].to_numpy(dtype=float)
                                            mae_x = float(np.mean(np.abs(p_x - e_x)))
                                            mae_y = float(np.mean(np.abs(p_y - e_y)))
                                            step_metrics[joint] = 0.5 * (mae_x + mae_y)
                                # Build row
                                row = {
                                    "patient": patient,
                                    "exercise": ex,
                                    "rep": rep_i,
                                    "step_index": idx,
                                    "step_name": step_name,
                                }
                                # Add per-joint metrics
                                for joint in JOINT_ORDER:
                                    row[f"mae_{joint}"] = step_metrics.get(joint, np.nan)
                                per_patient_stepwise_rows.append(row)

        # save per-patient summary
        if per_patient_rows:
            df_p = pd.DataFrame(per_patient_rows)
            out_p = os.path.join(OUT_PATIENT_SUM, f"{patient}.csv")
            df_p.to_csv(out_p, index=False)
            print(f"[OK] wrote per-exercise summary for {patient}: {out_p}")

        # save per-patient QUALITY summary (per exercise)
        if per_exercise_quality_rows:
            df_q = pd.DataFrame(per_exercise_quality_rows)
            out_q = os.path.join(OUT_QUALITY_PATIENT, f"{patient}.csv")
            df_q.to_csv(out_q, index=False)
            print(f"[OK] wrote quality summary for {patient}: {out_q}")

        # save per-patient STEPWISE summary (per exercise and rep)
        if per_patient_stepwise_rows:
            df_s = pd.DataFrame(per_patient_stepwise_rows)
            out_s = os.path.join(OUT_STEPWISE_PATIENT, f"{patient}.csv")
            df_s.to_csv(out_s, index=False)
            print(f"[OK] wrote stepwise summary for {patient}: {out_s}")

    # Save master CSV
    if master_rows:
        df_master = pd.DataFrame(master_rows)
        out_master = os.path.join(OUT_ROOT, "all_patients_master.csv")
        df_master.to_csv(out_master, index=False)
        print(f"[OK] wrote master CSV: {out_master}")

        # Overall patient-level averages (across all exercises & joints)
        agg = (
            df_master
            .groupby("patient", as_index=False)
            .agg({
                "mae_x":"mean","rmse_x":"mean","corr_x":"mean",
                "mae_y":"mean","rmse_y":"mean","corr_y":"mean"
            })
            .rename(columns={
                "mae_x":"avg_mae_x","rmse_x":"avg_rmse_x","corr_x":"avg_corr_x",
                "mae_y":"avg_mae_y","rmse_y":"avg_rmse_y","corr_y":"avg_corr_y"
            })
        )
        out_overall = os.path.join(OUT_ROOT, "overall_patient_summary.csv")
        agg.to_csv(out_overall, index=False)
        print(f"[OK] wrote overall patient summary: {out_overall}")

        # Optional: also pivot by joint for quick glance
        pivot = (
            df_master
            .groupby(["patient","joint"], as_index=False)
            .agg({"mae_x":"mean","mae_y":"mean","rmse_x":"mean","rmse_y":"mean","corr_x":"mean","corr_y":"mean"})
        )
        out_pivot = os.path.join(OUT_ROOT, "overall_by_joint.csv")
        pivot.to_csv(out_pivot, index=False)
        print(f"[OK] wrote joint-level summary: {out_pivot}")

        # Save QUALITY metrics master & overall summaries
        try:
            q_files = [os.path.join(OUT_QUALITY_PATIENT, f) for f in os.listdir(OUT_QUALITY_PATIENT) if f.endswith('.csv')]
            quality_frames = [pd.read_csv(fp) for fp in q_files] if q_files else []
            if quality_frames:
                df_quality_master = pd.concat(quality_frames, ignore_index=True)
                out_q_master = os.path.join(OUT_QUALITY_ROOT, 'all_patients_quality_master.csv')
                df_quality_master.to_csv(out_q_master, index=False)
                print(f"[OK] wrote quality master: {out_q_master}")

                # Overall patient-level averages for quality metrics
                q_agg = (
                    df_quality_master.groupby('patient', as_index=False)
                    .agg({
                        'elbow_angle_mae_deg':'mean',
                        'shoulder_angle_mae_deg':'mean',
                        'pos_err_pct_mean':'mean',
                        'dtw_distance':'mean',
                        'peak_lag_s':'mean',
                        'sym_dev_abs_mean':'mean',
                        'jerk_rms':'mean',
                    })
                )
                out_q_overall = os.path.join(OUT_QUALITY_ROOT, 'overall_patient_quality_summary.csv')
                q_agg.to_csv(out_q_overall, index=False)
                print(f"[OK] wrote overall patient quality summary: {out_q_overall}")
            else:
                print("[INFO] No per-exercise quality rows found; quality master not written.")
        except Exception as e:
            print(f"[WARN] Could not write quality master/summary: {e}")
    else:
        print("[INFO] No rows collected — check your directory structure & file names.")


if __name__ == "__main__":
    main()