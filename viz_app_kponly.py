import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# ====== CONFIG ===========
# =========================
PATIENT_DIR = "split_dataset_by_joint_cleaned"
EXPERT_DIR  = "split_dataset_by_joint_cleaned_expert"
JOINT_ORDER = ["Right_Shoulder", "Left_Shoulder", "Right_Elbow", "Left_Elbow", "Right_Wrist", "Left_Wrist"]

st.set_page_config(layout="wide")
st.title("🧠 Patient vs Expert — Keypoint Comparison")

# =========================
# ====== HELPERS ==========
# =========================
def list_dirs(root):
    if not os.path.exists(root) or not os.path.isdir(root):
        return []
    return sorted([p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))])

def load_joint_csv(base_dir, patient_id, exercise, joint):
    path = os.path.join(base_dir, patient_id, exercise, f"{joint}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # Expecting at least 2 columns: X, Y (first two)
        if df.empty or len(df.columns) < 2:
            return None
        # Keep only first two columns for keypoints (X, Y)
        df2 = df.iloc[:, :2].copy()
        df2.columns = ["X", "Y"]
        return df2.reset_index(drop=True)
    except Exception:
        return None

def global_min_max(dfs):
    gmin, gmax = np.inf, -np.inf
    for df in dfs:
        if df is None: 
            continue
        loc_min = df.min().min()
        loc_max = df.max().max()
        gmin = min(gmin, loc_min)
        gmax = max(gmax, loc_max)
    if not np.isfinite(gmin) or not np.isfinite(gmax):
        gmin, gmax = 0.0, 1.0
    if gmin == gmax:
        gmin, gmax = gmin - 0.5, gmax + 0.5
    pad = (gmax - gmin) * 0.05
    return gmin - pad, gmax + pad

def quick_metrics(a: pd.Series, b: pd.Series):
    # Align length
    n = min(len(a), len(b))
    if n == 0:
        return np.nan, np.nan, np.nan
    a = a.iloc[:n].astype(float)
    b = b.iloc[:n].astype(float)
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    return mae, rmse, corr

# =========================
# ===== SIDEBAR UI ========
# =========================
st.sidebar.header("🔎 Select Data")

# Directory checks
if not os.path.isdir(PATIENT_DIR):
    st.error(f"Patient data directory not found: **{PATIENT_DIR}**")
    st.stop()
if not os.path.isdir(EXPERT_DIR):
    st.warning(f"Expert data directory not found: **{EXPERT_DIR}** — plots will show patient only if expert missing.")

patients = list_dirs(PATIENT_DIR)
if not patients:
    st.error(f"No patient folders in **{PATIENT_DIR}**.")
    st.stop()

selected_patient = st.sidebar.selectbox("👤 Patient", patients)

# Exercises: intersect with expert (to avoid confusion) but allow all patient ones
patient_exercises = list_dirs(os.path.join(PATIENT_DIR, selected_patient))
selected_exercise = st.sidebar.selectbox("💪 Exercise", patient_exercises)

show_legend = st.sidebar.checkbox("Show legend", value=False)
sync_length = st.sidebar.checkbox("Trim to common length (X/Y)", value=True)
show_metrics = st.sidebar.checkbox("Show per-joint metrics table", value=True)

st.caption(
    "Expert lines are **bold solid**; Patient lines are **dotted**. "
    "Each joint panel overlays X and Y."
)

# =========================
# ====== LOAD DATA ========
# =========================
patient_dfs = {}
expert_dfs = {}

for joint in JOINT_ORDER:
    patient_dfs[joint] = load_joint_csv(PATIENT_DIR, selected_patient, selected_exercise, joint)
    expert_dfs[joint]  = load_joint_csv(EXPERT_DIR,  selected_patient, selected_exercise, joint)

# Global y range across all (patient + expert)
all_for_range = []
for j in JOINT_ORDER:
    if patient_dfs[j] is not None: all_for_range.append(patient_dfs[j])
    if expert_dfs[j] is not None:  all_for_range.append(expert_dfs[j])
ymin, ymax = global_min_max(all_for_range)

# =========================
# ======== TABS ===========
# =========================
tab_compare, tab_delta = st.tabs(["📊 Compare (Patient vs Expert)", "➖ Delta (Patient − Expert)"])

# =========================
# === TAB 1: COMPARE ======
# =========================
with tab_compare:
    fig = make_subplots(
        rows=len(JOINT_ORDER),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{j.replace('_', ' ')} (X & Y)" for j in JOINT_ORDER],
        vertical_spacing=0.06,
    )

    for i, joint in enumerate(JOINT_ORDER, start=1):
        p_df = patient_dfs[joint]
        e_df = expert_dfs[joint]

        # Optionally trim to common length to keep lines aligned visually
        if sync_length and p_df is not None and e_df is not None:
            n = min(len(p_df), len(e_df))
            p_df = p_df.iloc[:n].reset_index(drop=True)
            e_df = e_df.iloc[:n].reset_index(drop=True)

        # Expert first (bold solid)
        if e_df is not None:
            fig.add_trace(
                go.Scatter(
                    y=e_df["X"], mode="lines",
                    name=f"{joint} Expert X",
                    line=dict(width=3, dash="solid"),
                    hovertemplate="Expert X: %{y:.4f}<extra></extra>"
                ),
                row=i, col=1
            )
            fig.add_trace(
                go.Scatter(
                    y=e_df["Y"], mode="lines",
                    name=f"{joint} Expert Y",
                    line=dict(width=3, dash="solid"),
                    hovertemplate="Expert Y: %{y:.4f}<extra></extra>"
                ),
                row=i, col=1
            )

        # Patient next (dotted)
        if p_df is not None:
            fig.add_trace(
                go.Scatter(
                    y=p_df["X"], mode="lines",
                    name=f"{joint} Patient X",
                    line=dict(width=2, dash="dot"),
                    hovertemplate="Patient X: %{y:.4f}<extra></extra>"
                ),
                row=i, col=1
            )
            fig.add_trace(
                go.Scatter(
                    y=p_df["Y"], mode="lines",
                    name=f"{joint} Patient Y",
                    line=dict(width=2, dash="dot"),
                    hovertemplate="Patient Y: %{y:.4f}<extra></extra>"
                ),
                row=i, col=1
            )

        fig.update_yaxes(range=[ymin, ymax], title_text="Norm.", row=i, col=1)

    fig.update_layout(
        height=2000,
        title_text=f"Patient vs Expert — {selected_patient} · {selected_exercise}",
        showlegend=show_legend,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="Frame Index", row=len(JOINT_ORDER), col=1)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# === TAB 2: DELTAS =======
# =========================
with tab_delta:
    st.markdown("#### Δ (Patient − Expert) per Joint and Axis")
    fig_d = make_subplots(
        rows=len(JOINT_ORDER),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{j.replace('_', ' ')} ΔX & ΔY" for j in JOINT_ORDER],
        vertical_spacing=0.06,
    )

    metrics_rows = []
    for i, joint in enumerate(JOINT_ORDER, start=1):
        p_df = patient_dfs[joint]
        e_df = expert_dfs[joint]

        if p_df is None or e_df is None:
            # Show empty panel / skip metrics
            fig_d.add_trace(go.Scatter(y=[]), row=i, col=1)
            metrics_rows.append({
                "Joint": joint, "MAE X": np.nan, "RMSE X": np.nan, "Corr X": np.nan,
                "MAE Y": np.nan, "RMSE Y": np.nan, "Corr Y": np.nan
            })
            continue

        n = min(len(p_df), len(e_df)) if sync_length else max(len(p_df), len(e_df))
        p_df = p_df.iloc[:n].reset_index(drop=True)
        e_df = e_df.iloc[:n].reset_index(drop=True)

        dx = p_df["X"] - e_df["X"]
        dy = p_df["Y"] - e_df["Y"]

        # Plot ΔX and ΔY
        fig_d.add_trace(
            go.Scatter(
                y=dx, mode="lines", name=f"{joint} ΔX",
                line=dict(width=2, dash="solid"),
                hovertemplate="ΔX: %{y:.4f}<extra></extra>"
            ),
            row=i, col=1
        )
        fig_d.add_trace(
            go.Scatter(
                y=dy, mode="lines", name=f"{joint} ΔY",
                line=dict(width=2, dash="dash"),
                hovertemplate="ΔY: %{y:.4f}<extra></extra>"
            ),
            row=i, col=1
        )

        # Metrics
        mae_x, rmse_x, corr_x = quick_metrics(p_df["X"], e_df["X"])
        mae_y, rmse_y, corr_y = quick_metrics(p_df["Y"], e_df["Y"])
        metrics_rows.append({
            "Joint": joint,
            "MAE X": round(mae_x, 4), "RMSE X": round(rmse_x, 4), "Corr X": round(corr_x, 4) if not np.isnan(corr_x) else np.nan,
            "MAE Y": round(mae_y, 4), "RMSE Y": round(rmse_y, 4), "Corr Y": round(corr_y, 4) if not np.isnan(corr_y) else np.nan
        })

    # Auto range for deltas (center around 0)
    # Collect all deltas
    all_dx_dy = []
    for i, joint in enumerate(JOINT_ORDER, start=1):
        p_df, e_df = patient_dfs[joint], expert_dfs[joint]
        if p_df is None or e_df is None:
            continue
        n = min(len(p_df), len(e_df)) if sync_length else max(len(p_df), len(e_df))
        p_df = p_df.iloc[:n].reset_index(drop=True)
        e_df = e_df.iloc[:n].reset_index(drop=True)
        all_dx_dy.append((p_df["X"] - e_df["X"]).values)
        all_dx_dy.append((p_df["Y"] - e_df["Y"]).values)

    if all_dx_dy:
        stacked = np.concatenate(all_dx_dy)
        dmin, dmax = stacked.min(), stacked.max()
        pad = (dmax - dmin) * 0.05 if dmax > dmin else 0.1
        d_range = [dmin - pad, dmax + pad]
    else:
        d_range = [-1, 1]

    for i in range(1, len(JOINT_ORDER) + 1):
        fig_d.update_yaxes(range=d_range, title_text="Δ", row=i, col=1)
    fig_d.update_layout(
        height=2000,
        title_text=f"Deltas — {selected_patient} · {selected_exercise}",
        showlegend=show_legend,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig_d.update_xaxes(title_text="Frame Index", row=len(JOINT_ORDER), col=1)
    st.plotly_chart(fig_d, use_container_width=True)

    if show_metrics:
        st.markdown("##### Per-Joint Metrics (lower MAE/RMSE and higher Corr are better)")
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)