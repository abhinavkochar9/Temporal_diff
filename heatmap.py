import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import List, Tuple, Optional
import mediapipe as mp

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# ---------- CONFIG ----------
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_IMG_DIR = SCRIPT_DIR / "Step_Images" / "WHAM"
CSV_PATH = SCRIPT_DIR / "outputs" / "batch_reports" / "stepwise_summaries" / "TOLFB01.csv"
OUT_DIR = SCRIPT_DIR / "arm_heatmaps_aligned"

# --- Visualization Tweaks ---
ALPHA = 0.60
VMIN, VMAX = 0.0, 0.20
GAUSSIAN_BLUR_KERNEL = (121, 121)
JOINT_RADIUS = 25
ARM_THICKNESS = 32

# Thresholded colouring: errors below this MAE are treated as "no heat"
THRESHOLD = 0.15  # MAE units
GAMMA = 1.0       # >1 darkens mid-range, <1 brightens; 1.0 = linear

# --- CLI Defaults ---
DEFAULT_CSV = None  # path to a single CSV
DEFAULT_CSV_DIR = SCRIPT_DIR / "outputs" / "batch_reports" / "stepwise_summaries"  # directory of many CSVs

# ---------- HELPERS ----------

def get_pose_landmarks(image):
    h, w, _ = image.shape
    # Process the image to find pose
    # ✨ THIS IS THE CORRECTED LINE
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    l_sho = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
    r_sho = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    l_elb = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))
    r_elb = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
    l_wri = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h))
    r_wri = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))
    
    return [l_sho, r_sho, l_elb, r_elb, l_wri, r_wri]

def body_mask_silhouette(img_bgr, thresh=245):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < thresh).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    return mask

def nz_norm(x, vmin, vmax):
    return float(np.clip((x - vmin) / (max(1e-9, vmax - vmin)), 0.0, 1.0))

def thresh_norm(x: float, t: float, vmin: float, vmax: float, gamma: float = 1.0) -> float:
    """
    Piecewise mapping for MAE -> [0,1]:
      - x <= t  -> 0 (no colour below threshold)
      - x >  t  -> linear map from t..vmax to 0..1, then optional gamma
    """
    if x <= t:
        return 0.0
    # shift baseline up to threshold so that t maps to 0
    base_min = max(vmin, t)
    val = nz_norm(x, base_min, vmax)
    # gamma adjustment (>=0). If gamma>1 darkens mid, if <1 brightens mid.
    if gamma != 1.0 and val > 0.0:
        val = float(np.clip(val, 0.0, 1.0)) ** float(max(1e-9, gamma))
    return float(np.clip(val, 0.0, 1.0))

def draw_heatmap_on_image(img, points, maes, thickness, radius, blur_kernel):
    h, w = img.shape[:2]
    heat_canvas = np.zeros((h, w), dtype=np.float32)
    L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = points
    Ls, Rs, Le, Re, Lw, Rw = maes
    
    for p1, p2, p3, m1, m2, m3 in [(L_SHO, L_ELB, L_WRI, Ls, Le, Lw), (R_SHO, R_ELB, R_WRI, Rs, Re, Rw)]:
        pts = np.array([p1, p2, p3], dtype=np.float32)
        ma_vals = np.array([m1, m2, m3], dtype=np.float32)
        for i in range(2):
            a, b = pts[i], pts[i+1]
            ma, mb = ma_vals[i], ma_vals[i+1]
            n = max(2, int(np.linalg.norm(b - a) / 6))
            for t in np.linspace(0, 1, n):
                p = a * (1 - t) + b * t; m = ma * (1 - t) + mb
                cv2.circle(heat_canvas, (int(p[0]), int(p[1])), thickness // 2, m, -1)
                
    for point, mae in zip(points, maes):
        cv2.circle(heat_canvas, point, radius, mae, -1)
        
    heat_canvas = cv2.GaussianBlur(heat_canvas, blur_kernel, 0)
    
    heat_canvas_u8 = (255 * (heat_canvas / heat_canvas.max())).astype(np.uint8) if heat_canvas.max() > 0 else heat_canvas.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat_canvas_u8, cv2.COLORMAP_JET)
    return heatmap_color

# ---------- MAIN UTILITIES ----------

def find_csv_files(single_csv: Optional[Path], csv_dir: Optional[Path]) -> list[Path]:
    """Return a list of CSV files to process."""
    files: list[Path] = []
    if single_csv:
        if single_csv.exists():
            files.append(single_csv)
        else:
            logging.warning("CSV path does not exist: %s", single_csv)
    if csv_dir:
        if csv_dir.exists():
            files.extend(sorted(csv_dir.glob("*.csv")))
        else:
            logging.warning("CSV directory does not exist: %s", csv_dir)
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq

def compute_global_vmax(csv_files: list[Path]) -> float:
    """Compute a global VMAX (95th percentile) across all MAE values in all CSVs."""
    all_vals = []
    mae_cols_cache = None
    for p in csv_files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logging.warning("Failed to read %s: %s", p, e)
            continue
        mae_cols = [c for c in df.columns if c.startswith("mae_")]
        if not mae_cols:
            logging.warning("No MAE columns found in %s; skipping for VMAX.", p)
            continue
        mae_cols_cache = mae_cols
        all_vals.append(df[mae_cols].values.flatten())
    if not all_vals:
        logging.warning("No MAE values found across provided CSVs. Falling back to default VMAX=%.3f", VMAX)
        return VMAX
    stacked = np.concatenate(all_vals)
    vmax = float(np.percentile(stacked, 95))
    logging.info("Global VMAX (95th percentile across %d files): %.6f", len(csv_files), vmax)
    return vmax

def process_dataframe(df: pd.DataFrame, base_img_dir: Path, out_dir: Path,
                      alpha: float, vmin: float, vmax: float, threshold_mae: float, gamma_val: float) -> None:
    """Process one dataframe: render and save all heatmaps."""
    mae_order = [
        "mae_Left_Shoulder", "mae_Right_Shoulder",
        "mae_Left_Elbow",    "mae_Right_Elbow",
        "mae_Left_Wrist",    "mae_Right_Wrist"
    ]
    # basic column sanity
    missing_cols = [c for c in ["patient","exercise","rep","step_index"] + mae_order if c not in df.columns]
    if missing_cols:
        logging.warning("Missing expected columns %s; skipping this dataframe.", missing_cols)
        return

    for index, row in df.iterrows():
        exercise_name = row['exercise']
        step_idx = row['step_index']
        img_name = f"{exercise_name}_S{step_idx}.png"
        img_path = base_img_dir / exercise_name / img_name

        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning("Skipping row %d: image not found at %s", index, img_path)
            continue

        points = get_pose_landmarks(img)
        if points is None:
            logging.warning("Skipping row %d: no person detected in %s", index, img_name)
            continue

        mask = body_mask_silhouette(img)
        maes = [
            thresh_norm(float(row[col]), threshold_mae, vmin, vmax, gamma_val)
            for col in mae_order
        ]

        heatmap_color = draw_heatmap_on_image(
            img, points, maes, ARM_THICKNESS, JOINT_RADIUS, GAUSSIAN_BLUR_KERNEL
        )
        heatmap_color[mask == 0] = 0
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
        final_img = np.where(np.repeat(mask[..., None], 3, axis=2) > 0, overlay, img)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = out_dir / f"arm_{row['patient']}_{row['exercise']}_rep{row['rep']}_step{row['step_index']}.png"
        cv2.imwrite(str(out_name), final_img)

def configure_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate arm heatmaps for all patients.")
    parser.add_argument("--csv", type=str, default=str(CSV_PATH), help="Single CSV file to process (overrides --csv-dir if both given).")
    parser.add_argument("--csv-dir", type=str, default=str(DEFAULT_CSV_DIR), help="Directory containing multiple patient CSVs.")
    parser.add_argument("--base-img-dir", type=str, default=str(BASE_IMG_DIR), help="Directory of step images.")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="Directory to save outputs.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Overlay alpha.")
    parser.add_argument("--vmin", type=float, default=VMIN, help="Lower bound for MAE normalisation.")
    parser.add_argument("--vmax", type=float, default=-1.0, help="Upper bound for MAE normalisation; if negative, compute global 95th percentile.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="MAE threshold: values <= threshold render as 0 (no heat)")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help="Gamma for post-threshold intensity curve (>1 darkens mid, <1 brightens)")
    return parser.parse_args()

if __name__ == "__main__":
    configure_logging()
    args = parse_args()

    single_csv = Path(args.csv) if args.csv else None
    csv_dir = Path(args.csv_dir) if args.csv_dir else None
    base_img_dir = Path(args.base_img_dir)
    out_dir = Path(args.out_dir)

    csv_files = find_csv_files(single_csv, csv_dir)
    if not csv_files:
        logging.error("No CSV files found. Nothing to do.")
        raise SystemExit(1)

    # Determine VMAX
    if args.vmax is not None and args.vmax >= 0:
        vmax_global = float(args.vmax)
        logging.info("Using user-provided VMAX: %.6f", vmax_global)
    else:
        vmax_global = compute_global_vmax(csv_files)
        logging.info("Using computed global VMAX: %.6f", vmax_global)

    threshold_mae = float(args.threshold)
    gamma_val = float(args.gamma)
    logging.info("Using threshold=%.4f (MAE), gamma=%.3f", threshold_mae, gamma_val)

    # Process each CSV
    for csv_path in csv_files:
        logging.info("Processing %s ...", csv_path.name)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.warning("Failed to read %s: %s", csv_path, e)
            continue
        process_dataframe(df, base_img_dir, out_dir, args.alpha, args.vmin, vmax_global, threshold_mae, gamma_val)

    logging.info("All done. Outputs saved to %s", out_dir)