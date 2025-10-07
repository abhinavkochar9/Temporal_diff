import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont
import json
import re
import glob
_num_re = re.compile(r'(\d+)')
def _natural_key(s: str):
    s = os.path.basename(str(s))
    return [int(t) if t.isdigit() else t.lower() for t in _num_re.split(s)]

# --- Normalized name helpers for robust directory matching ---
def _norm_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

# --- Robust list matcher for query param to sidebar selection ---
def _best_match_from_list(options: list[str], target: str | None) -> str | None:
    """Return best match from options for target using alnum-only, case-insensitive matching.
    Prefers exact normalized match, then contains, then reverse-contains.
    """
    if not target or not options:
        return None
    want = _norm_name(target)
    # exact normalized match
    for o in options:
        if _norm_name(o) == want:
            return o
    # contains: option contains target
    for o in options:
        if want and _norm_name(o).find(want) != -1:
            return o
    # reverse contains: target contains option
    for o in options:
        if want and want.find(_norm_name(o)) != -1:
            return o
    return None

def _best_match_subdir(parent: str, exercise_name: str) -> str | None:
    if not parent or not os.path.isdir(parent):
        return None
    want = _norm_name(exercise_name)
    subdirs = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
    # exact normalized match
    for d in subdirs:
        if _norm_name(d) == want:
            return os.path.join(parent, d)
    # contains match
    for d in subdirs:
        if want and _norm_name(d).find(want) != -1:
            return os.path.join(parent, d)
    return None

# --- Regex helpers for step index and combo splitting ---
_step_idx_re = re.compile(r'(?:^|_)s(\d+)$', re.IGNORECASE)   # matches ..._S1 / ..._s10 at end of step folder
_combo_split_re = re.compile(r'[+_\-]')                       # split joint names by +, _ or -

# --- Generated overlays: filename matcher and loader ---
_gen_name_re = re.compile(r'^arm_(?P<pat>.+?)_(?P<ex>.+?)_rep(?P<rep>\d+)_step(?P<step>\d+)\.(?:png|jpg|jpeg|webp|bmp)$', re.IGNORECASE)

def _find_generated_images(root: str, patient: str, exercise: str):
    """Return (rep1_paths, rep2_paths) for generated overlays under `root`.
    Expects files named: arm_{patient}_{exercise}_rep{rep}_step{step}.png
    """
    if not root:
        return [], []
    base = root
    if not os.path.isabs(base):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), base)
    if not os.path.isdir(base):
        return [], []
    pattern = os.path.join(base, f"arm_{patient}_{exercise}_rep*_step*.*")
    paths = sorted(glob.glob(pattern), key=_natural_key)
    rep1, rep2 = [], []
    for p in paths:
        m = _gen_name_re.match(os.path.basename(p))
        if not m:
            continue
        rep = int(m.group("rep")); step = int(m.group("step"))
        if rep == 1:
            rep1.append((step, p))
        elif rep == 2:
            rep2.append((step, p))
    rep1 = [p for step, p in sorted(rep1)]
    rep2 = [p for step, p in sorted(rep2)]
    return rep1, rep2

# --- Pillow text size helper (compat across many versions, draw-free) ---
def _text_size(text: str, font: ImageFont.ImageFont):
    """
    Return (width, height) for text using font-only methods (no draw.* calls),
    compatible across old/new Pillow versions.
    """
    # Pillow >= 8: font.getbbox
    if hasattr(font, "getbbox"):
        try:
            l, t, r, b = font.getbbox(text)
            return (r - l, b - t)
        except Exception:
            pass
    # Fallback: font.getsize (older Pillow)
    if hasattr(font, "getsize"):
        try:
            return font.getsize(text)
        except Exception:
            pass
    # Last resort heuristic
    return (max(1, int(len(text) * 6)), 12)

# --- Fonts & drawing utilities -------------------------------------------------
DEFAULT_FONT = "arial.ttf"

def _load_font(size: int, fallback: bool = True) -> ImageFont.ImageFont:
    """Best-effort TTF load with graceful fallback to Pillow's default font."""
    try:
        return ImageFont.truetype(DEFAULT_FONT, size)
    except Exception:
        return ImageFont.load_default() if fallback else None
# --- Feedback loader & text wrapping -----------------------------------------
import textwrap

def _find_col(cols, *candidates):
    """Return the first matching column name (case-insensitive, strip spaces/underscores)."""
    norm = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in cols}
    for cand in candidates:
        key = "".join(ch for ch in cand.lower() if ch.isalnum())
        if key in norm:
            return norm[key]
    return None

def _load_feedback(csv_path: str, patient: str, exercise: str):
    """Load summary/grade/similarity for (patient, exercise) from a CSV.
    Returns dict with keys: summary, grade, similarity, patient_score, or None if not found."""
    try:
        df_fb = pd.read_csv(csv_path)
    except Exception:
        return None
    if df_fb is None or df_fb.empty:
        return None
    cols = list(df_fb.columns)
    # Prioritize exact labels first
    col_patient   = _find_col(cols, "patient_id", "patient", "subject", "user")
    col_exercise  = _find_col(cols, "exercise")
    col_summary   = _find_col(cols, "exercise_summary", "exercise summary", "summary", "feedback_summary", "llm_summary")
    col_grade     = _find_col(cols, "letter_grade", "letter grade", "grade", "final_grade")
    col_similarity= _find_col(cols, "overall_similarity", "overall similarity", "similarity", "score", "overall_score")
    if not col_patient or not col_exercise:
        return None

    # strict match first
    mask = (df_fb[col_patient].astype(str).str.strip().str.lower() == str(patient).strip().lower()) & \
           (df_fb[col_exercise].astype(str).str.strip().str.lower() == str(exercise).strip().lower())
    rows = df_fb[mask]
    if rows.empty:
        # relaxed match on exercise (alnum-only)
        ex_norm = "".join(ch for ch in str(exercise).lower() if ch.isalnum())
        rows = df_fb[df_fb[col_patient].astype(str).str.strip().str.lower() == str(patient).strip().lower()]
        if not rows.empty and col_exercise in rows.columns:
            exer_series = rows[col_exercise].astype(str).str.lower().apply(lambda s: "".join(ch for ch in s if ch.isalnum()))
            rows = rows[exer_series == ex_norm]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return {
        "summary":       (str(r[col_summary]) if col_summary in rows.columns else None),
        "grade":         (str(r[col_grade]) if col_grade in rows.columns else None),
        "similarity":    (float(r[col_similarity]) if col_similarity in rows.columns else None),
        "patient_score": (float(r[col_similarity]) if col_similarity in rows.columns else None),
    }

def _wrap_text_lines(text: str, font: ImageFont.ImageFont, max_width_px: int) -> list[str]:
    """Greedy wrap text to fit within max_width_px using font metrics."""
    if not text:
        return []
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        tw, _ = _text_size(trial, font)
        if tw <= max_width_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
# --- Heatmap helpers -----------------------------------------------------------
def _normalize_for_colors(vals: list[float]) -> tuple[float, float]:
    """Return robust (vmin, vmax) using 5th–95th percentiles when possible."""
    if not vals:
        return (0.0, 1.0)
    # ensure floats and finite
    clean = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not clean:
        return (0.0, 1.0)
    if len(clean) >= 3:
        vmin = float(np.percentile(clean, 5))
        vmax = float(np.percentile(clean, 95))
    else:
        vmin, vmax = float(min(clean)), float(max(clean))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + (abs(vmin) if abs(vmin) > 1e-6 else 1.0)
    return (vmin, vmax)

def _color_gyr(v: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    """Green→Yellow→Red color ramp for an error value v in [vmin, vmax]."""
    t = (v - vmin) / (vmax - vmin)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    if t < 0.5:
        a = t / 0.5
        r = int(0 + a * (255 - 0))
        g = int(200 + a * (220 - 200))
        b = 0
    else:
        a = (t - 0.5) / 0.5
        r = int(255 + a * (220 - 255))
        g = int(220 + a * (0 - 220))
        b = 0
    return (max(0, min(255, r)), max(0, min(255, g)), b)

def _draw_smooth_heatbar(draw: ImageDraw.ImageDraw, left_pad: int, top_y: int,
                         width: int, height: int, values: list[float],
                         vmin_override: float | None = None,
                         vmax_override: float | None = None) -> int:
    """
    Draw a smooth (per-pixel) heatbar across `width` using `values` (list of step-wise errors).
    Returns the new y position after drawing the bar (top_y + height + 8 spacing).
    """
    if not values or height <= 0 or width <= 0:
        return top_y
    vals = [float(v) if v is not None and np.isfinite(v) else 0.0 for v in values]
    if (vmin_override is not None and vmax_override is not None
        and np.isfinite(vmin_override) and np.isfinite(vmax_override)
        and float(vmax_override) > float(vmin_override)):
        vmin, vmax = float(vmin_override), float(vmax_override)
    else:
        vmin, vmax = _normalize_for_colors(vals)

    nseg = len(vals)
    y_top = top_y + 4  # small top gap
    for px in range(width):
        u = (px / max(1, width - 1)) * (nseg - 1 if nseg > 1 else 1)
        i0 = int(np.floor(u))
        i1 = min(nseg - 1, i0 + 1)
        t = u - i0
        v = (1.0 - t) * vals[i0] + t * vals[i1] if nseg > 1 else vals[0]
        color = _color_gyr(v, vmin, vmax)
        x = left_pad + px
        draw.line([(x, y_top), (x, y_top + height)], fill=color, width=1)

    # Subtle separators at step boundaries for readability
    if nseg > 1:
        for k in range(1, nseg):
            xk = left_pad + int(round(k * (width / nseg)))
            draw.line([(xk, y_top), (xk, y_top + height)], fill=(225, 225, 225), width=1)

    # Light outline
    draw.rectangle([left_pad, y_top, left_pad + width - 1, y_top + height - 1], outline=(180, 180, 180))
    return y_top + height + 4  # bottom gap

# =========================
# ====== CONFIG ===========
# =========================
PATIENT_DIR = "Patient_Temporal"
EXPERT_DIR  = "Expert_Temporal"
JOINT_ORDER = ["Right_Shoulder", "Left_Shoulder", "Right_Elbow", "Left_Elbow", "Right_Wrist", "Left_Wrist"]

# EMG dataset directory (wide CSV format: EMG_Dataset_TOLF-B/{patient}/{exercise}.csv)
EMG_DIR = "EMG_Dataset_TOLF-B"  # structure: EMG_Dataset_TOLF-B/{patient}/{exercise}.csv (wide format)

st.set_page_config(layout="wide")

st.title("🧠 Patient vs Expert — Keypoint Comparison (aggressive patient despike)")

# ------------------------
# Query params helper
# ------------------------
from urllib.parse import quote
def _get_query_params():
    """Return URL query params as a simple {str: str} dict via Streamlit's new API.
    Handles both plain dicts and QueryParams objects.
    """
    try:
        qp = st.query_params  # Mapping-like
        # Prefer official conversion if available
        if hasattr(qp, "to_dict") and callable(getattr(qp, "to_dict")):
            return {k: (v[0] if isinstance(v, list) else v) for k, v in qp.to_dict().items()}
        # Else handle like a mapping
        if hasattr(qp, "items"):
            return {k: (v[0] if isinstance(v, list) else v) for k, v in qp.items()}
        # Last resort
        return dict(qp)
    except Exception:
        return {}

def _qp_get_one(qp, key):
    if not isinstance(qp, dict):
        return None
    v = qp.get(key)
    if v is None:
        return None
    if isinstance(v, list):
        return v[0] if v else None
    return v

# Default feedback CSV path for homepage
_HOMEPAGE_FB_DEFAULT = "outputs/batch_reports/llm_reports5/summary/final_llm_feedback.csv"

# Lightweight color util using existing G→Y→R ramp
def _val_to_hex(v, vmin, vmax):
    r, g, b = _color_gyr(v, vmin, vmax)
    return f"#{r:02x}{g:02x}{b:02x}"

# Reversed G→Y→R mapping for homepage matrix (high values = green, low = red)
def _val_to_hex_rev(v, vmin, vmax):
    """Reverse the G→Y→R mapping so high values appear red→yellow→green becomes green→yellow→red flipped."""
    inv = (vmin + vmax) - float(v)
    r, g, b = _color_gyr(inv, vmin, vmax)
    return f"#{r:02x}{g:02x}{b:02x}"

# ------------------------
# HOMEPAGE (Patient × Exercise matrix)
# If query params ?patient=...&exercise=... are missing, show homepage
# Clicking a cell navigates back to this app with those query params.
# ------------------------
qp = _get_query_params()
qp_patient = _qp_get_one(qp, "patient")
qp_exercise = _qp_get_one(qp, "exercise")

def _render_homepage():
    st.header("🏠 Home — Patient × Exercise Matrix")
    # Prefer a path stored in session (set by the sidebar input), else fall back to default
    fb_path = st.session_state.get('fb_csv_path', _HOMEPAGE_FB_DEFAULT)
    # Try to load feedback CSV; if missing, show guidance
    if not os.path.isfile(fb_path):
        st.info(
            "Feedback CSV not found at the default path. Ensure your feedback file exists at: "
            f"**{fb_path}**. The homepage derives the *Patient Score* from this CSV."
        )
        return
    try:
        df = pd.read_csv(fb_path)
    except Exception as e:
        st.error(f"Failed to read feedback CSV for homepage: {e}")
        return
    if df is None or df.empty:
        st.info("Feedback CSV is empty.")
        return
    cols = list(df.columns)
    col_patient = _find_col(cols, "patient_id", "patient", "subject", "user")
    col_ex      = _find_col(cols, "exercise")
    col_score   = _find_col(cols, "overall_similarity", "overall similarity", "similarity", "score", "overall_score")
    if not col_patient or not col_ex or not col_score:
        st.warning("CSV must include patient, exercise, and overall similarity/score columns.")
        return
    # Build matrix by reusing the same logic as the detailed page: call _load_feedback()
    # This guarantees the score in the matrix == the "Patient Score" shown on the report.
    df_pairs = df[[col_patient, col_ex]].drop_duplicates(keep="first")
    rows = []
    for _, r in df_pairs.iterrows():
        pat = str(r[col_patient])
        ex  = str(r[col_ex])
        fb  = _load_feedback(fb_path, pat, ex)
        if fb and (fb.get("patient_score") is not None):
            rows.append({col_patient: pat, col_ex: ex, "__score__": float(fb["patient_score"])})
    if not rows:
        st.info("No (patient, exercise) scores found in CSV.")
        return
    df_scores = pd.DataFrame(rows)
    pivot = df_scores.pivot(index=col_patient, columns=col_ex, values="__score__").sort_index()
    # Determine color scale bounds (robust percentiles)
    vals = pivot.values.flatten()
    vals = np.array([float(x) for x in vals if pd.notna(x)], dtype=float)
    if vals.size == 0:
        st.info("No numeric scores in CSV to render.")
        return
    vmin = float(np.percentile(vals, 5)) if vals.size >= 5 else float(np.min(vals))
    vmax = float(np.percentile(vals, 95)) if vals.size >= 5 else float(np.max(vals))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    
    # Render as a clickable HTML table so each cell links to this page with query params
    # Basic styles
    st.markdown(
        """
        <style>
        table.matrix { border-collapse: collapse; width: 100%; }
        table.matrix th, table.matrix td { border: 1px solid #ddd; padding: 6px 8px; text-align: center; }
        table.matrix th.sticky { position: sticky; top: 0; background: #fafafa; z-index: 2; }
        table.matrix th.rowhdr { position: sticky; left: 0; background: #fafafa; z-index: 1; text-align: right; }
        table.matrix td a { display:block; width:100%; text-decoration:none; color:#111; font-weight:600; }
        table.matrix td.empty { background:#f5f5f5; color:#aaa; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Build header row
    ex_names = [str(c) for c in pivot.columns]
    html = ["<table class='matrix'>"]
    html.append("<thead><tr><th class='sticky rowhdr'>Patient \\ Exercise</th>" + "".join([f"<th class='sticky'>{quote(en) if False else en}</th>" for en in ex_names]) + "</tr></thead>")
    html.append("<tbody>")
    for pat, row in pivot.iterrows():
        pat_str = str(pat)
        tds = [f"<th class='rowhdr'>{pat_str}</th>"]
        for ex in pivot.columns:
            val = row.get(ex, np.nan)
            if pd.isna(val):
                tds.append("<td class='empty'>—</td>")
            else:
                color = _val_to_hex_rev(float(val), vmin, vmax)
                # Link back to this same app with query params
                href = f"?patient={quote(pat_str)}&exercise={quote(str(ex))}"
                tds.append(f"<td style='background:{color};'><a href='{href}' title='Open {pat_str} · {ex}'>{val:.3f}</a></td>")
        html.append("<tr>" + "".join(tds) + "</tr>")
    html.append("</tbody></table>")
    st.markdown("\n".join(html), unsafe_allow_html=True)
    st.caption("Click a score to open that patient’s exercise page. Colors follow Green→Yellow→Red by score range.")

# If either patient or exercise is missing, show homepage and stop.
if not (qp_patient and qp_exercise):
    _render_homepage()
    st.stop()

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
        if df.empty or len(df.columns) < 2:
            return None
        df2 = df.iloc[:, :2].copy()
        df2.columns = ["X", "Y"]
        for c in ["X", "Y"]:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2 = df2.interpolate(limit_direction="both")
        return df2.reset_index(drop=True)
    except Exception:
        return None

# --- EMG helpers for wide CSV format ---
def load_emg_table(base_dir: str, patient_id: str, exercise: str):
    """Load one EMG CSV for (patient, exercise) in wide format.
    Expected columns: a time column (e.g., 'Time_Index') plus multiple channels like
    'DELTOID_Affected', 'DELTOID_NonAffected', 'BICEPS_Affected', etc.
    Returns (df, time_col, channel_cols) where channel_cols excludes the time column.
    """
    path = os.path.join(base_dir, patient_id, f"{exercise}.csv")
    if not os.path.exists(path):
        return None, None, []
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None, []
    if df is None or df.empty:
        return None, None, []
    # Identify time column
    time_candidates = [c for c in df.columns if str(c).strip().lower() in {"time", "time_index", "frame", "t"}]
    time_col = time_candidates[0] if time_candidates else None
    # Coerce numeric for all non-time columns and drop empty channels
    channel_cols = [c for c in df.columns if c != time_col]
    for c in channel_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if time_col is not None:
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    channel_cols = [c for c in channel_cols if df[c].notna().any()]
    return df, time_col, channel_cols

def list_emg_channels(base_dir: str, patient_id: str, exercise: str):
    """Return available EMG channel names from the wide CSV for (patient, exercise)."""
    _, _, channel_cols = load_emg_table(base_dir, patient_id, exercise)
    return channel_cols or []

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

# -------- Exercise step images (strip) --------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

def _list_image_files(folder: str):
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().endswith(IMG_EXTS)]
    )

def _build_strip(paths, repeat=2, target_h=160, pad=8, bg=(255, 255, 255)):
    """
    Make one horizontal strip: [step1..stepN] repeated `repeat` times.
    Keeps aspect ratio, equal target height, small spacing between images.
    """
    if not paths:
        return None
    seq = paths * repeat
    imgs = []
    for p in seq:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        new_w = int(w * (target_h / float(h)))
        imgs.append(im.resize((new_w, target_h)))

    total_w = sum(im.size[0] for im in imgs) + pad * (len(imgs) - 1)
    strip = Image.new("RGB", (total_w, target_h), bg)
    x = 0
    for im in imgs:
        strip.paste(im, (x, 0))
        x += im.size[0] + pad
    return strip

# ---- Helpers for timeline strip ----
def _first_available_frames_count(base_dir_patient: str, base_dir_expert: str, patient_id: str, exercise: str) -> int | None:
    """
    Try to infer total frame count for this exercise by reading the first
    available joint CSV (patient first, then expert). Returns None if none found.
    """
    for base in [base_dir_patient, base_dir_expert]:
        if not base:
            continue
        for joint in JOINT_ORDER:
            csv_path = os.path.join(base, patient_id, exercise, f"{joint}.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    # use length of first two columns (X,Y) as proxy for frames
                    if not df.empty:
                        return len(df)
                except Exception:
                    continue
    return None

def _build_strip_with_timeline(paths, repeat=2, target_h=160, pad=8, fps=25.0, total_frames: int | None = None,
                               bg=(255, 255, 255), tick_every_s: float = 1.0, timeline_h: int = 36):
    """
    Build a horizontal strip of step images (repeated) and draw a clean seconds timeline.
    Improvements vs previous:
      - left/right padding so first/last labels aren't squashed
      - capped number of labels (~8–10)
      - major & minor ticks
      - thicker baseline and larger font
    """
    if not paths:
        return None

    # 1) Build the base strip of images
    seq = paths * repeat
    imgs = []
    for p in seq:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        new_w = int(w * (target_h / float(h)))
        imgs.append(im.resize((new_w, target_h)))

    total_w = sum(im.size[0] for im in imgs) + pad * (len(imgs) - 1)
    base_strip = Image.new("RGB", (total_w, target_h), bg)
    x = 0
    for im in imgs:
        base_strip.paste(im, (x, 0))
        x += im.size[0] + pad

    # 2) Extend canvas for timeline with horizontal padding
    W, H = base_strip.size
    left_pad = 18
    right_pad = 18
    canvas = Image.new("RGB", (W + left_pad + right_pad, H + timeline_h), bg)
    canvas.paste(base_strip, (left_pad, 0))

    # 3) Prepare to draw
    draw = ImageDraw.Draw(canvas)
    # Font: default; try a bit larger if available
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    y0 = H + 10               # baseline y
    major_len = 8             # shorter major tick
    minor_len = 4             # shorter minor tick
    y_major = y0 + major_len  # major tick bottom
    y_minor = y0 + minor_len  # minor tick bottom

    # Baseline
    draw.line([(left_pad, y0), (left_pad + W, y0)], fill=(110, 110, 110), width=2)

    # 4) Determine seconds range
    seconds_total = None
    if total_frames is not None and fps and fps > 0:
        seconds_total = max(0.0, float(total_frames) / float(fps))

    if seconds_total is None:
        seconds_total = 10.0  # fallback

    # 5) Choose tick spacing to target ~8 labels
    desired_labels = 8
    approx_px_per_s = W / max(seconds_total, 1e-6)
    # candidate spacings in seconds
    candidates = [0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 60]
    tick_step = candidates[-1]
    for c in candidates:
        if (seconds_total / c) <= desired_labels and approx_px_per_s * c >= 70:
            tick_step = c
            break

    # minor ticks at 1/5 of major spacing (if reasonably apart)
    minor_step = tick_step / 5.0 if tick_step >= 1.0 else max(0.1, tick_step / 2.0)

    # 6) Draw ticks
    def sec_to_x(sec: float) -> int:
        # map seconds to pixel on baseline with padding
        if seconds_total <= 0:
            return left_pad
        frac = min(1.0, max(0.0, sec / seconds_total))
        return int(left_pad + frac * W)

    # Minor ticks (no labels)
    s = 0.0
    while s <= seconds_total + 1e-9:
        x_pos = sec_to_x(s)
        draw.line([(x_pos, y0), (x_pos, y_minor)], fill=(160, 160, 160), width=1)
        s += minor_step

    # Major ticks with labels
    s = 0.0
    while s <= seconds_total + 1e-9:
        x_pos = sec_to_x(s)
        draw.line([(x_pos, y0), (x_pos, y_major)], fill=(110, 110, 110), width=2)
        label = f"{s:.0f}s" if tick_step >= 1 else f"{s:.1f}s"
        tw, th = _text_size(label, font)
        label_x = max(left_pad, min(left_pad + W - tw, x_pos - tw // 2))
        label_y = y_major + 2  # draw below the tick to avoid overlap
        draw.text((label_x, label_y), label, fill=(70, 70, 70), font=font)
        s += tick_step

    # Also ensure "0s" label at left if spacing skipped it
    if tick_step > 0.5:
        label = "0s"
        tw, th = _text_size(label, font)
        draw.text((left_pad, y_major + 2), label, fill=(70, 70, 70), font=font)

    return canvas

# ---- Robust exercise step images finder ----
def _canonicalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())

# --- Explicit exercise dir resolver for Expert or WHAM sublabel ---
def _resolve_ex_dir_by_subfolder(base_dir: str, exercise_name: str, sublabel: str):
    """
    Return best-matching `<root>/Step_Images/<sublabel>/<ExerciseVariant>` directory.
    Tries {script dir, CWD, parent} and uses normalized-name matching to tolerate case/underscore/space differences.
    """
    roots = [base_dir, os.getcwd(), os.path.dirname(base_dir)]
    for root in roots:
        parent = os.path.join(root, "Step_Images", sublabel)
        match = _best_match_subdir(parent, exercise_name)
        if match:
            return match
    return None

# --- List step folders and original images (works for Expert & WHAM) ---
def _list_step_folders_with_originals(ex_dir: str):
    """
    Returns a list of dicts:
      { "folder": <abs path>, "name": <folder name>, "index": <int step idx if found>, "original": <abs path or None>, "base_prefix": <stem used before __ in files> }
    Sorting is natural by index if available; otherwise by folder name.
    A step folder is any directory directly under ex_dir that contains a file matching "<foldername>__original.*".
    """
    out = []
    if not ex_dir or not os.path.isdir(ex_dir):
        return out
    for entry in os.listdir(ex_dir):
        step_dir = os.path.join(ex_dir, entry)
        if not os.path.isdir(step_dir):
            continue
        # candidate original
        # try "<folder>/<folder>__original.(png|jpg|jpeg|webp|bmp|tif|tiff)"
        base_prefix = entry
        orig = None
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
            cand = os.path.join(step_dir, f"{base_prefix}__original{ext}")
            if os.path.exists(cand):
                orig = cand
                break
        # capture numeric step index at end like "..._S3"
        idx = None
        m = re.search(r'_S(\d+)$', entry, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1))
        out.append({"folder": step_dir, "name": entry, "index": idx, "original": orig, "base_prefix": base_prefix})
    # sort
    def _sort_key(d):
        return (d["index"] if d["index"] is not None else 10**9, _natural_key(d["name"]))
    out.sort(key=_sort_key)
    return out

# --- Build a strip from a list of originals (for Expert top row) ---
def _build_strip_from_originals(step_items, repeat=2, target_h=160, pad=8, bg=(255,255,255)):
    paths = [d["original"] for d in step_items if d.get("original")]
    if not paths:
        return None
    return _build_strip(paths, repeat=repeat, target_h=target_h, pad=pad, bg=bg)

# --- Overlay resolver: tolerant to joint order/separator ---
def _find_overlay_image_for_step_any_order(step_folder: str, base_prefix: str, subset_overlay_keys: list[str]):
    """
    Look inside <step_folder>/error_combinations for a file whose joint-key SET matches subset_overlay_keys,
    ignoring order and separator type (+, _, -). If no subset given, return the step original if present.
    """
    if not subset_overlay_keys:
        # no errors => prefer original
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
            orig = os.path.join(step_folder, f"{base_prefix}__original{ext}")
            if os.path.exists(orig):
                return orig
        return None

    comb_dir = os.path.join(step_folder, "error_combinations")
    if not os.path.isdir(comb_dir):
        # fallback to original
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
            orig = os.path.join(step_folder, f"{base_prefix}__original{ext}")
            if os.path.exists(orig):
                return orig
        return None

    want = set(subset_overlay_keys)
    # scan all combination files under comb_dir for this base
    pattern = os.path.join(comb_dir, f"{base_prefix}__*")
    for p in sorted(glob.glob(pattern)):
        fname = os.path.basename(p)
        # split after the double underscore and before extension
        tail = os.path.splitext(fname)[0].split("__", 1)
        if len(tail) != 2:
            continue
        combo_str = tail[1]
        have = set(filter(None, _combo_split_re.split(combo_str)))
        if have == want:
            return p

    # if exact set not found, fallback to original
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
        orig = os.path.join(step_folder, f"{base_prefix}__original{ext}")
        if os.path.exists(orig):
            return orig
    return None

def _find_step_images(base_dir: str, exercise_name: str):
    """
    Try multiple reasonable folders for step images and name variants:
      - <base>/Step_Images/Expert/<exercise_name>/
      - <base>/Step_Images/<exercise_name>/
      - <cwd>/Step_Images/Expert/<exercise_name>/
      - <cwd>/Step_Images/<exercise_name>/
      - <base parent>/Step_Images/Expert/<exercise_name>/
      - <base parent>/Step_Images/<exercise_name>/
    where <exercise_name> is tried with several variants:
      raw, underscores, no spaces, canonical (alnum only), titlecased.
    Returns (paths, folder_checked_list).
    """
    variants = []
    raw = exercise_name
    variants.extend([
        raw,
        raw.replace("_", " "),
        raw.replace(" ", "_"),
        raw.replace(" ", ""),
    ])
    # canonical form (alnum only) e.g., 'ClaspandSpread'
    canon = _canonicalize(raw)
    variants.append(canon)
    # titlecased (helps when folders are CamelCase without spaces)
    variants.append(raw.title().replace(" ", ""))  # e.g., "Clasp And Spread" -> "ClaspAndSpread"

    candidate_roots = [
        base_dir,
        os.getcwd(),
        os.path.dirname(base_dir)
    ]
    checked = []
    for root in candidate_roots:
        for v in variants:
            for sub in [
                os.path.join("Step_Images", "Expert", v),
                os.path.join("Step_Images", v),
            ]:
                folder = os.path.join(root, sub)
                checked.append(folder)
                files = _list_image_files(folder)
                if files:
                    return files, checked
    return [], checked

# ---- Step image helpers for WHAM structure ----
def _find_wham_exercise_dir(base_dir: str, exercise_name: str):
    """
    Return the best-matching folder for WHAM step images:
      <base|cwd|parent>/Step_Images/WHAM/<ExerciseVariant>/
    """
    variants = []
    raw = exercise_name
    variants.extend([
        raw,
        raw.replace("_", " "),
        raw.replace(" ", "_"),
        raw.replace(" ", ""),
        _canonicalize(raw),
        raw.title().replace(" ", ""),   # CamelCase
    ])
    roots = [base_dir, os.getcwd(), os.path.dirname(base_dir)]
    for root in roots:
        for v in variants:
            cand = os.path.join(root, "Step_Images", "WHAM", v)
            if os.path.isdir(cand):
                return cand
    return None

def _list_step_images_in_dir(ex_dir: str):
    """List step image files that live directly under the exercise directory (not inside step folders)."""
    if not ex_dir or not os.path.isdir(ex_dir):
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [os.path.join(ex_dir, f) for f in os.listdir(ex_dir) if os.path.isfile(os.path.join(ex_dir, f))]
    files = [p for p in files if os.path.splitext(p)[1].lower() in exts]
    files.sort(key=_natural_key)
    return files

# ---- Pain reports helpers ----------------------------------------------------
def _resolve_pain_reports_dir(base_dir: str) -> str | None:
    """Return `<root>/Pain_mapping/pain_reports` where root ∈ {script dir, CWD, parent}."""
    roots = [base_dir, os.getcwd(), os.path.dirname(base_dir)]
    for root in roots:
        cand = os.path.join(root, "Pain_mapping", "pain_reports")
        if os.path.isdir(cand):
            return cand
    return None


def _list_pain_images_for_patient(pain_dir: str, patient: str) -> list[str]:
    """List pain report images for `patient` by fuzzy filename match (alnum-only contains)."""
    if not pain_dir or not os.path.isdir(pain_dir):
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    try:
        files = [os.path.join(pain_dir, f) for f in os.listdir(pain_dir)
                 if os.path.splitext(f)[1].lower() in exts]
    except Exception:
        return []
    p_norm = "".join(ch for ch in str(patient).lower() if ch.isalnum())
    out = []
    for p in sorted(files, key=_natural_key):
        fname = os.path.basename(p).lower()
        f_norm = "".join(ch for ch in fname if ch.isalnum())
        if p_norm and p_norm in f_norm:
            out.append(p)
    return out
# ---- Step image helpers for WHAM structure ----
def _resolve_step_images(base_dir: str, exercise_name: str):
    """
    Return (exercise_dir, step_image_paths, checked_paths) by searching these locations in order:
      1) <root>/Step_Images/Expert/<ExerciseVariant>/
      2) <root>/Step_Images/WHAM/<ExerciseVariant>/
      3) <root>/Step_Images/<ExerciseVariant>/
    where <root> ∈ {script folder, CWD, parent of script folder}
    Variants include: raw, underscores, no-spaces, canonical alnum, CamelCase.
    """
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
    subdirs_in_priority = [
        os.path.join("Step_Images", "Expert"),
        os.path.join("Step_Images", "WHAM"),
        os.path.join("Step_Images"),
    ]
    checked = []
    for root in roots:
        for sub in subdirs_in_priority:
            for v in variants:
                ex_dir = os.path.join(root, sub, v)
                checked.append(ex_dir)
                step_paths = _list_step_images_in_dir(ex_dir)
                if step_paths:
                    return ex_dir, step_paths, checked
    return None, [], checked

def _step_name_from_path(p: str):
    """Step name is the stem without extension, e.g., 'S3' from '.../S3.png'."""
    return os.path.splitext(os.path.basename(p))[0]

# Map plotting joint names -> overlay joint keys used by try.py outputs
_OVERLAY_JOINT_MAP = {
    "Right_Shoulder": "RShoulder",
    "Left_Shoulder":  "LShoulder",
    "Right_Elbow":    "RElbow",
    "Left_Elbow":     "LElbow",
    "Right_Wrist":    "RWrist",
    "Left_Wrist":     "LWrist",
}

def _pick_overlay_subset_for_step(mae_xy_per_joint: dict, thresh: float, top_pct: float | None = None):
    """
    mae_xy_per_joint: {'Right_Shoulder': 0.18, ...} (already averaged X/Y)
    Rule: include any joint >= thresh. If none, and top_pct is given, include top pct% (at least 1).
    Returns list of overlay keys like ['RShoulder','RWrist'] (sorted).
    """
    # Primary rule: fixed threshold
    chosen = [j for j, v in mae_xy_per_joint.items() if np.isfinite(v) and v >= thresh]
    if not chosen and top_pct is not None and len(mae_xy_per_joint) > 0:
        # Fallback: top pct of joints
        k = max(1, int(round(len(mae_xy_per_joint) * (top_pct / 100.0))))
        chosen = [j for j, _ in sorted(mae_xy_per_joint.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=-1), kv[0]), reverse=True)[:k]]
    # Map to overlay keys & sort for consistent filename
    ov = sorted([_OVERLAY_JOINT_MAP[j] for j in chosen if j in _OVERLAY_JOINT_MAP])
    return ov

def _find_overlay_image_for_step(ex_dir: str, step_name: str, subset_overlay_keys: list):
    """
    Locate the overlay image produced by try.py for this step.
    It should be under:
      <ex_dir>/<StepName>/error_combinations/<StepName>__<Key+Key+...>.png
    If subset is empty, try to return the '<StepName>__original.png' or the base step image.
    """
    step_folder = os.path.join(ex_dir, step_name)
    comb_dir    = os.path.join(step_folder, "error_combinations")
    if subset_overlay_keys:
        fname = f"{step_name}__{'+'.join(subset_overlay_keys)}.png"
        cand  = os.path.join(comb_dir, fname)
        if os.path.exists(cand):
            return cand
    # Empty subset (no issues) or missing combo: fallbacks
    orig = os.path.join(step_folder, f"{step_name}__original.png")
    if os.path.exists(orig):
        return orig
    base = os.path.join(ex_dir, f"{step_name}.png")
    return base if os.path.exists(base) else None

def _build_patient_strip_by_steps(ex_dir: str, step_paths: list, rep_subsets: list,
                                  repeat: int = 1, target_h: int = 160, pad: int = 8, bg=(255,255,255)):
    """
    Build two strips (Rep 1 and Rep 2) of patient overlay images from WHAM directory structure:
      <ex_dir>/<StepFolder>/error_combinations/<base_prefix>__J1+J2.png  (order-insensitive; supports +/_/-)
    - step_paths: not used directly for file names; only to preserve ordering.
    """
    if not step_paths or not rep_subsets or len(rep_subsets) != 2:
        return None, None

    # Instead of step_paths files, list WHAM step folders to align with overlay storage.
    step_items = _list_step_folders_with_originals(ex_dir)
    if not step_items:
        return None, None

    # Keep only as many steps as we have subsets for
    for_strip = step_items[:len(rep_subsets[0])]

    def _build_one(rep_idx: int):
        imgs = []
        for item, subset in zip(for_strip, rep_subsets[rep_idx]):
            img_path = _find_overlay_image_for_step_any_order(item["folder"], item["base_prefix"], subset)
            if not img_path or not os.path.exists(img_path):
                # as last resort, use original if present
                img_path = item.get("original")
            if not img_path:
                continue
            im = Image.open(img_path).convert("RGB")
            w, h = im.size
            new_w = int(w * (target_h / float(h)))
            imgs.append(im.resize((new_w, target_h)))
        if not imgs:
            return None
        total_w = sum(im.size[0] for im in imgs) + pad * (len(imgs) - 1)
        strip = Image.new("RGB", (total_w, target_h), bg)
        x = 0
        for im in imgs:
            strip.paste(im, (x, 0))
            x += im.size[0] + pad
        return strip

    rep1 = _build_one(0)
    rep2 = _build_one(1)
    return rep1, rep2


# === Helper: Concatenate two images horizontally with padding ===
def _concat_horizontal(img_left: Image.Image, img_right: Image.Image, pad: int = 8, bg=(255, 255, 255)) -> Image.Image:
    """Concatenate two PIL images of the same height with a small gap."""
    if img_left is None and img_right is None:
        return None
    if img_left is None:
        return img_right
    if img_right is None:
        return img_left
    h = img_left.size[1]
    if img_right.size[1] != h:
        rw = int(img_right.size[0] * (h / float(img_right.size[1])))
        img_right = img_right.resize((rw, h))
    w = img_left.size[0] + pad + img_right.size[0]
    out = Image.new("RGB", (w, h), bg)
    out.paste(img_left, (0, 0))
    out.paste(img_right, (img_left.size[0] + pad, 0))
    return out


# === Helper: Compose expert+patient strips with timeline ===
def _compose_expert_patient_with_timeline(step_paths, patient_rep1_strip, patient_rep2_strip,
                                          fps=25.0, total_frames: int | None = None,
                                          target_h=160, pad=8, bg=(255,255,255), timeline_h=36,
                                          expert_strip_img=None,
                                          heat_values: list | None = None,
                                          heat_h: int = 16,
                                          heat_vmin: float | None = None,
                                          heat_vmax: float | None = None,
                                          fb_summary: str | None = None,
                                          fb_grade: str | None = None,
                                          fb_similarity: float | None = None):
    """
    Build a composite header:
      [Expert steps rep1+rep2]  (top row)
      [Patient overlays rep1 | rep2]  (middle row)
      [timeline]  (bottom)
    Returns a PIL.Image or None.
    """
    # Allow building a composite if we have either expert strip or patient strips
    if (not step_paths) and (expert_strip_img is None) and (patient_rep1_strip is None and patient_rep2_strip is None):
        return None
    # Expert strip with 2 repetitions (if not provided)
    if expert_strip_img is not None:
        expert_strip = expert_strip_img
    else:
        expert_strip = _build_strip(step_paths, repeat=2, target_h=target_h, pad=pad, bg=bg)
        if expert_strip is None:
            return None

    # Patient combined strip (rep1 + rep2)
    patient_combined = _concat_horizontal(patient_rep1_strip, patient_rep2_strip, pad=pad, bg=bg)
    if patient_combined is None:
        patient_combined = Image.new("RGB", expert_strip.size, bg)
    else:
        if patient_combined.size[0] != expert_strip.size[0]:
            new_w = expert_strip.size[0]
            new_h = int(patient_combined.size[1] * (new_w / float(patient_combined.size[0])))
            patient_combined = patient_combined.resize((new_w, new_h))

    # Stack: expert (H), patient (H), then add timeline area (timeline_h)
    W = expert_strip.size[0]
    H1 = expert_strip.size[1]
    H2 = patient_combined.size[1]
    # wider left pad to host row labels ("Expert", "Patient (Rep1 | Rep2)")
    left_pad = 110
    right_pad = 18
    hm_h = max(0, int(heat_h)) if heat_values else 0

    # Determine a safe timeline block height based on font metrics so labels never clip
    font_tl = _load_font(12)
    tmp_w, tmp_h = _text_size("00s", font_tl)
    tl_h_used = max(int(timeline_h), int(tmp_h) + 18)  # labels + ticks + spacing
    bottom_pad = 6  # extra breathing room below timeline

    total_h = H1 + H2 + hm_h + tl_h_used + bottom_pad
    canvas = Image.new("RGB", (W + left_pad + right_pad, total_h), bg)

    # Paste rows
    y = 0
    canvas.paste(expert_strip, (left_pad, y)); y += H1
    canvas.paste(patient_combined, (left_pad, y)); y += H2

    # Row labels in the left margin
    draw = ImageDraw.Draw(canvas)
    # Fonts for labels and timeline
    font_lbl = _load_font(14)
    # Expert label (centered vertically in expert row)
    label_expert = "Expert"
    tw_e, th_e = _text_size(label_expert, font_lbl)
    x_e = max(4, left_pad - tw_e - 6)
    y_e = (H1 - th_e) // 2
    draw.text((x_e, y_e), label_expert, fill=(40, 40, 40), font=font_lbl)
    # Patient label (centered vertically in patient row)
    label_patient = "Patient"
    tw_p, th_p = _text_size(label_patient, font_lbl)
    x_p = max(4, left_pad - tw_p - 6)
    y_p = H1 + (H2 - th_p) // 2
    draw.text((x_p, y_p), label_patient, fill=(40, 40, 40), font=font_lbl)

    # --- Heatmap bar (refactored helper) ---
    if heat_values and hm_h > 0:
        y = _draw_smooth_heatbar(draw, left_pad, y, W, hm_h, heat_values, heat_vmin, heat_vmax)



    # Draw timeline along the bottom width W
    # Use the preloaded timeline font and adjusted block height
    font = font_tl
    y0 = y + 8
    major_len = 8
    minor_len = 4
    y_major = y0 + major_len
    y_minor = y0 + minor_len

    # Baseline
    draw.line([(left_pad, y0), (left_pad + W, y0)], fill=(110, 110, 110), width=2)

    # Seconds total
    seconds_total = None
    if total_frames is not None and fps and fps > 0:
        seconds_total = max(0.0, float(total_frames) / float(fps))
    if seconds_total is None:
        seconds_total = 10.0

    # Choose tick spacing
    desired_labels = 8
    approx_px_per_s = W / max(seconds_total, 1e-6)
    candidates = [0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 60]
    tick_step = candidates[-1]
    for c in candidates:
        if (seconds_total / c) <= desired_labels and approx_px_per_s * c >= 70:
            tick_step = c
            break
    minor_step = tick_step / 5.0 if tick_step >= 1.0 else max(0.1, tick_step / 2.0)

    def sec_to_x(sec: float) -> int:
        if seconds_total <= 0:
            return left_pad
        frac = min(1.0, max(0.0, sec / seconds_total))
        return int(left_pad + frac * W)

    # Minor ticks
    s = 0.0
    while s <= seconds_total + 1e-9:
        x_pos = sec_to_x(s)
        draw.line([(x_pos, y0), (x_pos, y_minor)], fill=(160, 160, 160), width=1)
        s += minor_step

    # Major ticks + labels
    s = 0.0
    while s <= seconds_total + 1e-9:
        x_pos = sec_to_x(s)
        draw.line([(x_pos, y0), (x_pos, y_major)], fill=(110, 110, 110), width=2)
        label = f"{s:.0f}s" if tick_step >= 1 else f"{s:.1f}s"
        tw, th = _text_size(label, font)
        label_x = max(left_pad, min(left_pad + W - tw, x_pos - tw // 2))
        label_y = y_major + 2
        draw.text((label_x, label_y), label, fill=(70, 70, 70), font=font)
        s += tick_step

    # Ensure 0s label
    if tick_step > 0.5:
        label = "0s"
        tw, th = _text_size(label, font)
        draw.text((left_pad, y_major + 2), label, fill=(70, 70, 70), font=font)

    return canvas

# ---------------- Patient-only aggressive despike ----------------
def _neighbor_mean(x, i, k):
    """
    Mean of k neighbors on each side, excluding x[i].
    Example: k=2 uses {i-2,i-1,i+1,i+2} clipped to bounds.
    """
    n = len(x)
    left = max(0, i - k)
    right = min(n, i + k + 1)
    idx = list(range(left, i)) + list(range(i+1, right))
    if not idx:
        return x[i]
    return float(np.nanmean(x[idx]))

def _local_stats(x, i, half_win):
    n = len(x)
    l = max(0, i - half_win)
    r = min(n, i + half_win + 1)
    seg = x[l:r]
    med = np.nanmedian(seg)
    mad = np.nanmedian(np.abs(seg - med)) + 1e-9
    # robust local range (5th–95th)
    low = np.nanpercentile(seg, 5) if np.isfinite(np.nanpercentile(seg, 5)) else med
    high = np.nanpercentile(seg, 95) if np.isfinite(np.nanpercentile(seg, 95)) else med
    loc_range = max(1e-9, high - low)
    return med, mad, loc_range

def _despike_series(
    s: pd.Series,
    win: int = 9,           # local window to estimate MAD/range
    k_neighbors: int = 2,   # neighbor count each side for mean
    z_thresh: float = 2.5,  # lower => more aggressive
    jump_abs: float | None = None,    # absolute jump threshold vs prev/next (unit of data)
    jump_pct_range: float | None = 25.0,  # % of local range; 25 means 0.25*local_range
    passes: int = 2,        # repeat to catch 2–3 point bursts
    post_ma: int | None = 3 # optional small moving-average after despike
) -> pd.Series:
    if s is None or len(s) < 3:
        return s
    x = pd.to_numeric(s, errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    n = len(x)
    win = max(3, int(win))
    if win % 2 == 0: win += 1
    half = win // 2

    out = x.copy()
    for _ in range(max(1, passes)):
        x = out.copy()
        for i in range(1, n - 1):
            nb_mean = _neighbor_mean(x, i, k_neighbors)
            med, mad, loc_rng = _local_stats(x, i, half)

            # thresholds
            thr_mad = z_thresh * mad
            thr_jump = 0.0
            if jump_abs is not None:
                thr_jump = max(thr_jump, float(jump_abs))
            if jump_pct_range is not None:
                thr_jump = max(thr_jump, float(jump_pct_range) / 100.0 * loc_rng)

            # conditions
            dev_nb = abs(x[i] - nb_mean)
            jump_prev = abs(x[i] - x[i-1])
            jump_next = abs(x[i] - x[i+1])

            cond_mad = dev_nb > thr_mad
            cond_jump = (jump_prev > thr_jump) and (jump_next > thr_jump)

            if cond_mad or cond_jump:
                out[i] = nb_mean

    # optional small moving average to soften edges
    if post_ma and post_ma >= 3:
        k = int(post_ma)
        if k % 2 == 0: k += 1
        pad = k // 2
        padded = np.pad(out, (pad, pad), mode="reflect")
        kernel = np.ones(k) / k
        out = np.convolve(padded, kernel, mode="valid")

    return pd.Series(out, index=s.index)

def clean_patient_df(
    df: pd.DataFrame,
    win: int = 9,
    k_neighbors: int = 2,
    z_thresh: float = 2.5,
    jump_abs: float | None = None,
    jump_pct_range: float = 25.0,
    passes: int = 2,
    post_ma: int | None = 3
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    tmp = df.copy()
    tmp["X"] = _despike_series(tmp["X"], win, k_neighbors, z_thresh, jump_abs, jump_pct_range, passes, post_ma)
    tmp["Y"] = _despike_series(tmp["Y"], win, k_neighbors, z_thresh, jump_abs, jump_pct_range, passes, post_ma)
    return tmp.interpolate(limit_direction="both").reset_index(drop=True)

# =========================
# ===== SIDEBAR UI ========
# =========================

# ------------------------
# Navigation: Back to Home button (top of sidebar)
# ------------------------
st.markdown("""<style>.sidebar .block-container{padding-top:1rem !important;}</style>""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    if st.button("🏠 Go to Homepage", use_container_width=True):
        st.query_params.clear()
        st.rerun()


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

# Use query params with robust matching against available folders
_pat_default = _best_match_from_list(patients, qp_patient) or (patients[0] if patients else None)
selected_patient = st.sidebar.selectbox(
    "👤 Patient", patients,
    index=(patients.index(_pat_default) if (_pat_default in patients) else 0)
)

patient_exercises = list_dirs(os.path.join(PATIENT_DIR, selected_patient))
_ex_default = _best_match_from_list(patient_exercises, qp_exercise) or (patient_exercises[0] if patient_exercises else None)
selected_exercise = st.sidebar.selectbox(
    "💪 Exercise", patient_exercises,
    index=(patient_exercises.index(_ex_default) if (_ex_default in patient_exercises) else 0)
)

show_legend = st.sidebar.checkbox("Show legend", value=False)
sync_length = st.sidebar.checkbox("Trim to common length (X/Y)", value=True)
show_metrics = st.sidebar.checkbox("Show per-joint metrics table", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🩹 Patient despike (aggressive)")
enable_fix   = st.sidebar.checkbox("Enable", value=True)
win          = st.sidebar.slider("Local window (odd)", 5, 41, 35, step=2)   # default 35
k_neighbors  = st.sidebar.slider("Neighbor count each side", 1, 10, 10, step=1)  # default 10
z_thresh     = st.sidebar.slider("MAD multiplier (lower = stronger)", 1.0, 5.0, 1.5, 0.1)  # default 1.5
jump_abs     = st.sidebar.number_input("Absolute jump threshold (optional, data units)", value=0.0, step=0.1)
jump_pct     = st.sidebar.slider("Jump as % of local range", 0.0, 100.0, 25.0, 1.0)  # default 25
passes       = st.sidebar.slider("Passes (repeat cleaning)", 1, 5, 2, 1)   # default 2
post_ma      = st.sidebar.slider("Post moving-average (0=off)", 0, 9, 3, 1)   # default 3

st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Step deviation rules")
step_thresh = st.sidebar.slider("Joint MAE threshold", 0.00, 1.00, 0.20, 0.01)
step_top_pct = st.sidebar.slider("If none pass threshold, pick top % joints", 0.0, 100.0, 0.0, 5.0)

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Heatmap settings")
show_heatmap = st.sidebar.checkbox("Show heatmap bar", value=True)
# OLD: heat_h_ctrl = st.sidebar.slider("Heatmap height (px)", 8, 48, 24, 1)
heat_h_ctrl = st.sidebar.slider("Heatmap height (px)", 6, 36, 16, 1)

# --- Global scaling controls for heatmap ---
st.sidebar.markdown("---")
st.sidebar.subheader("📏 Heatmap scale")
heat_scale_mode = st.sidebar.selectbox(
    "Normalization", ["Per-exercise (robust)", "Global (fixed)"], index=0,
    help=(
        "Per-exercise uses robust percentiles (5th–95th) within the current exercise; "
        "Global applies fixed bounds so colors are comparable across exercises/patients."
    )
)

global_min_mae = None
global_max_mae = None
if heat_scale_mode == "Global (fixed)":
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        global_min_mae = st.number_input("Min MAE", value=0.00, step=0.01)
    with col_b:
        global_max_mae = st.number_input("Max MAE", value=0.50, step=0.01)
    if global_max_mae <= global_min_mae:
        st.sidebar.warning("Max MAE must be greater than Min MAE for global scaling.")

# normalize inputs
if jump_abs == 0.0:
    jump_abs_val = None
else:
    jump_abs_val = float(jump_abs)
jump_pct_val = float(jump_pct)

# --- Feedback CSV path sidebar ---
fb_csv_path = st.sidebar.text_input(
    "Path to feedback CSV",
    value="outputs/batch_reports/llm_reports5/summary/final_llm_feedback.csv",
    help="CSV containing summary, grade, and overall_similarity per patient & exercise"
)
# Keep the feedback path in session so the homepage can access it
st.session_state['fb_csv_path'] = fb_csv_path

# --- Generated overlays UI ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ EMG Signals")
enable_emg = st.sidebar.checkbox("Enable EMG tab", value=True)
emg_base_dir = st.sidebar.text_input(
    "EMG directory",
    value=EMG_DIR,
    help="Folder: EMG_Dataset_TOLF-B/{patient}/{exercise}.csv",
)
emg_norm_mode = st.sidebar.selectbox(
    "Normalization",
    ["None", "Z-score per channel", "Min-Max per channel"],
    index=1,
)
emg_smooth = st.sidebar.slider(
    "Envelope smoothing (MA window)", 1, 201, 21, 2,
    help="Apply moving-average to EMG (samples). Set 1 to disable.")

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Generated overlays")
use_generated = st.sidebar.checkbox("Use generated images directory", value=True)
generated_dir = st.sidebar.text_input("Generated images directory", value="arm_heatmaps_aligned",
                                      help="Folder containing arm_{patient}_{exercise}_rep{rep}_step{step}.png")

base_dir = os.path.dirname(os.path.abspath(__file__))
ex_dir, step_paths, searched_step_dirs = _resolve_step_images(base_dir, selected_exercise)
est_frames = _first_available_frames_count(PATIENT_DIR, EXPERT_DIR, selected_patient, selected_exercise)
fps_val = 25.0
# --- New: resolve Expert and WHAM dirs explicitly ---
expert_dir = _resolve_ex_dir_by_subfolder(base_dir, selected_exercise, "Expert")
wham_dir   = _resolve_ex_dir_by_subfolder(base_dir, selected_exercise, "WHAM")

# Pain reports
pain_dir = _resolve_pain_reports_dir(base_dir)
pain_images_for_patient = _list_pain_images_for_patient(pain_dir, selected_patient)

# --- Load feedback for the selected patient/exercise (CSV from sidebar) ------
fb_summary, fb_grade, fb_patient_score = None, None, None
if fb_csv_path and os.path.isfile(fb_csv_path):
    fb_row = _load_feedback(fb_csv_path, selected_patient, selected_exercise)
    if fb_row:
        fb_summary = fb_row.get("summary")
        fb_grade = fb_row.get("grade")
        fb_patient_score = fb_row.get("patient_score")

st.caption(
    "Expert is **untouched** (bold). Patient is **dotted** and cleaned. "
    "This cleaner catches single spikes and small bursts. Increase passes or lower MAD multiplier for stronger effect."
)

# =========================
# ====== LOAD DATA ========
# =========================
patient_dfs = {}
expert_dfs = {}

for joint in JOINT_ORDER:
    p = load_joint_csv(PATIENT_DIR, selected_patient, selected_exercise, joint)
    e = load_joint_csv(EXPERT_DIR,  selected_patient, selected_exercise, joint)

    if enable_fix:
        p = clean_patient_df(
            p,
            win=win,
            k_neighbors=k_neighbors,
            z_thresh=z_thresh,
            jump_abs=jump_abs_val,
            jump_pct_range=jump_pct_val,
            passes=passes,
            post_ma=post_ma if post_ma >= 3 else None
        )

    patient_dfs[joint] = p
    expert_dfs[joint]  = e

# Global y range across all (patient + expert)
all_for_range = []
for j in JOINT_ORDER:
    if patient_dfs[j] is not None: all_for_range.append(patient_dfs[j])
    if expert_dfs[j] is not None:  all_for_range.append(expert_dfs[j])
ymin, ymax = global_min_max(all_for_range)

# ---------- Patient step overlays (2 reps) ----------
def _mae_step(joint, a, b):
    p_df = patient_dfs.get(joint); e_df = expert_dfs.get(joint)
    if p_df is None or e_df is None:
        return np.nan
    a0 = max(0, min(len(p_df)-1, a)); b0 = max(a0+1, min(len(p_df), b))
    a1 = max(0, min(len(e_df)-1, a)); b1 = max(a1+1, min(len(e_df), b))
    n = min(b0 - a0, b1 - a1)
    if n <= 0: return np.nan
    p_x = p_df["X"].iloc[a0:a0+n].to_numpy(dtype=float)
    p_y = p_df["Y"].iloc[a0:a0+n].to_numpy(dtype=float)
    e_x = e_df["X"].iloc[a1:a1+n].to_numpy(dtype=float)
    e_y = e_df["Y"].iloc[a1:a1+n].to_numpy(dtype=float)
    mae_x = float(np.mean(np.abs(p_x - e_x)))
    mae_y = float(np.mean(np.abs(p_y - e_y)))
    return 0.5 * (mae_x + mae_y)

# Prefer generated overlays if available
rep1_gen, rep2_gen = [], []
if use_generated:
    rep1_gen, rep2_gen = _find_generated_images(generated_dir, selected_patient, selected_exercise)

if step_paths:
    N = len(step_paths)
    ref_joint = next((j for j in JOINT_ORDER if patient_dfs.get(j) is not None and expert_dfs.get(j) is not None), None)
    if ref_joint is not None:
        p_len = len(patient_dfs[ref_joint])
        e_len = len(expert_dfs[ref_joint])
        L = min(p_len, e_len) if sync_length else max(p_len, e_len)
        if L >= 2 * N:
            rep_len = L // 2
            def _segments(rep_start):
                segs = []
                for s in range(N):
                    a = rep_start + int(np.floor(s * (rep_len / N)))
                    b = rep_start + int(np.floor((s + 1) * (rep_len / N)))
                    segs.append((a, b))
                return segs
            segs_r1 = _segments(0)
            segs_r2 = _segments(rep_len)

            per_step_joint_mae = {}
            rep_subsets = []
            for rep_label, segs in zip(['rep1', 'rep2'], (segs_r1, segs_r2)):
                subsets_this_rep = []
                for idx, (a, b) in enumerate(segs, start=1):
                    per_joint = {}
                    for j in JOINT_ORDER:
                        per_joint[j] = _mae_step(j, a, b)
                    per_step_joint_mae[(rep_label, idx)] = per_joint
                    subset_keys = _pick_overlay_subset_for_step(per_joint, thresh=step_thresh,
                                                                top_pct=(step_top_pct if step_top_pct > 0 else None))
                    subsets_this_rep.append(subset_keys)
                rep_subsets.append(subsets_this_rep)

            # Build heatmap values: mean MAE across joints per step (rep1 then rep2)
            def _mean_mae_for(rep_key, s_idx):
                maes = per_step_joint_mae.get((rep_key, s_idx), {})
                arr = [v for v in maes.values() if v is not None and np.isfinite(v)]
                return float(np.nanmean(arr)) if arr else 0.0

            heat_values = []
            for rep_key in ['rep1', 'rep2']:
                for s_idx in range(1, N + 1):
                    heat_values.append(_mean_mae_for(rep_key, s_idx))

            # Respect sidebar toggle
            heat_values_to_use = heat_values if show_heatmap and len(heat_values) > 0 else None
            heat_h_to_use = int(heat_h_ctrl) if show_heatmap else 0
            # Decide global overrides based on sidebar mode
            heat_vmin_override = (global_min_mae if heat_scale_mode == "Global (fixed)" else None)
            heat_vmax_override = (global_max_mae if heat_scale_mode == "Global (fixed)" else None)

            # Build strips: prefer generated overlays if present; otherwise fall back to WHAM/Expert discovery
# Build strips: always use Expert originals on the top row (as before).
            if rep1_gen or rep2_gen:
                # Generated patient strips
                rep1_strip = _build_strip(rep1_gen, repeat=1, target_h=160, pad=8)
                rep2_strip = _build_strip(rep2_gen, repeat=1, target_h=160, pad=8)

                # Top row: Expert originals (preferred), else fall back gracefully
                expert_items = _list_step_folders_with_originals(expert_dir) if expert_dir else []
                expert_strip = _build_strip_from_originals(expert_items, repeat=2, target_h=160, pad=8)
                if expert_strip is None:
                    # fallback to resolved step images if available
                    if step_paths:
                        expert_strip = _build_strip(step_paths, repeat=2, target_h=160, pad=8)
                    else:
                        # last resort: shape-match to generated images
                        top_src = rep1_gen if rep1_gen else rep2_gen
                        expert_strip = _build_strip(top_src, repeat=2, target_h=160, pad=8)

                # keep step_paths for width/timeline purposes when available
                step_paths_for_width = step_paths
                header_title = f"Expert & Patient Steps — {selected_exercise}"
            else:
                # Original Expert/WHAM resolution for patient overlays
                expert_items = _list_step_folders_with_originals(expert_dir) if expert_dir else []
                expert_strip = _build_strip_from_originals(expert_items, repeat=2, target_h=160, pad=8)
                patient_source_dir = wham_dir if wham_dir else ex_dir
                rep1_strip, rep2_strip = _build_patient_strip_by_steps(
                    patient_source_dir, step_paths, rep_subsets, repeat=1, target_h=160, pad=8
                )
                step_paths_for_width = step_paths
                header_title = f"Expert & Patient Steps — {selected_exercise}"

            st.subheader(header_title)
            composite = _compose_expert_patient_with_timeline(
                step_paths_for_width, rep1_strip, rep2_strip,
                fps=fps_val, total_frames=est_frames,
                target_h=160, pad=8, expert_strip_img=expert_strip,
                heat_values=heat_values_to_use, heat_h=heat_h_to_use,
                heat_vmin=heat_vmin_override, heat_vmax=heat_vmax_override
            )
            if composite is not None:
                st.image(
                    composite,
                    use_container_width=True,
                    caption=(
                        "Top: Generated overlays (2× repeat) • Middle: Generated Rep1 | Rep2" if (rep1_gen or rep2_gen)
                        else f"Top: Expert originals (2 reps) • Middle: Patient overlays (Rep1 | Rep2)"
                    ) + (
                        f" • Timeline: {fps_val:g} fps" + (f" • ~{est_frames/fps_val:.1f}s" if est_frames else "")
                    )
                )
                # --- Blue info panel: Summary + Grade + Patient Score --------------------
                if (fb_summary or fb_grade or fb_patient_score is not None):
                    _summary_text = fb_summary or ""
                    _grade_text = fb_grade or "N/A"
                    _score_text = (str(fb_patient_score) if fb_patient_score is not None else "N/A")
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #1f77b4; border-radius: 8px; background: rgba(31,119,180,0.06); padding: 14px 16px; margin: 10px auto; max-width: 70%; text-align:center;">
                          <div style="font-weight:600; color:#1f77b4; font-size:16px; margin-bottom:6px;">LLM-based Exercise Feedback</div>
                          <div style="margin-bottom:8px; line-height:1.4; color:#222;">{_summary_text}</div>
                          <div style="display:flex; justify-content:center; gap:18px; flex-wrap:wrap; color:#222;">
                            <div><b>Grade:</b> {_grade_text}</div>
                            <div><b>Patient Score:</b> {_score_text}</div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif fb_csv_path:
                    # Helpful hint if a CSV is provided but no row matched
                    st.caption("No feedback found for this patient/exercise in the selected CSV.")
            else:
                st.info("Could not build composite strip (missing images).")

            def _build_step_metrics_df(selected_patient, selected_exercise, step_paths, rep_subsets, per_step_joint_mae):
                rows = []
                step_names = [os.path.splitext(os.path.basename(p))[0] for p in step_paths]
                for rep_i, rep_key in enumerate(['rep1','rep2'], start=1):
                    for s_idx, step_name in enumerate(step_names, start=1):
                        maes = per_step_joint_mae.get((rep_key, s_idx), {})
                        chosen_overlay_keys = rep_subsets[rep_i-1][s_idx-1] if rep_subsets and len(rep_subsets) >= rep_i else []
                        rows.append({
                            "patient": selected_patient,
                            "exercise": selected_exercise,
                            "rep": rep_i,
                            "step_index": s_idx,
                            "step_name": step_name,
                            "chosen_overlay_keys": "+".join(chosen_overlay_keys) if chosen_overlay_keys else "",
                            **{f"mae_{j}": float(maes.get(j, float('nan'))) for j in JOINT_ORDER}
                        })
                return pd.DataFrame(rows)

            def _persist_step_results(df: pd.DataFrame, selection_dir="outputs/step_deltas"):
                os.makedirs(selection_dir, exist_ok=True)
                if df.empty:
                    return None, None
                p = df["patient"].iloc[0]; ex = df["exercise"].iloc[0]
                csv_path = os.path.join(selection_dir, f"{p}__{ex}__step_metrics.csv")
                json_path = os.path.join(selection_dir, f"{p}__{ex}__overlay_selection.json")
                df.to_csv(csv_path, index=False)
                sel = {}
                for _, r in df.iterrows():
                    key = f"rep{int(r['rep'])}_step{int(r['step_index'])}"
                    sel[key] = {"subset": r["chosen_overlay_keys"]}
                with open(json_path, "w") as f:
                    json.dump(sel, f, indent=2)
                return csv_path, json_path

            df_steps = _build_step_metrics_df(selected_patient, selected_exercise, step_paths, rep_subsets, per_step_joint_mae)
            csv_path, json_path = _persist_step_results(df_steps)

        else:
            st.info(f"Not enough frames to split into 2×{N} steps (got {L}).")
    else:
        st.info("Cannot compute step deviations — no joint with both patient & expert data.")
else:
    with st.expander("⚠️ No step images found — paths I checked"):
        st.write("\n".join(searched_step_dirs))

# =========================
# ======== TABS ===========
# =========================
tab_compare, tab_delta, tab_steps, tab_pain, tab_emg = st.tabs(
    [
        "📊 Compare (Patient vs Expert)",
        "➖ Delta (Patient − Expert)",
        "🧾 Stepwise Feedback",
        "🩹 Pain Reports",
        "⚡ EMG",
    ]
)


# =========================
# === TAB 1: COMPARE ======
# =========================
with tab_compare:
    # Compact toggle & height
    compact = st.sidebar.checkbox("Compact 3×2 grid", value=True, help="Show all 6 joints in a 3×2 grid.")
    fig_height = 950 if compact else 1800

    # 3x2 grid of subplots
    rows, cols = 3, 2
    order = JOINT_ORDER  # keep original order
    titles = [f"{j.replace('_', ' ')} (X & Y)" for j in order]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.12 if compact else 0.15,
        horizontal_spacing=0.08 if compact else 0.1,
        shared_xaxes=False
    )

    for idx, joint in enumerate(order):
        r = idx // cols + 1
        c = idx % cols + 1
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
                row=r, col=c
            )
            fig.add_trace(
                go.Scatter(
                    y=e_df["Y"], mode="lines",
                    name=f"{joint} Expert Y",
                    line=dict(width=3, dash="solid"),
                    hovertemplate="Expert Y: %{y:.4f}<extra></extra>"
                ),
                row=r, col=c
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
                row=r, col=c
            )
            fig.add_trace(
                go.Scatter(
                    y=p_df["Y"], mode="lines",
                    name=f"{joint} Patient Y",
                    line=dict(width=2, dash="dot"),
                    hovertemplate="Patient Y: %{y:.4f}<extra></extra>"
                ),
                row=r, col=c
            )

        fig.update_yaxes(range=[ymin, ymax], title_text="Norm.", row=r, col=c)

    # Label bottom row x-axes
    for c in range(1, cols + 1):
        fig.update_xaxes(title_text="Frame Index", row=rows, col=c)

    fig.update_layout(
        height=fig_height,
        title_text=f"Patient vs Expert — {selected_patient} · {selected_exercise}",
        showlegend=show_legend,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# === TAB 2: DELTAS =======
# =========================
with tab_delta:
    # Match the layout controls to the compare tab, but separate toggle so users can choose independently
    compact_delta = st.sidebar.checkbox("Compact 3×2 grid (Δ)", value=True, help="Show all 6 joints in a 3×2 grid for Δ.")
    fig_height_delta = 950 if compact_delta else 1800

    st.markdown("#### Δ (Patient − Expert) per Joint and Axis")

    rows, cols = 3, 2
    order = JOINT_ORDER
    titles = [f"{j.replace('_', ' ')} ΔX & ΔY" for j in order]

    fig_d = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        shared_xaxes=False,
        vertical_spacing=0.12 if compact_delta else 0.15,
        horizontal_spacing=0.08 if compact_delta else 0.1,
    )

    metrics_rows = []
    all_dx_dy = []

    for idx, joint in enumerate(order):
        r = idx // cols + 1
        c = idx % cols + 1
        p_df = patient_dfs[joint]
        e_df = expert_dfs[joint]

        if p_df is None or e_df is None:
            fig_d.add_trace(go.Scatter(y=[]), row=r, col=c)
            metrics_rows.append({"Joint": joint, "MAE X": np.nan, "RMSE X": np.nan, "Corr X": np.nan,
                                 "MAE Y": np.nan, "RMSE Y": np.nan, "Corr Y": np.nan})
            continue

        n = min(len(p_df), len(e_df)) if sync_length else max(len(p_df), len(e_df))
        p_df = p_df.iloc[:n].reset_index(drop=True)
        e_df = e_df.iloc[:n].reset_index(drop=True)

        dx = p_df["X"] - e_df["X"]
        dy = p_df["Y"] - e_df["Y"]

        fig_d.add_trace(
            go.Scatter(
                y=dx, mode="lines", name=f"{joint} ΔX",
                line=dict(width=2, dash="solid"),
                hovertemplate="ΔX: %{y:.4f}<extra></extra>"
            ), row=r, col=c
        )
        fig_d.add_trace(
            go.Scatter(
                y=dy, mode="lines", name=f"{joint} ΔY",
                line=dict(width=2, dash="dash"),
                hovertemplate="ΔY: %{y:.4f}<extra></extra>"
            ), row=r, col=c
        )

        mae_x, rmse_x, corr_x = quick_metrics(p_df["X"], e_df["X"])
        mae_y, rmse_y, corr_y = quick_metrics(p_df["Y"], e_df["Y"])
        metrics_rows.append({"Joint": joint,
                             "MAE X": round(mae_x, 4), "RMSE X": round(rmse_x, 4),
                             "Corr X": round(corr_x, 4) if not np.isnan(corr_x) else np.nan,
                             "MAE Y": round(mae_y, 4), "RMSE Y": round(rmse_y, 4),
                             "Corr Y": round(corr_y, 4) if not np.isnan(corr_y) else np.nan})

        all_dx_dy.append(dx.values); all_dx_dy.append(dy.values)

    # Uniform y-range across all Δ subplots for consistency
    if all_dx_dy:
        stacked = np.concatenate(all_dx_dy)
        dmin, dmax = stacked.min(), stacked.max()
        pad = (dmax - dmin) * 0.05 if dmax > dmin else 0.1
        d_range = [dmin - pad, dmax + pad]
    else:
        d_range = [-1, 1]

    for idx, joint in enumerate(order):
        r = idx // cols + 1
        c = idx % cols + 1
        fig_d.update_yaxes(range=d_range, title_text="Δ", row=r, col=c)

    # Label bottom row x-axes
    for c in range(1, cols + 1):
        fig_d.update_xaxes(title_text="Frame Index", row=rows, col=c)

    fig_d.update_layout(
        height=fig_height_delta,
        title_text=f"Deltas — {selected_patient} · {selected_exercise}",
        showlegend=show_legend,
        margin=dict(l=60, r=20, t=60, b=40)
    )
    st.plotly_chart(fig_d, use_container_width=True)

    if show_metrics:
        st.markdown("##### Per-Joint Metrics (lower MAE/RMSE and higher Corr are better)")
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

# =========================
# === PER-STEP METRICS ====
# =========================
if 'df_steps' in locals() and isinstance(df_steps, pd.DataFrame) and not df_steps.empty:
    st.markdown("### Per-step joint MAE & chosen overlays")
    st.dataframe(df_steps, use_container_width=True, hide_index=True)
    # Optional download (if persisted earlier)
    if 'csv_path' in locals() and csv_path and os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button("⬇️ Download per-step CSV", f, file_name=os.path.basename(csv_path))    

# =========================
# === TAB X: EMG ==========
# =========================
if enable_emg:
    with tab_emg:
        st.markdown("### ⚡ EMG — Wide CSV per Exercise")
        df_emg, time_col, ch_cols = load_emg_table(emg_base_dir, selected_patient, selected_exercise)
        if df_emg is None:
            st.info(
                "No EMG CSV found for this selection. Expected file: "
                f"{os.path.join(emg_base_dir, selected_patient, selected_exercise + '.csv')}"
            )
        else:
            # Default selection: prefer DELTOID/BICEPS if present
            default_sel = [c for c in ch_cols if any(k in c.upper() for k in ("DELTOID", "BICEPS"))] or ch_cols[:6]
            sel = st.multiselect("Channels", ch_cols, default=default_sel)
            if not sel:
                st.warning("Select at least one channel to plot.")
            else:
                def _norm_apply(s):
                    if emg_norm_mode == "Z-score per channel":
                        mu, sd = float(s.mean()), float(s.std(ddof=0)) or 1.0
                        return (s - mu) / sd
                    elif emg_norm_mode == "Min-Max per channel":
                        mn, mx = float(s.min()), float(s.max())
                        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                            return s * 0.0
                        return (s - mn) / (mx - mn)
                    return s
                def _smooth_apply(s, k):
                    k = int(max(1, k))
                    if k <= 1:
                        return s
                    if k % 2 == 0:
                        k += 1
                    pad = k // 2
                    arr = s.to_numpy(dtype=float)
                    padded = np.pad(arr, (pad, pad), mode='reflect')
                    kernel = np.ones(k) / k
                    out = np.convolve(padded, kernel, mode='valid')
                    return pd.Series(out)

                # X-axis: use time if present, else index
                if time_col and df_emg[time_col].notna().any():
                    x = pd.to_numeric(df_emg[time_col], errors='coerce')
                else:
                    x = pd.Series(np.arange(len(df_emg)), name='idx')
                    time_col = 'idx'

                fig = go.Figure()
                for c in sel:
                    s = pd.to_numeric(df_emg[c], errors='coerce').interpolate(limit_direction='both')
                    s = _norm_apply(s)
                    s = _smooth_apply(s, emg_smooth)
                    fig.add_trace(go.Scatter(x=x, y=s, mode='lines', name=c))
                fig.update_layout(
                    title=f"EMG — {selected_patient} · {selected_exercise}",
                    xaxis_title=time_col,
                    yaxis_title="EMG (normalized)",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Preview data"):
                    st.dataframe(df_emg[[time_col] + sel].head(50), use_container_width=True, hide_index=True)
else:
    with tab_emg:
        st.info("EMG tab is disabled in the sidebar.")

# =========================
# === TAB 3: STEPWISE =====
# =========================
with tab_steps:
    st.markdown("### LLM-based Stepwise Feedback")
    if not fb_csv_path or not os.path.isfile(fb_csv_path):
        st.info("No feedback CSV selected or file not found.")
    else:
        try:
            df_fb_all = pd.read_csv(fb_csv_path)
        except Exception as e:
            st.error(f"Failed to read feedback CSV: {e}")
            df_fb_all = None
        if df_fb_all is None or df_fb_all.empty:
            st.info("Feedback CSV is empty.")
        else:
            # Resolve columns (favor exact labels you provided)
            def _find_col(cols, *cands):
                m = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in cols}
                for c in cands:
                    k = "".join(ch for ch in c.lower() if ch.isalnum())
                    if k in m:
                        return m[k]
                return None

            cols = list(df_fb_all.columns)
            col_patient  = _find_col(cols, "patient_id", "patient")
            col_exercise = _find_col(cols, "exercise")
            col_grade    = _find_col(cols, "letter_grade", "grade")
            col_summary  = _find_col(cols, "exercise_summary", "summary")
            col_overall  = _find_col(cols, "overall_similarity", "similarity")

            if not col_patient or not col_exercise:
                st.warning("Could not find required columns 'patient_id' and 'exercise' in CSV.")
            else:
                mask = (df_fb_all[col_patient].astype(str).str.strip().str.lower() == str(selected_patient).strip().lower()) & \
                       (df_fb_all[col_exercise].astype(str).str.strip().str.lower() == str(selected_exercise).strip().lower())
                df_row = df_fb_all[mask]
                if df_row.empty:
                    # --- Relaxed fallback: alnum-only normalized match for both patient and exercise ---
                    def _norm(s):
                        return "".join(ch for ch in str(s).lower() if ch.isalnum())

                    ex_norm_sel = _norm(selected_exercise)
                    pat_norm_sel = _norm(selected_patient)
                    ex_norm_col = df_fb_all[col_exercise].astype(str).str.lower().apply(lambda s: "".join(ch for ch in s if ch.isalnum()))
                    pat_norm_col = df_fb_all[col_patient].astype(str).str.lower().apply(lambda s: "".join(ch for ch in s if ch.isalnum()))

                    mask_relaxed = (pat_norm_col == pat_norm_sel) & (ex_norm_col == ex_norm_sel)
                    df_row = df_fb_all[mask_relaxed]

                    if df_row.empty:
                        st.caption("No stepwise feedback found for this patient/exercise in the selected CSV.")
                        # Provide helpful hints and nearest matches
                        with st.expander("Show matching hints"):
                            st.write({
                                "selected_patient": selected_patient,
                                "selected_exercise": selected_exercise,
                                "csv_path": fb_csv_path,
                            })
                            preview_cols = [c for c in [col_patient, col_exercise, col_grade, col_overall] if c]
                            near_pat = df_fb_all[pat_norm_col == pat_norm_sel]
                            near_ex = df_fb_all[ex_norm_col == ex_norm_sel]
                            if not near_pat.empty:
                                st.markdown("**Rows with the same patient_id (normalized):**")
                                st.dataframe(near_pat[preview_cols].head(10), use_container_width=True, hide_index=True)
                            if not near_ex.empty:
                                st.markdown("**Rows with the same exercise (normalized):**")
                                st.dataframe(near_ex[preview_cols].head(10), use_container_width=True, hide_index=True)
                        st.stop()

                # If we get here, df_row is non-empty (matched strictly or relaxed)
                r = df_row.iloc[0]
                # Header card (centered, consistent with the blue card style)
                header_summary = str(r[col_summary]) if col_summary in df_row.columns else ""
                header_grade   = str(r[col_grade]) if col_grade in df_row.columns else "N/A"
                header_score   = (str(r[col_overall]) if col_overall in df_row.columns else "N/A")

                st.markdown(
                    f"""
                    <div style=\"border: 2px solid #1f77b4; border-radius: 8px; background: rgba(31,119,180,0.06); padding: 14px 16px; margin: 10px auto; max-width: 90%; text-align:center;\">
                      <div style=\"font-weight:600; color:#1f77b4; font-size:16px; margin-bottom:6px;\">LLM-based Exercise Feedback — Overview</div>
                      <div style=\"margin-bottom:8px; line-height:1.4; color:#222;\">{header_summary}</div>
                      <div style=\"display:flex; justify-content:center; gap:18px; flex-wrap:wrap; color:#222;\">
                        <div><b>Grade:</b> {header_grade}</div>
                        <div><b>Overall Similarity:</b> {header_score}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Extract dynamic stepwise fields: review_r{rep}_s{step}, similarity_r{rep}_s{step}
                import re
                review_pat = re.compile(r"^review_r(\d+)_s(\d+)$", re.IGNORECASE)
                sim_pat    = re.compile(r"^similarity_r(\d+)_s(\d+)$", re.IGNORECASE)

                # Map of rep -> { step -> {"review": str|None, "similarity": float|None} }
                reps = {}
                for c in cols:
                    c_norm = c.strip().lower()
                    m = review_pat.match(c_norm)
                    if m:
                        rep = int(m.group(1)); step = int(m.group(2))
                        reps.setdefault(rep, {}).setdefault(step, {})["review"] = r.get(c, None)
                        continue
                    m = sim_pat.match(c_norm)
                    if m:
                        rep = int(m.group(1)); step = int(m.group(2))
                        val = r.get(c, None)
                        try:
                            val = float(val) if val is not None and str(val).strip() != "" else None
                        except Exception:
                            val = None
                        reps.setdefault(rep, {}).setdefault(step, {})["similarity"] = val

                if not reps:
                    st.caption("No per-step fields (review/similarity) found in CSV header for this row.")
                else:
                    # Sort reps and steps
                    for rep in sorted(reps.keys()):
                        st.markdown(f"#### Rep {rep}")
                        steps = reps[rep]
                        for step_idx in sorted(steps.keys()):
                            data = steps[step_idx]
                            review = data.get("review")
                            simv   = data.get("similarity")
                            if (review is None or str(review).strip() == "") and (simv is None):
                                continue  # skip empty entries
                            sim_txt = (f"{simv}" if simv is not None else "N/A")
                            # Pretty card per step
                            st.markdown(
                                f"""
                                <div style=\"border:1px solid #d1e3f8; background:#f6fbff; border-radius:8px; padding:10px 12px; margin:6px 0;\">
                                  <div style=\"font-weight:600; color:#1f77b4; margin-bottom:4px;\">Step {step_idx}</div>
                                  <div style=\"color:#222; margin-bottom:6px; line-height:1.4;\">{review if review else ''}</div>
                                  <div style=\"color:#222;\"><b>Similarity:</b> {sim_txt}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

# =========================
# === TAB 4: PAIN REPORTS =
# =========================
with tab_pain:
    st.markdown("### 🩹 Pain report(s)")
    if pain_images_for_patient:
        for pimg in pain_images_for_patient:
            st.image(pimg, use_container_width=True, caption=os.path.basename(pimg))
    else:
        st.caption("No pain report image found for this patient.")
