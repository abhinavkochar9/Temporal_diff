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
import pathlib
_num_re = re.compile(r'(\d+)')
def _natural_key(s: str):
    s = os.path.basename(str(s))
    return [int(t) if t.isdigit() else t.lower() for t in _num_re.split(s)]

# --- Normalized name helpers for robust directory matching ---
def _norm_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

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

# =========================
# ====== CONFIG ===========
# =========================
PATIENT_DIR = "Patient_Temporal"
EXPERT_DIR  = "Expert_Temporal"
JOINT_ORDER = ["Right_Shoulder", "Left_Shoulder", "Right_Elbow", "Left_Elbow", "Right_Wrist", "Left_Wrist"]

st.set_page_config(layout="wide")
st.title("🧠 Patient vs Expert — Keypoint Comparison (aggressive patient despike)")

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
                                          expert_strip_img=None):
    """
    Build a composite header:
      [Expert steps rep1+rep2]  (top row)
      [Patient overlays rep1 | rep2]  (middle row)
      [timeline]  (bottom)
    Returns a PIL.Image or None.
    """
    if not step_paths:
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
    left_pad = 18
    right_pad = 18
    total_h = H1 + H2 + timeline_h
    canvas = Image.new("RGB", (W + left_pad + right_pad, total_h), bg)

    # Paste rows
    y = 0
    canvas.paste(expert_strip, (left_pad, y)); y += H1
    canvas.paste(patient_combined, (left_pad, y)); y += H2

    # Draw timeline along the bottom width W
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    y0 = y + 10  # baseline
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
patient_exercises = list_dirs(os.path.join(PATIENT_DIR, selected_patient))
selected_exercise = st.sidebar.selectbox("💪 Exercise", patient_exercises)

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

# normalize inputs
if jump_abs == 0.0:
    jump_abs_val = None
else:
    jump_abs_val = float(jump_abs)
jump_pct_val = float(jump_pct)

base_dir = os.path.dirname(os.path.abspath(__file__))
ex_dir, step_paths, searched_step_dirs = _resolve_step_images(base_dir, selected_exercise)
est_frames = _first_available_frames_count(PATIENT_DIR, EXPERT_DIR, selected_patient, selected_exercise)
fps_val = 25.0
# --- New: resolve Expert and WHAM dirs explicitly ---
expert_dir = _resolve_ex_dir_by_subfolder(base_dir, selected_exercise, "Expert")
wham_dir   = _resolve_ex_dir_by_subfolder(base_dir, selected_exercise, "WHAM")

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

            # --- NEW: Build strips using explicit Expert/WHAM dirs and any-order overlay resolver ---
            # Determine expert originals (top row): MUST come from Step_Images/Expert
            expert_items = _list_step_folders_with_originals(expert_dir) if expert_dir else []
            expert_strip = _build_strip_from_originals(expert_items, repeat=2, target_h=160, pad=8)

            # Build patient overlay strips specifically from WHAM dir (as requested)
            patient_source_dir = wham_dir if wham_dir else ex_dir  # fallback to resolved dir if WHAM missing
            rep1_strip, rep2_strip = _build_patient_strip_by_steps(patient_source_dir, step_paths, rep_subsets,
                                                                   repeat=1, target_h=160, pad=8)

            st.subheader(f"Expert & Patient Steps — {selected_exercise}")
            composite = _compose_expert_patient_with_timeline(step_paths, rep1_strip, rep2_strip,
                                                              fps=fps_val, total_frames=est_frames,
                                                              target_h=160, pad=8, expert_strip_img=expert_strip)
            if composite is not None:
                st.image(
                    composite,
                    use_container_width=True,
                    caption=f"Top: Expert originals (2 reps) from Expert  •  Middle: Patient overlays from WHAM (Rep1 | Rep2)  •  Timeline: {fps_val:g} fps{f' • ~{est_frames/fps_val:.1f}s' if est_frames else ''}"
                )
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
tab_compare, tab_delta = st.tabs(["📊 Compare (Patient vs Expert)", "➖ Delta (Patient − Expert)"])

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
    st.markdown("#### Δ (Patient − Expert) per Joint and Axis")
    fig_d = make_subplots(
        rows=len(JOINT_ORDER),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{j.replace('_', ' ')} ΔX & ΔY" for j in JOINT_ORDER],
        vertical_spacing=0.06,
    )

    metrics_rows = []
    all_dx_dy = []

    for i, joint in enumerate(JOINT_ORDER, start=1):
        p_df = patient_dfs[joint]
        e_df = expert_dfs[joint]

        if p_df is None or e_df is None:
            fig_d.add_trace(go.Scatter(y=[]), row=i, col=1)
            metrics_rows.append({"Joint": joint, "MAE X": np.nan, "RMSE X": np.nan, "Corr X": np.nan,
                                 "MAE Y": np.nan, "RMSE Y": np.nan, "Corr Y": np.nan})
            continue

        n = min(len(p_df), len(e_df)) if sync_length else max(len(p_df), len(e_df))
        p_df = p_df.iloc[:n].reset_index(drop=True)
        e_df = e_df.iloc[:n].reset_index(drop=True)

        dx = p_df["X"] - e_df["X"]
        dy = p_df["Y"] - e_df["Y"]

        fig_d.add_trace(go.Scatter(y=dx, mode="lines", name=f"{joint} ΔX",
                                   line=dict(width=2, dash="solid"),
                                   hovertemplate="ΔX: %{y:.4f}<extra></extra>"), row=i, col=1)
        fig_d.add_trace(go.Scatter(y=dy, mode="lines", name=f"{joint} ΔY",
                                   line=dict(width=2, dash="dash"),
                                   hovertemplate="ΔY: %{y:.4f}<extra></extra>"), row=i, col=1)

        mae_x, rmse_x, corr_x = quick_metrics(p_df["X"], e_df["X"])
        mae_y, rmse_y, corr_y = quick_metrics(p_df["Y"], e_df["Y"])
        metrics_rows.append({"Joint": joint,
                             "MAE X": round(mae_x, 4), "RMSE X": round(rmse_x, 4),
                             "Corr X": round(corr_x, 4) if not np.isnan(corr_x) else np.nan,
                             "MAE Y": round(mae_y, 4), "RMSE Y": round(rmse_y, 4),
                             "Corr Y": round(corr_y, 4) if not np.isnan(corr_y) else np.nan})

        all_dx_dy.append(dx.values); all_dx_dy.append(dy.values)

    if all_dx_dy:
        stacked = np.concatenate(all_dx_dy)
        dmin, dmax = stacked.min(), stacked.max()
        pad = (dmax - dmin) * 0.05 if dmax > dmin else 0.1
        d_range = [dmin - pad, dmax + pad]
    else:
        d_range = [-1, 1]

    for i in range(1, len(JOINT_ORDER) + 1):
        fig_d.update_yaxes(range=d_range, title_text="Δ", row=i, col=1)
    fig_d.update_layout(height=2000,
                        title_text=f"Deltas — {selected_patient} · {selected_exercise}",
                        showlegend=show_legend,
                        margin=dict(l=60, r=20, t=60, b=40))
    fig_d.update_xaxes(title_text="Frame Index", row=len(JOINT_ORDER), col=1)
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