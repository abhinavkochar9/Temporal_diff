# make_pain_maps_mediapipe_fixed_paths.py
# ------------------------------------------------------------
# Per-patient pain heatmaps using MediaPipe Pose landmarks.
# Edit ONLY the CONFIG block below for your paths & options.

import os, re, zipfile
import pandas as pd
import numpy as np
import cv2

# ===================== CONFIG (EDIT ME) ======================
EXCEL_PATH      = r"FFE_survey_patient.xlsx"
FRONT_IMG_PATH  = r"Front.png"         # front template
BACK_IMG_PATH   = r"Back.png"       # back template
OUT_DIR         = r"pain_reports"                   # will be created
ZIP_NAME        = ""                         # set to "" to skip ZIP

# Heatmap look/feel
GAUSS_SIGMA           = 12     # blur for smooth look (↑ = softer)
STAMP_RADIUS          = 8      # joint/point stamp radius (px)
CORRIDOR_WIDTH_FRAC   = 0.22   # limb corridor width as fraction of segment length
THIGH_WIDTH_FRAC      = 0.28
SHIN_WIDTH_FRAC       = 0.22

SPINE_WIDTH_FRAC      = 0.18

# Point-only heat style (natural, localized)
POINT_STAMP_RADIUS    = 7      # px, size of each pain point stamp
SHORT_LEN_FRAC        = 0.30   # fraction of a limb segment to cover for short capsules (mid-segment)
SHORT_WIDTH_FRAC      = 0.14   # width fraction for short capsules
SPINE_DISK_COUNT      = 3      # number of small disks along the spine (for code 28)
GAUSS_LOCAL_SIGMA     = 6      # per-blob gaussian softness (px)

# Arm-specific short capsule sizing (to avoid bleeding into torso)
ARM_LEN_FRAC         = 0.22
ARM_WIDTH_FRAC       = 0.10

COLORMAP              = cv2.COLORMAP_TURBO  # or JET/HOT
ALPHA_FIXED           = 0.65   # constant blending; color (yellow→red) reflects severity

# Consistent coloring & body-only overlay
USE_GLOBAL_NORMALIZATION = True   # if True, do not renormalize per image; clip to [0,1]
GLOBAL_HEAT_SCALE = 1.0           # multiply heat before clipping; keep at 1.0 for consistency
BODY_MASK_THRESH = 240            # grayscale threshold to detect the body (non-white) on the templates
HEAT_EPS = 1e-3           # only blend where heat > HEAT_EPS (prevents whole-body yellow)

# Severity → opacity (based on pain1_1 / pain3_1 text)
SEVERITY_ALPHA = {
    "none": 0.00,
    "mild": 0.35,
    "moderate": 0.55,
    "severe": 0.80,
    "very severe": 0.95
}
# =============================================================

def get_pose_landmarks(image_bgr):
    """Return list of 33 (x,y) points using MediaPipe Pose; fallback if needed."""
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError("Please `pip install mediapipe==0.10.11`") from e

    mp_pose = mp.solutions.pose
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1,
                      enable_segmentation=False, min_detection_confidence=0.5) as pose:
        res = pose.process(rgb)
        if not res.pose_landmarks:
            return None
        pts = []
        for lm in res.pose_landmarks.landmark:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pts.append((px, py))
        return pts

def fallback_template_landmarks(image):
    """Ratios tuned for clean front/back gray templates if MP fails."""
    h, w = image.shape[:2]
    P = lambda rx, ry: (int(rx*w), int(ry*h))
    Ls, Rs = P(0.405,0.245), P(0.595,0.245)
    Le, Re = P(0.43,0.435),  P(0.57,0.435)
    Lw, Rw = P(0.46,0.63),   P(0.54,0.63)
    Lh, Rh = P(0.465,0.54),  P(0.535,0.54)
    Lk, Rk = P(0.485,0.77),  P(0.515,0.77)
    La, Ra = P(0.495,0.93),  P(0.505,0.93)
    LhE,RhE= P(0.49,0.96),   P(0.51,0.96)
    Lfi,Rfi= P(0.49,0.97),   P(0.51,0.97)
    pts = [None]*33
    pts[11],pts[12]=Ls,Rs; pts[13],pts[14]=Le,Re; pts[15],pts[16]=Lw,Rw
    pts[23],pts[24]=Lh,Rh; pts[25],pts[26]=Lk,Rk; pts[27],pts[28]=La,Ra
    pts[29],pts[30]=LhE,RhE; pts[31],pts[32]=Lfi,Rfi
    return pts

def get_template_pose_or_fallback(image_bgr):
    pts = get_pose_landmarks(image_bgr)
    return pts if pts is not None else fallback_template_landmarks(image_bgr)

# ---------- Heat primitives ----------
def stamp_disk(heat, center, radius, value=1.0, sigma=None):
    """Add a Gaussian disk (not flat fill) so each blob has smooth shades.
    radius controls window size; sigma controls falloff (defaults based on radius or GAUSS_LOCAL_SIGMA).
    """
    h, w = heat.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r = int(max(2, radius))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    X, Y = np.meshgrid(xs, ys)
    sig = float(sigma if sigma is not None else max(1.0, (radius * 0.55)))
    G = np.exp(-(X*X + Y*Y) / (2.0 * sig * sig)).astype(np.float32)
    heat[y0:y1, x0:x1] += value * G

def stamp_corridor(heat, p1, p2, width_frac, value=1.0):
    p1 = np.array(p1, np.float32); p2 = np.array(p2, np.float32)
    v = p2 - p1
    L = np.linalg.norm(v) + 1e-6
    n = np.array([-v[1], v[0]], np.float32) / L
    half_w = max(3.0, (width_frac * L) / 2.0)
    q1 = (p1 + n*half_w); q2 = (p2 + n*half_w); q3 = (p2 - n*half_w); q4 = (p1 - n*half_w)
    cv2.fillConvexPoly(heat, np.int32([q1,q2,q3,q4]), value)
    cv2.circle(heat, tuple(np.int32(p1)), int(half_w), value, -1)
    cv2.circle(heat, tuple(np.int32(p2)), int(half_w), value, -1)

def stamp_polygon(heat, pts, value=1.0):
    cv2.fillConvexPoly(heat, np.array(pts, dtype=np.int32), value)

def gaussianize(heat, sigma):
    return cv2.GaussianBlur(heat, (0,0), sigma) if sigma>0 else heat

def colorize_overlay(base_bgr, heat, alpha, sev_scalar=0.55):
    """Colorize heat with a severity-aware red gradient.
    - `heat` in [0,1] encodes spatial falloff (Gaussian → hombré: dark center, lighter edge).
    - `sev_scalar` in [0,1] boosts saturation/intesity so higher severity looks deeper red.
    """
    if heat is None or np.max(heat) <= 0:
        return base_bgr

    # Spatial falloff (already Gaussian from stamps)
    t = np.clip(heat.astype(np.float32), 0.0, 1.0)

    # Boost by severity (so high severity pushes towards deeper red even at same radius)
    # Base 0.55 keeps existing look; 0.3..1.0 from severity_to_value pushes range.
    boost = 0.50 + 0.90 * float(sev_scalar)   # 0.50..1.40
    t_eff = np.clip(t * boost, 0.0, 1.0)

    # Gentle gamma → darker core, smoother edge
    t_eff = np.power(t_eff, 0.85)

    # Pure red family map (cool light red → deep red)
    # Lower GB at higher t_eff to reduce pinkness; raise R strongly with t_eff
    # R: 150..255, G/B: 110..0 gives broader perceived contrast
    R = np.uint8(np.round(150.0 + 105.0 * t_eff))
    GB = np.uint8(np.round(110.0 * (1.0 - t_eff)))
    heat_color = np.dstack((GB, GB, R))  # (B,G,R)

    return cv2.addWeighted(heat_color, float(alpha), base_bgr, 1.0 - float(alpha), 0)

# --- Body mask helpers ---
def make_body_mask(img_bgr, thresh=BODY_MASK_THRESH):
    """Return boolean mask where the body exists (True = body, False = background).
    Works for light-background, gray body templates. Adjust thresh if needed.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray < int(thresh)
    return mask

def colorize_overlay_masked(base_bgr, heat, alpha, body_mask, sev_scalar=0.55):
    """Colorize heat with fixed scaling and blend ONLY where heat>HEAT_EPS AND within body_mask."""
    heat_proc = heat.copy()
    if USE_GLOBAL_NORMALIZATION:
        heat_proc = heat_proc * float(GLOBAL_HEAT_SCALE)
        heat_proc = np.clip(heat_proc, 0.0, 1.0)
    else:
        m = float(np.max(heat_proc))
        heat_proc = (heat_proc / (m + 1e-8)) if m > 0 else heat_proc

    # Early exit if there is effectively no pain signal
    if np.max(heat_proc) <= HEAT_EPS:
        return base_bgr

    # Build blended image once, then apply only at active pain pixels
    blended = colorize_overlay(base_bgr, heat_proc, alpha, sev_scalar=sev_scalar)

    # Active pain mask: heat above epsilon AND inside body
    if body_mask.dtype != np.bool_:
        body_mask = body_mask.astype(bool)
    active = (heat_proc > HEAT_EPS) & body_mask

    out = base_bgr.copy()
    out[active] = blended[active]
    return out

# --- Localized stamp helpers (for point-only heatmaps) ---
def poly_center(poly):
    poly = np.array(poly, dtype=np.float32)
    cx = float(np.mean(poly[:,0])); cy = float(np.mean(poly[:,1]))
    return (int(round(cx)), int(round(cy)))

def stamp_mid_capsule(heat, p1, p2, len_frac=SHORT_LEN_FRAC, width_frac=SHORT_WIDTH_FRAC, value=1.0):
    p1 = np.array(p1, np.float32); p2 = np.array(p2, np.float32)
    v = p2 - p1
    L = float(np.linalg.norm(v)) + 1e-6
    if L <= 1.0:
        stamp_disk(heat, (int(p1[0]), int(p1[1])), max(2, int(POINT_STAMP_RADIUS*0.8)), value=value, sigma=GAUSS_LOCAL_SIGMA)
        return
    mid = (p1 + p2) * 0.5
    half_len = (len_frac * L) * 0.5
    dirv = v / L
    a = mid - dirv * half_len
    b = mid + dirv * half_len
    # 5 small Gaussians along the short segment
    for t in np.linspace(0.0, 1.0, 5):
        pt = a*(1.0-t) + b*t
        stamp_disk(heat, (int(pt[0]), int(pt[1])), int(POINT_STAMP_RADIUS), value=value, sigma=GAUSS_LOCAL_SIGMA)

def stamp_chain_along(heat, p1, p2, count=3, radius=POINT_STAMP_RADIUS, value=1.0):
    p1 = np.array(p1, np.float32); p2 = np.array(p2, np.float32)
    for t in np.linspace(0.0, 1.0, num=max(2, int(count))):
        pt = p1*(1.0-t) + p2*t
        stamp_disk(heat, (int(pt[0]), int(pt[1])), int(radius), value=value, sigma=GAUSS_LOCAL_SIGMA)

# ---------- Geometry helpers ----------
def L(pts, i): return pts[i] if (pts and i<len(pts) and pts[i] is not None) else None
def mid(a,b):  return (int((a[0]+b[0])//2), int((a[1]+b[1])//2))

def torso_bands(pts):
    Ls, Rs = L(pts,11), L(pts,12)
    Lh, Rh = L(pts,23), L(pts,24)
    neck   = mid(Ls, Rs); pelvis = mid(Lh, Rh)
    upper_y = int(neck[1] + (pelvis[1]-neck[1])*0.33)
    mid_y   = int(neck[1] + (pelvis[1]-neck[1])*0.66)
    chest_L = [(Ls[0], neck[1]), (neck[0], neck[1]), (neck[0], upper_y), (Ls[0], upper_y)]
    chest_R = [(neck[0], neck[1]), (Rs[0], neck[1]), (Rs[0], upper_y), (neck[0], upper_y)]
    upabd_L = [(Ls[0], upper_y), (neck[0], upper_y), (neck[0], mid_y), (Lh[0], mid_y)]
    upabd_R = [(neck[0], upper_y), (Rs[0], upper_y), (Rh[0], mid_y), (neck[0], mid_y)]
    pelvis_L= [(Lh[0], mid_y), (neck[0], mid_y), (neck[0], Lh[1]), (Lh[0], Lh[1])]
    pelvis_R= [(neck[0], mid_y), (Rh[0], mid_y), (Rh[0], Rh[1]), (neck[0], Rh[1])]
    return dict(chest_L=chest_L, chest_R=chest_R, upabd_L=upabd_L,
                upabd_R=upabd_R, pelvis_L=pelvis_L, pelvis_R=pelvis_R), neck, pelvis

# ---------- Code→region stamping ----------
def add_region_heat(heat, code, pts, value=1.0):
    polys, neck, pelvis = torso_bands(pts)
    Ls, Rs = L(pts,11), L(pts,12); Le, Re = L(pts,13), L(pts,14)
    Lw, Rw = L(pts,15), L(pts,16); Lh, Rh = L(pts,23), L(pts,24)
    Lk, Rk = L(pts,25), L(pts,26); La, Ra = L(pts,27), L(pts,28)
    LhE,RhE= L(pts,29), L(pts,30); Lfi,Rfi= L(pts,31), L(pts,32)

    # centers for torso rectangles
    chest_L_c = poly_center(polys['chest_L'])
    chest_R_c = poly_center(polys['chest_R'])
    upabd_L_c = poly_center(polys['upabd_L'])
    upabd_R_c = poly_center(polys['upabd_R'])
    pelvis_L_c= poly_center(polys['pelvis_L'])
    pelvis_R_c= poly_center(polys['pelvis_R'])

    # FRONT (1–23)
    if code == 1 and neck and Rs:  stamp_mid_capsule(heat, neck, Rs, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 2 and Rs and Re:  stamp_mid_capsule(heat, Rs, Re, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 3 and Re and Rw:  stamp_mid_capsule(heat, Re, Rw, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 4 and Rw:         stamp_disk(heat, Rw, STAMP_RADIUS, value=value)
    elif code == 5:                stamp_disk(heat, mid(chest_L_c, chest_R_c), POINT_STAMP_RADIUS, value=value)
    elif code == 6 and neck and Ls: stamp_mid_capsule(heat, neck, Ls, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 7 and Ls and Le:   stamp_mid_capsule(heat, Ls, Le, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 8 and Le and Lw:   stamp_mid_capsule(heat, Le, Lw, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 9 and Lw:          stamp_disk(heat, Lw, STAMP_RADIUS, value=value)
    elif code == 10:                stamp_disk(heat, chest_R_c, POINT_STAMP_RADIUS, value=value)
    elif code == 11:                stamp_disk(heat, chest_L_c, POINT_STAMP_RADIUS, value=value)
    elif code == 12:                stamp_disk(heat, upabd_R_c, POINT_STAMP_RADIUS, value=value)
    elif code == 13:                stamp_disk(heat, upabd_L_c, POINT_STAMP_RADIUS, value=value)
    elif code == 14:                stamp_disk(heat, pelvis_R_c, POINT_STAMP_RADIUS, value=value)
    elif code == 15:                stamp_disk(heat, pelvis_L_c, POINT_STAMP_RADIUS, value=value)
    elif code == 16 and Rh and Rk:  stamp_mid_capsule(heat, Rh, Rk, len_frac=SHORT_LEN_FRAC, width_frac=SHORT_WIDTH_FRAC, value=value)
    elif code == 17 and Lh and Lk:  stamp_mid_capsule(heat, Lh, Lk, len_frac=SHORT_LEN_FRAC, width_frac=SHORT_WIDTH_FRAC, value=value)
    elif code == 18 and Rk:         stamp_disk(heat, Rk, int(STAMP_RADIUS*1.2), value=value)
    elif code == 19 and Lk:         stamp_disk(heat, Lk, int(STAMP_RADIUS*1.2), value=value)
    elif code == 20 and Rk and Ra:  stamp_mid_capsule(heat, Rk, Ra, len_frac=SHORT_LEN_FRAC, width_frac=SHORT_WIDTH_FRAC, value=value)
    elif code == 21 and Lk and La:  stamp_mid_capsule(heat, Lk, La, len_frac=SHORT_LEN_FRAC, width_frac=SHORT_WIDTH_FRAC, value=value)
    elif code == 22 and Rfi:        stamp_disk(heat, Rfi, STAMP_RADIUS, value=value)
    elif code == 23 and Lfi:        stamp_disk(heat, Lfi, STAMP_RADIUS, value=value)

    # BACK (24–46)
    elif code == 24 and neck and Ls: stamp_mid_capsule(heat, neck, Ls, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 28 and neck and pelvis: stamp_chain_along(heat, neck, pelvis, count=SPINE_DISK_COUNT, radius=POINT_STAMP_RADIUS, value=value)
    elif code == 29 and neck and Rs: stamp_mid_capsule(heat, neck, Rs, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 25 and Ls and Le:   stamp_mid_capsule(heat, Ls, Le, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 26 and Le and Lw:   stamp_mid_capsule(heat, Le, Lw, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 27 and Lw:          stamp_disk(heat, Lw, STAMP_RADIUS, value=value)
    elif code == 30 and Rs and Re:   stamp_mid_capsule(heat, Rs, Re, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 31 and Re and Rw:   stamp_mid_capsule(heat, Re, Rw, len_frac=ARM_LEN_FRAC, width_frac=ARM_WIDTH_FRAC, value=value)
    elif code == 32 and Rw:          stamp_disk(heat, Rw, STAMP_RADIUS, value=value)
    elif code == 33:                 stamp_disk(heat, chest_L_c, POINT_STAMP_RADIUS, value=value)
    elif code == 34:                 stamp_disk(heat, chest_R_c, POINT_STAMP_RADIUS, value=value)
    elif code == 35:                 stamp_disk(heat, upabd_L_c, POINT_STAMP_RADIUS, value=value)
    elif code == 36:                 stamp_disk(heat, upabd_R_c, POINT_STAMP_RADIUS, value=value)
    elif code == 37:                 stamp_disk(heat, pelvis_L_c, POINT_STAMP_RADIUS, value=value)
    elif code == 38:                 stamp_disk(heat, pelvis_R_c, POINT_STAMP_RADIUS, value=value)
    elif code == 39 and Lh and Lk:   stamp_mid_capsule(heat, Lh, Lk, len_frac=SHORT_LEN_FRAC, width_frac=SHORT_WIDTH_FRAC, value=value)
    elif code == 40 and Rh and Rk:   stamp_mid_capsule(heat, Rh, Rk, len_frac=SHORT_LEN_FRAC, width_frac=SHORT_WIDTH_FRAC, value=value)
    elif code == 41 and Lk:          stamp_disk(heat, Lk, int(STAMP_RADIUS*1.2), value=value)
    elif code == 42 and Rk:          stamp_disk(heat, Rk, int(STAMP_RADIUS*1.2), value=value)
    elif code == 43 and Lk and La:   stamp_mid_capsule(heat, Lk, La, len_frac=SHORT_LEN_FRAC, width_frac=SHORT_WIDTH_FRAC, value=value)
    elif code == 44 and Rk and Ra:   stamp_mid_capsule(heat, Rk, Ra, len_frac=SHORT_LEN_FRAC, width_frac=SHORT_WIDTH_FRAC, value=value)
    elif code == 45 and LhE:         stamp_disk(heat, LhE, STAMP_RADIUS, value=value)
    elif code == 46 and RhE:         stamp_disk(heat, RhE, STAMP_RADIUS, value=value)
# Map severity text to a value in [0,1] so Yellow(low)→Red(high) reflects severity
def severity_to_value(pain3_1=None):
    txt = str(pain3_1).strip().lower() if pd.notna(pain3_1) else ""
    # Map to [0,1] so Yellow(low)→Red(high) reflects severity
    if "very severe" in txt: return 1.0
    if "severe" in txt:      return 0.85
    if "moderate" in txt:    return 0.55
    if "mild" in txt:        return 0.30
    if "none" in txt or "no pain" in txt: return 0.0
    return 0.55

# ---------- Excel parsing ----------
def severity_to_alpha(pain1_1=None, pain3_1=None):
    txt = ""
    for t in (pain1_1, pain3_1):
        if pd.notna(t): txt += " " + str(t).strip().lower()
    if "very severe" in txt: return SEVERITY_ALPHA["very severe"]
    if "severe" in txt:      return SEVERITY_ALPHA["severe"]
    if "moderate" in txt:    return SEVERITY_ALPHA["moderate"]
    if "mild" in txt:        return SEVERITY_ALPHA["mild"]
    if "none" in txt or "no pain" in txt: return SEVERITY_ALPHA["none"]
    return 0.55

def parse_codes(raw):
    if pd.isna(raw): return []
    return [int(x) for x in re.findall(r"\d+", str(raw)) if 1 <= int(x) <= 46]

def annotate_title_strip(canvas, title):
    h, w = canvas.shape[:2]
    strip_h = max(40, h//18)
    strip = np.full((strip_h, w, 3), 255, dtype=np.uint8)
    cv2.putText(strip, title, (10, strip_h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
    return np.vstack([strip, canvas])

# ---------- Main ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_excel(EXCEL_PATH, sheet_name=0)[['QID2','pain3_1','pain4']].copy()

    front = cv2.imread(FRONT_IMG_PATH, cv2.IMREAD_COLOR)
    back  = cv2.imread(BACK_IMG_PATH,  cv2.IMREAD_COLOR)
    if front is None: raise FileNotFoundError(FRONT_IMG_PATH)
    if back  is None: raise FileNotFoundError(BACK_IMG_PATH)

    # Build body masks so the overlay only affects the body area
    front_body_mask = make_body_mask(front, BODY_MASK_THRESH)
    back_body_mask  = make_body_mask(back,  BODY_MASK_THRESH)

    front_pts = get_template_pose_or_fallback(front)
    back_pts  = get_template_pose_or_fallback(back)

    saved = []

    for i, row in df.iterrows():
        pid = str(row['QID2']) if pd.notna(row['QID2']) else f"row{i:03d}"
        codes = parse_codes(row['pain4'])
        if not codes: 
            continue

        # Compute severity value for coloring
        sev_val = severity_to_value(row.get('pain3_1'))

        # FRONT
        front_heat = np.zeros(front.shape[:2], dtype=np.float32)
        for c in [c for c in codes if 1 <= c <= 23]:
            add_region_heat(front_heat, c, front_pts, value=sev_val)
        front_heat = gaussianize(front_heat, GAUSS_SIGMA)
        front_heat *= 1.0  # keep 1.0; increase to boost all heats uniformly
        # Clip heat to [0,1] for consistent scaling before colorizing
        front_heat = np.clip(front_heat, 0.0, 1.0)
        front_overlay = colorize_overlay_masked(front.copy(), front_heat, alpha=ALPHA_FIXED, body_mask=front_body_mask, sev_scalar=sev_val)

        # BACK
        back_heat = np.zeros(back.shape[:2], dtype=np.float32)
        for c in [c for c in codes if 24 <= c <= 46]:
            add_region_heat(back_heat, c, back_pts, value=sev_val)
        back_heat = gaussianize(back_heat, GAUSS_SIGMA)
        back_heat *= 1.0
        back_heat = np.clip(back_heat, 0.0, 1.0)
        back_overlay = colorize_overlay_masked(back.copy(), back_heat, alpha=ALPHA_FIXED, body_mask=back_body_mask, sev_scalar=sev_val)

        # side-by-side
        h = max(front_overlay.shape[0], back_overlay.shape[0])
        f_res = cv2.resize(front_overlay, (int(front_overlay.shape[1]*h/front_overlay.shape[0]), h))
        b_res = cv2.resize(back_overlay,  (int(back_overlay.shape[1]*h/back_overlay.shape[0]),  h))
        canvas = np.full((h, f_res.shape[1] + b_res.shape[1], 3), 255, dtype=np.uint8)
        canvas[:, :f_res.shape[1]] = f_res
        canvas[:, f_res.shape[1]:] = b_res

        title = f"Patient: {pid} | severity(pain3_1): {row.get('pain3_1')} | codes: {codes}"
        final_img = annotate_title_strip(canvas, title)
        out_path  = os.path.join(OUT_DIR, f"{pid}_painmap.png")
        cv2.imwrite(out_path, final_img)
        saved.append(out_path)

    if ZIP_NAME.strip() and saved:
        zip_path = os.path.join(OUT_DIR, ZIP_NAME)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in saved:
                zf.write(f, arcname=os.path.basename(f))

    print(f"Done. Saved {len(saved)} images to {OUT_DIR}.",
          f'ZIP: {os.path.join(OUT_DIR, ZIP_NAME)}' if ZIP_NAME.strip() else "")

if __name__ == "__main__":
    # deps: pip install mediapipe==0.10.11 opencv-python pandas openpyxl numpy
    main()