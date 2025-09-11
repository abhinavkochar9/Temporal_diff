"""
Highlight shoulders, elbows, wrists with bigger translucent dots (no labels).
- Input dir: Step_Images/WHAM/<Exercise>/<StepImages>
- Output (in-place): Step_Images/WHAM/<Exercise>/<StepName>/
      ├── <StepName>__original.png
      └── error_combinations/
            <StepName>__LShoulder.png
            <StepName>__LShoulder+RShoulder.png
            ...
"""

from pathlib import Path
import itertools
import cv2
import mediapipe as mp
import re

# ---------- CONFIG ----------
# Process all exercises inside WHAM
BASE_STEPS_DIR = Path("Step_Images") / "WHAM"
DOT_RADIUS = 18                # bigger dots
DOT_ALPHA  = 0.45              # 0..1, transparency (higher = more opaque)
MIN_VIS    = 0.25           # lower threshold so fewer steps are skipped
COLOR_BGR  = (0, 0, 255)       # red
GENERATE_PERMUTATIONS = False  # keep False unless you truly need permutations

mp_pose = mp.solutions.pose

JOINTS = {
    "LShoulder": 11, "RShoulder": 12,
    "LElbow":    13, "RElbow":    14,
    "LWrist":    15, "RWrist":    16,
}
JOINT_KEYS = list(JOINTS.keys())

_num_re = re.compile(r'(\d+)')
def _natural_key(s: str):
    s = str(s)
    return [int(t) if t.isdigit() else t.lower() for t in _num_re.split(s)]


def read_images(indir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(indir.iterdir(), key=lambda x: _natural_key(x.name)):
        if p.suffix.lower() in exts and p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                yield p, img
            else:
                print(f"  [WARN] Could not read image: {p.name}")


def iter_exercise_dirs(base: Path):
    """
    Yield exercise subdirectories under `base` that contain at least one image file.
    Example structure:
      Step_Images/WHAM/
        ClaspandSpread/
        DeepBreathing/
        HorizontalPumping/
        OverheadPumping/
        PushdownPumping/
        ShoulderRolls/
    """
    if not base.exists():
        return
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for d in sorted(base.iterdir()):
        if d.is_dir():
            try:
                has_img = any((p.suffix.lower() in exts) for p in d.iterdir() if p.is_file())
            except FileNotFoundError:
                has_img = False
            if has_img:
                yield d


def detect_points(img_bgr, pose):
    h, w = img_bgr.shape[:2]
    res = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if not res.pose_landmarks:
        return {}

    pts = {}
    lm = res.pose_landmarks.landmark
    for name, idx in JOINTS.items():
        l = lm[idx]
        if (l.visibility or 0) < MIN_VIS:
            continue
        pts[name] = (int(round(l.x * w)), int(round(l.y * h)))
    return pts


def draw_subset_translucent(base_img, subset_names, pts):
    """
    Draw translucent filled circles for the subset on a copy of the image.
    No labels are added.
    """
    overlay = base_img.copy()
    for name in subset_names:
        if name in pts:
            cv2.circle(overlay, pts[name], DOT_RADIUS, COLOR_BGR, -1, lineType=cv2.LINE_AA)
    # blend once for all dots (crisper than per-dot blending)
    out = base_img.copy()
    cv2.addWeighted(overlay, DOT_ALPHA, out, 1 - DOT_ALPHA, 0, out)
    return out


def safe(name: str) -> str:
    return name.replace(" ", "").replace("/", "_").replace("+", "_").replace("-", "_").replace(",", "_")


def main():
    if not BASE_STEPS_DIR.exists():
        raise SystemExit(f"[ERROR] Base steps dir not found: {BASE_STEPS_DIR} (did you mean to create Step_Images/WHAM/… ?)")

    total_images = 0
    total_combos = 0
    total_skipped = 0

    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=2,
                      enable_segmentation=False,
                      min_detection_confidence=0.5) as pose:
        for ex_dir in iter_exercise_dirs(BASE_STEPS_DIR):
            print(f"\n[EXERCISE] {ex_dir}")
            for img_path, img in read_images(ex_dir):
                total_images += 1
                out_dir = ex_dir / img_path.stem
                comb_dir = out_dir / "error_combinations"
                out_dir.mkdir(parents=True, exist_ok=True)
                comb_dir.mkdir(parents=True, exist_ok=True)

                ok_write = cv2.imwrite(str(out_dir / f"{img_path.stem}__original.png"), img)
                if not ok_write:
                    print(f"  [WARN] Failed to write original for {img_path.name} to {out_dir}")
                else:
                    print(f"  [SAVE] {out_dir / f'{img_path.stem}__original.png'}")

                pts = detect_points(img, pose)
                if not pts:
                    print(f"  [WARN] No pose landmarks detected in {img_path.name} (visibility>={MIN_VIS})")
                    total_skipped += 1
                    continue

                # ---- combinations (63) ----
                count_comb = 0
                for r in range(1, len(JOINT_KEYS) + 1):
                    for combo in itertools.combinations(JOINT_KEYS, r):
                        out = draw_subset_translucent(img, combo, pts)
                        fname = f"{img_path.stem}__{safe('+'.join(combo))}.png"
                        dest = comb_dir / fname
                        ok = cv2.imwrite(str(dest), out)
                        if ok:
                            count_comb += 1
                        else:
                            print(f"  [WARN] Failed to write {dest}")
                if count_comb != 63:
                    print(f"  [NOTE] Saved {count_comb} combinations (expected 63) for {img_path.name}")

                # ---- permutations (optional) ----
                if GENERATE_PERMUTATIONS:
                    perm_dir = comb_dir / "permutations"
                    perm_dir.mkdir(exist_ok=True)
                    count_perm = 0
                    for r in range(1, len(JOINT_KEYS) + 1):
                        for perm in itertools.permutations(JOINT_KEYS, r):
                            out = draw_subset_translucent(img, perm, pts)
                            fname = f"{img_path.stem}__{safe('-'.join(perm))}.png"
                            ok = cv2.imwrite(str(perm_dir / fname), out)
                            if ok:
                                count_perm += 1
                    print(f"  [{img_path.name}] saved {count_comb} combinations and {count_perm} permutations")
                else:
                    print(f"  [{img_path.name}] saved {count_comb} combinations")
                total_combos += count_comb

    print(f"\n[SUMMARY] images processed: {total_images}, combinations saved: {total_combos}, no-landmark skips: {total_skipped}")
    print("[INFO] Outputs saved next to source images under each step folder (see 'error_combinations').")
    print("[TIP] Open one step folder and confirm you see 1 original + 63 overlays. If not, share the console log for that step.")


if __name__ == "__main__":
    main()

#updates