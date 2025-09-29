#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Report Generator (Gemini) — Step-wise Feedback + Exercise Summary

What this does
--------------
- Reads 6 exercise reference TXT files (step descriptions, cues).
- Ingests a single CSV containing per-patient, per-exercise, per-step inputs.
- Calls Gemini to produce:
  (a) step-wise feedback ("good/bad + why + fix") and
  (b) a concise exercise-level summary.
- Preserves and outputs repetition numbers per step (e.g., rep 2, step 1).
- Writes per-patient/per-exercise JSON reports + one consolidated CSV.

Input CSV schema (wide-open; minimally these columns)
-----------------------------------------------------
patient_id, exercise, step_index, step_input
- patient_id: e.g., TOLF-B01
- exercise: one of ["Clasp_and_Spread","Deep_Breathing","Horizontal_Pumping",
                    "Overhead_Pumping","Push_Down_Pumping","Shoulder_Roll"]
  (names can be flexible; a mapping is provided below)
- step_index: integer step number (1..N) for that exercise instance
- step_input: freeform text OR serialized metrics you want the LLM to consider
  Examples: "Right elbow under-extended ~12°, wrist 8% lower than baseline"
            or a JSON-like blob string from your pipeline

Usage
-----
python llm_report_generator.py \
  --api-key "YOUR_GEMINI_KEY" \
  --input-csv /path/to/all_patients_steps.csv \
  --exercises-dir /path/to/exercise_txts \
  --out-dir /path/to/output_reports \
  --model "gemini-1.5-flash-latest"

You can also set the key via env var: GEMINI_API_KEY.
If both are set, --api-key takes precedence.

Outputs
-------
- out_dir/summary/final_llm_feedback.csv  (one row per patient×exercise; fixed grid of review/sim columns for rep1/rep2 × steps1..5, plus letter_grade and exercise_summary)

Notes
-----
- Includes simple caching keyed by (patient_id, exercise, step_index, step_input hash)
- Retries + exponential backoff for rate limits (429) and transient errors.
- Deterministic prompts so results are reproducible (temperature=0.2 by default).
"""

import os
import re
import json
import time
import logging
# =========================
# ===== USER CONFIG =======
# =========================
CONFIG = {
    # Paste your Gemini API key here
    "API_KEY": "",
    "API_KEYS": [
        "AIzaSyBnYI9xav-ciNcKtHmJBAtWX4akGlr_3_U",
        "AIzaSyC_fjcXTRoDVKyO0ai5mnR13XQZegAEumA",
        "AIzaSyCiRXA-KaDqV5gca4_dbzMkBqeCxZ_lZA4",
        "AIzaSyABKNhTd7Ps8soqRImlRv61lYDWkhPwthg"
    ],

    # Folder that contains 30 CSV files (one per patient)
    "CSV_INPUT_DIR": "outputs/batch_reports/stepwise_summaries",

    # If set, process exactly this one CSV (typical: a single patient's data)
    "SINGLE_PATIENT_CSV": "",

    # Folder that contains the 6 exercise TXT files
    "EXERCISES_DIR": "Exercise_Descriptions",

    # Output folder (final CSV will be placed under /summary)
    "OUT_DIR": "outputs/batch_reports/llm_reports5",

    # Gemini model name
    "MODEL": "gemini-2.5-flash-lite",

    # Similarity mapping parameter (mean MAE that maps to 0% similarity)
    "SIMILARITY_MAX_MAE": 10.0,

    # Always write the detailed per-step CSV
    "WRITE_DETAILED_CSV": True,

    # Auto-tune similarity scaling from your dataset (overrides SIMILARITY_MAX_MAE when True)
    "AUTO_TUNE_SIMILARITY": True,
    # Percentile of per-step mean MAE to use as the 0% similarity anchor when auto-tuning
    "SIMILARITY_PCTL": 0.95,

    # Balance similarity so "good" ≈ >90 and "bad" ≈ 60s
    "BALANCE_SIMILARITY": True,
    # Quantiles to anchor MAE→similarity mapping (low→~95, high→~60)
    "SIM_PCTL_LOW": 0.20,
    "SIM_PCTL_HIGH": 0.80,
    # Target scores for anchors
    "SIM_SCORE_HIGH": 95.0,
    "SIM_SCORE_LOW": 60.0,
}
from typing import Dict, List, Any, Tuple

# ----------------------------
# API Key Rotator and Model Factory
# ----------------------------
class _KeyRotator:
    def __init__(self, keys: List[str]):
        self.keys = [k.strip() for k in keys if k and str(k).strip()]
        self.i = 0
    def current(self) -> str:
        if not self.keys:
            return ""
        return self.keys[self.i % len(self.keys)]
    def advance(self):
        if self.keys:
            self.i = (self.i + 1) % len(self.keys)

def _make_model_for_key(api_key: str, model_name: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
        }
    )
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from statistics import mean

import pandas as pd

# Optional: install google-generativeai with: pip install google-generativeai
try:
    import google.generativeai as genai
except Exception as e:
    genai = None

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=getattr(logging, os.getenv("LLM_REPORT_LOGLEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ----------------------------
# Constants / Mappings
# ----------------------------
# Normalized exercise name mapping (case/spacing robust)
# We normalize any incoming name by lowercasing and removing non-letters.
NORM_EXERCISE_MAP = {
    "claspandspread": "Clasp_and_Spread",
    "deepbreathing": "Deep_Breathing",
    "horizontalpumping": "Horizontal_Pumping",
    "overheadpumping": "Overhead_Pumping",
    "pushdownpumping": "Push_Down_Pumping",
    "shoulderroll": "Shoulder_Roll",
}

def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())

DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_MAX_OUTPUT_TOKENS = 2048

# Similarity calculation defaults (percentage derived from MAE across keypoints)
DEFAULT_MAX_MAE_FOR_ZERO_SIM = 20.0  # if mean MAE >= this, similarity -> 0%
MAE_PREFIX = "mae_"  # any column starting with this is treated as a keypoint MAE

# ----------------------------
# Data classes
# ----------------------------
@dataclass
class StepFeedback:
    step_index: int
    verdict: str          # "good" or "needs_improvement"
    feedback: str
    similarity_percent: float | None = None
    rep: int | None = None

@dataclass
class ExerciseReport:
    patient_id: str
    exercise: str
    step_feedback: List[StepFeedback]
    exercise_summary: str
    letter_grade: str = ""

# ----------------------------
# Helpers
# ----------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _extract_maes_from_row(row: pd.Series) -> list[float]:
    """Collect MAE values for this step.
    Priority 1: any numeric columns starting with MAE_PREFIX.
    Priority 2: parse JSON-like payload inside step_input and look for a top-level
                key 'mae' (list of numbers) or 'mae_by_kp' (dict of numbers).
    Returns a list of floats (possibly empty)."""
    maes = []
    # Priority 1: columns named mae_*
    for col in row.index:
        if isinstance(col, str) and col.startswith(MAE_PREFIX):
            try:
                raw = row[col]
                if isinstance(raw, str) and raw.strip().lower() in {"", "nan", "none", "null"}:
                    continue
                v = float(raw)
                if not pd.isna(v):
                    maes.append(v)
            except Exception:
                continue
    if maes:
        return maes
    # Priority 2: parse within step_input JSON/text
    payload = str(row.get("step_input", "")).strip()
    try:
        # tolerate single quotes
        payload_json = json.loads(payload.replace("'", '"')) if payload else None
        if isinstance(payload_json, dict):
            if "mae" in payload_json and isinstance(payload_json["mae"], (list, tuple)):
                for v in payload_json["mae"]:
                    try:
                        if isinstance(v, str) and v.strip().lower() in {"", "nan", "none", "null"}:
                            continue
                        fv = float(v)
                        if not pd.isna(fv):
                            maes.append(fv)
                    except Exception:
                        continue
            elif "mae_by_kp" in payload_json and isinstance(payload_json["mae_by_kp"], dict):
                for v in payload_json["mae_by_kp"].values():
                    try:
                        if isinstance(v, str) and v.strip().lower() in {"", "nan", "none", "null"}:
                            continue
                        fv = float(v)
                        if not pd.isna(fv):
                            maes.append(fv)
                    except Exception:
                        continue
    except Exception:
        pass
    return maes
_REP_RE = re.compile(r"(\d+)")

def _parse_rep(val) -> int:
    """Coerce rep from diverse formats: 1, 1.0, '1', 'R1', 'rep 2', etc. Defaults to 1."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 1
        if isinstance(val, (int,)):
            return int(val)
        if isinstance(val, float):
            return int(round(val))
        s = str(val).strip()
        m = _REP_RE.search(s)
        if m:
            return int(m.group(1))
        return 1
    except Exception:
        return 1

# ----------------------------
# Balanced similarity helpers
# ----------------------------

def _compute_mae_quantiles(df: pd.DataFrame, q_low: float, q_high: float, fallback: Tuple[float, float] = (5.0, 15.0)) -> Tuple[float, float]:
    vals = []
    for _, row in df.iterrows():
        maes = _extract_maes_from_row(row)
        if maes:
            try:
                vals.append(float(mean(maes)))
            except Exception:
                continue
    if not vals:
        return fallback
    try:
        import numpy as np
        arr = np.array(vals, dtype=float)
        lo = float(np.quantile(arr, q_low))
        hi = float(np.quantile(arr, q_high))
        if hi <= lo:  # avoid degenerate anchors
            return fallback
        return lo, hi
    except Exception:
        return fallback

def _similarity_balanced_from_mae(avg_mae: float, mae_lo: float, mae_hi: float, score_hi: float, score_lo: float) -> float:
    """Continuous linear mapping using [mae_lo, mae_hi] → [score_hi, score_lo].
    Allows gentle extrapolation beyond the anchors; final value is clipped to [0, 100].
    This avoids a hard ceiling at score_hi (e.g., 95): very small MAE can yield >95 up to 100.
    """
    # Guard against degenerate anchors
    if mae_hi == mae_lo:
        return float(max(0.0, min(100.0, score_hi)))
    slope = (score_lo - score_hi) / (mae_hi - mae_lo)
    sim = score_hi + slope * (avg_mae - mae_lo)
    # Clip only to valid percentage bounds
    sim = max(0.0, min(100.0, sim))
    return float(sim)

def compute_similarity_percent(row: pd.Series, max_mae: float) -> float | None:
    maes = _extract_maes_from_row(row)
    if not maes:
        return None
    try:
        avg_mae = float(mean(maes))
        if bool(CONFIG.get("BALANCE_SIMILARITY", False)) and ("_SIM_MAE_LO" in CONFIG) and ("_SIM_MAE_HI" in CONFIG):
            return _similarity_balanced_from_mae(
                avg_mae=avg_mae,
                mae_lo=float(CONFIG["_SIM_MAE_LO"]),
                mae_hi=float(CONFIG["_SIM_MAE_HI"]),
                score_hi=float(CONFIG.get("SIM_SCORE_HIGH", 95.0)),
                score_lo=float(CONFIG.get("SIM_SCORE_LOW", 60.0)),
            )
        # fallback: single-anchor linear mapping to [0,100]
        ratio = max(0.0, min(1.0, avg_mae / max_mae))
        sim = (1.0 - ratio) * 100.0
        return float(sim)
    except Exception:
        return None

# ----------------------------
# Schema normalization helper
# ----------------------------
def _normalize_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce various source schemas into the expected columns:
    patient_id, exercise, step_index, step_input.
    If 'patient' exists, rename to 'patient_id'. If 'step_input' is missing,
    synthesize a compact JSON string with optional context and mae_by_kp.
    """
    df = df.copy()

    # 1) patient -> patient_id
    if 'patient_id' not in df.columns and 'patient' in df.columns:
        df = df.rename(columns={'patient': 'patient_id'})

    # 2) ensure step_index is integer when present
    if 'step_index' in df.columns:
        try:
            df['step_index'] = df['step_index'].astype(int)
        except Exception:
            pass

    # 3) synthesize step_input if missing: include rep, step_name, and mae_by_kp
    if 'step_input' not in df.columns:
        mae_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(MAE_PREFIX)]
        def _build(row: pd.Series) -> str:
            payload = {}
            if 'rep' in row.index and not pd.isna(row['rep']):
                try:
                    payload['rep'] = int(row['rep'])
                except Exception:
                    payload['rep'] = row['rep']
            if 'step_name' in row.index and not pd.isna(row['step_name']):
                payload['step_name'] = str(row['step_name'])
            if mae_cols:
                mae_map = {}
                for col in mae_cols:
                    try:
                        v = float(row[col])
                        if not pd.isna(v):
                            mae_map[col.replace(MAE_PREFIX, "")] = v
                    except Exception:
                        continue
                if mae_map:
                    payload['mae_by_kp'] = mae_map
            # Fallback text if empty
            if not payload:
                return "{}"
            return json.dumps(payload, ensure_ascii=False)
        df['step_input'] = df.apply(_build, axis=1)

    return df

# ----------------------------
# Similarity auto-tuning helper
# ----------------------------


def _auto_tune_similarity_anchor(df: pd.DataFrame, pct: float = 0.95, fallback: float = DEFAULT_MAX_MAE_FOR_ZERO_SIM) -> float:
    """Compute a data-driven max_mae so similarity% spreads better.
    We compute the mean of all MAE values per row (using the same extraction logic)
    and return the percentile value (default 95th). If not enough data, fallback.
    """
    vals = []
    for _, row in df.iterrows():
        maes = _extract_maes_from_row(row)
        if maes:
            try:
                vals.append(float(mean(maes)))
            except Exception:
                continue
    if not vals:
        return fallback
    try:
        import numpy as np
        p = float(np.quantile(np.array(vals, dtype=float), pct))
        # avoid degenerate tiny anchors
        if p < 1e-6:
            return fallback
        return float(p)
    except Exception:
        return fallback

# ----------------------------
# Grade mapping (A+ .. F) from overall similarity
# ----------------------------

def _grade_from_similarity(sim: float) -> str:
    """Map a 0–100 similarity score to an A+/A/A- … D-/F letter grade.
    Bands follow common US-style thresholds with plus/minus steps.
    """
    if sim is None:
        return ""
    try:
        s = float(sim)
    except Exception:
        return ""
    # Clamp to [0,100] just for safety
    s = max(0.0, min(100.0, s))
    if s >= 97: return "A+"
    if s >= 93: return "A"
    if s >= 90: return "A-"
    if s >= 87: return "B+"
    if s >= 83: return "B"
    if s >= 80: return "B-"
    if s >= 77: return "C+"
    if s >= 73: return "C"
    if s >= 70: return "C-"
    if s >= 67: return "D+"
    if s >= 63: return "D"
    if s >= 60: return "D-"
    return "F"

def load_exercise_texts(exercises_dir: Path) -> Dict[str, str]:
    """Loads *.txt files and returns {canonical_stem: content}"""
    out = {}
    for stem in [
        "Clasp_and_Spread",
        "Deep_Breathing",
        "Horizontal_Pumping",
        "Overhead_Pumping",
        "Push_Down_Pumping",
        "Shoulder_Roll",
    ]:
        p = exercises_dir / f"{stem}.txt"
        if not p.exists():
            logging.warning("Missing exercise file: %s", p)
            continue
        out[stem] = p.read_text(encoding="utf-8")
    return out

def canonical_exercise(name: str) -> str:
    key = _norm_key(name)
    return NORM_EXERCISE_MAP.get(key, name.strip())

def ensure_dirs(base: Path) -> Dict[str, Path]:
    reports_dir = base / "reports"
    summary_dir = base / "summary"
    cache_dir = base / "cache"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {"reports": reports_dir, "summary": summary_dir, "cache": cache_dir}

def default_model_config(model_name: str) -> Dict[str, Any]:
    return {
        "model": model_name,
        "generation_config": {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
        },
        "safety_settings": {
            # keep defaults; user content is clinical-exercise-like
        },
    }

def format_prompt(exercise_name: str, exercise_text: str, patient_id: str, step_payloads: List[Tuple[int, str, int | None]]) -> str:
    """Create a deterministic prompt instructing JSON-only output."""
    step_lines = "\n".join([
        (f"- rep {rep}, step_index {idx}: {content}" if rep is not None else f"- step_index {idx}: {content}")
        for (idx, content, rep) in step_payloads
    ])
    sys = (
        "You are a rehabilitation coach for post-surgery lymphatic exercises. "
        "Given the exercise definition and step-wise inputs for a single patient, "
        "return a STRICT JSON object with three keys: "
        "'step_feedback' (a list of objects with keys: rep, step_index, feedback), "
        "'exercise_summary' (a concise paragraph), and 'letter_grade' (one of A+, A, A-, B+, B, B-, C+, C, C-, D+, D, D-, F). "
        "Grade the overall execution quality on standard academic scale (A best, F worst) using your judgement from the steps. "
        "You MUST output exactly one item in 'step_feedback' for every line in STEP_INPUTS, in the SAME ORDER. "
        "Include the exact 'rep' number (1 or 2) and the 'step_index' shown in each line. "
        "Write the exercise_summary in plain, friendly, non-technical language, about 50 words (40–60), 2–3 short sentences. "
        "Be specific, actionable, and reference the exercise instructions. "
        "Do not include any text outside the JSON. "
        "Do not invent percentages; similarity percentages are computed separately. "
        "Avoid medical diagnoses; focus on exercise form and cues."
    )
    prompt = f"""{sys}

EXERCISE_NAME: {exercise_name}

EXERCISE_DEFINITION:
\"\"\"
{exercise_text.strip()}
\"\"\"

PATIENT_ID: {patient_id}

STEP_INPUTS:
{step_lines}

Expected JSON schema:
{{
  "step_feedback": [
    {{"rep": 1, "step_index": 1, "feedback": "..."}},
    {{"rep": 1, "step_index": 2, "feedback": "..."}}
  ],
  "exercise_summary": "...",
  "letter_grade": "A-"
}}
"""
    return prompt

def call_gemini_with_retries(rotator: _KeyRotator, model_name: str, prompt: str, cache_path: Path, cache_key: str, max_retries: int = 10) -> Dict[str, Any]:
    """Cache -> call -> parse JSON with retries/backoff and API key rotation.
    Rotates keys when we detect quota / 429 / exhausted errors.
    """
    hit = try_cache(cache_path, cache_key)
    if hit is not None:
        return hit

    backoff = 1.5
    for attempt in range(1, max_retries + 1):
        api_key = rotator.current()
        if not api_key:
            raise SystemExit("No API key available. Set CONFIG['API_KEY'] or CONFIG['API_KEYS'].")
        try:
            model = _make_model_for_key(api_key, model_name)
            resp = model.generate_content(prompt)

            # Robust extraction: concatenate all text parts from the first candidate
            text = None
            try:
                if getattr(resp, "candidates", None):
                    cand0 = resp.candidates[0]
                    # Log finish_reason for debugging (e.g., 2 = no content / blocked / max tokens)
                    fr = getattr(cand0, "finish_reason", None)
                    logging.debug("Gemini finish_reason: %s", str(fr))
                    parts = getattr(getattr(cand0, "content", None), "parts", []) or []
                    buf = []
                    for p in parts:
                        t = getattr(p, "text", None)
                        if t:
                            buf.append(t)
                    if buf:
                        text = "".join(buf)
            except Exception:
                text = None

            if not text:
                # As a fallback, try the quick accessor; if still empty, trigger rotation/retry
                try:
                    text = getattr(resp, "text", None)
                except Exception:
                    text = None

            if not text:
                # As a last resort, after retries, allow downstream fallback by returning empty JSON
                if attempt == max_retries:
                    logging.warning("Empty response text from Gemini (no parts) after max retries; returning empty JSON for fallback.")
                    data = {}
                    write_cache(cache_path, cache_key, data)
                    return data
                raise RuntimeError("Empty response text from Gemini (no parts)")

            if "{" in text and "}" in text:
                try:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    text = text[start:end]
                except Exception:
                    pass

            data = json.loads(text)
            write_cache(cache_path, cache_key, data)
            return data
        except Exception as e:
            msg = str(e).lower()
            rotate = any(t in msg for t in [
                "429", "rate", "quota", "exhaust", "billing",
                "resource has been exhausted", "finish_reason", "empty response"
            ])
            logging.warning("Gemini call failed (attempt %d/%d, key idx %d): %s", attempt, max_retries, rotator.i, e)
            if rotate:
                rotator.advance()
                logging.info("Rotating to next API key (index now %d)", rotator.i)
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 30.0)

def try_cache(cache_path: Path, key: str):
    f = cache_path / f"{key}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def write_cache(cache_path: Path, key: str, data: Dict[str, Any]):
    f = cache_path / f"{key}.json"
    f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def save_report(out_reports_dir: Path, report: ExerciseReport):
    patient_dir = out_reports_dir / report.patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    f = patient_dir / f"{report.exercise}.json"
    payload = {
        "patient_id": report.patient_id,
        "exercise": report.exercise,
        "step_feedback": [
            {
                **{k: v for k, v in asdict(s).items() if k in ("step_index", "verdict", "feedback", "similarity_percent", "rep")}
            }
            for s in report.step_feedback
        ],
        "exercise_summary": report.exercise_summary,
        "letter_grade": report.letter_grade,
    }
    f.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Wrote %s", f)

# Utility: clip summary to 40–60 words as a safety net
def _clip_summary_words(text: str, min_w: int = 40, max_w: int = 60) -> str:
    words = text.split()
    if not words:
        return text
    if len(words) <= max_w:
        return text
    return " ".join(words[:max_w])

def append_summary_row(rows: List[Dict[str, Any]], report: ExerciseReport):
    good = sum(1 for s in report.step_feedback if s.verdict == "good")
    bad = sum(1 for s in report.step_feedback if s.verdict != "good")
    rows.append({
        "patient_id": report.patient_id,
        "exercise": report.exercise,
        "num_steps": len(report.step_feedback),
        "num_good": good,
        "num_needs_improvement": bad,
        "summary_preview": (report.exercise_summary[:200] + "...") if report.exercise_summary else "",
    })

def _load_already_processed(final_csv: Path) -> set[tuple[str, str]]:
    """If a progressive CSV exists, return a set of (patient_id, exercise) pairs to skip."""
    if not final_csv.exists():
        return set()
    try:
        df_done = pd.read_csv(final_csv, usecols=["patient_id", "exercise"])  # small, fast
        return set((str(r.patient_id), str(r.exercise)) for r in df_done.itertuples(index=False))
    except Exception:
        return set()


def _append_progress_row(final_csv: Path, row_out: Dict[str, Any]):
    """Append one row to the progressive CSV, creating it with header if needed.
    Handles schema changes by backing up old file and starting fresh if header mismatches.
    """
    try:
        df_one = pd.DataFrame([row_out])
        header_needed = not final_csv.exists()
        # If file exists, verify header schema; if mismatched, back up and start fresh
        if not header_needed:
            try:
                with final_csv.open("r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                existing_cols = [c.strip() for c in first_line.split(",")]
                new_cols = list(df_one.columns)
                if existing_cols != new_cols:
                    bak = final_csv.with_suffix(".bak.csv")
                    logging.warning(
                        "Schema change for %s. Backing up old file to %s and starting fresh.",
                        final_csv, bak
                    )
                    final_csv.rename(bak)
                    header_needed = True
            except Exception:
                header_needed = True
        df_one.to_csv(final_csv, mode="a", header=header_needed, index=False)
    except Exception as e:
        logging.warning(
            "Failed to append progress row for %s/%s: %s",
            row_out.get("patient_id"), row_out.get("exercise"), e
        )


def main():
    # --- Read settings from CONFIG ---
    csv_input_dir = Path(CONFIG.get("CSV_INPUT_DIR", "")).expanduser()
    ex_dir = Path(CONFIG.get("EXERCISES_DIR", "")).expanduser()
    out_dir = Path(CONFIG.get("OUT_DIR", "")).expanduser()
    model_name = str(CONFIG.get("MODEL", "gemini-2.5-flash"))
    similarity_max_mae = float(CONFIG.get("SIMILARITY_MAX_MAE", DEFAULT_MAX_MAE_FOR_ZERO_SIM))
    write_detailed_csv = bool(CONFIG.get("WRITE_DETAILED_CSV", True))

    if genai is None:
        raise SystemExit("google-generativeai is not installed. Run: pip install google-generativeai")

    # API key rotation setup
    keys = []
    # Prefer a provided list of keys; fallback to single API_KEY
    cfg_keys = CONFIG.get("API_KEYS", []) or []
    if isinstance(cfg_keys, list):
        keys.extend([str(k).strip() for k in cfg_keys if str(k).strip()])
    single = str(CONFIG.get("API_KEY", "")).strip()
    if single and single not in keys:
        keys.append(single)
    if not keys:
        raise SystemExit("No API key provided. Set CONFIG['API_KEYS'] or CONFIG['API_KEY'].")
    rotator = _KeyRotator(keys)

    # IO from CONFIG
    paths = ensure_dirs(out_dir)
    reports_dir, summary_dir, cache_dir = paths["reports"], paths["summary"], paths["cache"]

    # Load exercise reference texts
    exercise_texts = load_exercise_texts(ex_dir)
    if not exercise_texts:
        raise SystemExit(f"No exercise TXT files found in: {ex_dir}")

    # Load CSVs: prefer a single-patient file if provided
    single_csv = str(CONFIG.get("SINGLE_PATIENT_CSV", "")).strip()
    frames = []
    if single_csv:
        fp = Path(single_csv).expanduser()
        if not fp.exists():
            raise SystemExit(f"SINGLE_PATIENT_CSV not found: {fp}")
        try:
            frames.append(pd.read_csv(fp))
            logging.info("Loaded single CSV %s", fp)
        except Exception as e:
            raise SystemExit(f"Failed to read SINGLE_PATIENT_CSV: {fp}: {e}")
    else:
        csv_files = sorted(csv_input_dir.glob("*.csv"))
        if not csv_files:
            raise SystemExit(f"No CSV files found in: {csv_input_dir}")
        for fp in csv_files:
            try:
                frames.append(pd.read_csv(fp))
                logging.info("Loaded %s", fp)
            except Exception as e:
                logging.warning("Skipping %s due to read error: %s", fp, e)
        if not frames:
            raise SystemExit("No readable CSV files were found.")
    df = pd.concat(frames, ignore_index=True)
    # Normalize incoming schema
    df = _normalize_input_schema(df)
    # Auto-tune similarity anchor to increase variation if enabled
    if bool(CONFIG.get("AUTO_TUNE_SIMILARITY", False)):
        tuned = _auto_tune_similarity_anchor(df, pct=float(CONFIG.get("SIMILARITY_PCTL", 0.95)), fallback=similarity_max_mae)
        logging.info("Auto-tuned SIMILARITY_MAX_MAE from %.3f to %.3f (pct=%.2f)", similarity_max_mae, tuned, float(CONFIG.get("SIMILARITY_PCTL", 0.95)))
        similarity_max_mae = tuned
    # Compute balanced similarity anchors if requested
    if bool(CONFIG.get("BALANCE_SIMILARITY", False)):
        mae_lo, mae_hi = _compute_mae_quantiles(
            df,
            q_low=float(CONFIG.get("SIM_PCTL_LOW", 0.20)),
            q_high=float(CONFIG.get("SIM_PCTL_HIGH", 0.80)),
            fallback=(similarity_max_mae * 0.3, similarity_max_mae * 0.9),
        )
        CONFIG["_SIM_MAE_LO"], CONFIG["_SIM_MAE_HI"] = mae_lo, mae_hi
        logging.info("Balanced similarity anchors: mae_lo=%.3f → ~%.1f, mae_hi=%.3f → ~%.1f", mae_lo, float(CONFIG.get("SIM_SCORE_HIGH", 95.0)), mae_hi, float(CONFIG.get("SIM_SCORE_LOW", 60.0)))
    required_cols = {"patient_id", "exercise", "step_index", "step_input"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV is missing columns: {sorted(missing)}")

    # Normalize types
    df["exercise"] = df["exercise"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df["step_index"] = df["step_index"].astype(int)
    df = df.sort_values(["patient_id", "exercise", "step_index"])

    # Progressive output CSV path (write-as-you-go)
    final_csv = summary_dir / "final_llm_feedback.csv"
    already = _load_already_processed(final_csv)
    if already:
        logging.info("Resuming: %d patient×exercise rows already present in %s", len(already), final_csv)
    global _OUTPUT_COLUMNS
    _OUTPUT_COLUMNS = None

    # Determine dynamic maximum number of steps for this run (across all exercises in this file)
    try:
        max_step_this_run = int(pd.to_numeric(df["step_index"], errors="coerce").max())
        if max_step_this_run < 1:
            max_step_this_run = 1
        # put a sensible upper bound to avoid runaway schemas
        max_step_this_run = min(max_step_this_run, 20)
    except Exception:
        max_step_this_run = 5
    logging.info("Dynamic max steps for this run: %d", max_step_this_run)

    # List to collect wide CSV rows
    wide_rows: List[Dict[str, Any]] = []

    # Group by patient + exercise
    for (patient_id, ex_raw), g in df.groupby(["patient_id", "exercise"]):
        ex_canon = canonical_exercise(ex_raw)
        if ex_canon not in exercise_texts:
            logging.warning(
                "Skipping (unknown exercise): patient=%s exercise=%s (normalized=%s canonical=%s)",
                patient_id, ex_raw, _norm_key(ex_raw), ex_canon
            )
            continue

        # Resume: skip if already in progressive CSV
        if (str(patient_id), str(ex_canon)) in already:
            logging.info("Skipping already-processed row (resume): patient=%s exercise=%s", patient_id, ex_canon)
            continue

        # Index rows by exact (rep, step) and then order keys as rep=1..2, steps ascending
        rows_by_key: Dict[Tuple[int, int], pd.Series] = {}
        for _, row in g.iterrows():
            idx = int(row["step_index"]) if not pd.isna(row.get("step_index")) else 0
            rep_val = _parse_rep(row.get("rep"))
            key = (rep_val, idx)
            # keep the first occurrence only (dedupe safety)
            if key not in rows_by_key:
                rows_by_key[key] = row

        # Build ordered keys: rep 1 then rep 2; within each, sort steps actually present for that rep
        keys_in_order: List[Tuple[int, int]] = []
        for rep_val in (1, 2):
            steps = sorted([s for (r, s) in rows_by_key.keys() if r == rep_val and 1 <= s <= 5])
            keys_in_order.extend([(rep_val, s) for s in steps])

        # Maps for payload and similarity
        payload_by_key: Dict[Tuple[int, int], str] = {}
        sim_by_key: Dict[Tuple[int, int], float | None] = {}
        for key in keys_in_order:
            row = rows_by_key[key]
            payload_by_key[key] = str(row.get("step_input", "")).strip()
            sim_by_key[key] = compute_similarity_percent(row, similarity_max_mae)

        logging.debug("%s/%s keys_in_order: %s", patient_id, ex_canon, keys_in_order)
        for k in keys_in_order:
            if sim_by_key.get(k) is None:
                row = rows_by_key[k]
                mae_cols = [c for c in g.columns if isinstance(c, str) and c.startswith(MAE_PREFIX)]
                present = {}
                for c in mae_cols:
                    try:
                        present[c] = row[c]
                    except Exception:
                        present[c] = None
                logging.debug("Missing similarity for key=%s; mae cols: %s", k, present)

        # For the prompt sent to the LLM (order must match keys_in_order)
        step_payloads: List[Tuple[int, str, int | None]] = [
            (idx, payload_by_key[(rep, idx)], rep) for (rep, idx) in keys_in_order
        ]

        # Track present keys for downstream logic and debugging
        present_keys = set(keys_in_order)
        logging.debug("%s/%s present_keys: %s", patient_id, ex_canon, sorted(present_keys))

        # Prompt
        prompt = format_prompt(
            exercise_name=ex_canon,
            exercise_text=exercise_texts[ex_canon],
            patient_id=patient_id,
            step_payloads=step_payloads
        )

        # Cache key
        cache_key = sha1(f"{patient_id}::{ex_canon}::" + sha1(json.dumps(step_payloads, ensure_ascii=False)))
        # LLM call with caching + retries
        data = call_gemini_with_retries(rotator, model_name, prompt, cache_dir, cache_key)

        # Parse LLM output, assign feedback to the exact (rep, step) pair, even if model returns fewer/more items.
        step_feedback_list: List[StepFeedback] = []
        llm_items = list(data.get("step_feedback", []))
        # Pair i-th LLM item to i-th (rep, step) key; if missing, make empty feedback
        for i in range(len(keys_in_order)):
            rep_i, step_i = keys_in_order[i]
            item = llm_items[i] if i < len(llm_items) else {}
            feedback_text = str(item.get("feedback", "")).strip()
            # Prefer explicit rep from the model if present and valid
            rep_model = item.get("rep")
            try:
                rep_model = int(rep_model)
                if rep_model not in (1, 2):
                    rep_model = rep_i
            except Exception:
                rep_model = rep_i
            step_feedback_list.append(
                StepFeedback(
                    step_index=int(item.get("step_index", step_i)),
                    verdict=str(item.get("verdict", "")).strip() or "",
                    feedback=feedback_text,
                    similarity_percent=sim_by_key.get((rep_i, step_i)),
                    rep=rep_model,
                )
            )

        # Fallback: if any feedback is blank, synthesize a short tip using the largest-MAE joints
        for fb in step_feedback_list:
            if not fb.feedback:
                # Find the row for this (rep, step) and extract MAEs
                row_src = rows_by_key.get((fb.rep if fb.rep is not None else 1, fb.step_index))
                kp_maes = {}
                if row_src is not None:
                    for c in row_src.index:
                        if isinstance(c, str) and c.startswith(MAE_PREFIX):
                            try:
                                v = float(row_src[c])
                                if not pd.isna(v):
                                    kp_maes[c.replace(MAE_PREFIX, "")] = v
                            except Exception:
                                continue
                if kp_maes:
                    # Take top 2 deviating joints
                    top = sorted(kp_maes.items(), key=lambda kv: kv[1], reverse=True)[:2]
                    joints = ", ".join(n.replace("_", " ") for n, _ in top)
                    fb.feedback = f"Focus on steadier alignment at {joints}. Keep movements smooth and follow the cue for this step."
                else:
                    fb.feedback = "Keep form steady and follow the cue for this step."

        summary_text = _clip_summary_words(str(data.get("exercise_summary", "")).strip(), min_w=40, max_w=60)
        # Keep the original pairing order (rep, step) to avoid any downstream ambiguity
        report = ExerciseReport(
            patient_id=patient_id,
            exercise=ex_canon,
            step_feedback=step_feedback_list,
            exercise_summary=summary_text,
            letter_grade=str(data.get("letter_grade", "")).strip(),
        )

        # Build wide row
        row_out: Dict[str, Any] = {
            "patient_id": report.patient_id,
            "exercise": report.exercise,
            "letter_grade": report.letter_grade,
            "exercise_summary": report.exercise_summary,
            "overall_similarity": "",
        }
        # Initialize all review/sim fields blank for r in {1,2}, s in {1..max_step_this_run}
        for rep_n in (1, 2):
            for s_idx in range(1, max_step_this_run + 1):
                row_out[f"review_r{rep_n}_s{s_idx}"] = ""
                row_out[f"similarity_r{rep_n}_s{s_idx}"] = ""
        # Fill from feedback list, preserving existing values and only writing when key is present
        for s in report.step_feedback:
            rep_val = s.rep if s.rep is not None else 1
            key = (rep_val, s.step_index)
            if key in present_keys and rep_val in (1, 2) and 1 <= s.step_index <= max_step_this_run:
                row_out[f"review_r{rep_val}_s{s.step_index}"] = s.feedback or ""
                if s.similarity_percent is not None:
                    row_out[f"similarity_r{rep_val}_s{s.step_index}"] = s.similarity_percent

        # Compute overall similarity (mean of available per-step similarities), restrict to actually present keys
        sims = []
        for (rep_n, s_idx) in present_keys:
            if rep_n in (1, 2) and 1 <= s_idx <= max_step_this_run:
                v = row_out.get(f"similarity_r{rep_n}_s{s_idx}")
                try:
                    if v != "" and v is not None:
                        sims.append(float(v))
                except Exception:
                    continue
        row_out["overall_similarity"] = (sum(sims) / len(sims)) if sims else ""
        # Deterministic grade from overall similarity using A+/A/A- … D-/F bands
        if row_out["overall_similarity"] != "":
            row_out["letter_grade"] = _grade_from_similarity(row_out["overall_similarity"])

        # Cleanup: ensure no fabricated cells for non-present keys
        for rep_n in (1, 2):
            for s_idx in range(1, max_step_this_run + 1):
                if (rep_n, s_idx) not in present_keys:
                    row_out[f"review_r{rep_n}_s{s_idx}"] = ""
                    row_out[f"similarity_r{rep_n}_s{s_idx}"] = ""

        # Ensure stable output column order
        if _OUTPUT_COLUMNS is None:
            cols = ["patient_id", "exercise", "letter_grade", "exercise_summary", "overall_similarity"]
            # Order columns as rep 1 (all steps), then rep 2 (all steps)
            for rep_n in (1, 2):
                for s_idx in range(1, max_step_this_run + 1):
                    cols.append(f"review_r{rep_n}_s{s_idx}")
                    cols.append(f"similarity_r{rep_n}_s{s_idx}")
            _OUTPUT_COLUMNS = cols
        row_out = {k: row_out.get(k, "") for k in _OUTPUT_COLUMNS}

        _append_progress_row(final_csv, row_out)
        wide_rows.append(row_out)

        # polite pacing to avoid RPM bursts
        time.sleep(0.2)



if __name__ == "__main__":
    main()
