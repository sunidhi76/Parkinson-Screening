"""Central configuration: paths, seeds, disclaimers and thresholds."""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
VOICE_FILE: Path = DATA_DIR / "parkinsons.data"
DRAWINGS_DIR: Path = DATA_DIR / "drawings"
MODELS_DIR: Path = ROOT / "models"
RESULTS_DIR: Path = ROOT / "experiments" / "results"

SEED: int = 42
N_BOOT: int = 1000
CV_FOLDS: int = 5
CV_REPEATS: int = 10

DISCLAIMER: str = (
    "Screening support, not diagnosis. This output is not a medical diagnosis and must not replace "
    "evaluation by a qualified clinician."
)
SYNTHETIC_STAMP: str = "SYNTHETIC: pipeline validation only, not scientific results."
UNVALIDATED_VOICE_STAMP: str = "unvalidated demo, not comparable to the training data."

# Risk bands applied to a predicted probability (screening-support heuristics, not clinical cut-offs).
BAND_LOW_MAX: float = 0.35
BAND_HIGH_MIN: float = 0.65

IMG_SIZE: int = 64  # CNN input resolution


def risk_band(p: float) -> tuple[str, bool]:
    """Map a probability to (band, retest_flag). 'uncertain' means: please retest."""
    if p < BAND_LOW_MAX:
        return "low", False
    if p > BAND_HIGH_MIN:
        return "elevated", False
    return "uncertain", True
