"""FastAPI backend. Two INDEPENDENT screening-support modules; there is deliberately NO combined endpoint.

Uploaded audio/images are processed in memory and never written to disk or logged."""
from __future__ import annotations

import io
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.config import DISCLAIMER, MODELS_DIR, ROOT, UNVALIDATED_VOICE_STAMP, risk_band
from src.drawing.features import FEATURE_NAMES as DRAW_FEATURES
from src.drawing.features import features_matrix
from src.drawing.load import preprocess_image
from src.voice.load import BASIC_FEATURES, FEATURES

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
STATE: dict[str, Any] = {}


def _load_models(models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Load whatever artifacts exist; missing ones simply disable that endpoint (clear error, no crash)."""
    st: dict[str, Any] = {"voice": {}, "voice_meta": None, "voice_basic": None, "drawing": None, "drawing_meta": None}
    vm = models_dir / "voice_meta.json"
    if vm.exists():
        st["voice_meta"] = json.loads(vm.read_text())
        for f in models_dir.glob("voice_*.joblib"):
            name = f.stem.replace("voice_", "")
            (st.__setitem__("voice_basic", joblib.load(f)) if name == "basic7" else st["voice"].__setitem__(name, joblib.load(f)))
    dm = models_dir / "drawing_meta.json"
    if dm.exists() and (models_dir / "drawing_handcrafted.joblib").exists():
        st["drawing_meta"] = json.loads(dm.read_text()); st["drawing"] = joblib.load(models_dir / "drawing_handcrafted.joblib")
    return st


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.clear(); STATE.update(_load_models())
    yield


app = FastAPI(title="parkinsons-screening", version="1.0.0", lifespan=lifespan,
              description=f"{DISCLAIMER} Voice and drawing modules are independent and never fused.")


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:  # never leak a stack trace or crash
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "Unexpected server error.",
                                                  "disclaimer": DISCLAIMER})


@app.get("/health")
def health() -> dict:
    vm, dm = STATE.get("voice_meta"), STATE.get("drawing_meta")
    return {"status": "ok", "disclaimer": DISCLAIMER,
            "voice": {"loaded": bool(STATE.get("voice")), "models": sorted(STATE.get("voice", {})),
                      "default_model": vm and vm["default_model"], "training_subjects": vm and vm["n_subjects"],
                      "basic7_demo_model_loaded": STATE.get("voice_basic") is not None},
            "drawing": {"loaded": STATE.get("drawing") is not None, "trained_on": dm and dm["trained_on"],
                        "warning": dm and dm["warning"]}}


# ---------------------------------------------------------------- voice
class VoiceRequest(BaseModel):
    features: dict[str, float] | None = Field(None, description="feature name -> value (UCI column names)")
    row: str | None = Field(None, description="a pasted UCI-format row (comma/tab/space separated)")
    model: str | None = Field(None, description="logreg | svm_rbf | random_forest | xgboost")


_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def parse_uci_row(text: str) -> dict[str, float]:
    """Accept 'name,f1..f22' with or without the status column (UCI order has status after HNR)."""
    toks = [t for t in re.split(r"[,\t; ]+", text.strip()) if t != ""]
    if toks and not _NUM.match(toks[0]):
        toks = toks[1:]                                           # drop the recording name
    if any(not _NUM.match(t) for t in toks):
        raise HTTPException(422, "Row contains non-numeric values after the optional name column.")
    v = [float(t) for t in toks]
    if len(v) == 23:                                              # raw UCI order includes `status` at index 16
        v = v[:16] + v[17:]
    if len(v) != 22:
        raise HTTPException(422, f"Expected 22 feature values (or 23 incl. UCI 'status'); got {len(v)}.")
    return dict(zip(FEATURES, v))


def _range_warnings(feats: dict[str, float]) -> list[str]:
    ranges = STATE["voice_meta"]["feature_ranges"]; out = []
    for k, v in feats.items():
        r = ranges.get(k)
        if r and (v < r["min"] or v > r["max"]):
            out.append(f"{k}={v:g} is outside the training range [{r['min']:g}, {r['max']:g}]")
    return out


@app.post("/predict/voice-features")
def predict_voice(req: VoiceRequest) -> dict:
    if not STATE.get("voice"):
        raise HTTPException(503, "Voice models not found. Run `make train` first.")
    if req.row is not None:
        feats = parse_uci_row(req.row)
    elif req.features is not None:
        feats = {k: float(v) for k, v in req.features.items()}
    else:
        raise HTTPException(422, "Provide either `features` (dict) or `row` (string).")
    bad = [k for k, v in feats.items() if not np.isfinite(v)]
    if bad:
        raise HTTPException(422, f"Non-finite values for: {bad}")
    unknown = [k for k in feats if k not in FEATURES]
    if unknown:
        raise HTTPException(422, f"Unknown feature names: {unknown}. Expected UCI names such as 'MDVP:Fo(Hz)'.")
    full = all(f in feats for f in FEATURES)
    basic = all(f in feats for f in BASIC_FEATURES)
    if full:
        name = req.model or STATE["voice_meta"]["default_model"]
        if name not in STATE["voice"]:
            raise HTTPException(422, f"Unknown model {name!r}; available: {sorted(STATE['voice'])}")
        model, order, mode = STATE["voice"][name], FEATURES, "uci_format_features"
        note = ("Model trained on 195 UCI recordings from 32 subject codes; subject-wise AUC about 0.7-0.8 with wide "
                "CIs (see README). Valid only for features computed by the same pipeline as UCI.")
        label = "UCI-format features: real model prediction, still screening support only."
    elif basic and STATE.get("voice_basic") is not None:
        name, model, order, mode = "basic7_demo", STATE["voice_basic"], BASIC_FEATURES, "basic7_unvalidated_demo"
        note = ("Only 7 basic features supplied. This demo model was trained on the same 7 UCI columns, but values "
                "extracted from a browser recording are NOT computed like the UCI features.")
        label = UNVALIDATED_VOICE_STAMP
    else:
        missing = [f for f in FEATURES if f not in feats]
        raise HTTPException(422, f"Need all 22 UCI features (or the 7 basic ones). Missing: {missing[:8]}"
                                 f"{'...' if len(missing) > 8 else ''}")
    x = np.array([[feats[f] for f in order]], dtype=float)
    p = float(model.predict_proba(x)[0, 1]); band, retest = risk_band(p)
    warns = _range_warnings({k: feats[k] for k in order})
    return {"module": "voice", "probability": round(p, 4), "risk_band": band, "uncertain_please_retest": retest,
            "model": name, "input_mode": mode, "label": label, "validity_note": note,
            "quality": {"out_of_training_range": warns, "ok": not warns},
            "disclaimer": DISCLAIMER}


# ---------------------------------------------------------------- drawing
def quality_checks(info: dict, centerline_ok: bool) -> tuple[list[str], bool]:
    """Human-readable input-quality warnings; second value = whether a score is allowed."""
    w, ok = [], True
    if info["blank"]:
        return ["No stroke detected (blank image)."], False
    if min(info["orig_height"], info["orig_width"]) < 64:
        w.append("Image is very small (<64 px): detail is lost."); ok = False
    if info["contrast"] < 0.25:
        w.append("Low contrast between pen and paper.")
    if info["ink_fraction"] < 0.002:
        w.append("Very little ink detected: stroke may be too faint or thin."); ok = False
    if info["ink_fraction"] > 0.35:
        w.append("Too much ink / dark background: this does not look like a single spiral."); ok = False
    if info["n_components"] > 5:
        w.append(f"{info['n_components']} separate ink components: noisy scan, broken stroke or extra marks.")
    if info["touches_border"]:
        w.append("Drawing touches the image border: it may be cropped.")
    if not centerline_ok:
        w.append("Could not trace the spiral turns (need a clear, roughly 3+ turn spiral). No score produced."); ok = False
    return w, ok


@app.post("/predict/drawing")
async def predict_drawing(file: UploadFile = File(...)) -> dict:
    if STATE.get("drawing") is None:
        raise HTTPException(503, "Drawing model not found. Run `make train` first.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 5 MB).")
    if not data:
        raise HTTPException(400, "Empty upload.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load(); arr, info = preprocess_image(im)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(415, "Could not read the file as an image (use PNG/JPEG/BMP).")
    finally:
        del data                                                    # nothing is stored
    dm = STATE["drawing_meta"]
    F = features_matrix(arr[None]) if not info["blank"] else np.full((1, len(DRAW_FEATURES)), np.nan)
    center_ok = bool(F[0, DRAW_FEATURES.index("centerline_ok")] == 1.0) if not info["blank"] else False
    warns, allowed = quality_checks(info, center_ok)
    resp: dict[str, Any] = {"module": "drawing", "model": "handcrafted_rf",
                            "model_trained_on": dm["trained_on"], "model_provenance": dm["warning"],
                            "quality": {"warnings": warns, "ok": allowed, "ink_fraction": round(info["ink_fraction"], 4),
                                        "contrast": round(info["contrast"], 3)},
                            "disclaimer": DISCLAIMER}
    if dm["trained_on"] == "synthetic":
        resp["SYNTHETIC_MODEL_WARNING"] = "This model was trained on SYNTHETIC spirals only. The probability is NOT meaningful."
    if not allowed:
        resp.update(probability=None, risk_band="unavailable", uncertain_please_retest=True)
        return resp
    p = float(STATE["drawing"].predict_proba_features(F)[0, 1]); band, retest = risk_band(p)
    resp.update(probability=round(p, 4), risk_band=band, uncertain_please_retest=retest)
    return resp


WEB = ROOT / "web"
if WEB.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
