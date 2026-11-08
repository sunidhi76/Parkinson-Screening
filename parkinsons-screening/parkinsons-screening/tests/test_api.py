import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from src.config import MODELS_DIR
from src.voice.load import BASIC_FEATURES, FEATURES

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "voice_meta.json").exists(), reason="models not trained (make train)")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_has_disclaimer(client):
    j = client.get("/health").json()
    assert j["status"] == "ok" and "not diagnosis" in j["disclaimer"]


def test_no_combined_endpoint(client):
    paths = {r.path for r in app.routes}
    assert not any("combined" in p or "fusion" in p or p == "/predict" for p in paths)


def test_voice_full_row(client):
    r = client.post("/predict/voice-features", json={"features": {f: 0.5 for f in FEATURES}})
    assert r.status_code == 200 and "disclaimer" in r.json() and r.json()["input_mode"] == "uci_format_features"
    assert r.json()["quality"]["out_of_training_range"]       # 0.5 for every column is far out of range


def test_voice_row_string_with_status_and_name(client):
    row = "phon_R01_S01_1,119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,1,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654"
    r = client.post("/predict/voice-features", json={"row": row})
    assert r.status_code == 200 and 0 <= r.json()["probability"] <= 1


def test_voice_basic7_is_labelled_unvalidated(client):
    r = client.post("/predict/voice-features", json={"features": {f: 1.0 for f in BASIC_FEATURES}})
    assert r.status_code == 200 and "unvalidated demo" in r.json()["label"]


@pytest.mark.parametrize("body", [{}, {"row": "1,2,3"}, {"row": "abc,def"}, {"features": {"nope": 1}},
                                  {"features": {"MDVP:Fo(Hz)": 100}}])
def test_voice_bad_input_gives_clear_4xx(client, body):
    r = client.post("/predict/voice-features", json=body)
    assert 400 <= r.status_code < 500 and r.json()


def test_voice_nan_rejected(client):
    r = client.post("/predict/voice-features", content='{"features": {"MDVP:Fo(Hz)": NaN}}', headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def _png(arr) -> bytes:
    b = io.BytesIO(); Image.fromarray(arr).save(b, format="PNG"); return b.getvalue()


def test_drawing_blank_image_no_score(client):
    r = client.post("/predict/drawing", files={"file": ("x.png", _png(np.full((200, 200), 255, np.uint8)), "image/png")})
    j = r.json()
    assert r.status_code == 200 and j["probability"] is None and j["quality"]["warnings"] and "disclaimer" in j


def test_drawing_valid_spiral(client, tmp_path):
    from src.synthetic import generate_dataset
    meta = generate_dataset(tmp_path, 1, 1, 5)
    r = client.post("/predict/drawing", files={"file": ("s.png", open(meta.image_path[0], "rb").read(), "image/png")})
    j = r.json()
    assert r.status_code == 200 and j["risk_band"] in {"low", "uncertain", "elevated", "unavailable"}
    if j["model_trained_on"] == "synthetic":
        assert "SYNTHETIC_MODEL_WARNING" in j


def test_drawing_not_an_image(client):
    r = client.post("/predict/drawing", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 415 and "image" in r.json()["detail"]


def test_drawing_empty_and_too_large(client):
    assert client.post("/predict/drawing", files={"file": ("x.png", b"", "image/png")}).status_code == 400
    assert client.post("/predict/drawing", files={"file": ("x.png", b"0" * (5 * 1024 * 1024 + 10), "image/png")}).status_code == 413


def test_nothing_written_to_disk(client, tmp_path):
    import os
    before = set(os.listdir(MODELS_DIR))
    client.post("/predict/drawing", files={"file": ("x.png", _png(np.zeros((100, 100), np.uint8)), "image/png")})
    assert set(os.listdir(MODELS_DIR)) == before
