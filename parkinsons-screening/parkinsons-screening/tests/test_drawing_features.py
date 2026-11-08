import numpy as np
import pandas as pd
import pytest

from src.drawing.features import FEATURE_NAMES, extract_features, features_matrix
from src.drawing.load import build_metadata, load_images
from src.synthetic import generate_dataset


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    meta = generate_dataset(d, 5, 2, 3)
    imgs, info = load_images(meta)
    return meta, imgs, info


def test_features_finite_and_complete(synth):
    meta, imgs, _ = synth
    F = features_matrix(imgs)
    assert F.shape == (len(meta), len(FEATURE_NAMES))
    assert np.isfinite(F[:, FEATURE_NAMES.index("ink_fraction")]).all()
    assert (F[:, FEATURE_NAMES.index("centerline_ok")] == 1).mean() > 0.8


def test_tremor_raises_radial_deviation(synth):
    meta, imgs, _ = synth
    df = pd.DataFrame(features_matrix(imgs), columns=FEATURE_NAMES); df["y"] = meta.label.values
    m = df.groupby("y")["radial_dev_rms"].mean()
    assert m[1] > m[0]                                   # synthetic sanity check only


def test_blank_image_does_not_crash():
    f = extract_features(np.zeros((128, 128), np.float32))
    assert f["centerline_ok"] == 0.0


def test_metadata_ids_prefixed_and_unique_across_sources(tmp_path):
    from PIL import Image
    for src in ("handpd", "newhandpd"):
        for c in ("healthy", "parkinson"):
            (tmp_path / src / c).mkdir(parents=True)
            Image.new("L", (80, 80), 255).save(tmp_path / src / c / f"{c}_a.png")
    meta = build_metadata(tmp_path)
    assert list(meta.columns) == ["image_path", "source_dataset", "subject_id", "label", "subject_id_source"]
    assert meta.subject_id.is_unique and all(s.startswith(d) for s, d in zip(meta.subject_id, meta.source_dataset))
    # a subject mapped to images of BOTH classes must be rejected
    pd.DataFrame({"filename": ["healthy_a.png", "parkinson_a.png"], "subject_id": ["P1", "P1"], "source": ["handpd", "handpd"]}).to_csv(tmp_path / "subjects.csv", index=False)
    with pytest.raises(AssertionError, match="conflicting labels"):
        build_metadata(tmp_path)


def test_empty_folder_gives_empty_metadata(tmp_path):
    assert len(build_metadata(tmp_path)) == 0
