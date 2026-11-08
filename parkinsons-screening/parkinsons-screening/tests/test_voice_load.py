import pandas as pd
import pytest

from src.config import VOICE_FILE
from src.voice.load import FEATURES, load_voice, parse_subject, subject_summary


def test_parse_subject():
    assert parse_subject("phon_R01_S01_1") == "S01"
    assert parse_subject("phon_R01_S35_7") == "S35"
    with pytest.raises(ValueError):
        parse_subject("garbage")


def test_load_voice_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_voice(tmp_path / "nope.data")


def test_load_voice_fixture(tmp_path):
    cols = ["name"] + FEATURES + ["status"]
    rows = [[f"phon_R01_S0{s}_{r}"] + [0.1 * (s + r)] * len(FEATURES) + [int(s > 1)] for s in (1, 2, 3) for r in (1, 2)]
    p = tmp_path / "x.data"; pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    df = load_voice(p); s = subject_summary(df)
    assert s["n_subjects"] == 3 and s["n_recordings"] == 6 and s["subjects_with_mixed_labels"] == 0


def test_real_data_counts():
    if not VOICE_FILE.exists():
        pytest.skip("voice data not downloaded")
    s = subject_summary(load_voice())
    assert s["n_recordings"] == 195 and s["subjects_with_mixed_labels"] == 0
    assert sum(s["subjects_per_class"].values()) == s["n_subjects"]
