import numpy as np
import pytest

from src.splits import (
    assert_cv_partition,
    assert_disjoint_groups,
    group_kfold,
    leave_one_dataset_out,
    leave_one_group_out,
    make_splits,
    random_kfold,
)


def toy(n_sub=24, per=6, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat([f"S{i:02d}" for i in range(n_sub)], per)
    lab = (np.arange(n_sub) % 3 != 0).astype(int)          # subject-level labels, imbalanced
    return np.repeat(lab, per), g, rng


@pytest.mark.parametrize("seed", range(5))
def test_group_kfold_no_subject_in_both(seed):
    y, g, _ = toy()
    sp = group_kfold(y, g, 5, seed)
    assert_disjoint_groups(g, sp); assert_cv_partition(len(y), sp)


def test_logo_no_subject_in_both():
    y, g, _ = toy()
    sp = leave_one_group_out(g)
    assert len(sp) == len(set(g)); assert_disjoint_groups(g, sp); assert_cv_partition(len(y), sp)


def test_random_split_DOES_leak():
    """Sanity check that the leaky baseline really leaks (and the assertion catches it)."""
    y, g, _ = toy()
    sp = random_kfold(y, 5, 0)
    with pytest.raises(AssertionError):
        assert_disjoint_groups(g, sp)


def test_leave_one_dataset_out_no_subject_overlap():
    y, g, _ = toy()
    src = np.where(np.arange(len(y)) < len(y) // 2, "handpd", "newhandpd")
    g = np.array([f"{s}_{x}" for s, x in zip(src, g)])      # prefixed ids, as in drawing.load
    seen = []
    for held, tr, te in leave_one_dataset_out(src):
        assert set(src[te]) == {held} and held not in set(src[tr])
        assert_disjoint_groups(g, [(tr, te)]); seen.append(held)
    assert sorted(seen) == ["handpd", "newhandpd"]


def test_make_splits_dispatch_and_error():
    y, g, _ = toy()
    for k in ("group", "logo"):
        assert_disjoint_groups(g, make_splits(k, y, g))
    with pytest.raises(ValueError):
        make_splits("nope", y, g)


def test_real_voice_splits_are_subject_wise():
    from src.config import VOICE_FILE
    if not VOICE_FILE.exists():
        pytest.skip("voice data not downloaded")
    from src.voice.load import get_xyg, load_voice
    _, y, g = get_xyg(load_voice())[0], *get_xyg(load_voice())[1:]
    for s in range(10):
        sp = make_splits("group", y, g, s); assert_disjoint_groups(g, sp); assert_cv_partition(len(y), sp)
    assert_disjoint_groups(g, make_splits("logo", y, g))
