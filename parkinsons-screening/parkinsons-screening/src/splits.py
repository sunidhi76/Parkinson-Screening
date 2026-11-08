"""Split strategies. Every subject-wise strategy can be verified with ``assert_disjoint_groups``."""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

Split = tuple[np.ndarray, np.ndarray]


def random_kfold(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[Split]:
    """Recording-level stratified K-fold that IGNORES subjects (leaky on repeated-measure data)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


def group_kfold(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[Split]:
    """Subject-wise, class-stratified K-fold (StratifiedGroupKFold): no subject spans train and test."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in sgkf.split(np.zeros(len(y)), y, groups)]


def leave_one_group_out(groups: np.ndarray) -> list[Split]:
    """Leave-one-subject-out."""
    return [(tr, te) for tr, te in LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups)]


def leave_one_dataset_out(sources: np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held_out_source, train_idx, test_idx): train on all other sources, test on one."""
    sources = np.asarray(sources)
    for s in sorted(set(sources.tolist())):
        te = np.where(sources == s)[0]
        tr = np.where(sources != s)[0]
        yield s, tr, te


def make_splits(kind: str, y: np.ndarray, groups: np.ndarray, seed: int = 42, n_splits: int = 5) -> list[Split]:
    """Dispatch by name: 'random' | 'group' | 'logo'."""
    if kind == "random":
        return random_kfold(y, n_splits, seed)
    if kind == "group":
        return group_kfold(y, groups, n_splits, seed)
    if kind == "logo":
        return leave_one_group_out(groups)
    raise ValueError(f"unknown split kind: {kind!r}")


def assert_disjoint_groups(groups: Sequence, splits: Sequence[Split]) -> None:
    """Raise AssertionError if any group appears on both sides of any split."""
    g = np.asarray(groups)
    for i, (tr, te) in enumerate(splits):
        shared = set(g[tr].tolist()) & set(g[te].tolist())
        assert not shared, f"split {i}: groups in BOTH train and test: {sorted(shared)[:5]}"


def assert_cv_partition(n: int, splits: Sequence[Split]) -> None:
    """Every sample is tested exactly once and never in its own training set."""
    seen = np.zeros(n, dtype=int)
    for tr, te in splits:
        assert not (set(tr.tolist()) & set(te.tolist())), "train/test index overlap"
        seen[te] += 1
    assert (seen == 1).all(), "each sample must be in exactly one test fold"
