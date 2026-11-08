"""Voice classifiers as sklearn Pipelines (imputer + scaler are fit inside every training fold)."""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import SEED

# SVC(probability=True) is deprecated in scikit-learn 1.9 (removal planned for 1.11). Versions are pinned.
warnings.filterwarnings("ignore", message=".*probability.*parameter was deprecated.*", category=FutureWarning)

MODEL_NAMES = ["logreg", "svm_rbf", "random_forest", "xgboost"]


class BalancedXGB(BaseEstimator, ClassifierMixin):
    """XGBoost whose ``scale_pos_weight`` is computed from the TRAINING fold only (class-imbalance handling)."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.05,
                 subsample: float = 0.8, random_state: int = SEED) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BalancedXGB":
        y = np.asarray(y)
        spw = float((y == 0).sum() / max((y == 1).sum(), 1))
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate,
            subsample=self.subsample, scale_pos_weight=spw, random_state=self.random_state,
            eval_metric="logloss", n_jobs=1, verbosity=0,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _clf(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed)
    if name == "svm_rbf":
        return SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2,
                                      random_state=seed, n_jobs=1)
    if name == "xgboost":
        return BalancedXGB(random_state=seed)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def build_model(name: str, seed: int = SEED) -> Pipeline:
    """Impute (median) -> standardise -> classifier. Fixed, un-tuned hyper-parameters (no selection bias)."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", _clf(name, seed))])


def factory(name: str, seed: int = SEED) -> Callable[[], Pipeline]:
    return lambda: build_model(name, seed)
