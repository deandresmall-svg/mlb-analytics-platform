from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Metrics:
    rows: int
    brier: float | None = None
    roc_auc: float | None = None
    log_loss: float | None = None
    accuracy: float | None = None
    mae: float | None = None


class _FeatureAwareModel:
    def __init__(self) -> None:
        self.meta: dict = {}

    @property
    def trained_features(self) -> list[str]:
        value = self.meta.get("features", [])
        return list(value) if isinstance(value, (list, tuple)) else []

    def feature_frame(
        self,
        frame: pd.DataFrame,
        fallback_features: list[str],
    ) -> pd.DataFrame:
        features = self.trained_features or list(fallback_features)
        output = frame.copy()
        for feature in features:
            if feature not in output.columns:
                output[feature] = np.nan
        return output[features]


class BinaryTimeModel(_FeatureAwareModel):
    def __init__(self, method: str = "sigmoid") -> None:
        super().__init__()
        base = Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self.model = CalibratedClassifierCV(base, method=method, cv=3)

    def fit(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
        label: str,
    ) -> Metrics:
        data = dataframe.dropna(subset=[label]).sort_values("game_date")
        cut = max(int(len(data) * 0.8), 1)
        train, test = data.iloc[:cut], data.iloc[cut:]
        if len(test) < 10:
            raise ValueError("Not enough chronological holdout rows")
        self.model.fit(train[features], train[label])
        probability = self.model.predict_proba(test[features])[:, 1]
        self.meta["features"] = list(features)
        return Metrics(
            rows=len(data),
            brier=brier_score_loss(test[label], probability),
            roc_auc=(
                roc_auc_score(test[label], probability)
                if test[label].nunique() > 1
                else None
            ),
            log_loss=log_loss(test[label], probability),
            accuracy=accuracy_score(test[label], probability >= 0.5),
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

    def predict_frame(
        self,
        frame: pd.DataFrame,
        fallback_features: list[str],
    ) -> np.ndarray:
        return self.predict(self.feature_frame(frame, fallback_features))

    def save(self, path: Path, meta: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        merged_meta = {**self.meta, **meta}
        joblib.dump({"model": self.model, "meta": merged_meta}, path)

    @classmethod
    def load(cls, path: Path) -> "BinaryTimeModel":
        output = cls()
        payload = joblib.load(path)
        output.model = payload["model"]
        output.meta = payload.get("meta", {}) or {}
        return output


class CountTimeModel(_FeatureAwareModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                (
                    "model",
                    PoissonRegressor(alpha=0.2, max_iter=2000),
                ),
            ]
        )

    def fit(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
        label: str,
    ) -> Metrics:
        data = dataframe.dropna(subset=[label]).sort_values("game_date")
        cut = max(int(len(data) * 0.8), 1)
        train, test = data.iloc[:cut], data.iloc[cut:]
        if len(test) < 10:
            raise ValueError("Not enough chronological holdout rows")
        self.model.fit(train[features], train[label])
        prediction = np.maximum(self.model.predict(test[features]), 0)
        self.meta["features"] = list(features)
        return Metrics(
            rows=len(data),
            mae=mean_absolute_error(test[label], prediction),
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.model.predict(features), 0)

    def predict_frame(
        self,
        frame: pd.DataFrame,
        fallback_features: list[str],
    ) -> np.ndarray:
        return self.predict(self.feature_frame(frame, fallback_features))

    def save(self, path: Path, meta: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        merged_meta = {**self.meta, **meta}
        joblib.dump({"model": self.model, "meta": merged_meta}, path)

    @classmethod
    def load(cls, path: Path) -> "CountTimeModel":
        output = cls()
        payload = joblib.load(path)
        output.model = payload["model"]
        output.meta = payload.get("meta", {}) or {}
        return output
