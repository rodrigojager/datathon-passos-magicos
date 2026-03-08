from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import select
import json

import pandas as pd

from src.config import TRANSITION_SCORES_PATH
from src.database import record_event, session_scope
from src.db_models import TransitionScoreRun
from src.feature_engineering import CATEGORICAL_FEATURES, STRATEGY_B_FEATURES
from src.utils import sanitize_record


TRANSITION_MODEL_NAME = "random_forest"
TRANSITION_STRATEGY_NAME = "strategy_b_segmented"


@dataclass
class TransitionScoreResult:
    transition_score_run_id: int
    training_run_id: int
    validation_reference_year: int
    strategy_name: str
    entry_metrics: dict[str, float]
    recovery_metrics: dict[str, float]
    data_version: str
    entry_bundle: dict[str, object]
    recovery_bundle: dict[str, object]


def _build_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_numeric_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="constant", fill_value=-1, keep_empty_features=True)
    except TypeError:  # pragma: no cover
        return SimpleImputer(strategy="constant", fill_value=-1)


def _build_categorical_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    except TypeError:  # pragma: no cover
        return SimpleImputer(strategy="most_frequent")


def _build_pipeline(feature_frame: pd.DataFrame, model) -> Pipeline:
    categorical_columns = [column for column in feature_frame.columns if column in CATEGORICAL_FEATURES]
    numeric_columns = [column for column in feature_frame.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", _build_categorical_imputer()),
                        ("encoder", _build_ohe()),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", _build_numeric_imputer()),
                    ]
                ),
                numeric_columns,
            ),
        ],
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _coerce_feature_types(feature_frame: pd.DataFrame) -> pd.DataFrame:
    working = feature_frame.copy()
    numeric_columns = [column for column in working.columns if column not in CATEGORICAL_FEATURES]
    categorical_columns = [column for column in working.columns if column in CATEGORICAL_FEATURES]

    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    for column in categorical_columns:
        working[column] = working[column].astype("string").fillna("<NA>")

    return working


def _safe_average_precision(y_true: pd.Series, scores) -> float:
    if len(pd.Series(y_true).unique()) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _safe_roc_auc(y_true: pd.Series, scores) -> float:
    if len(pd.Series(y_true).unique()) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _compute_metrics(y_true: pd.Series, probabilities, predictions) -> dict[str, float]:
    return {
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "pr_auc": _safe_average_precision(y_true, probabilities),
        "roc_auc": _safe_roc_auc(y_true, probabilities),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def _build_estimator() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )


def _fit_segment_model(
    train_subset: pd.DataFrame,
    validation_subset: pd.DataFrame,
    *,
    target_column: str,
    label_name: str,
) -> tuple[dict[str, float], dict[str, object]]:
    if train_subset.empty or validation_subset.empty:
        return (
            {
                "status": "unavailable",
                "reason": f"Sem dados suficientes para treinar o segmento {label_name}.",
            },
            {
                "model_name": TRANSITION_MODEL_NAME,
                "strategy": TRANSITION_STRATEGY_NAME,
                "label_name": label_name,
                "feature_columns": [column for column in STRATEGY_B_FEATURES if column in train_subset.columns],
                "threshold": 0.5,
                "pipeline": None,
            },
        )

    if train_subset[target_column].nunique() < 2 or validation_subset[target_column].nunique() < 2:
        return (
            {
                "status": "unavailable",
                "reason": f"O segmento {label_name} nao possui classes suficientes para treino e validacao.",
            },
            {
                "model_name": TRANSITION_MODEL_NAME,
                "strategy": TRANSITION_STRATEGY_NAME,
                "label_name": label_name,
                "feature_columns": [column for column in STRATEGY_B_FEATURES if column in train_subset.columns],
                "threshold": 0.5,
                "pipeline": None,
            },
        )

    feature_columns = [column for column in STRATEGY_B_FEATURES if column in train_subset.columns]
    x_train = _coerce_feature_types(train_subset[feature_columns].copy())
    y_train = train_subset[target_column].astype(int)
    x_valid = _coerce_feature_types(validation_subset[feature_columns].copy())
    y_valid = validation_subset[target_column].astype(int)

    pipeline = _build_pipeline(x_train, clone(_build_estimator()))
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = _compute_metrics(y_valid, probabilities, predictions)

    full_subset = pd.concat([train_subset, validation_subset], ignore_index=True)
    x_full = _coerce_feature_types(full_subset[feature_columns].copy())
    y_full = full_subset[target_column].astype(int)
    full_pipeline = _build_pipeline(x_full, clone(_build_estimator()))
    full_pipeline.fit(x_full, y_full)

    bundle = {
        "model_name": TRANSITION_MODEL_NAME,
        "strategy": TRANSITION_STRATEGY_NAME,
        "label_name": label_name,
        "feature_columns": feature_columns,
        "threshold": 0.5,
        "pipeline": full_pipeline,
    }
    return metrics, bundle


def train_transition_scores(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    training_run_id: int,
    validation_reference_year: int,
    data_version: str,
) -> TransitionScoreResult:
    entry_train = train_frame.loc[pd.to_numeric(train_frame["defasagem"], errors="coerce") >= 0].copy()
    entry_valid = validation_frame.loc[pd.to_numeric(validation_frame["defasagem"], errors="coerce") >= 0].copy()
    entry_train["entered_risk_next_year"] = entry_train["target_future_risk"].astype(int)
    entry_valid["entered_risk_next_year"] = entry_valid["target_future_risk"].astype(int)

    recovery_train = train_frame.loc[pd.to_numeric(train_frame["defasagem"], errors="coerce") < 0].copy()
    recovery_valid = validation_frame.loc[pd.to_numeric(validation_frame["defasagem"], errors="coerce") < 0].copy()
    recovery_train["recovered_next_year"] = (1 - recovery_train["target_future_risk"].astype(int)).astype(int)
    recovery_valid["recovered_next_year"] = (1 - recovery_valid["target_future_risk"].astype(int)).astype(int)

    entry_metrics, entry_bundle = _fit_segment_model(
        entry_train,
        entry_valid,
        target_column="entered_risk_next_year",
        label_name="entrada_em_risco",
    )
    recovery_metrics, recovery_bundle = _fit_segment_model(
        recovery_train,
        recovery_valid,
        target_column="recovered_next_year",
        label_name="recuperacao",
    )

    with session_scope() as session:
        run = TransitionScoreRun(
            training_run_id=training_run_id,
            validation_reference_year=validation_reference_year,
            entry_model_name=TRANSITION_MODEL_NAME,
            recovery_model_name=TRANSITION_MODEL_NAME,
            strategy_name=TRANSITION_STRATEGY_NAME,
            entry_metrics_json=sanitize_record(entry_metrics),
            recovery_metrics_json=sanitize_record(recovery_metrics),
            data_version=data_version,
        )
        session.add(run)
        session.flush()
        run_id = run.id

    record_event(
        "INFO",
        __name__,
        "Scores de transicao treinados",
        {
            "transition_score_run_id": run_id,
            "training_run_id": training_run_id,
            "validation_reference_year": validation_reference_year,
            "data_version": data_version,
        },
    )
    return TransitionScoreResult(
        transition_score_run_id=run_id,
        training_run_id=training_run_id,
        validation_reference_year=validation_reference_year,
        strategy_name=TRANSITION_STRATEGY_NAME,
        entry_metrics=entry_metrics,
        recovery_metrics=recovery_metrics,
        data_version=data_version,
        entry_bundle=entry_bundle,
        recovery_bundle=recovery_bundle,
    )


def read_latest_transition_score_summary() -> dict[str, object]:
    with session_scope() as session:
        run = session.execute(
            select(TransitionScoreRun).order_by(TransitionScoreRun.id.desc())
        ).scalars().first()
        if run is None:
            return {}
        payload = {
            "transition_score_run_id": run.id,
            "training_run_id": run.training_run_id,
            "validation_reference_year": run.validation_reference_year,
            "entry_model_name": run.entry_model_name,
            "recovery_model_name": run.recovery_model_name,
            "strategy_name": run.strategy_name,
            "entry_metrics": run.entry_metrics_json,
            "recovery_metrics": run.recovery_metrics_json,
            "data_version": run.data_version,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
    TRANSITION_SCORES_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
