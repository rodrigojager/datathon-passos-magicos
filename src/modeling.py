from __future__ import annotations

from dataclasses import dataclass
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

from src.config import (
    CANDIDATE_BUNDLES_PATH,
    LATEST_EVALUATION_PATH,
    MODEL_BUNDLE_PATH,
    PREDICTIONS_DIR,
    TRAINING_REPORT_PATH,
    VALIDATION_CANDIDATE_BUNDLES_PATH,
    ensure_directories,
)
from src.baseline_analysis import analyze_baseline_residuals
from src.database import record_event, session_scope
from src.db_models import EvaluationRow, EvaluationRun, PredictionRow, PredictionRun, TrainingRun
from src.feature_engineering import CATEGORICAL_FEATURES, STRATEGY_A_FEATURES, STRATEGY_B_FEATURES
from src.monitoring import save_training_reference
from src.storage import current_data_version, load_consolidated_dataframe
from src.transition_scoring import train_transition_scores
from src.utils import dataframe_to_records, get_logger, sanitize_record


logger = get_logger(__name__)


MODEL_CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        random_state=42,
        max_depth=6,
    ),
}


class ModelingError(ValueError):
    pass


@dataclass
class TrainingResult:
    selected_model: str
    selected_strategy: str
    metrics: dict[str, float]
    candidates: list[dict[str, object]]
    training_rows: int
    validation_rows: int
    training_run_id: int
    baseline_analysis_run_id: int
    transition_score_run_id: int
    transition_scores: dict[str, object]
    data_version: str


def candidate_model_id(model_name: str, strategy: str) -> str:
    return f"{model_name}__{strategy}"


def _safe_average_precision(y_true: pd.Series, scores: np.ndarray) -> float:
    if len(pd.Series(y_true).unique()) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _safe_roc_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    if len(pd.Series(y_true).unique()) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


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


def _prepare_model_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    required_columns = [
        "ra",
        "ano_referencia",
        "defasagem",
        "fase",
        "turma",
        "idade",
        "genero",
        "ano_ingresso",
        "instituicao_ensino",
        "numero_avaliacoes",
        "inde_atual",
        "iaa",
        "ieg",
        "ips",
        "ipp",
        "ida",
        "nota_matematica",
        "nota_portugues",
        "nota_ingles",
        "ipv",
        "ian",
        "fase_ideal",
    ]
    for column in required_columns:
        if column not in df.columns:
            df[column] = pd.NA

    df["flag_ipp_ausente"] = df["ipp"].isna().astype(int)

    next_year = df[["ra", "ano_referencia", "defasagem"]].copy()
    next_year["ano_referencia"] = pd.to_numeric(next_year["ano_referencia"], errors="coerce")
    next_year["ano_referencia"] = next_year["ano_referencia"] - 1
    next_year = next_year.rename(columns={"defasagem": "defasagem_futura"})

    merged = df.merge(next_year, on=["ra", "ano_referencia"], how="left")
    merged["defasagem_futura"] = pd.to_numeric(merged["defasagem_futura"], errors="coerce")
    merged = merged.loc[merged["defasagem_futura"].notna()].copy()
    merged["target_future_risk"] = (merged["defasagem_futura"] < 0).astype(int)
    merged["ano_referencia"] = pd.to_numeric(merged["ano_referencia"], errors="coerce")
    numeric_feature_columns = set(STRATEGY_B_FEATURES) - CATEGORICAL_FEATURES
    for column in numeric_feature_columns | {"defasagem"}:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


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


def _compute_metrics(y_true: pd.Series, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "pr_auc": _safe_average_precision(y_true, probabilities),
        "roc_auc": _safe_roc_auc(y_true, probabilities),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def _export_training_report(run_id: int) -> None:
    with session_scope() as session:
        run = session.get(TrainingRun, run_id)
        if run is None:
            return
        payload = {
            "training_run_id": run.id,
            "selected_model": run.selected_model,
            "selected_strategy": run.selected_strategy,
            "metrics": run.metrics_json,
            "training_rows": run.training_rows,
            "validation_rows": run.validation_rows,
            "candidates": run.candidates_json,
            "data_version": run.data_version,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
    TRAINING_REPORT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train_and_select_model() -> TrainingResult:
    ensure_directories()
    consolidated = load_consolidated_dataframe()
    model_frame = _prepare_model_frame(consolidated)

    if model_frame.empty:
        raise ModelingError("Nao ha pares temporais suficientes para treinar o modelo.")

    year_min = model_frame["ano_referencia"].min()
    year_max = model_frame["ano_referencia"].max()
    train_frame = model_frame.loc[model_frame["ano_referencia"] == year_min].copy()
    validation_frame = model_frame.loc[model_frame["ano_referencia"] == year_max].copy()

    if train_frame.empty or validation_frame.empty:
        raise ModelingError(
            "Nao foi possivel montar treino e validacao temporal. "
            "Verifique se ha pelo menos dois anos consecutivos consolidados."
        )

    y_valid = validation_frame["target_future_risk"].astype(int)
    candidates: list[dict[str, object]] = []

    baseline_scores = (pd.to_numeric(validation_frame["defasagem"], errors="coerce") < 0).astype(int)
    baseline_metrics = _compute_metrics(
        y_valid,
        baseline_scores.to_numpy(dtype=float),
        baseline_scores.to_numpy(dtype=int),
    )
    candidates.append(
        {
            "model_name": "baseline_persistencia",
            "strategy": "baseline_current_defasagem",
            "metrics": baseline_metrics,
        }
    )

    strategy_map = {
        "strategy_a": STRATEGY_A_FEATURES,
        "strategy_b": STRATEGY_B_FEATURES,
    }

    best_result: dict[str, object] | None = None
    validation_candidate_bundles: dict[str, dict[str, object]] = {
        candidate_model_id("baseline_persistencia", "baseline_current_defasagem"): {
            "kind": "baseline",
            "model_name": "baseline_persistencia",
            "strategy": "baseline_current_defasagem",
            "feature_columns": ["defasagem"],
            "validation_reference_year": int(year_max),
        }
    }

    for strategy_name, feature_columns in strategy_map.items():
        for model_name, estimator in MODEL_CANDIDATES.items():
            x_train = _coerce_feature_types(train_frame[feature_columns].copy())
            y_train = train_frame["target_future_risk"].astype(int)
            x_valid = _coerce_feature_types(validation_frame[feature_columns].copy())

            pipeline = _build_pipeline(x_train, clone(estimator))
            pipeline.fit(x_train, y_train)

            probabilities = pipeline.predict_proba(x_valid)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            metrics = _compute_metrics(y_valid, probabilities, predictions)
            result = {
                "model_name": model_name,
                "strategy": strategy_name,
                "metrics": metrics,
            }
            candidates.append(result)
            validation_candidate_bundles[candidate_model_id(model_name, strategy_name)] = {
                "kind": "pipeline",
                "model_name": model_name,
                "strategy": strategy_name,
                "feature_columns": feature_columns,
                "threshold": 0.5,
                "pipeline": pipeline,
                "validation_reference_year": int(year_max),
            }

            if best_result is None or metrics["f2"] > best_result["metrics"]["f2"]:
                best_result = result

    if best_result is None:
        raise ModelingError("Nao foi possivel selecionar um modelo vencedor.")

    full_training_frame = model_frame.copy()
    y_full = full_training_frame["target_future_risk"].astype(int)
    candidate_bundles: dict[str, dict[str, object]] = {
        candidate_model_id("baseline_persistencia", "baseline_current_defasagem"): {
            "kind": "baseline",
            "model_name": "baseline_persistencia",
            "strategy": "baseline_current_defasagem",
            "feature_columns": ["defasagem"],
        }
    }

    for strategy_name, feature_columns in strategy_map.items():
        x_full = _coerce_feature_types(full_training_frame[feature_columns].copy())
        for model_name, estimator in MODEL_CANDIDATES.items():
            pipeline = _build_pipeline(x_full, clone(estimator))
            pipeline.fit(x_full, y_full)
            candidate_bundles[candidate_model_id(model_name, strategy_name)] = {
                "kind": "pipeline",
                "model_name": model_name,
                "strategy": strategy_name,
                "feature_columns": feature_columns,
                "threshold": 0.5,
                "pipeline": pipeline,
            }

    best_bundle = candidate_bundles[candidate_model_id(best_result["model_name"], best_result["strategy"])]
    best_pipeline = best_bundle["pipeline"]
    best_features = best_bundle["feature_columns"]
    x_best = _coerce_feature_types(full_training_frame[best_features].copy())
    best_pipeline.fit(x_best, y_full)

    bundle = {
        "pipeline": best_pipeline,
        "selected_model": best_result["model_name"],
        "selected_strategy": best_result["strategy"],
        "feature_columns": best_features,
        "threshold": 0.5,
        "transition_scores": None,
    }
    joblib.dump(candidate_bundles, CANDIDATE_BUNDLES_PATH)
    joblib.dump(validation_candidate_bundles, VALIDATION_CANDIDATE_BUNDLES_PATH)

    data_version = current_data_version()
    with session_scope() as session:
        training_run = TrainingRun(
            selected_model=best_result["model_name"],
            selected_strategy=best_result["strategy"],
            metrics_json=sanitize_record(best_result["metrics"]),
            candidates_json=[sanitize_record(candidate) for candidate in candidates],
            selected_feature_columns_json=list(best_features),
            training_rows=len(train_frame),
            validation_rows=len(validation_frame),
            data_version=data_version,
        )
        session.add(training_run)
        session.flush()
        save_training_reference(training_run.id, x_best, session=session)
        training_run_id = training_run.id

    _export_training_report(training_run_id)
    baseline_analysis = analyze_baseline_residuals(
        validation_frame,
        training_run_id=training_run_id,
        validation_reference_year=int(year_max),
        data_version=data_version,
    )
    transition_scores = train_transition_scores(
        train_frame,
        validation_frame,
        training_run_id=training_run_id,
        validation_reference_year=int(year_max),
        data_version=data_version,
    )
    bundle["transition_scores"] = {
        "entry_bundle": transition_scores.entry_bundle,
        "recovery_bundle": transition_scores.recovery_bundle,
        "strategy_name": transition_scores.strategy_name,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    training_result = TrainingResult(
        selected_model=best_result["model_name"],
        selected_strategy=best_result["strategy"],
        metrics=best_result["metrics"],
        candidates=candidates,
        training_rows=len(train_frame),
        validation_rows=len(validation_frame),
        training_run_id=training_run_id,
        baseline_analysis_run_id=baseline_analysis.baseline_analysis_run_id,
        transition_score_run_id=transition_scores.transition_score_run_id,
        transition_scores={
            "strategy_name": transition_scores.strategy_name,
            "entry_metrics": transition_scores.entry_metrics,
            "recovery_metrics": transition_scores.recovery_metrics,
        },
        data_version=data_version,
    )
    logger.info(
        "Treinamento concluido | modelo_selecionado=%s | estrategia=%s | f2=%.4f",
        training_result.selected_model,
        training_result.selected_strategy,
        training_result.metrics["f2"],
    )
    record_event(
        "INFO",
        __name__,
        "Treinamento concluido",
        {
            "training_run_id": training_run_id,
            "selected_model": training_result.selected_model,
            "selected_strategy": training_result.selected_strategy,
            "f2": training_result.metrics["f2"],
            "data_version": data_version,
            "baseline_analysis_run_id": training_result.baseline_analysis_run_id,
            "transition_score_run_id": training_result.transition_score_run_id,
        },
    )
    return training_result


def _load_bundle() -> dict[str, object]:
    if not MODEL_BUNDLE_PATH.exists():
        raise ModelingError(
            "Modelo treinado nao encontrado. Execute o endpoint /train antes de usar /predict."
        )
    return joblib.load(MODEL_BUNDLE_PATH)


def load_candidate_bundles() -> dict[str, dict[str, object]]:
    if not CANDIDATE_BUNDLES_PATH.exists():
        raise ModelingError(
            "Modelos candidatos nao encontrados. Execute o endpoint /train antes de usar esta funcionalidade."
        )
    return joblib.load(CANDIDATE_BUNDLES_PATH)


def load_validation_candidate_bundles() -> dict[str, dict[str, object]]:
    if not VALIDATION_CANDIDATE_BUNDLES_PATH.exists():
        raise ModelingError(
            "Modelos candidatos de validacao nao encontrados. Execute o endpoint /train antes de usar esta funcionalidade."
        )
    return joblib.load(VALIDATION_CANDIDATE_BUNDLES_PATH)


def _predict_with_bundle(dataframe: pd.DataFrame, bundle: dict[str, object]) -> pd.DataFrame:
    feature_columns: list[str] = bundle["feature_columns"]
    working = dataframe.copy()
    for column in feature_columns:
        if column not in working.columns:
            working[column] = pd.NA

    features = _coerce_feature_types(working[feature_columns].copy())
    probabilities = bundle["pipeline"].predict_proba(features)[:, 1]
    threshold = float(bundle["threshold"])
    predictions = (probabilities >= threshold).astype(int)

    output = working[["ra", "ano_referencia", "source_schema", "source_file"]].copy()
    output["prediction_risco_defasagem_futura"] = predictions
    output["probabilidade_risco_defasagem_futura"] = probabilities
    output["threshold"] = threshold
    output["modelo_utilizado"] = bundle.get("selected_model", bundle.get("model_name"))
    output["estrategia_utilizada"] = bundle.get("selected_strategy", bundle.get("strategy"))
    return output


def _persist_prediction_run(
    predictions: pd.DataFrame,
    source_filename: str,
    detected_schema: str,
    reference_year: int,
    selected_model: str,
    selected_strategy: str,
) -> int:
    data_version = current_data_version()
    with session_scope() as session:
        prediction_run = PredictionRun(
            source_file=source_filename,
            detected_schema=detected_schema,
            reference_year=reference_year,
            selected_model=selected_model,
            selected_strategy=selected_strategy,
            row_count=len(predictions),
            data_version=data_version,
        )
        session.add(prediction_run)
        session.flush()
        for record in dataframe_to_records(predictions):
            session.add(PredictionRow(prediction_run_id=prediction_run.id, payload_json=record))
        return prediction_run.id


def _resolve_prediction_bundle(selected_model_id: str | None) -> dict[str, object]:
    if not selected_model_id:
        return _load_bundle()

    candidate_bundles = load_candidate_bundles()
    selected_bundle = candidate_bundles.get(selected_model_id)
    if selected_bundle is None:
        raise ModelingError(
            f"Modelo solicitado '{selected_model_id}' nao encontrado. Execute /train ou selecione um candidato valido."
        )

    return dict(selected_bundle)


def predict_from_normalized_dataframe(
    dataframe: pd.DataFrame,
    source_filename: str = "arquivo.csv",
    detected_schema: str = "unknown",
    reference_year: int | None = None,
    selected_model_id: str | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    primary_bundle = _resolve_prediction_bundle(selected_model_id)
    if primary_bundle.get("kind") == "baseline":
        predictions = _predict_with_candidate_bundle(dataframe, primary_bundle)
    else:
        predictions = _predict_with_bundle(dataframe, primary_bundle)
    working = dataframe.copy()

    defasagem_atual = pd.to_numeric(working.get("defasagem"), errors="coerce")
    predictions["atualmente_defasado"] = (defasagem_atual < 0).where(defasagem_atual.notna(), pd.NA)
    predictions["grupo_baseline_atual"] = pd.Series(pd.NA, index=predictions.index, dtype="object")
    predictions.loc[predictions["atualmente_defasado"] == True, "grupo_baseline_atual"] = "ja_defasado"
    predictions.loc[predictions["atualmente_defasado"] == False, "grupo_baseline_atual"] = "nao_defasado"

    predictions["score_entrada_em_risco"] = pd.NA
    predictions["score_recuperacao"] = pd.NA
    predictions["score_persistencia_defasagem"] = pd.NA
    predictions["score_transicao_relevante"] = pd.NA
    predictions["tipo_transicao_relevante"] = pd.NA

    operational_bundle = _load_bundle()
    transition_bundles = operational_bundle.get("transition_scores")
    if transition_bundles:
        entry_bundle = transition_bundles.get("entry_bundle")
        recovery_bundle = transition_bundles.get("recovery_bundle")

        if entry_bundle is not None:
            entry_mask = predictions["grupo_baseline_atual"] == "nao_defasado"
            if bool(entry_mask.any()) and entry_bundle.get("pipeline") is not None:
                entry_features = working.loc[entry_mask].copy()
                for column in entry_bundle["feature_columns"]:
                    if column not in entry_features.columns:
                        entry_features[column] = pd.NA
                x_entry = _coerce_feature_types(entry_features[entry_bundle["feature_columns"]].copy())
                entry_scores = entry_bundle["pipeline"].predict_proba(x_entry)[:, 1]
                predictions.loc[entry_mask, "score_entrada_em_risco"] = entry_scores
                predictions.loc[entry_mask, "score_transicao_relevante"] = entry_scores
                predictions.loc[entry_mask, "tipo_transicao_relevante"] = "entrada_em_risco"

        if recovery_bundle is not None:
            recovery_mask = predictions["grupo_baseline_atual"] == "ja_defasado"
            if bool(recovery_mask.any()) and recovery_bundle.get("pipeline") is not None:
                recovery_features = working.loc[recovery_mask].copy()
                for column in recovery_bundle["feature_columns"]:
                    if column not in recovery_features.columns:
                        recovery_features[column] = pd.NA
                x_recovery = _coerce_feature_types(recovery_features[recovery_bundle["feature_columns"]].copy())
                recovery_scores = recovery_bundle["pipeline"].predict_proba(x_recovery)[:, 1]
                predictions.loc[recovery_mask, "score_recuperacao"] = recovery_scores
                predictions.loc[recovery_mask, "score_persistencia_defasagem"] = 1 - recovery_scores
                predictions.loc[recovery_mask, "score_transicao_relevante"] = recovery_scores
                predictions.loc[recovery_mask, "tipo_transicao_relevante"] = "recuperacao"

    if persist:
        selected_model = str(primary_bundle.get("selected_model", primary_bundle.get("model_name")))
        selected_strategy = str(primary_bundle.get("selected_strategy", primary_bundle.get("strategy")))
        prediction_run_id = _persist_prediction_run(
            predictions,
            source_filename=source_filename,
            detected_schema=detected_schema,
            reference_year=reference_year or int(
                pd.to_numeric(dataframe.get("ano_referencia"), errors="coerce").dropna().mode().iloc[0]
            ),
            selected_model=selected_model,
            selected_strategy=selected_strategy,
        )
        record_event(
            "INFO",
            __name__,
            "Predicao concluida",
            {
                "prediction_run_id": prediction_run_id,
                "rows": len(predictions),
                "selected_model": selected_model,
                "selected_strategy": selected_strategy,
            },
        )

    selected_model = str(primary_bundle.get("selected_model", primary_bundle.get("model_name")))
    selected_strategy = str(primary_bundle.get("selected_strategy", primary_bundle.get("strategy")))
    logger.info(
        "Predicao concluida | linhas=%s | modelo=%s | estrategia=%s",
        len(predictions),
        selected_model,
        selected_strategy,
    )
    return predictions.where(pd.notna(predictions), None)


def _predict_with_candidate_bundle(
    dataframe: pd.DataFrame,
    candidate_bundle: dict[str, object],
) -> pd.DataFrame:
    working = dataframe.copy()
    if candidate_bundle["kind"] == "baseline":
        scores = (pd.to_numeric(working["defasagem"], errors="coerce") < 0).astype(float)
        predictions = scores.astype(int)
        model_name = candidate_bundle.get("model_name", candidate_bundle.get("selected_model"))
        strategy = candidate_bundle.get("strategy", candidate_bundle.get("selected_strategy"))
        threshold = 0.5
    else:
        feature_columns: list[str] = candidate_bundle["feature_columns"]
        for column in feature_columns:
            if column not in working.columns:
                working[column] = pd.NA
        features = _coerce_feature_types(working[feature_columns].copy())
        scores = candidate_bundle["pipeline"].predict_proba(features)[:, 1]
        threshold = float(candidate_bundle["threshold"])
        predictions = (scores >= threshold).astype(int)
        model_name = candidate_bundle.get("model_name", candidate_bundle.get("selected_model"))
        strategy = candidate_bundle.get("strategy", candidate_bundle.get("selected_strategy"))

    output = working[["ra", "ano_referencia", "source_schema", "source_file"]].copy()
    output["prediction_risco_defasagem_futura"] = predictions
    output["probabilidade_risco_defasagem_futura"] = scores
    output["threshold"] = threshold
    output["modelo_utilizado"] = model_name
    output["estrategia_utilizada"] = strategy
    return output


def _export_evaluation_rows(run_id: int, model_id: str | None, output_path) -> str:
    with session_scope() as session:
        query = select(EvaluationRow.payload_json).where(EvaluationRow.evaluation_run_id == run_id)
        if model_id is not None:
            query = query.where(EvaluationRow.model_id == model_id)
        rows = session.execute(query).scalars().all()
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe.to_csv(output_path, index=False)
    else:
        output_path.write_text("", encoding="utf-8")
    return str(output_path)


def _persist_evaluation_run(
    *,
    source_file: str,
    detected_schema: str,
    current_year: int,
    next_year: int,
    evaluation_mode: str,
    selected_model_id: str | None,
    metrics: dict[str, object] | None,
    summaries: list[dict[str, object]] | None,
    detailed_payloads: list[tuple[str, dict[str, object]]],
) -> int:
    data_version = current_data_version()
    with session_scope() as session:
        evaluation_run = EvaluationRun(
            source_file=source_file,
            detected_schema=detected_schema,
            current_year=current_year,
            next_year=next_year,
            evaluation_mode=evaluation_mode,
            selected_model_id=selected_model_id,
            metrics_json=None if metrics is None else sanitize_record(metrics),
            summaries_json=None if summaries is None else [sanitize_record(item) for item in summaries],
            compared_rows=len(detailed_payloads),
            data_version=data_version,
        )
        session.add(evaluation_run)
        session.flush()
        for model_id, payload in detailed_payloads:
            session.add(
                EvaluationRow(
                    evaluation_run_id=evaluation_run.id,
                    model_id=model_id,
                    payload_json=sanitize_record(payload),
                )
            )
        return evaluation_run.id


def evaluate_predictions_against_next_year(
    current_dataframe: pd.DataFrame,
    next_year_dataframe: pd.DataFrame,
    source_filename: str = "comparacao.csv",
    detected_schema: str = "uploaded_pair",
) -> dict[str, object]:
    predictions = predict_from_normalized_dataframe(
        current_dataframe,
        source_filename=source_filename,
        detected_schema=detected_schema,
        persist=False,
    )
    actual = next_year_dataframe[["ra", "defasagem"]].copy()
    actual["real_risco_defasagem"] = (pd.to_numeric(actual["defasagem"], errors="coerce") < 0).astype("Int64")
    merged = predictions.merge(
        actual[["ra", "real_risco_defasagem"]],
        on="ra",
        how="inner",
    )
    if merged.empty:
        raise ModelingError("Nao houve interseccao de alunos entre os arquivos de avaliacao.")

    y_true = merged["real_risco_defasagem"].astype(int)
    y_pred = merged["prediction_risco_defasagem_futura"].astype(int)
    y_score = merged["probabilidade_risco_defasagem_futura"].astype(float)
    metrics = _compute_metrics(y_true, y_score.to_numpy(), y_pred.to_numpy())

    merged["acerto"] = (
        merged["real_risco_defasagem"].astype(int)
        == merged["prediction_risco_defasagem_futura"].astype(int)
    )
    model_id = candidate_model_id(
        predictions["modelo_utilizado"].iloc[0],
        predictions["estrategia_utilizada"].iloc[0],
    )
    current_year = int(pd.to_numeric(current_dataframe["ano_referencia"], errors="coerce").dropna().mode().iloc[0])
    next_year = int(pd.to_numeric(next_year_dataframe["ano_referencia"], errors="coerce").dropna().mode().iloc[0])

    evaluation_run_id = _persist_evaluation_run(
        source_file=source_filename,
        detected_schema=detected_schema,
        current_year=current_year,
        next_year=next_year,
        evaluation_mode="uploaded_pair",
        selected_model_id=model_id,
        metrics=metrics,
        summaries=None,
        detailed_payloads=[(model_id, row) for row in dataframe_to_records(merged)],
    )
    output_file = _export_evaluation_rows(evaluation_run_id, model_id, LATEST_EVALUATION_PATH)
    logger.info(
        "Avaliacao concluida | linhas_comparadas=%s | f2=%.4f",
        len(merged),
        metrics["f2"],
    )
    record_event(
        "INFO",
        __name__,
        "Avaliacao contra o ano seguinte concluida",
        {
            "evaluation_run_id": evaluation_run_id,
            "compared_rows": len(merged),
            "f2": metrics["f2"],
            "model_id": model_id,
        },
    )
    return {
        "evaluation_run_id": evaluation_run_id,
        "metrics": metrics,
        "compared_rows": len(merged),
        "output_file": output_file,
        "preview": merged.head(10).to_dict(orient="records"),
    }


def evaluate_uploaded_file_against_consolidated(
    current_dataframe: pd.DataFrame,
    source_filename: str = "arquivo.csv",
    detected_schema: str = "unknown",
    selected_model_id: str | None = None,
) -> dict[str, object]:
    consolidated = load_consolidated_dataframe()
    if consolidated.empty:
        raise ModelingError("Base consolidada nao encontrada. Execute /data/bootstrap ou /data/ingest.")

    if "ano_referencia" not in current_dataframe.columns or current_dataframe["ano_referencia"].dropna().empty:
        raise ModelingError("O arquivo enviado nao permite inferir o ano de referencia.")

    current_year = int(pd.to_numeric(current_dataframe["ano_referencia"], errors="coerce").dropna().mode().iloc[0])
    next_year = current_year + 1
    actual = consolidated.loc[pd.to_numeric(consolidated["ano_referencia"], errors="coerce") == next_year].copy()
    if actual.empty:
        raise ModelingError(
            f"Nao ha dados consolidados do ano seguinte ({next_year}) para comparar previsto vs real."
        )

    candidate_bundles = load_validation_candidate_bundles()
    summaries: list[dict[str, object]] = []
    detailed_rows: list[dict[str, object]] = []
    detailed_payloads: list[tuple[str, dict[str, object]]] = []
    detailed_model_id = selected_model_id

    final_bundle = _load_bundle()
    if detailed_model_id is None:
        detailed_model_id = candidate_model_id(
            final_bundle["selected_model"],
            final_bundle["selected_strategy"],
        )

    for model_id, bundle in candidate_bundles.items():
        if bundle.get("validation_reference_year") not in (None, current_year):
            continue
        predictions = _predict_with_candidate_bundle(current_dataframe, bundle)
        merged = predictions.merge(
            actual[["ra", "defasagem"]],
            on="ra",
            how="inner",
        )
        if merged.empty:
            continue

        merged["real_risco_defasagem"] = (pd.to_numeric(merged["defasagem"], errors="coerce") < 0).astype(int)
        y_true = merged["real_risco_defasagem"].astype(int)
        y_pred = merged["prediction_risco_defasagem_futura"].astype(int)
        y_score = merged["probabilidade_risco_defasagem_futura"].astype(float)
        metrics = _compute_metrics(y_true, y_score.to_numpy(), y_pred.to_numpy())

        summary = {
            "model_id": model_id,
            "model_name": bundle["model_name"],
            "strategy": bundle["strategy"],
            "compared_rows": len(merged),
            **metrics,
        }
        summaries.append(summary)

        merged["acerto"] = (
            merged["real_risco_defasagem"].astype(int)
            == merged["prediction_risco_defasagem_futura"].astype(int)
        )
        serialized_rows = dataframe_to_records(merged)
        detailed_payloads.extend((model_id, row) for row in serialized_rows)

        if model_id == detailed_model_id:
            detailed_rows = serialized_rows[:250]

    if not summaries:
        raise ModelingError("Nao houve interseccao entre o arquivo enviado e o consolidado do ano seguinte.")

    summaries = sorted(summaries, key=lambda item: item["f2"], reverse=True)
    evaluation_run_id = _persist_evaluation_run(
        source_file=source_filename,
        detected_schema=detected_schema,
        current_year=current_year,
        next_year=next_year,
        evaluation_mode="validation_frozen_candidates",
        selected_model_id=detailed_model_id,
        metrics=None,
        summaries=summaries,
        detailed_payloads=detailed_payloads,
    )

    enriched_summaries: list[dict[str, object]] = []
    for summary in summaries:
        output_path = PREDICTIONS_DIR / f"evaluation_{summary['model_id']}_{current_year}_vs_{next_year}.csv"
        output_file = _export_evaluation_rows(evaluation_run_id, summary["model_id"], output_path)
        enriched = dict(summary)
        enriched["output_file"] = output_file
        enriched_summaries.append(enriched)

    with session_scope() as session:
        run = session.get(EvaluationRun, evaluation_run_id)
        if run is not None:
            run.summaries_json = [sanitize_record(item) for item in enriched_summaries]

    record_event(
        "INFO",
        __name__,
        "Candidatos congelados de validacao avaliados",
        {
            "evaluation_run_id": evaluation_run_id,
            "current_year": current_year,
            "next_year": next_year,
            "models_compared": len(enriched_summaries),
            "selected_model_id": detailed_model_id,
        },
    )
    return {
        "evaluation_run_id": evaluation_run_id,
        "current_year": current_year,
        "next_year": next_year,
        "evaluation_mode": "validation_frozen_candidates",
        "selected_model_id": detailed_model_id,
        "summaries": enriched_summaries,
        "detailed_rows": detailed_rows,
    }
