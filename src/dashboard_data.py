from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import func, select

from src.config import DRIFT_REPORT_PATH, DRIFT_SUMMARY_PATH, LATEST_EVALUATION_PATH, TRAINING_REPORT_PATH
from src.database import session_scope
from src.db_models import (
    AppEvent,
    BaselineAnalysisRun,
    DriftRun,
    EvaluationRow,
    EvaluationRun,
    StudentRecord,
    TrainingRun,
)
from src.transition_scoring import read_latest_transition_score_summary


def read_training_report() -> dict[str, object]:
    with session_scope() as session:
        run = session.execute(
            select(TrainingRun).order_by(TrainingRun.id.desc())
        ).scalars().first()
        if run is None:
            return {}
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
            "report_path": str(TRAINING_REPORT_PATH),
        }
    TRAINING_REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def read_drift_summary() -> dict[str, object]:
    with session_scope() as session:
        run = session.execute(
            select(DriftRun).order_by(DriftRun.id.desc())
        ).scalars().first()
        if run is None:
            return {}
        payload = {
            "drift_run_id": run.id,
            "report_path": str(DRIFT_REPORT_PATH),
            "feature_rows": run.feature_rows,
            "alert_count": run.alert_count,
            "details": run.details_json,
            "data_version": run.data_version,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
    DRIFT_SUMMARY_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def read_latest_drift_report_html() -> str | None:
    with session_scope() as session:
        run = session.execute(
            select(DriftRun).order_by(DriftRun.id.desc())
        ).scalars().first()
        return None if run is None else run.report_html


def read_latest_baseline_analysis_summary() -> dict[str, object]:
    with session_scope() as session:
        run = session.execute(
            select(BaselineAnalysisRun).order_by(BaselineAnalysisRun.id.desc())
        ).scalars().first()
        if run is None:
            return {}
        return {
            "baseline_analysis_run_id": run.id,
            "training_run_id": run.training_run_id,
            "validation_reference_year": run.validation_reference_year,
            "baseline_metrics": run.baseline_metrics_json,
            "error_counts": run.error_counts_json,
            "false_negative_importances": run.false_negative_importances_json,
            "recovery_importances": run.recovery_importances_json,
            "data_version": run.data_version,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }


def read_latest_evaluation_summary() -> dict[str, object]:
    with session_scope() as session:
        run = session.execute(
            select(EvaluationRun).order_by(EvaluationRun.id.desc())
        ).scalars().first()
        if run is None:
            return {}
        rows = session.execute(
            select(EvaluationRow.payload_json).where(EvaluationRow.evaluation_run_id == run.id)
        ).scalars().all()

    if not rows:
        return {}

    dataframe = pd.DataFrame(rows)
    if run.selected_model_id and "modelo_utilizado" in dataframe.columns and "estrategia_utilizada" in dataframe.columns:
        selected_model_name, _, selected_strategy = run.selected_model_id.partition("__")
        filtered = dataframe.loc[
            (dataframe["modelo_utilizado"] == selected_model_name)
            & (dataframe["estrategia_utilizada"] == selected_strategy)
        ].copy()
        if not filtered.empty:
            dataframe = filtered
    export_path = None
    if not dataframe.empty:
        dataframe.to_csv(LATEST_EVALUATION_PATH, index=False)
        export_path = str(LATEST_EVALUATION_PATH)
    accuracy = float((dataframe["prediction_risco_defasagem_futura"] == dataframe["real_risco_defasagem"]).mean())
    return {
        "evaluation_run_id": run.id,
        "rows": len(dataframe),
        "accuracy": accuracy,
        "positive_predictions": int(dataframe["prediction_risco_defasagem_futura"].sum()),
        "positive_real": int(dataframe["real_risco_defasagem"].sum()),
        "evaluation_mode": run.evaluation_mode,
        "current_year": run.current_year,
        "next_year": run.next_year,
        "data_version": run.data_version,
        "export_path": export_path,
    }


def read_recent_logs(limit: int = 200) -> list[dict[str, str]]:
    with session_scope() as session:
        rows = session.execute(
            select(AppEvent).order_by(AppEvent.id.desc()).limit(limit)
        ).scalars().all()
    parsed = []
    for row in rows:
        parsed.append(
            {
                "timestamp": row.created_at.isoformat() if row.created_at else "",
                "level": row.level,
                "logger": row.logger,
                "message": row.message,
            }
        )
    return list(reversed(parsed))


def build_overview_metrics() -> dict[str, object]:
    training = read_training_report()
    drift = read_drift_summary()
    evaluation = read_latest_evaluation_summary()
    baseline_analysis = read_latest_baseline_analysis_summary()
    transition_scores = read_latest_transition_score_summary()
    with session_scope() as session:
        rows = session.scalar(select(func.count()).select_from(StudentRecord)) or 0
        unique_students = session.scalar(select(func.count(func.distinct(StudentRecord.ra)))) or 0
        reference_years = session.execute(
            select(StudentRecord.ano_referencia).distinct().order_by(StudentRecord.ano_referencia)
        ).scalars().all()
    data_version = training.get("data_version") or drift.get("data_version") or evaluation.get("data_version")
    return {
        "consolidated_rows": int(rows),
        "unique_students": int(unique_students),
        "reference_years": [int(year) for year in reference_years if year is not None],
        "selected_model": training.get("selected_model"),
        "selected_strategy": training.get("selected_strategy"),
        "training_metrics": training.get("metrics", {}),
        "candidate_count": len(training.get("candidates", [])),
        "drift_alert_count": drift.get("alert_count", 0),
        "latest_evaluation": evaluation,
        "baseline_analysis": baseline_analysis,
        "transition_scores": transition_scores,
        "data_version": data_version,
        "training_run_id": training.get("training_run_id"),
        "drift_run_id": drift.get("drift_run_id"),
    }
