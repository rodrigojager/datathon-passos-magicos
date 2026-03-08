from __future__ import annotations

from dataclasses import dataclass
import html
import json

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from src.config import DRIFT_REPORT_PATH, DRIFT_SUMMARY_PATH, MODEL_BUNDLE_PATH, ensure_directories
from src.database import record_event, session_scope
from src.db_models import DriftRun, TrainingReferenceRow, TrainingRun
from src.feature_engineering import CATEGORICAL_FEATURES
from src.storage import current_data_version
from src.utils import dataframe_to_records, sanitize_record


class MonitoringError(ValueError):
    pass


@dataclass
class DriftResult:
    drift_run_id: int
    report_path: str
    feature_rows: int
    alert_count: int
    details: list[dict[str, object]]
    data_version: str


def save_training_reference(training_run_id: int, reference_frame: pd.DataFrame, session=None) -> None:
    records = dataframe_to_records(reference_frame)
    if session is not None:
        for record in records:
            session.add(
                TrainingReferenceRow(
                    training_run_id=training_run_id,
                    payload_json=record,
                )
            )
        return

    with session_scope() as new_session:
        for record in records:
            new_session.add(
                TrainingReferenceRow(
                    training_run_id=training_run_id,
                    payload_json=record,
                )
            )


def _load_reference_and_features() -> tuple[pd.DataFrame, list[str], int]:
    if not MODEL_BUNDLE_PATH.exists():
        raise MonitoringError(
            "Bundle de modelo nao encontrado. Execute /train antes de gerar drift."
        )
    with session_scope() as session:
        training_run = session.execute(
            select(TrainingRun).order_by(TrainingRun.id.desc())
        ).scalars().first()
        if training_run is None:
            raise MonitoringError(
                "Base de referencia de treino nao encontrada. Execute /train antes de gerar drift."
            )
        rows = session.execute(
            select(TrainingReferenceRow.payload_json).where(TrainingReferenceRow.training_run_id == training_run.id)
        ).scalars().all()
    reference = pd.DataFrame(rows)
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    return reference, bundle["feature_columns"], training_run.id


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _numeric_drift_row(feature: str, reference: pd.Series, current: pd.Series) -> dict[str, object]:
    ref = pd.to_numeric(reference, errors="coerce")
    cur = pd.to_numeric(current, errors="coerce")
    psi = _psi(ref, cur)
    ref_mean = float(ref.mean()) if ref.notna().any() else 0.0
    cur_mean = float(cur.mean()) if cur.notna().any() else 0.0
    mean_shift = abs(cur_mean - ref_mean)
    alert = psi >= 0.2 or mean_shift >= max(abs(ref_mean) * 0.25, 0.5)
    return {
        "feature": feature,
        "type": "numeric",
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
        "psi": psi,
        "alert": bool(alert),
    }


def _categorical_drift_row(feature: str, reference: pd.Series, current: pd.Series) -> dict[str, object]:
    ref = reference.astype("string").fillna("<NA>")
    cur = current.astype("string").fillna("<NA>")
    ref_dist = ref.value_counts(normalize=True)
    cur_dist = cur.value_counts(normalize=True)
    categories = sorted(set(ref_dist.index) | set(cur_dist.index))
    total_variation = 0.0
    for category in categories:
        total_variation += abs(ref_dist.get(category, 0.0) - cur_dist.get(category, 0.0))
    total_variation *= 0.5
    new_category_rate = float(cur[~cur.isin(ref_dist.index)].shape[0] / max(len(cur), 1))
    alert = total_variation >= 0.2 or new_category_rate >= 0.1
    return {
        "feature": feature,
        "type": "categorical",
        "reference_top": ref.mode().iloc[0] if not ref.mode().empty else "<NA>",
        "current_top": cur.mode().iloc[0] if not cur.mode().empty else "<NA>",
        "distribution_shift": float(total_variation),
        "new_category_rate": new_category_rate,
        "alert": bool(alert),
    }


def _render_html(details: list[dict[str, object]]) -> str:
    rows = []
    for row in details:
        cells = "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in row.values()
        )
        rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in details[0].keys()) if details else ""
    table_rows = "".join(rows)
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Relatório de Drift</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #faf7f0; color: #222; }}
    h1 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #ece6d8; }}
  </style>
</head>
<body>
  <h1>Relatório de Drift</h1>
  <p>Este painel compara a base de referência do treino com o arquivo enviado para monitoramento.</p>
  <table>
    <thead><tr>{headers}</tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>"""


def _export_drift_run(run_id: int) -> None:
    with session_scope() as session:
        run = session.get(DriftRun, run_id)
        if run is None:
            return
        summary = {
            "drift_run_id": run.id,
            "report_path": str(DRIFT_REPORT_PATH),
            "feature_rows": run.feature_rows,
            "alert_count": run.alert_count,
            "details": run.details_json,
            "data_version": run.data_version,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
        report_html = run.report_html
    DRIFT_REPORT_PATH.write_text(report_html, encoding="utf-8")
    DRIFT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_drift_report(
    current_dataframe: pd.DataFrame,
    source_filename: str = "arquivo.csv",
    detected_schema: str = "unknown",
    reference_year: int | None = None,
) -> DriftResult:
    ensure_directories()
    reference, feature_columns, training_run_id = _load_reference_and_features()
    details: list[dict[str, object]] = []

    for feature in feature_columns:
        if feature not in current_dataframe.columns:
            current_dataframe[feature] = pd.NA
        if feature not in reference.columns:
            reference[feature] = pd.NA

        if feature in CATEGORICAL_FEATURES:
            details.append(_categorical_drift_row(feature, reference[feature], current_dataframe[feature]))
        else:
            details.append(_numeric_drift_row(feature, reference[feature], current_dataframe[feature]))

    report_html = _render_html(details)
    data_version = current_data_version()
    effective_year = reference_year
    if effective_year is None and "ano_referencia" in current_dataframe.columns:
        years = pd.to_numeric(current_dataframe["ano_referencia"], errors="coerce").dropna()
        effective_year = int(years.mode().iloc[0]) if not years.empty else 0

    with session_scope() as session:
        drift_run = DriftRun(
            source_file=source_filename,
            detected_schema=detected_schema,
            reference_year=effective_year or 0,
            feature_rows=len(details),
            alert_count=sum(1 for row in details if row["alert"]),
            details_json=[sanitize_record(item) for item in details],
            report_html=report_html,
            data_version=data_version,
        )
        session.add(drift_run)
        session.flush()
        drift_run_id = drift_run.id

    _export_drift_run(drift_run_id)
    record_event(
        "INFO",
        __name__,
        "Analise de drift concluida",
        {
            "drift_run_id": drift_run_id,
            "training_run_id": training_run_id,
            "alert_count": sum(1 for row in details if row["alert"]),
            "feature_rows": len(details),
            "data_version": data_version,
        },
    )
    return DriftResult(
        drift_run_id=drift_run_id,
        report_path=str(DRIFT_REPORT_PATH),
        feature_rows=len(details),
        alert_count=sum(1 for row in details if row["alert"]),
        details=details,
        data_version=data_version,
    )
