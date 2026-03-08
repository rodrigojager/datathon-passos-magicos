from __future__ import annotations

from dataclasses import dataclass
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import select

from src.config import BASELINE_ANALYSIS_PATH, ensure_directories
from src.database import record_event, session_scope
from src.db_models import BaselineAnalysisRun
from src.feature_engineering import CATEGORICAL_FEATURES, STRATEGY_B_FEATURES
from src.utils import sanitize_record, sanitize_scalar


NUMERIC_SUMMARY_COLUMNS = [
    "inde_atual",
    "ipv",
    "ian",
    "ipp",
    "ida",
    "nota_matematica",
    "nota_portugues",
    "nota_ingles",
    "ieg",
    "ips",
    "iaa",
    "numero_avaliacoes",
    "idade",
]

CATEGORICAL_SUMMARY_COLUMNS = [
    "fase",
    "fase_ideal",
    "instituicao_ensino",
    "genero",
]

ACTIONABLE_FEATURES = {
    "ida": {
        "label": "IDA",
        "higher_is_better": True,
        "action_hint": "reforço acadêmico dirigido e acompanhamento das notas",
    },
    "nota_matematica": {
        "label": "Nota de matemática",
        "higher_is_better": True,
        "action_hint": "reforço em matemática e revisão de lacunas de conteúdo",
    },
    "nota_portugues": {
        "label": "Nota de português",
        "higher_is_better": True,
        "action_hint": "reforço em leitura, escrita e interpretação",
    },
    "nota_ingles": {
        "label": "Nota de inglês",
        "higher_is_better": True,
        "action_hint": "apoio pedagógico complementar em inglês",
    },
    "ieg": {
        "label": "IEG",
        "higher_is_better": True,
        "action_hint": "aumentar engajamento, participação e vínculo com as atividades",
    },
    "iaa": {
        "label": "IAA",
        "higher_is_better": True,
        "action_hint": "trabalhar percepção de pertencimento, autoestima e relação com os estudos",
    },
    "ips": {
        "label": "IPS",
        "higher_is_better": True,
        "action_hint": "fortalecer acompanhamento psicossocial e suporte emocional",
    },
    "ipp": {
        "label": "IPP",
        "higher_is_better": True,
        "action_hint": "intensificar acompanhamento psicopedagógico e estratégias de aprendizagem",
    },
    "ipv": {
        "label": "IPV",
        "higher_is_better": True,
        "action_hint": "estimular continuidade de evolução e acompanhamento longitudinal",
    },
    "numero_avaliacoes": {
        "label": "Número de avaliações",
        "higher_is_better": True,
        "action_hint": "aumentar frequência de acompanhamento e registro do progresso",
    },
}


@dataclass
class BaselineAnalysisResult:
    baseline_analysis_run_id: int
    training_run_id: int
    validation_reference_year: int
    baseline_metrics: dict[str, float]
    error_counts: dict[str, int]
    numeric_summary: list[dict[str, object]]
    categorical_summary: dict[str, object]
    false_negative_importances: list[dict[str, object]]
    recovery_importances: list[dict[str, object]]
    data_version: str


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


def _categorical_breakdown(dataframe: pd.DataFrame, column: str, error_type: str, top_n: int = 5) -> list[dict[str, object]]:
    subset = dataframe.loc[dataframe["error_type"] == error_type, column].astype("string").fillna("<NA>")
    if subset.empty:
        return []
    counts = subset.value_counts(normalize=True).head(top_n)
    return [
        {
            "category": str(category),
            "share": float(share),
        }
        for category, share in counts.items()
    ]


def _build_numeric_summary(validation_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in NUMERIC_SUMMARY_COLUMNS:
        if column not in validation_frame.columns:
            continue
        grouped = validation_frame.groupby("error_type")[column].mean(numeric_only=True)
        rows.append(
            sanitize_record(
                {
                    "feature": column,
                    "correct_mean": grouped.get("correct"),
                    "false_negative_mean": grouped.get("false_negative"),
                    "false_positive_mean": grouped.get("false_positive"),
                    "false_negative_minus_correct": (
                        grouped.get("false_negative", 0) - grouped.get("correct", 0)
                    ),
                    "false_positive_minus_correct": (
                        grouped.get("false_positive", 0) - grouped.get("correct", 0)
                    ),
                }
            )
        )
    return rows


def _encode_subset(subset: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_frame = subset.copy()
    available_features = [column for column in STRATEGY_B_FEATURES if column in feature_frame.columns]
    working = feature_frame[available_features].copy()
    categorical_columns = [column for column in available_features if column in CATEGORICAL_FEATURES]
    numeric_columns = [column for column in available_features if column not in categorical_columns]

    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(-1)
    for column in categorical_columns:
        working[column] = working[column].astype("string").fillna("<NA>")

    encoded = pd.get_dummies(working, columns=categorical_columns, dummy_na=False)
    return encoded, available_features


def _map_encoded_feature(encoded_feature: str, base_features: list[str]) -> str:
    if encoded_feature in base_features:
        return encoded_feature
    for feature in sorted(base_features, key=len, reverse=True):
        prefix = f"{feature}_"
        if encoded_feature.startswith(prefix):
            return feature
    return encoded_feature


def _build_importance_study(subset: pd.DataFrame, target_column: str) -> list[dict[str, object]]:
    if subset.empty or target_column not in subset.columns or subset[target_column].nunique() < 2:
        return []

    encoded, base_features = _encode_subset(subset)
    if encoded.empty:
        return []

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(encoded, subset[target_column].astype(int))

    grouped: dict[str, float] = {}
    for encoded_feature, importance in zip(encoded.columns, model.feature_importances_, strict=True):
        feature = _map_encoded_feature(encoded_feature, base_features)
        grouped[feature] = grouped.get(feature, 0.0) + float(importance)

    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in ranked[:10]
    ]


def _export_baseline_analysis(run_id: int) -> None:
    with session_scope() as session:
        run = session.get(BaselineAnalysisRun, run_id)
        if run is None:
            return
        payload = {
            "baseline_analysis_run_id": run.id,
            "training_run_id": run.training_run_id,
            "validation_reference_year": run.validation_reference_year,
            "baseline_metrics": run.baseline_metrics_json,
            "error_counts": run.error_counts_json,
            "numeric_summary": run.numeric_summary_json,
            "categorical_summary": run.categorical_summary_json,
            "false_negative_importances": run.false_negative_importances_json,
            "recovery_importances": run.recovery_importances_json,
            "data_version": run.data_version,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }

    BASELINE_ANALYSIS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def analyze_baseline_residuals(
    validation_frame: pd.DataFrame,
    *,
    training_run_id: int,
    validation_reference_year: int,
    data_version: str,
) -> BaselineAnalysisResult:
    ensure_directories()
    working = validation_frame.copy()
    working["baseline_pred"] = (pd.to_numeric(working["defasagem"], errors="coerce") < 0).astype(int)
    working["error_type"] = "correct"
    working.loc[
        (working["baseline_pred"] == 0) & (working["target_future_risk"] == 1),
        "error_type",
    ] = "false_negative"
    working.loc[
        (working["baseline_pred"] == 1) & (working["target_future_risk"] == 0),
        "error_type",
    ] = "false_positive"

    y_true = working["target_future_risk"].astype(int)
    baseline_scores = working["baseline_pred"].astype(float)
    baseline_predictions = working["baseline_pred"].astype(int)
    baseline_metrics = _compute_metrics(y_true, baseline_scores, baseline_predictions)

    error_counts = {
        "correct": int((working["error_type"] == "correct").sum()),
        "false_negative": int((working["error_type"] == "false_negative").sum()),
        "false_positive": int((working["error_type"] == "false_positive").sum()),
    }

    numeric_summary = _build_numeric_summary(working)
    categorical_summary = {
        "false_negative": {
            column: _categorical_breakdown(working, column, "false_negative")
            for column in CATEGORICAL_SUMMARY_COLUMNS
            if column in working.columns
        },
        "false_positive": {
            column: _categorical_breakdown(working, column, "false_positive")
            for column in CATEGORICAL_SUMMARY_COLUMNS
            if column in working.columns
        },
    }

    false_negative_subset = working.loc[working["baseline_pred"] == 0].copy()
    false_negative_subset["entered_risk_despite_baseline"] = false_negative_subset["target_future_risk"].astype(int)
    false_negative_importances = _build_importance_study(
        false_negative_subset,
        "entered_risk_despite_baseline",
    )

    recovery_subset = working.loc[working["baseline_pred"] == 1].copy()
    recovery_subset["recovered_despite_baseline"] = (1 - recovery_subset["target_future_risk"].astype(int)).astype(int)
    recovery_importances = _build_importance_study(
        recovery_subset,
        "recovered_despite_baseline",
    )

    with session_scope() as session:
        run = BaselineAnalysisRun(
            training_run_id=training_run_id,
            validation_reference_year=validation_reference_year,
            baseline_metrics_json=sanitize_record(baseline_metrics),
            error_counts_json=error_counts,
            numeric_summary_json=numeric_summary,
            categorical_summary_json=categorical_summary,
            false_negative_importances_json=false_negative_importances,
            recovery_importances_json=recovery_importances,
            data_version=data_version,
        )
        session.add(run)
        session.flush()
        run_id = run.id

    _export_baseline_analysis(run_id)
    record_event(
        "INFO",
        __name__,
        "Analise residual do baseline concluida",
        {
            "baseline_analysis_run_id": run_id,
            "training_run_id": training_run_id,
            "validation_reference_year": validation_reference_year,
            "data_version": data_version,
        },
    )
    return BaselineAnalysisResult(
        baseline_analysis_run_id=run_id,
        training_run_id=training_run_id,
        validation_reference_year=validation_reference_year,
        baseline_metrics=baseline_metrics,
        error_counts=error_counts,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        false_negative_importances=false_negative_importances,
        recovery_importances=recovery_importances,
        data_version=data_version,
    )


def read_latest_baseline_analysis() -> dict[str, object]:
    with session_scope() as session:
        run = session.execute(
            select(BaselineAnalysisRun).order_by(BaselineAnalysisRun.id.desc())
        ).scalars().first()
        if run is None:
            return {}
        payload = {
            "baseline_analysis_run_id": run.id,
            "training_run_id": run.training_run_id,
            "validation_reference_year": run.validation_reference_year,
            "baseline_metrics": run.baseline_metrics_json,
            "error_counts": run.error_counts_json,
            "numeric_summary": run.numeric_summary_json,
            "categorical_summary": run.categorical_summary_json,
            "false_negative_importances": run.false_negative_importances_json,
            "recovery_importances": run.recovery_importances_json,
            "data_version": run.data_version,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
    BASELINE_ANALYSIS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def build_actionable_insights(analysis: dict[str, object]) -> dict[str, object]:
    if not analysis:
        return {}

    numeric_summary = analysis.get("numeric_summary", [])
    numeric_by_feature = {
        row.get("feature"): row
        for row in numeric_summary
        if row.get("feature") in ACTIONABLE_FEATURES
    }

    def _build_signal_rows(
        importance_rows: list[dict[str, object]],
        *,
        delta_key: str,
        scenario: str,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for item in importance_rows:
            feature = item.get("feature")
            if feature not in ACTIONABLE_FEATURES:
                continue
            metadata = ACTIONABLE_FEATURES[feature]
            summary = numeric_by_feature.get(feature, {})
            delta = summary.get(delta_key)
            if delta is None:
                direction = "associacao_indisponivel"
                interpretation = "sem evidência numérica adicional suficiente nesta execução"
            else:
                delta_value = float(delta)
                if metadata["higher_is_better"]:
                    if scenario == "risk_entry":
                        direction = "valores_mais_baixos_associados_a_piora" if delta_value < 0 else "valores_mais_altos_associados_a_piora"
                        interpretation = "valores mais baixos parecem associados a maior chance de piora" if delta_value < 0 else "valores mais altos parecem associados a maior chance de piora"
                    else:
                        direction = "valores_mais_altos_associados_a_melhora" if delta_value > 0 else "valores_mais_baixos_associados_a_melhora"
                        interpretation = "valores mais altos parecem associados a maior chance de melhora" if delta_value > 0 else "valores mais baixos parecem associados a maior chance de melhora"
                else:
                    if scenario == "risk_entry":
                        direction = "valores_mais_altos_associados_a_piora" if delta_value > 0 else "valores_mais_baixos_associados_a_piora"
                        interpretation = "valores mais altos parecem associados a maior chance de piora" if delta_value > 0 else "valores mais baixos parecem associados a maior chance de piora"
                    else:
                        direction = "valores_mais_baixos_associados_a_melhora" if delta_value < 0 else "valores_mais_altos_associados_a_melhora"
                        interpretation = "valores mais baixos parecem associados a maior chance de melhora" if delta_value < 0 else "valores mais altos parecem associados a maior chance de melhora"

            rows.append(
                sanitize_record(
                    {
                        "feature": feature,
                        "label": metadata["label"],
                        "importance": item.get("importance"),
                        "direction": direction,
                        "interpretation": interpretation,
                        "action_hint": metadata["action_hint"],
                        "comparison_delta": delta,
                    }
                )
            )
        return rows

    worsening_signals = _build_signal_rows(
        analysis.get("false_negative_importances", []),
        delta_key="false_negative_minus_correct",
        scenario="risk_entry",
    )
    recovery_signals = _build_signal_rows(
        analysis.get("recovery_importances", []),
        delta_key="false_positive_minus_correct",
        scenario="recovery",
    )

    return {
        "baseline_analysis_run_id": analysis.get("baseline_analysis_run_id"),
        "validation_reference_year": analysis.get("validation_reference_year"),
        "data_version": analysis.get("data_version"),
        "worsening_signals": worsening_signals,
        "recovery_signals": recovery_signals,
        "notes": [
            "As variáveis listadas aqui foram filtradas para destacar apenas sinais mais acionáveis pela operação pedagógica e psicossocial.",
            "Foram removidos proxies estruturais como fase, fase ideal, IAN e defasagem, porque eles descrevem muito diretamente o próprio atraso escolar.",
            "Esses sinais ajudam a priorizar intervenção, mas não devem ser tratados como prova causal isolada.",
        ],
    }
