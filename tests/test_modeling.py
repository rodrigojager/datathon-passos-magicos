import pandas as pd

import src.baseline_analysis as baseline_analysis_module
import src.modeling as modeling_module
import src.monitoring as monitoring_module


def _synthetic_consolidated() -> pd.DataFrame:
    rows = []
    for year, risk_pattern in [
        (2022, {"RA-1": -1, "RA-2": 0, "RA-3": -2, "RA-4": 1}),
        (2023, {"RA-1": -1, "RA-2": 0, "RA-3": -1, "RA-4": 1}),
        (2024, {"RA-1": -2, "RA-2": 0, "RA-3": -1, "RA-4": 1}),
    ]:
        for idx, (ra, defasagem) in enumerate(risk_pattern.items(), start=1):
            rows.append(
                {
                    "ra": ra,
                    "ano_referencia": year,
                    "fase": f"{idx}",
                    "turma": "A",
                    "idade": 10 + idx,
                    "genero": "Masculino" if idx % 2 == 0 else "Feminino",
                    "ano_ingresso": 2020 + (idx % 2),
                    "instituicao_ensino": "Publica" if idx <= 2 else "Privada",
                    "numero_avaliacoes": 3 + idx,
                    "inde_atual": 5.0 + idx + (0.2 if defasagem >= 0 else -0.4),
                    "iaa": 6.0 + idx,
                    "ieg": 5.0 + idx,
                    "ips": 5.5 + idx,
                    "ipp": 6.0 + idx if year >= 2023 else pd.NA,
                    "ida": 4.5 + idx,
                    "nota_matematica": 4.0 + idx,
                    "nota_portugues": 4.5 + idx,
                    "nota_ingles": 5.0 + idx,
                    "ipv": 5.0 + idx,
                    "ian": 2.5 if defasagem < 0 else 10.0,
                    "fase_ideal": "Fase 8" if defasagem < 0 else "Fase 7",
                    "defasagem": defasagem,
                    "source_schema": f"pede_{year}_like",
                    "source_file": f"pede_{year}.csv",
                }
            )
    return pd.DataFrame(rows)


def _patch_model_paths(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "training_report.json"
    baseline_analysis_path = tmp_path / "baseline_analysis.json"
    evaluation_path = tmp_path / "latest_evaluation.csv"
    drift_path = tmp_path / "drift_report.html"
    drift_summary_path = tmp_path / "drift_summary.json"
    validation_bundles_path = tmp_path / "validation_candidate_models.joblib"
    predictions_dir = tmp_path / "predictions"

    def _ensure_dirs():
        predictions_dir.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        drift_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(modeling_module, "MODEL_BUNDLE_PATH", model_path)
    monkeypatch.setattr(modeling_module, "CANDIDATE_BUNDLES_PATH", tmp_path / "candidate_models.joblib")
    monkeypatch.setattr(modeling_module, "VALIDATION_CANDIDATE_BUNDLES_PATH", validation_bundles_path)
    monkeypatch.setattr(modeling_module, "TRAINING_REPORT_PATH", report_path)
    monkeypatch.setattr(modeling_module, "LATEST_EVALUATION_PATH", evaluation_path)
    monkeypatch.setattr(modeling_module, "PREDICTIONS_DIR", predictions_dir)
    monkeypatch.setattr(modeling_module, "ensure_directories", _ensure_dirs)

    monkeypatch.setattr(baseline_analysis_module, "BASELINE_ANALYSIS_PATH", baseline_analysis_path)
    monkeypatch.setattr(baseline_analysis_module, "ensure_directories", _ensure_dirs)

    monkeypatch.setattr(monitoring_module, "MODEL_BUNDLE_PATH", model_path)
    monkeypatch.setattr(monitoring_module, "DRIFT_REPORT_PATH", drift_path)
    monkeypatch.setattr(monitoring_module, "DRIFT_SUMMARY_PATH", drift_summary_path)
    monkeypatch.setattr(monitoring_module, "ensure_directories", _ensure_dirs)

    return model_path, report_path, baseline_analysis_path, evaluation_path, drift_path


def test_training_prediction_evaluation_and_drift(monkeypatch, tmp_path):
    consolidated = _synthetic_consolidated()
    model_path, report_path, baseline_analysis_path, evaluation_path, drift_path = _patch_model_paths(
        monkeypatch,
        tmp_path,
    )
    monkeypatch.setattr(modeling_module, "load_consolidated_dataframe", lambda: consolidated)
    monkeypatch.setattr(modeling_module, "current_data_version", lambda: "v9-r12")
    monkeypatch.setattr(monitoring_module, "current_data_version", lambda: "v9-r12")

    training = modeling_module.train_and_select_model()
    assert training.selected_model in {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert model_path.exists()
    assert report_path.exists()
    assert baseline_analysis_path.exists()
    assert training.baseline_analysis_run_id >= 1
    assert training.transition_score_run_id >= 1
    assert "entry_metrics" in training.transition_scores
    assert "recovery_metrics" in training.transition_scores

    current = consolidated.loc[consolidated["ano_referencia"] == 2023].copy()
    next_year = consolidated.loc[consolidated["ano_referencia"] == 2024].copy()

    predictions = modeling_module.predict_from_normalized_dataframe(
        current,
        source_filename="2023.csv",
        detected_schema="pede_2023_like",
        reference_year=2023,
    )
    assert len(predictions) == 4
    assert "probabilidade_risco_defasagem_futura" in predictions.columns
    assert "score_entrada_em_risco" in predictions.columns
    assert "score_recuperacao" in predictions.columns

    evaluation = modeling_module.evaluate_predictions_against_next_year(
        current,
        next_year,
        source_filename="2023_vs_2024.csv",
        detected_schema="uploaded_pair",
    )
    assert evaluation["compared_rows"] == 4
    assert evaluation_path.exists()

    model_comparison = modeling_module.evaluate_uploaded_file_against_consolidated(
        current,
        source_filename="2023.csv",
        detected_schema="pede_2023_like",
    )
    assert model_comparison["current_year"] == 2023
    assert model_comparison["next_year"] == 2024
    assert len(model_comparison["summaries"]) >= 2

    drift = monitoring_module.generate_drift_report(
        current,
        source_filename="2024.csv",
        detected_schema="pede_2024_like",
        reference_year=2024,
    )
    assert drift.feature_rows > 0
    assert drift_path.exists()
