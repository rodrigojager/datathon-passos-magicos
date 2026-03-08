import src.dashboard_data as dashboard_data
from src.database import session_scope
from src.db_models import AppEvent, DriftRun, EvaluationRow, EvaluationRun, StudentRecord, TrainingRun


def test_dashboard_data_reads_database_and_builds_overview():
    with session_scope() as session:
        session.add_all(
            [
                StudentRecord(
                    ra="RA-1",
                    ano_referencia=2023,
                    source_schema="pede_2023_like",
                    source_file="2023.csv",
                    source_row_number=1,
                    canonical_payload={"ra": "RA-1", "ano_referencia": 2023},
                ),
                StudentRecord(
                    ra="RA-2",
                    ano_referencia=2024,
                    source_schema="pede_2024_like",
                    source_file="2024.csv",
                    source_row_number=1,
                    canonical_payload={"ra": "RA-2", "ano_referencia": 2024},
                ),
            ]
        )
        training_run = TrainingRun(
            selected_model="random_forest",
            selected_strategy="strategy_b",
            metrics_json={"f2": 0.5},
            candidates_json=[{"model_name": "random_forest", "strategy": "strategy_b", "metrics": {"f2": 0.5}}],
            selected_feature_columns_json=["idade"],
            training_rows=10,
            validation_rows=5,
            data_version="v1-r2",
        )
        session.add(training_run)
        session.flush()
        evaluation_run = EvaluationRun(
            source_file="arquivo.csv",
            detected_schema="pede_2023_like",
            current_year=2023,
            next_year=2024,
            evaluation_mode="validation_frozen_candidates",
            selected_model_id="random_forest__strategy_b",
            metrics_json=None,
            summaries_json=[],
            compared_rows=2,
            data_version="v1-r2",
        )
        session.add(evaluation_run)
        session.flush()
        session.add_all(
            [
                EvaluationRow(
                    evaluation_run_id=evaluation_run.id,
                    model_id="random_forest__strategy_b",
                    payload_json={
                        "modelo_utilizado": "random_forest",
                        "estrategia_utilizada": "strategy_b",
                        "prediction_risco_defasagem_futura": 1,
                        "real_risco_defasagem": 1,
                    },
                ),
                EvaluationRow(
                    evaluation_run_id=evaluation_run.id,
                    model_id="random_forest__strategy_b",
                    payload_json={
                        "modelo_utilizado": "random_forest",
                        "estrategia_utilizada": "strategy_b",
                        "prediction_risco_defasagem_futura": 0,
                        "real_risco_defasagem": 1,
                    },
                ),
            ]
        )
        session.add(
            DriftRun(
                source_file="arquivo.csv",
                detected_schema="pede_2024_like",
                reference_year=2024,
                feature_rows=10,
                alert_count=3,
                details_json=[{"feature": "idade", "alert": True}],
                report_html="<html>ok</html>",
                data_version="v1-r2",
            )
        )
        session.add_all(
            [
                AppEvent(level="INFO", logger="src.storage", message="Bootstrap completed", context_json={}),
                AppEvent(level="INFO", logger="src.modeling", message="Training completed", context_json={}),
            ]
        )

    assert dashboard_data.read_training_report()["selected_model"] == "random_forest"
    assert dashboard_data.read_drift_summary()["alert_count"] == 3
    assert dashboard_data.read_latest_evaluation_summary()["rows"] == 2
    assert len(dashboard_data.read_recent_logs(limit=10)) == 2

    overview = dashboard_data.build_overview_metrics()
    assert overview["consolidated_rows"] == 2
    assert overview["unique_students"] == 2
    assert overview["selected_model"] == "random_forest"
