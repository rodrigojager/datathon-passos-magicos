from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from src.modeling import ModelingError


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dashboard_pages_render():
    for path in [
        "/dashboard",
        "/dashboard/models",
        "/dashboard/predict",
        "/dashboard/docs",
        "/dashboard/docs/01_visao_geral",
        "/dashboard/drift",
        "/dashboard/logs",
        "/docs",
    ]:
        response = client.get(path)
        assert response.status_code == 200
        assert "Passos" in response.text


def test_normalize_endpoint_returns_downloadable_csv():
    content = (
        "RA,Fase,Turma,Idade 22,Gênero,Ano ingresso,Instituição de ensino,INDE 22,IAA,IEG,IPS,IDA,Matem,Portug,Inglês,IPV,IAN,Fase ideal,Defas\n"
        "RA-1,7,A,19,Menina,2016,Escola Pública,\"5,783\",\"8,3\",\"4,1\",\"5,6\",\"4,0\",\"2,7\",\"3,5\",\"6,0\",\"7,278\",\"5,000\",Fase 8,-1\n"
    ).encode("utf-8")
    response = client.post(
        "/data/normalize",
        files={"file": ("PEDE2022.csv", content, "text/csv")},
    )
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"].lower()
    assert "ra,ano_referencia" in response.text.lower()


def test_predict_endpoint_uses_shared_normalization(monkeypatch):
    normalized = pd.DataFrame(
        {
            "ra": ["RA-1"],
            "ano_referencia": [2024],
            "source_schema": ["pede_2024_like"],
            "source_file": ["arquivo.csv"],
        }
    )
    monkeypatch.setattr(
        "app.main._normalize_or_422",
        lambda upload, reference_year: SimpleNamespace(
            dataframe=normalized,
            detected_schema="pede_2024_like",
            reference_year=2024,
            source_filename="arquivo.csv",
        ),
    )
    monkeypatch.setattr(
        "app.main.predict_from_normalized_dataframe",
        lambda dataframe, source_filename="arquivo.csv", detected_schema="pede_2024_like", reference_year=2024, selected_model_id=None: normalized.assign(
            prediction_risco_defasagem_futura=[1],
            probabilidade_risco_defasagem_futura=[0.9],
            threshold=[0.5],
            modelo_utilizado=["random_forest"],
            estrategia_utilizada=["strategy_b"],
        ),
    )

    response = client.post(
        "/predict?selected_model_id=random_forest__strategy_b",
        files={"file": ("qualquer.csv", b"ra\nRA-1\n", "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["rows"] == 1
    assert response.json()["selected_model_id"] == "random_forest__strategy_b"


def test_predict_endpoint_returns_409_when_model_missing(monkeypatch):
    monkeypatch.setattr(
        "app.main._normalize_or_422",
        lambda upload, reference_year: SimpleNamespace(
            dataframe=pd.DataFrame({"ra": ["RA-1"], "ano_referencia": [2024]}),
            detected_schema="pede_2024_like",
            reference_year=2024,
            source_filename="arquivo.csv",
        ),
    )
    monkeypatch.setattr(
        "app.main.predict_from_normalized_dataframe",
        lambda dataframe, source_filename="arquivo.csv", detected_schema="pede_2024_like", reference_year=2024, selected_model_id=None: (_ for _ in ()).throw(ModelingError("modelo ausente")),
    )

    response = client.post(
        "/predict",
        files={"file": ("qualquer.csv", b"ra\nRA-1\n", "text/csv")},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "modelo ausente"


def test_monitor_metrics_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.main.build_overview_metrics",
        lambda: {"consolidated_rows": 10, "selected_model": "random_forest"},
    )
    response = client.get("/monitor/metrics")
    assert response.status_code == 200
    assert response.json()["selected_model"] == "random_forest"


def test_data_status_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.main.get_data_status_summary",
        lambda: {"rows": 3, "unique_students": 2, "reference_years": [2022, 2023, 2024], "data_version": "v1-r3"},
    )
    response = client.get("/data/status")
    assert response.status_code == 200
    assert response.json()["unique_students"] == 2


def test_monitor_logs_and_models_endpoints(monkeypatch):
    monkeypatch.setattr(
        "app.main.read_recent_logs",
        lambda limit=200: [{"timestamp": "t", "level": "INFO", "logger": "x", "message": "ok"}],
    )
    monkeypatch.setattr(
        "app.main.read_training_report",
        lambda: {"selected_model": "random_forest", "selected_strategy": "strategy_b", "candidates": [{"model_name": "random_forest", "strategy": "strategy_b", "metrics": {"f2": 0.5}}]},
    )
    logs_response = client.get("/monitor/logs?limit=5")
    models_response = client.get("/monitor/models/comparison")
    assert logs_response.status_code == 200
    assert models_response.status_code == 200
    assert logs_response.json()["count"] == 1
    assert models_response.json()["selected_model"] == "random_forest"
    assert models_response.json()["candidates"][0]["model_id"] == "random_forest__strategy_b"


def test_monitor_baseline_analysis_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.main.read_latest_baseline_analysis",
        lambda: {
            "baseline_analysis_run_id": 3,
            "training_run_id": 1,
            "error_counts": {"false_negative": 10, "false_positive": 20},
        },
    )
    response = client.get("/monitor/baseline-analysis")
    assert response.status_code == 200
    assert response.json()["baseline_analysis_run_id"] == 3


def test_monitor_actionable_insights_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.main.read_latest_baseline_analysis",
        lambda: {"baseline_analysis_run_id": 4, "numeric_summary": []},
    )
    monkeypatch.setattr(
        "app.main.build_actionable_insights",
        lambda analysis: {
            "baseline_analysis_run_id": analysis["baseline_analysis_run_id"],
            "worsening_signals": [{"feature": "ida", "label": "IDA"}],
            "recovery_signals": [{"feature": "ipp", "label": "IPP"}],
        },
    )
    response = client.get("/monitor/actionable-insights")
    assert response.status_code == 200
    assert response.json()["baseline_analysis_run_id"] == 4


def test_monitor_transition_scores_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.main.read_latest_transition_score_summary",
        lambda: {
            "transition_score_run_id": 4,
            "entry_model_name": "random_forest",
            "recovery_model_name": "random_forest",
            "strategy_name": "strategy_b_segmented",
        },
    )
    response = client.get("/monitor/transition-scores")
    assert response.status_code == 200
    assert response.json()["transition_score_run_id"] == 4


def test_monitor_drift_summary_and_report(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.read_drift_summary", lambda: {"alert_count": 2, "feature_rows": 4})
    monkeypatch.setattr("app.main.read_latest_drift_report_html", lambda: "<html>ok</html>")
    summary_response = client.get("/monitor/drift/summary")
    report_response = client.get("/monitor/drift/report")
    assert summary_response.status_code == 200
    assert summary_response.json()["alert_count"] == 2
    assert report_response.status_code == 200
    assert "ok" in report_response.text


def test_monitor_drift_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.main._normalize_or_422",
        lambda upload, reference_year: SimpleNamespace(
            dataframe=pd.DataFrame({"ra": ["RA-1"], "ano_referencia": [2024]}),
            detected_schema="pede_2024_like",
            reference_year=2024,
            source_filename="arquivo.csv",
        ),
    )
    monkeypatch.setattr(
        "app.main.generate_drift_report",
        lambda dataframe, source_filename="arquivo.csv", detected_schema="pede_2024_like", reference_year=2024: SimpleNamespace(
            drift_run_id=7,
            report_path="artifacts/drift_report.html",
            feature_rows=3,
            alert_count=1,
            details=[{"feature": "idade", "alert": True}],
            data_version="v1-r10",
        ),
    )
    response = client.post(
        "/monitor/drift",
        files={"file": ("arquivo.csv", b"ra\nRA-1\n", "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["alert_count"] == 1
