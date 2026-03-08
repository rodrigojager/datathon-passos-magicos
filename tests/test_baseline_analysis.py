import pandas as pd

from src.baseline_analysis import analyze_baseline_residuals, read_latest_baseline_analysis


def test_baseline_analysis_persists_summary(tmp_path, monkeypatch):
    import src.baseline_analysis as baseline_analysis_module

    analysis_path = tmp_path / "baseline_analysis.json"
    monkeypatch.setattr(baseline_analysis_module, "BASELINE_ANALYSIS_PATH", analysis_path)
    monkeypatch.setattr(baseline_analysis_module, "ensure_directories", lambda: analysis_path.parent.mkdir(parents=True, exist_ok=True))

    validation_frame = pd.DataFrame(
        {
            "ra": ["A", "B", "C", "D", "E", "F"],
            "ano_referencia": [2023] * 6,
            "defasagem": [-1, -1, 0, 0, 0, -1],
            "target_future_risk": [1, 0, 1, 0, 0, 1],
            "fase": ["ALFA", "ALFA", "FASE 1", "FASE 2", "FASE 2", "FASE 3"],
            "turma": ["A", "A", "B", "B", "B", "C"],
            "idade": [8, 8, 9, 10, 10, 11],
            "genero": ["Feminino", "Masculino", "Feminino", "Masculino", "Feminino", "Masculino"],
            "ano_ingresso": [2023, 2023, 2022, 2022, 2022, 2021],
            "instituicao_ensino": ["Publica", "Publica", "Publica", "Privada", "Privada", "Publica"],
            "numero_avaliacoes": [2, 3, 2, 4, 4, 3],
            "inde_atual": [6.0, 7.5, 6.2, 7.8, 8.0, 5.9],
            "iaa": [6.5, 7.0, 6.4, 7.2, 7.3, 6.1],
            "ieg": [7.2, 8.3, 7.0, 8.5, 8.6, 6.8],
            "ips": [5.0, 5.5, 5.7, 5.1, 5.0, 5.8],
            "ipp": [6.0, 7.5, 6.2, 8.0, 8.1, 5.8],
            "ida": [5.9, 7.0, 6.0, 7.4, 7.6, 5.7],
            "nota_matematica": [5.8, 7.1, 5.9, 7.5, 7.7, 5.5],
            "nota_portugues": [6.0, 7.3, 6.2, 7.8, 7.9, 5.8],
            "nota_ingles": [4.8, 6.9, 4.9, 7.3, 7.4, 4.5],
            "ipv": [6.1, 8.0, 6.0, 8.1, 8.3, 5.8],
            "ian": [5.5, 9.0, 5.6, 9.1, 9.2, 5.2],
            "fase_ideal": ["ALFA", "ALFA", "Fase 1", "Fase 2", "Fase 2", "Fase 3"],
            "flag_ipp_ausente": [0, 0, 0, 0, 0, 0],
        }
    )

    result = analyze_baseline_residuals(
        validation_frame,
        training_run_id=1,
        validation_reference_year=2023,
        data_version="v1-r6",
    )

    assert result.error_counts["false_negative"] == 1
    assert result.error_counts["false_positive"] == 1
    assert analysis_path.exists()

    latest = read_latest_baseline_analysis()
    assert latest["training_run_id"] == 1
    assert latest["error_counts"]["false_negative"] == 1
