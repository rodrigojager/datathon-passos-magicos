from src.schema_registry import detect_schema, normalize_label


def test_normalize_label_removes_accents_and_spaces():
    assert normalize_label("Instituição de ensino") == "instituicao_de_ensino"


def test_detects_2022_like_schema():
    detected = detect_schema({"idade_22", "ano_nasc", "matem", "portug", "defas"})
    assert detected is not None
    assert detected.name == "pede_2022_like"


def test_detects_2024_like_schema():
    detected = detect_schema({"inde_2024", "escola", "ativo_inativo"})
    assert detected is not None
    assert detected.name == "pede_2024_like"
