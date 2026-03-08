from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


def normalize_label(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(label))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return re.sub(r"_+", "_", lowered).strip("_")


CANONICAL_COLUMN_ALIASES = {
    "ra": {"ra", "registro_aluno"},
    "ano_referencia": {"ano_referencia", "ano", "ano_base"},
    "fase": {"fase"},
    "turma": {"turma"},
    "nome_anonimizado": {"nome", "nome_anonimizado", "nome_aluno"},
    "ano_nascimento": {"ano_nasc", "ano_nascimento"},
    "data_nascimento": {"data_de_nasc", "data_nascimento"},
    "idade": {"idade", "idade_22", "idade_aluno"},
    "genero": {"genero", "sexo"},
    "ano_ingresso": {"ano_ingresso"},
    "instituicao_ensino": {
        "instituicao_de_ensino",
        "instituicao_ensino",
        "instituicao_de_ensino_aluno",
    },
    "escola": {"escola"},
    "status_matricula": {"ativo_inativo", "ativo_inativo_1", "status_matricula"},
    "pedra_20": {"pedra_20"},
    "pedra_21": {"pedra_21"},
    "pedra_22": {"pedra_22"},
    "pedra_23": {"pedra_23"},
    "pedra_atual": {"pedra_2023", "pedra_2024", "pedra_atual"},
    "inde_22": {"inde_22"},
    "inde_23": {"inde_23", "inde_2023"},
    "inde_atual": {"inde_2024", "inde_atual"},
    "cg": {"cg"},
    "cf": {"cf"},
    "ct": {"ct"},
    "numero_avaliacoes": {"n_av", "no_av", "n_avs", "numero_av", "numero_avaliacoes"},
    "avaliador_1": {"avaliador1"},
    "avaliador_2": {"avaliador2"},
    "avaliador_3": {"avaliador3"},
    "avaliador_4": {"avaliador4"},
    "avaliador_5": {"avaliador5"},
    "avaliador_6": {"avaliador6"},
    "rec_av_1": {"rec_av1"},
    "rec_av_2": {"rec_av2"},
    "rec_av_3": {"rec_av3"},
    "rec_av_4": {"rec_av4"},
    "rec_psicologia": {"rec_psicologia"},
    "iaa": {"iaa"},
    "ieg": {"ieg"},
    "ips": {"ips"},
    "ipp": {"ipp"},
    "ida": {"ida"},
    "nota_matematica": {"matem", "mat", "matematica", "nota_matematica"},
    "nota_portugues": {"portug", "por", "portugues", "nota_portugues"},
    "nota_ingles": {"ingles", "ing", "nota_ingles"},
    "indicado": {"indicado"},
    "atingiu_pv": {"atingiu_pv"},
    "ipv": {"ipv"},
    "ian": {"ian"},
    "fase_ideal": {"fase_ideal"},
    "defasagem": {"defas", "defasagem"},
    "destaque_ieg": {"destaque_ieg"},
    "destaque_ida": {"destaque_ida"},
    "destaque_ipv": {"destaque_ipv", "destaque_ipv_1"},
    "source_schema": {"source_schema"},
    "source_file": {"source_file"},
    "source_row_number": {"source_row_number"},
}

CANONICAL_COLUMN_ORDER = [
    "ra",
    "ano_referencia",
    "fase",
    "turma",
    "nome_anonimizado",
    "ano_nascimento",
    "data_nascimento",
    "idade",
    "genero",
    "ano_ingresso",
    "instituicao_ensino",
    "escola",
    "status_matricula",
    "pedra_20",
    "pedra_21",
    "pedra_22",
    "pedra_23",
    "pedra_atual",
    "inde_22",
    "inde_23",
    "inde_atual",
    "cg",
    "cf",
    "ct",
    "numero_avaliacoes",
    "avaliador_1",
    "avaliador_2",
    "avaliador_3",
    "avaliador_4",
    "avaliador_5",
    "avaliador_6",
    "rec_av_1",
    "rec_av_2",
    "rec_av_3",
    "rec_av_4",
    "rec_psicologia",
    "iaa",
    "ieg",
    "ips",
    "ipp",
    "ida",
    "nota_matematica",
    "nota_portugues",
    "nota_ingles",
    "indicado",
    "atingiu_pv",
    "ipv",
    "ian",
    "fase_ideal",
    "defasagem",
    "destaque_ieg",
    "destaque_ida",
    "destaque_ipv",
    "source_schema",
    "source_file",
    "source_row_number",
]

NUMERIC_COLUMNS = {
    "ano_referencia",
    "ano_nascimento",
    "idade",
    "ano_ingresso",
    "inde_22",
    "inde_23",
    "inde_atual",
    "cg",
    "cf",
    "ct",
    "numero_avaliacoes",
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
    "defasagem",
}

BOOLEAN_COLUMNS = {
    "indicado",
    "atingiu_pv",
    "status_matricula",
}

REQUIRED_FOR_NORMALIZATION = {"ra", "fase"}


@dataclass(frozen=True)
class SchemaDefinition:
    name: str
    signature_columns: set[str]
    fallback_year: int | None = None


SCHEMA_DEFINITIONS = (
    SchemaDefinition(
        name="pede_2024_like",
        signature_columns={"escola", "ativo_inativo", "inde_2024"},
        fallback_year=2024,
    ),
    SchemaDefinition(
        name="pede_2023_like",
        signature_columns={"inde_2023", "pedra_2023", "defasagem"},
        fallback_year=2023,
    ),
    SchemaDefinition(
        name="pede_2022_like",
        signature_columns={"idade_22", "ano_nasc", "matem", "portug", "defas"},
        fallback_year=2022,
    ),
)


def alias_lookup() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in CANONICAL_COLUMN_ALIASES.items():
        for alias in aliases:
            mapping[normalize_label(alias)] = canonical
    return mapping


def detect_schema(normalized_raw_columns: set[str]) -> SchemaDefinition | None:
    best_match: SchemaDefinition | None = None
    best_score = 0
    for definition in SCHEMA_DEFINITIONS:
        score = len(definition.signature_columns & normalized_raw_columns)
        if score > best_score:
            best_match = definition
            best_score = score
    return best_match if best_score > 0 else None
