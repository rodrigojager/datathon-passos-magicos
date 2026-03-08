from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re

import pandas as pd

from src.schema_registry import (
    BOOLEAN_COLUMNS,
    CANONICAL_COLUMN_ORDER,
    NUMERIC_COLUMNS,
    REQUIRED_FOR_NORMALIZATION,
    alias_lookup,
    detect_schema,
    normalize_label,
)


TRUE_VALUES = {"sim", "true", "1", "yes", "y", "cursando"}
FALSE_VALUES = {"nao", "não", "false", "0", "no", "n", "inativo"}


@dataclass
class NormalizationResult:
    dataframe: pd.DataFrame
    detected_schema: str
    detected_schema_year: int | None
    reference_year: int
    renamed_columns: dict[str, str]
    unknown_columns: list[str]
    source_filename: str


class NormalizationError(ValueError):
    pass


def _try_read_csv(content: bytes) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "latin1")
    separators = (",", ";")
    last_error: Exception | None = None
    for encoding in encodings:
        for sep in separators:
            try:
                return pd.read_csv(BytesIO(content), encoding=encoding, sep=sep)
            except Exception as error:  # pragma: no cover
                last_error = error
    raise NormalizationError(f"Nao foi possivel ler o CSV: {last_error}") from last_error


def _normalize_boolean(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    normalized = normalize_label(str(value))
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return pd.NA


def _normalize_numeric(series: pd.Series) -> pd.Series:
    textual = series.astype("string").str.strip()
    has_comma = textual.str.contains(",", regex=False, na=False)
    has_dot = textual.str.contains(".", regex=False, na=False)
    textual = textual.where(~(has_comma & has_dot), textual.str.replace(".", "", regex=False))
    textual = textual.where(~has_comma, textual.str.replace(",", ".", regex=False))
    return pd.to_numeric(textual, errors="coerce")


def _infer_year_from_filename(filename: str) -> int | None:
    match = re.search(r"(20\d{2})", filename or "")
    return int(match.group(1)) if match else None


def normalize_uploaded_csv(
    content: bytes,
    filename: str,
    reference_year: int | None = None,
) -> NormalizationResult:
    dataframe = _try_read_csv(content)
    raw_columns = list(dataframe.columns)
    normalized_raw_columns = [normalize_label(column) for column in raw_columns]
    schema_definition = detect_schema(set(normalized_raw_columns))

    if schema_definition is None:
        raise NormalizationError(
            "Nao foi possivel detectar um schema compativel com os padroes conhecidos "
            "(2022, 2023 ou 2024-like)."
        )

    lookup = alias_lookup()
    renamed_columns: dict[str, str] = {}
    unknown_columns: list[str] = []
    final_columns: list[str] = []

    for original, normalized in zip(raw_columns, normalized_raw_columns, strict=True):
        canonical = lookup.get(normalized)
        if canonical:
            renamed_columns[original] = canonical
            final_columns.append(canonical)
        else:
            unknown_columns.append(original)
            final_columns.append(normalized)

    dataframe.columns = final_columns

    if len(set(final_columns)) != len(final_columns):
        dataframe = dataframe.loc[:, ~pd.Index(final_columns).duplicated(keep="first")]

    missing_required = sorted(REQUIRED_FOR_NORMALIZATION - set(dataframe.columns))
    if missing_required:
        raise NormalizationError(
            f"Arquivo reconhecido, mas faltam colunas obrigatorias para normalizacao: {missing_required}"
        )

    effective_year = (
        reference_year
        or _infer_year_from_filename(filename)
        or schema_definition.fallback_year
    )
    if effective_year is None:
        raise NormalizationError(
            "Nao foi possivel inferir o ano de referencia. Envie o parametro reference_year."
        )

    dataframe["ano_referencia"] = effective_year
    dataframe["source_schema"] = schema_definition.name
    dataframe["source_file"] = filename
    dataframe["source_row_number"] = range(1, len(dataframe) + 1)

    if "inde_atual" not in dataframe.columns:
        dataframe["inde_atual"] = pd.NA

    for column in NUMERIC_COLUMNS & set(dataframe.columns):
        dataframe[column] = _normalize_numeric(dataframe[column])

    for column in BOOLEAN_COLUMNS & set(dataframe.columns):
        dataframe[column] = dataframe[column].apply(_normalize_boolean)

    if effective_year == 2022 and dataframe["inde_atual"].isna().all() and "inde_22" in dataframe.columns:
        dataframe["inde_atual"] = dataframe["inde_22"]
    if effective_year == 2023 and dataframe["inde_atual"].isna().all() and "inde_23" in dataframe.columns:
        dataframe["inde_atual"] = dataframe["inde_23"]

    for column in CANONICAL_COLUMN_ORDER:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA

    dataframe = dataframe[CANONICAL_COLUMN_ORDER]

    return NormalizationResult(
        dataframe=dataframe,
        detected_schema=schema_definition.name,
        detected_schema_year=schema_definition.fallback_year,
        reference_year=effective_year,
        renamed_columns=renamed_columns,
        unknown_columns=unknown_columns,
        source_filename=filename,
    )


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def normalized_output_name(filename: str, year: int) -> str:
    stem = Path(filename).stem
    return f"{stem}__normalized_{year}.csv"
