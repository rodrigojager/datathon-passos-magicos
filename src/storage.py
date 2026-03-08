from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import func, select, tuple_

from src.config import RAW_DATA_DIR, ensure_directories
from src.database import record_event, session_scope
from src.db_models import IngestionRun, StudentRecord
from src.normalization import NormalizationResult, normalize_uploaded_csv
from src.schema_registry import CANONICAL_COLUMN_ORDER
from src.utils import dataframe_to_records, get_logger, sanitize_record


logger = get_logger(__name__)


@dataclass
class IngestionSummary:
    ingestion_run_id: int
    source_file: str
    detected_schema: str
    reference_year: int
    input_rows: int
    inserted_rows: int
    duplicate_rows: int
    unknown_columns: list[str]
    data_version: str


def _build_data_version(session) -> str:
    row_count = session.scalar(select(func.count()).select_from(StudentRecord)) or 0
    latest_ingestion_id = session.scalar(select(func.max(IngestionRun.id))) or 0
    return f"v{latest_ingestion_id}-r{row_count}"


def current_data_version() -> str:
    with session_scope() as session:
        return _build_data_version(session)


def load_consolidated_dataframe() -> pd.DataFrame:
    ensure_directories()
    with session_scope() as session:
        rows = session.execute(
            select(StudentRecord.canonical_payload).order_by(StudentRecord.ano_referencia, StudentRecord.ra)
        ).scalars().all()

    if not rows:
        return pd.DataFrame(columns=CANONICAL_COLUMN_ORDER)

    dataframe = pd.DataFrame(rows)
    for column in CANONICAL_COLUMN_ORDER:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA
    return dataframe[CANONICAL_COLUMN_ORDER]


def get_data_status_summary() -> dict[str, object]:
    with session_scope() as session:
        rows = session.scalar(select(func.count()).select_from(StudentRecord)) or 0
        unique_students = session.scalar(select(func.count(func.distinct(StudentRecord.ra)))) or 0
        years = session.execute(
            select(StudentRecord.ano_referencia).distinct().order_by(StudentRecord.ano_referencia)
        ).scalars().all()
        latest_ingestion = session.execute(
            select(IngestionRun).order_by(IngestionRun.id.desc())
        ).scalars().first()
        return {
            "rows": int(rows),
            "unique_students": int(unique_students),
            "reference_years": [int(year) for year in years if year is not None],
            "data_version": _build_data_version(session),
            "latest_ingestion": None
            if latest_ingestion is None
            else {
                "ingestion_run_id": latest_ingestion.id,
                "source_file": latest_ingestion.source_file,
                "reference_year": latest_ingestion.reference_year,
                "inserted_rows": latest_ingestion.inserted_rows,
                "duplicate_rows": latest_ingestion.duplicate_rows,
                "created_at": latest_ingestion.created_at.isoformat() if latest_ingestion.created_at else None,
            },
        }


def ingest_normalized_dataframe(result: NormalizationResult) -> IngestionSummary:
    ensure_directories()
    incoming = result.dataframe.copy()
    incoming["ra"] = incoming["ra"].astype("string")
    incoming["ano_referencia"] = pd.to_numeric(incoming["ano_referencia"], errors="coerce").astype("Int64")
    incoming = incoming.loc[incoming["ra"].notna() & incoming["ano_referencia"].notna()].copy()
    incoming["ano_referencia"] = incoming["ano_referencia"].astype(int)

    keys = [
        (str(row["ra"]), int(row["ano_referencia"]))
        for row in incoming[["ra", "ano_referencia"]].to_dict(orient="records")
    ]

    with session_scope() as session:
        existing_keys: set[tuple[str, int]] = set()
        if keys:
            existing = session.execute(
                select(StudentRecord.ra, StudentRecord.ano_referencia).where(
                    tuple_(StudentRecord.ra, StudentRecord.ano_referencia).in_(keys)
                )
            ).all()
            existing_keys = {(str(ra), int(year)) for ra, year in existing}

        new_records = incoming.loc[
            ~incoming.apply(lambda row: (str(row["ra"]), int(row["ano_referencia"])) in existing_keys, axis=1)
        ].copy()

        ingestion_run = IngestionRun(
            source_file=result.source_filename,
            detected_schema=result.detected_schema,
            reference_year=result.reference_year,
            input_rows=len(result.dataframe),
            inserted_rows=len(new_records),
            duplicate_rows=len(result.dataframe) - len(new_records),
            unknown_columns_json=result.unknown_columns,
            data_version="pending",
        )
        session.add(ingestion_run)
        session.flush()

        for record in dataframe_to_records(new_records):
            session.add(
                StudentRecord(
                    ingestion_run_id=ingestion_run.id,
                    ra=str(record["ra"]),
                    ano_referencia=int(record["ano_referencia"]),
                    source_schema=record.get("source_schema"),
                    source_file=record.get("source_file"),
                    source_row_number=record.get("source_row_number"),
                    canonical_payload=record,
                )
            )

        ingestion_run.data_version = _build_data_version(session)
        session.flush()
        summary = IngestionSummary(
            ingestion_run_id=ingestion_run.id,
            source_file=result.source_filename,
            detected_schema=result.detected_schema,
            reference_year=result.reference_year,
            input_rows=len(result.dataframe),
            inserted_rows=len(new_records),
            duplicate_rows=len(result.dataframe) - len(new_records),
            unknown_columns=result.unknown_columns,
            data_version=ingestion_run.data_version,
        )

    logger.info(
        "Ingestao concluida | arquivo=%s | ano=%s | inseridas=%s | duplicadas=%s",
        result.source_filename,
        result.reference_year,
        summary.inserted_rows,
        summary.duplicate_rows,
    )
    record_event(
        "INFO",
        __name__,
        "Ingestao concluida",
        sanitize_record(summary.__dict__),
    )
    return summary


def bootstrap_raw_directory() -> list[IngestionSummary]:
    ensure_directories()
    summaries: list[IngestionSummary] = []
    for file_path in sorted(RAW_DATA_DIR.glob("*.csv")):
        result = normalize_uploaded_csv(
            file_path.read_bytes(),
            filename=file_path.name,
        )
        summaries.append(ingest_normalized_dataframe(result))
    logger.info("Carga inicial concluida | arquivos=%s", len(summaries))
    record_event("INFO", __name__, "Carga inicial concluida", {"arquivos_processados": len(summaries)})
    return summaries
