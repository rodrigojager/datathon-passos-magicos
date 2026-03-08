from __future__ import annotations

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_file: Mapped[str] = mapped_column(String(512), nullable=False)
    detected_schema: Mapped[str] = mapped_column(String(128), nullable=False)
    reference_year: Mapped[int] = mapped_column(Integer, nullable=False)
    input_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    inserted_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    duplicate_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    unknown_columns_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    student_records: Mapped[list["StudentRecord"]] = relationship(back_populates="ingestion_run")


class StudentRecord(Base):
    __tablename__ = "student_records"
    __table_args__ = (UniqueConstraint("ra", "ano_referencia", name="uq_student_records_ra_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingestion_run_id: Mapped[int | None] = mapped_column(ForeignKey("ingestion_runs.id"), nullable=True)
    ra: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    ano_referencia: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    source_schema: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_row_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    canonical_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    ingestion_run: Mapped[IngestionRun | None] = relationship(back_populates="student_records")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    selected_model: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_strategy: Mapped[str] = mapped_column(String(128), nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    candidates_json: Mapped[list] = mapped_column(JSON, nullable=False)
    selected_feature_columns_json: Mapped[list] = mapped_column(JSON, nullable=False)
    training_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    reference_rows: Mapped[list["TrainingReferenceRow"]] = relationship(back_populates="training_run")
    baseline_analysis_runs: Mapped[list["BaselineAnalysisRun"]] = relationship(back_populates="training_run")
    transition_score_runs: Mapped[list["TransitionScoreRun"]] = relationship(back_populates="training_run")


class TrainingReferenceRow(Base):
    __tablename__ = "training_reference_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False, index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    training_run: Mapped[TrainingRun] = relationship(back_populates="reference_rows")


class BaselineAnalysisRun(Base):
    __tablename__ = "baseline_analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False, index=True)
    validation_reference_year: Mapped[int] = mapped_column(Integer, nullable=False)
    baseline_metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    error_counts_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    numeric_summary_json: Mapped[list] = mapped_column(JSON, nullable=False)
    categorical_summary_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    false_negative_importances_json: Mapped[list] = mapped_column(JSON, nullable=False)
    recovery_importances_json: Mapped[list] = mapped_column(JSON, nullable=False)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    training_run: Mapped[TrainingRun] = relationship(back_populates="baseline_analysis_runs")


class TransitionScoreRun(Base):
    __tablename__ = "transition_score_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False, index=True)
    validation_reference_year: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    recovery_model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False)
    entry_metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    recovery_metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    training_run: Mapped[TrainingRun] = relationship(back_populates="transition_score_runs")


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_file: Mapped[str] = mapped_column(String(512), nullable=False)
    detected_schema: Mapped[str] = mapped_column(String(128), nullable=False)
    reference_year: Mapped[int] = mapped_column(Integer, nullable=False)
    selected_model: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_strategy: Mapped[str] = mapped_column(String(128), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    prediction_rows: Mapped[list["PredictionRow"]] = relationship(back_populates="prediction_run")


class PredictionRow(Base):
    __tablename__ = "prediction_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_run_id: Mapped[int] = mapped_column(ForeignKey("prediction_runs.id"), nullable=False, index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    prediction_run: Mapped[PredictionRun] = relationship(back_populates="prediction_rows")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_file: Mapped[str] = mapped_column(String(512), nullable=False)
    detected_schema: Mapped[str] = mapped_column(String(128), nullable=False)
    current_year: Mapped[int] = mapped_column(Integer, nullable=False)
    next_year: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluation_mode: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_model_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    summaries_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    compared_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    evaluation_rows: Mapped[list["EvaluationRow"]] = relationship(back_populates="evaluation_run")


class EvaluationRow(Base):
    __tablename__ = "evaluation_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluation_run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    evaluation_run: Mapped[EvaluationRun] = relationship(back_populates="evaluation_rows")


class DriftRun(Base):
    __tablename__ = "drift_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_file: Mapped[str] = mapped_column(String(512), nullable=False)
    detected_schema: Mapped[str] = mapped_column(String(128), nullable=False)
    reference_year: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    alert_count: Mapped[int] = mapped_column(Integer, nullable=False)
    details_json: Mapped[list] = mapped_column(JSON, nullable=False)
    report_html: Mapped[str] = mapped_column(Text, nullable=False)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AppEvent(Base):
    __tablename__ = "app_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    level: Mapped[str] = mapped_column(String(32), nullable=False)
    logger: Mapped[str] = mapped_column(String(128), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
