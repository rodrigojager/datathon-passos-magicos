import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "dados"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ROOT_DIR / "app" / "model"
LOG_DIR = ARTIFACTS_DIR / "logs"
EXPORTS_DIR = ARTIFACTS_DIR / "exports"
PREDICTIONS_DIR = EXPORTS_DIR / "predictions"
REPORTS_DIR = EXPORTS_DIR / "reports"

default_sqlite_path = (ROOT_DIR / "datathon.db").as_posix()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{default_sqlite_path}")
ROOT_PATH_PREFIX = os.getenv("ROOT_PATH", "")

MODEL_BUNDLE_PATH = MODEL_DIR / "student_risk_model.joblib"
CANDIDATE_BUNDLES_PATH = MODEL_DIR / "candidate_models.joblib"
VALIDATION_CANDIDATE_BUNDLES_PATH = MODEL_DIR / "validation_candidate_models.joblib"
TRAINING_REPORT_PATH = REPORTS_DIR / "training_report.json"
BASELINE_ANALYSIS_PATH = REPORTS_DIR / "baseline_analysis.json"
TRANSITION_SCORES_PATH = REPORTS_DIR / "transition_scores.json"
LATEST_EVALUATION_PATH = REPORTS_DIR / "latest_evaluation.csv"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.html"
DRIFT_SUMMARY_PATH = REPORTS_DIR / "drift_summary.json"
APP_LOG_PATH = LOG_DIR / "app.log"


def ensure_directories() -> None:
    for directory in (
        ARTIFACTS_DIR,
        MODEL_DIR,
        LOG_DIR,
        EXPORTS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
