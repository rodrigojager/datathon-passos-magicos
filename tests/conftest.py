import os
import sys
import tempfile
from pathlib import Path

import pytest


TEST_DB_PATH = Path(tempfile.gettempdir()) / "datathon_test.sqlite3"
if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database import Base, get_engine, init_db  # noqa: E402


@pytest.fixture(autouse=True)
def reset_database():
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)
