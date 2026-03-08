import pandas as pd

from src.normalization import NormalizationResult
from src.storage import ingest_normalized_dataframe, load_consolidated_dataframe


def test_ingest_deduplicates_by_ra_and_reference_year():
    dataframe = pd.DataFrame(
        {
            "ra": ["RA-1", "RA-2"],
            "ano_referencia": [2023, 2023],
            "fase": ["7", "8"],
            "source_schema": ["pede_2023_like", "pede_2023_like"],
            "source_file": ["arquivo.csv", "arquivo.csv"],
            "source_row_number": [1, 2],
        }
    )
    result = NormalizationResult(
        dataframe=dataframe,
        detected_schema="pede_2023_like",
        detected_schema_year=2023,
        reference_year=2023,
        renamed_columns={"RA": "ra"},
        unknown_columns=[],
        source_filename="arquivo.csv",
    )

    first = ingest_normalized_dataframe(result)
    second = ingest_normalized_dataframe(result)

    consolidated = load_consolidated_dataframe()
    assert first.inserted_rows == 2
    assert second.inserted_rows == 0
    assert second.duplicate_rows == 2
    assert len(consolidated) == 2
    assert first.data_version.startswith("v")
