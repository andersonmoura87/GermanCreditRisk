"""Great Expectations suite smoke tests."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data.ge_suite import build_german_credit_suite, validate_dataframe_with_suite


def test_build_suite_filters_missing_columns() -> None:
    df = pd.DataFrame({"credit_risk": [0, 1], "credit_amount": [100.0, 200.0]})
    suite = build_german_credit_suite(df)
    assert suite.name == "german_credit_suite"
    assert len(suite.expectations) >= 1


@pytest.mark.slow
def test_validate_dataframe_with_suite_passes() -> None:
    df = pd.DataFrame(
        {
            "credit_risk": [0, 1],
            "credit_amount": [100.0, 200.0],
            "age": [30, 40],
            "credit_duration": [12, 24],
            "account_status": [1, 2],
            "installment_rate": [2, 3],
        }
    )
    result = validate_dataframe_with_suite(df)
    assert result.success
