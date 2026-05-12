"""Great Expectations suite for the German Credit interim table (GX Core 1.x)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import great_expectations as gx
import pandas as pd
from great_expectations.core.expectation_validation_result import ExpectationSuiteValidationResult

logger = logging.getLogger(__name__)

SUITE_NAME = "german_credit_suite"


def _column_from_expectation(exp: gx.expectations.Expectation) -> str | None:
    """Return the column name if the expectation targets a single column."""
    kwargs = getattr(exp, "kwargs", {}) or {}
    return kwargs.get("column")


def build_german_credit_suite(df: pd.DataFrame) -> gx.ExpectationSuite:
    """
    Build the expectation suite for German Credit, skipping columns absent from ``df``.

    Args:
        df: Interim modeling frame (defines available columns).

    Returns:
        ``ExpectationSuite`` with domain checks aligned to credit modeling.
    """
    available = set(df.columns)
    candidates: list[Any] = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="credit_risk"),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="credit_risk", min_value=0, max_value=1
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="credit_amount"),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="credit_amount", min_value=0, max_value=2_000_000
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="age"),
        gx.expectations.ExpectColumnValuesToBeBetween(column="age", min_value=16, max_value=120),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="credit_duration"),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="credit_duration", min_value=1, max_value=120
        ),
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1, max_value=50_000),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="account_status", min_value=1, max_value=4
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="installment_rate", min_value=1, max_value=4
        ),
    ]
    expectations: list[Any] = []
    for exp in candidates:
        col = _column_from_expectation(cast(gx.expectations.Expectation, exp))
        if col is None:
            expectations.append(exp)
        elif col in available:
            expectations.append(exp)
    return gx.ExpectationSuite(name=SUITE_NAME, expectations=expectations)


def suite_json_path(project_root: Path) -> Path:
    """Path to the committed JSON export of the suite (for review / tooling)."""
    return project_root / "great_expectations" / "expectations" / f"{SUITE_NAME}.json"


def export_suite_json(project_root: Path) -> Path:
    """
    Write ``suite_json_path`` from the Python-defined expectations.

    Uses a minimal empty frame to list column-based expectations present in schema.

    Args:
        project_root: Repository root containing ``great_expectations/``.

    Returns:
        Path written.
    """
    path = suite_json_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    placeholder = pd.DataFrame(
        {
            "credit_risk": [0, 1],
            "credit_amount": [1.0, 2.0],
            "age": [30, 40],
            "credit_duration": [12, 24],
            "account_status": [1, 2],
            "installment_rate": [2, 3],
        }
    )
    suite = build_german_credit_suite(placeholder)
    path.write_text(json.dumps(suite.to_json_dict(), indent=2), encoding="utf-8")
    logger.info("Suíte GX exportada para %s", path)
    return path


def validate_dataframe_with_suite(df: pd.DataFrame) -> ExpectationSuiteValidationResult:
    """
    Run the German Credit suite against an in-memory DataFrame (ephemeral context).

    Args:
        df: Interim modeling frame.

    Returns:
        GX validation result object (check ``.success``).

    Raises:
        RuntimeError: If validation fails.
    """
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("german_credit_interim")
    asset = datasource.add_dataframe_asset("interim_frame")
    batch_def = asset.add_batch_definition_whole_dataframe("whole_frame")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    suite = build_german_credit_suite(df)
    result = batch.validate(suite)
    if not result.success:
        stats: dict[str, Any] = getattr(result, "statistics", {}) or {}
        logger.error("Great Expectations falhou: %s", stats)
        raise RuntimeError(f"Great Expectations validation failed: {stats}")
    logger.info("Great Expectations OK (%s expectativas).", len(suite.expectations))
    return result


def main() -> None:
    """CLI: exporta JSON da suíte para ``great_expectations/expectations/``."""
    logging.basicConfig(level=logging.INFO)
    export_suite_json(Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
