"""Pandera schema validation for the interim German Credit table."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml
from pandera import Check, Column, DataFrameSchema

from src.data.ge_suite import validate_dataframe_with_suite

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def german_credit_schema() -> DataFrameSchema:
    """
    Build a Pandera schema for the cleaned German Credit interim table.

    Returns:
        ``DataFrameSchema`` with core credit-risk columns enforced.
    """
    # Após clean_german_credit, colunas numéricas + credit_risk
    return DataFrameSchema(
        {
            "credit_risk": Column(int, Check.isin([0, 1]), coerce=True),
            "credit_amount": Column(float, Check.ge(0), coerce=True),
            "age": Column(int, Check.in_range(16, 120), coerce=True),
            "credit_duration": Column(int, Check.gt(0), coerce=True),
        },
        strict=False,
    )


def validate_interim(interim_path: Path) -> pd.DataFrame:
    """
    Validate interim Parquet against the Pandera schema.

    Args:
        interim_path: Path to ``german_credit.parquet``.

    Returns:
        Validated dataframe.

    Raises:
        pa.errors.SchemaError: If validation fails.
    """
    df = pd.read_parquet(interim_path)
    schema = german_credit_schema()
    schema.validate(df)
    validate_dataframe_with_suite(df)
    return df


def run() -> Path:
    """
    Validate interim dataset and write a success flag for DVC orchestration.

    Returns:
        Path to the validation flag file.

    Note:
        Great Expectations roda em memória via ``src.data.ge_suite``; JSON versionado em
        ``great_expectations/expectations/german_credit_suite.json`` (exportar com
        ``python -m src.data.ge_suite``).
    """
    logging.basicConfig(level=logging.INFO)
    params = _load_params()
    interim = PROJECT_ROOT / params["paths"]["interim_parquet"]
    validate_interim(interim)
    flag = interim.parent / "validation_ok.flag"
    flag.write_text("ok\n", encoding="utf-8")
    logger.info("Validação Pandera OK. Flag em %s", flag)
    return flag


def main() -> None:
    run()


if __name__ == "__main__":
    main()
