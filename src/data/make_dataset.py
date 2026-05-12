"""Load raw German Credit CSV and persist a cleaned interim Parquet dataset."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    path = PROJECT_ROOT / "params.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = bytes(pd.util.hash_pandas_object(df, index=True).values)  # type: ignore[arg-type]
    return hashlib.sha256(payload).hexdigest()[:16]


def load_raw_csv(raw_path: Path) -> pd.DataFrame:
    """
    Read the semicolon-separated German Credit dataset.

    Args:
        raw_path: Path to ``credit_risk_dataset.csv``.

    Returns:
        Raw table as a DataFrame.

    Raises:
        FileNotFoundError: If ``raw_path`` does not exist.
    """
    if not raw_path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {raw_path}")
    # encoding padrão; dataset público UCI / South German Credit
    df = pd.read_csv(raw_path, sep=";")
    return df


def clean_german_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce types and harmonize sparse categorical columns for downstream modeling.

    Args:
        df: Raw dataframe from ``load_raw_csv``.

    Returns:
        Cleaned dataframe with ``credit_risk`` as int {0,1}.
    """
    out = df.copy()
    if "credit_risk" not in out.columns:
        raise ValueError("Coluna 'credit_risk' obrigatória ausente.")

    yn_map = {"yes": 1, "no": 0}
    if "foreign_worker" in out.columns:
        out["foreign_worker"] = out["foreign_worker"].map(yn_map).fillna(-1).astype(int)

    if "telephone" in out.columns:
        out["telephone"] = pd.to_numeric(out["telephone"], errors="coerce").fillna(-1).astype(int)

    for col in out.columns:
        if col in ("foreign_worker", "telephone", "credit_risk"):
            continue
        if out[col].dtype == object:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.fillna(-1)
    out["credit_risk"] = out["credit_risk"].astype(int)
    return out


def run(raw_path: Path | None = None, interim_path: Path | None = None) -> Path:
    """
    End-to-end ingest: read raw CSV, clean, write Parquet.

    Args:
        raw_path: Optional override for raw CSV path.
        interim_path: Optional override for output Parquet path.

    Returns:
        Path to the written interim Parquet file.
    """
    logging.basicConfig(level=logging.INFO)
    params = _load_params()
    raw = Path(raw_path or params["paths"]["raw_csv"])
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    interim = Path(interim_path or params["paths"]["interim_parquet"])
    if not interim.is_absolute():
        interim = PROJECT_ROOT / interim

    interim.parent.mkdir(parents=True, exist_ok=True)
    df = load_raw_csv(raw)
    cleaned = clean_german_credit(df)
    fp = _dataset_fingerprint(cleaned)
    logger.info("Fingerprint do dataset (sha256 truncado): %s", fp)
    cleaned.to_parquet(interim, index=False)
    logger.info("Interim gravado em %s", interim)
    return interim


def main() -> None:
    run()


if __name__ == "__main__":
    main()
