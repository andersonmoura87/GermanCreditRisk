"""Fetch Brazilian macro series (BACEN SGS) via python-bcb and persist snapshots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_macro_br(
    start: str | None = None,
    series_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Download macro series from BACEN SGS.

    Args:
        start: ISO start date (defaults to ``params.yaml`` ``macro_br.start_date``).
        series_map: Mapping of friendly names to SGS codes (defaults to params).

    Returns:
        Wide dataframe indexed by date with one column per series.

    Raises:
        ImportError: If ``bcb`` is not installed.
        RuntimeError: If the remote API call fails (rede indisponível, etc.).
    """
    try:
        from bcb import sgs  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - ambiente sem rede/pacote
        raise ImportError("Instale python-bcb para usar fetch_macro_br.") from exc

    params = _load_params()
    macro_cfg: dict[str, Any] = params.get("macro_br", {})
    codes = series_map or macro_cfg.get(
        "series",
        {
            "selic": 432,
            "ipca": 433,
            "desemprego": 24369,
            "pib": 4380,
            "credito_pf": 20539,
            "inadimplencia": 21082,
        },
    )
    start_date = start or macro_cfg.get("start_date", "2015-01-01")
    df = sgs.get(codes, start=start_date)
    logger.info("Macro BR baixada: %s linhas, %s colunas", df.shape[0], df.shape[1])
    return df


def save_macro_snapshot(df: pd.DataFrame, path: Path | None = None) -> Path:
    """
    Persist macro data under ``data/raw/macro_brasil/``.

    Args:
        df: Macro dataframe from ``fetch_macro_br``.
        path: Optional explicit output path.

    Returns:
        Path written.
    """
    out = path or (PROJECT_ROOT / "data" / "raw" / "macro_brasil" / "macro_series.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=True)
    logger.info("Snapshot macro gravado em %s", out)
    return out
