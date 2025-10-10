from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

# Default fallback category labels per column when unmapped values appear.
_DEFAULT_CATEGORY_LABELS: Dict[str, str] = {
    "A/C ON/OFF": "OFF",
    "A/C Mode": "FAN",
    "A/C Fan Speed": "Low",
    "A/C Status": "OFF",
}


def _normalize_key(value: Any) -> str:
    """Normalize a categorical value for lookup (case-insensitive, trimmed)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().upper()
    if pd.isna(value):  # Handles NaN-like objects
        return ""
    return str(value).strip().upper()


@lru_cache(maxsize=None)
def load_category_mappings() -> Dict[str, Dict[str, int]]:
    """Load category mappings from `category_mapping.json` once and cache the result."""
    mapping_path = Path(__file__).resolve().parents[2] / "config/category_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as fp:
        raw_mapping: Dict[str, Dict[str, int]] = json.load(fp)
    return raw_mapping


def get_category_mapping(column: str) -> Dict[str, int]:
    """Return a copy of the mapping dictionary for the specified column."""
    mapping = load_category_mappings().get(column)
    if mapping is None:
        raise KeyError(f"No category mapping defined for column '{column}'.")
    return dict(mapping)


def get_inverse_category_mapping(column: str) -> Dict[int, str]:
    """Return a value-to-label mapping for the specified column."""
    mapping = get_category_mapping(column)
    inverse: Dict[int, str] = {}
    for label, code in mapping.items():
        inverse.setdefault(code, label)
    return inverse


def get_normalized_category_mapping(column: str) -> Dict[str, int]:
    """Return a normalized-key mapping for the specified column."""
    mapping = get_category_mapping(column)
    return {_normalize_key(label): code for label, code in mapping.items()}


def get_default_category_value(column: str) -> int | None:
    """Return a default numeric value for the column, if defined."""
    mapping = get_category_mapping(column)
    default_label = _DEFAULT_CATEGORY_LABELS.get(column)
    if default_label and default_label in mapping:
        return mapping[default_label]
    return next(iter(mapping.values())) if mapping else None


def map_category_series(
    series: pd.Series, column: str
) -> Tuple[pd.Series, Dict[str, int], Dict[str, int]]:
    """Map a pandas Series to numeric codes using the configured category mapping.

    Returns the mapped series, a dictionary of applied mappings (original value -> code),
    and a dictionary of unmapped value counts for logging/debugging purposes.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    # If the series is already numeric and aligned with defined codes, return as-is.
    if is_numeric_dtype(series):
        defined_codes = set(get_category_mapping(column).values())
        present_codes = set(series.dropna().unique())
        if present_codes <= defined_codes:
            applied_mapping = {str(code): int(code) for code in present_codes}
            return series, applied_mapping, {}

    normalized_mapping = get_normalized_category_mapping(column)

    mapped_series = series.map(lambda value: normalized_mapping.get(_normalize_key(value)))

    unmapped_mask = mapped_series.isna() & series.notna()
    unmapped_values: Dict[str, int] = (
        series.loc[unmapped_mask].value_counts(dropna=False).astype(int).to_dict()
        if unmapped_mask.any()
        else {}
    )

    applied_mapping: Dict[str, int] = {}
    for original_value in series.dropna().unique():
        code = normalized_mapping.get(_normalize_key(original_value))
        if code is not None:
            applied_mapping[str(original_value)] = code

    return mapped_series, applied_mapping, unmapped_values


def normalize_candidate_values(
    column: str,
    candidates: Optional[Iterable[Any]],
    preferred_labels: Sequence[str] = (),
) -> List[int]:
    """Normalize candidate values for a categorical column using configured mappings."""
    mapping = get_category_mapping(column)
    normalized_mapping = get_normalized_category_mapping(column)
    available_codes_ordered = list(mapping.values())
    available_codes = set(available_codes_ordered)

    normalized: List[int] = []
    if candidates is not None:
        for value in candidates:
            code: Optional[int] = None
            if isinstance(value, str):
                code = normalized_mapping.get(_normalize_key(value))
            else:
                try:
                    candidate_int = int(value)
                except (TypeError, ValueError):
                    code = None
                else:
                    if candidate_int in available_codes:
                        code = candidate_int
            if code is not None:
                normalized.append(code)

    if normalized:
        return normalized

    for label in preferred_labels:
        code = mapping.get(label)
        if code is not None:
            normalized.append(code)

    if normalized:
        return normalized

    return available_codes_ordered


__all__ = [
    "get_category_mapping",
    "get_inverse_category_mapping",
    "get_normalized_category_mapping",
    "get_default_category_value",
    "map_category_series",
    "normalize_candidate_values",
    "load_category_mappings",
]
