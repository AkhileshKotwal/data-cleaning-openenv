"""Single-task dataset specification and grading helpers for Ops Workbench."""

from __future__ import annotations

import copy
import math
import re
from datetime import datetime
from statistics import mean, median
from typing import Any, Dict, List, Tuple


DIRTY_DATASET: List[Dict[str, Any]] = [
    {
        "Customer ID": "1001",
        "First Name": "maya",
        "signupDate": "2024-01-10",
        "country": "usa",
        "purchase_amount": "120.50",
        "Age": "29",
        "notes": "vip",
    },
    {
        "Customer ID": "1002",
        "First Name": "Jordan",
        "signupDate": "Jan 12 2024",
        "country": "USA",
        "purchase_amount": None,
        "Age": "31",
        "notes": "",
    },
    {
        "Customer ID": "1002",
        "First Name": "Jordan",
        "signupDate": "Jan 12 2024",
        "country": "USA",
        "purchase_amount": None,
        "Age": "31",
        "notes": "",
    },
    {
        "Customer ID": "1003",
        "First Name": "ana",
        "signupDate": "2024/01/14",
        "country": "U.S.A",
        "purchase_amount": "85",
        "Age": None,
        "notes": "new",
    },
    {
        "Customer ID": "1004",
        "First Name": None,
        "signupDate": "2024-01-18",
        "country": "Canada",
        "purchase_amount": "210.0",
        "Age": "40",
        "notes": "returning",
    },
]

EXPECTED_COLUMNS = [
    "customer_id",
    "first_name",
    "signup_date",
    "country",
    "purchase_amount",
    "age",
    "notes",
]

EXPECTED_DTYPES = {
    "customer_id": "int",
    "first_name": "str",
    "signup_date": "datetime",
    "country": "str",
    "purchase_amount": "float",
    "age": "int",
    "notes": "str",
}

CONSISTENCY_FIXES = {
    "country": {"usa": "USA", "U.S.A": "USA"},
    "first_name": {"maya": "Maya", "ana": "Ana"},
}

NULL_STRATEGIES = {
    "first_name": "mode",
    "purchase_amount": "median",
    "age": "median",
}

TASK_NAME = "data_cleaning"


def load_dataset() -> List[Dict[str, Any]]:
    return copy.deepcopy(DIRTY_DATASET)


def is_snake_case(name: str) -> bool:
    return bool(re.fullmatch(r"[a-z]+[a-z0-9_]*", name))


def canonicalize_row(row: Dict[str, Any], columns: List[str]) -> Tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def duplicate_row_count(rows: List[Dict[str, Any]], columns: List[str]) -> int:
    seen = set()
    duplicates = 0
    for row in rows:
        signature = canonicalize_row(row, columns)
        if signature in seen:
            duplicates += 1
        else:
            seen.add(signature)
    return duplicates


def infer_dtype(values: List[Any]) -> str:
    filtered = [value for value in values if value not in (None, "")]
    if not filtered:
        return "str"
    if all(isinstance(value, datetime) for value in filtered):
        return "datetime"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in filtered):
        return "int"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in filtered):
        return "float" if any(isinstance(value, float) for value in filtered) else "int"
    return "str"


def column_values(rows: List[Dict[str, Any]], column: str) -> List[Any]:
    return [row.get(column) for row in rows]


def non_null_values(rows: List[Dict[str, Any]], column: str) -> List[Any]:
    return [value for value in column_values(rows, column) if value not in (None, "")]


def safe_parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%b %d %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(float(str(value).strip()))


def snake_case_name(name: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def compute_grade(rows: List[Dict[str, Any]], columns: List[str]) -> Tuple[float, Dict[str, float]]:
    row_count = max(len(rows), 1)
    total_cells = max(len(columns) * row_count, 1)
    unresolved_nulls = sum(1 for row in rows for column in columns if row.get(column) in (None, ""))
    null_resolution = 1.0 - (unresolved_nulls / total_cells)

    dtype_matches = 0
    for column, expected_dtype in EXPECTED_DTYPES.items():
        if column in columns:
            actual_dtype = infer_dtype(column_values(rows, column))
            if actual_dtype == expected_dtype:
                dtype_matches += 1
    type_score = dtype_matches / len(EXPECTED_DTYPES)

    duplicates = duplicate_row_count(rows, columns)
    duplicate_score = 1.0 - min(duplicates / row_count, 1.0)

    consistency_total = 0
    consistency_matches = 0
    for column, replacements in CONSISTENCY_FIXES.items():
        if column not in columns:
            continue
        consistency_total += 1
        values = set(str(value) for value in non_null_values(rows, column))
        if not any(source in values for source in replacements):
            consistency_matches += 1
    consistency_score = consistency_matches / max(consistency_total, 1)

    snake_matches = sum(1 for column in columns if is_snake_case(column))
    snake_score = snake_matches / max(len(columns), 1)

    weighted = (
        0.30 * max(0.0, null_resolution)
        + 0.25 * type_score
        + 0.15 * duplicate_score
        + 0.20 * consistency_score
        + 0.10 * snake_score
    )
    # Clamp to strict (0,1) for task validator
    clipped = min(max(weighted, 0.01), 0.98)
    return clipped, {
        "null_resolution": round(null_resolution, 4),
        "correct_data_types": round(type_score, 4),
        "no_duplicate_rows": round(duplicate_score, 4),
        "value_consistency": round(consistency_score, 4),
        "snake_case_column_names": round(snake_score, 4),
    }


def inspect_dataset(rows: List[Dict[str, Any]], columns: List[str]) -> Dict[str, Any]:
    return {
        "shape": [len(rows), len(columns)],
        "columns": columns,
        "dtypes": {column: infer_dtype(column_values(rows, column)) for column in columns},
        "null_counts": {
            column: sum(1 for value in column_values(rows, column) if value in (None, "")) for column in columns
        },
        "duplicate_rows": duplicate_row_count(rows, columns),
    }


def inspect_column(rows: List[Dict[str, Any]], column: str) -> Dict[str, Any]:
    values = column_values(rows, column)
    non_null = [value for value in values if value not in (None, "")]
    result: Dict[str, Any] = {
        "dtype": infer_dtype(values),
        "null_count": len(values) - len(non_null),
        "null_percent": round(((len(values) - len(non_null)) / max(len(values), 1)) * 100, 2),
        "unique_count": len({str(value) for value in non_null}),
        "sample_values": [str(value) for value in non_null[:3]],
    }
    dtype = result["dtype"]
    if dtype in {"int", "float"}:
        numeric = [float(value) for value in non_null]
        result["min"] = min(numeric) if numeric else None
        result["max"] = max(numeric) if numeric else None
        result["mean"] = round(mean(numeric), 4) if numeric else None
    else:
        counts: Dict[str, int] = {}
        for value in non_null:
            text = str(value)
            counts[text] = counts.get(text, 0) + 1
        result["value_counts"] = counts
    return result


def recommended_strategy(column: str) -> str | None:
    return NULL_STRATEGIES.get(column)
