"""
models.py — Typed Pydantic models for the Data Cleaning OpenEnv environment.
Defines Action, Observation, and State following the OpenEnv spec.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, field_validator


class DataCleaningAction(BaseModel):
    """
    An action the agent can take to clean the dataset.

    action_type choices:
        - fill_nulls        : fill missing values in a column
        - drop_duplicates   : remove duplicate rows
        - cast_column       : change a column's data type
        - remove_outliers   : drop rows where column value is an outlier
        - normalize_text    : strip whitespace and lowercase a text column
        - finish            : signal that the agent is done cleaning
    """
    action_type: str
    column: Optional[str] = None
    strategy: Optional[str] = None      # mean / median / mode / drop / ffill
    dtype: Optional[str] = None         # int / float / str / datetime / bool
    method: Optional[str] = None        # iqr / zscore
    metadata: Dict[str, Any] = {}


class ColumnInfo(BaseModel):
    """Summary statistics for a single column."""
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: List[Any]


class DataCleaningObservation(BaseModel):
    """
    What the agent sees after each step.
    Provides a structured view of the current dataset state.
    """
    task_id: str
    step: int
    columns: List[ColumnInfo]
    total_rows: int
    duplicate_rows: int
    outlier_counts: Dict[str, int]
    score_so_far: float
    done: bool
    message: str
    metadata: Dict[str, Any] = {}

    @field_validator("score_so_far", mode="before")
    @classmethod
    def clamp_score_so_far(cls, v: float) -> float:
        """Hard boundary at serialization: score must be strictly (0, 1)."""
        return max(0.01, min(0.99, round(float(v), 4)))


class DataCleaningState(BaseModel):
    """Internal episode metadata returned by state() endpoint."""
    episode_id: str
    task_id: str
    step_count: int
    max_steps: int
    actions_taken: List[str]
    current_score: float
    metadata: Dict[str, Any] = {}

    @field_validator("current_score", mode="before")
    @classmethod
    def clamp_current_score(cls, v: float) -> float:
        """Hard boundary at serialization: score must be strictly (0, 1)."""
        return max(0.01, min(0.99, round(float(v), 4)))
