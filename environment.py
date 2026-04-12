"""Single-task data cleaning environment used by the server wrapper and inference script."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from .models import WorkbenchAction, WorkbenchObservation, WorkbenchState
    from .tasks import (
        CONSISTENCY_FIXES,
        EXPECTED_COLUMNS,
        TASK_NAME,
        as_float,
        as_int,
        canonicalize_row,
        column_values,
        compute_grade,
        inspect_column,
        inspect_dataset,
        load_dataset,
        non_null_values,
        recommended_strategy,
        safe_parse_datetime,
    )
except ImportError:
    from models import WorkbenchAction, WorkbenchObservation, WorkbenchState
    from tasks import (
        CONSISTENCY_FIXES,
        EXPECTED_COLUMNS,
        TASK_NAME,
        as_float,
        as_int,
        canonicalize_row,
        column_values,
        compute_grade,
        inspect_column,
        inspect_dataset,
        load_dataset,
        non_null_values,
        recommended_strategy,
        safe_parse_datetime,
    )


OBJECTIVE = (
    "Produce a clean, consistent, analysis-ready customer dataset by inspecting the loaded table, "
    "resolving nulls, correcting data types, removing duplicates, standardizing values, and ensuring "
    "all column names use snake_case."
)

TOOLS = [
    "inspect",
    "fill_nulls",
    "rename_column",
    "cast_type",
    "remove_duplicates",
    "fix_value",
    "normalize",
    "drop_column",
    "submit",
]

GRADING_CRITERIA = {
    "null_resolution": "Handle all nulls with the right strategy for each column.",
    "correct_data_types": "Ensure dates are datetime, IDs/counts are int, decimals are float, and text is str.",
    "no_duplicate_rows": "Remove all exact duplicate rows before submission.",
    "value_consistency": "Standardize inconsistent categorical values to a single canonical form.",
    "snake_case_column_names": "Rename every column to lowercase snake_case.",
}


class OpsWorkbenchEnv:
    """Deterministic data cleaning environment with shaped rewards."""

    def __init__(self, default_task: str = TASK_NAME) -> None:
        if default_task != TASK_NAME:
            raise ValueError(f"Unknown task: {default_task}")
        self._rows: List[Dict[str, Any]] = []
        self._columns: List[str] = []
        self._inspected_columns: set[str] = set()
        self._tool_result: Dict[str, Any] = {}
        self._state = WorkbenchState()
        self.reset(task_name=default_task)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **_: Any,
    ) -> WorkbenchObservation:
        del seed
        if task_name is not None and task_name != TASK_NAME:
            raise ValueError(f"Unknown task: {task_name}")
        self._rows = load_dataset()
        self._columns = list(self._rows[0].keys()) if self._rows else []
        self._inspected_columns = set()
        self._tool_result = {}
        initial_score, _ = compute_grade(self._rows, self._columns)
        self._state = WorkbenchState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_task=TASK_NAME,
            max_steps=30,
            submitted=False,
            consecutive_errors=0,
            progress_score=initial_score,
            last_action_error=None,
            history=[],
        )
        return self._build_observation(reward=0.0, done=False)

    def state(self) -> WorkbenchState:
        return self._state

    def step(
        self,
        action: WorkbenchAction,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> Tuple[WorkbenchObservation, float, bool, Dict[str, Any]]:
        del timeout_s
        self._state.step_count += 1
        previous_score, previous_breakdown = compute_grade(self._rows, self._columns)
        penalty = 0.0
        error: Optional[str] = None
        self._tool_result = {}

        if self._state.step_count == 1 and action.action_type != "inspect":
            penalty -= 0.10
            error = "The first action must be inspect(column=None)."
        elif action.action_type == "inspect":
            error = self._handle_inspect(action)
        elif action.action_type == "fill_nulls":
            error = self._handle_fill_nulls(action)
        elif action.action_type == "rename_column":
            error = self._handle_rename(action)
        elif action.action_type == "cast_type":
            error = self._handle_cast(action)
        elif action.action_type == "remove_duplicates":
            error = self._handle_remove_duplicates()
        elif action.action_type == "fix_value":
            error = self._handle_fix_value(action)
        elif action.action_type == "normalize":
            error = self._handle_normalize(action)
        elif action.action_type == "drop_column":
            error = self._handle_drop_column(action)
        elif action.action_type == "submit":
            self._state.submitted = True
        else:
            error = f"Unsupported action: {action.action_type}"

        if error:
            penalty -= 0.05
            self._state.consecutive_errors += 1
        else:
            self._state.consecutive_errors = 0

        if self._state.consecutive_errors > 5:
            penalty -= 0.10

        current_score, breakdown = compute_grade(self._rows, self._columns)
        done = False
        if self._state.submitted:
            done = True
            if self._state.step_count < 3:
                penalty -= 0.08
            if self._state.step_count > 25:
                penalty -= 0.05
        elif self._state.step_count >= 30:
            done = True
            penalty -= 0.08

        reward = round((current_score - previous_score) + penalty, 4)
        # Clamp reward: validator requires strictly != 0.0. Inspect steps produce 0.0 reward.
        # Nudge tiny values to a small positive so the log is never exactly 0.00.
        if abs(reward) < 0.005:
            reward = 0.005 if reward >= 0 else -0.005
        reward = round(max(-0.98, min(0.98, reward)), 4)
        self._state.progress_score = current_score
        self._state.last_action_error = error
        self._state.history.append(
            {
                "step": self._state.step_count,
                "action_type": action.action_type,
                "reward": reward,
                "score": current_score,
                "error": error,
                "breakdown": breakdown,
                "previous_breakdown": previous_breakdown,
            }
        )

        observation = self._build_observation(reward=reward, done=done, error=error)
        info = {"score": current_score, "grading_breakdown": breakdown}
        return observation, reward, done, info

    def grade(self) -> float:
        score, _ = compute_grade(self._rows, self._columns)
        return score

    def close(self) -> None:
        return None

    def _handle_inspect(self, action: WorkbenchAction) -> Optional[str]:
        if action.column is None:
            self._tool_result = inspect_dataset(self._rows, self._columns)
            return None
        if action.column not in self._columns:
            return f"Unknown column: {action.column}"
        self._inspected_columns.add(action.column)
        self._tool_result = inspect_column(self._rows, action.column)
        return None

    def _handle_fill_nulls(self, action: WorkbenchAction) -> Optional[str]:
        if not action.column or action.column not in self._columns:
            return f"Unknown column: {action.column}"
        if action.column not in self._inspected_columns:
            return f"Inspect column {action.column} before modifying it."
        if action.strategy is None:
            return "fill_nulls requires a strategy."
        values = non_null_values(self._rows, action.column)
        if action.strategy == "drop":
            self._rows = [row for row in self._rows if row.get(action.column) not in (None, "")]
        elif not values:
            return f"Column {action.column} has no non-null values to infer a fill value."
        else:
            fill_value: Any
            if action.strategy == "mean":
                fill_value = sum(float(value) for value in values) / len(values)
            elif action.strategy == "median":
                sorted_values = sorted(float(value) for value in values)
                midpoint = len(sorted_values) // 2
                fill_value = (
                    sorted_values[midpoint]
                    if len(sorted_values) % 2 == 1
                    else (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2
                )
                if all(float(value).is_integer() for value in sorted_values):
                    fill_value = int(fill_value)
            elif action.strategy == "mode":
                counts: Dict[str, int] = {}
                for value in values:
                    key = str(value)
                    counts[key] = counts.get(key, 0) + 1
                fill_value = max(sorted(counts.items()), key=lambda item: item[1])[0]
            else:
                return f"Unsupported strategy: {action.strategy}"
            for row in self._rows:
                if row.get(action.column) in (None, ""):
                    row[action.column] = fill_value
        self._tool_result = {"filled_column": action.column, "strategy": action.strategy}
        recommended = recommended_strategy(action.column)
        if recommended and recommended != action.strategy:
            return f"{action.strategy} is not the recommended strategy for {action.column}."
        return None

    def _handle_rename(self, action: WorkbenchAction) -> Optional[str]:
        if not action.old_name or action.old_name not in self._columns:
            return f"Unknown column: {action.old_name}"
        if not action.new_name:
            return "rename_column requires new_name."
        if action.new_name in self._columns:
            return f"Column already exists: {action.new_name}"
        for row in self._rows:
            row[action.new_name] = row.pop(action.old_name)
        self._columns = [action.new_name if column == action.old_name else column for column in self._columns]
        if action.old_name in self._inspected_columns:
            self._inspected_columns.remove(action.old_name)
            self._inspected_columns.add(action.new_name)
        self._tool_result = {"renamed": [action.old_name, action.new_name]}
        return None

    def _handle_cast(self, action: WorkbenchAction) -> Optional[str]:
        if not action.column or action.column not in self._columns:
            return f"Unknown column: {action.column}"
        if action.column not in self._inspected_columns:
            return f"Inspect column {action.column} before modifying it."
        if action.dtype is None:
            return "cast_type requires dtype."
        try:
            for row in self._rows:
                value = row.get(action.column)
                if action.dtype == "int":
                    row[action.column] = as_int(value)
                elif action.dtype == "float":
                    row[action.column] = as_float(value)
                elif action.dtype == "str":
                    row[action.column] = "" if value is None else str(value)
                elif action.dtype == "datetime":
                    parsed = safe_parse_datetime(value)
                    if value not in (None, "") and parsed is None:
                        return f"Could not parse datetime value in {action.column}."
                    row[action.column] = parsed
        except ValueError:
            return f"Failed to cast column {action.column} to {action.dtype}."
        self._tool_result = {"cast_column": action.column, "dtype": action.dtype}
        return None

    def _handle_remove_duplicates(self) -> Optional[str]:
        seen = set()
        cleaned: List[Dict[str, Any]] = []
        removed = 0
        for row in self._rows:
            signature = canonicalize_row(row, self._columns)
            if signature in seen:
                removed += 1
                continue
            seen.add(signature)
            cleaned.append(row)
        self._rows = cleaned
        self._tool_result = {"duplicates_removed": removed}
        return None

    def _handle_fix_value(self, action: WorkbenchAction) -> Optional[str]:
        if not action.column or action.column not in self._columns:
            return f"Unknown column: {action.column}"
        if action.column not in self._inspected_columns:
            return f"Inspect column {action.column} before modifying it."
        if action.find is None or action.replace is None:
            return "fix_value requires find and replace."
        replacements = 0
        for row in self._rows:
            if row.get(action.column) == action.find:
                row[action.column] = action.replace
                replacements += 1
        self._tool_result = {"column": action.column, "replacements": replacements}
        allowed = CONSISTENCY_FIXES.get(action.column, {})
        if action.find in allowed and allowed[action.find] != action.replace:
            return f"{action.replace} is not the canonical replacement for {action.find} in {action.column}."
        return None

    def _handle_normalize(self, action: WorkbenchAction) -> Optional[str]:
        if not action.column or action.column not in self._columns:
            return f"Unknown column: {action.column}"
        values = [as_float(value) for value in column_values(self._rows, action.column)]
        numeric = [value for value in values if value is not None]
        if not numeric:
            return f"Column {action.column} is not numeric."
        min_value = min(numeric)
        max_value = max(numeric)
        if max_value == min_value:
            return f"Column {action.column} cannot be normalized because all values are identical."
        for row in self._rows:
            value = as_float(row.get(action.column))
            if value is not None:
                row[action.column] = round((value - min_value) / (max_value - min_value), 6)
        self._tool_result = {"normalized_column": action.column}
        return None

    def _handle_drop_column(self, action: WorkbenchAction) -> Optional[str]:
        if not action.column or action.column not in self._columns:
            return f"Unknown column: {action.column}"
        for row in self._rows:
            row.pop(action.column, None)
        self._columns = [column for column in self._columns if column != action.column]
        self._inspected_columns.discard(action.column)
        self._tool_result = {"dropped_column": action.column}
        return None

    def _preview(self) -> List[Dict[str, str]]:
        preview_rows: List[Dict[str, str]] = []
        for row in self._rows[:3]:
            preview_rows.append(
                {
                    column: (
                        row[column].isoformat() if hasattr(row[column], "isoformat") else str(row[column])
                    )
                    for column in self._columns
                }
            )
        return preview_rows

    def _build_observation(
        self,
        reward: float,
        done: bool,
        error: Optional[str] = None,
    ) -> WorkbenchObservation:
        dataset_info = inspect_dataset(self._rows, self._columns)
        score, _ = compute_grade(self._rows, self._columns)
        # Enforce strict open interval (0, 1) as required by the hackathon validator.
        safe_score = round(max(0.01, min(0.98, score)), 4)
        return WorkbenchObservation(
            objective=OBJECTIVE,
            tools=TOOLS,
            grading_criteria=GRADING_CRITERIA,
            dataset_columns=list(self._columns),
            dataset_preview=self._preview(),
            current_dtypes=dataset_info["dtypes"],
            null_counts=dataset_info["null_counts"],
            duplicate_rows=dataset_info["duplicate_rows"],
            tool_result=copy.deepcopy(self._tool_result),
            inspected_columns=sorted(self._inspected_columns),
            progress=safe_score,
            last_action_error=error,
            reward=reward,
            done=done,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "max_steps": self._state.max_steps,
                "expected_columns": EXPECTED_COLUMNS,
            },
        )
