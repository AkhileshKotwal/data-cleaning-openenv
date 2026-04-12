"""
server/environment.py — Data Cleaning OpenEnv v4.0

4 Tasks:
  fix_nulls       (Easy)   : Fill missing values correctly
  fix_types       (Medium) : Fix dtypes + remove duplicates
  full_clean      (Hard)   : Nulls + types + duplicates + outliers + text
  deceptive_clean (Expert) : Conflicting signals, hidden traps, protected data

Features:
  - Strategic adversary that targets your best column
  - Catastrophic penalties for destroying critical data
  - Action cost penalises brute-force cleaning
  - Protected nulls, valid outliers, hidden traps
  - Hints system reveals dataset structure without giving answers
"""

import uuid
import random
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ColumnInfo,
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
)

ACTION_COST = 0.04  # every step costs this — rewards efficient agents

# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "fix_nulls": {
        "max_steps": 10,
        "protected_nulls": [],
        "valid_outlier_cols": [],
        "required_strategies": {
            "age":    ["mean", "median"],
            "salary": ["mean", "median"],
            "score":  ["mean", "median"],
            "city":   ["mode"],
        },
        "critical_cols": [],
        "drop_penalty_cols": [],
    },
    "fix_types": {
        "max_steps": 15,
        "protected_nulls": ["join_date"],
        "valid_outlier_cols": [],
        "required_strategies": {
            "age":        ["mean", "median"],
            "salary":     ["mean", "median"],
            "department": ["mode"],
        },
        "critical_cols": ["user_id"],
        "drop_penalty_cols": ["user_id"],
    },
    "full_clean": {
        "max_steps": 20,
        "protected_nulls": ["email"],
        "valid_outlier_cols": ["rating"],
        "required_strategies": {
            "age":        ["mean", "median"],
            "salary":     ["median"],
            "rating":     ["mean", "median"],
            "department": ["mode"],
        },
        "critical_cols": ["emp_id"],
        "drop_penalty_cols": ["emp_id"],
    },
    "deceptive_clean": {
        "max_steps": 15,
        "protected_nulls": ["notes"],
        "valid_outlier_cols": ["revenue"],
        "required_strategies": {
            "age":     ["median"],
            "revenue": ["median"],
            "region":  ["mode"],
        },
        "critical_cols": ["customer_id"],
        "drop_penalty_cols": ["customer_id", "revenue"],
    },
}

MAX_STEPS = {k: v["max_steps"] for k, v in TASK_CONFIGS.items()}


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def _make_easy_dataset(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n = 100
    return pd.DataFrame({
        "age":    np.where(np.random.rand(n) < 0.15, np.nan,
                           np.random.randint(18, 70, n).astype(float)),
        "salary": np.where(np.random.rand(n) < 0.10, np.nan,
                           np.random.randint(30000, 120000, n).astype(float)),
        "score":  np.where(np.random.rand(n) < 0.20, np.nan,
                           np.round(np.random.uniform(0, 100, n), 2)),
        "city":   pd.array(np.where(np.random.rand(n) < 0.12, None,
                           np.random.choice(["Mumbai", "Delhi", "Pune", "Bangalore"], n)),
                           dtype=object),
    })


def _make_medium_dataset(seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    n = 120
    df = pd.DataFrame({
        "user_id":    [str(i) for i in range(n)],
        "age":        np.where(np.random.rand(n) < 0.15, np.nan,
                               np.random.randint(18, 70, n).astype(float)),
        "salary":     np.where(np.random.rand(n) < 0.10, np.nan,
                               np.random.randint(30000, 120000, n).astype(float)),
        "is_active":  np.random.choice(["True", "False", "true", "false", "1", "0"], n),
        "join_date":  np.random.choice(["2021-01-15", "2020-06-30", "2022-11-01",
                                        "not-a-date", None], n),
        "department": pd.array(np.where(np.random.rand(n) < 0.10, None,
                               np.random.choice(["HR", "Eng", "Sales", "Mktg"], n)),
                               dtype=object),
    })
    dups = df.sample(15, random_state=1).copy()
    return pd.concat([df, dups], ignore_index=True)


def _make_hard_dataset(seed: int = 99) -> pd.DataFrame:
    np.random.seed(seed)
    n = 150
    salaries = np.random.randint(30000, 120000, n).astype(float)
    salaries[np.random.choice(n, 8, replace=False)] = np.random.choice([5, 9999999], 8)
    ratings = np.round(np.random.uniform(1, 4.5, n), 1)
    ratings[np.random.choice(n, 5, replace=False)] = 5.0
    df = pd.DataFrame({
        "emp_id":     [str(i) for i in range(n)],
        "name":       np.random.choice(["  Alice ", "BOB", "charlie  ", " Diana", "EVE  "], n),
        "age":        np.where(np.random.rand(n) < 0.15, np.nan,
                               np.random.randint(18, 70, n).astype(float)),
        "salary":     np.where(np.random.rand(n) < 0.10, np.nan, salaries),
        "rating":     np.where(np.random.rand(n) < 0.12, np.nan, ratings),
        "department": pd.array(np.where(np.random.rand(n) < 0.10, None,
                               np.random.choice(["  HR", "Engineering ", "Sales", " Marketing "], n)),
                               dtype=object),
        "email":      pd.array(np.where(np.random.rand(n) < 0.08, None,
                               np.random.choice(["alice@co.com", "BOB@CO.COM",
                                                 "charlie@co.com", "valid@co.com"], n)),
                               dtype=object),
    })
    dups = df.sample(20, random_state=5).copy()
    return pd.concat([df, dups], ignore_index=True)


def _make_deceptive_dataset(seed: int = 13) -> pd.DataFrame:
    """
    Expert dataset with conflicting signals designed to punish naive agents.

    Traps:
    - revenue looks like outliers (high values) but they ARE real high earners
    - age is right-skewed — mean misleads, use median
    - notes column ~40% null — intentional (no note = no issue), DO NOT fill
    - ~10 rows look like duplicates but have different customer_ids (repeat purchases)
    - customer_id drop = instant catastrophic episode death
    - region has messy text — normalize IS correct here
    """
    np.random.seed(seed)
    n = 130

    revenue = np.random.randint(10000, 100000, n).astype(float)
    rich_idx = np.random.choice(n, 8, replace=False)
    revenue[rich_idx] = np.random.randint(500000, 2000000, 8).astype(float)

    age = np.random.exponential(scale=15, size=n) + 20
    age = np.clip(age, 18, 85).round(0)
    age = np.where(np.random.rand(n) < 0.12, np.nan, age)

    notes = pd.array(np.where(np.random.rand(n) < 0.40, None,
                     np.random.choice(["VIP", "at risk", "new", "churned"], n)), dtype=object)

    region = np.random.choice(["  North", "South  ", "EAST", "west ", " Central"], n)
    region = np.where(np.random.rand(n) < 0.08, None, region)

    satisfaction = np.where(np.random.rand(n) < 0.15, np.nan,
                            np.round(np.random.uniform(1, 10, n), 1))

    df = pd.DataFrame({
        "customer_id":  [f"C{i:04d}" for i in range(n)],
        "age":          age,
        "revenue":      np.where(np.random.rand(n) < 0.08, np.nan, revenue),
        "region":       pd.array(region, dtype=object),
        "notes":        notes,
        "satisfaction": satisfaction,
    })

    # Near-duplicates: same customer data, different customer_id (repeat purchases)
    near_dups = df.sample(10, random_state=7).copy()
    near_dups["revenue"] = near_dups["revenue"] * np.random.uniform(0.8, 1.2, 10)
    near_dups["customer_id"] = [f"C{i:04d}" for i in range(n, n + 10)]
    return pd.concat([df, near_dups], ignore_index=True)


TASK_DATASETS = {
    "fix_nulls":       _make_easy_dataset,
    "fix_types":       _make_medium_dataset,
    "full_clean":      _make_hard_dataset,
    "deceptive_clean": _make_deceptive_dataset,
}


# ---------------------------------------------------------------------------
# Strategic adversary — targets the column with most improvement
# ---------------------------------------------------------------------------

class StrategicAdversary:
    """
    After every agent step, targets the column where the agent made the most
    progress and undoes it. Much harder than random corruption.
    """

    def __init__(self, difficulty: float = 0.3):
        self.difficulty = max(0.0, min(1.0, difficulty))

    def corrupt(self, df: pd.DataFrame, original_df: pd.DataFrame,
                protected_nulls: List[str]) -> Tuple[pd.DataFrame, str]:
        if self.difficulty == 0.0:
            return df, ""

        n_attacks = max(1, int(self.difficulty * 3))
        messages = []

        for _ in range(n_attacks):
            target_col = self._find_best_target(df, original_df, protected_nulls)
            attack = self._choose_attack(df, target_col)
            try:
                if attack == "inject_null" and target_col:
                    idx = random.randint(0, len(df) - 1)
                    df = df.copy()
                    df.at[idx, target_col] = None
                    messages.append(f"re-nulled '{target_col}'")

                elif attack == "add_duplicate":
                    dup = df.sample(1, random_state=random.randint(0, 9999))
                    df = pd.concat([df, dup], ignore_index=True)
                    messages.append("re-introduced duplicate")

                elif attack == "corrupt_numeric" and target_col:
                    if pd.api.types.is_numeric_dtype(df[target_col]):
                        idx = random.randint(0, len(df) - 1)
                        df = df.copy()
                        q3 = df[target_col].quantile(0.75)
                        df.at[idx, target_col] = q3 * 100
                        messages.append(f"injected outlier in '{target_col}'")
            except Exception:
                pass

        msg = " | ".join(messages)
        return df, f"⚔️ Adversary: {msg}" if msg else ""

    def _find_best_target(self, df, original_df, protected_nulls):
        best_col, best_improvement = None, 0
        for col in df.columns:
            if col in protected_nulls or col not in original_df.columns:
                continue
            improvement = original_df[col].isnull().sum() - df[col].isnull().sum()
            if improvement > best_improvement:
                best_improvement = improvement
                best_col = col
        return best_col

    def _choose_attack(self, df, target_col):
        if target_col and pd.api.types.is_numeric_dtype(
                df.get(target_col, pd.Series(dtype=float))):
            return random.choice(["inject_null", "corrupt_numeric", "add_duplicate"])
        return random.choice(["inject_null", "add_duplicate"])


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _null_score(df, original_df, protected_nulls):
    orig = sum(original_df[c].isnull().sum() for c in original_df.columns
               if c not in protected_nulls)
    rem  = sum(df[c].isnull().sum() for c in df.columns
               if c not in protected_nulls and c in original_df.columns)
    return 1.0 if orig == 0 else max(0.0, 1.0 - rem / orig)


def _duplicate_score(df, original_df):
    orig = original_df.duplicated().sum()
    return 1.0 if orig == 0 else max(0.0, 1.0 - df.duplicated().sum() / orig)


def _outlier_score(df, original_df, numeric_cols, valid_outlier_cols):
    total_orig = total_rem = 0
    for col in numeric_cols:
        if col not in df.columns or col not in original_df.columns:
            continue
        if col in valid_outlier_cols:
            continue
        q1, q3 = original_df[col].quantile(0.25), original_df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        total_orig += int(((original_df[col] < lo) | (original_df[col] > hi)).sum())
        total_rem  += int(((df[col] < lo) | (df[col] > hi)).sum())
    return 1.0 if total_orig == 0 else max(0.0, 1.0 - total_rem / total_orig)


def _text_score(df, text_cols):
    total = clean = 0
    for col in text_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        total += len(s)
        clean += int((s == s.str.strip().str.lower()).sum())
    raw = clean / total if total else 1.0  # no text cols → perfectly clean
    return max(0.0, min(1.0, raw))


def _clamp(score: float) -> float:
    """Clamp score strictly between 0 and 1 (exclusive) as required by validator.
    Bounds are (0.01, 0.99) — well away from 0 and 1, and NOT at the sub-scorer
    ceiling (which is now 1.0 raw), so a perfect clean returns 0.99, not a sentinel."""
    s = float(score)
    if s <= 0.0:
        return 0.01
    if s >= 1.0:
        return 0.99
    return max(0.01, min(0.99, s))


def _clamp_reward(reward: float) -> float:
    """
    Clamp reward so it is never exactly 0.0, 1.0, -1.0 or ±∞.
    Negative rewards ARE valid per openenv.yaml spec (range: [-1.0, 1.0]).
    Near-zero values (|r| < 0.005) are nudged to ±0.01.
    Bounds: (-0.99, 0.99) — well inside (-1, 1).
    """
    r = float(reward)
    if r <= -0.99:
        return -0.99
    if r >= 0.99:
        return 0.99
    if -0.005 < r < 0.005:
        return 0.01 if r >= 0 else -0.01
    return r


def grade(task_id: str, df: pd.DataFrame, original_df: pd.DataFrame) -> float:
    cfg = TASK_CONFIGS[task_id]
    p = cfg["protected_nulls"]
    v = cfg["valid_outlier_cols"]

    if task_id == "fix_nulls":
        return _clamp(_null_score(df, original_df, p))

    elif task_id == "fix_types":
        return _clamp(0.5 * _null_score(df, original_df, p)
              + 0.5 * _duplicate_score(df, original_df))

    elif task_id == "full_clean":
        return _clamp(0.30 * _null_score(df, original_df, p)
              + 0.25 * _duplicate_score(df, original_df)
              + 0.25 * _outlier_score(df, original_df, ["salary", "age", "rating"], v)
              + 0.20 * _text_score(df, ["name", "department"]))

    elif task_id == "deceptive_clean":
        return _clamp(0.30 * _null_score(df, original_df, p)
              + 0.25 * _outlier_score(df, original_df, ["revenue", "age", "satisfaction"], v)
              + 0.25 * _text_score(df, ["region"])
              + 0.20 * _duplicate_score(df, original_df))

    return 0.01  # unknown task_id — safe fallback


# ---------------------------------------------------------------------------
# Penalty engine
# ---------------------------------------------------------------------------

CATASTROPHIC = -1.0
SEVERE       = -0.35
MODERATE     = -0.20
MINOR        = -0.12


def compute_penalty(action: DataCleaningAction, df: pd.DataFrame,
                    task_id: str) -> Tuple[float, str, bool]:
    """Returns (penalty, message, is_catastrophic)."""
    cfg = TASK_CONFIGS[task_id]
    col = action.column

    # Catastrophic: dropping a critical column via fill_nulls drop strategy
    if action.action_type == "fill_nulls" and action.strategy == "drop":
        if col in cfg.get("critical_cols", []):
            return (CATASTROPHIC,
                    f"CATASTROPHIC: Dropped critical column '{col}'. "
                    f"This is a primary key — never drop it. Episode terminated.",
                    True)

    # Catastrophic: removing outliers from a valid-outlier column
    if action.action_type == "remove_outliers" and col in cfg["valid_outlier_cols"]:
        return (CATASTROPHIC,
                f"CATASTROPHIC: Removed valid data from '{col}'. "
                f"These extreme values are real. Episode terminated.",
                True)

    # Severe: filling a protected null column
    if action.action_type == "fill_nulls" and col in cfg["protected_nulls"]:
        return (SEVERE,
                f"PENALTY -{abs(SEVERE)}: '{col}' nulls are intentional. "
                f"Filling them corrupts data meaning.",
                False)

    # Severe/moderate: wrong strategy for a column
    if action.action_type == "fill_nulls" and col and action.strategy:
        required = cfg["required_strategies"].get(col, [])
        if required and action.strategy not in required and action.strategy != "drop":
            severity = SEVERE if col in cfg.get("critical_cols", []) else MODERATE
            return (severity,
                    f"PENALTY {severity}: Wrong strategy '{action.strategy}' for '{col}'. "
                    f"Correct options: {required}.",
                    False)

    # Moderate: mean on skewed salary/revenue
    if (action.action_type == "fill_nulls"
            and col in ["salary", "revenue"]
            and action.strategy == "mean"
            and task_id in ["full_clean", "deceptive_clean"]):
        return (MODERATE,
                f"PENALTY {MODERATE}: '{col}' is right-skewed. "
                f"Mean is misleading here — use median.",
                False)

    return 0.0, "", False


# ---------------------------------------------------------------------------
# Hints generator
# ---------------------------------------------------------------------------

def generate_hints(task_id: str, df: pd.DataFrame) -> List[str]:
    cfg = TASK_CONFIGS[task_id]
    hints = []
    if cfg["protected_nulls"]:
        hints.append(
            f"WARNING: Columns {cfg['protected_nulls']} have intentional nulls. "
            f"Filling them is penalised."
        )
    if cfg["valid_outlier_cols"]:
        hints.append(
            f"WARNING: Columns {cfg['valid_outlier_cols']} contain real extreme values. "
            f"Removing them triggers catastrophic termination."
        )
    if task_id in ["full_clean", "deceptive_clean"]:
        hints.append(
            "HINT: Salary/revenue distributions are right-skewed. "
            "Use median, not mean."
        )
    if cfg["critical_cols"]:
        hints.append(
            f"DANGER: Dropping {cfg['critical_cols']} with strategy='drop' "
            f"ends the episode with a -1.0 penalty."
        )
    if task_id == "deceptive_clean":
        hints.append(
            "TRAP: High revenue rows look like outliers — they are real high earners. "
            "Do NOT remove them."
        )
        hints.append(
            "TRAP: Some near-duplicate rows have different customer_ids. "
            "They are repeat purchases, not true duplicates."
        )
    return hints


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def _build_observation(task_id, step, df, original_df, done, message,
                       adversary_active=False, hints=None,
                       catastrophic=False) -> DataCleaningObservation:
    columns = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        columns.append(ColumnInfo(
            name=col,
            dtype=str(df[col].dtype),
            null_count=null_count,
            null_pct=round(null_count / max(len(df), 1), 4),
            unique_count=int(df[col].nunique()),
            sample_values=df[col].dropna().head(5).tolist(),
        ))

    outlier_counts: Dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_counts[col] = int(
            ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        )

    cfg = TASK_CONFIGS[task_id]
    score = grade(task_id, df, original_df)

    return DataCleaningObservation(
        task_id=task_id,
        step=step,
        columns=columns,
        total_rows=len(df),
        duplicate_rows=int(df.duplicated().sum()),
        outlier_counts=outlier_counts,
        score_so_far=_clamp(score),
        done=done,
        message=message,
        metadata={
            "adversary_active": adversary_active,
            "protected_nulls":  cfg["protected_nulls"],
            "valid_outlier_cols": cfg["valid_outlier_cols"],
            "critical_cols":    cfg.get("critical_cols", []),
            "action_cost":      ACTION_COST,
            "hints":            hints or [],
            "catastrophic_termination": catastrophic,
        },
    )


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class DataCleaningEnvironment:
    """
    OpenEnv-compatible Data Cleaning environment.
    Each instance is fully isolated — safe for concurrent sessions.

    Tasks: fix_nulls | fix_types | full_clean | deceptive_clean
    Modes: standard | adversarial
    """

    def __init__(self, task_id: str = "fix_nulls", adversarial: bool = False,
                 adversary_difficulty: float = 0.3, seed: int = 42):
        assert task_id in TASK_DATASETS, \
            f"Unknown task '{task_id}'. Choose from {list(TASK_DATASETS.keys())}"
        self.task_id = task_id
        self.adversarial = adversarial
        self.adversary = StrategicAdversary(adversary_difficulty) if adversarial else None
        self.seed = seed
        self._episode_id: str = ""
        self._step: int = 0
        self._df: Optional[pd.DataFrame] = None
        self._original_df: Optional[pd.DataFrame] = None
        self._actions_taken: List[str] = []
        self._max_steps: int = MAX_STEPS[task_id]
        self._prev_score: float = 0.0
        self._total_penalty: float = 0.0
        self._total_cost: float = 0.0
        self._catastrophic: bool = False

    def reset(self) -> DataCleaningObservation:
        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._actions_taken = []
        self._df = TASK_DATASETS[self.task_id](seed=self.seed)
        self._original_df = self._df.copy()
        self._total_penalty = 0.0
        self._total_cost = 0.0
        self._catastrophic = False
        # Initialize _prev_score from actual dirty-dataset grade so step-1
        # score_delta is not inflated by the gap between 0.0 and the real initial score.
        self._prev_score = _clamp(grade(self.task_id, self._df, self._original_df))

        mode = "adversarial" if self.adversarial else "standard"
        hints = generate_hints(self.task_id, self._df)
        return _build_observation(
            self.task_id, self._step, self._df, self._original_df,
            done=False,
            message=(f"Episode started ({mode} mode). "
                     f"Action cost: {ACTION_COST} per step. Read hints carefully."),
            adversary_active=self.adversarial,
            hints=hints,
        )

    def step(self, action: DataCleaningAction) -> Tuple[DataCleaningObservation, float, bool, Dict]:
        assert self._df is not None, "Call reset() first."
        self._step += 1
        self._actions_taken.append(action.action_type)

        # 1. Penalty check (before applying action)
        penalty, penalty_msg, catastrophic = compute_penalty(action, self._df, self.task_id)
        self._total_penalty += penalty
        self._catastrophic = catastrophic

        # 2. Apply action (even on catastrophic — penalty already applied)
        agent_msg = ""
        try:
            agent_msg = self._apply_action(action)
        except Exception as e:
            agent_msg = f"Action failed: {e}"

        # 3. Adversary turn (skipped on catastrophic or finish)
        adversary_msg = ""
        if self.adversarial and self.adversary and not catastrophic \
                and action.action_type != "finish":
            cfg = TASK_CONFIGS[self.task_id]
            self._df, adversary_msg = self.adversary.corrupt(
                self._df, self._original_df, cfg["protected_nulls"])

        # 4. Score and reward
        score_now = _clamp(grade(self.task_id, self._df, self._original_df))
        score_delta = score_now - self._prev_score
        self._prev_score = score_now
        reward = _clamp_reward(score_delta - ACTION_COST + penalty)
        self._total_cost += ACTION_COST

        # 5. Done condition
        done = (
            catastrophic
            or action.action_type == "finish"
            or self._step >= self._max_steps
            or score_now >= 0.999
        )

        full_msg = " | ".join(p for p in [agent_msg, penalty_msg, adversary_msg] if p)
        hints = generate_hints(self.task_id, self._df)

        obs = _build_observation(
            self.task_id, self._step, self._df, self._original_df,
            done=done, message=full_msg,
            adversary_active=self.adversarial,
            hints=hints,
            catastrophic=catastrophic,
        )
        return obs, reward, done, {
            "score":         _clamp(score_now),
            "score_delta":   round(score_delta, 4),
            "action_cost":   ACTION_COST,
            "penalty":       penalty,
            "catastrophic":  catastrophic,
            "total_penalty": round(self._total_penalty, 4),
        }

    def state(self) -> DataCleaningState:
        return DataCleaningState(
            episode_id=self._episode_id,
            task_id=self.task_id,
            step_count=self._step,
            max_steps=self._max_steps,
            actions_taken=self._actions_taken,
            current_score=_clamp(max(0.01, self._prev_score)),
            metadata={
                "adversarial":             self.adversarial,
                "total_penalty":           round(self._total_penalty, 4),
                "total_cost":              round(self._total_cost, 4),
                "catastrophic_termination": self._catastrophic,
            },
        )

    def _apply_action(self, action: DataCleaningAction) -> str:
        df = self._df
        atype = action.action_type

        if atype == "fill_nulls":
            col = action.column
            strategy = action.strategy or "mean"
            if col not in df.columns:
                return f"Column '{col}' not found."
            before = int(df[col].isnull().sum())
            if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == "drop":
                self._df = df.dropna(subset=[col]).reset_index(drop=True)
                df = self._df
            elif strategy == "ffill":
                df[col] = df[col].ffill()
            else:
                return f"Unknown strategy '{strategy}'."
            after = int(df[col].isnull().sum()) if col in df.columns else 0
            return f"fill_nulls '{col}' ({strategy}): {before}→{after} nulls."

        elif atype == "drop_duplicates":
            before = len(df)
            self._df = df.drop_duplicates().reset_index(drop=True)
            return f"drop_duplicates: removed {before - len(self._df)} rows."

        elif atype == "cast_column":
            col = action.column
            dtype = action.dtype or "str"
            if col not in df.columns:
                return f"Column '{col}' not found."
            if dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "str":
                df[col] = df[col].astype(str)
            elif dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "bool":
                df[col] = df[col].map({
                    "True": True, "False": False,
                    "true": True, "false": False,
                    "1": True, "0": False
                })
            return f"cast_column '{col}' → {dtype}."

        elif atype == "remove_outliers":
            col = action.column
            method = action.method or "iqr"
            if col not in df.columns:
                return f"Column '{col}' not found."
            if not pd.api.types.is_numeric_dtype(df[col]):
                return f"Column '{col}' is not numeric."
            before = len(df)
            if method == "iqr":
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                mask = (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)
                self._df = df[mask | df[col].isnull()].reset_index(drop=True)
            elif method == "zscore":
                from scipy import stats
                z = np.abs(stats.zscore(df[col].dropna()))
                outlier_idx = df[col].dropna().index[z > 3]
                self._df = df.drop(index=outlier_idx).reset_index(drop=True)
            return f"remove_outliers '{col}' ({method}): removed {before - len(self._df)} rows."

        elif atype == "normalize_text":
            col = action.column
            if col not in df.columns:
                return f"Column '{col}' not found."
            before_unique = df[col].nunique()
            df[col] = df[col].astype(str).str.strip().str.lower()
            return f"normalize_text '{col}': {before_unique}→{df[col].nunique()} unique values."

        elif atype == "finish":
            return "Agent signalled finish."

        return f"Unknown action type: '{atype}'."
# v4.1
