import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


REQUIRED_COLUMNS = ["datetime"]


@dataclass
class ValidationReport:
    ok: bool
    missing_columns: List[str]
    parseable_datetime: bool
    numeric_targets: List[Tuple[str, bool]]
    n_rows: int


def validate_dataset(csv_path: Path, targets: Iterable[str]) -> ValidationReport:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    parseable_datetime = True
    if "datetime" in df.columns:
        try:
            df["datetime"].astype("datetime64[ns]")
        except Exception:
            parseable_datetime = False

    numeric_targets = []
    for t in targets:
        numeric_targets.append((t, t in df.columns and pd.api.types.is_numeric_dtype(df[t])))

    ok = (not missing) and parseable_datetime and all(flag for _, flag in numeric_targets)
    return ValidationReport(
        ok=ok,
        missing_columns=missing,
        parseable_datetime=parseable_datetime,
        numeric_targets=numeric_targets,
        n_rows=len(df),
    )


def write_split_config(config_path: Path, train_frac: float, val_frac: float, test_frac: float, lake_name: str) -> None:
    payload: Dict[str, List] = {
        "train_lakes": [lake_name],
        "train_fractions": [train_frac],
        "val_lakes": [lake_name],
        "val_fractions": [val_frac],
        "test_lakes": [lake_name],
        "test_fractions": [test_frac],
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_executor_command(workspace_root: Path, params: Dict[str, str]) -> List[str]:
    executor_path = workspace_root / "Transfer_Learning" / "executor.py"
    cmd = [sys.executable, "-u", str(executor_path)]
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        cmd.extend([f"--{key}", str(value)])
    return cmd


def run_experiment(cmd: List[str], cwd: Path):
    env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        yield line.rstrip("\n")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Experiment failed with exit code {process.returncode}")
