from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import json

import pandas as pd


@dataclass
class SubmissionLog:
    path: Path
    payload: dict
    activities: pd.DataFrame
    trades: pd.DataFrame


def _to_frame(value: object) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return pd.DataFrame()
        # Most submission logs store CSV blocks with ';' delimiter.
        try:
            return pd.read_csv(io.StringIO(text), sep=";")
        except Exception:
            # Fallback: sometimes logs contain JSON arrays serialized as text.
            try:
                return pd.DataFrame(json.loads(text))
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


def load_submission_log(path: Path) -> SubmissionLog:
    payload = json.loads(path.read_text())

    if "activitiesLog" not in payload:
        raise ValueError(f"{path} is missing activitiesLog.")

    activities = _to_frame(payload.get("activitiesLog"))
    trades = _to_frame(payload.get("tradeHistory"))

    return SubmissionLog(
        path=path,
        payload=payload,
        activities=activities,
        trades=trades,
    )

