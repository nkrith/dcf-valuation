from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_outputs_dir(path: str = "outputs") -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def save_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
