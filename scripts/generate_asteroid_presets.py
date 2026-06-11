from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "nasa.csv"
DEFAULT_FEATURES_PATH = BASE_DIR / "feature_names.pkl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "asteroid_presets.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Streamlit asteroid presets from the NASA dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the NASA CSV dataset. Default: data/nasa.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to feature_names.pkl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path. Default: asteroid_presets.json",
    )
    return parser.parse_args()


def normalize_hazardous_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "tehlikeli"}


def get_optional_value(row: pd.Series, column_names: list[str]) -> Any:
    for column_name in column_names:
        if column_name in row and not pd.isna(row[column_name]):
            value = row[column_name]
            if hasattr(value, "item"):
                return value.item()
            return value
    return None


def row_to_preset(row: pd.Series, feature_names: list[str], is_hazardous: bool) -> dict[str, Any]:
    features = {}
    for feature_name in feature_names:
        value = row[feature_name]
        if pd.isna(value):
            value = 0.0
        features[feature_name] = float(value)

    return {
        "features": features,
        "is_hazardous": is_hazardous,
        "neo_reference_id": get_optional_value(row, ["Neo Reference ID", "id", "ID"]),
        "name": get_optional_value(row, ["Name", "name"]),
    }


def build_presets(dataset_path: Path, features_path: Path) -> dict[str, dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Feature list file not found: {features_path}")

    df = pd.read_csv(dataset_path)
    feature_names = list(joblib.load(features_path))

    missing_columns = [name for name in feature_names if name not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required feature columns: {missing_text}")

    if "Hazardous" not in df.columns:
        raise ValueError("Dataset must contain a 'Hazardous' column.")

    df = df.copy()
    df["_is_hazardous"] = df["Hazardous"].map(normalize_hazardous_value)

    presets: dict[str, dict[str, Any]] = {}

    hazardous_df = df[df["_is_hazardous"]].reset_index(drop=True)
    safe_df = df[~df["_is_hazardous"]].reset_index(drop=True)

    for index, row in hazardous_df.iterrows():
        presets[f"tehlikeli-{index + 1}"] = row_to_preset(row, feature_names, True)

    for index, row in safe_df.iterrows():
        presets[f"tehlikesiz-{index + 1}"] = row_to_preset(row, feature_names, False)

    return presets


def main() -> None:
    args = parse_args()
    presets = build_presets(args.dataset, args.features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(presets, output_file, ensure_ascii=False, indent=2)

    hazardous_count = sum(1 for value in presets.values() if value["is_hazardous"])
    safe_count = len(presets) - hazardous_count
    print(f"Created {args.output}")
    print(f"Hazardous presets: {hazardous_count}")
    print(f"Non-hazardous presets: {safe_count}")
    print(f"Total presets: {len(presets)}")


if __name__ == "__main__":
    main()
