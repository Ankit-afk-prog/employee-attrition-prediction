"""Score employees with the saved attrition model.

Usage (from the repository root):
    python src/predict.py                                   # scores data/IBM_HR_Analytics_Employee_Attrition.csv
    python src/predict.py path/to/employees.csv out.csv     # custom input / output

The input file must contain the same feature columns as the training data
(comma-, tab- or semicolon-separated). Output columns: RiskScore, FlaggedAtRisk.
Scores are relative risk estimates from a model trained on fictional data;
do not use them to make decisions about individual people.
"""
import csv
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # makes attrition_features importable for the pickled model

DEFAULT_INPUT = ROOT / "data" / "IBM_HR_Analytics_Employee_Attrition.csv"
MODEL_PATH = ROOT / "models" / "final_model.joblib"


def load_table(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        delimiter = csv.Sniffer().sniff(handle.read(4096), delimiters=",\t;").delimiter
    return pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")


def main(argv):
    input_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_INPUT
    output_path = Path(argv[2]) if len(argv) > 2 else None

    bundle = joblib.load(MODEL_PATH)
    data = load_table(input_path)
    missing = [c for c in bundle["schema"]["columns"] if c not in data.columns]
    if missing:
        raise SystemExit(f"Input is missing required columns: {missing}")

    risk = bundle["pipeline"].predict_proba(data[bundle["schema"]["columns"]])[:, 1]
    result = data.copy()
    result["RiskScore"] = risk.round(4)
    result["FlaggedAtRisk"] = risk >= bundle["threshold"]

    print(f"Model: {bundle['model_name']}  |  threshold: {bundle['threshold']}")
    print(f"Scored {len(result):,} employees; flagged {int(result['FlaggedAtRisk'].sum()):,} as at risk.")
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main(sys.argv)
