"""Audit numeric equivalence without replacing the Qwen judge verdicts."""
import argparse
from pathlib import Path
import sys

from evaluate import read, validate_rows, write
from judge_core import JUDGE_PROTOCOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Output from evaluate.py judge")
    parser.add_argument("--toolstar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = read(args.input)
    dataset = payload["dataset"]
    if dataset not in ("gsm8k", "math500", "aime24"):
        raise ValueError("Numeric checks only apply to the three mathematics datasets")
    if payload["judge_protocol"] != JUDGE_PROTOCOL:
        raise ValueError("Unexpected judge protocol")
    rows = validate_rows(payload["rows"], dataset)
    if any(row["verdict"] not in ("correct", "incorrect") for row in rows):
        raise ValueError("Resolve invalid judgments before numeric cross-checking")
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        raise FileExistsError("Numeric audit must use a new output file")
    upstream = args.toolstar_root / "evaluation/evaluate/scripts"
    if not (upstream / "utils/math_equivalence.py").is_file():
        raise FileNotFoundError("Expected Tool-Star's original math_equivalence.py")
    sys.path.insert(0, str(upstream))
    from utils.math_equivalence import is_equiv
    disagreements = []
    numeric_correct = 0
    for row in rows:
        numeric_equal = bool(is_equiv(str(row["prediction"] or ""), str(row["reference"])))
        numeric_correct += numeric_equal
        if numeric_equal != (row["verdict"] == "correct"):
            disagreements.append({"index": row["index"], "prediction": row["prediction"],
                                  "reference": row["reference"], "judge_verdict": row["verdict"],
                                  "numeric_equal": numeric_equal})
    report = dict(dataset=dataset, count=len(rows), numeric_correct=numeric_correct,
                  disagreement_count=len(disagreements), disagreements=disagreements)
    write(args.output, report)
    print(f"{dataset}: {len(disagreements)} disagreements; Qwen verdicts unchanged")


if __name__ == "__main__":
    main()
