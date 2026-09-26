"""Evaluate saved KIM answers or reproduce the released aggregate scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import judge_core as judge

DATASETS = {"gsm8k": 200, "math500": 200, "aime24": 30,
            "hotpotqa": 200, "2wiki": 200, "musique": 200, "bamboogle": 125}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_rows(rows, dataset):
    if not isinstance(rows, list) or len(rows) != DATASETS[dataset]:
        raise ValueError(f"Expected {DATASETS[dataset]} rows for {dataset}")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every row must be an object")
    if any(type(row.get("index")) is not int for row in rows):
        raise ValueError("Question indices must be integers")
    if {row["index"] for row in rows} != set(range(DATASETS[dataset])):
        raise ValueError("Question indices are incomplete or duplicated")
    if any(row.get("dataset") != dataset for row in rows):
        raise ValueError("Dataset labels do not match")
    return sorted(rows, key=lambda row: row["index"])


def aggregate(cells):
    by_name = {row["dataset"]: row for row in cells}
    if len(cells) != len(DATASETS) or set(by_name) != set(DATASETS):
        raise ValueError("Exactly the seven datasets are required")
    result = []
    for dataset, count in DATASETS.items():
        row = by_name[dataset]
        correct = row["correct"]
        if row["count"] != count or type(correct) is not int or not 0 <= correct <= count:
            raise ValueError(f"Invalid counts for {dataset}")
        accuracy = correct / count * 100
        if "accuracy" in row and abs(row["accuracy"] - accuracy) > 1e-9:
            raise ValueError(f"Stored accuracy does not match counts: {dataset}")
        result.append(dict(dataset=dataset, count=count, correct=correct, accuracy=accuracy))
    scores = [row["accuracy"] for row in result]
    return dict(datasets=result, math_macro=mean(scores[:3]), qa_macro=mean(scores[3:]),
                macro=mean(scores), question_count=sum(DATASETS.values()))


def summarize(args):
    if args.judgments:
        cells = []
        for dataset in DATASETS:
            payload = read(args.judgments / f"{dataset}.json")
            if payload["judge_protocol"] != judge.JUDGE_PROTOCOL:
                raise ValueError("Do not combine different judge protocols")
            rows = validate_rows(payload["rows"], dataset)
            if any(row["verdict"] not in ("correct", "incorrect") for row in rows):
                raise ValueError(f"Resolve invalid judgments before reporting {dataset}")
            cells.append(dict(dataset=dataset, count=len(rows),
                              correct=sum(row["verdict"] == "correct" for row in rows)))
        result = aggregate(cells)
    else:
        payload = read(args.results)
        if payload["operator_count"] != 16 or payload["judge_protocol"] != judge.JUDGE_PROTOCOL:
            raise ValueError("Expected the KIM16 reference-comparison release")
        result = aggregate(payload["datasets"])
        for name in ("math_macro", "qa_macro", "macro", "question_count"):
            if abs(result[name] - payload[name]) > 1e-9:
                raise ValueError(f"Stored aggregate does not match counts: {name}")
    print("Dataset       Correct     Accuracy (%)")
    for row in result["datasets"]:
        print(f"{row['dataset']:<13} {row['correct']:>3}/{row['count']:<3}     {row['accuracy']:.2f}")
    for name in ("math_macro", "qa_macro", "macro"):
        print(f"{name:<13} {result[name]:.2f}")
    if args.output:
        write(args.output, {"judge_protocol": judge.JUDGE_PROTOCOL, **result})


def run_judge(args):
    rows = validate_rows(read(args.input), args.dataset)
    for row in rows:
        if not all(key in row for key in ("question", "answer", "Output")):
            raise ValueError("Each answer needs question, answer and Output fields")
        if row.get("status") == "error":
            raise ValueError("Resolve generation errors before judging")
    if args.output.exists():
        raise FileExistsError("Choose a new output path; existing verdicts are never overwritten")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BF16 Qwen3 judging")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = judge.load_judge_model(args.model, torch=torch, model_class=AutoModelForCausalLM)
    details = []
    for row in rows:
        prediction = row["Output"]
        missing = judge.is_missing_prediction(prediction)
        detail = {"dataset": args.dataset, "index": row["index"], "question": row["question"],
                  "reference": row["answer"], "prediction": prediction,
                  "verdict": "incorrect", "judge_state": "missing_answer", "judge_response": None}
        if not missing:
            prompt = judge.render_prompt(tokenizer, question=row["question"],
                                         reference=row["answer"], prediction=prediction)
            result = judge.generate_verdicts(model=model, tokenizer=tokenizer, torch=torch,
                     prompts=[prompt], batch_size=1, max_new_tokens=128)[0]
            detail["initial_verdict"] = dict(result)
            if result["verdict"] == "invalid":
                result = judge.generate_verdicts(model=model, tokenizer=tokenizer, torch=torch,
                         prompts=[prompt], batch_size=1, max_new_tokens=1024)[0]
                detail["retried_max_new_tokens"] = 1024
            detail.update(result)
            detail["judge_state"] = "valid" if result["verdict"] != "invalid" else "invalid_format_after_retry"
        details.append(detail)
        print(f"{args.dataset}: {len(details)}/{len(rows)} {detail['verdict']}", flush=True)
    payload = {"judge_protocol": judge.JUDGE_PROTOCOL, "dataset": args.dataset,
               "judge_model": "Qwen/Qwen3-30B-A3B", "rows": details}
    write(args.output, payload)
    if any(row["verdict"] == "invalid" for row in details):
        raise RuntimeError("Invalid verdicts remain; output saved for inspection, not aggregation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    inputs = summary.add_mutually_exclusive_group()
    inputs.add_argument("--results", type=Path, default=Path(__file__).parent / "results/kim16.json")
    inputs.add_argument("--judgments", type=Path)
    summary.add_argument("--output", type=Path)
    scoring = commands.add_parser("judge")
    scoring.add_argument("--input", type=Path, required=True)
    scoring.add_argument("--dataset", choices=DATASETS, required=True)
    scoring.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"summarize": summarize, "judge": run_judge}[args.command](args)


if __name__ == "__main__":
    main()
