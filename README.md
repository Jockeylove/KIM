# KIM: Knowledge-Augmenting Implicit Memory

Minimal results and saved-answer evaluation release for **KIM with 16 operators**
on Qwen2.5-3B-Instruct. This repository contains no other model variants,
ablations, training runs, scheduling files, or manuscript sources.

## Results

| Dataset | Correct / Questions | Accuracy (%) |
| --- | ---: | ---: |
| GSM8K | 154 / 200 | 77.00 |
| MATH500 | 124 / 200 | 62.00 |
| AIME24 | 5 / 30 | 16.67 |
| HotpotQA | 87 / 200 | 43.50 |
| 2WikiMultiHopQA | 77 / 200 | 38.50 |
| MuSiQue | 36 / 200 | 18.00 |
| Bamboogle | 44 / 125 | 35.20 |
| Math average | | 51.89 |
| QA average | | 33.80 |
| Overall average | | **41.55** |

There are 1,155 questions and 527 correct answers. The reported overall average
assigns equal weight to each of the seven datasets; it is not pooled accuracy.
Math averages the first three datasets, and QA averages the last four.
All reported scores use Qwen3-30B-A3B generated reference-comparison judgments.

## Recompute the Table (No GPU)

Python 3.10 or later is sufficient. No third-party packages are needed:

```sh
python evaluate.py summarize
python -m unittest discover -s tests -v
```

`results/kim16.json` contains the original aggregate counts and unrounded scores.
Individual model responses and verdicts are not included in this release.
Thus the command above verifies aggregate arithmetic, not individual judgments.

## Judge Saved Answers

Install the GPU dependencies into your own environment:

```sh
pip install -r requirements.txt
python evaluate.py judge --dataset gsm8k --input inputs/gsm8k.json --output outputs/judge/gsm8k.json
```

The input is a JSON array with exactly the dataset's number of questions.
Each record must include:

```json
{
  "dataset": "gsm8k",
  "index": 0,
  "question": "The original question text",
  "answer": "The reference answer",
  "Output": "The extracted candidate answer"
}
```

Indices are zero-based and contiguous. `answer` preserves the original reference
format, including alias lists for QA. `Output` is the already extracted answer;
use `null` for a missing answer. The evaluator does not re-extract answers from
reasoning traces. Do not substitute a different question subset when comparing
with the released counts.

The judge defaults to `Qwen/Qwen3-30B-A3B`; `--model` can point to a local copy
of that same model. It uses BF16 without quantization, greedy decoding,
non-thinking mode, batch size 1, and 128 generated tokens. Invalid verdict
formatting is retried with 1,024 tokens. Missing answers count as incorrect;
unresolved verdicts prevent score aggregation. GPU weights must fit entirely
in the selected GPU(s), without CPU or disk offloading. Set
`CUDA_VISIBLE_DEVICES` to select the available devices.

`judge_core.py` retains the original comparison prompt, verdict parser and
generation functions. The portable command-line wrapper preserves the
experiment's per-answer judging and retry settings. Package versions in
`requirements.txt` are compatible dependency bounds, not a frozen environment
lock; rerunning inference on a different stack need not be bitwise identical.

After evaluating all seven datasets into `outputs/judge/<dataset>.json`:

```sh
python evaluate.py summarize --judgments outputs/judge --output outputs/scores.json
```

Existing judgment output files are not overwritten. The evaluator never starts
training or answer generation. Checkpoints, generation code, retrieval indices,
and benchmark question files are outside this release's scope. Recorded KIM
geometry and generation budgets are provided in `protocol.json` for context.

## Mathematics Cross-Check

For mathematics, the numeric audit uses the original `is_equiv` implementation
from [Tool-Star](https://github.com/RUC-NLPIR/Tool-Star). Obtain that repository
and install the dependencies required by its evaluation utilities, then run:

```sh
python crosscheck_math.py --input outputs/judge/gsm8k.json --toolstar-root /path/to/Tool-Star --output outputs/gsm8k_numeric_audit.json
```

Repeat for MATH500 and AIME24. Disagreements are reported for inspection;
the audit does not replace Qwen judgments or change the reported protocol.
Tool-Star code and benchmark data are not redistributed here and retain their
respective licenses and terms.
