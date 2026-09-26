"""CPU-only checks for counts, aggregation and unchanged judge behavior."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import aggregate, run_judge, summarize, validate_rows
from judge_core import is_missing_prediction, parse_verdict, render_prompt


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "results/kim16.json"
        self.result = json.loads(path.read_text())

    def test_released_scores(self):
        result = aggregate(self.result["datasets"])
        self.assertEqual(result["question_count"], 1155)
        self.assertEqual(sum(row["correct"] for row in result["datasets"]), 527)
        self.assertAlmostEqual(result["macro"], 41.55238095238095)
        self.assertAlmostEqual(result["math_macro"], 51.888888888888886)
        self.assertAlmostEqual(result["qa_macro"], 33.8)

    def test_macro_is_not_pooled_accuracy(self):
        result = aggregate(self.result["datasets"])
        self.assertNotAlmostEqual(result["macro"], 527 / 1155 * 100)

    def test_aggregate_rejects_missing_duplicate_or_bad_counts(self):
        cells = self.result["datasets"]
        for bad in (cells[:-1], cells[:-1] + [cells[0]]):
            with self.assertRaises(ValueError):
                aggregate(bad)
        bad = copy.deepcopy(cells)
        bad[0]["correct"] = 201
        with self.assertRaises(ValueError):
            aggregate(bad)
        bad = copy.deepcopy(cells)
        bad[0]["accuracy"] += 1
        with self.assertRaises(ValueError):
            aggregate(bad)

    def test_indices(self):
        rows = [dict(dataset="aime24", index=i) for i in range(30)]
        self.assertEqual(validate_rows(rows[::-1], "aime24"), rows)
        for bad in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                validate_rows(bad, "aime24")

    def test_missing_answers(self):
        for value in (None, "", " \n"):
            self.assertTrue(is_missing_prediction(value))
        self.assertFalse(is_missing_prediction("0"))

    def test_verdict_parsing(self):
        self.assertEqual(parse_verdict("Same value.\n<verdict>correct</verdict>"), "correct")
        self.assertEqual(parse_verdict("<verdict>incorrect</verdict>"), "incorrect")
        self.assertEqual(parse_verdict("looks right"), "invalid")

    def test_prompt_uses_non_thinking_reference_comparison(self):
        class Tokenizer:
            def apply_chat_template(inner, messages, **kwargs):
                self.assertFalse(kwargs["enable_thinking"])
                self.assertFalse(kwargs["tokenize"])
                self.assertTrue(kwargs["add_generation_prompt"])
                return messages
        messages = render_prompt(Tokenizer(), question="What is 1+1?", reference="2", prediction="02")
        self.assertEqual([r["role"] for r in messages], ["system", "user"])
        self.assertIn("Do not solve the original problem again", messages[0]["content"])
        self.assertIn("Candidate answer:\n02", messages[1]["content"])

    def test_judge_retry_and_missing_answer(self):
        rows = [dict(dataset="aime24", index=i, question=f"Question {i}",
                     answer="2", Output=None) for i in range(30)]
        rows[0]["Output"] = "02"
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
            AutoModelForCausalLM=object)
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "answers.json", Path(folder) / "judgments.json"
            source.write_text(json.dumps(rows))
            args = SimpleNamespace(input=source, output=output, dataset="aime24", model="Qwen/Qwen3-30B-A3B")
            replies = [[dict(verdict="invalid", judge_response="unfinished")],
                       [dict(verdict="correct", judge_response="<verdict>correct</verdict>")]]
            with patch.dict(sys.modules, torch=fake_torch, transformers=fake_transformers), \
                 patch("evaluate.judge.load_judge_model", return_value=object()), \
                 patch("evaluate.judge.render_prompt", return_value="prompt"), \
                 patch("evaluate.judge.generate_verdicts", side_effect=replies) as generate, \
                 patch("builtins.print"):
                run_judge(args)
            self.assertEqual([c.kwargs["max_new_tokens"] for c in generate.call_args_list], [128, 1024])
            result = json.loads(output.read_text())["rows"]
            self.assertEqual(result[0]["verdict"], "correct")
            self.assertEqual(result[0]["retried_max_new_tokens"], 1024)
            self.assertTrue(all(r["verdict"] == "incorrect" for r in result[1:]))
            with self.assertRaises(FileExistsError):
                run_judge(args)

    def test_summary_rejects_mixed_protocol_and_invalid_verdicts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "gsm8k.json"
            args = SimpleNamespace(judgments=root, output=None)
            path.write_text(json.dumps(dict(judge_protocol="another_protocol", rows=[])))
            with self.assertRaisesRegex(ValueError, "protocol"):
                summarize(args)
            rows = [dict(dataset="gsm8k", index=i, verdict="invalid") for i in range(200)]
            path.write_text(json.dumps(dict(judge_protocol="qwen3_30b_a3b_reference_comparison_v3", rows=rows)))
            with self.assertRaisesRegex(ValueError, "invalid"):
                summarize(args)


if __name__ == "__main__":
    unittest.main()
