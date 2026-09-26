"""Original Qwen3 generated-comparison scoring primitives."""
from __future__ import annotations

import json
import re
from typing import Any


JUDGE_PROTOCOL = "qwen3_30b_a3b_reference_comparison_v3"


SYSTEM_PROMPT = """You are an impartial answer-correctness judge.
The supplied reference answer is authoritative. Do not solve the original problem again.
Only compare the candidate answer with the reference answer in the context of the question.

Accept harmless differences in notation, formatting, leading zeros, aliases, or wording when
the mathematical value or factual meaning is the same. Reject a different numerical value,
entity, relation, or conclusion. Extra explanation is acceptable unless it contradicts the
answer.

Give one short comparison sentence, then end with exactly one tag on the final line:
<verdict>correct</verdict>
<verdict>incorrect</verdict>"""


VERDICT_PATTERN = re.compile(
    r"<verdict>\s*(correct|incorrect)\s*</verdict>", re.IGNORECASE
)


def render_prompt(
    tokenizer: Any,
    *,
    question: Any,
    reference: Any,
    prediction: Any,
) -> str:
    user_prompt = (
        f"Question context:\n{question}\n\n"
        "Authoritative reference answer:\n"
        f"{json.dumps(reference, ensure_ascii=False)}\n\n"
        f"Candidate answer:\n{prediction}\n\n"
        "Compare only the reference and candidate; do not re-solve the question."
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_verdict(response: str) -> str:
    matches = VERDICT_PATTERN.findall(response)
    return matches[-1].lower() if matches else "invalid"


def is_missing_prediction(prediction: Any) -> bool:
    return prediction is None or (
        isinstance(prediction, str) and not prediction.strip()
    )


def generate_verdicts(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        for response in responses:
            results.append(
                {
                    "verdict": parse_verdict(response),
                    "judge_response": response,
                }
            )
    return results


def load_judge_model(model_path: str, *, torch: Any, model_class: Any) -> Any:
    kwargs = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    multi_gpu = torch.cuda.device_count() > 1
    if multi_gpu:
        kwargs["device_map"] = "balanced"
    model = model_class.from_pretrained(model_path, **kwargs)
    if multi_gpu:
        placement = model.hf_device_map
        if any(str(device) in {"cpu", "disk"} for device in placement.values()):
            raise RuntimeError("Judge weights must fit on the allocated GPUs without offloading")
        print(json.dumps({"judge_device_map": placement}), flush=True)
    else:
        model = model.to("cuda")
    model.eval()
    return model
