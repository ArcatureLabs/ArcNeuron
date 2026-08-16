"""Overnight deterministic evaluation harness for ArcNeuron (TEMPORARY).

This file is NOT part of the permanent ArcNeuron repository.  It loads a
checkpoint from the Modal volume, runs deterministic greedy decoding, and writes
results.json to the volume so run_modal.py can bring the scores back to the
local machine.  It contains no answer dictionary that helps the model; Python
only holds ground-truth labels to score model *output* after generation, never
to produce it.

Benchmarks (all deterministic, no LLM judge):
  - held-out language-model loss
  - simple + multi-step arithmetic (regex-verified answers)
  - code generation with exec-based unit tests
  - five-observation novel-concept induction (auto-scored)
  - counterfactual / insufficient-information / contradiction (auto-scored)
  - Vietnamese coherence sample (the exact demo prompt)
  - recurrent-depth curve (depth 1/2/4/8, aggregate across reasoning tasks)
"""

import argparse  # The checkpoint path is supplied by run_modal.py::ev.
import json  # Structured scores are written to the volume for the local machine to read.
import math  # Loss averaging uses basic math.
import pathlib  # Volume and checkpoint paths stay explicit.
import re  # Deterministic answer extraction uses simple regular expressions, no LLM judge.

import torch  # The model and its tokenizer live in PyTorch tensors and a SentencePiece model.
from torch.nn import functional as F  # Cross entropy is the same objective used during training.

import sys
sys.path.insert(0, "/root/arcneuron")  # The baked repo source is the only ArcNeuron code path.
from arcneuron import ArcNeuron, ArcNeuronConfig
from tokenizer import ArcTokenizer


VOL_PATH = pathlib.Path("/vol")
EVAL_DIR = VOL_PATH / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "arcneuron")  # The saved arch tag selects which architecture class owns these weights.
    if arch == "baseline":  # A baseline checkpoint must reconstruct the non-recurrent stack, not ArcNeuron.
        from baseline_transformer import BaselineTransformer, BaselineConfig
        config = BaselineConfig(**checkpoint["model_config"])
        model = BaselineTransformer(config).to(device)
    else:  # Normal ArcNeuron reconstructs the recurrent-depth architecture.
        config = ArcNeuronConfig(**checkpoint["model_config"])
        model = ArcNeuron(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokenizer = ArcTokenizer.from_bytes(checkpoint["tokenizer_model"])
    training_plan = checkpoint.get("training_plan", {})
    return model, tokenizer, config, training_plan


@torch.inference_mode()
def greedy_generate(model, tokenizer, prompt: str, depth: int, max_new_tokens: int = 128) -> str:
    # Deterministic greedy decoding is the primary comparison mode requested by the brief.
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated = list(prompt_ids)
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        visible = generated[-model.config.max_seq_len :]
        x = torch.tensor([visible], dtype=torch.long, device=device)
        logits = model(x, depth=depth)
        next_id = int(torch.argmax(logits[0, -1]).item())
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    # Return only the continuation, not the prompt, to keep scoring simple.
    full = tokenizer.decode(generated)
    decoded_prompt = tokenizer.decode(prompt_ids)
    if full.startswith(decoded_prompt):
        return full[len(decoded_prompt) :].lstrip()
    return tokenizer.decode(generated[len(prompt_ids) :]).lstrip()


@torch.inference_mode()
def held_out_loss(model, tokenizer, val_path: str, device, max_batches: int = 16, context: int = 128) -> float:
    # Language-model loss on a held-out file measures raw text quality.
    if not pathlib.Path(val_path).is_file():
        return float("nan")
    text = pathlib.Path(val_path).read_text(encoding="utf-8")
    if not text.strip():
        return float("nan")
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    if len(ids) < context + 2:
        return float("nan")
    total = torch.tensor(ids, dtype=torch.long, device=device)
    losses = []
    for i in range(max_batches):
        start = (i * context) % max(1, len(ids) - context - 1)
        if start + context + 1 >= len(ids):
            break
        x = total[start : start + context].unsqueeze(0)
        y = total[start + 1 : start + context + 1].unsqueeze(0)
        logits = model(x, depth=4)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss))
    return sum(losses) / len(losses) if losses else float("nan")


# ---------------------------------------------------------------------------
# Deterministic task suites.  Each returns (score, details) where score is in
# [0, 1].  Ground truth lives here ONLY for scoring model output, never for
# generating it.
# ---------------------------------------------------------------------------

ARITH_SIMPLE = [
    # answer extraction: the last integer (possibly negative) appearing in the text.
    ("Ba cong hai bang may?", "6"),
    ("Ba nhan bon bang bao nhieu?", "12"),
    ("Muoi hai chia bon bang may?", "3"),
    ("Nam cong nam bang bao nhieu?", "10"),
    ("Bay tru ba bang may?", "4"),
    ("What is two plus five?", "7"),
    ("What is three times four?", "12"),
    ("What is ten minus six?", "4"),
]

ARITH_MULTI = [
    # multi-step: model should chain.  We only check the final integer answer.
    ("Ba cong hai roi nhan hai. Ket qua cuoi la bao nhieu?", "10"),  # (3+2)*2=10
    ("Hai nhan ba roi cong mot. Ket qua cuoi la bao nhieu?", "7"),  # 2*3+1=7
    ("Muoi hai chia hai roi tru ba. Ket qua cuoi la bao nhieu?", "3"),  # 12/2-3=3
    ("What is (two plus three) times two? Give the final number.", "10"),
    ("What is four times three minus five? Give the final number.", "7"),
]


def extract_number(text: str) -> str | None:
    # Find the last integer (with optional sign) in the generated text.
    matches = re.findall(r"-?\b\d+\b", text)
    return matches[-1] if matches else None


def score_arithmetic(model, tokenizer, items, depth: int) -> tuple[float, list]:
    correct = 0
    details = []
    for prompt, gold in items:
        out = greedy_generate(model, tokenizer, prompt, depth=depth, max_new_tokens=48)
        pred = extract_number(out)
        ok = pred is not None and pred == gold
        correct += int(ok)
        details.append({"prompt": prompt, "gold": gold, "pred": pred, "raw": out[:120], "ok": ok})
    return correct / len(items), details


# Code generation: very small tasks with executable unit tests.  We extract the
# first ```python ...``` block or the whole text and run a verifier function.
CODE_TASKS = [
    {
        "prompt": "Write a Python function `add(a, b)` that returns the sum of a and b. Reply with code in a ```python block.",
        "verifier": "def _verify(code):\n    ns={}\n    exec(code, ns)\n    return ns['add'](2,3)==5 and ns['add'](0,0)==0 and ns['add'](-1,1)==0",
        "name": "add",
    },
    {
        "prompt": "Write a Python function `is_even(n)` that returns True when n is even and False otherwise. Reply with code in a ```python block.",
        "verifier": "def _verify(code):\n    ns={}\n    exec(code, ns)\n    return ns['is_even'](2)==True and ns['is_even'](3)==False and ns['is_even'](0)==True",
        "name": "is_even",
    },
]


def score_code(model, tokenizer, depth: int) -> tuple[float, list]:
    correct = 0
    details = []
    for task in CODE_TASKS:
        out = greedy_generate(model, tokenizer, task["prompt"], depth=depth, max_new_tokens=160)
        # Extract the first fenced python block; fall back to the whole text.
        block = out
        m = re.search(r"```python\s*(.*?)```", out, re.DOTALL)
        if m:
            block = m.group(1)
        ok = False
        err = ""
        try:
            ns: dict = {}
            exec(block, ns)
            ok = bool(eval(task["verifier"], {"ns": ns}))
        except Exception as e:  # noqa: BLE001 - verifier failures are expected and reported, not fatal.
            err = f"{type(e).__name__}: {e}"
        correct += int(ok)
        details.append({"name": task["name"], "ok": ok, "error": err, "raw": out[:200]})
    return correct / len(CODE_TASKS), details


# ---------------------------------------------------------------------------
# Five-observation novel-concept benchmark.
#
# IMPORTANT: these concept names are randomly generated and NEVER appear in
# pretraining.  Each concept has 3-5 observations; questions test recall,
# composition, counterfactual, and insufficient information.  Python holds the
# answer keys ONLY to score model output, never to produce it.
# ---------------------------------------------------------------------------

NOVEL_CONCEPTS = [
    {
        "name": "nolari",
        "observations": [
            "A nolari is a small creature found in cold caves.",
            "Nolari have a thick insulating coat.",
            "They eat mineral-rich moss.",
            "Their eyesight is poor.",
            "They navigate mainly through hearing.",
        ],
        "questions": [
            # composition / counterfactual: losing coat -> harder to stay warm (insulation)
            {"q": "If a nolari loses most of its coat, what difficulty might it face in a cold cave?", "type": "composition",
             "keywords_any": ["warm", "heat", "cold", "insulat", "nhiet", "lanh", "mat nhiet"]},
            # direct recall
            {"q": "How do nolari mainly navigate?", "type": "recall",
             "keywords_any": ["hearing", "hear", "am thanh", "thinh", "nghe"]},
            # insufficient information: we never stated color
            {"q": "What color is a nolari?", "type": "insufficient",
             "keywords_any": ["not enough", "unknown", "do not know", "no information", "khong du", "khong biet", "khong ro"]},
            # composition: hearing impaired -> navigation affected
            {"q": "If a nolari's hearing became impaired, which known ability would be affected?", "type": "composition",
             "keywords_any": ["navigate", "navigation", "move", "hearing", "dinh huong", "dang"]},
            # recall
            {"q": "What do nolari eat?", "type": "recall",
             "keywords_any": ["moss", "rau", "xoang", "thuc vat"]},
        ],
    },
    {
        "name": "tembril",
        "observations": [
            "A tembril is a hard synthetic material with low thermal conductivity.",
            "Tembril is used as an insulating layer in ovens.",
            "It is brittle if dropped from a height.",
            "Tembril is dark grey in color.",
        ],
        "questions": [
            {"q": "Why is tembril used as an insulating layer in ovens?", "type": "composition",
             "keywords_any": ["conduct", "insulat", "heat", "nhiet", "dẫn", "cách"]},
            {"q": "What happens if a tembril panel is dropped from a height?", "type": "recall",
             "keywords_any": ["brittle", "break", "shatter", "vo", "cứng", "brise"]},
            {"q": "Is tembril a good electrical conductor?", "type": "insufficient",
             "keywords_any": ["not enough", "unknown", "do not know", "no information", "khong du", "khong biet"]},
            {"q": "What color is tembril?", "type": "recall",
             "keywords_any": ["grey", "gray", "xam", "den"]},
            {"q": "If tembril were a good thermal conductor, would it still be a good oven insulator?", "type": "counterfactual",
             "keywords_any": ["no", "not", "would not", "wouldn", "khong", "se khong", "khong con"]},
        ],
    },
]


def score_novel_concepts(model, tokenizer, depth: int) -> tuple[float, dict]:
    # For each concept we build a context of observations, then ask each question.
    total = 0
    correct = 0
    per_type = {}
    details = []
    for concept in NOVEL_CONCEPTS:
        context = " ".join(concept["observations"])
        for q in concept["questions"]:
            prompt = f"{context}\n\n{q['q']}"
            out = greedy_generate(model, tokenizer, prompt, depth=depth, max_new_tokens=64)
            text = out.lower()
            ok = any(kw.lower() in text for kw in q["keywords_any"])
            total += 1
            correct += int(ok)
            per_type.setdefault(q["type"], [0, 0])
            per_type[q["type"]][0] += int(ok)
            per_type[q["type"]][1] += 1
            details.append({"concept": concept["name"], "type": q["type"], "q": q["q"], "ok": ok, "raw": out[:150]})
    overall = correct / total if total else 0.0
    by_type = {k: v[0] / v[1] for k, v in per_type.items() if v[1]}
    return overall, {"by_type": by_type, "counts": per_type, "details": details}


# The exact Vietnamese demo prompt required by the final report.
DEMO_PROMPT_VN = "Một con mèo bị mất một chân có còn là động vật có vú không? Giải thích."

# Additional Vietnamese coherence prompts to avoid cherry-picking one.
VN_COHERENCE_PROMPTS = [
    "Viết một câu giải thích vì sao lớp lông giúp động vật giữ nhiệt.",
    "Nếu một con vật sống trong hang lạnh và mất gần hết lông, điều gì có thể xảy ra?",
    "Số nguyên tố nhỏ nhất là số nào?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-file", default="/vol/data/val_mix.txt")
    parser.add_argument("--depths", default="1,2,4,8")
    parser.add_argument("--out", default="")  # When set, write results to this path so multiple checkpoints do not overwrite each other.
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    model, tokenizer, config, training_plan = load_model(args.checkpoint, device)
    depths = [int(d) for d in args.depths.split(",") if d.strip()]

    results = {
        "checkpoint": args.checkpoint,
        "model_config": {k: getattr(config, k, None) for k in ["vocab_size", "dim", "n_heads", "n_kv_heads", "ffn_dim", "prelude_layers", "core_layers", "coda_layers", "n_layers", "max_seq_len"]},
        "training_plan": training_plan,
        "held_out_loss": held_out_loss(model, tokenizer, args.val_file, device),
        "depth_curve": {},
        "vietnamese": {},
    }

    for depth in depths:
        arith1, arith1_det = score_arithmetic(model, tokenizer, ARITH_SIMPLE, depth)
        arith2, arith2_det = score_arithmetic(model, tokenizer, ARITH_MULTI, depth)
        code, code_det = score_code(model, tokenizer, depth)
        novel, novel_det = score_novel_concepts(model, tokenizer, depth)
        # A composite reasoning score aggregates the four reasoning axes.
        reasoning = (arith2 + code + novel) / 3.0  # multi-step arith, code, novel-concept
        results["depth_curve"][str(depth)] = {
            "arithmetic_simple": arith1,
            "arithmetic_multi": arith2,
            "code": code,
            "novel_concept": novel,
            "reasoning_composite": reasoning,
            "novel_concept_by_type": novel_det["by_type"],
        }

    # Vietnamese demo (deterministic greedy, depth 4 by default).
    demo_depth = 4
    results["vietnamese"]["demo_prompt"] = DEMO_PROMPT_VN
    results["vietnamese"]["demo_answer"] = greedy_generate(model, tokenizer, DEMO_PROMPT_VN, depth=demo_depth, max_new_tokens=120)
    results["vietnamese"]["demo_depth"] = demo_depth
    results["vietnamese"]["extra_samples"] = [
        {"prompt": p, "answer": greedy_generate(model, tokenizer, p, depth=demo_depth, max_new_tokens=100)}
        for p in VN_COHERENCE_PROMPTS
    ]

    out_path = EVAL_DIR / "results.json" if not args.out else pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval_overnight] wrote {out_path}", flush=True)
    # Also print a compact summary so the run_modal log shows it live.
    summary = {
        "held_out_loss": results["held_out_loss"],
        "depth_curve": {d: {k: round(v, 3) for k, v in dc.items() if isinstance(v, (int, float))} for d, dc in results["depth_curve"].items()},
        "demo_answer": results["vietnamese"]["demo_answer"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
