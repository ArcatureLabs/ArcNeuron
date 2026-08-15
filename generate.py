"""Generate text from a trained ArcNeuron checkpoint.

This file contains decoding policy only. It loads the neural checkpoint, asks the
model for next-token logits, applies generic sampling controls, and feeds sampled
tokens back autoregressively. It contains no fact database, symbolic reasoning
rule, answer template, prompt classifier, or task-specific branch.
"""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from arcneuron import ArcNeuron, ArcNeuronConfig
from tokenizer import ArcTokenizer


def parse_args() -> argparse.Namespace:
    """Read generation settings."""

    parser = argparse.ArgumentParser(
        description="Generate text with ArcNeuron"
    )

    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument(
        "--checkpoint",
        default="arcneuron-tuned.pt",
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.92,
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.08,
    )
    parser.add_argument(
        "--repeat-window",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def choose_device() -> torch.device:
    """Prefer CUDA automatically."""

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_model(
    path: str | Path,
    device: torch.device,
) -> tuple[ArcNeuron, ArcTokenizer]:
    """Reconstruct the exact neural model and tokenizer from one checkpoint."""

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = ArcNeuronConfig(
        **checkpoint["model_config"]
    )

    model = ArcNeuron(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer = ArcTokenizer.from_bytes(
        checkpoint["tokenizer_model"]
    )

    return model, tokenizer


def apply_repetition_penalty(
    logits: torch.Tensor,
    recent_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Discourage immediate repetition without adding any semantic rule."""

    if penalty < 1.0:
        raise ValueError(
            "repetition_penalty must be at least 1.0"
        )

    if penalty == 1.0 or not recent_ids:
        return logits

    logits = logits.clone()
    unique_ids = torch.tensor(
        list(set(recent_ids)),
        dtype=torch.long,
        device=logits.device,
    )

    selected = logits[unique_ids]
    selected = torch.where(
        selected < 0,
        selected * penalty,
        selected / penalty,
    )
    logits[unique_ids] = selected

    return logits


def apply_top_k(
    logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Keep only the strongest k candidate logits when requested."""

    if top_k <= 0 or top_k >= logits.numel():
        return logits

    threshold = torch.topk(
        logits,
        top_k,
    ).values[-1]

    return logits.masked_fill(
        logits < threshold,
        float("-inf"),
    )


def apply_top_p(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """Nucleus-filter a logit vector while retaining at least one token."""

    if not 0.0 < top_p <= 1.0:
        raise ValueError(
            "top_p must be in the interval (0, 1]"
        )

    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )
    sorted_probabilities = F.softmax(
        sorted_logits,
        dim=-1,
    )
    cumulative = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    sorted_logits = sorted_logits.masked_fill(
        remove,
        float("-inf"),
    )

    filtered = torch.full_like(
        logits,
        float("-inf"),
    )
    filtered.scatter_(
        0,
        sorted_indices,
        sorted_logits,
    )

    return filtered


def sample_next(
    logits: torch.Tensor,
    recent_ids: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> int:
    """Choose one next token from ArcNeuron's learned distribution."""

    if temperature < 0.0:
        raise ValueError(
            "temperature cannot be negative"
        )

    logits = apply_repetition_penalty(
        logits,
        recent_ids,
        repetition_penalty,
    )

    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)

    probabilities = F.softmax(
        logits,
        dim=-1,
    )

    return int(
        torch.multinomial(
            probabilities,
            num_samples=1,
        ).item()
    )


@torch.inference_mode()
def generate(
    model: ArcNeuron,
    tokenizer: ArcTokenizer,
    prompt: str,
    depth: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    top_p: float = 0.92,
    repetition_penalty: float = 1.08,
    repeat_window: int = 128,
    include_prompt: bool = False,
) -> str:
    """Generate one autoregressive continuation."""

    if depth < 1:
        raise ValueError(
            "depth must be at least 1"
        )

    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens cannot be negative"
        )

    if repeat_window < 0:
        raise ValueError(
            "repeat_window cannot be negative"
        )

    prompt_ids = tokenizer.encode(
        prompt,
        add_bos=True,
        add_eos=False,
    )

    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        visible = generated[
            -model.config.max_seq_len :
        ]

        x = torch.tensor(
            [visible],
            dtype=torch.long,
            device=device,
        )

        logits = model(
            x,
            depth=depth,
        )

        recent_ids = (
            generated[-repeat_window:]
            if repeat_window > 0
            else []
        )

        next_id = sample_next(
            logits[0, -1],
            recent_ids,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )

        generated.append(next_id)

        if next_id == tokenizer.eos_id:
            break

    full_text = tokenizer.decode(generated)

    if include_prompt:
        return full_text

    decoded_prompt = tokenizer.decode(prompt_ids)

    if full_text.startswith(decoded_prompt):
        return full_text[len(decoded_prompt) :].lstrip()

    return tokenizer.decode(
        generated[len(prompt_ids) :]
    ).lstrip()


def main() -> None:
    """Run one-shot or interactive generation."""

    args = parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device = choose_device()

    checkpoint = args.checkpoint

    if (
        not Path(checkpoint).is_file()
        and checkpoint == "arcneuron-tuned.pt"
        and Path("arcneuron.pt").is_file()
    ):
        checkpoint = "arcneuron.pt"

    model, tokenizer = load_model(
        checkpoint,
        device,
    )

    def run(prompt: str) -> str:
        return generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            depth=args.depth,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            repeat_window=args.repeat_window,
            include_prompt=args.include_prompt,
            device=device,
        )

    if args.prompt is not None:
        print(run(args.prompt))
        return

    print(
        f"ArcNeuron loaded from {checkpoint} on {device}. "
        "Empty input exits."
    )

    while True:
        prompt = input("\n> ")

        if not prompt:
            break

        print(run(prompt))


if __name__ == "__main__":
    main()
