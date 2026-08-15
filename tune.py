"""Continue training ArcNeuron on a small hand-guided natural-text corpus.

"Tuning" is continued next-token training on the exact same neural weights.
There is no adapter, reward model, classifier, symbolic reasoner, or alternate
objective. A small amount of base-text replay can be mixed in to reduce
forgetting.

The script prints live loss, learning rate, gradient norm, throughput, replay
usage, VRAM use, and ETA. Output is flushed immediately for Google Colab.
"""

import argparse
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from arcneuron import ArcNeuron, ArcNeuronConfig
from tokenizer import ArcTokenizer


def parse_args() -> argparse.Namespace:
    """Read tuning settings."""

    parser = argparse.ArgumentParser(
        description="Continue ArcNeuron training on tune.txt"
    )

    parser.add_argument("--checkpoint", default="arcneuron.pt")
    parser.add_argument("--data", default="tune.txt")
    parser.add_argument("--replay-data", default="train.txt")
    parser.add_argument("--out", default="arcneuron-tuned.pt")

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=4)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--replay-ratio", type=float, default=0.20)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)

    return parser.parse_args()


def choose_device() -> torch.device:
    """Prefer CUDA automatically."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_text(path: str | Path) -> str:
    """Read a plain UTF-8 text corpus."""

    text = Path(path).read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"text corpus is empty: {path}")

    return text


def to_tokens(
    tokenizer: ArcTokenizer,
    text: str,
) -> Tensor:
    """Encode text with the tokenizer already stored in the base checkpoint."""

    ids = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    return torch.tensor(ids, dtype=torch.long)


def sample_batch(
    tokens: Tensor,
    batch_size: int,
    context: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample contiguous next-token windows."""

    if tokens.numel() < context + 2:
        raise ValueError(
            "text is too small for --context; "
            "lower the context or add more natural text"
        )

    highest_start = tokens.numel() - context - 1
    starts = torch.randint(0, highest_start, (batch_size,))

    x = torch.stack(
        [tokens[start : start + context] for start in starts]
    )
    y = torch.stack(
        [tokens[start + 1 : start + context + 1] for start in starts]
    )

    return (
        x.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
    )


def make_optimizer(
    model: ArcNeuron,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Create a fresh low-rate AdamW state for tuning."""

    decay: list[Tensor] = []
    no_decay: list[Tensor] = []

    for parameter in model.parameters():
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )


def learning_rate(
    step: int,
    total_steps: int,
    warmup: int,
    maximum: float,
    minimum: float,
) -> float:
    """Use a short warmup and cosine decay for gentle continued training."""

    if warmup > 0 and step < warmup:
        return maximum * (step + 1) / warmup

    if total_steps <= warmup:
        return maximum

    progress = min(
        1.0,
        max(0.0, (step - warmup) / (total_steps - warmup)),
    )

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + cosine * (maximum - minimum)


def autocast_context(
    device: torch.device,
    dtype: torch.dtype,
):
    """Use mixed precision only on CUDA."""

    if device.type != "cuda":
        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


def save_checkpoint(
    path: str | Path,
    model: ArcNeuron,
    optimizer: torch.optim.Optimizer,
    tokenizer: ArcTokenizer,
    tune_step: int,
    base_step: int,
) -> None:
    """Write a tuned checkpoint that can run without the base checkpoint."""

    checkpoint = {
        "format": 1,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer_model": tokenizer.to_bytes(),
        "step": base_step,
        "tune_step": tune_step,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_random_state"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)


def format_eta(seconds: float) -> str:
    """Format ETA compactly."""

    if not math.isfinite(seconds) or seconds < 0:
        return "?"

    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:d}h{minutes:02d}m"

    if minutes:
        return f"{minutes:d}m{seconds:02d}s"

    return f"{seconds:d}s"


def main() -> None:
    """Run tuning."""

    args = parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be positive")
    if args.max_depth < 1:
        raise ValueError("--max-depth must be at least 1")
    if not 0.0 <= args.replay_ratio <= 1.0:
        raise ValueError("--replay-ratio must be between 0 and 1")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.save_every < 0:
        raise ValueError("--save-every cannot be negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device = choose_device()

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    tokenizer = ArcTokenizer.from_bytes(
        checkpoint["tokenizer_model"]
    )

    config = ArcNeuronConfig(
        **checkpoint["model_config"]
    )

    if args.context > config.max_seq_len:
        raise ValueError(
            "--context cannot exceed the checkpoint max_seq_len"
        )

    model = ArcNeuron(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.train()

    optimizer = make_optimizer(
        model,
        args.lr,
        args.weight_decay,
    )

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
    )

    tune_tokens = to_tokens(
        tokenizer,
        read_text(args.data),
    )

    replay_tokens = (
        to_tokens(
            tokenizer,
            read_text(args.replay_data),
        )
        if args.replay_ratio > 0.0
        else None
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"device={device} "
        f"params={parameter_count:,} "
        f"context={args.context} "
        f"tune_tokens={tune_tokens.numel():,} "
        f"replay_ratio={args.replay_ratio:.2f}",
        flush=True,
    )

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)

        print(
            f"gpu={torch.cuda.get_device_name(device)} "
            f"vram={properties.total_memory / 1024**3:.1f}GiB "
            f"amp={str(amp_dtype).replace('torch.', '')}",
            flush=True,
        )

    if tune_tokens.numel() < 10_000:
        print(
            "note: tune.txt is deliberately small. "
            "Tuning should shape response behavior, not teach the whole world.",
            flush=True,
        )

    started = time.perf_counter()
    base_step = int(checkpoint.get("step", 0))

    for step in range(args.steps):
        current_lr = learning_rate(
            step,
            args.steps,
            args.warmup,
            args.lr,
            args.min_lr,
        )

        for group in optimizer.param_groups:
            group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        depth = random.randint(1, args.max_depth)
        accumulated_loss = 0.0
        replay_batches = 0

        for _ in range(args.grad_accum):
            use_replay = (
                replay_tokens is not None
                and random.random() < args.replay_ratio
            )

            source = (
                replay_tokens
                if use_replay
                else tune_tokens
            )

            replay_batches += int(use_replay)

            x, y = sample_batch(
                source,
                args.batch_size,
                args.context,
                device,
            )

            with autocast_context(device, amp_dtype):
                logits = model(x, depth=depth)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                scaled_loss = loss / args.grad_accum

            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach())

        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.clip,
        )

        scaler.step(optimizer)
        scaler.update()

        completed = step + 1

        should_log = (
            completed == 1
            or completed % args.log_every == 0
            or completed == args.steps
        )

        if should_log:
            elapsed = max(
                time.perf_counter() - started,
                1e-9,
            )

            processed = (
                completed
                * args.batch_size
                * args.context
                * args.grad_accum
            )

            tok_per_second = processed / elapsed
            seconds_per_step = elapsed / completed
            eta = format_eta(
                seconds_per_step * (args.steps - completed)
            )

            mean_loss = accumulated_loss / args.grad_accum
            progress = 100.0 * completed / args.steps

            memory_text = ""

            if device.type == "cuda":
                allocated = (
                    torch.cuda.memory_allocated(device)
                    / 1024**3
                )
                memory_text = f" vram={allocated:.2f}GiB"

            print(
                f"tune={completed}/{args.steps} "
                f"({progress:6.2f}%) "
                f"depth={depth} "
                f"loss={mean_loss:.4f} "
                f"lr={current_lr:.2e} "
                f"grad={float(grad_norm):.3f} "
                f"replay={replay_batches}/{args.grad_accum} "
                f"tok/s={tok_per_second:,.0f} "
                f"eta={eta}"
                f"{memory_text}",
                flush=True,
            )

        should_save = (
            (args.save_every > 0 and completed % args.save_every == 0)
            or completed == args.steps
        )

        if should_save:
            save_checkpoint(
                args.out,
                model,
                optimizer,
                tokenizer,
                completed,
                base_step,
            )

            print(
                f"saved {args.out} at tune step {completed}",
                flush=True,
            )


if __name__ == "__main__":
    main()
