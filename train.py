"""Train ArcNeuron from raw natural text with ordinary next-token prediction.

Python is responsible only for feeding tensors, choosing the recurrent depth used
for a complete batch, computing the language-model loss, updating neural weights,
and saving checkpoints. It contains no semantic dictionary, symbolic reasoner,
answer rule, or task-specific branch.

The training loop prints live progress with loss, learning rate, gradient norm,
throughput, validation loss, and ETA. Every print is flushed immediately so it is
visible in Google Colab while training is still running.
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
    """Read all experiment settings from the command line."""

    parser = argparse.ArgumentParser(description="Train ArcNeuron from natural text")
    parser.add_argument("--data", default="train.txt")
    parser.add_argument("--out", default="arcneuron.pt")
    parser.add_argument("--resume", default=None)

    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--context", type=int, default=1024)

    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=1408)
    parser.add_argument("--prelude-layers", type=int, default=1)
    parser.add_argument("--core-layers", type=int, default=2)
    parser.add_argument("--coda-layers", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def choose_device() -> torch.device:
    """Prefer CUDA automatically while keeping a CPU smoke-test path."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed every random stream used directly by this trainer."""

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_text(path: str | Path) -> str:
    """Read the corpus exactly as UTF-8 natural text."""

    text = Path(path).read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"training corpus is empty: {path}")

    return text


def split_tokens(token_ids: list[int], context: int) -> tuple[Tensor, Tensor]:
    """Reserve the final five percent of the token stream for validation."""

    tokens = torch.tensor(token_ids, dtype=torch.long)
    minimum = context + 2
    val_size = max(int(tokens.numel() * 0.05), minimum)
    split = tokens.numel() - val_size

    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    if train_tokens.numel() < minimum or val_tokens.numel() < minimum:
        raise ValueError(
            "train.txt is too small for this context length; "
            "lower --context or add more natural text"
        )

    return train_tokens, val_tokens


def sample_batch(
    tokens: Tensor,
    batch_size: int,
    context: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample ordinary contiguous next-token windows from one token stream."""

    highest_start = tokens.numel() - context - 1

    if highest_start <= 0:
        raise ValueError("token stream is too short for the requested context")

    starts = torch.randint(0, highest_start, (batch_size,))
    x = torch.stack([tokens[start : start + context] for start in starts])
    y = torch.stack([tokens[start + 1 : start + context + 1] for start in starts])

    return (
        x.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
    )


def make_optimizer(
    model: ArcNeuron,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Create AdamW without decaying one-dimensional normalization scales."""

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

    fused = torch.cuda.is_available()

    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=(0.9, 0.95),
        fused=fused,
    )


def learning_rate(
    step: int,
    total_steps: int,
    warmup: int,
    maximum: float,
    minimum: float,
) -> float:
    """Linear warmup followed by cosine decay to a non-zero floor."""

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


def autocast_context(device: torch.device, dtype: torch.dtype):
    """Use mixed precision only on CUDA."""

    if device.type != "cuda":
        return nullcontext()

    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    tokens: Tensor,
    batch_size: int,
    context: int,
    depth: int,
    batches: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    """Measure held-out next-token cross entropy at one fixed recurrence depth."""

    model.eval()
    losses: list[float] = []

    for _ in range(batches):
        x, y = sample_batch(tokens, batch_size, context, device)

        with autocast_context(device, amp_dtype):
            logits = model(x, depth=depth)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

        losses.append(float(loss))

    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    path: str | Path,
    model: ArcNeuron,
    optimizer: torch.optim.Optimizer,
    tokenizer: ArcTokenizer,
    step: int,
) -> None:
    """Save everything required to reconstruct and resume the neural model."""

    checkpoint = {
        "format": 1,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer_model": tokenizer.to_bytes(),
        "step": step,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_random_state"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)


def format_eta(seconds: float) -> str:
    """Turn an ETA in seconds into a compact human-readable duration."""

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
    """Run base pretraining."""

    args = parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be positive")
    if args.max_depth < 1:
        raise ValueError("--max-depth must be at least 1")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.eval_every < 0:
        raise ValueError("--eval-every cannot be negative")
    if args.save_every < 0:
        raise ValueError("--save-every cannot be negative")

    seed_everything(args.seed)
    device = choose_device()

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    corpus_text = read_text(args.data)

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        tokenizer = ArcTokenizer.from_bytes(checkpoint["tokenizer_model"])
        config = ArcNeuronConfig(**checkpoint["model_config"])

        model = ArcNeuron(config).to(device)
        model.load_state_dict(checkpoint["model_state"])

        optimizer = make_optimizer(
            model,
            args.lr,
            args.weight_decay,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_step = int(checkpoint["step"])

        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])

        if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    else:
        tokenizer = ArcTokenizer.train(
            args.data,
            vocab_size=args.vocab_size,
        )

        config = ArcNeuronConfig(
            vocab_size=tokenizer.vocab_size,
            dim=args.dim,
            n_heads=args.heads,
            n_kv_heads=args.kv_heads,
            ffn_dim=args.ffn_dim,
            max_seq_len=args.context,
            prelude_layers=args.prelude_layers,
            core_layers=args.core_layers,
            coda_layers=args.coda_layers,
        )

        model = ArcNeuron(config).to(device)
        optimizer = make_optimizer(
            model,
            args.lr,
            args.weight_decay,
        )
        start_step = 0

    if start_step >= args.steps:
        raise ValueError(
            f"checkpoint is already at step {start_step}, "
            f"which is not below requested --steps={args.steps}"
        )

    token_ids = tokenizer.encode(
        corpus_text,
        add_bos=True,
        add_eos=True,
    )

    train_tokens, val_tokens = split_tokens(
        token_ids,
        config.max_seq_len,
    )

    raw_model = model
    run_model = torch.compile(model) if args.compile else model

    parameter_count = sum(
        parameter.numel()
        for parameter in raw_model.parameters()
    )

    print(
        f"device={device} "
        f"params={parameter_count:,} "
        f"vocab={config.vocab_size:,} "
        f"context={config.max_seq_len} "
        f"train_tokens={train_tokens.numel():,} "
        f"val_tokens={val_tokens.numel():,}",
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

    if train_tokens.numel() < 100_000:
        print(
            "warning: this is a very small corpus. "
            "It is useful for pipeline experiments and overfitting checks, "
            "not for producing a generally capable language model.",
            flush=True,
        )

    raw_model.train()

    started = time.perf_counter()
    local_steps = 0
    last_val_loss: float | None = None

    for step in range(start_step, args.steps):
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

        for _ in range(args.grad_accum):
            x, y = sample_batch(
                train_tokens,
                args.batch_size,
                config.max_seq_len,
                device,
            )

            with autocast_context(device, amp_dtype):
                logits = run_model(x, depth=depth)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                scaled_loss = loss / args.grad_accum

            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach())

        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(),
            args.clip,
        )

        scaler.step(optimizer)
        scaler.update()

        completed_step = step + 1
        local_steps += 1

        should_eval = (
            args.eval_every > 0
            and (
                completed_step % args.eval_every == 0
                or completed_step == args.steps
            )
        )

        if should_eval:
            eval_depth = min(args.max_depth, 4)
            last_val_loss = estimate_loss(
                run_model,
                val_tokens,
                args.batch_size,
                config.max_seq_len,
                eval_depth,
                args.eval_batches,
                device,
                amp_dtype,
            )

        should_log = (
            local_steps == 1
            or completed_step % args.log_every == 0
            or completed_step == args.steps
            or should_eval
        )

        if should_log:
            elapsed = max(
                time.perf_counter() - started,
                1e-9,
            )

            tokens_processed = (
                local_steps
                * args.batch_size
                * config.max_seq_len
                * args.grad_accum
            )

            tok_per_second = tokens_processed / elapsed
            seconds_per_step = elapsed / local_steps
            remaining_steps = args.steps - completed_step
            eta = format_eta(seconds_per_step * remaining_steps)

            mean_loss = accumulated_loss / args.grad_accum
            progress = 100.0 * completed_step / args.steps

            val_text = (
                f" val={last_val_loss:.4f}"
                if last_val_loss is not None
                else ""
            )

            memory_text = ""

            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_text = f" vram={allocated:.2f}GiB"

            print(
                f"step={completed_step}/{args.steps} "
                f"({progress:6.2f}%) "
                f"depth={depth} "
                f"loss={mean_loss:.4f}"
                f"{val_text} "
                f"lr={current_lr:.2e} "
                f"grad={float(grad_norm):.3f} "
                f"tok/s={tok_per_second:,.0f} "
                f"eta={eta}"
                f"{memory_text}",
                flush=True,
            )

        should_save = (
            (args.save_every > 0 and completed_step % args.save_every == 0)
            or completed_step == args.steps
        )

        if should_save:
            save_checkpoint(
                args.out,
                raw_model,
                optimizer,
                tokenizer,
                completed_step,
            )

            print(
                f"saved {args.out} at step {completed_step}",
                flush=True,
            )


if __name__ == "__main__":
    main()
