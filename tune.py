"""Continue training an ArcNeuron checkpoint on a small hand-guided text corpus.

"Tuning" here remains literal continued language-model training. It is not a
classifier, LoRA adapter, reward model, symbolic reasoner, or alternate objective.
The exact same ArcNeuron weights keep learning from exact same next-token cross
entropy at a lower learning rate.

The only automatic planning here is the amount of exposure to tune.txt. A tiny
behavior corpus should shape response behavior for a few passes, not be repeated
hundreds of times until the model becomes a parrot.
"""

import argparse  # Command-line arguments make tuning convenient in both Colab cells and ordinary terminals.
import math  # Automatic step calculation turns target corpus passes into optimizer updates.
import random  # Replay selection and recurrent depth stay stochastic while remaining seedable.
import time  # Wall-clock timing makes live throughput visible.
from contextlib import nullcontext  # CPU tuning uses the same loop without a mixed-precision context.
from dataclasses import asdict  # The unchanged ArcNeuron architecture remains plain checkpoint metadata.
from pathlib import Path  # Explicit paths keep checkpoint and corpus handling easy to audit.

import torch  # Tuning updates the exact same neural tensors created by base training.
from torch import Tensor  # Tensor annotations document plain token-stream batches and optimizer parameter groups.
from torch.nn import functional as F  # Cross entropy remains the one and only learning objective.

from arcneuron import ArcNeuron, ArcNeuronConfig  # Tuning reconstructs exactly the same neural architecture as the base checkpoint.
from tokenizer import ArcTokenizer  # The base tokenizer is restored rather than retrained on tuning text.


AUTO = "auto"  # A single literal marks values that should be derived from the actual tuning corpus.
DEFAULT_TARGET_PASSES = 3.0  # A small behavior corpus normally needs only a few equivalent passes to nudge response style.
DEFAULT_MAX_BATCH_SIZE = 8  # Tuning defaults to a modest microbatch because response examples may use relatively long natural passages.
DEFAULT_WINDOWS_PER_PASS = 12  # Automatic sizing keeps several optimizer updates in each equivalent tuning pass instead of swallowing the corpus at once.
HEAVY_REPETITION_PASSES = 20.0  # Larger explicit exposure requires an opt-in because it is likely to memorize tune.txt rather than merely guide behavior.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue ArcNeuron training on tune.txt")  # Keep every continued-training control discoverable through --help.
    parser.add_argument("--checkpoint", default="arcneuron.pt")  # Tuning starts from an already trained ArcNeuron checkpoint.
    parser.add_argument("--data", default="tune.txt")  # Human-guided natural text remains a plain UTF-8 file.
    parser.add_argument("--replay-data", default="train.txt")  # A small amount of base text replay helps resist catastrophic forgetting.
    parser.add_argument("--out", default="arcneuron-tuned.pt")  # Never overwrite the base model unless the caller explicitly requests that path.

    parser.add_argument("--steps", default=AUTO)  # Auto derives updates from actual tune-token exposure while a numeric value still enables controlled experiments.
    parser.add_argument("--target-passes", type=float, default=DEFAULT_TARGET_PASSES)  # Automatic tuning aims for roughly this many exposures to tune.txt itself.
    parser.add_argument("--batch-size", default=AUTO)  # Auto keeps tiny tuning corpora from disappearing into one huge microbatch.
    parser.add_argument("--grad-accum", type=int, default=1)  # Effective batch size can grow without increasing activation memory.
    parser.add_argument("--context", default=AUTO)  # Auto selects a shorter context from tune.txt while never exceeding the base checkpoint limit.
    parser.add_argument("--max-depth", type=int, default=4)  # Continue exercising multiple recurrent compute budgets during behavior tuning.

    parser.add_argument("--lr", type=float, default=5e-5)  # A lower rate nudges behavior without aggressively overwriting base representations.
    parser.add_argument("--weight-decay", type=float, default=0.05)  # Light regularization remains active during continued training.
    parser.add_argument("--clip", type=float, default=1.0)  # Recurrent gradients receive the same stability bound used during base training.
    parser.add_argument("--replay-ratio", type=float, default=0.20)  # Roughly this fraction of microbatches come from base text instead of tune.txt.

    parser.add_argument("--log-every", type=int, default=5)  # Frequent flushed logs make small Colab tuning runs easy to watch.
    parser.add_argument("--allow-heavy-repetition", action="store_true")  # An explicit opt-in is required when numeric steps would recycle tune.txt excessively.
    parser.add_argument("--seed", type=int, default=2026)  # A separate fixed seed makes tuning runs reproducible and independent from base pretraining.
    return parser.parse_args()  # One namespace keeps all continued-training policy visible at the entry point.


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use the GPU automatically whenever the current runtime exposes one.


def read_text(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")  # Tuning examples remain ordinary UTF-8 prose exactly as written.
    if not text.strip():  # An empty behavior corpus should fail before optimizer state or GPU memory is touched.
        raise ValueError(f"text corpus is empty: {path}")
    return text  # No prompt labels, hidden rules, or semantic preprocessing are injected here.


def to_tokens(tokenizer: ArcTokenizer, text: str) -> Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)  # Reuse the base model's immutable token-ID geometry and explicit sequence boundaries.
    return torch.tensor(ids, dtype=torch.long)  # Keep the complete small corpus on CPU and move only sampled windows to the accelerator.


def parse_positive_integer_or_auto(value: str, name: str) -> int | None:
    if value.lower() == AUTO:  # None tells the caller to calculate this value from the real tuning-token count.
        return None
    try:
        parsed = int(value)  # Numeric overrides remain simple and predictable for controlled experiments.
    except ValueError as error:
        raise ValueError(f"{name} must be 'auto' or a positive integer") from error
    if parsed <= 0:  # Zero or negative sizes cannot form a valid continued-training schedule.
        raise ValueError(f"{name} must be positive")
    return parsed  # A real integer marks a deliberate caller override.


def largest_power_of_two_not_above(value: int) -> int:
    if value < 1:  # Automatic context sizing needs at least one available token position.
        raise ValueError("value must be positive")
    return 1 << (value.bit_length() - 1)  # Power-of-two contexts keep GPU shapes convenient without changing any semantics.


def resolve_context(requested: str, tune_token_count: int, checkpoint_max_context: int) -> int:
    explicit = parse_positive_integer_or_auto(requested, "--context")  # An exact tuning context remains available when an experiment needs it.
    if explicit is not None:
        if explicit > checkpoint_max_context:  # Continued training cannot exceed the RoPE range stored by the base architecture.
            raise ValueError("--context cannot exceed the checkpoint max_seq_len")
        if explicit >= tune_token_count - 1:  # Every training window needs at least one shifted target token.
            raise ValueError("--context is too large for tune.txt")
        return explicit
    usable = max(32, tune_token_count // DEFAULT_WINDOWS_PER_PASS)  # Leave many independent windows in a tiny tune corpus so each optimizer update stays small.
    bounded = min(checkpoint_max_context, usable, tune_token_count - 2)  # Never exceed either the learned architecture limit or the available token stream.
    if bounded < 32:  # Extremely tiny tune files are better expanded than trained with almost no linguistic context.
        raise ValueError("tune.txt is too small for the minimum automatic context; add more natural tuning text")
    return largest_power_of_two_not_above(bounded)  # The resolved context is transparent and printed before tuning begins.


def resolve_batch_size(requested: str, tune_token_count: int, context: int) -> int:
    explicit = parse_positive_integer_or_auto(requested, "--batch-size")  # Known GPU budgets can always override conservative automatic sizing.
    if explicit is not None:
        return explicit
    windows_in_corpus = max(1, tune_token_count // context)  # Roughly measure how many context-sized windows exist in the unique tuning stream.
    batch_from_data = max(1, windows_in_corpus // DEFAULT_WINDOWS_PER_PASS)  # Aim for several optimizer updates during each tuning pass.
    return min(DEFAULT_MAX_BATCH_SIZE, batch_from_data)  # Dataset size may reduce a batch but cannot prove that larger batches fit GPU memory.


def resolve_steps(requested: str, target_passes: float, tune_token_count: int, tokens_per_step: int, replay_ratio: float, allow_heavy_repetition: bool) -> tuple[int, float]:
    if target_passes <= 0.0:  # Automatic tuning exposure must be positive.
        raise ValueError("--target-passes must be positive")
    tune_fraction = 1.0 - replay_ratio  # Only this expected fraction of microbatches actually exposes the optimizer to tune.txt.
    if tune_fraction <= 0.0:  # A hundred-percent replay run never performs tuning and therefore cannot satisfy a tune-pass target.
        raise ValueError("--replay-ratio must be smaller than 1 when --steps is auto")
    expected_tune_tokens_per_step = tokens_per_step * tune_fraction  # This expected value converts overall optimizer work into tune.txt-specific exposure.
    automatic_steps = max(1, math.ceil(tune_token_count * target_passes / expected_tune_tokens_per_step))  # Calculate enough total updates for roughly target_passes of the behavior corpus despite replay.
    explicit = parse_positive_integer_or_auto(requested, "--steps")  # Numeric steps remain available for deliberate controlled experiments.
    steps = automatic_steps if explicit is None else explicit  # Data-aware sizing applies only when an exact number was not requested.
    equivalent_tune_passes = steps * expected_tune_tokens_per_step / tune_token_count  # Report expected tune.txt exposure rather than counting replay tokens as behavior examples.
    if explicit is not None and equivalent_tune_passes > HEAVY_REPETITION_PASSES and not allow_heavy_repetition:  # Huge stale step values must not silently turn tuning into memorization.
        raise ValueError(
            f"{steps} tuning steps would expose tune.txt to about {equivalent_tune_passes:.1f} equivalent passes; "
            f"use --steps auto or add --allow-heavy-repetition only when that repetition is intentional"
        )
    return steps, equivalent_tune_passes  # Both values are printed before any neural weight is changed.


def sample_batch(tokens: Tensor, batch_size: int, context: int, device: torch.device) -> tuple[Tensor, Tensor]:
    if tokens.numel() < context + 2:  # The selected tuning context must fit at least one shifted next-token example.
        raise ValueError("text is too small for the resolved context")
    highest_start = tokens.numel() - context - 1  # Targets consume one token beyond every input chunk.
    starts = torch.randint(0, highest_start, (batch_size,))  # Each sequence starts at an independently sampled corpus position.
    x = torch.stack([tokens[start : start + context] for start in starts])  # Inputs remain plain contiguous natural-language windows.
    y = torch.stack([tokens[start + 1 : start + context + 1] for start in starts])  # Targets remain the exact next real token at every position.
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # Only the current microbatch consumes accelerator memory.


def make_optimizer(model: ArcNeuron, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay = []  # Learned matrices and the embedding table receive decoupled weight decay.
    no_decay = []  # One-dimensional RMSNorm scales remain free from shrinkage.
    for parameter in model.parameters():  # Tuning updates every neural parameter directly rather than attaching an adapter beside ArcNeuron.
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)  # Tensor rank provides one explicit auditable grouping rule.
    groups = [  # Both groups share Adam moments and learning rate while differing only in decay.
        {"params": decay, "weight_decay": weight_decay},  # Apply light regularization to matrix-shaped learned tensors.
        {"params": no_decay, "weight_decay": 0.0},  # Preserve normalization-scale flexibility.
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())  # CUDA receives fused AdamW automatically when PyTorch supports it.


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":  # CPU tuning is primarily a correctness and smoke-test path.
        return nullcontext()  # Avoid unnecessary mixed-precision conversion outside CUDA.
    return torch.autocast(device_type="cuda", dtype=dtype)  # GPU uses BF16 when possible and FP16 otherwise.


def save_checkpoint(path: str | Path, model: ArcNeuron, optimizer: torch.optim.Optimizer, tokenizer: ArcTokenizer, tune_step: int, base_step: int, tuning_plan: dict[str, int | float | str]) -> None:
    checkpoint = {  # The tuned checkpoint remains independently runnable without the base checkpoint sitting beside it.
        "format": 2,  # Format two includes transparent data-aware tuning metadata while model reconstruction remains unchanged.
        "model_config": asdict(model.config),  # generate.py reconstructs the exact neural tensor graph from these dimensions.
        "model_state": model.state_dict(),  # Every learned behavior change produced by tuning lives in these neural tensors.
        "optimizer_state": optimizer.state_dict(),  # The tuning run itself can continue later if needed.
        "tokenizer_model": tokenizer.to_bytes(),  # Preserve the base token-to-ID mapping exactly.
        "step": base_step,  # Keep original base-training progress for provenance.
        "tune_step": tune_step,  # Record the number of continued-training optimizer updates applied afterward.
        "tuning_plan": tuning_plan,  # Human-readable exposure numbers are checkpoint metadata and are never used by model.forward().
    }
    torch.save(checkpoint, path)  # One portable PyTorch file remains enough to run the tuned neural model later.


def main() -> None:
    args = parse_args()  # Read all continued-training choices before loading potentially large tensors.
    if not 0.0 <= args.replay_ratio < 1.0:  # Tuning requires at least some actual tune.txt batches.
        raise ValueError("--replay-ratio must be between 0 inclusive and 1 exclusive")
    if args.grad_accum <= 0:  # One optimizer update must contain at least one microbatch.
        raise ValueError("--grad-accum must be positive")
    if args.max_depth < 1:  # ArcNeuron's recurrent core must execute at least once.
        raise ValueError("--max-depth must be at least 1")
    if args.log_every <= 0:  # Progress cadence cannot be zero.
        raise ValueError("--log-every must be positive")

    random.seed(args.seed)  # Replay selection and recurrent-depth choice become reproducible.
    torch.manual_seed(args.seed)  # Random token-window sampling becomes reproducible.
    if torch.cuda.is_available():  # CUDA multinomial/tensor randomness has a separate generator stream.
        torch.cuda.manual_seed_all(args.seed)  # Seed every visible GPU consistently.
        torch.set_float32_matmul_precision("high")  # Prefer efficient high-precision CUDA matrix kernels where PyTorch supports them.

    device = choose_device()  # Select CUDA automatically when a Colab GPU runtime is active.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)  # Load the complete base checkpoint directly onto the active runtime device.
    tokenizer = ArcTokenizer.from_bytes(checkpoint["tokenizer_model"])  # Never retrain tokenization during tuning because embedding IDs must remain fixed.
    config = ArcNeuronConfig(**checkpoint["model_config"])  # The checkpoint alone defines the base model architecture.
    model = ArcNeuron(config).to(device)  # Instantiate only the same neural graph before restoring learned behavior.
    try:
        model.load_state_dict(checkpoint["model_state"])  # Strict loading prevents accidentally tuning a checkpoint from an incompatible older ArcNeuron graph.
    except RuntimeError as error:
        raise RuntimeError("checkpoint is incompatible with the current ArcNeuron architecture; train a fresh base checkpoint after architecture changes") from error
    model.train()  # Continued training updates those same neural weights directly.

    tune_tokens = to_tokens(tokenizer, read_text(args.data))  # Human-guided tuning prose is tokenized with the unchanged base tokenizer.
    replay_tokens = to_tokens(tokenizer, read_text(args.replay_data)) if args.replay_ratio > 0.0 else None  # Base-text replay is optional and never introduces a second objective.
    context = resolve_context(args.context, tune_tokens.numel(), config.max_seq_len)  # Automatic context respects both tune.txt size and the base model's fixed RoPE ceiling.
    batch_size = resolve_batch_size(args.batch_size, tune_tokens.numel(), context)  # Tiny tune files automatically use smaller microbatches to preserve multiple optimizer updates per pass.
    tokens_per_step = batch_size * context * args.grad_accum  # This is total token exposure per optimizer update before replay selection.
    total_steps, equivalent_tune_passes = resolve_steps(args.steps, args.target_passes, tune_tokens.numel(), tokens_per_step, args.replay_ratio, args.allow_heavy_repetition)  # Convert tune.txt size into a transparent behavior-exposure budget.

    optimizer = make_optimizer(model, args.lr, args.weight_decay)  # A fresh low-rate AdamW state avoids carrying high-rate pretraining momentum into behavior tuning.
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16  # Prefer BF16 on modern accelerators for its wider exponent range.
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16  # FP16 gradients may underflow whereas BF16 generally does not require loss scaling.
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # This object becomes inert for BF16 and CPU execution.

    parameter_count = sum(parameter.numel() for parameter in model.parameters())  # Tuning never adds parameters; this count must match the base neural model exactly.
    tuning_plan: dict[str, int | float | str] = {  # Record the actual tuning budget for experiment provenance without creating an inference dependency.
        "tune_tokens": int(tune_tokens.numel()),  # Unique tune.txt token count under the base tokenizer.
        "replay_tokens": int(replay_tokens.numel()) if replay_tokens is not None else 0,  # Base replay stream length when replay is enabled.
        "parameters": int(parameter_count),  # Exact neural parameter count being updated.
        "context": int(context),  # Resolved tuning context, always within the checkpoint architecture limit.
        "batch_size": int(batch_size),  # Resolved microbatch size.
        "grad_accum": int(args.grad_accum),  # Number of microbatches contributing to one optimizer update.
        "tokens_per_step": int(tokens_per_step),  # Total token positions processed by each optimizer update.
        "steps": int(total_steps),  # Planned continued-training optimizer updates.
        "expected_tune_passes": float(equivalent_tune_passes),  # Expected tune.txt exposure after accounting for replay probability.
        "replay_ratio": float(args.replay_ratio),  # Expected fraction of microbatches reserved for base-text replay.
        "schedule_source": AUTO if args.steps.lower() == AUTO else "explicit",  # Record whether data or the caller selected the step count.
    }

    print(
        f"device={device} params={parameter_count:,} tune_tokens={tune_tokens.numel():,} "
        f"context={context} batch={batch_size} grad_accum={args.grad_accum}",
        flush=True,
    )  # The first line shows the exact neural/data scale before any base weights are changed.
    print(
        f"tokens/step={tokens_per_step:,} steps={total_steps:,} "
        f"expected_tune_exposure={equivalent_tune_passes:.2f} passes replay={args.replay_ratio:.2f}",
        flush=True,
    )  # The second line makes the automatically derived continued-training budget explicit instead of hiding a magic 600-step constant.

    started = time.perf_counter()  # Measure practical continued-training throughput from the first optimizer update.
    base_step = int(checkpoint.get("step", 0))  # Preserve base-pretraining progress for checkpoint provenance.

    for step in range(total_steps):  # Each iteration ends in one direct update of ArcNeuron's own weights.
        optimizer.zero_grad(set_to_none=True)  # Start accumulated gradients from an allocation-efficient empty state.
        depth = random.randint(1, args.max_depth)  # Tuning preserves the ability to operate at several recurrent compute budgets.
        accumulated_loss = 0.0  # This Python scalar exists only for live logging.
        replay_batches = 0  # Count replay selections so the observed run can be compared with the requested probability.

        for _ in range(args.grad_accum):  # Multiple independent natural-text windows may contribute to one optimizer update.
            use_replay = replay_tokens is not None and random.random() < args.replay_ratio  # Replay is selected at microbatch level instead of mixing hidden labels into token sequences.
            source = replay_tokens if use_replay else tune_tokens  # The neural objective stays exactly the same regardless of which natural-text stream supplied the batch.
            replay_batches += int(use_replay)  # Logging replay frequency does not participate in gradients.
            source_context = min(context, source.numel() - 2)  # A smaller replay stream may require a shorter window while remaining valid for the same model.
            x, y = sample_batch(source, batch_size, source_context, device)  # Draw one ordinary causal-language-model batch.
            with autocast_context(device, amp_dtype):  # Use efficient GPU precision without changing the network equation or target.
                logits = model(x, depth=depth)  # The same recurrent neural core performs every extra computation step.
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))  # No instruction mask, reward term, or separate tuning objective exists.
                scaled_loss = loss / args.grad_accum  # Keep effective update scale independent of gradient accumulation.
            scaler.scale(scaled_loss).backward()  # Backpropagate through the actual ArcNeuron neurons and every recurrent application.
            accumulated_loss += float(loss.detach())  # Store only a detached scalar for the progress line.

        scaler.unscale_(optimizer)  # Reveal true gradient magnitudes before clipping.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # Limit rare unstable recurrent updates without adding task logic.
        scaler.step(optimizer)  # AdamW changes the exact same weights generate.py later loads.
        scaler.update()  # FP16 scaling adapts to numerical range while BF16/CPU paths remain unchanged.
        completed = step + 1  # Human-facing tuning progress starts from one.

        if completed == 1 or completed % args.log_every == 0 or completed == total_steps:  # Compact frequent logs are enough for short behavior-tuning runs.
            elapsed = max(time.perf_counter() - started, 1e-9)  # Avoid division by zero during an extremely fast smoke test.
            processed = completed * tokens_per_step  # Count all input token positions processed so far, including replay.
            tok_per_second = processed / elapsed  # End-to-end speed includes recurrent compute and optimizer work.
            eta_seconds = elapsed / completed * (total_steps - completed)  # Average step time gives a simple stable remaining-time estimate.
            mean_loss = accumulated_loss / args.grad_accum  # Report average raw CE across accumulated microbatches.
            memory_text = ""  # CPU tuning has no CUDA allocation number.
            if device.type == "cuda":  # CUDA users benefit from seeing actual current allocation while adjusting batch/context.
                allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert allocated bytes into GiB.
                memory_text = f" vram={allocated:.2f}GiB"  # Keep memory on the same compact line.
            minutes, seconds = divmod(int(max(0.0, eta_seconds)), 60)  # Tuning runs are normally short enough for a minute/second ETA.
            eta = f"{minutes:d}m{seconds:02d}s" if minutes else f"{seconds:d}s"  # Compact ETA remains easy to read in Colab.
            print(
                f"tune={completed}/{total_steps} depth={depth} loss={mean_loss:.4f} "
                f"grad={float(grad_norm):.3f} replay={replay_batches}/{args.grad_accum} "
                f"tok/s={tok_per_second:,.0f} eta={eta}{memory_text}",
                flush=True,
            )  # flush=True makes progress visible immediately through a notebook subprocess.

    save_checkpoint(args.out, model, optimizer, tokenizer, total_steps, base_step, tuning_plan)  # Save one completely self-contained tuned neural checkpoint after the planned exposure budget.
    print(f"saved {args.out}", flush=True)  # Make the final artifact path obvious before an ephemeral Colab runtime disappears.


if __name__ == "__main__":  # Merely importing tune.py must never modify a checkpoint.
    main()  # Direct execution performs continued plain next-token training.
