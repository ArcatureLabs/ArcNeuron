"""Continue training an ArcNeuron checkpoint on a small hand-guided text corpus.

"Tuning" here is intentionally literal continued language-model training.  It is
not a separate classifier, LoRA adapter, reward model, symbolic reasoner, or
special instruction objective.  The same ArcNeuron weights keep learning with
the same next-token cross-entropy loss, only with a lower learning rate and a
small amount of replay from train.txt to reduce forgetting.
"""

import argparse  # Tuning budget and file paths should be easy to change from one Colab command.
import random  # Random corpus selection and recurrent depth keep tuning examples from becoming a fixed execution script.
import time  # Timing reveals whether a chosen context/depth combination is practical on the current GPU.
from contextlib import nullcontext  # CPU execution uses the same loop without mixed-precision context management.
from dataclasses import asdict  # Save the unchanged neural architecture back into the tuned checkpoint.
from pathlib import Path  # Explicit paths make checkpoint and corpus handling easy to audit.

import torch  # Tuning updates the same real neural parameters created during base training.
from torch import Tensor  # Tensor annotations document the plain token-stream batches.
from torch.nn import functional as F  # Cross entropy remains the one and only learning objective.

from arcneuron import ArcNeuron, ArcNeuronConfig  # The tuned model is exactly the same architecture as the base model.
from tokenizer import ArcTokenizer  # The base tokenizer is restored rather than retrained on tuning text.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue ArcNeuron training on tune.txt")  # Keep the tuning entry point discoverable without reading source first.
    parser.add_argument("--checkpoint", default="arcneuron.pt")  # Tuning always starts from an already trained ArcNeuron checkpoint.
    parser.add_argument("--data", default="tune.txt")  # Human-guided natural text lives in a plain text file.
    parser.add_argument("--replay-data", default="train.txt")  # A little base text replay helps preserve general language behavior.
    parser.add_argument("--out", default="arcneuron-tuned.pt")  # Never overwrite the base model unless the user deliberately asks for that path.
    parser.add_argument("--steps", type=int, default=1000)  # Tuning should be much shorter than base pretraining by default.
    parser.add_argument("--batch-size", type=int, default=8)  # A modest microbatch leaves room for longer reasoning-like prose examples.
    parser.add_argument("--grad-accum", type=int, default=1)  # Effective batch size can grow without changing the model or corpus format.
    parser.add_argument("--context", type=int, default=512)  # Tuning may use shorter chunks than the model's maximum trained context.
    parser.add_argument("--max-depth", type=int, default=4)  # Continue exercising multiple recurrent depths instead of collapsing to one fixed depth.
    parser.add_argument("--lr", type=float, default=5e-5)  # A lower rate nudges behavior without aggressively overwriting base representations.
    parser.add_argument("--weight-decay", type=float, default=0.05)  # Keep light regularization while adapting the existing model.
    parser.add_argument("--clip", type=float, default=1.0)  # Recurrent gradients receive the same safety bound as base training.
    parser.add_argument("--replay-ratio", type=float, default=0.15)  # Roughly fifteen percent of batches can come from base text to resist forgetting.
    parser.add_argument("--save-every", type=int, default=100)  # Colab users should not lose a long tuning run when the runtime disconnects.
    parser.add_argument("--seed", type=int, default=2026)  # A separate fixed seed makes tuning runs comparable while remaining independent of pretraining RNG.
    return parser.parse_args()  # One namespace holds every runtime choice without hiding policy in global constants.


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use the GPU automatically when Colab exposes one.


def read_text(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")  # Tuning examples remain ordinary UTF-8 prose exactly as written by a human.
    if not text.strip():  # An empty tuning file should fail before optimizer state is touched.
        raise ValueError(f"text corpus is empty: {path}")
    return text  # No labels, templates, or reasoning annotations are created in Python.


def to_tokens(tokenizer: ArcTokenizer, text: str) -> Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)  # Reuse the base model's immutable token-ID geometry.
    return torch.tensor(ids, dtype=torch.long)  # Keep the whole corpus on CPU and move only sampled chunks to the accelerator.


def sample_batch(tokens: Tensor, batch_size: int, context: int, device: torch.device) -> tuple[Tensor, Tensor]:
    if tokens.numel() < context + 2:  # The chosen tuning context must fit at least one shifted training example.
        raise ValueError("tuning text is too small for --context; lower the context or add more natural text")
    highest_start = tokens.numel() - context - 1  # Targets consume one token beyond each input chunk.
    starts = torch.randint(0, highest_start, (batch_size,))  # Random contiguous chunks avoid hand-engineered example boundaries.
    x = torch.stack([tokens[start : start + context] for start in starts])  # Input remains plain language-model context.
    y = torch.stack([tokens[start + 1 : start + context + 1] for start in starts])  # Target remains plain next-token continuation.
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # Only the current batch uses GPU memory.


def make_optimizer(model: ArcNeuron, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay = []  # Learned matrices and embeddings receive decoupled weight decay.
    no_decay = []  # RMSNorm scales remain free from shrinkage.
    for parameter in model.parameters():  # Tuning updates every neural parameter rather than attaching an adapter beside the model.
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)  # Matrix rank is a simple transparent parameter-group rule.
    groups = [  # Both groups use identical Adam moments and learning rate.
        {"params": decay, "weight_decay": weight_decay},  # Regularize matrix-shaped weights lightly during adaptation.
        {"params": no_decay, "weight_decay": 0.0},  # Preserve normalization-scale flexibility.
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())  # CUDA gets fused AdamW automatically when available.


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":  # CPU tuning is primarily a correctness path.
        return nullcontext()  # Avoid unnecessary precision conversion on CPU.
    return torch.autocast(device_type="cuda", dtype=dtype)  # GPU uses BF16 when possible and FP16 otherwise.


def save_checkpoint(path: str | Path, model: ArcNeuron, optimizer: torch.optim.Optimizer, tokenizer: ArcTokenizer, tune_step: int, base_step: int) -> None:
    checkpoint = {  # The tuned file remains independently runnable without the base checkpoint sitting beside it.
        "format": 1,  # Keep the same simple checkpoint format family as train.py.
        "model_config": asdict(model.config),  # Generate.py reconstructs the exact architecture from this dictionary.
        "model_state": model.state_dict(),  # All behavioral changes produced by tuning are inside these neural tensors.
        "optimizer_state": optimizer.state_dict(),  # The tuning run itself can be continued if needed.
        "tokenizer_model": tokenizer.to_bytes(),  # Preserve the base token-to-ID mapping exactly.
        "step": base_step,  # Keep the original base-training progress for provenance.
        "tune_step": tune_step,  # Record how many continued-training updates were applied afterward.
        "python_random_state": random.getstate(),  # Save batch-source and recurrence RNG state for reproducible continuation.
        "torch_random_state": torch.get_rng_state(),  # Save tensor sampling RNG state as well.
    }
    torch.save(checkpoint, path)  # One portable file is enough to run the tuned model later.


def main() -> None:
    args = parse_args()  # Read tuning choices before loading potentially large tensors.
    if not 0.0 <= args.replay_ratio <= 1.0:  # Replay probability is meaningful only inside a valid probability interval.
        raise ValueError("--replay-ratio must be between 0 and 1")
    random.seed(args.seed)  # Make replay selection and recurrent-depth selection reproducible.
    torch.manual_seed(args.seed)  # Make random chunk selection reproducible.
    device = choose_device()  # Select CUDA automatically when available.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)  # Load the complete base model state on the active runtime device.
    tokenizer = ArcTokenizer.from_bytes(checkpoint["tokenizer_model"])  # Never retrain tokenization during tuning because embedding IDs must stay fixed.
    config = ArcNeuronConfig(**checkpoint["model_config"])  # Reconstruct the exact base architecture rather than trusting today's defaults.
    if args.context > config.max_seq_len:  # Tuning cannot exceed the rotary-position range the model was built to process.
        raise ValueError("--context cannot exceed the checkpoint max_seq_len")
    model = ArcNeuron(config).to(device)  # Instantiate only neural architecture before restoring learned behavior.
    model.load_state_dict(checkpoint["model_state"])  # These base weights contain all language and reasoning ability acquired so far.
    model.train()  # Continued training updates those same weights directly.
    optimizer = make_optimizer(model, args.lr, args.weight_decay)  # A fresh low-rate AdamW state avoids carrying high-rate pretraining momentum into tuning.
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16  # Prefer BF16 on modern accelerators.
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16  # Only FP16 requires dynamic loss scaling here.
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # This object is inert for BF16 and CPU execution.
    tune_tokens = to_tokens(tokenizer, read_text(args.data))  # Hand-guided tuning prose is tokenized with the unchanged base tokenizer.
    replay_tokens = to_tokens(tokenizer, read_text(args.replay_data)) if args.replay_ratio > 0.0 else None  # Base text is optional and contributes no new objective.
    parameter_count = sum(parameter.numel() for parameter in model.parameters())  # Tuning never adds parameters; this count should match the base model exactly.
    print(f"device={device} params={parameter_count:,} tune_tokens={tune_tokens.numel():,} replay={args.replay_ratio:.2f}")  # Confirm what will actually be updated.
    started = time.perf_counter()  # Measure practical continued-training throughput.
    base_step = int(checkpoint.get("step", 0))  # Old checkpoints without the field still have a sensible provenance fallback.

    for step in range(args.steps):  # Every iteration ends in one direct update of ArcNeuron's own weights.
        optimizer.zero_grad(set_to_none=True)  # Start the accumulated gradient from an allocation-efficient empty state.
        depth = random.randint(1, args.max_depth)  # Tuning preserves the model's ability to operate at multiple recurrent compute budgets.
        accumulated_loss = 0.0  # This Python scalar exists only for logging.
        replay_batches = 0  # Logging how often replay happened makes forgetting experiments easier to interpret.

        for _ in range(args.grad_accum):  # Multiple text chunks may contribute to the same optimizer step.
            use_replay = replay_tokens is not None and random.random() < args.replay_ratio  # Replay is stochastic at batch level rather than mixed token-by-token.
            source = replay_tokens if use_replay else tune_tokens  # The neural objective is identical regardless of which natural-text corpus supplied the chunk.
            replay_batches += int(use_replay)  # Count replay selections without affecting gradients.
            x, y = sample_batch(source, args.batch_size, args.context, device)  # Draw a normal contiguous language-model batch.
            with autocast_context(device, amp_dtype):  # Use efficient GPU precision without changing the network equation.
                logits = model(x, depth=depth)  # The same recurrent neural core performs all additional computation.
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))  # No instruction mask or separate tuning loss exists.
                scaled_loss = loss / args.grad_accum  # Keep effective gradient scale constant when accumulation changes.
            scaler.scale(scaled_loss).backward()  # Backpropagate through the actual ArcNeuron neurons and recurrent applications.
            accumulated_loss += float(loss.detach())  # Store only a detached scalar for the progress line.

        scaler.unscale_(optimizer)  # Reveal true gradient magnitude before clipping.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # Limit rare unstable recurrent updates.
        scaler.step(optimizer)  # AdamW changes the same weights that generate text at inference time.
        scaler.update()  # FP16 scaling adapts to numerical range; BF16/CPU paths remain unchanged.
        completed = step + 1  # Present tuning progress in human-friendly one-based steps.

        if completed == 1 or completed % 10 == 0:  # Compact frequent logs are useful during small experimental runs.
            elapsed = max(time.perf_counter() - started, 1e-9)  # Avoid division by zero in an extremely fast smoke test.
            processed = completed * args.batch_size * args.context * args.grad_accum  # Count every input token seen by the tuning loop.
            mean_loss = accumulated_loss / args.grad_accum  # Average raw CE across accumulation microbatches.
            print(f"tune_step={completed} depth={depth} loss={mean_loss:.4f} grad={float(grad_norm):.3f} replay_batches={replay_batches} tok/s={processed / elapsed:,.0f}")  # Show behavior, stability, replay, and speed in one line.

        if completed % args.save_every == 0 or completed == args.steps:  # The final requested step is always saved even when it misses the normal interval.
            save_checkpoint(args.out, model, optimizer, tokenizer, completed, base_step)  # Produce a completely self-contained tuned checkpoint.
            print(f"saved {args.out}")  # Make the artifact location obvious in a notebook output cell.


if __name__ == "__main__":  # Merely importing tune.py must never modify a model checkpoint.
    main()  # Command-line execution performs continued plain next-token training.
