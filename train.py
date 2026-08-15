"""Train ArcNeuron from raw natural text with ordinary next-token prediction.

There is deliberately no symbolic reasoning code here.  Python chooses tensor
batches, recurrent depth, optimization settings, and checkpoint timing.  The
neural network alone must learn language, knowledge, and useful recurrent
computation from the prediction objective.
"""

import argparse  # Command-line arguments make the same file convenient on Colab, a workstation, or a rented GPU.
import math  # Cosine learning-rate decay needs pi and cos.
import random  # Python chooses one recurrent depth for each complete microbatch.
import time  # Wall-clock timing makes GPU throughput visible while experimenting.
from contextlib import nullcontext  # CPU training does not need an autocast context, so nullcontext keeps one code path.
from dataclasses import asdict  # The architecture dataclass is serialized as plain checkpoint metadata.
from pathlib import Path  # Paths stay readable and work on Linux, Windows, and Colab.

import torch  # PyTorch owns tensors, automatic differentiation, optimizers, and CUDA execution.
from torch import Tensor  # Tensor annotations make batch helpers easier to audit.
from torch.nn import functional as F  # Cross entropy is the only training objective used by ArcNeuron R1.

from arcneuron import ArcNeuron, ArcNeuronConfig  # This import is the complete neural architecture and nothing else.
from tokenizer import ArcTokenizer  # Tokenization compresses text but contributes no task or reasoning logic.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ArcNeuron from train.txt")  # Keep the entry point self-documenting in terminals and notebooks.
    parser.add_argument("--data", default="train.txt")  # Raw natural text is the only required training corpus.
    parser.add_argument("--out", default="arcneuron.pt")  # One PyTorch checkpoint stores weights, config, tokenizer, and optimizer state.
    parser.add_argument("--resume", default=None)  # An interrupted Colab session can continue from an existing checkpoint when desired.
    parser.add_argument("--steps", type=int, default=5000)  # Total optimizer steps are explicit because corpus size and GPU budget vary wildly.
    parser.add_argument("--batch-size", type=int, default=16)  # Microbatch size is the first knob to lower when CUDA memory is tight.
    parser.add_argument("--grad-accum", type=int, default=1)  # Gradient accumulation increases effective batch size without increasing activation memory.
    parser.add_argument("--context", type=int, default=1024)  # Training chunks never exceed the architecture's rotary-position limit.
    parser.add_argument("--vocab-size", type=int, default=8192)  # SentencePiece may return a smaller actual vocabulary for tiny corpora.
    parser.add_argument("--dim", type=int, default=512)  # Default width targets a compact proof-of-concept rather than a huge first experiment.
    parser.add_argument("--heads", type=int, default=8)  # Eight query heads give the 512-wide model 64 features per head.
    parser.add_argument("--kv-heads", type=int, default=2)  # Grouped-query attention shares each KV head across four query heads.
    parser.add_argument("--ffn-dim", type=int, default=1408)  # A SwiGLU hidden width near 2.75x model width is efficient for this small model.
    parser.add_argument("--prelude-layers", type=int, default=1)  # One input block is enough for the first controlled recurrent experiment.
    parser.add_argument("--core-layers", type=int, default=2)  # Two shared core blocks give each recurrent iteration meaningful compute.
    parser.add_argument("--coda-layers", type=int, default=1)  # One output block converts the recurrent state back toward token prediction.
    parser.add_argument("--max-depth", type=int, default=4)  # Training varies recurrence between one and this value instead of hardcoding one depth.
    parser.add_argument("--lr", type=float, default=3e-4)  # A small-model AdamW learning rate that is easy to override for scaling experiments.
    parser.add_argument("--min-lr", type=float, default=3e-5)  # Cosine decay stops at a nonzero floor instead of collapsing learning completely.
    parser.add_argument("--warmup", type=int, default=200)  # Early warmup avoids an abrupt full-rate update from randomly initialized weights.
    parser.add_argument("--weight-decay", type=float, default=0.1)  # Matrix weights receive decoupled L2-style regularization through AdamW.
    parser.add_argument("--clip", type=float, default=1.0)  # Gradient clipping protects recurrent training from occasional exploding updates.
    parser.add_argument("--eval-every", type=int, default=100)  # Frequent validation is useful while architecture experiments are still cheap.
    parser.add_argument("--eval-batches", type=int, default=10)  # A few random validation chunks are enough for a lightweight progress signal.
    parser.add_argument("--seed", type=int, default=1337)  # Fixed seeds make two architecture runs much easier to compare.
    parser.add_argument("--compile", action="store_true")  # torch.compile is optional because tiny Colab runs may not amortize compilation time.
    return parser.parse_args()  # Returning one namespace keeps configuration visible at the top of main().


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA is preferred automatically; CPU remains a functional smoke-test path.


def seed_everything(seed: int) -> None:
    random.seed(seed)  # Recurrent-depth sampling becomes reproducible.
    torch.manual_seed(seed)  # CPU parameter initialization and random batches become reproducible.
    if torch.cuda.is_available():  # CUDA has a separate random-number stream that should use the same experiment seed.
        torch.cuda.manual_seed_all(seed)  # Seed every visible GPU in case the runtime exposes more than one device.


def read_text(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")  # Preserve the corpus exactly as UTF-8 natural text.
    if not text.strip():  # An empty corpus can only produce meaningless batches and confusing downstream errors.
        raise ValueError(f"training corpus is empty: {path}")
    return text  # No cleaning, dictionary expansion, or semantic preprocessing is performed here.


def split_tokens(token_ids: list[int], context: int) -> tuple[Tensor, Tensor]:
    tokens = torch.tensor(token_ids, dtype=torch.long)  # Token IDs remain integers until ArcNeuron's learned embedding sees them.
    minimum = context + 2  # Each sampled sequence needs context inputs plus at least one shifted target position.
    val_size = max(int(tokens.numel() * 0.05), minimum)  # Keep five percent normally, but guarantee at least one full validation window for small demo corpora.
    split = tokens.numel() - val_size  # Everything before the reserved suffix becomes the optimizer-visible training stream.
    train_tokens = tokens[:split]  # Random training chunks are sampled only from this prefix.
    val_tokens = tokens[split:]  # Validation chunks come only from the held-out suffix.
    if train_tokens.numel() < minimum or val_tokens.numel() < minimum:  # Tiny corpora should fail with a useful message instead of an indexing exception.
        raise ValueError("train.txt is too small for this context length; lower --context or add more natural text")
    return train_tokens, val_tokens  # Both tensors stay on CPU so the full corpus does not consume scarce GPU memory.


def sample_batch(tokens: Tensor, batch_size: int, context: int, device: torch.device) -> tuple[Tensor, Tensor]:
    highest_start = tokens.numel() - context - 1  # One extra token is required because targets are shifted by one position.
    starts = torch.randint(0, highest_start, (batch_size,))  # Every sequence begins at an independently sampled corpus position.
    x = torch.stack([tokens[start : start + context] for start in starts])  # Inputs are contiguous natural-text windows, not engineered feature vectors.
    y = torch.stack([tokens[start + 1 : start + context + 1] for start in starts])  # Targets are exactly the next token at every position.
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # Transfer only the current batch to the accelerator.


def make_optimizer(model: ArcNeuron, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay = []  # Matrix parameters benefit from weight decay.
    no_decay = []  # One-dimensional normalization scales should not be shrunk toward zero.
    for parameter in model.parameters():  # The optimizer sees every trainable neuron parameter in the architecture.
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)  # Shape alone cleanly separates matrices/embeddings from scale vectors.
    groups = [  # AdamW can apply different regularization while sharing every other optimizer setting.
        {"params": decay, "weight_decay": weight_decay},  # Regularize learned matrices and the tied embedding/output table.
        {"params": no_decay, "weight_decay": 0.0},  # Leave RMSNorm scale parameters unregularized.
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())  # Fused AdamW is used automatically on CUDA when PyTorch supports it.


def learning_rate(step: int, total_steps: int, warmup: int, maximum: float, minimum: float) -> float:
    if warmup > 0 and step < warmup:  # The first updates should grow gradually from almost zero.
        return maximum * (step + 1) / warmup  # Linear warmup reaches the requested maximum exactly at the warmup boundary.
    if total_steps <= warmup:  # Degenerate short experiments should simply stay at the maximum rate after warmup logic.
        return maximum
    progress = min(1.0, (step - warmup) / (total_steps - warmup))  # Clamp progress so resumed or extended runs never create an invalid cosine phase.
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # Smoothly move from one to zero across the remaining training schedule.
    return minimum + cosine * (maximum - minimum)  # Scale the cosine into the requested [minimum, maximum] learning-rate interval.


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":  # CPU smoke tests prioritize correctness and simplicity over mixed-precision speed.
        return nullcontext()  # A no-op context lets the training loop stay identical on both devices.
    return torch.autocast(device_type="cuda", dtype=dtype)  # GPU matrix operations use BF16 or FP16 while sensitive operations may remain FP32.


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, tokens: Tensor, batch_size: int, context: int, depth: int, batches: int, device: torch.device, amp_dtype: torch.dtype) -> float:
    model.eval()  # Disable training-only behavior before measuring validation quality.
    losses = []  # Averaging several random chunks is less noisy than trusting one arbitrary slice.
    for _ in range(batches):  # Validation remains deliberately lightweight so most GPU time goes to learning.
        x, y = sample_batch(tokens, batch_size, context, device)  # Draw text the optimizer never trained on.
        with autocast_context(device, amp_dtype):  # Match the numerical path used during training when running on GPU.
            logits = model(x, depth=depth)  # Validation uses a fixed depth so values remain comparable across checkpoints.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))  # Evaluate the same next-token objective used for optimization.
        losses.append(float(loss))  # Move only a scalar back to Python rather than storing validation graphs or logits.
    model.train()  # Restore training mode for the caller's next optimizer step.
    return sum(losses) / len(losses)  # Report one human-readable held-out cross-entropy number.


def save_checkpoint(path: str | Path, model: ArcNeuron, optimizer: torch.optim.Optimizer, tokenizer: ArcTokenizer, step: int) -> None:
    checkpoint = {  # Everything required to reconstruct the same neural training state lives in one file.
        "format": 1,  # A tiny format version gives future code a clean migration point.
        "model_config": asdict(model.config),  # Architectural dimensions are data, not hidden assumptions in generate.py.
        "model_state": model.state_dict(),  # These tensors contain the learned behavior of ArcNeuron.
        "optimizer_state": optimizer.state_dict(),  # Resume can continue AdamW momentum instead of restarting optimization dynamics.
        "tokenizer_model": tokenizer.to_bytes(),  # Exact subword compression state travels with the weights.
        "step": step,  # Training resumes from the correct schedule position.
        "python_random_state": random.getstate(),  # Recurrent-depth sampling can continue from the same Python RNG stream.
        "torch_random_state": torch.get_rng_state(),  # CPU random batching and other tensor randomness can also resume exactly.
    }
    torch.save(checkpoint, path)  # PyTorch serializes tensor storage efficiently and map_location can relocate it when loading.


def main() -> None:
    args = parse_args()  # Parse every experiment knob once before touching random state or files.
    seed_everything(args.seed)  # Reproducibility starts before tokenizer-independent model initialization.
    device = choose_device()  # Colab automatically lands on CUDA when a GPU runtime is enabled.
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16  # Prefer BF16 for its wider exponent range; use FP16 on older GPUs.
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16  # FP16 gradients may underflow, whereas BF16 generally does not need loss scaling.
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # The scaler is a no-op unless FP16 CUDA training is active.
    corpus_text = read_text(args.data)  # Read raw natural language without injecting questions, labels, or symbolic features.

    if args.resume:  # Resume reconstructs the exact tokenizer and architecture stored by the earlier run.
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)  # map_location makes a GPU checkpoint loadable on any current device.
        tokenizer = ArcTokenizer.from_bytes(checkpoint["tokenizer_model"])  # Token IDs must retain exactly the same meanings as the embedding rows.
        config = ArcNeuronConfig(**checkpoint["model_config"])  # The checkpoint, not today's CLI defaults, defines the resumed network shape.
        model = ArcNeuron(config).to(device)  # Build the same architecture before loading its learned tensors.
        model.load_state_dict(checkpoint["model_state"])  # Restore all neurons with no Python-side reconstruction of behavior.
        optimizer = make_optimizer(model, args.lr, args.weight_decay)  # Recreate parameter groups around the newly instantiated model objects.
        optimizer.load_state_dict(checkpoint["optimizer_state"])  # Restore AdamW momentum and variance estimates for a true continuation.
        start_step = int(checkpoint["step"])  # Continue the learning-rate schedule from the saved optimizer step.
        random.setstate(checkpoint["python_random_state"])  # Resume depth sampling from the saved Python RNG state.
        torch.set_rng_state(checkpoint["torch_random_state"])  # Resume CPU tensor randomness from the saved state as well.
    else:  # A fresh run learns tokenizer compression first and then initializes ArcNeuron from random neural weights.
        tokenizer = ArcTokenizer.train(args.data, vocab_size=args.vocab_size)  # BPE learns only recurring byte sequences in train.txt.
        config = ArcNeuronConfig(  # Every shape that changes the neural architecture is written explicitly here.
            vocab_size=tokenizer.vocab_size,  # Embedding rows exactly match the trained tokenizer's actual vocabulary.
            dim=args.dim,  # Use the requested hidden width.
            n_heads=args.heads,  # Use the requested query-head count.
            n_kv_heads=args.kv_heads,  # Use the requested GQA key/value-head count.
            ffn_dim=args.ffn_dim,  # Use the requested SwiGLU hidden width.
            max_seq_len=args.context,  # The first implementation trains and serves exactly the configured context ceiling.
            prelude_layers=args.prelude_layers,  # Configure non-recurrent input depth.
            core_layers=args.core_layers,  # Configure shared recurrent depth per iteration.
            coda_layers=args.coda_layers,  # Configure non-recurrent output depth.
        )
        model = ArcNeuron(config).to(device)  # Move all learnable weights to the selected accelerator in one operation.
        optimizer = make_optimizer(model, args.lr, args.weight_decay)  # AdamW now owns every model parameter and no external reasoning parameter exists.
        start_step = 0  # A fresh model begins at the first learning-rate schedule step.

    token_ids = tokenizer.encode(corpus_text, add_bos=True, add_eos=True)  # The corpus becomes a single causal token stream with explicit boundaries.
    train_tokens, val_tokens = split_tokens(token_ids, config.max_seq_len)  # Validation remains held out from gradient updates.
    raw_model = model  # Checkpointing always uses the ordinary ArcNeuron object even if a compiled wrapper is used for execution.
    run_model = torch.compile(model) if args.compile else model  # Compilation is a runtime optimization and cannot alter the model's learned semantics.
    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())  # Count unique shared parameters, not effective unrolled recurrent depth.
    print(f"device={device} params={parameter_count:,} vocab={config.vocab_size} train_tokens={train_tokens.numel():,} val_tokens={val_tokens.numel():,}")  # One line confirms the real experiment before GPU time is spent.
    raw_model.train()  # Explicitly enter training mode before the first forward pass.
    started = time.perf_counter()  # Throughput statistics use a stable monotonic wall clock.

    for step in range(start_step, args.steps):  # Each loop iteration performs exactly one optimizer update after optional gradient accumulation.
        current_lr = learning_rate(step, args.steps, args.warmup, args.lr, args.min_lr)  # Compute this step's warmup/cosine learning rate.
        for group in optimizer.param_groups:  # Every parameter group shares the same schedule even though weight decay differs.
            group["lr"] = current_lr  # Update the optimizer in place before any gradients are accumulated.
        optimizer.zero_grad(set_to_none=True)  # Setting gradients to None saves memory writes compared with explicitly filling them with zeros.
        depth = random.randint(1, args.max_depth)  # One depth for the whole microbatch keeps GPU work aligned instead of leaving samples idle.
        accumulated_loss = 0.0  # This scalar is only for logging; gradient scaling happens on the tensor loss below.

        for _ in range(args.grad_accum):  # Multiple independent text batches can contribute to one optimizer update.
            x, y = sample_batch(train_tokens, args.batch_size, config.max_seq_len, device)  # Sample ordinary contiguous language from the corpus.
            with autocast_context(device, amp_dtype):  # Mixed precision accelerates the matrix-heavy neural forward pass on supported GPUs.
                logits = run_model(x, depth=depth)  # The only "reasoning control" supplied by Python is how many times the same neural core is evaluated.
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))  # ArcNeuron R1 learns solely by predicting the next real token.
                scaled_loss = loss / args.grad_accum  # Averaging accumulated gradients keeps their scale independent of accumulation count.
            scaler.scale(scaled_loss).backward()  # Autograd computes gradients through every recurrent application of the shared core.
            accumulated_loss += float(loss.detach())  # Logging never participates in gradient computation.

        scaler.unscale_(optimizer)  # Gradient clipping must see real gradient magnitudes rather than FP16-scaled values.
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip)  # Cap the total gradient norm to make recurrent experiments harder to destabilize.
        scaler.step(optimizer)  # Update every neural weight using the accumulated gradients if they are finite.
        scaler.update()  # Adjust FP16 loss scale for the next optimizer step; this is a no-op in BF16/CPU mode.

        completed_step = step + 1  # Human-facing step counts are easier to read starting at one rather than zero.
        if completed_step == 1 or completed_step % 10 == 0:  # Frequent compact logs make Colab experiments easy to watch without flooding output.
            elapsed = max(time.perf_counter() - started, 1e-9)  # Protect throughput division from an impossibly tiny timer interval.
            processed = completed_step * args.batch_size * config.max_seq_len * args.grad_accum  # Count input tokens processed since this process started.
            tok_per_second = processed / elapsed  # This includes optimizer and logging overhead, giving a practical end-to-end speed number.
            mean_loss = accumulated_loss / args.grad_accum  # Report the average raw CE across accumulated microbatches.
            print(f"step={completed_step} depth={depth} loss={mean_loss:.4f} lr={current_lr:.2e} grad={float(grad_norm):.3f} tok/s={tok_per_second:,.0f}")  # One line contains the main stability and speed signals.

        should_eval = completed_step % args.eval_every == 0 or completed_step == args.steps  # Always evaluate the final requested step even if it misses the normal interval.
        if should_eval:  # Validation and checkpointing happen together so every saved state has a nearby quality measurement.
            eval_depth = min(args.max_depth, 4)  # A stable modest depth makes validation comparable while avoiding an unnecessarily expensive sweep during training.
            val_loss = estimate_loss(run_model, val_tokens, args.batch_size, config.max_seq_len, eval_depth, args.eval_batches, device, amp_dtype)  # Measure held-out next-token modeling quality.
            print(f"validation step={completed_step} depth={eval_depth} loss={val_loss:.4f}")  # Keep the metric plain so notebook users can grep or plot it easily.
            save_checkpoint(args.out, raw_model, optimizer, tokenizer, completed_step)  # Save the real model weights and runtime state after validation.
            print(f"saved {args.out}")  # Confirm the checkpoint path because Colab runtimes are ephemeral.


if __name__ == "__main__":  # Importing train.py for inspection must never start an expensive training run automatically.
    main()  # Command-line execution begins the complete raw-text pretraining path.
