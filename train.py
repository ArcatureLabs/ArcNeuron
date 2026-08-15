"""Train ArcNeuron from raw natural text with ordinary next-token prediction.

There is deliberately no symbolic reasoning code here. Python reads text, turns
it into ordinary token batches, chooses a recurrent depth, computes next-token
cross entropy, and updates the neural weights.

The trainer also owns the few calculations that a trainer must know anyway:
the real token count after tokenization, the exact parameter count of the actual
ArcNeuron instance, a data-sized context and microbatch when "auto" is requested,
and a step budget expressed as corpus-equivalent token exposure. Nothing from
those calculations is needed by the trained model at inference time.
"""

import argparse  # Command-line arguments keep one trainer useful on Colab, a workstation, or a rented GPU.
import math  # Cosine learning-rate decay, power-of-two context sizing, and exposure calculations need basic mathematics.
import random  # Python chooses held-out paragraphs, training windows, and one recurrent depth for each complete microbatch.
import time  # Wall-clock timing makes live GPU throughput and ETA visible while the experiment is running.
from contextlib import nullcontext  # CPU training does not need autocast, so nullcontext keeps one readable code path.
from dataclasses import asdict  # The architecture dataclass is stored as plain checkpoint metadata beside the learned tensors.
from pathlib import Path  # Explicit paths stay readable and work on Linux, Windows, and Google Colab.

import torch  # PyTorch owns tensors, automatic differentiation, optimizers, mixed precision, and CUDA execution.
from torch import Tensor  # Tensor annotations make batch helpers and parameter groups easier to audit.
from torch.nn import functional as F  # Cross entropy remains the only optimization objective used by ArcNeuron.

from arcneuron import ArcNeuron, ArcNeuronConfig  # This import is the complete neural architecture and contains no trainer logic.
from tokenizer import ArcTokenizer  # Tokenization compresses text into IDs but contributes no semantic rule or answer logic.


AUTO = "auto"  # One literal value lets context, batch size, warmup, evaluation cadence, and steps be selected from the actual data.
DEFAULT_TARGET_PASSES = 4.0  # Four corpus-equivalent token passes are enough for a tiny research corpus to reveal learning without silently repeating it hundreds of times.
DEFAULT_MAX_BATCH_SIZE = 16  # Automatic microbatch sizing never exceeds this conservative value because GPU memory cannot be inferred from dataset size alone.
DEFAULT_WINDOWS_PER_PASS = 16  # Automatic batch sizing aims for roughly this many optimizer steps per corpus-equivalent pass when the data is small.
HEAVY_REPETITION_PASSES = 20.0  # A numeric override beyond this many equivalent passes requires an explicit opt-in because it is usually a memorization experiment.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ArcNeuron from train.txt")  # Keep every experiment knob discoverable through --help.
    parser.add_argument("--data", default="train.txt")  # Raw natural text remains the only base-training corpus.
    parser.add_argument("--out", default="arcneuron.pt")  # The best held-out checkpoint is written to one predictable path.
    parser.add_argument("--resume", default=None)  # An interrupted compatible run can continue from an existing checkpoint.

    parser.add_argument("--steps", default=AUTO)  # Auto derives optimizer updates from real token exposure; a positive integer still allows deliberate controlled experiments.
    parser.add_argument("--target-passes", type=float, default=DEFAULT_TARGET_PASSES)  # Auto steps expose the optimizer to approximately this many corpus-equivalent tokens.
    parser.add_argument("--batch-size", default=AUTO)  # Auto chooses a small microbatch from corpus size; an integer overrides it when GPU memory is already known.
    parser.add_argument("--grad-accum", type=int, default=1)  # Gradient accumulation grows effective batch size without increasing activation memory.
    parser.add_argument("--context", default=AUTO)  # Auto chooses a power-of-two context that leaves many independent windows in the training corpus.

    parser.add_argument("--vocab-size", type=int, default=8192)  # SentencePiece may return a smaller real vocabulary when a tiny corpus cannot support the requested cap.
    parser.add_argument("--dim", type=int, default=512)  # Hidden width stays an explicit architecture decision rather than being silently changed by the trainer.
    parser.add_argument("--heads", type=int, default=8)  # Query-head count stays explicit because it changes the neural tensor graph.
    parser.add_argument("--kv-heads", type=int, default=2)  # Grouped-query KV-head count also remains an architecture decision.
    parser.add_argument("--ffn-dim", type=int, default=1408)  # SwiGLU width is part of the model, not a data-loader heuristic.
    parser.add_argument("--prelude-layers", type=int, default=1)  # Non-recurrent input depth remains explicitly controlled by the experiment.
    parser.add_argument("--core-layers", type=int, default=2)  # Shared recurrent depth per iteration remains explicitly controlled by the experiment.
    parser.add_argument("--coda-layers", type=int, default=1)  # Non-recurrent output depth remains explicitly controlled by the experiment.
    parser.add_argument("--max-depth", type=int, default=4)  # Training samples recurrence between one and this value so one checkpoint learns multiple compute budgets.

    parser.add_argument("--lr", type=float, default=3e-4)  # AdamW peak learning rate remains easy to override for scaling experiments.
    parser.add_argument("--min-lr", type=float, default=3e-5)  # Cosine decay stops at a nonzero floor instead of collapsing updates completely.
    parser.add_argument("--warmup", default=AUTO)  # Auto warmup scales with the final step budget; an integer still provides exact control.
    parser.add_argument("--weight-decay", type=float, default=0.1)  # Matrix weights receive decoupled AdamW regularization.
    parser.add_argument("--clip", type=float, default=1.0)  # Gradient clipping protects recurrent training from occasional exploding updates.

    parser.add_argument("--val-ratio", type=float, default=0.10)  # Whole blank-line-separated paragraphs are held out so validation is not merely the adjacent tail of one token stream.
    parser.add_argument("--patience", type=int, default=3)  # Stop after this many validation checks without a meaningful improvement.
    parser.add_argument("--min-delta", type=float, default=0.01)  # Tiny validation noise below this amount does not reset early-stopping patience.
    parser.add_argument("--log-every", type=int, default=5)  # Frequent flushed logs keep Colab visibly alive without printing every optimizer update.
    parser.add_argument("--eval-every", default=AUTO)  # Auto evaluates about ten times across the planned run unless an exact cadence is requested.
    parser.add_argument("--eval-batches", type=int, default=8)  # Several random held-out windows make validation less noisy than one arbitrary slice.
    parser.add_argument("--save-every", default=AUTO)  # Auto snapshots around five times across the run while args.out always keeps the best validation state.

    parser.add_argument("--allow-heavy-repetition", action="store_true")  # This explicit switch is required when a numeric step override would recycle a tiny corpus excessively.
    parser.add_argument("--seed", type=int, default=1337)  # Fixed random streams make architecture comparisons easier to reproduce.
    parser.add_argument("--compile", action="store_true")  # torch.compile stays optional because compilation cost may not amortize on tiny experiments.
    return parser.parse_args()  # Returning one namespace keeps all runtime policy visible near the entry point.


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA is preferred automatically while CPU remains a functional correctness path.


def seed_everything(seed: int) -> None:
    random.seed(seed)  # Paragraph splitting and recurrent-depth sampling become reproducible.
    torch.manual_seed(seed)  # Model initialization and random token-window sampling become reproducible.
    if torch.cuda.is_available():  # CUDA keeps a separate random-number stream from the CPU.
        torch.cuda.manual_seed_all(seed)  # Seed every visible GPU in case a runtime exposes more than one device.


def read_text(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")  # Preserve the corpus exactly as UTF-8 natural text.
    if not text.strip():  # An empty corpus can only create meaningless batches and confusing downstream exceptions.
        raise ValueError(f"training corpus is empty: {path}")
    return text  # No semantic cleanup, dictionary expansion, answer labeling, or reasoning annotation happens here.


def split_documents(text: str, val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < val_ratio < 0.5:  # Validation must reserve some data without consuming most of a small training corpus.
        raise ValueError("--val-ratio must be greater than 0 and smaller than 0.5")
    documents = [part.strip() for part in text.split("\n\n") if part.strip()]  # Blank lines define readable natural-text units that can be held out independently.
    if len(documents) < 2:  # One giant paragraph cannot produce a meaningful paragraph-level held-out split.
        raise ValueError("train.txt needs at least two blank-line-separated paragraphs for validation")
    indices = list(range(len(documents)))  # Shuffle indices instead of mutating the user's original text order.
    random.Random(seed).shuffle(indices)  # A local seeded RNG makes the split deterministic without consuming the training-depth RNG stream.
    val_count = max(1, round(len(documents) * val_ratio))  # Always hold out at least one complete paragraph.
    val_indices = set(indices[:val_count])  # Membership checks stay cheap and explicit while rebuilding each split.
    train_documents = [document for index, document in enumerate(documents) if index not in val_indices]  # Only these paragraphs contribute gradients.
    val_documents = [document for index, document in enumerate(documents) if index in val_indices]  # These paragraphs are never sampled by the optimizer.
    return train_documents, val_documents  # Tokenization happens after the split so token counts describe the data each side actually contains.


def encode_documents(tokenizer: ArcTokenizer, documents: list[str]) -> Tensor:
    token_ids: list[int] = []  # All documents become one ordinary causal token stream after each one receives explicit sequence boundaries.
    for document in documents:  # Encode one paragraph at a time so one paragraph ending cannot silently become another paragraph beginning.
        token_ids.extend(tokenizer.encode(document, add_bos=True, add_eos=True))  # The tokenizer only maps text to subword IDs; it contributes no training target beyond the original text.
    return torch.tensor(token_ids, dtype=torch.long)  # Keep the complete token stream on CPU so a tiny corpus does not consume accelerator memory permanently.


def parse_positive_integer_or_auto(value: str, name: str) -> int | None:
    if value.lower() == AUTO:  # None means the caller should derive the value from actual corpus statistics.
        return None
    try:
        parsed = int(value)  # Numeric CLI overrides remain simple and predictable.
    except ValueError as error:
        raise ValueError(f"{name} must be 'auto' or a positive integer") from error
    if parsed <= 0:  # Zero or negative sizes cannot form a valid training schedule.
        raise ValueError(f"{name} must be positive")
    return parsed  # Returning an integer marks an explicit user override that automatic sizing must respect.


def largest_power_of_two_not_above(value: int) -> int:
    if value < 1:  # A power-of-two context cannot be derived from zero available positions.
        raise ValueError("value must be positive")
    return 1 << (value.bit_length() - 1)  # Bit length gives the largest exact power of two no larger than the supplied integer.


def resolve_context(requested: str, train_token_count: int) -> int:
    explicit = parse_positive_integer_or_auto(requested, "--context")  # Explicit contexts must remain untouched because they are part of the model architecture.
    if explicit is not None:
        if explicit >= train_token_count - 1:  # At least one shifted target token must remain beyond every sampled input window.
            raise ValueError("--context is too large for the training token stream")
        return explicit
    usable = max(32, train_token_count // DEFAULT_WINDOWS_PER_PASS)  # Leave roughly sixteen non-overlapping windows in a tiny corpus instead of making one optimizer step consume most of it.
    bounded = min(1024, usable, train_token_count - 2)  # R1 keeps automatic contexts modest and never exceeds the number of available shifted positions.
    if bounded < 32:  # Fewer than about thirty-two positions is too small even for a meaningful language-model smoke experiment.
        raise ValueError("training corpus is too small even for the minimum automatic context; add more natural text")
    return largest_power_of_two_not_above(bounded)  # Power-of-two contexts are convenient for GPU shapes without changing any semantics.


def resolve_batch_size(requested: str, train_token_count: int, context: int) -> int:
    explicit = parse_positive_integer_or_auto(requested, "--batch-size")  # A known GPU budget can always override the conservative automatic choice.
    if explicit is not None:
        return explicit
    windows_in_corpus = max(1, train_token_count // context)  # This rough count measures how many context-sized chunks the unique corpus can hold.
    batch_from_data = max(1, windows_in_corpus // DEFAULT_WINDOWS_PER_PASS)  # Aim for many optimizer updates per corpus-equivalent pass instead of swallowing a tiny corpus in one huge batch.
    return min(DEFAULT_MAX_BATCH_SIZE, batch_from_data)  # Dataset size can safely reduce a batch but cannot prove that an arbitrarily large batch fits accelerator memory.


def resolve_steps(requested: str, target_passes: float, train_token_count: int, tokens_per_step: int, allow_heavy_repetition: bool) -> tuple[int, float]:
    if target_passes <= 0.0:  # An automatic exposure budget must represent a positive amount of training.
        raise ValueError("--target-passes must be positive")
    automatic_steps = max(1, math.ceil(train_token_count * target_passes / tokens_per_step))  # Convert desired corpus-equivalent token exposure directly into optimizer updates.
    explicit = parse_positive_integer_or_auto(requested, "--steps")  # A numeric value remains available for a deliberate experiment.
    steps = automatic_steps if explicit is None else explicit  # Automatic sizing is only used when the user did not request an exact step count.
    equivalent_passes = steps * tokens_per_step / train_token_count  # This is token exposure, not a claim that random overlapping windows form exact dataset epochs.
    if explicit is not None and equivalent_passes > HEAVY_REPETITION_PASSES and not allow_heavy_repetition:  # Stale notebook values such as 3000 steps must not silently turn a tiny corpus into a memorization benchmark.
        raise ValueError(
            f"{steps} steps would expose the optimizer to about {equivalent_passes:.1f} corpus-equivalent passes; "
            f"use --steps auto or add --allow-heavy-repetition only when that repetition is intentional"
        )
    return steps, equivalent_passes  # The same calculated exposure is printed before any expensive GPU work begins.


def resolve_cadence(requested: str, total_steps: int, divisions: int, name: str) -> int:
    explicit = parse_positive_integer_or_auto(requested, name)  # Exact logging/evaluation behavior remains available when an experiment needs it.
    if explicit is not None:
        return explicit
    return max(1, total_steps // divisions)  # Automatic cadence scales with run length instead of assuming every experiment lasts thousands of steps.


def resolve_warmup(requested: str, total_steps: int) -> int:
    explicit = parse_positive_integer_or_auto(requested, "--warmup")  # Exact warmup remains available for controlled comparisons.
    if explicit is not None:
        return min(explicit, max(total_steps - 1, 0))  # Warmup cannot consume more updates than the entire run.
    if total_steps <= 1:  # A one-step smoke run cannot meaningfully warm up before its only update.
        return 0
    return max(1, min(200, math.ceil(total_steps * 0.05)))  # Default warmup uses five percent of the real schedule instead of a fixed 200 steps on a 20-step corpus.


def sample_batch(tokens: Tensor, batch_size: int, context: int, device: torch.device) -> tuple[Tensor, Tensor]:
    highest_start = tokens.numel() - context - 1  # One extra token is required because targets are shifted by one position.
    if highest_start <= 0:  # Fail with a useful explanation before torch.randint receives an invalid interval.
        raise ValueError("token stream is too short for the requested context length")
    starts = torch.randint(0, highest_start, (batch_size,))  # Every sequence begins at an independently sampled point in the allowed corpus stream.
    x = torch.stack([tokens[start : start + context] for start in starts])  # Inputs are contiguous natural-text windows, not engineered feature vectors.
    y = torch.stack([tokens[start + 1 : start + context + 1] for start in starts])  # Targets are exactly the next real token at every position.
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # Transfer only the current batch to the accelerator.


def make_optimizer(model: ArcNeuron, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay = []  # Matrix parameters benefit from decoupled weight decay.
    no_decay = []  # One-dimensional normalization scales should not be shrunk toward zero.
    for parameter in model.parameters():  # The optimizer sees every trainable neural parameter exactly once even though the recurrent core is executed many times.
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)  # Tensor rank cleanly separates matrices/embeddings from scale vectors in this architecture.
    groups = [  # AdamW can apply different decay while sharing learning-rate and moment settings.
        {"params": decay, "weight_decay": weight_decay},  # Regularize learned matrices and the tied embedding/output table.
        {"params": no_decay, "weight_decay": 0.0},  # Leave RMSNorm scale parameters free from shrinkage.
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())  # Fused AdamW is used automatically when CUDA supports the path.


def learning_rate(step: int, total_steps: int, warmup: int, maximum: float, minimum: float) -> float:
    if warmup > 0 and step < warmup:  # The first updates grow gradually instead of immediately applying the full peak learning rate.
        return maximum * (step + 1) / warmup  # Linear warmup reaches the requested maximum at the warmup boundary.
    if total_steps <= warmup:  # A degenerate tiny run simply stays at the maximum after warmup handling.
        return maximum
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))  # Clamp schedule progress so resume or short-run arithmetic cannot leave the cosine interval.
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # Smoothly move from one to zero over the remaining optimizer updates.
    return minimum + cosine * (maximum - minimum)  # Scale the cosine into the requested nonzero learning-rate interval.


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":  # CPU execution prioritizes correctness and simple debugging over mixed-precision speed.
        return nullcontext()  # A no-op context keeps one training loop for both CPU and CUDA.
    return torch.autocast(device_type="cuda", dtype=dtype)  # CUDA matrix operations use BF16 when available and FP16 otherwise.


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, tokens: Tensor, batch_size: int, context: int, depth: int, batches: int, device: torch.device, amp_dtype: torch.dtype) -> float:
    eval_context = min(context, tokens.numel() - 2)  # Held-out data may be smaller than the model context, but the same model can always evaluate a shorter sequence.
    if eval_context < 8:  # A handful of tokens is too little to provide a useful validation signal.
        raise ValueError("validation split is too small; increase train.txt or lower --val-ratio only if more training data remains")
    eval_batch_size = max(1, min(batch_size, max(1, tokens.numel() // eval_context)))  # Validation never needs a microbatch larger than the number of context-sized held-out windows.
    model.eval()  # Disable training-only behavior before measuring held-out language modeling quality.
    losses = []  # Averaging several random windows is less noisy than trusting one arbitrary slice.
    for _ in range(batches):  # Validation remains deliberately lightweight so most accelerator time goes to learning.
        x, y = sample_batch(tokens, eval_batch_size, eval_context, device)  # Draw text that was excluded from optimizer-visible paragraphs.
        with autocast_context(device, amp_dtype):  # Match the numerical path used by training on the same accelerator.
            logits = model(x, depth=depth)  # Validation fixes depth so loss values remain comparable across checkpoints.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))  # Evaluate the exact same next-token objective without backpropagation.
        losses.append(float(loss))  # Move only one scalar to Python instead of storing validation logits or graphs.
    model.train()  # Restore training mode for the caller's next optimizer update.
    return sum(losses) / len(losses)  # Return one human-readable held-out cross-entropy value.


def save_checkpoint(path: str | Path, model: ArcNeuron, optimizer: torch.optim.Optimizer, tokenizer: ArcTokenizer, step: int, best_val_loss: float, training_plan: dict[str, int | float | str]) -> None:
    checkpoint = {  # Everything required to reconstruct the neural training state lives in one portable file.
        "format": 2,  # Format two records the data-aware schedule metadata while keeping model reconstruction straightforward.
        "model_config": asdict(model.config),  # Architecture dimensions travel with the weights instead of being duplicated in generate.py.
        "model_state": model.state_dict(),  # These tensors contain the learned behavior used at inference time.
        "optimizer_state": optimizer.state_dict(),  # Resume can continue AdamW momentum instead of restarting optimization dynamics.
        "tokenizer_model": tokenizer.to_bytes(),  # The exact token-to-ID mapping travels with the neural embedding rows.
        "step": step,  # Resume and experiment reports know which optimizer update produced this state.
        "best_val_loss": best_val_loss,  # Early stopping can continue from the best held-out score already observed.
        "training_plan": training_plan,  # Human-auditable schedule numbers are metadata only and are never consulted by model.forward().
        "python_random_state": random.getstate(),  # Recurrent-depth sampling can continue from the saved Python RNG stream.
        "torch_random_state": torch.get_rng_state(),  # CPU tensor-window sampling can continue from the saved PyTorch RNG stream.
    }
    if torch.cuda.is_available():  # CUDA has its own random state that matters for exact continuation on the same accelerator setup.
        checkpoint["cuda_random_state"] = torch.cuda.get_rng_state_all()  # Save every visible CUDA generator state without adding any inference dependency.
    torch.save(checkpoint, path)  # PyTorch serializes tensor storage efficiently and map_location can relocate it later.


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:  # Invalid timing arithmetic should display uncertainty instead of a nonsense duration.
        return "?"
    seconds = int(seconds)  # Whole seconds are precise enough for a human-facing Colab estimate.
    hours, remainder = divmod(seconds, 3600)  # Separate long runs into hours and a remainder.
    minutes, seconds = divmod(remainder, 60)  # Split the remainder into minutes and seconds.
    if hours:  # Long runs are easier to scan in compact h/m form.
        return f"{hours:d}h{minutes:02d}m"
    if minutes:  # Medium runs display minutes and seconds.
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"  # Very short runs only need seconds.


def main() -> None:
    args = parse_args()  # Read every experiment choice before touching random state, files, or accelerator memory.
    if args.grad_accum <= 0:  # Gradient accumulation must contain at least one microbatch.
        raise ValueError("--grad-accum must be positive")
    if args.max_depth < 1:  # ArcNeuron's recurrent core must execute at least once.
        raise ValueError("--max-depth must be at least 1")
    if args.patience <= 0:  # Early stopping needs at least one unsuccessful validation check before it can stop.
        raise ValueError("--patience must be positive")
    if args.log_every <= 0:  # Live logging cadence cannot be zero.
        raise ValueError("--log-every must be positive")

    seed_everything(args.seed)  # Reproducibility starts before tokenizer-independent model initialization.
    device = choose_device()  # Colab automatically lands on CUDA when a GPU runtime is enabled.
    if device.type == "cuda":  # Modern NVIDIA GPUs benefit from high-precision TensorFloat-style matrix paths where PyTorch allows them.
        torch.set_float32_matmul_precision("high")  # This changes kernel precision policy, not ArcNeuron's learned equations.

    corpus_text = read_text(args.data)  # Read the user's natural-text corpus without injecting hidden instructions or semantic labels.
    train_documents, val_documents = split_documents(corpus_text, args.val_ratio, args.seed)  # Reserve whole paragraphs before any optimizer-visible token stream is built.

    if args.resume:  # Resume must restore the exact tokenizer and architecture that created the checkpoint's learned tensor geometry.
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)  # map_location makes one checkpoint portable across current devices.
        tokenizer = ArcTokenizer.from_bytes(checkpoint["tokenizer_model"])  # Token IDs must retain exactly the same meaning as the saved embedding rows.
        config = ArcNeuronConfig(**checkpoint["model_config"])  # The checkpoint, not today's CLI architecture defaults, defines the resumed neural graph.
        context = config.max_seq_len  # Context length is part of the saved architecture and therefore cannot be silently recalculated during resume.
    else:  # A fresh run first learns only statistical subword compression from the supplied text.
        tokenizer = ArcTokenizer.train(args.data, vocab_size=args.vocab_size)  # SentencePiece sees raw text only and adds no semantic dictionary.
        provisional_train_tokens = encode_documents(tokenizer, train_documents)  # Real token count after tokenization is required before automatic context sizing can be honest.
        context = resolve_context(args.context, provisional_train_tokens.numel())  # Automatic context now comes from what the model will actually see rather than a fixed notebook guess.
        config = ArcNeuronConfig(  # Every value below changes the neural tensor graph and is therefore stored in the checkpoint.
            vocab_size=tokenizer.vocab_size,  # Embedding rows exactly match the tokenizer's actual vocabulary size.
            dim=args.dim,  # Use the explicitly requested hidden width.
            n_heads=args.heads,  # Use the explicitly requested query-head count.
            n_kv_heads=args.kv_heads,  # Use the explicitly requested grouped-query key/value-head count.
            ffn_dim=args.ffn_dim,  # Use the explicitly requested SwiGLU hidden width.
            max_seq_len=context,  # Automatic sizing may choose context from data, but once training begins this is part of the model architecture.
            prelude_layers=args.prelude_layers,  # Configure non-recurrent input depth.
            core_layers=args.core_layers,  # Configure shared recurrent blocks per iteration.
            coda_layers=args.coda_layers,  # Configure non-recurrent output depth.
        )

    train_tokens = encode_documents(tokenizer, train_documents)  # Only these token IDs contribute gradients.
    val_tokens = encode_documents(tokenizer, val_documents)  # These token IDs are reserved for held-out validation and early stopping.
    batch_size = resolve_batch_size(args.batch_size, train_tokens.numel(), context)  # Small corpora automatically use smaller batches so one optimizer step does not swallow most of the unique data.
    tokens_per_step = batch_size * context * args.grad_accum  # This is the actual number of input token positions contributing gradients per optimizer update.
    total_steps, equivalent_passes = resolve_steps(args.steps, args.target_passes, train_tokens.numel(), tokens_per_step, args.allow_heavy_repetition)  # Convert data size into a transparent token-exposure budget.
    warmup_steps = resolve_warmup(args.warmup, total_steps)  # Warmup scales with the real schedule instead of assuming a five-thousand-step run.
    eval_every = resolve_cadence(args.eval_every, total_steps, 10, "--eval-every")  # Automatic validation happens roughly ten times across whatever run length the data produced.
    save_every = resolve_cadence(args.save_every, total_steps, 5, "--save-every")  # Automatic snapshots happen roughly five times without controlling model quality.

    model = ArcNeuron(config).to(device)  # Move the complete neural architecture to the selected accelerator.
    optimizer = make_optimizer(model, args.lr, args.weight_decay)  # AdamW owns every neural parameter and no external reasoning state exists.
    start_step = 0  # A fresh run begins before the first optimizer update.
    best_val_loss = math.inf  # The first validation measurement necessarily becomes the initial best checkpoint.
    stale_evaluations = 0  # Early stopping begins with no failed held-out improvements.

    if args.resume:  # Restore learned tensors and optimizer dynamics only after the matching architecture object exists.
        try:
            model.load_state_dict(checkpoint["model_state"])  # Strict loading intentionally rejects checkpoints from an older incompatible ArcNeuron graph.
        except RuntimeError as error:
            raise RuntimeError("resume checkpoint is incompatible with the current ArcNeuron architecture; train a fresh checkpoint after architecture changes") from error
        optimizer.load_state_dict(checkpoint["optimizer_state"])  # Restore AdamW moments for a true continuation.
        start_step = int(checkpoint.get("step", 0))  # Continue human-facing and schedule step numbers from the saved update.
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))  # Older checkpoints without this metadata simply establish a best value on their next evaluation.
        if "python_random_state" in checkpoint:  # Format-one checkpoints may or may not carry every newer metadata field.
            random.setstate(checkpoint["python_random_state"])  # Resume recurrent-depth sampling from the saved Python stream when available.
        if "torch_random_state" in checkpoint:  # Restore CPU tensor randomness only when the checkpoint provides it.
            torch.set_rng_state(checkpoint["torch_random_state"])  # Random corpus windows then continue from the saved sequence.
        if torch.cuda.is_available() and "cuda_random_state" in checkpoint:  # CUDA exact continuation is possible when its generator state was saved.
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])  # Restore every visible CUDA random generator.

    if start_step >= total_steps:  # Automatic scheduling may correctly conclude that an already-trained checkpoint has consumed its intended token budget.
        raise ValueError(f"checkpoint is already at step {start_step}, not below planned total step count {total_steps}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())  # Count shared recurrent weights once; running the same core four times increases compute but not unique parameters.
    tokens_per_parameter = train_tokens.numel() / parameter_count  # This simple data/model ratio immediately reveals absurd cases such as millions of parameters trained from only a few thousand unique tokens.
    training_plan: dict[str, int | float | str] = {  # Save the numbers printed below so later experiment reports can recover the exact training budget.
        "train_tokens": int(train_tokens.numel()),  # Unique optimizer-visible token stream length after tokenization and paragraph holdout.
        "val_tokens": int(val_tokens.numel()),  # Held-out token stream length used only for validation.
        "parameters": int(parameter_count),  # Exact unique neural parameter count of this instantiated checkpoint.
        "context": int(context),  # Model context used by every training microbatch.
        "batch_size": int(batch_size),  # Resolved microbatch size after optional data-aware automatic sizing.
        "grad_accum": int(args.grad_accum),  # Number of microbatches contributing to one optimizer update.
        "tokens_per_step": int(tokens_per_step),  # Actual token positions contributing gradients per optimizer update.
        "steps": int(total_steps),  # Planned total number of optimizer updates.
        "equivalent_passes": float(equivalent_passes),  # Total planned token exposure divided by unique training-token count.
        "tokens_per_parameter": float(tokens_per_parameter),  # Unique data/model balance reported without pretending it is a universal law.
        "schedule_source": AUTO if args.steps.lower() == AUTO else "explicit",  # Record whether the user or the data-aware default chose the step count.
    }

    print(
        f"device={device} params={parameter_count:,} vocab={config.vocab_size:,} "
        f"train_tokens={train_tokens.numel():,} val_tokens={val_tokens.numel():,} "
        f"tokens/param={tokens_per_parameter:.6f}",
        flush=True,
    )  # The first live line describes the actual data/model balance before spending meaningful GPU time.
    print(
        f"context={context} batch={batch_size} grad_accum={args.grad_accum} "
        f"tokens/step={tokens_per_step:,} steps={total_steps:,} "
        f"exposure={equivalent_passes:.2f} corpus-passes warmup={warmup_steps} "
        f"eval_every={eval_every}",
        flush=True,
    )  # The second line makes every automatically resolved training-budget decision auditable instead of hiding it in a separate planner.
    if train_tokens.numel() < parameter_count:  # Less than one unique token per parameter is an obvious red flag even without assuming one specific scaling law.
        print(
            "CRITICAL DATA WARNING: the model has more parameters than unique training tokens. "
            "This run can test the numerical pipeline or deliberate memorization, but it is not enough evidence for general language or reasoning ability.",
            flush=True,
        )  # Tiny research text is still allowed, but the trainer refuses to pretend that repeating it creates new knowledge.

    raw_model = model  # Checkpointing always saves the ordinary ArcNeuron object even when a compiled wrapper executes forward passes.
    run_model = torch.compile(model) if args.compile else model  # Compilation is only a runtime optimization and does not change learned semantics.
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16  # Prefer BF16 for its wider exponent range, then fall back to FP16 on older GPUs.
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16  # FP16 may need dynamic loss scaling whereas BF16 normally does not.
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # This object becomes a no-op outside FP16 CUDA training.
    raw_model.train()  # Enter training mode explicitly before the first gradient-producing forward pass.
    started = time.perf_counter()  # One monotonic clock feeds throughput and ETA statistics.
    local_steps = 0  # Resume throughput should count only work performed by this process rather than pretending the old runtime never stopped.
    last_val_loss: float | None = None  # Validation output remains absent until the first scheduled held-out check.

    for step in range(start_step, total_steps):  # Each outer iteration performs exactly one optimizer update after optional gradient accumulation.
        current_lr = learning_rate(step, total_steps, warmup_steps, args.lr, args.min_lr)  # Compute this update's warmup/cosine learning rate from the resolved schedule.
        for group in optimizer.param_groups:  # Both decay groups follow the same learning-rate schedule.
            group["lr"] = current_lr  # Update AdamW before any microbatch contributes gradients.
        optimizer.zero_grad(set_to_none=True)  # None gradients save memory writes compared with explicitly filling old gradient tensors with zeros.
        depth = random.randint(1, args.max_depth)  # One recurrent depth for the whole microbatch keeps GPU work aligned and teaches several inference compute budgets.
        accumulated_loss = 0.0  # This Python scalar exists only for human-facing progress logs.

        for _ in range(args.grad_accum):  # Several independent natural-text windows may contribute to one optimizer update.
            x, y = sample_batch(train_tokens, batch_size, context, device)  # Draw an ordinary contiguous next-token batch from optimizer-visible paragraphs.
            with autocast_context(device, amp_dtype):  # Mixed precision accelerates matrix-heavy neural computation without changing the objective.
                logits = run_model(x, depth=depth)  # All learned reasoning computation happens inside ArcNeuron's neural tensor graph.
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))  # Next-token cross entropy is still the one and only training loss.
                scaled_loss = loss / args.grad_accum  # Averaging accumulated gradients keeps update scale stable when grad accumulation changes.
            scaler.scale(scaled_loss).backward()  # Autograd differentiates through every execution of the shared recurrent core.
            accumulated_loss += float(loss.detach())  # Only a detached scalar leaves the graph for logging.

        scaler.unscale_(optimizer)  # Gradient clipping must inspect real magnitudes instead of FP16-scaled values.
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip)  # Cap rare unstable recurrent updates without changing normal gradients.
        scaler.step(optimizer)  # AdamW changes the actual neural weights later used by generate.py.
        scaler.update()  # FP16 loss scaling adapts to numerical range while BF16/CPU paths remain unchanged.
        completed_step = step + 1  # Human-facing progress counts optimizer updates from one instead of zero.
        local_steps += 1  # Throughput statistics count only updates performed since this process started.

        should_eval = completed_step % eval_every == 0 or completed_step == total_steps  # The final requested state is always validated even when cadence does not divide the run evenly.
        if should_eval:  # Held-out validation decides whether training is still learning something beyond the optimizer-visible paragraphs.
            eval_depth = min(args.max_depth, 4)  # A fixed modest recurrence depth keeps validation comparable and cheap across checkpoints.
            last_val_loss = estimate_loss(run_model, val_tokens, batch_size, context, eval_depth, args.eval_batches, device, amp_dtype)  # Measure next-token quality on paragraphs excluded from gradient updates.
            if last_val_loss < best_val_loss - args.min_delta:  # Only a meaningful improvement should replace the best model and reset patience.
                best_val_loss = last_val_loss  # Remember the strongest held-out score observed by this run.
                stale_evaluations = 0  # A real improvement proves that early stopping should give training more time.
                save_checkpoint(args.out, raw_model, optimizer, tokenizer, completed_step, best_val_loss, training_plan)  # args.out always points to the best validation checkpoint rather than merely the latest weights.
                print(f"best checkpoint saved: step={completed_step} val={best_val_loss:.4f} -> {args.out}", flush=True)  # Colab users can see exactly when the durable model improved.
            else:  # Training loss may keep falling even while generalization no longer improves.
                stale_evaluations += 1  # Count consecutive validation checks that failed to beat the best model by min_delta.

        should_log = (
            local_steps == 1
            or completed_step % args.log_every == 0
            or completed_step == total_steps
            or should_eval
        )  # Important events always print immediately while ordinary updates follow the requested compact cadence.
        if should_log:  # Live progress must remain useful without flooding a notebook with one line per token batch.
            elapsed = max(time.perf_counter() - started, 1e-9)  # Protect throughput division from an impossibly tiny timer interval.
            processed = local_steps * tokens_per_step  # Count only token positions processed by this invocation, which keeps resume throughput honest.
            tok_per_second = processed / elapsed  # End-to-end throughput includes recurrent compute, optimizer work, and logging overhead.
            seconds_per_step = elapsed / local_steps  # Average local optimizer-update time provides a stable ETA after the first few steps.
            eta = format_eta(seconds_per_step * (total_steps - completed_step))  # Remaining schedule length becomes a compact human-readable duration.
            mean_loss = accumulated_loss / args.grad_accum  # Average raw CE across all microbatches contributing to this optimizer update.
            val_text = f" val={last_val_loss:.4f}" if last_val_loss is not None else ""  # Before the first validation check the log stays compact instead of printing fake data.
            best_text = f" best={best_val_loss:.4f}" if math.isfinite(best_val_loss) else ""  # The current best held-out state is visible once one has been measured.
            memory_text = ""  # CPU training has no CUDA memory statistic to print.
            if device.type == "cuda":  # CUDA users benefit from seeing actual model/batch allocation while tuning sizes.
                allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert current allocated bytes to human-readable GiB.
                memory_text = f" vram={allocated:.2f}GiB"  # Keep memory on the same compact progress line.
            print(
                f"step={completed_step}/{total_steps} depth={depth} loss={mean_loss:.4f}"
                f"{val_text}{best_text} lr={current_lr:.2e} grad={float(grad_norm):.3f} "
                f"tok/s={tok_per_second:,.0f} eta={eta}{memory_text}",
                flush=True,
            )  # flush=True is essential when the trainer is running inside a Colab subprocess.

        if completed_step % save_every == 0 and completed_step != total_steps:  # Periodic recovery snapshots are independent from the best-validation checkpoint.
            snapshot = Path(args.out).with_name(f"{Path(args.out).stem}.step{completed_step}{Path(args.out).suffix}")  # A distinct filename prevents a worse recovery state from overwriting the best model.
            save_checkpoint(snapshot, raw_model, optimizer, tokenizer, completed_step, best_val_loss, training_plan)  # Snapshot full optimizer/RNG state so an interrupted Colab run can resume.
            print(f"snapshot saved: {snapshot}", flush=True)  # Make ephemeral-runtime recovery files easy to find.

        if should_eval and stale_evaluations >= args.patience:  # Rising validation loss should stop a tiny corpus before it is memorized for hundreds of equivalent passes.
            print(
                f"early stop at step {completed_step}: validation failed to improve for {stale_evaluations} checks; "
                f"best_val={best_val_loss:.4f}",
                flush=True,
            )  # The best checkpoint already lives at args.out, so stopping does not lose the strongest observed state.
            break

    if not Path(args.out).is_file():  # Extremely short or unusual schedules still need one self-contained model file when training finishes.
        fallback_val = last_val_loss if last_val_loss is not None else math.inf  # Preserve whatever held-out information exists without inventing a score.
        save_checkpoint(args.out, raw_model, optimizer, tokenizer, min(total_steps, max(start_step + local_steps, 0)), fallback_val, training_plan)  # The fallback is only used when no scheduled validation ever saved a best checkpoint.
        print(f"final checkpoint saved: {args.out}", flush=True)  # Make the artifact path explicit in terminal and notebook output.


if __name__ == "__main__":  # Importing train.py for inspection must never start an expensive training run automatically.
    main()  # Direct execution begins the complete data-aware raw-text pretraining path.
