"""Generate text from a trained ArcNeuron checkpoint.

This file contains only numerical decoding policy: load the checkpoint, run the
neural network, sample its next-token distribution, and feed the sampled token
back as the next input.  It contains no factual database, reasoning rule,
prompt classifier, tool call, answer template, or task-specific branch.
"""

import argparse  # A tiny CLI makes generation convenient in both Colab cells and normal terminals.
from pathlib import Path  # Checkpoint existence is checked with an explicit platform-independent path.

import torch  # ArcNeuron inference is tensor computation executed by PyTorch.
from torch.nn import functional as F  # Softmax turns the model's learned logits into a sampling distribution.

from arcneuron import ArcNeuron, ArcNeuronConfig  # Reconstruct the exact neural architecture stored by the checkpoint.
from tokenizer import ArcTokenizer  # Decode and encode with the exact tokenizer stored beside those weights.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with ArcNeuron")  # Keep all decoding controls discoverable through --help.
    parser.add_argument("prompt", nargs="?", default=None)  # A positional prompt makes one-shot terminal generation pleasantly short.
    parser.add_argument("--checkpoint", default="arcneuron-tuned.pt")  # Prefer the tuned model while allowing the base checkpoint explicitly.
    parser.add_argument("--depth", type=int, default=4)  # Recurrent depth is the direct test-time compute knob of ArcNeuron.
    parser.add_argument("--max-new-tokens", type=int, default=256)  # Hard output limits prevent an unfinished small model from generating forever.
    parser.add_argument("--temperature", type=float, default=0.8)  # Values below one sharpen the neural distribution without changing model weights.
    parser.add_argument("--top-k", type=int, default=50)  # Optional top-k sampling removes extremely unlikely tails from each next-token draw.
    parser.add_argument("--seed", type=int, default=42)  # Deterministic sampling is useful when comparing recurrent depths on the same prompt.
    return parser.parse_args()  # Generation behavior is now fully visible in one plain namespace.


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA is used automatically when a Colab GPU runtime is active.


def load_model(path: str | Path, device: torch.device) -> tuple[ArcNeuron, ArcTokenizer]:
    checkpoint_path = Path(path)  # Normalize the user-supplied checkpoint path before opening it.
    if not checkpoint_path.is_file():  # A clear early error is better than a long torch.load traceback for a mistyped path.
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # Load learned tensors directly onto the active inference device.
    config = ArcNeuronConfig(**checkpoint["model_config"])  # Exact architectural sizes come from training, not from duplicated constants here.
    model = ArcNeuron(config).to(device)  # Build only the neural graph described by the saved configuration.
    model.load_state_dict(checkpoint["model_state"])  # Every learned behavior is restored from neural weights.
    model.eval()  # ArcNeuron R1 has no dropout, but eval mode is still the correct inference contract.
    tokenizer = ArcTokenizer.from_bytes(checkpoint["tokenizer_model"])  # Token IDs must match the embedding rows learned by this exact checkpoint.
    return model, tokenizer  # Generation now has only a neural model and a reversible text codec.


def sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature < 0.0:  # Negative temperatures have no meaningful probabilistic interpretation.
        raise ValueError("temperature cannot be negative")
    if temperature == 0.0:  # Zero temperature is defined here as deterministic greedy decoding.
        return int(torch.argmax(logits).item())  # Choose exactly the most probable token predicted by the neural model.
    scaled = logits / temperature  # Temperature rescales confidence while preserving the model's ranking of token logits.
    if top_k > 0 and top_k < scaled.numel():  # Keep all tokens when top_k is zero or larger than the whole vocabulary.
        values, indices = torch.topk(scaled, top_k)  # Extract the strongest learned candidates for this one decoding step.
        probabilities = F.softmax(values, dim=-1)  # Convert those candidate logits into a normalized categorical distribution.
        chosen = torch.multinomial(probabilities, num_samples=1)  # Randomness samples one candidate according to the model's own probabilities.
        return int(indices[chosen].item())  # Map the sampled position back to the original vocabulary ID.
    probabilities = F.softmax(scaled, dim=-1)  # Without top-k, normalize the complete model vocabulary.
    return int(torch.multinomial(probabilities, num_samples=1).item())  # Draw one next token directly from ArcNeuron's distribution.


@torch.inference_mode()
def generate(model: ArcNeuron, tokenizer: ArcTokenizer, prompt: str, depth: int, max_new_tokens: int, temperature: float, top_k: int, device: torch.device) -> str:
    if depth < 1:  # ArcNeuron's recurrent core must execute at least once.
        raise ValueError("depth must be at least 1")
    if max_new_tokens < 0:  # A negative loop count would be a caller mistake rather than a useful decoding mode.
        raise ValueError("max_new_tokens cannot be negative")
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)  # The user prompt is encoded with no hidden instruction or semantic preprocessing.
    generated = list(token_ids)  # Keep the complete autoregressive history because each new token conditions on all previous tokens in context.

    for _ in range(max_new_tokens):  # One iteration asks the neural model for exactly one additional token.
        visible = generated[-model.config.max_seq_len :]  # Crop only when history exceeds the architecture's declared context window.
        x = torch.tensor([visible], dtype=torch.long, device=device)  # Shape one integer token sequence into a batch of size one.
        logits = model(x, depth=depth)  # All language understanding and reasoning computation happens inside ArcNeuron's learned neurons.
        next_id = sample_next(logits[0, -1], temperature, top_k)  # Sampling policy chooses from the model's final next-token distribution.
        generated.append(next_id)  # Autoregressive decoding feeds the chosen token back into the next neural forward pass.
        if next_id == tokenizer.eos_id:  # EOS is the only learned sequence-boundary condition used to stop early.
            break  # No domain-specific answer rule participates in stopping.

    return tokenizer.decode(generated)  # Decode the complete prompt and continuation exactly through the checkpoint tokenizer.


def main() -> None:
    args = parse_args()  # Read checkpoint and sampling controls once at process start.
    torch.manual_seed(args.seed)  # Make multinomial draws reproducible for fair depth or checkpoint comparisons.
    if torch.cuda.is_available():  # CUDA has a separate RNG stream used by GPU multinomial sampling.
        torch.cuda.manual_seed_all(args.seed)  # Seed every visible GPU with the same requested generation seed.
    device = choose_device()  # Use the current Colab GPU automatically when present.
    checkpoint = args.checkpoint  # Keep the requested path visible before fallback handling.
    if not Path(checkpoint).is_file() and checkpoint == "arcneuron-tuned.pt" and Path("arcneuron.pt").is_file():  # Fresh users often have only a base checkpoint before running tune.py.
        checkpoint = "arcneuron.pt"  # Fall back only by filename existence, never by prompt content or task type.
    model, tokenizer = load_model(checkpoint, device)  # Restore the self-contained neural checkpoint.

    if args.prompt is not None:  # A supplied positional prompt runs one non-interactive generation and exits.
        print(generate(model, tokenizer, args.prompt, args.depth, args.max_new_tokens, args.temperature, args.top_k, device))  # Print exactly what the model and tokenizer produced.
        return  # Do not enter interactive input after satisfying the one-shot request.

    print(f"ArcNeuron loaded from {checkpoint} on {device}. Empty input exits.")  # Interactive mode shows only runtime state, not any synthetic assistant persona.
    while True:  # Reuse the same loaded neural weights for multiple manual experiments.
        prompt = input("\n> ")  # The terminal passes raw user text straight to the tokenizer.
        if not prompt:  # An empty line is a simple local-interface exit convention.
            break  # Exit without sending any magic command token through the model.
        result = generate(model, tokenizer, prompt, args.depth, args.max_new_tokens, args.temperature, args.top_k, device)  # Run the same generic neural continuation path for every prompt.
        print(result)  # Show the complete decoded prompt plus continuation so token-boundary spacing remains faithful.


if __name__ == "__main__":  # Importing helpers for experiments must never start an interactive terminal loop.
    main()  # Direct execution provides either one-shot or interactive generation.
