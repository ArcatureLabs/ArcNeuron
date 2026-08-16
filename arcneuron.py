"""ArcNeuron neural architecture and nothing else.

This file intentionally contains only modules that are part of the neural network
itself.  It has no tokenizer, data loader, optimizer, loss function, checkpoint
helper, generation loop, prompt rule, symbolic rule, dictionary, or task logic.

The architecture is a decoder-only Transformer with three stages:

    token embedding -> prelude -> recurrent core x depth -> coda -> LM head

The recurrent core reuses exactly the same parameters at every iteration.  The
original prelude representation is injected into every recurrent iteration so
that extra reasoning depth does not require the recurrent state to preserve the
input perfectly by itself.
"""

from dataclasses import dataclass  # A small immutable-looking container keeps architectural sizes together.

import torch  # ArcNeuron is a neural network, so all state and computation are tensors.
from torch import Tensor, nn  # Tensor documents shapes; nn provides trainable neural modules.
from torch.nn import functional as F  # Functional SDPA and SiLU map directly to efficient PyTorch kernels.


@dataclass(frozen=True)
class ArcNeuronConfig:
    """All values that change the shape or mathematics of ArcNeuron."""

    vocab_size: int  # Number of tokenizer symbols seen by the embedding and language-model head.
    dim: int = 512  # Width of every token representation inside the network.
    n_heads: int = 8  # Number of query attention heads.
    n_kv_heads: int = 2  # Number of key/value heads; fewer KV heads reduce attention memory and bandwidth.
    ffn_dim: int = 1408  # Hidden width of the gated feed-forward network.
    max_seq_len: int = 1024  # Longest sequence for which rotary positions are precomputed.
    prelude_layers: int = 1  # Non-recurrent Transformer blocks that first encode the token stream.
    core_layers: int = 2  # Transformer blocks reused inside every recurrent reasoning iteration.
    coda_layers: int = 1  # Non-recurrent Transformer blocks that turn the final state into an answer-ready state.
    rope_theta: float = 10_000.0  # Base frequency used by rotary positional embeddings.
    # --- Optional recurrent-depth research knobs (off by default = original ArcNeuron behavior). ---
    # These exist only so each architectural choice can be turned on or off for a controlled ablation
    # without rewriting the math; none of them adds a new reasoning subsystem to the model.
    sandwich_norm: bool = False  # When true, core blocks use post-residual RMSNorm (4 norms/block) so residual magnitude cannot grow across many recurrent iterations.
    emb_scale: bool = False  # When true, scale the token embedding by sqrt(dim) so the recurrent state and the reinjected context enter the mixer at comparable magnitudes (Huginn's balance).
    random_state_init: bool = False  # When true, start the recurrent state from a truncated-normal tensor of variance 0.4 instead of copying the prelude context, so iterative computation is not merely a perturbation of the input.
    out_proj_shrink_init: bool = False  # When true, initialize recurrent residual outputs with std = sqrt(2/(5*dim))/sqrt(2*L) instead of zeros, where L is total effective depth; this is Huginn's identity-friendly alternative to zeroing.
    max_depth_default: int = 4  # Nominal training recurrence used only to size the out-proj shrink factor L; it does not change inference depth, which is still a runtime argument.

    def __post_init__(self) -> None:
        """Reject shapes that cannot form a valid grouped-query Transformer."""

        if self.vocab_size <= 0:  # An embedding table cannot have zero or a negative number of symbols.
            raise ValueError("vocab_size must be positive")
        if self.dim <= 0:  # Every hidden representation needs at least one feature.
            raise ValueError("dim must be positive")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:  # Attention needs at least one query and one KV head.
            raise ValueError("attention head counts must be positive")
        if self.dim % self.n_heads != 0:  # Every query head must receive exactly the same number of features.
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:  # PyTorch GQA requires query heads to be divisible by KV heads.
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2 != 0:  # RoPE rotates feature pairs, so a head width must be even.
            raise ValueError("attention head dimension must be even for RoPE")
        if self.ffn_dim <= 0:  # SwiGLU needs a positive hidden width.
            raise ValueError("ffn_dim must be positive")
        if self.max_seq_len <= 0:  # Position tables need at least one valid sequence position.
            raise ValueError("max_seq_len must be positive")
        if self.prelude_layers < 0 or self.core_layers <= 0 or self.coda_layers < 0:  # The core must exist; outer stages may be empty.
            raise ValueError("core_layers must be positive and outer layer counts cannot be negative")


class RMSNorm(nn.Module):
    """Root-mean-square normalization with one learned scale per feature."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()  # Register this object as a normal PyTorch module.
        self.eps = eps  # A tiny positive value prevents division by zero.
        self.weight = nn.Parameter(torch.ones(dim))  # The network can relearn the useful scale of every feature.

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype  # The normalized result should leave this layer with the caller's dtype.
        x_float = x.float()  # Accumulating the mean square in FP32 is more stable than FP16/BF16 accumulation.
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)  # Compute RMS statistics independently for every token.
        normalized = x_float * torch.rsqrt(mean_square + self.eps)  # Divide by the root mean square without computing a separate sqrt then reciprocal.
        return normalized.to(original_dtype) * self.weight  # Restore the activation dtype and apply the learned feature scales.


class RotaryEmbedding(nn.Module):
    """Precomputed rotary-position frequencies shared by attention calls."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()  # Buffers created below should move automatically with the model.
        pair_index = torch.arange(0, head_dim, 2, dtype=torch.float32)  # Each RoPE angle controls one even/odd feature pair.
        inv_freq = 1.0 / (theta ** (pair_index / head_dim))  # Higher feature pairs rotate at progressively slower frequencies.
        positions = torch.arange(max_seq_len, dtype=torch.float32)  # Every legal token position receives its own phase.
        angles = torch.outer(positions, inv_freq)  # Shape [sequence, head_dim/2] stores the angle for every pair.
        self.register_buffer("cos", angles.cos(), persistent=False)  # Cosines are deterministic and therefore do not belong in checkpoints.
        self.register_buffer("sin", angles.sin(), persistent=False)  # Sines are kept beside cosines for the same reason.

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = q.size(-2)  # Query and key sequences have identical lengths in this causal self-attention implementation.
        if seq_len > self.cos.size(0):  # Running past the configured context would silently produce wrong positions.
            raise ValueError("sequence length exceeds ArcNeuronConfig.max_seq_len")
        cos = self.cos[:seq_len].to(dtype=q.dtype, device=q.device)[None, None, :, :]  # Broadcast one angle table across batch and heads.
        sin = self.sin[:seq_len].to(dtype=q.dtype, device=q.device)[None, None, :, :]  # Match the current activation dtype and device before rotation.
        q_even, q_odd = q[..., 0::2], q[..., 1::2]  # Split query features into the pairs that RoPE rotates together.
        k_even, k_odd = k[..., 0::2], k[..., 1::2]  # Split key features in exactly the same way so dot products keep relative positions.
        q_rotated = torch.stack((q_even * cos - q_odd * sin, q_even * sin + q_odd * cos), dim=-1).flatten(-2)  # Apply a 2D rotation to every query pair.
        k_rotated = torch.stack((k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1).flatten(-2)  # Apply the matching rotation to every key pair.
        return q_rotated, k_rotated  # Attention receives position-aware queries and keys but unchanged values.


class CausalSelfAttention(nn.Module):
    """Grouped-query causal self-attention backed by PyTorch SDPA."""

    def __init__(self, config: ArcNeuronConfig, zero_output: bool) -> None:
        super().__init__()  # All projections below become trainable parameters of this module.
        self.n_heads = config.n_heads  # Query heads determine how many independent attention views are produced.
        self.n_kv_heads = config.n_kv_heads  # Shared KV heads reduce parameter count and inference bandwidth.
        self.head_dim = config.dim // config.n_heads  # Every head owns an equal slice of the model width.
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)  # Map token states into query vectors.
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)  # Map token states into the smaller set of key vectors.
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)  # Map token states into the smaller set of value vectors.
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)  # Merge all attended query heads back into model width.
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)  # RoPE gives attention relative-position information without learned tables.
        if zero_output:  # Recurrent residual branches begin close to identity so repeated application starts numerically gentle.
            nn.init.zeros_(self.o_proj.weight)  # Only the residual output is zeroed; query/key/value projections remain normally initialized.

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape  # Attention operates on [batch, time, feature] activations.
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # Shape queries as [batch, query_head, time, head_dim].
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # Shape keys as [batch, kv_head, time, head_dim].
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # Shape values identically to keys.
        q, k = self.rope(q, k)  # Rotate queries and keys before their dot products are formed.
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, enable_gqa=True)  # Let PyTorch choose an efficient fused causal-attention kernel when available.
        merged = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)  # Put heads beside each other again for the output projection.
        return self.o_proj(merged)  # Return only the residual branch; the caller adds the original activation.


class SwiGLU(nn.Module):
    """Gated feed-forward network used after attention in every block."""

    def __init__(self, config: ArcNeuronConfig, zero_output: bool) -> None:
        super().__init__()  # The three linear maps below are the trainable feed-forward parameters.
        self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)  # Produce values that decide which hidden features should pass.
        self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)  # Produce the candidate hidden features themselves.
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)  # Compress the gated hidden representation back to model width.
        if zero_output:  # The recurrent feed-forward residual also starts as an identity-preserving zero update.
            nn.init.zeros_(self.down_proj.weight)  # Upstream gate/up weights remain random so they can become useful once this output learns away from zero.

    def forward(self, x: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)  # SwiGLU multiplies a smooth learned gate by a learned content projection.
        return self.down_proj(gated)  # Return the residual update in the original model width.


class TransformerBlock(nn.Module):
    """Pre-normalized causal-attention block with a SwiGLU residual path."""

    def __init__(self, config: ArcNeuronConfig, zero_residual_outputs: bool = False) -> None:
        super().__init__()  # Norms, attention, and feed-forward modules become a single Transformer block.
        self.sandwich = config.sandwich_norm  # Sandwich normalization keeps the residual stream bounded across many recurrent iterations by normalizing after each residual addition.
        self.attn_norm = RMSNorm(config.dim)  # Normalize the state before attention so residual scale stays predictable.
        self.attn = CausalSelfAttention(config, zero_output=zero_residual_outputs)  # Compute contextual token interactions.
        self.ffn_norm = RMSNorm(config.dim)  # Normalize again before the nonlinear feed-forward transformation.
        self.ffn = SwiGLU(config, zero_output=zero_residual_outputs)  # Transform features independently at every token position.
        self.attn_out_norm = RMSNorm(config.dim) if self.sandwich else None  # An extra norm after the attention residual re-normalizes the stream that will be iterated.
        self.ffn_out_norm = RMSNorm(config.dim) if self.sandwich else None  # An extra norm after the feed-forward residual re-normalizes the state before the next iteration.

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))  # The first residual path lets attention refine rather than replace the current representation.
        if self.sandwich:  # Normalizing after the residual addition stops residual magnitude from drifting across deep recurrence (Huginn uses this at scale).
            x = self.attn_out_norm(x)
        x = x + self.ffn(self.ffn_norm(x))  # The second residual path lets the MLP refine the attention-updated representation.
        if self.sandwich:  # The matching post-MLP norm keeps the stream that the next iteration will receive on a controlled scale.
            x = self.ffn_out_norm(x)
        return x  # The block keeps exactly the same [batch, time, dim] shape it received.


class RecurrentCore(nn.Module):
    """Shared-weight reasoning core that can be applied an arbitrary number of times."""

    def __init__(self, config: ArcNeuronConfig) -> None:
        super().__init__()  # The mixer, blocks, and final normalization below are the only parameters reused across recurrent iterations.
        self.mix = nn.Linear(config.dim * 2, config.dim, bias=False)  # Fuse the evolving state with the untouched prelude context at every iteration instead of letting either source silently replace the other.
        self.blocks = nn.ModuleList(TransformerBlock(config, zero_residual_outputs=True) for _ in range(config.core_layers))  # Core blocks start near identity so the same residual stack can be repeated without immediately destroying the representation.
        self.output_norm = RMSNorm(config.dim)  # Normalize the state before it is fed into another recurrent iteration so hidden magnitude cannot drift simply because inference requested more depth.

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        x = self.mix(torch.cat((state, context), dim=-1))  # The neural mixer receives both the current latent state and the original encoded input on every recurrent pass.
        for block in self.blocks:  # Every recurrent iteration traverses the exact same shared Transformer blocks in the same order.
            x = block(x)  # These exact weights are reused again when the caller requests another reasoning iteration.
        return self.output_norm(x)  # The next recurrent iteration receives a controlled-scale neural state rather than an activation whose magnitude can grow with depth.


class ArcNeuron(nn.Module):
    """Complete ArcNeuron decoder-only recurrent-depth language model."""

    def __init__(self, config: ArcNeuronConfig) -> None:
        super().__init__()  # Register every architectural component under one trainable model.
        self.config = config  # Keeping shape information with the model makes checkpoint reconstruction exact.
        self.embedding = nn.Embedding(config.vocab_size, config.dim)  # Convert discrete tokenizer IDs into learned continuous vectors.
        self.prelude = nn.ModuleList(TransformerBlock(config) for _ in range(config.prelude_layers))  # Encode the input before recurrent reasoning begins.
        self.core = RecurrentCore(config)  # Reuse one shared reasoning core instead of allocating new parameters for every reasoning step.
        self.coda = nn.ModuleList(TransformerBlock(config) for _ in range(config.coda_layers))  # Refine the final recurrent state before token prediction.
        self.final_norm = RMSNorm(config.dim)  # Put the final hidden state on a stable scale before the vocabulary projection.
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)  # Turn each hidden token state into logits over the tokenizer vocabulary.
        self.lm_head.weight = self.embedding.weight  # Weight tying reduces parameters and keeps input/output token geometry in the same learned space.
        self.apply(self._init_weights)  # Give ordinary linear and embedding weights a small zero-centered initialization.
        self._stabilize_core_initialization()  # Reapply identity-friendly recurrent initialization after the generic initializer above.

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):  # Linear projections dominate Transformer parameters.
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # A conventional small normal initialization works well for compact decoder models.
        elif isinstance(module, nn.Embedding):  # Token embeddings use the same initial scale as linear projections.
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # No pretrained lexical meaning is injected into the embedding table.

    def _stabilize_core_initialization(self) -> None:
        nn.init.zeros_(self.core.mix.weight)  # Clear the generic random mixer initialization so the recurrent path starts from a known and auditable state/context balance.
        with torch.no_grad():  # Initialization writes parameter values directly and therefore must not create an autograd graph.
            identity = torch.eye(self.config.dim, device=self.core.mix.weight.device, dtype=self.core.mix.weight.dtype)  # Build one identity map in the exact dtype and device used by the recurrent mixer.
            self.core.mix.weight[:, : self.config.dim].copy_(identity * 0.5)  # Give the evolving recurrent state half of the initial mixer contribution instead of letting it own the whole path.
            self.core.mix.weight[:, self.config.dim :].copy_(identity * 0.5)  # Give the untouched prelude context the other half so context reinjection is real from the very first optimizer step rather than initially being multiplied by zero.
        if self.config.out_proj_shrink_init:  # Huginn's alternative to zeroing: shrink recurrent residual outputs by 1/sqrt(2L) instead of zeroing them, so gradients flow immediately while the core still starts near identity.
            dim = self.config.dim
            base_std = (2.0 / (5.0 * dim)) ** 0.5  # The Nguyen-Salazar variance baseline used by Huginn's Takase init.
            effective_depth = self.config.prelude_layers + self.config.core_layers * self.config.max_depth_default + self.config.coda_layers  # Total effective depth at the nominal training recurrence; shrinkage scales with it.
            shrink = 1.0 / (2.0 * effective_depth) ** 0.5  # The 1/sqrt(2L) factor keeps repeated untrained passes gentle like zeroing, but lets the branch learn away from zero immediately.
            for block in self.core.blocks:  # Only recurrent residual outputs receive the shrinkage; ordinary prelude/coda blocks keep their normal init.
                nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=base_std * shrink)  # Shrunken attention residual output instead of zeros.
                nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=base_std * shrink)  # Shrunken feed-forward residual output instead of zeros.
        else:  # The original ArcNeuron choice: zero recurrent residual outputs so the untrained core is exactly identity.
            for block in self.core.blocks:  # Only recurrent residual branches need the identity-friendly zero-output initialization.
                nn.init.zeros_(block.attn.o_proj.weight)  # Zero attention residual output makes that branch initially contribute nothing while its output projection can immediately receive gradients.
                nn.init.zeros_(block.ffn.down_proj.weight)  # Zero feed-forward residual output does the same for the nonlinear branch and keeps repeated untrained passes numerically gentle.

    def forward(self, tokens: Tensor, depth: int = 3) -> Tensor:
        if tokens.ndim != 2:  # A language-model batch must be [batch, sequence], not a pre-embedded tensor or a flat list.
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.size(1) > self.config.max_seq_len:  # RoPE was intentionally configured for a finite maximum context.
            raise ValueError("token sequence exceeds ArcNeuronConfig.max_seq_len")
        if depth < 1:  # At least one recurrent pass is part of the ArcNeuron architecture by definition.
            raise ValueError("depth must be at least 1")
        x = self.embedding(tokens)  # Learned embeddings are the only conversion from token IDs into neural activations.
        if self.config.emb_scale:  # Huginn scales the embedding by sqrt(dim) so the reinjected context and the recurrent state enter the mixer at comparable magnitudes.
            x = x * (self.config.dim ** 0.5)
        for block in self.prelude:  # Prelude blocks build a stable contextual representation of the current sequence.
            x = block(x)  # No external rule or feature is injected; everything comes from the learned tensors.
        context = x  # Preserve the prelude representation so every recurrent iteration can revisit the original encoded input.
        if self.config.random_state_init:  # Start the recurrent state from a fixed-variance random tensor instead of copying the prelude output, so iteration is not merely a perturbation of the input.
            state = torch.zeros_like(x)  # Allocate the state on the same device and shape as the prelude output before filling it.
            std = (2.0 / 5.0) ** 0.5  # Huginn's state variance is 2/5, matching the expected hidden-activation scale under its init scheme.
            with torch.no_grad():  # State initialization writes tensors directly and must not build an autograd graph.
                torch.nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)  # A bounded random state gives the core a meaningfully different starting point every recurrent run.
            state = state.to(x.dtype)  # Match the activation dtype so the mixer's two inputs stay on the same numeric path.
        else:  # The original ArcNeuron choice: reasoning starts from the same representation as the context.
            state = context  # Reasoning starts from the same representation rather than from a separately engineered memory structure.
        for _ in range(depth):  # More iterations spend more computation while reusing exactly the same core parameters.
            state = self.core(state, context)  # The evolving hidden state is the model's only recurrent reasoning state.
        for block in self.coda:  # Coda blocks convert the final recurrent representation into a state suited for next-token prediction.
            state = block(state)  # Coda is ordinary neural computation and contains no decoding heuristic.
        state = self.final_norm(state)  # Normalize once before projecting into vocabulary space.
        return self.lm_head(state)  # Return raw logits; training and generation policies intentionally live outside this architecture file.
