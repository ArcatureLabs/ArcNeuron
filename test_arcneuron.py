"""Smoke tests for ArcNeuron architecture and research knobs.

These tests verify the neural primitives work correctly with and without the
optional recurrent-depth research knobs, that gradients flow through the
recurrent loop at every tested depth, and that checkpoints round-trip.

Run with pytest, or directly:  python test_arcneuron.py
"""

import io  # In-memory checkpoint buffers avoid touching the filesystem.
import pathlib  # Checkpoint round-trip uses a temp file.

try:  # pytest is optional; the file also runs standalone via its __main__ block.
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


# Lightweight shims so the tests run without pytest installed.
if pytest is None:
    class _Parametrize:
        def __init__(self, arg, values):
            self.arg = arg
            self.values = values
        def __call__(self, fn):
            fn._param_values = self.values  # __main__ iterates these
            fn._param_arg = self.arg
            return fn  # return the original function so __main__ can call it per-value

    class _RaisesCtx:
        def __init__(self, exc):
            self.exc = exc
        def __enter__(self):
            return self
        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"expected {self.exc.__name__} but none was raised")
            if not issubclass(et, self.exc):
                return False
            self.value = ev
            return True

    class _Pytest:
        class _Mark:
            parametrize = _Parametrize
        mark = _Mark()
        @staticmethod
        def raises(exc, *args, **kwargs):
            return _RaisesCtx(exc)

    pytest = _Pytest()

import torch  # All tensors and autograd live in PyTorch.

from arcneuron import ArcNeuron, ArcNeuronConfig  # The architecture under test.
from baseline_transformer import BaselineTransformer, BaselineConfig  # The non-recurrent comparison model.
from tokenizer import ArcTokenizer  # Tokenizer smoke test uses the real SentencePiece wrapper.


# A tiny config keeps every test fast while still exercising the full math path.
def _tiny_config(**kw):
    base = dict(
        vocab_size=64,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=128,
        prelude_layers=1,
        core_layers=2,
        coda_layers=1,
    )
    base.update(kw)
    return ArcNeuronConfig(**base)


def _tiny_baseline_config():
    return BaselineConfig(
        vocab_size=64,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=128,
        n_layers=4,
    )


# ---------------------------------------------------------------------------
# Forward pass: shapes and finiteness.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_forward_default_depth(depth):
    """Default config (all knobs off = original ArcNeuron) produces valid logits at every depth."""
    cfg = _tiny_config()
    model = ArcNeuron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x, depth=depth)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_forward_emb_scale():
    """emb_scale multiplies the embedding by sqrt(dim) but output stays valid."""
    cfg = _tiny_config(emb_scale=True)
    model = ArcNeuron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x, depth=4)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_forward_sandwich_norm():
    """sandwich_norm adds post-residual RMSNorms in core blocks."""
    cfg = _tiny_config(sandwich_norm=True)
    model = ArcNeuron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x, depth=4)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_forward_random_state_init():
    """random_state_init starts recurrence from a random tensor instead of context."""
    cfg = _tiny_config(random_state_init=True)
    model = ArcNeuron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x, depth=4)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_forward_out_proj_shrink_init():
    """out_proj_shrink_init uses non-zero init for recurrent residual outputs."""
    cfg = _tiny_config(out_proj_shrink_init=True, max_depth_default=4)
    model = ArcNeuron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x, depth=4)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_forward_all_knobs_combined():
    """All four research knobs on at once still produces valid output."""
    cfg = _tiny_config(
        sandwich_norm=True,
        emb_scale=True,
        random_state_init=True,
        out_proj_shrink_init=True,
        max_depth_default=4,
    )
    model = ArcNeuron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x, depth=8)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Backward pass: gradient flow through the recurrent loop.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_backward_gradient_flow(depth):
    """Gradients flow through every recurrent iteration back to the core parameters."""
    cfg = _tiny_config()
    model = ArcNeuron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x, depth=depth)
    # Use cross-entropy against a random target so the loss is non-trivial and
    # gradients flow through the full LM head -> coda -> recurrent core path.
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size).float(), targets.reshape(-1)
    )
    loss.backward()
    # The core blocks are the shared recurrent parameters; they must receive gradients.
    core_params = list(model.core.parameters())
    assert len(core_params) > 0
    grads = [p.grad for p in core_params if p.grad is not None]
    assert len(grads) == len(core_params), "not all core parameters received gradients"
    for g in grads:
        assert torch.isfinite(g).all()


def test_backward_knobs_combined():
    """Gradients flow with all research knobs enabled."""
    cfg = _tiny_config(
        sandwich_norm=True,
        emb_scale=True,
        random_state_init=True,
        out_proj_shrink_init=True,
    )
    model = ArcNeuron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x, depth=4)
    loss = logits.float().sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# Depth-dependent behavior: more iterations change the output.
# ---------------------------------------------------------------------------

def test_depth_changes_output():
    """Depth > 1 must produce a different output than depth = 1 (recurrence does something)."""
    cfg = _tiny_config()
    model = ArcNeuron(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits_d1 = model(x, depth=1)
        logits_d4 = model(x, depth=4)
    assert not torch.allclose(logits_d1, logits_d4), "depth=4 output is identical to depth=1"


def test_depth_zero_raises():
    """Depth < 1 must raise a ValueError (at least one pass is required by definition)."""
    cfg = _tiny_config()
    model = ArcNeuron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with pytest.raises(ValueError):
        model(x, depth=0)


# ---------------------------------------------------------------------------
# Checkpoint round-trip with arch tag.
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_arcneuron(tmp_path):
    """An ArcNeuron checkpoint saves and reloads with the correct arch tag and weights."""
    cfg = _tiny_config()
    model = ArcNeuron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        original = model(x, depth=4)

    ckpt_path = tmp_path / "test.pt"
    torch.save(
        {
            "arch": "arcneuron",
            "model_config": cfg.__dict__,
            "model_state": model.state_dict(),
        },
        ckpt_path,
    )
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["arch"] == "arcneuron"
    cfg2 = ArcNeuronConfig(**loaded["model_config"])
    model2 = ArcNeuron(cfg2).eval()
    model2.load_state_dict(loaded["model_state"])
    with torch.no_grad():
        replayed = model2(x, depth=4)
    assert torch.allclose(original, replayed, atol=1e-5)


def test_checkpoint_roundtrip_baseline(tmp_path):
    """A BaselineTransformer checkpoint saves and reloads with arch='baseline'."""
    cfg = _tiny_baseline_config()
    model = BaselineTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        original = model(x, depth=1)

    ckpt_path = tmp_path / "test_baseline.pt"
    torch.save(
        {
            "arch": "baseline",
            "model_config": cfg.__dict__,
            "model_state": model.state_dict(),
        },
        ckpt_path,
    )
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["arch"] == "baseline"
    cfg2 = BaselineConfig(**loaded["model_config"])
    model2 = BaselineTransformer(cfg2).eval()
    model2.load_state_dict(loaded["model_state"])
    with torch.no_grad():
        replayed = model2(x, depth=1)
    assert torch.allclose(original, replayed, atol=1e-5)


def test_baseline_ignores_depth():
    """BaselineTransformer must return the same output regardless of the depth argument."""
    cfg = _tiny_baseline_config()
    model = BaselineTransformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        d1 = model(x, depth=1)
        d4 = model(x, depth=4)
        d8 = model(x, depth=8)
    assert torch.allclose(d1, d4, atol=1e-5)
    assert torch.allclose(d1, d8, atol=1e-5)


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------

def test_config_rejects_bad_dim():
    """dim not divisible by n_heads must raise."""
    with pytest.raises(ValueError):
        ArcNeuronConfig(vocab_size=64, dim=65, n_heads=8, n_kv_heads=2, ffn_dim=128)


def test_config_rejects_zero_core():
    """core_layers=0 must raise (the core must exist)."""
    with pytest.raises(ValueError):
        ArcNeuronConfig(vocab_size=64, dim=64, n_heads=4, n_kv_heads=2, ffn_dim=128, core_layers=0)


# ---------------------------------------------------------------------------
# Tokenizer smoke test (requires sentencepiece).
# ---------------------------------------------------------------------------

def test_tokenizer_character_coverage(tmp_path):
    """ArcTokenizer.train accepts a character_coverage argument and round-trips text."""
    text = "The quick brown fox. Con mèo den. 12345."
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    tok = ArcTokenizer.train(
        str(corpus),
        vocab_size=512,
        character_coverage=0.9999,
    )
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert len(ids) > 0
    decoded = tok.decode(ids)
    # BPE with byte fallback should recover most of the original text.
    assert "fox" in decoded or "quick" in decoded


if __name__ == "__main__":
    # Allow running without pytest: execute every test_* function, expanding
    # parametrized cases stored by the shim's _Parametrize decorator.
    import inspect
    import tempfile

    passed = 0
    failed = 0
    for name in sorted(globals()):
        if not name.startswith("test_"):
            continue
        fn = globals()[name]
        sig = inspect.signature(fn)
        has_tmp = "tmp_path" in sig.parameters

        # Build the list of argument-tuples to call the test with.
        cases = []
        if hasattr(fn, "_param_values"):
            for v in fn._param_values:
                cases.append((v,) if not isinstance(v, tuple) else v)
        else:
            cases.append(())

        for case in cases:
            label = name if not case else f"{name}[{','.join(str(c) for c in case)}]"
            try:
                if has_tmp:
                    with tempfile.TemporaryDirectory() as td:
                        fn(pathlib.Path(td), *case)
                else:
                    fn(*case)
                print(f"  PASS {label}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {label}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
