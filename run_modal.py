"""Overnight Modal execution harness for ArcNeuron (TEMPORARY, not committed).

This file is NOT part of the permanent ArcNeuron repository.  It only exists to
run the heavy AI work (weights + PyTorch) on Modal GPUs while keeping results on
the local machine and in the repo.  It will be deleted before the final commits.

It mounts the ArcNeuron source into a remote container, attaches a persistent
Modal Volume for data and checkpoints, and forwards work to train.py / tune.py /
generate.py plus a small evaluation script.  Nothing here changes the neural
architecture or the training objective.
"""

import json  # JSON results travel back from the container to the local machine.
import pathlib  # Path filtering keeps the repo mount small by excluding weights and logs.
import subprocess  # The trainer runs as the real train.py entrypoint so the architecture file stays the source of truth.
import sys  # The container's python is used to launch the shipped scripts.
import time  # Wall-clock seconds feed a transparent cost estimate for every run.

import modal  # Modal is the only remote-compute dependency this harness needs.

# One app name keeps every overnight experiment grouped in the Modal dashboard.
app = modal.App("arcneuron-overnight")

# Decide which local files travel into the container image.  add_local_dir's
# `ignore` callback returns True for files that should NOT be uploaded.
def _ignore_repo_file(path: pathlib.Path) -> bool:
    parts = path.parts
    if ".git" in parts:  # Never upload the repository history to a remote container.
        return True
    name = path.name
    if name.endswith(".pt"):  # Checkpoints live on the volume, not in the source mount.
        return True
    if name.endswith(".pdf"):  # The paper PDF is large and irrelevant to training.
        return True
    if name.endswith(".log"):  # Local logs never need to run inside the container.
        return True
    if name in {"RESEARCH_LOG.md", "run_modal.py", "run_modal.pyc", "COST.md"}:  # Local-only scratch.
        return True
    if name in {"vol", "local_results", "scratch"}:  # Local-only output directories.
        return True
    if name.startswith("."):  # Skip dotfiles such as editor swap files.
        return True
    return False  # Source files, notebooks, the paper TeX, and the tiny example corpora all travel.


# A Debian image plus pip packages is enough; the default PyPI torch wheel ships CUDA.
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")  # build tools stay available even though wheels are preferred.
    .pip_install(
        "torch==2.5.1",  # A stable torch release with bundled CUDA kernels for T4/L4/A10.
        "sentencepiece==0.2.0",  # The same tokenizer backend used by ArcNeuron.
        "datasets==3.6.0",  # datasets downloads TinyStories and Vietnamese Wikipedia cleanly.
        "huggingface_hub==0.26.5",  # HF client pinned to avoid pulling a much newer API mid-run.
        "numpy==2.1.3",  # Deterministic numeric helpers for the small evaluation suite.
    )
    .add_local_dir(".", remote_path="/root/arcneuron", copy=True, ignore=_ignore_repo_file)  # Bake the readable source into the image (copy=True so later steps are fine).
)

# One persistent volume holds downloaded corpora, tokenized caches, checkpoints, and logs.
VOLUME = modal.Volume.from_name("arcneuron-overnight", create_if_missing=True)

REPO_PATH = "/root/arcneuron"  # The baked ArcNeuron source tree lives here inside the container.
VOL_PATH = "/vol"  # The volume mount point for data and checkpoints.


def _run_script(script_name: str, argv: list[str], gpu: str, timeout: int) -> dict:
    """Run one ArcNeuron script as the container's real entrypoint and stream logs.

    The container inherits stdout/stderr so progress lines with flush=True appear
    live in the local terminal exactly as if training ran locally on a GPU.
    """

    start = time.perf_counter()  # One monotonic clock feeds an honest cost estimate.
    cmd = [sys.executable, f"{REPO_PATH}/{script_name}"] + [str(a) for a in argv]
    print(f"[run_modal] exec: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=REPO_PATH)  # Inherited fds stream logs to Modal and onward to the user.
    elapsed = time.perf_counter() - start
    price = _gpu_price_per_hour(gpu)
    est_cost = elapsed / 3600.0 * price
    print(f"[run_modal] done in {elapsed:.1f}s returncode={completed.returncode} est_cost=${est_cost:.4f} (gpu={gpu} @ ${price:.2f}/hr)", flush=True)
    VOLUME.commit()  # Persist any checkpoint the script wrote into the volume mount.
    return {
        "script": script_name,
        "argv": argv,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "gpu": gpu,
        "estimated_cost_usd": est_cost,
    }


# Price table mirrors modal.com/pricing so each run prints an honest cost estimate.
_GPU_PRICES = {
    "T4": 0.59,
    "L4": 0.80,
    "A10": 1.10,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100": 2.50,
    "H100": 3.95,
}


def _gpu_price_per_hour(gpu: str) -> float:
    # Fall back to the T4 price if an experimental GPU string is used.
    return _GPU_PRICES.get(gpu, 0.59)


@app.function(image=IMAGE, volumes={VOL_PATH: VOLUME}, cpu=2, memory=8192, timeout=3600)
def prepare_data(ts_articles: int, viwiki_articles: int, seed: int) -> dict:
    """Download TinyStories + Vietnamese Wikipedia and write a mixed training corpus.

    The mixed file keeps TinyStories for small-model English coherence and a
    Vietnamese Wikipedia slice so the final Vietnamese demo is not word salad.
    """

    import random  # Deterministic shuffling keeps the corpus reproducible.

    data_dir = pathlib.Path(VOL_PATH) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset  # Imported inside the container only.

    rng = random.Random(seed)  # One seeded stream controls subsampling and the train/val split.

    # TinyStories is the proven small-model English corpus (CDLA-Sharing 1.0).
    ts = load_dataset("roneneldan/TinyStories", split="train")
    ts_indices = list(range(len(ts)))
    rng.shuffle(ts_indices)
    ts_pick = ts_indices[:ts_articles]
    ts_texts = [ts[i]["text"].strip() for i in ts_pick]
    ts_texts = [t for t in ts_texts if t]  # Drop any empty rows the shuffle happened to surface.

    # Vietnamese Wikipedia is openly licensed (CC-BY-SA 3.0) natural Vietnamese text.
    vi = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
    vi_texts = []
    for idx, row in enumerate(vi):
        if idx >= viwiki_articles:
            break
        text = row.get("text", "").strip()
        if text:
            vi_texts.append(text)
        if (idx + 1) % 5000 == 0:
            print(f"[prepare_data] viwiki scanned {idx + 1} articles, kept {len(vi_texts)}", flush=True)

    print(f"[prepare_data] TinyStories kept {len(ts_texts)} stories; viwiki kept {len(vi_texts)} articles", flush=True)

    # Normalize to plain paragraph-style text: collapse per-article whitespace so each
    # article becomes one document separated by a blank line, which train.py splits on.
    def normalize(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return " ".join(lines)  # One clean paragraph per document matches train.py's splitter.

    ts_docs = [normalize(t) for t in ts_texts]
    vi_docs = [normalize(t) for t in vi_texts]

    all_docs = ts_docs + vi_docs
    rng.shuffle(all_docs)  # Interleave English and Vietnamese so batches see both languages.

    # Hold out whole documents (not adjacent tokens) for a honest validation split.
    val_count = max(1, round(len(all_docs) * 0.05))
    val_docs = all_docs[:val_count]
    train_docs = all_docs[val_count:]

    train_path = data_dir / "train_mix.txt"
    val_path = data_dir / "val_mix.txt"
    train_path.write_text("\n\n".join(train_docs), encoding="utf-8")
    val_path.write_text("\n\n".join(val_docs), encoding="utf-8")

    train_chars = sum(len(d) for d in train_docs)
    val_chars = sum(len(d) for d in val_docs)
    print(f"[prepare_data] wrote {train_path}: {len(train_docs)} docs, {train_chars} chars", flush=True)
    print(f"[prepare_data] wrote {val_path}: {len(val_docs)} docs, {val_chars} chars", flush=True)
    print(f"[prepare_data] estimate ~{train_chars // 4} train tokens, ~{val_chars // 4} val tokens (rough 4 chars/token)", flush=True)

    meta = {
        "tinystories_articles": len(ts_docs),
        "viwiki_articles": len(vi_docs),
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "train_chars": train_chars,
        "val_chars": val_chars,
        "seed": seed,
        "licenses": {
            "tinystories": "CDLA-Sharing-1.0",
            "viwiki": "CC-BY-SA-3.0",
        },
        "sources": {
            "tinystories": "roneneldan/TinyStories (HF)",
            "viwiki": "wikimedia/wikipedia config 20231101.vi (HF)",
        },
    }
    (data_dir / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    VOLUME.commit()
    return meta


@app.function(image=IMAGE, volumes={VOL_PATH: VOLUME}, gpu="T4", timeout=7200)
def train(argv: list[str], gpu: str, timeout: int) -> dict:
    """Forward to train.py with the real data-aware trainer untouched."""
    return _run_script("train.py", argv, gpu, timeout)


@app.function(image=IMAGE, volumes={VOL_PATH: VOLUME}, gpu="T4", timeout=7200)
def tune(argv: list[str], gpu: str, timeout: int) -> dict:
    """Forward to tune.py for continued training on the small behavior corpus."""
    return _run_script("tune.py", argv, gpu, timeout)


@app.function(image=IMAGE, volumes={VOL_PATH: VOLUME}, gpu="T4", timeout=3600)
def generate(argv: list[str], gpu: str, timeout: int) -> dict:
    """Forward to generate.py for one-shot or scripted generation."""
    return _run_script("generate.py", argv, gpu, timeout)


@app.function(image=IMAGE, volumes={VOL_PATH: VOLUME}, gpu="T4", timeout=3600)
def eval_run(checkpoint: str, eval_script: str, gpu: str, timeout: int, out: str = "") -> dict:
    """Run the temporary evaluation script and return its results.json contents."""
    # The eval script writes results.json inside the volume so this returns structured scores.
    start = time.perf_counter()
    cmd = [sys.executable, f"{REPO_PATH}/{eval_script}", "--checkpoint", checkpoint]
    if out:
        cmd += ["--out", out]
    print(f"[run_modal] eval exec: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=REPO_PATH)
    elapsed = time.perf_counter() - start
    price = _gpu_price_per_hour(gpu)
    results = {}
    if out:
        # out may be an absolute volume path (/vol/eval/...) or a relative one (eval/...).
        results_path = pathlib.Path(out) if out.startswith("/") else pathlib.Path(VOL_PATH) / out
    else:
        results_path = pathlib.Path(VOL_PATH) / "eval" / "results.json"
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    VOLUME.commit()
    print(f"[run_modal] eval done in {elapsed:.1f}s returncode={completed.returncode} est_cost=${elapsed/3600*price:.4f}", flush=True)
    return {"returncode": completed.returncode, "elapsed_seconds": elapsed, "estimated_cost_usd": elapsed / 3600 * price, "results": results}


@app.function(image=IMAGE, volumes={VOL_PATH: VOLUME}, timeout=600)
def list_volume(path: str = "/") -> dict:
    """List files on the volume so checkpoints and data can be found from the local machine."""
    root = pathlib.Path(VOL_PATH) / path.lstrip("/")
    listing = []
    for entry in sorted(root.rglob("*")):
        if entry.is_file():
            listing.append({"path": str(entry.relative_to(VOL_PATH)), "size": entry.stat().st_size})
    return listing


@app.function(image=IMAGE, volumes={VOL_PATH: VOLUME}, timeout=600)
def read_volume_file(path: str) -> bytes:
    """Read a small file (results JSON, logs) from the volume."""
    return (pathlib.Path(VOL_PATH) / path.lstrip("/")).read_bytes()


@app.function(image=IMAGE, volumes={VOL_PATH: VOLUME}, timeout=600)
def delete_volume_path(path: str) -> dict:
    """Delete a path on the volume to reclaim space between experiments."""
    target = pathlib.Path(VOL_PATH) / path.lstrip("/")
    if target.is_dir():
        import shutil
        shutil.rmtree(target)
    elif target.is_file():
        target.unlink()
    VOLUME.commit()
    return {"deleted": path}


@app.function(image=IMAGE, volumes={VOL_PATH: VOLUME}, timeout=600)
def copy_volume_file(src: str, dst: str) -> dict:
    """Copy a file on the volume so a checkpoint can be saved under a canonical name."""
    import shutil
    src_path = pathlib.Path(VOL_PATH) / src.lstrip("/")
    dst_path = pathlib.Path(VOL_PATH) / dst.lstrip("/")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    VOLUME.commit()
    return {"copied": f"{src} -> {dst}", "size": dst_path.stat().st_size}


@app.function(image=IMAGE, volumes={VOL_PATH: VOLUME}, cpu=2, memory=8192, timeout=1800)
def tokenizer_probe(data: str, max_chars: int, vocab_sizes: str, coverages: str, seed: int) -> dict:
    """Run the tokenizer measurement script and return its JSON results."""
    start = time.perf_counter()
    argv = [sys.executable, f"{REPO_PATH}/tokenizer_probe.py", "--data", data, "--max-chars", str(max_chars), "--vocab-sizes", vocab_sizes, "--coverages", coverages, "--seed", str(seed)]
    print(f"[run_modal] probe exec: {' '.join(argv)}", flush=True)
    completed = subprocess.run(argv, cwd=REPO_PATH)
    elapsed = time.perf_counter() - start
    results_path = pathlib.Path(VOL_PATH) / "eval" / "tokenizer_probe.json"
    results = {}
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    VOLUME.commit()
    print(f"[run_modal] probe done in {elapsed:.1f}s returncode={completed.returncode}", flush=True)
    return {"returncode": completed.returncode, "elapsed_seconds": elapsed, "results": results}


# ---------------------------------------------------------------------------
# Local entrypoints: these run on the local machine and dispatch to the cloud.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def prepare(
    ts_articles: int = 120000,
    viwiki_articles: int = 20000,
    seed: int = 1337,
):
    """Prepare the mixed training corpus on the volume."""
    meta = prepare_data.remote(ts_articles, viwiki_articles, seed)
    print(json.dumps(meta, indent=2))


@app.local_entrypoint()
def tr(
    data: str = "/vol/data/train_mix.txt",
    out: str = "/vol/ckpt/baseline/arcneuron.pt",
    steps: str = "auto",
    batch_size: str = "auto",
    context: str = "auto",
    vocab_size: int = 4096,
    dim: int = 256,
    heads: int = 8,
    kv_heads: int = 2,
    ffn_dim: int = 704,
    prelude_layers: int = 1,
    core_layers: int = 2,
    coda_layers: int = 1,
    max_depth: int = 4,
    arch: str = "arcneuron",  # Select "baseline" for the mandatory non-recurrent Transformer comparison.
    baseline_layers: int = 4,  # Total stacked blocks for the non-recurrent baseline.
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup: str = "auto",
    grad_accum: int = 1,
    target_passes: float = 4.0,
    weight_decay: float = 0.1,
    clip: float = 1.0,
    val_ratio: float = 0.10,
    seed: int = 1337,
    compile: bool = False,
    allow_heavy_repetition: bool = False,
    character_coverage: float = 1.0,  # SentencePiece coverage; lower it when a multilingual corpus exceeds the requested vocab.
    sandwich_norm: bool = False,  # Recurrent-depth ablation: post-residual RMSNorm in core blocks.
    emb_scale: bool = False,  # Recurrent-depth ablation: scale embedding by sqrt(dim).
    random_state_init: bool = False,  # Recurrent-depth ablation: random initial recurrent state.
    out_proj_shrink_init: bool = False,  # Recurrent-depth ablation: shrunken recurrent residual init.
    max_depth_default: int = 4,  # Nominal recurrence for the out-proj shrink factor L.
    resume: str = "",  # Resume from an existing checkpoint to continue an interrupted run.
    gpu: str = "T4",
    timeout: int = 7200,
):
    """Train ArcNeuron remotely (forwards every useful train.py knob)."""
    argv = [
        "--data", data,
        "--out", out,
        "--steps", steps,
        "--batch-size", batch_size,
        "--context", context,
        "--vocab-size", str(vocab_size),
        "--dim", str(dim),
        "--heads", str(heads),
        "--kv-heads", str(kv_heads),
        "--ffn-dim", str(ffn_dim),
        "--prelude-layers", str(prelude_layers),
        "--core-layers", str(core_layers),
        "--coda-layers", str(coda_layers),
        "--max-depth", str(max_depth),
        "--arch", arch,
        "--baseline-layers", str(baseline_layers),
        "--character-coverage", str(character_coverage),
        "--max-depth-default", str(max_depth_default),
        "--lr", str(lr),
        "--min-lr", str(min_lr),
        "--warmup", warmup,
        "--grad-accum", str(grad_accum),
        "--target-passes", str(target_passes),
        "--weight-decay", str(weight_decay),
        "--clip", str(clip),
        "--val-ratio", str(val_ratio),
        "--seed", str(seed),
    ]
    if compile:
        argv.append("--compile")
    if allow_heavy_repetition:
        argv.append("--allow-heavy-repetition")
    if sandwich_norm:
        argv.append("--sandwich-norm")
    if emb_scale:
        argv.append("--emb-scale")
    if random_state_init:
        argv.append("--random-state-init")
    if out_proj_shrink_init:
        argv.append("--out-proj-shrink-init")
    if resume:
        argv += ["--resume", resume]
    result = train.remote(argv, gpu=gpu, timeout=timeout)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def tn(
    checkpoint: str = "/vol/ckpt/baseline/arcneuron.pt",
    data: str = "/root/arcneuron/tune.txt",
    replay_data: str = "/vol/data/train_mix.txt",
    out: str = "/vol/ckpt/baseline/arcneuron-tuned.pt",
    steps: str = "auto",
    context: str = "auto",
    batch_size: str = "auto",
    max_depth: int = 4,
    lr: float = 5e-5,
    replay_ratio: float = 0.20,
    seed: int = 2026,
    gpu: str = "T4",
    timeout: int = 3600,
):
    """Tune ArcNeuron remotely on the small behavior corpus."""
    argv = [
        "--checkpoint", checkpoint,
        "--data", data,
        "--replay-data", replay_data,
        "--out", out,
        "--steps", steps,
        "--context", context,
        "--batch-size", batch_size,
        "--max-depth", str(max_depth),
        "--lr", str(lr),
        "--replay-ratio", str(replay_ratio),
        "--seed", str(seed),
    ]
    result = tune.remote(argv, gpu=gpu, timeout=timeout)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def gen(
    prompt: str,
    checkpoint: str = "/vol/ckpt/baseline/arcneuron.pt",
    depth: int = 4,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    top_k: int = 40,
    top_p: float = 0.92,
    repetition_penalty: float = 1.08,
    include_prompt: bool = False,
    seed: int = 42,
    gpu: str = "T4",
    timeout: int = 1800,
):
    """Generate text from a remote checkpoint (temperature 0 = deterministic greedy)."""
    argv = [
        prompt,
        "--checkpoint", checkpoint,
        "--depth", str(depth),
        "--max-new-tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--top-k", str(top_k),
        "--top-p", str(top_p),
        "--repetition-penalty", str(repetition_penalty),
        "--seed", str(seed),
    ]
    if include_prompt:
        argv.append("--include-prompt")
    result = generate.remote(argv, gpu=gpu, timeout=timeout)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def ev(
    checkpoint: str = "/vol/ckpt/baseline/arcneuron.pt",
    eval_script: str = "eval_overnight.py",
    gpu: str = "T4",
    timeout: int = 3600,
    out: str = "",
):
    """Run the evaluation suite on a remote checkpoint."""
    result = eval_run.remote(checkpoint=checkpoint, eval_script=eval_script, gpu=gpu, timeout=timeout, out=out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


@app.local_entrypoint()
def eval_all(
    checkpoints: str = "baseline_A,baseline_B,stageC/sandwich_norm,stageC/emb_scale,stageC/random_state,stageC/out_proj_shrink,stageC/combined",
    eval_script: str = "eval_overnight.py",
    gpu: str = "T4",
    timeout: int = 3600,
):
    """Run the evaluation suite on several checkpoints, writing per-checkpoint results.json."""
    names = [c.strip() for c in checkpoints.split(",") if c.strip()]
    for name in names:
        ckpt = f"/vol/ckpt/{name}/arcneuron.pt"
        out = f"/vol/eval/{name.replace('/', '_')}_results.json"
        print(f"===== EVAL {name} ({ckpt}) =====", flush=True)
        result = eval_run.remote(checkpoint=ckpt, eval_script=eval_script, gpu=gpu, timeout=timeout, out=out)
        # Print a compact one-line summary so all checkpoints are comparable in one scroll.
        r = result.get("results", {})
        hol = r.get("held_out_loss", float("nan"))
        dc = r.get("depth_curve", {})
        depth_summary = " ".join(f"d{d}={(dc.get(d, {}) or {}).get('reasoning_composite', float('nan')):.2f}" for d in sorted(dc.keys())) if dc else "(none)"
        demo = (r.get("vietnamese", {}) or {}).get("demo_answer", "")[:80].replace("\n", " ")
        print(f"[summary] {name}: held_out={hol:.3f} | {depth_summary} | demo={demo!r}", flush=True)


@app.local_entrypoint()
def vls(path: str = "/"):
    """List files on the volume."""
    listing = list_volume.remote(path)
    for entry in listing:
        print(f"{entry['size']:>12}  /{entry['path']}")


@app.local_entrypoint()
def vget(path: str, local_dest: str = "local_results"):
    """Download a file from the volume to the local machine."""
    data = read_volume_file.remote(path)
    pathlib.Path(local_dest).mkdir(parents=True, exist_ok=True)
    dest = pathlib.Path(local_dest) / pathlib.Path(path).name
    dest.write_bytes(data)
    print(f"saved {len(data)} bytes to {dest}")


@app.local_entrypoint()
def vrm(path: str):
    """Delete a path on the volume."""
    print(json.dumps(delete_volume_path.remote(path), indent=2))


@app.local_entrypoint()
def vcp(src: str, dst: str):
    """Copy a file on the volume."""
    print(json.dumps(copy_volume_file.remote(src, dst), indent=2))


@app.function(image=IMAGE, cpu=2, memory=4096, timeout=600)
def run_tests() -> dict:
    """Run the test suite inside the container and return the exit code."""
    start = time.perf_counter()
    argv = [sys.executable, f"{REPO_PATH}/test_arcneuron.py"]
    print(f"[run_modal] test exec: {' '.join(argv)}", flush=True)
    completed = subprocess.run(argv, cwd=REPO_PATH, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, flush=True)
    print(f"[run_modal] tests done in {elapsed:.1f}s returncode={completed.returncode}", flush=True)
    return {"returncode": completed.returncode, "elapsed_seconds": elapsed, "stdout": completed.stdout, "stderr": completed.stderr}


@app.local_entrypoint()
def tests():
    """Run the ArcNeuron smoke tests on Modal."""
    result = run_tests.remote()
    print(json.dumps({"returncode": result["returncode"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])


@app.local_entrypoint()
def probe(
    data: str = "/vol/data/train_mix.txt",
    max_chars: int = 2_000_000,
    vocab_sizes: str = "1024,2048,4096,8192",
    coverages: str = "0.9995,0.9999,1.0",
    seed: int = 1337,
):
    """Measure tokenizer vocab sizes and write results to the volume."""
    result = tokenizer_probe.remote(data=data, max_chars=max_chars, vocab_sizes=vocab_sizes, coverages=coverages, seed=seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
