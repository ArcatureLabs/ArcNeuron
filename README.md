# ArcNeuron
- Open In Colab(Vietnamese):
[![Open In Colab(Vietnamese)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArcatureLabs/ArcNeuron/blob/main/ArcNeuron.ipynb)
- Open In Colab(English):
[![Open In Colab(English)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArcatureLabs/ArcNeuron/blob/main/ArcNeuron-English.ipynb)

ArcNeuron is a small recurrent-depth language model experiment from ArcatureLabs.

The project is built around one narrow idea: **reuse the same neural core several times so the model can spend more compute on a hard token without adding a new set of parameters for every extra step**.

It is still a normal causal language model. It reads text, predicts the next token, and generates text autoregressively. There is no Python dictionary behind the answers, no symbolic rule engine, no retrieval layer, no hidden calculator, no prompt classifier, and no agent runtime inside the model.

The first goal is not to claim that recurrence magically creates reasoning. The goal is to make that claim easy to test and easy to reject if it is wrong.

## Repository

```text
ArcNeuron/
├── arcneuron.py
├── train.py
├── tune.py
├── ArcNeuron.tex
├── generate.py
├── tokenizer.py
├── train.txt
├── tune.txt
└── README.md
```

`arcneuron.py` contains **only the neural architecture**. It has no tokenizer, loss, optimizer, checkpoint helper, generation loop, data loader, prompt rule, or task logic.

`train.py` performs base next-token training from plain `train.txt`.

`tune.py` continues training the exact same weights at a lower learning rate on `tune.txt`. In this project this phase is simply called **tuning**: cầm tay chỉ việc một ít, then keep training the same model. It does not attach LoRA, a reward model, a classifier, or another neural network.

`generate.py` loads one checkpoint and samples the logits produced by ArcNeuron. It contains no knowledge or answer logic.

`tokenizer.py` wraps SentencePiece BPE with byte fallback. BPE is used only to compress text into a shorter reversible sequence. It is not a semantic dictionary.

`train.txt` and `tune.txt` are tiny readable examples included so the pipeline can be tested immediately. They are **not** enough to train a useful general model.

`ArcNeuron.tex` is the paper draft describing the hypothesis, design, training discipline, experiments, and limitations.

## Architecture

The complete neural path is deliberately small:

```text
tokens
  |
  v
embedding
  |
  v
prelude
  |
  +---------------- context
  |                     |
  v                     |
state ------------------+
  |
  v
recurrent core
  |
  v
recurrent core
  |
  v
... repeated N times with the SAME weights
  |
  v
coda
  |
  v
LM head
  |
  v
next-token logits
```

For one recurrent step:

```text
current hidden state + original prelude context
                    |
                    v
                 linear mix
                    |
                    v
              Transformer block
                    |
                    v
              Transformer block
                    |
                    v
               next hidden state
```

The default Colab-scale model uses:

```text
model width      512
query heads      8
KV heads         2
SwiGLU width     1408
prelude blocks   1
shared core      2 blocks
coda blocks      1
context          1024
train depth      1..4
```

With an 8192-token vocabulary this is about **16 million unique parameters**. At recurrent depth 4 the shared core is executed four times, but those executions do not create four copies of its weights.

Every Transformer block is ordinary:

```text
RMSNorm
causal GQA attention + RoPE
residual
RMSNorm
SwiGLU
residual
```

Attention uses `torch.nn.functional.scaled_dot_product_attention`, so PyTorch can use fused CUDA implementations where the current GPU supports them.

The recurrent residual outputs are zero-initialized and the recurrent input mixer begins by copying the previous state. This makes the untrained recurrent path start close to an identity transformation instead of repeatedly destroying its own representation before learning has begun.

## The important boundary

ArcNeuron's intelligence must live in its learned tensors.

Python is allowed to do this:

```text
read text
encode tokens
sample a tensor batch
choose recurrent depth
run matrix operations
calculate cross entropy
backpropagate
update weights
sample logits
```

Python is **not** allowed to do this:

```text
"chi" means "chân"
if mammal then animal
if question is math then use calculator
if prompt asks why then emit a stored explanation
look up the answer in a concept dictionary
```

A trained checkpoint contains the architecture configuration, model `state_dict`, and exact serialized tokenizer. `train.py` and `tune.py` are not needed to answer anything once the model is trained. Another runtime that correctly implements the same tensor graph and tokenizer can run the same learned model without carrying any Python-side knowledge with it.

`depth` is a compute budget, not a reasoning rule. Asking for depth 8 only means running the same learned core eight times.

## Data

Base data is plain natural text.

There is no required format such as:

```text
Q:
A:
Reasoning:
```

The model would merely learn those strings as part of the language anyway.

A useful ArcNeuron corpus should contain normal prose with useful structure: definitions, explanations, counterexamples, uncertainty, correction, code, mathematics, causal arguments, and cases where the correct conclusion is that there is not enough information.

For example, instead of training only this:

```text
Lunari has thick fur.
Lunari lives in cold mountains.
```

natural text can also contain:

```text
Lunari lives in cold mountains and its thick fur helps retain heat. If an
individual loses most of that fur, it may have more difficulty staying warm.
That information alone is not enough to conclude that the animal will die or
that it has a particular disease.
```

The reasoning is still text. No label tells the network which sentence is a fact and which sentence is a conclusion. Next-token prediction has to model the whole relation.

For few-observation concept learning, the training distribution should also contain context-dependent new concepts. An invented word can describe an animal in one independent passage and a material in another. The point is to make a fixed memorized meaning unreliable, so using the current context becomes useful.

Do not confuse **five facts** with **five training tokens**. A concept may have only five pieces of new information while the corpus contains many natural situations that exercise those same pieces of information in different combinations.

## Training

Google Colab already ships PyTorch in normal GPU runtimes. SentencePiece is the only small extra dependency if the runtime does not already have it:

```bash
pip install sentencepiece
```

Then run:

```bash
python train.py
```

A more practical first Colab run on a 16 GB GPU is:

```bash
python train.py \
  --batch-size 8 \
  --grad-accum 2 \
  --steps 5000 \
  --compile
```

If memory is tight, lower `--batch-size` first. If the corpus is still small while debugging, lower `--context` as well.

The trainer performs only ordinary causal LM training:

```python
logits = model(x, depth=depth)
loss = cross_entropy(logits, next_tokens)
loss.backward()
optimizer.step()
```

One recurrent depth is sampled for the whole microbatch. The default range is 1 through 4. This avoids making one sample wait inside a batch while another sample runs many more recurrent iterations.

The checkpoint is saved as `arcneuron.pt` and contains:

```text
model architecture config
model weights
optimizer state
tokenizer model
training step
RNG state
```

Resume with:

```bash
python train.py --resume arcneuron.pt
```

## Tuning

Tuning continues from the base checkpoint:

```bash
python tune.py
```

It uses the same tokenizer, same model, same forward pass, and same next-token cross entropy. The main differences are a smaller corpus, lower learning rate, and optional replay of base text.

The default output is:

```text
arcneuron-tuned.pt
```

A simple run is:

```bash
python tune.py \
  --checkpoint arcneuron.pt \
  --steps 1000 \
  --context 512
```

`tune.txt` should mostly teach behavior: answer the main point, explain when explanation is useful, check unsupported assumptions, revise an error, and say when information is insufficient. It should not become a secret second encyclopedia.

## Generation

After training:

```bash
python generate.py "Mèo là gì?" --checkpoint arcneuron.pt
```

After tuning:

```bash
python generate.py "Mèo là gì?" --checkpoint arcneuron-tuned.pt
```

Recurrent depth is an inference knob:

```bash
python generate.py "Giải thích vì sao..." --depth 1
python generate.py "Giải thích vì sao..." --depth 2
python generate.py "Giải thích vì sao..." --depth 4
python generate.py "Giải thích vì sao..." --depth 8
```

That depth sweep is not just a feature. It is one of the main experiments. If deeper recurrence does not improve tasks that actually need more composition, ArcNeuron has not justified the recurrent core.

`generate.py` intentionally uses a very plain autoregressive loop in R1. It recomputes the visible context for each new token instead of carrying a complicated KV-cache implementation. That is slower than a production runtime, but it keeps the first architecture audit small and makes it much harder for inference-only machinery to accidentally change the model equation. KV caching is a runtime optimization and can be added after the reasoning hypothesis survives basic experiments.

## What to benchmark

Do not judge ArcNeuron only by training loss or by one cherry-picked chat.

The first serious comparison should train a normal Transformer and ArcNeuron with the same tokenizer, data, approximate parameter budget, and optimizer budget. Test at least:

- paraphrases that change the wording but preserve the information
- concepts introduced by only a few observations
- answers that require combining multiple pieces of information
- counterfactual changes to one property
- contradictions
- cases with insufficient information
- natural explanations, not only true/false labels
- mathematics with independently checkable results
- code with tests
- the same ArcNeuron checkpoint at depth 1, 2, 4, and 8

A good recurrent-depth curve would improve until some useful depth and may later degrade from overthinking. A flat curve means the recurrent mechanism has not learned to use extra compute.

## Current status

The repository is a runnable R1 research scaffold, not a pretrained model release.

The included code has been smoke-tested end to end with a tiny configuration:

```text
train.txt
  -> SentencePiece tokenizer
  -> ArcNeuron forward
  -> recurrent backpropagation
  -> AdamW update
  -> checkpoint
  -> continued tuning
  -> tuned checkpoint
  -> autoregressive generation
```

A one- or two-step smoke model will of course emit nonsense. The smoke test proves that the numerical training and inference path works; it does not prove the reasoning hypothesis.

The hypothesis only becomes interesting after a real corpus and controlled baselines are run.

## Paper

The architecture and research plan are described in `ArcNeuron.tex`.

Compile it with:

```bash
xelatex ArcNeuron.tex
xelatex ArcNeuron.tex
```

The paper deliberately makes no benchmark claim that has not been measured yet.
