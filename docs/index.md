# openjev

One-pass option scoring with a local Gemma 3 4B on Apple silicon via MLX.

Given a context and a list of pre-written options, the model prefills the
context once, expands that KV cache across the option batch, and scores every
option in a single padded forward pass. No decoding. The score is the
log-probability of the option tokens given the context; a softmax over the
option scores gives a probability per option, like `jevlike-predict`.

## Why it is fast

The context is encoded once and its KV cache is reused for every candidate, so
adding options costs one padded forward pass rather than one pass per option.

Measured on an M5 Pro, 64 GB (bf16, mlx-lm), with a 202-token context, 8
options and 242 option tokens in total:

| path | median latency |
|---|---|
| context cached once, options batched | 0.17 s |
| context re-encoded per option, no cache | 0.68 s |

Model load is about 1 s from a warm disk. The first forward pass adds under a
second of warm-up.

## What you can do with it

- **Rank options from the CLI** — [`openjev score`](cli.md#score) and
  [`openjev eval`](cli.md#eval) for accuracy on jevlike-style JSONL.
- **Serve it over HTTP** — [`openjev serve`](cli.md#serve) loads the model once
  and answers `POST /score` and `POST /v1/systemone`, a
  [TypeSafe System One](https://docs.typesafe.ai)-compatible contract.
- **Train a per-task head** — cache frozen-Gemma features and fit jevlike's
  cross-attention head in MLX. See [Training a head](training.md).
- **Watch it play Doom** — the server picks every action from a text
  description of the frame. See the [Doom demo](doom-demo.md).

!!! note "macOS only"
    openjev runs natively on macOS with Metal. There is no container path,
    because Linux containers cannot reach the Apple GPU.

## Quick links

- [Getting started](getting-started.md) — install, download the model, first run.
- [CLI reference](cli.md) — every subcommand and flag.
- [Training a head](training.md) — features, training, evaluation.
- [Doom demo](doom-demo.md) — Doom in the terminal, scored by the server.
- Design notes: [one-pass option scoring](design/one-pass-option-scoring.md)
  and [per-task fine-tuning with Gemma](design/per-task-finetuning-with-gemma.md).
- [API reference](api.md) — the Python package.

## Python in thirty seconds

```python
from openjev import OptionScorer

s = OptionScorer("models/gemma-3-4b-it", batch_size=8)
for r in s.score("The capital of France is", [" Paris", " Berlin"], norm="mean"):
    print(r.option, r.probability, r.logprob_sum, r.n_tokens)
print(s.last_timing)
```
