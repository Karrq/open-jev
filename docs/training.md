# Training a head

Zero-shot scoring is a good default, but softmaxed next-token likelihoods are
not calibrated judgements. When you have labelled rows, you can freeze Gemma
and train a small cross-attention head on top of its features — jevlike's
Route A, ported to MLX.

The background is in
[Per-task fine-tuning with Gemma](design/per-task-finetuning-with-gemma.md).

## The three steps

```sh
make features    # one prefix-shared pass per example, cached to runs/feats/{train,validation,test}.npz
make train       # jevlike's cross-attention head in MLX, listwise cross-entropy -> runs/head.safetensors
make eval-head   # top-1/top-3, ECE, and the shuffled-context control on the test split
```

`make features` expects `$(DATA_DIR)/{train,validation,test}.jsonl`
(`data/synthetic` by default) and writes one `.npz` per split into `$(FEATS)`
(`runs/feats`). `make train` reads the train and validation caches and writes
`$(HEAD)` (`runs/head.safetensors`). `make eval-head` scores `$(HEAD)` against
the test cache.

## 1. Cache the features

```sh
.venv/bin/openjev features DATA --out F.npz
```

This stops Gemma before the LM head and keeps two things per example: every
context token's final hidden state, and the masked-mean of each option's
tokens (float16).

Options are encoded on their own, as in jevlike, so the head has to do the
matching itself.

!!! warning "`--contextual` invalidates the control"
    `--contextual` encodes options as continuations of the context instead.
    The features are stronger, but the match leaks into the option vectors and
    the shuffled-context control below stops meaning anything.

## 2. Train the head

```sh
.venv/bin/openjev train runs/feats/train.npz \
    --validation runs/feats/validation.npz --out runs/head.safetensors
```

The defaults are AdamW at 5e-4, weight decay 1e-4, gradient clipping at 1.0, 8
epochs, batch 64, keeping the best validation epoch. The objective is listwise
cross-entropy over each row's options.

!!! note "Why 5e-4"
    2e-3 diverges on Gemma features, whose norms are around 115.

The checkpoint is `head.safetensors` plus a `head.json` recording the rank,
hidden size and training config.

## 3. Evaluate, and check the control

```sh
.venv/bin/openjev eval-head runs/head.safetensors runs/feats/test.npz
```

This reports what `jevlike-eval` reports: top-1, top-3, 10-bin expected
calibration error, and the same metrics with every example paired with another
example's context.

The shuffled-context control is the important one. If the shuffled number does
not collapse toward chance, the head is reading option priors rather than the
state, and the headline accuracy is not measuring what you think it is.

## Reference result

On the synthetic split (2000 train / 400 validation / 400 test, 2026-09-17).
Feature extraction ran at 77 ms per example; head training took about 15 s for
8 epochs.

| | top-1 | top-3 | ECE |
|---|---|---|---|
| head on frozen Gemma features, test | 0.970 | 1.000 | 0.027 |
| same head, shuffled-context control | 0.258 | 0.698 | 0.724 |
| jevlike tiny byte encoder, test | 0.998 | 1.000 | 0.008 |
| Gemma zero-shot, `--norm sum`, test | 1.000 | 1.000 | n/a |

The control sits at chance (about 4.5 options per row), so the head really does
match options to the state, and its ECE of 0.03 is what a calibrated
`confidence` looks like.

## Zero-shot or trained?

The synthetic task above is easy for both, so it does not settle the question.
The real validation is your own labelled rows: run
[`openjev eval`](cli.md#eval) on them zero-shot, and compare against a
`jevlike-train` / `jevlike-eval` run on the same split.

- If Gemma zero-shot is close to the trained head, Route B (zero-shot
  likelihood scoring) is enough.
- If not, train a head with `make features && make train && make eval-head`.

## Validating against jevlike

`jevlike` can be installed into the same venv
(`uv pip install -e ../../vinnylarouge/jevlike`), so both CLIs read the same
JSONL. Generate its synthetic menu set, then score it both ways:

```sh
.venv/bin/jevlike-data synthetic --output data/synthetic
.venv/bin/openjev eval data/synthetic/test.jsonl --norm sum --sep $'\nChoice: '
.venv/bin/jevlike-train data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
    --output runs/synthetic-tiny.pt --device mps
.venv/bin/jevlike-eval runs/synthetic-tiny.pt data/synthetic/test.jsonl --device mps
```

Results on the 400-row synthetic test set (2026-09-16):

| scorer | training | top-1 | top-3 | median latency / example |
|---|---|---|---|---|
| openjev, Gemma 3 4B zero-shot, `--norm sum` | none | 1.000 | 1.000 | 0.086 s |
| openjev, `--norm mean` | none | 0.988 | 1.000 | 0.086 s |
| openjev, `--norm pmi` | none | 0.988 | 1.000 | 0.153 s |
| jevlike tiny byte encoder + head | 2000 rows, 8 epochs | 0.998 | 1.000 | well under 10 ms |
