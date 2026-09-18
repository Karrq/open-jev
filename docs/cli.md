# CLI reference

The package installs one entry point, `openjev`:

```
openjev {score,eval,bench,check,features,train,eval-head,serve} ...
```

In a `make setup` checkout it lives at `.venv/bin/openjev`.

## Common flags

`score`, `eval`, `bench`, `check` and `features` all accept the same scoring
options:

| flag | default | meaning |
|---|---|---|
| `--model MODEL` | `models/gemma-3-4b-it` | local dir or HF repo id |
| `--batch-size N` | `8` | options per forward pass |
| `--norm {mean,sum,pmi}` | `mean` | how option log-probs are normalised |
| `--chat` | off | wrap the context in Gemma's chat template; options score as the reply |
| `--sep SEP` | `""` | string inserted between context and option |

`serve` accepts `--model` and `--batch-size` with the same defaults, but none
of the other scoring flags.

### Normalisation (`--norm`)

| norm | score | use when |
|---|---|---|
| `mean` (default) | sum of option-token log-probs divided by token count | options differ in length |
| `sum` | total log-prob | options are the same length or you want raw likelihood |
| `pmi` | sum minus the option's unconditional log-prob (BOS-only context) | options differ in base-rate plausibility; costs one extra batched pass |

!!! warning "Let options carry their own leading space"
    Passing `--sep " "` makes the tokenizer emit a lone space token, which
    almost never occurs mid-sentence in training data. Every option then
    scores around -20 nats and the ranking becomes noise. Put the leading
    space inside the option string instead (`" Paris"`, not `"Paris"`).

---

## `score`

Score options for one context.

```
openjev score [common flags] --context CONTEXT
              [--option OPTION ...] [--options-file OPTIONS_FILE] [--json]
```

| flag | meaning |
|---|---|
| `--context CONTEXT` | required; the context to score against |
| `--option OPTION` | repeat for each option |
| `--options-file FILE` | text file with one predefined option per line |
| `--json` | emit JSON instead of the table |

```sh
# Rank options for one context (prints probability, score, raw sum, token count)
.venv/bin/openjev score --context "The capital of France is" \
    --option " Paris" --option " Berlin" --option " Lyon"

# Chat template (context as user turn, options scored as the reply) + PMI normalisation
.venv/bin/openjev score --chat --norm pmi --context "..." --option "..." --option "..."

# Predefined options: one per line in a text file, reused for every context
.venv/bin/openjev score --options-file options.txt --context "..."
```

There is also a Makefile wrapper:

```sh
make score CONTEXT="The capital of France is" OPTIONS="--option ' Paris' --option ' Berlin'"
```

## `eval`

Top-k accuracy on jevlike-style JSONL rows,
`{"context": ..., "options": [...], "label": 0}`.

```
openjev eval [common flags] [--limit LIMIT] [--fixed-options FILE] [--verbose] data
```

| argument | meaning |
|---|---|
| `data` | positional; path to the JSONL file |
| `--limit N` | stop after N rows (`0`, the default, means all) |
| `--fixed-options FILE` | text file of options used for every row lacking an `options` field |
| `--verbose` | per-row output |

```sh
# Top-1 / top-3 accuracy
.venv/bin/openjev eval data.jsonl --norm mean

# Rows need only {"context": ...}; add "label" for accuracy
.venv/bin/openjev eval contexts.jsonl --fixed-options options.txt
```

Or through the Makefile, which passes `--model` and `--norm` for you:

```sh
make eval DATA=data/synthetic/test.jsonl
```

## `bench`

Latency of prefix-cached batched scoring versus naive re-encoding, on a
synthetic workload.

```
openjev bench [common flags] [--context-tokens N] [--options N]
              [--option-tokens N] [--repeat N] [--tol TOL]
```

| flag | default |
|---|---|
| `--context-tokens N` | `200` |
| `--options N` | `8` |
| `--option-tokens N` | `30` |
| `--repeat N` | `5` |
| `--tol TOL` | `0.5` |

```sh
.venv/bin/openjev bench --context-tokens 200 --options 8 --option-tokens 30
make bench
```

## `check`

Verify that cached batched scores match naive full-sequence scores. It takes
the same flags as `bench`; `--tol` is the maximum absolute log-probability
difference allowed before the check fails.

```sh
.venv/bin/openjev check
make check
```

## `features`

Cache frozen-Gemma features for a JSONL file into an `.npz`. See
[Training a head](training.md) for how the cache is used.

```
openjev features [common flags] --out OUT [--limit LIMIT] [--contextual] data
```

| argument | meaning |
|---|---|
| `data` | positional; jevlike-style JSONL |
| `--out OUT` | required; destination `.npz` |
| `--limit N` | stop after N rows (`0`, the default, means all) |
| `--contextual` | encode options as continuations of the context (leaks the match into option features) |

```sh
.venv/bin/openjev features data/synthetic/train.jsonl --out runs/feats/train.npz
make features
```

## `train`

Train the cross-attention head on cached features.

```
openjev train --validation VALIDATION [--out OUT] [--rank RANK] [--epochs EPOCHS]
              [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--seed SEED] train
```

| argument | default | meaning |
|---|---|---|
| `train` | — | positional; the training `.npz` |
| `--validation FILE` | required | validation `.npz` |
| `--out OUT` | `runs/head.safetensors` | checkpoint path |
| `--rank RANK` | `256` | head rank |
| `--epochs EPOCHS` | `8` | training epochs |
| `--batch-size BATCH_SIZE` | `64` | examples per step |
| `--learning-rate LR` | `5e-4` | AdamW learning rate |
| `--seed SEED` | `7` | random seed |

```sh
.venv/bin/openjev train runs/feats/train.npz \
    --validation runs/feats/validation.npz --out runs/head.safetensors
make train
```

!!! note
    `train` reads cached features, not the model, so it does not take the
    common scoring flags.

## `eval-head`

Top-k, ECE and the shuffled-context control for a trained head.

```
openjev eval-head checkpoint features
```

Both arguments are positional: the checkpoint written by `train`, and the test
feature `.npz`.

```sh
.venv/bin/openjev eval-head runs/head.safetensors runs/feats/test.npz
make eval-head
```

## `serve`

HTTP server with the model loaded once. Exposes `GET /health`, `POST /score`
and `POST /v1/systemone`.

```
openjev serve [--model MODEL] [--batch-size BATCH_SIZE] [--host HOST] [--port PORT]
```

| flag | default |
|---|---|
| `--model MODEL` | `models/gemma-3-4b-it` |
| `--batch-size BATCH_SIZE` | `8` |
| `--host HOST` | `127.0.0.1` |
| `--port PORT` | `8000` |

```sh
.venv/bin/openjev serve --port 8000
make serve
```

Set `OPENJEV_API_KEY` before starting the server to require
`Authorization: Bearer <key>` on `/v1/systemone`.

---

## Makefile targets

`make help` lists them. In full:

| target | what it does |
|---|---|
| `setup` | create the venv and download the model |
| `venv` | `uv sync --extra torch` |
| `model` | download `$(HF_REPO)` into `$(MODEL)` if missing |
| `serve` | run the HTTP server |
| `health` | curl the running server's health endpoint |
| `request` | example scoring request against the running server |
| `doom` | play Doom in the terminal, the running server picks every action |
| `systemone` | TypeSafe quickstart example against the running server |
| `score` | one-off CLI scoring |
| `check` | verify prefix-cached batched scores match naive re-encoding |
| `bench` | latency: cached and batched vs naive |
| `eval` | zero-shot top-k accuracy on `$(DATA)` |
| `features` | cache frozen-Gemma features for `$(DATA_DIR)/{train,validation,test}.jsonl` |
| `train` | train the attention head into `$(HEAD)` |
| `eval-head` | top-k, ECE and shuffled-context control of `$(HEAD)` |
| `clean` | remove caches and run artefacts, keeping the venv and model |

Overridable variables: `MODEL`, `HF_REPO`, `HOST`, `PORT`, `NORM`, `DATA`,
`API_KEY`, `DATA_DIR`, `FEATS`, `HEAD`, `SEP`.
