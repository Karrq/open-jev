# Getting started

## Requirements

- macOS on Apple silicon (the model runs on the GPU through Metal).
- [uv](https://docs.astral.sh/uv/) for the Python 3.12 environment.
- A Hugging Face login with access to `google/gemma-3-4b-it`, which is a gated
  repository.

!!! warning "Xcode licence"
    `make` on macOS needs the Xcode licence accepted
    (`sudo xcodebuild -license accept`), or use Homebrew's `gmake`.

## Setup

```sh
make setup       # uv sync (arm64 Python 3.12 venv) + download google/gemma-3-4b-it into models/
```

`make setup` runs two targets:

| target | what it does |
|---|---|
| `make venv` | `uv sync --extra torch` into `.venv/` |
| `make model` | `hf download google/gemma-3-4b-it --local-dir models/gemma-3-4b-it` if it is missing |

The optional `torch` extra is installed because `make setup` asks for it; it is
used only for the jevlike comparison and Hugging Face cross-checks, not for
scoring.

Both the model directory and the Hugging Face repo id are overridable:

```sh
make setup MODEL=models/my-gemma HF_REPO=google/gemma-3-4b-it
```

## First run

Rank three options for one context:

```sh
.venv/bin/openjev score --context "The capital of France is" \
    --option " Paris" --option " Berlin" --option " Lyon"
```

Each option prints with its probability, normalised score, raw log-probability
sum, and token count. Note the leading space inside each option: options carry
their own leading whitespace rather than relying on a separator.

Check that the prefix-cached batched path agrees with naive re-encoding, then
benchmark it:

```sh
make check       # verify cached batched scoring against naive re-encoding
make bench       # latency benchmark
```

## Run the server

```sh
make serve       # = .venv/bin/openjev serve --host 127.0.0.1 --port 8000 --model models/gemma-3-4b-it
```

In another terminal:

```sh
make health      # curl /health
make request     # example curl against /score
make systemone   # TypeSafe-style request against /v1/systemone
```

`POST /score` takes a context and a list of options:

```sh
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "context": "Customer: my order arrived broken. Agent:",
  "options": [" I am sorry, I will send a replacement.", " Please read our returns policy."],
  "norm": "mean", "chat": false, "sep": ""
}'
```

The response has `best`, `best_index`, per-option `probability` /
`logprob_sum` / `n_tokens`, and `timing`.

### TypeSafe System One

`POST /v1/systemone` implements the request/response shape documented at
[docs.typesafe.ai](https://docs.typesafe.ai): a `state` (string, object or
array) plus a map of typed `questions`, answered against that state.

```sh
make systemone     # sends examples/systemone-quickstart.json
```

| question `type` | request `criteria` | answer fields |
|---|---|---|
| `choice` | map of option name to description (string, object, array or null); up to 255 options | `choice`, `probabilities` (sum to 1), `confidence` |
| `score` | ordered array of level descriptions | `score` (probability-weighted mean of level index), `probabilities` keyed `"0".."n-1"`, `legend`, `confidence` |
| `noul` | optional `{"true": ..., "false": ...}` | `noul` = probability of yes |

The response is
`{"model", "answers": {id: answer}, "usage": {"input_tokens", "output_tokens"}}`.
Set `OPENJEV_API_KEY` before `make serve` to require
`Authorization: Bearer <key>` on this route.

Each question is rendered to a plain-text prompt (state, instructions, options
or levels, then `Answer:` and a newline), and the option names, level numbers
or `yes`/`no` are scored as continuations in one prefix-shared batched pass per
question. `confidence` is 1 minus the normalised entropy of the distribution;
TypeSafe does not publish its formula, so treat it as an approximation.
`usage.output_tokens` counts the candidate-label tokens that were scored.

!!! note "Zero-shot probabilities are not calibrated"
    On the docs' quick-start request, Gemma 3 4B zero-shot reaches the same
    routing decision as Jev but is markedly over-confident and disagrees on
    judgement calls. Softmaxed next-token likelihoods are not calibrated
    judgements — closing the gap means labelled data and a
    [trained head](training.md).

## Evaluate on your own data

```sh
make eval DATA=data/synthetic/test.jsonl
```

Rows are jevlike-style JSONL: `{"context": ..., "options": [...], "label": 0}`.
See the [CLI reference](cli.md#eval) for the full set of flags.

## Cleaning up

```sh
make clean       # remove runs/ and __pycache__, keeping the venv and the model
```

## Next steps

- [CLI reference](cli.md) for every subcommand.
- [Training a head](training.md) when zero-shot is not accurate or calibrated enough.
- [Design notes](design/one-pass-option-scoring.md) for why this works.
