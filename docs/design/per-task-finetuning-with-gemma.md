# Per-task fine-tuning with Gemma: what jevlike does, what openjev must add

Written 2026-09-16. Companion to [one-pass-option-scoring.md](one-pass-option-scoring.md).

Today openjev answers System One questions zero-shot: it renders a prompt and reads Gemma's
next-token likelihoods. That gives the right decision often but meaningless probabilities
(see the README comparison: 1.00 where Jev says 0.84). Turning it into a per-task, calibrated
scorer means training something on labelled examples. This document lists exactly what has to
change, task by task, and what can be copied from `jevlike`.

---

## 1. What jevlike trains, and what it does not

| Piece | jevlike (PyTorch) | Status in openjev (MLX) |
|---|---|---|
| Encoder | Frozen `AutoModel` (default Qwen 0.5B) or a tiny byte encoder | Gemma 3 4B via mlx-lm, used only through its LM head |
| Features | Context: all token hidden states. Option: mean-pooled hidden states | Not extracted; only logits are read |
| Head | One cross-attention layer, width `--rank`: option vector queries context tokens, dot product gives a logit per option | None |
| Loss | Listwise softmax cross-entropy over the options of one example | None |
| Data | JSONL `{"context", "options", "label"}` | Same format accepted by `openjev eval` |
| Calibration | Falls out of cross-entropy; measured as ECE in `jevlike-eval` | None |
| Caching | None: encoder re-run every epoch | Prefix-cached inference only |
| Checkpoint | Head weights + encoder name | n/a |

Everything below is the delta.

## 2. Feature extraction from Gemma (new)

Add a `hidden_states=True` path to `OptionScorer` that returns the final-layer hidden state
for every token instead of logits. In mlx-lm this is `model.model(tokens, cache=cache)`
before the LM head; shape `(batch, tokens, 2560)` for Gemma 3 4B.

- Reuse the prefix-shared batched pass: context once, options batched. The features come
  from the same forward pass that scoring already does, so extraction is not slower than
  inference.
- Pool options by masked mean, as jevlike does. Keep all context tokens.
- **Encode options on their own, not as continuations of the context.** If option tokens
  attend to the context (prefix-shared cache), each pooled option vector already contains the
  match: the head reaches 100% in one epoch and the shuffled-context control also reads 100%,
  so nothing can be verified. Standalone options cost the same and keep the control honest.
  (Implemented: `openjev features`, with `--contextual` as the leaky alternative.)
- **Cache to disk** as `.npz` per example: `context_h (Lc, 2560)`, `options_h (N, 2560)`,
  `label`. One pass over the dataset, then head training never touches Gemma again.
  At about 100 ms per example, 5,000 examples take under 10 minutes.
- Which layer: start with the last. If the head under-performs, try a middle layer
  (roughly two thirds of the depth); intermediate layers often carry better task features
  than the final, LM-tuned one.
- Decide `-it` vs `-pt` here, once. Base weights usually give softer, more general features.

## 3. The head (port from jevlike, ~60 lines of MLX)

Same architecture, in `mlx.nn`:

```
LayerNorm(context), LayerNorm(options)           # fp32
q = Linear(2560 -> rank)(options)                # one query per option
k, v = Linear(2560 -> rank)(context tokens)
attn = softmax(q @ k.T / sqrt(rank), masked by context padding)
ctx_vec = attn @ v                               # (N, rank), one per option
logit_i = (q_i * ctx_vec_i).sum() / sqrt(rank)   # (N,)
```

`rank` 256 worked for Qwen in jevlike; start there for Gemma. Padded option slots get
`-inf`. Trainable parameters: about 2 M, so it fits in any memory and trains in seconds per
epoch on cached features.

## 4. Per-task targets: one head per question type

The System One API has three question types. Each needs its own target and loss, but they
share the encoder features and the head architecture. Train **one head per task** (or per
question id when a task has its own labelled set); the encoder pass is shared.

| Question type | Training example | Head output | Loss | Answer fields |
|---|---|---|---|---|
| `choice` | state + option names/descriptions, `label` = index | one logit per option | listwise cross-entropy | `choice` = argmax, `probabilities` = softmax, `confidence` from the distribution |
| `score` | state + ordered level texts, `label` = level index | one logit per level | cross-entropy over levels (ordinal variant: CORAL or squared error on the expected index) | `score` = sum(i * p_i), `probabilities`, `legend` |
| `noul` | state + true/false descriptions, `label` in {0,1} | one logit for "yes" | binary cross-entropy | `noul` = sigmoid |

Notes per type:

- **choice**: this is exactly jevlike's setting. Options are the rendered option name plus
  description, so the head sees the same text the zero-shot prompt shows.
- **score**: levels are options too, so the same head applies. Plain cross-entropy trains
  well; if you need the score to respect ordering, add a small penalty on
  `|expected_index - label|`.
- **noul**: the two options are the `true` and `false` descriptions (or the literals "yes"
  and "no" when no criteria are given). Binary cross-entropy on the "yes" logit.

Rendering must be **identical** at train and serve time. Factor the prompt/option renderers
in `systemone.py` so the feature extractor and the endpoint call the same functions.

## 5. Training loop (port `train.py`, drop the encoder)

- AdamW, weight decay 1e-4, grad-norm clip 1.0, 8 epochs, batch 64, as in jevlike, but
  **lr 5e-4, not 2e-3**: Gemma's final hidden states have norms around 115 and the higher
  rate diverges on some seeds (loss spikes to about 75 in epoch 1, val top-1 stuck at chance).
- Batches are padded feature tensors from the cache, not text.
- Validation each epoch; keep the best checkpoint.
- Checkpoint: head weights as `.safetensors` plus a JSON config with encoder name, layer,
  rank, task type, and the renderer version. The encoder is not stored.

## 6. Evaluation (port `eval.py`)

Report what jevlike reports so results are comparable:

- top-1 and top-3 accuracy;
- **shuffled-context control**: score each option set against a different example's
  context. If accuracy does not drop, the head is reading option priors, not the state;
- **ECE** (10-bin expected calibration error) so `confidence` and `probabilities` mean
  something. This is the number that separates a trained head from zero-shot softmax.

Add these to `openjev eval` so zero-shot and trained heads are compared with one command.

## 7. Serving

`/v1/systemone` gains a `mode` per question, or a server-wide default: `zero-shot` (today) or
`head:<name>`. In head mode the endpoint runs the same feature pass, applies the loaded head,
and fills the answer fields from the head's softmax. Latency is unchanged: one prefill of the
state plus one batched pass over the options, then microseconds for the head.

## 8. When to unfreeze Gemma (Route C)

Only after a head on frozen features plateaus and you have thousands of examples. mlx-lm has
LoRA built in. Apply to attention and MLP projections, backbone lr about 10x lower than the
head, warmup, and the same listwise loss. Keep the frozen-head checkpoint as the baseline you
must beat.

## 9. Data you need

Per task, a JSONL of `{"context", "options", "label"}` where `context` is the rendered
state plus instructions and `options` are the rendered option texts. Rough guide from
jevlike's numbers: a few hundred examples show whether the signal is there, a few thousand
give a usable head. Labels can come from human review, from downstream outcomes (which
department actually resolved the ticket), or from a strong model used as a judge, in which
case the head distils that judge into a 100 ms local scorer.

## 10. Order of work

1. Feature extraction with disk cache (section 2). Verify shapes on 10 examples.
2. Head in MLX plus training loop on the jevlike synthetic set (sections 3 and 5); it must
   reach about 99% top-1 like jevlike's own head does.
3. Eval with shuffled control and ECE (section 6).
4. Per-task datasets and heads for choice, score, noul (section 4).
5. Serve heads behind `/v1/systemone` (section 7).
6. LoRA only if needed (section 8).
