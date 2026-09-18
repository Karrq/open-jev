# One-pass option scoring with a pretrained language model

How to rank pre-written options with a transformer in a single forward pass, why
that is fast even though transformers generate text sequentially, and how to
replicate the `jevlike` scorer with a regular Gemma 4B, with or without
training.

Written 2026-09-16 from the design of
the `vinnylarouge/jevlike` reference implementation.

---

## 1. Why a transformer can score options quickly

Transformers are only sequential when they **produce** new tokens. Token N+1
depends on token N, so decoding is a loop. **Reading** input is fully parallel:
one forward pass over the whole sequence yields a hidden state and a
next-token distribution at every position at once.

Ranking pre-written options never needs the loop. The context and each option
are already known text, so the model reads them in one pass and we read a
number off the result. There are two places to read that number from:

| Source of the score | Needs training | What "better" means |
|---|---|---|
| A small scorer head on top of hidden states | Yes, on labelled pairs | Whatever your labels say |
| The model's own language-model head (log-probs) | No | "Likely text" |

Even when a model does generate, three things keep it fast: the prompt is
prefilled in parallel, each decode step reuses a KV cache, and decode is
memory-bound so small models stream their weights quickly. None of that is
needed for scoring, which is prefill only.

## 2. What `jevlike` does

`jevlike` is a scorer, not a generator. It takes a context and a list of
candidate options and outputs one score per option.

- **Encoder.** Either its own byte-level model trained from scratch (default,
  used for the Doom and chess examples with a convolutional stem), or a frozen
  Hugging Face model loaded with `AutoModel` via `--encoder hf --hf-model ...`.
  The Qwen2.5-0.5B path is an example of the latter.
- **Head.** A small trainable module (`--rank` sets its width) that pools the
  per-token hidden states for context and options into a score per option.
  This is the only thing gradient descent touches on the HF path, and the only
  thing stored in the checkpoint besides the encoder name.
- **Training.** Supervised on labelled option sets. The frozen encoder is a
  feature extractor; the head learns the task-specific definition of "better".

This is the linear-probe / frozen-backbone pattern. Cheap (no backprop through
the encoder), stable (small data cannot wreck pretrained features), reusable
(one encoder, many heads). The ceiling is set by whether the encoder's
representations already contain the signal that separates good from bad.

Reported numbers from the starter: about 98% on synthetic menus with the byte
encoder, 26% on target-disjoint Wikispeedia next-click with frozen Qwen 0.5B
plus head, against about 8% for shuffled and random-encoder controls.

## 3. Three routes to a scorer

### Route A: frozen encoder plus trained head (what `jevlike` ships)

1. Load the pretrained model with `AutoModel` (no LM head).
2. Freeze it. Run one forward pass over context plus options.
3. Pool hidden states, feed the head, train the head with a ranking loss.

Because the encoder is frozen, precompute and cache its features once over the
dataset. Head training then takes seconds per epoch regardless of encoder size.

### Route B: zero-shot likelihood scoring (no training, no decoding)

1. Load with `AutoModelForCausalLM` so the LM head is present.
2. Concatenate context and option. One forward pass.
3. At each position covering an option token, read the log-probability the
   model assigned to the token that is actually there.
4. Sum (or average) over the option tokens. That is the score.

Handle these:

- **Length bias.** Longer options accumulate more negative log-prob. Divide by
  token count, or subtract the option's unconditional score (same option, no
  context).
- **Plausible is not correct.** The score measures how natural the text is as
  a continuation. Good for factual answers, weak for idiosyncratic notions of
  "better".
- **Text only.** Useless for raw bytes such as Doom frames or chess boards.

### Route C: fine-tune the backbone

Attach the head as in Route A but unfreeze the backbone with LoRA on attention
and MLP projections. Train end to end with a pairwise or listwise loss over the
option set. Keep the backbone learning rate roughly an order of magnitude
below the head's and use warmup, or the first few steps destroy the
pretrained representations. Only worth it once Route A plateaus and there is
enough data that unfreezing will not simply overfit.

### Which to try first

1. Route B. Costs nothing but inference and tells you whether the model already
   has the signal.
2. Route A if you have labelled pairs. Cheap and stable.
3. Route C last.

A bigger model helps most on Route B for tasks close to pretraining data. It
does not help at all on non-text inputs; there the trained byte encoder wins.

## 4. Replicating with regular Gemma 4B

All three routes port directly. Gemma-specific points:

- **Model class.** Use `Gemma3ForCausalLM` (text-only path) for Route B, or
  the bare model for Route A.
- **Precision.** bf16, not fp16. Gemma's activations overflow in fp16.
- **Vocabulary.** About 262k tokens, so the LM head is expensive. Compute
  logits only at option positions, never at every position.
- **Attention.** For LoRA training use the eager attention implementation;
  Gemma's logit soft-capping does not work with SDPA in some versions.
- **Memory.** bf16 weights are about 8 GB. A 64 GB Mac has ample room for
  batching.

### Reference implementation sketch for Route B

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it", torch_dtype=torch.bfloat16
).eval()

@torch.inference_mode()
def score_options(context: str, options: list[str]) -> list[float]:
    ctx_ids = tok(context, return_tensors="pt").input_ids[0]
    # Run the context once and cache its KV.
    out = model(ctx_ids[None], use_cache=True)
    cache = out.past_key_values
    last_ctx_logit = out.logits[0, -1]

    scores = []
    for opt in options:
        opt_ids = tok(opt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        o = model(opt_ids[None], past_key_values=cache, use_cache=True)
        # Logits predicting each option token: last context position, then
        # every option position except the final one.
        logits = torch.cat([last_ctx_logit[None], o.logits[0, :-1]], dim=0)
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = logp[torch.arange(len(opt_ids)), opt_ids]
        scores.append(tok_lp.mean().item())  # mean = length-normalised
        # Note: reusing `cache` across options requires it to be immutable or
        # copied per option. With DynamicCache, deep-copy it, or batch all
        # options with padding and a shared prefix instead.
    return scores
```

Batching all options in one padded batch, with the context prefix expanded
across the batch dimension, is the production version. Prefix sharing plus
batching is where nearly all the latency savings come from.

## 5. Latency expectations for Gemma 4B

Scoring is pure prefill, so it is compute-bound and roughly linear in total
tokens. A 4B model is about 8 GFLOP per token, so 1,000 tokens is about 8
TFLOP of work.

Workload: one 200-token context, eight 30-token options.

| Setup | Tokens per example | Expected latency (M5 Pro, 20-core GPU) |
|---|---|---|
| PyTorch on MPS, context re-encoded per option | 1,840 | 1.5 to 4 s |
| PyTorch on MPS, context KV cached once | 440 | 0.4 to 1 s |
| MLX, context re-encoded per option | 1,840 | 0.4 to 0.8 s |
| MLX, context KV cached once | 440 | 0.1 to 0.2 s |

These are estimates extrapolated from similar Apple silicon parts. Measure
before committing, and exclude the first call's warm-up of a few seconds.

What moves the numbers:

- **Share the context prefix.** About a 4x saving on this workload.
- **Batch the options.** Same total compute, much better GPU utilisation.
- **Use MLX on Apple silicon.** Several times faster than PyTorch MPS for
  prefill. MLX also has LoRA support if Route C is needed outside the
  `jevlike` PyTorch loop.
- **Cache features for Route A.** One pass over the dataset, then head
  training is nearly free.

For scale: the `jevlike` byte encoder scores an example in well under 10 ms on
the same machine, and Qwen 0.5B in roughly a tenth of Gemma 4B's time. Gemma
4B is the slowest option by a wide margin but still interactive at a few
examples per second with prefix caching.

## 6. Decision summary

| Approach | Data needed | Cost per score | Ceiling |
|---|---|---|---|
| Frozen small encoder + head (Route A) | Labelled pairs | Tiny | Encoder features |
| Zero-shot likelihood, bigger model (Route B) | None | One prefill of a big model | Prompt / likelihood mismatch |
| LoRA fine-tune + head (Route C) | Labelled pairs, more of them | One prefill of a big model | Highest |
| Prompted judging with generation | None | Sequential decode | Highest quality, slowest |

Start with Route B on Gemma 4B. If it is good enough, stop. If it is weak, or
you find yourself writing elaborate prompts to explain what "better" means,
that is the signal to label pairs and move to Route A, keeping Gemma as the
frozen encoder via `--hf-model`.
