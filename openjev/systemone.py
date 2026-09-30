"""TypeSafe-compatible System One API (https://docs.typesafe.ai) on top of the local scorer.

POST /v1/systemone
  {"state": <string|object|array>, "model": "...", "questions": {id: Choice|Score|Noul}}
  -> {"model": "...", "answers": {id: answer}, "usage": {"input_tokens": n, "output_tokens": n}}

Two readouts, both without generating anything:

- ``letters`` (default): SemIf's prompt (github.com/TheoLeeCJ/SemIf). A system instruction
  asks for one option letter; the user turn is JSON with the state as ``evidence``, the
  question as ``criterion`` and lettered options, rendered through the model's chat
  template. The answer distribution is the next-token distribution restricted to the
  option letters. On LocalLLaMA/typed-decisions this scores 0.738 with Gemma 4 26B-A4B
  against 0.636 for ``continuation``. Questions with more options than letters fall back
  to ``continuation``.
- ``continuation``: a plain-text prompt ending in "Answer:\\n", with each candidate label
  scored as a continuation in one prefix-shared batched forward pass per question.

Confidence is 1 - normalised entropy of the probability distribution, an approximation
of TypeSafe's "how spread out is the distribution" definition.

Every prompt opens with the same rendered State block, so it is prefilled once per request
and each question only encodes its own instructions and labels. This matches TypeSafe's
documented behaviour, where the state is ingested once and every question is evaluated
against it. Without it a request with many questions re-encodes the whole state per
question, which dominates cost as soon as the state is large. With ``letters``, the
prefilled state can also be kept across requests (prefix_store.py), and all questions
are read in packed forward passes over that one copy (packed.py).
"""
from __future__ import annotations

import json
import math
import time
from typing import Any, Literal, Union

import mlx.core as mx
from pydantic import BaseModel, Field, model_validator

from . import prefix_store
from .prefix_store import PrefixStore
from .scorer import OptionScorer, PrefixCache, _softmax

Entry = Union[str, dict, list, None]
MAX_CHOICE_OPTIONS = 255
READOUTS = ("letters", "continuation")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTER_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


# ------------------------------------------------------------------ request
class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Entry = None
    criteria: dict[str, Entry] = Field(min_length=2, max_length=MAX_CHOICE_OPTIONS)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Entry = None
    criteria: list[Entry] = Field(min_length=2)


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Entry = None
    criteria: dict[str, Entry] | None = None  # optional {"true": ..., "false": ...}

    @model_validator(mode="after")
    def _keys(self):
        if self.criteria and not set(self.criteria) <= {"true", "false"}:
            raise ValueError("noul criteria keys must be 'true' and/or 'false'")
        return self


Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


class SystemOneRequest(BaseModel):
    state: Union[str, dict, list]
    model: str | None = None
    questions: dict[str, Question] = Field(min_length=1)


# ----------------------------------------------------------------- response
class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, Entry]
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]]
    usage: Usage


# ---------------------------------------------------------------- rendering
def render_entry(e: Entry) -> str:
    if e is None:
        return ""
    if isinstance(e, str):
        return e
    return json.dumps(e, indent=2, ensure_ascii=False)


def _block(title: str, e: Entry) -> str:
    body = render_entry(e)
    return f"{title}:\n{body}\n\n" if body else ""


def state_prefix(state: Entry) -> str:
    """The leading text every question prompt for this state shares."""
    return _block("State", state)


def render_choice(state: Entry, q: ChoiceQuestion) -> tuple[str, list[str]]:
    lines = []
    for name, desc in q.criteria.items():
        d = render_entry(desc)
        lines.append(f"- {name}: {d}" if d else f"- {name}")
    prompt = (
        _block("State", state)
        + _block("Question", q.instructions)
        + "Choose exactly one option.\nOptions:\n" + "\n".join(lines)
        + "\n\nAnswer:\n"
    )
    return prompt, list(q.criteria)


def render_score(state: Entry, q: ScoreQuestion) -> tuple[str, list[str]]:
    lines = []
    for i, desc in enumerate(q.criteria):
        d = render_entry(desc)
        lines.append(f"{i}: {d}" if d else f"{i}")
    prompt = (
        _block("State", state)
        + _block("Question", q.instructions)
        + "Rate on the following ordered levels and answer with the level number only.\nLevels:\n"
        + "\n".join(lines)
        + "\n\nAnswer:\n"
    )
    return prompt, [str(i) for i in range(len(q.criteria))]


def render_noul(state: Entry, q: NoulQuestion) -> tuple[str, list[str]]:
    crit = ""
    if q.criteria:
        t, f = render_entry(q.criteria.get("true")), render_entry(q.criteria.get("false"))
        crit = ("Answer yes when:\n" + t + "\n\n" if t else "") + ("Answer no when:\n" + f + "\n\n" if f else "")
    prompt = (
        _block("State", state)
        + _block("Question", q.instructions)
        + crit
        + "Answer yes or no.\n\nAnswer:\n"
    )
    return prompt, ["yes", "no"]


def option_descriptions(q: Question) -> tuple[list[str], list[str]]:
    """Answer labels and the option text shown for each, for the ``letters`` prompt."""
    if isinstance(q, ChoiceQuestion):
        labels = list(q.criteria)
        descs = [f"{k}: {render_entry(d)}" if render_entry(d) else k for k, d in q.criteria.items()]
    elif isinstance(q, ScoreQuestion):
        labels = [str(i) for i in range(len(q.criteria))]
        descs = [render_entry(d) for d in q.criteria]
    else:
        crit = q.criteria or {}
        labels = ["true", "false"]
        descs = [
            render_entry(crit.get("true")) or "Yes, the statement is true.",
            render_entry(crit.get("false")) or "No, the statement is false.",
        ]
    return labels, descs


def letter_messages(state: Entry, instructions: Entry, descs: list[str]) -> list[dict]:
    payload = {
        "evidence": state,
        "criterion": render_entry(instructions),
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(descs)],
    }
    return [
        {"role": "system", "content": LETTER_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def letter_ids(scorer: OptionScorer, conversations: list[list[dict]]) -> list[list[int]]:
    texts = [
        scorer.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for m in conversations
    ]
    hf = scorer.tok._tokenizer
    # One batched call is ~5x faster than encoding each prompt, which matters when
    # hundreds of questions each repeat a long state. Tokenizers that rewrite text
    # inside encode() (mlx-lm's NewlineTokenizer) keep the per-prompt path.
    if hasattr(hf, "_preprocess_text"):
        return [scorer.tok.encode(t, add_special_tokens=False) for t in texts]
    return hf(texts, add_special_tokens=False)["input_ids"]


# ------------------------------------------------------------------- maths
def confidence(probs: list[float]) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(n)))


# ------------------------------------------------------------------ answer
def _answer(q: Question, labels: list[str], probs: list[float]):
    if isinstance(q, ChoiceQuestion):
        best = max(range(len(labels)), key=lambda i: probs[i])
        return ChoiceAnswer(
            choice=labels[best],
            probabilities=dict(zip(labels, probs)),
            confidence=confidence(probs),
        )
    if isinstance(q, ScoreQuestion):
        return ScoreAnswer(
            score=sum(i * p for i, p in enumerate(probs)),
            confidence=confidence(probs),
            legend={str(i): c for i, c in enumerate(q.criteria)},
            probabilities={str(i): p for i, p in enumerate(probs)},
        )
    return NoulAnswer(noul=probs[0])


def system_one(
    scorer: OptionScorer,
    req: SystemOneRequest,
    model_name: str,
    norm: str = "sum",
    readout: str = "letters",
    store: PrefixStore | None = None,
) -> SystemOneResponse:
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}")
    questions = req.questions
    answers: dict[str, Any] = {}
    in_tok = out_tok = 0
    if readout == "letters":
        lettered = {qid: q for qid, q in questions.items() if len(option_descriptions(q)[0]) <= len(LETTERS)}
        if lettered:
            answers, in_tok, out_tok = _letters(scorer, req.state, lettered, store)
        questions = {qid: q for qid, q in questions.items() if qid not in lettered}
    if questions:
        more, i, o = _continuation(scorer, req.state, questions, norm)
        answers.update(more)
        in_tok, out_tok = in_tok + i, out_tok + o
    answers = {qid: answers[qid] for qid in req.questions}
    return SystemOneResponse(model=model_name, answers=answers, usage=Usage(input_tokens=in_tok, output_tokens=out_tok))


def _letters(scorer: OptionScorer, state: Entry, questions: dict[str, Question], store: PrefixStore | None):
    """All questions share one prefilled prefix and are read in packed passes."""
    t0 = time.perf_counter()
    slots = []
    for letter in LETTERS:
        ids = scorer.tok.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"option letter {letter!r} is not a single token for this tokenizer")
        slots.append(ids[0])
    rows, conversations = [], []
    for qid, q in questions.items():
        labels, descs = option_descriptions(q)
        rows.append((qid, q, labels))
        conversations.append(letter_messages(state, q.instructions, descs))
    # A probe question marks where the state ends even when the request has only one
    # question, so the stored prefix is the state, not the state plus that question.
    conversations.append(letter_messages(state, "\u2063", ["", ""]))
    *encoded, probe = letter_ids(scorer, conversations)
    rows = [(*r, ids) for r, ids in zip(rows, encoded)]
    n = min(prefix_store.common_len(probe, ids) for *_, ids in rows)
    cache, info = prefix_store.prefill(scorer, store, probe[:n])
    t1 = time.perf_counter()
    logits = scorer.last_logits(cache, [ids[n:] for *_, ids in rows])
    answers = {}
    for (qid, q, labels, _), lg in zip(rows, logits):
        picked = lg[mx.array(slots[: len(labels)])].tolist()
        answers[qid] = _answer(q, labels, _softmax(picked))
    scorer.last_timing = {
        **info,
        "questions_s": time.perf_counter() - t1,
        "total_s": time.perf_counter() - t0,
        "prefix_tokens": n,
        "suffix_tokens": sum(len(ids) - n for *_, ids in rows),
    }
    return answers, sum(len(ids) for *_, ids in rows), len(rows)


def _continuation(scorer: OptionScorer, state: Entry, questions: dict[str, Question], norm: str):
    answers: dict[str, Any] = {}
    in_tok = out_tok = 0
    # pmi needs an unconditional pass per question and cannot reuse a prefix.
    prefix: PrefixCache | None = None
    if norm != "pmi" and len(questions) > 1:
        prefix = scorer.prefill_prefix(state_prefix(state))
    for qid, q in questions.items():
        if isinstance(q, ChoiceQuestion):
            prompt, labels = render_choice(state, q)
        elif isinstance(q, ScoreQuestion):
            prompt, labels = render_score(state, q)
        else:
            prompt, labels = render_noul(state, q)
        if prefix is None:
            res = scorer.score(prompt, labels, norm=norm, chat=False, sep="")
        else:
            res = scorer.score_with_prefix(prefix, prompt, labels, norm=norm)
        in_tok += int(scorer.last_timing["context_tokens"])
        out_tok += int(scorer.last_timing["option_tokens"])
        answers[qid] = _answer(q, labels, [r.probability for r in res])
    return answers, in_tok, out_tok
