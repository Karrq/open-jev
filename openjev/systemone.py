"""TypeSafe-compatible System One API (https://docs.typesafe.ai) on top of the local scorer.

POST /v1/systemone
  {"state": <string|object|array>, "model": "...", "questions": {id: Choice|Score|Noul}}
  -> {"model": "...", "answers": {id: answer}, "usage": {"input_tokens": n, "output_tokens": n}}

Every question is rendered into a plain-text prompt ending in "Answer:\\n" and the candidate
labels are scored as continuations with one prefix-shared batched forward pass per question.
Nothing is generated. Confidence is 1 - normalised entropy of the probability distribution,
an approximation of TypeSafe's "how spread out is the distribution" definition.

Every prompt opens with the same rendered State block, so it is prefilled once per request
and each question only encodes its own instructions and labels. This matches TypeSafe's
documented behaviour, where the state is ingested once and every question is evaluated
against it. Without it a request with many questions re-encodes the whole state per
question, which dominates cost as soon as the state is large.
"""
from __future__ import annotations

import json
import math
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

from .scorer import OptionScorer, PrefixCache

Entry = Union[str, dict, list, None]
MAX_CHOICE_OPTIONS = 255


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


# ------------------------------------------------------------------- maths
def confidence(probs: list[float]) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(n)))


# ------------------------------------------------------------------ answer
def system_one(scorer: OptionScorer, req: SystemOneRequest, model_name: str, norm: str = "sum") -> SystemOneResponse:
    answers: dict[str, Any] = {}
    in_tok = out_tok = 0
    # pmi needs an unconditional pass per question and cannot reuse a prefix.
    prefix: PrefixCache | None = None
    if norm != "pmi" and len(req.questions) > 1:
        prefix = scorer.prefill_prefix(state_prefix(req.state))
    for qid, q in req.questions.items():
        if isinstance(q, ChoiceQuestion):
            prompt, labels = render_choice(req.state, q)
        elif isinstance(q, ScoreQuestion):
            prompt, labels = render_score(req.state, q)
        else:
            prompt, labels = render_noul(req.state, q)
        if prefix is None:
            res = scorer.score(prompt, labels, norm=norm, chat=False, sep="")
        else:
            res = scorer.score_with_prefix(prefix, prompt, labels, norm=norm)
        probs = [r.probability for r in res]
        in_tok += int(scorer.last_timing["context_tokens"])
        out_tok += int(scorer.last_timing["option_tokens"])

        if isinstance(q, ChoiceQuestion):
            best = max(range(len(labels)), key=lambda i: probs[i])
            answers[qid] = ChoiceAnswer(
                choice=labels[best],
                probabilities=dict(zip(labels, probs)),
                confidence=confidence(probs),
            )
        elif isinstance(q, ScoreQuestion):
            answers[qid] = ScoreAnswer(
                score=sum(i * p for i, p in enumerate(probs)),
                confidence=confidence(probs),
                legend={str(i): c for i, c in enumerate(q.criteria)},
                probabilities={str(i): p for i, p in enumerate(probs)},
            )
        else:
            answers[qid] = NoulAnswer(noul=probs[0])
    return SystemOneResponse(model=model_name, answers=answers, usage=Usage(input_tokens=in_tok, output_tokens=out_tok))
