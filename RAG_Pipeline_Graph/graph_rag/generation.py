"""
Answer generation backends.

    ExtractiveGenerator  (default) deterministic, offline. Composes the answer from the *resolved*
                         graph facts (so priority rules are guaranteed) or quotes the best
                         passage sentences. It cannot fabricate: every word comes from evidence.
    FlanT5Generator      google/flan-t5-* over the fused prompt (open-source, no API key). Its
                         output is checked by validation.validate_graph_answer.

A generator either exposes `answer(retrieval, query) -> GenerationOutput` (structured) or
just `generate(prompt) -> str` (any LLM wrapper, including test doubles).
"""
from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .models import REFUSAL, ResolvedFact, Retrieval
from .retrieval import is_narrative_question
from .text import content_tokens, flat, split_sentences

logger = logging.getLogger("graph_rag.generation")

PREDICATE_LABELS = {
    "on_call_engineer": "on-call engineer", "owner_team": "owner team", "status": "status",
    "version": "version", "environment": "environment", "sla_tier": "SLA tier",
    "last_deployed_at": "last deployment time", "manager": "manager", "slack_channel": "Slack channel",
    "escalation_contact": "escalation contact", "member": "members", "tier": "tier",
    "depends_on": "dependencies",
}
#: Words the templates add around evidence values (validation treats them as grounded).
TEMPLATE_VOCAB = {"depends", "depend", "owns", "own", "members", "member", "facts", "known", "time",
                  "deployment", "last", "dependencies", "engineer", "call", "on", "team", "owner"}

_PROCEDURAL_RE = re.compile(r"^\s*(?:how\s+(?:do|can|should|to)|what\s+are\s+the\s+steps|steps|procedure)\b", re.I)


def label(predicate: str) -> str:
    return PREDICATE_LABELS.get(predicate, predicate.replace("_", " "))


def join_names(names: Sequence[str]) -> str:
    names = list(OrderedDict.fromkeys(names))
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


@dataclass
class GenerationOutput:
    text: str
    used_facts: List[ResolvedFact] = field(default_factory=list)
    used_passages: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "facts"        # facts | passages | refusal


class ExtractiveGenerator:
    """Deterministic answer composer over resolved facts and retrieved passages."""

    name = "extractive"

    # ---- facts ---------------------------------------------------------------------------
    @staticmethod
    def _forward(subject: str, predicate: str, values: List[str]) -> str:
        if predicate == "depends_on":
            return f"{subject} depends on {join_names(values)}."
        if predicate == "member":
            return f"The members of {subject} are {join_names(values)}."
        return f"The {label(predicate)} for {subject} is {join_names(values)}."

    @staticmethod
    def _reverse(value: str, predicate: str, subjects: List[str]) -> str:
        if predicate == "depends_on":
            verb = "depends" if len(set(subjects)) == 1 else "depend"
            return f"{join_names(subjects)} {verb} on {value}."
        if predicate == "owner_team":
            return f"{value} owns {join_names(subjects)}."
        if predicate == "member":
            return f"{value} is one of the members of {join_names(subjects)}."
        return f"{value} is the {label(predicate)} for {join_names(subjects)}."

    def _answer_from_facts(self, r: Retrieval, query: str) -> Optional[GenerationOutput]:
        ents = {flat(e): e for e in r.entities}
        facts = [rf for rf in r.resolved if rf.predicate not in ("type", "mentioned_in")]
        if not facts or not ents:
            return None
        sentences: List[str] = []
        used: List[ResolvedFact] = []

        def add(sentence: str, rfs: List[ResolvedFact]) -> None:
            if sentence not in sentences:
                sentences.append(sentence)
                used.extend(rf for rf in rfs if rf not in used)

        if r.predicates:
            for pred in r.predicates:
                fwd = [rf for rf in facts if rf.predicate == pred and flat(rf.subject) in ents]
                rev = [rf for rf in facts if rf.predicate == pred and flat(rf.value) in ents
                       and flat(rf.subject) not in ents]
                reverse_first = pred == "depends_on" and any(
                    re.search(r"depend\w*\s+(?:up)?on\s+(?:the\s+)?" + re.escape(e).replace(r"\-", r"[\s-]"),
                              query, re.I) for e in r.entities)
                if fwd and not reverse_first:
                    for subj in OrderedDict.fromkeys(rf.subject for rf in fwd):
                        group = [rf for rf in fwd if rf.subject == subj]
                        add(self._forward(subj, pred, [rf.value for rf in group]), group)
                elif rev:
                    for val in OrderedDict.fromkeys(rf.value for rf in rev):
                        group = [rf for rf in rev if rf.value == val]
                        add(self._reverse(val, pred, [rf.subject for rf in group]), group)
            # multi-hop: follow a used fact's value to a fact about it (team -> manager, ...)
            for rf in list(used):
                for pred in r.predicates:
                    hop = [h for h in facts if flat(h.subject) == flat(rf.value) and h.predicate == pred
                           and pred != rf.predicate]
                    if hop:
                        add(self._forward(hop[0].subject, pred, [h.value for h in hop]), hop)
            if not sentences:
                return None
        else:
            # No specific attribute asked: summarise what the graph knows about the entities.
            for key, ent in ents.items():
                about = [rf for rf in facts if flat(rf.subject) == key][:8]
                if about:
                    parts = [f"{label(rf.predicate)}: {rf.value}" for rf in about]
                    add(f"Known facts for {about[0].subject} - " + "; ".join(parts) + ".", about)
            if not sentences:
                return None
        return GenerationOutput(" ".join(sentences), used, [], "facts")

    # ---- passages ------------------------------------------------------------------------
    def _answer_from_passages(self, r: Retrieval, query: str) -> Optional[GenerationOutput]:
        passages = sorted(r.passages, key=lambda p: p["distance"])
        if not passages:
            return None
        q_tokens = set(content_tokens(query))
        need = max(1, min(3, math.ceil(0.4 * len(q_tokens))))
        best = passages[0]
        if _PROCEDURAL_RE.search(query) and best.get("tier") == "wiki":
            body = best["text"].split("\n", 1)[-1].strip()
            words = body.split()
            if len(set(content_tokens(body)) & q_tokens) >= need:
                text = " ".join(words[:90]) + (" ..." if len(words) > 90 else "")
                return GenerationOutput(text, [], [best], "passages")
        # Best-matching sentence across the top passages, plus the sentence after it when that one
        # continues the thought (shares a query word) - both from the SAME passage, so one answer
        # never stitches together unrelated documents.
        best_hit = None
        for rank, p in enumerate(passages[:3]):
            body = p["text"].split("\n", 1)[-1] if p.get("tier") == "wiki" else p["text"]
            sentences = split_sentences(body)
            for i, s in enumerate(sentences):
                overlap = len(set(content_tokens(s)) & q_tokens)
                if overlap >= need and (best_hit is None or (overlap, -rank) > (best_hit[0], -best_hit[1])):
                    best_hit = (overlap, rank, i, sentences, p)
        if best_hit is None:
            return None
        _, _, i, sentences, p = best_hit
        chosen = [sentences[i]]
        if i + 1 < len(sentences) and set(content_tokens(sentences[i + 1])) & q_tokens:
            chosen.append(sentences[i + 1])
        return GenerationOutput(" ".join(chosen), [], [p], "passages")

    def answer(self, r: Retrieval, query: str) -> GenerationOutput:
        order = ([self._answer_from_passages, self._answer_from_facts] if is_narrative_question(query)
                 else [self._answer_from_facts, self._answer_from_passages])
        for fn in order:
            out = fn(r, query)
            if out is not None:
                return out
        return GenerationOutput(REFUSAL, mode="refusal")

    def generate(self, prompt: str, retrieval: Optional[Retrieval] = None, query: str = "") -> str:
        if retrieval is None:
            return REFUSAL                                # cannot answer from a bare prompt
        return self.answer(retrieval, query).text


class FlanT5Generator:
    """FLAN-T5 seq2seq generation; the model is loaded once on first use."""

    def __init__(self, model_name: str, max_input_tokens: int = 712, max_new_tokens: int = 150):
        self.model_name = model_name
        self.name = f"flan-t5:{model_name}"
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self._tok = self._model = self._device = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self._device)

    def count_tokens(self, text: str) -> int:
        """Exact prompt length in model tokens (used by prompting.fit_prompt)."""
        if self._model is None:
            self._load()
        return len(self._tok(text, add_special_tokens=True)["input_ids"])

    def generate(self, prompt: str) -> str:
        if self._model is None:
            self._load()
        import torch
        inputs = self._tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self._tok.decode(out[0], skip_special_tokens=True).strip()


_LLM_CACHE: Dict[str, FlanT5Generator] = {}


def get_llm(model_name: str, config: Dict[str, Any]) -> FlanT5Generator:
    """Process-wide FLAN-T5 instance per model name."""
    if model_name not in _LLM_CACHE:
        _LLM_CACHE[model_name] = FlanT5Generator(
            model_name, int(config["max_context_tokens"]) + 200, int(config["max_answer_tokens"]))
    return _LLM_CACHE[model_name]


def make_generator(config: Dict[str, Any]):
    if config.get("llm_backend", "extractive") == "flan-t5":
        return get_llm(config["generation_model"], config)
    return ExtractiveGenerator()
