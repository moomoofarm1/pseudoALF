#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leave-one-file-out few-shot pseudonymization NER with a native Hugging Face
encoder, a small PyTorch ProtoBERT implementation, and SuperCD/StructShot-style
NNShot + Viterbi decoding.

Why this v3 exists
------------------
The preceding v2 preserved thunlp/Few-NERD's complete runtime. That gave strong
methodological fidelity, but it also forced modern Hugging Face encoders through
Few-NERD's old BERTWordEncoder / episode framework and required compatibility
and memory patches.

This version keeps the therapy/pseudonymization evaluation design but removes
Few-NERD as a runtime dependency:

    AutoTokenizer + AutoModel
             |
      word representations
      (first subword only)
          /       \\
  ProtoBERT       NNShot
  prototypes      emissions
                     |
              StructShot Viterbi

The StructShot path follows the logic used by chen700564/supercd and the
original asappresearch/structshot implementation:
  * support/query token representations come directly from a Hugging Face model
  * first-subword labels are the supervised token positions
  * NNShot uses normalized negative squared Euclidean similarity by default
  * per-class emission = best support-token similarity for that class
  * StructShot adds abstract transition probabilities and Viterbi decoding

ProtoBERT is deliberately small and transparent: one mean support prototype per
observed class (including O), negative squared Euclidean logits, argmax.
There is no fine-tuning, episodic training, or learned classification head.

Input
-----
ONE directory of Presidio-preprocessed .txt/.alfrttm files with inline labels:

    [PersonData]Peter[PER]

Each file acts once as the single support session; all remaining files are the
query set for that fold.

Dependencies
------------
Required for modelling/evaluation: torch, transformers, numpy, nervaluate. No Flair. No Few-NERD tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import importlib.util
import math
import random
import re
import sys
from importlib import metadata as importlib_metadata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Directory containing this script and its helper modules (config_pipe.py, loger.py).
SCRIPT_DIR = Path(__file__).resolve().parent

try:
    import torch
    import torch.nn.functional as F
except Exception as _torch_exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _torch_exc
else:
    _TORCH_IMPORT_ERROR = None


try:
    from nervaluate.evaluator import Evaluator as NervaluateEvaluator
except Exception as _nervaluate_exc:  # pragma: no cover
    NervaluateEvaluator = None  # type: ignore[assignment]
    _NERVALUATE_IMPORT_ERROR = _nervaluate_exc
else:
    _NERVALUATE_IMPORT_ERROR = None

try:
    NERVALUATE_VERSION = importlib_metadata.version("nervaluate")
except importlib_metadata.PackageNotFoundError:  # pragma: no cover
    NERVALUATE_VERSION = None


SCRIPT_VERSION = "2026-09-15-hf-protobert-supercd-structshot-v3.3-nervaluate-hfauth-log"
NERVALUATE_SCENARIOS = ("strict", "exact", "partial", "ent_type")
SUPERCD_REPO = "https://github.com/chen700564/supercd"
SUPERCD_REF = "c9208b5fe19f30c007882d17c3c3ad1c634aaa90"
STRUCTSHOT_REPO = "https://github.com/asappresearch/structshot"
STRUCTSHOT_REF = "2bf53794b3ffd55b9970eb7e3c4b68847b1bd4eb"

# ---------------------------------------------------------------------------
# GDPR-ALF child-level schema
# ---------------------------------------------------------------------------
GDPR_ALF_PARENT_CHILDREN = {
    "PER": ("PER", "STAFF", "PATIENT"),
    "LOC": ("LOC", "ADDRESS", "POSTCODE"),
    "ORG": ("ORG", "HOSPITAL"),
    "CONTACT": ("CONTACT", "PHONE", "EMAIL"),
    "ID": ("ID", "CPR"),
    "DATE": ("DATE", "TIME", "DURATION"),
    "DEM": ("DEM", "ETHNICITY", "RELIGION", "POLITICS", "SEXUALITY", "AGE"),
    "HEALTH": ("HEALTH", "DIAGNOSIS", "MEDICATION", "CONDITION"),
    "MISC": tuple(),
}
GDPR_ALF_TARGET_LABELS = frozenset(
    child for children in GDPR_ALF_PARENT_CHILDREN.values() for child in children
)

DEFAULT_ANNOTATION_ALIASES = {
    "NAME": "PER",
    "PERSON": "PER",
    "ORGANIZATION": "ORG",
    "GPE": "LOC",
    "LOCATION": "LOC",
    "FACILITY": "LOC",
    "LANGUAGE": "DEM",
}

ANNOTATION_TEMPLATE = (
    r"\[{marker}\](?P<text>[^\[\]\r\n]+?)"
    r"\[(?P<label>[A-Za-z][A-Za-z0-9_ ]*)\]"
)
DEFAULT_WORD_REGEX = r"\w+|[^\w\s]"


# ===========================================================================
# Data model
# ===========================================================================
@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    text: str
    raw_label: str = ""


@dataclass
class ValidationRecord:
    uid: str
    line_no: int
    text: str
    reference_entities: List[Entity]
    unsupported_entities: List[Entity]


@dataclass(frozen=True)
class Word:
    start: int
    end: int
    text: str


@dataclass
class EncodedSet:
    uid: str
    line_no: int
    words: List[Word]
    word_labels: List[str]
    boundary_expansions: int
    # For every original word: [first_subword, last_subword+1) in the concatenated
    # text-subword stream. (-1, -1) means the tokenizer dropped the word.
    word_subword_span: List[Tuple[int, int]]
    # Text-subword token ids, without special tokens, split to model-sized chunks.
    chunks: List[List[int]]
    # Filled once by embed_bundles(); CPU float32 [n_words, hidden]. Dropped words
    # are zero rows and valid_word_mask marks which rows are usable.
    word_reps: Optional["torch.Tensor"] = None
    valid_word_mask: Optional[List[bool]] = None


@dataclass
class FileBundle:
    name: str
    input_path: Path
    records: List[ValidationRecord]
    encoded: List[EncodedSet] = field(default_factory=list)
    tokens_dropped_records: int = 0
    dropped_words: int = 0
    boundary_expansions: int = 0
    n_chunks: int = 0
    n_text_subwords: int = 0


# ===========================================================================
# Requirements / device
# ===========================================================================
def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch could not be imported in this interpreter "
            f"({_TORCH_IMPORT_ERROR}). Run inside the torch + transformers env."
        )


def require_transformers():
    try:
        import transformers
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The `transformers` package is not importable in this interpreter "
            f"({exc}). Run this script in the configured torch/transformers env."
        ) from exc
    return transformers



def _load_python_module_from_path(path: Path, module_name: str):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Python helper module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hf_token_from_config(config_path: Path) -> Optional[str]:
    """Load HG_TOKEN from config_pipe.py without ever logging the token itself."""
    module = _load_python_module_from_path(config_path, "fewshot_config_pipe")
    token = getattr(module, "HG_TOKEN", None)
    if token is None:
        return None
    token = str(token).strip()
    if not token or token.upper() == "YOUR_TOKEN":
        return None
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return token


def setup_trace_logging(logger_module_path: Path, log_file: Path):
    """Load user-provided loger.py and initialize file + console logging."""
    log_file = Path(log_file).expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    module = _load_python_module_from_path(logger_module_path, "fewshot_loger")
    setup = getattr(module, "setup_logging", None)
    if not callable(setup):
        raise AttributeError(
            f"{logger_module_path} does not define callable setup_logging(log_file, level)"
        )
    setup(str(log_file), level=logging.INFO)
    logger = logging.getLogger("fewshot_pseudonymization")
    logger.info("Logging initialized: %s", log_file)
    return logger, log_file


def require_nervaluate() -> None:
    if NervaluateEvaluator is None:
        raise RuntimeError(
            "nervaluate could not be imported in this interpreter "
            f"({_NERVALUATE_IMPORT_ERROR}). Install it in the same environment with: "
            "uv pip install 'nervaluate>=1.2.0'"
        )

def resolve_device(requested: str):
    require_torch()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("--device mps requested, but MPS is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_torch_dtype(name: str):
    require_torch()
    if name == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


# ===========================================================================
# Annotation parsing
# ===========================================================================
def canonical_label(label: str, aliases: Dict[str, str]) -> str:
    x = label.strip().upper()
    if x == "WORK_OF_ART":
        x = "WORK OF ART"
    return aliases.get(x, x)


def make_annotation_re(marker: str) -> re.Pattern:
    return re.compile(ANNOTATION_TEMPLATE.format(marker=re.escape(marker)))


def extract_utterance(
    line: str, strict: bool, path: Path, line_no: int
) -> Optional[str]:
    first = line.find('"')
    last = line.rfind('"')
    if first >= 0 and last > first:
        utterance = line[first + 1 : last]
        return utterance if utterance.strip() else None
    if strict and line.strip():
        raise ValueError(
            f"{path}:{line_no}: ALFRRTM line has no quoted utterance. "
            "Fix the row or omit --strict-quoted-utterance to skip it."
        )
    return None


def parse_annotated_utterance(
    utterance: str,
    ann_re: re.Pattern,
    aliases: Dict[str, str],
) -> Tuple[str, List[Entity]]:
    out: List[str] = []
    entities: List[Entity] = []
    cursor = 0
    out_len = 0
    for m in ann_re.finditer(utterance):
        prefix = utterance[cursor : m.start()]
        out.append(prefix)
        out_len += len(prefix)
        mention = m.group("text")
        raw_label = m.group("label").strip()
        label = canonical_label(raw_label, aliases)
        start = out_len
        out.append(mention)
        out_len += len(mention)
        entities.append(
            Entity(start, out_len, label, mention, raw_label=raw_label)
        )
        cursor = m.end()
    out.append(utterance[cursor:])
    return "".join(out), entities


def load_input_validation_records(
    path: Path,
    marker: str,
    aliases: Dict[str, str],
    target_labels: set,
    strict_quoted_utterance: bool,
) -> Tuple[List[ValidationRecord], List[dict]]:
    ann_re = make_annotation_re(marker)
    records: List[ValidationRecord] = []
    audit_rows: List[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    is_alfrttm = path.suffix.lower() == ".alfrttm"
    for i, line in enumerate(lines, start=1):
        if is_alfrttm:
            utterance = extract_utterance(line, strict_quoted_utterance, path, i)
            if utterance is None:
                continue
        else:
            utterance = line.rstrip("\r\n")
            if not utterance.strip():
                continue
        plain, raw_entities = parse_annotated_utterance(utterance, ann_re, aliases)
        reference_entities: List[Entity] = []
        unsupported_entities: List[Entity] = []
        for ent in raw_entities:
            in_schema = ent.label in target_labels
            audit_rows.append(
                {
                    "file": path.name,
                    "record_uid": f"{path.name}::line={i}",
                    "line_no": i,
                    "raw_label": ent.raw_label,
                    "canonical_label": ent.label,
                    "entity_text": ent.text,
                    "status": (
                        "in_gdpr_alf_child_schema"
                        if in_schema
                        else "out_of_gdpr_alf_child_schema"
                    ),
                }
            )
            (reference_entities if in_schema else unsupported_entities).append(ent)
        records.append(
            ValidationRecord(
                uid=f"{path.name}::line={i}",
                line_no=i,
                text=plain,
                reference_entities=reference_entities,
                unsupported_entities=unsupported_entities,
            )
        )
    return records, audit_rows


def discover_files(
    directory: Path,
    extensions: set,
    include_substrings: Sequence[str],
) -> List[Path]:
    paths = []
    for p in directory.rglob("*"):
        if ".ipynb_checkpoints" in p.parts or not p.is_file():
            continue
        if p.suffix.lower() not in extensions:
            continue
        if include_substrings and not any(x in p.name for x in include_substrings):
            continue
        paths.append(p)
    return sorted(paths, key=lambda p: str(p).casefold())


# ===========================================================================
# Word/subword encoding
# ===========================================================================
_WORD_RE_CACHE: Dict[str, re.Pattern] = {}


def word_pattern(pattern: str) -> re.Pattern:
    if pattern not in _WORD_RE_CACHE:
        _WORD_RE_CACHE[pattern] = re.compile(pattern)
    return _WORD_RE_CACHE[pattern]


def tokenize_to_words(text: str, pattern: str) -> List[Word]:
    rx = word_pattern(pattern)
    return [Word(m.start(), m.end(), m.group(0)) for m in rx.finditer(text)]


def gold_word_labels(
    text: str,
    words: Sequence[Word],
    entities: Sequence[Entity],
) -> Tuple[List[str], int]:
    labels = ["O"] * len(words)
    for ent in entities:
        for i, w in enumerate(words):
            if w.end <= ent.start:
                continue
            if w.start >= ent.end:
                break
            if labels[i] != "O" and labels[i] != ent.label:
                raise ValueError(
                    "Overlapping reference entities with conflicting labels: "
                    f"word={w.text!r}, {labels[i]} vs {ent.label}"
                )
            labels[i] = ent.label
    expansions = 0
    for ent in entities:
        starts_on_boundary = any(w.start == ent.start for w in words)
        ends_on_boundary = any(w.end == ent.end for w in words)
        if not (starts_on_boundary and ends_on_boundary):
            expansions += 1
    return labels, expansions


def encode_words_hf_batch(tokenizer, words: Sequence[Word]) -> Tuple[List[int], List[int]]:
    enc = tokenizer(
        [w.text for w in words],
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    ids = list(enc["input_ids"])
    if not hasattr(enc, "word_ids"):
        raise RuntimeError(
            "Tokenizer has no word_ids(); use --tokenization per-word or a fast tokenizer."
        )
    wids = enc.word_ids()
    if wids is None or len(wids) != len(ids):
        raise RuntimeError("Fast tokenizer returned an inconsistent word_ids() map")
    mapped = [(-1 if w is None else int(w)) for w in wids]
    return [int(x) for x in ids], mapped


def encode_words_per_word(tokenizer, words: Sequence[Word]) -> Tuple[List[int], List[int]]:
    ids: List[int] = []
    mapped: List[int] = []
    for wi, w in enumerate(words):
        piece_ids = tokenizer.encode(w.text, add_special_tokens=False)
        if not piece_ids:
            continue
        ids.extend(int(x) for x in piece_ids)
        mapped.extend([wi] * len(piece_ids))
    return ids, mapped


def chunk_subwords(
    ids: Sequence[int], mapped: Sequence[int], max_content_tokens: int
) -> Tuple[List[List[int]], List[List[int]]]:
    if max_content_tokens < 1:
        raise ValueError("max_content_tokens must be >= 1")
    chunks_ids: List[List[int]] = []
    chunks_mapped: List[List[int]] = []
    for start in range(0, max(len(ids), 1), max_content_tokens):
        piece_ids = list(ids[start : start + max_content_tokens])
        if not piece_ids:
            break
        chunks_ids.append(piece_ids)
        chunks_mapped.append(list(mapped[start : start + max_content_tokens]))
    return chunks_ids, chunks_mapped


def encode_record(
    rec: ValidationRecord,
    tokenizer,
    max_length: int,
    max_content_tokens: int,
    word_regex: str,
    tokenization: str,
    audit_rows: List[dict],
) -> Optional[EncodedSet]:
    words = tokenize_to_words(rec.text, word_regex)
    if not words:
        return None
    word_labels, boundary_expansions = gold_word_labels(
        rec.text, words, rec.reference_entities
    )
    if tokenization == "hf-batch":
        ids, mapped = encode_words_hf_batch(tokenizer, words)
    elif tokenization == "per-word":
        ids, mapped = encode_words_per_word(tokenizer, words)
    else:
        raise ValueError(f"unknown tokenization mode {tokenization!r}")
    if not ids:
        return None
    chunks_ids, chunks_mapped = chunk_subwords(ids, mapped, max_content_tokens)

    first_pos: Dict[int, int] = {}
    last_pos: Dict[int, int] = {}
    position = 0
    for cm in chunks_mapped:
        for wi in cm:
            if wi >= 0:
                first_pos.setdefault(wi, position)
                last_pos[wi] = position
            position += 1
    spans = [
        (first_pos.get(wi, -1), last_pos.get(wi, -2) + 1 if wi in last_pos else -1)
        for wi in range(len(words))
    ]
    dropped_words = sum(1 for a, _ in spans if a < 0)
    audit_rows.append(
        {
            "record_uid": rec.uid,
            "line_no": rec.line_no,
            "n_words": len(words),
            "n_text_subwords": len(ids),
            "n_chunks": len(chunks_ids),
            "dropped_words_no_subword": dropped_words,
            "reference_entities_not_on_word_boundaries": boundary_expansions,
            "max_length": max_length,
            "max_content_tokens": max_content_tokens,
        }
    )
    return EncodedSet(
        uid=rec.uid,
        line_no=rec.line_no,
        words=words,
        word_labels=word_labels,
        boundary_expansions=boundary_expansions,
        word_subword_span=spans,
        chunks=chunks_ids,
    )


# ===========================================================================
# Hugging Face encoder
# ===========================================================================
def resolve_max_length_cap(tokenizer, config) -> Tuple[Optional[int], str]:
    candidates: List[Tuple[int, str]] = []
    tok = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok, int) and 2 < tok < 1_000_000:
        candidates.append((tok, "tokenizer.model_max_length"))
    for attr in ("max_position_embeddings", "n_positions"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and 2 < value < 1_000_000:
            # RoBERTa/XLM-R position tables commonly include reserved positions;
            # tokenizer.model_max_length is the safer cap when both are present.
            candidates.append((value, f"config.{attr}"))
    if not candidates:
        return None, "unknown"
    value, src = min(candidates, key=lambda x: x[0])
    return int(value), src


class HFWordEncoder:
    """Native AutoModel encoder returning first-subword word representations."""

    def __init__(
        self,
        model_name: str,
        device,
        hidden_representation: str = "last",
        model_dtype: str = "auto",
        trust_remote_code: bool = False,
        hf_token: Optional[str] = None,
    ):
        require_torch()
        require_transformers()
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.hidden_representation = hidden_representation
        self.hf_token = hf_token
        auth_kwargs = {"token": hf_token} if hf_token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            **auth_kwargs,
        )
        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            **auth_kwargs,
        )
        dtype = parse_torch_dtype(model_dtype)
        kwargs = {"trust_remote_code": trust_remote_code, **auth_kwargs}
        if dtype is not None:
            # Transformers v5 documents `dtype`; v4 used `torch_dtype`. Current
            # v5 still accepts the legacy name, but choosing by major version
            # avoids a deprecation warning while retaining v4 compatibility.
            try:
                major = int(str(transformers.__version__).split(".", 1)[0])
            except Exception:
                major = 4
            kwargs["dtype" if major >= 5 else "torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.to(device)
        self.model.eval()

        # Transformers v5 removed prepare_for_model() and
        # build_inputs_with_special_tokens() from TokenizersBackend tokenizers
        # (including the current XLMRobertaTokenizer).  We therefore learn the
        # *single-sequence* special-token wrapper once through the public
        # tokenizer call API, then apply that prefix/suffix directly to already
        # tokenized IDs.  This preserves the exact subword IDs and avoids a
        # lossy decode -> retokenize round trip.
        (
            self.special_prefix_ids,
            self.special_suffix_ids,
        ) = self._infer_single_sequence_special_wrapper()
        self.special_tokens = len(self.special_prefix_ids) + len(self.special_suffix_ids)

    @property
    def hidden_size(self) -> int:
        value = getattr(self.config, "hidden_size", None)
        if value is None:
            value = getattr(self.config, "d_model", None)
        if value is None:
            raise RuntimeError("Could not determine encoder hidden size from config")
        return int(value)

    def _infer_single_sequence_special_wrapper(self) -> Tuple[List[int], List[int]]:
        """Infer prefix/suffix special IDs through the public tokenizer API.

        Hugging Face Transformers v5 moved ``prepare_for_model`` and
        ``build_inputs_with_special_tokens`` away from TokenizersBackend.  The
        current XLM-R tokenizer is such a backend.  For this pipeline we only
        need single sequences, so we can safely learn the model's wrapper by
        comparing the same probe with and without special tokens.

        The check below is intentionally strict: all non-special probe IDs must
        survive unchanged and contiguously inside the wrapped sequence.  If a
        future tokenizer violates that assumption, we fail loudly rather than
        silently shifting first-subword alignment.
        """
        probes = ("a", "hello", "test", "1")
        last_error: Optional[str] = None

        for probe in probes:
            try:
                bare_enc = self.tokenizer(
                    probe,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                )
                wrapped_enc = self.tokenizer(
                    probe,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_special_tokens_mask=True,
                )
                bare = [int(x) for x in bare_enc["input_ids"]]
                wrapped = [int(x) for x in wrapped_enc["input_ids"]]
                special_mask = [int(x) for x in wrapped_enc["special_tokens_mask"]]
            except Exception as exc:
                last_error = f"probe {probe!r}: {type(exc).__name__}: {exc}"
                continue

            if not bare:
                last_error = f"probe {probe!r}: tokenizer produced no content IDs"
                continue
            if len(wrapped) != len(special_mask):
                last_error = (
                    f"probe {probe!r}: input_ids/special_tokens_mask length mismatch "
                    f"({len(wrapped)} != {len(special_mask)})"
                )
                continue

            content_positions = [i for i, flag in enumerate(special_mask) if flag == 0]
            if not content_positions:
                last_error = f"probe {probe!r}: wrapped sequence has no content positions"
                continue

            first, last = content_positions[0], content_positions[-1]
            expected_positions = list(range(first, last + 1))
            if content_positions != expected_positions:
                last_error = (
                    f"probe {probe!r}: special tokens are interleaved with content; "
                    "this pipeline requires a prefix/suffix single-sequence wrapper"
                )
                continue

            wrapped_content = wrapped[first : last + 1]
            if wrapped_content != bare:
                last_error = (
                    f"probe {probe!r}: adding special tokens changed content IDs "
                    f"({bare!r} -> {wrapped_content!r})"
                )
                continue

            prefix = wrapped[:first]
            suffix = wrapped[last + 1 :]

            # Cross-check against num_special_tokens_to_add when the tokenizer
            # implements it.  Do not depend on it for construction because the
            # public tokenization result is the source of truth in v5.
            try:
                reported = int(self.tokenizer.num_special_tokens_to_add(pair=False))
            except Exception:
                reported = len(prefix) + len(suffix)
            observed = len(prefix) + len(suffix)
            if reported != observed:
                last_error = (
                    f"probe {probe!r}: tokenizer reports {reported} special tokens "
                    f"but public encoding produced {observed}"
                )
                continue

            return prefix, suffix

        raise RuntimeError(
            "Could not infer the tokenizer's single-sequence special-token wrapper "
            "without changing content token IDs. This is required to preserve NER "
            "word/subword alignment. "
            + (f"Last tokenizer error: {last_error}" if last_error else "")
        )

    def _prepare_chunk(self, token_ids: Sequence[int]) -> dict:
        content = [int(x) for x in token_ids]
        input_ids = self.special_prefix_ids + content + self.special_suffix_ids
        special_tokens_mask = (
            [1] * len(self.special_prefix_ids)
            + [0] * len(content)
            + [1] * len(self.special_suffix_ids)
        )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "special_tokens_mask": special_tokens_mask,
        }

    def _select_hidden(self, outputs):
        if self.hidden_representation == "last":
            return outputs.last_hidden_state
        states = outputs.hidden_states
        if states is None or len(states) < 4:
            raise RuntimeError(
                f"--hidden-representation {self.hidden_representation} needs at "
                "least four hidden states"
            )
        stack = torch.stack(states[-4:], dim=0)
        if self.hidden_representation == "sum-last4":
            return stack.sum(dim=0)
        if self.hidden_representation == "mean-last4":
            return stack.mean(dim=0)
        raise ValueError(self.hidden_representation)

    def encode_chunk_batch(self, chunks: Sequence[Sequence[int]]) -> List["torch.Tensor"]:
        if not chunks:
            return []
        prepared = [self._prepare_chunk(c) for c in chunks]
        max_len = max(len(x["input_ids"]) for x in prepared)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        input_ids = []
        attention = []
        special_masks = []
        for x in prepared:
            n = len(x["input_ids"])
            pad = max_len - n
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            attention.append(x["attention_mask"] + [0] * pad)
            # Treat padding as special/non-text so it can never be returned.
            special_masks.append(x["special_tokens_mask"] + [1] * pad)

        ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attn_t = torch.tensor(attention, dtype=torch.long, device=self.device)
        need_hidden = self.hidden_representation != "last"
        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids_t,
                attention_mask=attn_t,
                output_hidden_states=need_hidden,
                return_dict=True,
            )
            reps = self._select_hidden(outputs)

        out: List[torch.Tensor] = []
        for i, (chunk, sm) in enumerate(zip(chunks, special_masks)):
            content_positions = [
                j for j, flag in enumerate(sm) if flag == 0 and attention[i][j] == 1
            ]
            if len(content_positions) != len(chunk):
                raise RuntimeError(
                    "Tokenizer special-token accounting failed: expected "
                    f"{len(chunk)} text tokens, found {len(content_positions)}"
                )
            idx = torch.tensor(content_positions, dtype=torch.long, device=reps.device)
            # Cache as CPU float32 so different folds can reuse embeddings without
            # keeping the entire corpus on GPU/MPS.
            out.append(reps[i].index_select(0, idx).float().cpu())
        return out


def embed_bundles(
    bundles: Sequence[FileBundle],
    encoder: HFWordEncoder,
    encoder_batch_size: int,
) -> dict:
    """Encode every text chunk exactly once and cache word reps on CPU."""
    flat: List[Tuple[EncodedSet, int, List[int]]] = []
    for bundle in bundles:
        for item in bundle.encoded:
            for ci, chunk in enumerate(item.chunks):
                flat.append((item, ci, chunk))

    chunk_reps: Dict[Tuple[int, int], torch.Tensor] = {}
    n_forward = 0
    batch_size = max(1, int(encoder_batch_size))
    for start in range(0, len(flat), batch_size):
        block = flat[start : start + batch_size]
        outputs = encoder.encode_chunk_batch([x[2] for x in block])
        n_forward += 1
        for (item, ci, _), rep in zip(block, outputs):
            chunk_reps[(id(item), ci)] = rep
        print(
            f"\rEncoding HF chunks: {min(start + len(block), len(flat))}/{len(flat)}",
            end="",
            flush=True,
        )
    if flat:
        print()

    embedded_words = 0
    dropped_words = 0
    for bundle in bundles:
        for item in bundle.encoded:
            parts = [chunk_reps[(id(item), ci)] for ci in range(len(item.chunks))]
            subword_reps = torch.cat(parts, dim=0) if parts else torch.empty((0, encoder.hidden_size))
            word_reps = torch.zeros(
                (len(item.words), encoder.hidden_size), dtype=torch.float32
            )
            valid = [False] * len(item.words)
            for wi, (first, _last) in enumerate(item.word_subword_span):
                if 0 <= first < subword_reps.size(0):
                    word_reps[wi] = subword_reps[first]
                    valid[wi] = True
                    embedded_words += 1
                else:
                    dropped_words += 1
            item.word_reps = word_reps
            item.valid_word_mask = valid

    return {
        "chunks": len(flat),
        "encoder_forward_calls": n_forward,
        "embedded_words": embedded_words,
        "dropped_words": dropped_words,
        "cache_device": "cpu",
        "cache_dtype": "float32",
    }


# ===========================================================================
# Few-shot heads: simple ProtoBERT + SuperCD-style NNShot
# ===========================================================================
def maybe_normalize(x: "torch.Tensor", enabled: bool) -> "torch.Tensor":
    if not enabled:
        return x
    return F.normalize(x, p=2, dim=-1)


def negative_squared_euclidean(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Return [len(a), len(b)] without constructing [A,B,H]."""
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1).unsqueeze(0)
    d2 = (a2 + b2 - 2.0 * (a @ b.transpose(0, 1))).clamp_min_(0.0)
    return -d2


class ProtoBERTHead:
    """Inference-only prototypical token classifier implemented in plain torch."""

    def __init__(
        self,
        support_reps: "torch.Tensor",
        support_labels: "torch.Tensor",
        n_classes: int,
        normalize_embeddings: bool,
    ):
        self.normalize_embeddings = normalize_embeddings
        support_reps = maybe_normalize(support_reps, normalize_embeddings)
        prototypes = []
        for c in range(n_classes):
            mask = support_labels == c
            if not bool(mask.any()):
                raise ValueError(f"ProtoBERT class {c} has no support token")
            prototypes.append(support_reps[mask].mean(dim=0))
        self.prototypes = torch.stack(prototypes, dim=0)

    def logits(self, query_reps: "torch.Tensor") -> "torch.Tensor":
        q = maybe_normalize(query_reps, self.normalize_embeddings)
        return negative_squared_euclidean(q, self.prototypes)

    def predict(self, query_reps: "torch.Tensor") -> "torch.Tensor":
        return self.logits(query_reps).argmax(dim=1)


class NNShotHead:
    """Nearest-support-token class emissions, following SuperCD/StructShot logic."""

    def __init__(
        self,
        support_reps: "torch.Tensor",
        support_labels: "torch.Tensor",
        n_classes: int,
        normalize_embeddings: bool,
        distance_budget_mb: int,
    ):
        self.normalize_embeddings = normalize_embeddings
        self.support_reps = maybe_normalize(support_reps, normalize_embeddings)
        self.support_labels = support_labels
        self.n_classes = int(n_classes)
        self.budget_bytes = max(0, int(distance_budget_mb)) * 1024 * 1024
        self.class_support = []
        for c in range(self.n_classes):
            reps = self.support_reps[self.support_labels == c]
            if reps.numel() == 0:
                raise ValueError(f"NNShot class {c} has no support token")
            self.class_support.append(reps)

    def _rows_for(self, n_support_class: int, elem_size: int) -> Optional[int]:
        if self.budget_bytes <= 0:
            return None
        per_row = max(1, int(n_support_class) * int(elem_size))
        return max(1, self.budget_bytes // per_row)

    def emissions(self, query_reps: "torch.Tensor") -> "torch.Tensor":
        q = maybe_normalize(query_reps, self.normalize_embeddings)
        out = torch.empty((q.size(0), self.n_classes), dtype=q.dtype, device=q.device)
        for c, support_c in enumerate(self.class_support):
            rows = self._rows_for(support_c.size(0), q.element_size())
            if rows is None or rows >= q.size(0):
                out[:, c] = negative_squared_euclidean(q, support_c).max(dim=1).values
            else:
                vals = []
                for start in range(0, q.size(0), rows):
                    scores = negative_squared_euclidean(q[start : start + rows], support_c)
                    vals.append(scores.max(dim=1).values)
                out[:, c] = torch.cat(vals, dim=0)
        return out


# ===========================================================================
# StructShot decoder, adapted from ASAPP / SuperCD
# ===========================================================================
START_ID = 0
O_ID = 1


class ViterbiDecoder:
    """Generalized StructShot Viterbi decoder."""

    def __init__(self, n_tag: int, abstract_transitions: Sequence[float], tau: float):
        if n_tag < 4:
            raise ValueError(
                "StructShot needs START + O + at least two entity classes (n_tag >= 4)"
            )
        self.transitions = self.project_target_transitions(
            n_tag, abstract_transitions, tau
        )

    @staticmethod
    def project_target_transitions(
        n_tag: int, abstract_transitions: Sequence[float], tau: float
    ) -> "torch.Tensor":
        s_o, s_i, o_o, o_i, i_o, i_i, x_y = [float(x) for x in abstract_transitions]
        a = torch.eye(n_tag, dtype=torch.float32) * i_i
        b = torch.ones((n_tag, n_tag), dtype=torch.float32) * x_y / (n_tag - 3)
        c = torch.eye(n_tag, dtype=torch.float32) * x_y / (n_tag - 3)
        transitions = a + b - c
        transitions[START_ID, O_ID] = s_o
        transitions[START_ID, O_ID + 1 :] = s_i / (n_tag - 2)
        transitions[O_ID, O_ID] = o_o
        transitions[O_ID, O_ID + 1 :] = o_i / (n_tag - 2)
        transitions[O_ID + 1 :, O_ID] = i_o
        transitions[:, START_ID] = 0.0
        powered = torch.pow(transitions.clamp_min(0.0), float(tau))
        summed = powered.sum(dim=1, keepdim=True).clamp_min(1e-12)
        transitions = powered / summed
        transitions = torch.where(
            transitions > 0,
            transitions,
            torch.full_like(transitions, 1e-6),
        )
        return torch.log(transitions)

    def forward(self, scores: "torch.Tensor") -> "torch.Tensor":
        batch_size, sentence_len, _ = scores.size()
        transitions = self.transitions.to(scores.device, dtype=scores.dtype)
        transitions = transitions.expand(batch_size, sentence_len, -1, -1)
        emissions = scores.unsqueeze(2).expand_as(transitions)
        return transitions + emissions

    @staticmethod
    def viterbi(features: "torch.Tensor") -> "torch.Tensor":
        batch_size, sentence_len, ntags, _ = features.size()
        delta_t = features[:, 0, START_ID, :]
        deltas = [delta_t]
        for t in range(1, sentence_len):
            f_t = features[:, t]
            delta_t, _ = torch.max(f_t + delta_t.unsqueeze(2).expand_as(f_t), 1)
            deltas.append(delta_t)
        sequences = [torch.argmax(deltas[-1], 1, keepdim=True)]
        for t in reversed(range(sentence_len - 1)):
            f_prev = features[:, t + 1].gather(
                2, sequences[-1].unsqueeze(2).expand(batch_size, ntags, 1)
            ).squeeze(2)
            sequences.append(torch.argmax(f_prev + deltas[t], 1, keepdim=True))
        sequences.reverse()
        return torch.cat(sequences, dim=1)


def structshot_decode(
    emissions: "torch.Tensor",
    decoder: ViterbiDecoder,
) -> Tuple["torch.Tensor", int]:
    """
    Decode one word sequence. Emissions columns are [O, entity1, ...].
    StructShot adds a START column before Viterbi, exactly as the original code.
    Returns class ids in [0..C-1] and the number of unexpected START predictions.
    """
    if emissions.size(0) == 0:
        return torch.empty(0, dtype=torch.long, device=emissions.device), 0
    sent_probs = torch.softmax(emissions, dim=1)
    start_probs = torch.full(
        (emissions.size(0), 1), 1e-6, dtype=emissions.dtype, device=emissions.device
    )
    probs = torch.cat((start_probs, sent_probs), dim=1)
    features = decoder.forward(torch.log(probs.clamp_min(1e-12)).unsqueeze(0))
    vit = decoder.viterbi(features).view(-1)
    start_count = int((vit == START_ID).sum().item())
    # Valid StructShot labels are 1..C. START is not a target class; if numerical
    # degeneracy ever selects it, map it conservatively to O after counting it.
    class_ids = torch.where(vit > 0, vit - 1, torch.zeros_like(vit))
    return class_ids, start_count


# ===========================================================================
# Abstract transitions
# ===========================================================================
def _transition_counts(sequences: Sequence[Sequence[str]], smoothing: float = 0.0) -> dict:
    s_o = s_i = o_o = o_i = i_o = i_i = x_y = float(smoothing)
    for tags in sequences:
        if not tags:
            continue
        if tags[0] == "O":
            s_o += 1
        else:
            s_i += 1
        for p, n in zip(tags[:-1], tags[1:]):
            if p == "O":
                if n == "O":
                    o_o += 1
                else:
                    o_i += 1
            else:
                if n == "O":
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1
    return {
        "s_o": s_o,
        "s_i": s_i,
        "o_o": o_o,
        "o_i": o_i,
        "i_o": i_o,
        "i_i": i_i,
        "x_y": x_y,
        "smoothing": float(smoothing),
    }


def _counts_to_transitions(counts: dict) -> Optional[List[float]]:
    s_o, s_i = counts["s_o"], counts["s_i"]
    o_o, o_i = counts["o_o"], counts["o_i"]
    i_o, i_i, x_y = counts["i_o"], counts["i_i"], counts["x_y"]
    den = (s_o + s_i, o_o + o_i, i_o + i_i + x_y)
    if any(x <= 0 for x in den):
        return None
    return [
        s_o / den[0],
        s_i / den[0],
        o_o / den[1],
        o_i / den[1],
        i_o / den[2],
        i_i / den[2],
        x_y / den[2],
    ]


def abstract_transitions_from_sequences(
    sequences: Sequence[Sequence[str]], smoothing: float
) -> Tuple[List[float], dict]:
    raw = _transition_counts(sequences, 0.0)
    trans = _counts_to_transitions(raw)
    if trans is not None and all(math.isfinite(x) for x in trans):
        return trans, {
            "smoothed_fallback_used": False,
            "counts": raw,
            "smoothing_parameter": float(smoothing),
        }
    effective = float(smoothing) if smoothing > 0 else 1e-6
    counts = _transition_counts(sequences, effective)
    trans = _counts_to_transitions(counts)
    if trans is None:
        raise RuntimeError("Could not construct abstract StructShot transitions")
    return trans, {
        "smoothed_fallback_used": True,
        "counts": counts,
        "smoothing_parameter": effective,
    }


# ===========================================================================
# Prediction -> character spans
# ===========================================================================
def labels_to_entities(
    text: str,
    words: Sequence[Word],
    labels: Sequence[str],
    merge_gap: str,
) -> List[Entity]:
    ents: List[Entity] = []
    active_label: Optional[str] = None
    active_start: Optional[int] = None
    active_end: Optional[int] = None

    def flush():
        nonlocal active_label, active_start, active_end
        if active_label is not None and active_start is not None and active_end is not None:
            ents.append(
                Entity(
                    active_start,
                    active_end,
                    active_label,
                    text[active_start:active_end],
                    raw_label=active_label,
                )
            )
        active_label = None
        active_start = None
        active_end = None

    for w, lab in zip(words, labels):
        if lab == "O":
            flush()
            continue
        if active_label == lab and active_end is not None:
            gap = text[active_end : w.start]
            mergeable = gap.strip() == "" if merge_gap == "whitespace" else True
            if mergeable:
                active_end = w.end
                continue
        flush()
        active_label = lab
        active_start = w.start
        active_end = w.end
    flush()
    return ents


def ids_to_full_word_labels(
    item: EncodedSet,
    valid_word_indices: Sequence[int],
    pred_ids: Sequence[int],
    id2label: Dict[int, str],
) -> List[str]:
    labels = ["O"] * len(item.words)
    for wi, pid in zip(valid_word_indices, pred_ids):
        labels[int(wi)] = id2label.get(int(pid), "O")
    return labels


# ===========================================================================
# Metrics
# ===========================================================================
def entity_counter(
    file_name: str,
    uid: str,
    entities: Sequence[Entity],
    allowed_labels: set,
) -> Counter:
    return Counter(
        (file_name, uid, ent.start, ent.end, ent.label)
        for ent in entities
        if ent.label in allowed_labels
    )


def _zero_nervaluate_score(scenario: str) -> dict:
    return {
        "scenario": scenario,
        "correct": 0,
        "incorrect": 0,
        "partial": 0,
        "missed": 0,
        "spurious": 0,
        "actual": 0,
        "possible": 0,
        "reference_n": 0,
        "pred_n": 0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def _pack_nervaluate_result(result, scenario: str) -> dict:
    partial_credit = scenario in {"partial", "ent_type"}
    tp = float(result.correct) + (0.5 * float(result.partial) if partial_credit else 0.0)
    actual = int(result.actual)
    possible = int(result.possible)
    return {
        "scenario": scenario,
        "correct": int(result.correct),
        "incorrect": int(result.incorrect),
        "partial": int(result.partial),
        "missed": int(result.missed),
        "spurious": int(result.spurious),
        "actual": actual,
        "possible": possible,
        "reference_n": possible,
        "pred_n": actual,
        "tp": tp,
        "fp": float(actual) - tp,
        "fn": float(possible) - tp,
        "precision": float(result.precision),
        "recall": float(result.recall),
        "f1": float(result.f1),
    }


def _counter_to_nervaluate_docs(
    reference: Counter, pred: Counter
) -> Tuple[List[List[dict]], List[List[dict]]]:
    """Convert internal character spans to nervaluate dict-loader documents.

    Internal spans use Python's [start, end) convention. nervaluate uses an
    inclusive end offset, so end is converted to end-1. Each (file, record_uid)
    is kept as a separate document so entities from different utterances can
    never match each other.
    """
    doc_keys = sorted(
        {(k[0], k[1]) for k in reference} | {(k[0], k[1]) for k in pred}
    )
    if not doc_keys:
        doc_keys = [("__EMPTY__", "__EMPTY__")]

    def one_side(counter: Counter) -> List[List[dict]]:
        grouped: Dict[Tuple[str, str], List[dict]] = {key: [] for key in doc_keys}
        for key, count in counter.items():
            file_name, uid, start, end, label = key
            if int(end) <= int(start):
                continue
            entity = {
                "label": str(label),
                "start": int(start),
                "end": int(end) - 1,
            }
            for _ in range(int(count)):
                grouped[(file_name, uid)].append(dict(entity))
        for ents in grouped.values():
            ents.sort(key=lambda e: (e["start"], e["end"], e["label"]))
        return [grouped[key] for key in doc_keys]

    return one_side(reference), one_side(pred)


def nervaluate_evaluate_counters(
    reference: Counter,
    pred: Counter,
    allowed_labels: Sequence[str],
    min_overlap_percentage: float = 1.0,
) -> dict:
    """Run nervaluate once and return overall and per-label results."""
    require_nervaluate()
    labels = sorted(set(str(x) for x in allowed_labels))
    true_docs, pred_docs = _counter_to_nervaluate_docs(reference, pred)
    evaluator = NervaluateEvaluator(
        true_docs,
        pred_docs,
        tags=labels,
        loader="dict",
        min_overlap_percentage=float(min_overlap_percentage),
    )
    raw = evaluator.evaluate()

    overall = {}
    for scenario in NERVALUATE_SCENARIOS:
        result = raw["overall"].get(scenario)
        overall[scenario] = (
            _pack_nervaluate_result(result, scenario)
            if result is not None
            else _zero_nervaluate_score(scenario)
        )

    entities: Dict[str, Dict[str, dict]] = {}
    raw_entities = raw.get("entities", {})
    for label in labels:
        per_label = raw_entities.get(label, {})
        entities[label] = {}
        for scenario in NERVALUATE_SCENARIOS:
            result = per_label.get(scenario)
            entities[label][scenario] = (
                _pack_nervaluate_result(result, scenario)
                if result is not None
                else _zero_nervaluate_score(scenario)
            )

    return {"overall": overall, "entities": entities}


def filter_counter_label(counter: Counter, label: str) -> Counter:
    return Counter({k: v for k, v in counter.items() if k[-1] == label})


def write_csv(
    path: Path,
    rows: Sequence[dict],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            return
        seen: List[str] = []
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(x) for x in values]
    if not vals:
        return 0.0, 0.0
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, sd


def _pooled(rows: Sequence[dict]) -> Tuple[float, float, float]:
    tp = sum(float(r["tp"]) for r in rows)
    fp = sum(float(r["fp"]) for r in rows)
    fn = sum(float(r["fn"]) for r in rows)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def aggregate_average_metrics(fold_rows: Sequence[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in fold_rows:
        groups[(row["method"], row["scope"], row["scenario"])].append(row)
    out = []
    for (method, scope, scenario), rows in sorted(groups.items()):
        p_mean, p_sd = mean_sd([r["precision"] for r in rows])
        r_mean, r_sd = mean_sd([r["recall"] for r in rows])
        f_mean, f_sd = mean_sd([r["f1"] for r in rows])
        pooled_p, pooled_r, pooled_f = _pooled(rows)
        out.append(
            {
                "method": method,
                "scope": scope,
                "scenario": scenario,
                "n_folds": len(rows),
                "precision_mean": p_mean,
                "precision_sd": p_sd,
                "recall_mean": r_mean,
                "recall_sd": r_sd,
                "f1_mean": f_mean,
                "f1_sd": f_sd,
                "tp_pooled": sum(float(r["tp"]) for r in rows),
                "fp_pooled": sum(float(r["fp"]) for r in rows),
                "fn_pooled": sum(float(r["fn"]) for r in rows),
                "precision_pooled": pooled_p,
                "recall_pooled": pooled_r,
                "f1_pooled": pooled_f,
            }
        )
    return out


def aggregate_average_label_metrics(label_rows: Sequence[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in label_rows:
        groups[(row["method"], row["scope"], row["label"], row["scenario"])].append(row)
    out = []
    for (method, scope, label, scenario), rows in sorted(groups.items()):
        p_mean, p_sd = mean_sd([r["precision"] for r in rows])
        r_mean, r_sd = mean_sd([r["recall"] for r in rows])
        f_mean, f_sd = mean_sd([r["f1"] for r in rows])
        pooled_p, pooled_r, pooled_f = _pooled(rows)
        out.append(
            {
                "method": method,
                "scope": scope,
                "label": label,
                "scenario": scenario,
                "n_folds": len(rows),
                "precision_mean": p_mean,
                "precision_sd": p_sd,
                "recall_mean": r_mean,
                "recall_sd": r_sd,
                "f1_mean": f_mean,
                "f1_sd": f_sd,
                "tp_pooled": sum(float(r["tp"]) for r in rows),
                "fp_pooled": sum(float(r["fp"]) for r in rows),
                "fn_pooled": sum(float(r["fn"]) for r in rows),
                "precision_pooled": pooled_p,
                "recall_pooled": pooled_r,
                "f1_pooled": pooled_f,
            }
        )
    return out


def aggregate_tau_sweep(fold_rows: Sequence[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in fold_rows:
        groups[(row["method"], row["scope"], row["tau"], row["scenario"])].append(row)
    out = []
    for (method, scope, tau, scenario), rows in sorted(groups.items()):
        f_mean, f_sd = mean_sd([r["f1"] for r in rows])
        pooled_p, pooled_r, pooled_f = _pooled(rows)
        out.append(
            {
                "method": method,
                "scope": scope,
                "scenario": scenario,
                "tau": tau,
                "n_folds": len(rows),
                "precision_mean": mean_sd([r["precision"] for r in rows])[0],
                "recall_mean": mean_sd([r["recall"] for r in rows])[0],
                "f1_mean": f_mean,
                "f1_sd": f_sd,
                "precision_pooled": pooled_p,
                "recall_pooled": pooled_r,
                "f1_pooled": pooled_f,
            }
        )
    return out


def print_average_table(rows: Sequence[dict]) -> None:
    if not rows:
        return
    print("\nAverage nervaluate entity-level metrics across support folds")
    print("=" * 118)
    print(
        f"{'method':<14} {'scope':<18} {'scenario':<10} "
        f"{'P mean+-SD':>17} {'R mean+-SD':>17} {'F1 mean+-SD':>17} {'F1 pooled':>10}"
    )
    print("-" * 118)
    for r in rows:
        print(
            f"{r['method']:<14} {r['scope']:<18} {r['scenario']:<10} "
            f"{r['precision_mean']:.4f}+-{r['precision_sd']:.4f} "
            f"{r['recall_mean']:.4f}+-{r['recall_sd']:.4f} "
            f"{r['f1_mean']:.4f}+-{r['f1_sd']:.4f} {r['f1_pooled']:.4f}"
        )
    print("=" * 118)


def print_tau_table(rows: Sequence[dict]) -> None:
    if not rows:
        return
    print("\nStructShot tau sweep (nervaluate; SuperCD/ASAPP-style decoder)")
    print("=" * 94)
    print(
        f"{'tau':>8} {'scope':<18} {'scenario':<10} "
        f"{'P pooled':>10} {'R pooled':>10} {'F1 pooled':>10}"
    )
    print("-" * 94)
    for r in rows:
        print(
            f"{r['tau']:>8} {r['scope']:<18} {r['scenario']:<10} "
            f"{r['precision_pooled']:>10.4f} {r['recall_pooled']:>10.4f} "
            f"{r['f1_pooled']:>10.4f}"
        )
    print("=" * 94)


# ===========================================================================
# CLI / mappings
# ===========================================================================
def parse_tau_list(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _validated_target_mapping(mapping: dict, source_name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in mapping.items():
        key = str(k).strip().upper()
        if key == "WORK_OF_ART":
            key = "WORK OF ART"
        value = str(v).strip().upper()
        if value not in GDPR_ALF_TARGET_LABELS:
            raise ValueError(
                f"{source_name}: destination {value!r} is not a GDPR-ALF child target"
            )
        out[key] = value
    return out


def load_annotation_aliases(args) -> Dict[str, str]:
    aliases = _validated_target_mapping(
        DEFAULT_ANNOTATION_ALIASES, "DEFAULT_ANNOTATION_ALIASES"
    )
    if args.annotation_label_map_json is not None:
        with args.annotation_label_map_json.open("r", encoding="utf-8") as f:
            custom = json.load(f)
        if not isinstance(custom, dict):
            raise ValueError("--annotation-label-map-json must contain a JSON object")
        aliases.update(_validated_target_mapping(custom, "--annotation-label-map-json"))
    return aliases


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "LOO few-shot pseudonymization NER: native Hugging Face AutoModel + "
            "simple Torch ProtoBERT + SuperCD/StructShot NNShot/Viterbi."
        )
    )
    p.add_argument("--version", action="version", version=SCRIPT_VERSION)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--extensions", nargs="+", default=[".txt", ".alfrttm"])
    p.add_argument("--include-substrings", nargs="*", default=[])
    p.add_argument("--expected-files", type=int, default=0)
    p.add_argument("--annotation-marker", default="PersonData")
    p.add_argument("--strict-quoted-utterance", action="store_true", default=False)
    p.add_argument("--annotation-label-map-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--config-pipe", type=Path, default=SCRIPT_DIR / "config_pipe.py",
        help=("Path to config_pipe.py; default is the copy beside this script "
              "under pseudopipe/. HG_TOKEN is used for authenticated HF Hub requests."),
    )
    p.add_argument(
        "--logger-module", type=Path, default=SCRIPT_DIR / "loger.py",
        help=("Path to loger.py; default is the copy beside this script under "
              "pseudopipe/. It must define setup_logging(log_file, level)."),
    )
    p.add_argument(
        "--log-file", type=Path, default=None,
        help="Trace log path; default <output-dir>/fewshot_pseudonymization.log.",
    )

    p.add_argument(
        "--encoder-model",
        default="FacebookAI/xlm-roberta-large",
        help=(
            "Any Hugging Face AutoModel-compatible encoder. XLM-R checkpoints are "
            "first-class; no architecture flag is needed."
        ),
    )
    p.add_argument(
        "--hidden-representation",
        choices=["last", "mean-last4", "sum-last4"],
        default="last",
        help="'last' follows SuperCD/ASAPP; last-4 options are provided for comparison.",
    )
    p.add_argument(
        "--model-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument(
        "--tokenization", choices=["hf-batch", "per-word"], default="hf-batch"
    )
    p.add_argument("--word-regex", default=DEFAULT_WORD_REGEX)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument(
        "--encoder-batch-size",
        type=int,
        default=32,
        help="Number of model-sized text chunks encoded per HF forward pass.",
    )
    p.add_argument(
        "--query-chunk",
        type=int,
        default=64,
        help="Number of query records scored together after embeddings are cached; 0=all.",
    )
    p.add_argument(
        "--distance-budget-mb",
        type=int,
        default=256,
        help=(
            "NNShot score-matrix budget. Unlike v2 this bounds only [query, support] "
            "class-specific score matrices, never [query, support, hidden]. 0=unchunked."
        ),
    )
    p.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        default=True,
        help=(
            "Disable L2 normalization before ProtoBERT/NNShot distances. Default ON "
            "to follow the SuperCD/original StructShot NNShot metric."
        ),
    )
    p.add_argument(
        "--io-merge-gap",
        choices=["whitespace", "adjacent"],
        default="whitespace",
    )

    p.add_argument("--structshot-tau", type=float, default=0.05)
    p.add_argument("--tau-sweep", type=str, default="")
    p.add_argument(
        "--transition-source",
        choices=["auto", "support", "corpus", "json"],
        default="auto",
    )
    p.add_argument("--transition-corpus-dir", type=Path, default=None)
    p.add_argument("--transition-json", type=Path, default=None)
    p.add_argument(
        "--transition-smoothing",
        type=float,
        default=1.0,
        help="Fallback additive smoothing if an unsmoothed abstract ratio is undefined.",
    )
    p.add_argument(
        "--nervaluate-min-overlap",
        type=float,
        default=1.0,
        help=(
            "Minimum percentage overlap used by nervaluate for non-exact matches "
            "(1-100; default 1.0, matching nervaluate)."
        ),
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check input annotations and write audits without loading the encoder.",
    )
    return p


# ===========================================================================
# Fold helpers
# ===========================================================================
def collect_support_tensors(
    support_items: Sequence[EncodedSet],
    tag2id: Dict[str, int],
    device,
) -> Tuple["torch.Tensor", "torch.Tensor", Counter]:
    reps = []
    labels = []
    counts = Counter()
    for item in support_items:
        if item.word_reps is None or item.valid_word_mask is None:
            raise RuntimeError("Internal error: embeddings have not been cached")
        idx = [i for i, ok in enumerate(item.valid_word_mask) if ok]
        if not idx:
            continue
        reps.append(item.word_reps[idx])
        for i in idx:
            lab = item.word_labels[i]
            if lab not in tag2id:
                raise RuntimeError(f"support label {lab!r} missing from tag2id")
            labels.append(tag2id[lab])
            counts[lab] += 1
    if not reps:
        raise RuntimeError("support fold has no valid word representations")
    return (
        torch.cat(reps, dim=0).to(device),
        torch.tensor(labels, dtype=torch.long, device=device),
        counts,
    )


def valid_query_reps(item: EncodedSet, device) -> Tuple["torch.Tensor", List[int]]:
    if item.word_reps is None or item.valid_word_mask is None:
        raise RuntimeError("Internal error: embeddings have not been cached")
    idx = [i for i, ok in enumerate(item.valid_word_mask) if ok]
    if not idx:
        return torch.empty((0, item.word_reps.size(1)), device=device), []
    return item.word_reps[idx].to(device), idx


def load_transition_corpus_sequences(
    directory: Path,
    extensions: set,
    marker: str,
    aliases: Dict[str, str],
    strict: bool,
    word_regex: str,
) -> List[List[str]]:
    paths = discover_files(directory, extensions, [])
    if not paths:
        raise RuntimeError(f"No transition corpus files found under {directory}")
    seqs: List[List[str]] = []
    for path in paths:
        recs, _ = load_input_validation_records(
            path, marker, aliases, set(GDPR_ALF_TARGET_LABELS), strict
        )
        for rec in recs:
            words = tokenize_to_words(rec.text, word_regex)
            if not words:
                continue
            labels, _ = gold_word_labels(rec.text, words, rec.reference_entities)
            seqs.append(labels)
    if not seqs:
        raise RuntimeError("Transition corpus produced no usable word sequences")
    return seqs


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resolved_log_file = args.log_file or (args.output_dir / "fewshot_pseudonymization.log")
    logger, resolved_log_file = setup_trace_logging(args.logger_module, resolved_log_file)
    logger.info("Script version: %s", SCRIPT_VERSION)
    logger.info("Input directory: %s", args.input_dir)
    logger.info("Output directory: %s", args.output_dir)

    hf_token = load_hf_token_from_config(args.config_pipe)
    if hf_token:
        logger.info("Hugging Face authentication configured from %s", args.config_pipe)
    else:
        logger.warning(
            "No usable HG_TOKEN in %s; Hugging Face Hub requests will be unauthenticated",
            args.config_pipe,
        )

    extensions = {
        x.lower() if x.startswith(".") else "." + x.lower() for x in args.extensions
    }
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    input_paths = discover_files(
        args.input_dir, extensions=extensions, include_substrings=args.include_substrings
    )
    if not input_paths:
        raise RuntimeError(f"No input files found under {args.input_dir}")
    if args.expected_files > 0 and len(input_paths) != args.expected_files:
        raise RuntimeError(
            f"Expected {args.expected_files} files, found {len(input_paths)}:\n"
            + "\n".join(map(str, input_paths))
        )

    logger.info("Runtime: native Hugging Face + local Torch ProtoBERT + StructShot")
    logger.info("Input sessions / support folds: %d", len(input_paths))
    print(f"Script version: {SCRIPT_VERSION}")
    print("Runtime: native Hugging Face + local Torch ProtoBERT + StructShot")
    print(f"Input sessions / support folds: {len(input_paths)}")
    for path in input_paths:
        print(f"  - {path}")

    aliases = load_annotation_aliases(args)
    bundles: List[FileBundle] = []
    input_annotation_audit: List[dict] = []
    input_annotation_summary: List[dict] = []
    raw_annotation_count = 0
    in_schema_annotation_count = 0

    for input_path in input_paths:
        records, audit = load_input_validation_records(
            input_path,
            marker=args.annotation_marker,
            aliases=aliases,
            target_labels=set(GDPR_ALF_TARGET_LABELS),
            strict_quoted_utterance=args.strict_quoted_utterance,
        )
        status_counts = Counter(r["status"] for r in audit)
        raw_annotation_count += len(audit)
        in_schema_annotation_count += status_counts["in_gdpr_alf_child_schema"]
        input_annotation_audit.extend(audit)
        input_annotation_summary.append(
            {
                "file": input_path.name,
                "input_path": str(input_path),
                "records": len(records),
                "annotations_total": len(audit),
                "annotations_in_gdpr_alf_child_schema": status_counts[
                    "in_gdpr_alf_child_schema"
                ],
                "annotations_out_of_gdpr_alf_child_schema": status_counts[
                    "out_of_gdpr_alf_child_schema"
                ],
            }
        )
        bundles.append(FileBundle(input_path.name, input_path, records))

    if raw_annotation_count == 0:
        raise RuntimeError(
            "No inline [PersonData]mention[LABEL] annotations were found."
        )
    if in_schema_annotation_count == 0:
        raise RuntimeError("Annotations were found, but none match the GDPR-ALF child schema")

    write_csv(args.output_dir / "input_annotation_audit.csv", input_annotation_audit)
    write_csv(args.output_dir / "input_annotation_summary.csv", input_annotation_summary)
    schema_rows = []
    for parent, children in GDPR_ALF_PARENT_CHILDREN.items():
        if children:
            for child in children:
                schema_rows.append(
                    {"parent_label": parent, "child_label": child, "is_child_target": True}
                )
        else:
            schema_rows.append(
                {"parent_label": parent, "child_label": "", "is_child_target": False}
            )
    write_csv(args.output_dir / "gdpr_alf_target_schema.csv", schema_rows)

    if args.preflight_only:
        logger.info("Preflight complete; encoder not loaded")
        print("Preflight complete; encoder not loaded.")
        return 0

    require_torch()
    require_nervaluate()
    device = resolve_device(args.device)
    logger.info("Selected device: %s", device)
    print(f"\nDevice: {device}")
    logger.info("Loading Hugging Face encoder: %s", args.encoder_model)
    encoder = HFWordEncoder(
        args.encoder_model,
        device=device,
        hidden_representation=args.hidden_representation,
        model_dtype=args.model_dtype,
        trust_remote_code=args.trust_remote_code,
        hf_token=hf_token,
    )
    logger.info(
        "Encoder loaded: model_type=%s hidden_size=%s representation=%s",
        getattr(encoder.config, "model_type", "?"), encoder.hidden_size, args.hidden_representation,
    )
    tokenizer = encoder.tokenizer
    if args.tokenization == "hf-batch" and not getattr(tokenizer, "is_fast", False):
        print(
            "NOTE: tokenizer is not fast; switching --tokenization to per-word.",
            file=sys.stderr,
        )
        args.tokenization = "per-word"

    cap, cap_src = resolve_max_length_cap(tokenizer, encoder.config)
    if cap is not None and args.max_length > cap:
        raise SystemExit(
            f"ERROR: --max-length {args.max_length} exceeds {args.encoder_model} "
            f"capacity {cap} ({cap_src})"
        )
    max_content_tokens = args.max_length - encoder.special_tokens
    if max_content_tokens < 1:
        raise SystemExit(
            f"ERROR: --max-length {args.max_length} leaves no text positions after "
            f"{encoder.special_tokens} special tokens"
        )

    print(
        f"Encoder: {args.encoder_model}\n"
        f"  model_type             {getattr(encoder.config, 'model_type', '?')}\n"
        f"  hidden_size            {encoder.hidden_size}\n"
        f"  hidden representation  {args.hidden_representation}\n"
        f"  tokenizer fast         {getattr(tokenizer, 'is_fast', False)}\n"
        f"  max_length             {args.max_length}\n"
        f"  special tokens         {encoder.special_tokens}\n"
        f"  normalize embeddings   {args.normalize_embeddings}"
    )

    # ------------------------------- tokenize / align once
    encoding_audit: List[dict] = []
    logger.info("Tokenizing annotation-stripped utterances")
    print("\nTokenizing annotation-stripped utterances...")
    for i, bundle in enumerate(bundles, start=1):
        print(f"  [{i}/{len(bundles)}] {bundle.name}")
        encoded = []
        for rec in bundle.records:
            item = encode_record(
                rec,
                tokenizer,
                args.max_length,
                max_content_tokens,
                args.word_regex,
                args.tokenization,
                encoding_audit,
            )
            if item is None:
                bundle.tokens_dropped_records += 1
                continue
            bundle.dropped_words += sum(1 for a, _ in item.word_subword_span if a < 0)
            bundle.boundary_expansions += item.boundary_expansions
            bundle.n_chunks += len(item.chunks)
            bundle.n_text_subwords += sum(len(c) for c in item.chunks)
            encoded.append(item)
        bundle.encoded = encoded

    logger.info("Encoding all chunks once; embeddings cached on CPU and reused across folds")
    print("\nEncoding all chunks once; embeddings are cached on CPU and reused across folds...")
    embedding_stats = embed_bundles(bundles, encoder, args.encoder_batch_size)

    boundary_audit = []
    for bundle in bundles:
        valid_words = sum(
            sum(bool(x) for x in (item.valid_word_mask or [])) for item in bundle.encoded
        )
        boundary_audit.append(
            {
                "file": bundle.name,
                "records_encoded": len(bundle.encoded),
                "records_skipped_no_words": bundle.tokens_dropped_records,
                "words_without_subword": bundle.dropped_words,
                "hf_chunks": bundle.n_chunks,
                "text_subwords": bundle.n_text_subwords,
                "valid_first_subword_words": valid_words,
                "reference_entities_not_on_word_boundaries": bundle.boundary_expansions,
            }
        )

    # ------------------------------- transitions external setup
    transition_source = args.transition_source
    if transition_source == "auto":
        if args.transition_json is not None:
            transition_source = "json"
        elif args.transition_corpus_dir is not None:
            transition_source = "corpus"
        else:
            transition_source = "support"
    if transition_source == "json" and args.transition_json is None:
        raise RuntimeError("--transition-source json requires --transition-json")
    if transition_source == "corpus" and args.transition_corpus_dir is None:
        raise RuntimeError("--transition-source corpus requires --transition-corpus-dir")

    transition_values: Optional[List[float]] = None
    corpus_sequences: Optional[List[List[str]]] = None
    if transition_source == "json":
        with args.transition_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        raw = payload.get("abstract_transitions", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw, (list, tuple)) or len(raw) != 7:
            raise ValueError("--transition-json must contain 7 abstract transition values")
        transition_values = [float(x) for x in raw]
    elif transition_source == "corpus":
        corpus_sequences = load_transition_corpus_sequences(
            args.transition_corpus_dir,
            extensions,
            args.annotation_marker,
            aliases,
            args.strict_quoted_utterance,
            args.word_regex,
        )
    logger.info("Abstract transition source: %s", transition_source)
    print(f"Abstract transition source: {transition_source}")

    tau_list = [float(args.structshot_tau)]
    for t in parse_tau_list(args.tau_sweep):
        if all(abs(t - u) > 1e-12 for u in tau_list):
            tau_list.append(t)

    rng = random.Random(args.seed)
    support_order = list(range(len(bundles)))
    rng.shuffle(support_order)

    pair_rows: List[dict] = []
    fold_rows: List[dict] = []
    label_rows: List[dict] = []
    tau_sweep_rows: List[dict] = []
    support_shot_rows: List[dict] = []
    transition_rows: List[dict] = []
    fold_run_rows: List[dict] = []
    start_predictions_total = 0

    methods = ["protobert", "structshot"]

    # ------------------------------- LOO folds
    for fold_number, support_idx in enumerate(support_order, start=1):
        support_bundle = bundles[support_idx]
        query_bundles = [b for j, b in enumerate(bundles) if j != support_idx]
        support_items = support_bundle.encoded
        if not support_items:
            print(f"Fold {fold_number}: support {support_bundle.name} has no encoded items; skip")
            continue

        # Only labels with at least one surviving first-subword support word are
        # predictable. O is pinned to id 0.
        support_valid_labels = []
        for item in support_items:
            valid = item.valid_word_mask or []
            for wi, ok in enumerate(valid):
                if ok:
                    support_valid_labels.append(item.word_labels[wi])
        support_label_counts = Counter(support_valid_labels)
        if support_label_counts["O"] == 0:
            print(
                f"Fold {fold_number}: support={support_bundle.name} has no valid O "
                "word after tokenization; fold is methodologically invalid and is skipped.",
                file=sys.stderr,
            )
            continue

        observed_entity_classes = sorted(
            x for x in support_label_counts if x != "O" and support_label_counts[x] > 0
        )
        tag2id = {"O": 0, **{lab: i + 1 for i, lab in enumerate(observed_entity_classes)}}
        id2tag = {v: k for k, v in tag2id.items()}
        observed_labels = set(observed_entity_classes)

        logger.info(
            "Fold %d/%d support=%s query_files=%d observed_entity_labels=%d",
            fold_number, len(bundles), support_bundle.name, len(query_bundles), len(observed_entity_classes),
        )
        print(
            f"\nFold {fold_number}/{len(bundles)} support={support_bundle.name}\n"
            f"  observed entity labels ({len(observed_entity_classes)}): "
            + (", ".join(observed_entity_classes) if observed_entity_classes else "<none>")
        )

        support_reps, support_ids, support_counts = collect_support_tensors(
            support_items, tag2id, device
        )
        for lab in ["O"] + observed_entity_classes:
            support_shot_rows.append(
                {
                    "fold": fold_number,
                    "support_file": support_bundle.name,
                    "label": lab,
                    "label_id": tag2id[lab],
                    "support_word_count": support_counts[lab],
                    "support_record_count": len(support_items),
                }
            )

        proto = ProtoBERTHead(
            support_reps,
            support_ids,
            n_classes=len(tag2id),
            normalize_embeddings=args.normalize_embeddings,
        )
        nnshot = NNShotHead(
            support_reps,
            support_ids,
            n_classes=len(tag2id),
            normalize_embeddings=args.normalize_embeddings,
            distance_budget_mb=args.distance_budget_mb,
        )

        structshot_available = len(observed_entity_classes) >= 2
        methods_this_fold = ["protobert"] + (["structshot"] if structshot_available else [])
        tau_decoders: Dict[float, ViterbiDecoder] = {}
        abstract = None
        trans_info = None
        if structshot_available:
            if transition_source == "json":
                abstract = list(transition_values or [])
                trans_info = {
                    "smoothed_fallback_used": False,
                    "counts": {},
                    "smoothing_parameter": None,
                }
            else:
                if transition_source == "corpus":
                    sequences = corpus_sequences or []
                else:
                    sequences = [item.word_labels for item in support_items]
                abstract, trans_info = abstract_transitions_from_sequences(
                    sequences, args.transition_smoothing
                )
            for tau in tau_list:
                tau_decoders[tau] = ViterbiDecoder(
                    len(tag2id) + 1, abstract, tau
                )
            counts = (trans_info or {}).get("counts", {})
            transition_rows.append(
                {
                    "fold": fold_number,
                    "support_file": support_bundle.name,
                    "transition_source": transition_source,
                    "n_entity_classes": len(observed_entity_classes),
                    "entity_classes": "|".join(observed_entity_classes),
                    "tau_default": args.structshot_tau,
                    "tau_values": "|".join(str(t) for t in tau_list),
                    "s_o": abstract[0],
                    "s_i": abstract[1],
                    "o_o": abstract[2],
                    "o_i": abstract[3],
                    "i_o": abstract[4],
                    "i_i": abstract[5],
                    "x_y": abstract[6],
                    "smoothed_fallback_used": (trans_info or {}).get(
                        "smoothed_fallback_used", False
                    ),
                    "count_s_o": counts.get("s_o", ""),
                    "count_s_i": counts.get("s_i", ""),
                    "count_o_o": counts.get("o_o", ""),
                    "count_o_i": counts.get("o_i", ""),
                    "count_i_o": counts.get("i_o", ""),
                    "count_i_i": counts.get("i_i", ""),
                    "count_x_y": counts.get("x_y", ""),
                }
            )
        else:
            print(
                "  StructShot skipped: it needs at least two support entity classes "
                "because the canonical transition projection divides by N_entity-1.",
                file=sys.stderr,
            )

        all_query_items: List[Tuple[FileBundle, EncodedSet, ValidationRecord]] = []
        for qb in query_bundles:
            by_uid = {r.uid: r for r in qb.records}
            for item in qb.encoded:
                rec = by_uid.get(item.uid)
                if rec is not None:
                    all_query_items.append((qb, item, rec))

        chunk_size = args.query_chunk if args.query_chunk > 0 else max(1, len(all_query_items))
        per_record_pred: Dict[str, Dict[str, List[Entity]]] = defaultdict(dict)
        per_record_tau_pred: Dict[float, Dict[str, List[Entity]]] = defaultdict(dict)
        n_scored_records = 0

        with torch.inference_mode():
            for start in range(0, len(all_query_items), max(1, chunk_size)):
                block = all_query_items[start : start + max(1, chunk_size)]
                reps_parts = []
                meta = []
                for qb, item, rec in block:
                    q, valid_idx = valid_query_reps(item, device)
                    reps_parts.append(q)
                    meta.append((qb, item, rec, valid_idx, int(q.size(0))))
                nonempty = [x for x in reps_parts if x.size(0) > 0]
                if nonempty:
                    q_all = torch.cat(nonempty, dim=0)
                    proto_all = proto.predict(q_all)
                    nn_emissions_all = nnshot.emissions(q_all) if structshot_available else None
                else:
                    q_all = torch.empty((0, encoder.hidden_size), device=device)
                    proto_all = torch.empty(0, dtype=torch.long, device=device)
                    nn_emissions_all = None

                cursor = 0
                for qb, item, rec, valid_idx, n in meta:
                    if n > 0:
                        pids = proto_all[cursor : cursor + n].detach().cpu().tolist()
                    else:
                        pids = []
                    p_labels = ids_to_full_word_labels(item, valid_idx, pids, id2tag)
                    per_record_pred[rec.uid]["protobert"] = labels_to_entities(
                        rec.text, item.words, p_labels, args.io_merge_gap
                    )

                    if structshot_available:
                        em = nn_emissions_all[cursor : cursor + n]
                        for tau in tau_list:
                            sids, start_count = structshot_decode(em, tau_decoders[tau])
                            start_predictions_total += start_count
                            s_labels = ids_to_full_word_labels(
                                item,
                                valid_idx,
                                sids.detach().cpu().tolist(),
                                id2tag,
                            )
                            ents = labels_to_entities(
                                rec.text, item.words, s_labels, args.io_merge_gap
                            )
                            per_record_tau_pred[tau][rec.uid] = ents
                            if abs(tau - args.structshot_tau) <= 1e-12:
                                per_record_pred[rec.uid]["structshot"] = ents
                    cursor += n
                    n_scored_records += 1

        # --------------------------- scoring via nervaluate
        fold_reference = {scope: Counter() for scope in ("support_observed", "all_schema")}
        fold_pred = {
            (m, scope): Counter()
            for m in methods_this_fold
            for scope in ("support_observed", "all_schema")
        }

        for qb in query_bundles:
            pair_reference = {scope: Counter() for scope in ("support_observed", "all_schema")}
            pair_pred = {
                (m, scope): Counter()
                for m in methods_this_fold
                for scope in ("support_observed", "all_schema")
            }
            by_uid = {r.uid: r for r in qb.records}
            for item in qb.encoded:
                rec = by_uid.get(item.uid)
                if rec is None:
                    continue
                allowed_by_scope = {
                    "support_observed": observed_labels,
                    "all_schema": set(GDPR_ALF_TARGET_LABELS),
                }
                for scope, allowed in allowed_by_scope.items():
                    g = entity_counter(qb.name, rec.uid, rec.reference_entities, allowed)
                    pair_reference[scope].update(g)
                    fold_reference[scope].update(g)
                    for method in methods_this_fold:
                        p = entity_counter(
                            qb.name,
                            rec.uid,
                            per_record_pred[rec.uid].get(method, []),
                            allowed,
                        )
                        pair_pred[(method, scope)].update(p)
                        fold_pred[(method, scope)].update(p)

            for scope in ("support_observed", "all_schema"):
                labels_for_scope = (
                    sorted(observed_labels)
                    if scope == "support_observed"
                    else sorted(GDPR_ALF_TARGET_LABELS)
                )
                for method in methods_this_fold:
                    evaluation = nervaluate_evaluate_counters(
                        pair_reference[scope],
                        pair_pred[(method, scope)],
                        labels_for_scope,
                        args.nervaluate_min_overlap,
                    )
                    for scenario in NERVALUATE_SCENARIOS:
                        pair_rows.append(
                            {
                                "fold": fold_number,
                                "support_file": support_bundle.name,
                                "query_file": qb.name,
                                "method": method,
                                "scope": scope,
                                "scenario": scenario,
                                "n_observed_support_labels": len(observed_labels),
                                "observed_support_labels": "|".join(sorted(observed_labels)),
                                **evaluation["overall"][scenario],
                            }
                        )

        for scope in ("support_observed", "all_schema"):
            labels_for_scope = (
                sorted(observed_labels)
                if scope == "support_observed"
                else sorted(GDPR_ALF_TARGET_LABELS)
            )
            for method in methods_this_fold:
                evaluation = nervaluate_evaluate_counters(
                    fold_reference[scope],
                    fold_pred[(method, scope)],
                    labels_for_scope,
                    args.nervaluate_min_overlap,
                )
                for scenario in NERVALUATE_SCENARIOS:
                    fold_rows.append(
                        {
                            "fold": fold_number,
                            "support_file": support_bundle.name,
                            "method": method,
                            "scope": scope,
                            "scenario": scenario,
                            "n_query_files": len(query_bundles),
                            "n_observed_support_labels": len(observed_labels),
                            "observed_support_labels": "|".join(sorted(observed_labels)),
                            "tau": args.structshot_tau if method == "structshot" else "",
                            **evaluation["overall"][scenario],
                        }
                    )
                for label in labels_for_scope:
                    for scenario in NERVALUATE_SCENARIOS:
                        label_rows.append(
                            {
                                "fold": fold_number,
                                "support_file": support_bundle.name,
                                "method": method,
                                "scope": scope,
                                "label": label,
                                "scenario": scenario,
                                **evaluation["entities"][label][scenario],
                            }
                        )

        if structshot_available:
            for tau in tau_list:
                for scope in ("support_observed", "all_schema"):
                    allowed = (
                        observed_labels
                        if scope == "support_observed"
                        else set(GDPR_ALF_TARGET_LABELS)
                    )
                    labels_for_scope = sorted(allowed)
                    ref_c = Counter()
                    pred_c = Counter()
                    for qb, item, rec in all_query_items:
                        ref_c.update(
                            entity_counter(qb.name, rec.uid, rec.reference_entities, allowed)
                        )
                        pred_c.update(
                            entity_counter(
                                qb.name,
                                rec.uid,
                                per_record_tau_pred[tau].get(rec.uid, []),
                                allowed,
                            )
                        )
                    evaluation = nervaluate_evaluate_counters(
                        ref_c, pred_c, labels_for_scope, args.nervaluate_min_overlap
                    )
                    for scenario in NERVALUATE_SCENARIOS:
                        tau_sweep_rows.append(
                            {
                                "fold": fold_number,
                                "support_file": support_bundle.name,
                                "method": "structshot",
                                "scope": scope,
                                "scenario": scenario,
                                "tau": tau,
                                "tau_is_default": abs(tau - args.structshot_tau) <= 1e-12,
                                "n_observed_support_labels": len(observed_labels),
                                **evaluation["overall"][scenario],
                            }
                        )

        fold_run_rows.append(
            {
                "fold": fold_number,
                "support_file": support_bundle.name,
                "support_words": int(support_reps.size(0)),
                "n_observed_support_labels": len(observed_labels),
                "observed_support_labels": "|".join(sorted(observed_labels)),
                "query_records": n_scored_records,
                "query_chunk_records": args.query_chunk,
                "structshot_available": structshot_available,
                "tau_default": args.structshot_tau,
                "tau_values": "|".join(str(t) for t in tau_list),
                "normalize_embeddings": args.normalize_embeddings,
                "distance_budget_mb": args.distance_budget_mb,
            }
        )

        # Free per-fold GPU tensors before the next support file.
        del support_reps, support_ids, proto, nnshot
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------- outputs
    average_rows = aggregate_average_metrics(fold_rows)
    average_label_rows = aggregate_average_label_metrics(label_rows)
    tau_average_rows = aggregate_tau_sweep(tau_sweep_rows)

    write_csv(args.output_dir / "pair_metrics.csv", pair_rows)
    write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "fold_label_metrics.csv", label_rows)
    write_csv(args.output_dir / "average_metrics.csv", average_rows)
    write_csv(args.output_dir / "average_label_metrics.csv", average_label_rows)
    write_csv(args.output_dir / "structshot_tau_sweep_fold_metrics.csv", tau_sweep_rows)
    write_csv(
        args.output_dir / "structshot_tau_sweep_average_metrics.csv", tau_average_rows
    )
    write_csv(args.output_dir / "support_shots.csv", support_shot_rows)
    write_csv(args.output_dir / "structshot_transition_audit.csv", transition_rows)
    write_csv(args.output_dir / "fold_run_audit.csv", fold_run_rows)
    write_csv(args.output_dir / "hf_encoding_audit.csv", encoding_audit)
    write_csv(args.output_dir / "token_boundary_audit.csv", boundary_audit)

    summary = {
        "script_version": SCRIPT_VERSION,
        "runtime": "native_huggingface_no_fewnerd_runtime",
        "hf_authentication": {
            "config_pipe": str(Path(args.config_pipe)),
            "token_configured": bool(hf_token),
        },
        "logging": {
            "logger_module": str(Path(args.logger_module)),
            "log_file": str(resolved_log_file),
        },
        "methods": methods,
        "metric": "nervaluate SemEval-style entity evaluation",
        "evaluation": {
            "backend": "nervaluate",
            "version": NERVALUATE_VERSION,
            "scenarios": list(NERVALUATE_SCENARIOS),
            "primary_strict_interpretation": (
                "strict = exact entity boundaries plus correct entity label"
            ),
            "min_overlap_percentage": args.nervaluate_min_overlap,
            "offset_unit": "character",
            "internal_span_convention": "[start,end) exclusive end",
            "nervaluate_span_convention": "[start,end] inclusive end",
        },
        "loo_design": "one file support; all remaining files query",
        "encoder": {
            "model": args.encoder_model,
            "model_type": getattr(encoder.config, "model_type", None),
            "hidden_size": encoder.hidden_size,
            "hidden_representation": args.hidden_representation,
            "device": str(device),
            "model_dtype_argument": args.model_dtype,
            "tokenization": args.tokenization,
            "word_representation": "first subword only",
            "max_length": args.max_length,
            "special_tokens_per_single_sequence": encoder.special_tokens,
            "normalize_embeddings_before_distance": args.normalize_embeddings,
            "embedding_cache": embedding_stats,
        },
        "implementation": {
            "protobert": (
                "Local plain-PyTorch inference head: mean support representation per "
                "observed class including O; negative squared Euclidean logits; argmax."
            ),
            "nnshot": (
                "SuperCD/StructShot logic: class emission is the maximum negative "
                "squared Euclidean similarity to any support token of that class."
            ),
            "structshot": (
                "NNShot emissions + abstract transition projection + generalized "
                "Viterbi decoder, adapted from chen700564/supercd and "
                "asappresearch/structshot. Emissions are softmaxed, START=1e-6 is "
                "prepended, then decoded in log space."
            ),
            "distance_memory": (
                "No [query,support,hidden] broadcast tensor is created. Pairwise "
                "negative squared Euclidean scores use norm terms + matrix multiply; "
                "NNShot is additionally chunked by class/query rows under the budget."
            ),
            "supercd_repo": SUPERCD_REPO,
            "supercd_ref": SUPERCD_REF,
            "structshot_repo": STRUCTSHOT_REPO,
            "structshot_ref": STRUCTSHOT_REF,
        },
        "scopes": {
            "support_observed": (
                "Only entity labels with at least one surviving first-subword support word."
            ),
            "all_schema": (
                "Whole GDPR-ALF child schema; labels absent from support are false "
                "negatives when present in query because they cannot be predicted."
            ),
        },
        "transition_source": transition_source,
        "transition_smoothing": args.transition_smoothing,
        "structshot_start_predictions_mapped_to_O": start_predictions_total,
        "annotation_marker": args.annotation_marker,
        "annotation_to_gdpr_alf_aliases": aliases,
        "gdpr_alf_target_labels": sorted(GDPR_ALF_TARGET_LABELS),
        "n_files": len(bundles),
        "support_order": [bundles[i].name for i in support_order],
        "files": [
            {
                "name": b.name,
                "input_path": str(b.input_path),
                "records": len(b.records),
                "encoded_records": len(b.encoded),
                "hf_chunks": b.n_chunks,
                "text_subwords": b.n_text_subwords,
                "records_skipped_no_words": b.tokens_dropped_records,
                "words_without_subword": b.dropped_words,
            }
            for b in bundles
        ],
        "outputs": [
            Path(resolved_log_file).name,
            "average_metrics.csv",
            "average_label_metrics.csv",
            "fold_metrics.csv",
            "fold_label_metrics.csv",
            "pair_metrics.csv",
            "support_shots.csv",
            "structshot_tau_sweep_fold_metrics.csv",
            "structshot_tau_sweep_average_metrics.csv",
            "structshot_transition_audit.csv",
            "fold_run_audit.csv",
            "hf_encoding_audit.csv",
            "token_boundary_audit.csv",
            "input_annotation_summary.csv",
            "input_annotation_audit.csv",
            "gdpr_alf_target_schema.csv",
        ],
    }
    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_average_table(average_rows)
    print_tau_table([r for r in tau_average_rows if r["scope"] == "all_schema"])
    logger.info("Results written to: %s", args.output_dir.resolve())
    logger.info("Primary table: average_metrics.csv")
    logger.info("Trace log: %s", resolved_log_file)
    print(f"\nResults written to: {args.output_dir.resolve()}")
    print("Primary table: average_metrics.csv")
    print(f"Trace log: {resolved_log_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logging.getLogger("fewshot_pseudonymization").exception("Fatal unhandled error")
        raise
