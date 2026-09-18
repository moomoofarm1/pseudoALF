#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Leave-one-file-out few-shot validation for Danish pseudonymization NER.

Single-directory design
-----------------------
The script uses ONE directory only: --input-dir.

Each selected file in --input-dir must already be the Presidio-preprocessed
version used for modeling and must contain inline child annotations such as:

    [PersonData]Peter[PER]

For every record, annotation wrappers are removed before the text is passed to
DaCy. The annotations are retained internally only to:
  * construct the support representations in the current leave-one-out fold;
  * score the held-out files after prediction.

There is no separate gold directory, validation directory, or Presidio/source
pairing step.

Validation design
-----------------
For N selected sessions:
  1. Each session is used once as the support session.
  2. The remaining N-1 sessions are query sessions.
  3. Evaluate:
       - dacy_zero_shot
       - protobert
       - structshot
  4. Report exact entity-level precision, recall, and F1.

The target class inventory follows the supplied GDPR-ALF PDF at CHILD level.
The MISC row in the PDF explicitly has no child tag, so MISC is not a child
target. Labels outside the PDF child inventory are audited and excluded from
child-level support/scoring rather than silently remapped.

Only the quoted utterance is passed to the model for .alfrttm files. Blank or
unquoted metadata rows are skipped unless --strict-quoted-utterance is enabled.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_VERSION = "2026-09-04-fewshot-pseudonymization-loo-v4-single-input-dir"

# Native labels exposed by chcaa/da_dacy_large_ner_fine_grained. These are
# encoder/baseline labels, not the pseudonymization evaluation schema.
EXPECTED_DACY_LABELS = {
    "CARDINAL", "DATE", "EVENT", "FACILITY", "GPE", "LANGUAGE", "LAW",
    "LOCATION", "MONEY", "NORP", "ORDINAL", "ORGANIZATION", "PERCENT",
    "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK OF ART",
}

# Exact CHILD-level GDPR-ALF schema from the supplied PDF.
# MISC is a parent row with "No child tag" and therefore is deliberately not
# included in GDPR_ALF_TARGET_LABELS.
GDPR_ALF_PARENT_CHILDREN = {
    "PER": ("PER", "STAFF", "PATIENT"),
    "LOC": ("LOC", "ADDRESS", "POSTCODE"),
    "ORG": ("ORG", "HOSPITAL"),
    "CONTACT": ("CONTACT", "PHONE", "EMAIL"),
    "ID": ("ID", "CPR"),
    "DATE": ("DATE", "TIME", "DURATION"),
    "DEM": ("DEM", "ETHNICITY", "RELIGION", "POLITICS", "SEXUALITY", "AGE"),
    "HEALTH": ("HEALTH", "DIAGNOSIS", "MEDICATION", "CONDITION"),
    "MISC": tuple(),  # PDF: no child tag
}
GDPR_ALF_TARGET_LABELS = frozenset(
    child
    for children in GDPR_ALF_PARENT_CHILDREN.values()
    for child in children
)

# Only unambiguous legacy/native aliases that map to an actual PDF child label.
# Labels such as CARDINAL/NUM/PERCENT/RELATION/PRODUCT/MISC are NOT child tags
# in the supplied PDF and are therefore left out-of-schema.
DEFAULT_ANNOTATION_ALIASES = {
    "NAME": "PER",
    "PERSON": "PER",
    "ORGANIZATION": "ORG",
    "GPE": "LOC",
    "LOCATION": "LOC",
    "FACILITY": "LOC",
    "LANGUAGE": "DEM",
}

# Conservative native-DaCy -> GDPR-ALF mapping for the zero-shot baseline only.
# Few-shot ProtoBERT/StructShot learn all schema labels directly from support.
DEFAULT_DACY_TO_GDPR_CHILD = {
    "PERSON": "PER",
    "GPE": "LOC",
    "LOCATION": "LOC",
    "FACILITY": "LOC",
    "ORGANIZATION": "ORG",
    "DATE": "DATE",
    "TIME": "TIME",
    "LANGUAGE": "DEM",
    "NORP": "DEM",
}

ANNOTATION_TEMPLATE = (
    r"\[{marker}\](?P<text>[^\[\]\r\n]+?)"
    r"\[(?P<label>[A-Za-z][A-Za-z0-9_ ]*)\]"
)

COMMON_PRESIDIO_SUFFIXES = (
    "_deidentified",
    "-deidentified",
    ".deidentified",
    "_anonymized",
    "-anonymized",
    "_anonymised",
    "-anonymised",
)

@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    text: str
    raw_label: str = ""

@dataclass
class AnnotatedRecord:
    uid: str
    line_no: int
    original_plain: str
    raw_entities: List[Entity]

@dataclass
class ValidationRecord:
    uid: str
    line_no: int
    text: str
    reference_entities: List[Entity]
    dropped_entities: List[Entity]
    unsupported_entities: List[Entity]

@dataclass
class TokenInfo:
    start: int
    end: int
    text: str
    is_space: bool

@dataclass
class EncodedRecord:
    uid: str
    line_no: int
    text: str
    tokens: List[TokenInfo]
    vectors: np.ndarray
    reference_entities: List[Entity]
    base_entities: List[Entity]

@dataclass
class FileBundle:
    name: str
    input_path: Path
    records: List[ValidationRecord]
    encoded: List[EncodedRecord] = field(default_factory=list)

def canonical_label(label: str, aliases: Dict[str, str]) -> str:
    x = label.strip().upper()
    # Normalize only the known work-of-art spelling automatically.
    if x == "WORK_OF_ART":
        x = "WORK OF ART"
    return aliases.get(x, x)

def make_annotation_re(marker: str) -> re.Pattern:
    return re.compile(ANNOTATION_TEMPLATE.format(marker=re.escape(marker)))

def extract_utterance(line: str, strict: bool, path: Path, line_no: int) -> Optional[str]:
    """Return the quoted utterance from one ALFRRTM row.

    In normal mode, blank/unquoted metadata rows are skipped. With
    --strict-quoted-utterance, a non-empty unquoted row raises. This prevents
    metadata or empty rows from being sent through the transformer.
    """
    first = line.find('"')
    last = line.rfind('"')
    if first >= 0 and last > first:
        utterance = line[first + 1:last]
        return utterance if utterance.strip() else None

    if strict and line.strip():
        raise ValueError(
            f"{path}:{line_no}: ALFRRTM line has no quoted utterance. "
            "Fix the row or omit --strict-quoted-utterance to skip unquoted "
            "ALFRRTM metadata/malformed rows."
        )
    return None

def parse_annotated_utterance(
    utterance: str,
    ann_re: re.Pattern,
    aliases: Dict[str, str],
) -> Tuple[str, List[Entity]]:
    """Strip annotation wrappers while preserving entity text and char spans."""
    out: List[str] = []
    entities: List[Entity] = []
    cursor = 0
    out_len = 0

    for m in ann_re.finditer(utterance):
        prefix = utterance[cursor:m.start()]
        out.append(prefix)
        out_len += len(prefix)

        mention = m.group("text")
        raw_label = m.group("label").strip()
        label = canonical_label(raw_label, aliases)

        start = out_len
        out.append(mention)
        out_len += len(mention)
        end = out_len

        entities.append(
            Entity(
                start=start,
                end=end,
                label=label,
                text=mention,
                raw_label=raw_label,
            )
        )
        cursor = m.end()

    tail = utterance[cursor:]
    out.append(tail)
    return "".join(out), entities

def load_annotated_records(
    path: Path,
    marker: str,
    aliases: Dict[str, str],
    strict_quoted_utterance: bool,
) -> List[AnnotatedRecord]:
    ann_re = make_annotation_re(marker)
    records: List[AnnotatedRecord] = []

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    is_alfrttm = path.suffix.lower() == ".alfrttm"

    for i, line in enumerate(lines, start=1):
        if is_alfrttm:
            utterance = extract_utterance(
                line, strict_quoted_utterance, path, i
            )
            if utterance is None:
                continue
        else:
            utterance = line.rstrip("\r\n")

        plain, entities = parse_annotated_utterance(utterance, ann_re, aliases)
        records.append(
            AnnotatedRecord(
                uid=f"{path.name}::line={i}",
                line_no=i,
                original_plain=plain,
                raw_entities=entities,
            )
        )

    return records

def load_input_validation_records(
    path: Path,
    marker: str,
    aliases: Dict[str, str],
    target_labels: set,
    strict_quoted_utterance: bool,
) -> Tuple[List[ValidationRecord], List[dict]]:
    """Load one already-Presidio-preprocessed, inline-labelled LOO session.

    The inline annotation wrappers are stripped before model inference. The
    resulting plain text is therefore the only text seen by DaCy. Child labels
    are retained internally for support construction and held-out scoring.
    """
    ann_re = make_annotation_re(marker)
    records: List[ValidationRecord] = []
    audit_rows: List[dict] = []

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    is_alfrttm = path.suffix.lower() == ".alfrttm"

    for i, line in enumerate(lines, start=1):
        if is_alfrttm:
            utterance = extract_utterance(
                line, strict_quoted_utterance, path, i
            )
            if utterance is None:
                continue
        else:
            utterance = line.rstrip("\r\n")
            if not utterance.strip():
                continue

        plain, raw_entities = parse_annotated_utterance(
            utterance, ann_re, aliases
        )

        reference_entities: List[Entity] = []
        unsupported_entities: List[Entity] = []

        for ent in raw_entities:
            in_schema = ent.label in target_labels
            audit_rows.append({
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
            })
            if in_schema:
                reference_entities.append(ent)
            else:
                unsupported_entities.append(ent)

        records.append(
            ValidationRecord(
                uid=f"{path.name}::line={i}",
                line_no=i,
                text=plain,
                reference_entities=reference_entities,
                dropped_entities=[],
                unsupported_entities=unsupported_entities,
            )
        )

    return records, audit_rows


def load_presidio_records(
    path: Path,
    strict_quoted_utterance: bool,
) -> List[Tuple[int, str]]:
    records: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    is_alfrttm = path.suffix.lower() == ".alfrttm"
    for i, line in enumerate(lines, start=1):
        if is_alfrttm:
            utterance = extract_utterance(
                line, strict_quoted_utterance, path, i
            )
            if utterance is None:
                continue
            records.append((i, utterance))
        else:
            records.append((i, line.rstrip("\r\n")))
    return records

def map_entity_through_presidio(
    original: str,
    processed: str,
    ent: Entity,
) -> Optional[Entity]:
    """
    Map an annotated reference span only when it survived Presidio unchanged.

    SequenceMatcher provides equal blocks between original plain text and the
    Presidio-preprocessed text. The entity is retained only when its complete
    character span lies inside one equal block.
    """
    if ent.start < 0 or ent.end > len(original) or ent.start >= ent.end:
        return None

    sm = difflib.SequenceMatcher(
        a=original,
        b=processed,
        autojunk=False,
    )
    for block in sm.get_matching_blocks():
        a0, b0, size = block
        if size <= 0:
            continue
        if ent.start >= a0 and ent.end <= a0 + size:
            mapped_start = b0 + (ent.start - a0)
            mapped_end = mapped_start + (ent.end - ent.start)
            mapped_text = processed[mapped_start:mapped_end]
            if mapped_text == ent.text:
                return Entity(
                    start=mapped_start,
                    end=mapped_end,
                    label=ent.label,
                    text=mapped_text,
                    raw_label=ent.raw_label,
                )
    return None

def align_annotations_to_presidio(
    annotated_records: List[AnnotatedRecord],
    presidio_records: List[Tuple[int, str]],
    target_labels: set,
) -> Tuple[List[ValidationRecord], List[dict]]:
    """Align child-labelled validation records to Presidio input by line number."""
    annotated_by_line = {g.line_no: g for g in annotated_records}
    pres_by_line = {line_no: txt for line_no, txt in presidio_records}
    if len(annotated_by_line) != len(annotated_records):
        raise ValueError("Duplicate annotated record line numbers detected.")
    if len(pres_by_line) != len(presidio_records):
        raise ValueError("Duplicate Presidio record line numbers detected.")

    annotated_lines = set(annotated_by_line)
    pres_lines = set(pres_by_line)
    if annotated_lines != pres_lines:
        missing = sorted(annotated_lines - pres_lines)
        extra = sorted(pres_lines - annotated_lines)
        raise ValueError(
            "Annotated and Presidio files do not contain the same model-text line "
            "numbers after ALFRRTM blank/unquoted-row filtering. "
            f"Missing in Presidio={missing[:20]}; extra in Presidio={extra[:20]}."
        )

    residual: List[ValidationRecord] = []
    audit_rows: List[dict] = []

    for line_no in sorted(annotated_lines):
        g = annotated_by_line[line_no]
        p_text = pres_by_line[line_no]
        kept: List[Entity] = []
        dropped: List[Entity] = []
        unsupported: List[Entity] = []

        for ent in g.raw_entities:
            if ent.label not in target_labels:
                unsupported.append(ent)
                audit_rows.append({
                    "record_uid": g.uid,
                    "annotated_line_no": g.line_no,
                    "presidio_line_no": line_no,
                    "raw_label": ent.raw_label,
                    "canonical_label": ent.label,
                    "entity_text": ent.text,
                    "status": "out_of_gdpr_alf_schema",
                })
                continue

            mapped = map_entity_through_presidio(g.original_plain, p_text, ent)
            if mapped is None:
                dropped.append(ent)
                audit_rows.append({
                    "record_uid": g.uid,
                    "annotated_line_no": g.line_no,
                    "presidio_line_no": line_no,
                    "raw_label": ent.raw_label,
                    "canonical_label": ent.label,
                    "entity_text": ent.text,
                    "status": "changed_or_removed_by_presidio",
                })
            else:
                kept.append(mapped)
                audit_rows.append({
                    "record_uid": g.uid,
                    "annotated_line_no": g.line_no,
                    "presidio_line_no": line_no,
                    "raw_label": ent.raw_label,
                    "canonical_label": ent.label,
                    "entity_text": ent.text,
                    "status": "survived_presidio",
                })

        residual.append(
            ValidationRecord(
                uid=g.uid,
                line_no=line_no,
                text=p_text,
                reference_entities=kept,
                dropped_entities=dropped,
                unsupported_entities=unsupported,
            )
        )
    return residual, audit_rows

def normalized_file_key(path: Path) -> str:
    key = path.stem.casefold()
    for suffix in COMMON_PRESIDIO_SUFFIXES:
        if key.endswith(suffix):
            key = key[:-len(suffix)]
    key = re.sub(r"\s+", " ", key).strip()
    return key

def discover_files(
    directory: Path,
    extensions: set,
    include_substrings: Sequence[str],
) -> List[Path]:
    paths = []
    for p in directory.rglob("*"):
        if ".ipynb_checkpoints" in p.parts:
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() not in extensions:
            continue
        if include_substrings and not any(
            x in p.name for x in include_substrings
        ):
            continue
        paths.append(p)
    return sorted(paths, key=lambda p: str(p).casefold())

def match_presidio_file(annotated_path: Path, presidio_candidates: List[Path]) -> Path:
    same_name = [
        p for p in presidio_candidates
        if p.name.casefold() == annotated_path.name.casefold()
    ]
    if len(same_name) == 1:
        return same_name[0]
    if len(same_name) > 1:
        raise ValueError(
            f"Multiple Presidio files match basename {annotated_path.name}: "
            + ", ".join(map(str, same_name))
        )

    gkey = normalized_file_key(annotated_path)
    same_key = [
        p for p in presidio_candidates
        if normalized_file_key(p) == gkey
        and p.suffix.lower() == annotated_path.suffix.lower()
    ]
    if len(same_key) == 1:
        return same_key[0]
    if len(same_key) > 1:
        raise ValueError(
            f"Multiple Presidio files match normalized key {gkey!r}: "
            + ", ".join(map(str, same_key))
        )

    raise FileNotFoundError(
        f"No Presidio counterpart found for: {annotated_path}"
    )

def _to_numpy(x) -> np.ndarray:
    # CuPy arrays expose .get(); NumPy arrays do not.
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x)

def _ragged_token_means(ragged, n_tokens: int) -> Optional[np.ndarray]:
    """Mean-pool a curated-transformer Ragged output to spaCy tokens."""
    data = getattr(ragged, "dataXd", None)
    lengths = getattr(ragged, "lengths", None)
    if data is None or lengths is None:
        return None
    arr = _to_numpy(data).astype(np.float32, copy=False)
    lens = _to_numpy(lengths).astype(np.int64, copy=False).ravel()
    if arr.ndim != 2 or len(lens) != n_tokens:
        return None

    result = np.zeros((n_tokens, arr.shape[-1]), dtype=np.float32)
    cursor = 0
    for i, length in enumerate(lens):
        length = int(length)
        if length > 0:
            end = cursor + length
            if end > len(arr):
                return None
            result[i] = arr[cursor:end].mean(axis=0)
            cursor = end
    return result


def _flatten_hidden_candidate(hidden) -> Optional[np.ndarray]:
    """Convert common hidden-state containers to a 2D [pieces, width] matrix."""
    if hidden is None:
        return None
    if isinstance(hidden, (list, tuple)):
        parts = []
        for item in hidden:
            arr = _flatten_hidden_candidate(item)
            if arr is not None and arr.size:
                parts.append(arr)
        if not parts:
            return None
        width = parts[0].shape[-1]
        if any(x.shape[-1] != width for x in parts):
            return None
        return np.vstack(parts).astype(np.float32, copy=False)

    try:
        arr = _to_numpy(hidden).astype(np.float32, copy=False)
    except Exception:
        return None
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2:
        return arr
    return None


def token_vectors_from_doc(doc) -> np.ndarray:
    """Return one contextual transformer vector per spaCy token.

    Supports both:
      * spacy-transformers TransformerData (model_output/tensors + align), and
      * spacy-curated-transformers DocTransformerOutput
        (last_hidden_layer_state Ragged).

    Doc.tensor is used only as a token-aligned fallback.
    """
    if len(doc) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    data = getattr(doc._, "trf_data", None)
    if data is not None:
        # spaCy 3.7+ curated-transformer API.
        try:
            curated_last = getattr(data, "last_hidden_layer_state", None)
        except Exception:
            curated_last = None
        if curated_last is not None:
            pooled = _ragged_token_means(curated_last, len(doc))
            if pooled is not None and pooled.shape[1] > 0:
                return pooled

        # spacy-transformers 1.1+ API.
        model_output = getattr(data, "model_output", None)
        hidden = None
        if model_output is not None:
            hidden = getattr(model_output, "last_hidden_state", None)
            if hidden is None and hasattr(model_output, "get"):
                hidden = model_output.get("last_hidden_state")
        flat = _flatten_hidden_candidate(hidden)

        # Older spacy-transformers compatibility tuple.
        if flat is None:
            try:
                tensors = getattr(data, "tensors", ())
            except Exception:
                tensors = ()
            candidates = []
            for tensor in tensors or ():
                arr = _flatten_hidden_candidate(tensor)
                if arr is not None and arr.size:
                    candidates.append(arr)
            if candidates:
                flat = max(candidates, key=lambda x: x.shape[-1])

        if flat is not None:
            align = getattr(data, "align", None)
            if align is not None:
                width = flat.shape[-1]
                result = np.zeros((len(doc), width), dtype=np.float32)
                aligned_any = False
                for i in range(len(doc)):
                    try:
                        aligned = _to_numpy(align[i].dataXd).astype(
                            np.int64, copy=False
                        ).ravel()
                    except Exception:
                        aligned = np.zeros((0,), dtype=np.int64)
                    aligned = aligned[(aligned >= 0) & (aligned < len(flat))]
                    if len(aligned):
                        result[i] = flat[aligned].mean(axis=0)
                        aligned_any = True
                if aligned_any:
                    return result

            # Some outputs are already token-aligned.
            if flat.shape[0] == len(doc):
                return flat.astype(np.float32, copy=False)

    # Safe fallback for pipelines whose annotation setter stores contextual
    # token features in Doc.tensor.
    try:
        doc_tensor = _to_numpy(doc.tensor).astype(np.float32, copy=False)
    except Exception:
        doc_tensor = None
    if (
        doc_tensor is not None
        and doc_tensor.ndim == 2
        and doc_tensor.shape[0] == len(doc)
        and doc_tensor.shape[1] > 0
    ):
        return doc_tensor

    data_type = (
        f"{type(data).__module__}.{type(data).__name__}"
        if data is not None else "None"
    )
    attrs = []
    if data is not None:
        for name in (
            "model_output", "tensors", "align", "all_outputs",
            "last_hidden_layer_state"
        ):
            try:
                if hasattr(data, name):
                    attrs.append(name)
            except Exception:
                pass
    raise RuntimeError(
        "Could not obtain token-aligned transformer hidden states. "
        f"trf_data_type={data_type}; available_attributes={attrs}; "
        f"len(doc)={len(doc)}. The validator supports both "
        "spacy-transformers TransformerData and spacy-curated-transformers "
        "DocTransformerOutput."
    )

def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms

def encode_bundle(
    nlp,
    bundle: FileBundle,
    batch_size: int,
    normalize_vecs: bool,
    dacy_to_target: Dict[str, str],
) -> None:
    """Encode Presidio text and map DaCy baseline entities into GDPR-ALF."""
    model_records = [r for r in bundle.records if r.text and r.text.strip()]
    texts = [r.text for r in model_records]
    encoded: List[EncodedRecord] = []
    expected_width: Optional[int] = None

    for rec, doc in zip(
        model_records,
        nlp.pipe(texts, batch_size=batch_size),
    ):
        vecs = token_vectors_from_doc(doc)
        if vecs.shape[0] != len(doc):
            raise RuntimeError(
                f"Token/vector mismatch for {rec.uid}: "
                f"{len(doc)} tokens vs {vecs.shape[0]} vector rows."
            )
        if len(doc) and vecs.shape[1] == 0:
            raise RuntimeError(f"Zero-width transformer vectors for {rec.uid}.")
        if len(doc):
            if expected_width is None:
                expected_width = int(vecs.shape[1])
            elif vecs.shape[1] != expected_width:
                raise RuntimeError(
                    f"Transformer width changed within {bundle.name}: "
                    f"{expected_width} -> {vecs.shape[1]} at {rec.uid}."
                )

        if normalize_vecs and len(vecs):
            vecs = normalize_embeddings(vecs)

        tokens = [
            TokenInfo(
                start=t.idx,
                end=t.idx + len(t),
                text=t.text,
                is_space=bool(t.is_space),
            )
            for t in doc
        ]

        # Zero-shot DaCy is a baseline only. Convert only native classes with
        # deterministic correspondence to the GDPR-ALF schema.
        base_entities: List[Entity] = []
        for ent in doc.ents:
            native = ent.label_.strip().upper()
            if native == "WORK_OF_ART":
                native = "WORK OF ART"
            target = dacy_to_target.get(native)
            if target is not None:
                base_entities.append(
                    Entity(
                        start=ent.start_char,
                        end=ent.end_char,
                        label=target,
                        text=ent.text,
                        raw_label=ent.label_,
                    )
                )

        encoded.append(
            EncodedRecord(
                uid=rec.uid,
                line_no=rec.line_no,
                text=rec.text,
                tokens=tokens,
                vectors=vecs,
                reference_entities=rec.reference_entities,
                base_entities=base_entities,
            )
        )

    bundle.encoded = encoded

def reference_token_labels(
    record: EncodedRecord,
) -> Tuple[List[str], int]:
    labels = ["O"] * len(record.tokens)
    boundary_expansions = 0

    for ent in record.reference_entities:
        overlapping = []
        exact_left = False
        exact_right = False
        for i, tok in enumerate(record.tokens):
            if tok.is_space:
                continue
            if tok.start == ent.start:
                exact_left = True
            if tok.end == ent.end:
                exact_right = True
            if tok.start < ent.end and ent.start < tok.end:
                overlapping.append(i)

        if not (exact_left and exact_right):
            boundary_expansions += 1

        for i in overlapping:
            if labels[i] != "O" and labels[i] != ent.label:
                raise ValueError(
                    f"Overlapping reference entities with conflicting labels in "
                    f"{record.uid}: token={record.tokens[i].text!r}, "
                    f"{labels[i]} vs {ent.label}"
                )
            labels[i] = ent.label

    return labels, boundary_expansions

def labels_to_entities(
    record: EncodedRecord,
    labels: Sequence[str],
) -> List[Entity]:
    """Convert IO token labels to contiguous entity spans."""
    ents: List[Entity] = []
    active_label: Optional[str] = None
    active_start: Optional[int] = None
    active_end: Optional[int] = None

    def flush():
        nonlocal active_label, active_start, active_end
        if active_label is not None and active_start is not None and active_end is not None:
            text = record.text[active_start:active_end]
            ents.append(
                Entity(
                    start=active_start,
                    end=active_end,
                    label=active_label,
                    text=text,
                    raw_label=active_label,
                )
            )
        active_label = None
        active_start = None
        active_end = None

    for tok, label in zip(record.tokens, labels):
        if tok.is_space or label == "O":
            flush()
            continue
        if active_label == label and active_end is not None:
            # Merge adjacent/whitespace-separated tokens of the same IO label.
            gap = record.text[active_end:tok.start]
            if gap.strip() == "":
                active_end = tok.end
                continue
        flush()
        active_label = label
        active_start = tok.start
        active_end = tok.end

    flush()
    return ents

def squared_euclidean_to_matrix(
    query: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """
    Pairwise squared Euclidean distances, shape [n_query, n_support].
    """
    q2 = np.sum(query * query, axis=1, keepdims=True)
    s2 = np.sum(support * support, axis=1, keepdims=True).T
    d = q2 + s2 - 2.0 * (query @ support.T)
    np.maximum(d, 0.0, out=d)
    return d

def build_support(
    support_bundle: FileBundle,
    rng: random.Random,
    max_o_support: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    List[List[str]],
    List[dict],
    int,
]:
    """
    Return:
      prototypes, structshot support matrices, token vectors by class,
      support label sequences, support-shot audit rows, boundary-expansion count
    """
    vectors_by_label: Dict[str, List[np.ndarray]] = defaultdict(list)
    label_sequences: List[List[str]] = []
    shot_rows: List[dict] = []
    boundary_expansions = 0

    mention_counts = Counter()
    token_counts = Counter()

    for rec in support_bundle.encoded:
        labels, n_expanded = reference_token_labels(rec)
        boundary_expansions += n_expanded
        label_sequences.append(labels)

        for ent in rec.reference_entities:
            mention_counts[ent.label] += 1

        for i, (tok, label) in enumerate(zip(rec.tokens, labels)):
            if tok.is_space:
                continue
            vectors_by_label[label].append(rec.vectors[i])
            token_counts[label] += 1

    observed_labels = sorted(
        label for label in vectors_by_label
        if label != "O" and len(vectors_by_label[label]) > 0
    )

    if "O" not in vectors_by_label:
        raise RuntimeError(
            f"Support file {support_bundle.name} produced no O tokens."
        )

    # ProtoBERT-style prototypes use all available support tokens.
    prototypes: Dict[str, np.ndarray] = {}
    for label in ["O"] + observed_labels:
        arr = np.vstack(vectors_by_label[label]).astype(np.float32)
        prototypes[label] = arr.mean(axis=0)

    # StructShot-style nearest-neighbour support. The O class can dominate a
    # full interview, so cap it reproducibly to control runtime/memory.
    struct_support: Dict[str, np.ndarray] = {}
    for label in ["O"] + observed_labels:
        arr = np.vstack(vectors_by_label[label]).astype(np.float32)
        if label == "O" and max_o_support > 0 and len(arr) > max_o_support:
            idx = list(range(len(arr)))
            rng.shuffle(idx)
            idx = sorted(idx[:max_o_support])
            arr = arr[idx]
        struct_support[label] = arr

    for label in sorted(set(mention_counts) | set(token_counts)):
        shot_rows.append({
            "label": label,
            "mention_count": mention_counts[label],
            "token_count": token_counts[label],
            "structshot_support_token_count": (
                len(struct_support[label])
                if label in struct_support else 0
            ),
        })

    return (
        prototypes,
        struct_support,
        {k: np.vstack(v).astype(np.float32) for k, v in vectors_by_label.items()},
        label_sequences,
        shot_rows,
        boundary_expansions,
    )

def proto_predict(
    record: EncodedRecord,
    prototypes: Dict[str, np.ndarray],
) -> List[str]:
    classes = list(prototypes)
    proto_mat = np.vstack([prototypes[c] for c in classes])
    q = record.vectors

    # Squared Euclidean distance to each prototype.
    q2 = np.sum(q * q, axis=1, keepdims=True)
    p2 = np.sum(proto_mat * proto_mat, axis=1, keepdims=True).T
    dist = q2 + p2 - 2.0 * (q @ proto_mat.T)
    pred_idx = np.argmin(dist, axis=1)
    labels = [classes[i] for i in pred_idx]

    for i, tok in enumerate(record.tokens):
        if tok.is_space:
            labels[i] = "O"
    return labels

def log_softmax_rows(scores: np.ndarray) -> np.ndarray:
    m = np.max(scores, axis=1, keepdims=True)
    z = scores - m
    return z - np.log(np.sum(np.exp(z), axis=1, keepdims=True) + 1e-30)

def structshot_emissions(
    record: EncodedRecord,
    support: Dict[str, np.ndarray],
    classes: Sequence[str],
) -> np.ndarray:
    """
    NNShot-style emission log-probabilities:
    negative nearest-support-token squared Euclidean distance per class.
    """
    q = record.vectors
    raw = np.empty((len(q), len(classes)), dtype=np.float32)

    for j, label in enumerate(classes):
        s = support[label]
        # Utterances are short, so one matrix per class is usually modest.
        d = squared_euclidean_to_matrix(q, s)
        raw[:, j] = -np.min(d, axis=1)

    logp = log_softmax_rows(raw)

    # Force whitespace to O.
    o_idx = classes.index("O")
    for i, tok in enumerate(record.tokens):
        if tok.is_space:
            logp[i, :] = -1e9
            logp[i, o_idx] = 0.0

    return logp

def build_abstract_transition_matrix(
    label_sequences: Sequence[Sequence[str]],
    entity_classes: Sequence[str],
    smoothing: float,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build StructShot-style abstract IO transition probabilities.

    We estimate:
      O->O, O->I,
      I->O, I->same-I, I->different-I,
    then distribute abstract entity transitions over concrete target labels.
    """
    n_ent = len(entity_classes)
    classes = ["O"] + list(entity_classes)
    n = len(classes)

    start_o = smoothing
    start_i = smoothing

    oo = smoothing
    oi = smoothing
    io = smoothing
    ii_same = smoothing
    ii_diff = smoothing

    for seq in label_sequences:
        seq = [x for x in seq if x is not None]
        if not seq:
            continue
        if seq[0] == "O":
            start_o += 1
        else:
            start_i += 1

        for a, b in zip(seq[:-1], seq[1:]):
            if a == "O":
                if b == "O":
                    oo += 1
                else:
                    oi += 1
            else:
                if b == "O":
                    io += 1
                elif b == a:
                    ii_same += 1
                else:
                    ii_diff += 1

    p_start_o = start_o / (start_o + start_i)
    p_start_i = 1.0 - p_start_o

    p_oo = oo / (oo + oi)
    p_oi = 1.0 - p_oo

    denom_i = io + ii_same + ii_diff
    p_io = io / denom_i
    p_same = ii_same / denom_i
    p_diff = ii_diff / denom_i

    start = np.zeros(n, dtype=np.float64)
    start[0] = p_start_o

    trans = np.zeros((n, n), dtype=np.float64)
    trans[0, 0] = p_oo

    if n_ent > 0:
        start[1:] = p_start_i / n_ent
        trans[0, 1:] = p_oi / n_ent

        for i in range(1, n):
            trans[i, 0] = p_io
            if n_ent == 1:
                trans[i, i] = p_same + p_diff
            else:
                trans[i, i] = p_same
                for j in range(1, n):
                    if j != i:
                        trans[i, j] = p_diff / (n_ent - 1)
    else:
        start[0] = 1.0
        trans[0, 0] = 1.0

    # Numerical normalization.
    start = start / start.sum()
    trans = trans / trans.sum(axis=1, keepdims=True)

    abstract = {
        "p_start_O": p_start_o,
        "p_start_I": p_start_i,
        "p_O_to_O": p_oo,
        "p_O_to_I": p_oi,
        "p_I_to_O": p_io,
        "p_I_to_same_I": p_same,
        "p_I_to_different_I": p_diff,
    }
    return start, trans, abstract

def viterbi_decode(
    emission_logp: np.ndarray,
    start_prob: np.ndarray,
    trans_prob: np.ndarray,
    tau: float,
) -> np.ndarray:
    t_len, n_states = emission_logp.shape
    if t_len == 0:
        return np.zeros((0,), dtype=np.int64)

    log_start = np.log(np.maximum(start_prob, 1e-30))
    log_trans = np.log(np.maximum(trans_prob, 1e-30))

    dp = np.empty((t_len, n_states), dtype=np.float64)
    back = np.zeros((t_len, n_states), dtype=np.int64)

    dp[0] = emission_logp[0] + tau * log_start

    for t in range(1, t_len):
        scores = dp[t - 1][:, None] + tau * log_trans
        back[t] = np.argmax(scores, axis=0)
        dp[t] = emission_logp[t] + scores[back[t], np.arange(n_states)]

    path = np.zeros(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(t_len - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path

def structshot_predict(
    record: EncodedRecord,
    support: Dict[str, np.ndarray],
    label_sequences: Sequence[Sequence[str]],
    tau: float,
    smoothing: float,
) -> Tuple[List[str], dict]:
    entity_classes = sorted([x for x in support if x != "O"])
    classes = ["O"] + entity_classes

    start, trans, abstract = build_abstract_transition_matrix(
        label_sequences=label_sequences,
        entity_classes=entity_classes,
        smoothing=smoothing,
    )
    emission = structshot_emissions(record, support, classes)
    path = viterbi_decode(
        emission_logp=emission,
        start_prob=start,
        trans_prob=trans,
        tau=tau,
    )
    labels = [classes[i] for i in path]
    return labels, abstract

def entity_counter(
    file_name: str,
    record: EncodedRecord,
    entities: Sequence[Entity],
    allowed_labels: set,
) -> Counter:
    return Counter(
        (
            file_name,
            record.uid,
            ent.start,
            ent.end,
            ent.label,
        )
        for ent in entities
        if ent.label in allowed_labels
    )

def score_counters(reference: Counter, pred: Counter) -> dict:
    tp = sum((reference & pred).values())
    reference_n = sum(reference.values())
    pred_n = sum(pred.values())
    fp = pred_n - tp
    fn = reference_n - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "reference_n": reference_n,
        "pred_n": pred_n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def filter_counter_label(counter: Counter, label: str) -> Counter:
    return Counter(
        {k: v for k, v in counter.items() if k[-1] == label}
    )

def write_csv(path: Path, rows: Sequence[dict], fieldnames: Optional[Sequence[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            return
        seen = []
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.append(k)
        fieldnames = seen

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
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

def aggregate_average_metrics(fold_rows: Sequence[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in fold_rows:
        groups[(row["method"], row["scope"])].append(row)

    out = []
    for (method, scope), rows in sorted(groups.items()):
        p_mean, p_sd = mean_sd([r["precision"] for r in rows])
        r_mean, r_sd = mean_sd([r["recall"] for r in rows])
        f_mean, f_sd = mean_sd([r["f1"] for r in rows])

        tp = sum(int(r["tp"]) for r in rows)
        fp = sum(int(r["fp"]) for r in rows)
        fn = sum(int(r["fn"]) for r in rows)
        # Pool the entity-level contingency counts across the LOO folds.
        pooled_p = tp / (tp + fp) if (tp + fp) else 0.0
        pooled_r = tp / (tp + fn) if (tp + fn) else 0.0
        pooled_f = (
            2 * pooled_p * pooled_r / (pooled_p + pooled_r)
            if (pooled_p + pooled_r) else 0.0
        )

        out.append({
            "method": method,
            "scope": scope,
            "n_folds": len(rows),
            "precision_mean": p_mean,
            "precision_sd": p_sd,
            "recall_mean": r_mean,
            "recall_sd": r_sd,
            "f1_mean": f_mean,
            "f1_sd": f_sd,
            "tp_pooled": tp,
            "fp_pooled": fp,
            "fn_pooled": fn,
            "precision_pooled": pooled_p,
            "recall_pooled": pooled_r,
            "f1_pooled": pooled_f,
        })
    return out

def aggregate_average_label_metrics(label_rows: Sequence[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in label_rows:
        groups[(row["method"], row["scope"], row["label"])].append(row)

    out = []
    for (method, scope, label), rows in sorted(groups.items()):
        p_mean, p_sd = mean_sd([r["precision"] for r in rows])
        r_mean, r_sd = mean_sd([r["recall"] for r in rows])
        f_mean, f_sd = mean_sd([r["f1"] for r in rows])

        tp = sum(int(r["tp"]) for r in rows)
        fp = sum(int(r["fp"]) for r in rows)
        fn = sum(int(r["fn"]) for r in rows)
        pooled_p = tp / (tp + fp) if (tp + fp) else 0.0
        pooled_r = tp / (tp + fn) if (tp + fn) else 0.0
        pooled_f = (
            2 * pooled_p * pooled_r / (pooled_p + pooled_r)
            if (pooled_p + pooled_r) else 0.0
        )

        out.append({
            "method": method,
            "scope": scope,
            "label": label,
            "n_folds": len(rows),
            "precision_mean": p_mean,
            "precision_sd": p_sd,
            "recall_mean": r_mean,
            "recall_sd": r_sd,
            "f1_mean": f_mean,
            "f1_sd": f_sd,
            "tp_pooled": tp,
            "fp_pooled": fp,
            "fn_pooled": fn,
            "precision_pooled": pooled_p,
            "recall_pooled": pooled_r,
            "f1_pooled": pooled_f,
        })
    return out

def print_average_table(rows: Sequence[dict]) -> None:
    if not rows:
        return
    print("\nAverage exact entity-level metrics across support folds")
    print("=" * 98)
    print(
        f"{'method':<18} {'scope':<18} "
        f"{'P mean±SD':>16} {'R mean±SD':>16} {'F1 mean±SD':>16} "
        f"{'F1 pooled':>10}"
    )
    print("-" * 98)
    for r in rows:
        print(
            f"{r['method']:<18} {r['scope']:<18} "
            f"{r['precision_mean']:.4f}±{r['precision_sd']:.4f} "
            f"{r['recall_mean']:.4f}±{r['recall_sd']:.4f} "
            f"{r['f1_mean']:.4f}±{r['f1_sd']:.4f} "
            f"{r['f1_pooled']:.4f}"
        )
    print("=" * 98)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Leave-one-file-out few-shot pseudonymization validation in the "
            "supplied GDPR-ALF target schema using DaCy as contextual encoder."
        )
    )
    p.add_argument("--version", action="version", version=SCRIPT_VERSION)

    p.add_argument(
        "--input-dir", type=Path, required=True,
        help=(
            "Single directory containing the already-Presidio-preprocessed "
            "LOO sessions with inline [PersonData]mention[LABEL] annotations. "
            "No separate validation/gold directory is used."
        ),
    )
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument(
        "--model",
        default="da_dacy_large_ner_fine_grained",
        help="Installed spaCy/DaCy pipeline name.",
    )
    p.add_argument(
        "--extensions", nargs="+", default=[".txt", ".alfrttm"],
    )
    p.add_argument(
        "--include-substrings",
        nargs="*",
        default=[],
        help=(
            "Optional filename substrings. If provided, only input files whose "
            "basename contains at least one substring are used."
        ),
    )
    p.add_argument(
        "--expected-files",
        type=int,
        default=0,
        help="Require this many selected input sessions; 0 disables the count check.",
    )
    p.add_argument("--annotation-marker", default="PersonData")
    p.add_argument(
        "--strict-quoted-utterance",
        action="store_true",
        default=False,
        help=(
            "For ALFRRTM, raise on a non-empty row without a quoted utterance. "
            "Default behavior skips such metadata/malformed rows."
        ),
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--gpu-id",
        type=int,
        default=-2,
        help="-2: prefer GPU automatically; -1: CPU; >=0: require that GPU.",
    )
    p.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help=(
            "L2-normalize token embeddings before Euclidean-distance methods. "
            "Off by default to stay closer to ProtoBERT's Euclidean logic."
        ),
    )
    p.add_argument(
        "--max-o-support",
        type=int,
        default=2048,
        help=(
            "Maximum O-class support tokens for StructShot nearest-neighbour "
            "emissions. Entity-class tokens are never capped; 0 means no cap."
        ),
    )
    p.add_argument(
        "--structshot-tau",
        type=float,
        default=0.32,
        help="Transition-weight parameter for Viterbi decoding.",
    )
    p.add_argument(
        "--transition-smoothing",
        type=float,
        default=1.0,
        help="Additive smoothing for support-derived abstract transitions.",
    )

    p.add_argument(
        "--annotation-label-map-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping raw/legacy reference labels into the supplied "
            "GDPR-ALF target labels."
        ),
    )
    p.add_argument(
        "--dacy-label-map-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping native DaCy labels into GDPR-ALF labels for "
            "the zero-shot baseline only."
        ),
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Check files/model and inline child annotations, write audits, but skip "
            "transformer embeddings and few-shot evaluation."
        ),
    )
    return p


def _validated_target_mapping(mapping: dict, source_name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in mapping.items():
        key = str(k).strip().upper()
        if key == "WORK_OF_ART":
            key = "WORK OF ART"
        value = str(v).strip().upper()
        if value not in GDPR_ALF_TARGET_LABELS:
            raise ValueError(
                f"{source_name}: destination {value!r} is not a valid "
                "GDPR-ALF target label."
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
            raise ValueError("--annotation-label-map-json must contain a JSON object.")
        aliases.update(
            _validated_target_mapping(custom, "--annotation-label-map-json")
        )
    return aliases


def load_dacy_target_map(args) -> Dict[str, str]:
    mapping = _validated_target_mapping(
        DEFAULT_DACY_TO_GDPR_CHILD, "DEFAULT_DACY_TO_GDPR_CHILD"
    )
    if args.dacy_label_map_json is not None:
        with args.dacy_label_map_json.open("r", encoding="utf-8") as f:
            custom = json.load(f)
        if not isinstance(custom, dict):
            raise ValueError("--dacy-label-map-json must contain a JSON object.")
        mapping.update(
            _validated_target_mapping(custom, "--dacy-label-map-json")
        )
    return mapping


def load_spacy_model(args):
    import spacy

    if args.gpu_id == -2:
        used_gpu = bool(spacy.prefer_gpu())
    elif args.gpu_id == -1:
        used_gpu = False
    else:
        spacy.require_gpu(args.gpu_id)
        used_gpu = True

    nlp = spacy.load(args.model)

    transformer_pipes = [
        name for name in nlp.pipe_names
        if "transformer" in name.casefold()
    ]
    if not transformer_pipes:
        raise RuntimeError(
            f"{args.model!r} has no transformer component: {nlp.pipe_names}"
        )
    if "ner" not in nlp.pipe_names:
        raise RuntimeError(
            f"{args.model!r} has no 'ner' component: {nlp.pipe_names}"
        )

    raw_labels = list(nlp.get_pipe("ner").labels)
    model_labels = set()
    for x in raw_labels:
        y = str(x).strip().upper()
        if y == "WORK_OF_ART":
            y = "WORK OF ART"
        model_labels.add(y)

    return nlp, model_labels, used_gpu, raw_labels, transformer_pipes

def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {
        x.lower() if x.startswith(".") else "." + x.lower()
        for x in args.extensions
    }

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    input_paths = discover_files(
        args.input_dir,
        extensions=extensions,
        include_substrings=args.include_substrings,
    )
    if not input_paths:
        raise RuntimeError(f"No input files found under {args.input_dir}")
    if args.expected_files > 0 and len(input_paths) != args.expected_files:
        raise RuntimeError(
            f"Expected {args.expected_files} input files, found {len(input_paths)}:\n"
            + "\n".join(map(str, input_paths))
        )

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Input sessions / LOO folds: {len(input_paths)}")
    for pth in input_paths:
        print(f"  - {pth}")

    print("\nLoading DaCy model...")
    nlp, model_labels, used_gpu, raw_model_labels, transformer_pipes = load_spacy_model(args)
    print(f"Model: {args.model}")
    print(f"GPU active: {used_gpu}")
    print("Model NER labels (native DaCy space):")
    print("  " + ", ".join(sorted(model_labels)))
    print("Transformer component(s): " + ", ".join(transformer_pipes))
    print("GDPR-ALF CHILD target labels (evaluation space):")
    print("  " + ", ".join(sorted(GDPR_ALF_TARGET_LABELS)))

    missing_expected = EXPECTED_DACY_LABELS - model_labels
    extra_model = model_labels - EXPECTED_DACY_LABELS
    if missing_expected:
        print(
            "WARNING: installed model is missing expected labels: "
            + ", ".join(sorted(missing_expected)),
            file=sys.stderr,
        )
    if extra_model:
        print(
            "NOTE: installed model has extra labels: "
            + ", ".join(sorted(extra_model)),
            file=sys.stderr,
        )

    aliases = load_annotation_aliases(args)
    dacy_to_target = load_dacy_target_map(args)

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
        in_schema_annotation_count += status_counts[
            "in_gdpr_alf_child_schema"
        ]
        input_annotation_audit.extend(audit)

        input_annotation_summary.append({
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
        })

        bundles.append(
            FileBundle(
                name=input_path.name,
                input_path=input_path,
                records=records,
            )
        )

    if raw_annotation_count == 0:
        raise RuntimeError(
            "No inline [PersonData]mention[LABEL] annotations were found in "
            "--input-dir. In this single-directory design, the same already-"
            "Presidio-preprocessed files must contain the few-shot child labels."
        )
    if in_schema_annotation_count == 0:
        raise RuntimeError(
            "Annotations were found, but none match the CHILD labels in the "
            "supplied GDPR-ALF schema. See input_annotation_audit.csv."
        )

    write_csv(
        args.output_dir / "input_annotation_audit.csv",
        input_annotation_audit,
    )
    write_csv(
        args.output_dir / "input_annotation_summary.csv",
        input_annotation_summary,
    )

    # Persist the PDF-aligned schema. MISC is included as a parent row with
    # no child target, exactly as specified in the supplied table.
    schema_rows = []
    for parent, children in GDPR_ALF_PARENT_CHILDREN.items():
        if children:
            for child in children:
                schema_rows.append({
                    "parent_label": parent,
                    "child_label": child,
                    "is_child_target": True,
                })
        else:
            schema_rows.append({
                "parent_label": parent,
                "child_label": "",
                "is_child_target": False,
            })
    write_csv(
        args.output_dir / "gdpr_alf_target_schema.csv",
        schema_rows,
    )

    if args.preflight_only:
        print(
            "\nPreflight complete. Input annotation audit written; "
            "evaluation skipped."
        )
        return 0

    print("\nEncoding annotation-stripped Presidio-preprocessed utterances with DaCy...")
    for i, bundle in enumerate(bundles, start=1):
        print(f"  [{i}/{len(bundles)}] {bundle.name}")
        encode_bundle(
            nlp,
            bundle,
            batch_size=args.batch_size,
            normalize_vecs=args.normalize_embeddings,
            dacy_to_target=dacy_to_target,
        )

    # Release pipeline reference only after all vectors/base predictions exist.
    # The encoded records retain NumPy token vectors, not transformer tensors.
    rng_master = random.Random(args.seed)
    support_order = list(range(len(bundles)))
    rng_master.shuffle(support_order)

    pair_rows: List[dict] = []
    fold_rows: List[dict] = []
    label_rows: List[dict] = []
    support_shot_rows: List[dict] = []
    struct_transition_rows: List[dict] = []
    boundary_rows: List[dict] = []

    methods = ("dacy_zero_shot", "protobert", "structshot")

    for fold_number, support_idx in enumerate(support_order, start=1):
        support_bundle = bundles[support_idx]
        query_bundles = [
            b for j, b in enumerate(bundles) if j != support_idx
        ]

        fold_rng = random.Random(
            (args.seed + 1) * 1000003 + support_idx
        )

        (
            prototypes,
            struct_support,
            _vectors_by_label,
            support_label_sequences,
            shot_rows,
            boundary_expansions,
        ) = build_support(
            support_bundle,
            rng=fold_rng,
            max_o_support=args.max_o_support,
        )

        observed_labels = set(prototypes) - {"O"}

        for row in shot_rows:
            x = dict(row)
            x["fold"] = fold_number
            x["support_file"] = support_bundle.name
            x["observed_entity_class"] = (
                x["label"] in observed_labels and x["label"] != "O"
            )
            support_shot_rows.append(x)

        boundary_rows.append({
            "fold": fold_number,
            "support_file": support_bundle.name,
            "support_reference_entities_not_exactly_on_spacy_token_boundaries": (
                boundary_expansions
            ),
        })

        print(
            f"\nFold {fold_number}/{len(bundles)} "
            f"support={support_bundle.name}"
        )
        print(
            f"  observed supported labels ({len(observed_labels)}): "
            + (", ".join(sorted(observed_labels)) if observed_labels else "<none>")
        )

        # StructShot transition stats are identical for all query records in fold.
        _, _, abstract_transition = build_abstract_transition_matrix(
            support_label_sequences,
            sorted(observed_labels),
            args.transition_smoothing,
        )
        struct_transition_rows.append({
            "fold": fold_number,
            "support_file": support_bundle.name,
            "tau": args.structshot_tau,
            "smoothing": args.transition_smoothing,
            "transition_source": "support_only",
            **abstract_transition,
        })

        # Fold-level counters by method/scope.
        fold_reference = {
            scope: Counter()
            for scope in ("support_observed", "all_schema")
        }
        fold_pred = {
            (method, scope): Counter()
            for method in methods
            for scope in ("support_observed", "all_schema")
        }

        for query_bundle in query_bundles:
            pair_reference = {
                scope: Counter()
                for scope in ("support_observed", "all_schema")
            }
            pair_pred = {
                (method, scope): Counter()
                for method in methods
                for scope in ("support_observed", "all_schema")
            }

            for rec in query_bundle.encoded:
                # Held-out reference annotations and zero-shot baseline.
                base_ents = rec.base_entities

                # Few-shot methods. If no entity labels occur in support, both
                # methods can only emit O.
                if observed_labels:
                    proto_labels = proto_predict(rec, prototypes)
                    proto_ents = labels_to_entities(rec, proto_labels)

                    struct_labels, _ = structshot_predict(
                        rec,
                        support=struct_support,
                        label_sequences=support_label_sequences,
                        tau=args.structshot_tau,
                        smoothing=args.transition_smoothing,
                    )
                    struct_ents = labels_to_entities(rec, struct_labels)
                else:
                    proto_ents = []
                    struct_ents = []

                pred_by_method = {
                    "dacy_zero_shot": base_ents,
                    "protobert": proto_ents,
                    "structshot": struct_ents,
                }

                allowed_by_scope = {
                    "support_observed": observed_labels,
                    "all_schema": set(GDPR_ALF_TARGET_LABELS),
                }

                for scope, allowed in allowed_by_scope.items():
                    g = entity_counter(
                        query_bundle.name,
                        rec,
                        rec.reference_entities,
                        allowed,
                    )
                    pair_reference[scope].update(g)
                    fold_reference[scope].update(g)

                    for method, ents in pred_by_method.items():
                        p = entity_counter(
                            query_bundle.name,
                            rec,
                            ents,
                            allowed,
                        )
                        pair_pred[(method, scope)].update(p)
                        fold_pred[(method, scope)].update(p)

            for scope in ("support_observed", "all_schema"):
                for method in methods:
                    s = score_counters(
                        pair_reference[scope],
                        pair_pred[(method, scope)],
                    )
                    pair_rows.append({
                        "fold": fold_number,
                        "support_file": support_bundle.name,
                        "query_file": query_bundle.name,
                        "method": method,
                        "scope": scope,
                        "n_observed_support_labels": len(observed_labels),
                        "observed_support_labels": "|".join(
                            sorted(observed_labels)
                        ),
                        **s,
                    })

        # Fold-level exact metrics and label metrics.
        for scope in ("support_observed", "all_schema"):
            labels_for_scope = (
                sorted(observed_labels)
                if scope == "support_observed"
                else sorted(GDPR_ALF_TARGET_LABELS)
            )

            for method in methods:
                s = score_counters(
                    fold_reference[scope],
                    fold_pred[(method, scope)],
                )
                fold_rows.append({
                    "fold": fold_number,
                    "support_file": support_bundle.name,
                    "method": method,
                    "scope": scope,
                    "n_query_files": len(query_bundles),
                    "n_observed_support_labels": len(observed_labels),
                    "observed_support_labels": "|".join(
                        sorted(observed_labels)
                    ),
                    **s,
                })

                for label in labels_for_scope:
                    sg = filter_counter_label(fold_reference[scope], label)
                    sp = filter_counter_label(
                        fold_pred[(method, scope)], label
                    )
                    sl = score_counters(sg, sp)
                    label_rows.append({
                        "fold": fold_number,
                        "support_file": support_bundle.name,
                        "method": method,
                        "scope": scope,
                        "label": label,
                        **sl,
                    })

    average_rows = aggregate_average_metrics(fold_rows)
    average_label_rows = aggregate_average_label_metrics(label_rows)

    write_csv(args.output_dir / "pair_metrics.csv", pair_rows)
    write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "fold_label_metrics.csv", label_rows)
    write_csv(args.output_dir / "average_metrics.csv", average_rows)
    write_csv(
        args.output_dir / "average_label_metrics.csv",
        average_label_rows,
    )
    write_csv(args.output_dir / "support_shots.csv", support_shot_rows)
    write_csv(
        args.output_dir / "structshot_transition_audit.csv",
        struct_transition_rows,
    )
    write_csv(
        args.output_dir / "token_boundary_audit.csv",
        boundary_rows,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "dacy_native_labels": sorted(model_labels),
        "raw_model_labels": raw_model_labels,
        "gdpr_alf_target_labels": sorted(GDPR_ALF_TARGET_LABELS),
        "dacy_to_gdpr_alf_mapping": dacy_to_target,
        "transformer_pipes": transformer_pipes,
        "gpu_active": used_gpu,
        "seed": args.seed,
        "n_files": len(bundles),
        "support_order": [bundles[i].name for i in support_order],
        "methods": list(methods),
        "metric": "exact entity-level span + label match",
        "scopes": {
            "support_observed": (
                "Classic few-shot-style scope: only entity labels with at least "
                "one surviving support example in the support file."
            ),
            "all_schema": (
                "Whole supplied GDPR-ALF CHILD target schema. For ProtoBERT/StructShot, "
                "schema labels absent from the current support file cannot be "
                "predicted in that fold and therefore become false negatives "
                "when they occur in query files."
            ),
        },
        "input_policy": (
            "A single --input-dir is used. Each file is already Presidio-"
            "preprocessed and contains inline child labels. Annotation wrappers "
            "are stripped before model inference; no separate gold/validation "
            "directory or cross-file alignment step exists."
        ),
        "structshot_transition_source": "support_only",
        "structshot_tau": args.structshot_tau,
        "transition_smoothing": args.transition_smoothing,
        "max_o_support": args.max_o_support,
        "normalize_embeddings": args.normalize_embeddings,
        "annotation_marker": args.annotation_marker,
        "annotation_to_gdpr_alf_aliases": aliases,
        "files": [
            {
                "name": b.name,
                "input_path": str(b.input_path),
            }
            for b in bundles
        ],
        "outputs": [
            "average_metrics.csv",
            "average_label_metrics.csv",
            "fold_metrics.csv",
            "fold_label_metrics.csv",
            "pair_metrics.csv",
            "support_shots.csv",
            "input_annotation_summary.csv",
            "input_annotation_audit.csv",
            "gdpr_alf_target_schema.csv",
            "structshot_transition_audit.csv",
            "token_boundary_audit.csv",
        ],
    }
    with (args.output_dir / "run_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_average_table(average_rows)
    print(f"\nResults written to: {args.output_dir.resolve()}")
    print("Primary table: average_metrics.csv")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
