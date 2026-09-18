#!/usr/bin/env python3
"""
Stage 3: Danish fine-grained PII / named-entity recognition with
chcaa/da_dacy_large_ner_fine_grained.

Expected upstream annotation format
-----------------------------------
    [PersonData]text[LABEL]

Cascade policy
--------------
* Existing upstream [PersonData]text[LABEL] spans are preserved and hidden
  from DaCy.
* Labels specified by --reprocess-labels (default: MISC) are exceptions:
  their wrapper is hidden, but their inner text is visible to DaCy.
* Standalone placeholders such as [IBAN], [URL], [IP] are hidden from DaCy.
* This stage performs recognition only. It does not de-identify or replace PII.
* Optional tagged copies use the same format:
      [PersonData]text[LABEL]

ALFRRTM safety
--------------
For .alfrttm files, and for .txt lines which look like ALFRRTM records, ONLY
text inside the outer quotation marks is sent to DaCy. Metadata such as
    start=0.1s stop=3.2s speaker_CLINICIAN
is never model input.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import runpy
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import spacy


HF_MODEL_NAME = "chcaa/da_dacy_large_ner_fine_grained"
SPACY_MODEL_NAME = "da_dacy_large_ner_fine_grained"
SCRIPT_VERSION = "2026-08-31-stage3-persondata-v4"
DEFAULT_EXTENSIONS = (".txt", ".alfrttm")

# Detect ALFRRTM-like records even if stored with a .txt suffix.
ALFRRTM_PREFIX_RE = re.compile(
    r'^\s*start\s*=\s*\S+\s+stop\s*=\s*\S+\s+speaker_[^\s]+(?:\s+|$)',
    flags=re.IGNORECASE,
)

# Generic bracket token, e.g. [PersonData], [PER], [/PER], [WORK_OF_ART].
TAG_RE = re.compile(r"\[(?P<close>/)?(?P<label>[A-Za-z][A-Za-z0-9_ -]*)\]")

EXPECTED_LABELS = {
    "CARDINAL",
    "DATE",
    "EVENT",
    "FACILITY",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOCATION",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORGANIZATION",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK OF ART",
}


@dataclass(frozen=True)
class TagToken:
    index: int
    label: str
    start: int
    end: int
    is_closing: bool


@dataclass(frozen=True)
class UpstreamSpan:
    label: str
    style: str  # persondata or legacy_pair
    full_start: int
    full_end: int
    inner_start: int
    inner_end: int
    left_wrapper_start: int
    left_wrapper_end: int
    right_wrapper_start: int
    right_wrapper_end: int


@dataclass(frozen=True)
class StandaloneToken:
    label: str
    start: int
    end: int
    is_closing: bool


def annotation_safe_label(label: str) -> str:
    return label.strip().upper().replace(" ", "_")


def normalize_model_label(label: str) -> str:
    return label.strip().upper().replace("_", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-3 recognition with chcaa/da_dacy_large_ner_fine_grained; "
            "protect upstream annotations and process only transcript utterances."
        )
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annonydata") / "dacy_finegrained_pii",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Optional DaCy label filter. Default keeps all model labels. "
            "Use WORK_OF_ART for WORK OF ART."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--reprocess-labels",
        nargs="+",
        default=["MISC"],
        help=(
            "Upstream labels whose inner text is reprocessed by DaCy. "
            "All other upstream annotations are protected. Default: MISC."
        ),
    )
    parser.add_argument(
        "--annotation-marker",
        default="PersonData",
        help=(
            "Inline annotation marker. Default PersonData gives "
            "[PersonData]text[LABEL]."
        ),
    )
    parser.add_argument(
        "--strict-quoted-utterance",
        action="store_true",
        help=(
            "For ALFRRTM-like records, skip malformed lines that do not contain "
            "a valid quoted utterance. Metadata is never passed to DaCy."
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--write-tagged-files",
        action="store_true",
        help=(
            "Write Stage-3 annotated transcript copies using "
            "[PersonData]text[LABEL]. Inputs are not modified."
        ),
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "dacy_finegrained_pii.log"

    try:
        from loger import setup_logging as project_setup_logging  # type: ignore

        return project_setup_logging(str(log_path))
    except Exception:
        logger = logging.getLogger("dacy_finegrained_pii")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(sh)
        return logger


def find_gpu_cleaner() -> Path:
    script_dir = Path(__file__).resolve().parent
    project_dir = Path.cwd().resolve()
    candidates = [
        script_dir / "tool_clean_gpu.py",
        project_dir / "pseudopipe" / "tool_clean_gpu.py",
        project_dir / "tool_clean_gpu.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find tool_clean_gpu.py. Searched:\n  - " + searched
    )


def run_gpu_cleaner(cleaner: Path, logger: logging.Logger, stage: str) -> None:
    logger.info("GPU cleanup (%s): %s", stage, cleaner)
    gc.collect()
    runpy.run_path(str(cleaner), run_name=f"__gpu_cleanup_{stage}__")


def normalize_extensions(values: Iterable[str]) -> set[str]:
    return {
        v.lower() if v.startswith(".") else f".{v.lower()}"
        for v in values
    }


def normalize_requested_labels(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {normalize_model_label(v) for v in values}
    unknown = normalized - EXPECTED_LABELS
    if unknown:
        raise ValueError(
            "Unknown --labels value(s): "
            + ", ".join(sorted(unknown))
            + ". Valid labels: "
            + ", ".join(sorted(EXPECTED_LABELS))
        )
    return normalized


def normalize_reprocess_labels(values: Iterable[str]) -> set[str]:
    labels = {annotation_safe_label(v) for v in values if v.strip()}
    if not labels:
        raise ValueError("--reprocess-labels must contain at least one label.")
    return labels


def validate_annotation_marker(marker: str) -> str:
    marker = marker.strip()
    if not marker:
        raise ValueError("--annotation-marker cannot be empty.")
    if any(ch in marker for ch in "[]\r\n"):
        raise ValueError("--annotation-marker must not contain [, ], or newlines.")
    return marker


def iter_input_files(
    input_dir: Path,
    extensions: set[str],
    output_dir: Path,
) -> list[Path]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    files: list[Path] = []

    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        try:
            path.resolve().relative_to(output_dir)
            continue
        except ValueError:
            pass
        files.append(path)
    return sorted(files)


def looks_like_alfrttm(raw: str, suffix: str) -> bool:
    return suffix.lower() == ".alfrttm" or bool(ALFRRTM_PREFIX_RE.match(raw))


def extract_quoted_utterance(raw: str) -> tuple[str, int, int] | None:
    """Return text between the first and final double quote."""
    first = raw.find('"')
    last = raw.rfind('"')
    if first < 0 or last <= first:
        return None
    return raw[first + 1 : last], first + 1, last


def extract_segment(
    line: str,
    suffix: str,
) -> tuple[str, int, int, bool]:
    """
    Return (model_segment, start_in_line, end_in_line, is_alfrttm_like).

    ALFRRTM metadata is NEVER returned as model text.
    """
    raw = line.rstrip("\r\n")
    is_alfrttm = looks_like_alfrttm(raw, suffix)

    if is_alfrttm:
        quoted = extract_quoted_utterance(raw)
        if quoted is None:
            return "", 0, 0, True
        segment, start, end = quoted
        return segment, start, end, True

    return raw, 0, len(raw), False


def tokenize_markup(text: str) -> list[TagToken]:
    tokens: list[TagToken] = []
    for idx, match in enumerate(TAG_RE.finditer(text)):
        tokens.append(
            TagToken(
                index=idx,
                label=annotation_safe_label(match.group("label")),
                start=match.start(),
                end=match.end(),
                is_closing=bool(match.group("close")),
            )
        )
    return tokens


def parse_upstream_markup(
    text: str,
    annotation_marker: str,
) -> tuple[list[UpstreamSpan], list[StandaloneToken]]:
    """
    Parse both:
        [PersonData]text[LABEL]
    and legacy:
        [LABEL]text[/LABEL]

    Any unmatched tag is treated as a standalone placeholder and hidden.
    """
    tokens = tokenize_markup(text)
    marker_label = annotation_safe_label(annotation_marker)
    consumed: set[int] = set()
    spans: list[UpstreamSpan] = []

    # Current format: [PersonData]inner[LABEL]
    for pos, token in enumerate(tokens):
        if token.index in consumed:
            continue
        if token.is_closing or token.label != marker_label:
            continue

        terminal: TagToken | None = None
        for candidate in tokens[pos + 1 :]:
            if candidate.index in consumed:
                continue
            if candidate.is_closing:
                continue
            # A new marker means the current annotation is malformed; do not
            # swallow another annotation while searching for a terminal label.
            if candidate.label == marker_label:
                break
            terminal = candidate
            break

        if terminal is None:
            continue

        consumed.add(token.index)
        consumed.add(terminal.index)
        spans.append(
            UpstreamSpan(
                label=terminal.label,
                style="persondata",
                full_start=token.start,
                full_end=terminal.end,
                inner_start=token.end,
                inner_end=terminal.start,
                left_wrapper_start=token.start,
                left_wrapper_end=token.end,
                right_wrapper_start=terminal.start,
                right_wrapper_end=terminal.end,
            )
        )

    # Legacy [LABEL]inner[/LABEL] for backward compatibility.
    stack: list[TagToken] = []
    for token in tokens:
        if token.index in consumed:
            continue
        if token.label == marker_label:
            continue

        if not token.is_closing:
            stack.append(token)
            continue

        open_pos = None
        for idx in range(len(stack) - 1, -1, -1):
            if stack[idx].label == token.label:
                open_pos = idx
                break
        if open_pos is None:
            continue

        opener = stack.pop(open_pos)
        consumed.add(opener.index)
        consumed.add(token.index)
        spans.append(
            UpstreamSpan(
                label=opener.label,
                style="legacy_pair",
                full_start=opener.start,
                full_end=token.end,
                inner_start=opener.end,
                inner_end=token.start,
                left_wrapper_start=opener.start,
                left_wrapper_end=opener.end,
                right_wrapper_start=token.start,
                right_wrapper_end=token.end,
            )
        )

    standalone = [
        StandaloneToken(
            label=t.label,
            start=t.start,
            end=t.end,
            is_closing=t.is_closing,
        )
        for t in tokens
        if t.index not in consumed
    ]

    return sorted(spans, key=lambda s: (s.full_start, s.full_end)), standalone


def build_masked_model_text(
    text: str,
    annotation_marker: str,
    reprocess_labels: set[str],
) -> tuple[str, list[bool], list[UpstreamSpan], list[StandaloneToken]]:
    """
    Create same-length text for DaCy.

    * all annotation syntax is masked;
    * non-reprocessed upstream spans are masked completely;
    * reprocessed spans (MISC) expose only their inner text.
    """
    spans, standalone = parse_upstream_markup(text, annotation_marker)
    protected = [False] * len(text)

    for span in spans:
        # Wrappers are always model-invisible.
        for i in range(span.left_wrapper_start, span.left_wrapper_end):
            protected[i] = True
        for i in range(span.right_wrapper_start, span.right_wrapper_end):
            protected[i] = True

        # Upstream entities are also model-invisible unless explicitly reopened.
        if span.label not in reprocess_labels:
            for i in range(span.inner_start, span.inner_end):
                protected[i] = True

    # Standalone placeholders/unmatched markup are model-invisible.
    for token in standalone:
        for i in range(token.start, token.end):
            protected[i] = True

    chars = list(text)
    for i, is_protected in enumerate(protected):
        if is_protected and chars[i] not in "\r\n":
            chars[i] = " "

    return "".join(chars), protected, spans, standalone


def entity_overlaps_protected(start: int, end: int, protected: list[bool]) -> bool:
    if start < 0 or end > len(protected) or start >= end:
        return True
    return any(protected[start:end])


def containing_reprocessed_labels(
    start: int,
    end: int,
    spans: list[UpstreamSpan],
    reprocess_labels: set[str],
) -> list[str]:
    labels = {
        span.label
        for span in spans
        if span.label in reprocess_labels
        and start >= span.inner_start
        and end <= span.inner_end
    }
    return sorted(labels)


def configure_device(force_cpu: bool, logger: logging.Logger) -> str:
    if force_cpu:
        spacy.require_cpu()
        logger.info("Inference device: CPU (--cpu requested)")
        return "cpu"

    using_gpu = bool(spacy.prefer_gpu())
    device = "gpu" if using_gpu else "cpu"
    logger.info("Inference device selected by spaCy: %s", device.upper())
    return device


def load_model(logger: logging.Logger):
    logger.info("Loading model: %s", HF_MODEL_NAME)
    logger.info("spaCy package name: %s", SPACY_MODEL_NAME)
    try:
        nlp = spacy.load(SPACY_MODEL_NAME)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load spaCy model package '{SPACY_MODEL_NAME}'. "
            "Install it inside vdeidspacy first."
        ) from exc

    if "ner" not in nlp.pipe_names:
        raise RuntimeError(f"Loaded model has no NER component: {nlp.pipe_names}")

    labels = set(nlp.get_pipe("ner").labels)
    logger.info("Pipeline components: %s", ", ".join(nlp.pipe_names))
    logger.info("NER labels (%d): %s", len(labels), ", ".join(sorted(labels)))
    return nlp


def entities_from_doc(
    doc,
    selected_labels: set[str] | None,
    protected: list[bool],
    spans: list[UpstreamSpan],
    reprocess_labels: set[str],
    original_text: str,
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []

    for ent in doc.ents:
        model_label = normalize_model_label(str(ent.label_))
        if selected_labels is not None and model_label not in selected_labels:
            continue

        start = int(ent.start_char)
        end = int(ent.end_char)
        if entity_overlaps_protected(start, end, protected):
            continue

        entity_text = original_text[start:end]
        if not entity_text.strip():
            continue

        upstream_labels = containing_reprocessed_labels(
            start, end, spans, reprocess_labels
        )
        entities.append(
            {
                "label": model_label,
                "annotation_label": annotation_safe_label(model_label),
                "text": entity_text,
                "start": start,
                "end": end,
                "inside_reprocessed_upstream_label": bool(upstream_labels),
                "reprocessed_upstream_labels": "|".join(upstream_labels),
            }
        )
    return entities


def build_tagged_segment(
    text: str,
    entities: list[dict[str, Any]],
    spans: list[UpstreamSpan],
    reprocess_labels: set[str],
    annotation_marker: str,
) -> str:
    """
    Emit [PersonData]text[LABEL].

    Non-MISC upstream annotations are preserved byte-for-byte. A reprocessed
    MISC wrapper is removed only if Stage 3 actually found at least one entity
    inside that MISC span; otherwise the original MISC annotation is retained.
    """
    remove = [False] * len(text)

    # Determine which reprocessed spans actually yielded a finer entity.
    for span in spans:
        if span.label not in reprocess_labels:
            continue
        has_inner_entity = any(
            int(ent["start"]) >= span.inner_start
            and int(ent["end"]) <= span.inner_end
            for ent in entities
        )
        if not has_inner_entity:
            continue
        for i in range(span.left_wrapper_start, span.left_wrapper_end):
            remove[i] = True
        for i in range(span.right_wrapper_start, span.right_wrapper_end):
            remove[i] = True

    # Insert Stage-3 annotations at original character offsets.
    events: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for ent in entities:
        start = int(ent["start"])
        end = int(ent["end"])
        label = str(ent["annotation_label"])
        events[start].append((1, f"[{annotation_marker}]"))
        events[end].append((0, f"[{label}]"))

    out: list[str] = []
    for pos in range(len(text) + 1):
        if pos in events:
            for _, token in sorted(events[pos], key=lambda x: x[0]):
                out.append(token)
        if pos < len(text) and not remove[pos]:
            out.append(text[pos])
    return "".join(out)


def update_upstream_counts(
    spans: list[UpstreamSpan],
    standalone: list[StandaloneToken],
    reprocess_labels: set[str],
    counter: Counter,
) -> None:
    for span in spans:
        policy = (
            "REPROCESS_INNER_TEXT"
            if span.label in reprocess_labels
            else "PROTECT_AND_IGNORE"
        )
        counter[(span.label, span.style, policy)] += 1

    for token in standalone:
        kind = "unmatched_closing_tag" if token.is_closing else "standalone_placeholder"
        counter[(token.label, kind, "PROTECT_AND_IGNORE")] += 1


def process_file(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    nlp,
    selected_labels: set[str] | None,
    reprocess_labels: set[str],
    annotation_marker: str,
    strict_quoted_utterance: bool,
    batch_size: int,
    write_tagged: bool,
    upstream_counter: Counter,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    relative = path.resolve().relative_to(input_dir.resolve())
    logger.info("Processing: %s", relative)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    jobs: list[dict[str, Any]] = []

    for line_number, line in enumerate(lines, start=1):
        segment, segment_start, _, is_alfrttm = extract_segment(line, path.suffix)

        if is_alfrttm and not segment:
            raw = line.rstrip("\r\n")
            if strict_quoted_utterance and raw.strip():
                logger.warning(
                    "Skipping malformed ALFRRTM-like line without quoted utterance: %s:%d",
                    relative,
                    line_number,
                )
            # Crucial: never fall back to processing the metadata line.
            continue

        if not segment.strip():
            continue

        masked_text, protected, spans, standalone = build_masked_model_text(
            text=segment,
            annotation_marker=annotation_marker,
            reprocess_labels=reprocess_labels,
        )
        update_upstream_counts(spans, standalone, reprocess_labels, upstream_counter)

        jobs.append(
            {
                "line_number": line_number,
                "segment": segment,
                "segment_start": segment_start,
                "masked_text": masked_text if masked_text.strip() else None,
                "protected": protected,
                "spans": spans,
            }
        )

    if not jobs:
        return []

    model_jobs = [job for job in jobs if job["masked_text"] is not None]
    if model_jobs:
        longest = max(len(str(job["masked_text"])) for job in model_jobs)
        if longest >= nlp.max_length:
            nlp.max_length = longest + 1000
            logger.warning("Raised nlp.max_length to %d for %s", nlp.max_length, relative)

    docs_by_line: dict[int, Any] = {}
    if model_jobs:
        docs = nlp.pipe(
            [str(job["masked_text"]) for job in model_jobs],
            batch_size=batch_size,
        )
        for job, doc in zip(model_jobs, docs):
            docs_by_line[int(job["line_number"])] = doc

    records: list[dict[str, Any]] = []
    tagged_by_line: dict[int, str] = {}

    for job in jobs:
        line_number = int(job["line_number"])
        original_segment = str(job["segment"])
        segment_start = int(job["segment_start"])
        spans = job["spans"]

        doc = docs_by_line.get(line_number)
        if doc is None:
            entities: list[dict[str, Any]] = []
        else:
            entities = entities_from_doc(
                doc=doc,
                selected_labels=selected_labels,
                protected=job["protected"],
                spans=spans,
                reprocess_labels=reprocess_labels,
                original_text=original_segment,
            )

        for ent in entities:
            records.append(
                {
                    "file": str(relative),
                    "line_number": line_number,
                    "model": HF_MODEL_NAME,
                    "label": ent["label"],
                    "annotation_label": ent["annotation_label"],
                    "text": ent["text"],
                    "start_in_segment": int(ent["start"]),
                    "end_in_segment": int(ent["end"]),
                    "start_in_line": segment_start + int(ent["start"]),
                    "end_in_line": segment_start + int(ent["end"]),
                    "inside_reprocessed_upstream_label": ent[
                        "inside_reprocessed_upstream_label"
                    ],
                    "reprocessed_upstream_labels": ent[
                        "reprocessed_upstream_labels"
                    ],
                    "annotation": (
                        f"[{annotation_marker}]"
                        f"{ent['text']}"
                        f"[{ent['annotation_label']}]"
                    ),
                }
            )

        if write_tagged:
            tagged_by_line[line_number] = build_tagged_segment(
                text=original_segment,
                entities=entities,
                spans=spans,
                reprocess_labels=reprocess_labels,
                annotation_marker=annotation_marker,
            )

    if write_tagged:
        tagged_lines: list[str] = []
        for line_number, line in enumerate(lines, start=1):
            raw_no_newline = line.rstrip("\r\n")
            newline = line[len(raw_no_newline) :]

            if line_number not in tagged_by_line:
                tagged_lines.append(line)
                continue

            segment, segment_start, _, _ = extract_segment(line, path.suffix)
            tagged_segment = tagged_by_line[line_number]
            rebuilt = (
                raw_no_newline[:segment_start]
                + tagged_segment
                + raw_no_newline[segment_start + len(segment) :]
            )
            tagged_lines.append(rebuilt + newline)

        # Preserve the Stage-2 relative filename exactly.
        tagged_path = output_dir / "tagged" / relative
        tagged_path.parent.mkdir(parents=True, exist_ok=True)
        tagged_path.write_text("".join(tagged_lines), encoding="utf-8")

    return records


def write_outputs(
    records: list[dict[str, Any]],
    upstream_counter: Counter,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "dacy_entities.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    fieldnames = [
        "file",
        "line_number",
        "model",
        "label",
        "annotation_label",
        "text",
        "start_in_segment",
        "end_in_segment",
        "start_in_line",
        "end_in_line",
        "inside_reprocessed_upstream_label",
        "reprocessed_upstream_labels",
        "annotation",
    ]
    csv_path = output_dir / "dacy_entities.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    counts = Counter(str(record["label"]) for record in records)
    summary_path = output_dir / "dacy_entity_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "count"])
        writer.writeheader()
        for label, count in sorted(counts.items()):
            writer.writerow({"label": label, "count": count})

    prior_summary_path = output_dir / "dacy_prior_tag_summary.csv"
    with prior_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["upstream_label", "kind", "count", "stage3_policy"],
        )
        writer.writeheader()
        for (label, kind, policy), count in sorted(upstream_counter.items()):
            writer.writerow(
                {
                    "upstream_label": label,
                    "kind": kind,
                    "count": count,
                    "stage3_policy": policy,
                }
            )


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    selected_labels = normalize_requested_labels(args.labels)
    reprocess_labels = normalize_reprocess_labels(args.reprocess_labels)
    annotation_marker = validate_annotation_marker(args.annotation_marker)
    extensions = normalize_extensions(args.extensions)

    logger = setup_logging(output_dir)
    cleaner = find_gpu_cleaner()
    files = iter_input_files(input_dir, extensions, output_dir)

    logger.info("Script version: %s", SCRIPT_VERSION)
    logger.info("Project cwd: %s", Path.cwd().resolve())
    logger.info("Stage-3 input directory: %s", input_dir)
    logger.info("Stage-3 output directory: %s", output_dir)
    logger.info("Matched transcript files: %d", len(files))
    logger.info("Annotation format: [%s]text[LABEL]", annotation_marker)
    logger.info(
        "Protected upstream annotations; reprocess inner text for: %s",
        ", ".join(sorted(reprocess_labels)),
    )
    logger.info(
        "ALFRRTM policy: ONLY quoted utterance goes to DaCy; metadata excluded"
    )

    run_gpu_cleaner(cleaner, logger, "before")

    nlp = None
    try:
        configure_device(args.cpu, logger)
        nlp = load_model(logger)

        all_records: list[dict[str, Any]] = []
        upstream_counter: Counter = Counter()

        for path in files:
            all_records.extend(
                process_file(
                    path=path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    nlp=nlp,
                    selected_labels=selected_labels,
                    reprocess_labels=reprocess_labels,
                    annotation_marker=annotation_marker,
                    strict_quoted_utterance=args.strict_quoted_utterance,
                    batch_size=args.batch_size,
                    write_tagged=args.write_tagged_files,
                    upstream_counter=upstream_counter,
                    logger=logger,
                )
            )

        write_outputs(all_records, upstream_counter, output_dir)

        logger.info("New Stage-3 entity spans: %d", len(all_records))
        logger.info("JSONL: %s", output_dir / "dacy_entities.jsonl")
        logger.info("CSV: %s", output_dir / "dacy_entities.csv")
        if args.write_tagged_files:
            logger.info("Tagged copies: %s", output_dir / "tagged")
        return 0

    finally:
        nlp = None
        gc.collect()
        try:
            run_gpu_cleaner(cleaner, logger, "after")
        except Exception:
            logger.exception("Post-run GPU cleanup failed.")


if __name__ == "__main__":
    raise SystemExit(main())
