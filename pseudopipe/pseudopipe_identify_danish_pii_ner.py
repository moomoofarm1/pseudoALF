#!/usr/bin/env python3
"""
Stage 2: Danish PER / ORG / LOC recognition with
thomasbeste/danish-xlmr-ner-large.

Input
-----
Stage-1 Presidio output transcripts (.txt and/or .alfrttm).

Important transcript rule
-------------------------
For ALFRRTM-style lines such as
    start=0.1s stop=3.2s speaker_CLINICIAN "Jeg hedder Peter."
ONLY the text inside the outer quotation marks is sent to the NER model.
This is enforced both for .alfrttm files and for .txt files whose lines look
like ALFRRTM records.

Inline annotation format
------------------------
New detections are written as:
    [PersonData]Peter Jensen[PER]
    [PersonData]Novo Nordisk[ORG]
    [PersonData]København[LOC]

The marker name is configurable with --annotation-marker and defaults to
PersonData.

Outputs
-------
<output_dir>/pii_entities.jsonl
<output_dir>/pii_entities.csv
<output_dir>/tagged/<relative input path>.tagged<suffix>

The script performs recognition/tagging only; it does not replace detected
PER/ORG/LOC text. GPU cleanup is run before model loading and after inference.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import runpy
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


MODEL_NAME = "thomasbeste/danish-xlmr-ner-large"
DEFAULT_EXTENSIONS = (".txt", ".alfrttm")

# Detect ALFRRTM by content as well as by suffix. This avoids feeding metadata
# to the model when an ALFRRTM-formatted transcript happens to be stored as .txt.
ALFRRTM_PREFIX_RE = re.compile(
    r"^\s*start\s*=\s*\S+\s+stop\s*=\s*\S+\s+speaker_[^\s]+\s+\"",
    flags=re.IGNORECASE,
)

# Existing Stage-1 placeholders such as [IBAN], [URL], [IP] are masked before
# NER so the model does not waste predictions on annotation tokens.
BRACKET_TOKEN_RE = re.compile(r"\[[A-Za-z][A-Za-z0-9_ -]*\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 Danish PER/ORG/LOC recognition with "
            "thomasbeste/danish-xlmr-ner-large."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annonydata") / "danish_per_loc_org_pii",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--annotation-marker",
        default="PersonData",
        help=(
            "Prefix marker used for inline annotations. Default: PersonData, "
            "yielding [PersonData]text[PER]."
        ),
    )
    parser.add_argument(
        "--strict-quoted-utterance",
        action="store_true",
        help=(
            "For ALFRRTM-like lines, skip malformed lines that do not contain a "
            "valid outer quoted utterance. Metadata is never sent to the model."
        ),
    )
    parser.add_argument(
        "--no-tagged-files",
        action="store_true",
        help="Do not write inline-annotated transcript copies.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "danish_xlmr_pii.log"

    try:
        from loger import setup_logging as project_setup_logging  # type: ignore

        return project_setup_logging(str(log_path))
    except Exception:
        logger = logging.getLogger("danish_xlmr_pii")
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
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in values
    }


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
    """Extract text between the first and last double quote, preserving offsets."""
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


def mask_existing_bracket_tokens(text: str) -> tuple[str, list[bool]]:
    """Mask existing [LABEL]-style tokens with spaces while preserving offsets."""
    protected = [False] * len(text)
    chars = list(text)
    for match in BRACKET_TOKEN_RE.finditer(text):
        for i in range(match.start(), match.end()):
            protected[i] = True
            chars[i] = " "
    return "".join(chars), protected


def span_overlaps_mask(start: int, end: int, protected: list[bool]) -> bool:
    if start < 0 or end > len(protected) or start >= end:
        return True
    return any(protected[start:end])


def build_ner(force_cpu: bool, logger: logging.Logger):
    use_cuda = torch.cuda.is_available() and not force_cpu
    device = 0 if use_cuda else -1

    logger.info("Loading model: %s", MODEL_NAME)
    logger.info("Inference device: %s", "cuda:0" if use_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model_kwargs: dict[str, Any] = {}
    if use_cuda:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )
    ner = pipeline(
        task="ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )
    return tokenizer, model, ner


def identify_segment(
    ner,
    original_text: str,
    model_text: str,
    protected: list[bool],
    stride: int,
    min_score: float,
) -> list[dict[str, Any]]:
    if not model_text.strip():
        return []

    predictions = ner(model_text, stride=stride)
    entities: list[dict[str, Any]] = []

    for pred in predictions:
        score = float(pred["score"])
        if score < min_score:
            continue

        start = int(pred["start"])
        end = int(pred["end"])
        if span_overlaps_mask(start, end, protected):
            continue

        label = str(pred.get("entity_group", pred.get("entity", ""))).upper()
        if label not in {"PER", "ORG", "LOC"}:
            continue

        entity_text = original_text[start:end]
        if not entity_text.strip():
            continue

        entities.append(
            {
                "label": label,
                "text": entity_text,
                "score": score,
                "start": start,
                "end": end,
            }
        )

    return entities


def annotation_text(marker: str, entity_text: str, label: str) -> str:
    return f"[{marker}]{entity_text}[{label}]"


def tag_segment(
    text: str,
    entities: list[dict[str, Any]],
    marker: str,
) -> str:
    """Insert [PersonData]entity[LABEL] annotations from right to left."""
    tagged = text
    for ent in sorted(
        entities,
        key=lambda item: (int(item["start"]), int(item["end"])),
        reverse=True,
    ):
        start = int(ent["start"])
        end = int(ent["end"])
        label = str(ent["label"])
        tagged = (
            tagged[:start]
            + f"[{marker}]"
            + tagged[start:end]
            + f"[{label}]"
            + tagged[end:]
        )
    return tagged


def process_file(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    ner,
    stride: int,
    min_score: float,
    annotation_marker: str,
    strict_quoted_utterance: bool,
    write_tagged: bool,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    relative = path.resolve().relative_to(input_dir.resolve())
    logger.info("Processing: %s", relative)

    records: list[dict[str, Any]] = []
    tagged_lines: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            segment, segment_start, _, is_alfrttm = extract_segment(line, path.suffix)

            if is_alfrttm and not segment and strict_quoted_utterance:
                raw = line.rstrip("\r\n")
                if raw.strip():
                    logger.warning(
                        "Skipping malformed ALFRRTM-like line without a valid quoted "
                        "utterance: %s:%d",
                        relative,
                        line_number,
                    )

            masked_segment, protected = mask_existing_bracket_tokens(segment)
            entities = identify_segment(
                ner=ner,
                original_text=segment,
                model_text=masked_segment,
                protected=protected,
                stride=stride,
                min_score=min_score,
            )

            for ent in entities:
                records.append(
                    {
                        "file": str(relative),
                        "line_number": line_number,
                        "label": ent["label"],
                        "text": ent["text"],
                        "score": round(float(ent["score"]), 6),
                        "start_in_segment": int(ent["start"]),
                        "end_in_segment": int(ent["end"]),
                        "start_in_line": segment_start + int(ent["start"]),
                        "end_in_line": segment_start + int(ent["end"]),
                        "annotation": annotation_text(
                            annotation_marker,
                            str(ent["text"]),
                            str(ent["label"]),
                        ),
                    }
                )

            if write_tagged:
                raw_no_newline = line.rstrip("\r\n")
                newline = line[len(raw_no_newline) :]

                if segment:
                    tagged_segment = tag_segment(
                        segment,
                        entities,
                        annotation_marker,
                    )
                    rebuilt = (
                        raw_no_newline[:segment_start]
                        + tagged_segment
                        + raw_no_newline[segment_start + len(segment) :]
                    )
                else:
                    rebuilt = raw_no_newline

                tagged_lines.append(rebuilt + newline)

    if write_tagged:
        tagged_root = output_dir / "tagged"
        tagged_path = (
            tagged_root
            / relative.parent
            / f"{relative.stem}.tagged{relative.suffix}"
        )
        tagged_path.parent.mkdir(parents=True, exist_ok=True)
        tagged_path.write_text("".join(tagged_lines), encoding="utf-8")

    return records


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "pii_entities.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = output_dir / "pii_entities.csv"
    fieldnames = [
        "file",
        "line_number",
        "label",
        "text",
        "score",
        "start_in_segment",
        "end_in_segment",
        "start_in_line",
        "end_in_line",
        "annotation",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    annotation_marker = validate_annotation_marker(args.annotation_marker)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if not 0.0 <= args.min_score <= 1.0:
        raise ValueError("--min-score must be between 0 and 1.")
    if args.stride < 0:
        raise ValueError("--stride must be >= 0.")

    logger = setup_logging(output_dir)
    cleaner = find_gpu_cleaner()
    extensions = normalize_extensions(args.extensions)
    files = iter_input_files(input_dir, extensions, output_dir)

    logger.info("Project cwd: %s", Path.cwd().resolve())
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Matched files: %d", len(files))
    logger.info(
        "Inline annotation format: [%s]text[LABEL]",
        annotation_marker,
    )
    logger.info(
        "ALFRRTM policy: model receives quoted utterance only; metadata is excluded"
    )

    run_gpu_cleaner(cleaner, logger, "before")

    tokenizer = None
    model = None
    ner = None
    try:
        tokenizer, model, ner = build_ner(args.cpu, logger)

        model_max_length = int(getattr(tokenizer, "model_max_length", 512))
        if args.stride >= model_max_length:
            raise ValueError(
                f"--stride ({args.stride}) must be smaller than tokenizer "
                f"model_max_length ({model_max_length})."
            )

        all_records: list[dict[str, Any]] = []
        for path in files:
            all_records.extend(
                process_file(
                    path=path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    ner=ner,
                    stride=args.stride,
                    min_score=args.min_score,
                    annotation_marker=annotation_marker,
                    strict_quoted_utterance=args.strict_quoted_utterance,
                    write_tagged=not args.no_tagged_files,
                    logger=logger,
                )
            )

        write_outputs(all_records, output_dir)

        logger.info("Detected entity spans: %d", len(all_records))
        logger.info("JSONL: %s", output_dir / "pii_entities.jsonl")
        logger.info("CSV: %s", output_dir / "pii_entities.csv")
        if not args.no_tagged_files:
            logger.info("Tagged copies: %s", output_dir / "tagged")
        return 0

    finally:
        ner = None
        model = None
        tokenizer = None
        gc.collect()
        try:
            run_gpu_cleaner(cleaner, logger, "after")
        except Exception:
            logger.exception("Post-run GPU cleanup failed.")


if __name__ == "__main__":
    raise SystemExit(main())
