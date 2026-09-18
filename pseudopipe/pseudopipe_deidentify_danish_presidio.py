#!/usr/bin/env python3
"""
De-identify Danish transcript files using Microsoft Presidio.

Designed to be launched from a notebook/project root using a dedicated conda
environment, e.g.:

    conda run --no-capture-output -n vdeidpresidio \
        python pseudopipe/preproc_deidentify_danish_presidio.py \
        --input-dir transcripts \
        --output-dir annonydata/presidio_deidentified \
        --seed 20260831

Detection methods
-----------------
CPR
    Custom Presidio PatternRecognizer + Danish CPR structural/date validation.
Danish phone
    Presidio PhoneRecognizer configured with supported_regions=["DK"].
Email
    Presidio EmailRecognizer (pattern/validation logic from Presidio).
IBAN
    Presidio IbanRecognizer (pattern + checksum validation).
URL
    Presidio UrlRecognizer.
IP
    Presidio IpRecognizer.
Danish postal code
    Custom Presidio PatternRecognizer plus local Danish context filtering.
Dates
    Presidio DateRecognizer plus a Danish textual-month PatternRecognizer.

Replacement plan
----------------
The replacements below implement the supplied project proposal:
- CPR: random 10-digit string
- PHONE: +45 + random 8-digit string
- EMAIL: one of the proposal's common names + @gmail.com
- POSTCODE: random 2-digit number + 00
- DATE: random day/month/year with day 1-30, month 1-12, year 2000-2026

The proposal does not define replacement values for IBAN, URL, or IP. Those are
therefore conservatively replaced with literal placeholders [IBAN], [URL], and
[IP].

Important
---------
The input transcript is assumed NOT to contain pre-existing PII tags. Presidio
first identifies spans and Presidio AnonymizerEngine then performs replacement.

For .alfrttm files, only the final quoted utterance is analyzed/de-identified;
timestamps and speaker metadata are preserved exactly.

Outputs
-------
<output_dir>/deidentified/<same relative input path>
<output_dir>/presidio_entities.jsonl
<output_dir>/presidio_entities.csv
<output_dir>/presidio_summary.csv
<output_dir>/presidio_deid.log
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

try:
    from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult
    from presidio_analyzer.predefined_recognizers import (
        DateRecognizer,
        EmailRecognizer,
        IbanRecognizer,
        IpRecognizer,
        PhoneRecognizer,
        UrlRecognizer,
    )
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except ImportError as exc:  # pragma: no cover - executed only in target env
    raise SystemExit(
        "Presidio is not available in this Python environment. Run this script "
        "inside the vdeidpresidio environment and install presidio-analyzer "
        "and presidio-anonymizer.\nOriginal import error: " + str(exc)
    ) from exc


DEFAULT_EXTENSIONS = (".txt", ".alfrttm")
QUOTE_RE = re.compile(r'"(?P<utterance>.*)"\s*$')

# Names specified in the uploaded replacement proposal.
PDF_EMAIL_NAMES = (
    "Anne",
    "Mette",
    "Kirsten",
    "Hanne",
    "Anna",
    "Peter",
    "Michael",
    "Lars",
    "Thomas",
    "Jens",
)

# Cross-entity overlap priority. A 10-digit CPR may otherwise contain an
# 8-digit substring which PhoneRecognizer regards as a Danish phone number.
ENTITY_PRIORITY = {
    "DK_CPR": 100,
    "EMAIL_ADDRESS": 90,
    "IBAN_CODE": 90,
    "URL": 85,
    "IP_ADDRESS": 85,
    "DK_PHONE_NUMBER": 75,
    "DK_POSTAL_CODE": 65,
    "DATE_TIME": 55,
}

ENTITY_OUTPUT_LABEL = {
    "DK_CPR": "CPR",
    "DK_PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "IBAN_CODE": "IBAN",
    "URL": "URL",
    "IP_ADDRESS": "IP",
    "DK_POSTAL_CODE": "POSTCODE",
    "DATE_TIME": "DATE",
}


# ===========================================================================
# Danish Presidio recognizers
# ===========================================================================

class DanishCprRecognizer(PatternRecognizer):
    """Recognize Danish CPR numbers using Presidio + Danish validation.

    Structure: DDMMYY-SSSS, where the hyphen/space may be omitted in ASR text.

    Validation uses:
      1. exactly ten digits after removing separator;
      2. the official relationship between YY and the first serial digit to
         infer the birth century;
      3. an actual calendar birth date which is not in the future.

    Modulus-11 is intentionally NOT required: Danish authorities have assigned
    fully valid CPR numbers without a modulus-11 control digit since 2007.
    """

    COUNTRY_CODE = "dk"

    CPR_PATTERN = Pattern(
        name="Danish CPR DDMMYY-SSSS",
        regex=(
            r"(?<!\d)"
            r"(?:0[1-9]|[12]\d|3[01])"
            r"(?:0[1-9]|1[0-2])"
            r"\d{2}"
            r"[- ]?"
            r"\d{4}"
            r"(?!\d)"
        ),
        score=0.75,
    )

    def __init__(self) -> None:
        super().__init__(
            supported_entity="DK_CPR",
            name="DanishCprRecognizer",
            supported_language="da",
            patterns=[self.CPR_PATTERN],
            context=[
                "cpr",
                "cpr-nummer",
                "cpr nummer",
                "personnummer",
                "personnr",
                "person nr",
            ],
        )

    @staticmethod
    def _birth_year(yy: int, serial_first_digit: int) -> int | None:
        """Infer four-digit birth year from CPR YY + position 7."""
        if 0 <= serial_first_digit <= 3:
            return 1900 + yy

        if serial_first_digit == 4:
            return 2000 + yy if 0 <= yy <= 36 else 1900 + yy

        if 5 <= serial_first_digit <= 8:
            return 2000 + yy if 0 <= yy <= 57 else 1800 + yy

        if serial_first_digit == 9:
            return 2000 + yy if 0 <= yy <= 36 else 1900 + yy

        return None

    def validate_result(self, pattern_text: str):
        digits = re.sub(r"\D", "", pattern_text)
        if len(digits) != 10:
            return False

        day = int(digits[0:2])
        month = int(digits[2:4])
        yy = int(digits[4:6])
        serial_first = int(digits[6])

        year = self._birth_year(yy, serial_first)
        if year is None:
            return False

        try:
            birth_date = date(year, month, day)
        except ValueError:
            return False

        return birth_date <= date.today()


class DanishPostalCodeRecognizer(PatternRecognizer):
    """Recognize Danish four-digit postcodes with context filtering.

    A bare four-digit token is too ambiguous in transcripts (year, quantity,
    room number, etc.). The candidate is therefore kept only when either:
      - nearby Danish address/postcode context exists;
      - a DK- prefix occurs immediately before it; or
      - it is followed by a capitalized city-like token, e.g. 2100 København.

    Use --postal-without-context to disable this conservative filter.
    """

    COUNTRY_CODE = "dk"

    POSTAL_PATTERN = Pattern(
        name="Danish four-digit postcode",
        regex=r"(?<!\d)\d{4}(?!\d)",
        score=0.35,
    )

    CONTEXT_RE = re.compile(
        r"\b(?:postnummer|postnr\.?|post\s*nr\.?|adresse|bopæl|"
        r"bor\s+i|bor\s+på|bosat\s+i|postdistrikt|kommune)\b",
        flags=re.IGNORECASE,
    )

    CITY_AFTER_RE = re.compile(
        r"^\s+(?:[A-ZÆØÅ][A-Za-zÆØÅæøå.'’-]{1,})"
    )

    DK_PREFIX_RE = re.compile(r"DK-\s*$", flags=re.IGNORECASE)

    def __init__(self, require_context: bool = True, context_window: int = 45) -> None:
        self.require_context = require_context
        self.context_window = context_window
        super().__init__(
            supported_entity="DK_POSTAL_CODE",
            name="DanishPostalCodeRecognizer",
            supported_language="da",
            patterns=[self.POSTAL_PATTERN],
            context=["postnummer", "postnr", "adresse", "bopæl", "bor"],
        )

    def analyze(self, text, entities, nlp_artifacts=None, regex_flags=None):
        results = super().analyze(
            text=text,
            entities=entities,
            nlp_artifacts=nlp_artifacts,
            regex_flags=regex_flags,
        )

        filtered = []
        for result in results:
            value = int(text[result.start:result.end])
            if not 1 <= value <= 9999:
                continue

            left = max(0, result.start - self.context_window)
            right = min(len(text), result.end + self.context_window)
            window = text[left:right]
            before = text[left:result.start]
            after = text[result.end:right]

            has_context = bool(self.CONTEXT_RE.search(window))
            has_dk_prefix = bool(self.DK_PREFIX_RE.search(before))
            has_city_after = bool(self.CITY_AFTER_RE.search(after))

            if self.require_context and not (
                has_context or has_dk_prefix or has_city_after
            ):
                continue

            if has_context or has_dk_prefix or has_city_after:
                result.score = max(float(result.score), 0.85)

            filtered.append(result)

        return filtered


class DanishTextDateRecognizer(PatternRecognizer):
    """Supplement Presidio DateRecognizer with Danish month names."""

    MONTHS = (
        r"januar|februar|marts|april|maj|juni|juli|august|"
        r"september|oktober|november|december|"
        r"jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|"
        r"sep\.?|sept\.?|okt\.?|nov\.?|dec\."
    )

    PATTERNS = [
        Pattern(
            name="Danish written date: day month year",
            regex=(
                rf"\b(?:0?[1-9]|[12]\d|3[01])\.?\s+"
                rf"(?:{MONTHS})\s+(?:18|19|20|21)\d{{2}}\b"
            ),
            score=0.80,
        ),
        Pattern(
            name="Danish written date: den day month year",
            regex=(
                rf"\bden\s+(?:0?[1-9]|[12]\d|3[01])\.?\s+"
                rf"(?:{MONTHS})\s+(?:18|19|20|21)\d{{2}}\b"
            ),
            score=0.85,
        ),
        Pattern(
            name="Danish month and year",
            regex=rf"\b(?:{MONTHS})\s+(?:18|19|20|21)\d{{2}}\b",
            score=0.55,
        ),
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="DATE_TIME",
            name="DanishTextDateRecognizer",
            supported_language="da",
            patterns=self.PATTERNS,
            context=[
                "dato",
                "født",
                "fødselsdato",
                "den",
                "d.",
                "fra",
                "til",
                "siden",
            ],
        )


# ===========================================================================
# Replacement policy
# ===========================================================================

class ReplacementPolicy:
    """Generate per-occurrence replacements from the project proposal."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def _digits(self, n: int) -> str:
        return "".join(str(self.rng.randint(0, 9)) for _ in range(n))

    def replacement_for(self, entity_type: str, original_text: str) -> str:
        del original_text  # currently not needed; kept for future policy extensions

        if entity_type == "DK_CPR":
            # PDF: random 10 digit number
            return self._digits(10)

        if entity_type == "DK_PHONE_NUMBER":
            # PDF: +45 + random 8 digit number
            return "+45 " + self._digits(8)

        if entity_type == "EMAIL_ADDRESS":
            # PDF: common Danish name + @gmail.com
            return f"{self.rng.choice(PDF_EMAIL_NAMES)}@gmail.com"

        if entity_type == "DK_POSTAL_CODE":
            # PDF: random 2 digit number + 00
            return f"{self.rng.randint(10, 99)}00"

        if entity_type == "DATE_TIME":
            # PDF literal ranges: day 1-30, month 1-12, year 2000-2026.
            # This deliberately follows the proposal rather than silently
            # changing the policy to calendar-valid random dates.
            day = self.rng.randint(1, 30)
            month = self.rng.randint(1, 12)
            year = self.rng.randint(2000, 2026)
            return f"{day}/{month}/{year}"

        # No replacement rule was supplied in the PDF for these entities.
        if entity_type == "IBAN_CODE":
            return "[IBAN]"
        if entity_type == "URL":
            return "[URL]"
        if entity_type == "IP_ADDRESS":
            return "[IP]"

        return f"[{ENTITY_OUTPUT_LABEL.get(entity_type, entity_type)}]"


# ===========================================================================
# CLI and pipeline helpers
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify and de-identify Danish PII in .txt/.alfrttm transcripts "
            "using Microsoft Presidio."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing transcript files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annonydata") / "presidio_deidentified",
        help="Output directory (default: annonydata/presidio_deidentified).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Extensions scanned recursively (default: .txt .alfrttm).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.20,
        help="Minimum Presidio confidence retained (default: 0.20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional seed for reproducible random replacements. If omitted, "
            "Python seeds from system entropy."
        ),
    )
    parser.add_argument(
        "--audit-raw-pii",
        action="store_true",
        help=(
            "Store original PII strings in audit CSV/JSONL. Default is OFF; "
            "only SHA-256 digests are retained."
        ),
    )
    parser.add_argument(
        "--postal-without-context",
        action="store_true",
        help=(
            "Treat every four-digit candidate as a possible Danish postcode. "
            "Not recommended because years create many false positives."
        ),
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "presidio_deid.log"

    try:
        from loger import setup_logging as project_setup_logging  # type: ignore

        return project_setup_logging(str(log_path))
    except Exception:
        logger = logging.getLogger("presidio_deid")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(console_formatter)
        logger.addHandler(sh)

        return logger


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def normalize_extensions(values: Iterable[str]) -> set[str]:
    return {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in values
    }


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

        # Avoid recursively processing this pipeline's output if output_dir is
        # placed beneath input_dir.
        try:
            path.resolve().relative_to(output_dir)
            continue
        except ValueError:
            pass

        files.append(path)

    return sorted(files)


def extract_segment(line: str, suffix: str) -> tuple[str, int, int]:
    """Return (text_to_analyze, start_in_line, end_in_line)."""
    raw = line.rstrip("\r\n")

    if suffix.lower() == ".alfrttm":
        match = QUOTE_RE.search(raw)
        if not match:
            return "", 0, 0
        return (
            match.group("utterance"),
            match.start("utterance"),
            match.end("utterance"),
        )

    return raw, 0, len(raw)


def create_recognizers(postal_require_context: bool) -> list:
    """Instantiate the Presidio recognizers used by this pipeline."""
    recognizers = [
        DanishCprRecognizer(),
        PhoneRecognizer(
            supported_language="da",
            supported_entity="DK_PHONE_NUMBER",
            supported_regions=["DK"],
            context=[
                "telefon",
                "telefonnummer",
                "tlf",
                "mobil",
                "mobilnummer",
                "nummer",
                "ring",
                "kontakt",
            ],
        ),
        EmailRecognizer(
            supported_language="da",
            context=["email", "e-mail", "mail", "mailadresse", "epost"],
        ),
        IbanRecognizer(
            supported_language="da",
            context=["iban", "bank", "konto", "kontonummer", "bankkonto"],
        ),
        UrlRecognizer(
            supported_language="da",
            context=["url", "website", "webside", "hjemmeside", "link"],
        ),
        IpRecognizer(
            supported_language="da",
            context=["ip", "ip-adresse", "server", "netværk"],
        ),
        DanishPostalCodeRecognizer(require_context=postal_require_context),
        DateRecognizer(
            supported_language="da",
            context=[
                "dato",
                "født",
                "fødselsdato",
                "den",
                "d.",
                "fra",
                "til",
                "siden",
            ],
        ),
        DanishTextDateRecognizer(),
    ]

    # Presidio recognizers support lazy loading. Explicit loading here makes
    # failures occur during initialization rather than half-way through files.
    for recognizer in recognizers:
        recognizer.load()

    return recognizers


def spans_overlap(a: RecognizerResult, b: RecognizerResult) -> bool:
    return a.start < b.end and b.start < a.end


def resolve_overlaps(results: Sequence[RecognizerResult]) -> list[RecognizerResult]:
    """Resolve conflicting cross-entity spans with privacy-first priorities."""
    ordered = sorted(
        results,
        key=lambda result: (
            -ENTITY_PRIORITY.get(result.entity_type, 0),
            -float(result.score),
            -(result.end - result.start),
            result.start,
            result.end,
        ),
    )

    accepted: list[RecognizerResult] = []
    for result in ordered:
        if any(spans_overlap(result, existing) for existing in accepted):
            continue
        accepted.append(result)

    return sorted(accepted, key=lambda result: (result.start, result.end))


def analyze_text(
    text: str,
    recognizers: Sequence,
    min_score: float,
) -> list[RecognizerResult]:
    """Run the configured Presidio recognizers over raw transcript text."""
    if not text.strip():
        return []

    candidates: list[RecognizerResult] = []

    for recognizer in recognizers:
        results = recognizer.analyze(
            text=text,
            entities=list(recognizer.supported_entities),
            nlp_artifacts=None,
        )
        candidates.extend(
            result for result in results if float(result.score) >= min_score
        )

    return resolve_overlaps(candidates)


def recognizer_name(result: RecognizerResult) -> str:
    metadata = getattr(result, "recognition_metadata", None) or {}
    return str(
        metadata.get("recognizer_name")
        or metadata.get(RecognizerResult.RECOGNIZER_NAME_KEY, "")
    )


def deidentify_segment(
    text: str,
    recognizers: Sequence,
    anonymizer: AnonymizerEngine,
    policy: ReplacementPolicy,
    min_score: float,
) -> tuple[str, list[tuple[RecognizerResult, str]]]:
    """Detect PII, generate replacements, and apply them with Presidio.

    Each result is anonymized separately from right to left. This lets each PII
    occurrence receive its own random replacement while keeping original span
    offsets valid for all yet-to-be-processed spans.
    """
    results = analyze_text(text, recognizers, min_score)
    if not results:
        return text, []

    current_text = text
    replacement_by_key: dict[tuple[int, int, str], str] = {}

    for result in sorted(results, key=lambda r: (r.start, r.end), reverse=True):
        original = text[result.start:result.end]
        replacement = policy.replacement_for(result.entity_type, original)

        operator = OperatorConfig("replace", {"new_value": replacement})
        anonymized = anonymizer.anonymize(
            text=current_text,
            analyzer_results=[result],
            operators={result.entity_type: operator},
        )
        current_text = anonymized.text
        replacement_by_key[(result.start, result.end, result.entity_type)] = replacement

    paired = [
        (
            result,
            replacement_by_key[(result.start, result.end, result.entity_type)],
        )
        for result in results
    ]

    return current_text, paired


def split_newline(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1], line[-1]
    return line, ""


def process_file(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    recognizers: Sequence,
    anonymizer: AnonymizerEngine,
    policy: ReplacementPolicy,
    min_score: float,
    audit_records: list[dict],
    audit_raw_pii: bool,
) -> int:
    relative_path = input_path.resolve().relative_to(input_root.resolve())
    output_path = output_root / "deidentified" / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    output_lines: list[str] = []
    entity_count = 0

    for line_number, line in enumerate(lines, start=1):
        segment, seg_start, seg_end = extract_segment(line, input_path.suffix)

        if seg_end <= seg_start or not segment:
            output_lines.append(line)
            continue

        deidentified_segment, detections = deidentify_segment(
            text=segment,
            recognizers=recognizers,
            anonymizer=anonymizer,
            policy=policy,
            min_score=min_score,
        )

        raw_no_newline, newline = split_newline(line)
        reconstructed = (
            raw_no_newline[:seg_start]
            + deidentified_segment
            + raw_no_newline[seg_end:]
            + newline
        )
        output_lines.append(reconstructed)

        for result, replacement in detections:
            original = segment[result.start:result.end]
            audit_records.append(
                {
                    "file": str(relative_path),
                    "line_number": line_number,
                    "entity_type": result.entity_type,
                    "pii_label": ENTITY_OUTPUT_LABEL.get(
                        result.entity_type, result.entity_type
                    ),
                    "original_text": original if audit_raw_pii else "",
                    "original_text_sha256": hashlib.sha256(
                        original.encode("utf-8")
                    ).hexdigest(),
                    "replacement": replacement,
                    "score": round(float(result.score), 6),
                    "recognizer": recognizer_name(result),
                    "start_in_segment": result.start,
                    "end_in_segment": result.end,
                    "start_in_line": seg_start + result.start,
                    "end_in_line": seg_start + result.end,
                }
            )
            entity_count += 1

    output_path.write_text("".join(output_lines), encoding="utf-8")
    return entity_count


def write_audit_files(output_dir: Path, records: list[dict]) -> None:
    jsonl_path = output_dir / "presidio_entities.jsonl"
    csv_path = output_dir / "presidio_entities.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    fieldnames = [
        "file",
        "line_number",
        "entity_type",
        "pii_label",
        "original_text",
        "original_text_sha256",
        "replacement",
        "score",
        "recognizer",
        "start_in_segment",
        "end_in_segment",
        "start_in_line",
        "end_in_line",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(output_dir: Path, records: list[dict]) -> None:
    counts = Counter(record["pii_label"] for record in records)
    summary_path = output_dir / "presidio_summary.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pii_label", "count", "replacement_policy"])
        writer.writerow(["CPR", counts.get("CPR", 0), "random 10 digits"])
        writer.writerow([
            "PHONE",
            counts.get("PHONE", 0),
            "+45 + random 8 digits",
        ])
        writer.writerow([
            "EMAIL",
            counts.get("EMAIL", 0),
            "proposal name + @gmail.com",
        ])
        writer.writerow([
            "IBAN",
            counts.get("IBAN", 0),
            "[IBAN] (not specified in PDF)",
        ])
        writer.writerow([
            "URL",
            counts.get("URL", 0),
            "[URL] (not specified in PDF)",
        ])
        writer.writerow([
            "IP",
            counts.get("IP", 0),
            "[IP] (not specified in PDF)",
        ])
        writer.writerow([
            "POSTCODE",
            counts.get("POSTCODE", 0),
            "random 2 digits + 00",
        ])
        writer.writerow([
            "DATE",
            counts.get("DATE", 0),
            "random day 1-30 / month 1-12 / year 2000-2026",
        ])


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    logger.info("Presidio analyzer version: %s", package_version("presidio-analyzer"))
    logger.info(
        "Presidio anonymizer version: %s",
        package_version("presidio-anonymizer"),
    )
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Minimum score: %.3f", args.min_score)
    logger.info("Replacement seed: %s", args.seed)
    logger.info("Audit stores raw PII: %s", args.audit_raw_pii)
    logger.info(
        "Postal-code context required: %s",
        not args.postal_without_context,
    )

    extensions = normalize_extensions(args.extensions)
    files = iter_input_files(input_dir, extensions, output_dir)
    logger.info("Transcript files found: %d", len(files))

    if not files:
        logger.warning("No matching transcript files found. Nothing to do.")
        return

    recognizers = create_recognizers(
        postal_require_context=not args.postal_without_context
    )
    anonymizer = AnonymizerEngine()
    policy = ReplacementPolicy(seed=args.seed)

    logger.info(
        "Presidio recognizers: %s",
        ", ".join(type(recognizer).__name__ for recognizer in recognizers),
    )

    audit_records: list[dict] = []
    total_entities = 0

    for index, input_path in enumerate(files, start=1):
        count = process_file(
            input_path=input_path,
            input_root=input_dir,
            output_root=output_dir,
            recognizers=recognizers,
            anonymizer=anonymizer,
            policy=policy,
            min_score=args.min_score,
            audit_records=audit_records,
            audit_raw_pii=args.audit_raw_pii,
        )
        total_entities += count
        logger.info(
            "[%d/%d] %s -> %d PII span(s)",
            index,
            len(files),
            input_path.relative_to(input_dir),
            count,
        )

    write_audit_files(output_dir, audit_records)
    write_summary(output_dir, audit_records)

    logger.info("De-identification complete.")
    logger.info("Files processed: %d", len(files))
    logger.info("PII spans replaced: %d", total_entities)
    logger.info("De-identified transcripts: %s", output_dir / "deidentified")
    logger.info("Audit CSV: %s", output_dir / "presidio_entities.csv")
    logger.info("Audit JSONL: %s", output_dir / "presidio_entities.jsonl")
    logger.info("Summary CSV: %s", output_dir / "presidio_summary.csv")


if __name__ == "__main__":
    main()
