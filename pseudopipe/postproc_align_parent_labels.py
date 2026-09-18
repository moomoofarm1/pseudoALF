#!/usr/bin/env python3
"""Stage 4: collapse Stage-2/3 PII labels to GDPR-ALF parent labels.

Input: Stage-3 tagged .alfrttm/.txt files containing
       [PersonData]text[LABEL]
Output: same files/relative paths, but LABEL is replaced by its parent label.

No external packages are required.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LABEL_TO_PARENT = {
    # PER
    "PER": "PER",
    "STAFF": "PER",
    "PATIENT": "PER",
    "PERSON": "PER",

    # LOC
    "LOC": "LOC",
    "ADDRESS": "LOC",
    "POSTCODE": "LOC",
    "GPE": "LOC",
    "LOCATION": "LOC",
    "FACILITY": "LOC",

    # ORG
    "ORG": "ORG",
    "HOSPITAL": "ORG",
    "ORGANIZATION": "ORG",

    # CONTACT
    "CONTACT": "CONTACT",
    "PHONE": "CONTACT",
    "EMAIL": "CONTACT",

    # ID
    "ID": "ID",
    "CPR": "ID",
    "IBAN": "ID",
    "IP": "ID",
    "URL": "ID",

    # DATE
    "DATE": "DATE",
    "TIME": "DATE",
    "DURATION": "DATE",

    # DEM
    "DEM": "DEM",
    "ETHNICITY": "DEM",
    "RELIGION": "DEM",
    "POLITICS": "DEM",
    "SEXUALITY": "DEM",
    "AGE": "DEM",
    "LANGUAGE": "DEM",
    "NORP": "DEM",

    # HEALTH
    "HEALTH": "HEALTH",
    "DIAGNOSIS": "HEALTH",
    "MEDICATION": "HEALTH",
    "CONDITION": "HEALTH",

    # NUM
    "CARDINAL": "NUM",
    "MONEY": "MONEY",
    "ORDINAL": "NUM",
    "PERCENT": "PERCENT",
    "QUANTITY": "NUM",

    # Fine-grained labels without a more specific GDPR-ALF parent
    "MISC": "MISC",
    "EVENT": "MISC",
    "LAW": "MISC",
    "PRODUCT": "MISC",
    "WORK_OF_ART": "MISC",
}

# Explicit child/model-label -> GDPR-ALF parent-label mapping.
# ENG transform
# LABEL_TO_PARENT = {
#     # PER
#     "PER": "PER",
#     "STAFF": "STAFF",
#     "PATIENT": "PATIENT",
#     "PERSON": "PER",

#     # LOC
#     "LOC": "LOC",
#     "ADDRESS": "ADDRESS",
#     "POSTCODE": "POSTCODE",
#     "GPE": "GPE",
#     "LOCATION": "LOC",
#     "FACILITY": "FACILITY",

#     # ORG
#     "ORG": "ORG",
#     "HOSPITAL": "HOSPITAL",
#     "ORGANIZATION": "ORGANIZATION",

#     # CONTACT
#     "CONTACT": "CONTACT",
#     "PHONE": "PHONE",
#     "EMAIL": "EMAIL",

#     # ID
#     "ID": "ID",
#     "CPR": "CPR",
#     "IBAN": "IBAN",
#     "IP": "IP",
#     "URL": "URL",

#     # DATE
#     "DATE": "DATE",
#     "TIME": "TIME",
#     "DURATION": "DURATION",

#     # DEM
#     "DEM": "DEM",
#     "ETHNICITY": "ETHNICITY",
#     "RELIGION": "RELIGION",
#     "POLITICS": "POLITICS",
#     "SEXUALITY": "SEXUALITY",
#     "AGE": "AGE",
#     "LANGUAGE": "LANGUAGE",
#     "NORP": "NORP",

#     # HEALTH
#     "HEALTH": "HEALTH",
#     "DIAGNOSIS": "DIAGNOSIS",
#     "MEDICATION": "MEDICATION",
#     "CONDITION": "CONDITION",

#     # NUM
#     "CARDINAL": "NUM",
#     "MONEY": "MONEY",
#     "ORDINAL": "NUM",
#     "PERCENT": "PERCENT",
#     "QUANTITY": "NUM",

#     # Fine-grained labels without a more specific GDPR-ALF parent
#     "MISC": "PERSONOPLYSNING",
#     "EVENT": "EVENT",
#     "LAW": "LAW",
#     "PRODUCT": "PRODUCT",
#     "WORK_OF_ART": "WORK_OF_ART",
# }

# danish mapping, the parent is due to the initial idea. Keep the fine-grained labels help the LLM training.
# LABEL_TO_PARENT = {
#     # PER — PERSON
#     "PER": "PERSON",
#     "STAFF": "PERSONALE",
#     "PATIENT": "PATIENT",
#     "PERSON": "PERSON",

#     # LOC — LOCATION
#     "LOC": "STED",
#     "ADDRESS": "ADRESSE",
#     "POSTCODE": "POSTNUMMER",
#     "GPE": "GEOPOLITISK_ENHED",
#     "LOCATION": "STED",
#     "FACILITY": "FACILITET",

#     # ORG — ORGANIZATION
#     "ORG": "ORGANISATION",
#     "HOSPITAL": "HOSPITAL",
#     "ORGANIZATION": "ORGANISATION",

#     # CONTACT — CONTACT INFORMATION
#     "CONTACT": "KONTAKT",
#     "PHONE": "TELEFON",
#     "EMAIL": "EMAIL",

#     # ID — IDENTIFIER
#     "ID": "IDENTIFIKATOR",
#     "CPR": "CPR",
#     "IBAN": "IBAN",
#     "IP": "IP",
#     "URL": "URL",

#     # DATE — DATE / TIME
#     "DATE": "DATO",
#     "TIME": "TID",
#     "DURATION": "VARIGHED",

#     # DEM — DEMOGRAPHIC INFORMATION
#     "DEM": "DEMOGRAFI",
#     "ETHNICITY": "ETNICITET",
#     "RELIGION": "RELIGION",
#     "POLITICS": "POLITIK",
#     "SEXUALITY": "SEKSUALITET",
#     "AGE": "ALDER",
#     "LANGUAGE": "SPROG",
#     "NORP": "GRUPPETILHØRSFORHOLD",

#     # HEALTH — HEALTH INFORMATION
#     "HEALTH": "HELBRED",
#     "DIAGNOSIS": "DIAGNOSE",
#     "MEDICATION": "MEDICIN",
#     "CONDITION": "TILSTAND",

#     # NUM — NUMERIC INFORMATION
#     "CARDINAL": "TAL",
#     "MONEY": "BELØB",
#     "ORDINAL": "TAL",
#     "PERCENT": "PROCENT",
#     "QUANTITY": "TAL",

#     # MISC — MISCELLANEOUS / OTHER PERSONAL INFORMATION
#     "MISC": "PERSONOPLYSNING",
#     "EVENT": "BEGIVENHED",
#     "LAW": "LOV",
#     "PRODUCT": "PRODUKT",
#     "WORK_OF_ART": "VÆRK",
# }

ANNOTATION_RE = re.compile(
    r"\[PersonData\](?P<text>.*?)\[(?P<label>[A-Za-z][A-Za-z0-9_ ]*)\]"
)
STANDALONE_RE = re.compile(r"\[(?P<label>IBAN|URL|IP)\]")


def normalize_label(label: str) -> str:
    return label.strip().upper().replace(" ", "_")


def align_text(text: str) -> str:
    def replace_annotation(match: re.Match[str]) -> str:
        label = normalize_label(match.group("label"))
        parent = LABEL_TO_PARENT.get(label, "MISC")
        return f"[PersonData]{match.group('text')}[{parent}]"

    def replace_standalone(match: re.Match[str]) -> str:
        label = normalize_label(match.group("label"))
        return f"[{LABEL_TO_PARENT.get(label, 'MISC')}]"

    text = ANNOTATION_RE.sub(replace_annotation, text)
    return STANDALONE_RE.sub(replace_standalone, text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map Stage-3 PII labels to GDPR-ALF parent labels."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".alfrttm", ".txt"}
    )

    for src in files:
        rel = src.relative_to(input_dir)

        # Remove a leading "tagged/" folder from the relative path
        if rel.parts and rel.parts[0] == "tagged":
            rel = Path(*rel.parts[1:])

        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(align_text(src.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"{rel} -> {dst}")

    print(f"Done: {len(files)} file(s).")


if __name__ == "__main__":
    main()
