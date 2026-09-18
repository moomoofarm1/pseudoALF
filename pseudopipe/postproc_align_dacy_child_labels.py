#!/usr/bin/env python3
"""Project DaCy fine-grained NER labels to GDPR-ALF child labels.

Expected inline annotation format:
    [PersonData]text[LABEL]

The mapping is deliberately conservative:
- DaCy labels with a defensible GDPR-ALF child-level equivalent are projected.
- DaCy labels without a defensible equivalent are kept as themselves.
- Unknown labels are also preserved rather than collapsed to MISC.
- MISC is preserved only when the input label is already MISC.

No external packages are required.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


# -----------------------------------------------------------------------------
# DaCy -> GDPR-ALF CHILD-LABEL PROJECTION
# -----------------------------------------------------------------------------
# DaCy model:
#   chcaa/da_dacy_large_ner_fine_grained
#
# Important design choice:
#   DO NOT use MISC as a fallback. Labels that cannot be mapped defensibly to a
#   GDPR-ALF child label are retained unchanged. This prevents unrelated entity
#   types (e.g. PRODUCT or EVENT) from being incorrectly turned into MISC.
#
# Input labels are normalized to uppercase with spaces replaced by underscores
# before lookup, so DaCy's "WORK OF ART" becomes "WORK_OF_ART".
DACY_TO_CHILD = {
    # Direct / high-confidence projections
    "PERSON": "PER",
    "ORGANIZATION": "ORG",
    "DATE": "DATE",
    "TIME": "TIME",

    # Location-like DaCy labels -> broad GDPR-ALF location child label
    "GPE": "LOC",
    "LOCATION": "LOC",
    "FACILITY": "LOC",

    # Demographic-like DaCy labels -> broad GDPR-ALF demographic child label
    "NORP": "DEM",
    "LANGUAGE": "DEM",

    # No defensible GDPR-ALF child-level equivalent: keep the DaCy label
    "CARDINAL": "CARDINAL",
    "ORDINAL": "ORDINAL",
    "MONEY": "MISC",
    "PERCENT": "PERCENT",
    "QUANTITY": "MISC",
    "EVENT": "MISC",
    "LAW": "MISC",
    "PRODUCT": "MISC",
    "WORK_OF_ART": "MISC",

    # "CARDINAL": "CARDINAL",
    # "ORDINAL": "ORDINAL",
    # "MONEY": "MONEY",
    # "PERCENT": "PERCENT",
    # "QUANTITY": "QUANTITY",
    # "EVENT": "EVENT",
    # "LAW": "LAW",
    # "PRODUCT": "PRODUCT",
    # "WORK_OF_ART": "WORK_OF_ART",

    # DaCy does not normally emit MISC, but preserve it if it is present.
    # MISC is therefore never created merely because a label is unknown.
    "MISC": "MISC",
}


ANNOTATION_RE = re.compile(
    r"\[PersonData\](?P<text>.*?)\[(?P<label>[A-Za-z][A-Za-z0-9_ ]*)\]"
)


def normalize_label(label: str) -> str:
    """Normalize a label for mapping while keeping the label semantics intact."""
    return label.strip().upper().replace(" ", "_")


def align_text(text: str, counts: Counter[str], unmapped: Counter[str]) -> str:
    """Project inline DaCy labels to GDPR-ALF child labels."""

    def replace_annotation(match: re.Match[str]) -> str:
        source_label = normalize_label(match.group("label"))
        target_label = DACY_TO_CHILD.get(source_label)

        # Preserve unknown labels; never use MISC as a generic fallback.
        if target_label is None:
            target_label = source_label
            unmapped[source_label] += 1

        counts[f"{source_label} -> {target_label}"] += 1
        return f"[PersonData]{match.group('text')}[{target_label}]"

    return ANNOTATION_RE.sub(replace_annotation, text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map DaCy fine-grained NER labels to GDPR-ALF child labels."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    files = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".alfrttm", ".txt"}
    )

    mapping_counts: Counter[str] = Counter()
    unmapped_counts: Counter[str] = Counter()

    for src in files:
        rel = src.relative_to(input_dir)

        # Match the directory behavior of the existing parent-alignment script.
        if rel.parts and rel.parts[0] == "tagged":
            rel = Path(*rel.parts[1:])

        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        aligned = align_text(
            src.read_text(encoding="utf-8"),
            counts=mapping_counts,
            unmapped=unmapped_counts,
        )
        dst.write_text(aligned, encoding="utf-8")
        print(f"{rel} -> {dst}")

    print(f"\nDone: {len(files)} file(s).")

    if mapping_counts:
        print("\nLabel projection counts:")
        for mapping, n in sorted(mapping_counts.items()):
            print(f"  {mapping}: {n}")

    if unmapped_counts:
        print("\nWARNING: labels not listed in DACY_TO_CHILD were preserved unchanged:")
        for label, n in sorted(unmapped_counts.items()):
            print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
