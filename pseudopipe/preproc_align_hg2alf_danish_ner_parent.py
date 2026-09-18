from pathlib import Path
import json
import re
import shutil

import requests
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


# ============================================================
# Configuration
# ============================================================

OVERWRITE = True

# Prefer the directory that already contains danish_ner_datasets.
candidate_roots = [
    Path("annonydata"),
    Path("anonydata"),
]

DATA_ROOT = None
for root in candidate_roots:
    candidate = root / "danish_ner_datasets"
    if candidate.exists():
        DATA_ROOT = candidate
        break

if DATA_ROOT is None:
    # Your current project uses "annonydata", so use that as default.
    DATA_ROOT = Path("annonydata") / "danish_ner_datasets"

CHCAA_INPUT = DATA_ROOT / "chcaa" / "dataset"
BPLANK_INPUT = DATA_ROOT / "bplank" / "dataset"

OUTPUT_ROOT = DATA_ROOT / "parent_aligned"
CHCAA_OUTPUT = OUTPUT_ROOT / "chcaa" / "dataset"
BPLANK_OUTPUT = OUTPUT_ROOT / "bplank" / "dataset"


# ============================================================
# Target parent schema
# ============================================================

PARENT_LABELS = [
    "DATE",
    "DEM",
    "HEALTH",
    "LOC",
    "MISC",
    "NUM",
    "ORG",
    "PER",
    "PERCENT",
    "RELATION",
    "ID",
]

BIO_LABELS = ["O"]
for label in PARENT_LABELS:
    BIO_LABELS.extend([f"B-{label}", f"I-{label}"])

LABEL2ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


# ============================================================
# Label normalization
# ============================================================

def normalize_entity_name(label):
    """
    Normalize source label spelling without changing its meaning.

    Examples:
        WORK OF ART  -> WORK OF ART
        WORK_OF_ART  -> WORK OF ART
        WORK-OF-ART  -> WORK OF ART
        organization -> ORGANIZATION
    """
    label = str(label).strip().upper()
    label = label.replace("_", " ")
    label = label.replace("-", " ")
    label = re.sub(r"\s+", " ", label)
    return label


# DANSK / OntoNotes-style entity type -> user's parent type.
# Keys are written in normalized form.
CHCAA_TO_PARENT = {
    # Person
    "PERSON": "PER",
    "PER": "PER",

    # Location / geopolitical / facility
    "GPE": "LOC",
    "LOCATION": "LOC",
    "LOC": "LOC",
    "FACILITY": "LOC",
    "FAC": "LOC",

    # Organization
    "ORGANIZATION": "ORG",
    "ORG": "ORG",

    # Temporal
    "DATE": "DATE",
    "TIME": "DATE",

    # Numeric
    "CARDINAL": "NUM",
    "ORDINAL": "NUM",
    "MONEY": "NUM",
    "QUANTITY": "NUM",
    "NUM": "NUM",

    # Percentage
    "PERCENT": "PERCENT",
    "PERCENTAGE": "PERCENT",

    # Demographic-like OntoNotes classes
    "NORP": "DEM",
    "LANGUAGE": "DEM",
    "DEM": "DEM",

    # Residual named-entity categories
    "EVENT": "MISC",
    "LAW": "MISC",
    "PRODUCT": "MISC",
    "WORK OF ART": "MISC",
    "MISC": "MISC",

    # Exact passthroughs if encountered in a future version
    "HEALTH": "HEALTH",
    "RELATION": "RELATION",
    "ID": "ID",
}

# Plank coarse NER -> user's parent type.
BPLANK_TO_PARENT = {
    "PER": "PER",
    "PERSON": "PER",

    "LOC": "LOC",
    "LOCATION": "LOC",
    "GPE": "LOC",

    "ORG": "ORG",
    "ORGANIZATION": "ORG",

    "MISC": "MISC",

    # Defensive passthroughs if a locally converted version contains them.
    "DATE": "DATE",
    "DEM": "DEM",
    "HEALTH": "HEALTH",
    "NUM": "NUM",
    "PERCENT": "PERCENT",
    "RELATION": "RELATION",
    "ID": "ID",
}


# ============================================================
# Generic utilities
# ============================================================

def save_dataset(dataset, path):
    path = Path(path)

    if path.exists():
        if OVERWRITE:
            print(f"Removing previous output: {path}")
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"{path} already exists. Set OVERWRITE=True to replace it."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))


def make_output_row(text, tokens, ner_tags, source):
    return {
        "text": text,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "ner_tag_ids": [LABEL2ID[tag] for tag in ner_tags],
        "source": source,
    }


# ============================================================
# CHCAA / DANSK
# ============================================================

def load_chcaa():
    if CHCAA_INPUT.exists():
        print("Loading local CHCAA dataset:")
        print(f"  {CHCAA_INPUT.resolve()}")
        return load_from_disk(str(CHCAA_INPUT))

    print("\nLocal CHCAA dataset not found.")
    print("Downloading chcaa/dansk-ner from Hugging Face...")

    cache_dir = DATA_ROOT / "chcaa" / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "chcaa/dansk-ner",
        cache_dir=str(cache_dir),
    )

    CHCAA_INPUT.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(CHCAA_INPUT))
    return ds


def get_chcaa_tokens(row):
    """
    Convert the DANSK token-offset structure into token strings + spans.
    """
    text = row["text"]
    raw_tokens = row["tokens"]

    token_strings = []
    token_spans = []

    for tok in raw_tokens:
        if isinstance(tok, dict) and "start" in tok and "end" in tok:
            start = int(tok["start"])
            end = int(tok["end"])
            token_strings.append(text[start:end])
            token_spans.append((start, end))
        elif isinstance(tok, str):
            # Fallback for an alternate dataset representation.
            raise TypeError(
                "CHCAA tokens are plain strings rather than character-offset "
                "dictionaries. This script expects the locally downloaded "
                "chcaa/dansk-ner representation with token start/end offsets."
            )
        else:
            raise TypeError(
                "Unexpected CHCAA token representation:\n"
                f"  type={type(tok)}\n"
                f"  value={tok}"
            )

    return token_strings, token_spans


def map_chcaa_label(raw_label):
    normalized = normalize_entity_name(raw_label)

    if normalized not in CHCAA_TO_PARENT:
        raise ValueError(
            "\nUnknown CHCAA label encountered:\n"
            f"  raw       = {raw_label!r}\n"
            f"  normalized= {normalized!r}\n\n"
            "Add this normalized label explicitly to CHCAA_TO_PARENT "
            "rather than silently inventing a mapping."
        )

    return CHCAA_TO_PARENT[normalized]


def align_chcaa_row(row):
    text = row["text"]
    tokens, token_spans = get_chcaa_tokens(row)

    entities = []
    for entity_index, ent in enumerate(row["ents"]):
        parent = map_chcaa_label(ent["label"])

        entities.append({
            "entity_index": entity_index,
            "start": int(ent["start"]),
            "end": int(ent["end"]),
            "parent": parent,
        })

    # Assign each token to the entity with the greatest positive overlap.
    token_entity = []

    for token_start, token_end in token_spans:
        best = None
        best_overlap = 0

        for ent in entities:
            overlap = max(
                0,
                min(token_end, ent["end"]) - max(token_start, ent["start"]),
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best = ent

        token_entity.append(best if best_overlap > 0 else None)

    # Convert span membership to standard BIO.
    ner_tags = []
    previous_entity_index = None

    for ent in token_entity:
        if ent is None:
            ner_tags.append("O")
            previous_entity_index = None
            continue

        entity_index = ent["entity_index"]
        parent = ent["parent"]

        prefix = "B" if entity_index != previous_entity_index else "I"
        ner_tags.append(f"{prefix}-{parent}")
        previous_entity_index = entity_index

    if len(tokens) != len(ner_tags):
        raise RuntimeError("CHCAA token/tag length mismatch.")

    return make_output_row(
        text=text,
        tokens=tokens,
        ner_tags=ner_tags,
        source="chcaa/dansk-ner",
    )


def align_chcaa_dataset(ds):
    aligned = {}

    for split, split_ds in ds.items():
        print(f"\nAligning CHCAA split: {split}")

        rows = []
        total = len(split_ds)

        for i, row in enumerate(split_ds):
            rows.append(align_chcaa_row(row))

            if (i + 1) % 5000 == 0 or (i + 1) == total:
                print(f"  {i + 1:,}/{total:,}")

        aligned[split] = Dataset.from_list(rows)

    return DatasetDict(aligned)


# ============================================================
# BPLANK
# ============================================================

BPLANK_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "bplank/danish_ner_transfer/master/data"
)

BPLANK_FILES = {
    "train_5k": "da_ddt-ud-ner-train-5k.conll",
    "train_10k": "da_ddt-ud-ner-train-10k.conll",
    "dev": "da_ddt-ud-ner-dev.conll",
    "test": "da_ddt-ud-ner-test.conll",
}


def conll_to_dataset(filepath):
    rows = []
    tokens = []
    tags = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")

            if not line.strip():
                if tokens:
                    rows.append({
                        "text": " ".join(tokens),
                        "tokens": tokens.copy(),
                        "ner_tags": tags.copy(),
                    })
                    tokens = []
                    tags = []
                continue

            # Original files are tab-separated. rsplit is intentionally used
            # in case the token itself contains unexpected whitespace.
            token, tag = line.rsplit("\t", 1)
            tokens.append(token)
            tags.append(tag)

    if tokens:
        rows.append({
            "text": " ".join(tokens),
            "tokens": tokens.copy(),
            "ner_tags": tags.copy(),
        })

    return Dataset.from_list(rows)


def download_bplank():
    raw_dir = DATA_ROOT / "bplank" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    splits = {}

    for split, filename in BPLANK_FILES.items():
        filepath = raw_dir / filename

        if not filepath.exists() or filepath.stat().st_size == 0:
            url = f"{BPLANK_BASE_URL}/{filename}"
            print(f"Downloading: {url}")

            response = requests.get(url, timeout=120)
            response.raise_for_status()
            filepath.write_bytes(response.content)

        splits[split] = conll_to_dataset(filepath)

    ds = DatasetDict(splits)

    BPLANK_INPUT.parent.mkdir(parents=True, exist_ok=True)

    # Avoid save_to_disk collision if a partially created folder exists.
    if BPLANK_INPUT.exists():
        shutil.rmtree(BPLANK_INPUT)

    ds.save_to_disk(str(BPLANK_INPUT))
    return ds


def load_bplank():
    if BPLANK_INPUT.exists():
        print("\nLoading local Plank dataset:")
        print(f"  {BPLANK_INPUT.resolve()}")
        return load_from_disk(str(BPLANK_INPUT))

    print("\nLocal Plank dataset not found; downloading it.")
    return download_bplank()


def parse_source_tag(tag):
    """
    Parse BIO/BILOU-like source tags while preserving entity names that
    themselves may contain spaces, underscores, or hyphens.
    """
    tag = str(tag).strip()

    if tag.upper() == "O":
        return "O", None

    # Only treat the first prefix separator as BIO syntax.
    match = re.match(r"^([BIESUL])[-_](.+)$", tag, flags=re.IGNORECASE)

    if match:
        prefix = match.group(1).upper()
        entity = normalize_entity_name(match.group(2))
    else:
        prefix = "B"
        entity = normalize_entity_name(tag)

    return prefix, entity


def align_bplank_tags(source_tags):
    aligned = []
    previous_parent = None

    for source_tag in source_tags:
        prefix, source_label = parse_source_tag(source_tag)

        if prefix == "O":
            aligned.append("O")
            previous_parent = None
            continue

        if source_label not in BPLANK_TO_PARENT:
            raise ValueError(
                "\nUnknown Plank label encountered:\n"
                f"  raw       = {source_tag!r}\n"
                f"  normalized= {source_label!r}\n\n"
                "Add this normalized label explicitly to BPLANK_TO_PARENT."
            )

        parent = BPLANK_TO_PARENT[source_label]

        if prefix in {"B", "S", "U"}:
            new_prefix = "B"
        elif prefix in {"I", "E", "L"}:
            # Repair an illegal continuation after O or another entity class.
            new_prefix = "I" if previous_parent == parent else "B"
        else:
            raise ValueError(f"Unknown BIO/BILOU prefix: {prefix!r}")

        aligned.append(f"{new_prefix}-{parent}")
        previous_parent = parent

    return aligned


def align_bplank_row(row):
    tokens = list(row["tokens"])
    source_tags = list(row["ner_tags"])
    ner_tags = align_bplank_tags(source_tags)

    if len(tokens) != len(ner_tags):
        raise RuntimeError("BPLANK token/tag length mismatch.")

    return make_output_row(
        text=row.get("text", " ".join(tokens)),
        tokens=tokens,
        ner_tags=ner_tags,
        source="bplank/danish_ner_transfer",
    )


def align_bplank_dataset(ds):
    aligned = {}

    for split, split_ds in ds.items():
        print(f"\nAligning Plank split: {split}")

        rows = []
        total = len(split_ds)

        for i, row in enumerate(split_ds):
            rows.append(align_bplank_row(row))

            if (i + 1) % 5000 == 0 or (i + 1) == total:
                print(f"  {i + 1:,}/{total:,}")

        aligned[split] = Dataset.from_list(rows)

    return DatasetDict(aligned)


# ============================================================
# Diagnostics
# ============================================================

def count_labels(ds):
    counts = {}

    for split_ds in ds.values():
        for row in split_ds:
            for tag in row["ner_tags"]:
                counts[tag] = counts.get(tag, 0) + 1

    return counts


def print_summary(name, ds):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(ds)

    counts = count_labels(ds)

    print("\nBIO label counts:")
    for label in BIO_LABELS:
        count = counts.get(label, 0)
        if count:
            print(f"{label:15s} {count:>10,d}")


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PARENT NER ALIGNMENT")
    print("=" * 70)

    print("Data root:")
    print(f"  {DATA_ROOT.resolve()}")

    print("\nParent entity labels:")
    print(PARENT_LABELS)

    print("\nBIO label vocabulary:")
    for label, idx in LABEL2ID.items():
        print(f"{idx:2d}  {label}")

    # CHCAA
    chcaa = load_chcaa()
    chcaa_aligned = align_chcaa_dataset(chcaa)
    save_dataset(chcaa_aligned, CHCAA_OUTPUT)
    print_summary("CHCAA -> parent schema", chcaa_aligned)

    # BPLANK
    bplank = load_bplank()
    bplank_aligned = align_bplank_dataset(bplank)
    save_dataset(bplank_aligned, BPLANK_OUTPUT)
    print_summary("BPLANK -> parent schema", bplank_aligned)

    # Save schema/mapping metadata
    schema = {
        "parent_entity_labels": PARENT_LABELS,
        "bio_labels": BIO_LABELS,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "chcaa_to_parent": CHCAA_TO_PARENT,
        "bplank_to_parent": BPLANK_TO_PARENT,
        "normalization": (
            "uppercase; underscores/hyphens converted to spaces; "
            "repeated whitespace collapsed"
        ),
    }

    schema_path = OUTPUT_ROOT / "parent_schema.json"

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("\nCHCAA aligned dataset:")
    print(f"  {CHCAA_OUTPUT.resolve()}")

    print("\nBPLANK aligned dataset:")
    print(f"  {BPLANK_OUTPUT.resolve()}")

    print("\nShared schema:")
    print(f"  {schema_path.resolve()}")


if __name__ == "__main__":
    main()
