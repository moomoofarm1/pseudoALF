import os
from pathlib import Path

import requests
from datasets import load_dataset, Dataset, DatasetDict


# ============================================================
# Paths
# ============================================================

# Use the directory from which the script is launched.
# This makes it convenient when called from a Jupyter notebook.
PROJECT_DIR = Path.cwd()

BASE_DIR = PROJECT_DIR / "annonydata" / "danish_ner_datasets"

CHCAA_DIR = BASE_DIR / "chcaa"
CHCAA_DATASET_DIR = CHCAA_DIR / "dataset"
CHCAA_PREVIEW_DIR = CHCAA_DIR / "previews"
CHCAA_CACHE_DIR = CHCAA_DIR / ".hf_cache"

BPLANK_DIR = BASE_DIR / "bplank"
BPLANK_RAW_DIR = BPLANK_DIR / "raw"
BPLANK_DATASET_DIR = BPLANK_DIR / "dataset"
BPLANK_PREVIEW_DIR = BPLANK_DIR / "previews"


for directory in [
    CHCAA_DIR,
    CHCAA_PREVIEW_DIR,
    CHCAA_CACHE_DIR,
    BPLANK_DIR,
    BPLANK_RAW_DIR,
    BPLANK_PREVIEW_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("OUTPUT DIRECTORY")
print("=" * 70)
print(BASE_DIR.resolve())


# ============================================================
# Dataset 1: chcaa/dansk-ner
# ============================================================

print("\n" + "=" * 70)
print("Downloading: chcaa/dansk-ner")
print("=" * 70)

ner1 = load_dataset(
    "chcaa/dansk-ner",
    cache_dir=str(CHCAA_CACHE_DIR),
)

print(ner1)

for split in ner1.keys():
    print(f"\n[{split}]")
    print("Rows:", len(ner1[split]))
    print("Features:")
    print(ner1[split].features)

    print(
        ner1[split]
        .select(range(min(3, len(ner1[split]))))
        .to_pandas()
    )


# Save full Hugging Face DatasetDict
if CHCAA_DATASET_DIR.exists():
    print(
        f"\nWARNING: {CHCAA_DATASET_DIR} already exists. "
        "Hugging Face save_to_disk() may refuse to overwrite it."
    )
else:
    ner1.save_to_disk(str(CHCAA_DATASET_DIR))


# Save previews
for split in ner1.keys():
    preview_path = CHCAA_PREVIEW_DIR / f"{split}_preview.csv"

    ner1[split].select(
        range(min(20, len(ner1[split])))
    ).to_pandas().to_csv(
        preview_path,
        index=False,
    )


print("\nchcaa/dansk-ner saved to:")
print(CHCAA_DIR.resolve())


# ============================================================
# Dataset 2: bplank/danish_ner_transfer
# ============================================================

print("\n" + "=" * 70)
print("Downloading: bplank/danish_ner_transfer")
print("=" * 70)

GH_BASE = (
    "https://raw.githubusercontent.com/"
    "bplank/danish_ner_transfer/master/data"
)


BPLANK_FILES = {
    "train_5k": "da_ddt-ud-ner-train-5k.conll",
    "train_10k": "da_ddt-ud-ner-train-10k.conll",
    "dev": "da_ddt-ud-ner-dev.conll",
    "test": "da_ddt-ud-ner-test.conll",
}


def download_file(url, output_path):
    """
    Download file only if it does not already exist.
    """
    output_path = Path(output_path)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already downloaded: {output_path.name}")
        return output_path

    print(f"Downloading: {url}")

    response = requests.get(
        url,
        timeout=120,
    )
    response.raise_for_status()

    output_path.write_bytes(response.content)

    print(f"Saved: {output_path}")

    return output_path


def conll_to_dataset(filepath):
    """
    Convert the downloaded two-column CoNLL file into
    a Hugging Face Dataset.
    """
    filepath = Path(filepath)

    rows = []
    tokens = []
    tags = []

    with filepath.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                if tokens:
                    rows.append(
                        {
                            "text": " ".join(tokens),
                            "tokens": tokens.copy(),
                            "ner_tags": tags.copy(),
                        }
                    )

                    tokens = []
                    tags = []

                continue

            token, tag = line.rsplit("\t", 1)

            tokens.append(token)
            tags.append(tag)

    # Final sentence if file does not end with blank line
    if tokens:
        rows.append(
            {
                "text": " ".join(tokens),
                "tokens": tokens.copy(),
                "ner_tags": tags.copy(),
            }
        )

    return Dataset.from_list(rows)


# ------------------------------------------------------------
# Download original CoNLL files
# ------------------------------------------------------------

local_files = {}

for split, filename in BPLANK_FILES.items():

    url = f"{GH_BASE}/{filename}"
    local_path = BPLANK_RAW_DIR / filename

    local_files[split] = download_file(
        url,
        local_path,
    )


# ------------------------------------------------------------
# Convert to Hugging Face DatasetDict
# ------------------------------------------------------------

ner2 = DatasetDict(
    {
        split: conll_to_dataset(filepath)
        for split, filepath in local_files.items()
    }
)

print("\n" + "=" * 70)
print("bplank/danish_ner_transfer")
print("=" * 70)

print(ner2)

for split in ner2.keys():

    print(f"\n[{split}]")
    print("Rows:", len(ner2[split]))
    print("Features:")
    print(ner2[split].features)

    print(
        ner2[split]
        .select(range(min(3, len(ner2[split]))))
        .to_pandas()
    )


# Save full DatasetDict
if BPLANK_DATASET_DIR.exists():
    print(
        f"\nWARNING: {BPLANK_DATASET_DIR} already exists. "
        "Hugging Face save_to_disk() may refuse to overwrite it."
    )
else:
    ner2.save_to_disk(
        str(BPLANK_DATASET_DIR)
    )


# Save previews
for split in ner2.keys():

    preview_path = (
        BPLANK_PREVIEW_DIR
        / f"{split}_preview.csv"
    )

    ner2[split].select(
        range(min(20, len(ner2[split])))
    ).to_pandas().to_csv(
        preview_path,
        index=False,
    )


print("\nbplank/danish_ner_transfer saved to:")
print(BPLANK_DIR.resolve())


# ============================================================
# Final directory tree
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(f"\nAll data stored under:\n{BASE_DIR.resolve()}\n")

for root, dirs, files in os.walk(BASE_DIR):

    root_path = Path(root)
    level = len(root_path.relative_to(BASE_DIR).parts)

    indent = "  " * level

    print(f"{indent}{root_path.name}/")

    for filename in files:
        print(f"{indent}  {filename}")