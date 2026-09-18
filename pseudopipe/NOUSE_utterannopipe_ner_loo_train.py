from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

# -----------------------------------------------------------------------------
# Project-local configuration / logging
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config_pipe as _config
except ImportError:
    _config = None

try:
    from loger import setup_logging
except ImportError as exc:
    raise ImportError(
        "Could not import project logger 'loger.py'. Keep loger.py in the notebook "
        "parent folder and this script in pseudopipe/."
    ) from exc


def cfg(name: str, default: Any) -> Any:
    return getattr(_config, name, default) if _config is not None else default


DEFAULT_MODELS = (
    # pure roberta
    "thomasbeste/danish-xlmr-ner-large",
    "FacebookAI/xlm-roberta-large",
    # sentence transformer
    "KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align",
    "microsoft/harrier-oss-v1-0.6b",
)

# Table 10 defaults from the uploaded report.
DEFAULT_LEARNING_RATE = float(cfg("PSEUDO_NER_LEARNING_RATE", 1e-5))
DEFAULT_TRAIN_BATCH_SIZE = int(cfg("PSEUDO_NER_TRAIN_BATCH_SIZE", 2))
DEFAULT_EVAL_BATCH_SIZE = int(cfg("PSEUDO_NER_EVAL_BATCH_SIZE", 2))
DEFAULT_GRAD_ACCUM = int(cfg("PSEUDO_NER_GRAD_ACCUM_STEPS", 8))
DEFAULT_EPOCHS = int(cfg("PSEUDO_NER_EPOCHS", 3))
DEFAULT_WEIGHT_DECAY = float(cfg("PSEUDO_NER_WEIGHT_DECAY", 0.01))

DEFAULT_MAX_LENGTH = int(cfg("PSEUDO_NER_MAX_LENGTH", 512))
DEFAULT_STRIDE = int(cfg("PSEUDO_NER_STRIDE", 64))
DEFAULT_SEED = int(cfg("PSEUDO_NER_SEED", 42))
DEFAULT_DEVICE = str(cfg("PSEUDO_NER_DEVICE", "auto"))
DEFAULT_GRADIENT_CHECKPOINTING = bool(cfg("PSEUDO_NER_GRADIENT_CHECKPOINTING", False))
DEFAULT_OUTPUT_ROOT = Path(
    cfg("PSEUDO_NER_OUTPUT_DIR", Path.cwd() / "out_pseudonymization_ner_loo")
).expanduser()
DEFAULT_LOG_FILE = Path(
    cfg("PSEUDO_NER_LOG_FILE", Path.cwd() / "pseudo_ner_loo.log")
).expanduser()
DEFAULT_GPU_CLEAN_SCRIPT = Path(
    cfg("PSEUDO_NER_GPU_CLEAN_SCRIPT", PROJECT_ROOT / "tool_clean_gpu.py")
).expanduser()

# Gold markup. The alternate leading ']' is accepted defensively because it was
# present in the task description, while normal data use [PersonData].
PERSONDATA_RE = re.compile(r"(?:\[|\])\s*PersonData\s*\]", flags=re.IGNORECASE)
GENERIC_TAG_RE = re.compile(r"\[([^\[\]\r\n]+)\]")
BIO_PREFIX_RE = re.compile(r"^[BIESU]-", flags=re.IGNORECASE)
QUOTED_ALF_RE = re.compile(
    r'^\s*start=.*?\s+stop=.*?\s+speaker_[^\s]+\s+"(?P<text>.*)"\s*$',
    flags=re.IGNORECASE,
)


def normalize_label(label: str) -> str:
    label = BIO_PREFIX_RE.sub("", str(label).strip()).upper()
    return re.sub(r"[\s\-]+", "_", label)


def normalize_labels(labels: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for label in labels:
        value = normalize_label(label)
        if not value or value == "O":
            continue
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one non-O target label is required.")
    return normalized


def safe_slug(value: str, max_len: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return (slug or "item")[:max_len]


def fold_directory_name(fold_index: int, holdout_file: str | Path) -> str:
    return f"fold_{fold_index:02d}_holdout_{safe_slug(Path(holdout_file).stem, 80)}"


def extract_transcript_text(line: str) -> str:
    """Extract only the transcript field from ALF/ALFRTTM-like input."""
    line = line.rstrip("\r\n")
    if not line.strip():
        return ""

    match = QUOTED_ALF_RE.match(line)
    if match:
        return match.group("text")

    if "\t" in line:
        return line.rsplit("\t", 1)[-1].strip().strip('"')

    if line.lstrip().startswith("SPEAKER "):
        fields = line.split(maxsplit=10)
        return fields[10].strip().strip('"') if len(fields) == 11 else ""

    return line.strip().strip('"')


def parse_persondata_markup(
    raw_text: str,
    target_labels: set[str],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Remove inline markup and return plain text plus target gold spans."""
    output: list[str] = []
    gold_spans: list[dict[str, Any]] = []
    observed_labels: list[str] = []
    output_length = 0
    i = 0

    while i < len(raw_text):
        opener = PERSONDATA_RE.match(raw_text, i)
        if opener:
            closing_tag = GENERIC_TAG_RE.search(raw_text, opener.end())
            if closing_tag is None:
                # Malformed opener: remove the markup but preserve following text.
                i = opener.end()
                continue

            surface = raw_text[opener.end() : closing_tag.start()]
            raw_label = normalize_label(closing_tag.group(1))
            observed_labels.append(raw_label)

            span_base = output_length
            output.append(surface)
            output_length += len(surface)

            if raw_label in target_labels and surface.strip():
                left_trim = len(surface) - len(surface.lstrip())
                right_edge = len(surface.rstrip())
                start = span_base + left_trim
                end = span_base + right_edge
                if end > start:
                    gold_spans.append({"start": start, "end": end, "label": raw_label})

            i = closing_tag.end()
            continue

        generic_tag = GENERIC_TAG_RE.match(raw_text, i)
        if generic_tag:
            # Generic markup not attached to PersonData is metadata, not text.
            i = generic_tag.end()
            continue

        output.append(raw_text[i])
        output_length += 1
        i += 1

    plain_text = "".join(output)
    gold_spans.sort(key=lambda x: (int(x["start"]), int(x["end"]), str(x["label"])))
    for left, right in zip(gold_spans, gold_spans[1:]):
        if int(left["end"]) > int(right["start"]):
            raise ValueError(f"Overlapping gold spans are not supported: {left!r} vs {right!r}")
    return plain_text, gold_spans, observed_labels


def load_records(
    files: Sequence[str | Path],
    labels: Sequence[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    target_set = set(labels)
    records: list[dict[str, Any]] = []
    observed = Counter()

    for file_path in files:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw_text = extract_transcript_text(line)
                if not raw_text:
                    continue
                text, spans, raw_labels = parse_persondata_markup(raw_text, target_set)
                observed.update(raw_labels)
                if not text.strip():
                    continue
                records.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "text": text,
                        "gold": spans,
                    }
                )

    if not records:
        raise ValueError("No non-empty transcript records were loaded from the selected files.")
    return records, observed


def make_label_maps(labels: Sequence[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    bio_labels = ["O"]
    for label in labels:
        bio_labels.extend((f"B-{label}", f"I-{label}"))
    label2id = {label: idx for idx, label in enumerate(bio_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return bio_labels, label2id, id2label


def set_reproducible_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN path without forcing unsupported deterministic kernels.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str):
    import torch

    value = requested.strip().lower()
    if value == "gpu":
        value = "cuda"
    if value == "auto":
        if torch.cuda.is_available():
            value = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            value = "mps"
        else:
            value = "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if value == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("MPS requested but unavailable.")
        return torch.device("mps")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError("--device must be one of auto, cpu, cuda, gpu, mps")


def align_offsets_to_bio(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[dict[str, Any]],
    label2id: dict[str, int],
) -> list[int]:
    labels: list[int] = []
    previous_span_index: int | None = None

    for token_start, token_end in offsets:
        token_start = int(token_start)
        token_end = int(token_end)
        if token_end <= token_start:  # special token / padding placeholder
            labels.append(-100)
            previous_span_index = None
            continue

        matched_index: int | None = None
        for span_index, span in enumerate(spans):
            span_start = int(span["start"])
            span_end = int(span["end"])
            if token_start < span_end and token_end > span_start:
                matched_index = span_index
                break

        if matched_index is None:
            labels.append(label2id["O"])
            previous_span_index = None
            continue

        entity_label = str(spans[matched_index]["label"])
        prefix = "I" if previous_span_index == matched_index else "B"
        labels.append(label2id[f"{prefix}-{entity_label}"])
        previous_span_index = matched_index

    return labels


def tokenize_records(
    records: Sequence[dict[str, Any]],
    tokenizer,
    label2id: dict[str, int],
    max_length: int,
    stride: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        encoded = tokenizer(
            str(record["text"]),
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
            padding=False,
        )
        if "offset_mapping" not in encoded:
            raise RuntimeError(
                "A fast tokenizer with offset_mapping support is required for exact NER spans."
            )

        n_chunks = len(encoded["input_ids"])
        for chunk_index in range(n_chunks):
            offsets = encoded["offset_mapping"][chunk_index]
            example: dict[str, Any] = {
                key: encoded[key][chunk_index]
                for key in tokenizer.model_input_names
                if key in encoded
            }
            example["labels"] = align_offsets_to_bio(offsets, record["gold"], label2id)
            example["record_index"] = record_index
            example["chunk_index"] = chunk_index
            examples.append(example)

    if not examples:
        raise ValueError("Tokenization produced no training examples.")
    return examples


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV columns for empty rows: {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_gpu_cleaner(clean_script: Path, logger: logging.Logger) -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
    if hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except (RuntimeError, AttributeError):
            pass

    if clean_script.is_file():
        logger.info("Running GPU cleanup script: %s", clean_script)
        result = subprocess.run([sys.executable, str(clean_script)], check=False)
        if result.returncode != 0:
            logger.warning("GPU cleanup script exited with code %s", result.returncode)
    else:
        logger.warning("GPU cleanup script not found: %s", clean_script)


def model_parameter_counts(model) -> tuple[int, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_fold(
    *,
    model_id: str,
    model_revision: str,
    train_records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    fold_dir: Path,
    holdout_file: Path,
    train_files: Sequence[Path],
    device,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    grad_accum_steps: int,
    epochs: int,
    weight_decay: float,
    max_length: int,
    stride: int,
    seed: int,
    gradient_checkpointing: bool,
    hf_token: str | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification

    set_reproducible_seed(seed)
    bio_labels, label2id, id2label = make_label_maps(labels)

    logger.info("Loading tokenizer/model: %s (revision=%s)", model_id, model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        token=hf_token,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(f"{model_id} did not provide a fast tokenizer; offsets are required.")

    backbone = AutoModel.from_pretrained(
        model_id,
        revision=model_revision,
        token=hf_token,
    )
    if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False

    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError(f"Could not determine hidden_size for {model_id}.")

    class TokenClassifier(nn.Module):
        def __init__(self, backbone_model, n_labels: int, hidden: int):
            super().__init__()
            self.backbone = backbone_model
            self.classifier = nn.Linear(hidden, n_labels)

        def forward(self, labels_tensor=None, **inputs):
            outputs = self.backbone(**inputs, return_dict=True)
            logits = self.classifier(outputs.last_hidden_state)
            loss = None
            if labels_tensor is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels_tensor.reshape(-1))
            return loss, logits

    model = TokenClassifier(backbone, len(bio_labels), int(hidden_size))
    model.to(device)

    tokenized_examples = tokenize_records(
        train_records,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
        stride=stride,
    )

    class NerDataset(Dataset):
        def __init__(self, examples: Sequence[dict[str, Any]]):
            self.examples = list(examples)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = self.examples[index]
            return {
                key: value
                for key, value in item.items()
                if key not in {"record_index", "chunk_index"}
            }

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        NerDataset(tokenized_examples),
        batch_size=train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_params, trainable_params = model_parameter_counts(model)
    logger.info(
        "Fold holdout=%s | records=%d | chunks=%d | params=%d | trainable=%d | device=%s",
        holdout_file.name,
        len(train_records),
        len(tokenized_examples),
        total_params,
        trainable_params,
        device,
    )

    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    global_optimizer_steps = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        epoch_optimizer_steps = 0

        for batch_index, batch in enumerate(loader, start=1):
            labels_tensor = batch.pop("labels").to(device)
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            loss, _ = model(labels_tensor=labels_tensor, **model_inputs)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch={epoch}, batch={batch_index}: {loss}")

            raw_loss = float(loss.detach().cpu())
            (loss / grad_accum_steps).backward()
            running_loss += raw_loss
            batches += 1

            should_step = (batch_index % grad_accum_steps == 0) or (batch_index == len(loader))
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_optimizer_steps += 1
                epoch_optimizer_steps += 1

            if batch_index == 1 or batch_index == len(loader) or batch_index % 25 == 0:
                logger.info(
                    "model=%s holdout=%s epoch=%d/%d batch=%d/%d loss=%.6f",
                    model_id,
                    holdout_file.name,
                    epoch,
                    epochs,
                    batch_index,
                    len(loader),
                    raw_loss,
                )

        mean_loss = running_loss / max(batches, 1)
        history.append(
            {
                "model": model_id,
                "holdout_file": str(holdout_file),
                "epoch": epoch,
                "mean_train_loss": mean_loss,
                "batches": batches,
                "optimizer_steps": epoch_optimizer_steps,
                "global_optimizer_steps": global_optimizer_steps,
            }
        )
        logger.info(
            "Completed model=%s holdout=%s epoch=%d mean_loss=%.6f",
            model_id,
            holdout_file.name,
            epoch,
            mean_loss,
        )

    # Save a self-contained fine-tuned backbone + tokenizer and the linear NER head.
    fold_dir.mkdir(parents=True, exist_ok=True)
    backbone_dir = fold_dir / "backbone"
    tokenizer_dir = fold_dir / "tokenizer"
    model.backbone.save_pretrained(backbone_dir, safe_serialization=True)
    tokenizer.save_pretrained(tokenizer_dir)
    torch.save(model.classifier.state_dict(), fold_dir / "classifier_head.pt")

    write_csv(
        fold_dir / "labels.csv",
        [{"label_id": idx, "bio_label": label} for idx, label in enumerate(bio_labels)],
        fieldnames=("label_id", "bio_label"),
    )
    write_csv(fold_dir / "training_history.csv", history)

    target_counts = Counter(
        str(span["label"])
        for record in train_records
        for span in record["gold"]
    )
    write_csv(
        fold_dir / "training_label_counts.csv",
        [
            {"label": label, "entity_count": int(target_counts.get(label, 0))}
            for label in labels
        ],
        fieldnames=("label", "entity_count"),
    )

    absent = [label for label in labels if target_counts.get(label, 0) == 0]
    if absent:
        logger.warning(
            "No positive training examples for labels in holdout %s: %s",
            holdout_file.name,
            ", ".join(absent),
        )

    source_commit = str(getattr(model.backbone.config, "_commit_hash", "") or "")
    manifest = {
        "model": model_id,
        "requested_revision": model_revision,
        "resolved_commit": source_commit,
        "holdout_file": str(holdout_file),
        "train_files": " | ".join(str(path) for path in train_files),
        "target_labels": " | ".join(labels),
        "seed": seed,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "max_length": max_length,
        "stride": stride,
        "gradient_checkpointing": int(gradient_checkpointing),
        "device": str(device),
        "train_records": len(train_records),
        "train_chunks": len(tokenized_examples),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "labels_absent_from_training": " | ".join(absent),
        "python": sys.version.split()[0],
    }
    write_csv(fold_dir / "fold_manifest.csv", [manifest], fieldnames=list(manifest.keys()))

    del optimizer, loader, collator, tokenized_examples, model, backbone, tokenizer
    gc.collect()
    return manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="File-level leave-one-out NER fine-tuning for inline [PersonData]... [LABEL] ALFRTTM data."
    )
    parser.add_argument("--files", nargs="+", required=True, help="Selected trainfiles only.")
    parser.add_argument("--model", required=True, choices=DEFAULT_MODELS)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=("auto", "cpu", "cuda", "gpu", "mps"))
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GRADIENT_CHECKPOINTING,
    )
    parser.add_argument("--gpu-clean-script", type=Path, default=DEFAULT_GPU_CLEAN_SCRIPT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    labels = normalize_labels(args.labels)
    files = sorted({Path(path).expanduser().resolve() for path in args.files}, key=lambda p: str(p).lower())
    if len(files) < 2:
        raise ValueError("Leave-one-out training requires at least two files.")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(args.log_file.expanduser().resolve()), level=logging.INFO)
    logger.info("Starting LOO fine-tuning with Python: %s", sys.executable)
    logger.info("Model: %s", args.model)
    logger.info("Files (%d): %s", len(files), " | ".join(str(p) for p in files))
    logger.info("Target labels: %s", " | ".join(labels))
    logger.info(
        "Table-10 defaults/current values: AdamW lr=%g train_bs=%d eval_bs=%d grad_accum=%d epochs=%d weight_decay=%g",
        args.learning_rate,
        args.train_batch_size,
        args.eval_batch_size,
        args.grad_accum_steps,
        args.epochs,
        args.weight_decay,
    )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token and _config is not None:
        token_from_cfg = getattr(_config, "HG_TOKEN", None)
        if token_from_cfg and token_from_cfg != "YOUR_TOKEN":
            hf_token = str(token_from_cfg)

    device = resolve_device(args.device)
    model_root = output_root / safe_slug(args.model)
    model_root.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []

    for fold_index, holdout_file in enumerate(files, start=1):
        fold_dir = model_root / fold_directory_name(fold_index, holdout_file)
        required_artifacts = (
            fold_dir / "backbone" / "config.json",
            fold_dir / "tokenizer",
            fold_dir / "classifier_head.pt",
            fold_dir / "labels.csv",
        )
        if not args.overwrite and all(path.exists() for path in required_artifacts):
            logger.info("Skipping existing fold: %s", fold_dir)
            continue

        train_files = [path for path in files if path != holdout_file]
        train_records, observed = load_records(train_files, labels)
        logger.info(
            "Fold %d/%d holdout=%s observed annotation labels in training=%s",
            fold_index,
            len(files),
            holdout_file.name,
            dict(observed),
        )

        try:
            manifest = train_one_fold(
                model_id=args.model,
                model_revision=args.model_revision,
                train_records=train_records,
                labels=labels,
                fold_dir=fold_dir,
                holdout_file=holdout_file,
                train_files=train_files,
                device=device,
                learning_rate=args.learning_rate,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                grad_accum_steps=args.grad_accum_steps,
                epochs=args.epochs,
                weight_decay=args.weight_decay,
                max_length=args.max_length,
                stride=args.stride,
                seed=args.seed,
                gradient_checkpointing=args.gradient_checkpointing,
                hf_token=hf_token,
                logger=logger,
            )
            manifest["fold"] = fold_index
            manifests.append(manifest)
        finally:
            # Required after every fold/fine-tuning run.
            run_gpu_cleaner(args.gpu_clean_script.expanduser().resolve(), logger)

    if manifests:
        manifest_fields = ["fold"] + [key for key in manifests[0].keys() if key != "fold"]
        write_csv(model_root / "training_manifest.csv", manifests, fieldnames=manifest_fields)
    logger.info("Finished LOO fine-tuning for %s", args.model)


if __name__ == "__main__":
    main()
