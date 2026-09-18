from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
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
    cfg("PSEUDO_NER_GPU_CLEAN_SCRIPT", SCRIPT_DIR / "tool_clean_gpu.py")
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



SCENARIOS = ("strict", "exact", "partial", "ent_type")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def spans_to_character_bio(text: str, spans: Sequence[dict[str, Any]]) -> list[str]:
    """Match the character-BIO nervaluate logic in the user's existing evaluator."""
    tags = ["O"] * len(text)
    for span in sorted(spans, key=lambda x: (int(x["start"]), int(x["end"]))):
        start = int(span["start"])
        end = int(span["end"])
        label = str(span["label"])
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"Invalid span {span!r} for text length {len(text)}")
        if any(tag != "O" for tag in tags[start:end]):
            raise ValueError(f"Overlapping span detected: {span!r}")
        tags[start] = f"B-{label}"
        for position in range(start + 1, end):
            tags[position] = f"I-{label}"
    return tags


def result_fields(result: Any) -> dict[str, Any]:
    return {
        "correct": int(result.correct),
        "incorrect": int(result.incorrect),
        "partial": int(result.partial),
        "missed": int(result.missed),
        "spurious": int(result.spurious),
        "precision": float(result.precision),
        "recall": float(result.recall),
        "f1": float(result.f1),
    }


def empty_result_fields() -> dict[str, Any]:
    return {
        "correct": 0,
        "incorrect": 0,
        "partial": 0,
        "missed": 0,
        "spurious": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def evaluate_sequences(
    *,
    model_name: str,
    labels: Sequence[str],
    records: Sequence[dict[str, Any]],
    predictions: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from nervaluate.evaluator import Evaluator

    gold_bio = [spans_to_character_bio(str(record["text"]), record["gold"]) for record in records]
    pred_bio = [
        spans_to_character_bio(str(record["text"]), predicted)
        for record, predicted in zip(records, predictions, strict=True)
    ]
    evaluator = Evaluator(true=gold_bio, pred=pred_bio, tags=list(labels), loader="list")
    results = evaluator.evaluate()

    overall_rows = [
        {"model": model_name, "scenario": scenario, **result_fields(result)}
        for scenario, result in results["overall"].items()
    ]

    label_rows: list[dict[str, Any]] = []
    for label in labels:
        entity_results = results["entities"].get(label, {})
        for scenario in SCENARIOS:
            result = entity_results.get(scenario)
            label_rows.append(
                {
                    "model": model_name,
                    "label": label,
                    "scenario": scenario,
                    **(result_fields(result) if result is not None else empty_result_fields()),
                }
            )
    return overall_rows, label_rows


def token_predictions_to_spans(
    offsets: Sequence[Sequence[int]],
    pred_ids: Sequence[int],
    confidences: Sequence[float],
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and int(current["end"]) > int(current["start"]):
            scores = current.pop("_scores")
            current["score"] = float(sum(scores) / len(scores)) if scores else 0.0
            spans.append(current)
        current = None

    for (start, end), pred_id, confidence in zip(offsets, pred_ids, confidences, strict=True):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        tag = id2label[int(pred_id)]
        if tag == "O" or "-" not in tag:
            flush()
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or current is None or str(current["label"]) != label:
            flush()
            current = {
                "start": start,
                "end": end,
                "label": label,
                "_scores": [float(confidence)],
            }
        else:
            current["end"] = max(int(current["end"]), end)
            current["_scores"].append(float(confidence))
    flush()
    return spans


def resolve_prediction_overlaps(
    spans: Sequence[dict[str, Any]],
    text_length: int,
) -> list[dict[str, Any]]:
    unique: dict[tuple[int, int, str], dict[str, Any]] = {}
    for span in spans:
        start = max(0, min(int(span["start"]), text_length))
        end = max(start, min(int(span["end"]), text_length))
        if end <= start:
            continue
        candidate = {
            **span,
            "start": start,
            "end": end,
            "score": float(span.get("score", 0.0)),
        }
        key = (start, end, str(candidate["label"]))
        if key not in unique or candidate["score"] > unique[key]["score"]:
            unique[key] = candidate

    candidates = sorted(
        unique.values(),
        key=lambda x: (
            -float(x["score"]),
            -(int(x["end"]) - int(x["start"])),
            int(x["start"]),
            int(x["end"]),
            str(x["label"]),
        ),
    )
    occupied = [False] * text_length
    accepted: list[dict[str, Any]] = []
    for span in candidates:
        start, end = int(span["start"]), int(span["end"])
        if any(occupied[start:end]):
            continue
        accepted.append(span)
        for position in range(start, end):
            occupied[position] = True
    return sorted(accepted, key=lambda x: (int(x["start"]), int(x["end"]), str(x["label"])))


def predict_records_with_model(
    *,
    records: Sequence[dict[str, Any]],
    tokenizer,
    model,
    id2label: dict[int, str],
    device,
    batch_size: int,
    max_length: int,
    stride: int,
) -> list[list[dict[str, Any]]]:
    import torch

    chunks: list[dict[str, Any]] = []
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
            raise RuntimeError("Fast-tokenizer offset mapping is required for exact span evaluation.")
        for chunk_index in range(len(encoded["input_ids"])):
            inputs = {
                key: encoded[key][chunk_index]
                for key in tokenizer.model_input_names
                if key in encoded
            }
            chunks.append(
                {
                    "record_index": record_index,
                    "chunk_index": chunk_index,
                    "offsets": encoded["offset_mapping"][chunk_index],
                    "inputs": inputs,
                }
            )

    spans_by_record: list[list[dict[str, Any]]] = [[] for _ in records]
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            padded = tokenizer.pad(
                [chunk["inputs"] for chunk in batch_chunks],
                padding=True,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(device) for key, value in padded.items()}
            _, logits = model(**model_inputs)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            pred = pred.detach().cpu()
            conf = conf.detach().cpu()

            for row_index, chunk in enumerate(batch_chunks):
                offsets = chunk["offsets"]
                token_count = len(offsets)
                chunk_spans = token_predictions_to_spans(
                    offsets=offsets,
                    pred_ids=pred[row_index, :token_count].tolist(),
                    confidences=conf[row_index, :token_count].tolist(),
                    id2label=id2label,
                )
                spans_by_record[int(chunk["record_index"])].extend(chunk_spans)

    return [
        resolve_prediction_overlaps(spans, len(str(record["text"])))
        for record, spans in zip(records, spans_by_record, strict=True)
    ]


def serialize_spans(spans: Sequence[dict[str, Any]]) -> str:
    payload = [
        {
            "start": int(span["start"]),
            "end": int(span["end"]),
            "label": str(span["label"]),
            **({"score": round(float(span["score"]), 8)} if "score" in span else {}),
        }
        for span in spans
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def deserialize_spans(value: str) -> list[dict[str, Any]]:
    payload = json.loads(value)
    if not isinstance(payload, list):
        raise ValueError("Span CSV payload must be a JSON list.")
    return [dict(item) for item in payload]


def fold_summary(rows: Sequence[dict[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[k]) for k in group_keys)
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        result = {name: value for name, value in zip(group_keys, key, strict=True)}
        result["n_folds"] = len(group)
        for metric in ("precision", "recall", "f1"):
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        for count_name in ("correct", "incorrect", "partial", "missed", "spurious"):
            result[f"{count_name}_sum"] = sum(int(float(row[count_name])) for row in group)
        summaries.append(result)
    return summaries


def train_and_validate_one_fold(
    *,
    model_id: str,
    model_revision: str,
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    fold_index: int,
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
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[list[dict[str, Any]]]]:
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
        "Fold %d holdout=%s | train_records=%d | validation_records=%d | chunks=%d | params=%d | trainable=%d | device=%s",
        fold_index,
        holdout_file.name,
        len(train_records),
        len(validation_records),
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
                raise RuntimeError(
                    f"Non-finite training loss at epoch={epoch}, batch={batch_index}: {loss}"
                )

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
                "fold": fold_index,
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

    # The held-out file has never entered the optimizer. Evaluate it now, while
    # the fold-specific model is still resident in memory.
    logger.info("Evaluating held-out validation file immediately: %s", holdout_file.name)
    predictions = predict_records_with_model(
        records=validation_records,
        tokenizer=tokenizer,
        model=model,
        id2label=id2label,
        device=device,
        batch_size=eval_batch_size,
        max_length=max_length,
        stride=stride,
    )
    overall_rows, label_rows = evaluate_sequences(
        model_name=model_id,
        labels=labels,
        records=validation_records,
        predictions=predictions,
    )

    for row in overall_rows:
        row.update(
            {
                "fold": fold_index,
                "holdout_file": str(holdout_file),
                "n_records": len(validation_records),
            }
        )
    for row in label_rows:
        row.update(
            {
                "fold": fold_index,
                "holdout_file": str(holdout_file),
                "n_records": len(validation_records),
            }
        )

    prediction_rows = [
        {
            "model": model_id,
            "fold": fold_index,
            "holdout_file": str(holdout_file),
            "source_file": str(record["file"]),
            "line": int(record["line"]),
            "text": str(record["text"]),
            "gold_spans": serialize_spans(record["gold"]),
            "predicted_spans": serialize_spans(predicted),
        }
        for record, predicted in zip(validation_records, predictions, strict=True)
    ]

    # Save model checkpoint and all fold-level CSV outputs together.
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
    write_csv(fold_dir / "validation_predictions.csv", prediction_rows)
    write_csv(fold_dir / "validation_overall_metrics.csv", overall_rows)
    write_csv(fold_dir / "validation_by_label_metrics.csv", label_rows)

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

    validation_counts = Counter(
        str(span["label"])
        for record in validation_records
        for span in record["gold"]
    )
    write_csv(
        fold_dir / "validation_label_counts.csv",
        [
            {"label": label, "entity_count": int(validation_counts.get(label, 0))}
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
        "fold": fold_index,
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
        "validation_records": len(validation_records),
        "train_chunks": len(tokenized_examples),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "labels_absent_from_training": " | ".join(absent),
        "python": sys.version.split()[0],
    }
    write_csv(fold_dir / "fold_manifest.csv", [manifest], fieldnames=list(manifest.keys()))

    strict = [row for row in overall_rows if row["scenario"] == "strict"]
    if strict:
        logger.info(
            "Fold %d held-out strict: precision=%.6f recall=%.6f f1=%.6f",
            fold_index,
            float(strict[0]["precision"]),
            float(strict[0]["recall"]),
            float(strict[0]["f1"]),
        )

    del optimizer, loader, collator, tokenized_examples, model, backbone, tokenizer
    gc.collect()
    return manifest, overall_rows, label_rows, predictions


def load_completed_fold(
    *,
    fold_dir: Path,
    model_id: str,
    fold_index: int,
    holdout_file: Path,
    validation_records: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[list[dict[str, Any]]]]:
    manifest_rows = read_csv_rows(fold_dir / "fold_manifest.csv")
    overall_rows = [dict(row) for row in read_csv_rows(fold_dir / "validation_overall_metrics.csv")]
    label_rows = [dict(row) for row in read_csv_rows(fold_dir / "validation_by_label_metrics.csv")]
    prediction_rows = read_csv_rows(fold_dir / "validation_predictions.csv")

    if len(manifest_rows) != 1:
        raise ValueError(f"Expected one manifest row in {fold_dir}")
    if len(prediction_rows) != len(validation_records):
        raise ValueError(
            f"Existing fold predictions ({len(prediction_rows)}) do not match current validation "
            f"records ({len(validation_records)}) in {holdout_file}. Use --overwrite."
        )

    predictions = [deserialize_spans(row["predicted_spans"]) for row in prediction_rows]
    manifest: dict[str, Any] = dict(manifest_rows[0])
    manifest["fold"] = fold_index

    for row in overall_rows:
        row["fold"] = fold_index
        row["holdout_file"] = str(holdout_file)
        row["n_records"] = len(validation_records)
    for row in label_rows:
        row["fold"] = fold_index
        row["holdout_file"] = str(holdout_file)
        row["n_records"] = len(validation_records)

    return manifest, overall_rows, label_rows, predictions


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combined file-level leave-one-out NER pipeline: for each fold, train on N-1 files, "
            "immediately evaluate the held-out file with nervaluate, then aggregate out-of-fold CSV metrics."
        )
    )
    parser.add_argument("--files", nargs="+", required=True, help="Selected human-labelled files only.")
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
    files = sorted(
        {Path(path).expanduser().resolve() for path in args.files},
        key=lambda p: str(p).lower(),
    )
    if len(files) < 2:
        raise ValueError("Leave-one-out requires at least two files.")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(args.log_file.expanduser().resolve()), level=logging.INFO)
    logger.info("Starting COMBINED LOO train+validation with Python: %s", sys.executable)
    logger.info("Model: %s", args.model)
    logger.info("Files (%d): %s", len(files), " | ".join(str(p) for p in files))
    logger.info("Target labels: %s", " | ".join(labels))
    logger.info(
        "Table-10 values: AdamW lr=%g train_bs=%d eval_bs=%d grad_accum=%d epochs=%d weight_decay=%g",
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
    all_overall_fold_rows: list[dict[str, Any]] = []
    all_label_fold_rows: list[dict[str, Any]] = []
    pooled_records: list[dict[str, Any]] = []
    pooled_predictions: list[list[dict[str, Any]]] = []
    all_prediction_rows: list[dict[str, Any]] = []

    for fold_index, holdout_file in enumerate(files, start=1):
        fold_name = fold_directory_name(fold_index, holdout_file)
        fold_dir = model_root / fold_name
        train_files = [path for path in files if path != holdout_file]

        # Load held-out data separately. It is never passed into tokenized training data.
        validation_records, validation_observed = load_records([holdout_file], labels)
        logger.info(
            "Fold %d/%d: TRAIN on %d files -> VALIDATE on %s | validation labels=%s",
            fold_index,
            len(files),
            len(train_files),
            holdout_file.name,
            dict(validation_observed),
        )

        complete_artifacts = (
            fold_dir / "fold_manifest.csv",
            fold_dir / "validation_predictions.csv",
            fold_dir / "validation_overall_metrics.csv",
            fold_dir / "validation_by_label_metrics.csv",
        )

        if not args.overwrite and all(path.is_file() for path in complete_artifacts):
            logger.info("Resuming from completed fold CSVs: %s", fold_dir)
            manifest, overall_rows, label_rows, predictions = load_completed_fold(
                fold_dir=fold_dir,
                model_id=args.model,
                fold_index=fold_index,
                holdout_file=holdout_file,
                validation_records=validation_records,
            )
        else:
            train_records, training_observed = load_records(train_files, labels)
            logger.info(
                "Fold %d training annotation labels=%s",
                fold_index,
                dict(training_observed),
            )
            try:
                manifest, overall_rows, label_rows, predictions = train_and_validate_one_fold(
                    model_id=args.model,
                    model_revision=args.model_revision,
                    train_records=train_records,
                    validation_records=validation_records,
                    labels=labels,
                    fold_index=fold_index,
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
            finally:
                # Required after each fold's fine-tuning/evaluation cycle.
                run_gpu_cleaner(args.gpu_clean_script.expanduser().resolve(), logger)

        manifests.append(manifest)
        all_overall_fold_rows.extend(overall_rows)
        all_label_fold_rows.extend(label_rows)
        pooled_records.extend(validation_records)
        pooled_predictions.extend(predictions)

        fold_prediction_rows = [
            {
                "model": args.model,
                "fold": fold_index,
                "holdout_file": str(holdout_file),
                "source_file": str(record["file"]),
                "line": int(record["line"]),
                "text": str(record["text"]),
                "gold_spans": serialize_spans(record["gold"]),
                "predicted_spans": serialize_spans(predicted),
            }
            for record, predicted in zip(validation_records, predictions, strict=True)
        ]
        all_prediction_rows.extend(fold_prediction_rows)
        write_csv(model_root / "predictions" / f"{fold_name}.csv", fold_prediction_rows)

    # Pooled OOF: each record is scored exactly once by a model that never trained on its file.
    pooled_overall, pooled_by_label = evaluate_sequences(
        model_name=args.model,
        labels=labels,
        records=pooled_records,
        predictions=pooled_predictions,
    )
    for row in pooled_overall:
        row.update({"n_files": len(files), "n_records": len(pooled_records)})
    for row in pooled_by_label:
        row.update({"n_files": len(files), "n_records": len(pooled_records)})

    overall_summary = fold_summary(all_overall_fold_rows, group_keys=("model", "scenario"))
    label_summary = fold_summary(all_label_fold_rows, group_keys=("model", "label", "scenario"))

    # Model-level CSVs.
    write_csv(model_root / "training_manifest.csv", manifests)
    write_csv(model_root / "overall_by_fold.csv", all_overall_fold_rows)
    write_csv(model_root / "overall_oof.csv", pooled_overall)
    write_csv(model_root / "overall_fold_summary.csv", overall_summary)
    write_csv(model_root / "by_label_by_fold.csv", all_label_fold_rows)
    write_csv(model_root / "by_label_oof.csv", pooled_by_label)
    write_csv(model_root / "by_label_fold_summary.csv", label_summary)
    write_csv(model_root / "predictions" / "oof_predictions.csv", all_prediction_rows)

    # Reproducible dedicated output folders.
    write_csv(model_root / "overall" / "fold_metrics.csv", all_overall_fold_rows)
    write_csv(model_root / "overall" / "oof_metrics.csv", pooled_overall)
    write_csv(model_root / "overall" / "summary_across_folds.csv", overall_summary)

    for label in labels:
        label_dir = model_root / "labels" / safe_slug(label)
        write_csv(
            label_dir / "fold_metrics.csv",
            [row for row in all_label_fold_rows if str(row["label"]) == label],
        )
        write_csv(
            label_dir / "oof_metrics.csv",
            [row for row in pooled_by_label if str(row["label"]) == label],
        )
        write_csv(
            label_dir / "summary_across_folds.csv",
            [row for row in label_summary if str(row["label"]) == label],
        )

    strict = [row for row in pooled_overall if row["scenario"] == "strict"]
    if strict:
        logger.info(
            "FINAL pooled OOF strict for %s: precision=%.6f recall=%.6f f1=%.6f",
            args.model,
            float(strict[0]["precision"]),
            float(strict[0]["recall"]),
            float(strict[0]["f1"]),
        )
    logger.info("Finished combined LOO train+validation. CSV outputs: %s", model_root)


if __name__ == "__main__":
    main()
