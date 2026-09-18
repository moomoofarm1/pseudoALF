from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

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

# Reuse parsing/fold naming from the training module in the same pseudopipe folder.
from utterannopipe_ner_loo_train import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_STRIDE,
    fold_directory_name,
    load_records,
    make_label_maps,
    normalize_labels,
    resolve_device,
    safe_slug,
    set_reproducible_seed,
)


def cfg(name: str, default: Any) -> Any:
    return getattr(_config, name, default) if _config is not None else default


DEFAULT_RESULTS_ROOT = Path(
    cfg("PSEUDO_NER_EVAL_DIR", Path.cwd() / "out_pseudonymization_ner_loo_csv")
).expanduser()
DEFAULT_LOG_FILE = Path(
    cfg("PSEUDO_NER_LOG_FILE", Path.cwd() / "pseudo_ner_loo.log")
).expanduser()
SCENARIOS = ("strict", "exact", "partial", "ent_type")


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


def read_labels_csv(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty label mapping: {path}")
    rows.sort(key=lambda row: int(row["label_id"]))
    return [str(row["bio_label"]) for row in rows]


def spans_to_character_bio(text: str, spans: Sequence[dict[str, Any]]) -> list[str]:
    """Same character-BIO representation used by evaluate_alfrttm_ner_multivenv_conda.py."""
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[str]], list[list[str]]]:
    from nervaluate.evaluator import Evaluator

    gold_bio = [spans_to_character_bio(record["text"], record["gold"]) for record in records]
    pred_bio = [
        spans_to_character_bio(record["text"], predicted)
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
    return overall_rows, label_rows, gold_bio, pred_bio


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
        if tag == "O":
            flush()
            continue
        if "-" not in tag:
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
        else:  # I-same-label
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
        candidate = {**span, "start": start, "end": end, "score": float(span.get("score", 0.0))}
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


def load_finetuned_model(fold_dir: Path, device, expected_labels: Sequence[str]):
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    bio_labels, _, expected_id2label = make_label_maps(expected_labels)
    saved_bio_labels = read_labels_csv(fold_dir / "labels.csv")
    if saved_bio_labels != bio_labels:
        raise ValueError(
            f"Evaluation labels do not match checkpoint label space in {fold_dir}.\n"
            f"checkpoint={saved_bio_labels}\nrequested={bio_labels}"
        )

    tokenizer = AutoTokenizer.from_pretrained(fold_dir / "tokenizer", use_fast=True)
    backbone = AutoModel.from_pretrained(fold_dir / "backbone")
    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError(f"Could not determine hidden_size for checkpoint {fold_dir}")
    classifier = nn.Linear(int(hidden_size), len(bio_labels))
    try:
        state = torch.load(fold_dir / "classifier_head.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(fold_dir / "classifier_head.pt", map_location="cpu")
    classifier.load_state_dict(state)
    backbone.to(device).eval()
    classifier.to(device).eval()
    return tokenizer, backbone, classifier, expected_id2label


def predict_records(
    *,
    records: Sequence[dict[str, Any]],
    tokenizer,
    backbone,
    classifier,
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
    with torch.inference_mode():
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            padded = tokenizer.pad(
                [chunk["inputs"] for chunk in batch_chunks],
                padding=True,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(device) for key, value in padded.items()}
            outputs = backbone(**model_inputs, return_dict=True)
            logits = classifier(outputs.last_hidden_state)
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
            result[f"{count_name}_sum"] = sum(int(row[count_name]) for row in group)
        summaries.append(result)
    return summaries


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate file-level LOO fine-tuned NER checkpoints with nervaluate and CSV outputs."
    )
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--model", required=True, choices=DEFAULT_MODELS)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--device", default=str(cfg("PSEUDO_NER_DEVICE", "auto")), choices=("auto", "cpu", "cuda", "gpu", "mps"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    labels = normalize_labels(args.labels)
    files = sorted({Path(path).expanduser().resolve() for path in args.files}, key=lambda p: str(p).lower())
    if len(files) < 2:
        raise ValueError("LOO evaluation requires at least two files.")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)

    logger = setup_logging(str(args.log_file.expanduser().resolve()), level=logging.INFO)
    logger.info("Starting LOO evaluation with Python: %s", sys.executable)
    logger.info("Model: %s | labels: %s", args.model, " | ".join(labels))
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)

    model_slug = safe_slug(args.model)
    checkpoint_model_root = args.checkpoint_root.expanduser().resolve() / model_slug
    result_model_root = args.results_root.expanduser().resolve() / model_slug
    result_model_root.mkdir(parents=True, exist_ok=True)

    all_overall_fold_rows: list[dict[str, Any]] = []
    all_label_fold_rows: list[dict[str, Any]] = []
    pooled_records: list[dict[str, Any]] = []
    pooled_predictions: list[list[dict[str, Any]]] = []

    for fold_index, holdout_file in enumerate(files, start=1):
        fold_name = fold_directory_name(fold_index, holdout_file)
        fold_dir = checkpoint_model_root / fold_name
        if not fold_dir.is_dir():
            raise FileNotFoundError(f"Missing checkpoint fold directory: {fold_dir}")

        records, observed = load_records([holdout_file], labels)
        logger.info(
            "Evaluating fold %d/%d holdout=%s records=%d observed_labels=%s",
            fold_index,
            len(files),
            holdout_file.name,
            len(records),
            dict(observed),
        )

        tokenizer = backbone = classifier = None
        try:
            tokenizer, backbone, classifier, id2label = load_finetuned_model(
                fold_dir, device=device, expected_labels=labels
            )
            predictions = predict_records(
                records=records,
                tokenizer=tokenizer,
                backbone=backbone,
                classifier=classifier,
                id2label=id2label,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                stride=args.stride,
            )
        finally:
            if classifier is not None:
                del classifier
            if backbone is not None:
                del backbone
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        overall_rows, label_rows, _, _ = evaluate_sequences(
            model_name=args.model,
            labels=labels,
            records=records,
            predictions=predictions,
        )
        for row in overall_rows:
            row.update(
                {
                    "fold": fold_index,
                    "holdout_file": str(holdout_file),
                    "n_records": len(records),
                }
            )
            all_overall_fold_rows.append(row)
        for row in label_rows:
            row.update(
                {
                    "fold": fold_index,
                    "holdout_file": str(holdout_file),
                    "n_records": len(records),
                }
            )
            all_label_fold_rows.append(row)

        prediction_rows = []
        for record, predicted in zip(records, predictions, strict=True):
            prediction_rows.append(
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
            )
        write_csv(result_model_root / "predictions" / f"{fold_name}.csv", prediction_rows)

        pooled_records.extend(records)
        pooled_predictions.extend(predictions)

    # Pooled out-of-fold evaluation: every source file is predicted only by the model
    # that did not train on that file.
    pooled_overall, pooled_by_label, _, _ = evaluate_sequences(
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

    # Flat per-model CSVs for easy comparison/concatenation.
    write_csv(result_model_root / "overall_by_fold.csv", all_overall_fold_rows)
    write_csv(result_model_root / "overall_oof.csv", pooled_overall)
    write_csv(result_model_root / "overall_fold_summary.csv", overall_summary)
    write_csv(result_model_root / "by_label_by_fold.csv", all_label_fold_rows)
    write_csv(result_model_root / "by_label_oof.csv", pooled_by_label)
    write_csv(result_model_root / "by_label_fold_summary.csv", label_summary)

    # Requested separate subfolders for overall and each individual label.
    write_csv(result_model_root / "overall" / "fold_metrics.csv", all_overall_fold_rows)
    write_csv(result_model_root / "overall" / "oof_metrics.csv", pooled_overall)
    write_csv(result_model_root / "overall" / "summary_across_folds.csv", overall_summary)

    for label in labels:
        label_dir = result_model_root / "labels" / safe_slug(label)
        fold_rows = [row for row in all_label_fold_rows if row["label"] == label]
        oof_rows = [row for row in pooled_by_label if row["label"] == label]
        summary_rows = [row for row in label_summary if row["label"] == label]
        write_csv(label_dir / "fold_metrics.csv", fold_rows)
        write_csv(label_dir / "oof_metrics.csv", oof_rows)
        write_csv(label_dir / "summary_across_folds.csv", summary_rows)

    logger.info("Saved CSV evaluation outputs under: %s", result_model_root)
    strict = [row for row in pooled_overall if row["scenario"] == "strict"]
    if strict:
        logger.info(
            "Primary pooled OOF strict result for %s: precision=%.6f recall=%.6f f1=%.6f",
            args.model,
            float(strict[0]["precision"]),
            float(strict[0]["recall"]),
            float(strict[0]["f1"]),
        )


if __name__ == "__main__":
    main()
