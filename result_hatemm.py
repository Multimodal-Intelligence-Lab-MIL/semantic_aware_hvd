#!/usr/bin/env python3
"""Evaluate saved HateMM outputs without treating unreadable verdicts as nonhate.

The default recovered mode reparses final segment judgments and, when needed,
their fallback votes before applying the original video OR rule. This is an
offline reconstruction from saved answers, not a rerun of the reasoning model.
Use --prediction-source stored to evaluate the original saved video labels.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent

# datasplit 文件和结果目录都与本脚本放在同一个目录下
RESULT_DIR_NAME = "HateMM_LLaVA_1_5_7B_K=10_S=5"
DEFAULT_RESULT_DIR = SCRIPT_DIR / RESULT_DIR_NAME
DEFAULT_SPLIT_JSON = SCRIPT_DIR / "HateMM_datasplit_5folds.json"
METRIC_NAMES = ("acc", "macrof1", "macroprecision", "macrorecall")
LABELS = (0, 1)


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I)
    # An unfinished thinking block is not a final answer.
    return "" if re.search(r"<think>", text, flags=re.I) else text.strip()


def repair_markdown_escapes(text: str) -> str:
    # LLaVA sometimes writes Markdown escapes inside JSON keys, e.g. verdict\_label.
    # Preserve valid JSON escapes, including escaped backslashes and quotes.
    return re.sub(r"(?<!\\)\\([_*`])", r"\1", text)


def safe_json_loads(text: Any, repair_escapes: bool = True) -> Optional[Any]:
    if not isinstance(text, str):
        return None

    value = strip_thinking(text)
    if not value:
        return None

    candidates = [value]
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.I)
    if fence:
        candidates.append(fence.group(1).strip())

    start_candidates = [idx for idx in (value.find("{"), value.find("[")) if idx >= 0]
    end_candidates = [idx for idx in (value.rfind("}"), value.rfind("]")) if idx >= 0]
    if start_candidates and end_candidates:
        start = min(start_candidates)
        end = max(end_candidates)
        if end > start:
            candidates.append(value[start : end + 1])

    if repair_escapes:
        candidates += [repair_markdown_escapes(candidate) for candidate in candidates]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue

    return None


def nested_get(data: Any, keys: Sequence[str]) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def normalize_label(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int) and value in LABELS:
        return value

    if isinstance(value, float) and value.is_integer() and int(value) in LABELS:
        return int(value)

    if isinstance(value, str):
        cleaned = value.strip().lower()
        cleaned = cleaned.strip("`'\".,:;()[]{}")
        normalized = re.sub(r"[\\\s_-]+", "", cleaned)

        if normalized in {"0", "nonhate", "nonoffensive", "negative", "nothate", "nothateful", "benign"}:
            return 0
        if normalized in {"1", "hate", "hateful", "offensive", "hateoffensive", "positive"}:
            return 1

    return None


def normalize_verdict(value: Any) -> Optional[str]:
    label = normalize_label(value)
    if label is not None:
        return "hate" if label else "nonhate"
    if isinstance(value, str) and value.strip().lower() in {"uncertain", "unsure", "unknown"}:
        return "uncertain"
    return None


def verdict_from_fields(fields: Sequence[Tuple[str, Any]]) -> Optional[str]:
    values = [
        normalize_verdict(value)
        for key, value in fields
        if re.sub(r"[\\\s_-]+", "", str(key).lower()) in {"verdictlabel", "label"}
    ]
    # Do not select one of two contradictory labels or accept schema placeholders.
    return values[0] if values and values[0] is not None and len(set(values)) == 1 else None


def partial_json_fields(text: str) -> List[Tuple[str, Any]]:
    """Read complete top-level fields before a truncated value, without guessing it."""
    start = text.find("{")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    fields = []
    position = start + 1
    while position < len(text):
        position += len(text[position:]) - len(text[position:].lstrip())
        try:
            key, position = decoder.raw_decode(text, position)
            if not isinstance(key, str):
                break
            position += len(text[position:]) - len(text[position:].lstrip())
            if position >= len(text) or text[position] != ":":
                break
            position += 1
            position += len(text[position:]) - len(text[position:].lstrip())
            value, position = decoder.raw_decode(text, position)
        except (ValueError, TypeError):
            break
        position += len(text[position:]) - len(text[position:].lstrip())
        if position < len(text) and text[position] not in ",}":
            break
        fields.append((key, value))
        if position >= len(text) or text[position] != ",":
            # Multiple answer objects are ambiguous, even if the first is valid.
            if position < len(text) and "{" in text[position + 1:]:
                return []
            break
        position += 1
    return fields


def parse_raw_verdict(raw: Any) -> Tuple[Optional[str], str]:
    """Only use explicit final-answer fields; never search for the word 'hate'."""
    if not isinstance(raw, str) or not strip_thinking(raw):
        return None, "unreadable"
    text = strip_thinking(raw)
    fields = partial_json_fields(repair_markdown_escapes(text))
    for repair in (False, True):
        parsed = safe_json_loads(text, repair_escapes=repair)
        if isinstance(parsed, dict):
            return verdict_from_fields(fields), "repaired_json" if repair else "json"
        if isinstance(parsed, (str, int, float, bool)):
            verdict = normalize_verdict(parsed)
            return verdict, "plain_label" if verdict is not None else "unreadable"
    if fields:
        verdict = verdict_from_fields(fields)
        return verdict, "partial_json" if verdict is not None else "unreadable"
    if text.startswith(('"', "{", "[")):
        return None, "unreadable"
    verdict = normalize_verdict(text)
    return verdict, "plain_label" if verdict is not None else "unreadable"


def count_diagnostic(diagnostics: Optional[Dict[str, Any]], key: str) -> None:
    if diagnostics is not None:
        counts = diagnostics.setdefault("counts", {})
        counts[key] = counts.get(key, 0) + 1


def recover_record_verdict(record: Any, diagnostics: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(record, dict):
        return None
    if "raw_model_output" not in record:
        return normalize_verdict(record.get("verdict_label"))
    verdict, source = parse_raw_verdict(record["raw_model_output"])
    count_diagnostic(diagnostics, "raw_" + source)
    if verdict is None:
        count_diagnostic(diagnostics, "unresolved_raw_verdicts")
    elif verdict == "uncertain":
        count_diagnostic(diagnostics, "explicit_uncertain_verdicts")
    elif normalize_label(verdict) != normalize_label(record.get("verdict_label")):
        count_diagnostic(diagnostics, "raw_vs_stored_verdict_changes")
    # A present but empty/unreadable output must not inherit a default nonhate.
    return verdict


def recover_segment_label(segment: Any, diagnostics: Optional[Dict[str, Any]] = None) -> Optional[int]:
    if not isinstance(segment, dict):
        return None
    steps = segment.get("context_steps") or []
    if not isinstance(steps, list):
        return None
    verdict = (
        recover_record_verdict(steps[-1], diagnostics)
        if steps
        else normalize_verdict(segment.get("final_label_before_fallback"))
    )
    label = normalize_label(verdict)
    if label is not None:
        return label

    votes = segment.get("fallback_votes") or []
    if not isinstance(votes, list):
        return None
    if votes:
        labels = [normalize_label(recover_record_verdict(vote, diagnostics)) for vote in votes]
        hate_votes, nonhate_votes = labels.count(1), labels.count(0)
        # Failed votes remain unknown; they neither count as nonhate nor shrink
        # the denominator so that a single readable vote becomes a majority.
        if hate_votes > len(votes) / 2:
            return 1
        if nonhate_votes > len(votes) / 2:
            return 0
        if None not in labels and hate_votes == nonhate_votes:
            return 1  # Original pipeline's tie rule, only when all votes are known.
        return None

    if not steps:
        # Compatibility with older results that have no per-stage raw outputs.
        return normalize_label(segment.get("final_label_after_fallback"))
    return None


def recover_video_label(data: Dict[str, Any], diagnostics: Optional[Dict[str, Any]] = None) -> Optional[int]:
    segments = data["segment_results"]
    labels = [recover_segment_label(segment, diagnostics) for segment in segments]
    unresolved_ids = [
        segment.get("evidence_segment_id", segment.get("merged_segment_id", index))
        if isinstance(segment, dict) else index
        for index, (segment, label) in enumerate(zip(segments, labels))
        if label is None
    ]
    if unresolved_ids and diagnostics is not None:
        diagnostics.setdefault("videos_with_unresolved_segments", []).append(
            {"video_id": data.get("video_id"), "segment_ids": unresolved_ids}
        )

    title = data.get("title_reasoning")
    title_label = 0
    if isinstance(title, dict) and title.get("title_present", True):
        title_label = normalize_label(recover_record_verdict(title, diagnostics))
    elif str(data.get("video_title") or "").strip() and title is None:
        title_label = None

    # Preserve the original any-hate-segment OR hateful-title aggregation.
    if 1 in labels or title_label == 1:
        return 1
    if (
        unresolved_ids or title_label is None or not segments
        or data.get("unprocessed_segment_ids")
        or data.get("localization_complete") is False
    ):
        return None
    return 0


def extract_label_from_data(
    data: Any,
    prediction_source: str = "recovered",
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    if prediction_source == "recovered" and isinstance(data, dict) and isinstance(data.get("segment_results"), list):
        return recover_video_label(data, diagnostics)

    label_paths = (
        ("final_video_label",),
        ("summary", "final_evaluation", "final_decision", "label"),
        ("multi_turn_results", "turn_4", "parsed_response", "final_decision", "label"),
        ("parsed_response", "final_decision", "label"),
        ("final_decision", "label"),
        ("verdict_label",),
        ("label",),
    )

    for path in label_paths:
        label = normalize_label(nested_get(data, path))
        if label is not None:
            return label

    raw_paths = (
        ("multi_turn_results", "turn_4", "raw_response"),
        ("raw_response",),
    )
    for path in raw_paths:
        raw_response = nested_get(data, path)
        parsed = safe_json_loads(raw_response)
        if parsed is not None:
            label = extract_label_from_data(parsed, prediction_source)
            if label is not None:
                return label

        verdict, _ = parse_raw_verdict(raw_response)
        label = normalize_label(verdict)
        if label is not None:
            return label

    return None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_video_prediction(
    video_dir: Path, prediction_source: str, diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[int], Optional[int]]:
    for name in ("video_result.json", "mars_result.json", "mars_step_4_final_synthesis_and_evaluation.json"):
        path = video_dir / name
        if not path.is_file():
            continue
        try:
            data = read_json(path)
        except (OSError, ValueError) as exc:
            print(f"Warning: failed to read {path}: {exc}")
            continue
        video_id = str(data.get("video_id") or video_dir.name) if isinstance(data, dict) else video_dir.name
        stored = extract_label_from_data(data, "stored")
        label = extract_label_from_data(data, prediction_source, diagnostics)
        if label is not None or (isinstance(data, dict) and isinstance(data.get("segment_results"), list)):
            # An unresolved modern result must not be replaced by a stale legacy file.
            return video_id, label, stored
    return video_dir.name, None, None


def extract_label_for_video_dir(video_dir: Path, prediction_source: str = "recovered") -> Optional[int]:
    return read_video_prediction(video_dir, prediction_source)[1]


def load_predictions(
    result_dir: Path,
    prediction_source: str = "recovered",
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, int], List[str]]:
    if prediction_source not in {"recovered", "stored"}:
        raise ValueError(f"Unknown prediction source: {prediction_source}")
    predictions: Dict[str, int] = {}
    skipped: List[str] = []

    for video_dir in sorted(path for path in result_dir.iterdir() if path.is_dir()):
        result_file = video_dir / "video_result.json"
        legacy_result_file = video_dir / "mars_result.json"
        step4_file = video_dir / "mars_step_4_final_synthesis_and_evaluation.json"
        if not any(path.is_file() for path in (result_file, legacy_result_file, step4_file)):
            continue

        video_id, label, stored = read_video_prediction(video_dir, prediction_source, diagnostics)

        if label is None:
            skipped.append(video_dir.name)
            if diagnostics is not None:
                diagnostics.setdefault("unresolved_predictions", []).append(video_id)
            continue

        if diagnostics is not None and stored is not None and stored != label:
            diagnostics.setdefault("changed_predictions", []).append(
                {"video_id": video_id, "stored_label": stored, "recovered_label": label}
            )
        predictions[video_id] = label

    return predictions, skipped


def unpack_split_items(split_items: Any, fold_name: str) -> List[Tuple[str, int]]:
    if isinstance(split_items, list) and (
        not split_items or isinstance(split_items[0], dict)
    ):
        output = []
        for item in split_items:
            video_id = item.get("video_id") or item.get("id")
            label = normalize_label(item.get("label"))
            if video_id is None or label is None:
                raise ValueError(f"Missing video_id/label in {fold_name}: {item}")
            output.append((str(video_id), label))
        return output

    if isinstance(split_items, (list, tuple)) and len(split_items) == 2:
        ids, labels = split_items
        return [(str(video_id), int(label)) for video_id, label in zip(ids, labels)]

    if isinstance(split_items, dict):
        id_keys = ("ids", "video_ids", "video_id", "data", "x")
        label_keys = ("labels", "label", "y", "targets", "target")
        ids = next((split_items[key] for key in id_keys if key in split_items), None)
        labels = next((split_items[key] for key in label_keys if key in split_items), None)
        if ids is not None and labels is not None:
            return [(str(video_id), int(label)) for video_id, label in zip(ids, labels)]

    raise ValueError(f"Unsupported split format in {fold_name}: {type(split_items).__name__}")


def load_test_folds(split_json: Path) -> List[Dict[str, Any]]:
    split_data = read_json(split_json)
    raw_folds = split_data.get("folds")
    if not isinstance(raw_folds, list):
        raise ValueError(f"Split JSON missing list field 'folds': {split_json}")

    folds = []
    for index, raw_fold in enumerate(raw_folds):
        fold_id = raw_fold.get("fold", index)
        fold_name = f"fold_{fold_id}"
        folds.append(
            {
                "fold": fold_id,
                "test": unpack_split_items(raw_fold["test"], fold_name),
            }
        )

    return folds


def lookup_prediction(video_id: str, predictions: Dict[str, int]) -> Optional[int]:
    if video_id in predictions:
        return predictions[video_id]

    prefixed = [
        predictions[key]
        for key in (f"hate_{video_id}", f"non_hate_{video_id}", f"nonhate_{video_id}")
        if key in predictions
    ]
    if len(prefixed) == 1:
        return prefixed[0]

    return None


def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("No matched samples for metric calculation")

    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    acc = correct / len(y_true)

    precisions = []
    recalls = []
    f1s = []
    for label in LABELS:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": acc,
        "macrof1": sum(f1s) / len(f1s),
        "macroprecision": sum(precisions) / len(precisions),
        "macrorecall": sum(recalls) / len(recalls),
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def evaluate(
    result_dir: Path, split_json: Path, strict_missing: bool,
    prediction_source: str = "recovered",
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "counts": {}, "changed_predictions": [], "unresolved_predictions": [],
        "videos_with_unresolved_segments": [],
    }
    predictions, skipped = load_predictions(result_dir, prediction_source, diagnostics)
    folds = load_test_folds(split_json)

    print(f"Result dir: {result_dir}")
    print(f"Split json: {split_json}")
    print(f"Prediction source: {prediction_source}")
    print(f"Loaded predictions: {len(predictions)}")
    print(f"Prediction counts: nonhate={list(predictions.values()).count(0)}, hate={list(predictions.values()).count(1)}")
    if prediction_source == "recovered":
        print(f"Changed from stored video labels: {len(diagnostics['changed_predictions'])}")
        counts = diagnostics["counts"]
        print(
            "Raw verdict recovery: "
            f"JSON={counts.get('raw_json', 0)}, "
            f"repaired JSON={counts.get('raw_repaired_json', 0)}, "
            f"complete labels in truncated JSON={counts.get('raw_partial_json', 0)}, "
            f"unresolved={counts.get('unresolved_raw_verdicts', 0)}"
        )
    if skipped:
        print(f"Skipped video dirs without valid prediction: {len(skipped)}")
        print("Unresolved labels are excluded, not counted as nonhate; check test coverage below.")
    print()

    fold_results = []
    pooled_true: List[int] = []
    pooled_pred: List[int] = []
    missing_test_ids: List[str] = []
    missing_by_label = {0: 0, 1: 0}
    for fold in folds:
        y_true = []
        y_pred = []
        missing = []

        for video_id, true_label in fold["test"]:
            pred_label = lookup_prediction(video_id, predictions)
            if pred_label is None:
                missing.append(video_id)
                missing_by_label[true_label] += 1
                continue
            y_true.append(true_label)
            y_pred.append(pred_label)

        missing_test_ids.extend(missing)
        pooled_true.extend(y_true)
        pooled_pred.extend(y_pred)

        if strict_missing and missing:
            raise RuntimeError(
                f"Fold {fold['fold']} has {len(missing)} missing predictions; "
                f"first missing ids: {', '.join(missing[:10])}"
            )

        if not y_true:
            print(f"Fold {fold['fold']}: no matched samples, skipped")
            continue

        metrics = calculate_metrics(y_true, y_pred)
        result = {
            "fold": fold["fold"],
            "n_matched": len(y_true),
            "n_test": len(fold["test"]),
            "n_missing": len(missing),
            "missing_video_ids": missing,
            **metrics,
        }
        fold_results.append(result)

        print(
            f"Fold {fold['fold']}: "
            f"n={result['n_matched']}/{result['n_test']}, "
            f"acc={result['acc']:.4f}, "
            f"macrof1={result['macrof1']:.4f}, "
            f"macroprecision={result['macroprecision']:.4f}, "
            f"macrorecall={result['macrorecall']:.4f}"
        )

    if not fold_results:
        raise RuntimeError("No folds had matched predictions.")

    print()
    print(f"{len(fold_results)}-fold mean ± std (matched samples only)")
    aggregate = {}
    for metric_name in METRIC_NAMES:
        values = [fold_result[metric_name] for fold_result in fold_results]
        metric_mean = mean(values)
        metric_variance = variance(values)
        metric_std = metric_variance ** 0.5
        aggregate[metric_name] = {
            "mean": metric_mean,
            "std": metric_std,
            "variance": metric_variance,
        }
        print(f"{metric_name}: {metric_mean:.4f} ± {metric_std:.4f}")

    total_matched = sum(fold_result["n_matched"] for fold_result in fold_results)
    total_test = sum(len(fold["test"]) for fold in folds)
    print(f"Total matched samples: {total_matched}/{total_test}")
    print(f"Missing/unresolved test samples: nonhate={missing_by_label[0]}, hate={missing_by_label[1]}")
    confusion = [[0, 0], [0, 0]]
    for true, pred in zip(pooled_true, pooled_pred):
        confusion[true][pred] += 1
    hate_total = sum(confusion[1])
    hate_recall = confusion[1][1] / hate_total if hate_total else 0.0
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {confusion}")
    print(f"Hate recall on matched samples: {hate_recall:.4f}")
    if prediction_source == "recovered":
        print("Recovery uses saved outputs only; it cannot restore missing context or rerun model reasoning.")

    return {
        "result_dir": str(result_dir),
        "split_json": str(split_json),
        "prediction_source": prediction_source,
        "loaded_predictions": len(predictions),
        "skipped_prediction_dirs": skipped,
        "fold_results": fold_results,
        "aggregate": aggregate,
        "total_matched": total_matched,
        "total_test": total_test,
        "missing_test_video_ids": missing_test_ids,
        "missing_test_by_label": missing_by_label,
        "confusion_matrix": confusion,
        "hate_recall": hate_recall,
        "pooled_metrics": calculate_metrics(pooled_true, pooled_pred),
        "recovery_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate HateMM results, recovering explicit raw verdicts before video aggregation."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing one subdirectory per video with video_result.json.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=DEFAULT_SPLIT_JSON,
        help="Path to HateMM_datasplit_5folds.json.",
    )
    parser.add_argument(
        "--prediction-source",
        choices=("recovered", "stored"),
        default="recovered",
        help="recovered: reparse final raw verdicts and fallback votes (default); stored: use saved final labels.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any test item is missing a prediction.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save detailed fold metrics as JSON.",
    )
    args = parser.parse_args()

    results = evaluate(
        result_dir=args.result_dir.resolve(),
        split_json=args.split_json.resolve(),
        strict_missing=args.strict_missing,
        prediction_source=args.prediction_source,
    )

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {args.save_json}")


if __name__ == "__main__":
    main()
