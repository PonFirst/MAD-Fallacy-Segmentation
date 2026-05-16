import json
import os
import re
from collections import defaultdict


# ── span utilities ────────────────────────────────────────────────

def find_span_in_dialogue(span_text, dialogue):
    """Resolve predicted span text to character offsets in the dialogue."""
    span_text = re.sub(r'\s+', ' ', str(span_text)).strip()
    dialogue_norm = re.sub(r'\s+', ' ', str(dialogue))
    idx = dialogue_norm.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)
    return -1, -1


def span_iou(pred_start, pred_end, gold_start, gold_end):
    """Intersection over Union of two character spans."""
    if pred_end <= pred_start or gold_end <= gold_start:
        return 0.0
    overlap = max(0, min(pred_end, gold_end) - max(pred_start, gold_start))
    union = max(pred_end, gold_end) - min(pred_start, gold_start)
    return overlap / union if union > 0 else 0.0


def span_overlap(pred_start, pred_end, gold_start, gold_end):
    """Returns True if spans share at least 1 character."""
    if pred_end <= pred_start or gold_end <= gold_start:
        return False
    return min(pred_end, gold_end) > max(pred_start, gold_start)


# ── greedy one-to-one matching ────────────────────────────────────

def greedy_match(predictions, gold_annotations, dialogue, iou_threshold=0.0):
    """
    Match predictions to gold annotations using greedy one-to-one matching.

    Args:
        predictions:      list of dicts with keys suspected_fallacy, span_text
        gold_annotations: list of dicts with keys fallacy, span_start, span_end
        dialogue:         full dialogue text for resolving predicted spans
        iou_threshold:    0.0 for strict (any overlap), 0.5 for lenient

    Returns:
        tp: number of true positives
        fp: number of false positives
        fn: number of false negatives
        matched_pairs: list of (pred, gold) tuples that were matched
    """
    resolved_preds = []
    for p in predictions:
        if "error" in p:
            continue
        pred_start, pred_end = find_span_in_dialogue(
            p.get("span_text", ""), dialogue
        )
        resolved_preds.append({
            "fallacy": p.get("suspected_fallacy", ""),
            "span_start": pred_start,
            "span_end": pred_end,
            "span_text": p.get("span_text", ""),
            "original": p
        })

    scored_pairs = []
    for pi, pred in enumerate(resolved_preds):
        for gi, gold in enumerate(gold_annotations):
            label_match = (
                pred["fallacy"].strip().lower() ==
                gold["fallacy"].strip().lower()
            )
            if not label_match:
                continue
            if pred["span_start"] == -1:
                continue

            if iou_threshold == 0.0:
                hit = span_overlap(
                    pred["span_start"], pred["span_end"],
                    gold["span_start"], gold["span_end"]
                )
                score = 1.0 if hit else 0.0
            else:
                score = span_iou(
                    pred["span_start"], pred["span_end"],
                    gold["span_start"], gold["span_end"]
                )
                score = score if score >= iou_threshold else 0.0

            if score > 0:
                scored_pairs.append((score, pi, gi))

    scored_pairs.sort(key=lambda x: -x[0])

    matched_preds = set()
    matched_golds = set()
    matched_pairs = []

    for score, pi, gi in scored_pairs:
        if pi in matched_preds or gi in matched_golds:
            continue
        matched_preds.add(pi)
        matched_golds.add(gi)
        matched_pairs.append((resolved_preds[pi], gold_annotations[gi]))

    tp = len(matched_pairs)
    fp = len(resolved_preds) - tp
    fn = len(gold_annotations) - tp

    return tp, fp, fn, matched_pairs


# ── per-class tracking ────────────────────────────────────────────

def evaluate_file(jsonl_path, iou_threshold=0.0, label=""):
    """
    Evaluate a results .jsonl file.
    Each line must have: dialogue, gold_annotations, predictions
    Returns dict with overall and per-class metrics.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    with open(jsonl_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for record in records:
        dialogue = record["dialogue"]
        gold = record["gold_annotations"]
        preds = record["predictions"]

        tp, fp, fn, matched = greedy_match(
            preds, gold, dialogue, iou_threshold
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        matched_gold_ids = {id(g) for _, g in matched}

        for _, gold_ann in matched:
            per_class_tp[gold_ann["fallacy"]] += 1

        for gold_ann in gold:
            if id(gold_ann) not in matched_gold_ids:
                per_class_fn[gold_ann["fallacy"]] += 1

        matched_pred_spans = {p["span_text"] for p, _ in matched}
        for pred in preds:
            if "error" in pred:
                continue
            if pred.get("span_text", "") not in matched_pred_spans:
                per_class_fp[pred.get("suspected_fallacy", "unknown")] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    print(f"\n{'='*60}")
    print(f"System: {label or jsonl_path}")
    print(f"Threshold: {'strict (any overlap)' if iou_threshold == 0.0 else f'lenient (IoU >= {iou_threshold})'}")
    print(f"Dialogues evaluated: {len(records)}")
    print(f"{'='*60}")
    print(f"TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print(f"\n--- Per-class breakdown ---")
    all_classes = set(
        list(per_class_tp.keys()) +
        list(per_class_fn.keys())
    )
    for cls in sorted(all_classes):
        tp_c = per_class_tp[cls]
        fp_c = per_class_fp[cls]
        fn_c = per_class_fn[cls]
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        f_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0
        print(f"  {cls:<25} P={p_c:.3f}  R={r_c:.3f}  F1={f_c:.3f}  "
              f"(TP={tp_c} FP={fp_c} FN={fn_c})")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }


# ── compare multiple systems ──────────────────────────────────────

def compare_systems(systems, iou_threshold=0.0):
    """
    systems: list of (label, jsonl_path) tuples
    Prints a comparison table.
    """
    results = []
    for label, path in systems:
        if not os.path.exists(path):
            print(f"File not found: {path} — skipping")
            continue
        r = evaluate_file(path, iou_threshold, label)
        results.append((label, r))

    print(f"\n{'='*60}")
    print(f"COMPARISON TABLE — "
          f"{'strict' if iou_threshold == 0.0 else f'IoU>={iou_threshold}'}")
    print(f"{'='*60}")
    print(f"{'System':<30} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"{'-'*50}")
    for label, r in results:
        print(f"{label:<30} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")


# ── main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    systems = [
        ("Zero-shot (10 dialogues)", "results/zero_shot.jsonl"),
        ("Zero-shot (55 dialogues)", "results/zero_shot_full.jsonl"),
        # ("Few-shot baseline",        "results/few_shot.jsonl"),
        # ("Generic MAD",              "results/generic_mad.jsonl"),
        # ("MAD full system",          "results/mad_full.jsonl"),
    ]

    print("\n=== STRICT EVALUATION (any span overlap + label match) ===")
    compare_systems(systems, iou_threshold=0.0)

    print("\n=== LENIENT EVALUATION (IoU >= 0.5 + label match) ===")
    compare_systems(systems, iou_threshold=0.5)