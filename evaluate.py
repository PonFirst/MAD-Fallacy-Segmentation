import json
import os
import re
import csv
from collections import defaultdict


# =============================================================================
# Span Utilities
# =============================================================================

def find_span_in_dialogue(span_text, dialogue):
    """
    Find the character offsets of span_text inside dialogue.
    Normalizes whitespace on both sides before searching.
    Returns (start, end) or (-1, -1) if not found.
    """
    span_text = re.sub(r'\s+', ' ', str(span_text)).strip()
    dialogue_norm = re.sub(r'\s+', ' ', str(dialogue))
    idx = dialogue_norm.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)
    return -1, -1


def resolve_predictions(preds, dialogue):
    """
    Convert raw prediction dicts (with span_text) into resolved dicts
    that include character offsets. Skips predictions that errored.
    Returns a list of resolved prediction dicts.
    """
    resolved = []
    for p in preds:
        if "error" in p:
            continue
        start, end = find_span_in_dialogue(p.get("span_text", ""), dialogue)
        resolved.append({
            "fallacy":    p.get("suspected_fallacy", ""),
            "span_start": start,
            "span_end":   end,
            "span_text":  p.get("span_text", ""),
            "original":   p,
        })
    return resolved


def span_iou(pred_start, pred_end, gold_start, gold_end):
    """
    Intersection over Union between two character spans.
    Used for lenient matching (IoU >= threshold).
    """
    if pred_end <= pred_start or gold_end <= gold_start:
        return 0.0
    overlap = max(0, min(pred_end, gold_end) - max(pred_start, gold_start))
    union   = max(pred_end, gold_end) - min(pred_start, gold_start)
    return overlap / union if union > 0 else 0.0


def span_overlap(pred_start, pred_end, gold_start, gold_end):
    """
    Returns True if two spans share at least one character.
    Used for strict matching (any overlap counts).
    """
    if pred_end <= pred_start or gold_end <= gold_start:
        return False
    return min(pred_end, gold_end) > max(pred_start, gold_start)


# =============================================================================
# Greedy One-to-One Matching
# =============================================================================

def greedy_match(predictions, gold_annotations, dialogue, iou_threshold=0.0):
    """
    Match predictions to gold annotations using greedy one-to-one matching.

    Each prediction is matched to at most one gold annotation, and each gold
    annotation is matched to at most one prediction. This prevents a single
    correct prediction from claiming credit for multiple gold annotations.

    Args:
        predictions:      raw prediction list from the results jsonl file
        gold_annotations: gold annotation list with fallacy, span_start, span_end
        dialogue:         full dialogue text for resolving predicted spans
        iou_threshold:    0.0 = strict (any character overlap)
                          0.5 = lenient (IoU must be >= 0.5)

    Returns:
        tp:                  number of true positives
        fp:                  number of false positives
        fn:                  number of false negatives
        matched_pairs:       list of (resolved_pred, gold_ann) matched tuples
        matched_pred_indices: set of indices into resolved_preds that were matched
        resolved_preds:      the resolved prediction list (reused for FP tracking)
    """
    resolved_preds = resolve_predictions(predictions, dialogue)

    # build all valid candidate pairs (label match + span threshold)
    scored_pairs = []
    for pi, pred in enumerate(resolved_preds):
        for gi, gold in enumerate(gold_annotations):

            # labels must match exactly (case-insensitive)
            label_match = (
                pred["fallacy"].strip().lower() ==
                gold["fallacy"].strip().lower()
            )
            if not label_match:
                continue

            # skip predictions whose span text was not found in the dialogue
            if pred["span_start"] == -1:
                continue

            # score the pair based on the chosen threshold
            if iou_threshold == 0.0:
                hit   = span_overlap(pred["span_start"], pred["span_end"],
                                     gold["span_start"], gold["span_end"])
                score = 1.0 if hit else 0.0
            else:
                score = span_iou(pred["span_start"], pred["span_end"],
                                 gold["span_start"], gold["span_end"])
                score = score if score >= iou_threshold else 0.0

            if score > 0:
                scored_pairs.append((score, pi, gi))

    # greedy assignment: take the highest-scoring unmatched pair first
    scored_pairs.sort(key=lambda x: -x[0])

    matched_pred_indices = set()
    matched_gold_indices = set()
    matched_pairs        = []

    for score, pi, gi in scored_pairs:
        if pi in matched_pred_indices or gi in matched_gold_indices:
            continue
        matched_pred_indices.add(pi)
        matched_gold_indices.add(gi)
        matched_pairs.append((resolved_preds[pi], gold_annotations[gi]))

    tp = len(matched_pairs)
    fp = len(resolved_preds) - tp
    fn = len(gold_annotations) - tp

    return tp, fp, fn, matched_pairs, matched_pred_indices, resolved_preds


# =============================================================================
# File Evaluation
# =============================================================================

def evaluate_file(jsonl_path, iou_threshold=0.0, label=""):
    """
    Load a results jsonl file and compute overall and per-class metrics.

    Each line in the file must contain:
        dialogue          - the full input text
        gold_annotations  - list of {fallacy, span_start, span_end, snippet}
        predictions       - list of {suspected_fallacy, span_text, reasoning}

    Returns a dict with precision, recall, f1, weighted_f1, tp, fp, fn.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    error_count = 0

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    with open(jsonl_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for record in records:
        dialogue = record["dialogue"]
        gold     = record["gold_annotations"]
        preds    = record["predictions"]

        # count dialogues where the API completely failed
        if preds and "error" in preds[0]:
            error_count += 1

        tp, fp, fn, matched, matched_pred_indices, resolved_preds = greedy_match(
            preds, gold, dialogue, iou_threshold
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # per-class TP — based on which gold annotations were matched
        matched_gold_ids = {id(g) for _, g in matched}

        for _, gold_ann in matched:
            per_class_tp[gold_ann["fallacy"]] += 1

        # per-class FN — gold annotations that had no matching prediction
        for gold_ann in gold:
            if id(gold_ann) not in matched_gold_ids:
                per_class_fn[gold_ann["fallacy"]] += 1

        # per-class FP — predictions that did not match any gold annotation
        # uses index-based tracking to avoid span_text collision bugs
        for pi, pred in enumerate(resolved_preds):
            if pi not in matched_pred_indices:
                per_class_fp[pred["fallacy"] or "unknown"] += 1

    # overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # weighted F1 — weights per-class F1 by gold annotation count
    # more meaningful than macro F1 when classes are imbalanced
    all_classes = sorted(set(list(per_class_tp.keys()) + list(per_class_fn.keys())))
    total_gold_count = sum(per_class_tp[c] + per_class_fn[c] for c in all_classes)

    weighted_f1 = 0.0
    for cls in all_classes:
        gold_count = per_class_tp[cls] + per_class_fn[cls]
        tp_c = per_class_tp[cls]
        fp_c = per_class_fp[cls]
        fn_c = per_class_fn[cls]
        p_c  = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r_c  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f_c  = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
        weighted_f1 += f_c * (gold_count / total_gold_count)

    # print results
    threshold_label = ("strict (any overlap)"
                       if iou_threshold == 0.0
                       else f"lenient (IoU >= {iou_threshold})")

    print(f"\n{'='*60}")
    print(f"System:    {label or jsonl_path}")
    print(f"Threshold: {threshold_label}")
    print(f"Dialogues: {len(records)} evaluated, {error_count} with API errors")
    print(f"{'='*60}")
    print(f"TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"Macro F1:     {f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")

    print(f"\n--- Per-class breakdown ---")
    print(f"  {'Class':<25} {'P':>6} {'R':>6} {'F1':>6}  counts")
    print(f"  {'-'*65}")
    for cls in all_classes:
        tp_c = per_class_tp[cls]
        fp_c = per_class_fp[cls]
        fn_c = per_class_fn[cls]
        p_c  = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r_c  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f_c  = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
        print(f"  {cls:<25} {p_c:>6.3f} {r_c:>6.3f} {f_c:>6.3f}  "
              f"(TP={tp_c} FP={fp_c} FN={fn_c})")

    return {
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
        "weighted_f1":  weighted_f1,
        "tp":           total_tp,
        "fp":           total_fp,
        "fn":           total_fn,
        "errors":       error_count,
        "dialogues":    len(records),
    }


# =============================================================================
# Label-Only Accuracy (Span Ignored)
# =============================================================================

def label_only_accuracy(jsonl_path, label=""):
    """
    Compute classification accuracy ignoring span boundaries entirely.

    For each dialogue, checks whether each predicted fallacy label matches
    any remaining gold label in that dialogue. Uses greedy label matching
    so a single gold label cannot be claimed by multiple predictions.

    This metric provides a comparison point against prior work (Goffredo et al.
    2023) that reports token-level classification F1 without span evaluation.
    """
    total_pred  = 0
    total_hit   = 0
    total_gold  = 0

    per_class_hit  = defaultdict(int)
    per_class_gold = defaultdict(int)

    with open(jsonl_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for record in records:
        gold  = record["gold_annotations"]
        preds = record["predictions"]

        # count available gold labels for this dialogue
        gold_label_counts = defaultdict(int)
        for ann in gold:
            gold_label_counts[ann["fallacy"]] += 1
            per_class_gold[ann["fallacy"]]    += 1
        total_gold += len(gold)

        # greedily match predictions to gold labels
        remaining = dict(gold_label_counts)
        for p in preds:
            if "error" in p:
                continue
            lbl = p.get("suspected_fallacy", "")
            total_pred += 1
            if remaining.get(lbl, 0) > 0:
                total_hit           += 1
                remaining[lbl]      -= 1
                per_class_hit[lbl]  += 1

    precision = total_hit / total_pred if total_pred > 0 else 0.0
    recall    = total_hit / total_gold if total_gold > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n--- Label-only accuracy: {label or jsonl_path} ---")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"  {'Class':<25} {'Recall':>8}  (hit / gold)")
    print(f"  {'-'*45}")
    for cls in sorted(per_class_gold.keys()):
        r = per_class_hit[cls] / per_class_gold[cls] if per_class_gold[cls] > 0 else 0.0
        print(f"  {cls:<25} {r:>8.3f}  ({per_class_hit[cls]} / {per_class_gold[cls]})")

    return {"precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# System Comparison Table
# =============================================================================

def compare_systems(systems, iou_threshold=0.0):
    """
    Evaluate multiple result files and print a side-by-side comparison table.

    Args:
        systems: list of (label, jsonl_path) tuples
        iou_threshold: 0.0 for strict, 0.5 for lenient
    """
    results = []
    for label, path in systems:
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        r = evaluate_file(path, iou_threshold, label)
        results.append((label, r))

    threshold_label = "strict" if iou_threshold == 0.0 else f"IoU>={iou_threshold}"

    print(f"\n{'='*60}")
    print(f"COMPARISON TABLE — {threshold_label}")
    print(f"{'='*60}")
    print(f"{'System':<30} {'P':>6} {'R':>6} {'MacroF1':>9} {'WtdF1':>7}")
    print(f"{'-'*57}")
    for label, r in results:
        print(f"{label:<30} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1']:>9.3f} {r['weighted_f1']:>7.3f}")


# =============================================================================
# Save Metrics to CSV (for LaTeX tables)
# =============================================================================

def save_metrics_csv(systems, output_path="results/metrics_summary.csv"):
    """
    Run evaluation on all systems and save a flat CSV for easy import into
    LaTeX tables or spreadsheets.
    """
    rows = []
    for label, path in systems:
        if not os.path.exists(path):
            continue
        for threshold, threshold_label in [(0.0, "strict"), (0.5, "lenient")]:
            r = evaluate_file(path, threshold, label)
            rows.append({
                "system":       label,
                "threshold":    threshold_label,
                "precision":    round(r["precision"],   4),
                "recall":       round(r["recall"],      4),
                "macro_f1":     round(r["f1"],          4),
                "weighted_f1":  round(r["weighted_f1"], 4),
                "tp":           r["tp"],
                "fp":           r["fp"],
                "fn":           r["fn"],
                "errors":       r["errors"],
                "dialogues":    r["dialogues"],
            })

    if not rows:
        print("No results to save.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetrics saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    systems = [
        ("Zero-shot",    "results/zero_shot_full_v2.jsonl"),
        # ("Few-shot",     "results/few_shot_full.jsonl"),
        # ("Generic MAD",  "results/generic_mad_full.jsonl"),
        # ("MAD no debate","results/mad_no_debate_full.jsonl"),
        # ("MAD full",     "results/mad_full.jsonl"),
    ]

    print("\n=== STRICT EVALUATION (any span overlap + label match) ===")
    compare_systems(systems, iou_threshold=0.0)

    print("\n=== LENIENT EVALUATION (IoU >= 0.5 + label match) ===")
    compare_systems(systems, iou_threshold=0.5)

    print("\n=== LABEL-ONLY ACCURACY (span ignored) ===")
    for label, path in systems:
        if os.path.exists(path):
            label_only_accuracy(path, label)

    save_metrics_csv(systems)