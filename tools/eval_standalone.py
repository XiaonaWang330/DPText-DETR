#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone CTW1500 evaluator. Replicates text_eval_script_det.py logic using Shapely.
No adet/Polygon3 required. Uses COCO-format GT JSON (datasets/ctw1500/test_poly.json).

Usage:
    python tools/eval_standalone.py --json ensemble_fused.json
    python tools/eval_standalone.py --json output/r_50_poly/ctw1500/cura_a100/text_results.json
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon


def poly_to_shapely(poly):
    """Convert polygon (flat or nested) to Shapely Polygon."""
    if not poly or len(poly) < 6:
        return None
    if isinstance(poly[0], (list, tuple)):
        pts = [(float(p[0]), float(p[1])) for p in poly]
    else:
        pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
    if len(pts) < 3:
        return None
    try:
        p = Polygon(pts)
        if not p.is_valid:
            p = p.buffer(0)
        return p
    except Exception:
        return None


def compute_iou(pred_poly, gt_poly):
    """IoU between two Shapely polygons."""
    if pred_poly is None or gt_poly is None:
        return 0.0
    if pred_poly.is_empty or gt_poly.is_empty:
        return 0.0
    try:
        inter = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def evaluate(predictions, gt_annotations, iou_thresh=0.5, area_precision=0.5):
    """
    Replicates the official CTW1500 evaluation logic.

    Args:
        predictions: list of dicts with 'image_id', 'polys', 'score'
        gt_annotations: list of dicts with 'image_id', 'polys', 'id'
    Returns:
        (precision, recall, f1)
    """
    # Index GT by image_id
    gt_by_image = defaultdict(list)
    for ann in gt_annotations:
        gt_by_image[ann['image_id']].append(ann)

    # Index predictions by image_id
    pred_by_image = defaultdict(list)
    for i, pred in enumerate(predictions):
        pred_by_image[pred['image_id']].append((i, pred))

    total_matched = 0
    total_gt_care = 0
    total_det_care = 0

    for img_id in pred_by_image:
        pred_list = pred_by_image[img_id]
        gt_list = gt_by_image.get(img_id, [])

        # Convert polygons
        gt_polys = []
        gt_valid = []
        for gt in gt_list:
            p = poly_to_shapely(gt['polys'])
            gt_polys.append(p)
            if p is not None:
                gt_valid.append(p)
            else:
                gt_valid.append(None)

        det_polys = []
        det_scores = []
        det_valid = []
        for pi, (pred_idx, pred) in enumerate(pred_list):
            p = poly_to_shapely(pred['polys'])
            det_polys.append(p)
            det_scores.append(pred.get('score', 1.0))
            if p is not None:
                det_valid.append(p)
            else:
                det_valid.append(None)

        # Identify "don't care" GT (for CTW1500, all GT are care)
        gt_dont_care = set()

        # Identify "don't care" detections (those overlapping don't-care GT)
        det_dont_care = set()
        for didx in range(len(det_polys)):
            if det_polys[didx] is None or det_polys[didx].is_empty:
                det_dont_care.add(didx)
                continue
            for gidx in gt_dont_care:
                if gt_polys[gidx] is None:
                    continue
                inter = det_polys[didx].intersection(gt_polys[gidx]).area
                pd_area = det_polys[didx].area
                if pd_area > 0 and inter / pd_area > area_precision:
                    det_dont_care.add(didx)
                    break

        # Compute IoU matrix
        num_gt = len(gt_polys)
        num_det = len(det_polys)
        iou_mat = np.zeros((num_gt, num_det))

        for gi in range(num_gt):
            if gt_polys[gi] is None:
                continue
            for di in range(num_det):
                if det_polys[di] is None:
                    continue
                iou_mat[gi, di] = compute_iou(det_polys[di], gt_polys[gi])

        # Greedy matching
        gt_matched = np.zeros(num_gt, dtype=bool)
        det_matched = np.zeros(num_det, dtype=bool)

        # Sort all pairs by IoU descending
        pairs = []
        for gi in range(num_gt):
            for di in range(num_det):
                if iou_mat[gi, di] > iou_thresh:
                    pairs.append((iou_mat[gi, di], gi, di))
        pairs.sort(key=lambda x: -x[0])

        det_matched_count = 0
        for iou, gi, di in pairs:
            if not gt_matched[gi] and not det_matched[di]:
                if gi not in gt_dont_care and di not in det_dont_care:
                    gt_matched[gi] = True
                    det_matched[di] = True
                    det_matched_count += 1

        num_gt_care = num_gt - len(gt_dont_care)
        num_det_care = num_det - len(det_dont_care)

        total_matched += det_matched_count
        total_gt_care += num_gt_care
        total_det_care += num_det_care

    recall = total_matched / total_gt_care if total_gt_care > 0 else 1.0
    precision = total_matched / total_det_care if total_det_care > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, total_matched, total_gt_care, total_det_care


def main():
    parser = argparse.ArgumentParser(description="Standalone CTW1500 evaluation")
    parser.add_argument('--json', required=True, help='Path to text_results.json')
    parser.add_argument('--gt', default='datasets/ctw1500/test_poly.json',
                        help='Path to GT COCO JSON')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    args = parser.parse_args()

    # Load predictions
    with open(args.json, 'r') as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from {args.json}")

    # Load GT
    with open(args.gt, 'r') as f:
        gt_data = json.load(f)
    gt_annotations = gt_data['annotations']
    print(f"Loaded {len(gt_annotations)} GT annotations from {args.gt}")

    P, R, F1, matched, gt_care, det_care = evaluate(predictions, gt_annotations, iou_thresh=args.iou)

    print(f"\n{'='*55}")
    print(f"  matched={matched}  gt_care={gt_care}  det_care={det_care}")
    print(f"  Precision: {P:.4f}  |  Recall: {R:.4f}  |  F1: {F1:.4f}")
    print(f"{'='*55}")
    print(f"\n{P:.4f}\t{R:.4f}\t{F1:.4f}")


if __name__ == '__main__':
    main()
