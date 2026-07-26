#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a prediction JSON file using the official CTW1500 evaluation pipeline.
No model loading required — just evaluates from saved predictions.

Usage:
    # Evaluate a single json
    python tools/eval_json.py --json fused_results.json --gt datasets/evaluation/gt_ctw1500.zip

    # Evaluate with a specific output name
    python tools/eval_json.py --json output/ensemble/text_results.json --gt datasets/evaluation/gt_ctw1500.zip
"""

import os
import re
import json
import glob
import shutil
import zipfile
import argparse
import numpy as np
from shapely.geometry import Polygon, LinearRing
from adet.evaluation import text_eval_script_det


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def polygon_is_valid_and_ccw(poly_pts):
    """Check validity and reorder to CCW. Returns (is_valid, ccw_points)."""
    try:
        n = len(poly_pts)
        pts = [(int(poly_pts[j][0]), int(poly_pts[j][1])) for j in range(n)]
        pgt = Polygon(pts)
        if not pgt.is_valid:
            return False, pts
        pRing = LinearRing(pts)
        if pRing.is_ccw:
            pts.reverse()
        return True, pts
    except Exception:
        return False, None


def to_eval_format(json_path, temp_dir="temp_det_results"):
    """
    Convert text_results.json to per-image .txt files for official evaluation.
    Replicates TextDetEvaluator.to_eval_format + sort_detection.
    Returns path to det.zip.
    """
    data = load_json(json_path)
    print(f"Loaded {len(data)} predictions from {json_path}")

    # Phase 1: Write per-image .txt files
    dirn = os.path.abspath(temp_dir)
    os.makedirs(dirn, exist_ok=True)

    # Group by image_id
    by_image = {}
    for entry in data:
        img_id = entry['image_id']
        if img_id not in by_image:
            by_image[img_id] = []
        by_image[img_id].append(entry)

    total_written = 0
    total_skipped = 0
    for img_id, entries in by_image.items():
        filename = '{:07d}.txt'.format(int(img_id))
        out_path = os.path.join(dirn, filename)
        with open(out_path, 'w') as fout:
            for entry in entries:
                score = entry.get('score', 1.0)
                polys = entry.get('polys', [])

                if score <= 0.1:
                    total_skipped += 1
                    continue

                valid, pts = polygon_is_valid_and_ccw(polys)
                if not valid:
                    total_skipped += 1
                    continue

                # Write: x1,y1,x2,y2,...xN,yN,####
                outstr = ','.join(f'{int(p[0])},{int(p[1])}' for p in pts[:-1])
                outstr += f',{int(pts[-1][0])},{int(pts[-1][1])}'
                outstr += ',####\n'
                fout.write(outstr)
                total_written += 1

    print(f"  Wrote {total_written} valid detections, skipped {total_skipped}")

    # Phase 2: Create det.zip (sort_detection equivalent but simpler)
    # Actually the official eval expects zip, so let's create one
    det_zip_path = os.path.abspath("det.zip")
    with zipfile.ZipFile(det_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for fname in sorted(os.listdir(dirn)):
            fpath = os.path.join(dirn, fname)
            zipf.write(fpath, arcname=fname)

    # Cleanup temp
    shutil.rmtree(dirn, ignore_errors=True)

    print(f"  Created {det_zip_path}")
    return det_zip_path


def evaluate_predictions(json_path, gt_path):
    """
    Full evaluation pipeline: JSON → eval → {P, R, F1}.
    """
    result_path = to_eval_format(json_path)
    text_result = text_eval_script_det.text_eval_main_det(
        det_file=result_path, gt_file=gt_path
    )
    os.remove(result_path)

    # Parse result string: "DET_RESULT: precision: 0.9013, recall: 0.8560, hmean: 0.8781"
    result_str = text_result['det_method']
    match = re.match(r"DET_RESULT: precision: ([\d.]+), recall: ([\d.]+), hmean: ([\d.]+)", result_str)
    if match:
        precision = float(match.group(1))
        recall = float(match.group(2))
        f1 = float(match.group(3))
    else:
        # Fallback parsing
        print(f"WARNING: Could not parse result string: {result_str}")
        precision = recall = f1 = 0.0

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction JSON against GT")
    parser.add_argument('--json', required=True, help='Path to text_results.json')
    parser.add_argument('--gt', default='datasets/evaluation/gt_ctw1500.zip',
                        help='Path to GT zip file (default: datasets/evaluation/gt_ctw1500.zip)')
    args = parser.parse_args()

    P, R, F1 = evaluate_predictions(args.json, args.gt)

    print(f"\n{'='*50}")
    print(f"Results: P={P:.4f}, R={R:.4f}, F1={F1:.4f}")
    print(f"{'='*50}")

    # Also print formatted for easy copying
    print(f"\n{P:.4f}\t{R:.4f}\t{F1:.4f}")


if __name__ == '__main__':
    main()
