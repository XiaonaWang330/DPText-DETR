"""
Per-sample comparison: Baseline vs Scale-Aware Gate on CTW1500 test set.
Uses polygon IoU matching to compute per-image P/R/F1 and identifies
improvement/regression patterns.
"""
import json
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
import sys

# ── helpers ──
def poly_iou(poly_a, poly_b):
    """IoU between two shapely Polygons."""
    try:
        if not poly_a.is_valid:
            poly_a = poly_a.buffer(0)
        if not poly_b.is_valid:
            poly_b = poly_b.buffer(0)
        inter = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def build_polygon(flat_list):
    """Build shapely Polygon from flat list [x0,y0,x1,y1,...].
    Returns None if invalid."""
    if len(flat_list) < 6:
        return None
    pts = [(float(flat_list[i]), float(flat_list[i+1])) for i in range(0, len(flat_list), 2)]
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if poly.is_valid and not poly.is_empty else None
    except Exception:
        return None


def match_predictions(gt_polys, pred_polys, pred_scores, iou_thresh=0.5):
    """
    Greedy matching: sort preds by score, match to best-IoU GT.
    Returns TP, FP, FN, FN_indices, FP_indices, matched_pairs.
    """
    n_gt = len(gt_polys)
    n_pred = len(pred_polys)
    matched_gt = set()
    matched_pred = set()
    matched_pairs = []  # (gt_idx, pred_idx, iou)
    
    sorted_idx = sorted(range(n_pred), key=lambda i: pred_scores[i], reverse=True)
    
    for pi in sorted_idx:
        best_iou = 0.0
        best_gi = -1
        for gi in range(n_gt):
            if gi in matched_gt:
                continue
            iou = poly_iou(pred_polys[pi], gt_polys[gi])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_thresh and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            matched_pairs.append((best_gi, pi, best_iou))
    
    TP = len(matched_gt)
    FP = n_pred - TP
    FN = n_gt - TP
    
    FN_set = set(range(n_gt)) - matched_gt
    FP_set = set(range(n_pred)) - matched_pred
    
    return TP, FP, FN, FN_set, FP_set, matched_pairs


# ── Load GT ──
print("Loading GT...")
with open(r'f:\Code\OCR\DPText-DETR\datasets\ctw1500\test_poly.json') as f:
    gt_data = json.load(f)

gt_by_image = defaultdict(list)
for ann in gt_data['annotations']:
    img_id = ann['image_id']
    flat_polys = ann.get('polys', None)
    if flat_polys is None or len(flat_polys) < 6:
        continue
    poly = build_polygon(flat_polys)
    if poly is not None:
        gt_by_image[img_id].append({
            'poly': poly,
            'area': poly.area,
            'bbox': ann.get('bbox', None),
        })

img_info = {img['id']: img for img in gt_data['images']}
n_gt_total = sum(len(v) for v in gt_by_image.values())
print(f"GT: {len(gt_data['images'])} images, {len(gt_data['annotations'])} annotations")
print(f"Valid GT polygons: {n_gt_total}")

# ── Load predictions ──
def load_preds(path):
    print(f"Loading predictions from: {path}")
    with open(path) as f:
        data = json.load(f)
    pred_by_image = defaultdict(list)
    for item in data:
        # item['polys'] is list of [x, y] points (16 points)
        pts = item['polys']
        # normalize to flat list
        if isinstance(pts[0], list):
            flat = []
            for p in pts:
                flat.extend([float(p[0]), float(p[1])])
        else:
            flat = [float(v) for v in pts]
        poly = build_polygon(flat)
        if poly is not None:
            pred_by_image[item['image_id']].append({
                'poly': poly,
                'score': item['score'],
            })
    return pred_by_image

baseline_preds = load_preds(r'f:\Code\OCR\DPText-DETR\output\r_50_poly\ctw1500\baseline_a100\text_results.json')
sag_preds = load_preds(r'f:\Code\OCR\DPText-DETR\output\r_50_poly\ctw1500\Scale-Aware Gate\text_results.json')

print(f"Baseline predictions: {sum(len(v) for v in baseline_preds.values())} across {len(baseline_preds)} images")
print(f"Scale-Aware Gate predictions: {sum(len(v) for v in sag_preds.values())} across {len(sag_preds)} images")

# ── Per-sample analysis ──
all_image_ids = sorted(gt_by_image.keys())
results = []

print(f"\nAnalyzing {len(all_image_ids)} images...")
for idx, img_id in enumerate(all_image_ids):
    if idx % 200 == 0:
        print(f"  Progress: {idx}/{len(all_image_ids)}")
    
    gt_list = gt_by_image.get(img_id, [])
    bl_list = baseline_preds.get(img_id, [])
    sag_list = sag_preds.get(img_id, [])
    
    if not gt_list:
        continue
    
    gt_polys = [g['poly'] for g in gt_list]
    gt_areas = [g['area'] for g in gt_list]
    
    # Baseline matching
    bl_polys_list = [p['poly'] for p in bl_list]
    bl_scores = [p['score'] for p in bl_list]
    bl_TP, bl_FP, bl_FN, bl_FN_set, bl_FP_set, _ = match_predictions(gt_polys, bl_polys_list, bl_scores)
    
    bl_P = bl_TP / (bl_TP + bl_FP) if (bl_TP + bl_FP) > 0 else 0.0
    bl_R = bl_TP / (bl_TP + bl_FN) if (bl_TP + bl_FN) > 0 else 0.0
    bl_F1 = 2 * bl_P * bl_R / (bl_P + bl_R) if (bl_P + bl_R) > 0 else 0.0
    
    # SAG matching
    sag_polys_list = [p['poly'] for p in sag_list]
    sag_scores = [p['score'] for p in sag_list]
    sag_TP, sag_FP, sag_FN, sag_FN_set, sag_FP_set, _ = match_predictions(gt_polys, sag_polys_list, sag_scores)
    
    sag_P = sag_TP / (sag_TP + sag_FP) if (sag_TP + sag_FP) > 0 else 0.0
    sag_R = sag_TP / (sag_TP + sag_FN) if (sag_TP + sag_FN) > 0 else 0.0
    sag_F1 = 2 * sag_P * sag_R / (sag_P + sag_R) if (sag_P + sag_R) > 0 else 0.0
    
    # Size metrics for GT texts in this image
    gt_areas_sorted = sorted(gt_areas)
    min_area = gt_areas_sorted[0] if gt_areas_sorted else 0
    avg_area = np.mean(gt_areas) if gt_areas else 0
    
    # GT indices missed by each model
    bl_missed = set(bl_FN_set)
    sag_missed = set(sag_FN_set)
    
    recovered = bl_missed - sag_missed  # missed by baseline, caught by SAG
    lost = sag_missed - bl_missed      # caught by baseline, missed by SAG
    both_missed = bl_missed & sag_missed  # missed by both
    
    recovered_areas = [gt_areas[i] for i in recovered]
    lost_areas = [gt_areas[i] for i in lost]
    both_missed_areas = [gt_areas[i] for i in both_missed]
    
    results.append({
        'image_id': img_id,
        'n_gt': len(gt_list),
        'n_pred_bl': len(bl_list),
        'n_pred_sag': len(sag_list),
        'bl_TP': bl_TP, 'bl_FP': bl_FP, 'bl_FN': bl_FN,
        'bl_P': bl_P, 'bl_R': bl_R, 'bl_F1': bl_F1,
        'sag_TP': sag_TP, 'sag_FP': sag_FP, 'sag_FN': sag_FN,
        'sag_P': sag_P, 'sag_R': sag_R, 'sag_F1': sag_F1,
        'delta_F1': sag_F1 - bl_F1,
        'delta_P': sag_P - bl_P,
        'delta_R': sag_R - bl_R,
        'n_recovered': len(recovered),
        'n_lost': len(lost),
        'n_both_missed': len(both_missed),
        'recovered_areas': recovered_areas,
        'lost_areas': lost_areas,
        'both_missed_areas': both_missed_areas,
        'min_gt_area': min_area,
        'avg_gt_area': avg_area,
        'gt_areas': gt_areas,
    })

print(f"  Progress: {len(all_image_ids)}/{len(all_image_ids)} - Done!")

# ── Overall stats ──
total_bl_TP = sum(r['bl_TP'] for r in results)
total_bl_FP = sum(r['bl_FP'] for r in results)
total_bl_FN = sum(r['bl_FN'] for r in results)
total_sag_TP = sum(r['sag_TP'] for r in results)
total_sag_FP = sum(r['sag_FP'] for r in results)
total_sag_FN = sum(r['sag_FN'] for r in results)

denom_bl = total_bl_TP + total_bl_FP
bl_P = total_bl_TP / denom_bl if denom_bl > 0 else 0.0
denom_bl_r = total_bl_TP + total_bl_FN
bl_R = total_bl_TP / denom_bl_r if denom_bl_r > 0 else 0.0
bl_F1 = 2 * bl_P * bl_R / (bl_P + bl_R) if (bl_P + bl_R) > 0 else 0.0

denom_sag = total_sag_TP + total_sag_FP
sag_P = total_sag_TP / denom_sag if denom_sag > 0 else 0.0
denom_sag_r = total_sag_TP + total_sag_FN
sag_R = total_sag_TP / denom_sag_r if denom_sag_r > 0 else 0.0
sag_F1 = 2 * sag_P * sag_R / (sag_P + sag_R) if (sag_P + sag_R) > 0 else 0.0

print("\n" + "="*80)
print("OVERALL METRICS (IoU=0.5)")
print("="*80)
print(f"{'':>25} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
print(f"{'Baseline':>25} {bl_P:>8.2%} {bl_R:>8.2%} {bl_F1:>8.2%} {total_bl_TP:>6} {total_bl_FP:>6} {total_bl_FN:>6}")
print(f"{'Scale-Aware Gate':>25} {sag_P:>8.2%} {sag_R:>8.2%} {sag_F1:>8.2%} {total_sag_TP:>6} {total_sag_FP:>6} {total_sag_FN:>6}")
print(f"{'Delta':>25} {sag_P-bl_P:>+8.2%} {sag_R-bl_R:>+8.2%} {sag_F1-bl_F1:>+8.2%}")
print(f"\nOfficial baseline (from metrics.json): P=90.77  R=85.64  F1=88.13")
print(f"Official SAG (best iter 7999):        P=90.27  R=86.61  F1=88.40")

# ── Improvement / Regression analysis ──
improved = [r for r in results if r['delta_F1'] > 0.001]
regressed = [r for r in results if r['delta_F1'] < -0.001]
unchanged = [r for r in results if abs(r['delta_F1']) <= 0.001]

print(f"\n{'='*80}")
print(f"PER-IMAGE ANALYSIS")
print(f"{'='*80}")
print(f"Improved:  {len(improved)} images (F1↑)")
print(f"Regressed: {len(regressed)} images (F1↓)")
print(f"Unchanged: {len(unchanged)} images (F1=)")

# ── Top improved images ──
print(f"\n--- Top 20 Most Improved Images ---")
improved_sorted = sorted(improved, key=lambda r: r['delta_F1'], reverse=True)[:20]
for r in improved_sorted:
    recov_info = ""
    if r['recovered_areas']:
        recov_info = f" recov_areas=[{','.join(f'{a:.0f}' for a in sorted(r['recovered_areas']))}]"
    print(f"  img={r['image_id']} | GT={r['n_gt']:3d} | "
          f"BL: P={r['bl_P']:.2%} R={r['bl_R']:.2%} F1={r['bl_F1']:.2%} | "
          f"SAG: P={r['sag_P']:.2%} R={r['sag_R']:.2%} F1={r['sag_F1']:.2%} | "
          f"ΔF1={r['delta_F1']:+.2%} | "
          f"recov={r['n_recovered']} lost={r['n_lost']}{recov_info}")

# ── Top regressed images ──
print(f"\n--- Top 20 Most Regressed Images ---")
regressed_sorted = sorted(regressed, key=lambda r: r['delta_F1'])[:20]
for r in regressed_sorted:
    lost_info = ""
    if r['lost_areas']:
        lost_info = f" lost_areas=[{','.join(f'{a:.0f}' for a in sorted(r['lost_areas']))}]"
    print(f"  img={r['image_id']} | GT={r['n_gt']:3d} | "
          f"BL: P={r['bl_P']:.2%} R={r['bl_R']:.2%} F1={r['bl_F1']:.2%} | "
          f"SAG: P={r['sag_P']:.2%} R={r['sag_R']:.2%} F1={r['sag_F1']:.2%} | "
          f"ΔF1={r['delta_F1']:+.2%} | "
          f"recov={r['n_recovered']} lost={r['n_lost']}{lost_info}")

# ── Recovered / Lost text analysis ──
all_recovered = []
all_lost = []
all_both_missed = []
for r in results:
    for a in r['recovered_areas']:
        all_recovered.append((r['image_id'], a))
    for a in r['lost_areas']:
        all_lost.append((r['image_id'], a))
    for a in r['both_missed_areas']:
        all_both_missed.append((r['image_id'], a))

print(f"\n{'='*80}")
print(f"RECOVERED vs LOST TEXT ANALYSIS")
print(f"{'='*80}")
print(f"Recovered (baseline missed, SAG caught): {len(all_recovered)} texts")
print(f"Lost (baseline caught, SAG missed):     {len(all_lost)} texts")
print(f"Still missed by both:                   {len(all_both_missed)} texts")
print(f"Net gain:                                {len(all_recovered) - len(all_lost):+d} texts")

if all_recovered:
    areas_r = [a for _, a in all_recovered]
    print(f"\nRECOVERED text size distribution ({len(areas_r)} texts):")
    print(f"  Mean area: {np.mean(areas_r):.0f}  Median: {np.median(areas_r):.0f}  Std: {np.std(areas_r):.0f}")
    print(f"  Min: {np.min(areas_r):.0f}  Max: {np.max(areas_r):.0f}")
    
    for lo, hi, label in [(0, 100, "Micro (<100)"), (100, 200, "Tiny (100-200)"), 
                            (200, 500, "Small (200-500)"), (500, 1000, "Medium (500-1000)"),
                            (1000, 5000, "Large (1000-5000)"), (5000, 999999, "Very Large (5000+)")]:
        cnt = sum(1 for a in areas_r if lo <= a < hi)
        if cnt > 0:
            print(f"  {label:20s}: {cnt:4d} ({cnt/len(areas_r):5.1%})")

if all_lost:
    areas_l = [a for _, a in all_lost]
    print(f"\nLOST text size distribution ({len(areas_l)} texts):")
    print(f"  Mean area: {np.mean(areas_l):.0f}  Median: {np.median(areas_l):.0f}  Std: {np.std(areas_l):.0f}")
    print(f"  Min: {np.min(areas_l):.0f}  Max: {np.max(areas_l):.0f}")
    
    for lo, hi, label in [(0, 100, "Micro (<100)"), (100, 200, "Tiny (100-200)"), 
                            (200, 500, "Small (200-500)"), (500, 1000, "Medium (500-1000)"),
                            (1000, 5000, "Large (1000-5000)"), (5000, 999999, "Very Large (5000+)")]:
        cnt = sum(1 for a in areas_l if lo <= a < hi)
        if cnt > 0:
            print(f"  {label:20s}: {cnt:4d} ({cnt/len(areas_l):5.1%})")

if all_both_missed:
    areas_b = [a for _, a in all_both_missed]
    print(f"\nSTILL MISSED BY BOTH - size distribution ({len(areas_b)} texts):")
    print(f"  Mean area: {np.mean(areas_b):.0f}  Median: {np.median(areas_b):.0f}  Std: {np.std(areas_b):.0f}")
    print(f"  Min: {np.min(areas_b):.0f}  Max: {np.max(areas_b):.0f}")
    
    for lo, hi, label in [(0, 100, "Micro (<100)"), (100, 200, "Tiny (100-200)"), 
                            (200, 500, "Small (200-500)"), (500, 1000, "Medium (500-1000)"),
                            (1000, 5000, "Large (1000-5000)"), (5000, 999999, "Very Large (5000+)")]:
        cnt = sum(1 for a in areas_b if lo <= a < hi)
        if cnt > 0:
            print(f"  {label:20s}: {cnt:4d} ({cnt/len(areas_b):5.1%})")

# ── FP analysis ──
print(f"\n{'='*80}")
print(f"FALSE POSITIVE ANALYSIS")
print(f"{'='*80}")
total_bl_FP_count = sum(r['bl_FP'] for r in results)
total_sag_FP_count = sum(r['sag_FP'] for r in results)
print(f"Baseline total FP: {total_bl_FP_count}")
print(f"SAG total FP:      {total_sag_FP_count}")
print(f"FP delta:          {total_sag_FP_count - total_bl_FP_count:+d}")

fp_increased = [r for r in results if r['sag_FP'] > r['bl_FP']]
fp_decreased = [r for r in results if r['sag_FP'] < r['bl_FP']]
fp_same = [r for r in results if r['sag_FP'] == r['bl_FP']]
print(f"\nImages with MORE FP (SAG > BL):    {len(fp_increased)}")
print(f"Images with FEWER FP (SAG < BL):   {len(fp_decreased)}")
print(f"Images with SAME FP:               {len(fp_same)}")

# Sanity check: images with high FP increase
worst_fp = sorted(fp_increased, key=lambda r: r['sag_FP'] - r['bl_FP'], reverse=True)[:10]
print(f"\nTop 10 FP-increased images:")
for r in worst_fp:
    print(f"  img={r['image_id']} | GT={r['n_gt']} | BL_FP={r['bl_FP']} SAG_FP={r['sag_FP']} | ΔFP={r['sag_FP']-r['bl_FP']:+d}")

# ── Size bucket analysis ──
print(f"\n{'='*80}")
print(f"SIZE BUCKET ANALYSIS (by min GT area in image)")
print(f"{'='*80}")

buckets = [
    ("Very Small (min<100)", 0, 100),
    ("Tiny (min 100-200)", 100, 200),
    ("Small (min 200-500)", 200, 500),
    ("Medium (min 500-1000)", 500, 1000),
    ("Large (min 1000-5000)", 1000, 5000),
    ("Very Large (min>=5000)", 5000, 999999),
]

for bname, lo, hi in buckets:
    bucket_imgs = [r for r in results if lo <= r['min_gt_area'] < hi]
    if not bucket_imgs:
        continue
    avg_dF1 = np.mean([r['delta_F1'] for r in bucket_imgs])
    avg_dP = np.mean([r['delta_P'] for r in bucket_imgs])
    avg_dR = np.mean([r['delta_R'] for r in bucket_imgs])
    n_improved = sum(1 for r in bucket_imgs if r['delta_F1'] > 0)
    n_regressed = sum(1 for r in bucket_imgs if r['delta_F1'] < 0)
    avg_recovered = np.mean([r['n_recovered'] for r in bucket_imgs])
    avg_lost = np.mean([r['n_lost'] for r in bucket_imgs])
    total_recov = sum(r['n_recovered'] for r in bucket_imgs)
    total_lost = sum(r['n_lost'] for r in bucket_imgs)
    print(f"\n{bname}: {len(bucket_imgs)} images")
    print(f"  Avg ΔF1: {avg_dF1:+.2%}  Avg ΔP: {avg_dP:+.2%}  Avg ΔR: {avg_dR:+.2%}")
    print(f"  Improved: {n_improved}  Regressed: {n_regressed}  Unchanged: {len(bucket_imgs)-n_improved-n_regressed}")
    print(f"  Total recovered: {total_recov}  Total lost: {total_lost}  Net: {total_recov-total_lost:+d}")
    print(f"  Avg recovered/img: {avg_recovered:.2f}  Avg lost/img: {avg_lost:.2f}")

# ── Correlation: image GT count vs delta F1 ──
print(f"\n{'='*80}")
print(f"IMAGE-LEVEL CORRELATION ANALYSIS")
print(f"{'='*80}")

# By n_gt (number of GT texts in image)
for lo, hi, label in [(0, 3, "Very few GT (0-2)"), (3, 6, "Few GT (3-5)"), 
                        (6, 10, "Many GT (6-9)"), (10, 20, "Crowded (10-19)"),
                        (20, 999, "Very crowded (20+)")]:
    imgs = [r for r in results if lo <= r['n_gt'] < hi]
    if not imgs:
        continue
    avg_dF1 = np.mean([r['delta_F1'] for r in imgs])
    avg_dR = np.mean([r['delta_R'] for r in imgs])
    avg_recov = np.mean([r['n_recovered'] for r in imgs])
    print(f"  {label:25s}: {len(imgs):4d} imgs | avg ΔF1={avg_dF1:+.2%} avg ΔR={avg_dR:+.2%} avg recov/img={avg_recov:.2f}")

# ── Hard images: still many missed ──
print(f"\n{'='*80}")
print(f"HARD IMAGES (most texts still missed by SAG)")
print(f"{'='*80}")
hard_imgs = sorted(results, key=lambda r: r['n_both_missed'], reverse=True)[:20]
for r in hard_imgs:
    if r['n_both_missed'] > 0:
        missed_areas = sorted(r['both_missed_areas'])
        areas_str = ','.join(f'{a:.0f}' for a in missed_areas[:10])
        if len(missed_areas) > 10:
            areas_str += ',...'
        print(f"  img={r['image_id']} | GT={r['n_gt']:3d} | both_missed={r['n_both_missed']:2d} | "
              f"SAG_F1={r['sag_F1']:.2%} | "
              f"missed_areas=[{areas_str}]")

# ── Summary stats ──
print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
n_net_positive = sum(1 for r in results if r['n_recovered'] > r['n_lost'])
n_net_negative = sum(1 for r in results if r['n_lost'] > r['n_recovered'])
n_net_zero = sum(1 for r in results if r['n_recovered'] == r['n_lost'])
print(f"Images with net recovered > lost: {n_net_positive}")
print(f"Images with net lost > recovered: {n_net_negative}")
print(f"Images with net zero change:      {n_net_zero}")

# Check if recovered texts tend to come from images with many small texts
imgs_with_recovery = [r for r in results if r['n_recovered'] > 0]
if imgs_with_recovery:
    avg_min_area_recovery = np.mean([r['min_gt_area'] for r in imgs_with_recovery])
    avg_min_area_no_recovery = np.mean([r['min_gt_area'] for r in results if r['n_recovered'] == 0])
    print(f"\nAvg min GT area of images WITH recoveries:  {avg_min_area_recovery:.0f}")
    print(f"Avg min GT area of images WITHOUT recoveries: {avg_min_area_no_recovery:.0f}")

print("\nDone.")
