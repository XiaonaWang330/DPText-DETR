"""
Generate Scale-Aware Gate visualization images for manual analysis.
Overlays: GT (green), matched preds (blue), FP (red), FN (yellow).
"""
import json
import os
import cv2
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.patches as mpatches

# ── Config ──
IMG_DIR = r'f:\Code\OCR\DPText-DETR\datasets\ctw1500\test_images'
GT_PATH = r'f:\Code\OCR\DPText-DETR\datasets\ctw1500\test_poly.json'
SAG_PATH = r'f:\Code\OCR\DPText-DETR\output\r_50_poly\ctw1500\Scale-Aware Gate\text_results.json'
BL_PATH  = r'f:\Code\OCR\DPText-DETR\output\r_50_poly\ctw1500\baseline_a100\text_results.json'
OUT_DIR = r'f:\Code\OCR\DPText-DETR\output\r_50_poly\ctw1500\vis_sag_analysis'

os.makedirs(OUT_DIR, exist_ok=True)

# ── Polygon helpers ──
def build_poly(flat_list):
    if len(flat_list) < 6:
        return None
    pts = [(float(flat_list[i]), float(flat_list[i+1])) for i in range(0, len(flat_list), 2)]
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if poly.is_valid and not poly.is_empty else None
    except:
        return None

def poly_iou(poly_a, poly_b):
    try:
        if not poly_a.is_valid: poly_a = poly_a.buffer(0)
        if not poly_b.is_valid: poly_b = poly_b.buffer(0)
        inter = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return inter / union if union > 0 else 0.0
    except:
        return 0.0

def match_predictions(gt_polys, pred_polys, pred_scores, iou_thresh=0.5):
    matched_gt = set()
    matched_pred = set()
    pairs = []
    sorted_idx = sorted(range(len(pred_polys)), key=lambda i: pred_scores[i], reverse=True)
    for pi in sorted_idx:
        best_iou, best_gi = 0.0, -1
        for gi in range(len(gt_polys)):
            if gi in matched_gt: continue
            iou = poly_iou(pred_polys[pi], gt_polys[gi])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thresh and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            pairs.append((best_gi, pi, best_iou))
    TP = len(matched_gt)
    FP = n_pred - TP if (n_pred := len(pred_polys)) else 0
    FN = len(gt_polys) - TP
    FN_set = set(range(len(gt_polys))) - matched_gt
    FP_set = set(range(len(pred_polys))) - matched_pred
    return TP, FP, FN, FN_set, FP_set, pairs

# ── Precompute data from script ──
print("Loading GT...")
with open(GT_PATH) as f:
    gt_data = json.load(f)

gt_by_image = {}
for ann in gt_data['annotations']:
    img_id = ann['image_id']
    flat_polys = ann.get('polys', None)
    if flat_polys is None: continue
    poly = build_poly(flat_polys)
    if poly is None: continue
    gt_by_image.setdefault(img_id, []).append({
        'poly': poly, 'area': poly.area,
        'polys_flat': flat_polys
    })

def load_preds(path):
    with open(path) as f:
        data = json.load(f)
    pred_by_image = defaultdict(list)
    for item in data:
        pts = item['polys']
        if isinstance(pts[0], list):
            flat = []; [flat.extend([float(p[0]), float(p[1])]) for p in pts]
        else:
            flat = [float(v) for v in pts]
        poly = build_poly(flat)
        if poly is not None:
            pred_by_image[item['image_id']].append({
                'poly': poly, 'score': item['score'], 'polys_flat': flat
            })
    return dict(pred_by_image)

print("Loading predictions...")
sag_all = load_preds(SAG_PATH)
bl_all  = load_preds(BL_PATH)

all_image_ids = sorted(gt_by_image.keys())
print(f"Total images: {len(all_image_ids)}")

# ── Compute per-image results ──
print("Computing per-image metrics...")
results = {}
for img_id in all_image_ids:
    gts = gt_by_image.get(img_id, [])
    bls = bl_all.get(img_id, [])
    sags = sag_all.get(img_id, [])
    if not gts: continue
    
    gt_polys = [g['poly'] for g in gts]
    
    bl_polys_list = [p['poly'] for p in bls]
    bl_scores = [p['score'] for p in bls]
    bl_TP, bl_FP, bl_FN, bl_FN_set, bl_FP_set, bl_pairs = match_predictions(gt_polys, bl_polys_list, bl_scores)
    bl_P = bl_TP / (bl_TP + bl_FP) if (bl_TP + bl_FP) > 0 else 0
    bl_R = bl_TP / (bl_TP + bl_FN) if (bl_TP + bl_FN) > 0 else 0
    bl_F1 = 2 * bl_P * bl_R / (bl_P + bl_R) if (bl_P + bl_R) > 0 else 0
    
    sag_polys_list = [p['poly'] for p in sags]
    sag_scores = [p['score'] for p in sags]
    sag_TP, sag_FP, sag_FN, sag_FN_set, sag_FP_set, sag_pairs = match_predictions(gt_polys, sag_polys_list, sag_scores)
    sag_P = sag_TP / (sag_TP + sag_FP) if (sag_TP + sag_FP) > 0 else 0
    sag_R = sag_TP / (sag_TP + sag_FN) if (sag_TP + sag_FN) > 0 else 0
    sag_F1 = 2 * sag_P * sag_R / (sag_P + sag_R) if (sag_P + sag_R) > 0 else 0
    
    bl_missed = set(bl_FN_set)
    sag_missed = set(sag_FN_set)
    recovered = bl_missed - sag_missed
    lost = sag_missed - bl_missed
    both_missed = bl_missed & sag_missed
    
    results[img_id] = {
        'bl_TP': bl_TP, 'bl_FP': bl_FP, 'bl_FN': bl_FN,
        'bl_P': bl_P, 'bl_R': bl_R, 'bl_F1': bl_F1,
        'bl_matched_gt': set(p[0] for p in bl_pairs),
        'bl_FN_set': bl_FN_set, 'bl_FP_set': bl_FP_set,
        'sag_TP': sag_TP, 'sag_FP': sag_FP, 'sag_FN': sag_FN,
        'sag_P': sag_P, 'sag_R': sag_R, 'sag_F1': sag_F1,
        'sag_matched_gt': set(p[0] for p in sag_pairs),
        'sag_FN_set': sag_FN_set, 'sag_FP_set': sag_FP_set,
        'recovered': recovered, 'lost': lost, 'both_missed': both_missed,
        'delta_F1': sag_F1 - bl_F1, 'delta_P': sag_P - bl_P, 'delta_R': sag_R - bl_R,
        'n_gt': len(gts), 'n_pred_sag': len(sags), 'n_pred_bl': len(bls),
        'gt_areas': [g['area'] for g in gts],
    }

# ── Drawing function ──
def draw_polygon(ax, flat_list, color, alpha=0.4, linewidth=1.5, label=None):
    """Draw a polygon from flat list [x0,y0,x1,y1,...]."""
    pts = [(flat_list[i], flat_list[i+1]) for i in range(0, len(flat_list), 2)]
    pts.append(pts[0])  # close
    xs, ys = zip(*pts)
    if label:
        ax.fill(xs, ys, alpha=alpha, color=color)
        ax.plot(xs, ys, color=color, linewidth=linewidth, label=label)
    else:
        ax.fill(xs, ys, alpha=alpha, color=color)
        ax.plot(xs, ys, color=color, linewidth=linewidth)

def draw_pred_boxes(ax, preds, fp_indices, matched_indices_dict, score_thresh=0.5):
    """Draw all predictions: matched in blue, FP in red, with score text."""
    for pi, pred in enumerate(preds):
        score = pred['score']
        if score < score_thresh:
            continue
        flat = pred['polys_flat']
        if pi in fp_indices:
            draw_polygon(ax, flat, 'red', alpha=0.15, linewidth=1.0)
        else:
            draw_polygon(ax, flat, 'dodgerblue', alpha=0.15, linewidth=1.0)
        # Score text at first point
        x, y = flat[0], flat[1]
        color = 'red' if pi in fp_indices else 'dodgerblue'
        ax.text(x, y-2, f'{score:.2f}', fontsize=4, color=color, 
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, lw=0.3))

def create_visualization(img_id, save_path, tag=""):
    """Generate single visualization image."""
    r = results.get(img_id)
    if r is None:
        return False
    
    img_path = os.path.join(IMG_DIR, f'{img_id}.jpg')
    if not os.path.exists(img_path):
        return False
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    gts = gt_by_image.get(img_id, [])
    sags = sag_all.get(img_id, [])
    bls = bl_all.get(img_id, [])
    
    n_gt = len(gts)
    
    # Left: GT + SAG predictions
    # Right: GT + Baseline predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, h/w*9)))
    
    for ax in [ax1, ax2]:
        ax.imshow(img)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis('off')
    
    # ── Left: Scale-Aware Gate ──
    # Draw GT: green for matched, yellow for FN
    sag_fn = r['sag_FN_set']
    sag_matched = r['sag_matched_gt']
    for gi, gt in enumerate(gts):
        flat = gt['polys_flat']
        if gi in sag_fn:
            draw_polygon(ax1, flat, 'yellow', alpha=0.3, linewidth=1.5)
        elif gi in sag_matched:
            draw_polygon(ax1, flat, 'lime', alpha=0.25, linewidth=1.5)
    
    # Draw SAG predictions
    draw_pred_boxes(ax1, sags, r['sag_FP_set'], None)
    
    # Recovered indicator
    recovered = r.get('recovered', set())
    for gi in recovered:
        flat = gts[gi]['polys_flat']
        cx = flat[0]
        cy = flat[1]
        ax1.scatter(cx, cy, s=30, c='cyan', marker='*', edgecolors='black', linewidths=0.3, zorder=10)
    
    ax1.set_title(f"Scale-Aware Gate  |  img={img_id}  GT={n_gt}\n"
                  f"P={r['sag_P']:.2%}  R={r['sag_R']:.2%}  F1={r['sag_F1']:.2%}  "
                  f"TP={r['sag_TP']}  FP={r['sag_FP']}  FN={r['sag_FN']}\n"
                  f"recov={len(recovered)}  lost={len(r.get('lost',set()))}",
                  fontsize=9, fontweight='bold')
    
    # ── Right: Baseline ──
    bl_fn = r['bl_FN_set']
    bl_matched = r['bl_matched_gt']
    for gi, gt in enumerate(gts):
        flat = gt['polys_flat']
        if gi in bl_fn:
            draw_polygon(ax2, flat, 'yellow', alpha=0.3, linewidth=1.5)
        elif gi in bl_matched:
            draw_polygon(ax2, flat, 'lime', alpha=0.25, linewidth=1.5)
    
    draw_pred_boxes(ax2, bls, r['bl_FP_set'], None)
    
    ax2.set_title(f"Baseline  |  img={img_id}  GT={n_gt}\n"
                  f"P={r['bl_P']:.2%}  R={r['bl_R']:.2%}  F1={r['bl_F1']:.2%}  "
                  f"TP={r['bl_TP']}  FP={r['bl_FP']}  FN={r['bl_FN']}",
                  fontsize=9, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='lime', alpha=0.3, label='GT: Matched'),
        mpatches.Patch(color='yellow', alpha=0.3, label='GT: Missed (FN)'),
        mpatches.Patch(color='dodgerblue', alpha=0.3, label='Pred: Matched (TP)'),
        mpatches.Patch(color='red', alpha=0.3, label='Pred: False Positive'),
    ]
    if len(recovered) > 0:
        legend_elements.append(mpatches.Patch(color='cyan', alpha=0.7, label='★ Recovered by SAG'))
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=6, ncol=2, 
               framealpha=0.9, bbox_to_anchor=(0.01, -0.02))
    
    # Top-right summary box
    dF1 = r['delta_F1']
    dP = r['delta_P']
    dR = r['delta_R']
    status = "▲ IMPROVED" if dF1 > 0 else ("▼ REGRESSED" if dF1 < 0 else "► SAME")
    color = 'green' if dF1 > 0 else ('red' if dF1 < 0 else 'gray')
    
    min_area = min(r['gt_areas']) if r['gt_areas'] else 0
    avg_area = np.mean(r['gt_areas']) if r['gt_areas'] else 0
    
    textstr = (f"ΔF1: {dF1:+.2%}  |  ΔP: {dP:+.2%}  |  ΔR: {dR:+.2%}\n"
               f"Status: {status}\n"
               f"GT={n_gt}  min_area={min_area:.0f}  avg_area={avg_area:.0f}\n"
               f"SAG_pred={r['n_pred_sag']}  BL_pred={r['n_pred_bl']}")
    fig.text(0.98, 0.98, textstr, fontsize=7, family='monospace',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, lw=2),
             transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True

# ── Generate visualizations ──
print("\nRanking images...")

# 1. Top 20 most improved
improved = sorted([(k, v) for k, v in results.items() if v['delta_F1'] > 0.001], 
                   key=lambda x: x[1]['delta_F1'], reverse=True)
# 2. Top 20 most regressed
regressed = sorted([(k, v) for k, v in results.items() if v['delta_F1'] < -0.001], 
                    key=lambda x: x[1]['delta_F1'])
# 3. Top 20 hardest (most both_missed)
hardest = sorted([(k, v) for k, v in results.items()], 
                  key=lambda x: len(x[1]['both_missed']), reverse=True)
# 4. Most recovered
most_recov = sorted([(k, v) for k, v in results.items()], 
                     key=lambda x: len(x[1]['recovered']), reverse=True)

# ── Generate category folders ──
categories = {
    '01_improved': improved[:20],
    '02_regressed': regressed[:20],
    '03_hardest': hardest[:20],
    '04_most_recovered': most_recov[:20],
}

for cat_name, img_list in categories.items():
    cat_dir = os.path.join(OUT_DIR, cat_name)
    os.makedirs(cat_dir, exist_ok=True)
    print(f"\nGenerating {cat_name} ({len(img_list)} images)...")
    for rank, (img_id, r) in enumerate(img_list):
        fname = f"{rank+1:02d}_{img_id}_dF1_{r['delta_F1']:+.3f}_nGT{r['n_gt']}_recov{len(r['recovered'])}_lost{len(r['lost'])}.png"
        save_path = os.path.join(cat_dir, fname)
        if create_visualization(img_id, save_path):
            print(f"  [{rank+1}/{len(img_list)}] {fname}")

# 5. Quick montage: first 4 from each category for overview
print("\nGenerating overview montage...")
fig, axes = plt.subplots(4, 4, figsize=(32, 32))
cat_labels = ['Improved', 'Regressed', 'Hardest', 'Most Recovered']
img_lists = [improved[:4], regressed[:4], hardest[:4], most_recov[:4]]

for row, (cat_label, img_list) in enumerate(zip(cat_labels, img_lists)):
    for col, (img_id, r) in enumerate(img_list):
        ax = axes[row, col]
        img_path = os.path.join(IMG_DIR, f'{img_id}.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            
            gts = gt_by_image.get(img_id, [])
            sags = sag_all.get(img_id, [])
            
            # GT in lime
            for gt in gts:
                draw_polygon(ax, gt['polys_flat'], 'lime', alpha=0.2, linewidth=0.8)
            
            # SAG preds: matched blue, FP red
            for pi, pred in enumerate(sags):
                if pred['score'] < 0.5:
                    continue
                color = 'red' if pi in r['sag_FP_set'] else 'dodgerblue'
                draw_polygon(ax, pred['polys_flat'], color, alpha=0.12, linewidth=0.6)
            
            # Recovered stars
            for gi in r.get('recovered', set()):
                flat = gts[gi]['polys_flat']
                ax.scatter(flat[0], flat[1], s=40, c='cyan', marker='*', 
                          edgecolors='black', linewidths=0.3, zorder=10)
        
        ax.set_title(f"#{row*4+col+1} {img_id}  ΔF1={r['delta_F1']:+.2%}  recov={len(r['recovered'])}\n"
                     f"P:{r['sag_P']:.2%} R:{r['sag_R']:.2%} F1:{r['sag_F1']:.2%}  GT:{r['n_gt']}",
                     fontsize=7)
        ax.axis('off')

plt.suptitle("Scale-Aware Gate vs Baseline — Per-Sample Analysis Overview\n"
             "Green=GT  Blue=Match  Red=FP  ★=Recovered by SAG",
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '00_overview_montage.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nAll visualizations saved to: {OUT_DIR}")
print("Done!")
