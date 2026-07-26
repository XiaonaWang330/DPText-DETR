"""
P3 vs Baseline Visualization Script
====================================
Generates 2×2 comparison images for P3-DRTP vs baseline on CTW1500 test set.

Output format (matching vis_cura/ style):
  ┌─────────────────┬─────────────────┐
  │ P3 Detections   │ Ground Truth    │
  ├─────────────────┼─────────────────┤
  │ Baseline Heatmap│ P3 Heatmap      │
  │ + GT Boxes      │ + GT Boxes      │
  └─────────────────┴─────────────────┘

Usage:
  python tools/visualize_p3.py \
      --config configs/DPText_DETR/CTW1500/R_50_poly_drtp_ablation_p3_A100.yaml \
      --p3-weights output/r_50_poly/ctw1500/DRTP_P3/model_best.pth \
      --baseline-weights output/r_50_poly/ctw1500/baseline_a100/baseline_model_best.pth \
      --output output/vis_p3 \
      --num-images 50
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.modeling import build_model as d2_build_model
from detectron2.checkpoint import DetectionCheckpointer

from adet.config import get_cfg as adet_get_cfg
from adet.data import DatasetMapperWithBasis  # triggers dataset registration
from adet.utils.misc import NestedTensor


# ── Color palette ──
POLYGON_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255),
]


def setup_cfg(config_file, weights_file):
    """Build detectron2 config from yaml and inject weights path."""
    cfg = adet_get_cfg()
    _merge_from_file_utf8(cfg, config_file)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


def _merge_from_file_utf8(cfg, filename):
    """Workaround for GBK encoding error on Windows when reading YAML configs."""
    from detectron2.config import CfgNode as D2CfgNode
    from fvcore.common.config import CfgNode as FvCfgNode

    # Save originals
    d2_orig = D2CfgNode._open_cfg.__func__
    fv_orig = FvCfgNode._open_cfg.__func__

    @classmethod
    def _open_cfg_utf8(cls, fname):
        return open(fname, "r", encoding="utf-8")

    D2CfgNode._open_cfg = _open_cfg_utf8
    FvCfgNode._open_cfg = _open_cfg_utf8

    try:
        cfg.merge_from_file(filename)
    finally:
        D2CfgNode._open_cfg = classmethod(d2_orig)
        FvCfgNode._open_cfg = classmethod(fv_orig)


class FeatureHooks:
    """Simple storage for forward-hook captured feature tensors."""

    def __init__(self):
        self.features = {}

    def clear(self):
        self.features.clear()


def draw_polygons(image, polygons, color=None, thickness=2, show_label=False):
    """Draw polygon detections on image."""
    vis = image.copy()
    for i, poly_data in enumerate(polygons):
        if isinstance(poly_data, dict):
            poly = poly_data["polygon"]
            score = poly_data.get("score", 1.0)
        else:
            poly = poly_data
            score = 1.0
        c = color if color else POLYGON_COLORS[i % len(POLYGON_COLORS)]
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=c, thickness=thickness)
        if show_label and score < 1.0:
            cx, cy = int(poly[0][0]), int(poly[0][1])
            cv2.putText(vis, f"{score:.2f}", (cx, cy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    return vis


def heatmap_overlay(image_bgr, heatmap, alpha=0.45):
    """Overlay heatmap on BGR image."""
    h, w = image_bgr.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, hm_color, alpha, 0)


def load_gt_polygons(data_dict):
    """Extract GT polygons from CTW1500 dataset dict."""
    polygons = []
    for ann in data_dict.get("annotations", []):
        segm = ann.get("segmentation", [])
        if segm:
            for seg in segm:
                coords = np.array(seg).reshape(-1, 2)
                polygons.append({"polygon": coords, "score": 1.0})
        else:
            x1, y1, w_box, h_box = ann.get("bbox", [0, 0, 0, 0])
            bbox_poly = np.array([[x1, y1], [x1 + w_box, y1],
                                  [x1 + w_box, y1 + h_box], [x1, y1 + h_box]])
            polygons.append({"polygon": bbox_poly, "score": 1.0})
    return polygons


def create_2x2_figure(orig_bgr, p3_det, gt_polys, bl_heatmap, p3_heatmap, image_id, save_path):
    """Create and save 2×2 comparison grid."""
    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # Top-left: P3 Detections
    p3_panel = cv2.cvtColor(draw_polygons(orig_bgr, p3_det, thickness=2), cv2.COLOR_BGR2RGB)

    # Top-right: Ground Truth
    gt_panel = cv2.cvtColor(draw_polygons(orig_bgr, gt_polys, color=(0, 255, 0), thickness=1), cv2.COLOR_BGR2RGB)

    # Bottom-left: Baseline Heatmap + GT
    if bl_heatmap is not None:
        bl_hm = heatmap_overlay(orig_bgr, bl_heatmap)
        bl_hm = draw_polygons(bl_hm, gt_polys, color=(0, 255, 0), thickness=1)
        bl_hm = cv2.cvtColor(bl_hm, cv2.COLOR_BGR2RGB)
    else:
        bl_hm = np.zeros((orig_bgr.shape[0], orig_bgr.shape[1], 3), dtype=np.uint8)

    # Bottom-right: P3 Heatmap + GT
    if p3_heatmap is not None:
        p3_hm = heatmap_overlay(orig_bgr, p3_heatmap)
        p3_hm = draw_polygons(p3_hm, gt_polys, color=(0, 255, 0), thickness=1)
        p3_hm = cv2.cvtColor(p3_hm, cv2.COLOR_BGR2RGB)
    else:
        p3_hm = np.zeros((orig_bgr.shape[0], orig_bgr.shape[1], 3), dtype=np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    titles = [
        f"P3-DRTP Detections ({len(p3_det)} boxes)",
        f"Ground Truth ({len(gt_polys)} texts)",
        "Baseline Projected P3 (no CLIP)",
        "P3-DRTP After CLIP Fusion (P3 active)",
    ]
    images = [p3_panel, gt_panel, bl_hm, p3_hm]

    for ax, img_t, title in zip(axes, images, titles):
        ax.imshow(img_t)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle(f"P3 vs Baseline — {image_id}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="P3 vs Baseline Visualization")
    parser.add_argument("--config", required=True,
                        help="P3 config yaml (e.g., R_50_poly_drtp_ablation_p3_A100.yaml)")
    parser.add_argument("--baseline-config", default=None,
                        help="Baseline config yaml. Default: auto-detect from --config dir")
    parser.add_argument("--p3-weights", required=True,
                        help="Path to P3 model checkpoint")
    parser.add_argument("--baseline-weights", required=True,
                        help="Path to baseline model checkpoint")
    parser.add_argument("--output", default="output/vis_p3")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("[Step 1] Building models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Baseline config (separate file, no merge) ---
    if args.baseline_config:
        bl_config_file = args.baseline_config
    else:
        bl_config_file = os.path.join(os.path.dirname(args.config),
                                      "R_50_poly_baseline_A100.yaml")
    baseline_cfg = setup_cfg(bl_config_file, args.baseline_weights)

    baseline_model = d2_build_model(baseline_cfg)
    bl_checkpointer = DetectionCheckpointer(baseline_model)
    bl_checkpointer.resume_or_load(args.baseline_weights, resume=False)
    baseline_model.eval()
    baseline_model.to(device)
    print(f"  Baseline loaded: {bl_config_file}")

    # --- P3 config (separate, no merge) ---
    p3_cfg = setup_cfg(args.config, args.p3_weights)
    p3_model = d2_build_model(p3_cfg)
    p3_checkpointer = DetectionCheckpointer(p3_model)
    p3_checkpointer.resume_or_load(args.p3_weights, resume=False)
    p3_model.eval()
    p3_model.to(device)
    print(f"  P3 loaded: {args.config}")

    # Get dataset from P3 config
    from detectron2.data import build_detection_test_loader
    dataset_name = p3_cfg.DATASETS.TEST[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)

    # Build data loader with correct mapper
    mapper = DatasetMapperWithBasis(p3_cfg, is_train=False)
    data_loader = build_detection_test_loader(dataset_dicts, mapper=mapper)
    print(f"  Dataset: {dataset_name} ({len(dataset_dicts)} test images)")

    # Feature hooks
    bl_hooks = FeatureHooks()
    p3_hooks = FeatureHooks()

    # Register hooks to capture P3-level features BEFORE/AFTER CLIP fusion.
    # Baseline: input_proj[0] output (projected res3, no CLIP).
    # P3-DRTP:  clip_adapter output list[0] (after CLIP fusion at active level 0).
    def make_res3_hook(store):
        def _hook(m, i, o):
            store.features["p3"] = o.detach() if isinstance(o, torch.Tensor) else o
        return _hook

    def make_clip_hook(store):
        def _hook(m, i, o):
            # clip_adapter returns a list of tensors; level 0 is P3
            store.features["p3"] = o[0].detach() if isinstance(o, (list, tuple)) else o.detach()
        return _hook

    # Baseline: hook input_proj[0] inside DPText_DETR
    bl_handle = None
    for bl_name, bl_mod in baseline_model.named_modules():
        if bl_name.endswith("input_proj.0"):
            bl_handle = bl_mod.register_forward_hook(make_res3_hook(bl_hooks))
            break
    if bl_handle is None:
        print("  [WARN] Could not find baseline input_proj[0].")

    # P3: hook clip_adapter output
    p3_handle = None
    for p3_name, p3_mod in p3_model.named_modules():
        if p3_name.endswith("clip_adapter"):
            p3_handle = p3_mod.register_forward_hook(make_clip_hook(p3_hooks))
            break
    if p3_handle is None:
        print("  [WARN] Could not find P3 clip_adapter; falling back to input_proj[0].")
        for p3_name, p3_mod in p3_model.named_modules():
            if p3_name.endswith("input_proj.0"):
                p3_handle = p3_mod.register_forward_hook(make_res3_hook(p3_hooks))
                break

    print(f"\n[Step 2] Processing {min(args.num_images, len(dataset_dicts))} images...")

    summary = []
    for idx, batch in enumerate(tqdm(data_loader, desc="Visualizing", total=min(args.num_images, len(data_loader)))):
        if idx >= args.num_images:
            break
        # batch is [dict] (list of 1 dict for batch_size=1)
        data_dict = batch[0] if isinstance(batch, list) else batch

        image_path = data_dict.get("file_name", dataset_dicts[idx]["file_name"])
        image_id = os.path.basename(image_path).split(".")[0]

        # Original image for visualization
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            continue

        # Format input for model
        model_input = [{
            "image": data_dict["image"].to(device),
            "height": data_dict["height"],
            "width": data_dict["width"],
        }]

        # Baseline inference
        bl_hooks.clear()
        with torch.no_grad():
            bl_output = baseline_model(model_input)

        # P3 inference
        p3_hooks.clear()
        with torch.no_grad():
            p3_output = p3_model(model_input)

        # Parse outputs
        def parse_output(output):
            """Extract polygons + scores from model output."""
            results = []
            if not output:
                return results
            # output is list of dicts: [{"instances": Instances}, ...]
            inst = output[0].get("instances")
            if inst is None or not hasattr(inst, "polygons") or len(inst) == 0:
                return results
            polygons = inst.polygons  # [N, num_ctrl_points * 2], already in pixel coords
            scores = inst.scores      # [N]
            for i in range(len(inst)):
                if polygons[i].numel() == 0:
                    continue
                poly = polygons[i].cpu().numpy().reshape(-1, 2)
                s = float(scores[i].cpu())
                results.append({"polygon": poly, "score": s})
            return results

        p3_det = parse_output(p3_output)
        bl_det = parse_output(bl_output)

        # Filter by confidence
        p3_det_filtered = [d for d in p3_det if d["score"] >= args.conf_threshold]
        bl_det_filtered = [d for d in bl_det if d["score"] >= args.conf_threshold]

        # GT
        ds_entry = dataset_dicts[idx]
        gt_polys = load_gt_polygons(ds_entry)

        # Heatmaps: baseline = projected P3 feat, P3 = after CLIP fusion
        bl_hm = bl_hooks.features.get("p3")
        p3_hm = p3_hooks.features.get("p3")

        if bl_hm is not None:
            bl_hm = bl_hm.norm(dim=1).squeeze(0).cpu().numpy()
        if p3_hm is not None:
            p3_hm = p3_hm.norm(dim=1).squeeze(0).cpu().numpy()

        # Save comparison
        save_name = f"{image_id}.jpg"
        save_path = os.path.join(args.output, save_name)
        create_2x2_figure(orig_img, p3_det_filtered, gt_polys, bl_hm, p3_hm, image_id, save_path)

        summary.append({
            "image_id": image_id,
            "p3_num_det": len(p3_det_filtered),
            "bl_num_det": len(bl_det_filtered),
            "gt_num": len(gt_polys),
            "p3_avg_conf": float(np.mean([d["score"] for d in p3_det_filtered])) if p3_det_filtered else 0,
            "bl_avg_conf": float(np.mean([d["score"] for d in bl_det_filtered])) if bl_det_filtered else 0,
        })

    # Clean handles
    if bl_handle:
        bl_handle.remove()
    if p3_handle:
        p3_handle.remove()

    # Save summary
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print stats
    n = len(summary)
    if n > 0:
        avg_p = np.mean([s["p3_num_det"] for s in summary])
        avg_b = np.mean([s["bl_num_det"] for s in summary])
        avg_g = np.mean([s["gt_num"] for s in summary])
        avg_pc = np.mean([s["p3_avg_conf"] for s in summary])
        avg_bc = np.mean([s["bl_avg_conf"] for s in summary])

        print(f"\n{'='*60}")
        print(f"Aggregate stats ({n} images):")
        print(f"  {'':>20}  {'Baseline':>10}  {'P3-DRTP':>10}  {'GT':>10}")
        print(f"  {'Avg detections':>20}  {avg_b:>10.1f}  {avg_p:>10.1f}  {avg_g:>10.1f}")
        print(f"  {'Avg confidence':>20}  {avg_bc:>10.3f}  {avg_pc:>10.3f}")
        print(f"  {'Δ detections':>20}  {'':>10}  {avg_p - avg_b:>+10.1f}  ({'+more' if avg_p > avg_b else 'fewer'} than baseline)")
        print(f"  {'Δ confidence':>20}  {'':>10}  {avg_pc - avg_bc:>+10.3f}")
        print(f"\n  Output: {args.output}/")
        print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
