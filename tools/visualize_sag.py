"""
Scale-Aware Gate Visualization Script (vis_baseline style)
===========================================================
Generates 2x2 single-model visualization for SAG-DRTP on CTW1500 test set.

Output format (matching vis_baseline style):
  +-------------------+-------------------+
  | SAG Detections    | Ground Truth      |
  +-------------------+-------------------+
  | Feature Activation| Heatmap + Boxes   |
  | (After CLIP, P4)  | (overlay + GT)    |
  +-------------------+-------------------+

Usage:
  python tools/visualize_sag.py \
      --config configs/DPText_DETR/CTW1500/R_50_poly_drtp_scale_aware_A100.yaml \
      --weights output/r_50_poly/ctw1500/Scale-Aware Gate/model_best.pth \
      --output output/vis_sag \
      --num-images 200
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


def raw_heatmap_image(heatmap_small, target_size):
    """Create a raw heatmap image (no overlay) resized to target_size."""
    h, w = target_size
    hm = cv2.resize(heatmap_small, (w, h))
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)


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


def create_2x2_figure(orig_bgr, sag_det, gt_polys, sag_heatmap, image_id, save_path):
    """Create and save 2x2 single-model visualization (vis_baseline style)."""
    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = orig_bgr.shape[:2]

    # Top-left: SAG Detections
    det_panel = cv2.cvtColor(
        draw_polygons(orig_bgr, sag_det, thickness=2),
        cv2.COLOR_BGR2RGB,
    )

    # Top-right: Ground Truth
    gt_panel = cv2.cvtColor(
        draw_polygons(orig_bgr, gt_polys, color=(0, 255, 0), thickness=1),
        cv2.COLOR_BGR2RGB,
    )

    # Bottom-left: Feature Activation (raw heatmap, no overlay)
    if sag_heatmap is not None:
        feat_panel = raw_heatmap_image(sag_heatmap, (h_img, w_img))
    else:
        feat_panel = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    # Bottom-right: Heatmap + GT Boxes (overlay)
    if sag_heatmap is not None:
        hm_gt = heatmap_overlay(orig_bgr, sag_heatmap)
        hm_gt = draw_polygons(hm_gt, gt_polys, color=(0, 255, 0), thickness=1)
        hm_gt = cv2.cvtColor(hm_gt, cv2.COLOR_BGR2RGB)
    else:
        hm_gt = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    titles = [
        f"SAG Detections ({len(sag_det)} boxes)",
        f"Ground Truth ({len(gt_polys)} texts)",
        "Feature Activation (After CLIP Fusion, P4)",
        "Heatmap + GT Boxes",
    ]
    images = [det_panel, gt_panel, feat_panel, hm_gt]

    for ax, img_t, title in zip(axes, images, titles):
        ax.imshow(img_t)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle(f"Scale-Aware Gate — {image_id}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SAG Visualization (vis_baseline style)")
    parser.add_argument("--config", default="configs/DPText_DETR/CTW1500/R_50_poly_drtp_scale_aware_A100.yaml",
                        help="SAG config yaml")
    parser.add_argument("--weights", default="output/r_50_poly/ctw1500/Scale-Aware Gate/model_best.pth",
                        help="SAG model checkpoint")
    parser.add_argument("--output", default="output/vis_sag")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("[Step 1] Building SAG model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sag_cfg = setup_cfg(args.config, args.weights)
    sag_model = d2_build_model(sag_cfg)
    sag_checkpointer = DetectionCheckpointer(sag_model)
    sag_checkpointer.resume_or_load(args.weights, resume=False)
    sag_model.eval()
    sag_model.to(device)
    print(f"  SAG model loaded: {args.config}")

    # Get dataset
    dataset_name = sag_cfg.DATASETS.TEST[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)

    # Build data loader
    mapper = DatasetMapperWithBasis(sag_cfg, is_train=False)
    data_loader = build_detection_test_loader(dataset_dicts, mapper=mapper)
    print(f"  Dataset: {dataset_name} ({len(dataset_dicts)} test images)")

    # Feature hooks
    sag_hooks = FeatureHooks()

    # Hook clip_adapter output to capture post-CLIP-fusion P4 feature
    def make_clip_hook(store):
        def _hook(m, i, o):
            # clip_adapter returns a list of tensors; level 0 is P4 (stride 4)
            store.features["p4_fused"] = o[0].detach() if isinstance(o, (list, tuple)) else o.detach()
        return _hook

    sag_handle = None
    for sag_name, sag_mod in sag_model.named_modules():
        if sag_name.endswith("clip_adapter"):
            sag_handle = sag_mod.register_forward_hook(make_clip_hook(sag_hooks))
            print(f"  Hooked: {sag_name}")
            break

    if sag_handle is None:
        print("  [WARN] Could not find clip_adapter; falling back to input_proj[0].")
        def make_res3_hook(store):
            def _hook(m, i, o):
                store.features["p4_fused"] = o.detach() if isinstance(o, torch.Tensor) else o
            return _hook
        for sag_name, sag_mod in sag_model.named_modules():
            if sag_name.endswith("input_proj.0"):
                sag_handle = sag_mod.register_forward_hook(make_res3_hook(sag_hooks))
                print(f"  Hooked fallback: {sag_name}")
                break

    print(f"\n[Step 2] Processing {min(args.num_images, len(dataset_dicts))} images...")

    summary = []
    for idx, batch in enumerate(tqdm(data_loader, desc="Visualizing",
                                      total=min(args.num_images, len(data_loader)))):
        if idx >= args.num_images:
            break

        data_dict = batch[0] if isinstance(batch, list) else batch

        image_path = data_dict.get("file_name", dataset_dicts[idx]["file_name"])
        image_id = os.path.basename(image_path).split(".")[0]

        # Original image
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            print(f"  [SKIP] Cannot read: {image_path}")
            continue

        # Model input
        model_input = [{
            "image": data_dict["image"].to(device),
            "height": data_dict["height"],
            "width": data_dict["width"],
        }]

        # SAG inference
        sag_hooks.clear()
        with torch.no_grad():
            sag_output = sag_model(model_input)

        # Parse outputs
        def parse_output(output):
            results = []
            if not output:
                return results
            inst = output[0].get("instances")
            if inst is None or not hasattr(inst, "polygons") or len(inst) == 0:
                return results
            polygons = inst.polygons
            scores = inst.scores
            for i in range(len(inst)):
                if polygons[i].numel() == 0:
                    continue
                poly = polygons[i].cpu().numpy().reshape(-1, 2)
                s = float(scores[i].cpu())
                results.append({"polygon": poly, "score": s})
            return results

        sag_det = parse_output(sag_output)
        sag_det_filtered = [d for d in sag_det if d["score"] >= args.conf_threshold]

        # GT
        ds_entry = dataset_dicts[idx]
        gt_polys = load_gt_polygons(ds_entry)

        # Heatmap: post-CLIP-fusion P4 feature (L2 norm across channels)
        sag_hm = sag_hooks.features.get("p4_fused")
        if sag_hm is not None:
            sag_hm = sag_hm.norm(dim=1).squeeze(0).cpu().numpy()

        # Save
        save_name = f"{image_id}.jpg"
        save_path = os.path.join(args.output, save_name)
        create_2x2_figure(orig_img, sag_det_filtered, gt_polys, sag_hm, image_id, save_path)

        summary.append({
            "image_id": image_id,
            "sag_num_det": len(sag_det_filtered),
            "gt_num": len(gt_polys),
            "sag_avg_conf": float(np.mean([d["score"] for d in sag_det_filtered])) if sag_det_filtered else 0,
        })

    # Clean handles
    if sag_handle:
        sag_handle.remove()

    # Save summary
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print stats
    n = len(summary)
    if n > 0:
        avg_det = np.mean([s["sag_num_det"] for s in summary])
        avg_gt = np.mean([s["gt_num"] for s in summary])
        avg_conf = np.mean([s["sag_avg_conf"] for s in summary])

        print(f"\n{'='*60}")
        print(f"Aggregate stats ({n} images):")
        print(f"  {'':>20}  {'SAG':>10}  {'GT':>10}")
        print(f"  {'Avg detections':>20}  {avg_det:>10.1f}  {avg_gt:>10.1f}")
        print(f"  {'Avg confidence':>20}  {avg_conf:>10.3f}")
        print(f"\n  Output: {args.output}/")
        print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
