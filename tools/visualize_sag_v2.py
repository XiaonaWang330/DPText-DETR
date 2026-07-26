"""
Scale-Aware Gate Visualization Script v2 (vis_baseline style)
===============================================================
Generates 2x2 single-model visualization for SAG-DRTP on CTW1500 test set.

Fixed issues vs v1:
  1. GT: Use 'polys' field (16-point curve polygon) instead of COCO bbox fallback
  2. Feature Activation: Show pre-CLIP FPN-P4 (matches baseline vis) + post-CLIP fusion + GT

Output format:
  +-------------------+-------------------+
  | SAG Detections    | Ground Truth      |
  +-------------------+-------------------+
  | Feature Activation| Heatmap + GT polys|
  | (FPN-P4 no CLIP)  | (After CLIP+P4)  |
  +-------------------+-------------------+

Usage:
  python tools/visualize_sag_v2.py \
      --config configs/DPText_DETR/CTW1500/R_50_poly_drtp_scale_aware_A100.yaml \
      --weights output/r_50_poly/ctw1500/Scale-Aware Gate/model_best.pth \
      --output output/vis_sag_v2 \
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

# ── Color palette ──
POLYGON_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
                  (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                  (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
                  (128, 0, 255), (0, 128, 255)]


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


def load_gt_polys_from_dataset(dataset_entry):
    """
    Extract GT polygons from CTW1500 dataset dict.
    Uses 'polys' field (16-point curve polygon), NOT COCO segmentation.
    """
    polygons = []
    annotations = dataset_entry.get("annotations", [])
    for ann in annotations:
        polys = ann.get("polys", None)
        if polys is not None and len(polys) >= 6:
            # polys is a flat list: [x0,y0,x1,y1,...,x15,y15] (16 points, 32 values)
            pts = np.array(polys, dtype=np.float32).reshape(-1, 2)
            polygons.append({"polygon": pts, "score": 1.0})
        else:
            # Fallback: try standard COCO segmentation
            segm = ann.get("segmentation", [])
            if segm:
                for seg in segm:
                    coords = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    polygons.append({"polygon": coords, "score": 1.0})
            else:
                # Last resort: bbox
                x1, y1, w_box, h_box = ann.get("bbox", [0, 0, 0, 0])
                bbox_poly = np.array([[x1, y1], [x1 + w_box, y1],
                                      [x1 + w_box, y1 + h_box], [x1, y1 + h_box]],
                                    dtype=np.float32)
                polygons.append({"polygon": bbox_poly, "score": 1.0})
    return polygons


def draw_polygons(image, polygons, color=None, thickness=2):
    """Draw polygon detections on image (BGR)."""
    vis = image.copy()
    for i, poly_data in enumerate(polygons):
        if isinstance(poly_data, dict):
            poly = poly_data["polygon"]
        else:
            poly = poly_data
        c = color if color else POLYGON_COLORS[i % len(POLYGON_COLORS)]
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=c, thickness=thickness)
    return vis


def heatmap_overlay(image_bgr, heatmap_small, alpha=0.45):
    """Overlay heatmap (small feature map) on BGR image."""
    h, w = image_bgr.shape[:2]
    hm = cv2.resize(heatmap_small, (w, h))
    hm_min, hm_max = hm.min(), hm.max()
    if hm_max > hm_min:
        hm_norm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm_norm = np.zeros_like(hm)
    hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, hm_color, alpha, 0)


def raw_heatmap_rgb(heatmap_small, target_size):
    """Create a raw heatmap (no overlay) resized to target_size, return RGB."""
    h, w = target_size
    hm = cv2.resize(heatmap_small, (w, h))
    hm_min, hm_max = hm.min(), hm.max()
    if hm_max > hm_min:
        hm_norm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm_norm = np.zeros_like(hm)
    hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)


def feature_to_heatmap(feature_tensor):
    """
    Convert a 4D feature tensor [1, C, H, W] to a 2D heatmap by taking
    L2 norm across channel dimension.
    """
    if feature_tensor is None:
        return None
    t = feature_tensor
    if t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    elif t.ndim == 3:
        pass
    else:
        return None
    # L2 norm along channel dim
    hm = t.norm(dim=0).cpu().numpy()
    return hm


def create_2x2_figure(orig_bgr, det_polys, gt_polys,
                      pre_heatmap, post_heatmap,
                      image_id, save_path):
    """Create and save 2x2 single-model visualization."""
    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = orig_bgr.shape[:2]

    # Top-left: Detections
    det_panel = cv2.cvtColor(
        draw_polygons(orig_bgr, det_polys, thickness=2),
        cv2.COLOR_BGR2RGB,
    )

    # Top-right: Ground Truth (green, thicker for visibility of curve polys)
    gt_panel = cv2.cvtColor(
        draw_polygons(orig_bgr, gt_polys, color=(0, 255, 0), thickness=2),
        cv2.COLOR_BGR2RGB,
    )

    # Bottom-left: Pre-CLIP Feature Activation (FPN-P4, no CLIP influence)
    if pre_heatmap is not None:
        feat_panel = raw_heatmap_rgb(pre_heatmap, (h_img, w_img))
        # Overlay GT for reference
        feat_panel_bgr = cv2.cvtColor(feat_panel, cv2.COLOR_RGB2BGR)
        feat_panel_bgr = draw_polygons(feat_panel_bgr, gt_polys, color=(0, 255, 0), thickness=1)
        feat_panel = cv2.cvtColor(feat_panel_bgr, cv2.COLOR_BGR2RGB)
    else:
        feat_panel = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    # Bottom-right: Post-CLIP Fusion Heatmap + GT polys (overlay on image)
    if post_heatmap is not None:
        hm_gt = heatmap_overlay(orig_bgr, post_heatmap)
        hm_gt = draw_polygons(hm_gt, gt_polys, color=(0, 255, 0), thickness=1)
        hm_gt = cv2.cvtColor(hm_gt, cv2.COLOR_BGR2RGB)
    else:
        hm_gt = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    titles = [
        f"SAG Detections ({len(det_polys)} boxes)",
        f"Ground Truth ({len(gt_polys)} texts, 16-pt)",
        "Feature Activation (FPN-P4, no CLIP) + GT",
        "Feature Activation (After SAG CLIP Fusion, P4) + GT",
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
    parser = argparse.ArgumentParser(description="SAG Visualization v2")
    parser.add_argument("--config", default="configs/DPText_DETR/CTW1500/R_50_poly_drtp_scale_aware_A100.yaml")
    parser.add_argument("--weights", default="output/r_50_poly/ctw1500/Scale-Aware Gate/model_best.pth")
    parser.add_argument("--output", default="output/vis_sag_v2")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Step 1] Building SAG model...")

    sag_cfg = setup_cfg(args.config, args.weights)
    sag_model = d2_build_model(sag_cfg)
    sag_checkpointer = DetectionCheckpointer(sag_model)
    sag_checkpointer.resume_or_load(args.weights, resume=False)
    sag_model.eval()
    sag_model.to(device)

    dataset_name = sag_cfg.DATASETS.TEST[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)
    mapper = DatasetMapperWithBasis(sag_cfg, is_train=False)
    data_loader = build_detection_test_loader(dataset_dicts, mapper=mapper)
    print(f"  Dataset: {dataset_name} ({len(dataset_dicts)} images)")

    # ── Setup hooks ──
    store_pre = FeatureHooks()
    store_post = FeatureHooks()

    # Hook 1: input_proj.0 output => pre-CLIP FPN-P4 feature
    def hook_pre(m, i, o):
        store_pre.features["fpn_p4"] = o.detach()

    # Hook 2: clip_adapter output => post-CLIP-fusion feature list
    def hook_post(m, i, o):
        if isinstance(o, (list, tuple)):
            store_post.features["fused_p4"] = o[0].detach()
        else:
            store_post.features["fused_p4"] = o.detach()

    handle_pre = None
    handle_post = None

    for name, mod in sag_model.named_modules():
        if name == "dptext_detr.input_proj.0":
            handle_pre = mod.register_forward_hook(hook_pre)
            print(f"  Hooked pre-CLIP: {name}")
        if name == "dptext_detr.clip_adapter":
            handle_post = mod.register_forward_hook(hook_post)
            print(f"  Hooked post-CLIP: {name}")

    if handle_pre is None or handle_post is None:
        print("  [WARN] Some hooks not registered. Check module names.")
        if handle_pre is None:
            print("    -> input_proj.0 NOT FOUND (pre-CLIP)")

    # ── Inference loop ──
    max_images = min(args.num_images, len(data_loader))
    print(f"\n[Step 2] Processing {max_images} images...")

    summary = []
    for idx, batch in enumerate(tqdm(data_loader, desc="Visualizing", total=max_images)):
        if idx >= args.num_images:
            break

        data_dict = batch[0] if isinstance(batch, list) else batch
        image_path = data_dict.get("file_name", dataset_dicts[idx]["file_name"])
        image_id = os.path.basename(image_path).split(".")[0]
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            continue

        model_input = [{
            "image": data_dict["image"].to(device),
            "height": data_dict["height"],
            "width": data_dict["width"],
        }]

        store_pre.clear()
        store_post.clear()

        with torch.no_grad():
            sag_output = sag_model(model_input)

        # Parse detections
        def parse_output(output):
            results = []
            if not output:
                return results
            inst = output[0].get("instances")
            if inst is None or not hasattr(inst, "polygons") or len(inst) == 0:
                return results
            polygons = inst.polygons
            scores = inst.scores
            for i_poly in range(len(inst)):
                if polygons[i_poly].numel() == 0:
                    continue
                poly = polygons[i_poly].cpu().numpy().reshape(-1, 2)
                s = float(scores[i_poly].cpu())
                results.append({"polygon": poly, "score": s})
            return results

        sag_det = parse_output(sag_output)
        sag_det = [d for d in sag_det if d["score"] >= args.conf_threshold]

        # GT polygons (from 'polys' field now!)
        gt_polys = load_gt_polys_from_dataset(dataset_dicts[idx])

        # Heatmaps
        pre_hm = feature_to_heatmap(store_pre.features.get("fpn_p4"))
        post_hm = feature_to_heatmap(store_post.features.get("fused_p4"))

        save_path = os.path.join(args.output, f"{image_id}.jpg")
        create_2x2_figure(orig_img, sag_det, gt_polys, pre_hm, post_hm, image_id, save_path)

        summary.append({
            "image_id": image_id,
            "sag_num_det": len(sag_det),
            "gt_num": len(gt_polys),
            "sag_avg_conf": float(np.mean([d["score"] for d in sag_det])) if sag_det else 0,
        })

    # Cleanup
    if handle_pre:
        handle_pre.remove()
    if handle_post:
        handle_post.remove()

    # Save summary
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    n = len(summary)
    if n > 0:
        avg_det = np.mean([s["sag_num_det"] for s in summary])
        avg_gt = np.mean([s["gt_num"] for s in summary])
        avg_conf = np.mean([s["sag_avg_conf"] for s in summary])
        print(f"\n{'='*60}")
        print(f"  {n} images -> {args.output}/")
        print(f"  Avg det: {avg_det:.1f} | Avg GT: {avg_gt:.1f} | Avg conf: {avg_conf:.3f}")


if __name__ == "__main__":
    main()
