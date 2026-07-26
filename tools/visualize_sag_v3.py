"""
Scale-Aware Gate Visualization v3 (vis_baseline style, FIXED)
===============================================================
Fixed vs v2:
  1. GT: dataset key is 'polygons' (NOT 'polys') → 16-pt curve polygons now
  2. Feature Activation: channel-MEAN heatmap (NOT L2 norm) → visible difference
  3. Level: P3 (CLIP_ACTIVE_LEVELS=[0]) correctly labeled
  4. Diff map: bottom-right shows (post - pre) to highlight what CLIP changes

Output format (2×2):
  +-----------------------------+------------------------------+
  | SAG Detections              | Ground Truth (curve polys)   |
  +-----------------------------+------------------------------+
  | Pre-CLIP P3 Feature + GT    | CLIP Delta |post-pre| + GT   |
  | channel-mean heatmap overlay| magnitude heatmap overlay    |
  +-----------------------------+------------------------------+

Usage:
  python tools/visualize_sag_v3.py \
      --config configs/DPText_DETR/CTW1500/R_50_poly_drtp_scale_aware_A100.yaml \
      --weights output/r_50_poly/ctw1500/Scale-Aware Gate/model_best.pth \
      --output output/vis_sag_v3 \
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

from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.modeling import build_model as d2_build_model
from detectron2.checkpoint import DetectionCheckpointer

from adet.config import get_cfg as adet_get_cfg
from adet.data import DatasetMapperWithBasis

# ── Color palette ──
POLYGON_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255),
]


def setup_cfg(config_file, weights_file):
    cfg = adet_get_cfg()
    _merge_from_file_utf8(cfg, config_file)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


def _merge_from_file_utf8(cfg, filename):
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
    def __init__(self):
        self.features = {}

    def clear(self):
        self.features.clear()


# ──────── GT: polygons (FIXED) ────────
def load_gt_polygons(dataset_entry):
    """
    Extract GT polygons from CTW1500 dataset dict.
    
    CTW1500 JSON format: 'polygons' is a flat list of 32 floats per annotation
    (16 pts × 2), same for every text instance.
    """
    polygons = []
    annotations = dataset_entry.get("annotations", [])
    for ann in annotations:
        poly_data = ann.get("polygons", None)
        if poly_data is not None and len(poly_data) >= 6:
            # Flat list: [x0,y0, x1,y1, ..., x15,y15]
            # Check first element to distinguish flat list vs list-of-lists
            if isinstance(poly_data[0], (int, float)):
                pts = np.array(poly_data, dtype=np.float32).reshape(-1, 2)
                polygons.append({"polygon": pts, "score": 1.0})
            else:
                # list of polygons (each flat)
                for poly_flat in poly_data:
                    if len(poly_flat) >= 6:
                        pts = np.array(poly_flat, dtype=np.float32).reshape(-1, 2)
                        polygons.append({"polygon": pts, "score": 1.0})
        else:
            # Fallback: COCO segmentation
            segm = ann.get("segmentation", [])
            if segm:
                for seg in segm:
                    coords = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    polygons.append({"polygon": coords, "score": 1.0})
            else:
                # Last resort: bbox
                x1, y1, w_box, h_box = ann.get("bbox", [0, 0, 0, 0])
                bbox_poly = np.array(
                    [[x1, y1], [x1 + w_box, y1],
                     [x1 + w_box, y1 + h_box], [x1, y1 + h_box]],
                    dtype=np.float32,
                )
                polygons.append({"polygon": bbox_poly, "score": 1.0})
    return polygons


def draw_polygons(image, polygons, color=None, thickness=2):
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


def heatmap_rgb(heatmap_2d, target_hw, colormap=cv2.COLORMAP_JET):
    """Resize 2D heatmap to target size and return RGB."""
    h, w = target_hw
    hm = cv2.resize(heatmap_2d, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_min, hm_max = hm.min(), hm.max()
    if hm_max > hm_min:
        hm_norm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm_norm = np.zeros_like(hm)
    hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), colormap)
    return cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)


def heatmap_overlay(image_bgr, heatmap_2d, alpha=0.45, vmin=None, vmax=None):
    """Overlay heatmap on BGR image. Optional shared vmin/vmax for comparison."""
    h, w = image_bgr.shape[:2]
    hm = cv2.resize(heatmap_2d, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_min = vmin if vmin is not None else hm.min()
    hm_max = vmax if vmax is not None else hm.max()
    if hm_max > hm_min:
        hm_norm = (hm - hm_min) / (hm_max - hm_min)
        hm_norm = np.clip(hm_norm, 0.0, 1.0)
    else:
        hm_norm = np.zeros_like(hm)
    hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, hm_color, alpha, 0)


def tensor_to_channel_mean_heatmap(tensor):
    """Convert [1, C, H, W] tensor to 2D heatmap: MEAN across channels."""
    if tensor is None:
        return None
    t = tensor
    if t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    elif t.ndim == 3:
        pass
    else:
        return None
    return t.mean(dim=0).cpu().numpy()


def tensor_to_channel_std_heatmap(tensor):
    """Convert [1, C, H, W] tensor to 2D heatmap: STD across channels."""
    if tensor is None:
        return None
    t = tensor
    if t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    elif t.ndim == 3:
        pass
    else:
        return None
    return t.std(dim=0).cpu().numpy()


# ──────── 2×2 Figure ────────
def create_2x2_figure(orig_bgr, det_polys, gt_polys,
                      pre_hm, post_hm, diff_hm,
                      image_id, save_path):
    """
    2×2 layout:
      Top-left:  Detections
      Top-right: Ground Truth
      Bottom-left: Pre-CLIP channel-mean activation (P3) overlaid on image + GT
      Bottom-right: CLIP Delta magnitude |post-pre| overlaid on image + GT
    """
    h_img, w_img = orig_bgr.shape[:2]

    # Top-left: SAG Detections
    det_panel = cv2.cvtColor(
        draw_polygons(orig_bgr, det_polys, thickness=2),
        cv2.COLOR_BGR2RGB,
    )

    # Top-right: Ground Truth (green)
    gt_panel = cv2.cvtColor(
        draw_polygons(orig_bgr, gt_polys, color=(0, 255, 0), thickness=2),
        cv2.COLOR_BGR2RGB,
    )

    # Bottom-left: Pre-CLIP P3 feature activation overlaid on image + GT
    if pre_hm is not None:
        pre_panel = heatmap_overlay(orig_bgr, pre_hm, alpha=0.45)
        pre_panel = draw_polygons(pre_panel, gt_polys, color=(0, 255, 0), thickness=1)
        pre_panel = cv2.cvtColor(pre_panel, cv2.COLOR_BGR2RGB)
    else:
        pre_panel = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    # Bottom-right: CLIP Delta magnitude |post-pre| overlaid on image + GT
    if diff_hm is not None:
        diff_panel = heatmap_overlay(orig_bgr, diff_hm, alpha=0.55)
        diff_panel = draw_polygons(diff_panel, gt_polys, color=(0, 255, 0), thickness=1)
        diff_panel = cv2.cvtColor(diff_panel, cv2.COLOR_BGR2RGB)
    else:
        diff_panel = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    titles = [
        f"SAG Detections ({len(det_polys)} boxes)",
        f"Ground Truth ({len(gt_polys)} texts, 16-pt)",
        "Pre-CLIP P3 Feature (channel mean) + GT",
        "CLIP Delta |post-pre| + GT  (yellow=changed)",
    ]
    images = [det_panel, gt_panel, pre_panel, diff_panel]

    for ax, img_t, title in zip(axes, images, titles):
        ax.imshow(img_t)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle(f"Scale-Aware Gate — {image_id}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SAG Visualization v3 (FIXED GT + channel-mean)")
    parser.add_argument("--config", default="configs/DPText_DETR/CTW1500/R_50_poly_drtp_scale_aware_A100.yaml")
    parser.add_argument("--weights", default="output/r_50_poly/ctw1500/Scale-Aware Gate/model_best.pth")
    parser.add_argument("--output", default="output/vis_sag_v3")
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

    # ── Hooks: pre-CLIP (input_proj.0) + post-CLIP (clip_adapter output[0]) ──
    store = FeatureHooks()

    def hook_pre(m, i, o):
        store.features["pre_p3"] = o.detach()

    def hook_post(m, i, o):
        if isinstance(o, (list, tuple)):
            t = o[0]
            # Handle NestedTensor wrapping if present
            store.features["post_p3"] = (t.tensor if hasattr(t, 'tensor') else t).detach()
        else:
            store.features["post_p3"] = (o.tensor if hasattr(o, 'tensor') else o).detach()

    def hook_pre_full(m, i):
        """Capture input to clip_adapter (before CLIP fusion) for level 0."""
        # i[1] is the srcs list; i[1][0] is level 0 tensor or NestedTensor
        srcs = i[1]
        t = srcs[0]
        store.features["pre_p3_clip"] = (t.tensor if hasattr(t, 'tensor') else t).detach()

    handle_pre = None
    handle_post = None
    handle_pre_full = None
    for name, mod in sag_model.named_modules():
        if name == "dptext_detr.input_proj.0":
            handle_pre = mod.register_forward_hook(hook_pre)
            print("  Hooked pre-CLIP (proj): dptext_detr.input_proj.0")
        if name == "dptext_detr.clip_adapter":
            handle_post = mod.register_forward_hook(hook_post)
            handle_pre_full = mod.register_forward_pre_hook(hook_pre_full)
            print("  Hooked clip_adapter (pre-hook + post-hook)")

    # ── Run inference ──
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

        store.clear()
        with torch.no_grad():
            sag_output = sag_model(model_input)

        # Parse detections
        inst = sag_output[0].get("instances") if sag_output else None
        sag_det = []
        if inst is not None and hasattr(inst, "polygons") and len(inst) > 0:
            polys = inst.polygons
            scores = inst.scores
            for i_poly in range(len(inst)):
                if polys[i_poly].numel() == 0:
                    continue
                poly = polys[i_poly].cpu().numpy().reshape(-1, 2)
                s = float(scores[i_poly].cpu())
                if s >= args.conf_threshold:
                    sag_det.append({"polygon": poly, "score": s})

        # GT — Fixed: uses 'polygons' key
        gt_polys = load_gt_polygons(dataset_dicts[idx])

        # Heatmaps: channel-mean
        # pre_p3 = input_proj.0 output (raw projected P3)
        # pre_p3_clip = exact input to clip_adapter (may differ: NestedTensor wrapping)
        pre_t = store.features.get("pre_p3")
        pre_clip_t = store.features.get("pre_p3_clip")
        post_t = store.features.get("post_p3")

        pre_hm = tensor_to_channel_mean_heatmap(pre_clip_t if pre_clip_t is not None else pre_t)
        post_hm = tensor_to_channel_mean_heatmap(post_t)
        diff_hm = None

        if pre_clip_t is not None and post_t is not None:
            t_pre = pre_clip_t
            t_post = post_t
            if t_pre.ndim == 4 and t_pre.shape[0] == 1:
                t_pre = t_pre.squeeze(0)
            if t_post.ndim == 4 and t_post.shape[0] == 1:
                t_post = t_post.squeeze(0)
            diff_t = (t_post - t_pre).mean(dim=0).cpu().numpy()
            diff_hm = np.abs(diff_t)
            # Debug stats
            pre_norm = float(t_pre.norm().cpu())
            post_norm = float(t_post.norm().cpu())
            diff_norm = float((t_post - t_pre).norm().cpu())
            ratio = diff_norm / (pre_norm + 1e-8)
            if idx < 5:
                print(f"  [{image_id}] pre_norm={pre_norm:.1f} post_norm={post_norm:.1f} diff_norm={diff_norm:.1f} ratio={ratio:.4f}")

        save_path = os.path.join(args.output, f"{image_id}.jpg")
        create_2x2_figure(orig_img, sag_det, gt_polys, pre_hm, post_hm, diff_hm, image_id, save_path)

        summary.append({
            "image_id": image_id,
            "sag_num_det": len(sag_det),
            "gt_num": len(gt_polys),
            "sag_avg_conf": float(np.mean([d["score"] for d in sag_det])) if sag_det else 0,
        })

    if handle_pre:
        handle_pre.remove()
    if handle_post:
        handle_post.remove()

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
