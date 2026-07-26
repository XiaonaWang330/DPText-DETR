"""
FPS & FLOPs Benchmark for DPText-DETR Text Detection Model

Measurements:
  1. Params (M) - total and trainable parameters
  2. FLOPs (G) - floating-point operations (core forward pass via fvcore)
  3. Forward FPS - dptext_detr.forward() (backbone + transformer + heads)
  4. End-to-end FPS - preprocess + dptext_detr + inference + detector_postprocess

Datasets tested (architecture identical, all tested at fixed 640x640):
  - ICDAR2015: 640x640
  - CTW1500:   640x640
  - TotalText: 640x640

Usage:
  python tools/measure_fps.py

Requirements:
  pip install fvcore
"""

import time
import sys
import os

# Ensure DPText-DETR's adet is found first
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.nn as nn
import numpy as np

import detectron2.utils.comm as comm
from detectron2.config import get_cfg as d2_get_cfg
from detectron2.modeling import build_model
from detectron2.structures import ImageList

from adet.config import get_cfg
from adet.utils.misc import NestedTensor, nested_tensor_from_tensor_list


# ---------------------------------------------------------------------------
# Inline copy of detector_postprocess (avoid circular import from adet.modeling)
# ---------------------------------------------------------------------------

def _detector_postprocess(results, output_height, output_width):
    """Scale predictions to output resolution."""
    scale_x = output_width / results.image_size[1]
    scale_y = output_height / results.image_size[0]

    if results.has("beziers"):
        beziers = results.beziers
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y

    if results.has("polygons"):
        polygons = results.polygons
        polygons[:, 0::2] *= scale_x
        polygons[:, 1::2] *= scale_y

    return results


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_dptext_model(config_path, weights_path, device='cuda'):
    """Build DPText-DETR model from config and load weights."""
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device
    cfg.freeze()

    model = build_model(cfg)
    model.to(device).eval()

    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

    return model, cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_dummy_inputs(batch_size, height, width, device='cuda'):
    """Create dummy inputs in the format expected by DPText-DETR.

    Returns:
        batched_inputs: list of dicts with 'image', 'height', 'width' keys.
    """
    batched_inputs = []
    for _ in range(batch_size):
        img = torch.randn(3, height, width, device=device)
        batched_inputs.append({
            'image': img,
            'height': height,
            'width': width,
        })
    return batched_inputs


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def measure_flops(model, batched_inputs, device='cuda'):
    """Measure FLOPs for the core dptext_detr forward pass.

    Uses a wrapper that converts raw tensors into the ImageList format
    expected by dptext_detr, so fvcore/thop can profile it.
    """
    # Build a pre-normalized single-image tensor
    img = batched_inputs[0]['image'].unsqueeze(0)  # (1, C, H, W)
    normalized = model.normalizer(img)

    class _DPTextWrapper(nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, x):
            # x: (1, 3, H, W) normalized tensor
            images = [x[i] for i in range(x.shape[0])]
            image_list = ImageList.from_tensors(images)
            return self.detector(image_list)['pred_logits'][-1]

    wrapper = _DPTextWrapper(model.dptext_detr)

    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(wrapper, normalized)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total()
    except ImportError:
        pass

    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(wrapper, inputs=(normalized,), verbose=False)
        return flops
    except ImportError:
        print('    [WARN] Neither fvcore nor thop installed.')
        return None


def measure_forward_fps(model, batched_inputs, warmup=50, num_iters=300, device='cuda'):
    """Measure pure forward FPS: dptext_detr (backbone + transformer + heads).

    The model.dptext_detr.forward() accepts a list of tensors, wrapping
    them into a NestedTensor internally.
    """
    # Use preprocess_image to get properly normalized ImageList
    # then extract the individual tensors
    images = model.preprocess_image(batched_inputs)
    # images is ImageList; we can pass it directly to dptext_detr
    # DPText_DETR.forward checks isinstance(samples, (list, torch.Tensor))
    # ImageList is not list/Tensor, so it uses it as-is
    # But MaskedBackbone.forward accesses images.tensor (singular)
    # ImageList has .tensor, so this works!

    with torch.no_grad():
        for _ in range(warmup):
            _ = model.dptext_detr(images)
            if device == 'cuda':
                torch.cuda.synchronize()

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(num_iters):
            _ = model.dptext_detr(images)
            if device == 'cuda':
                torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / num_iters) * 1000
    avg_fps = num_iters / elapsed
    return avg_fps, avg_latency_ms


def measure_e2e_fps(model, batched_inputs, warmup=30, num_iters=100, device='cuda'):
    """Measure end-to-end FPS with breakdown."""
    # Warmup
    with torch.no_grad():
        for i in range(min(warmup, 5)):
            _ = model(batched_inputs)
            if device == 'cuda':
                torch.cuda.synchronize()
            print(f'\r    E2E warmup: {i + 1}/{min(warmup, 5)}', end='', flush=True)
    print()

    # 1) End-to-end
    if device == 'cuda':
        torch.cuda.synchronize()
    e2e_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(batched_inputs)
            if device == 'cuda':
                torch.cuda.synchronize()
    e2e_elapsed = time.perf_counter() - e2e_start

    # 2) Preprocess only
    if device == 'cuda':
        torch.cuda.synchronize()
    pre_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model.preprocess_image(batched_inputs)
            if device == 'cuda':
                torch.cuda.synchronize()
    pre_elapsed = time.perf_counter() - pre_start

    # 3) Core forward (dptext_detr)
    images = model.preprocess_image(batched_inputs)
    if device == 'cuda':
        torch.cuda.synchronize()
    fwd_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model.dptext_detr(images)
            if device == 'cuda':
                torch.cuda.synchronize()
    fwd_elapsed = time.perf_counter() - fwd_start

    # 4) Inference + postprocess
    output = model.dptext_detr(images)
    ctrl_point_cls = output["pred_logits"]
    ctrl_point_coord = output["pred_ctrl_points"]
    if device == 'cuda':
        torch.cuda.synchronize()
    post_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            results = model.inference(ctrl_point_cls, ctrl_point_coord, images.image_sizes)
            for r, bi, isz in zip(results, batched_inputs, images.image_sizes):
                h = bi.get("height", isz[0])
                w = bi.get("width", isz[1])
                _detector_postprocess(r, h, w)
            if device == 'cuda':
                torch.cuda.synchronize()
    post_elapsed = time.perf_counter() - post_start

    return dict(
        e2e_fps=(num_iters / e2e_elapsed),
        e2e_ms=(e2e_elapsed / num_iters) * 1000,
        pre_ms=(pre_elapsed / num_iters) * 1000,
        fwd_ms=(fwd_elapsed / num_iters) * 1000,
        post_ms=(post_elapsed / num_iters) * 1000,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print('WARNING: CUDA not available, falling back to CPU')
        device = 'cpu'
    else:
        device = 'cuda'
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA: {torch.version.cuda}  PyTorch: {torch.__version__}')
    print()

    config_path = 'configs/DPText_DETR/ICDAR2015/R_50_poly.yaml'
    weights_path = 'pretrain.pth'
    print(f'Config: {config_path}')
    print(f'Weights: {weights_path}\n')

    print('Building model ...')
    model, cfg = build_dptext_model(config_path, weights_path, device)
    print('Done.\n')

    total_params, trainable_params = count_parameters(model)
    print(f'Parameters: {total_params / 1e6:.2f}M total, '
          f'{trainable_params / 1e6:.2f}M trainable')
    print(f'Backbone: ResNet-50')
    print(f'Transformer: enc={cfg.MODEL.TRANSFORMER.ENC_LAYERS}, '
          f'dec={cfg.MODEL.TRANSFORMER.DEC_LAYERS}, '
          f'dim={cfg.MODEL.TRANSFORMER.HIDDEN_DIM}, '
          f'heads={cfg.MODEL.TRANSFORMER.NHEADS}, '
          f'queries={cfg.MODEL.TRANSFORMER.NUM_QUERIES}')
    print(f'Ctrl points: {cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS}  '
          f'EPQM: {cfg.MODEL.TRANSFORMER.EPQM}  '
          f'EFSA: {cfg.MODEL.TRANSFORMER.EFSA}')
    print()

    warmup = 50
    num_iters = 300

    test_configs = [
        ('ICDAR2015',   640, 640),
        ('CTW1500',     640, 640),
        ('TotalText',   640, 640),
    ]

    print('=' * 110)
    print(f'  Warmup: {warmup} | Measure: {num_iters} | Batch: 1 | FP32')
    print('=' * 110)

    flops_cache = {}
    results = []

    for name, h, w in test_configs:
        print(f'\n--- {name}  (Input: {h}x{w}) ---')

        batched_inputs = make_dummy_inputs(1, h, w, device)

        # FLOPs (cached by resolution)
        flops = None
        res_key = (h, w)
        if res_key not in flops_cache:
            try:
                flops = measure_flops(model, batched_inputs, device)
                if flops is not None:
                    flops_cache[res_key] = flops
                    print(f'  FLOPs: {flops / 1e9:.2f}G')
            except Exception as e:
                print(f'  [WARN] FLOPs failed: {e}')
        else:
            flops = flops_cache[res_key]
            print(f'  FLOPs: {flops / 1e9:.2f}G (cached)')

        # Forward FPS
        print('  Measuring Forward FPS (dptext_detr) ...')
        try:
            fwd_fps, fwd_lat = measure_forward_fps(
                model, batched_inputs, warmup=warmup, num_iters=num_iters, device=device
            )
            print(f'  Forward FPS: {fwd_fps:.1f}  |  Latency: {fwd_lat:.2f} ms')
        except Exception as e:
            fwd_fps, fwd_lat = 0.0, 0.0
            print(f'  [WARN] Forward FPS failed: {e}')

        # E2E FPS
        print('  Measuring End-to-End FPS ...')
        try:
            e2e = measure_e2e_fps(
                model, batched_inputs, warmup=warmup, num_iters=num_iters, device=device
            )
            print(f'  E2E FPS: {e2e["e2e_fps"]:.1f}  |  E2E Latency: {e2e["e2e_ms"]:.2f} ms')
            print(f'    pre={e2e["pre_ms"]:.2f}ms  fwd={e2e["fwd_ms"]:.2f}ms  '
                  f'post={e2e["post_ms"]:.2f}ms')
        except Exception as e:
            e2e = dict(e2e_fps=0, e2e_ms=0, pre_ms=0, fwd_ms=0, post_ms=0)
            print(f'  [WARN] E2E FPS failed: {e}')

        results.append((name, h, w, flops, fwd_fps, fwd_lat,
                        e2e['e2e_fps'], e2e['e2e_ms'],
                        e2e['pre_ms'], e2e['post_ms']))

        torch.cuda.empty_cache()

    # ---- Summary ----
    gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print('\n\n' + '=' * 110)
    print(f'{">>> SUMMARY (Batch=1, FP32, " + gpu_name + ") <<<":^110}')
    print('=' * 110)
    print(f'{"Dataset":<16} {"Res":<12} {"FLOPs(G)":<10} {"FPS(fwd)":<10} '
          f'{"ms(fwd)":<9} {"FPS(e2e)":<10} {"ms(e2e)":<9} '
          f'{"ms(pre)":<9} {"ms(post)":<9} {"Params(M)":<10}')
    print('-' * 110)
    for name, h, w, flops, fwd_fps, fwd_lat, e2e_fps, e2e_ms, pre_ms, post_ms in results:
        flops_str = f'{flops / 1e9:.2f}' if flops is not None else 'N/A'
        e2e_fps_str = f'{e2e_fps:.1f}' if e2e_fps > 0 else 'N/A'
        e2e_ms_str = f'{e2e_ms:.2f}' if e2e_ms > 0 else 'N/A'
        pre_str = f'{pre_ms:.2f}' if pre_ms > 0 else 'N/A'
        post_str = f'{post_ms:.2f}' if post_ms > 0 else 'N/A'
        print(f'{name:<16} {h}x{w:<8} {flops_str:<10} {fwd_fps:<10.1f} '
              f'{fwd_lat:<9.2f} {e2e_fps_str:<10} {e2e_ms_str:<9} '
              f'{pre_str:<9} {post_str:<9} {total_params/1e6:<10.2f}')
    print('=' * 110)

    print()
    print('Legend:')
    print('  FPS(fwd)  = dptext_detr.forward (backbone + transformer + heads)')
    print('  FPS(e2e)  = preprocess + dptext_detr + inference + detector_postprocess')
    print('  ms(pre)   = preprocess_image (normalize + pad to ImageList)')
    print('  ms(post)  = inference (threshold + coord scale) + detector_postprocess')
    print('  All results at batch=1, FP32, single GPU')


if __name__ == '__main__':
    main()
