"""
step-0 identity check: SA-CAPR (rho=0) vs P3-DRTP.
Runs first training batch through both models at step 0, compares loss.

Usage:
    python tools/debug_step0_identity.py
"""
import sys
import os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.data import build_detection_train_loader

from adet.config import get_cfg as adet_get_cfg
from adet.data.dataset_mapper import DatasetMapperWithBasis
import adet.data  # trigger dataset registration


def setup_cfg(config_file):
    cfg = adet_get_cfg()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    default_setup(cfg, {})
    return cfg


def main():
    # ── Configs ──
    config_sa = "configs/DPText_DETR/CTW1500/R_50_poly_sa_capr_p3_A100.yaml"
    config_pt = "configs/DPText_DETR/CTW1500/R_50_poly_mm_p3_A100.yaml"

    # Check config existence
    for cfg_path in [config_sa, config_pt]:
        full = os.path.join(_PROJ_ROOT, cfg_path)
        if not os.path.exists(full):
            print(f"WARNING: Config not found: {full}")
            return

    print("=" * 70)
    print("Step-0 Identity Check: SA-CAPR (rho=0) vs P3-DRTP")
    print("=" * 70)

    # ── Fix random seeds ──
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── Setup configs ──
    cfg_sa = setup_cfg(config_sa)
    cfg_pt = setup_cfg(config_pt)

    # ── Build models ──
    print("\n[1] Building SA-CAPR model (rho=0)...")
    model_sa = build_model(cfg_sa).cuda()
    model_sa.train()

    print("[2] Building P3-DRTP model (use_text_gate=False)...")
    model_pt = build_model(cfg_pt).cuda()
    model_pt.train()

    # ── Check rho / text_gate_scale ──
    clip_adapter_sa = model_sa.dptext_detr.clip_adapter
    clip_adapter_pt = model_pt.dptext_detr.clip_adapter
    rho_val = clip_adapter_sa.text_gate_scale.item()
    print(f"\n[3] SA-CAPR rho = {rho_val:.6f} (expected 0.0)")

    # ── Check that GateNet weights match (same seed → same init) ──
    gate_w_sa = clip_adapter_sa.level_gate_nets[0].state_dict()
    gate_w_pt = clip_adapter_pt.level_gate_nets[0].state_dict()

    print("\n[4] Comparing GateNet weights (should be identical with same seed):")
    all_match = True
    for k in sorted(set(gate_w_sa.keys()) | set(gate_w_pt.keys())):
        if k not in gate_w_sa:
            print(f"    SA-CAPR missing: {k}")
            all_match = False
        elif k not in gate_w_pt:
            print(f"    P3-DRTP missing: {k}")
            all_match = False
        else:
            diff = (gate_w_sa[k] - gate_w_pt[k]).abs().max().item()
            status = "OK" if diff < 1e-6 else f"MISMATCH diff={diff:.2e}"
            if diff >= 1e-6:
                all_match = False
            print(f"    {status}: {k}")

    if not all_match:
        print("\n    *** GateNet weights differ! Different random init? ***")
        print("    *** This explains the 37 vs 33 loss gap. ***")

    # ── Load one training batch ──
    print("\n[5] Loading one training batch...")
    mapper = DatasetMapperWithBasis(cfg_sa, is_train=True)
    loader_sa = build_detection_train_loader(cfg_sa, mapper=mapper)
    data_iter = iter(loader_sa)
    batch = next(data_iter)

    # Move batch to GPU
    for d in batch:
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    d[k] = v.cuda()

    # ── Forward both on same batch ──
    print("[6] Running SA-CAPR forward...")
    with torch.enable_grad():
        loss_dict_sa = model_sa(batch)
        loss_sa = sum(v for v in loss_dict_sa.values())

    print("[7] Running P3-DRTP forward...")
    with torch.enable_grad():
        loss_dict_pt = model_pt(batch)
        loss_pt = sum(v for v in loss_dict_pt.values())

    # ── Compare ──
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  SA-CAPR total_loss  = {loss_sa.item():.4f}")
    print(f"  P3-DRTP total_loss  = {loss_pt.item():.4f}")
    print(f"  Difference          = {(loss_sa - loss_pt).item():.4f}")
    print(f"  Relative diff       = {abs(loss_sa - loss_pt).item() / loss_pt.item() * 100:.2f}%")

    print("\nPer-loss breakdown:")
    for k in sorted(set(loss_dict_sa.keys()) | set(loss_dict_pt.keys())):
        v_sa = loss_dict_sa.get(k, torch.tensor(0.0)).item()
        v_pt = loss_dict_pt.get(k, torch.tensor(0.0)).item()
        d = v_sa - v_pt
        marker = " ***" if abs(d) > 0.01 else ""
        print(f"  {k:30s}  SA={v_sa:.4f}  PT={v_pt:.4f}  Δ={d:.4f}{marker}")

    if abs(loss_sa.item() - loss_pt.item()) < 0.01:
        print("\n>>> PASS: SA-CAPR (rho=0) matches P3-DRTP at step 0. <<<")
        print(">>> The 37 vs 33 difference is due to: different seeds OR different pretrained weights. <<<")
    else:
        print("\n>>> FAIL: SA-CAPR (rho=0) does NOT match P3-DRTP at step 0! <<<")
        print(">>> There may be a code bug in the SA-CAPR path. <<<")


if __name__ == "__main__":
    main()
