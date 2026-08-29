# -*- coding: utf-8 -*-
"""
CTP 代码恢复冒烟测试 (smoke test)。

目的：在不跑完整 13000 iter 训练的前提下，验证当前代码的 CTP 模块
(TextPrototypeAlignmentHead) 是否已恢复到 88.66 时代的结构。

验证内容：
  1. ta_head 的可训练参数名（应为 adapter.0/adapter.2/projector + alpha/beta/mod_dir
     五个标量，且【没有】 clip_cls_proj / gamma 等 V3/V4/V5 分支）。
  2. ta_head 的模块结构打印（应与 88.66 log.txt 一致：只有 adapter + projector）。
  3. 可选：跑几步，打印 loss_ta 是否正常产生。

用法（在服务器 det 环境）：
  python tools/smoke_test_ctp.py \
      --config-file configs/DPText_DETR/CTW1500/R_50_poly_ctp_t0p20_oldprompt_max_A100.yaml \
      --steps 5

判据（成功 = 代码已恢复到 88.66 结构）：
  - 输出里【出现】 "adapter", "projector", "alpha", "beta", "mod_dir"
  - 输出里【不出现】 "clip_cls_proj", "gamma", "clip_residual", "pixel_margin"
"""

import argparse
import os
import sys

# 确保 adet 优先于 PYTHONPATH 上的其它 adet
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch

from adet.config import get_cfg
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger


def load_cfg(config_file, opts):
    """加载 config（复用 train_net.py 的 utf8 + _BASE_ 递归合并逻辑）。"""
    import yaml
    from detectron2.config import CfgNode

    def _merge_from_file_utf8(cfg, filename):
        with open(filename, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if "_BASE_" in loaded:
            base_dir = os.path.dirname(os.path.abspath(filename))
            base_files = loaded.pop("_BASE_")
            if isinstance(base_files, str):
                base_files = [base_files]
            for bf in base_files:
                bf_path = bf if os.path.isabs(bf) else os.path.join(base_dir, bf)
                _merge_from_file_utf8(cfg, bf_path)
        cfg.merge_from_other_cfg(CfgNode(loaded))

    cfg = get_cfg()
    _merge_from_file_utf8(cfg, config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--steps", type=int, default=0,
                        help="跑几步训练验证 loss_ta（0 = 只检查结构不训练）")
    parser.add_argument("--eval-only", action="store_true",
                        help="只构建模型（不训练），等价 steps=0")
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="额外 config 覆盖，如 MODEL.WEIGHTS ...")
    args = parser.parse_args()

    cfg = load_cfg(args.config_file, args.opts)
    default_setup(cfg, args)
    setup_logger(cfg.OUTPUT_DIR, name="adet")

    # ── 1. 构建模型（detectron2 会打印完整模型结构到 log）──
    from tools.train_net import Trainer
    model = Trainer.build_model(cfg)
    model.eval()

    # ── 2. 提取 ta_head ──
    dptext = getattr(model, "dptext_detr", model)
    ta_head = getattr(dptext, "ta_head", None)

    print("\n" + "=" * 70)
    print("[SMOKE] CTP 结构检查")
    print("=" * 70)

    if ta_head is None:
        print("[SMOKE] [FAIL] 未找到 ta_head —— CLIP_TEXT_PRIOR.ENABLED 是否为 True？")
        sys.exit(1)

    # 可训练参数名（含 buffer 与否分开统计）
    trainable = [n for n, p in ta_head.named_parameters() if p.requires_grad]
    buffers = [n for n, _ in ta_head.named_buffers()]

    print("[SMOKE] ta_head 可训练参数（named_parameters, requires_grad）:")
    for n in trainable:
        print(f"        - {n}")
    print("[SMOKE] ta_head 冻结 buffer:")
    for n in buffers:
        print(f"        - {n}")

    print("\n[SMOKE] ta_head 模块结构:")
    print(ta_head)

    # ── 3. 判据 ──
    joined = " ".join(trainable) + " " + str(ta_head)
    expect_present = ["adapter", "projector", "alpha", "beta", "mod_dir"]
    expect_absent = ["clip_cls_proj", "gamma", "clip_residual", "pixel_margin",
                     "margin_feature", "projector_clip", "clip_vision"]

    print("\n[SMOKE] 判据检查:")
    ok = True
    for token in expect_present:
        present = token in joined
        print(f"        expect-present '{token}': {'PASS' if present else 'FAIL'}")
        ok = ok and present
    for token in expect_absent:
        absent = token not in joined
        print(f"        expect-absent  '{token}': {'PASS' if absent else 'FAIL'}")
        ok = ok and absent

    # ── 4. 可选：跑几步验证 loss_ta ──
    if args.steps > 0:
        print(f"\n[SMOKE] 跑 {args.steps} 步训练验证 loss_ta ...")
        model.train()
        # 复用 trainer 的 optimizer 与数据加载
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.model.train()
        for it in range(args.steps):
            losses = trainer.model(next(trainer._trainer._data_loader_iter))
            if isinstance(losses, torch.Tensor):
                loss_dict = {"total_loss": losses}
            else:
                loss_dict = {k: v.detach().item() for k, v in losses.items()}
            ta = loss_dict.get("loss_ta", None)
            print(f"        iter {it}: loss_ta = {ta}")
            # 反向 + 更新，验证无报错
            if isinstance(losses, torch.Tensor):
                total = losses
            else:
                total = sum(losses.values())
            trainer.optimizer.zero_grad()
            total.backward()
            trainer.optimizer.step()

    print("\n" + "=" * 70)
    if ok:
        print("[SMOKE] [PASS] 结构检查通过：CTP 代码已恢复到 88.66 时代结构")
    else:
        print("[SMOKE] [FAIL] 结构检查未通过：仍存在 V3/V4/V5 残留分支，")
        print("        请确认服务器上的 adet/ 已同步本次回退后的代码")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
