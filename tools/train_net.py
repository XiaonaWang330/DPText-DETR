# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import sys

# Ensure DPText-DETR's adet is found before any other adet on PYTHONPATH
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import time
from collections import OrderedDict
from typing import Any, Dict, List, Set
import torch
import itertools

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator,TextDetEvaluator


class EvalPeriodFinalCheckpointer(hooks.HookBase):
    """
    Save ``model_final_<iter>.pth`` after each evaluation period.
    Only the latest checkpoint is kept (previous one is deleted).

    This replaces the noisy ``model_0000999.pth`` periodic checkpoints
    with a single well-named snapshot that updates every eval period.
    """

    def __init__(self, eval_period: int, checkpointer, max_iter: int):
        self._period = eval_period
        self._checkpointer = checkpointer
        self._max_iter = max_iter
        self._prev_path = None
        self._logger = logging.getLogger(__name__)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0 and next_iter != self._max_iter:
            # Delete previous final checkpoint
            if self._prev_path and os.path.exists(self._prev_path):
                os.remove(self._prev_path)

            name = f"model_final_{next_iter}"
            self._checkpointer.save(name, iteration=next_iter)
            self._prev_path = os.path.join(self._checkpointer.save_dir, f"{name}.pth")

    def after_train(self):
        # Delete the intermediate final, save the true final
        if self._prev_path and os.path.exists(self._prev_path):
            os.remove(self._prev_path)
        name = f"model_final_{self.trainer.iter}"
        self._checkpointer.save(name, iteration=self.trainer.iter)


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.
        Add `BestCheckpointer` to save best model based on hmean (F1).
        Add `EvalPeriodFinalCheckpointer` to save final_<iter> after each eval.

        Hook execution order in after_step():
          - PeriodicCheckpointer       -> save "model_final" at end only (periodic saves disabled)
          - EvalHook                   -> run evaluation, store metrics in storage
          - BestCheckpointer           -> save "model_best" if hmean improved
          - EvalPeriodFinalCheckpointer -> save "model_final_<iter>" (latest only)
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                # Disable periodic saves (period > max_iter) — only keep
                # the end-of-training "model_final" save for backward compat.
                ret[i] = hooks.PeriodicCheckpointer(
                    self.checkpointer, self.max_iter + 1,
                    max_iter=self.max_iter
                )
            elif isinstance(ret[i], hooks.EvalHook):
                # Insert BestCheckpointer AFTER EvalHook so it can read freshly-stored metrics
                ret.insert(i + 1, hooks.BestCheckpointer(
                    eval_period=self.cfg.TEST.EVAL_PERIOD,
                    checkpointer=self.checkpointer,
                    val_metric="DET_RESULT/hmean",
                    mode="max",
                    file_prefix="model_best",
                ))
                # Insert EvalPeriodFinalCheckpointer after BestCheckpointer
                ret.insert(i + 2, EvalPeriodFinalCheckpointer(
                    eval_period=self.cfg.TEST.EVAL_PERIOD,
                    checkpointer=self.checkpointer,
                    max_iter=self.max_iter,
                ))
        return ret
    
    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def run_step(self):
        """
        Override run_step to support gradient accumulation with AMP.
        Effective batch size = IMS_PER_BATCH * MODEL.GA_STEPS.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        ga_steps = self.cfg.MODEL.GA_STEPS

        if ga_steps <= 1:
            return super().run_step()

        _trainer = self._trainer  # the inner SimpleTrainer instance

        # Lazy-init AMP GradScaler (used below and in optimizer.step)
        if not hasattr(self, "_amp_scaler"):
            self._amp_scaler = torch.cuda.amp.GradScaler()

        start = time.perf_counter()
        data = next(_trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        # Zero grads once per accumulation cycle
        _trainer.optimizer.zero_grad()

        # Accumulate gradients over ga_steps micro-batches
        total_loss_dict = None
        for _micro_step in range(ga_steps):
            with torch.cuda.amp.autocast():
                loss_dict = _trainer.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict_for_log = {"total_loss": loss_dict.detach()}
            else:
                losses = sum(loss_dict.values())
                loss_dict_for_log = {k: v.detach() for k, v in loss_dict.items()}

            # Scale loss so sum of ga_steps gradients = average of micro-batches
            losses = losses / ga_steps
            self._amp_scaler.scale(losses).backward()

            # Accumulate loss tensors for logging (detached)
            if total_loss_dict is None:
                total_loss_dict = {}
                for k, v in loss_dict_for_log.items():
                    total_loss_dict[k] = v.clone()
            else:
                for k, v in loss_dict_for_log.items():
                    total_loss_dict[k] += v

            # Fetch next batch for next micro-step (except last)
            if _micro_step < ga_steps - 1:
                data = next(_trainer._data_loader_iter)

        # Average accumulated loss values for logging
        if total_loss_dict is not None:
            total_loss_dict = {k: v / ga_steps for k, v in total_loss_dict.items()}

        _trainer.after_backward()
        _trainer._write_metrics(total_loss_dict, data_time)
        self._amp_scaler.step(_trainer.optimizer)
        self._amp_scaler.update()

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            if cfg.TEST.DET_ONLY:
                return TextDetEvaluator(dataset_name, cfg, True, output_folder)
            else:
                return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_optimizer(cls, cfg, model):
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY

            if match_name_keywords(key, cfg.SOLVER.LR_BACKBONE_NAMES):
                lr = cfg.SOLVER.LR_BACKBONE
            elif match_name_keywords(key, cfg.SOLVER.LR_LINEAR_PROJ_NAMES):
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.LR_LINEAR_PROJ_MULT

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
