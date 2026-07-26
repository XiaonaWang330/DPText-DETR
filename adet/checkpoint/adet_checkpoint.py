import pickle, os
from collections import OrderedDict

import torch
from fvcore.common.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer


class AdetCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectronCheckpointer`, plus:
    - Model zoo format conversion (LPF backbone, DLA, etc.)
    - CLIP frozen weight exclusion (V9): save without ~252MB CLIP params,
      reload from HF on every init.
    """

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                if "weight_order" in data:
                    del data["weight_order"]
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}

        basename = os.path.basename(filename).lower()
        if "lpf" in basename or "dla" in basename:
            loaded["matching_heuristics"] = True
        return loaded

    def _load_model(self, checkpoint):
        """
        Load model state dict, with automatic key migration (SATR→TACT rename)
        and CLIP frozen weight exclusion.
        """
        checkpoint_model = checkpoint.get("model", checkpoint)

        # ── Key migration: SATR → TACT rename ──
        # Old checkpoints store 'satr_module.*' and 'decoder.layers.X.satr.*'.
        # New code uses 'tact.*' and 'decoder.layers.X.tact.*'.
        # Always migrate on load → transparent to all callers.
        _migrated = {}
        for k, v in checkpoint_model.items():
            # Top-level module: dptext_detr.satr_module.* → dptext_detr.tact.*
            if "satr_module." in k:
                _migrated[k.replace("satr_module.", "tact.")] = v
            # Per-decoder-layer: ...layers.X.satr.* → ...layers.X.tact.*
            elif ".satr." in k:
                _migrated[k.replace(".satr.", ".tact.")] = v
            else:
                _migrated[k] = v
        checkpoint_model = _migrated

        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint_model)
            model_state = self.model.state_dict()
            for key in checkpoint_model:
                if key in model_state:
                    model_state[key] = checkpoint_model[key]
            missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        else:
            missing, unexpected = self.model.load_state_dict(
                checkpoint_model, strict=False
            )

        # Only warn about truly unexpected missing keys.
        # We exclude:
        #  - CLIP frozen weights (reload from HF / CLIP pretrained on every init)
        #  - CLIP adapter trainable params (new modules not in pretrained checkpoints)
        #  - TACT / legacy SATR modules
        def _is_expected_missing(key):
            return (
                "clip_text_model." in key   # CLIP Language Prior (reloaded from HF)
                or "clip_adapter." in key   # CLIP Dense Adapter (new module + frozen ViT)
                or ".tgsr." in key           # TGSR (new module, not in pretrained)
                or "tact." in key            # TACT (new module, not in pretrained)
                or ".satr." in key           # legacy SATR → TACT migration in progress
            )
        real_missing = [k for k in missing if not _is_expected_missing(k)]
        real_unexpected = [k for k in unexpected if not _is_expected_missing(k)]
        if real_missing:
            self.logger.warning(f"Missing keys: {real_missing}")
        if real_unexpected:
            self.logger.warning(f"Unexpected keys: {real_unexpected}")

    def save(self, name: str, **kwargs):
        """
        Save checkpoint, excluding CLIP frozen weights (~88MB ViT + ~60MB Text).
        They are reloaded from CLIP pretrained / HF on next init.
        Trainable adapter params (level_projectors, level_gate_nets, level_alphas)
        are kept — they are small (~3MB total).
        """
        data = {}
        data["model"] = OrderedDict(
            (k, v) for k, v in self.model.state_dict().items()
            if not "clip_text_model." in k              # CLIP Language Prior (reloaded from HF)
            and not "clip_adapter.clip_vision." in k    # CLIP ViT frozen (reloaded from pretrained)
            and not "clip_adapter.visual_projection." in k  # CLIP vis proj frozen
            and not "clip_adapter.text_model." in k     # CLIP text encoder frozen (~60MB!)
            and not "clip_adapter.text_projection." in k # CLIP text proj frozen
            and not "clip_adapter.logit_scale." in k    # CLIP logit scale frozen
            and not "tgsr.clip_extractor." in k         # TGSR CLIP ViT frozen (reloaded from pretrained)
        )
        for key, obj in kwargs.items():
            data[key] = obj

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        with PathManager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)
