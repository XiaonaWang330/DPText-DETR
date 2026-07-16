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
        Load model state dict, allowing missing CLIP frozen params
        (they are reloaded from HF on each init).
        """
        if checkpoint.get("matching_heuristics", False):
            # Convert weights by name-matching heuristics (legacy)
            self._convert_ndarray_to_tensor(checkpoint["model"])
            model_state = self.model.state_dict()
            for key in checkpoint["model"]:
                if key in model_state:
                    model_state[key] = checkpoint["model"][key]
            missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        else:
            checkpoint_model = checkpoint["model"]
            missing, unexpected = self.model.load_state_dict(
                checkpoint_model, strict=False
            )

        # Only warn about non-CLIP missing keys (CLIP frozen weights reload from HF)
        def _is_clip_param(key):
            return (
                "clip_text_model." in key   # SFA, CLIP Language Prior
                or "clip_vision." in key     # CMFE, CSG
                or "clip_text." in key       # CMFE, CSG
            )
        real_missing = [k for k in missing if not _is_clip_param(k)]
        real_unexpected = [k for k in unexpected if not _is_clip_param(k)]
        if real_missing:
            self.logger.warning(f"Missing keys: {real_missing}")
        if real_unexpected:
            self.logger.warning(f"Unexpected keys: {real_unexpected}")

    def save(self, name: str, **kwargs):
        """
        Save checkpoint, excluding CLIP frozen weights (~600MB for ViT+Text).
        They are reloaded from huggingface on next init.
        """
        data = {}
        data["model"] = OrderedDict(
            (k, v) for k, v in self.model.state_dict().items()
            if not (
                "clip_text_model." in k   # SFA, CLIP Language Prior
                or "clip_vision." in k     # CMFE, CSG
                or "clip_text." in k       # CMFE, CSG (distinct from clip_text_model.)
            )
        )
        for key, obj in kwargs.items():
            data[key] = obj

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        with PathManager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)
