import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from adet.layers.deformable_transformer import DeformableTransformer_Det
from adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset
from .utils import MLP

# CLIP Language Prior — provides text prototypes for TACT
try:
    from .clip_language_prior import CLIPLanguagePrior
except Exception:
    CLIPLanguagePrior = None

# TACT: Topology-Aware Curvature Transform
try:
    from .tact import TACT
except ImportError:
    TACT = None

# GCR: Geometric Context Refinement
try:
    from .geometric_context_refinement import GeometricContextRefinement
except Exception:
    GeometricContextRefinement = None

# TGSR: Topology-Conditioned Semantic Routing
try:
    from .tgsr import TGSR, CLIPFeatureExtractor
except ImportError:
    TGSR = None
    CLIPFeatureExtractor = None

# CLIP Dense Fusion Adapter (DRTP-v2)
try:
    from .clip_dense_adapter import CLIPDenseAdapter
except ImportError:
    CLIPDenseAdapter = None


class DPText_DETR(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone

        self.d_model = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.num_proposals = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.pos_embed_scale = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_ctrl_points = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.num_classes = 1  # only text
        self.sigmoid_offset = not cfg.MODEL.TRANSFORMER.USE_POLYGON

        self.epqm = cfg.MODEL.TRANSFORMER.EPQM
        self.efsa = cfg.MODEL.TRANSFORMER.EFSA
        self.use_clip_lang_prior = cfg.MODEL.TRANSFORMER.USE_CLIP_LANG_PRIOR
        self.enhance = cfg.MODEL.TRANSFORMER.SGIFA.ENABLED
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)

        # ── CLIP Language Prior (provides text prototypes for TACT) ──
        if self.use_clip_lang_prior:
            if CLIPLanguagePrior is None:
                raise ImportError(
                    "USE_CLIP_LANG_PRIOR=True but clip_language_prior module failed to import."
                )
            clip_model_path = cfg.MODEL.TRANSFORMER.get("CLIP_MODEL_PATH", "")
            self.clip_lang_prior = CLIPLanguagePrior(
                d_model=self.d_model,
                num_ctrl_points=self.num_ctrl_points,
                clip_model_path=clip_model_path if clip_model_path else None,
            )

        # ── TACT: Topology-Aware Curvature Transform ──
        self.use_tact = cfg.MODEL.TRANSFORMER.get("USE_TACT", False)
        self.tact = None
        if self.use_tact:
            if TACT is None:
                raise ImportError("USE_TACT=True but tact module not found.")
            self.tact = TACT(
                d_model=self.d_model,
                num_ctrl_points=self.num_ctrl_points,
                sigma_relax=cfg.MODEL.TRANSFORMER.get("TACT_SIGMA_RELAX", False),
                decoupled=cfg.MODEL.TRANSFORMER.get("TACT_DECOUPLED", False),
                use_cura=cfg.MODEL.TRANSFORMER.get("TACT_USE_CURA", True),
            )

        # ── Transformer ──
        self.transformer = DeformableTransformer_Det(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels,
            dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points,
            num_proposals=self.num_proposals,
            num_ctrl_points=self.num_ctrl_points,
            epqm=self.epqm,
            efsa=self.efsa,
            use_clip_lang_prior=self.use_clip_lang_prior,
            enhance=self.enhance,
            tact_module=self.tact,
        )

        # ── Classification & Regression Heads ──
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)

        # ── GCR: Geometric Context Refinement ──
        self.use_gcr = cfg.MODEL.TRANSFORMER.GCR.ENABLED
        self.gcr_cls_bonus = cfg.MODEL.TRANSFORMER.GCR.CLS_BONUS
        self.gcr_num_layers = cfg.MODEL.TRANSFORMER.GCR.get("NUM_LAYERS", 1)
        self.gcr_coord_only = cfg.MODEL.TRANSFORMER.GCR.get("COORD_ONLY", False)
        self.gcr_use_attention = cfg.MODEL.TRANSFORMER.GCR.get("USE_ATTENTION", False)
        if self.use_gcr:
            if GeometricContextRefinement is None:
                raise ImportError("GCR.ENABLED=True but geometric_context_refinement module not found.")
            common_kwargs = dict(
                d_model=self.d_model,
                num_points=self.num_ctrl_points,
                hidden_dim=cfg.MODEL.TRANSFORMER.GCR.HIDDEN_DIM,
                cls_bonus=self.gcr_cls_bonus,
                coord_only=self.gcr_coord_only,
                use_attention=self.gcr_use_attention,
            )
            if self.gcr_num_layers > 1:
                self.gcr_layers = nn.ModuleList([
                    GeometricContextRefinement(**common_kwargs)
                    for _ in range(self.gcr_num_layers)
                ])
            else:
                self.gcr = GeometricContextRefinement(**common_kwargs)

        # ── TGSR: Topology-Conditioned Semantic Routing ──
        self.use_tgsr = cfg.MODEL.TRANSFORMER.get("USE_TGSR", False)
        self.tgsr_start_layer = cfg.MODEL.TRANSFORMER.get("TGSR_START_LAYER", 4)
        self.tgsr = None
        if self.use_tgsr:
            if TGSR is None:
                raise ImportError("USE_TGSR=True but tgsr module not found.")
            clip_extractor = CLIPFeatureExtractor(
                pretrained_path=cfg.MODEL.TRANSFORMER.get("TGSR_CLIP_PATH", "pretrain/clip-vit-base-patch16"),
                freeze=True,
            )
            self.tgsr = TGSR(
                clip_extractor=clip_extractor,
                d_model=self.d_model,
                num_ctrl_points=self.num_ctrl_points,
                beta_init=cfg.MODEL.TRANSFORMER.get("TGSR_BETA_INIT", 0.0),
                eta=cfg.MODEL.TRANSFORMER.get("TGSR_ETA", 2.0),
                sigma=cfg.MODEL.TRANSFORMER.get("TGSR_SIGMA", 0.3),
                temperature=cfg.MODEL.TRANSFORMER.get("TGSR_TEMP", 0.1),
                start_layer=self.tgsr_start_layer,
            )

        # ── CLIP Dense Fusion Adapter (DRTP-v2) ──
        # Fuses frozen CLIP patch tokens into FPN features BEFORE the encoder.
        # Read from yaml-only keys registered in defaults.py.
        self.use_clip = cfg.MODEL.TRANSFORMER.get("USE_CLIP", False)
        self.clip_adapter = None
        if self.use_clip:
            if CLIPDenseAdapter is None:
                raise ImportError("USE_CLIP=True but clip_dense_adapter module not found.")
            self.clip_adapter = CLIPDenseAdapter(
                clip_model_name=cfg.MODEL.TRANSFORMER.get("CLIP_PRETRAINED", "pretrain/clip-vit-base-patch16"),
                d_model=self.d_model,
                num_feature_levels=self.num_feature_levels,
                freeze_clip=cfg.MODEL.TRANSFORMER.get("CLIP_FREEZE", True),
                keep_aspect_ratio=cfg.MODEL.TRANSFORMER.get("CLIP_KEEP_ASPECT", True),
                shuffle=cfg.MODEL.TRANSFORMER.get("CLIP_SHUFFLE", False),
                replace_noise=cfg.MODEL.TRANSFORMER.get("CLIP_REPLACE_NOISE", False),
                token_mix=cfg.MODEL.TRANSFORMER.get("CLIP_TOKEN_MIX", False),
                mix_lambda=cfg.MODEL.TRANSFORMER.get("CLIP_TOKEN_MIX_LAMBDA", 0.1),
                # ── Phase 1 ablation ──
                active_levels=cfg.MODEL.TRANSFORMER.get("CLIP_ACTIVE_LEVELS", [0, 1, 2, 3]),
                use_gate=cfg.MODEL.TRANSFORMER.get("CLIP_USE_GATE", True),
                learnable_alpha=cfg.MODEL.TRANSFORMER.get("CLIP_LEARNABLE_ALPHA", True),
                fixed_alpha_value=cfg.MODEL.TRANSFORMER.get("CLIP_FIXED_ALPHA_VALUE", 0.5),
                shared_projector=cfg.MODEL.TRANSFORMER.get("CLIP_SHARED_PROJECTOR", False),
                directional_gate=cfg.MODEL.TRANSFORMER.get("CLIP_DIRECTIONAL_GATE", False),
                scale_aware_gate=cfg.MODEL.TRANSFORMER.get("CLIP_SCALE_AWARE_GATE", False),
            )

        # ── Input Projection (FPN → d_model) ──
        if self.num_feature_levels > 1:
            _resnet_ch_map = {"res3": 512, "res4": 1024, "res5": 2048}
            _resnet_st_map = {"res3": 8, "res4": 16, "res5": 32}
            _out_feats = cfg.MODEL.RESNETS.OUT_FEATURES
            num_channels = [_resnet_ch_map[f] for f in _out_feats]
            num_backbone_outs = len(num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            num_channels = [2048]
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )
            ])

        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        # ── Head Initialization ──
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        self.ctrl_point_class = nn.ModuleList([self.ctrl_point_class for _ in range(num_pred)])
        self.ctrl_point_coord = nn.ModuleList([self.ctrl_point_coord for _ in range(num_pred)])
        if self.epqm:
            self.transformer.decoder.ctrl_point_coord = self.ctrl_point_coord
        self.transformer.decoder.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

        self.to(self.device)

    def forward(self, samples: NestedTensor, clip_images=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            clip_images: optional list of (3, H_i, W_i) raw image tensors (geometrically
                         transformed but NOT detector-normalized) for TGSR CLIP feature extraction.
                         Only used when USE_TGSR=True.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        if self.num_feature_levels == 1:
            raise NotImplementedError

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[0]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # ── CLIP Dense Fusion: inject frozen CLIP tokens into FPN features ──
        if self.clip_adapter is not None and clip_images is not None:
            srcs = self.clip_adapter(clip_images, srcs, masks)

        # ── TGSR: Extract CLIP features (once, frozen) ──
        clip_feat_map = None
        if self.tgsr is not None and clip_images is not None:
            clip_tensor = CLIPFeatureExtractor.prepare_images(clip_images, device=self.device)
            clip_feat_map = self.tgsr.clip_extractor(clip_tensor)  # (B, 512, 14, 14)

        # n_pts, embed_dim --> n_q, n_pts, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        # CLIP language prior (provides text prototypes for TACT)
        c_lang, v_spatial = None, None
        if self.use_clip_lang_prior:
            c_lang, v_spatial = self.clip_lang_prior(srcs[-2], srcs[-1])

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, dec_unc = self.transformer(
            srcs, masks, pos, ctrl_point_embed, c_lang=c_lang, v_spatial=v_spatial
        )

        outputs_classes = []
        outputs_coords = []
        gcr_cls_bonus_last = None
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            # Save original [0,1] reference for TGSR before inverse_sigmoid
            ref_original = reference
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)

            h_cur = hs[lvl]

            outputs_class = self.ctrl_point_class[lvl](h_cur)
            tmp = self.ctrl_point_coord[lvl](h_cur)

            # ── GCR: Geometric Context Refinement ──
            if self.use_gcr:
                total_layers = hs.shape[0]
                gcr_start_layer = total_layers - self.gcr_num_layers
                if lvl >= gcr_start_layer:
                    if self.gcr_num_layers > 1:
                        gcr_mod = self.gcr_layers[lvl - gcr_start_layer]
                    else:
                        gcr_mod = self.gcr
                    need_cls_bonus = (
                        self.gcr_cls_bonus and lvl == total_layers - 1
                    )
                    correction, gcr_cls_bonus_last = gcr_mod(
                        tmp.detach(),
                        None if self.gcr_coord_only else h_cur.detach(),
                        return_cls_bonus=need_cls_bonus,
                    )
                    tmp = tmp + correction
                    if gcr_cls_bonus_last is not None:
                        outputs_class = outputs_class + gcr_cls_bonus_last

            # ── TGSR: Topology-Conditioned Semantic Routing ──
            # Applied at layers >= start_layer, ADDITIVE to classification logits
            if clip_feat_map is not None and lvl >= self.tgsr_start_layer:
                if lvl == self.tgsr_start_layer:
                    ref_prev = ref_original  # same → r=1 → local CLIP only
                else:
                    ref_prev = inter_references[lvl - 2]  # (B, N, K, 2)
                tgsr_correction = self.tgsr(clip_feat_map, ref_original, ref_prev)  # (B, N, 1)
                tgsr_correction = tgsr_correction.unsqueeze(-1)  # (B, N, 1, 1) → broadcast to (B, N, K, 1)
                outputs_class = outputs_class + tgsr_correction

            if reference.shape[-1] == 2:
                if self.epqm:
                    tmp += reference
                else:
                    tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                if self.epqm:
                    tmp += reference[..., :2]
                else:
                    tmp += reference[:, :, None, :2]
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_ctrl_points': outputs_coord[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {'pred_logits': a, 'pred_ctrl_points': b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
