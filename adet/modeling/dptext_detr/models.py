import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from adet.layers.deformable_transformer import DeformableTransformer_Det
from adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset
from .utils import MLP

# CLIP Dense Fusion Adapter (DRTP-v2)
try:
    from .clip_dense_adapter import CLIPDenseAdapter
except ImportError:
    CLIPDenseAdapter = None

# RICA: Residual-guided Instance Classification Attention (uses DRTP residuals)
# Legacy module; superseded by PRICA (kept for checkpoint compatibility).
try:
    from .residual_guided_cls_attention import ResidualGuidedClsAttention
except ImportError:
    ResidualGuidedClsAttention = None

# PRICA: Point-Residual Intra-query Classification Aggregation
# Unified module = RICA (point-wise residual evidence extraction) + PAGA
# (consistency-aware instance aggregation). Paper narrative: DRTP + PRICA.
try:
    from .prica import PRICA
except ImportError:
    PRICA = None

# CQR: CLIP-Conditioned Query Routing
# Each query reads local CLIP content at its predicted control points and
# emits a classification-only logit residual (see clip_query_routing.py).
try:
    from .clip_query_routing import CQR
except ImportError:
    CQR = None

# TACT: Topology-Aware Curvature Transform
# Pure-visual geometric module (control-point distance/curvature topology),
# injected into every decoder layer after intra-SA + circonv (see 结构/tact.txt).
try:
    from .tact import TACT
except ImportError:
    TACT = None

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
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)

        # -- TACT: Topology-Aware Curvature Transform --
        # Pure-visual geometric module injected into every decoder layer after
        # intra-SA + circonv: Gaussian topology aggregation + curvature-aware
        # gating + FiLM circonv modulation. All new params zero-init -> the
        # first training step is exactly the DPText-DETR baseline.
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

        # -- Transformer --
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
            tact_module=self.tact,
        )

        # -- Classification & Regression Heads --
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)

        # -- CLIP Dense Fusion Adapter (DRTP-v2) --
        # Fuses frozen CLIP patch tokens into FPN features BEFORE the encoder.
        # Read from yaml-only keys registered in defaults.py.
        # CLIP_DENSE_FUSION=False skips building the adapter entirely so RICA
        # can run alone on raw FPN features (RICA-only ablation).
        self.use_clip = cfg.MODEL.TRANSFORMER.get("USE_CLIP", False)
        self.clip_dense_fusion = cfg.MODEL.TRANSFORMER.get("CLIP_DENSE_FUSION", False)
        self.selfgen_fusion = cfg.MODEL.TRANSFORMER.get("SELF_GATED_FUSION", False)
        self.clip_adapter = None
        if (self.use_clip and self.clip_dense_fusion) or self.selfgen_fusion:
            if CLIPDenseAdapter is None:
                raise ImportError("USE_CLIP/SELF_GATED_FUSION=True but clip_dense_adapter module not found.")
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
                # -- Phase 1 ablation --
                active_levels=cfg.MODEL.TRANSFORMER.get("CLIP_ACTIVE_LEVELS", [0, 1, 2, 3]),
                use_gate=cfg.MODEL.TRANSFORMER.get("CLIP_USE_GATE", True),
                learnable_alpha=cfg.MODEL.TRANSFORMER.get("CLIP_LEARNABLE_ALPHA", True),
                fixed_alpha_value=cfg.MODEL.TRANSFORMER.get("CLIP_FIXED_ALPHA_VALUE", 0.5),
                shared_projector=cfg.MODEL.TRANSFORMER.get("CLIP_SHARED_PROJECTOR", False),
                directional_gate=cfg.MODEL.TRANSFORMER.get("CLIP_DIRECTIONAL_GATE", False),
                probe_mode=cfg.MODEL.TRANSFORMER.get("CLIP_PROBE_MODE", "none"),
                # -- Self-Gated Fusion (CLIP-free) --
                selfgen=self.selfgen_fusion,
                selfgen_active_levels=cfg.MODEL.TRANSFORMER.get("SGF_ACTIVE_LEVELS", [0, 1, 2]),
                selfgen_use_gate=cfg.MODEL.TRANSFORMER.get("SGF_USE_GATE", True),
                selfgen_directional_gate=cfg.MODEL.TRANSFORMER.get("SGF_DIRECTIONAL_GATE", True),
                selfgen_learnable_alpha=cfg.MODEL.TRANSFORMER.get("SGF_LEARNABLE_ALPHA", True),
                selfgen_fixed_alpha_value=cfg.MODEL.TRANSFORMER.get("SGF_FIXED_ALPHA_VALUE", 0.5),
            )

        # -- RICA: Residual-guided Instance Classification Attention --
        # Reuses the final decoder cross-attention sampling geometry to read
        # DRTP residuals, forming a classification-only feature h_cls:
        #   final_logits = class_embed(h_cls), final_points = point_embed(h_final)
        # No geometry changes, no auxiliary loss, no new hyper-parameters.
        self.use_rica = cfg.MODEL.TRANSFORMER.get("RICA_ENABLED", False) and self.use_clip
        self.rica = None
        if self.use_rica:
            if ResidualGuidedClsAttention is None:
                raise ImportError("RICA_ENABLED=True but residual_guided_cls_attention module not found.")
            self.rica = ResidualGuidedClsAttention(
                d_model=self.d_model,
                num_levels=len(cfg.MODEL.TRANSFORMER.get("CLIP_ACTIVE_LEVELS", [0, 1, 2])),
                n_heads=self.nhead,
                n_points=self.dec_n_points,
                active_cross_levels=cfg.MODEL.TRANSFORMER.get("CLIP_ACTIVE_LEVELS", [0, 1, 2]),
            )

        # -- PRICA: Point-Residual Intra-query Classification Aggregation --
        # The unified module (RICA + PAGA). PRICA extracts per-point residual
        # classification evidence, weights points by query-residual agreement,
        # and aggregates them into an instance-level refinement delta:
        #   h_cls = query + delta.unsqueeze(2)     # residual refinement
        #   final_logits = class_embed(h_cls)      # classification path only
        #   final_points = point_embed(query)      # geometry path untouched
        # delta is ZERO at init -> PRICA starts as the identity on the original
        # classification path (the RICA baseline is preserved, not replaced).
        self.use_prica = cfg.MODEL.TRANSFORMER.get("PRICA_ENABLED", False) and self.use_clip
        self.prica = None
        if self.use_prica:
            if PRICA is None:
                raise ImportError("PRICA_ENABLED=True but prica module not found.")
            self.prica = PRICA(
                d_model=self.d_model,
                num_levels=len(cfg.MODEL.TRANSFORMER.get("CLIP_ACTIVE_LEVELS", [0, 1, 2])),
                n_heads=self.nhead,
                n_points=self.dec_n_points,
                active_cross_levels=cfg.MODEL.TRANSFORMER.get("CLIP_ACTIVE_LEVELS", [0, 1, 2]),
                point_feat_mode=cfg.MODEL.TRANSFORMER.get(
                    "PRICA_POINT_FEAT", "rica+ctx+agree"),
            )

        # -- CQR: CLIP-Conditioned Query Routing --
        # Classification-only logit residual. Each query samples the raw CLIP
        # patch content at its predicted control-point locations (detached) and
        # modulates its class logit via a point-wise multiplicative interaction:
        #   delta_logit = cqr_cls_head(out_norm(interaction(h_cur, clip_at_pts)))
        #   final_logits = base_logits + delta_logit    (zero-init => identity)
        # The regression path (h_cur -> ctrl_point_coord) is untouched.
        self.use_cqr = cfg.MODEL.TRANSFORMER.get("CLIP_QUERY_ROUTING", False) and self.use_clip
        self.cqr = None
        if self.use_cqr:
            if CQR is None:
                raise ImportError("CLIP_QUERY_ROUTING=True but clip_query_routing module not found.")
            if self.clip_adapter is None:
                raise ImportError("CLIP_QUERY_ROUTING=True but no CLIP adapter (need CLIP_DENSE_FUSION=True).")
            self.cqr = CQR(
                d_model=self.d_model,
                clip_dim=self.clip_adapter.clip_dim,
                clip_patch_size=self.clip_adapter.clip_patch_size,
            )

        # -- CTP: CLIP Text Prior (CLIP text-prototype alignment) --
        # CLIP-derived text/confuser prototypes supervise the decoder query
        # features (softplus margin) as a training-time auxiliary loss, PLUS a
        # classification-only residual adapter (q_cls = q + alpha * A(sg(q)))
        # active at both training and inference. Regression path untouched.
        # Semantic prior -> pairs with TACT (geometric correction). Internal
        # identifiers below keep the historical ta_head / ta_* names.
        self.ta_enabled = cfg.MODEL.TRANSFORMER.get("CLIP_TEXT_PRIOR", None) is not None and \
            cfg.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.ENABLED
        self.ta_head = None
        if self.ta_enabled:
            from .clip_text_prior import TextPrototypeAlignmentHead
            ta_cfg = cfg.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR
            self.ta_head = TextPrototypeAlignmentHead(
                d_model=self.d_model,
                proto_dim=ta_cfg.PROTO_DIM,
                use_confuser=ta_cfg.USE_CONFUSER,
                temperature=ta_cfg.TEMPERATURE,
                clip_model_name=ta_cfg.CLIP_PRETRAINED,
                text_prompts=list(ta_cfg.TEXT_PROMPTS),
                confuser_prompts=list(ta_cfg.CONFUSER_PROMPTS),
                adapter_detach_input=ta_cfg.ADAPTER_DETACH_INPUT,
                alpha_init=ta_cfg.ADAPTER_ALPHA_INIT,
                use_adapter=ta_cfg.USE_ADAPTER,
                use_margin_mod=ta_cfg.USE_MARGIN_MOD,
                margin_mod_beta_init=ta_cfg.MARGIN_MOD_BETA_INIT,
                proto_pooling=ta_cfg.PROTO_POOLING,
            )
            self.ta_matcher_base_logits = ta_cfg.MATCHER_BASE_LOGITS
            self.ta_loss_detach = ta_cfg.LOSS_DETACH

        # -- GCR: Geometric Context Refinement (纯几何回归修正) --
        # 回归路径（FINAL layer）：tmp = ctrl_point_coord(h_cur) 后加
        # GCR(tmp.detach(), h_cur.detach())，RingConv 环形卷积修正。
        # 数据集无关（不依赖 CLIP），与 CTP（分类路径）互补。
        self.use_gcr = cfg.MODEL.TRANSFORMER.get("USE_GCR", False)
        self.gcr = None
        self.gcr_only_final = cfg.MODEL.TRANSFORMER.get("GCR_ONLY_FINAL", True)
        if self.use_gcr:
            from .geometric_context_refinement import GeometricContextRefinement
            self.gcr = GeometricContextRefinement(
                d_model=self.d_model,
                num_ctrl_points=self.num_ctrl_points,
                hidden_dim=cfg.MODEL.TRANSFORMER.get("GCR_HIDDEN_DIM", 128),
                coord_only=cfg.MODEL.TRANSFORMER.get("GCR_COORD_ONLY", False),
            )

        # -- GATP: Geometry-Aware Text Prototype (几何感知文本原型) --
        # 分类路径（FINAL layer）：用控制点曲率/尺度条件化的可学习文本
        # 原型 margin 注入分类 logits。纯几何、数据集无关。
        self.use_gatp = cfg.MODEL.TRANSFORMER.get("USE_GATP", False)
        self.gatp = None
        if self.use_gatp:
            from .geometry_aware_text_prototype import GeometryAwareTextPrototype
            self.gatp = GeometryAwareTextPrototype(
                d_model=self.d_model,
                proto_dim=cfg.MODEL.TRANSFORMER.get("GATP_PROTO_DIM", 256),
                num_ctrl_points=self.num_ctrl_points,
                num_confuser=cfg.MODEL.TRANSFORMER.get("GATP_NUM_CONFUSER", 6),
                temperature=cfg.MODEL.TRANSFORMER.get("GATP_TEMPERATURE", 0.10),
                beta_init=cfg.MODEL.TRANSFORMER.get("GATP_BETA_INIT", 0.05),
                beta_trainable=cfg.MODEL.TRANSFORMER.get("GATP_BETA_TRAINABLE", True),
            )

        # -- Input Projection (FPN → d_model) --
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

        # -- Head Initialization --
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
                         transformed but NOT detector-normalized) for CLIP adapter.
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

        # -- CLIP Dense Fusion: inject frozen CLIP tokens into FPN features --
        # RICA-only / PRICA-only mode (CLIP_DENSE_FUSION=False, RICA/PRICA
        # enabled): no adapter is built, so the module samples the RAW FPN
        # features directly to isolate its standalone contribution from DRTP.
        clip_residuals = None
        if self.clip_adapter is not None:
            if self.selfgen_fusion:
                # SGF: CLIP-free self-gated fusion (no clip_images needed)
                srcs = self.clip_adapter.forward_selfgen(srcs, masks)
            elif clip_images is not None:
                srcs = self.clip_adapter(clip_images, srcs, masks)
            # Pre-fusion per-level CLIP residuals for RICA (None if inactive level)
            clip_residuals = self.clip_adapter.get_last_residuals()
        elif (self.rica is not None or self.prica is not None) and not self.clip_dense_fusion:
            clip_residuals = srcs

        # n_pts, embed_dim --> n_q, n_pts, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs,
            masks,
            pos,
            ctrl_point_embed,
            clip_residuals=clip_residuals,
        )

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)

            h_cur = hs[lvl]

            # Classification feature: on the FINAL decoder layer only, enhance
            # h_cur by reading DRTP residuals at the original cross-attention
            # sampling sites. The geometry path keeps using the original h_cur
            # (classification residual never touches it).
            h_cls = h_cur
            if lvl == hs.shape[0] - 1 and clip_residuals is not None:
                sampling_locations = getattr(self.transformer, "_last_dec_sampling_locations", None)
                attention_weights = getattr(self.transformer, "_last_dec_attention_weights", None)
                if (self.prica is not None or self.rica is not None) and \
                        (sampling_locations is None or attention_weights is None):
                    raise RuntimeError(
                        "PRICA/RICA enabled but no decoder cross-attention details exposed")

                if self.prica is not None:
                    # PRICA (RICA + point-wise signed residual refinement — see
                    # prica.py docstring for the v1 -> v2 motivation and the
                    # E1 wiring fix):
                    #   rica_h       = query + rica_point   (point-wise RICA,
                    #                                        KEPT in the
                    #                                        classification path)
                    #   point_delta  = delta_proj(<mode-dependent per-point
                    #                              feature>) (B,K,N,C) SIGNED
                    #   h_cls        = rica_h + point_delta
                    # E1 fix: the validated RICA evidence path is no longer
                    # discarded (diagnosis: query + point_delta lost ~0.4 F1
                    # vs RICA 88.57 / DRTP+RICA 88.72 vs DRTP+PRICA 88.31);
                    # the signed per-point delta only refines rica_h.
                    rica_h, point_delta, point_weight = self.prica(
                        query=h_cur,
                        residual_features=clip_residuals,
                        sampling_locations=sampling_locations,
                        attention_weights=attention_weights,
                    )
                    h_cls = rica_h + point_delta
                    self._last_prica_point_weight = point_weight
                elif self.rica is not None:
                    # Legacy RICA (kept for checkpoint compatibility).
                    h_cls = self.rica(
                        h_final=h_cur,
                        residual_features=clip_residuals,
                        sampling_locations=sampling_locations,
                        attention_weights=attention_weights,
                    )

            # -- TA: classification-only residual adapter (FINAL layer only) --
            # q_cls = q + alpha * A(sg(q)); regression path (h_cur) untouched.
            # Exposes (i) base logits for the Hungarian matcher (E variant) and
            # (ii) instance-level query features for the TA auxiliary loss.
            if self.ta_enabled and lvl == hs.shape[0] - 1:
                if self.ta_matcher_base_logits:
                    self._ta_base_logits = self.ctrl_point_class[lvl](h_cls)
                ta_query_feat = h_cur.mean(dim=2)  # (B, N, d_model)
                if self.ta_loss_detach:
                    ta_query_feat = ta_query_feat.detach()
                self._ta_query_feat = ta_query_feat

                # 分类残差 adapter（q_cls = q + alpha * A(sg(q))），逐点广播
                h_cls = self.ta_head.forward_adapter(h_cls)
                outputs_class = self.ctrl_point_class[lvl](h_cls)
                # -- CTP margin modulation (DAG fix): route projector margin
                # into logits as a scalar beta*margin residual.
                if self.ta_head.use_margin_mod:
                    outputs_class = outputs_class + self.ta_head.margin_modulation(ta_query_feat.detach()).unsqueeze(2)
            else:
                outputs_class = self.ctrl_point_class[lvl](h_cls)
            tmp = self.ctrl_point_coord[lvl](h_cur)

            # -- GCR: geometric context refinement (regression path) --
            # tmp = tmp + RingConv-corrected delta. Inputs detached -> pure
            # geometric side-path; zero-init output_proj -> starts as baseline.
            # Dataset-agnostic (no CLIP) -> complements CTP across datasets.
            if self.gcr is not None and (
                    not self.gcr_only_final or lvl == hs.shape[0] - 1):
                gcr_delta = self.gcr(tmp.detach(), h_cur.detach())
                tmp = tmp + gcr_delta

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

            # -- CQR: query-conditioned CLIP logit residual (FINAL layer only) --
            # Samples raw CLIP patch content at the (detached) predicted control
            # points and adds a classification-only delta to the base logits.
            if self.use_cqr and lvl == hs.shape[0] - 1:
                if self.clip_adapter is not None:
                    clip_map, content_hw = self.clip_adapter.get_last_cqr_inputs()
                    if clip_map is not None:
                        cqr_delta = self.cqr(
                            h_cur=h_cur,
                            points=outputs_coord.detach(),
                            clip_map=clip_map,
                            content_hw=content_hw,
                        )
                        outputs_class = outputs_class + cqr_delta.type_as(outputs_class)

            # -- GATP: geometry-aware text prototype (FINAL layer only) --
            # 用预测控制点（detach）的曲率/尺度条件化可学习文本原型，
            # margin 注入分类 logits。纯几何、数据集无关。
            if self.gatp is not None and lvl == hs.shape[0] - 1:
                q_inst = h_cur.mean(dim=2)                    # (B, N, d_model)
                pts = outputs_coord.detach()                  # (B, N, P, 2)
                outputs_class = self.gatp.forward_logits(
                    outputs_class, q_inst, pts)

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # Matcher / classification loss / inference all use the SAME logits.
        # RICA only changed the feature feeding class_embed on the last layer
        # (h_cls), so no second logits set is needed.
        out = {
            'pred_logits': outputs_class[-1],
            'pred_ctrl_points': outputs_coord[-1],
        }

        # TA: expose base logits (matcher, E variant) and instance-level query
        # features (auxiliary loss). Only present when TA is enabled.
        if self.ta_enabled:
            out['pred_logits_base'] = self._ta_base_logits
            out['ta_query_feat'] = self._ta_query_feat

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        # Diagnostic: expose reference points for per-layer coverage analysis
        out['reference_points'] = {
            'init_reference': init_reference,
            'inter_references': inter_references,
        }

        # Diagnostic: PRICA point-level consistency weights (for TP/FP/FN
        # point-weight analysis in paper visualization). Detached already.
        if self.prica is not None and getattr(self, "_last_prica_point_weight", None) is not None:
            out['prica_point_weight'] = self._last_prica_point_weight

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {'pred_logits': a, 'pred_ctrl_points': b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
