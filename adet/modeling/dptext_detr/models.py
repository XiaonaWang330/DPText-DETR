import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from adet.layers.deformable_transformer import DeformableTransformer_Det
from adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset
from .utils import MLP
from .clip_language_prior import CLIPLanguagePrior
from .semantic_feature_alignment import SemanticFeatureAlignment
from .geometric_context_refinement import GeometricContextRefinement
from .spectral_curvature_refinement import SpectralCurvatureRefinement

# Optional modules: may not exist on server if only certain experiments are synced.
try:
    from .clip_spatial_guidance import CLIPSpatialGuidance
except ImportError:
    CLIPSpatialGuidance = None

try:
    from .clip_feature_enhancement import CrossModalFeatureEnhancement
except ImportError:
    CrossModalFeatureEnhancement = None

# SATR: Scale-Adaptive Topology Refinement
try:
    from .scale_adaptive_topology_refinement import SATR_Module
except ImportError:
    SATR_Module = None


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




        # V9: CLIP language prior with per-layer sigmoid gate
        if self.use_clip_lang_prior:
            clip_model_path = cfg.MODEL.TRANSFORMER.get("CLIP_MODEL_PATH", "")
            self.clip_lang_prior = CLIPLanguagePrior(
                d_model=self.d_model,
                num_ctrl_points=self.num_ctrl_points,
                clip_model_path=clip_model_path if clip_model_path else None,
            )

        # V13: Semantic Feature Alignment
        self.use_sfa = cfg.MODEL.TRANSFORMER.get("USE_SFA", False)
        self.sfa_num_layers = cfg.MODEL.TRANSFORMER.get("SFA_NUM_LAYERS", 1)
        # V51: detach h_cur input → SFA reads but does NOT write to backbone
        self.sfa_detach_input = cfg.MODEL.TRANSFORMER.get("SFA_DETACH_INPUT", False)
        # V55: FiLM fusion — SFA modulates GCR via gamma/beta (no cls injection)
        self.sfa_film_gcr = cfg.MODEL.TRANSFORMER.get("SFA_FILM_GCR", False)
        if self.use_sfa:
            sfa_clip_path = cfg.MODEL.TRANSFORMER.get("CLIP_MODEL_PATH", "")
            sfa_agg_mode = cfg.MODEL.TRANSFORMER.get("SFA_AGG_MODE", "point")
            self.sfa = SemanticFeatureAlignment(
                d_model=self.d_model,
                clip_model_path=sfa_clip_path if sfa_clip_path else None,
                agg_mode=sfa_agg_mode,
                use_film=self.sfa_film_gcr,   # V55: enable gamma/beta heads
            )

        # V61: CLIP Spatial Guidance — injects CLIP visual-semantic priors
        # at the FPN feature map level (NOT at decoder output, NOT as classifier).
        # Operates BEFORE the transformer → all downstream modules benefit.
        self.use_csg = cfg.MODEL.TRANSFORMER.get("USE_CSG", False)
        if self.use_csg:
            if CLIPSpatialGuidance is None:
                raise ImportError("USE_CSG=True but clip_spatial_guidance module not found. Sync the file to server.")
            csg_clip_path = cfg.MODEL.TRANSFORMER.get("CLIP_MODEL_PATH", "")
            self.csg = CLIPSpatialGuidance(
                clip_model_path=csg_clip_path if csg_clip_path else None,
                d_model=self.d_model,
            )

        # V62: Cross-Modal Feature Enhancement (CMFE) — TRUE multi-modal module.
        # Vision Encoder (ViT) + Text Encoder (multiple prompts) interact via
        # channel-wise modulation → 768D features preserved → injected at FPN.
        # Unlike SFA/CSG (1D scalar collapse → classification redundancy):
        #   CMFE does NOT answer "is this text?" → NO competition with cls_head.
        #   CMFE does "text concept guides vision encoding" → COMPLEMENTARY.
        self.use_cmfe = cfg.MODEL.TRANSFORMER.get("USE_CMFE", False)
        if self.use_cmfe:
            if CrossModalFeatureEnhancement is None:
                raise ImportError("USE_CMFE=True but clip_feature_enhancement module not found. Sync the file to server.")
            cmfe_clip_path = cfg.MODEL.TRANSFORMER.get("CLIP_MODEL_PATH", "")
            self.cmfe = CrossModalFeatureEnhancement(
                clip_model_path=cmfe_clip_path if cmfe_clip_path else None,
                d_model=self.d_model,
                use_sigmoid_scale=cfg.MODEL.TRANSFORMER.get('CMFE_USE_SIGMOID_SCALE', True),
                use_min_boost=cfg.MODEL.TRANSFORMER.get('CMFE_USE_MIN_BOOST', False),
                gate_mode=cfg.MODEL.TRANSFORMER.get('CMFE_GATE_MODE', 'sigmoid'),
            )

        # SATR: Scale-Adaptive Topology Refinement
        self.use_satr = cfg.MODEL.TRANSFORMER.get("USE_SATR", False)
        self.satr_module = None
        if self.use_satr:
            if SATR_Module is None:
                raise ImportError("USE_SATR=True but scale_adaptive_topology_refinement module not found.")
            self.satr_module = SATR_Module(
                d_model=self.d_model,
                num_ctrl_points=self.num_ctrl_points,
                sigma_relax=cfg.MODEL.TRANSFORMER.get("SATR_SIGMA_RELAX", False),
                decoupled=cfg.MODEL.TRANSFORMER.get("SATR_DECOUPLED", False),
                use_cura=cfg.MODEL.TRANSFORMER.get("SATR_USE_CURA", False),
            )


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
            satr_module=self.satr_module,
        )
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)

        # V40: GCR — Geometric Context Refinement
        # Detached ring-conv refinement on baseline regression output.
        # h_cur.detach() → zero gradient to h_cur → no P/R seesaw.
        # Zero-init output → starts as exact baseline → residual-safe.
        # V43: NUM_LAYERS > 1 → GCR applied on the last N decoder layers.
        # Each layer has its own GCR module (independent params) because
        # h_cur at different depths has different geometric semantics.
        self.use_gcr = cfg.MODEL.TRANSFORMER.GCR.ENABLED
        self.gcr_cls_bonus = cfg.MODEL.TRANSFORMER.GCR.CLS_BONUS
        self.gcr_num_layers = cfg.MODEL.TRANSFORMER.GCR.get("NUM_LAYERS", 1)
        self.gcr_coord_only = cfg.MODEL.TRANSFORMER.GCR.get("COORD_ONLY", False)
        self.gcr_use_attention = cfg.MODEL.TRANSFORMER.GCR.get("USE_ATTENTION", False)
        self.gcr_use_scgr = cfg.MODEL.TRANSFORMER.GCR.get("USE_SCGR", False)
        if self.use_gcr:
            # V56: SCGR replaces GCR's ring conv with spectral filtering + curvature gating
            RefinementModule = (
                SpectralCurvatureRefinement if self.gcr_use_scgr
                else GeometricContextRefinement
            )
            common_kwargs = dict(
                d_model=self.d_model,
                num_points=self.num_ctrl_points,
                hidden_dim=cfg.MODEL.TRANSFORMER.GCR.HIDDEN_DIM,
                cls_bonus=self.gcr_cls_bonus,
                coord_only=self.gcr_coord_only,
                use_attention=self.gcr_use_attention,
            )
            # v3: pass semantic gate flag to SCGR only (GCR doesn't accept it)
            if self.gcr_use_scgr:
                common_kwargs['semantic_gate'] = cfg.MODEL.TRANSFORMER.GCR.get(
                    'SCGR_SEMANTIC_GATE', True
                )
                common_kwargs['use_scalar_curvature'] = cfg.MODEL.TRANSFORMER.GCR.get(
                    'SCGR_USE_SCALAR_CURVATURE', False
                )
            if self.gcr_num_layers > 1:
                # V43: per-layer GCR modules (one per decoder depth)
                self.gcr_layers = nn.ModuleList([
                    RefinementModule(**common_kwargs)
                    for _ in range(self.gcr_num_layers)
                ])
            else:
                self.gcr = RefinementModule(**common_kwargs)

        # V41: SFA alignment loss pathway detach
        # When True, feat_proj in sfa_info is detached → alignment loss does not
        # shape h_cur. semantic_logit injection (path A) stays attached.
        self.sfa_align_detach = cfg.MODEL.TRANSFORMER.get("SFA_ALIGN_DETACH", False)

        # V47: Matching-Enhancement Decoupling (MED)
        # When True, the model additionally produces clean matching outputs
        # (without SFA/GCR) for the Hungarian matcher, while loss computation
        # still uses enhanced outputs. Eliminates matcher-instability collapse
        # when SFA+GCR operate simultaneously.
        self.match_decouple = cfg.MODEL.TRANSFORMER.get("MATCH_DECOUPLE", False)
        # V49: selective MED — when False, only decouple coord (GCR) from matching;
        # SFA-enhanced cls still participates in matching for semantic prior benefit.
        self.match_decouple_sfa = cfg.MODEL.TRANSFORMER.get("MATCH_DECOUPLE_SFA", True)






        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
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
                        nn.Conv2d(in_channels, self.d_model,kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )
            ])
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

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

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
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

        # n_pts, embed_dim --> n_q, n_pts, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        # V9: CLIP language prior
        c_lang, v_spatial = None, None
        if self.use_clip_lang_prior:
            c_lang, v_spatial = self.clip_lang_prior(srcs[-2], srcs[-1])

        # V61: CLIP Spatial Guidance — inject semantic spatial prior at feature map level
        # Unlike SFA (which competes with cls_head at decoder output),
        # CSG enhances FPN features BEFORE the transformer → all downstream benefit.
        if self.use_csg:
            srcs = self.csg(samples.tensor, srcs)

        # V62: CMFE — Cross-Modal Feature Enhancement (Vision + Language)
        # True multi-modal: Text Encoder modulates Vision Encoder features
        # via channel-wise semantics (768D → 768D, no scalar collapse).
        # Injected at FPN BEFORE transformer → encoder, decoder, cls, reg, GCR all benefit.
        # v3: Also exports text_score_map for SCGR semantic-guided correction.
        text_score_map = None
        if self.use_cmfe:
            srcs, text_score_map = self.cmfe(samples.tensor, srcs)
            # Detach: SCGR semantic gating must not backprop to CMFE.
            # This ensures gradient orthogonality: CMFE trained by detection loss
            # through enhanced features; SCGR trained through geometric correction.
            text_score_map = text_score_map.detach()

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, dec_unc = self.transformer(
            srcs, masks, pos, ctrl_point_embed, c_lang=c_lang, v_spatial=v_spatial
        )

        # V50: SFA info (computed per-layer inside the loop, exported from last SFA layer)
        sem_cos, sfa_info = None, None

        outputs_classes = []
        outputs_coords = []
        # V42: collect GCR cls_bonus from the last decoder layer
        gcr_cls_bonus_last = None
        # V47: MED — matching outputs (initialized None, filled at last decoder layer)
        matching_logits = None
        matching_ctrl_points = None
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)

            h_cur = hs[lvl]

            outputs_class = self.ctrl_point_class[lvl](h_cur)
            # V47: MED (v47 mode) — capture raw cls logit BEFORE SFA injection for matching
            if self.match_decouple and self.match_decouple_sfa and lvl == hs.shape[0] - 1:
                matching_logits = outputs_class.detach()
            # V13: SFA — inject semantic logit at the last decoder layer only
            # V55 FiLM mode: produce gamma/beta for GCR, skip cls injection
            _gamma, _beta = None, None
            if self.use_sfa and lvl == hs.shape[0] - 1:
                # V51: optionally detach h_cur → zero SFA→backbone gradient
                _sfa_input = h_cur.detach() if self.sfa_detach_input else h_cur
                if self.sfa_film_gcr:
                    # V55: FiLM mode — request gamma/beta, do NOT inject sem_logit
                    _sem_logit, _sem_cos, _sfa_info, _gamma, _beta = self.sfa(
                        _sfa_input, unc=dec_unc, return_film=True
                    )
                    # NO cls injection → outputs_class stays baseline
                else:
                    _sem_logit, _sem_cos, _sfa_info = self.sfa(
                        _sfa_input, unc=dec_unc
                    )
                    # V60: Multiplicative semantic gating (detached → no gradient competition)
                    # SFA provides per-point confidence ∈ [0.5, 1.0]:
                    #   text-like (cos_sim→1) → gate→1.0 (full cls output)
                    #   non-text  (cos_sim→-1)→ gate→0.5 (half cls, never zero)
                    # Detach ensures SFA is trained ONLY by alignment loss,
                    # NOT by focal loss → eliminates cls_head vs SFA gradient war.
                    # Uses raw _sem_cos (∈[-1,1]) rather than scaled _sem_logit
                    # to guarantee stable gate range regardless of logit_scale growth.
                    _sem_gate = (_sem_cos.detach() + 1.0) / 2.0   # (B,N,16) ∈ [0,1]
                    _sem_gate = 0.5 + 0.5 * _sem_gate              # (B,N,16) ∈ [0.5, 1.0]
                    _sem_gate = _sem_gate.unsqueeze(-1)            # (B,N,16,1)
                    outputs_class = outputs_class * _sem_gate
                sem_cos = _sem_cos
                sfa_info = _sfa_info
                if self.sfa_align_detach and sfa_info is not None:
                    sfa_info = {**sfa_info, 'feat_proj': sfa_info['feat_proj'].detach()}
            # V49: selective MED — capture ENHANCED cls (WITH SFA) for matching
            # SFA's semantic prior participates in matching → better query-GT assignment
            if self.match_decouple and not self.match_decouple_sfa and lvl == hs.shape[0] - 1:
                matching_logits = outputs_class.detach()
            tmp = self.ctrl_point_coord[lvl](h_cur)
            # V47/V49: MED — capture raw reg delta BEFORE GCR correction for matching
            # GCR does NOT participate in matching → avoids over-matching (v40 P crash)
            if self.match_decouple and lvl == hs.shape[0] - 1:
                matching_tmp = tmp.detach()
            # V40/V43: GCR — detached geometric refinement
            # Reads tmp.detach() (+ h_cur.detach() when not coord_only) → zero gradient to h_cur.
            # V49: coord_only=True → GCR uses ONLY coordinates, no h_cur dependency
            #      → eliminates read-after-write coupling with SFA
            # V43: multi-layer GCR — each decoder layer gets its own GCR module.
            # Layers from (total_layers - gcr_num_layers) upward get GCR applied.
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
                    # v3: Semantic-guided geometric refinement.
                    # When SCGR + CMFE are both active, compute estimated control
                    # point coordinates (before correction) and pass text_score_map
                    # to SCGR for semantic gating. This enables cross-level
                    # CLIP semantic guidance: CMFE's text-likelihood controls
                    # SCGR's correction strength per-point.
                    gcr_extra = {}
                    if self.gcr_use_scgr and text_score_map is not None:
                        _est = tmp.detach()
                        if reference.shape[-1] == 2:
                            if self.epqm:
                                _est = _est + reference
                            else:
                                _est = _est + reference[:, :, None, :]
                        else:
                            if self.epqm:
                                _est = _est + reference[..., :2]
                            else:
                                _est = _est + reference[:, :, None, :2]
                        _est = sigmoid_offset(_est, offset=self.sigmoid_offset)
                        if self.sigmoid_offset:
                            _est = _est + 0.5  # [-0.5, 0.5] → [0, 1]
                        gcr_extra['text_score_map'] = text_score_map
                        gcr_extra['point_coords'] = _est
                    correction, gcr_cls_bonus_last = gcr_mod(
                        tmp.detach(),
                        None if self.gcr_coord_only else h_cur.detach(),
                        return_cls_bonus=need_cls_bonus,
                        gamma=_gamma,    # V55: FiLM per-point scale
                        beta=_beta,      # V55: FiLM per-point shift
                        **gcr_extra,     # v3: semantic guidance for SCGR
                    )
                    tmp = tmp + correction
                    # V42: inject GCR geometric context into classification
                    if gcr_cls_bonus_last is not None:
                        outputs_class = outputs_class + gcr_cls_bonus_last
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
            # V47: MED — build clean matching_coord from raw delta (no GCR, same reference)
            if self.match_decouple and lvl == hs.shape[0] - 1:
                mt = matching_tmp  # already detached
                if reference.shape[-1] == 2:
                    if self.epqm:
                        mt = mt + reference
                    else:
                        mt = mt + reference[:, :, None, :]
                else:
                    if self.epqm:
                        mt = mt + reference[..., :2]
                    else:
                        mt = mt + reference[:, :, None, :2]
                matching_ctrl_points = sigmoid_offset(mt, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_ctrl_points': outputs_coord[-1]}

        # V47: MED — clean matching outputs for stable Hungarian matching
        # These are used by losses.py if present; model forward stays unchanged.
        if self.match_decouple and matching_logits is not None:
            out['matching_logits'] = matching_logits
            out['matching_ctrl_points'] = matching_ctrl_points

        # V13: attach SFA outputs for loss computation and inference fusion
        if self.use_sfa and sfa_info is not None:
            out['sfa_info'] = sfa_info
            out['sem_cos'] = sem_cos  # (B, N, 16) for inference

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {'pred_logits': a, 'pred_ctrl_points': b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
