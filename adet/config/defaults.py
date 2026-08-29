from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.GA_STEPS = 1  # gradient accumulation steps (effective batch = IMS_PER_BATCH * GA_STEPS)
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = False
_C.INPUT.CROP.CROP_INSTANCE = True

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# The options for the quality of box prediction
# It can be "ctrness" (as described in FCOS paper) or "iou"
# Using "iou" here generally has ~0.4 better AP on COCO
# Note that for compatibility, we still use the term "ctrness" in the code
_C.MODEL.FCOS.BOX_QUALITY = "ctrness"

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0

# The normalizer of the classification loss
# The normalizer can be "fg" (normalized by the number of the foreground samples),
# "moving_fg" (normalized by the MOVING number of the foreground samples),
# or "all" (normalized by the number of all samples)
_C.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
_C.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0

_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
_C.MODEL.FCOS.YIELD_PROPOSAL = False
_C.MODEL.FCOS.YIELD_BOX_FEATURES = False

# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.DLA = CN()
_C.MODEL.DLA.CONV_BODY = "DLA34"
_C.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.DLA.NORM = "FrozenBN"

# ---------------------------------------------------------------------------- #
# BAText Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BATEXT = CN()
_C.MODEL.BATEXT.VOC_SIZE = 96
_C.MODEL.BATEXT.NUM_CHARS = 25
_C.MODEL.BATEXT.POOLER_RESOLUTION = (8, 32)
_C.MODEL.BATEXT.IN_FEATURES = ["p2", "p3", "p4"]
_C.MODEL.BATEXT.POOLER_SCALES = (0.25, 0.125, 0.0625)
_C.MODEL.BATEXT.SAMPLING_RATIO = 1
_C.MODEL.BATEXT.CONV_DIM = 256
_C.MODEL.BATEXT.NUM_CONV = 2
_C.MODEL.BATEXT.RECOGNITION_LOSS = "ctc"
_C.MODEL.BATEXT.RECOGNIZER = "attn"
_C.MODEL.BATEXT.CANONICAL_SIZE = 96  # largest min_size for level 3 (stride=8)
_C.MODEL.BATEXT.USE_COORDCONV = False
_C.MODEL.BATEXT.USE_AET = False
_C.MODEL.BATEXT.CUSTOM_DICT = "" # Path to the class file.

# ---------------------------------------------------------------------------- #
# BlendMask Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BLENDMASK = CN()
_C.MODEL.BLENDMASK.ATTN_SIZE = 14
_C.MODEL.BLENDMASK.TOP_INTERP = "bilinear"
_C.MODEL.BLENDMASK.BOTTOM_RESOLUTION = 56
_C.MODEL.BLENDMASK.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO = 1
_C.MODEL.BLENDMASK.POOLER_SCALES = (0.25,)
_C.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT = 1.0
_C.MODEL.BLENDMASK.VISUALIZE = False

# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3

# ---------------------------------------------------------------------------- #
# MEInst Head
# ---------------------------------------------------------------------------- #
_C.MODEL.MEInst = CN()

# This is the number of foreground classes.
_C.MODEL.MEInst.NUM_CLASSES = 80
_C.MODEL.MEInst.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.MEInst.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.MEInst.PRIOR_PROB = 0.01
_C.MODEL.MEInst.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.MEInst.INFERENCE_TH_TEST = 0.05
_C.MODEL.MEInst.NMS_TH = 0.6
_C.MODEL.MEInst.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.MEInst.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.MEInst.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.MEInst.POST_NMS_TOPK_TEST = 100
_C.MODEL.MEInst.TOP_LEVELS = 2
_C.MODEL.MEInst.NORM = "GN"  # Support GN or none
_C.MODEL.MEInst.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.MEInst.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.MEInst.LOSS_ALPHA = 0.25
_C.MODEL.MEInst.LOSS_GAMMA = 2.0
_C.MODEL.MEInst.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.MEInst.USE_RELU = True
_C.MODEL.MEInst.USE_DEFORMABLE = False
_C.MODEL.MEInst.LAST_DEFORMABLE = False
_C.MODEL.MEInst.TYPE_DEFORMABLE = "DCNv1"  # or DCNv2.

# the number of convolutions used in the cls and bbox tower
_C.MODEL.MEInst.NUM_CLS_CONVS = 4
_C.MODEL.MEInst.NUM_BOX_CONVS = 4
_C.MODEL.MEInst.NUM_SHARE_CONVS = 0
_C.MODEL.MEInst.CENTER_SAMPLE = True
_C.MODEL.MEInst.POS_RADIUS = 1.5
_C.MODEL.MEInst.LOC_LOSS_TYPE = 'giou'

# ---------------------------------------------------------------------------- #
# Mask Encoding
# ---------------------------------------------------------------------------- #
# Whether to use mask branch.
_C.MODEL.MEInst.MASK_ON = True
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.MEInst.IOU_THRESHOLDS = [0.5]
_C.MODEL.MEInst.IOU_LABELS = [0, 1]
# Whether to use class_agnostic or class_specific.
_C.MODEL.MEInst.AGNOSTIC = True
# Some operations in mask encoding.
_C.MODEL.MEInst.WHITEN = True
_C.MODEL.MEInst.SIGMOID = True

# The number of convolutions used in the mask tower.
_C.MODEL.MEInst.NUM_MASK_CONVS = 4

# The dim of mask before/after mask encoding.
_C.MODEL.MEInst.DIM_MASK = 60
_C.MODEL.MEInst.MASK_SIZE = 28
# The default path for parameters of mask encoding.
_C.MODEL.MEInst.PATH_COMPONENTS = "datasets/coco/components/" \
                                   "coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz"
# An indicator for encoding parameters loading during training.
_C.MODEL.MEInst.FLAG_PARAMETERS = False
# The loss for mask branch, can be mse now.
_C.MODEL.MEInst.MASK_LOSS_TYPE = "mse"

# Whether to use gcn in mask prediction.
# Large Kernel Matters -- https://arxiv.org/abs/1703.02719
_C.MODEL.MEInst.USE_GCN_IN_MASK = False
_C.MODEL.MEInst.GCN_KERNEL_SIZE = 9
# Whether to compute loss on original mask (binary mask).
_C.MODEL.MEInst.LOSS_ON_MASK = False

# ---------------------------------------------------------------------------- #
# CondInst Options
# ---------------------------------------------------------------------------- #
_C.MODEL.CONDINST = CN()

# the downsampling ratio of the final instance masks to the input image
_C.MODEL.CONDINST.MASK_OUT_STRIDE = 4
_C.MODEL.CONDINST.BOTTOM_PIXELS_REMOVED = -1

# if not -1, we only compute the mask loss for MAX_PROPOSALS random proposals PER GPU
_C.MODEL.CONDINST.MAX_PROPOSALS = -1
# if not -1, we only compute the mask loss for top `TOPK_PROPOSALS_PER_IM` proposals
# PER IMAGE in terms of their detection scores
_C.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM = -1

_C.MODEL.CONDINST.MASK_HEAD = CN()
_C.MODEL.CONDINST.MASK_HEAD.CHANNELS = 8
_C.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 3
_C.MODEL.CONDINST.MASK_HEAD.USE_FP16 = False
_C.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS = False

_C.MODEL.CONDINST.MASK_BRANCH = CN()
_C.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = 8
_C.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.CONDINST.MASK_BRANCH.CHANNELS = 128
_C.MODEL.CONDINST.MASK_BRANCH.NORM = "BN"
_C.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS = 4
_C.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = False

# The options for BoxInst, which can train the instance segmentation model with box annotations only
# Please refer to the paper https://arxiv.org/abs/2012.02310
_C.MODEL.BOXINST = CN()
# Whether to enable BoxInst
_C.MODEL.BOXINST.ENABLED = False
_C.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 10

_C.MODEL.BOXINST.PAIRWISE = CN()
_C.MODEL.BOXINST.PAIRWISE.SIZE = 3
_C.MODEL.BOXINST.PAIRWISE.DILATION = 2
_C.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000
_C.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3

# ---------------------------------------------------------------------------- #
# TOP Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = "conv"
_C.MODEL.TOP_MODULE.DIM = 16

# ---------------------------------------------------------------------------- #
# BiFPN options
# ---------------------------------------------------------------------------- #

_C.MODEL.BiFPN = CN()
# Names of the input feature maps to be used by BiFPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.OUT_CHANNELS = 160
_C.MODEL.BiFPN.NUM_REPEATS = 6

# Options: "" (no norm), "GN"
_C.MODEL.BiFPN.NORM = ""

# ---------------------------------------------------------------------------- #
# SOLOv2 Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SOLOV2 = CN()

# Instance hyper-parameters
_C.MODEL.SOLOV2.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.SOLOV2.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
_C.MODEL.SOLOV2.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.SOLOV2.SIGMA = 0.2
# Channel size for the instance head.
_C.MODEL.SOLOV2.INSTANCE_IN_CHANNELS = 256
_C.MODEL.SOLOV2.INSTANCE_CHANNELS = 512
# Convolutions to use in the instance head.
_C.MODEL.SOLOV2.NUM_INSTANCE_CONVS = 4
_C.MODEL.SOLOV2.USE_DCN_IN_INSTANCE = False
_C.MODEL.SOLOV2.TYPE_DCN = 'DCN'
_C.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]
# Number of foreground classes.
_C.MODEL.SOLOV2.NUM_CLASSES = 80
_C.MODEL.SOLOV2.NUM_KERNELS = 256
_C.MODEL.SOLOV2.NORM = "GN"
_C.MODEL.SOLOV2.USE_COORD_CONV = True
_C.MODEL.SOLOV2.PRIOR_PROB = 0.01

# Mask hyper-parameters.
# Channel size for the mask tower.
_C.MODEL.SOLOV2.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.SOLOV2.MASK_IN_CHANNELS = 256
_C.MODEL.SOLOV2.MASK_CHANNELS = 128
_C.MODEL.SOLOV2.NUM_MASKS = 256

# Test cfg.
_C.MODEL.SOLOV2.NMS_PRE = 500
_C.MODEL.SOLOV2.SCORE_THR = 0.1
_C.MODEL.SOLOV2.UPDATE_THR = 0.05
_C.MODEL.SOLOV2.MASK_THR = 0.5
_C.MODEL.SOLOV2.MAX_PER_IMG = 100
# NMS type: matrix OR mask.
_C.MODEL.SOLOV2.NMS_TYPE = "matrix"
# Matrix NMS kernel type: gaussian OR linear.
_C.MODEL.SOLOV2.NMS_KERNEL = "gaussian"
_C.MODEL.SOLOV2.NMS_SIGMA = 2

# Loss cfg.
_C.MODEL.SOLOV2.LOSS = CN()
_C.MODEL.SOLOV2.LOSS.FOCAL_USE_SIGMOID = True
_C.MODEL.SOLOV2.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.SOLOV2.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT = 1.0
_C.MODEL.SOLOV2.LOSS.DICE_WEIGHT = 3.0


# ---------------------------------------------------------------------------- #
# (Deformable) Transformer Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.USE_POLYGON = False
_C.MODEL.TRANSFORMER.ENABLED = True
_C.MODEL.TRANSFORMER.INFERENCE_TH_TEST = 0.3
# GQ: geometric quality sqrt re-ranking (pure post-process, no training change)
_C.MODEL.TRANSFORMER.GQ_REORDER = False   # True = sqrt(base_score * geom_quality)
# V11: dual-path uncertainty (disabled by default, set in per-experiment configs)
_C.MODEL.TRANSFORMER.INFERENCE_CLS_UNC_WEIGHT = 0.0   # w_cls: 0 = off
_C.MODEL.TRANSFORMER.INFERENCE_REG_UNC_WEIGHT = 0.0   # w_reg: 0 = off
_C.MODEL.TRANSFORMER.INFERENCE_UNC_TEMP = 1.0
_C.MODEL.TRANSFORMER.INFERENCE_AUX_ENSEMBLE = False

_C.MODEL.TRANSFORMER.VOC_SIZE = 96
_C.MODEL.TRANSFORMER.NUM_CHARS = 25
_C.MODEL.TRANSFORMER.AUX_LOSS = True
_C.MODEL.TRANSFORMER.ENC_LAYERS = 6
_C.MODEL.TRANSFORMER.DEC_LAYERS = 6
_C.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 1024
_C.MODEL.TRANSFORMER.HIDDEN_DIM = 256
_C.MODEL.TRANSFORMER.DROPOUT = 0.1
_C.MODEL.TRANSFORMER.NHEADS = 8
_C.MODEL.TRANSFORMER.NUM_QUERIES = 100
_C.MODEL.TRANSFORMER.ENC_N_POINTS = 4
_C.MODEL.TRANSFORMER.DEC_N_POINTS = 4
_C.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE = 6.283185307179586  # 2 PI
_C.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS = 4
_C.MODEL.TRANSFORMER.NUM_CTRL_POINTS = 16

_C.MODEL.TRANSFORMER.EPQM = False # for DPText-DETR
_C.MODEL.TRANSFORMER.EFSA = False
_C.MODEL.TRANSFORMER.USE_CLIP_LANG_PRIOR = False  # legacy: CLIP language prior (removed)

# ----------------------------------------------------------------------
# CLIP Dense Fusion (DRTP-v2)
# ----------------------------------------------------------------------
# Injects frozen CLIP patch tokens into FPN via dense projection +
# spatial-aware fusion BEFORE the encoder. Enables the encoder to
# generate semantically-informed proposals.
_C.MODEL.TRANSFORMER.USE_CLIP = False                # master switch for CLIP feature extraction
_C.MODEL.TRANSFORMER.CLIP_DENSE_FUSION = False       # enable dense CLIP-to-FPN fusion
_C.MODEL.TRANSFORMER.CLIP_SHUFFLE = False            # (ablation) permute CLIP patch tokens
_C.MODEL.TRANSFORMER.CLIP_REPLACE_NOISE = False      # (ablation) replace CLIP tokens with noise
_C.MODEL.TRANSFORMER.CLIP_TOKEN_MIX = False          # training-only partial token mixing
_C.MODEL.TRANSFORMER.CLIP_TOKEN_MIX_LAMBDA = 0.1     # mixing strength (0=aligned, 1=shuffled)
_C.MODEL.TRANSFORMER.CLIP_PROMPT_GATE = False        # learnable prompt-based gating
_C.MODEL.TRANSFORMER.CLIP_QUERY_FUSION = False       # query-level CLIP fusion (alt path)
_C.MODEL.TRANSFORMER.CLIP_QUERY_ROUTING = False      # CQR: query-conditioned CLIP routing
_C.MODEL.TRANSFORMER.CLIP_PROBE_MODE = "none"        # eval-only intervention: none|cross_image|shuffle|mean_broadcast
_C.MODEL.TRANSFORMER.CLIP_BACKBONE = "ViT-B-16"      # CLIP model architecture
_C.MODEL.TRANSFORMER.CLIP_PRETRAINED = "pretrain/clip-vit-base-patch16"
_C.MODEL.TRANSFORMER.CLIP_FREEZE = True              # freeze CLIP backbone
_C.MODEL.TRANSFORMER.CLIP_GATE_INIT = 0.0            # initial gate value
_C.MODEL.TRANSFORMER.CLIP_DROPOUT = 0.0              # dropout on CLIP fusion
_C.MODEL.TRANSFORMER.CLIP_ALPHA_INIT = 0.5           # alpha init for projection
_C.MODEL.TRANSFORMER.CLIP_KEEP_ASPECT = True         # keep aspect ratio in CLIP preprocess

# -- DRTP Internal Ablation (Phase 1) --
_C.MODEL.TRANSFORMER.CLIP_ACTIVE_LEVELS = [0, 1, 2, 3]  # which FPN levels get CLIP (0=P3, 1=P4, ...)
_C.MODEL.TRANSFORMER.CLIP_USE_GATE = True                # False → skip GateNet, direct α*V injection
_C.MODEL.TRANSFORMER.CLIP_LEARNABLE_ALPHA = True         # False → α frozen at CLIP_FIXED_ALPHA_VALUE
_C.MODEL.TRANSFORMER.CLIP_FIXED_ALPHA_VALUE = 0.5        # scalar α when CLIP_LEARNABLE_ALPHA=False
_C.MODEL.TRANSFORMER.CLIP_SHARED_PROJECTOR = False       # True → one shared projector for all levels
_C.MODEL.TRANSFORMER.CLIP_DIRECTIONAL_GATE = False     # True → direction-aware GateNet (1×7 + 7×1 strip conv)

# ----------------------------------------------------------------------
# Self-Gated Fusion (SGF): CLIP-free variant of DRTP's dense gated residual.
# Reuses DRTP's per-level structure (DirectionalGateNet + learnable alpha +
# per-level independent enhancement) but generates the enhancement V from the
# FPN features themselves:
#     V_l = SelfGen_l(F_l);   F'_l = F_l + alpha_l * G_l * V_l
# Zero CLIP runtime cost. DRTP-noise ablation (F1 88.45 vs DRTP-P3 88.45)
# showed ~90% of DRTP's gain comes from the structure, not the CLIP content,
# so SGF preserves the gain while dropping the CLIP dependency.
_C.MODEL.TRANSFORMER.SELF_GATED_FUSION = False          # enable SGF (no CLIP needed)
_C.MODEL.TRANSFORMER.SGF_ACTIVE_LEVELS = [0, 1, 2]      # which FPN levels get SGF (0=P3, 1=P4, ...)
_C.MODEL.TRANSFORMER.SGF_USE_GATE = True                # False → skip GateNet, direct α*V injection
_C.MODEL.TRANSFORMER.SGF_DIRECTIONAL_GATE = True        # True → direction-aware GateNet (1×7 + 7×1 strip conv)
_C.MODEL.TRANSFORMER.SGF_LEARNABLE_ALPHA = True         # False → α frozen at SGF_FIXED_ALPHA_VALUE
_C.MODEL.TRANSFORMER.SGF_FIXED_ALPHA_VALUE = 0.5        # scalar α when SGF_LEARNABLE_ALPHA=False

# ----------------------------------------------------------------------
# RICA (Residual-guided Instance Classification Attention)
# ----------------------------------------------------------------------
# Reuses the FINAL decoder cross-attention sampling geometry to read DRTP
# residuals (P3/P4/P5), forming a classification-only feature h_cls:
#   h_cls  = h_final + rica_proj([h_final, r, h_final*r, |h_final-r|])
#   final_logits = class_embed(h_cls)
#   final_points = point_embed(h_final)      # geometry path untouched
# No auxiliary loss, no matcher change, no new hyper-parameters.
# sampling_locations / attention_weights are detached; residuals and h_final
# are NOT detached (DRTP/decoder can co-adapt with the classification task).
_C.MODEL.TRANSFORMER.RICA_ENABLED = False                # requires USE_CLIP=True

# ----------------------------------------------------------------------
# PRICA (Point-Residual Intra-query Classification Aggregation)
# ----------------------------------------------------------------------
# The unified module = RICA (point-wise residual evidence extraction) + PAGA
# (consistency-aware instance aggregation). Paper narrative: DRTP + PRICA.
#   rica_h       = query + rica_point        # point-wise RICA evidence
#   h_cls        = rica_h + point_delta      # RICA path KEPT + signed residual
#   final_logits = class_embed(h_cls)        # classification path only
#   final_points = point_embed(query)        # geometry path untouched
# delta is ZERO at init -> PRICA starts as the identity on the original
# classification path (preserves the RICA baseline, no warmup / no scale).
# No new loss, no temperature, no top-k, no gate, no threshold.
# PRICA_ENABLED takes precedence over RICA_ENABLED; requires USE_CLIP=True.
#
# PRICA_POINT_FEAT: ablation switch for the delta input (structure only,
# no new loss / no new hyper-parameter — same per-point focal loss):
#   "rica+ctx+agree" (E1, default) point_feature = [rica_point, mean/std ctx,
#                                                   agreement, -agreement]
#   "rica+agree"     (E2)          point_feature = [rica_point, agreement, -a]
#   "rica"           (E3)          point_feature = rica_point
# E1 keeps the v2 delta_proj input shape (3C+2) so old checkpoints load.
_C.MODEL.TRANSFORMER.PRICA_ENABLED = False
_C.MODEL.TRANSFORMER.PRICA_POINT_FEAT = "rica+ctx+agree"

# ----------------------------------------------------------------------
# CTP: CLIP Text Prior (CLIP text-prototype alignment of query cls logits)
# ----------------------------------------------------------------------
# CLIP-derived text/confuser prototypes supervise the decoder query
# features (softplus margin) as a training-time auxiliary loss, PLUS a
# classification-only residual adapter (q_cls = q + alpha * A(sg(q)))
# active at both training and inference. Regression path untouched.
# Semantic prior -> pairs with TACT (geometric correction).
# See 结构/ta.txt for the design history (v1 -> noneg-nodiffw -> C/D/E).
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR = CN()
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.ENABLED = False              # master switch
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.USE_ADAPTER = True           # classification residual adapter (q_cls = q + alpha*A(sg(q)))
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.WEIGHT = 0.01                # auxiliary loss weight
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.TEMPERATURE = 0.10           # softplus temperature
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.USE_CONFUSER = True          # use 6 confuser prototypes
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.DIFFICULTY_WEIGHT = False    # nodiffw: no difficulty weighting
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.USE_NEG_BRANCH = False       # noneg: no negative branch
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.PROTO_POOLING = "max"        # max|mean: aggregate over prototypes
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.LOSS_DETACH = True           # CTP loss input detach
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.ADAPTER_DETACH_INPUT = True  # adapter input detach
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.MATCHER_BASE_LOGITS = True   # matcher uses base logits (D/E)
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.IOU_GATE = 0.5               # E: only matched queries with pred-IoU >= gate
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.PROTO_DIM = 512              # CLIP text embedding dim
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.ADAPTER_ALPHA_INIT = 0.0     # alpha init (0 = identity at start)
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.USE_MARGIN_MOD = False       # DAG fix (E2'): route projector margin into cls logits (scalar, proven ~0)
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.MARGIN_MOD_BETA_INIT = 0.0   # beta init (0 = identity at start)
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.CLIP_PRETRAINED = "pretrain/clip-vit-base-patch16"
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.TEXT_PROMPTS = [
    "scene text",
    "a word in a natural scene",
    "vertical scene text",
    "artistic but readable lettering",
    "text on a signboard",
    "a street sign with text",
    "printed text on a wall",
    "a shop name written in letters",
    "horizontal text in an image",
    "a label with letters",
    "handwritten text",
    "text in a photograph",
]
_C.MODEL.TRANSFORMER.CLIP_TEXT_PRIOR.CONFUSER_PROMPTS = [
    "decorative graphic without readable text",
    "letter-like shapes that are not actual text",
    "a texture pattern without words",
    "an abstract logo design",
    "graphic elements that resemble letters",
    "ornamental shapes in an image",
]

# ----------------------------------------------------------------------
# TACT: Topology-Aware Curvature Transform
# ----------------------------------------------------------------------
# Pure-visual geometric module, injected at every decoder layer after
# intra-SA + circonv (see 结构/tact.txt):
#   1. Gaussian Topology Aggregation — control-point distance weights share features
#   2. Curvature-Aware Gating (Perona-Malik) — corners suppress, flats boost residual
#   3. FiLM Circonv Modulation — scale-conditioned circular-conv modulation
# All new params zero-init -> training step 1 == exact baseline.
_C.MODEL.TRANSFORMER.USE_TACT = False
_C.MODEL.TRANSFORMER.TACT_USE_CURA = True       # curvature-aware gating (default ON)
_C.MODEL.TRANSFORMER.TACT_SIGMA_RELAX = False
_C.MODEL.TRANSFORMER.TACT_DECOUPLED = False

# ----------------------------------------------------------------------
# GCR: Geometric Context Refinement (几何上下文细化)
# ----------------------------------------------------------------------
# 纯几何修正旁路，注入到回归路径（FINAL decoder layer）：
#   tmp = ctrl_point_coord(h_cur)          # (B,N,16,2) 未 sigmoid
#   tmp = tmp + GCR(tmp.detach(), h_cur.detach())
#   outputs_coord = sigmoid(tmp + reference)
# RingConv 环形卷积（k=3 DW→PW→k=5 DW, circular pad）+ zero-init
# output_proj → 训练起点 == baseline。数据集无关（不依赖 CLIP）。
# 历史 GCR-only = 88.67（最强单模块）；coord-only 太弱（V49=87.80）。
_C.MODEL.TRANSFORMER.USE_GCR = False
_C.MODEL.TRANSFORMER.GCR_COORD_ONLY = False   # False=读 h_cur 特征（推荐）；True=仅坐标
_C.MODEL.TRANSFORMER.GCR_HIDDEN_DIM = 128
_C.MODEL.TRANSFORMER.GCR_ONLY_FINAL = True    # 仅在 FINAL decoder layer 注入

# ----------------------------------------------------------------------
# GATP: Geometry-Aware Text Prototype (几何感知文本原型)
# ----------------------------------------------------------------------
# 用几何特征（曲率/尺度）条件化的可学习文本原型，margin 注入分类 logits
# （FINAL decoder layer，分类路径）。控制点 detach → 不影响回归。
# 纯几何、数据集无关 → 补 CTP 跨数据集泛化短板。
# 历史 GATP (SEED=42, CTW1500): P=90.29 / R=86.79 / F1=88.51；
# P 暴跌（原型太宽松）→ 本版收紧：温度 + 可学习 beta 非零初始化。
_C.MODEL.TRANSFORMER.USE_GATP = False
_C.MODEL.TRANSFORMER.GATP_PROTO_DIM = 256
_C.MODEL.TRANSFORMER.GATP_NUM_CONFUSER = 6
_C.MODEL.TRANSFORMER.GATP_TEMPERATURE = 0.10
_C.MODEL.TRANSFORMER.GATP_BETA_INIT = 0.05
_C.MODEL.TRANSFORMER.GATP_BETA_TRAINABLE = True

_C.MODEL.TRANSFORMER.LOSS = CN()
_C.MODEL.TRANSFORMER.LOSS.AUX_LOSS = True
_C.MODEL.TRANSFORMER.LOSS.POINT_CLASS_WEIGHT = 2.0
_C.MODEL.TRANSFORMER.LOSS.POINT_COORD_WEIGHT = 5.0
_C.MODEL.TRANSFORMER.LOSS.BOX_CLASS_WEIGHT = 2.0
_C.MODEL.TRANSFORMER.LOSS.BOX_COORD_WEIGHT = 5.0
_C.MODEL.TRANSFORMER.LOSS.BOX_GIOU_WEIGHT = 2.0
_C.MODEL.TRANSFORMER.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.TRANSFORMER.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.TRANSFORMER.LOSS.POINT_VAR_WEIGHT = 0.0  # V10: regression uncertainty (0 = off, enable in config)
_C.MODEL.TRANSFORMER.LOSS.CLS_VAR_WEIGHT = 0.0    # V11: classification uncertainty (0 = off)
_C.MODEL.TRANSFORMER.LOSS.REG_VAR_NEG_WEIGHT = 0.0  # V11: negative-sample reg uncertainty (0 = off)
_C.MODEL.TRANSFORMER.LOSS.REG_VAR_NEG_TARGET = 0.0  # V11: negative-sample reg log_var target
_C.MODEL.TRANSFORMER.LOSS.AUX_ENSEMBLE_LAYERS = 3


_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.LR_BACKBONE = 1e-5
_C.SOLVER.LR_BACKBONE_NAMES = []
_C.SOLVER.LR_LINEAR_PROJ_NAMES = []
_C.SOLVER.LR_LINEAR_PROJ_MULT = 0.1
# Semi-joint warm start (stage 2): multiplier applied to already-trained
# decoder/head params. New RG-DPU params use SOLVER.BASE_LR directly.
# 1.0 = disabled (all params use BASE_LR as usual).
_C.SOLVER.WARMSTART_OLD_LR_FACTOR = 1.0


_C.TEST.DET_ONLY = True
_C.TEST.USE_LEXICON = False
# 1 - Full lexicon (for totaltext, ctw1500...)
_C.TEST.LEXICON_TYPE = 1
_C.TEST.WEIGHTED_EDIT_DIST = False
