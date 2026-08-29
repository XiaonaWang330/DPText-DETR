"""
TA: Text Alignment（文本对齐辅助监督 + 分类专属残差 adapter）

训练期辅助损失：用冻结 CLIP text encoder 构建 text/confuser prototypes，
监督 decoder query 特征与"文本原型"的对齐 margin，迫使 query 语义上
"像文本、不像 confuser"。

设计（C/D/E 变体，详见 结构/ta.txt）：
- TA loss 仅训练期旁路，推理时整条旁路删除，零推理开销。
- 分类专属残差 adapter（训练+推理均生效，只影响分类路径）：
      q_cls = q + alpha * A(sg(q))   (q_box = q 不变)
- 双 detach 解耦：
      TA_LOSS_DETACH:        query_feat.detach() → TA loss 梯度止于 projector
      ADAPTER_DETACH_INPUT:  adapter 输入 detach → adapter 不反向推动共享层
- E 变体 IoU 门控：仅 pred-IoU >= IOU_GATE 的 matched query 参与 TA loss。
- E 变体 matcher 解耦：Hungarian matcher 分类 cost 用 base logits（不加 adapter）。
- margin 调制（DAG 修复，USE_MARGIN_MOD 开关，默认关）：
      logits += beta * m * mod_dir   （把 projector 的 margin 接回分类 logits）

原型构建：
    CLIPModel + CLIPTokenizer 编码 prompts → L2-normalized → 冻结 buffer。
    构建完成后 CLIP 从显存删除，训练/推理不占额外显存。

NOTE（88.66 复现）：本文件已回退到 88.66 时代版本，只保留
    adapter / projector / alpha / beta / mod_dir 五个可训练参数。
    V3（clip_cls_proj + gamma）、V4（use_margin_feature）、V5
    （clip_pixel_margin 像素注入）等后续实验分支已删除。
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPModel, CLIPTokenizer
    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    _HAS_TRANSFORMERS = False

# 默认 12 个文本原型 prompt（前 4 个来自原版 ta.txt，其余为合理补全）
TEXT_PROMPTS_DEFAULT = [
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

# 默认 6 个 confuser 原型 prompt（前 2 个来自原版 ta.txt，其余为合理补全）
CONFUSER_PROMPTS_DEFAULT = [
    "decorative graphic without readable text",
    "letter-like shapes that are not actual text",
    "a texture pattern without words",
    "an abstract logo design",
    "graphic elements that resemble letters",
    "ornamental shapes in an image",
]


class TextPrototypeAlignmentHead(nn.Module):
    """
    组件：
        projector:          Linear(d_model -> proto_dim)   唯一参与 TA loss 的可训练参数
        text_prototypes:    (N_t, proto_dim) 冻结，L2-normalized
        confuser_prototypes:(N_c, proto_dim) 冻结，L2-normalized
        adapter A:          MLP(d_model -> d_model -> d_model)  分类残差
        alpha:              可学习标量（初始 0 → 训练起点恒等于 baseline）
        beta/mod_dir:       margin 调制标量（USE_MARGIN_MOD 才启用，beta init=0）

    前向：
        forward_adapter(q): q_cls = q + alpha * A(sg(q))        （逐点共享 MLP）
        forward_margin(f):  m = max_j cos(z, t_j) - max_k cos(z, c_k)
    """

    def __init__(
        self,
        d_model=256,
        proto_dim=512,
        use_confuser=True,
        temperature=0.10,
        clip_model_name="pretrain/clip-vit-base-patch16",
        text_prompts=None,
        confuser_prompts=None,
        adapter_detach_input=True,
        alpha_init=0.0,
        use_adapter=True,
        use_margin_mod=False,
        margin_mod_beta_init=0.0,
        proto_pooling="max",
    ):
        super().__init__()
        self.proto_pooling = proto_pooling
        self.temperature = temperature
        self.adapter_detach_input = adapter_detach_input
        self.use_adapter = use_adapter
        self.use_margin_mod = use_margin_mod

        # margin 调制（DAG 修复，E2′）：beta 可学习标量初始 0（起点恒等于
        # baseline），mod_dir 为 logits 空间调制方向（单类 text → 一维向量，
        # 初始 +1）。实测 beta→0.0016≈0 被证伪，保留仅为实验对照。
        self.beta = nn.Parameter(torch.tensor(float(margin_mod_beta_init)))
        self.mod_dir = nn.Parameter(torch.ones(1))

        # 分类残差 adapter A（逐点共享 MLP，量级与 projector 相当 ~0.13M）
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # projector：TA loss 的可训练投影
        self.projector = nn.Linear(d_model, proto_dim)

        # ── 冻结原型（由 CLIP text encoder 构建，构建后 CLIP 删除）──
        text_prompts = text_prompts or TEXT_PROMPTS_DEFAULT
        confuser_prompts = confuser_prompts or CONFUSER_PROMPTS_DEFAULT
        text_feats, confuser_feats = self._build_prototypes(
            clip_model_name, text_prompts, confuser_prompts, proto_dim
        )
        self.register_buffer("text_prototypes", text_feats)          # (N_t, proto_dim)
        if use_confuser:
            self.register_buffer("confuser_prototypes", confuser_feats)  # (N_c, proto_dim)
        else:
            self.register_buffer("confuser_prototypes", torch.zeros(0, proto_dim))

    def _build_prototypes(self, clip_model_name, text_prompts, confuser_prompts, proto_dim):
        """用冻结 CLIP text encoder 编码 prompts，返回 L2-normalized 原型（CPU tensor）。"""
        def _rand():
            t = F.normalize(torch.randn(len(text_prompts), proto_dim), dim=-1)
            c = F.normalize(torch.randn(len(confuser_prompts), proto_dim), dim=-1)
            return t, c

        if not _HAS_TRANSFORMERS:
            print("[TA] WARNING: transformers unavailable, using RANDOM prototypes (smoke only)")
            return _rand()
        if not os.path.isdir(clip_model_name):
            print(f"[TA] WARNING: CLIP dir '{clip_model_name}' not found, using RANDOM prototypes")
            return _rand()

        clip = CLIPModel.from_pretrained(clip_model_name)
        tok = CLIPTokenizer.from_pretrained(clip_model_name)
        with torch.no_grad():
            def _enc(prompts):
                inputs = tok(prompts, padding=True, return_tensors="pt")
                feats = clip.get_text_features(**inputs)  # (N, 512)
                return F.normalize(feats, dim=-1)
            text_feats = _enc(text_prompts)
            confuser_feats = _enc(confuser_prompts)
        del clip, tok
        return text_feats.cpu(), confuser_feats.cpu()

    def forward_adapter(self, q):
        """分类残差 adapter：q_cls = q + alpha * A(sg(q))。q: (..., d_model)
        USE_ADAPTER=False (ablation): identity, cls logits unchanged."""
        if not self.use_adapter:
            return q
        q_det = q.detach() if self.adapter_detach_input else q
        return q + self.alpha * self.adapter(q_det)

    def _pool_sims(self, sims):
        """sims: (..., K) prototype similarity → (...,) aggregated score.
        max: 取最相似原型（V1-E 原版）
        mean: 所有原型平均 → 每个原型都有梯度"""
        if self.proto_pooling == "mean":
            return sims.mean(dim=-1)
        return sims.max(dim=-1).values

    def _margin_from_z(self, z):
        """z: (..., proto_dim) L2-normalized → margin (...,)。"""
        m_t = self._pool_sims(z @ self.text_prototypes.T)
        if self.confuser_prototypes.shape[0] > 0:
            m_c = self._pool_sims(z @ self.confuser_prototypes.T)
        else:
            m_c = torch.zeros_like(m_t)
        return m_t - m_c

    def forward_margin(self, query_feat):
        """实例级 query 特征 → margin。query_feat: (B, N, d_model) → (B, N)"""
        z = F.normalize(self.projector(query_feat), dim=-1)  # (B, N, proto_dim)
        return self._margin_from_z(z)

    def margin_modulation(self, query_feat):
        """margin 调制（修复断路，E2′）：把 projector 学到的文本对齐 margin 注入分类 logits。
        query_feat: (B, N, d_model) → (B, N, 1) logit 残差（单类 text）。
        query_feat 必须已 detach（与 TA loss 同源），共享层保持解耦。"""
        z = F.normalize(self.projector(query_feat), dim=-1)  # (B, N, proto_dim)
        m = self._margin_from_z(z)  # (B, N)
        return self.beta * m.unsqueeze(-1) * self.mod_dir
