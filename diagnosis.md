# DPText-DETR 改进全历程诊断文档

> 本文档提取全部实验记忆和当前困境，用于向其他模型/研究者求助。
> 生成时间: 2026-07-09

---

## 1. 任务与基座

**任务**: 弯曲文本检测 (CTW1500 数据集)
**基座模型**: DPText-DETR (Deformable DETR 变体, ResNet-50 backbone, 6层decoder, 16个控制点)

**Baseline 结果**: P=90.95, R=84.90, F1=87.82

**Decoder 结构 (每层)**:
- `h_cur = SelfAttn + CrossAttn + FFN` → (B, N, 16, 256)
- `cls_logit = Linear(h_cur)` → (B, N, 16, 1)
- `reg_delta = MLP(h_cur)` → (B, N, 16, 2) → +ref → sigmoid → pred_ctrl_points
- 匈牙利匹配 + FocalLoss(分类) + L1Loss(坐标回归)

---

## 2. 两条改进路线

### A. GCR (Geometric Context Refinement) — 纯几何修正

**核心思路**: 在 reg_head 输出端加独立几何修正旁路，用 `.detach()` 斩断梯度回流到 h_cur，避免 P/R 跷跷板。

**架构**:
```
h_cur → reg_head → tmp
                     │
tmp.detach() ──────→ GCR:
  ├── coord_embed(tmp坐标, 2→128)
  ├── feat_proj(h_cur.detach(), 256→128)  ← 可选, coord_only模式跳过
  ├── RingConv(k=3 DW→PW→k=5 DW, circular pad)
  └── output_proj(128→2, zero-init)
                       │
tmp = tmp + correction → sigmoid → pred
```

**关键**: zero-init → 训练初期等同baseline → 渐进学习几何修正 → 梯度完全不碰h_cur

### B. SFA (Semantic Feature Alignment) — 语义对齐增强

**核心思路**: 用 CLIP text encoder 提取"text"语义锚点，将控制点特征投影到 CLIP 语义空间，计算 cosine similarity 作为分类偏置。

**架构**:
```
CLIP Text Encoder (冻结)
  prompt: "a photo of text"
  → c_text_raw ∈ R^512
  → text_proj (512→256+LN) → c_text ∈ R^256

h_cur (最后一层decoder)
  → semantic_proj (256→ReLU→256)
  → feat_proj ∈ R^256
  → L2_norm(feat_proj) · L2_norm(c_text)
  → cos_sim ∈ [-1,+1]
  → logit_scale × cos_sim → semantic_logit

cls_logit = cls_head(h_cur) + semantic_logit
```

**额外损失**: `loss_sfa_align = mean(1 - cos(feat_proj_pos, c_text)) + bg_weight × relu(cos(feat_proj_neg, c_text) - margin)`

---

## 3. 完整实验结果

> 数据集: CTW1500, 全部 A100 训练

| 版本 | P | R | F1 | 关键改动 |
|------|------|------|------|------|
| **baseline** | 90.95 | 84.90 | **87.82** | 纯视觉, 无额外模块 |
| **SFA-only** (V22) | 91.38 | 85.60 | **88.40** | SFA 单层, 单prompt, 无残差 |
| **GCR-only** | 91.80 | 85.75 | **88.67** | 纯几何修正, 读取h_cur |
| **V40** (SFA+GCR) | 89.84 | 86.50 | 88.13 | SFA写h_cur → GCR读h_cur冲突 → P崩 |
| **V47** (SFA+GCR+MED) | 91.12 | 85.49 | 88.21 | MED全解耦 → R被砍 |
| **V49** (SFA+GCR+coord_only) | 90.75 | 85.04 | **87.80** | coord_only GCR太弱 + 匹配不一致 |
| **V50** (SFA V2) | 91.59 | 85.12 | **88.24** | multi-prompt+learnable_proto+3层 → 比V1更差 |

### 关键观测

1. **GCR 是最强单模块** (88.67), SFA 次之 (88.40)
2. **SFA + GCR 无法共存**: 所有组合版本 (V40/V47/V49) 均低于单独 GCR
3. **SFA V2 比 V1 更差**: 加了 3 个 prompt、可学残差、3 层投影 → R 从 85.60 掉到 85.12
4. **V49 甚至比什么都不加还差**: F1=87.80 < baseline 87.82

---

## 4. SFA V2 的根因分析

V50 比 SFA V1 差的三个技术原因:

### 4.1 multi-prompt max() 与对齐 loss 不一致
```python
# Forward: max 取 3 个原型中最高的
all_cos = einsum("bnpd,kd->bnpk", feat_proj_norm, c_text_norm)  # (B,N,16,3)
semantic_logit = all_cos.max(dim=-1).values  # 梯度只流向赢家

# Alignment loss: 只推 feat_proj 向 c_text[0] 对齐
c_text_loss = c_text[0]  # 永远是 "a photo of text"
```
→ proto[1] 和 proto[2] 没有对齐监督, 漂移出随机方向, max() 输出噪声

### 4.2 共享 prototype_residual 杀死多样性
```python
c = text_proj(raw)          # (3, 256) 三个不同方向
c = c + prototype_residual  # 同一个 256 维残差加到所有 3 个原型
```
→ 3 个原型被拉到同一方向, multi-prompt 多语义覆盖变伪命题

### 4.3 3 层投影仅 1 层有对齐 loss
层 3/4 的 `sfa_layer_projs[0]/[1]` 只收到 CE 梯度 (通过 semantic_logit → focal loss), 不收到对齐 loss。CE 梯度嘈杂 → 两个额外投影层是随机游走噪声源。

---

## 5. 代码架构

### 关键文件
| 文件 | 作用 |
|------|------|
| `semantic_feature_alignment.py` | SFA 模块 (CLIP+投影+对齐loss) |
| `geometric_context_refinement.py` | GCR 模块 (coord_embed+RingConv+残差修正) |
| `models.py` | 主模型, decoder循环, SFA/GCR注入点 |
| `losses.py` | SetCriterion, 匈牙利匹配, loss计算 |
| `defaults.py` | 所有config默认值 |

### SFA 当前设计 (V1, 已清理V50代码)
```python
class SemanticFeatureAlignment(nn.Module):
    def __init__(self, d_model, clip_model_path, agg_mode):
        self.clip_text_model   # CLIPTextModel, 冻结
        self._c_text_raw       # CLIP encode("a photo of text")
        self.text_proj         # Linear(512→256) + LayerNorm
        self.semantic_proj     # Linear→ReLU→Linear (256→256)
        self.logit_scale_raw   # nn.Parameter(1.0)

    def forward(self, feat, unc=None):
        c_text = text_proj(c_text_raw)           # (256,)
        feat_proj = semantic_proj(feat)           # (B,N,16,256)
        cos_sim = dot(L2(feat_proj), L2(c_text))  # ∈ [-1,1]
        return logit_scale * cos_sim, cos_sim, {"feat_proj":..., "c_text":...}

    def compute_alignment_loss(feat_proj, c_text, pos_idx, neg_mask, ...):
        # pos: mean(1 - cos(pos_feat, c_text))
        # neg: relu(cos(neg_feat, c_text) - margin).mean()
```

### SFA 注入点 (models.py)
```
Layer 0~4: cls_head(h_cur) → aux loss (无SFA)
Layer 5:   cls_head(h_cur) + SFA.forward(h_cur) → main loss + alignment loss
```

### GCR 设计
```python
class GeometricContextRefinement(nn.Module):
    def __init__(self, d_model, num_points, hidden_dim, coord_only=False):
        self.coord_embed   # Linear(2→128) + GELU + Linear(128→128)
        self.feat_proj     # Linear(256→128), coord_only模式下为None
        self.ring_conv     # DWConv(k=3)→PWConv→DWConv(k=5)
        self.output_proj   # Linear(128→2), zero-init

    def forward(self, tmp_detached, h_cur_detached=None):
        coord_feat = coord_embed(tmp)
        if not coord_only: coord_feat = coord_feat + feat_proj(h_cur)
        correction = ring_conv(coord_feat)
        correction = output_proj(correction)  # zero-init → 0 at start
        return correction, cls_bonus_optional
```

---

## 6. 当前困境

### 核心问题
**花了大量精力在 SFA 方向上迭代, 但始终追不上 GCR 的 88.67。SFA 的"天花板"似乎就是 ~88.4。**

### 尝试过但失败的路径
1. ❌ SFA + GCR 共存 (read-after-write 冲突)
2. ❌ coord-only GCR 解耦 (太弱, V49=87.80)
3. ❌ MED 全解耦 (R 被砍, V47=88.21)
4. ❌ 更多 CLIP 原型 (multi-prompt, V50=88.24)
5. ❌ 可学原型残差 (learnable_proto, V50 内含)
6. ❌ 多层 SFA 注入 (3 层, V50 内含)
7. ❌ SFA 读 encoder 特征 (SFA_V2 失败, 未记录)

### 未被回答的关键问题
1. **SFA 的 F1 天花板是否就是 ~88.4?** 如果是, CLIP 语义信息对此任务的价值有限
2. **GCR 为什么单模块就 88.67?** 纯几何信号 > 语义对齐, 说明 CTW1500 的核心难点在定位而非分类
3. **有没有能让 SFA 与 GCR 真正正交的融合方式?** (目前的加法注入不可行)
4. **是否应该放弃 SFA, 全力迭代 GCR?** 还是存在更根本的重新设计空间?

---

## 7. 约束条件

1. sigmoid 输出, 不用 tanh
2. CLIP backbone 保持冻结
3. SPP (空间金字塔池化) 必须保留
4. Post-FFN 注入 (不在 SelfAttn/CrossAttn 内注入)
5. 不改 h_cur 分布 (避免破坏 pretrained decoder 行为)

---

## 8. 求助问题

1. **如何判断 SFA 是否已经碰到天花板?** 是否应该定量分析 semantic_logit 对最终分类的实际贡献度?
2. **有没有新的融合范式可以替代"加法注入"?** 比如 attention-based 的软融合、门控机制、或者双塔结构?
3. **GCR 还有多大提升空间?** 当前的 RingConv 设计是否足够, 还是应该探索更强的几何编码 (如 GNN, 图卷积)?
4. **是否需要换一个角度?** 比如不改善分类/回归, 而改善匹配策略 (match cost)、后处理、或者 NMS?
