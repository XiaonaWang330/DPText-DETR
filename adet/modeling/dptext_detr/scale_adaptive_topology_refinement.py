"""
SATR: Scale-Adaptive Topology Refinement
=========================================
纯视觉几何模块，解决小文字特征塌缩和尺度自适应问题。

两个子模块:
1. Gaussian Intra-SA Bias: 基于控制点欧式距离的高斯加权邻居特征聚合
   - 小文字点距小 → 高斯核衰减快 → 防止特征同质化
   - 大文字点距大 → 高斯核保留基础权重 → 保持邻域通信

2. FiLM Circonv Modulation: 尺度因子 σ 通过仿射变换调制 circonv 输出
   - γ(σ), β(σ) 零初始化 → 初期输出 = 原版 circonv

版本谱系:
  V1 (decoupled=False, sigma_relax=False):
    共享 σ 预测器，保守带宽 [0.1, 0.2]，P=91.87 R=85.31 F1=88.46
  V2 (decoupled=False, sigma_relax=True):
    共享 σ + relax，带宽 [0.15, 0.35]，P=90.89 R=85.38 F1=88.05 (退化)
  V3 (decoupled=True):
    解耦 sigma + 几何面积先验 + 保守/自适应双轨

CURA 升级 (use_cura=True):
  - Perona-Malik 各向异性扩散: 曲率控制残差注入量
    · high curvature (corners) → modulation ≈ 0 → 保 P
    · low  curvature (smooth)  → modulation ≈ 1 → 救 R
  - kappa_gate 零初始化 → 训练第一步 = V1 完全一致
  - 仅 1 个新增参数 (~4 bytes), 比 V1 只多 2 行代码

所有新增参数零初始化 → 训练第一步与 baseline 完全一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SATR_Module(nn.Module):
    """
    Scale-Adaptive Topology Refinement Module.

    不修改 loss、不修改 matcher、不超过 0.15M 参数。
    """

    def __init__(self, d_model=256, num_ctrl_points=16,
                 sigma_relax=False, decoupled=False,
                 use_cura=False):
        super().__init__()
        self.d_model = d_model
        self.num_ctrl_points = num_ctrl_points
        self.sigma_relax = sigma_relax          # V2: dyn_sigma*2.0 + 0.05 floor
        self.decoupled = decoupled              # V3: decoupled sigma predictors + geo prior
        self.use_cura = use_cura                # CURA: curvature-aware residual aggregation

        # ── Sigma 预测器 ──
        if decoupled:
            # V3: 解耦双预测器
            # 高斯 sigma → 语义 + 几何面积，保守范围保 P
            self.gauss_sigma_pred = nn.Sequential(
                nn.Linear(d_model + 1, d_model // 4),   # +1 = bbox area
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )
            # FiLM scale → 纯语义，自适应范围
            self.film_scale_pred = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )
        else:
            # V1/V2: 共享 σ 预测器 (纯语义)
            self.scale_pred = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )

        # ── FiLM 尺度调制 (对 circonv 输出) ──
        # gamma/beta 全零初始化 → 训练第一步 h' = h * 1 + 0 = h
        self.film_gamma = nn.Linear(1, d_model)
        self.film_beta = nn.Linear(1, d_model)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # ── 高斯邻居特征聚合 ──
        # 高斯加权邻居特征投影后作为零初始化残差注入
        self.gauss_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gauss_proj.weight)
        nn.init.zeros_(self.gauss_proj.bias)

        # 高斯核的基带宽 (可学习, 初始化为合理值)
        self.base_sigma = nn.Parameter(torch.tensor(0.1))

        # ── CURA: 曲率感知残差门控 ──
        if use_cura:
            # kappa_gate 零初始化 → modulation=1 → 退化为 V1
            self.kappa_gate = nn.Parameter(torch.zeros(1))

    # ==================================================================
    # 几何特征提取
    # ==================================================================

    def compute_geometry_area(self, ref_points):
        """
        从控制点坐标计算外接矩形面积 (归一化坐标).
        Args:
            ref_points: (B, N, P, 2)
        Returns:
            area: (B, N, 1) ∈ [0, 1]
        """
        x_min = ref_points[..., 0].min(dim=2)[0]  # (B, N)
        x_max = ref_points[..., 0].max(dim=2)[0]
        y_min = ref_points[..., 1].min(dim=2)[0]
        y_max = ref_points[..., 1].max(dim=2)[0]
        w = x_max - x_min
        h = y_max - y_min
        area = (w * h).unsqueeze(-1)  # (B, N, 1)
        return area

    # ==================================================================
    # 高斯权重计算
    # ==================================================================

    def compute_gaussian_weights(self, ref_points, effective_sigma):
        """
        计算 16 个控制点之间的高斯距离权重 (行归一化，去对角线).

        Args:
            ref_points:      (B, N, P, 2) 归一化坐标 [0,1]
            effective_sigma: (B, N, 1, 1) 最终高斯带宽
        Returns:
            gauss_weights:   (B, N, P, P) 行归一化高斯权重矩阵
        """
        B, N, K, _ = ref_points.shape

        # 两两欧式距离平方
        pts = ref_points                                   # (B, N, P, 2)
        diff = pts.unsqueeze(3) - pts.unsqueeze(2)         # (B, N, P, P, 2)
        dist_sq = torch.sum(diff ** 2, dim=-1)              # (B, N, P, P)

        # 高斯核: exp(-d² / 2σ²)
        gauss = torch.exp(-dist_sq / (2.0 * effective_sigma ** 2 + 1e-8))

        # 去掉对角线 (不自引用)
        eye_mask = torch.eye(K, device=ref_points.device).unsqueeze(0).unsqueeze(0)
        gauss = gauss * (1.0 - eye_mask)

        # 行归一化 (每个点的邻居权重和为 1)
        gauss = gauss / (gauss.sum(dim=-1, keepdim=True) + 1e-8)

        return gauss

    # ==================================================================
    # 前向传播
    # ==================================================================

    def forward(self, h_intra, ref_points):
        """
        统一入口: 高斯残差 + 尺度预测.

        Args:
            h_intra:    (B, N, P, C) 特征 (intra-SA 之前的输入)
            ref_points: (B, N, P, 2)
        Returns:
            gauss_feat: (B, N, P, C) 零初始化高斯特征残差
            film_scale: (B, N, 1)    用于 FiLM 调制的尺度因子
        """
        # ── 防御性：以 ref_points 的 N 为规范源 ──
        # ref_points 是几何坐标 → 其 N 决定了 compute_gaussian_weights 的 dist_sq 形状
        # h_intra 的特征 N 必须与之匹配，否则所有 sigma/scale 预测都会错位
        B, N, K, C = h_intra.shape
        N_rp = ref_points.shape[1]
        assert N == N_rp, (
            f"SATR N-mismatch: h_intra N={N} vs ref_points N={N_rp}. "
            f"h_intra{tuple(h_intra.shape)}, ref_points{tuple(ref_points.shape)}. "
            f"This usually means SATR received inconsistent inputs. "
            f"Check the decoder layer that calls SATR."
        )

        h_inst = h_intra.mean(dim=2)    # (B, N, C) 实例语义特征

        # ── Sigma / scale 预测 ──
        if self.decoupled:
            # V3: 解耦预测
            geo_area = self.compute_geometry_area(ref_points)        # (B, N, 1)
            gauss_input = torch.cat([h_inst, geo_area], dim=-1)      # (B, N, C+1)
            gauss_sigma_raw = self.gauss_sigma_pred(gauss_input)     # (B, N, 1) ∈ [0,1]
            # 保守范围: base * [0.8, 1.5] = [0.08, 0.15] (base=0.1)
            # 注意: gauss_sigma_raw 已经是 3D (B,N,1)，只需 unsqueeze 一次 → 4D (B,N,1,1)
            # 之前误用两次 unsqueeze → 5D → 与 dist_sq (4D) 广播错位
            effective_sigma = self.base_sigma.abs() * (0.8 + 0.7 * gauss_sigma_raw)
            effective_sigma = effective_sigma.unsqueeze(-1)  # (B,N,1,1) — 4D
            assert effective_sigma.dim() == 4, (
                f"SATR V3 effective_sigma dim wrong: got {effective_sigma.dim()}D "
                f"shape={tuple(effective_sigma.shape)}, expected 4D (B,N,1,1)"
            )
            # FiLM scale: 纯语义，自适应
            film_scale = self.film_scale_pred(h_inst)                # (B, N, 1)
        else:
            # V1/V2: 共享 σ 预测器
            sigma = self.scale_pred(h_inst)                          # (B, N, 1)
            dyn_sigma = sigma.unsqueeze(-1)                          # (B, N, 1, 1)
            if self.sigma_relax:
                # V2: sigma*2 + 0.05 floor → [0.15, 0.35]
                effective_sigma = self.base_sigma.abs() * (1.0 + dyn_sigma * 2.0) + 0.05
            else:
                # V1: 保守 → [0.1, 0.2]
                effective_sigma = self.base_sigma.abs() * (1.0 + dyn_sigma) + 1e-6
            film_scale = sigma  # 共享 sigma 也用于 FiLM

        # ── 高斯加权邻居特征聚合 ──
        # 最终安全网：验证维度数 + N 值
        # dist_sq 是 4D (B,N,P,P)，effective_sigma 必须也是 4D (B,N,1,1) 才能正确广播
        assert effective_sigma.dim() == 4, (
            f"SATR effective_sigma must be 4D (B,N,1,1), got {effective_sigma.dim()}D "
            f"shape={tuple(effective_sigma.shape)}"
        )
        assert ref_points.shape[1] == effective_sigma.shape[1], (
            f"SATR N mismatch: ref_points N={ref_points.shape[1]} vs "
            f"effective_sigma N={effective_sigma.shape[1]}. "
            f"ref_points{tuple(ref_points.shape)}, "
            f"effective_sigma{tuple(effective_sigma.shape)}"
        )
        gauss_w = self.compute_gaussian_weights(ref_points, effective_sigma)  # (B,N,P,P)

        # 加权聚合: (B,N,P,P) × (B,N,P,C) → (B,N,P,C)
        gauss_w_exp = gauss_w.unsqueeze(-1)          # (B, N, P, P, 1)
        h_expanded = h_intra.unsqueeze(2)             # (B, N, 1, P, C)
        neighbor_feat = (gauss_w_exp * h_expanded).sum(dim=3)  # (B, N, P, C)

        # 投影 + 零初始化 → 残差
        gauss_feat = self.gauss_proj(neighbor_feat)  # (B, N, P, C)

        # ── CURA: 曲率感知残差调制 ──
        if self.use_cura:
            # 1. 控制点局部曲率 (环形二阶差分)
            pts_prev = torch.cat(
                [ref_points[:, :, -1:, :], ref_points[:, :, :-1, :]], dim=2
            )  # (B,N,K,2)
            pts_next = torch.cat(
                [ref_points[:, :, 1:, :], ref_points[:, :, :1, :]], dim=2
            )  # (B,N,K,2)
            delta2 = pts_prev + pts_next - 2 * ref_points  # (B,N,K,2)
            curvature = torch.norm(delta2, dim=-1)          # (B,N,K)

            # 2. Perona-Malik: 高曲率 → 抑制扩散, 低曲率 → 增强扩散
            #    kappa_gate=0 时 modulation=1 → 退化为 V1
            modulation = torch.exp(-self.kappa_gate * curvature)  # (B,N,K)
            modulation = modulation.unsqueeze(-1)                  # (B,N,K,1)

            # 3. 曲率门控残差: h_intra + modulation * delta
            gauss_feat = modulation * gauss_feat  # (B,N,K,C)

        return gauss_feat, film_scale

    # ==================================================================
    # FiLM 调制 (施加在 circonv 输出上)
    # ==================================================================

    def forward_film_circonv(self, h_circonv, film_scale):
        """
        FiLM 调制 circonv 输出.

        Args:
            h_circonv:  (B, N, P, C)
            film_scale: (B, N, 1) 尺度因子 (V1/V2=共享σ, V3=解耦film_scale)
        Returns:
            h_mod: (B, N, P, C)
        """
        gamma = self.film_gamma(film_scale).unsqueeze(2)    # (B, N, 1, C)
        beta = self.film_beta(film_scale).unsqueeze(2)      # (B, N, 1, C)

        # (1 + gamma) * h + beta  →  gamma=0, beta=0 时输出 = h
        return h_circonv * (1.0 + gamma) + beta
