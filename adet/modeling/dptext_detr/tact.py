"""
TACT: Topology-Aware Curvature Transform
=========================================
纯视觉几何模块，解决小文字特征塌缩和尺度自适应问题。

三个子机制:
1. Gaussian Topology Aggregation: 基于控制点欧式距离的高斯加权邻居特征聚合
   - 小文字点距小 → 高斯核衰减快 → 防止特征同质化
   - 大文字点距大 → 高斯核保留基础权重 → 保持邻域通信

2. Curvature-Aware Residual Gating (Perona-Malik 各向异性扩散):
   - high curvature (corners) → modulation ≈ 0 → preserve Precision
   - low  curvature (smooth)  → modulation ≈ 1 → boost Recall
   - kappa_gate 零初始化 → 训练第一步 = 无曲率门控

3. FiLM Circonv Modulation: 尺度因子 通过仿射变换调制 circonv 输出
   - ()(), ()() 零初始化 → 初期输出 = 原版 circonv

所有新增参数零初始化 → 训练第一步与 baseline 完全一致。

(原名 SATR/CURA, 统一更名为 TACT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TACT(nn.Module):
    """
    Topology-Aware Curvature Transform.

    Injected at each decoder layer after intra-SA + circonv.
    Does NOT modify loss, matcher, or exceed 0.15M parameters.
    """

    def __init__(self, d_model=256, num_ctrl_points=16,
                 sigma_relax=False, decoupled=False,
                 use_cura=True):
        super().__init__()
        self.d_model = d_model
        self.num_ctrl_points = num_ctrl_points
        self.sigma_relax = sigma_relax          # V2: dyn_sigma*2.0 + 0.05 floor
        self.decoupled = decoupled              # V3: decoupled sigma predictors + geo prior
        self.use_cura = use_cura                # Curvature-aware gating (default ON)

        # ── Sigma predictor ──
        if decoupled:
            self.gauss_sigma_pred = nn.Sequential(
                nn.Linear(d_model + 1, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )
            self.film_scale_pred = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )
        else:
            # V1/V2: shared sigma predictor
            self.scale_pred = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )

        # ── FiLM modulation (on circonv output) ──
        self.film_gamma = nn.Linear(1, d_model)
        self.film_beta = nn.Linear(1, d_model)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # ── Gaussian topology aggregation ──
        self.gauss_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gauss_proj.weight)
        nn.init.zeros_(self.gauss_proj.bias)

        # Base bandwidth (learnable, conservatively initialised)
        self.base_sigma = nn.Parameter(torch.tensor(0.1))

        # ── Curvature-aware residual gating ──
        if use_cura:
            self.kappa_gate = nn.Parameter(torch.zeros(1))

    # ==================================================================
    # Geometry utilities
    # ==================================================================

    def compute_geometry_area(self, ref_points):
        """Bounding-box area from normalised control points."""
        x_min = ref_points[..., 0].min(dim=2)[0]
        x_max = ref_points[..., 0].max(dim=2)[0]
        y_min = ref_points[..., 1].min(dim=2)[0]
        y_max = ref_points[..., 1].max(dim=2)[0]
        area = ((x_max - x_min) * (y_max - y_min)).unsqueeze(-1)
        return area

    # ==================================================================
    # Gaussian weights
    # ==================================================================

    def compute_gaussian_weights(self, ref_points, effective_sigma):
        """
        Row-normalised Gaussian distance weights (diagonal removed).

        Args:
            ref_points:      (B, N, K, 2)
            effective_sigma: (B, N, 1, 1)
        Returns:
            gauss_weights:   (B, N, K, K)
        """
        B, N, K, _ = ref_points.shape

        pts = ref_points
        diff = pts.unsqueeze(3) - pts.unsqueeze(2)         # (B,N,K,K,2)
        dist_sq = torch.sum(diff ** 2, dim=-1)               # (B,N,K,K)

        gauss = torch.exp(-dist_sq / (2.0 * effective_sigma ** 2 + 1e-8))

        eye_mask = torch.eye(K, device=ref_points.device).unsqueeze(0).unsqueeze(0)
        gauss = gauss * (1.0 - eye_mask)

        gauss = gauss / (gauss.sum(dim=-1, keepdim=True) + 1e-8)
        return gauss

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(self, h_intra, ref_points):
        """
        Args:
            h_intra:    (B, N, K, C)  features before intra-SA
            ref_points: (B, N, K, 2)  normalised control points
        Returns:
            gauss_feat: (B, N, K, C)  zero-init Gaussian residual
            film_scale: (B, N, 1)     scale factor for FiLM
        """
        B, N, K, C = h_intra.shape
        N_rp = ref_points.shape[1]
        assert N == N_rp, (
            f"TACT N-mismatch: h_intra N={N} vs ref_points N={N_rp}. "
            f"h_intra{tuple(h_intra.shape)}, ref_points{tuple(ref_points.shape)}."
        )

        h_inst = h_intra.mean(dim=2)    # (B, N, C)

        # ── Sigma / scale prediction ──
        if self.decoupled:
            geo_area = self.compute_geometry_area(ref_points)
            gauss_input = torch.cat([h_inst, geo_area], dim=-1)
            gauss_sigma_raw = self.gauss_sigma_pred(gauss_input)
            effective_sigma = self.base_sigma.abs() * (0.8 + 0.7 * gauss_sigma_raw)
            effective_sigma = effective_sigma.unsqueeze(-1)  # (B,N,1,1)
            assert effective_sigma.dim() == 4, (
                f"TACT effective_sigma: expected 4D, got {effective_sigma.dim()}D"
            )
            film_scale = self.film_scale_pred(h_inst)
        else:
            sigma = self.scale_pred(h_inst)
            dyn_sigma = sigma.unsqueeze(-1)
            if self.sigma_relax:
                effective_sigma = self.base_sigma.abs() * (1.0 + dyn_sigma * 2.0) + 0.05
            else:
                effective_sigma = self.base_sigma.abs() * (1.0 + dyn_sigma) + 1e-6
            film_scale = sigma

        assert effective_sigma.dim() == 4, (
            f"TACT effective_sigma must be 4D (B,N,1,1), got {effective_sigma.dim()}D"
        )
        assert ref_points.shape[1] == effective_sigma.shape[1], (
            f"TACT N mismatch: ref_points N={ref_points.shape[1]} vs "
            f"sigma N={effective_sigma.shape[1]}"
        )
        gauss_w = self.compute_gaussian_weights(ref_points, effective_sigma)

        # Weighted aggregation
        gauss_w_exp = gauss_w.unsqueeze(-1)
        h_expanded = h_intra.unsqueeze(2)
        neighbor_feat = (gauss_w_exp * h_expanded).sum(dim=3)

        gauss_feat = self.gauss_proj(neighbor_feat)

        # ── Curvature-aware gating ──
        if self.use_cura:
            pts_prev = torch.cat(
                [ref_points[:, :, -1:, :], ref_points[:, :, :-1, :]], dim=2
            )
            pts_next = torch.cat(
                [ref_points[:, :, 1:, :], ref_points[:, :, :1, :]], dim=2
            )
            delta2 = pts_prev + pts_next - 2 * ref_points
            curvature = torch.norm(delta2, dim=-1)
            modulation = torch.exp(-self.kappa_gate * curvature)
            modulation = modulation.unsqueeze(-1)
            gauss_feat = modulation * gauss_feat

        return gauss_feat, film_scale

    # ==================================================================
    # FiLM modulation (applied on circonv output)
    # ==================================================================

    def forward_film_circonv(self, h_circonv, film_scale):
        """
        Args:
            h_circonv:  (B, N, K, C)
            film_scale: (B, N, 1)
        Returns:
            h_mod: (B, N, K, C)
        """
        gamma = self.film_gamma(film_scale).unsqueeze(2)
        beta  = self.film_beta(film_scale).unsqueeze(2)
        return h_circonv * (1.0 + gamma) + beta
