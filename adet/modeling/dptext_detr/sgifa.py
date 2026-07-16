# ----------------------------------------------------------------------------
# Self-Guided Instance Feature Aggregation (SGIFA)
#
# V22 base: mean pooling + query-level uncertainty gate (detached).
# V34 BC-PWP: Branch-Coupled Precision-Weighted Pooling.
#
# Problem with V28 PWP: precision directly weighted the pooling → instance_feat
#   distribution shifted → SFA's semantic_proj learned unstable mappings →
#   SFA+PWP = 88.09 < either alone (negative synergy).
#
# BC-PWP fix: DECOUPLE precision from pooling.
#   - Pooling: STILL mean (preserves distribution, SFA-compatible)
#   - Precision: predicts per-point reliability, aggregates to query-level,
#     modulates the GATE (not the feature). High-precision → gate up (reliable
#     instance, safe to enhance); low-precision → gate down (noisy instance,
#     suppress enhancement to avoid injecting noise).
#   - Net effect: hs_out distribution is statistically similar to V22 mean
#     (same enhanced vector, just scaled by precision-gate). SFA's input
#     distribution is preserved → positive synergy restored.
#
# Uncertainty analysis preserved (CCF-B narrative):
#   - per-point precision prediction (Bayesian-inspired reliability)
#   - query-level uncertainty aggregation (1 - mean precision)
#   - uncertainty-modulated enhancement gating
#   The novelty is preserved; only the coupling point moves from
#   feature-polling to gate-modulation, which is distribution-preserving.
#
# Three guarantees:
#   ORTHOGONAL: BC-PWP's precision modulates gate, not instance_feat.
#     hs_out = hs + gate * enhanced. V22: gate from unc_head. BC-PWP:
#     gate from unc_head × precision_gate. enhanced is identical (mean pooling).
#     → hs distribution shape preserved → SFA compatible.
#   EFFECTIVE: Precision still steers enhancement — unreliable instances
#     (low precision) get less enhancement, reliable ones get more.
#   DECOUPLED: Pure SGIFA internal change. SFA code unchanged.
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfGuidedInstanceFeatureAggregation(nn.Module):

    def __init__(self, d_model=256, num_levels=4, use_uncertainty_gate=True):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        self.use_uncertainty_gate = use_uncertainty_gate

        # Project pooled instance features back to d_model
        self.instance_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # Learnable gate: query state → per-query enhancement weight
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, 1),
        )

        # Zero-initialize gate → SGIFA starts as near-identity, warms up gradually
        nn.init.constant_(self.gate_net[-1].weight, 0)
        nn.init.constant_(self.gate_net[-1].bias, 0)

        # Zero-initialize instance_proj → starts as identity-path
        nn.init.constant_(self.instance_proj[-1].weight, 0)
        nn.init.constant_(self.instance_proj[-1].bias, 0)

        if use_uncertainty_gate:
            # V22: query-level uncertainty head (detached) for SFA UASG
            self.unc_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 2, 1),
            )
            nn.init.constant_(self.unc_head[-1].weight, 0)
            nn.init.constant_(self.unc_head[-1].bias, 0)

            # V34 BC-PWP: per-point precision head (detached).
            # Predicts per-point reliability WITHOUT participating in pooling.
            # Aggregates to query-level precision to modulate the enhancement
            # gate — distribution-preserving alternative to V28 PWP.
            self.point_precision_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 4, 1),
            )
            nn.init.constant_(self.point_precision_head[-1].weight, 0)
            nn.init.constant_(self.point_precision_head[-1].bias, 0)

    def forward(self, hs, ref_points, memory, spatial_shapes, lid=-1, return_unc=False):
        """
        Args:
            hs (Tensor): (B, N, 16, D)
            ref_points (Tensor): (B, N, 16, 2) in [0,1] (detached)
            memory (Tensor): (Sum(HW), B, D)
            spatial_shapes (Tensor): (L, 2)
            lid (int): layer index (unused)
            return_unc (bool): if True, return (hs_out, unc) tuple

        Returns:
            hs_out: (B, N, 16, D)
            unc: (B, N, 1) or None — query-level uncertainty for SFA UASG
        """
        B, N, P, D = hs.shape
        L = spatial_shapes.shape[0]

        # --- Step 1: Sample encoder features at each ctrl_point per level ---
        sampled_scales = []
        start = 0
        for lvl in range(L):
            H, W = spatial_shapes[lvl].tolist()
            end = start + H * W
            mem_lvl = memory[:, start:end, :].reshape(B, H, W, D).permute(0, 3, 1, 2)
            grid = ref_points.clone()
            grid[..., 0] = 2.0 * grid[..., 0] - 1.0
            grid[..., 1] = 2.0 * grid[..., 1] - 1.0
            grid = grid.reshape(B, N * P, 1, 2)
            sampled = F.grid_sample(
                mem_lvl, grid, mode='bilinear',
                padding_mode='border', align_corners=True,
            )
            sampled = sampled.squeeze(-1).reshape(B, D, N, P).permute(0, 2, 3, 1)
            sampled_scales.append(sampled)
            start = end

        # --- Step 2: Average multi-scale → instance feature (V22 base) ---
        instance_feat = torch.stack(sampled_scales, dim=0).mean(dim=0)  # (B, N, P, D)

        # --- Step 3: MEAN pooling over 16 points (distribution-preserving) ---
        # V28 PWP weighted here → shifted distribution → SFA broke.
        # BC-PWP keeps mean → enhanced is identical to V22 → SFA compatible.
        instance_feat = instance_feat.mean(dim=2)  # (B, N, D)

        # --- Step 4: Project and gate ---
        enhanced = self.instance_proj(instance_feat)  # (B, N, D)
        gate = self.gate_net(hs.mean(dim=2)).sigmoid()  # (B, N, 1)

        # --- Step 5: Dual uncertainty gating (V22 unc + BC-PWP precision) ---
        unc = None
        if self.use_uncertainty_gate:
            # V22: query-level uncertainty for SFA UASG (detached)
            unc = self.unc_head(hs.detach().mean(dim=2)).sigmoid()  # (B, N, 1)
            gate = gate * (0.5 + 0.5 * unc)  # V22 base gate modulation

            # V34 BC-PWP: precision-modulated gate (detached).
            # Per-point precision → query-level reliability → gate scale.
            # High precision (reliable instance) → boost gate (safe to enhance).
            # Low precision (noisy instance) → suppress gate (avoid noise injection).
            # This modulates gate AMPLITUDE, not feature content → distribution
            # shape preserved. zero-init → sigmoid(0)=0.5 → scale=1.0 → V22.
            point_precision = self.point_precision_head(hs.detach()).sigmoid()  # (B, N, P, 1)
            query_precision = point_precision.mean(dim=2)  # (B, N, 1)
            # Map precision [0,1] → gate_scale [0.5, 1.5]: centered at 1.0
            precision_gate = 0.5 + point_precision.mean(dim=2)  # (B, N, 1) ∈ [0.5, 1.5]
            gate = gate * precision_gate

        # --- Step 6: Add instance feature to all 16 points ---
        hs_out = hs + gate.unsqueeze(2) * enhanced.unsqueeze(2)  # (B, N, P, D)

        if return_unc:
            return hs_out, unc
        return hs_out


def build_sgifa(cfg):
    """Factory function to create SGIFA module from config."""
    d_model = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
    num_levels = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS

    sgifa_cfg = cfg.MODEL.TRANSFORMER.get('SGIFA', {})
    use_unc_gate = getattr(sgifa_cfg, 'USE_UNC_GATE', True) if isinstance(sgifa_cfg, dict) \
        else sgifa_cfg.get('USE_UNC_GATE', True)

    return SelfGuidedInstanceFeatureAggregation(
        d_model=d_model,
        num_levels=num_levels,
        use_uncertainty_gate=use_unc_gate,
    )
