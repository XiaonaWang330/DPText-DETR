# ----------------------------------------------------------------------------
# Self-Guided Instance Feature Aggregation (SGIFA)
#
# A novel query enhancement mechanism for DPText-DETR that uses the model's
# own predicted control points to sample pixel-level features from encoder
# feature maps, then aggregates them into instance-level features to enhance
# decoder queries in subsequent layers.
#
# Key differences from SRFormer's MQE (AAAI 2024):
#   1. Uses ctrl_points as sampling coordinates (no extra mask branch)
#   2. Self-referential: model guides its own feature extraction
#   3. No additional supervision or loss needed
#   4. Uncertainty-aware gating selectively enhances uncertain queries
#
# For CCF-B submission:
#   "Uncertainty-Aware Scene Text Detection with
#    Self-Guided Instance Feature Aggregation"
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfGuidedInstanceFeatureAggregation(nn.Module):
    """
    SGIFA: ctrl_points → sample encoder features → instance-level pooling → enhance query.

    At each decoder layer (lid >= 1), the previously predicted ctrl_points (16 points
    defining the text instance polygon) are used to sample features from the encoder
    multi-scale feature maps (grid_sample at each point). The sampled 16-point features
    are averaged into an instance-level feature, projected, gated, and added back to
    all 16 positions of the query for the next decoder layer.

    Optionally, a lightweight uncertainty head predicts per-query uncertainty, which
    modulates the gate — uncertain queries (potential false negatives) receive stronger
    enhancement, while confident queries receive less.

    Args:
        d_model (int): Feature dimension (256)
        num_levels (int): Number of encoder feature levels (4)
        use_uncertainty_gate (bool): Enable uncertainty-modulated gating
    """

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

        # Optional: lightweight uncertainty head for adaptive gating
        if use_uncertainty_gate:
            self.unc_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 2, 1),
            )
            # Zero-init → uncertainty-gate starts at 0.5 weight
            nn.init.constant_(self.unc_head[-1].weight, 0)
            nn.init.constant_(self.unc_head[-1].bias, 0)

    def forward(self, hs, ref_points, memory, spatial_shapes, lid=-1):
        """
        Args:
            hs (Tensor): (B, N, 16, D) — decoder hidden state at current layer
            ref_points (Tensor): (B, N, 16, 2) — reference ctrl points in [0,1] (detached)
            memory (Tensor): (Sum(HW), B, D) — flattened multi-scale encoder features
            spatial_shapes (Tensor): (L, 2) — (H, W) per feature level
            lid (int): layer index (unused, kept for API compatibility)

        Returns:
            hs_enhanced (Tensor): (B, N, 16, D) — enhanced hidden state
        """
        B, N, P, D = hs.shape
        L = spatial_shapes.shape[0]

        # --- Step 1: Sample encoder features at each ctrl_point ---
        # memory: (B, sum_HW, D) — slice along spatial dim (dim 1)

        sampled_scales = []
        start = 0
        for lvl in range(L):
            H, W = spatial_shapes[lvl].tolist()
            end = start + H * W

            # Extract per-level feature: slice (B, H*W, D) → reshape to (B, H, W, D) → (B, D, H, W)
            mem_lvl = memory[:, start:end, :].reshape(B, H, W, D).permute(0, 3, 1, 2)

            # grid_sample coords: [0,1] → [-1,1]
            grid = ref_points.clone()
            grid[..., 0] = 2.0 * grid[..., 0] - 1.0  # x
            grid[..., 1] = 2.0 * grid[..., 1] - 1.0  # y
            grid = grid.reshape(B, N * P, 1, 2)

            # grid_sample: (B, D, H, W) with grid (B, N*P, 1, 2) → (B, D, N*P, 1)
            sampled = F.grid_sample(
                mem_lvl, grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True,
            )
            # Reshape: (B, D, N*P, 1) → (B, D, N, P) → (B, N, P, D)
            sampled = sampled.squeeze(-1).reshape(B, D, N, P).permute(0, 2, 3, 1)
            sampled_scales.append(sampled)

            start = end

        # --- Step 2: Average multi-scale features → instance feature ---
        # sampled_scales: list of L × (B, N, P, D)
        instance_feat = torch.stack(sampled_scales, dim=0).mean(dim=0)  # (B, N, P, D)

        # Pool across 16 ctrl_points → instance-level representation
        instance_feat = instance_feat.mean(dim=2)  # (B, N, D)

        # --- Step 3: Project and gate ---
        enhanced = self.instance_proj(instance_feat)  # (B, N, D)
        gate = self.gate_net(hs.mean(dim=2)).sigmoid()  # (B, N, 1)

        # --- Step 4: Optional uncertainty-modulated gating ---
        if self.use_uncertainty_gate:
            unc = self.unc_head(hs.detach().mean(dim=2)).sigmoid()  # (B, N, 1)
            gate = gate * (0.5 + 0.5 * unc)  # scale to [0.25, 0.75] × base gate

        # --- Step 5: Add instance feature to all 16 points of each query ---
        # gate:   (B, N, 1) → unsqueeze to (B, N, 1, 1)
        # enhanced: (B, N, D) → unsqueeze to (B, N, 1, D)
        hs_out = hs + gate.unsqueeze(2) * enhanced.unsqueeze(2)  # (B, N, P, D)

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
