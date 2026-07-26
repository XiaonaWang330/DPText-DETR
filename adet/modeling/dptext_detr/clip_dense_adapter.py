"""
CLIP Dense Adapter (DRTP-v2) + Scale-Aware Gate (SAG / v2.2)
==============================================================
Frozen CLIP ViT extracts patch tokens from the input image.
Patch tokens are projected to d_model, resized to each detector
feature level, and fused via a learnable gated residual:

    F'_l = F_l + alpha_eff * G_l * V_l

Scale-Aware Gate (SAG, v2.2):
    A per-sample global scale factor s ∈ (0,1) modulates the effective
    injection alpha:

        alpha_eff = alpha_base * scale

    scale is predicted from the backbone's global spatial statistics
    (avg-pool over the active level's features → MLP → sigmoid, init
    bias=2.0 → near 1 at the start of training). Intuition: when the
    backbone features are dominated by small/hard text (degraded),
    reduce CLIP injection to avoid the noise from CLIP's 14×14 patch
    tokens over dense small text.

Key design:
    - alpha_init > 0 (no tanh) — gate net gets gradient from step 1.
    - Standard GateNet (mask-aware 2D conv) produces gate G_l ∈ [0,1].
    - DirectionalGateNet is an ablation switch (CLIP_DIRECTIONAL_GATE).
    - Token-shuffle / token-mix / replace-noise are ablation switches.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

# CLIP ViT input normalization constants (RGB, range [0, 1])
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Standard ImageNet normalization (RGB, range [0, 1]) — for reference
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DirectionalGateNet(nn.Module):
    """
    Direction-aware gating network for DRTP.

    Replaces the final 1×1 conv in the original GateNet with factorized
    strip convolutions (1×7 horizontal + 7×1 vertical) to explicitly
    capture text direction during gating.

    Architecture:
        shared:  Conv1×1(512→256)→ReLU→Conv1×1(256→64)→ReLU
        h_branch: Conv2d(64→1, kernel=(1,7), pad=(0,3))  → horizontal context
        v_branch: Conv2d(64→1, kernel=(7,1), pad=(3,0))  → vertical context
        output = sigmoid(w_h * h + w_v * v)
        where [w_h, w_v] = softmax(direction_weight)
    """

    def __init__(self, in_ch: int = 512, d_model: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model // 4, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.h_branch = nn.Conv2d(d_model // 4, 1, kernel_size=(1, 7), padding=(0, 3))
        self.v_branch = nn.Conv2d(d_model // 4, 1, kernel_size=(7, 1), padding=(3, 0))
        self.direction_weight = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

        # kaiming init for h/v branches, bias=0
        for branch in [self.h_branch, self.v_branch]:
            nn.init.kaiming_normal_(branch.weight, mode='fan_out', nonlinearity='linear')
            nn.init.zeros_(branch.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 512, H, W) concatenated [norm_src, norm_clip]
        Returns:
            gate_logits: (B, 1, H, W)
            h_logits:    (B, 1, H, W) horizontal branch output
            v_logits:    (B, 1, H, W) vertical branch output
            dir_w:       (2,) softmax-normalized direction weights
        """
        feat = self.shared(x)                         # (B, 64, H, W)
        h_logits = self.h_branch(feat)                # (B, 1, H, W)
        v_logits = self.v_branch(feat)                # (B, 1, H, W)
        w = F.softmax(self.direction_weight, dim=0)   # [w_h, w_v]
        gate_logits = w[0] * h_logits + w[1] * v_logits
        return gate_logits, h_logits, v_logits, w.detach()


class ScaleAwareModulator(nn.Module):
    """Per-level scale-aware alpha modulator.

    Learns to reduce CLIP injection strength when backbone features indicate
    the image is dominated by small text (where CLIP 14×14 patches are noisy).

    Architecture:
        AdaptiveAvgPool2d(1) → Flatten → Linear(256→64) → ReLU → Linear(64→1) → Sigmoid
        Output ∈ [0, 1] multiplies base_alpha.
        Init bias=2.0 → sigmoid(2)≈0.88 → near-identity at step 0.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.modulator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        # Init to ~0.88 (near identity, avoids training instability)
        nn.init.zeros_(self.modulator[1].weight)
        nn.init.constant_(self.modulator[1].bias, 2.0)
        nn.init.zeros_(self.modulator[3].weight)
        nn.init.constant_(self.modulator[3].bias, 2.0)

    def forward(self, src: torch.Tensor):
        """src: (B, d_model, H, W) backbone feature → (B, 1, 1, 1) scale factor"""
        return self.modulator(self.pool(src)).view(-1, 1, 1, 1)  # (B, 1, 1, 1)


class CLIPDenseAdapter(nn.Module):
    """
    CLIP Dense Visual Adapter (Phase 1).

    Fuses frozen CLIP patch features into the detector's multi-scale
    feature pyramid via learnable per-level gated residuals.

    Zero-init design:
        alpha_l = 0 at init → F'_l == F_l exactly.
        Model can learn to gradually incorporate CLIP features if useful.
    """

    def __init__(
        self,
        clip_model_name: str = "pretrain/clip-vit-base-patch16",
        d_model: int = 256,
        num_feature_levels: int = 4,
        freeze_clip: bool = True,
        dropout: float = 0.0,
        gate_init_bias: float = 0.0,
        alpha_init: float = 0.5,
        keep_aspect_ratio: bool = True,
        shuffle: bool = False,
        replace_noise: bool = False,
        token_mix: bool = False,
        mix_lambda: float = 0.1,
        # ── Phase 1 ablation switches ──
        active_levels: list = None,          # which FPN levels get CLIP (None=all)
        use_gate: bool = True,               # False → skip GateNet, direct α*V injection
        learnable_alpha: bool = True,        # False → α frozen at alpha_init
        fixed_alpha_value: float = 0.5,      # used when learnable_alpha=False
        shared_projector: bool = False,      # True → one projector shared across levels
        directional_gate: bool = False,      # True → DirectionalGateNet (1×7 + 7×1 strip conv)
        scale_aware_gate: bool = False,     # True → per-level ScaleAwareModulator on alpha
    ):
        super().__init__()
        self.d_model = d_model
        self.num_feature_levels = num_feature_levels
        self.freeze_clip = freeze_clip
        self.keep_aspect_ratio = keep_aspect_ratio
        self.shuffle = shuffle
        self.replace_noise = replace_noise
        self.token_mix = token_mix
        self.mix_lambda = mix_lambda
        self.use_gate = use_gate
        self.directional_gate = directional_gate
        self.scale_aware_gate = scale_aware_gate
        self.learnable_alpha = learnable_alpha
        self.fixed_alpha_value = fixed_alpha_value
        self.shared_projector = shared_projector
        # Normalize active_levels: None → all, else list of ints
        if active_levels is None:
            self.active_levels = list(range(num_feature_levels))
        else:
            self.active_levels = sorted(set(active_levels))

        # ── Load frozen CLIP vision model ──
        clip_full = CLIPModel.from_pretrained(clip_model_name)
        self.clip_vision = clip_full.vision_model           # CLIPVisionModel
        self.visual_projection = clip_full.visual_projection # nn.Linear(768→512)
        del clip_full  # submodules are now owned by self

        self.clip_config = self.clip_vision.config
        self.clip_dim = self.clip_config.hidden_size          # 768 for ViT-B
        self.clip_patch_size = self.clip_config.patch_size     # 16 for ViT-B
        self.clip_image_size = self.clip_config.image_size     # 224 for ViT-B
        self.clip_num_layers = self.clip_config.num_hidden_layers  # 12 for ViT-B

        if freeze_clip:
            for p in self.clip_vision.parameters():
                p.requires_grad = False
            for p in self.visual_projection.parameters():
                p.requires_grad = False
            self.clip_vision.eval()

        # ── Projection (always create 4 per-level projectors for init determinism) ──
        self.level_projectors = nn.ModuleList()
        for _ in range(num_feature_levels):
            self.level_projectors.append(
                nn.Sequential(
                    nn.Conv2d(self.clip_dim, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model) if d_model >= 32 else nn.Identity(),
                )
            )
        # Optional shared projector (extra module, created after per-level for determinism)
        if shared_projector:
            self._shared_projector = nn.Sequential(
                nn.Conv2d(self.clip_dim, d_model, kernel_size=1),
                nn.GroupNorm(32, d_model) if d_model >= 32 else nn.Identity(),
            )
        else:
            self._shared_projector = None

        # ── Gate Network (always created for init determinism; forward skips if !use_gate) ──
        self.level_gate_nets = nn.ModuleList()
        if directional_gate:
            for _ in range(num_feature_levels):
                self.level_gate_nets.append(
                    DirectionalGateNet(in_ch=2 * d_model, d_model=d_model)
                )
        else:
            for _ in range(num_feature_levels):
                self.level_gate_nets.append(
                    nn.Sequential(
                        nn.Conv2d(2 * d_model, d_model, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(d_model, d_model // 4, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(d_model // 4, 1, kernel_size=1),
                    )
                )
            for gate_net in self.level_gate_nets:
                nn.init.constant_(gate_net[-1].bias, gate_init_bias)

        # ── Alpha (learnable or frozen) ──
        self.level_alphas = nn.ParameterList()
        alpha_init_val = alpha_init if learnable_alpha else fixed_alpha_value
        for _ in range(num_feature_levels):
            self.level_alphas.append(
                nn.Parameter(torch.tensor(alpha_init_val, dtype=torch.float32))
            )

        # ── Scale-Aware Modulators (per-level, used when scale_aware_gate=True) ──
        self.scale_modulators = nn.ModuleList()
        if scale_aware_gate:
            for _ in range(num_feature_levels):
                self.scale_modulators.append(ScaleAwareModulator(d_model=d_model))

        # ── Learnable noise (replaces CLIP when replace_noise=True) ──
        if replace_noise:
            self.noise_token = nn.Parameter(
                torch.randn(1, self.clip_dim, 14, 14) * 0.02
            )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # ── One-time diagnostic: print ablation config ──
        print(
            f"[DRTP Config] active_levels={self.active_levels} "
            f"| use_gate={self.use_gate} "
            f"| learnable_alpha={self.learnable_alpha} "
            f"| fixed_alpha_value={self.fixed_alpha_value} "
            f"| shared_projector={self.shared_projector} "
            f"| shuffle={self.shuffle} "
            f"| token_mix={self.token_mix} "
            f"| replace_noise={self.replace_noise}"
            f"| directional_gate={self.directional_gate}"
            f"| scale_aware_gate={self.scale_aware_gate}"
        )
        self._level_shapes_logged = False

    @torch.no_grad()
    def _extract_patch_tokens(
        self, clip_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract CLIP patch tokens (excluding CLS).

        Args:
            clip_tensor: (B, 3, H, W) CLIP-normalized images

        Returns:
            patch_tokens: (B, N, clip_dim) where N = H//P * W//P
        """
        was_training = self.clip_vision.training
        self.clip_vision.eval()

        outputs = self.clip_vision(clip_tensor, output_hidden_states=True)
        # Last hidden state: (B, 1 + N_patch, clip_dim)
        last_hidden = outputs.last_hidden_state
        # Drop CLS token (index 0)
        patch_tokens = last_hidden[:, 1:, :]  # (B, N_patch, clip_dim)

        if was_training:
            self.clip_vision.train()

        return patch_tokens

    @torch.no_grad()

    def _reshape_to_2d(
        self, patch_tokens: torch.Tensor, grid_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Reshape (B, N_patch, C) → (B, C, H_grid, W_grid).
        """
        B, N, C = patch_tokens.shape
        H_g, W_g = grid_shape
        # Ensure N matches grid
        assert N == H_g * W_g, f"Patch count {N} != grid {H_g}x{W_g}={H_g * W_g}"
        return patch_tokens.permute(0, 2, 1).reshape(B, C, H_g, W_g).contiguous()

    def forward(
        self,
        clip_images: List[torch.Tensor],
        srcs: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Fuse CLIP features into detector feature levels via gated residual.

        Args:
            clip_images: list of (3, H_i, W_i) raw image tensors (geometrically transformed,
                         NOT detector-normalized)
            srcs: list of (B, d_model, H_l, W_l) detector feature maps
            masks: list of (B, H_l, W_l) binary masks

        Returns:
            Modified srcs with CLIP features fused in (same shapes as input)
        """

        # ── One-time: print level shapes (index → stride verification) ──
        if not self._level_shapes_logged:
            print("[DRTP Level Shapes] index → stride mapping:")
            for i, src in enumerate(srcs):
                active_mark = " ← ACTIVE" if i in self.active_levels else ""
                print(f"  level[{i}] = {tuple(src.shape)}{active_mark}")
            self._level_shapes_logged = True

        # ── 1. Prepare CLIP input ──
        B = srcs[0].shape[0]
        if self.replace_noise:
            # Skip CLIP entirely: use learnable noise as "fake CLIP features"
            patch_tokens = self.noise_token.expand(B, -1, -1, -1).reshape(B, self.clip_dim, -1).permute(0, 2, 1)
            # patch_tokens: (B, 196, clip_dim)
            grid_shape = (14, 14)
            clip_mask = torch.zeros(
                B, 1, 14, 14,
                device=patch_tokens.device, dtype=torch.bool,
            )
        else:
            clip_tensor, clip_mask, grid_shape = self._process_image_for_clip(clip_images)

            # ── 2. Extract frozen CLIP features ──
            with torch.no_grad():
                patch_tokens = self._extract_patch_tokens(clip_tensor)

        # ── 2b. (Ablation) Shuffle patch tokens to destroy spatial-semantic alignment ──
        if self.shuffle:
            B, N, C = patch_tokens.shape
            for b in range(B):
                perm = torch.randperm(N, device=patch_tokens.device)
                patch_tokens[b] = patch_tokens[b, perm]

        # ── 2c. (Regularization) Partial token mixing — training only ──
        # Mix a small fraction of shuffled tokens to reduce overfitting to
        # exact CLIP patch correspondence. Inference uses 100% aligned tokens.
        if self.training and self.token_mix:
            B, N, C = patch_tokens.shape
            idx = torch.stack([
                torch.randperm(N, device=patch_tokens.device)
                for _ in range(B)
            ])
            shuffled = patch_tokens[torch.arange(B, device=patch_tokens.device).unsqueeze(1), idx]
            mixed = (
                (1.0 - self.mix_lambda) * patch_tokens
                + self.mix_lambda * shuffled
            )
            # ── One-time verification ──
            if not hasattr(self, '_token_mix_logged'):
                diff = (mixed - patch_tokens).abs().mean().item()
                cos_sim = torch.nn.functional.cosine_similarity(
                    patch_tokens.reshape(-1, C), mixed.reshape(-1, C), dim=-1
                ).mean().item()
                print(f"[CLIP DENSE] token_mix (train-only) ACTIVE "
                      f"| lambda={self.mix_lambda} "
                      f"| ||mixed-orig||_1={diff:.4f} "
                      f"| cos(orig,mixed)={cos_sim:.4f}")
                self._token_mix_logged = True
            patch_tokens = mixed

        # ── 3. Reshape to 2D feature map ──
        clip_feat_2d = self._reshape_to_2d(patch_tokens, grid_shape)  # (B, C_clip, H_g, W_g)

        # ── 4. Resize mask to match clip_feat_2d ──
        # clip_mask is (B, 1, H_g, W_g) True=padded
        clip_mask_valid = (~clip_mask).float()  # (B, 1, H_g, W_g), 1=valid, 0=padded

        # ── 5. For each feature level: project, resize, gate, fuse ──
        fused_srcs = []
        self._last_gate_mean = []
        self._last_gate_std = []
        self._last_gate_active_ratio = []
        self._last_delta_ratio = []
        self._last_alpha_gate_mean = []

        # ── Directional Gate stats ──
        self._last_dir_w_h = []        # w_h per level (from softmax)
        self._last_dir_w_v = []        # w_v per level
        self._last_dir_h_gate_mean = [] # h_branch gate mean per level
        self._last_dir_v_gate_mean = [] # v_branch gate mean per level

        # ── Scale-Aware stats ──
        self._last_scale_factor = []    # scale_factor per active level
        self._last_adaptive_alpha = []  # base_alpha * scale_factor per active level

        for lvl, src in enumerate(srcs):
            B, C, H_l, W_l = src.shape

            # ---- Check if this level is active ----
            if lvl not in self.active_levels:
                fused_srcs.append(src)
                self._last_gate_mean.append(0.0)
                self._last_gate_std.append(0.0)
                self._last_gate_active_ratio.append(0.0)
                self._last_delta_ratio.append(0.0)
                self._last_alpha_gate_mean.append(0.0)
                if self.scale_aware_gate:
                    self._last_scale_factor.append(0.0)
                    self._last_adaptive_alpha.append(0.0)
                continue

            # ---- 5a. Project CLIP features to d_model ----
            if self.shared_projector:
                V = self._shared_projector(clip_feat_2d)
            else:
                V = self.level_projectors[lvl](clip_feat_2d)

            # ---- 5b. Resize mask and CLIP features to this level's resolution ----
            if V.shape[-2:] != (H_l, W_l):
                V = F.interpolate(V, size=(H_l, W_l), mode="bilinear", align_corners=False)
                mask_valid = F.interpolate(clip_mask_valid, size=(H_l, W_l), mode="nearest")
            else:
                mask_valid = clip_mask_valid

            # ---- 5c. Gate (optional) ----
            if self.use_gate:
                # gate input: concat(detector_feat_normalized, clip_feat_normalized)
                src_nh = src.float().permute(0, 2, 3, 1)
                V_nh = V.float().permute(0, 2, 3, 1)
                src_norm = F.layer_norm(src_nh, [C]).permute(0, 3, 1, 2).type_as(src)
                V_norm = F.layer_norm(V_nh, [C]).permute(0, 3, 1, 2).type_as(V)
                gate_input = torch.cat([src_norm, V_norm], dim=1)

                if self.directional_gate:
                    gate_logits, h_logits, v_logits, dir_w = self.level_gate_nets[lvl](gate_input)
                    self._last_dir_w_h.append(dir_w[0].item())
                    self._last_dir_w_v.append(dir_w[1].item())
                    with torch.no_grad():
                        self._last_dir_h_gate_mean.append(torch.sigmoid(h_logits).mean().item())
                        self._last_dir_v_gate_mean.append(torch.sigmoid(v_logits).mean().item())
                else:
                    gate_logits = self.level_gate_nets[lvl](gate_input)

                G = torch.sigmoid(gate_logits) * mask_valid

                # ── Collect gate stats (common to both modes) ──
                G = self.dropout(G)
            else:
                # No gate: use mask_valid directly (still suppresses padding)
                G = mask_valid

            # ---- 5d. Alpha (scale-aware if enabled) ----
            if self.learnable_alpha:
                base_alpha = self.level_alphas[lvl]
            else:
                base_alpha = self.level_alphas[lvl].detach()  # frozen at init value

            if self.scale_aware_gate and lvl < len(self.scale_modulators):
                scale_factor = self.scale_modulators[lvl](src)  # (B, 1, 1, 1)
                alpha = base_alpha * scale_factor
                with torch.no_grad():
                    self._last_scale_factor.append(scale_factor.mean().item())
                    self._last_adaptive_alpha.append(alpha.mean().item())
            else:
                alpha = base_alpha

            residual = alpha * G * V

            # ---- 5e. Fuse ----
            F_new = src + residual
            fused_srcs.append(F_new)

            # ---- 5f. Collect stats ----
            with torch.no_grad():
                self._last_gate_mean.append(G.mean().item())
                self._last_gate_std.append(G.std().item())
                self._last_gate_active_ratio.append((G > 0.5).float().mean().item())
                self._last_alpha_gate_mean.append((alpha * G).mean().item())
                src_norm_val = src.norm(dim=1, keepdim=True).mean()
                res_norm_val = residual.norm(dim=1, keepdim=True).mean()
                ratio = (res_norm_val / (src_norm_val + 1e-8)).item()
                self._last_delta_ratio.append(ratio)

        return fused_srcs

    def get_stats(self) -> Dict[str, float]:
        """
        Return monitoring statistics for training logs.
        Call after each forward.
        """
        stats = {}
        for lvl in range(self.num_feature_levels):
            alpha_val = self.level_alphas[lvl].detach().item()
            stats[f"clip/alpha_level_{lvl}"] = alpha_val
            if lvl < len(self._last_gate_mean):
                stats[f"clip/gate_mean_level_{lvl}"] = self._last_gate_mean[lvl]
                stats[f"clip/gate_std_level_{lvl}"] = self._last_gate_std[lvl]
                stats[f"clip/gate_active_ratio_level_{lvl}"] = self._last_gate_active_ratio[lvl]
                stats[f"clip/alpha_gate_mean_level_{lvl}"] = self._last_alpha_gate_mean[lvl]
                stats[f"clip/delta_ratio_level_{lvl}"] = self._last_delta_ratio[lvl]
            # ── Directional Gate stats ──
            if self.directional_gate and lvl < len(self._last_dir_w_h):
                stats[f"clip/dir_w_h_level_{lvl}"] = self._last_dir_w_h[lvl]
                stats[f"clip/dir_w_v_level_{lvl}"] = self._last_dir_w_v[lvl]
                stats[f"clip/dir_h_gate_mean_level_{lvl}"] = self._last_dir_h_gate_mean[lvl]
                stats[f"clip/dir_v_gate_mean_level_{lvl}"] = self._last_dir_v_gate_mean[lvl]
            # ── Scale-Aware stats ──
            if self.scale_aware_gate and lvl < len(self._last_scale_factor):
                stats[f"clip/scale_factor_level_{lvl}"] = self._last_scale_factor[lvl]
                stats[f"clip/adaptive_alpha_level_{lvl}"] = self._last_adaptive_alpha[lvl]
        stats["clip/num_levels"] = float(self.num_feature_levels)
        stats["clip/keep_aspect"] = float(self.keep_aspect_ratio)
        return stats
