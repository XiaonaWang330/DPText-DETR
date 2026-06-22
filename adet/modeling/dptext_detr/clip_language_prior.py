"""
CLIP Language Prior for DPText-DETR (V9).

Components:
  1. Learnable soft prompts (fixed per-task, no adaptive offset)
  2. Frozen CLIP text encoder: soft prompts -> c_lang_clip (B, 512)
  3. Projection: 512 -> 256 = c_lang (B, 256)
  4. Global average pooling -> v_spatial (B, n_pts, 256)
  5. All new parameters zero-init for cold start.

V9: First CLIP integration. c_lang provides a global text-concept signal,
     v_spatial provides per-control-point spatial context.
     Decoder uses sigmoid gate: gamma = sigmoid(lang_to_gamma(c_lang)),
     tgt = tgt + v_scale * gamma * v_spatial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel


class CLIPLanguagePrior(nn.Module):
    """
    V9 CLIP-based language prior (simpler version without SPP/prompt_offset).

    Forward:
        src_s32  (B, 256, Hs, Ws)   stride-32 feature map
        src_s64  (B, 256, Hs, Ws)   stride-64 feature map
        ->
        c_lang    (B, 256)           language prior for decoder sigmoid gate
        v_spatial (B, n_pts, 256)   per-point spatial visual features
    """

    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch16",
        clip_model_path=None,
        d_model=256,
        num_soft_prompts=4,
        num_ctrl_points=16,
    ):
        super().__init__()
        self.num_soft_prompts = num_soft_prompts
        self.num_ctrl_points = num_ctrl_points

        # ---- Frozen CLIP text encoder ----
        load_source = clip_model_path if clip_model_path else clip_model_name
        self.clip_text_model = CLIPTextModel.from_pretrained(load_source)
        clip_dim = self.clip_text_model.config.hidden_size  # 512
        for p in self.clip_text_model.parameters():
            p.requires_grad = False

        # ---- Learnable soft prompts (N x clip_dim) ----
        self.soft_prompts = nn.Parameter(torch.randn(num_soft_prompts, clip_dim) * 0.02)

        # ---- CLIP (512) -> DETR (256) projection (zero-init) ----
        self.proj_to_detr = nn.Linear(clip_dim, d_model)
        nn.init.zeros_(self.proj_to_detr.weight)
        nn.init.zeros_(self.proj_to_detr.bias)

        # ---- GAP -> per-point v_spatial (zero-init) ----
        self.v_spatial_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.v_spatial_proj.weight)
        nn.init.zeros_(self.v_spatial_proj.bias)

        # CLIP attention mask (all tokens attend to all)
        self.register_buffer(
            "_clip_mask",
            torch.ones(1, num_soft_prompts, dtype=torch.long),
            persistent=False,
        )

    # ------------------------------------------------------------------
    #  CLIP encoding (frozen, no grad)
    # ------------------------------------------------------------------
    def _clip_encode(self, prompt_embeds):
        """Run frozen CLIP text model, return pooled output (B, clip_dim)."""
        from transformers.modeling_attn_mask_utils import (
            _create_4d_causal_attention_mask,
            _prepare_4d_attention_mask,
        )
        with torch.no_grad():
            txt = self.clip_text_model.text_model
            B, L, D = prompt_embeds.shape

            # 1. Token + position embeddings
            hidden = txt.embeddings(inputs_embeds=prompt_embeds)

            # 2. Build masks
            causal_attn_mask = _create_4d_causal_attention_mask(
                (B, L), hidden.dtype, device=hidden.device
            )
            attn_mask_2d = self._clip_mask.expand(B, -1)
            attn_mask = _prepare_4d_attention_mask(attn_mask_2d, hidden.dtype)

            # 3. Encoder
            enc_out = txt.encoder(
                hidden,
                attention_mask=attn_mask,
                causal_attention_mask=causal_attn_mask,
            )

            # 4. Final layer norm
            hidden = txt.final_layer_norm(enc_out.last_hidden_state)

            # 5. Pooled output: last valid token (EOT position)
            if torch.all(attn_mask_2d == 1):
                eos_pos = L - 1
            else:
                eos_pos = attn_mask_2d.to(dtype=torch.int).argmax(dim=-1)
            pooled = hidden[torch.arange(B, device=hidden.device), eos_pos]
        return pooled   # (B, clip_dim)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, src_s32, src_s64):
        B = src_s32.shape[0]
        clip_dim = self.soft_prompts.shape[1]

        # 1. Fixed soft prompts (B, N, clip_dim)
        adaptive_prompts = self.soft_prompts.unsqueeze(0).expand(B, -1, -1)

        # 2. CLIP encoding (frozen)
        c_lang_clip = self._clip_encode(adaptive_prompts)     # (B, clip_dim)

        # 3. Project to DETR space (zero-init → c_lang starts at 0)
        c_lang = self.proj_to_detr(c_lang_clip)               # (B, 256)

        # 4. GAP on stride-32 → per-point v_spatial (zero-init → v_spatial starts at 0)
        v_feat = F.adaptive_avg_pool2d(src_s32, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 256)
        v = self.v_spatial_proj(v_feat)                       # (B, 256)
        v_spatial = v.unsqueeze(1).expand(-1, self.num_ctrl_points, -1)  # (B, n_pts, 256)

        return c_lang, v_spatial

    def clip_param_names(self):
        """Return names of CLIP frozen params (to exclude from checkpoint)."""
        return {n for n, _ in self.named_parameters() if n.startswith('clip_text_model.')}
