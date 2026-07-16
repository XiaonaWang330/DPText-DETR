"""
Semantic Feature Alignment (SFA) for DPText-DETR.

Core idea:
  Text detection classification = visual decision + semantic alignment.
  We project control-point features into a CLIP-defined semantic space and
  align text-point features toward the "text" concept anchor.

Technical innovation:
  1. CLIP text encoder (frozen, transformers backend) -> semantic anchor
  2. Learnable projection: visual feature -> CLIP-aligned space
  3. Semantic logit bias: L2-normalized cos_sim added to cls_logit
  4. Contrastive alignment loss: text CPs -> anchor, non-text -> pushed away
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel

SFA_TEXT_PROMPT = "a photo of text"


class SemanticFeatureAlignment(nn.Module):
    """
    SFA: aligns control-point features with CLIP "text" semantic anchor.

    Forward returns:
        semantic_logit: (B, N, 16, 1) — semantic classification bias
        cos_sim: (B, N, 16) — similarity score for inference fusion
        sfa_info: dict with 'feat_proj' and 'c_text' for alignment loss
    """

    def __init__(
        self,
        d_model=256,
        clip_model_name="openai/clip-vit-base-patch16",
        clip_model_path=None,
        agg_mode="point",
        use_film=False,    # V55: FiLM — produce gamma/beta for GCR modulation
    ):
        super().__init__()
        self.agg_mode = agg_mode

        # ── Load frozen CLIP text encoder ──
        load_source = clip_model_path if clip_model_path else clip_model_name
        self.clip_text_model = CLIPTextModel.from_pretrained(load_source)
        clip_dim = self.clip_text_model.config.hidden_size  # typically 512
        for p in self.clip_text_model.parameters():
            p.requires_grad = False

        # ── Pre-compute frozen text anchor at init ──
        self._c_text_raw = self._encode_text_prompt(SFA_TEXT_PROMPT, clip_dim)

        # ── Learnable: project CLIP text (512) -> model dim (256) ──
        self.text_proj = nn.Sequential(
            nn.Linear(clip_dim, d_model),
            nn.LayerNorm(d_model),
        )
        nn.init.xavier_uniform_(self.text_proj[0].weight)
        nn.init.zeros_(self.text_proj[0].bias)

        # ── Learnable: project visual feature -> CLIP-aligned semantic space ──
        self.semantic_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        for m in self.semantic_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # ── Free logit_scale, init=1.0 (cos_sim bounded by [-1,1]) ──
        self.logit_scale_raw = nn.Parameter(torch.tensor(1.0))

        # ── V55 FiLM: per-point scalar gamma/beta for GCR modulation ──
        # gamma ∈ [0.7, 1.3], beta ∈ [-0.3, 0.3] via tanh constraint
        # Per-point granularity: 16 points × 1 scalar = 16 modulators
        self.use_film = use_film
        if use_film:
            self.gamma_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 4, 1),
            )
            self.beta_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 4, 1),
            )
            # gamma: bias=1.0 → gamma_init = 1 + 0.3*tanh(1) ≈ 1.23 … no, actually
            # with zero-init weight: gamma_proj output ≈ bias = 1.0
            # gamma = 1 + 0.3*tanh(1.0) ≈ 1.23, not 1.0
            # Fix: bias=0 → gamma = 1 + 0.3*tanh(0) = 1.0
            nn.init.constant_(self.gamma_proj[-1].weight, 0)
            nn.init.constant_(self.gamma_proj[-1].bias, 0)   # → gamma = 1.0
            nn.init.constant_(self.beta_proj[-1].weight, 0)
            nn.init.constant_(self.beta_proj[-1].bias, 0)    # → beta = 0.0

    def _encode_text_prompt(self, prompt, clip_dim):
        """Encode a single text prompt with frozen CLIP text encoder → pooled vector."""
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(
            self.clip_text_model.config._name_or_path
        )
        tokens = tokenizer(
            [prompt], return_tensors="pt", truncation=True, max_length=77
        )
        with torch.no_grad():
            outputs = self.clip_text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
        pooled = outputs.pooler_output[0]  # (clip_dim,)
        return pooled

    def get_text_anchor(self, device=None):
        """Return projected text anchor: (d_model,)."""
        c = self.text_proj(self._c_text_raw.to(device))
        return c

    def forward(self, feat, unc=None, return_film=False):
        """
        Args:
            feat: (B, N, 16, d_model) — control-point features from decoder layer
            unc:  (B, N, 1) or None — SGIFA per-query uncertainty (V18 UASG)
            return_film: if True and use_film=True, also return (gamma, beta)

        Returns:
            If return_film=False:
                semantic_logit: (B, N, 16, 1)
                cos_sim: (B, N, 16)
                sfa_info: dict
            If return_film=True:
                semantic_logit, cos_sim, sfa_info, gamma, beta
                gamma: (B, N, 16, 1) ∈ [0.7, 1.3]
                beta:  (B, N, 16, 1) ∈ [-0.3, 0.3]
        """
        B, N, P, D = feat.shape

        c_text = self.get_text_anchor(device=feat.device)  # (D,)

        feat_proj = self.semantic_proj(feat)  # (B, N, 16, D)

        # ── Compute cosine similarity ──
        feat_proj_norm = F.normalize(feat_proj, p=2, dim=-1)  # (B, N, 16, D)
        c_text_norm = F.normalize(c_text, p=2, dim=0)  # (D,)
        semantic_logit = torch.einsum(
            "bnpd,d->bnp", feat_proj_norm, c_text_norm
        ).unsqueeze(-1)  # (B, N, 16, 1) ∈ [-1, 1]

        # ── Per-query aggregation ──
        if self.agg_mode == "max":
            per_query = semantic_logit.max(dim=2, keepdim=True).values
            semantic_logit = per_query.expand(-1, -1, P, -1)
        elif self.agg_mode == "mean":
            per_query = semantic_logit.mean(dim=2, keepdim=True)
            semantic_logit = per_query.expand(-1, -1, P, -1)

        cos_sim = semantic_logit.squeeze(-1)  # (B, N, 16)

        sfa_info = {
            "feat_proj": feat_proj,
            "c_text": c_text,
            "cos_sim": cos_sim,
        }

        # ── Scale ──
        if unc is not None:
            scale = self.logit_scale_raw * (0.5 + unc).unsqueeze(2)
        else:
            scale = self.logit_scale_raw

        if return_film and self.use_film:
            # V55: FiLM modulation (per-point scalar, constrained range)
            gamma = self.gamma_proj(feat_proj)          # (B, N, 16, 1)
            beta  = self.beta_proj(feat_proj)            # (B, N, 16, 1)
            gamma = 1.0 + 0.3 * torch.tanh(gamma)       # ∈ [0.7, 1.3]
            beta  = 0.3 * torch.tanh(beta)               # ∈ [-0.3, 0.3]
            return scale * semantic_logit, cos_sim, sfa_info, gamma, beta

        return scale * semantic_logit, cos_sim, sfa_info

    def compute_alignment_loss(
        self,
        feat_proj,      # (B, N, 16, D)
        c_text,         # (D,)
        pos_idx,        # (batch_idx: [M], query_idx: [M])  — matched positives
        neg_mask,       # (B, N) bool — True for unmatched (background) queries
        num_inst,       # float — total GT instances across batch (unused)
        bg_margin=0.1,  # hinge margin: penalize bg cos_sim > margin
        bg_weight=0.1,  # relative weight of bg contrastive loss
    ):
        """
        Compute the semantic feature alignment loss.

        Per-point mean normalization:
        - loss_pos = mean(1 - cos_per_point)  ∈ [0, 2], batch-invariant
        - loss_neg = mean(relu(cos_per_point - m))  ∈ [0, 1-m], batch-invariant
        - Per-point gradient preserves fine-grained alignment for all 16 CPs.
        - Both terms normalized by their own point counts → no batch sensitivity.

        Args:
            feat_proj: projected features (B, N, 16, D)
            c_text: text anchor (D,)
            pos_idx: tuple of (batch_indices, query_indices) from matcher
            neg_mask: (B, N) boolean, True for background (unmatched) queries
            num_inst: total number of GT instances (unused)
            bg_margin: hinge threshold — bg cos_sim > margin gets penalized
            bg_weight: weight of bg contrastive term relative to positive term

        Returns:
            loss scalar tensor
        """
        c_text_norm = F.normalize(c_text, p=2, dim=0)  # (D,)

        # ── Positive: per-POINT cosine similarity ──
        pos_feat = feat_proj[pos_idx[0], pos_idx[1]]  # (M, 16, D)
        if pos_feat.shape[0] == 0:
            return torch.tensor(0.0, device=feat_proj.device)

        pos_feat_norm = F.normalize(pos_feat, p=2, dim=-1)  # (M, 16, D)
        cos_sim_pos = torch.einsum("mpd,d->mp", pos_feat_norm, c_text_norm)  # (M, 16)
        loss_pos = (1.0 - cos_sim_pos).mean()

        # ── Background contrastive: per-POINT, push away from c_text ──
        if bg_weight > 0 and neg_mask is not None:
            neg_feat = feat_proj[neg_mask]  # (K, 16, D) — K background queries
            if neg_feat.shape[0] > 0:
                neg_feat_norm = F.normalize(neg_feat, p=2, dim=-1)  # (K, 16, D)
                cos_sim_neg = torch.einsum("kpd,d->kp", neg_feat_norm, c_text_norm)  # (K, 16)
                loss_neg = F.relu(cos_sim_neg - bg_margin).mean()
            else:
                loss_neg = torch.tensor(0.0, device=feat_proj.device)
        else:
            loss_neg = torch.tensor(0.0, device=feat_proj.device)

        return loss_pos + bg_weight * loss_neg

    def compute_semantic_score(self, cos_sim):
        """
        Convert per-point cosine similarity to per-query semantic confidence.

        Args:
            cos_sim: (B, N, 16) — cosine similarity ∈ [-1, 1] per control point

        Returns:
            sem_score: (B, N) — semantic confidence ∈ [0, 1] per query
        """
        sem_score = cos_sim.mean(dim=-1)  # (B, N)
        sem_score = (sem_score + 1.0) / 2.0
        return sem_score
