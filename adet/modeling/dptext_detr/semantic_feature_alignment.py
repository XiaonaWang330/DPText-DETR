"""
Semantic Feature Alignment (SFA) for DPText-DETR.

Core idea:
  Text detection classification = visual decision + semantic alignment.
  We project control-point features into a CLIP-defined semantic space and
  align text-point features toward the "text" concept anchor.

Technical innovation:
  1. CLIP text encoder (frozen, transformers backend) → semantic anchor c_text in R^256
  2. Learnable projection: visual feature → CLIP-aligned space
  3. Semantic logit bias: L2-normalized dot(feat_proj_norm, c_text_norm) = cos_sim ∈ [-1,1] added to cls_logit (V19 fix: was unnormalized dot → P crash)
  4. Contrastive alignment loss: text CPs → c_text, non-text CPs → pushed away
  5. Full participation from iter 0 → model co-adapts classification + semantics

Key property: SFA operates in the classification decision space (logit level),
NOT in the query feature space — hence fully orthogonal to SGIFA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel

# Hard-coded text prompt: the semantic anchor for "text" concept
SFA_TEXT_PROMPT = "a photo of text"


class SemanticFeatureAlignment(nn.Module):
    """
    SFA: aligns control-point features with the CLIP "text" semantic anchor
    to provide a semantic prior for the classification decision.

    Forward returns:
        semantic_logit: (B, N, 16, 1) — zero-init-scaled dot-product with c_text
        sfa_info: dict with 'feat_proj' (B,N,16,256) and 'c_text' (256,)
                  for alignment-loss computation in SetCriterion.
    """

    def __init__(
        self,
        d_model=256,
        clip_model_name="openai/clip-vit-base-patch16",
        clip_model_path=None,
        agg_mode="point",
    ):
        super().__init__()
        self.agg_mode = agg_mode  # V21: "point" or "max"
        # Load frozen CLIP text encoder (transformers backend, consistent with V9/V21)
        load_source = clip_model_path if clip_model_path else clip_model_name
        self.clip_text_model = CLIPTextModel.from_pretrained(load_source)
        clip_dim = self.clip_text_model.config.hidden_size  # typically 512
        for p in self.clip_text_model.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # Pre-compute frozen text anchor at init time
        # ------------------------------------------------------------------
        self._c_text_raw = self._encode_text_prompt(SFA_TEXT_PROMPT, clip_dim)
        # self._c_text_raw shape: (clip_dim,) — frozen buffer

        # ------------------------------------------------------------------
        # Learnable: project CLIP text (512) → model dimension (256)
        # ------------------------------------------------------------------
        self.text_proj = nn.Sequential(
            nn.Linear(clip_dim, d_model),
            nn.LayerNorm(d_model),
        )
        nn.init.xavier_uniform_(self.text_proj[0].weight)
        nn.init.zeros_(self.text_proj[0].bias)

        # ------------------------------------------------------------------
        # Learnable: project visual feature → CLIP-aligned semantic space
        # ------------------------------------------------------------------
        # Normal init (no zero-init) so gradients flow through both layers.
        # Only logit_scale is zero-init to ensure the SFA branch contributes
        # zero to classification at t=0 (warm start identical to baseline).
        self.semantic_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        for m in self.semantic_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # ------------------------------------------------------------------
        # V19 / V21: free logit_scale (no sigmoid), init=1.0.
        #   Scale × L2-normalized cos_sim → bounded by |scale|.
        #   Safe because cos_sim ∈ [-1,1] has a hard ceiling.
        # ------------------------------------------------------------------
        self.logit_scale_raw = nn.Parameter(torch.tensor(1.0))

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
        # Use pooler_output (CLIP-style pooled representation)
        pooled = outputs.pooler_output[0]  # (clip_dim,)
        return pooled  # stays on CPU until used

    def get_text_anchor(self, device=None):
        """Return projected text anchor of shape (d_model,)."""
        c = self.text_proj(self._c_text_raw.to(device))
        return c

    def forward(self, feat, unc=None):
        """
        Args:
            feat: (B, N, 16, d_model) — control-point features from decoder layer
            unc:  (B, N, 1) or None — SGIFA per-query uncertainty (V18 UASG)

        Returns:
            semantic_logit: (B, N, 16, 1) — semantic classification bias
            cos_sim: (B, N, 16) — similarity score for inference fusion
            sfa_info: dict for alignment-loss computation
        """
        B, N, P, D = feat.shape
        c_text = self.get_text_anchor(device=feat.device)  # (D,)

        # Project features to semantic space
        feat_proj = self.semantic_proj(feat)  # (B, N, 16, D)

        # L2-normalize both → cosine similarity ∈ [-1, 1]
        feat_proj_norm = F.normalize(feat_proj, p=2, dim=-1)  # (B, N, 16, D)
        c_text_norm = F.normalize(c_text, p=2, dim=0)          # (D,)
        semantic_logit = torch.einsum(
            "bnpd,d->bnp", feat_proj_norm, c_text_norm
        ).unsqueeze(-1)  # (B, N, 16, 1) ∈ [-1, 1]

        # V21/V22: Per-query aggregation
        #   "max"   — most aggressive: 1 lucky point lifts whole query (R↑ but P↓)
        #   "mean"  — conservative: random noise cancels out for bg queries (P↑ R~)
        if self.agg_mode == "max":
            per_query = semantic_logit.max(dim=2, keepdim=True).values  # (B, N, 1, 1)
            semantic_logit = per_query.expand(-1, -1, P, -1)             # (B, N, 16, 1)
        elif self.agg_mode == "mean":
            per_query = semantic_logit.mean(dim=2, keepdim=True)        # (B, N, 1, 1)
            semantic_logit = per_query.expand(-1, -1, P, -1)             # (B, N, 16, 1)

        cos_sim = semantic_logit.squeeze(-1)  # (B, N, 16) ∈ [-1, 1]

        sfa_info = {
            "feat_proj": feat_proj,  # (B, N, 16, D) — for alignment loss
            "c_text": c_text,        # (D,)
            "cos_sim": cos_sim,      # (B, N, 16) — for V23 SGFL (semantic hard negative mining)
        }

        # V19: Free scale × cos_sim (no sigmoid)
        base_scale = self.logit_scale_raw
        if unc is not None:
            # UASG: per-query uncertainty modulation
            per_query_scale = base_scale * (0.5 + unc)  # (B, N, 1)
            scale = per_query_scale.unsqueeze(2)         # (B, N, 1, 1)
        else:
            scale = base_scale  # scalar, broadcasts

        return scale * semantic_logit, cos_sim, sfa_info

    def compute_alignment_loss(
        self,
        feat_proj,      # (B, N, 16, D)
        c_text,         # (D,)
        pos_idx,        # (batch_idx: [M], query_idx: [M])  — matched positives
        neg_mask,       # (B, N) bool — True for unmatched (background) queries
        num_inst,       # float — total GT instances across batch (normalization)
        bg_margin=0.1,  # hinge margin: penalize bg cos_sim > margin
        bg_weight=0.1,  # relative weight of bg contrastive loss
    ):
        """
        Compute the semantic feature alignment loss (V22: positive pull + background push).

        Positive: pull matched text-point features toward c_text (cos_sim → 1).
        Background: push unmatched query features away from c_text (cos_sim → < margin).

        This is the MISSING HALF of contrastive learning. Without bg push, background
        queries have random cos_sim ≈ 0 → some positive → false positive bias → P drops.
        With bg push, background gets consistently negative cos_sim → negative bias →
        FP suppressed → P recovers and can exceed baseline/V12.

        Args:
            feat_proj: projected features (B, N, 16, D)
            c_text: text anchor (D,)
            pos_idx: tuple of (batch_indices, query_indices) from matcher
            neg_mask: (B, N) boolean, True for background (unmatched) queries
            num_inst: total number of GT instances (normalization)
            bg_margin: hinge threshold — bg cos_sim > margin gets penalized
            bg_weight: weight of bg contrastive term relative to positive term

        Returns:
            loss scalar tensor
        """
        # ------------------------------------------------------------------
        # Positive alignment: pull text points → c_text (cos_sim → 1)
        # ------------------------------------------------------------------
        pos_feat = feat_proj[pos_idx[0], pos_idx[1]]  # (M, 16, D)
        M_total = pos_feat.shape[0] * pos_feat.shape[1]  # M × 16

        if M_total == 0:
            return torch.tensor(0.0, device=feat_proj.device)

        pos_feat_flat = pos_feat.reshape(-1, pos_feat.shape[-1])  # (M×16, D)
        c_text_exp_pos = c_text.unsqueeze(0).expand(M_total, -1)   # (M×16, D)

        # Cosine similarity: want it close to 1.0 for text points
        cos_sim_pos = F.cosine_similarity(pos_feat_flat, c_text_exp_pos, dim=-1)  # (M×16,)
        loss_pos = (1.0 - cos_sim_pos).sum() / num_inst

        # ------------------------------------------------------------------
        # Background contrastive: push bg points → away from c_text (cos_sim < margin)
        # Hinge loss: relu(cos_sim - margin) penalizes bg points that are
        #   too close to "text" semantic. Self-balancing: initially most bg
        #   cos_sim < margin → zero loss → only hard negatives penalized.
        # ------------------------------------------------------------------
        if bg_weight > 0 and neg_mask is not None:
            # Select background queries
            neg_feat = feat_proj[neg_mask]  # (K, 16, D) — K background queries
            if neg_feat.shape[0] > 0:
                K_total = neg_feat.shape[0] * neg_feat.shape[1]  # K × 16
                neg_feat_flat = neg_feat.reshape(-1, neg_feat.shape[-1])  # (K×16, D)
                c_text_exp_neg = c_text.unsqueeze(0).expand(K_total, -1)  # (K×16, D)

                cos_sim_neg = F.cosine_similarity(neg_feat_flat, c_text_exp_neg, dim=-1)  # (K×16,)
                # Hinge loss: only penalize bg points with cos_sim > margin
                # This pushes them below margin → trained bg gets negative bias → P↑
                loss_neg = F.relu(cos_sim_neg - bg_margin).sum() / num_inst
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
        # Mean-pool over 16 control points → per-query semantic score
        sem_score = cos_sim.mean(dim=-1)  # (B, N)
        # Map [-1, 1] → [0, 1] for multiplicative fusion
        sem_score = (sem_score + 1.0) / 2.0
        return sem_score
