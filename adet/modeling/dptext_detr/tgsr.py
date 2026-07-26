"""
TGSR: Topology-Conditioned Semantic Routing with Matcher Decoupling
===================================================================

Query-level CLIP semantic injection that replaces FPN-level DRTP.

Three sub-mechanisms:
  1. Dynamic-point CLIP sampling — reads CLIP at decoder control-point locations
  2. Topology-conditioned global-local mixing — Gaussian topology matrix aggregates
     local semantics; geometry-adaptive reliability blends with global CLIP
  3. Bounded semantic correction — beta*tanh(s_q) added to visual classification logits

Design invariants:
  - CLIP does NOT enter FPN → encoder/decoder features are pure visual
  - TACT topology conditions CLIP reading (W_geo.detach()) → causal direction
  - beta = 0 init → training step 1 == visual-only baseline
  - pred_logits_match (visual) feeds Hungarian; pred_logits (fused) feeds loss

Usage (in models.py forward):
  clip_feat_map = self.tgsr.extract_clip_features(clip_images)   # (B,512,14,14)
  ...
  for lvl in [4, 5]:
      correction = self.tgsr(clip_feat_map, ref_points[lvl], ref_points[lvl-1])
      outputs[lvl] = visual_logits + correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer


# CLIP preprocessing constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ============================================================================
# CLIP Feature Extractor — minimal, frozen CLIP ViT wrapper (HuggingFace)
# ============================================================================

class CLIPFeatureExtractor(nn.Module):
    """
    Minimal frozen CLIP ViT wrapper (HuggingFace transformers backend).
    Only extracts patch feature map (14x14 for ViT-B/16) and encodes text prompts.
    Does NOT inject into FPN — the caller uses patch features directly.
    """

    def __init__(
        self,
        pretrained_path="pretrain/clip-vit-base-patch16",
        freeze=True,
    ):
        super().__init__()
        # Load full CLIP model (vision + text) from HuggingFace checkpoint
        self.clip_model = CLIPModel.from_pretrained(pretrained_path)
        self.vision_model = self.clip_model.vision_model
        self.visual_projection = self.clip_model.visual_projection
        self.text_model = self.clip_model.text_model
        self.text_projection = self.clip_model.text_projection

        # Tokenizer for text encoding
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_path)

        self.clip_dim = self.clip_model.config.projection_dim  # 512 for ViT-B/16
        self.clip_resolution = self.clip_model.config.vision_config.image_size  # 224

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    @staticmethod
    def prepare_images(clip_images, device=None):
        """
        Prepare raw image list → CLIP-ready batched tensor.

        Args:
            clip_images: List[Tensor] of (3, H_i, W_i), raw uint8 or float [0,255]
                         from geometric transforms (NOT detector-normalised).
        Returns:
            clip_tensor: (B, 3, 224, 224) float32, CLIP-normalised
        """
        B = len(clip_images)
        if device is None:
            device = clip_images[0].device
        clip_tensor = torch.zeros(B, 3, 224, 224, device=device, dtype=torch.float32)

        mean = torch.tensor(CLIP_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(CLIP_STD,  device=device, dtype=torch.float32).view(1, 3, 1, 1)

        for i, img in enumerate(clip_images):
            # Normalise dynamic range
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            elif img.dtype != torch.float32:
                img = img.float()
                if img.max() > 2.0:
                    img = img / 255.0

            # Direct square resize (MVP — keep_aspect_ratio optional later)
            if img.shape[-2:] != (224, 224):
                img = F.interpolate(
                    img.unsqueeze(0), size=(224, 224),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)

            # CLIP normalisation
            clip_tensor[i] = (img.unsqueeze(0) - mean) / std

        return clip_tensor

    @torch.no_grad()
    def forward(self, clip_images):
        """
        Extract CLIP ViT patch feature map.

        Args:
            clip_images: (B, 3, 224, 224) – already resized + CLIP-normalised
        Returns:
            patch_feat: (B, clip_dim, 14, 14) – patch tokens reshaped to 2D
        """
        B = clip_images.shape[0]

        # --- Vision transformer (HuggingFace) ---
        vision_outputs = self.vision_model(clip_images, output_hidden_states=False,
                                            return_dict=True)
        # last_hidden_state: (B, 197, 768), drop CLS token
        patches = vision_outputs.last_hidden_state[:, 1:, :]  # (B, 196, 768)

        # --- Visual projection: 768 → 512 (per-patch) ---
        patches = self.visual_projection(patches)               # (B, 196, 512)

        # --- Reshape to spatial map ---
        H = W = int(patches.shape[1] ** 0.5)  # 14
        patch_feat = patches.permute(0, 2, 1).reshape(B, self.clip_dim, H, W)

        return patch_feat

    def encode_text_prompts(self, prompts):
        """
        Encode text prompts to CLIP text embeddings.

        Args:
            prompts: list[str] – e.g. ["a photo of text", "background"]
        Returns:
            text_feat: (N, clip_dim) – normalised text embeddings
        """
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_outputs = self.text_model(**inputs)
            # Take EOS-token hidden state (last non-padding position)
            eos_indices = inputs["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(len(prompts), device=device)
            x_eos = text_outputs.last_hidden_state[batch_idx, eos_indices]  # (N, 512)
            text_feat = self.text_projection(x_eos)                          # (N, 512)
            text_feat = F.normalize(text_feat.float(), dim=-1)

        return text_feat


# ============================================================================
# TGSR: Topology-Conditioned Semantic Routing
# ============================================================================

# Default text prompts (fixed, not learnable in MVP)
_POSITIVE_PROMPTS = [
    "a photo of scene text",
    "written characters in an image",
    "an arbitrary-shaped text instance",
]
_NEGATIVE_PROMPTS = [
    "a background region without text",
    "a natural object rather than text",
    "a non-text image region",
]


class TGSR(nn.Module):
    """
    Topology-Conditioned Semantic Routing.

    Routes CLIP global–local semantics conditioned on TACT geometry stability,
    then applies a bounded residual to classification logits.

    Architecture (per decoder layer):
      ref_points → grid_sample(CLIP map) → local CLIP evidence
                                             │
      Gaussian topology W (detached) ────────┤
                                             ▼
                          topology-aggregated local: z_local
                                             │
      CLIP patch mean ─── global: z_global ──┤
                                             ▼
      prev_ref_points ── reliability: r ────▶ r*z_local + (1-r)*z_global
                                             │
      text prototypes ── cosine contrast ────▶ s_q
                                             │
                                            beta * tanh(s_q) → correction
    """

    def __init__(
        self,
        clip_extractor,          # CLIPFeatureExtractor instance (shared)
        d_model=256,
        num_ctrl_points=16,
        start_layer=4,           # first decoder layer to apply routing
        beta_init=0.0,           # zero-initialised semantic scale
        eta=10.0,                # reliability decay (higher = sharper local↔global switch)
        sigma=0.1,               # Gaussian topology bandwidth
        temperature=0.07,        # cosine similarity temperature
    ):
        super().__init__()
        self.clip_extractor = clip_extractor
        self.clip_dim = clip_extractor.clip_dim
        self.d_model = d_model
        self.num_ctrl_points = num_ctrl_points
        self.start_layer = start_layer
        self.eta = eta
        self.sigma = sigma
        self.temperature = temperature

        # ── Bounded semantic scale (zero init → step 1 == visual baseline) ──
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

        # ── CLIP → decoder-space projector (for mixing) ──
        self.clip_projector = nn.Sequential(
            nn.Linear(self.clip_dim, self.clip_dim),
            nn.LayerNorm(self.clip_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.clip_dim, self.clip_dim),
        )

        # ── Text prototypes (computed once, frozen) ──
        self.register_buffer(
            "t_pos",
            torch.zeros(self.clip_dim),
        )
        self.register_buffer(
            "t_neg",
            torch.zeros(self.clip_dim),
        )
        self._prototypes_initialised = False

    def _init_prototypes(self, device):
        """One-time text prototype initialisation (buffer-safe)."""
        if self._prototypes_initialised:
            return

        t_pos = self.clip_extractor.encode_text_prompts(_POSITIVE_PROMPTS)
        t_neg = self.clip_extractor.encode_text_prompts(_NEGATIVE_PROMPTS)

        # In-place buffer update so state_dict stays consistent
        self.t_pos.data = F.normalize(t_pos.mean(0).float(), dim=-1).to(device)
        self.t_neg.data = F.normalize(t_neg.mean(0).float(), dim=-1).to(device)

        self._prototypes_initialised = True

    # ==================================================================
    # Gaussian topology weights (from control-point distances)
    # ==================================================================

    def _gaussian_topology(self, ref_points):
        """
        Compute row-normalised Gaussian distance weights.

        Args:
            ref_points: (B, N, K, 2) normalised [0,1] control points
        Returns:
            W: (B, N, K, K) row-normalised Gaussian weights (self-exclusive)
        """
        B, N, K, _ = ref_points.shape
        diff = ref_points.unsqueeze(3) - ref_points.unsqueeze(2)  # (B,N,K,K,2)
        dist_sq = (diff ** 2).sum(-1)                               # (B,N,K,K)

        sigma2 = 2.0 * self.sigma ** 2 + 1e-8
        gauss = torch.exp(-dist_sq / sigma2)

        # Remove self-connection
        eye = torch.eye(K, device=ref_points.device).view(1, 1, K, K)
        gauss = gauss * (1.0 - eye)

        # Row-normalise
        gauss = gauss / (gauss.sum(dim=-1, keepdim=True) + 1e-8)
        return gauss

    # ==================================================================
    # Reliability from control-point movement
    # ==================================================================

    def _reliability(self, ref_curr, ref_prev):
        """
        Geometry-adaptive reliability: stable points → trust local;
        unstable points → fall back to global.

        Args:
            ref_curr: (B, N, K, 2)
            ref_prev: (B, N, K, 2)
        Returns:
            r: (B, N, 1) ∈ [0, 1]
        """
        delta = (ref_curr - ref_prev).abs().mean(dim=(2, 3))  # (B, N)
        r = torch.exp(-self.eta * delta).unsqueeze(-1)         # (B, N, 1)
        return r

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(self, clip_feat_map, ref_curr, ref_prev):
        """
        Compute bounded semantic correction for a single decoder layer.

        Args:
            clip_feat_map: (B, C_clip, 14, 14) – CLIP patch feature map
            ref_curr:      (B, N, K, 2) – current-layer control points
            ref_prev:      (B, N, K, 2) – previous-layer control points
        Returns:
            correction: (B, N, 1) – beta * tanh(s_q), zero-init → 0 at step 1
        """
        B, N, K, _ = ref_curr.shape
        device = ref_curr.device
        Hclip, Wclip = clip_feat_map.shape[2], clip_feat_map.shape[3]  # 14, 14

        # ── 1. Dynamic point CLIP sampling ──
        # grid_sample wants (x, y) in [-1, 1]; ref_points are in [0, 1]
        grid = ref_curr * 2.0 - 1.0                     # (B, N, K, 2) → [-1, 1]
        grid = grid.flatten(1, 2).unsqueeze(2)           # (B, N*K, 1, 2)

        # clip_feat_map: (B, C, H, W) → tile N times
        clip_expanded = clip_feat_map.unsqueeze(1).expand(
            -1, N, -1, -1, -1
        ).reshape(B * N, clip_feat_map.shape[1], Hclip, Wclip)

        grid_expanded = grid.reshape(B * N, K, 1, 2)

        local_raw = F.grid_sample(
            clip_expanded, grid_expanded,
            mode="bilinear", padding_mode="border", align_corners=True,
        )  # (B*N, C, K, 1)
        local_raw = local_raw.squeeze(-1).permute(0, 2, 1)  # (B*N, K, C)
        local_raw = local_raw.reshape(B, N, K, self.clip_dim)  # (B, N, K, C)

        # ── 2. Topology-conditioned aggregation ──
        W = self._gaussian_topology(ref_curr).detach()  # (B, N, K, K) — DETACHED
        z_local = torch.einsum("bnij,bnjc->bnic", W, local_raw)  # (B, N, K, C→512)
        z_local = z_local.mean(dim=2)  # (B, N, C) — ave over control points

        z_local = self.clip_projector(z_local)
        z_local = F.normalize(z_local, dim=-1)

        # ── 3. Global CLIP (patch mean) ──
        z_global = clip_feat_map.mean(dim=(2, 3))  # (B, C)
        z_global = self.clip_projector(z_global.unsqueeze(1).expand(-1, N, -1))
        z_global = F.normalize(z_global, dim=-1)

        # ── 4. Reliability-based global-local mixing ──
        r = self._reliability(ref_curr.detach(), ref_prev.detach())  # (B, N, 1)
        z_q = r * z_local + (1.0 - r) * z_global                      # (B, N, C)
        z_q = F.normalize(z_q, dim=-1)

        # ── 5. Contrastive semantic score (text prototypes) ──
        self._init_prototypes(device)
        cos_pos = torch.matmul(z_q, self.t_pos)  # (B, N)
        cos_neg = torch.matmul(z_q, self.t_neg)  # (B, N)
        s_q = (cos_pos - cos_neg) / self.temperature  # (B, N)

        # ── 6. Bounded residual correction ──
        correction = self.beta * torch.tanh(s_q)  # (B, N)
        correction = correction.unsqueeze(-1)      # (B, N, 1)

        return correction
