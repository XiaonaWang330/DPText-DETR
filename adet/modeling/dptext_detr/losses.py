import torch
import torch.nn as nn
import torch.nn.functional as F
from adet.utils.misc import accuracy, generalized_box_iou, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, is_dist_avail_and_initialized
from detectron2.utils.comm import get_world_size


def _ctrl_to_xyxy(ctrl):
    """控制点 (..., 16, 2) → 轴对齐 bbox xyxy (..., 4)"""
    x = ctrl[..., 0]
    y = ctrl[..., 1]
    return torch.stack(
        [x.min(dim=-1).values, y.min(dim=-1).values,
         x.max(dim=-1).values, y.max(dim=-1).values],
        dim=-1,
    )


def _bbox_iou(b1, b2):
    """逐对 IoU：b1, b2 (N, 4) xyxy → (N,)"""
    x1 = torch.maximum(b1[:, 0], b2[:, 0])
    y1 = torch.maximum(b1[:, 1], b2[:, 1])
    x2 = torch.minimum(b1[:, 2], b2[:, 2])
    y2 = torch.minimum(b1[:, 3], b2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    union = a1 + a2 - inter
    return inter / union.clamp(min=1e-6)


def sigmoid_focal_loss(inputs, targets, num_inst, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss.ndim == 4:
        return loss.mean((1, 2)).sum() / num_inst
    elif loss.ndim == 3:
        return loss.mean(1).sum() / num_inst
    else:
        raise NotImplementedError(f"Unsupported dim {loss.ndim}")


class SetCriterion(nn.Module):
    def __init__(
            self,
            num_classes,
            enc_matcher,
            dec_matcher,
            weight_dict,
            enc_losses,
            dec_losses,
            num_ctrl_points,
            focal_alpha=0.25,
            focal_gamma=2.0,
            ta_head=None,
            ta_temperature=0.10,
            ta_iou_gate=0.5,
            ta_matcher_base_logits=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.enc_matcher = enc_matcher
        self.dec_matcher = dec_matcher
        self.weight_dict = weight_dict
        self.enc_losses = enc_losses
        self.dec_losses = dec_losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_ctrl_points = num_ctrl_points
        # CTP: CLIP Text Prior auxiliary loss (E variant)
        self.ta_head = ta_head
        self.ta_temperature = ta_temperature
        self.ta_iou_gate = ta_iou_gate
        self.ta_matcher_base_logits = ta_matcher_base_logits

    def loss_labels(self, outputs, targets, indices, num_inst, log=False):
        """Classification loss (Focal Loss)."""
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(
            src_logits.shape[:-1], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        shape = list(src_logits.shape)
        shape[-1] += 1
        target_classes_onehot = torch.zeros(
            shape, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device
        )
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]

        # Focal loss reduction
        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            src_logits, target_classes_onehot, reduction="none"
        )
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
        loss_per_element = ce_loss * ((1 - p_t) ** self.focal_gamma)

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * target_classes_onehot + (1 - self.focal_alpha) * (1 - target_classes_onehot)
            loss_per_element = alpha_t * loss_per_element

        if loss_per_element.ndim == 4:
            loss_ce = loss_per_element.mean((1, 2)).sum() / num_inst
        elif loss_per_element.ndim == 3:
            loss_ce = loss_per_element.mean(1).sum() / num_inst

        loss_ce = loss_ce * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_inst):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.mean(-2).argmax(-1) == 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def loss_boxes(self, outputs, targets, indices, num_inst):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_inst

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses['loss_giou'] = loss_giou.sum() / num_inst
        return losses

    def loss_ctrl_points(self, outputs, targets, indices, num_inst):
        """L1 regression loss for control points."""
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_ctrl_points'][idx]
        target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if src_ctrl_points.shape[0] == 0:
            return {'loss_ctrl_points': src_ctrl_points.sum() * 0.0}

        l1 = (src_ctrl_points - target_ctrl_points).abs()
        loss_ctrl_points = l1.sum() / num_inst
        return {'loss_ctrl_points': loss_ctrl_points}

    def loss_ta(self, outputs, targets, indices, num_inst):
        """TA auxiliary loss (E variant): softplus(-margin/T) averaged over
        matched queries whose predicted IoU (bbox from control points) >=
        ta_iou_gate. Gradients flow only into the TA projector; query_feat is
        already detached upstream (LOSS_DETACH), so the shared decoder is not
        touched by this loss."""
        if self.ta_head is None or 'ta_query_feat' not in outputs:
            return {}
        query_feat = outputs['ta_query_feat']   # (B, N, d_model)
        pred_ctrl = outputs['pred_ctrl_points']  # (B, N, 16, 2)
        loss_sum = None
        n_valid = 0
        for i, (src_i, tgt_i) in enumerate(indices):
            n = src_i.shape[0]
            if n == 0:
                continue
            qf = query_feat[i][src_i]  # (n, d_model)
            pb = _ctrl_to_xyxy(pred_ctrl[i][src_i])               # (n, 4)
            gb = _ctrl_to_xyxy(targets[i]['ctrl_points'][tgt_i])  # (n, 4)
            iou = _bbox_iou(pb, gb)                                # (n,)
            m = self.ta_head.forward_margin(qf)                    # (n,)
            per_sample = F.softplus(-m / self.ta_temperature)      # (n,)
            valid = iou >= self.ta_iou_gate
            if valid.sum() == 0:
                continue
            l = per_sample[valid].sum()
            loss_sum = l if loss_sum is None else loss_sum + l
            n_valid += int(valid.sum().item())
        if loss_sum is None:
            loss = torch.zeros((), device=query_feat.device)
        else:
            loss = loss_sum / max(n_valid, 1)
        return {'loss_ta': loss}

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_inst, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'ctrl_points': self.loss_ctrl_points,
            'boxes': self.loss_boxes,
            'ta': self.loss_ta,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_inst, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {
            k: v for k, v in outputs.items()
            if k not in ('aux_outputs', 'enc_outputs')
        }

        # Matching between outputs of the last layer and targets.
        # CTP variant (D/E): matcher classification cost uses BASE logits
        # (without the TA residual adapter) so matching is not perturbed by
        # the adapter; only the classification LOSS uses the fused logits.
        matching_outputs = outputs_without_aux
        if self.ta_matcher_base_logits and 'pred_logits_base' in outputs_without_aux:
            matching_outputs = dict(outputs_without_aux)
            matching_outputs['pred_logits'] = outputs_without_aux['pred_logits_base']
        indices = self.dec_matcher(matching_outputs, targets)

        # Average number of target boxes across all nodes
        num_inst = sum(len(t['ctrl_points']) for t in targets)
        num_inst = torch.as_tensor([num_inst], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()

        # Compute all requested losses
        losses = {}
        for loss in self.dec_losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_inst, **kwargs))

        # Auxiliary losses
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.dec_matcher(aux_outputs, targets)
                for loss in self.dec_losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_inst, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.enc_matcher(enc_outputs, targets)
            for loss in self.enc_losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_inst, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
