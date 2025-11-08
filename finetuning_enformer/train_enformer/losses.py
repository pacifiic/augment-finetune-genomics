# -*- coding: utf-8 -*-
from typing import Tuple
import torch
import torch.nn.functional as F


def _flatten_pred_target(output: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if output.dim() == 2 and output.size(-1) == 1:
        output = output.squeeze(-1)
    return output.contiguous().view(-1), target.contiguous().view(-1)


def smape_loss(output, target, epsilon=1e-6):
    output, target = _flatten_pred_target(output.float(), target.float())
    numerator = torch.abs(output - target)
    denominator = (torch.abs(output) + torch.abs(target)) / 2.0 + epsilon
    return numerator.div(denominator).mean()


def bradley_terry_loss(predA, predB, yA, yB, beta=1.0, eps=1e-8):
    if predA.dim() == 2 and predA.size(1) == 1:
        predA = predA.squeeze(1)
    if predB.dim() == 2 and predB.size(1) == 1:
        predB = predB.squeeze(1)
    logits = predA - predB
    target = (yA > yB).float()
    diff = (yA - yB).abs()
    mask = diff > eps
    if mask.sum() == 0:
        return logits.sum() * 0.0
    beta_t = torch.as_tensor(beta, device=logits.device, dtype=logits.dtype)
    scaled_logits = logits * beta_t
    return F.binary_cross_entropy_with_logits(scaled_logits[mask], target[mask])


class GeneBetaScheduler:
    def __init__(self, params_df, beta_min=0.5, beta_max=1.2):
        self.params_df = params_df
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta(self, yA, yB, gene):
        if gene not in self.params_df.index:
            center, k = 0.2, 5.0
        else:
            center = float(self.params_df.loc[gene, "center"])
            k = float(self.params_df.loc[gene, "k"])
        with torch.no_grad():
            mean_diff = (yA - yB).abs().mean().item()
            scale = torch.sigmoid(torch.tensor(k * (mean_diff - center))).item()
            beta = self.beta_min + (self.beta_max - self.beta_min) * scale
        return float(beta)