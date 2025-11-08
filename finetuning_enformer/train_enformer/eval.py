# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np
import torch
from scipy.stats import pearsonr
from .losses import _flatten_pred_target


def evaluate_model(model_engine, dataloader, device) -> Tuple[float, float]:
    model_engine.eval()
    G = len(dataloader.dataset.genes)
    flat_preds, flat_targs = [], []
    total_loss = 0.0
    with torch.no_grad():
        for seq1, seq2, target in dataloader:
            seq1 = seq1.to(device).to(torch.bfloat16)
            seq2 = seq2.to(device).to(torch.bfloat16)
            tgt = target.to(device).float()
            out = model_engine(seq1, seq2)
            out, tgt = _flatten_pred_target(out, tgt)
            numerator = torch.abs(out - tgt)
            denominator = (torch.abs(out) + torch.abs(tgt)) / 2.0 + 1e-6
            loss = (numerator / denominator).mean()
            total_loss += loss.item()
            flat_preds.extend(out.detach().cpu().view(-1).tolist())
            flat_targs.extend(tgt.detach().cpu().view(-1).tolist())
    avg_loss = total_loss / max(1, len(dataloader))
    preds_arr = np.array(flat_preds).reshape(-1, 2).mean(axis=1)
    targs_arr = np.array(flat_targs).reshape(-1, 2).mean(axis=1)
    preds_per_gene = preds_arr.reshape(-1, G).T
    targs_per_gene = targs_arr.reshape(-1, G).T
    rhos = []
    for p, t in zip(preds_per_gene, targs_per_gene):
        if len(p) >= 2:
            rhos.append(pearsonr(p, t)[0])
    mean_rho = float(np.mean(rhos)) if rhos else 0.0
    return avg_loss, mean_rho