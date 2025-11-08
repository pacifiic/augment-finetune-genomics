# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler
from .losses import smape_loss, bradley_terry_loss, GeneBetaScheduler
from .eval import evaluate_model
from .utils import reset_gpu_peak_memory, log_gpu_memory
from .checkpoints import save_checkpoint

def train_virtual_regression_loop(model_engine, train_loader, valid_loader, device, num_epochs: int, save_dir: str,
                                  loss_type: str = "smape", start_epoch: int = 0, seed: int = 42, dist=None):
    criterion = smape_loss if loss_type.lower() == "smape" else (lambda o, t: F.mse_loss(*_flatten_pred_target(o, t)))
    world_size = dist.get_world_size()
    for epoch in range(start_epoch, num_epochs):
        reset_gpu_peak_memory(device)
        epoch_seed = seed + epoch * world_size + dist.get_rank()
        torch.manual_seed(epoch_seed)
        model_engine.train()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            seq, y, _gene = batch
            seq = seq.to(device).to(torch.bfloat16)
            y   = y.to(device).float()
            pred = model_engine(seq, seq)
            loss = criterion(pred, y)
            model_engine.backward(loss)
            model_engine.step()
            running_loss += float(loss.detach().item())
            if dist.get_rank() == 0 and (step + 1) % 200 == 0:
                print(f"[VIRTUAL-REG][E{epoch + 1}] step {step + 1}/{len(train_loader)} | "
                      f"loss(avg)={running_loss / (step + 1):.4f}")
        avg_train_loss = running_loss / max(1, len(train_loader))
        val_loss, val_rho = evaluate_model(model_engine, valid_loader, device)
        save_checkpoint(epoch + 1, avg_train_loss, val_loss, val_rho, save_dir, model_engine)
        log_gpu_memory(f"VIRTUAL-REG Epoch {epoch + 1}", device)
        if dist.get_rank() == 0:
            print(f"[VIRTUAL-REG] Epoch {epoch + 1}/{num_epochs} | "
                  f"Train={avg_train_loss:.4f} | Val(SMAPE)={val_loss:.4f} | Pearson={val_rho:.4f}")


def train_virtual_pairwise_loop(model_engine, train_loader, valid_loader, device, num_epochs: int, save_dir: str,
                       virtual_beta_params_df, start_epoch: int = 0, seed: int = 42, dist=None):
    gene_beta = GeneBetaScheduler(virtual_beta_params_df, beta_min=0.5, beta_max=1.2)
    world_size = dist.get_world_size()
    for epoch in range(start_epoch, num_epochs):
        reset_gpu_peak_memory(device)
        epoch_seed = seed + epoch * world_size + dist.get_rank()
        torch.manual_seed(epoch_seed)
        model_engine.train()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            seqA, yA, seqB, yB, gene = batch
            seqA = seqA.to(device).to(torch.bfloat16)
            seqB = seqB.to(device).to(torch.bfloat16)
            yA = yA.to(device).float()
            yB = yB.to(device).float()
            genes = list(gene) if isinstance(gene, (list, tuple)) else [gene] * yA.shape[0]
            betas = [gene_beta.get_beta(yA[i:i+1], yB[i:i+1], g) for i, g in enumerate(genes)]
            beta = torch.tensor(betas, device=device, dtype=torch.float32)
            predA = model_engine(seqA, seqA)
            predB = model_engine(seqB, seqB)
            loss = bradley_terry_loss(predA, predB, yA, yB, beta=beta)
            model_engine.backward(loss)
            model_engine.step()
            running_loss += float(loss.detach().item())
            if dist.get_rank() == 0 and (step + 1) % 200 == 0:
                print(f"[VIRTUAL][E{epoch + 1}] step {step + 1}/{len(train_loader)} | loss(avg)={running_loss/(step+1):.4f}")
        avg_train_loss = running_loss / max(1, len(train_loader))
        val_loss, val_rho = evaluate_model(model_engine, valid_loader, device)
        save_checkpoint(epoch + 1, avg_train_loss, val_loss, val_rho, save_dir, model_engine)
        log_gpu_memory(f"VIRTUAL Epoch {epoch + 1}", device)
        if dist.get_rank() == 0:
            print(f"[VIRTUAL] Epoch {epoch + 1}/{num_epochs} | Train={avg_train_loss:.4f} | Val(SMAPE)={val_loss:.4f} | Pearson={val_rho:.4f}")


def train_real_pairwise_loop(model_engine, train_loader, valid_loader, device, num_epochs: int, save_dir: str,
                              real_beta_params_df, start_epoch: int = 0, seed: int = 42, dist=None):
    gene_beta = GeneBetaScheduler(real_beta_params_df, beta_min=0.5, beta_max=1.2)
    world_size = dist.get_world_size()
    for epoch in range(start_epoch, num_epochs):
        reset_gpu_peak_memory(device)
        epoch_seed = seed + epoch * world_size + dist.get_rank()
        torch.manual_seed(epoch_seed)
        model_engine.train()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            a1, a2, yA, b1, b2, yB, gene = batch
            a1 = a1.to(device).to(torch.bfloat16)
            a2 = a2.to(device).to(torch.bfloat16)
            b1 = b1.to(device).to(torch.bfloat16)
            b2 = b2.to(device).to(torch.bfloat16)
            yA = yA.to(device).float()
            yB = yB.to(device).float()
            genes = list(gene) if isinstance(gene, (list, tuple)) else [gene] * yA.shape[0]
            betas = [gene_beta.get_beta(yA[i:i+1], yB[i:i+1], g) for i, g in enumerate(genes)]
            beta = torch.tensor(betas, device=device, dtype=torch.float32)
            predA = model_engine(a1, a2)
            predB = model_engine(b1, b2)
            loss = bradley_terry_loss(predA, predB, yA, yB, beta=beta)
            model_engine.backward(loss)
            model_engine.step()
            running_loss += float(loss.detach().item())
            if dist.get_rank() == 0 and (step + 1) % 200 == 0:
                print(f"[REAL-PW][E{epoch + 1}] step {step + 1}/{len(train_loader)} | loss(avg)={running_loss/(step+1):.4f}")
        val_loss, val_rho = evaluate_model(model_engine, valid_loader, device)
        avg_train_loss = running_loss / max(1, len(train_loader))
        save_checkpoint(epoch + 1, avg_train_loss, val_loss, val_rho, save_dir, model_engine)
        log_gpu_memory(f"REAL-PAIRWISE Epoch {epoch + 1}", device)
        if dist.get_rank() == 0:
            print(f"[REAL-PAIRWISE] Epoch {epoch + 1}/{num_epochs} | Train(BT)={avg_train_loss:.4f} | Val(SMAPE)={val_loss:.4f} | Pearson={val_rho:.4f}")


def train_real_regression_loop(model_engine, train_loader, valid_loader, device, num_epochs: int, save_dir: str,
                    loss_type: str = "smape", start_epoch: int = 0, seed: int = 42, dist=None):
    criterion = smape_loss if loss_type.lower() == "smape" else (lambda o, t: F.mse_loss(*_flatten_pred_target(o, t)))
    from .losses import _flatten_pred_target  # local import to avoid circular
    for epoch in range(start_epoch, num_epochs):
        reset_gpu_peak_memory(device)
        model_engine.train()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            seq1, seq2, target = batch
            seq1 = seq1.to(device).to(torch.bfloat16)
            seq2 = seq2.to(device).to(torch.bfloat16)
            target = target.to(device).float()
            pred = model_engine(seq1, seq2)
            loss = criterion(pred, target)
            model_engine.backward(loss)
            model_engine.step()
            running_loss += float(loss.detach().item())
            if dist.get_rank() == 0 and (step + 1) % 200 == 0:
                print(f"[REAL][E{epoch + 1}] step {step + 1}/{len(train_loader)} | loss(avg)={running_loss/(step+1):.4f}")
        avg_train_loss = running_loss / max(1, len(train_loader))
        val_loss, val_rho = evaluate_model(model_engine, valid_loader, device)
        save_checkpoint(epoch + 1, avg_train_loss, val_loss, val_rho, save_dir, model_engine)
        log_gpu_memory(f"REAL Epoch {epoch + 1}", device)
        if dist.get_rank() == 0:
            print(f"[REAL] Epoch {epoch + 1}/{num_epochs} | Train={avg_train_loss:.4f} | Val(SMAPE)={val_loss:.4f} | Pearson={val_rho:.4f}")
