# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import random

# ========= Logging & seed =========
def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def log_trainable_params(model: nn.Module, title: str = "MODEL"):
    total, trainable = _count_parameters(model)
    ratio = (trainable / total) if total > 0 else 0.0
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[{title}] total_params={total:,} | trainable_params={trainable:,} | ratio={ratio:.4f}")


def reset_gpu_peak_memory(device):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def log_gpu_memory(prefix: str, device):
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        reserv = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        print(f"[{prefix}][RANK {dist.get_rank() if dist.is_initialized() else 0}] "
              f"peak_allocated={alloc:.2f} GB | peak_reserved={reserv:.2f} GB")


def safe_load_tensor(path: str):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)

# ========= Freezing =========

def _freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def freeze_stem_and_conv_tower(enf: nn.Module):
    frozen = set()
    for attr in ("stem", "conv_tower"):
        if hasattr(enf, attr) and isinstance(getattr(enf, attr), nn.Module):
            _freeze_module(getattr(enf, attr))
            frozen.add(attr)
    for name, module in enf.named_modules():
        last = name.split(".")[-1] if name else ""
        if last in ("stem", "conv_tower"):
            _freeze_module(module)
            frozen.add(name)
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        msg = ", ".join(sorted(frozen)) if frozen else "(none found)"
        print(f"[FREEZE] Frozen Enformer submodules: {msg}")
    return list(sorted(frozen))

# ========= PEFT wrapper =========
class EnformerPEFTWrapper(nn.Module):
    def __init__(self, enformer_model):
        super().__init__()
        self.enformer = enformer_model

    def forward(self, input_ids=None, seq=None, **kwargs):
        if seq is None:
            raise ValueError("Enformer expects 'seq' as input, but got None")
        return self.enformer(seq)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.enformer, name)

# ========= Dataloader worker seed =========

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)