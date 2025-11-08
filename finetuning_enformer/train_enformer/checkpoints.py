# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import os
import re
import json
from datetime import datetime
import torch.distributed as dist

_STEP_RE = re.compile(r"global_step(\d+)$")

def _extract_step(tag: str) -> int:
    m = _STEP_RE.search(tag)
    return int(m.group(1)) if m else -1


def pick_latest_checkpoint_dir(save_root: str) -> Optional[str]:
    if not os.path.isdir(save_root):
        return None
    base = os.path.basename(os.path.normpath(save_root))
    if base.startswith("checkpoint_"):
        return save_root
    best = (-1, None)
    for d in os.listdir(save_root):
        if not d.startswith("checkpoint_"):
            continue
        ckpt_dir = os.path.join(save_root, d)
        if not os.path.isdir(ckpt_dir):
            continue
        tags = [t for t in os.listdir(ckpt_dir)
                if os.path.isdir(os.path.join(ckpt_dir, t)) and t.startswith("global_step")]
        step = max(_extract_step(t) for t in tags) if tags else int(os.path.getmtime(ckpt_dir))
        if step > best[0]:
            best = (step, ckpt_dir)
    return best[1]


def resolve_resume_dir_across_ranks(load_from_root: Optional[str]) -> Optional[str]:
    path = None
    if load_from_root:
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            path = pick_latest_checkpoint_dir(load_from_root)
        if dist.is_initialized():
            buf = [path]
            dist.broadcast_object_list(buf, src=0)
            path = buf[0]
    return path


def _stepnum(t: str) -> int:
    try:
        return int(t.replace("global_step", ""))
    except Exception:
        return -1


def _find_latest_tag_and_start_epoch(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None, 0
    tags = [d for d in os.listdir(ckpt_dir) if d.startswith("global_step")]
    latest_tag = None
    if tags:
        tags.sort(key=_stepnum)
        latest_tag = tags[-1]
    start_epoch = 0
    meta_files = [f for f in os.listdir(ckpt_dir) if f.startswith("metadata_") and f.endswith(".json")]
    for mf in meta_files:
        try:
            with open(os.path.join(ckpt_dir, mf), "r") as f:
                meta = json.load(f)
            e = int(meta.get("epoch", 0))
            if e > start_epoch:
                start_epoch = e
        except Exception:
            pass
    return latest_tag, start_epoch


def load_latest_checkpoint(model_engine, resume_ckpt_dir: Optional[str], load_optimizer_states: bool = False):
    if not resume_ckpt_dir:
        return False, 0
    tag, start_epoch = _find_latest_tag_and_start_epoch(resume_ckpt_dir)
    if tag is None:
        if dist.get_rank() == 0:
            print(f"[CKPT] No 'global_step*' tag under: {resume_ckpt_dir}")
        return False, 0
    success, _ = model_engine.load_checkpoint(
        resume_ckpt_dir,
        tag=tag,
        load_optimizer_states=load_optimizer_states,
        load_lr_scheduler_states=load_optimizer_states,
    )
    dist.barrier()
    if dist.is_initialized():
        buf = [start_epoch]
        dist.broadcast_object_list(buf, src=0)
        start_epoch = buf[0]
    if dist.get_rank() == 0:
        print(f"[CKPT-LOAD] success={success} | from {resume_ckpt_dir}@{tag} "
              f"(opt={load_optimizer_states}, sched={load_optimizer_states}) | start_epoch={start_epoch}")
    return bool(success), start_epoch


def save_checkpoint(epoch, train_loss, valid_loss, pearson_rho, save_dir, model_engine):
    if dist.get_rank() == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        timestamp = None
    timestamp_list = [timestamp]
    dist.broadcast_object_list(timestamp_list, src=0)
    timestamp = timestamp_list[0]
    checkpoint_dir = os.path.join(save_dir, f'checkpoint_{timestamp}')
    if dist.get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    dist.barrier()
    metadata = {
        'epoch': epoch,
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'Pearson_correlation': pearson_rho,
        'timestamp': timestamp
    }
    metadata_path = os.path.join(checkpoint_dir, f'metadata_{timestamp}.json')
    if dist.get_rank() == 0:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"[CKPT-SAVE] Metadata -> {metadata_path}")
    model_engine.save_checkpoint(checkpoint_dir)
    if dist.get_rank() == 0:
        print(f"[CKPT-SAVE] Model/optimizer -> {checkpoint_dir}")