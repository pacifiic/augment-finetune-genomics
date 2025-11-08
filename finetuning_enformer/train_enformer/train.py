# -*- coding: utf-8 -*-
from typing import Optional
import math
import torch
import torch.distributed as dist
from torch.optim import AdamW
import deepspeed

from . import config
from .utils import set_seed_all, log_trainable_params
from .dataloaders import create_virtual_pair_loader, create_real_regression_loader, create_real_pair_loader, create_virtual_regression_loader
from .model import build_dual_enformer
from .checkpoints import load_latest_checkpoint, resolve_resume_dir_across_ranks
from .train_loops import train_virtual_pairwise_loop, train_real_pairwise_loop, train_real_regression_loop, train_virtual_regression_loop
import pandas as pd


def train_model_wrapper(local_rank: int,
                        training_data: str = "virtual",
                        num_epochs: int = 50,
                        gradient_accumulation_steps: int = 4,
                        use_lora: bool = True,
                        lora_rank: int = 32,
                        pairs_per_epoch: Optional[int] = None,
                        resume_ckpt_dir: Optional[str] = None,
                        load_opt_states_on_resume: bool = False,
                        save_root_dir: str = config.DEFAULT_SAVE_ROOT_DIR,
                        learning_rate: float = config.DEFAULT_LR,
                        weight_decay: float = config.DEFAULT_WD,
                        zero_stage: int = config.DEFAULT_ZERO_STAGE,
                        real_loss_type: str = "smape",
                        train_batch_size: int = 1,
                        eval_batch_size: int = 1,
                        real_vector_dir: Optional[str] = None,
                        objective: str = "regression",
                        load_from_root: Optional[str] = None,
                        seed: int = 42):
    set_seed_all(seed)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Load beta params once here and pass to loops
    virtual_beta_params_df = pd.read_csv(config.VIRTUAL_BETA_PARAMS_PATH); virtual_beta_params_df.set_index("gene", inplace=True)
    real_beta_params_df    = pd.read_csv(config.REAL_BETA_PARAMS_PATH);    real_beta_params_df.set_index("gene", inplace=True)

    if training_data.lower() == "virtual":
        data_dir = real_vector_dir or config.PROCESSED_VECTOR_DIR
        if objective.lower() == "pairwise":
            train_loader = create_virtual_pair_loader(config.TRAIN_GENES_FILE, config.VIRTUAL_VECTOR_DIR, config.PSEUDO_LABEL_PATH,
                                                   batch_size=train_batch_size, training=True, pairs_per_epoch=pairs_per_epoch)
            valid_loader = create_real_regression_loader(config.TRAIN_GENES_FILE, config.VALID_SAMPLES_FILE, data_dir,
                                              batch_size=eval_batch_size, training=False)
        else:
            train_loader = create_virtual_regression_loader(config.TRAIN_GENES_FILE, config.VIRTUAL_VECTOR_DIR, config.PSEUDO_LABEL_PATH,
                                              batch_size=train_batch_size, training=True, pairs_per_epoch=pairs_per_epoch)
            valid_loader = create_real_regression_loader(config.TRAIN_GENES_FILE, config.VALID_SAMPLES_FILE, data_dir,
                                              batch_size=eval_batch_size, training=False)
       
    elif training_data.lower() == "real":
        data_dir = real_vector_dir or config.PROCESSED_VECTOR_DIR
        if objective.lower() == "pairwise":
            train_loader = create_real_pair_loader(config.TRAIN_GENES_FILE, config.TRAIN_SAMPLES_FILE, data_dir,
                                                   batch_size=train_batch_size, training=True, pairs_per_epoch=pairs_per_epoch)
            valid_loader = create_real_regression_loader(config.TRAIN_GENES_FILE, config.VALID_SAMPLES_FILE, data_dir,
                                              batch_size=eval_batch_size, training=False)
        else:
            train_loader = create_real_regression_loader(config.TRAIN_GENES_FILE, config.TRAIN_SAMPLES_FILE, data_dir,
                                              batch_size=train_batch_size, training=True)
            valid_loader = create_real_regression_loader(config.TRAIN_GENES_FILE, config.VALID_SAMPLES_FILE, data_dir,
                                              batch_size=eval_batch_size, training=False)
    else:
        raise ValueError("training_data must be one of {'virtual','real'}")

    model = build_dual_enformer(use_lora=use_lora, lora_rank=lora_rank).to(device)
    log_trainable_params(model, title=f"DUAL-ENFORMER (vector-only, lora={'on' if use_lora else 'off'}, rank={lora_rank})")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    micro_steps_per_epoch = len(train_loader)
    global_micro_steps = micro_steps_per_epoch * num_epochs
    total_steps = max(1, (global_micro_steps + gradient_accumulation_steps - 1) // gradient_accumulation_steps)

    def lr_lambda(current_step):
        warmup_steps = config.DEFAULT_WARMUP_STEPS
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ds_config = {
        "train_batch_size": train_batch_size * dist.get_world_size() * gradient_accumulation_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_optimization": {"stage": int(zero_stage)},
        "gradient_clipping": 1.0,
    }
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), optimizer=optimizer, lr_scheduler=scheduler, config=ds_config
    )

    start_epoch = 0
    if resume_ckpt_dir:
        _, start_epoch = load_latest_checkpoint(model_engine, resume_ckpt_dir=resume_ckpt_dir,
                                                load_optimizer_states=load_opt_states_on_resume)
    elif load_from_root:
        resume_ckpt_dir = resolve_resume_dir_across_ranks(load_from_root)
        _, start_epoch = load_latest_checkpoint(model_engine, resume_ckpt_dir=resume_ckpt_dir,
                                                load_optimizer_states=load_opt_states_on_resume)
        if not load_opt_states_on_resume:
            num_epochs += start_epoch
        if (dist.get_rank() == 0) and (resume_ckpt_dir is None):
            print(f"[WARN] No checkpoint_* found under: {load_from_root}")

    if training_data.lower() == "virtual":
        if objective.lower() == "pairwise":
            train_virtual_pairwise_loop(model_engine, train_loader, valid_loader, device, num_epochs, save_root_dir,
                           virtual_beta_params_df=virtual_beta_params_df, start_epoch=start_epoch, seed=seed, dist=dist)
        else:
            train_virtual_regression_loop(model_engine, train_loader, valid_loader, device, num_epochs, save_root_dir,
                            loss_type=real_loss_type, start_epoch=start_epoch, seed=seed, dist=dist)
    else:
        if objective.lower() == "pairwise":
            train_real_pairwise_loop(model_engine, train_loader, valid_loader, device, num_epochs, save_root_dir,
                                     real_beta_params_df=real_beta_params_df, start_epoch=start_epoch, seed=seed, dist=dist)
        else:
            train_real_regression_loop(model_engine, train_loader, valid_loader, device, num_epochs, save_root_dir,
                            loss_type=real_loss_type, start_epoch=start_epoch, seed=seed, dist=dist)

    if dist.get_rank() == 0:
        print("Training completed.")
