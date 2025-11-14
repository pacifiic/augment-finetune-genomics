# -*- coding: utf-8 -*-
import os
import torch
import torch.distributed as dist
from .train import train_model_wrapper
from . import config


def run_experiment_suite(local_rank: int):
    base_root = "/path/to/checkpoint/dir"
    seeds = [42, 47, 52]

    for seed in seeds:
        print(f"\n================= [SEED {seed}] =================")

        # ==== Experiment 1 ====
        exp1_root = os.path.join(base_root, f"exp01_real_regression100_seed{seed}")
        exp1 = {
            "name": f"exp01_real_regression100_seed{seed}",
            "training_data": "real",
            "use_lora": True,
            "lora_rank": 32,
            "num_epochs": 100,
            "objective": "regression",
            "real_loss_type": "smape",
            "save_root": exp1_root,
            "seed": seed,
        }

        # ==== Experiment 2 ====
        exp2_base = os.path.join(base_root, f"exp02_real_virtual_mix_seed{seed}")
        exp2_1 = {
            "name": f"exp02_real_regression30_seed{seed}",
            "training_data": "real",
            "use_lora": True,
            "lora_rank": 32,
            "num_epochs": 30,
            "objective": "regression",
            "real_loss_type": "smape",
            "save_root": os.path.join(exp2_base, "phase1_real30"),
            "seed": seed,
        }
        exp2_2 = {
            "name": f"exp02_virtual_bt5_seed{seed}",
            "training_data": "virtual",
            "use_lora": True,
            "lora_rank": 32,
            "num_epochs": 5,
            "objective": "pairwise",
            "load_opt_states_on_resume": False,
            "load_from_root": exp2_1["save_root"],
            "save_root": os.path.join(exp2_base, "phase2_virtual5"),
            "seed": seed,
        }
        exp2_3 = {
            "name": f"exp02_real_regression30_2_seed{seed}",
            "training_data": "real",
            "use_lora": True,
            "lora_rank": 32,
            "num_epochs": 30,
            "objective": "regression",
            "real_loss_type": "smape",
            "load_from_root": exp2_2["save_root"],
            "save_root": os.path.join(exp2_base, "phase3_real30"),
            "seed": seed,
        }
        exp2_4 = {
            "name": f"exp02_virtual_bt5_2_seed{seed}",
            "training_data": "virtual",
            "use_lora": True,
            "lora_rank": 32,
            "num_epochs": 5,
            "objective": "pairwise",
            "load_opt_states_on_resume": False,
            "load_from_root": exp2_3["save_root"],
            "save_root": os.path.join(exp2_base, "phase4_virtual5"),
            "seed": seed,
        }
        exp2_5 = {
            "name": f"exp02_real_regression30_3_seed{seed}",
            "training_data": "real",
            "use_lora": True,
            "lora_rank": 32,
            "num_epochs": 30,
            "objective": "regression",
            "real_loss_type": "smape",
            "load_from_root": exp2_4["save_root"],
            "save_root": os.path.join(exp2_base, "phase5_real30"),
            "seed": seed,
        }

        # ==== Experiment 3 ====
        exp3_base = os.path.join(base_root, f"exp03_real_only_mix_seed{seed}")
        exp3_2 = {
            "name": f"exp03_real_bt5_seed{seed}",
            "training_data": "real",
            "use_lora": True,
            "lora_rank": 32,
            "num_epochs": 5,
            "objective": "pairwise",
            "load_opt_states_on_resume": False,
            "load_from_root": exp2_1["save_root"],
            "save_root": os.path.join(exp3_base, "phase2_real_bt5"),
            "seed": seed,
        }
        exp3_3 = {
            "name": f"exp03_real_regression30_2_seed{seed}",
            "training_data": "real",
            "use_lora": True,
            "lora_rank": 32,
            "num_epochs": 30,
            "objective": "regression",
            "real_loss_type": "smape",
            "load_from_root": exp3_2["save_root"],
            "save_root": os.path.join(exp3_base, "phase3_real30"),
            "seed": seed,
        }
        exp3_4 = {
            "name": f"exp03_real_bt5_2_seed{seed}",
            "training_data": "real",
            "use_lora": True,
            "lora_rank": 32,
            "num_epochs": 5,
            "objective": "pairwise",
            "load_opt_states_on_resume": False,
            "load_from_root": exp3_3["save_root"],
            "save_root": os.path.join(exp3_base, "phase4_real_bt5"),
            "seed": seed,
        }
        exp3_5 = {
            "name": f"exp03_real_regression30_3_seed{seed}",
            "training_data": "real",
            "use_lora": True,
            "lora_rank": 32,
            "num_epochs": 30,
            "objective": "regression",
            "real_loss_type": "smape",
            "load_from_root": exp3_4["save_root"],
            "save_root": os.path.join(exp3_base, "phase5_real30"),
            "seed": seed,
        }

        all_experiments = [
            exp1,
            exp2_1, exp2_2, exp2_3, exp2_4, exp2_5,
            exp3_2, exp3_3, exp3_4, exp3_5
        ]

        for cfg in all_experiments:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\n========== [RUN] {cfg['name']} ==========")

            train_model_wrapper(
                local_rank=local_rank,
                training_data=cfg["training_data"],
                num_epochs=cfg["num_epochs"],
                gradient_accumulation_steps=4,
                use_lora=cfg["use_lora"],
                lora_rank=cfg["lora_rank"],
                pairs_per_epoch=None,
                resume_ckpt_dir=cfg.get("resume_ckpt_dir"),
                load_opt_states_on_resume=cfg.get("load_opt_states_on_resume", False),
                save_root_dir=cfg["save_root"],
                learning_rate=config.DEFAULT_LR,
                weight_decay=config.DEFAULT_WD,
                zero_stage=config.DEFAULT_ZERO_STAGE,
                real_loss_type=cfg.get("real_loss_type", "smape"),
                train_batch_size=1,
                eval_batch_size=1,
                real_vector_dir=config.PROCESSED_VECTOR_DIR,
                objective=cfg.get("objective", "regression"),
                load_from_root=cfg.get("load_from_root"),
                seed=cfg["seed"],
            )

            torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.barrier()
