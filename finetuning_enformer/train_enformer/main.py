# -*- coding: utf-8 -*-
import argparse
from .suite import run_experiment_suite
from .train import train_model_wrapper
from . import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--training_data', type=str, default='virtual', choices=['virtual', 'real'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--ga_steps', type=int, default=4)
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--pairs_per_epoch', type=int, default=None)
    parser.add_argument('--resume_dir', type=str, default="")
    parser.add_argument('--resume_load_opt', action='store_true')
    parser.add_argument('--save_root', type=str, default=config.DEFAULT_SAVE_ROOT_DIR)
    parser.add_argument('--lr', type=float, default=config.DEFAULT_LR)
    parser.add_argument('--wd', type=float, default=config.DEFAULT_WD)
    parser.add_argument('--zero_stage', type=int, default=config.DEFAULT_ZERO_STAGE, choices=[0, 1, 2, 3])
    parser.add_argument('--real_loss', type=str, default='smape', choices=['smape', 'mse'])
    parser.add_argument('--train_batch', type=int, default=1)
    parser.add_argument('--eval_batch', type=int, default=1)
    parser.add_argument('--real_vector_dir', type=str, default=config.PROCESSED_VECTOR_DIR)
    parser.add_argument('--objective', type=str, default='regression', choices=['regression', 'pairwise'])
    parser.add_argument('--suite', action='store_true', help='Run the predefined 30R-5V-30R-5V-30R style suite')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    use_lora = not args.no_lora

    if args.suite:
        run_experiment_suite(local_rank=args.local_rank, seed=args.seed)
    else:
        train_model_wrapper(
            local_rank=args.local_rank,
            training_data=args.training_data,
            num_epochs=args.epochs,
            gradient_accumulation_steps=args.ga_steps,
            use_lora=use_lora,
            lora_rank=args.lora_rank,
            pairs_per_epoch=args.pairs_per_epoch,
            resume_ckpt_dir=args.resume_dir or None,
            load_opt_states_on_resume=args.resume_load_opt,
            save_root_dir=args.save_root,
            learning_rate=args.lr,
            weight_decay=args.wd,
            zero_stage=args.zero_stage,
            real_loss_type=args.real_loss,
            train_batch_size=args.train_batch,
            eval_batch_size=args.eval_batch,
            real_vector_dir=args.real_vector_dir,
            objective=args.objective,
            seed=args.seed,
        )

if __name__ == "__main__":
    main()