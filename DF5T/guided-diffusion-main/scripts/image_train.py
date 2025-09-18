import os
import argparse
import sys
import torch
import torch.multiprocessing as mp
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
    args = create_argparser().parse_args()

    # Spawn 3 processes for 3 GPUs
    mp.spawn(run_worker, nprocs=4, args=(args,))


def run_worker(local_rank, args):
    """
    This function runs training on each worker, with each worker assigned a different GPU.
    """
    # Set local_rank for each process and initialize the distributed environment
    os.environ["RANK"] = str(local_rank)  # Assign the rank of each process (0, 1, 2)
    os.environ["WORLD_SIZE"] = "4"  # Total number of processes (GPUs)
    dist_util.setup_dist()  # Initialize the distributed process group

    # Configure logger (only on rank 0)
    logger.configure()

    logger.log(f"[local_rank={local_rank}] creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())  # Move model to the appropriate device (GPU)

    # Create schedule sampler
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"[local_rank={local_rank}] creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log(f"[local_rank={local_rank}] training...")
    # Start training loop
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/root/dmz/dataset",
        schedule_sampler="uniform",
        lr=1e-5,
        weight_decay=1e-6,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=100000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()