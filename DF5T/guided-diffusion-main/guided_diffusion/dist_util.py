"""
Helpers for distributed training.
"""
import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist


import torch as th
import torch.distributed as dist
import os


def setup_dist():
    """
    Setup a distributed process group for multi-GPU training using NCCL backend.
    """
    if dist.is_initialized():
        return

    backend = "nccl"  # NCCL is optimized for multi-GPU communication

    local_rank = int(os.environ["RANK"])  # Local rank per GPU
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of GPUs used

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=local_rank)

    # Set the device for each process
    th.cuda.set_device(local_rank)


def dev():
    """
    Get the current device for each process.
    """
    if th.cuda.is_available():
        local_rank = int(os.environ["RANK"])
        return th.device(f"cuda:{local_rank}")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        p_data = p.data.clone()
        with th.no_grad():
            dist.broadcast(p_data, src=0)
            p.data.copy_(p_data)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()