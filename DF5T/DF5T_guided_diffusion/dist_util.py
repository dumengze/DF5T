import io
import os
import socket

try:
    import blobfile as bf
except Exception:
    bf = None
import torch as th
import torch.distributed as dist


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "gloo"  
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.

    On Windows, absolute paths like 'C:\\Users\\...' can cause blobfile to
    raise 'Unrecognized path'. For local files we bypass blobfile entirely.
    """
    # Prefer native torch.load for local filesystem paths
    if os.path.isfile(path):
        return th.load(path, **kwargs)

    # Fallback to blobfile for remote URLs (gs://, s3://, etc.)
    if bf is None:
        raise FileNotFoundError(f"Unable to load model path without blobfile support: {path}")
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
            dist.broadcast(p_data, 0)  
            p.data.copy_(p_data)  




def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
