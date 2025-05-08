import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager

from tools.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@contextmanager
def working_directory(path):
    """Context manager for temporarily changing working directory"""
    origin = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

def validate_file_path(path: str, check_exists: bool = True) -> Path:
    """Validate if a file path is accessible and optionally exists"""
    try:
        path_obj = Path(path).resolve()
        if check_exists and not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path_obj
    except Exception as e:
        logger.error(f"Invalid file path {path}: {str(e)}")
        raise

def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--doc", type=str, required=True, help="Name of the log folder")
    parser.add_argument("--deg", type=str, required=True, help="Degradation type")
    parser.add_argument("--sigma_0", type=float, required=True, help="Sigma_0 value")
    
    # Optional arguments with defaults
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp", help="Path for experiment data")
    parser.add_argument("--comment", type=str, default="", help="Experiment comment")
    parser.add_argument("--verbose", type=str, default="info", 
                       choices=["debug", "info", "warning", "error"], help="Verbose level")
    parser.add_argument("--sample", action="store_true", help="Produce samples from model")
    parser.add_argument("--image_folder", type=str, default="images", help="Folder for samples")
    parser.add_argument("--ni", action="store_true", help="Non-interactive mode")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--eta", type=float, default=0.85, help="Eta value")
    parser.add_argument("--etaB", type=float, default=1.0, help="Eta_b value")
    parser.add_argument("--subset_start", type=int, default=-1, help="Subset start index")
    parser.add_argument("--subset_end", type=int, default=-1, help="Subset end index")
    parser.add_argument("--gain_factor", type=float, default=0.1, help="Gain factor")

    try:
        args = parser.parse_args()
    except SystemExit as e:
        logger.error("Failed to parse arguments")
        raise

    # Validate arguments
    if args.timesteps <= 0:
        raise ValueError("Timesteps must be positive")
    if args.sigma_0 <= 0:
        raise ValueError("sigma_0 must be positive")
    if not 0 <= args.eta <= 1:
        raise ValueError("eta must be between 0 and 1")
    if args.subset_start != -1 and args.subset_end != -1 and args.subset_start > args.subset_end:
        raise ValueError("subset_start must be less than subset_end")

    # Setup paths
    args.exp = validate_file_path(args.exp, check_exists=False).as_posix()
    args.log_path = os.path.join(args.exp, "model", args.doc)
    args.image_folder = os.path.join(args.exp, "result", args.image_folder)

    # Setup logging
    log_level = getattr(logging, args.verbose.upper())
    logger.setLevel(log_level)
    
    # Create directories
    try:
        os.makedirs(args.log_path, exist_ok=True)
        os.makedirs(args.image_folder, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        raise

    # Handle existing image folder
    if os.path.exists(args.image_folder) and not args.ni:
        response = input(f"Image folder {args.image_folder} exists. Overwrite? (Y/N): ")
        if response.upper() != "Y":
            logger.info("Program halted by user")
            sys.exit(0)
        try:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        except Exception as e:
            logger.error(f"Failed to recreate image folder: {str(e)}")
            raise

    # Load and validate config
    try:
        config_path = validate_file_path(os.path.join("configs", args.config))
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Empty configuration file")
        new_config = dict2namespace(config)
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    new_config.device = device

    # Set random seeds
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    except Exception as e:
        logger.error(f"Failed to set random seeds: {str(e)}")
        raise

    return args, new_config

def dict2namespace(config):
    """Convert dictionary to namespace recursively"""
    namespace = argparse.Namespace()
    try:
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(namespace, key, dict2namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace
    except Exception as e:
        logger.error(f"Failed to convert config to namespace: {str(e)}")
        raise

def main():
    try:
        # Change to script directory to handle relative paths
        with working_directory(os.path.dirname(os.path.abspath(__file__))):
            args, config = parse_args_and_config()
            
        logger.info(f"Writing logs to: {args.log_path}")
        logger.info(f"Experiment PID: {os.getpid()}")
        logger.info(f"Experiment comment: {args.comment}")

        # Initialize and run diffusion model
        runner = Diffusion(args, config)
        runner.sample()

        logger.info("Experiment completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())