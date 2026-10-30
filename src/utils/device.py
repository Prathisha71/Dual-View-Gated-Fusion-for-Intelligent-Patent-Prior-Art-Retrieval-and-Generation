import torch
import logging

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Detect available hardware accelerator (CUDA, MPS, CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device
