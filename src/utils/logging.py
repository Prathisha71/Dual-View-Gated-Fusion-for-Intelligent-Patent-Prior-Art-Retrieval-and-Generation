import logging
import os
import sys

def setup_logger(name: str = "PAFIR", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup structured logging for console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
