import logging

# Configure logging
def setup_logging(log_file="pseudo_pipe_log.log", level=logging.INFO):
    """
    Setup logging configuration to write to both file and console.
    
    Parameters:
    - log_file: str - Name of the log file (saved in current directory)
    - level: logging level - Default is INFO
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)
    
    return logger