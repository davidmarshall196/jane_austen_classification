import logging
from datetime import datetime

# Define the name for the log file
log_file_name = datetime.now().strftime("../logs/jane_austen_%Y_%m_%d.log")

# Create a logger
logger = logging.getLogger("JaneAustenLogger")
logger.setLevel(logging.INFO)  # Set to your preferred logging level

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a file handler
fh = logging.FileHandler(log_file_name)
fh.setLevel(logging.INFO)

# Create a formatter and set it for the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)
logger.addHandler(fh)


def get_logger():
    return logger
