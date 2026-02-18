import logging
import os
from config import artifact_dir

os.makedirs(artifact_dir, exist_ok=True)

logger = logging.getLogger("GAD")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler(os.path.join(artifact_dir, "gad.log"))
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
