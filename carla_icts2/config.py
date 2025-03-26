"""Configuration module for setting up paths and logging."""

from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

ASSETS_DIR = PROJ_ROOT / "carla_icts2" / "assets"
OUTPUT_DIR = PROJ_ROOT / "output"
VIDEOS_DIR = OUTPUT_DIR / "videos"

# Make sure the paths exist
for path in [ASSETS_DIR]:
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        raise FileNotFoundError(f"Path does not exist: {path}")

# TZINFO
TZINFO = ZoneInfo("Europe/Berlin")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
