import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger()

logger.info("Project started.")
logger.info("Dataset loaded successfully.")
logger.warning("This is a warning message.")
logger.error("This is a test error message.")

print("\nLogger test completed.")