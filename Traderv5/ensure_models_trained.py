#!/usr/bin/env python3
"""Ensure models are trained before running live trading."""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from Traderv5.model.predictor import ModelPredictor

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_models():
    """Check if models are available and train if needed."""
    try:
        # Try to load existing models
        predictor = ModelPredictor.load_default()
        logger.info("✅ Trained models found and loaded successfully")
        return True
        
    except FileNotFoundError:
        logger.warning("❌ No trained models found")
        logger.info("Starting model training...")
        
        try:
            from Traderv5.model.training import main as train_models
            train_models()
            
            # Try to load again
            predictor = ModelPredictor.load_default()
            logger.info("✅ Models trained and loaded successfully")
            return True
            
        except Exception as exc:
            logger.error(f"❌ Failed to train models: {exc}")
            logger.exception("Training error")
            return False
            
    except Exception as exc:
        logger.error(f"❌ Error loading models: {exc}")
        logger.exception("Model loading error")
        return False


def main():
    """Main function to ensure models are ready."""
    logger.info("Checking TraderV5 model status...")
    
    if check_models():
        logger.info("🎉 Models are ready for live trading!")
        return 0
    else:
        logger.error("💥 Models are not ready. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())
