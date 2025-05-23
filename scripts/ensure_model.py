#!/usr/bin/env python3
# Script to ensure the PLaMo embedding model is downloaded

import os
import logging
import argparse
import sys

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from huggingface_hub import snapshot_download, HfFolder, HfApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_model_downloaded(model_path=None, force=False):
    """
    Ensure the PLaMo embedding model is downloaded
    
    Args:
        model_path: Path where the model should be stored
        force: Whether to force download even if the model already exists
        
    Returns:
        bool: Whether the model is available
    """
    model_path = model_path or config.EMBEDDING_MODEL_PATH
    
    # Fix potential Docker path reference (/app)
    if model_path.startswith('/app/'):
        # We're running locally, not in Docker, so replace /app with the actual base directory
        relative_path = model_path[5:]  # Remove /app/ prefix
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_path)
        logger.info(f"Detected Docker path, converted to local path: {model_path}")
    
    # Check if model already exists and contains essential files
    if not force and os.path.exists(model_path) and os.listdir(model_path):
        files = os.listdir(model_path)
        # Check if model has required files
        if "config.json" in files and any(f in files for f in ["model.safetensors", "pytorch_model.bin"]):
            logger.info(f"Model already exists at {model_path}")
            return True
        else:
            logger.warning(f"Model directory exists but may be incomplete: {files}")
    
    # Create the directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    try:
        logger.info(f"Downloading PLaMo-Embedding-1B model to {model_path}")
        
        # Try direct download with explicit parameters
        snapshot_download(
            repo_id="pfnet/plamo-embedding-1b", 
            local_dir=model_path,
            local_dir_use_symlinks=False,
            revision="main",
            ignore_patterns=["*.h5", "*.ot", "*.msgpack"],  # Skip large unnecessary files
            max_workers=2  # Use fewer workers to avoid connection issues
        )
        
        # Verify the download
        files = os.listdir(model_path)
        logger.info(f"Downloaded files: {files}")
        
        # Verify essential files
        missing_files = []
        for essential_file in ["config.json", "tokenizer.model"]:
            if essential_file not in files:
                missing_files.append(essential_file)
        
        # Check for model weight files
        has_weights = any(f in files for f in ["model.safetensors", "pytorch_model.bin"])
        if not has_weights:
            missing_files.append("model weights (model.safetensors or pytorch_model.bin)")
        
        if missing_files:
            logger.error(f"Download incomplete. Missing: {missing_files}")
            return False
            
        logger.info("Model downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PLaMo embedding model")
    parser.add_argument("--model_path", type=str, default=config.EMBEDDING_MODEL_PATH,
                      help=f"Where to save the model (default: {config.EMBEDDING_MODEL_PATH})")
    parser.add_argument("--force", action="store_true", 
                      help="Force download even if model already exists")
    
    args = parser.parse_args()
    
    success = ensure_model_downloaded(args.model_path, args.force)
    if success:
        print("Model is ready to use")
        sys.exit(0)
    else:
        print("Failed to download model")
        sys.exit(1)
