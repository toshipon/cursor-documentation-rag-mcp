#!/usr/bin/env python3
# Script to download the PLaMo embedding model

import os
import logging
import argparse
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(model_path):
    """Download the PLaMo embedding model from HuggingFace Hub"""
    try:
        logger.info(f"Downloading PLaMo-Embedding-1B model to {model_path}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Use snapshot_download with appropriate parameters
        snapshot_download(
            repo_id="pfnet/plamo-embedding-1b", 
            local_dir=model_path,
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
            revision="main",  # Use the main branch
        )
        logger.info("Model downloaded successfully")
        
        # Verify the downloaded files
        model_files = os.listdir(model_path)
        logger.info(f"Downloaded files: {model_files}")
        
        # Verify if essential files are present
        essential_files = ["config.json", "tokenizer.model", "tokenizer_config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        
        missing_essential = [f for f in essential_files if f not in model_files]
        if missing_essential:
            logger.warning(f"Some essential files are missing: {missing_essential}")
            
        # Check if at least one model file exists
        has_model_file = any(f in model_files for f in ["pytorch_model.bin", "model.safetensors"])
        if not has_model_file:
            logger.error("No model weight file found. This will cause runtime errors.")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PLaMo embedding model")
    parser.add_argument("--model-path", type=str, default="/app/models/plamo-embedding-1b",
                        help="Path to download the model to")
    args = parser.parse_args()
    
    success = download_model(args.model_path)
    if not success:
        exit(1)
