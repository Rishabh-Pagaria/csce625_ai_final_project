#!/usr/bin/env python3
"""
Download Gemma-2-2b model from HuggingFace Hub to local storage.
This script downloads the model files to hf_models/gemma-2-2b/
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    # Define model and local directory
    model_id = "google/gemma-2-2b-it"
    local_dir = "hf_models/gemma-2-2b-it"
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    print(f"[download_gemma] Target directory: {local_dir}")
    
    # Check if model already exists
    config_file = os.path.join(local_dir, "config.json")
    model_file = os.path.join(local_dir, "model.safetensors")
    
    if os.path.exists(config_file) and os.path.exists(model_file):
        print(f"[download_gemma] Model already exists in {local_dir}")
        print(f"[download_gemma] Delete the folder to re-download")
        return
    
    print(f"[download_gemma] Downloading {model_id}...")
    print(f"[download_gemma] This may take several minutes (~5GB)")
    print(f"[download_gemma] Note: You need to accept the license at https://huggingface.co/google/gemma-2-2b-it")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Don't use symlinks on Windows
            resume_download=True,  # Resume if interrupted
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip unnecessary files
        )
        
        print(f"[download_gemma] ✓ Download complete!")
        print(f"[download_gemma] Model saved to: {os.path.abspath(local_dir)}")
        
        # List downloaded files
        print(f"\n[download_gemma] Downloaded files:")
        for file in sorted(os.listdir(local_dir)):
            file_path = os.path.join(local_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")
                
    except Exception as e:
        print(f"\n[download_gemma] ✗ Error downloading model: {e}")
        print(f"[download_gemma] Make sure you:")
        print(f"  1. Have accepted the license at https://huggingface.co/google/gemma-2-2b")
        print(f"  2. Are logged in: huggingface-cli login")
        print(f"  3. Have sufficient disk space (~5GB)")
        raise

if __name__ == "__main__":
    main()
