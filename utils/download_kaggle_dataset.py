#!/usr/bin/env python3
"""
Download and prepare deepfake datasets from Kaggle
Requires: pip install kaggle
"""
import os
import subprocess
import argparse
from pathlib import Path

# Popular deepfake detection datasets on Kaggle
DATASETS = {
    'deepfake-detection': 'darubagus/deepfake-detection-dataset',
    'dfdc': 'c0d3kl0wn/deepfake-detection-challenge-dataset',
    'face-detection': 'ciplab/real-and-fake-face-detection',
}

def setup_kaggle():
    """Ensure Kaggle API is configured"""
    kaggle_dir = Path.home() / '.kaggle'
    if not kaggle_dir.exists():
        print("⚠️  Kaggle API not set up.")
        print("Steps to configure:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New Token' to download kaggle.json")
        print(f"3. Place kaggle.json at: {kaggle_dir / 'kaggle.json'}")
        return False
    return True

def download_dataset(dataset_key, output_dir='data/raw'):
    """Download a dataset from Kaggle"""
    if dataset_key not in DATASETS:
        print(f"Available datasets: {list(DATASETS.keys())}")
        return False
    
    dataset_name = DATASETS[dataset_key]
    output_path = Path(output_dir) / dataset_key
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading {dataset_key} from Kaggle...")
    print(f"   Dataset: {dataset_name}")
    print(f"   Output: {output_path}")
    
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(output_path), '--unzip']
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Downloaded to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        return False

def list_datasets():
    """Show available datasets"""
    print("Available Kaggle Datasets for Deepfake Detection:\n")
    for key, dataset in DATASETS.items():
        print(f"  • {key:30} → {dataset}")

def main():
    parser = argparse.ArgumentParser(description='Download deepfake datasets from Kaggle')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--dataset', type=str, help='Dataset key to download')
    parser.add_argument('--out', type=str, default='data/raw', help='Output directory')
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if not setup_kaggle():
        print("\nTo download datasets, configure Kaggle API first.")
        return
    
    if args.dataset:
        download_dataset(args.dataset, args.out)
    else:
        print("Specify a dataset with --dataset <name> or use --list to see options")

if __name__ == '__main__':
    main()
