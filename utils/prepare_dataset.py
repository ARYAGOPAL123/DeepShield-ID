import argparse
import os
from pathlib import Path
from PIL import Image
import io

def prepare_images(src_dir, out_dir, size=(224, 224)):
    """Resize and convert images from source to output directory"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img_file in Path(src_dir).rglob('*'):
        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            try:
                img = Image.open(img_file).convert('RGB')
                img = img.resize(size)
                out_file = out_path / f'img_{count:05d}.jpg'
                img.save(out_file, quality=95)
                count += 1
                if count % 10 == 0:
                    print(f"Processed {count} images...")
            except Exception as e:
                print(f"Skipped {img_file}: {e}")
    
    print(f"Total images processed: {count}")
    return count

def main(src_dir, out_dir, size=224):
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} not found. Please download dataset first.")
        print("\nQuick steps:")
        print("1. Kaggle: pip install kaggle && kaggle datasets download -d <dataset-name>")
        print("2. FaceForensics: Visit https://github.com/ondyari/FaceForensics")
        print("3. LFW: Download from http://vis-www.cs.umass.edu/lfw/")
        return
    
    prepare_images(src_dir, out_dir, size=(size, size))
    print(f"✓ Processed dataset saved to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare downloaded datasets for training')
    parser.add_argument('--src', required=True, help='Source directory with raw images')
    parser.add_argument('--out', required=True, help='Output directory for processed images')
    parser.add_argument('--size', type=int, default=224, help='Output image size (will be size x size)')
    args = parser.parse_args()
    main(args.src, args.out, args.size)
