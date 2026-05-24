# Dataset Download Guide for DeepShield-ID

## Option 1: Public Deepfake Datasets (Recommended for Demo)

### FaceForensics++ Dataset
- **Website**: https://github.com/ondyari/FaceForensics
- **Size**: ~1.8 TB (full), ~370 GB (compressed)
- **Description**: Large-scale dataset with real and deepfake videos
- **Download**:
  1. Visit https://github.com/ondyari/FaceForensics
  2. Request access (free academic use)
  3. Download the face extraction tool
  4. Extract key frames from downloaded videos

### DFDC (Deepfake Detection Challenge)
- **Website**: https://deepfakedetectionchallenge.ai/
- **Size**: ~500 GB
- **Description**: High-quality deepfakes and real videos
- **Note**: May require sign-up and agreement

---

## Option 2: Lightweight Alternative for Quick Testing

### Kaggle Deepfake Detection Dataset
- **URL**: https://www.kaggle.com/datasets/darubagus/deepfake-detection-dataset
- **Size**: ~2-5 GB
- **How to download**:
  1. Install kaggle CLI: `pip install kaggle`
  2. Set up API key from https://www.kaggle.com/settings/account
  3. Run:
     ```bash
     kaggle datasets download -d darubagus/deepfake-detection-dataset
     unzip deepfake-detection-dataset.zip
     ```

### LFW (Labeled Faces in the Wild)
- **URL**: http://vis-www.cs.umass.edu/lfw/
- **Size**: ~200 MB
- **Description**: Real face dataset (all "real" for training)

---

## Option 3: Synthetic ID Card Dataset (Current Project Default)

The current project generates synthetic data automatically. To enhance it:

```bash
# Generate more augmented samples
python data/generate_dataset.py --out data/processed --n-augment 500
```

---

## How to Integrate Downloaded Dataset

### Step 1: Organize dataset into the required folder structure:
```
data/processed/
├── real/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── fake/
    ├── deepfake_1.jpg
    ├── deepfake_2.jpg
    └── ...
```

### Step 2: Ensure images are in JPG/PNG format (224x224 minimum):

```python
# Run this script to resize and convert
import os
from PIL import Image
from pathlib import Path

def prepare_images(src_dir, out_dir, size=(224, 224)):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for idx, img_file in enumerate(Path(src_dir).glob('*')):
        try:
            img = Image.open(img_file).convert('RGB')
            img = img.resize(size)
            img.save(out_path / f'img_{idx}.jpg')
            print(f"Processed {img_file}")
        except Exception as e:
            print(f"Error with {img_file}: {e}")

# Example usage:
prepare_images('data/raw_real_ids', 'data/processed/real')
prepare_images('data/raw_fake_ids', 'data/processed/fake')
```

### Step 3: Re-train the model with new data:

```bash
python model/train.py --data data/processed --epochs 10 --out model/weights.pth
```

---

## Option 4: Quick Start with Sample Images

Use this Python script to download ~50 sample face images from public sources:

```bash
python utils/download_sample_dataset.py --out data/processed --count 50
```

---

## Recommended Workflow for Client Presentation

1. **Start with synthetic data** (current setup) → Show working UI
2. **Integrate Kaggle dataset** (lightweight) → Show improved accuracy
3. **Add real ID cards** (if you have them) → Final production-grade demo

---

## Notes for Production

- For a **production system**, use FaceForensics++ or proprietary ID scanning databases
- Always preprocess: resize to 224x224, normalize, check file integrity
- Split data: 70% train, 15% validation, 15% test
- Balance classes: roughly equal real/fake samples per batch
- Use data augmentation (rotation, blur, noise) for robustness
