# Quick Start: Download Real Dataset for DeepShield-ID

## ⚡ Fastest Option (Recommended): Kaggle Dataset

### Step 1: Install Kaggle CLI
```bash
pip install kaggle
```

### Step 2: Configure Kaggle API
1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section and click "Create New Token"
3. This downloads `kaggle.json`
4. On Windows, move it to: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
5. (Mac/Linux: `~/.kaggle/kaggle.json`)

### Step 3: Download Dataset
```bash
cd c:\Users\aryac\OneDrive\Desktop\DeepShield-ID

# Show available datasets
python utils/download_kaggle_dataset.py --list

# Download a dataset (pick one):
python utils/download_kaggle_dataset.py --dataset deepfake-detection
# OR
python utils/download_kaggle_dataset.py --dataset face-detection
```

### Step 4: Prepare the Dataset
```bash
# Resize and organize images
python utils/prepare_dataset.py --src data/raw/deepfake-detection --out data/processed
```

### Step 5: Retrain Model
```bash
python model/train.py --data data/processed --epochs 10 --out model/weights.pth
```

### Step 6: Test in UI
```bash
streamlit run app/streamlit_app.py
```

---

## 🎯 Alternative: Manual Download

### Option A: Google Drive Download
1. Search Kaggle for "deepfake dataset" → find dataset → click "Download"
2. Extract to `data/raw/` folder
3. Run: `python utils/prepare_dataset.py --src data/raw --out data/processed`

### Option B: Use LFW (Faces) Dataset
Only provides "real" faces, but good for testing:
```bash
# Download LFW
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf lfw.tgz

# Organize
python utils/prepare_dataset.py --src lfw --out data/processed/real
```

---

## 📊 Folder Structure After Download

After running the above commands, your structure should look like:

```
DeepShield-ID/
├── data/
│   ├── raw/                    # Downloaded raw dataset
│   │   └── deepfake-detection/
│   │       ├── real/
│   │       └── fake/
│   └── processed/              # Processed for training
│       ├── real/               # 224x224 JPEG images
│       │   ├── img_00000.jpg
│       │   ├── img_00001.jpg
│       │   └── ...
│       └── fake/
│           ├── img_00000.jpg
│           ├── img_00001.jpg
│           └── ...
├── model/
│   └── weights.pth             # Retrained model
└── app/
    └── streamlit_app.py
```

---

## ✅ Dataset Size Reference

| Dataset | Size | Training Time | Accuracy |
|---------|------|---------------|----------|
| Synthetic (current) | 30 samples | ~1 min | ~65% |
| Kaggle (face-detection) | 1000+ samples | ~5-10 min | ~85% |
| FaceForensics++ | 1000s of videos | Hours | ~95% |

---

## 🔧 Troubleshooting

### "kaggle: command not found"
```bash
pip install kaggle
# Restart terminal
```

### "Kaggle API not authenticated"
```bash
# Verify kaggle.json exists:
ls ~/.kaggle/kaggle.json  # Mac/Linux
dir C:\Users\<USER>\.kaggle\kaggle.json  # Windows
```

### "Permission denied" (Mac/Linux)
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Images not found after download
Check that images are in subfolders:
```bash
# Windows
dir data\raw\deepfake-detection

# Should show: real/ and fake/ folders
```

---

## 📝 Next Steps After Dataset

1. **Retrain model**: `python model/train.py --data data/processed --epochs 10`
2. **Evaluate**: Add validation metrics in `model/train.py`
3. **Deploy**: Upload model to production server
4. **Client demo**: Show real results with actual dataset in UI

---

## ⚠️ Important Notes

- **Min 500 images** per class recommended for good accuracy
- **Class balance** important: roughly equal real/fake samples
- **Preprocessing** crucial: all images must be 224x224 JPG/PNG
- **Privacy**: Ensure you have rights to use dataset (academic/commercial)
- **GPU**: Training on GPU ~10x faster. Enable with `device='cuda'` in train.py

---

For more dataset options, see `DATASET_GUIDE.md`.
