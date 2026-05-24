# DeepShield-ID: ID Card Deepfake Detection

DeepShield-ID is a professional-grade demo project showing a full pipeline for detecting fake or manipulated ID card images.
It includes dataset generation, preprocessing, training with validation, inference, explainability, and a polished Streamlit UI.

## What this project includes

- Synthetic ID card generation and dataset creation
- Train / re-train model from UI or CLI
- Validation metrics with accuracy, precision, recall, and F1 score
- GradCAM explanation overlay for model transparency
- Real dataset support via Kaggle/FaceForensics/DFDC
- Docker-ready deployment for production demos

## Quick Start (Windows)

1. Create a virtual environment and install dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Generate synthetic sample data and train a model:

```powershell
python data/synthesize_idcards.py --out data/processed --count 120 --fake-ratio 0.5
python model/train.py --data data/processed --epochs 5 --batch-size 16 --out model/weights.pth
```

3. Run the Streamlit UI:

```powershell
py -3 -m streamlit run app/streamlit_app.py
```

## Real Dataset Workflow

For a real-world client-ready demo, use `QUICKSTART_DATASETS.md` to:
- download authentic deepfake datasets
- prepare images with `utils/prepare_dataset.py`
- retrain the model with real/ fake ID samples
- review validation performance in the UI

## Useful commands

- Generate synthetic data: `python data/synthesize_idcards.py --out data/processed --count 120 --fake-ratio 0.5`
- Train the model: `python model/train.py --data data/processed --epochs 5 --out model/weights.pth`
- Predict one image from Python: `python model/predict.py`
- Build Docker container: `docker build -t deepshield-id .`
- Run with Docker: `docker-compose up --build`

## Project structure

- `app/streamlit_app.py` — main Streamlit UI
- `model/train.py` — training pipeline with validation and metrics export
- `model/predict.py` — inference and GradCAM explanation
- `model/engine.py` — data loaders, model building, training utilities
- `model/evaluate.py` — metrics and evaluation report support
- `data/synthesize_idcards.py` — synthetic ID card generator
- `data/generate_dataset.py` — demo image augmentation script
- `utils/prepare_dataset.py` — preparation for downloaded datasets
- `utils/download_kaggle_dataset.py` — Kaggle dataset downloader
- `Dockerfile` / `docker-compose.yml` — containerized deployment

## Notes

- This solution is designed as a real-world demo with expandable architecture.
- Replace synthetic data with real deepfake ID datasets for production-level results.
- Use the Streamlit UI to interactively show predictions and explainability.

For detailed dataset instructions, see `QUICKSTART_DATASETS.md` and `DATASET_GUIDE.md`.
