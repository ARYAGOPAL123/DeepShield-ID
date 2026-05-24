import json
import os
import random
import subprocess
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import pandas as pd
from model.predict import load_model, predict_image, predict_image_with_cam

st.set_page_config(page_title="DeepShield-ID", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "weights.pth"
METRICS_PATH = ROOT / "model" / "weights.metrics.json"
DATA_PATH = ROOT / "data" / "processed"
FEEDBACK_CSV = ROOT / "data" / "feedback.csv"


def load_model_data():
    if MODEL_PATH.exists():
        try:
            model, class_names = load_model(str(MODEL_PATH))
            return model, class_names
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    return None, None


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def dataset_counts():
    real_dir = DATA_PATH / 'real'
    fake_dir = DATA_PATH / 'fake'
    real_count = len(list(real_dir.glob('*.jpg'))) + len(list(real_dir.glob('*.png')))
    fake_count = len(list(fake_dir.glob('*.jpg'))) + len(list(fake_dir.glob('*.png')))
    return real_count, fake_count


def sample_image_path():
    valid_paths = []
    for label in ['real', 'fake']:
        path = DATA_PATH / label
        if path.exists():
            valid_paths.extend(list(path.glob('*.jpg')) + list(path.glob('*.png')))
    return random.choice(valid_paths) if valid_paths else None


def run_command(command):
    return subprocess.run([sys.executable] + command, check=False)


st.title("DeepShield-ID — ID Card Deepfake Detection")
st.write("A real-world ID card deepfake detection demo with dataset controls, training, metrics, and explainability.")

model, class_names = load_model_data()
metrics = load_metrics()
real_count, fake_count = dataset_counts()

with st.sidebar:
    st.header("Project Controls")
    st.markdown("Use these buttons to build a real-world demo data pipeline.")
    if st.button("Generate synthetic dataset"):
        run_command(["data/synthesize_idcards.py", "--out", "data/processed", "--count", "120", "--fake-ratio", "0.5"])
        st.success("Synthetic ID dataset created in data/processed.")
    if st.button("Train / Re-train model"):
        run_command(["model/train.py", "--data", "data/processed", "--epochs", "5", "--batch-size", "16", "--out", str(MODEL_PATH)])
        model, class_names = load_model_data()
        metrics = load_metrics()
        st.success("Model training finished. Reload the page if needed.")
    st.markdown("---")
    st.write("Dataset status")
    st.write(f"Real images: **{real_count}**")
    st.write(f"Fake images: **{fake_count}**")
    st.markdown("---")
    st.write("If you have a real dataset, download and prepare it using the `utils` scripts.")
    st.write("For production: use FaceForensics++, DFDC, or proprietary ID scans.")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Input and Prediction")
    uploaded = st.file_uploader("Upload an ID image", type=["png", "jpg", "jpeg"])
    use_sample = st.button("Use a random sample image")
    input_image = None
    if uploaded:
        input_image = Image.open(uploaded).convert('RGB')
    elif use_sample:
        sample_path = sample_image_path()
        if sample_path:
            input_image = Image.open(sample_path).convert('RGB')
            st.info(f"Using sample image: {sample_path.name}")
        else:
            st.warning("No sample image is available in data/processed.")

    if input_image is not None:
        st.image(input_image, caption="Input ID image", use_column_width=True)
        if model is None:
            st.warning("No model found. Train the model first using the sidebar.")
        else:
            with st.spinner("Running inference..."):
                probs = predict_image(model, input_image)
                # Build a DataFrame of probabilities
                prob_df = pd.DataFrame({'class': class_names, 'prob': probs})
                prob_df['percent'] = (prob_df['prob'] * 100).round(2).astype(str) + '%'
                # Sort for display
                prob_df = prob_df.sort_values('prob', ascending=False).reset_index(drop=True)
                label_index = int(prob_df.index[0])
                label = prob_df.loc[0, 'class']
                confidence = float(prob_df.loc[0, 'prob'])

                st.metric(label="Predicted class", value=label.upper(), delta=f"{confidence*100:.1f}% confidence")
                st.write("### Class probabilities")
                st.table(prob_df[['class', 'percent']])
                st.bar_chart(data=prob_df.set_index('class')['prob'])

                if st.checkbox("Show GradCAM explanation", value=True):
                    _, overlay = predict_image_with_cam(model, input_image)
                    st.image(overlay, caption="GradCAM explanation", use_column_width=True)

                # Feedback / correction
                st.write("---")
                st.write("Was this prediction correct?")
                true_label = st.selectbox("If incorrect, select the true label (or leave as-is)", options=["--",] + class_names)
                if st.button("Save feedback"):
                    FEEDBACK_DIR = ROOT / 'data' / 'feedback_images'
                    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
                    import time
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    # save uploaded image copy or sample
                    img_name = f"img_{ts}.jpg"
                    save_path = FEEDBACK_DIR / img_name
                    input_image.save(save_path)
                    # append to CSV
                    import pandas as _pd
                    row = {
                        'image': str(save_path.relative_to(ROOT)),
                        'predicted': label,
                        'confidence': confidence,
                        'true_label': (None if true_label == '--' else true_label),
                        'timestamp': ts
                    }
                    if not FEEDBACK_CSV.exists():
                        _pd.DataFrame([row]).to_csv(FEEDBACK_CSV, index=False)
                    else:
                        _pd.DataFrame([row]).to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)
                    st.success('Feedback saved — you can retrain the model using feedback data.')

with col2:
    st.subheader("Model and Dataset Report")
    if model is None:
        st.info("No trained model available. Click 'Train / Re-train model' in the sidebar.")
    else:
        if metrics:
            st.metric("Validation Accuracy", f"{metrics['validation_metrics']['accuracy']*100:.1f}%")
            st.metric("Validation F1", f"{metrics['validation_metrics']['f1_score']*100:.1f}%")
            st.write("### Validation Metrics")
            st.json({
                'precision': metrics['validation_metrics']['precision'],
                'recall': metrics['validation_metrics']['recall'],
                'confusion_matrix': metrics['validation_metrics']['confusion_matrix']
            })
        else:
            st.info("Model weights found, but no metrics report is available.")
    st.write("### Quick facts")
    st.write("- Model: ResNet-18 transfer learning")
    st.write("- Supports: real vs fake/manipulated ID cards")
    st.write("- Input: JPG/PNG ID images")
    st.write("- Built for client demo and production extension")

st.write("---")
st.write("### Included project features")
st.write(
    "- Dataset generation and synthetic ID card creation\n"
    "- Training and validation with performance report\n"
    "- Live inference and explainability overlay\n"
    "- Real dataset support via Kaggle and FaceForensics integration\n"
    "- Docker-ready deployment support"
)
