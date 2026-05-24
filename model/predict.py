from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from utils.gradcam import GradCAM


def load_model(path):
    ckpt = torch.load(path, map_location='cpu')
    num_classes = ckpt.get('num_classes', 2)
    class_names = ckpt.get('class_names', ['real', 'fake'])
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, class_names


def _build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_image(model, pil_img: Image.Image, img_size=224):
    transform = _build_transform(img_size)
    x = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).squeeze().tolist()
    return probs


def predict_image_with_cam(model, pil_img: Image.Image, img_size=224):
    transform = _build_transform(img_size)
    x = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).squeeze().tolist()
    target_idx = int(torch.argmax(torch.tensor(probs)))
    cam = GradCAM(model, model.layer4[-1])
    heatmap = cam(x, class_idx=target_idx)
    overlay = _overlay_heatmap(pil_img, heatmap)
    return probs, overlay


def _overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    heatmap = Image.fromarray(heatmap).resize(image.size)
    base = np.array(image).astype(np.float32)
    overlay = np.array(heatmap).astype(np.float32)
    combined = np.uint8(np.clip(base * (1 - alpha) + overlay * alpha, 0, 255))
    return Image.fromarray(combined)
