import io
from pathlib import Path
from PIL import Image


def load_image_from_bytes(data):
    return Image.open(io.BytesIO(data)).convert('RGB')


def load_image_from_path(path):
    return Image.open(Path(path)).convert('RGB')


def ensure_image_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_image(image, path, quality=90):
    image.save(path, format='JPEG', quality=quality)
