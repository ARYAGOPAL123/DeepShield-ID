import os
import argparse
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import io
import base64
import random
import requests
from pathlib import Path

SAMPLE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png",
]

FALLBACK_COLOR = (200, 200, 200)


def download_image(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception:
        return None


def synthesize_image(size=(400,250), color=(255,255,255)):
    img = Image.new('RGB', size, color)
    return img


def create_fake(img: Image.Image):
    # Apply manipulations to simulate a fake/manipulated ID
    out = img.copy()
    w,h = out.size
    # random crop and paste
    try:
        crop = out.crop((w//8, h//8, w*7//8, h*7//8)).resize((w,h))
        out.paste(crop, (0,0))
    except Exception:
        pass
    # blur and color jitter
    out = out.filter(ImageFilter.GaussianBlur(radius=2))
    enhancer = ImageEnhance.Color(out)
    out = enhancer.enhance(0.6)
    return out


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def main(out_dir, n_augment=50):
    out_dir = Path(out_dir)
    real_dir = out_dir / 'real'
    fake_dir = out_dir / 'fake'
    ensure_dir(real_dir)
    ensure_dir(fake_dir)

    # Try downloading sample images
    imgs = []
    for url in SAMPLE_URLS:
        img = download_image(url)
        if img:
            imgs.append(img)
    if not imgs:
        imgs.append(synthesize_image(color=(240,240,240)))

    # Save originals and augmented fakes
    i = 0
    for idx, img in enumerate(imgs):
        img = img.resize((600,380))
        img.save(real_dir / f'real_{idx}.jpg')
        for j in range(max(1, n_augment//len(imgs))):
            # create small augmentations for both classes
            fake = create_fake(img)
            real_aug = img.filter(ImageFilter.SHARPEN)
            fake.save(fake_dir / f'fake_{idx}_{j}.jpg')
            real_aug.save(real_dir / f'real_{idx}_{j}.jpg')
            i += 1
    print(f"Generated dataset at {out_dir} with ~{i} augmented images.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='Output directory for processed dataset')
    parser.add_argument('--n-augment', type=int, default=50, help='Approx number of augmentations to generate')
    args = parser.parse_args()
    main(args.out, args.n_augment)
