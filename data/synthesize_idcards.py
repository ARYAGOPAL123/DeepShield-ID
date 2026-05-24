import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def random_text(length):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choice(letters) for _ in range(length))


def draw_id_card(base_color=(245, 245, 240), width=500, height=320, is_fake=False):
    img = Image.new('RGB', (width, height), base_color)
    draw = ImageDraw.Draw(img)

    # header block
    draw.rectangle([0, 0, width, 70], fill=(18, 58, 99))
    draw.text((20, 18), 'CITIZEN ID CARD', fill='white')
    draw.rectangle([20, 90, 180, 170], outline=(80, 80, 80), width=2)
    draw.text((28, 98), 'PHOTO', fill=(80, 80, 80))

    # details section
    draw.text((210, 90), 'Name:', fill='black')
    draw.text((210, 120), 'ID No:', fill='black')
    draw.text((210, 150), 'DOB:', fill='black')
    draw.text((210, 180), 'Nationality:', fill='black')

    name = random.choice(['ALEXANDER SMITH', 'PRIYA KUMAR', 'ZHANG WEI', 'AISHWARYA DAS'])
    id_no = 'ID' + random_text(8)
    dob = random.choice(['12-07-1993', '22-11-1987', '05-03-1995', '18-09-1990'])
    nation = random.choice(['INDIA', 'USA', 'CANADA', 'AUSTRALIA'])

    draw.text((280, 90), name, fill='black')
    draw.text((280, 120), id_no, fill='black')
    draw.text((280, 150), dob, fill='black')
    draw.text((280, 180), nation, fill='black')

    draw.rectangle([20, 220, width-20, 260], fill=(220, 220, 220))
    draw.text((28, 228), 'Signature', fill=(100, 100, 100))

    # fake artifacts
    if is_fake:
        for _ in range(4):
            x1 = random.randint(200, width-40)
            y1 = random.randint(80, height-40)
            x2 = x1 + random.randint(20, 70)
            y2 = y1 + random.randint(10, 30)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        draw.line((210, 100, 410, 100), fill=(255, 0, 0), width=2)

    return img


def generate_dataset(output_dir, count=100, fake_ratio=0.5):
    output_dir = Path(output_dir)
    real_dir = output_dir / 'real'
    fake_dir = output_dir / 'fake'
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    real_count = int(count * (1 - fake_ratio))
    fake_count = count - real_count
    
    for i in range(real_count):
        img = draw_id_card(is_fake=False)
        img.save(real_dir / f'real_{i:04d}.jpg')
    for i in range(fake_count):
        img = draw_id_card(is_fake=True)
        img.save(fake_dir / f'fake_{i:04d}.jpg')
    print(f"Generated {real_count} real and {fake_count} fake synthetic ID cards in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic ID card dataset')
    parser.add_argument('--out', required=True, help='Output dataset directory')
    parser.add_argument('--count', type=int, default=120, help='Total number of cards')
    parser.add_argument('--fake-ratio', type=float, default=0.5, help='Share of fake cards')
    args = parser.parse_args()
    generate_dataset(args.out, args.count, args.fake_ratio)
