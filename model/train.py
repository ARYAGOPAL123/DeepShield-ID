import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.engine import create_dataloaders, build_model, get_device
from model.evaluate import compute_classification_metrics, save_metrics


def train(data_dir, epochs, out_path, batch_size=16, img_size=224, val_split=0.2, lr=1e-4, device=None):
    if device is None:
        device = get_device()
    train_loader, val_loader, class_names = create_dataloaders(data_dir, batch_size=batch_size, img_size=img_size, val_split=val_split)
    num_classes = len(class_names)
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    history = {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    print(f"Training {num_classes} classes on {device} with {len(train_loader.dataset)} train samples and {len(val_loader.dataset)} validation samples.")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += inputs.size(0)

        train_loss = epoch_loss / epoch_total
        train_accuracy = epoch_correct / epoch_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

        val_loss = val_loss / max(val_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        metrics = compute_classification_metrics(all_labels, all_preds, labels=class_names)

        history['epochs'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} | val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}")
        print(f"  Metrics: accuracy={metrics['accuracy']:.3f} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1_score']:.3f}")

    checkpoint = {
        'model_state': model.state_dict(),
        'num_classes': num_classes,
        'class_names': class_names,
        'image_size': img_size
    }
    Path(os.path.dirname(out_path) or '.').mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    print(f"Saved model checkpoint to {out_path}")

    metrics_path = Path(out_path).with_suffix('.metrics.json')
    model_report = {
        'training': {
            'data_dir': data_dir,
            'epochs': epochs,
            'batch_size': batch_size,
            'image_size': img_size,
            'device': device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'class_names': class_names,
        'history': history,
        'validation_metrics': metrics
    }
    save_metrics(model_report, metrics_path)
    print(f"Saved training report to {metrics_path}")
    return out_path, metrics_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the DeepShield-ID model')
    parser.add_argument('--data', required=True, help='Processed dataset directory (ImageFolder format)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--img-size', type=int, default=224, help='Image resize dimension')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split fraction')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--out', required=True, help='Output model checkpoint path (.pth)')
    args = parser.parse_args()
    train(args.data, args.epochs, args.out, batch_size=args.batch_size, img_size=args.img_size, val_split=args.val_split, lr=args.lr)
