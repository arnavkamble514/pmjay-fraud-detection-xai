"""
Component C — document tamper detection CNN.

Fine-tunes a pretrained ResNet-18 on CASIA 2.0 to classify images as
authentic vs tampered. Mirrors the approach in Nataraj et al. (2021),
paper #8 in your literature survey, which used a ResNet backbone for
the same task.

Saves tamper_cnn.pth, which Stage 2 loads to score generated documents.

Run:  python train.py
"""

import json
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

import io
import numpy as np
from PIL import Image


def ela_loader(path, quality=90, scale=15):
    """Error Level Analysis: re-compress the image and amplify the
    difference. Tampered regions were compressed a second/different
    time and show up as a brighter residual than untouched areas."""
    original = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    orig_arr = np.asarray(original, dtype=np.int16)
    resaved_arr = np.asarray(resaved, dtype=np.int16)
    diff = np.abs(orig_arr - resaved_arr) * scale
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return Image.fromarray(diff)


DATASET = "divg07/casia-20-image-tampering-detection-dataset"
HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
WEIGHTS = HERE / "tamper_cnn.pth"
METRICS = HERE / "training_metrics.json"

EPOCHS = 12
BATCH_SIZE = 32          # drop to 16 if you hit a CUDA OOM error
LR = 1e-4
IMG_SIZE = 224
VAL_SPLIT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_dataset():
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.jpg")):
        print(f"Dataset already present at {DATA_DIR}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET,
         "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    print("Download complete.")


def find_class_root(data_dir):
    """
    CASIA 2.0 unzips with authentic ('Au') and tampered ('Tp') folders,
    sometimes nested a level or two deep. Find the parent holding both.
    """
    for path in [data_dir] + [p for p in data_dir.rglob("*") if p.is_dir()]:
        subdirs = {d.name.lower() for d in path.iterdir() if d.is_dir()}
        has_auth = any(s.startswith("au") for s in subdirs)
        has_tamp = any(s.startswith("tp") for s in subdirs)
        if has_auth and has_tamp:
            return path
    raise RuntimeError(
        f"Could not find Au/Tp folders under {data_dir}. "
        "Check the extracted structure and pass the path manually."
    )


def build_loaders(root):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(root, transform=train_tf, loader=ela_loader)
    print(f"Classes found: {full.classes}  (total {len(full)} images)")

    n_val = int(len(full) * VAL_SPLIT)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_ds.dataset = datasets.ImageFolder(root, transform=val_tf, loader=ela_loader)

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        full.classes,
    )


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    probs, labels, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.tolist())

    acc = sum(int(a == b) for a, b in zip(preds, labels)) / len(labels)
    auc = roc_auc_score(labels, probs)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "auc": auc, "precision": prec,
            "recall": rec, "f1": f1}


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    download_dataset()
    root = find_class_root(DATA_DIR)
    print(f"Using class root: {root}")

    train_loader, val_loader, classes = build_loaders(root)
    model = build_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history, best_auc, patience_left = [], 0.0, 4

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)

        metrics = evaluate(model, val_loader)
        scheduler.step(metrics["auc"])

        row = {
            "epoch": epoch,
            "train_loss": running / seen,
            "seconds": round(time.time() - t0, 1),
            **{k: round(v, 4) for k, v in metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:2d} | loss {row['train_loss']:.4f} | "
            f"acc {row['accuracy']:.4f} | auc {row['auc']:.4f} | "
            f"f1 {row['f1']:.4f} | {row['seconds']}s"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            patience_left = 4
            torch.save(
                {"state_dict": model.state_dict(),
                 "classes": classes,
                 "img_size": IMG_SIZE,
                 "val_metrics": metrics},
                WEIGHTS,
            )
            print(f"  saved new best -> {WEIGHTS.name}")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  no AUC improvement in 4 epochs, stopping early.")
                break

    METRICS.write_text(json.dumps(history, indent=2))
    print(f"\nBest validation AUC: {best_auc:.4f}")
    print(f"Weights : {WEIGHTS}")
    print(f"Metrics : {METRICS}")
    print("\nYour synopsis targets >92% tamper-detection accuracy — "
          "compare against the best epoch above.")


if __name__ == "__main__":
    main()
