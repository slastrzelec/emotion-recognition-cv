"""
Training script — meant to run on a free GPU (Google Colab or Kaggle),
not on the Streamlit deployment host. Downloads FER-2013 via kagglehub,
trains either the baseline CNN or the transfer-learning ResNet under the
same pipeline (augmentation, class-weighted loss, LR scheduling, early
stopping), and saves the best checkpoint.

Usage (Colab):

    !pip install -q kagglehub scikit-learn
    !python train.py --model resnet --epochs 40

Usage (Kaggle notebook): same, kagglehub download is a no-op there since
the dataset is already mounted — see `resolve_data_root()`.

Requires Kaggle credentials for the download (kagglehub prompts for a
Kaggle API token the first time; on Colab, upload kaggle.json or set
KAGGLE_USERNAME / KAGGLE_KEY as environment variables/Colab secrets —
never hardcoded in this file or committed to the repo).
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import EmotionFolderDataset, compute_class_weights, get_transform
from model import build_model


def resolve_data_root() -> str:
    """Finds (downloading if necessary) the FER-2013 root folder, which
    should contain `train/` and `test/` subfolders, each with one
    folder per emotion class.
    """
    # Kaggle notebooks already have competition/dataset data mounted
    # under /kaggle/input — prefer that over re-downloading.
    kaggle_input = "/kaggle/input/fer2013"
    if os.path.isdir(kaggle_input):
        return kaggle_input

    import kagglehub

    path = kagglehub.dataset_download("msambare/fer2013")
    return path


def build_dataloaders(model_name: str, data_root: str, batch_size: int):
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    train_ds = EmotionFolderDataset(train_dir, transform=get_transform(model_name, train=True))
    test_ds = EmotionFolderDataset(test_dir, transform=get_transform(model_name, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_ds, test_ds, train_loader, test_loader


def train_one_model(
    model_name: str,
    data_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    out_dir: str,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_name}] device: {device}")

    train_ds, test_ds, train_loader, test_loader = build_dataloaders(model_name, data_root, batch_size)
    print(f"[{model_name}] train samples: {len(train_ds)}  test samples: {len(test_ds)}")

    class_weights = compute_class_weights(train_ds).to(device)
    print(f"[{model_name}] class weights: {class_weights.tolist()}")

    model = build_model(model_name, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)

    best_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, f"{model_name}_best.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = running_loss / len(train_ds)
        train_acc = 100.0 * train_correct / len(train_ds)

        model.eval()
        test_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_correct += (outputs.argmax(1) == labels).sum().item()
        test_acc = 100.0 * test_correct / len(test_ds)

        scheduler.step(test_acc)
        elapsed = time.time() - t0
        print(
            f"[{model_name}] epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%  "
            f"test_acc={test_acc:.2f}%  ({elapsed:.0f}s)"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": best_state, "model_name": model_name, "accuracy": best_acc, "epoch": epoch},
                checkpoint_path,
            )
            print(f"[{model_name}] saved new best checkpoint: {best_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{model_name}] no improvement for {patience} epochs — stopping early.")
                break

    print(f"[{model_name}] done. best test accuracy: {best_acc:.2f}%  -> {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["cnn", "resnet", "both"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6, help="early-stopping patience, in epochs")
    parser.add_argument("--out-dir", default="checkpoints")
    args = parser.parse_args()

    data_root = resolve_data_root()
    print(f"Data root: {data_root}")

    models_to_train = ["cnn", "resnet"] if args.model == "both" else [args.model]
    for model_name in models_to_train:
        train_one_model(
            model_name=model_name,
            data_root=data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
