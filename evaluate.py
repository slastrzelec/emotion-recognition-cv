"""
Measures real accuracy — overall and per-class — plus a confusion matrix,
for a trained checkpoint. Numbers quoted in the README come from running
this script, not from numbers remembered/asserted from the training run.

Library use (no CLI/file I/O):

    from evaluate import evaluate
    result = evaluate(model, dataloader, device)
    print(result.overall_accuracy, result.per_class)

CLI use:

    python evaluate.py --checkpoint checkpoints/resnet_best.pt \
        --model resnet --data-dir data/test
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from data import EmotionFolderDataset, get_transform
from model import EMOTION_LABELS, build_model


@dataclass
class EvalResult:
    overall_accuracy: float
    per_class: dict = field(default_factory=dict)  # label -> {precision, recall, f1, support}
    confusion: list = field(default_factory=list)  # 7x7, rows=true, cols=predicted


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    model.to(device)

    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    overall_accuracy = correct / len(all_labels) if all_labels else 0.0

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(EMOTION_LABELS))), zero_division=0
    )
    per_class = {
        EMOTION_LABELS[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(EMOTION_LABELS))
    }

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(EMOTION_LABELS))))

    return EvalResult(
        overall_accuracy=overall_accuracy,
        per_class=per_class,
        confusion=cm.tolist(),
    )


def _print_report(result: EvalResult) -> None:
    print(f"Overall accuracy: {result.overall_accuracy * 100:.2f}%\n")
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for label, m in result.per_class.items():
        print(
            f"{label:<10} {m['precision']:>10.3f} {m['recall']:>10.3f} "
            f"{m['f1']:>10.3f} {m['support']:>10d}"
        )
    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "        " + " ".join(f"{l[:4]:>5}" for l in EMOTION_LABELS)
    print(header)
    for label, row in zip(EMOTION_LABELS, result.confusion):
        print(f"{label:<8}" + " ".join(f"{v:>5d}" for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint from train.py")
    parser.add_argument("--model", required=True, choices=["cnn", "resnet"])
    parser.add_argument("--data-dir", required=True, help="Folder-per-class test set root")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    dataset = EmotionFolderDataset(args.data_dir, transform=get_transform(args.model, train=False))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    result = evaluate(model, dataloader, device)
    _print_report(result)


if __name__ == "__main__":
    main()
