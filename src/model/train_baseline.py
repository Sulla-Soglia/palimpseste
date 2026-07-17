"""Baseline: multi-label period classifier over facade photos.

ResNet-18 pretrained on ImageNet, final layer replaced with 6 sigmoid
outputs (one per sovereignty period). Two training phases:
  1. head-only  — backbone frozen, only the new classifier learns
  2. fine-tune  — everything unfrozen at a 10x lower learning rate

Loss is BCEWithLogitsLoss with pos_weight so rare periods (german_occupation,
french_modern) are not drowned out by frequent ones.

Usage:
    python src/model/train_baseline.py \
        --dataset data/processed/dataset.jsonl --images data/raw/images \
        --epochs-head 3 --epochs-full 8 --out results/baseline
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PERIODS = [
    "germanic_hre",
    "french_ancien",
    "german_reich",
    "french_interwar",
    "german_occupation",
    "french_modern",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FacadeDataset(Dataset):
    def __init__(self, rows, image_dir, train):
        self.rows = rows
        self.image_dir = Path(image_dir)
        if train:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.open(self.image_dir / row["image"]).convert("RGB")
        y = torch.zeros(len(PERIODS))
        for p in row["periods"]:
            y[PERIODS.index(p)] = 1.0
        return self.tf(img), y


def load_split(dataset_path):
    splits = {"train": [], "val": [], "test": []}
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            splits[row["split"]].append(row)
    return splits


def auc_score(scores, targets):
    """ROC-AUC for one class, computed from scratch (rank statistic).

    AUC = probability that a randomly chosen positive example gets a
    higher score than a randomly chosen negative one. 0.5 = coin flip.
    Returns None if the class has no positives or no negatives.
    """
    pos = scores[targets > 0.5]
    neg = scores[targets < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
    ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()
    return ((greater + 0.5 * ties) / (len(pos) * len(neg))).item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_targets = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_scores.append(torch.sigmoid(logits).cpu())
        all_targets.append(y)
    scores = torch.cat(all_scores)
    targets = torch.cat(all_targets)
    per_class = {p: auc_score(scores[:, i], targets[:, i])
                 for i, p in enumerate(PERIODS)}
    valid = [v for v in per_class.values() if v is not None]
    macro = sum(valid) / len(valid) if valid else float("nan")
    return macro, per_class


def run_epochs(model, loader, val_loader, criterion, optimizer, epochs,
               device, tag, best):
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        macro, per_class = evaluate(model, val_loader, device)
        marker = ""
        if macro > best["auc"]:
            best.update(auc=macro, state={k: v.cpu().clone() for k, v
                                          in model.state_dict().items()})
            marker = "  *best*"
        detail = " ".join(f"{p.split('_')[1][:3]}:{v:.2f}"
                          for p, v in per_class.items() if v is not None)
        print(f"[{tag}] epoch {epoch+1}/{epochs}  "
              f"loss {total/len(loader.dataset):.3f}  "
              f"val macro-AUC {macro:.3f}  ({detail}){marker}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/dataset.jsonl")
    ap.add_argument("--images", default="data/raw/images")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-full", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/baseline")
    args = ap.parse_args()

    torch.manual_seed(67)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_split(args.dataset)
    print(f"device {device} | train {len(splits['train'])} "
          f"val {len(splits['val'])} test {len(splits['test'])}")

    loaders = {
        name: DataLoader(FacadeDataset(rows, args.images, train=(name == "train")),
                         batch_size=args.batch_size, shuffle=(name == "train"),
                         num_workers=0)
        for name, rows in splits.items()
    }

    # pos_weight = negatives/positives per class, so each class's positive
    # examples carry as much total weight as its negatives
    counts = torch.zeros(len(PERIODS))
    for row in splits["train"]:
        for p in row["periods"]:
            counts[PERIODS.index(p)] += 1
    n_train = len(splits["train"])
    pos_weight = (n_train - counts) / counts
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(PERIODS))
    model.to(device)

    best = {"auc": -1.0, "state": None}

    # phase 1: head only
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    opt = torch.optim.AdamW(model.fc.parameters(), lr=args.lr)
    run_epochs(model, loaders["train"], loaders["val"], criterion, opt,
               args.epochs_head, device, "head", best)

    # phase 2: fine-tune everything, gently
    for param in model.parameters():
        param.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr / 10)
    run_epochs(model, loaders["train"], loaders["val"], criterion, opt,
               args.epochs_full, device, "full", best)

    model.load_state_dict(best["state"])
    macro, per_class = evaluate(model, loaders["test"], device)
    print(f"\nTEST macro-AUC {macro:.3f}")
    for p, v in per_class.items():
        print(f"  {p:20s} {'--' if v is None else f'{v:.3f}'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], out / "resnet18_best.pt")
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val_macro_auc": best["auc"], "test_macro_auc": macro,
                   "test_per_class": per_class, "periods": PERIODS,
                   "args": vars(args)}, f, indent=2)
    print(f"saved checkpoint + metrics to {out}")


if __name__ == "__main__":
    main()
