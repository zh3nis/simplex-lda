import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from lda import SimplexLDAHead


class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class SoftmaxHead(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.linear = nn.Linear(D, C)

    def forward(self, z):
        return self.linear(z)


class DeepLDA(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.encoder = Encoder(D)
        self.head = SimplexLDAHead(C, D)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class DeepClassifier(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.encoder = Encoder(D)
        self.head = SoftmaxHead(D, C)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class TinyImageNetTrain(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.wnids = self._load_wnids()
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.samples = []
        for wnid in self.wnids:
            images_dir = self.root / "train" / wnid / "images"
            for path in sorted(images_dir.glob("*.JPEG")):
                self.samples.append((path, self.wnid_to_idx[wnid]))

    def _load_wnids(self):
        wnids_path = self.root / "wnids.txt"
        with open(wnids_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.wnids = self._load_wnids()
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.samples = []
        ann_path = self.root / "val" / "val_annotations.txt"
        images_dir = self.root / "val" / "images"
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                fname, wnid = parts[0], parts[1]
                label = self.wnid_to_idx[wnid]
                self.samples.append((images_dir / fname, label))

    def _load_wnids(self):
        wnids_path = self.root / "wnids.txt"
        with open(wnids_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ok = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ok += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return ok / tot


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_root(data_root):
    root = Path(data_root)
    if (root / "train").exists() and (root / "val").exists():
        return root
    candidate = root / "tiny-imagenet-200"
    if (candidate / "train").exists() and (candidate / "val").exists():
        return candidate
    raise FileNotFoundError(f"Tiny ImageNet directory not found under {data_root}")


def build_loaders(data_root, batch_size, test_batch_size, num_workers):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    pin_memory = torch.cuda.is_available()

    train_tfm = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = TinyImageNetTrain(data_root, transform=train_tfm)
    test_ds = TinyImageNetVal(data_root, transform=test_tfm)
    train_ld = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    test_ld = DataLoader(
        test_ds, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_ld, test_ld


def train_model(model, loss_fn, train_ld, test_ld, device, epochs, tag):
    opt = torch.optim.Adam(model.parameters())
    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        model.train()
        acc_sum = n_sum = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = logits.argmax(1)
                acc_sum += (pred == y).sum().item()
                n_sum += y.size(0)
        tr_acc = acc_sum / n_sum
        te_acc = evaluate(model, test_ld, device)
        epoch_metrics.append({"epoch": epoch, "train_acc": tr_acc, "test_acc": te_acc})
        print(
            f"[{tag}][{epoch:03d}/{epochs:03d}] train acc={tr_acc:.4f} | test acc={te_acc:.4f}",
            flush=True,
        )

    return epoch_metrics


def main():
    parser = argparse.ArgumentParser(description="Train DeepLDA and Softmax on Tiny ImageNet.")
    parser.add_argument("--data-root", default="./data/tiny-imagenet-200", help="Dataset root directory.")
    parser.add_argument("--output", default="./results/tiny_imagenet_deeplda.json", help="JSON output file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--runs", type=int, default=3, help="Runs per head.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--test-batch-size", type=int, default=512, help="Test batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers.")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 200
    embedding_dim = num_classes - 1

    train_ld, test_ld = build_loaders(data_root, args.batch_size, args.test_batch_size, args.num_workers)

    results = {
        "meta": {
            "epochs": args.epochs,
            "runs": args.runs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "dataset": "TinyImageNet",
        },
        "results": {"SimplexLDA": [], "Softmax": []},
    }

    for run in range(1, args.runs + 1):
        seed_everything(1337 + run)
        lda_model = DeepLDA(C=num_classes, D=embedding_dim).to(device)
        lda_metrics = train_model(
            lda_model,
            nn.NLLLoss(),
            train_ld,
            test_ld,
            device,
            args.epochs,
            tag=f"TinyImageNet/SimplexLDA/run{run}",
        )
        results["results"]["SimplexLDA"].append({
            "run": run,
            "epoch_metrics": lda_metrics,
            "final_train_acc": lda_metrics[-1]["train_acc"],
            "final_test_acc": lda_metrics[-1]["test_acc"],
        })

        seed_everything(1337 + run)
        softmax_model = DeepClassifier(C=num_classes, D=embedding_dim).to(device)
        softmax_metrics = train_model(
            softmax_model,
            nn.CrossEntropyLoss(),
            train_ld,
            test_ld,
            device,
            args.epochs,
            tag=f"TinyImageNet/Softmax/run{run}",
        )
        results["results"]["Softmax"].append({
            "run": run,
            "epoch_metrics": softmax_metrics,
            "final_train_acc": softmax_metrics[-1]["train_acc"],
            "final_test_acc": softmax_metrics[-1]["test_acc"],
        })

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
