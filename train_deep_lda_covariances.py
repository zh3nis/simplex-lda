import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lda import SimplexLDAHead


class Encoder(nn.Module):
    def __init__(self, dim, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
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


class DeepLDA(nn.Module):
    def __init__(self, C, D, in_channels, covariance_type):
        super().__init__()
        self.encoder = Encoder(D, in_channels)
        self.head = SimplexLDAHead(C, D, covariance_type=covariance_type)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


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


def build_loaders(dataset_name, data_root, batch_size, test_batch_size):
    if dataset_name == "FashionMNIST":
        tfm = transforms.ToTensor()
        train_ds = datasets.FashionMNIST(root=data_root, train=True, transform=tfm, download=True)
        test_ds = datasets.FashionMNIST(root=data_root, train=False, transform=tfm, download=True)
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_ld = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)
        in_channels = 1
    elif dataset_name == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        pin_memory = torch.cuda.is_available()
        train_tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_ds = datasets.CIFAR10(root=data_root, train=True, transform=train_tfm, download=True)
        test_ds = datasets.CIFAR10(root=data_root, train=False, transform=test_tfm, download=True)
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
        test_ld = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
        in_channels = 3
    elif dataset_name == "CIFAR100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        pin_memory = torch.cuda.is_available()
        train_tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_ds = datasets.CIFAR100(root=data_root, train=True, transform=train_tfm, download=True)
        test_ds = datasets.CIFAR100(root=data_root, train=False, transform=test_tfm, download=True)
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
        test_ld = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
        in_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ld, test_ld, in_channels


def train_one_run(dataset_name, C, D, in_channels, covariance_type, device, epochs, train_ld, test_ld, optimize_head):
    model = DeepLDA(C=C, D=D, in_channels=in_channels, covariance_type=covariance_type).to(device)
    params = model.parameters() if optimize_head else model.encoder.parameters()
    opt = torch.optim.Adam(params)
    loss_fn = nn.NLLLoss()

    epoch_records = []
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
        epoch_records.append({"epoch": epoch, "train_acc": tr_acc, "test_acc": te_acc})
        print(
            f"[{dataset_name}][{covariance_type}][{epoch:03d}/{epochs:03d}] "
            f"train acc={tr_acc:.4f} | test acc={te_acc:.4f}",
            flush=True,
        )

    return epoch_records


def main():
    parser = argparse.ArgumentParser(description="Train Deep LDA with multiple covariance types.")
    parser.add_argument("--data-root", default="./data", help="Dataset root directory.")
    parser.add_argument("--output", default="./results/deep_lda_covariances.json", help="JSON output file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per run.")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per covariance type.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--test-batch-size", type=int, default=1024, help="Test batch size.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    covariance_types = ["spherical", "diag", "full"]
    datasets_cfg = [
        {"name": "FashionMNIST", "C": 10, "D": 9, "optimize_head": False},
        {"name": "CIFAR10", "C": 10, "D": 9, "optimize_head": True},
        {"name": "CIFAR100", "C": 100, "D": 99, "optimize_head": True},
    ]

    results = {
        "meta": {
            "epochs": args.epochs,
            "runs": args.runs,
            "covariance_types": covariance_types,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
        },
        "results": {},
    }

    for cfg in datasets_cfg:
        train_ld, test_ld, in_channels = build_loaders(
            cfg["name"], args.data_root, args.batch_size, args.test_batch_size
        )
        results["results"][cfg["name"]] = {}
        for cov in covariance_types:
            runs = []
            for run_idx in range(1, args.runs + 1):
                seed_everything(1337 + run_idx)
                epoch_records = train_one_run(
                    dataset_name=cfg["name"],
                    C=cfg["C"],
                    D=cfg["D"],
                    in_channels=in_channels,
                    covariance_type=cov,
                    device=device,
                    epochs=args.epochs,
                    train_ld=train_ld,
                    test_ld=test_ld,
                    optimize_head=cfg["optimize_head"],
                )
                final = epoch_records[-1]
                runs.append(
                    {
                        "run": run_idx,
                        "epoch_metrics": epoch_records,
                        "final_train_acc": final["train_acc"],
                        "final_test_acc": final["test_acc"],
                    }
                )
            results["results"][cfg["name"]][cov] = runs

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
