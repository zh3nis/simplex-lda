import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50

from lda import SimplexLDAHead


class ResNet50Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(2048, dim)

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
        self.encoder = ResNet50Encoder(D)
        self.head = SimplexLDAHead(C, D)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class DeepClassifier(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.encoder = ResNet50Encoder(D)
        self.head = SoftmaxHead(D, C)

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


def build_loaders(dataset_name, data_root, batch_size, test_batch_size):
    weights = ResNet50_Weights.IMAGENET1K_V2
    weights_tfm = weights.transforms()
    mean = weights_tfm.mean
    std = weights_tfm.std
    pin_memory = torch.cuda.is_available()

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset_map = {
        "CIFAR100": datasets.CIFAR100,
        "CIFAR10": datasets.CIFAR10,
        "FashionMNIST": datasets.FashionMNIST,
    }
    dataset_cls = dataset_map[dataset_name]
    if dataset_name == "FashionMNIST":
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    train_ds = dataset_cls(root=data_root, train=True, transform=train_tfm, download=True)
    test_ds = dataset_cls(root=data_root, train=False, transform=test_tfm, download=True)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    test_ld = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    return train_ld, test_ld


def train_model(model, loss_fn, train_ld, test_ld, device, epochs, tag):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
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
    parser = argparse.ArgumentParser(
        description="Train ResNet50 on CIFAR100/CIFAR10/FashionMNIST with LDA and Softmax heads."
    )
    parser.add_argument("--data-root", default="./data", help="Dataset root directory.")
    parser.add_argument("--output", default="./results/resnet50.json", help="JSON output file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--test-batch-size", type=int, default=512, help="Test batch size.")
    parser.add_argument("--runs", type=int, default=3, help="Runs per (head, dataset) pair.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "meta": {
            "epochs": args.epochs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "runs": args.runs,
        },
        "results": {},
    }
    datasets_cfg = {
        "CIFAR100": 100,
        "CIFAR10": 10,
        "FashionMNIST": 10,
    }

    for dataset_name, num_classes in datasets_cfg.items():
        train_ld, test_ld = build_loaders(dataset_name, args.data_root, args.batch_size, args.test_batch_size)
        results["results"][dataset_name] = {"SimplexLDA": [], "Softmax": []}
        for run in range(1, args.runs + 1):
            lda_model = DeepLDA(C=num_classes, D=num_classes - 1).to(device)
            lda_metrics = train_model(
                lda_model,
                nn.NLLLoss(),
                train_ld,
                test_ld,
                device,
                args.epochs,
                tag=f"{dataset_name}/SimplexLDA/run{run}",
            )
            results["results"][dataset_name]["SimplexLDA"].append({
                "run": run,
                "epoch_metrics": lda_metrics,
                "final_train_acc": lda_metrics[-1]["train_acc"],
                "final_test_acc": lda_metrics[-1]["test_acc"],
            })

            softmax_model = DeepClassifier(C=num_classes, D=num_classes - 1).to(device)
            softmax_metrics = train_model(
                softmax_model,
                nn.CrossEntropyLoss(),
                train_ld,
                test_ld,
                device,
                args.epochs,
                tag=f"{dataset_name}/Softmax/run{run}",
            )
            results["results"][dataset_name]["Softmax"].append({
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
