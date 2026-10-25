"""Train simplex Deep LDA with the notebook Fisher objective and Sigma = I."""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from lda import FisherSimplexLDAHead
from train_deep_lda_covariances import Encoder, build_loaders, evaluate, seed_everything


class DeepFisherLDA(nn.Module):
    def __init__(self, C, D, in_channels):
        super().__init__()
        self.encoder = Encoder(D, in_channels)
        self.head = FisherSimplexLDAHead(C, D, covariance_type="identity")

    def logits(self, x):
        return self.head.logits(self.encoder(x))

    def forward(self, x, y=None):
        z = self.encoder(x)
        return self.head.logits(z) if y is None else self.head(z, y)


def train_one_run(dataset_name, C, D, in_channels, device, epochs,
                  train_ld, test_ld, optimize_head):
    model = DeepFisherLDA(C, D, in_channels).to(device)
    # Match the covariance experiment's dataset-specific prior optimization.
    model.head.requires_grad_(optimize_head)
    opt = torch.optim.Adam(p for p in model.parameters() if p.requires_grad)

    epoch_records = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = acc_sum = n_sum = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            loss = model(x, y)  # Negative between / (within + eps), as in the notebook.
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                # The notebook measures training accuracy after the optimizer step.
                pred = model.logits(x).argmax(1)
                acc_sum += (pred == y).sum().item()
                n_sum += y.size(0)
                loss_sum += loss.item() * y.size(0)
        tr_acc = acc_sum / n_sum
        te_acc = evaluate(model, test_ld, device)
        fisher_loss = loss_sum / n_sum
        epoch_records.append({
            "epoch": epoch, "fisher_loss": fisher_loss,
            "train_acc": tr_acc, "test_acc": te_acc,
        })
        print(
            f"[{dataset_name}][identity][{epoch:03d}/{epochs:03d}] "
            f"fisher_loss={fisher_loss:.4f} | train acc={tr_acc:.4f} | test acc={te_acc:.4f}",
            flush=True,
        )
    return epoch_records


def main():
    parser = argparse.ArgumentParser(
        description="Train Deep LDA with the Fisher objective and fixed identity covariance."
    )
    parser.add_argument("--data-root", default="./data", help="Dataset root directory.")
    parser.add_argument("--output", default="./results/deep_lda_fisher.json", help="JSON output file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per run.")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per dataset.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--test-batch-size", type=int, default=1024, help="Test batch size.")
    args = parser.parse_args()
    for name in ("epochs", "runs", "batch_size", "test_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_cfg = [
        {"name": "FashionMNIST", "C": 10, "D": 9, "optimize_head": True},
        {"name": "CIFAR10", "C": 10, "D": 9, "optimize_head": True},
        {"name": "CIFAR100", "C": 100, "D": 99, "optimize_head": True},
    ]
    results = {
        "meta": {
            "epochs": args.epochs,
            "runs": args.runs,
            "objective": "fisher",
            "covariance_types": ["identity"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
        },
        "results": {},
    }

    for cfg in datasets_cfg:
        train_ld, test_ld, in_channels = build_loaders(
            cfg["name"], args.data_root, args.batch_size, args.test_batch_size
        )
        runs = []
        for run_idx in range(1, args.runs + 1):
            seed_everything(1337 + run_idx)
            epoch_records = train_one_run(
                dataset_name=cfg["name"], C=cfg["C"], D=cfg["D"],
                in_channels=in_channels, device=device, epochs=args.epochs,
                train_ld=train_ld, test_ld=test_ld, optimize_head=cfg["optimize_head"],
            )
            final = epoch_records[-1]
            runs.append({
                "run": run_idx,
                "epoch_metrics": epoch_records,
                "final_train_acc": final["train_acc"],
                "final_test_acc": final["test_acc"],
            })
        results["results"][cfg["name"]] = {"identity": runs}

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
