"""Prototype network from Li et al., AAAI 2018, Eqs. (1)-(7).

Reuse the covariance baseline's exact encoder and data protocol. Jointly learn
the encoder, decoder, latent prototypes, and bias-free distance classifier.
Reference: https://github.com/OscarcarLi/PrototypeDL (CAE_MNIST.py).
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_deep_lda_covariances import Encoder, build_loaders, evaluate, seed_everything


class PrototypeHead(nn.Module):
    """Eq. (1): squared distances, followed by learned W with no bias.

    Prototypes have no assigned class. W is unconstrained; there is no feature
    normalization, distance negation, or nearest-class-prototype inference.
    """

    def __init__(self, C, D, num_prototypes=15):
        super().__init__()
        if C < 2 or D < 1 or num_prototypes < 1:
            raise ValueError("C must be at least 2; D and num_prototypes must be positive")
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, D))
        self.classifier = nn.Linear(num_prototypes, C, bias=False)
        # Match the authors' initialization of prototypes and last_layer['w'].
        nn.init.uniform_(self.prototypes, 0.0, 1.0)
        nn.init.uniform_(self.classifier.weight, 0.0, 1.0)

    def squared_distances(self, z):
        # Matrix form avoids a batch x prototypes x latent_dim intermediate.
        return (z.square().sum(1, keepdim=True)
                + self.prototypes.square().sum(1).unsqueeze(0)
                - 2 * z @ self.prototypes.T).clamp_min(0)

    def forward(self, z):
        return self.classifier(self.squared_distances(z))


class Decoder(nn.Module):
    """Decode the baseline's vector bottleneck into a 28x28 or 32x32 image."""

    def __init__(self, D, in_channels, image_size):
        super().__init__()
        if image_size not in (28, 32):
            raise ValueError("image_size must be 28 or 32")
        self.side = image_size // 4
        self.proj = nn.Linear(D, 256 * self.side * self.side)
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.features(self.proj(z).reshape(-1, 256, self.side, self.side))


class PrototypeDL(nn.Module):
    def __init__(self, C, D, in_channels, image_size, num_prototypes=15):
        super().__init__()
        self.encoder = Encoder(D, in_channels)
        self.head = PrototypeHead(C, D, num_prototypes)
        self.decoder = Decoder(D, in_channels, image_size)

    def forward(self, x):
        # Classification does not need labels or the decoder.
        return self.head(self.encoder(x))

    def training_outputs(self, x):
        z = self.encoder(x)
        distances = self.head.squared_distances(z)
        return self.head.classifier(distances), self.decoder(z), distances


def pixel_targets(x, dataset_name):
    """Undo only loader normalization; retain the same random crop/flip target."""
    if dataset_name == "FashionMNIST":
        return x
    if dataset_name == "CIFAR10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    elif dataset_name == "CIFAR100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return x * x.new_tensor(std).view(1, -1, 1, 1) + x.new_tensor(mean).view(1, -1, 1, 1)


def prototype_loss(logits, labels, reconstruction, target, distances,
                   lambda_reconstruction=0.05, lambda_1=0.05, lambda_2=0.05):
    """Eq. (7), using the paper's minibatch approximation for R1.

    R sums pixels per image before averaging the batch (not mean pixel MSE).
    R1 averages each prototype's distance to its nearest example; R2 averages
    each example's distance to its nearest prototype. Neither uses labels.
    """
    terms = {
        "classification_loss": F.cross_entropy(logits, labels),
        "reconstruction_loss": (reconstruction - target).square().flatten(1).sum(1).mean(),
        "prototype_to_data_loss": distances.min(dim=0).values.mean(),
        "data_to_prototype_loss": distances.min(dim=1).values.mean(),
    }
    total = (terms["classification_loss"]
             + lambda_reconstruction * terms["reconstruction_loss"]
             + lambda_1 * terms["prototype_to_data_loss"]
             + lambda_2 * terms["data_to_prototype_loss"])
    return total, terms


def train_one_run(dataset_name, C, D, in_channels, image_size, device, epochs,
                  train_ld, test_ld, num_prototypes=15, lr=0.001,
                  lambda_reconstruction=0.05, lambda_1=0.05, lambda_2=0.05):
    model = PrototypeDL(C, D, in_channels, image_size, num_prototypes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_records = []
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {name: 0.0 for name in (
            "train_loss", "classification_loss", "reconstruction_loss",
            "prototype_to_data_loss", "data_to_prototype_loss",
        )}
        acc_sum = n_sum = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits, reconstruction, distances = model.training_outputs(x)
            loss, terms = prototype_loss(
                logits, y, reconstruction, pixel_targets(x, dataset_name), distances,
                lambda_reconstruction, lambda_1, lambda_2,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                acc_sum += (logits.argmax(1) == y).sum().item()
                n_sum += y.size(0)
                sums["train_loss"] += loss.item() * y.size(0)
                for name, value in terms.items():
                    sums[name] += value.item() * y.size(0)
        record = {"epoch": epoch, **{name: value / n_sum for name, value in sums.items()},
                  "train_acc": acc_sum / n_sum, "test_acc": evaluate(model, test_ld, device)}
        epoch_records.append(record)
        print(f"[{dataset_name}][PrototypeDL][{epoch:03d}/{epochs:03d}] "
              f"loss={record['train_loss']:.4f} | train acc={record['train_acc']:.4f} | "
              f"test acc={record['test_acc']:.4f}", flush=True)
    return epoch_records


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", type=Path, default=Path("./results/prototype_dl.json"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--datasets", nargs="+", choices=["FashionMNIST", "CIFAR10", "CIFAR100"],
                        default=["FashionMNIST", "CIFAR10", "CIFAR100"])
    parser.add_argument("--num-prototypes", type=int, default=15,
                        help="Number of learned prototypes, independent of class count (paper: 15).")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Adam learning rate (baseline: 0.001; paper Sec. 3.2: 0.0001).")
    parser.add_argument("--lambda-reconstruction", type=float, default=0.05)
    parser.add_argument("--lambda-1", type=float, default=0.05,
                        help="Weight of prototype-to-data regularizer R1.")
    parser.add_argument("--lambda-2", type=float, default=0.05,
                        help="Weight of data-to-prototype regularizer R2.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Run r (one-based) uses seed + r, matching the baseline.")
    args = parser.parse_args()
    for name in ("epochs", "runs", "batch_size", "test_batch_size", "num_prototypes"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    for name in ("lambda_reconstruction", "lambda_1", "lambda_2"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_cfg = [
        {"name": "FashionMNIST", "C": 10, "D": 9, "image_size": 28},
        {"name": "CIFAR10", "C": 10, "D": 9, "image_size": 32},
        {"name": "CIFAR100", "C": 100, "D": 99, "image_size": 32},
    ]
    datasets_cfg = [cfg for cfg in datasets_cfg if cfg["name"] in args.datasets]
    results = {
        "meta": {
            "epochs": args.epochs, "runs": args.runs, "objective": "prototype_dl",
            "paper": "Li et al., Deep Learning for Case-Based Reasoning through Prototypes, AAAI 2018",
            "reference_code": "https://github.com/OscarcarLi/PrototypeDL",
            "loss": "E + lambda_reconstruction * R + lambda_1 * R1 + lambda_2 * R2 (Eq. 7)",
            "lambda_reconstruction": args.lambda_reconstruction,
            "lambda_1": args.lambda_1, "lambda_2": args.lambda_2,
            "num_prototypes": args.num_prototypes,
            "prototype_initialization": "uniform_0_1",
            "classifier": "learned_bias_free_linear_on_squared_euclidean_distances",
            "classifier_initialization": "uniform_0_1",
            "reconstruction": "mean_batch_sum_squared_pixel_error_in_0_1_space",
            "R1": "mean_prototype_min_over_current_minibatch",
            "R2": "mean_example_min_over_prototypes",
            "optimized_parameters": "encoder_decoder_prototypes_classifier_jointly",
            "evaluation": "argmax_learned_weighted_squared_distances",
            "encoder": "train_deep_lda_covariances.Encoder",
            "decoder": "linear_to_256x(H/4)x(W/4)_convolutions_two_transposed_convolutions_sigmoid",
            "datasets": datasets_cfg,
            "optimizer": {"name": "Adam", "lr": args.lr, "betas": [0.9, 0.999],
                          "eps": 1e-8, "weight_decay": 0},
            "batch_size": args.batch_size, "test_batch_size": args.test_batch_size,
            "data_root": str(args.data_root), "seed_base": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device), "torch_version": str(torch.__version__),
        },
        "results": {}, "summary": {},
    }
    for cfg in datasets_cfg:
        train_ld, test_ld, in_channels = build_loaders(
            cfg["name"], args.data_root, args.batch_size, args.test_batch_size,
        )
        runs = []
        results["results"][cfg["name"]] = {"prototype_dl": runs}
        for run_idx in range(1, args.runs + 1):
            seed = args.seed + run_idx
            seed_everything(seed)
            print(f"=== {cfg['name']} | PrototypeDL | run {run_idx}/{args.runs} | seed={seed} ===",
                  flush=True)
            epoch_records = train_one_run(
                dataset_name=cfg["name"], C=cfg["C"], D=cfg["D"],
                in_channels=in_channels, image_size=cfg["image_size"], device=device,
                epochs=args.epochs, train_ld=train_ld, test_ld=test_ld,
                num_prototypes=args.num_prototypes, lr=args.lr,
                lambda_reconstruction=args.lambda_reconstruction,
                lambda_1=args.lambda_1, lambda_2=args.lambda_2,
            )
            final = epoch_records[-1]
            runs.append({
                "run": run_idx, "seed": seed, "epoch_metrics": epoch_records,
                "final_train_acc": final["train_acc"], "final_test_acc": final["test_acc"],
            })
            finals = [run["final_test_acc"] for run in runs]
            results["summary"][cfg["name"]] = {
                "completed_runs": len(runs), "final_test_mean": statistics.mean(finals),
                "final_test_std": statistics.pstdev(finals),
            }
            save_results(args.output, results)
        summary = results["summary"][cfg["name"]]
        print(f"{cfg['name']}: final test acc = {summary['final_test_mean']:.4f} "
              f"± {summary['final_test_std']:.4f} (population std)", flush=True)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
