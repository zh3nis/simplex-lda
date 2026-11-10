"""MM-LDA (Pang et al., ICML 2018, Sec. 3.2 / Algorithm 2).

Use the covariance baseline's encoder and training protocol. Fixed simplex
means satisfy Eq. (4); identity covariance and fixed uniform class priors give
the Gaussian posterior in Eq. (5). Train only the encoder with cross entropy.
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


class MMLDAHead(nn.Module):
    """Fixed MMD classifier; C is class count, not the paper's squared norm C.

    No feature normalization, learned means/covariance/priors, or target margin.
    Optional priors are fixed positive probabilities (uniform by default).
    """

    def __init__(self, C, D, mean_norm_squared=100.0, priors=None):
        super().__init__()
        if C < 2 or D < C - 1:
            raise ValueError("C must be at least 2 and D must be at least C-1")
        if not math.isfinite(mean_norm_squared) or mean_norm_squared <= 0:
            raise ValueError("mean_norm_squared must be finite and positive")
        self.C, self.D = C, D
        self.mean_norm_squared = mean_norm_squared
        # Closed form of Algorithm 1's triangular construction, omitting its
        # final zero coordinate so the minimal dimension D=C-1 is supported.
        # Build in float64 before casting to avoid accumulating roundoff.
        mu = torch.zeros(C, D, dtype=torch.float64)
        for j in range(C - 1):
            remaining = C - j
            mu[j, j] = math.sqrt((C / (C - 1)) * ((remaining - 1) / remaining))
            mu[j + 1:, j] = -math.sqrt(C / ((C - 1) * remaining * (remaining - 1)))
        mu *= math.sqrt(mean_norm_squared)
        dtype = torch.get_default_dtype()
        self.register_buffer("mu", mu.to(dtype=dtype))
        if priors is None:
            prior = torch.full((C,), 1 / C, dtype=torch.float64)
        else:
            prior = torch.as_tensor(priors, dtype=torch.float64).detach().clone()
            if (prior.shape != (C,) or not torch.isfinite(prior).all()
                    or not (prior > 0).all()
                    or not torch.isclose(prior.sum(), prior.new_tensor(1.0))):
                raise ValueError("priors must contain C finite positive probabilities summing to 1")
        self.register_buffer("log_priors", prior.log().to(dtype=dtype))

    def forward(self, z):
        # log pi_k - ||z-mu_k||^2/2, dropping the shared -||z||^2/2.
        # This linear form avoids cancellation for large feature norms and
        # returns logits, not probabilities, for stable CrossEntropyLoss.
        mu = self.mu.to(dtype=z.dtype)
        bias = self.log_priors.to(dtype=z.dtype) - 0.5 * mu.square().sum(dim=1)
        return F.linear(z, mu, bias)


class DeepMMLDA(nn.Module):
    def __init__(self, C, D, in_channels, mean_norm_squared=100.0, priors=None):
        super().__init__()
        self.encoder = Encoder(D, in_channels)
        self.head = MMLDAHead(C, D, mean_norm_squared=mean_norm_squared, priors=priors)

    def forward(self, x):
        return self.head(self.encoder(x))


def train_one_run(dataset_name, C, D, in_channels, device, epochs,
                  train_ld, test_ld, mean_norm_squared=100.0):
    model = DeepMMLDA(C, D, in_channels, mean_norm_squared=mean_norm_squared).to(device)
    opt = torch.optim.Adam(model.encoder.parameters())
    loss_fn = nn.CrossEntropyLoss()
    epoch_records = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = acc_sum = n_sum = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                acc_sum += (logits.argmax(1) == y).sum().item()
                n_sum += y.size(0)
                loss_sum += loss.item() * y.size(0)
        tr_acc = acc_sum / n_sum
        te_acc = evaluate(model, test_ld, device)
        train_loss = loss_sum / n_sum
        epoch_records.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_acc": tr_acc, "test_acc": te_acc,
        })
        print(f"[{dataset_name}][MM-LDA][{epoch:03d}/{epochs:03d}] "
              f"loss={train_loss:.4f} | train acc={tr_acc:.4f} | test acc={te_acc:.4f}",
              flush=True)
    return epoch_records


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", type=Path, default=Path("./results/mm_lda.json"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--datasets", nargs="+", choices=["FashionMNIST", "CIFAR10", "CIFAR100"],
                        default=["FashionMNIST", "CIFAR10", "CIFAR100"])
    parser.add_argument("--mean-norm-squared", type=float, default=100.0,
                        help="paper's C = ||mu_k||^2 (not class count or pairwise distance)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="run r (one-based) uses seed + r, matching the LDA baseline")
    args = parser.parse_args()
    for name in ("epochs", "runs", "batch_size", "test_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.mean_norm_squared) or args.mean_norm_squared <= 0:
        parser.error("--mean-norm-squared must be finite and positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_cfg = [
        {"name": "FashionMNIST", "C": 10, "D": 9},
        {"name": "CIFAR10", "C": 10, "D": 9},
        {"name": "CIFAR100", "C": 100, "D": 99},
    ]
    datasets_cfg = [cfg for cfg in datasets_cfg if cfg["name"] in args.datasets]
    results = {
        "meta": {
            "epochs": args.epochs, "runs": args.runs, "objective": "mm_lda",
            "paper": "Pang et al., Max-Mahalanobis Linear Discriminant Analysis Networks, ICML 2018",
            "loss": "cross_entropy_gaussian_posterior_eq_5",
            "mean_norm_squared": args.mean_norm_squared,
            "means": "fixed_regular_simplex_eq_4_algorithm_1_closed_form",
            "covariance": "fixed_identity", "class_priors": "fixed_uniform",
            "optimized_parameters": "encoder_only",
            "evaluation": "argmax_gaussian_posterior",
            "encoder": "train_deep_lda_covariances.Encoder",
            "datasets": datasets_cfg,
            "optimizer": {"name": "Adam", "lr": 0.001, "betas": [0.9, 0.999],
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
        results["results"][cfg["name"]] = {"mm_lda": runs}
        for run_idx in range(1, args.runs + 1):
            seed = args.seed + run_idx
            seed_everything(seed)
            print(f"=== {cfg['name']} | MM-LDA | run {run_idx}/{args.runs} | seed={seed} ===",
                  flush=True)
            epoch_records = train_one_run(
                dataset_name=cfg["name"], C=cfg["C"], D=cfg["D"],
                in_channels=in_channels, device=device, epochs=args.epochs,
                train_ld=train_ld, test_ld=test_ld, mean_norm_squared=args.mean_norm_squared,
            )
            final = epoch_records[-1]
            runs.append({
                "run": run_idx, "seed": seed, "epoch_metrics": epoch_records,
                "final_train_acc": final["train_acc"],
                "final_test_acc": final["test_acc"],
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
