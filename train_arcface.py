"""ArcFace (Deng et al., CVPR 2019, Eq. 3 / Algorithm 1) with the LDA encoder.

Keep the covariance baseline's architecture, data pipeline and training protocol.
Only the classifier and objective change. Evaluation uses margin-free cosine
classification, adapting the paper's face-verification objective to these datasets.
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


class ArcFaceHead(nn.Module):
    """Normalized, bias-free class weights with an additive target angular margin.

    This implements Algorithm 1 directly, with no easy-margin or piecewise
    monotonicity approximation. Clamp target cosines just inside [-1, 1] before
    acos to keep backward gradients finite at numerical endpoints.
    """

    def __init__(self, C, D, scale=64.0, margin=0.5):
        super().__init__()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not math.isfinite(margin) or not 0 <= margin < math.pi:
            raise ValueError("margin must be finite and in [0, pi) radians")
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(C, D))
        nn.init.xavier_uniform_(self.weight)

    def cosine(self, z):
        return F.linear(F.normalize(z, dim=1), F.normalize(self.weight, dim=1)).clamp(-1, 1)

    def margin_logits(self, cosine, labels):
        if self.margin == 0:
            return self.scale * cosine
        target = cosine.gather(1, labels.unsqueeze(1))
        eps = torch.finfo(cosine.dtype).eps
        theta = target.clamp(-1 + eps, 1 - eps).acos()
        marginal_target = (theta + self.margin).cos()
        # Scatter replaces only the ground-truth class logit in each row.
        return self.scale * cosine.scatter(1, labels.unsqueeze(1), marginal_target)

    def forward(self, z, labels=None):
        cosine = self.cosine(z)
        return self.scale * cosine if labels is None else self.margin_logits(cosine, labels)


class DeepArcFace(nn.Module):
    def __init__(self, C, D, in_channels, scale=64.0, margin=0.5):
        super().__init__()
        # Import the exact baseline class, including its final projection.
        self.encoder = Encoder(D, in_channels)
        self.head = ArcFaceHead(C, D, scale=scale, margin=margin)

    def forward(self, x, labels=None):
        return self.head(self.encoder(x), labels)


def train_one_run(dataset_name, C, D, in_channels, device, epochs,
                  train_ld, test_ld, scale=64.0, margin=0.5):
    model = DeepArcFace(C, D, in_channels, scale=scale, margin=margin).to(device)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    epoch_records = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = acc_sum = n_sum = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            cosine = model.head.cosine(model.encoder(x))
            loss = loss_fn(model.head.margin_logits(cosine, y), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                # As in the baseline, use pre-step predictions; report accuracy
                # without the training-only target margin, as at evaluation.
                acc_sum += (cosine.argmax(1) == y).sum().item()
                n_sum += y.size(0)
                loss_sum += loss.item() * y.size(0)
        tr_acc = acc_sum / n_sum
        te_acc = evaluate(model, test_ld, device)
        train_loss = loss_sum / n_sum
        epoch_records.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_acc": tr_acc, "test_acc": te_acc,
        })
        print(f"[{dataset_name}][ArcFace][{epoch:03d}/{epochs:03d}] "
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
    parser.add_argument("--output", type=Path, default=Path("./results/arcface.json"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--datasets", nargs="+", choices=["FashionMNIST", "CIFAR10", "CIFAR100"],
                        default=["FashionMNIST", "CIFAR10", "CIFAR100"])
    parser.add_argument("--scale", type=float, default=64.0,
                        help="ArcFace logit scale s (not the simplex pairwise distance)")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="additive angular margin m in radians")
    parser.add_argument("--seed", type=int, default=1337,
                        help="run r (one-based) uses seed + r, matching the LDA baseline")
    args = parser.parse_args()
    for name in ("epochs", "runs", "batch_size", "test_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be finite and positive")
    if not math.isfinite(args.margin) or not 0 <= args.margin < math.pi:
        parser.error("--margin must be finite and in [0, pi) radians")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_cfg = [
        {"name": "FashionMNIST", "C": 10, "D": 9},
        {"name": "CIFAR10", "C": 10, "D": 9},
        {"name": "CIFAR100", "C": 100, "D": 99},
    ]
    datasets_cfg = [cfg for cfg in datasets_cfg if cfg["name"] in args.datasets]
    results = {
        "meta": {
            "epochs": args.epochs, "runs": args.runs, "objective": "arcface",
            "scale": args.scale, "margin_radians": args.margin,
            "margin_implementation": "paper_algorithm_1_acos_with_endpoint_clamp",
            "evaluation": "margin_free_cosine_classifier",
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
        results["results"][cfg["name"]] = {"arcface": runs}
        for run_idx in range(1, args.runs + 1):
            seed = args.seed + run_idx
            seed_everything(seed)
            print(f"=== {cfg['name']} | ArcFace | run {run_idx}/{args.runs} | seed={seed} ===",
                  flush=True)
            epoch_records = train_one_run(
                dataset_name=cfg["name"], C=cfg["C"], D=cfg["D"],
                in_channels=in_channels, device=device, epochs=args.epochs,
                train_ld=train_ld, test_ld=test_ld, scale=args.scale, margin=args.margin,
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
            # Preserve completed runs if a later run is interrupted.
            save_results(args.output, results)
        summary = results["summary"][cfg["name"]]
        print(f"{cfg['name']}: final test acc = {summary['final_test_mean']:.4f} "
              f"± {summary['final_test_std']:.4f} (population std)", flush=True)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
