"""CIFAR-100 sensitivity to simplex pairwise distance s, for Simplex LDA only.

Reuse the latent-dimension sweep's encoder, data transforms, Adam optimizer,
and NLL objective. Hold latent dimension and identity covariance fixed.
"""
import argparse
import math
import pathlib
import random
import statistics

from CIFAR100_LatentDimSweep import (
    Encoder, build_loaders, plt, save_results_json, train_single,
)
import torch
import torch.nn as nn

from lda import SimplexLDAHead


class DeepLDA(nn.Module):
    def __init__(self, C: int, D: int, s: float):
        super().__init__()
        self.encoder = Encoder(D)
        self.head = SimplexLDAHead(C, D, covariance_type="identity", s=s)

    def forward(self, x):
        return self.head(self.encoder(x))


def run_sweep(scales, latent_dim, epochs, train_ld, test_ld, device, runs=3, seed=0):
    results = {}
    for s in scales:
        run_results = []
        for run in range(runs):
            # Match initialization and shuffled data across scales for each seed.
            run_seed = seed + run
            random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)
            print(f"=== Simplex LDA with s={s:g}, D={latent_dim}, "
                  f"run {run + 1}/{runs}, seed={run_seed} ===", flush=True)
            model = DeepLDA(C=100, D=latent_dim, s=s).to(device)
            opt = torch.optim.Adam(model.parameters())
            metrics = train_single(
                model, nn.NLLLoss(), opt, train_ld, test_ld, device, epochs,
            )
            run_results.append({"run": run + 1, "seed": run_seed, **metrics})
            del opt, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finals = [metrics["final_test"] for metrics in run_results]
        results[s] = {
            "runs": run_results,
            "final_test_mean": statistics.mean(finals),
            "final_test_std": statistics.pstdev(finals),
        }
    return results


def save_plot(path, scales, results, latent_dim):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort only for plotting so custom CLI order does not produce crossing lines.
    scales = sorted(scales)
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        scales, [results[s]["final_test_mean"] for s in scales],
        yerr=[results[s]["final_test_std"] for s in scales],
        marker="o", capsize=3, label="Simplex DeepLDA",
    )
    plt.xlabel("Simplex scale s (pairwise distance)")
    plt.ylabel("Final test accuracy (mean ± std)")
    plt.title(f"CIFAR-100 final test accuracy vs s (D={latent_dim})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved plot to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=float, nargs="+", default=[1, 2, 4, 6, 8, 10, 12],
                        help="simplex pairwise distances s to try (space-separated)")
    parser.add_argument("--latent-dim", type=int, default=99,
                        help="fixed latent dimension D (must be >= 99)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("./data"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("plots"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; run i uses seed + i (zero-based)")
    args = parser.parse_args()

    if any(not math.isfinite(s) or s <= 0 for s in args.scales):
        parser.error("every scale must be finite and positive")
    if len(set(args.scales)) != len(args.scales):
        parser.error("--scales must not contain duplicates")
    if args.latent_dim < 99:
        parser.error("CIFAR-100 SimplexLDA requires --latent-dim >= 99")
    if args.runs < 1 or args.epochs < 1:
        parser.error("--runs and --epochs must be positive")
    if args.batch_size < 1 or args.test_batch_size < 1 or args.workers < 0:
        parser.error("batch sizes must be positive and --workers must be nonnegative")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    train_ld, test_ld = build_loaders(
        args.data_root, args.batch_size, args.test_batch_size, args.workers,
    )
    results = run_sweep(
        args.scales, args.latent_dim, args.epochs, train_ld, test_ld, device,
        runs=args.runs, seed=args.seed,
    )
    payload = {
        "scales": args.scales,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "runs": args.runs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "workers": args.workers,
        "covariance_type": "identity",
        "device": str(device),
        "DeepLDA": results,
    }
    save_results_json(args.output_dir / "cifar100_simplex_scale_sweep.json", payload)
    save_plot(args.output_dir / "cifar100_simplex_scale_sweep.png",
              args.scales, results, args.latent_dim)
    print("Simplex LDA final test accuracy by s:")
    for s in args.scales:
        print(f"s={s:g}: final test acc = {results[s]['final_test_mean']:.4f} "
              f"± {results[s]['final_test_std']:.4f} ({args.runs} runs)")


if __name__ == "__main__":
    main()
