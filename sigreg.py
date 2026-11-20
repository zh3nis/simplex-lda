"""Class-conditional adaptation of LeJEPA Algorithm 1 (local LeJEPA.pdf).

The paper regularizes an entire embedding distribution toward N(0, I).
Here each class is tested separately after subtracting its fixed LDA center.
By default the targets are N(mu_c, I); optionally standardize by the model's
shared spherical standard deviation to target N(mu_c, sigma^2 I).
No empirical centering or covariance estimation is applied.
"""

import math

import torch
from torch import nn


class ClasswiseSIGReg(nn.Module):
    """Frequency-weighted mean of class-wise Epps--Pulley statistics.

    Defaults to the paper's 256 random unit projections, 17 trapezoidal
    integration points on [-5, 5], and exp(-t^2/2) as both target CF and window.
    Increase num_points for finer quadrature when standardized residuals are
    large: an evenly spaced frequency grid aliases translations by 2*pi/dt.
    Each class statistic retains Algorithm 1's sample-count multiplier n_c.
    Classes with fewer than two samples contribute zero without renormalizing
    other weights. Increase ``step`` each minibatch to resample projections.
    A private generator keeps sampling independent of model/data-loader RNGs.
    """

    def __init__(self, num_slices=256, seed=1729, *, num_points=17):
        super().__init__()
        if not isinstance(num_slices, int) or num_slices < 1:
            raise ValueError("num_slices must be a positive integer.")
        if not isinstance(num_points, int) or num_points < 3:
            raise ValueError("num_points must be an integer of at least 3.")
        self.num_slices = num_slices
        self.seed = int(seed)
        self.register_buffer("integration_grid", torch.linspace(-5, 5, num_points))

    def statistics(self, embeddings, target, centers, *, step=0, log_variance=None):
        """Return (per-class EP, counts); labels must be zero-based integers.

        embeddings: (N, D); target: (N,); fixed centers: (C, D).
        Projections are shared across classes, with separate empirical CFs.
        Returned statistics are zero for absent/singleton classes.
        Optional scalar log_variance is log(sigma^2) from the LDA head.
        Gradients propagate through this model variance as well as embeddings.
        """
        if embeddings.ndim != 2 or min(embeddings.shape) == 0:
            raise ValueError("embeddings must have nonempty shape (N, D).")
        if target.ndim != 1 or target.size(0) != embeddings.size(0):
            raise ValueError("target must have shape (N,) matching embeddings.")
        if centers.ndim != 2 or centers.size(0) == 0 or centers.size(1) != embeddings.size(1):
            raise ValueError("centers must have shape (C, D) matching embeddings.")

        centers = centers.to(embeddings)
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(self.seed + int(step))
        directions = torch.randn(
            embeddings.size(1), self.num_slices, generator=generator,
            device=embeddings.device, dtype=embeddings.dtype,
        )
        directions = nn.functional.normalize(directions, dim=0)
        t = self.integration_grid.to(embeddings)
        gaussian_cf = torch.exp(-0.5 * t.square())
        residuals = embeddings - centers[target]
        if log_variance is not None:
            log_variance = torch.as_tensor(
                log_variance, device=embeddings.device, dtype=embeddings.dtype,
            )
            if log_variance.numel() != 1:
                raise ValueError("log_variance must contain one shared spherical log variance.")
            residuals = residuals * torch.exp(-0.5 * log_variance.reshape(()))
        phases = (residuals @ directions).unsqueeze(-1) * t

        # Average cos/sin within each class, never across the class mixture.
        membership = nn.functional.one_hot(target, num_classes=centers.size(0)).T.to(embeddings)
        counts = membership.sum(dim=1)
        denominator = counts.clamp_min(1).view(-1, 1, 1)
        cf_real = (membership @ phases.cos().flatten(1)).view(
            centers.size(0), self.num_slices, t.numel()
        ) / denominator
        cf_imag = (membership @ phases.sin().flatten(1)).view_as(cf_real) / denominator
        error = ((cf_real - gaussian_cf).square() + cf_imag.square()) * gaussian_cf
        ep = counts * torch.trapezoid(error, t, dim=-1).mean(dim=-1)
        return ep * (counts >= 2).to(ep.dtype), counts

    def forward(self, embeddings, target, centers, *, step=0, log_variance=None):
        """Return sum_c (n_c / N) EP_c, differentiable through embeddings."""
        ep, counts = self.statistics(
            embeddings, target, centers, step=step, log_variance=log_variance,
        )
        return (counts / embeddings.size(0) * ep).sum()


class SIGRegNLLLoss(nn.Module):
    """Mean generative NLL plus weighted class-wise SIGReg.

    ``input`` contains (N, C) unnormalized generative log scores; no log_softmax
    is applied. Supply the same (N, D) embeddings and fixed (C, D) LDA centers.
    Advance ``step`` each minibatch to resample the regularizer's directions.
    A zero gaussianity_weight recovers nn.NLLLoss without evaluating SIGReg.
    Supply scalar log_variance=log(sigma^2) for a learned spherical head; SIGReg
    then tests (z - mu_c) / sigma. Omit it for identity covariance.
    """

    def __init__(self, gaussianity_weight=1.0, num_slices=256, seed=1729, *, num_points=17):
        super().__init__()
        self.gaussianity_weight = float(gaussianity_weight)
        if not math.isfinite(self.gaussianity_weight) or self.gaussianity_weight < 0:
            raise ValueError("gaussianity_weight must be finite and nonnegative.")
        self.sigreg = ClasswiseSIGReg(num_slices=num_slices, seed=seed, num_points=num_points)

    def forward(self, input, target, embeddings, centers, *, step=0, log_variance=None):
        if input.ndim != 2 or embeddings.ndim != 2 or input.size(0) != embeddings.size(0):
            raise ValueError("input and embeddings must be matrices with matching batch sizes.")
        if centers.ndim != 2 or centers.size(0) != input.size(1):
            raise ValueError("centers must have one row per input class.")
        nll = nn.functional.nll_loss(input, target)
        if self.gaussianity_weight == 0:
            return nll
        return nll + self.gaussianity_weight * self.sigreg(
            embeddings, target, centers, step=step, log_variance=log_variance,
        )
