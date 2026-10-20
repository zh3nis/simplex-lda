import math
import torch
import torch.nn as nn


class SimplexLDAHead(nn.Module):
    """Fixed-mean LDA classifier with configurable covariance and trainable priors.

    covariance_type="identity" fixes Sigma to I with no covariance parameters.
    The spherical, diag, and full options use trainable covariance parameters.
    """

    def __init__(self, C, D, covariance_type="spherical", min_scale=1e-4):
        super().__init__()
        if D < C - 1:
            raise ValueError(f"D must be at least C-1 to embed the simplex (got C={C}, D={D}).")
        self.C = C
        self.D = D
        cov_type = str(covariance_type).lower()
        if cov_type not in {"identity", "spherical", "diag", "full"}:
            raise ValueError(
                "covariance_type must be one of {'identity', 'spherical', 'diag', 'full'} "
                f"(got {covariance_type!r})."
            )
        self.covariance_type = cov_type
        self.min_scale = min_scale
        dtype = torch.get_default_dtype()
        mu = self._regular_simplex_vertices(C, D, dtype=dtype)
        pairwise_dist = math.sqrt(2.0 * C / (C - 1))
        scale = 6.0 / pairwise_dist
        mu = mu * scale
        self.register_buffer('mu', mu)
        if self.covariance_type == "spherical":
            self.log_cov = nn.Parameter(torch.zeros(1, dtype=dtype))
        elif self.covariance_type == "diag":
            self.log_cov_diag = nn.Parameter(torch.zeros(D, dtype=dtype))
        elif self.covariance_type == "full":
            self.raw_tril = nn.Parameter(torch.zeros(D, D, dtype=dtype))
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))

    @staticmethod
    def _regular_simplex_vertices(C, D, dtype):
        """Construct vertices of a regular simplex centered at the origin."""
        eye = torch.eye(C, dtype=dtype)
        centered = eye - eye.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh.transpose(-2, -1)[:, :C - 1]
        simplex = centered @ basis
        simplex = simplex / simplex.norm(dim=1, keepdim=True)
        if D > C - 1:
            pad = torch.zeros(C, D - (C - 1), dtype=dtype)
            simplex = torch.cat([simplex, pad], dim=1)
        return simplex

    @property
    def cov_diag(self):
        if self.covariance_type == "identity":
            return self.mu.new_ones(self.D)
        if self.covariance_type == "spherical":
            return torch.exp(self.log_cov).repeat(self.D)
        if self.covariance_type == "diag":
            return torch.exp(self.log_cov_diag)
        return torch.diagonal(self.covariance)

    @property
    def covariance(self):
        """Return fixed identity covariance or learned full covariance Sigma = L L^T."""
        if self.covariance_type == "identity":
            return torch.eye(self.D, dtype=self.mu.dtype, device=self.mu.device)
        if self.covariance_type != "full":
            raise AttributeError("covariance is only defined for identity or full covariance.")
        L = self._get_cholesky(self.raw_tril.dtype, self.raw_tril.device)
        return L @ L.transpose(-2, -1)

    def _get_cholesky(self, dtype, device):
        raw = torch.tril(self.raw_tril.to(device=device, dtype=dtype))
        diag = torch.diagonal(raw, 0)
        safe_diag = torch.nn.functional.softplus(diag) + self.min_scale
        L = raw - torch.diag(diag) + torch.diag(safe_diag)
        return L

    def forward(self, z):
        if self.covariance_type == "full":
            dtype = z.dtype
            device = z.device
            mu = self.mu.to(device=device, dtype=dtype)
            diff = z.unsqueeze(1) - mu.unsqueeze(0)
            L = self._get_cholesky(dtype, device)
            diff_flat = diff.reshape(-1, self.D).transpose(0, 1)
            solved = torch.linalg.solve_triangular(L, diff_flat, upper=False)
            m2 = (solved * solved).sum(dim=0).reshape(z.shape[0], self.C)
            log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
            log_prior = torch.log_softmax(
                self.prior_logits.to(device=device, dtype=dtype), dim=0
            )
            return log_prior.unsqueeze(0) - 0.5 * (m2 + log_det)

        mu = self.mu.to(z.dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        if self.covariance_type == "identity":
            # Sigma^{-1} = I and log det(Sigma) = 0.
            m2 = (diff * diff).sum(-1)
            log_prior = torch.log_softmax(self.prior_logits, dim=0)
            return log_prior.unsqueeze(0) - 0.5 * m2
        if self.covariance_type == "spherical":
            m2 = (diff * diff).sum(-1)
            log_cov = self.log_cov.to(z.dtype)
            var = torch.exp(log_cov)
            log_det = self.D * log_cov
            log_prior = torch.log_softmax(self.prior_logits, dim=0)
            return log_prior.unsqueeze(0) - 0.5 * (m2 / var + log_det)

        log_cov_diag = self.log_cov_diag.to(z.dtype)
        var = torch.exp(log_cov_diag)
        m2 = (diff * diff / var).sum(-1)
        log_det = log_cov_diag.sum()
        log_prior = torch.log_softmax(self.prior_logits, dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 + log_det)


class SimplexQDAHead(nn.Module):
    """Fixed-mean QDA classifier with per-class covariance and trainable priors."""

    def __init__(self, C, D, covariance_type="spherical", min_scale=1e-4):
        super().__init__()
        if D < C - 1:
            raise ValueError(f"D must be at least C-1 to embed the simplex (got C={C}, D={D}).")
        self.C = C
        self.D = D
        cov_type = str(covariance_type).lower()
        if cov_type not in {"spherical", "diag", "full"}:
            raise ValueError(
                "covariance_type must be one of {'spherical', 'diag', 'full'} "
                f"(got {covariance_type!r})."
            )
        self.covariance_type = cov_type
        self.min_scale = min_scale
        dtype = torch.get_default_dtype()
        mu = self._regular_simplex_vertices(C, D, dtype=dtype)
        pairwise_dist = math.sqrt(2.0 * C / (C - 1))
        scale = 6.0 / pairwise_dist
        mu = mu * scale
        self.register_buffer('mu', mu)
        if self.covariance_type == "spherical":
            self.log_cov = nn.Parameter(torch.zeros(C, dtype=dtype))
        elif self.covariance_type == "diag":
            self.log_cov_diag = nn.Parameter(torch.zeros(C, D, dtype=dtype))
        else:
            self.raw_tril = nn.Parameter(torch.zeros(C, D, D, dtype=dtype))
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))

    @staticmethod
    def _regular_simplex_vertices(C, D, dtype):
        """Construct vertices of a regular simplex centered at the origin."""
        eye = torch.eye(C, dtype=dtype)
        centered = eye - eye.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh.transpose(-2, -1)[:, :C - 1]
        simplex = centered @ basis
        simplex = simplex / simplex.norm(dim=1, keepdim=True)
        if D > C - 1:
            pad = torch.zeros(C, D - (C - 1), dtype=dtype)
            simplex = torch.cat([simplex, pad], dim=1)
        return simplex

    @property
    def cov_diag(self):
        if self.covariance_type == "spherical":
            return torch.exp(self.log_cov).unsqueeze(1).repeat(1, self.D)
        if self.covariance_type == "diag":
            return torch.exp(self.log_cov_diag)
        return torch.diagonal(self.covariance, dim1=-2, dim2=-1)

    @property
    def covariance(self):
        """Per-class full covariance matrices Sigma_k = L_k L_k^T."""
        if self.covariance_type != "full":
            raise AttributeError("covariance is only defined for full covariance.")
        L = self._get_cholesky(self.raw_tril.dtype, self.raw_tril.device)
        return L @ L.transpose(-2, -1)

    def _get_cholesky(self, dtype, device):
        raw = torch.tril(self.raw_tril.to(device=device, dtype=dtype))
        diag = torch.diagonal(raw, dim1=-2, dim2=-1)
        safe_diag = torch.nn.functional.softplus(diag) + self.min_scale
        # Avoid in-place updates on views to keep autograd happy.
        return raw - torch.diag_embed(diag) + torch.diag_embed(safe_diag)

    def forward(self, z):
        if self.covariance_type == "full":
            dtype = z.dtype
            device = z.device
            mu = self.mu.to(device=device, dtype=dtype)
            diff = z.unsqueeze(1) - mu.unsqueeze(0)
            L = self._get_cholesky(dtype, device)
            diff_trans = diff.transpose(0, 1)
            diff_flat = diff_trans.transpose(1, 2)
            solved = torch.linalg.solve_triangular(L, diff_flat, upper=False)
            m2 = (solved * solved).sum(dim=1).transpose(0, 1)
            log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=1)
            log_prior = torch.log_softmax(
                self.prior_logits.to(device=device, dtype=dtype), dim=0
            )
            return log_prior.unsqueeze(0) - 0.5 * (m2 + log_det.unsqueeze(0))

        mu = self.mu.to(z.dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        if self.covariance_type == "spherical":
            m2 = (diff * diff).sum(-1)
            log_cov = self.log_cov.to(z.dtype)
            var = torch.exp(log_cov)
            log_det = self.D * log_cov
            log_prior = torch.log_softmax(self.prior_logits, dim=0)
            return log_prior.unsqueeze(0) - 0.5 * (m2 / var + log_det)

        log_cov_diag = self.log_cov_diag.to(z.dtype)
        var = torch.exp(log_cov_diag)
        m2 = (diff * diff / var).sum(-1)
        log_det = log_cov_diag.sum(dim=1)
        log_prior = torch.log_softmax(self.prior_logits, dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 + log_det)


class FisherSimplexLDAHead(nn.Module):
    """
    Fixed-mean spherical LDA head trained with Fisher's discriminant criterion.

    Forward returns the negative Fisher ratio for a labeled batch. Use `logits`
    for evaluation-time class logits identical to SimplexLDAHead.
    """

    def __init__(self, C, D, fisher_eps=1e-8, prior_strength=0.5):
        super().__init__()
        if D < C - 1:
            raise ValueError(f"D must be at least C-1 to embed the simplex (got C={C}, D={D}).")
        if not (0.0 <= prior_strength <= 1.0):
            raise ValueError(f"prior_strength must be in [0,1] (got {prior_strength}).")
        self.C = C
        self.D = D
        self.fisher_eps = fisher_eps
        self.prior_strength = prior_strength
        dtype = torch.get_default_dtype()
        mu = self._regular_simplex_vertices(C, D, dtype=dtype)
        pairwise_dist = math.sqrt(2.0 * C / (C - 1))
        scale = 6.0 / pairwise_dist
        mu = mu * scale
        self.register_buffer('mu', mu)
        self.log_cov = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))

    @staticmethod
    def _regular_simplex_vertices(C, D, dtype):
        eye = torch.eye(C, dtype=dtype)
        centered = eye - eye.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh.transpose(-2, -1)[:, :C - 1]
        simplex = centered @ basis
        simplex = simplex / simplex.norm(dim=1, keepdim=True)
        if D > C - 1:
            pad = torch.zeros(C, D - (C - 1), dtype=dtype)
            simplex = torch.cat([simplex, pad], dim=1)
        return simplex

    @property
    def cov_diag(self):
        return torch.exp(self.log_cov).repeat(self.D)

    def logits(self, z):
        """Compute LDA logits for evaluation (same form as SimplexLDAHead.forward)."""
        mu = self.mu.to(z.dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        m2 = (diff * diff).sum(-1)
        var = torch.exp(self.log_cov).to(z.dtype)
        log_det = self.D * self.log_cov.to(z.dtype)
        log_prior = torch.log_softmax(self.prior_logits, dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 / var + log_det)

    def forward(self, z, y):
        """
        Return negative Fisher ratio for the batch.

        z: (B, D) embeddings
        y: (B,) labels in [0, C)
        """
        if y is None:
            raise ValueError("Labels y must be provided to compute the Fisher criterion.")
        if y.dim() != 1:
            raise ValueError(f"Expected y to be 1D (got shape {tuple(y.shape)}).")
        if z.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched batch sizes: z has {z.shape[0]}, y has {y.shape[0]}.")

        dtype = z.dtype
        device = z.device
        mu = self.mu.to(device=device, dtype=dtype)
        var = torch.exp(self.log_cov.to(device=device, dtype=dtype))

        # Within-class scatter measured by Mahalanobis distance to fixed class means.
        diff = z - mu[y]
        within = (diff.pow(2).sum(dim=1) / var).mean()

        # Between-class scatter of the fixed means around their prior-weighted centroid.
        counts = torch.bincount(y, minlength=self.C).to(device=device, dtype=dtype)
        total = counts.sum().clamp_min(1.0)
        data_pi = counts / total
        learned_pi = torch.softmax(self.prior_logits.to(device=device, dtype=dtype), dim=0)
        pi = self.prior_strength * learned_pi + (1.0 - self.prior_strength) * data_pi

        overall_mu = (pi.unsqueeze(1) * mu).sum(dim=0)
        centered = mu - overall_mu
        between_per_class = centered.pow(2).sum(dim=1) / var
        between = (pi * between_per_class).sum()

        fisher_ratio = between / (within + self.fisher_eps)
        return -fisher_ratio

def dnll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    lambda_reg: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    DNLL: Discriminative Negative Log-Likelihood

        L(x, y) = -input_y(x) + λ * sum_c exp(input_c(x))

    Applicable to any generative classifier with class-wise
    (unnormalized) log-density or log-joint scores.

    Args:
        input:  Tensor (N, C) of class scores δ_c(x).
        target: LongTensor (N,) with class indices in [0, C-1].
        lambda_reg: float ≥ 0, strength of discriminative penalty.
        reduction: "none" | "mean" | "sum".

    Returns:
        Loss reduced according to `reduction`.
    """
    # NLL part: -δ_y(x)
    nll = -input.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)

    # Discriminative penalty: λ * ∑_c exp(δ_c(x))
    reg = lambda_reg * input.exp().sum(dim=1)               # (N,)

    loss = nll + reg                                        # (N,)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

class DNLLLoss(nn.Module):
    r"""
    DNLL: Discriminative Negative Log-Likelihood

        L(x, y) = -input_y(x) + λ * sum_c exp(input_c(x))

    A drop-in loss module similar to nn.CrossEntropyLoss, but designed
    for generative classifiers whose outputs are log-density scores.
    """
    def __init__(self, lambda_reg: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.lambda_reg = float(lambda_reg)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dnll_loss(
            input=input,
            target=target,
            lambda_reg=self.lambda_reg,
            reduction=self.reduction,
        )


def class_covariance_penalty(
    embeddings: torch.Tensor, target: torch.Tensor, num_classes=None
) -> torch.Tensor:
    """Return sum_c (n_c / N) ||sample_cov(z_c) - I||_F^2.

    Covariances are centered on each class's empirical embedding mean and use
    denominator n_c - 1. Classes with fewer than two examples contribute zero;
    remaining batch-frequency weights are not renormalized. Targets are zero-based
    class indices. Gradients propagate through the class means and covariances.

    Supply num_classes to avoid discovering present classes with torch.unique.
    CovNLLLoss supplies it from the score matrix shape. All covariances are computed
    with one batched matrix multiplication, using O(C * N * D) temporary storage.
    """
    if embeddings.ndim != 2 or embeddings.size(0) == 0:
        raise ValueError("embeddings must have nonempty shape (N, D).")
    if target.ndim != 1 or target.size(0) != embeddings.size(0):
        raise ValueError("target must have shape (N,) matching embeddings.")
    if num_classes is None:
        classes, target = torch.unique(target, return_inverse=True)
        num_classes = classes.numel()

    # Membership has shape (C, N). The training path needs no dynamic indexing
    # or Python loop over classes, avoiding repeated small GPU kernel launches.
    membership = nn.functional.one_hot(target, num_classes=num_classes).T.to(embeddings.dtype)
    counts = membership.sum(dim=1)
    means = (membership @ embeddings) / counts.clamp_min(1).unsqueeze(1)
    centered = (embeddings.unsqueeze(0) - means.unsqueeze(1)) * membership.unsqueeze(2)
    covariance = torch.bmm(centered.transpose(1, 2), centered)
    covariance = covariance / (counts - 1).clamp_min(1).view(-1, 1, 1)

    identity = torch.eye(
        embeddings.size(1), dtype=embeddings.dtype, device=embeddings.device
    )
    squared_errors = (covariance - identity).square().sum(dim=(1, 2))
    weights = (counts / embeddings.size(0)) * (counts >= 2).to(embeddings.dtype)
    return (weights * squared_errors).sum()


class CovNLLLoss(nn.Module):
    """Mean NLL plus a class-wise penalty toward identity embedding covariance.

    input: (N, C) log probabilities or unnormalized generative log scores.
    target: (N,) zero-based class labels.
    embeddings: (N, D) embeddings from the same examples, on the same device.

    Like nn.NLLLoss, the NLL term is -mean(input[n, target[n]]); no log_softmax
    is applied. This preserves generative-score training in the LDA notebooks.
    Only mean reduction is used, keeping the scale of the covariance weight
    independent of batch size. Matching covariance does not ensure Gaussianity.
    """

    def __init__(self, covariance_weight: float = 1.0):
        super().__init__()
        self.covariance_weight = float(covariance_weight)
        if not math.isfinite(self.covariance_weight) or self.covariance_weight < 0:
            raise ValueError("covariance_weight must be finite and nonnegative.")

    def forward(self, input, target, embeddings):
        """Return the scalar mean NLL plus weighted covariance penalty."""
        if input.ndim != 2 or embeddings.ndim != 2 or input.size(0) != embeddings.size(0):
            raise ValueError("input and embeddings must be matrices with matching batch sizes.")
        nll = nn.functional.nll_loss(input, target)
        penalty = (
            class_covariance_penalty(embeddings, target, num_classes=input.size(1))
            if self.covariance_weight != 0 else embeddings.sum() * 0.0
        )
        return nll + self.covariance_weight * penalty
