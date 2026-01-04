import math
import torch
import torch.nn as nn


class SimplexLDAHead(nn.Module):
    """Fixed-mean LDA classifier with spherical covariance and trainable priors."""

    def __init__(self, C, D):
        super().__init__()
        if D < C - 1:
            raise ValueError(f"D must be at least C-1 to embed the simplex (got C={C}, D={D}).")
        self.C = C
        self.D = D
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
        """Return diagonal of the spherical covariance matrix."""
        return torch.exp(self.log_cov).repeat(self.D)

    def forward(self, z):
        mu = self.mu.to(z.dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        m2 = (diff * diff).sum(-1)
        var = torch.exp(self.log_cov).to(z.dtype)
        log_det = self.D * self.log_cov.to(z.dtype)
        log_prior = torch.log_softmax(self.prior_logits, dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 / var + log_det)


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
