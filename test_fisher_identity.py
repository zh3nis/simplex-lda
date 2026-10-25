"""Numerical checks: python -m unittest test_fisher_identity."""

import unittest

import torch

from lda import FisherSimplexLDAHead, SimplexLDAHead


class FisherIdentityTests(unittest.TestCase):
    def test_ratio_with_uneven_counts_and_absent_class(self):
        head = FisherSimplexLDAHead(3, 2, covariance_type="identity").double()
        with torch.no_grad():
            head.prior_logits.copy_(torch.tensor([0.2, -0.7, 1.1]))
        y = torch.tensor([0, 0, 2])
        offsets = torch.tensor([[1., 0.], [0., 2.], [1., 1.]], dtype=torch.double)
        z = head.mu[y] + offsets
        pi = 0.5 * head.prior_logits.softmax(0) + 0.5 * z.new_tensor([2 / 3, 0, 1 / 3])
        # The fixed simplex has equal squared radius and zero centroid.
        between = head.mu[0].square().sum() - (pi @ head.mu).square().sum()
        within = z.new_tensor(7 / 3)
        torch.testing.assert_close(head(z, y), -between / (within + head.fisher_eps))

    def test_matches_notebook_spherical_at_unit_variance_and_gradients(self):
        torch.manual_seed(7)
        identity = FisherSimplexLDAHead(4, 3, covariance_type="identity").double()
        spherical = FisherSimplexLDAHead(4, 3).double()
        with torch.no_grad():
            identity.prior_logits.normal_()
            spherical.prior_logits.copy_(identity.prior_logits)
        z = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
        y = torch.tensor([0, 0, 0, 1, 2, 2, 3])
        loss = identity(z, y)
        reference = spherical(z, y)
        torch.testing.assert_close(loss, reference)
        actual_grad = torch.autograd.grad(loss, (z, identity.prior_logits))
        expected_grad = torch.autograd.grad(reference, (z, spherical.prior_logits))
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected)
        self.assertGreater(actual_grad[0].norm().item(), 0)
        self.assertGreater(actual_grad[1].norm().item(), 0)
        torch.testing.assert_close(identity.logits(z), spherical.logits(z))

    def test_identity_stays_fixed_after_optimization(self):
        head = FisherSimplexLDAHead(3, 2, covariance_type="identity")
        self.assertEqual(set(dict(head.named_parameters())), {"prior_logits"})
        encoder = torch.nn.Linear(5, 2)
        opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()))
        x, y = torch.randn(6, 5), torch.tensor([0, 0, 0, 1, 2, 2])
        for _ in range(2):
            opt.zero_grad()
            head(encoder(x), y).backward()
            opt.step()
        torch.testing.assert_close(head.cov_diag, torch.ones(2))
        likelihood = SimplexLDAHead(3, 2, covariance_type="identity")
        likelihood.load_state_dict(head.state_dict())
        torch.testing.assert_close(head.logits(encoder(x)), likelihood(encoder(x)))


if __name__ == "__main__":
    unittest.main()
