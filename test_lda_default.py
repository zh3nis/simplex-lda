"""Default fixed-covariance checks: python -m unittest test_lda_default."""

import math
import unittest

import torch

from lda import SimplexLDAHead


class LDADefaultTests(unittest.TestCase):
    def test_covariance_uses_embedding_dimension_and_has_no_parameters(self):
        for classes, dimension in [(2, 1), (3, 2), (3, 7), (10, 9)]:
            with self.subTest(classes=classes, dimension=dimension):
                head = SimplexLDAHead(classes, dimension).double()
                self.assertEqual(head.covariance_type, 'fixed')
                self.assertEqual(set(dict(head.named_parameters())), {'prior_logits'})
                torch.testing.assert_close(head.covariance, torch.eye(dimension, dtype=torch.double) / dimension)
                torch.testing.assert_close(head.cov_diag, torch.full((dimension,), 1 / dimension, dtype=torch.double))
                torch.testing.assert_close(head.covariance.trace(), torch.ones((), dtype=torch.double))

    def test_scores_and_gradients_match_gaussian_density(self):
        torch.manual_seed(13)
        head = SimplexLDAHead(3, 7).double()
        with torch.no_grad():
            head.prior_logits.copy_(torch.tensor([0.3, -0.7, 1.2]))
        z = torch.randn(5, 7, dtype=torch.double, requires_grad=True)
        normal = torch.distributions.MultivariateNormal(head.mu, covariance_matrix=head.covariance)
        expected = (normal.log_prob(z[:, None, :]) + head.prior_logits.log_softmax(0)
                    + 0.5 * head.D * math.log(2 * math.pi))
        actual = head(z)
        torch.testing.assert_close(actual, expected)
        weights = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad((actual * weights).sum(), (z, head.prior_logits))
        expected_gradients = torch.autograd.grad((expected * weights).sum(), (z, head.prior_logits))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_covariance_stays_fixed_after_training_and_state_roundtrip(self):
        torch.manual_seed(17)
        encoder = torch.nn.Linear(4, 7)
        head = SimplexLDAHead(3, 7)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()))
        initial_covariance, initial_means = head.covariance.clone(), head.mu.clone()
        x, y = torch.randn(6, 4), torch.tensor([0, 0, 0, 1, 1, 2])
        for _ in range(3):
            optimizer.zero_grad()
            torch.nn.functional.nll_loss(head(encoder(x)), y).backward()
            optimizer.step()
        torch.testing.assert_close(head.covariance, initial_covariance, atol=0, rtol=0)
        torch.testing.assert_close(head.mu, initial_means, atol=0, rtol=0)
        restored = SimplexLDAHead(3, 7)
        restored.load_state_dict(head.state_dict(), strict=True)
        torch.testing.assert_close(restored(encoder(x)), head(encoder(x)))
        torch.testing.assert_close(restored.covariance, initial_covariance)

    def test_explicit_fixed_matches_default_on_available_devices(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                default = SimplexLDAHead(3, 7).double().to(device)
                explicit = SimplexLDAHead(3, 7, covariance_type='fixed').double().to(device)
                explicit.load_state_dict(default.state_dict())
                z = torch.randn(4, 7, dtype=torch.double, device=device)
                torch.testing.assert_close(default(z), explicit(z))
                self.assertEqual(default.cov_diag.device, z.device)
                self.assertEqual(default.covariance.dtype, z.dtype)

    def test_explicit_covariance_modes_preserve_their_behavior(self):
        z = torch.randn(4, 7, dtype=torch.double)
        identity = SimplexLDAHead(3, 7, covariance_type='identity').double()
        spherical = SimplexLDAHead(3, 7, covariance_type='spherical').double()
        diagonal = SimplexLDAHead(3, 7, covariance_type='diag').double()
        torch.testing.assert_close(identity.covariance, torch.eye(7, dtype=torch.double))
        torch.testing.assert_close(identity(z), spherical(z))
        torch.testing.assert_close(identity(z), diagonal(z))
        self.assertTrue(spherical.log_cov.requires_grad)
        self.assertTrue(diagonal.log_cov_diag.requires_grad)
        full = SimplexLDAHead(3, 7, covariance_type='full').double()
        normal = torch.distributions.MultivariateNormal(full.mu, covariance_matrix=full.covariance)
        expected = (normal.log_prob(z[:, None, :]) + full.prior_logits.log_softmax(0)
                    + 0.5 * full.D * math.log(2 * math.pi))
        torch.testing.assert_close(full(z), expected)
        self.assertTrue(full.raw_tril.requires_grad)


if __name__ == '__main__':
    unittest.main()
