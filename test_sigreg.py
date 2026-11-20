"""Numerical SIGReg checks: python -m unittest test_sigreg."""
import math
import unittest
from unittest import mock

import torch

from sigreg import ClasswiseSIGReg, SIGRegNLLLoss
from lda import SimplexLDAHead


class SIGRegTests(unittest.TestCase):
    def test_finer_quadrature_resolves_coarse_grid_translation_alias(self):
        z = torch.randn(512, 1, dtype=torch.double,
                        generator=torch.Generator().manual_seed(11))
        labels = torch.zeros(len(z), dtype=torch.long)
        centers = torch.zeros(1, 1, dtype=z.dtype)
        coarse = ClasswiseSIGReg(num_slices=1, num_points=17).double()
        fine = ClasswiseSIGReg(num_slices=1, num_points=65).double()
        shift = 2 * math.pi / (10 / 16)
        # The original grid has exactly this translation ambiguity in 1D.
        torch.testing.assert_close(coarse(z, labels, centers), coarse(z + shift, labels, centers))
        self.assertGreater(fine(z + shift, labels, centers).item(),
                           100 * fine(z, labels, centers).item())

    def test_dense_quadrature_matches_analytic_gaussian_window_integral(self):
        z = torch.tensor([[-1.2], [-0.5], [0.2], [0.7], [2.3]], dtype=torch.double)
        labels = torch.zeros(len(z), dtype=torch.long)
        centers = torch.zeros(1, 1, dtype=z.dtype)
        # Exact integral over the real line of |ECF - exp(-t^2/2)|^2 exp(-t^2/2).
        pairwise = z - z.T
        expected = (math.sqrt(2 * math.pi) * torch.exp(-0.5 * pairwise.square()).mean()
                    - 2 * math.sqrt(math.pi) * torch.exp(-z.square() / 4).mean()
                    + math.sqrt(2 * math.pi / 3))
        reg = ClasswiseSIGReg(num_slices=1, num_points=129).double()
        actual = reg(z, labels, centers) / len(z)
        # The numerical integral truncates the Gaussian window at +/-5.
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)

    def test_spherical_standardization_and_variance_gradients(self):
        torch.manual_seed(71)
        z = torch.randn(8, 2, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        centers = torch.randn(2, 2, dtype=torch.double)
        log_variance = torch.tensor([0.8], dtype=torch.double, requires_grad=True)
        regularizer = ClasswiseSIGReg(num_slices=11)
        actual = regularizer(z, labels, centers, step=3, log_variance=log_variance)
        inverse_sigma = torch.exp(-0.5 * log_variance)
        expected = regularizer(z * inverse_sigma, labels, centers * inverse_sigma, step=3)
        torch.testing.assert_close(actual, expected)
        actual_gradients = torch.autograd.grad(actual, (z, log_variance))
        expected_gradients = torch.autograd.grad(expected, (z, log_variance))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertNotEqual(actual_gradients[1].item(), 0.)
        self.assertTrue(torch.autograd.gradcheck(
            lambda e, v: regularizer(e, labels, centers, step=3, log_variance=v),
            (z, log_variance)))
        torch.testing.assert_close(regularizer(z, labels, centers), regularizer(
            z, labels, centers, log_variance=torch.zeros_like(log_variance)))

    def test_spherical_nll_updates_shared_variance(self):
        torch.manual_seed(73)
        head = SimplexLDAHead(3, 2, covariance_type='spherical').double()
        head.log_cov.data.fill_(0.4)
        z = torch.randn(9, 2, dtype=torch.double, requires_grad=True)
        labels = torch.arange(3).repeat_interleave(3)
        criterion = SIGRegNLLLoss(0.7, num_slices=11)
        scores = head(z)
        nll = torch.nn.functional.nll_loss(scores, labels)
        sigreg = criterion.sigreg(z, labels, head.mu, log_variance=head.log_cov)
        actual = criterion(scores, labels, z, head.mu, log_variance=head.log_cov)
        torch.testing.assert_close(actual, nll + 0.7 * sigreg)
        self.assertNotEqual(torch.autograd.grad(nll, head.log_cov, retain_graph=True)[0].item(), 0.)
        self.assertNotEqual(torch.autograd.grad(sigreg, head.log_cov, retain_graph=True)[0].item(), 0.)
        optimizer = torch.optim.Adam([head.log_cov])
        initial = head.log_cov.detach().clone()
        optimizer.zero_grad()
        actual.backward()
        optimizer.step()
        self.assertFalse(torch.equal(head.log_cov, initial))
        self.assertTrue(torch.isfinite(head.cov_diag).all() and (head.cov_diag > 0).all())

    def test_nll_combination_preserves_generative_scores(self):
        scores = torch.tensor([[-2., -5.], [-4., -3.], [-1., -2.], [-5., -2.]],
                              dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1])
        z = torch.randn(4, 2, dtype=torch.double, requires_grad=True)
        centers = torch.zeros(2, 2, dtype=torch.double)
        criterion = SIGRegNLLLoss(0.7, num_slices=7)
        actual = criterion(scores, labels, z, centers, step=9)
        expected = scores.new_tensor(2.) + 0.7 * criterion.sigreg(z, labels, centers, step=9)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.autograd.gradcheck(
            lambda s, e: criterion(s, labels, e, centers, step=9), (scores, z)))

    def test_zero_weight_matches_nll_value_and_gradients(self):
        scores = torch.randn(4, 2, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1])
        z = torch.randn(4, 2, dtype=torch.double, requires_grad=True)
        centers = torch.zeros(2, 2, dtype=torch.double)
        criterion = SIGRegNLLLoss(0.)
        with mock.patch.object(criterion.sigreg, 'forward', side_effect=AssertionError):
            actual = criterion(scores, labels, z, centers)
        expected = torch.nn.functional.nll_loss(scores, labels)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, scores)[0],
                                   torch.autograd.grad(expected, scores)[0])

    def test_invalid_gaussianity_weights(self):
        for weight in [-1., float('nan'), float('inf')]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                SIGRegNLLLoss(weight)

    def test_matches_paper_complex_cf_value_and_gradient(self):
        torch.manual_seed(4)
        z = torch.randn(9, 2, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 2, 2, 2, 2, 2, 3])
        centers = torch.randn(5, 2, dtype=torch.double)
        regularizer = ClasswiseSIGReg(num_slices=11, seed=31)
        generator = torch.Generator().manual_seed(31 + 7)
        directions = torch.randn(2, 11, generator=generator, dtype=z.dtype)
        directions = directions / directions.norm(dim=0)
        t = torch.linspace(-5, 5, 17, dtype=z.dtype)
        target_cf = torch.exp(-0.5 * t.square())
        expected = z.sum() * 0
        for c in labels.unique():
            points = z[labels == c] - centers[c]
            if len(points) < 2:
                continue
            phases = (points @ directions).unsqueeze(-1) * t
            ecf = (1j * phases).exp().mean(dim=0)
            ep = len(points) * torch.trapezoid((ecf - target_cf).abs().square() * target_cf, t).mean()
            expected = expected + (len(points) / len(z)) * ep
        actual = regularizer(z, labels, centers, step=7)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, z)[0],
                                   torch.autograd.grad(expected, z)[0])

    def test_gradcheck(self):
        z = torch.randn(6, 2, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        centers = torch.zeros(2, 2, dtype=z.dtype)
        regularizer = ClasswiseSIGReg(num_slices=7)
        self.assertTrue(torch.autograd.gradcheck(lambda x: regularizer(x, labels, centers), (z,)))

    def test_singletons_and_absent_classes(self):
        z = torch.randn(3, 2, requires_grad=True)
        regularizer = ClasswiseSIGReg(num_slices=7)
        value = regularizer(z, torch.arange(3), torch.zeros(5, 2))
        value.backward()
        torch.testing.assert_close(value, torch.zeros(()))
        torch.testing.assert_close(z.grad, torch.zeros_like(z))

    def test_resampling_reproducibility_and_independent_rng(self):
        z = torch.randn(40, 2)
        labels = torch.zeros(40, dtype=torch.long)
        centers = torch.zeros(1, 2)
        regularizer = ClasswiseSIGReg(num_slices=7)
        rng = torch.get_rng_state().clone()
        first = regularizer(z, labels, centers, step=3)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        torch.testing.assert_close(first, regularizer(z, labels, centers, step=3))
        self.assertNotEqual(first.item(), regularizer(z, labels, centers, step=4).item())

    def test_detects_shape_beyond_identity_covariance(self):
        # Paired X distribution: exactly Gaussian coordinate marginals, nearly
        # identity covariance, but non-Gaussian projections along diagonals.
        g = torch.Generator().manual_seed(17)
        a, b = torch.randn(2, 2048, generator=g)
        gaussian = torch.stack([torch.cat([a, a]), torch.cat([b, -b])], dim=1)
        x_shape = torch.stack([torch.cat([a, a]), torch.cat([a, -a])], dim=1)
        labels = torch.zeros(len(gaussian), dtype=torch.long)
        centers = torch.zeros(1, 2)
        regularizer = ClasswiseSIGReg(num_slices=64)
        self.assertGreater(regularizer(x_shape, labels, centers).item(),
                           10 * regularizer(gaussian, labels, centers).item())
        self.assertGreater(regularizer(torch.zeros_like(gaussian), labels, centers).item(),
                           regularizer(gaussian, labels, centers).item())
        self.assertGreater(regularizer(gaussian + 2, labels, centers).item(),
                           regularizer(gaussian, labels, centers).item())

    def test_classes_are_separate_and_centers_are_fixed(self):
        g = torch.Generator().manual_seed(11)
        residuals = torch.randn(200, 2, generator=g)
        labels = torch.arange(2).repeat_interleave(100)
        centers = torch.tensor([[-3., 0.], [3., 0.]])
        z = residuals + centers[labels]
        regularizer = ClasswiseSIGReg(num_slices=19)
        torch.testing.assert_close(regularizer(z, labels, centers),
                                   regularizer(residuals, labels, torch.zeros_like(centers)))
        ep, counts = regularizer.statistics(z, labels, centers)
        torch.testing.assert_close(regularizer(z, labels, centers), (ep * counts / len(z)).sum())


if __name__ == '__main__':
    unittest.main()
