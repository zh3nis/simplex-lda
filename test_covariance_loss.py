"""Numerical checks: python -m unittest test_covariance_loss (requires PyTorch)."""
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    raise unittest.SkipTest('PyTorch is required for numerical loss tests') from exc

from lda import CovNLLLoss, class_covariance_penalty


class CovarianceLossTests(unittest.TestCase):
    def test_identity_covariance_and_translation(self):
        a = (1.5) ** 0.5
        z = torch.tensor([[a, 0.], [-a, 0.], [0., a], [0., -a]], dtype=torch.double)
        labels = torch.zeros(4, dtype=torch.long)
        torch.testing.assert_close(class_covariance_penalty(z, labels), z.new_zeros(()), atol=1e-12, rtol=0)
        torch.testing.assert_close(class_covariance_penalty(z + 12, labels), z.new_zeros(()), atol=1e-12, rtol=0)

    def test_frequency_weights_and_singletons(self):
        z = torch.tensor([[-1.], [1.], [3.], [3.], [3.], [99.]], dtype=torch.double)
        labels = torch.tensor([0, 0, 1, 1, 1, 2])
        torch.testing.assert_close(class_covariance_penalty(z, labels), z.new_tensor(5 / 6))

    def test_all_singletons_have_zero_gradient(self):
        z = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
        penalty = class_covariance_penalty(z, torch.arange(3))
        penalty.backward()
        torch.testing.assert_close(penalty, z.new_zeros(()))
        torch.testing.assert_close(z.grad, torch.zeros_like(z))

    def test_total_loss_and_no_implicit_softmax(self):
        scores = torch.tensor([[-2., -5.], [-4., -3.], [-1., -2.]], dtype=torch.double)
        labels = torch.tensor([0, 1, 0])
        z = torch.tensor([[-1., 0.], [0., 2.], [1., 0.]], dtype=torch.double)
        criterion = CovNLLLoss(covariance_weight=2.5)
        total = criterion(scores, labels, z)
        nll = scores.new_tensor(2.0)
        self.assertEqual(total.ndim, 0)
        torch.testing.assert_close(total, nll + 2.5 * class_covariance_penalty(z, labels))
        torch.testing.assert_close(CovNLLLoss(0)(scores, labels, z), nll)

    def test_gradients_through_scores_and_embeddings(self):
        torch.manual_seed(17)
        z = torch.randn(6, 2, dtype=torch.double, requires_grad=True)
        scores = torch.randn(6, 2, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        criterion = CovNLLLoss(0.7)
        self.assertTrue(torch.autograd.gradcheck(lambda x, e: criterion(x, labels, e), (scores, z)))

    def test_batched_penalty_matches_loop_value_and_gradient(self):
        torch.manual_seed(23)
        # Uneven class counts, an absent class (1), and a singleton class (4).
        labels = torch.tensor([0, 0, 0, 2, 2, 3, 3, 3, 3, 4])
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                target = labels.to(device)
                z = (torch.randn(10, 9, dtype=torch.double, device=device) + 20).requires_grad_()
                expected = z.sum() * 0.0
                for c in target.unique():
                    points = z[target == c]
                    if len(points) < 2:
                        continue
                    centered = points - points.mean(0)
                    cov = centered.T @ centered / (len(points) - 1)
                    expected = expected + (len(points) / len(z)) * (
                        cov - torch.eye(9, dtype=z.dtype, device=device)
                    ).square().sum()
                actual = class_covariance_penalty(z, target, num_classes=6)
                torch.testing.assert_close(actual, expected)
                expected_gradient, = torch.autograd.grad(expected, z)
                actual_gradient, = torch.autograd.grad(actual, z)
                torch.testing.assert_close(actual_gradient, expected_gradient)
                torch.testing.assert_close(class_covariance_penalty(z, target), expected)

    def test_invalid_weights(self):
        for weight in [-1, float('nan'), float('inf')]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                CovNLLLoss(weight)


if __name__ == '__main__':
    unittest.main()
