"""Scale geometry checks: python3 -m unittest test_simplex_scale."""
import unittest

import torch

from lda import SimplexLDAHead


class SimplexScaleTests(unittest.TestCase):
    def test_pairwise_distance_and_center_for_padded_simplex(self):
        for s in (0.5, 6.0, 12.0):
            with self.subTest(s=s):
                head = SimplexLDAHead(4, 7, covariance_type="identity", s=s)
                torch.testing.assert_close(torch.pdist(head.mu), torch.full((6,), s))
                torch.testing.assert_close(head.mu.mean(0), torch.zeros(7), atol=1e-6, rtol=0)
                torch.testing.assert_close(head.covariance, torch.eye(7))
                self.assertEqual(set(dict(head.named_parameters())), {"prior_logits"})

    def test_default_and_state_dict_compatibility(self):
        default = SimplexLDAHead(4, 3, covariance_type="identity")
        explicit = SimplexLDAHead(4, 3, covariance_type="identity", s=6.0)
        torch.testing.assert_close(default.mu, explicit.mu)
        scaled = SimplexLDAHead(4, 3, covariance_type="identity", s=12.0)
        torch.testing.assert_close(scaled.mu, 2 * default.mu)
        # Existing checkpoints still define geometry through their mu buffer.
        scaled.load_state_dict(default.state_dict(), strict=True)
        z = torch.randn(5, 3, requires_grad=True)
        torch.testing.assert_close(scaled(z), default(z))
        torch.nn.NLLLoss()(scaled(z), torch.tensor([0, 1, 2, 3, 0])).backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_invalid_scales(self):
        for s in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(s=s), self.assertRaisesRegex(ValueError, "finite and positive"):
                SimplexLDAHead(4, 3, s=s)


if __name__ == "__main__":
    unittest.main()
