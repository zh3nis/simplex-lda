"""Paper-equation and synthetic training checks: python -m unittest test_mm_lda."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_mm_lda import DeepMMLDA, MMLDAHead, main
from train_deep_lda_covariances import Encoder


class MMLDATests(unittest.TestCase):
    def test_means_satisfy_eq_4_and_maximal_pairwise_distance(self):
        for C, D in [(2, 1), (10, 9), (10, 16), (100, 99)]:
            for norm_squared in (1.0, 100.0):
                head = MMLDAHead(C, D, mean_norm_squared=norm_squared)
                mu = head.mu.double()
                gram = torch.full((C, C), -norm_squared / (C - 1), dtype=torch.double)
                gram.fill_diagonal_(norm_squared)
                torch.testing.assert_close(mu @ mu.T, gram, rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(mu.sum(0), torch.zeros(D, dtype=torch.double),
                                           rtol=0, atol=2e-6)
                expected_distance = math.sqrt(2 * C * norm_squared / (C - 1))
                torch.testing.assert_close(torch.pdist(mu),
                                           torch.full((C * (C - 1) // 2,), expected_distance,
                                                      dtype=torch.double))
                self.assertEqual(list(head.parameters()), [])
                if D > C - 1:
                    self.assertEqual(mu[:, C - 1:].count_nonzero().item(), 0)

    def test_posterior_loss_and_gradients_match_eq_5(self):
        torch.manual_seed(9)
        for priors in (None, [0.1, 0.2, 0.3, 0.4]):
            head = MMLDAHead(4, 3, priors=priors).double()
            z = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
            labels = torch.arange(7) % 4
            logits = head(z)
            # Independent reference: full Gaussian class-conditional densities.
            normal = torch.distributions.Normal(head.mu, torch.ones_like(head.mu))
            joint = normal.log_prob(z[:, None, :]).sum(-1) + head.log_priors
            torch.testing.assert_close(logits.softmax(1), joint.softmax(1))
            loss = F.cross_entropy(logits, labels)
            reference = -(joint - joint.logsumexp(1, keepdim=True))[torch.arange(7), labels].mean()
            torch.testing.assert_close(loss, reference)
            actual_grad, = torch.autograd.grad(loss, z, retain_graph=True)
            expected_grad, = torch.autograd.grad(reference, z)
            torch.testing.assert_close(actual_grad, expected_grad)
            self.assertTrue(torch.isfinite(actual_grad).all())
            self.assertGreater(actual_grad.norm().item(), 0)
            # The fixed prior is recovered at the origin; features have no L2 normalization.
            torch.testing.assert_close(head(torch.zeros(1, 3, dtype=torch.double)).softmax(1)[0],
                                       head.log_priors.exp(), rtol=1e-5, atol=1e-6)

    def test_encoder_matches_baseline_and_head_remains_fixed_after_step(self):
        torch.manual_seed(1338)
        reference = Encoder(9, 1)
        torch.manual_seed(1338)
        model = DeepMMLDA(10, 9, 1)
        self.assertIs(type(model.encoder), Encoder)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.encoder.state_dict()[name], value, rtol=0, atol=0)
        before = {name: value.clone() for name, value in model.head.state_dict().items()}
        # Exercise the encoder projection and head without an expensive CNN pass.
        opt = torch.optim.Adam(model.encoder.parameters())
        loss = F.cross_entropy(model.head(model.encoder.proj(torch.randn(4, 256))), torch.arange(4))
        loss.backward()
        self.assertTrue(torch.isfinite(model.encoder.proj.weight.grad).all())
        opt.step()
        self.assertFalse(torch.equal(model.encoder.proj.weight, reference.proj.weight))
        for name, value in before.items():
            torch.testing.assert_close(model.head.state_dict()[name], value, rtol=0, atol=0)

    def test_invalid_configuration(self):
        for C, D in [(1, 1), (10, 8)]:
            with self.assertRaises(ValueError):
                MMLDAHead(C, D)
        for norm_squared in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                MMLDAHead(3, 2, mean_norm_squared=norm_squared)
        for priors in ([0.5, 0.5], [0, 0.5, 0.5], [-1, 1, 1],
                       [1, 1, 1], [float("nan"), 0.5, 0.5]):
            with self.assertRaises(ValueError):
                MMLDAHead(3, 2, priors=priors)

    def test_end_to_end_all_datasets_with_synthetic_images(self):
        def fake_loaders(name, data_root, batch_size, test_batch_size):
            channels, side = (1, 28) if name == "FashionMNIST" else (3, 32)
            ds = TensorDataset(torch.randn(4, channels, side, side), torch.arange(4))
            return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4), channels

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "results.json"
                argv = ["train_mm_lda.py", "--epochs", "1", "--runs", "2", "--output", str(output)]
                with patch("sys.argv", argv), patch("train_mm_lda.build_loaders", fake_loaders), \
                     patch("torch.cuda.is_available", return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()):
                    main()
                result = json.loads(output.read_text())
                self.assertEqual(set(result["results"]), {"FashionMNIST", "CIFAR10", "CIFAR100"})
                self.assertEqual(result["meta"]["mean_norm_squared"], 100)
                self.assertEqual(result["meta"]["optimized_parameters"], "encoder_only")
                for dataset, entries in result["results"].items():
                    runs = entries["mm_lda"]
                    self.assertEqual([run["seed"] for run in runs], [1338, 1339])
                    self.assertEqual(result["summary"][dataset]["completed_runs"], 2)
                    finals = [run["final_test_acc"] for run in runs]
                    self.assertEqual(result["summary"][dataset]["final_test_mean"], sum(finals) / 2)
                    for run in runs:
                        self.assertEqual(len(run["epoch_metrics"]), 1)
                        self.assertTrue(math.isfinite(run["epoch_metrics"][0]["train_loss"]))
                        self.assertTrue(0 <= run["final_test_acc"] <= 1)
                self.assertFalse(output.with_name(output.name + ".tmp").exists())
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
