"""Numerical and synthetic training checks: python -m unittest test_arcface."""
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

from train_arcface import ArcFaceHead, DeepArcFace, main
from train_deep_lda_covariances import Encoder


class ArcFaceTests(unittest.TestCase):
    def test_matches_paper_equation_and_gradient(self):
        torch.manual_seed(9)
        head = ArcFaceHead(4, 3).double()
        z = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 0])
        logits = head(z, labels)
        cosine = F.normalize(z, dim=1) @ F.normalize(head.weight, dim=1).T
        reference = cosine.clone()
        rows = torch.arange(len(labels))
        reference[rows, labels] = torch.cos(torch.acos(cosine[rows, labels]) + 0.5)
        torch.testing.assert_close(logits, 64 * reference)
        loss = F.cross_entropy(logits, labels)
        ref_loss = F.cross_entropy(64 * reference, labels)
        actual = torch.autograd.grad(loss, (z, head.weight), retain_graph=True)
        expected = torch.autograd.grad(ref_loss, (z, head.weight))
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e)
            self.assertTrue(torch.isfinite(a).all())
            self.assertGreater(a.norm().item(), 0)

    def test_inference_has_no_margin_and_is_norm_invariant(self):
        head = ArcFaceHead(3, 3)
        z = torch.randn(4, 3)
        cosine = F.normalize(z, dim=1) @ F.normalize(head.weight, dim=1).T
        torch.testing.assert_close(head(z), 64 * cosine)
        expected = head(z).detach()
        with torch.no_grad():
            head.weight.mul_(torch.tensor([[2.0], [3.0], [4.0]]))
        torch.testing.assert_close(head(7 * z), expected)
        zero_margin = ArcFaceHead(3, 3, margin=0)
        zero_margin.load_state_dict(head.state_dict())
        torch.testing.assert_close(zero_margin(z, torch.tensor([0, 1, 2, 0])), head(z))

    def test_finite_gradients_at_aligned_and_zero_features(self):
        head = ArcFaceHead(3, 3)
        with torch.no_grad():
            head.weight.copy_(torch.eye(3))
        z = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 0., 0.]], requires_grad=True)
        loss = F.cross_entropy(head(z, torch.tensor([0, 0, 1])), torch.tensor([0, 0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(torch.isfinite(head.weight.grad).all())

    def test_encoder_is_identical_to_baseline(self):
        for C, D, channels in [(10, 9, 1), (10, 9, 3), (100, 99, 3)]:
            torch.manual_seed(1338)
            reference = Encoder(D, channels)
            torch.manual_seed(1338)
            model = DeepArcFace(C, D, channels)
            self.assertIs(type(model.encoder), Encoder)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(model.encoder.state_dict()[name], value, rtol=0, atol=0)

    def test_invalid_hyperparameters(self):
        for scale in (0, -1, float("inf"), float("nan")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                ArcFaceHead(3, 2, scale=scale)
        for margin in (-1, math.pi, float("inf"), float("nan")):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                ArcFaceHead(3, 2, margin=margin)

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
                argv = ["train_arcface.py", "--epochs", "1", "--runs", "2", "--output", str(output)]
                with patch("sys.argv", argv), patch("train_arcface.build_loaders", fake_loaders), \
                     patch("torch.cuda.is_available", return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()):
                    main()
                result = json.loads(output.read_text())
                self.assertEqual(set(result["results"]), {"FashionMNIST", "CIFAR10", "CIFAR100"})
                for dataset, entries in result["results"].items():
                    runs = entries["arcface"]
                    self.assertEqual([run["seed"] for run in runs], [1338, 1339])
                    self.assertEqual(result["summary"][dataset]["completed_runs"], 2)
                    for run in runs:
                        self.assertEqual(len(run["epoch_metrics"]), 1)
                        self.assertTrue(math.isfinite(run["epoch_metrics"][0]["train_loss"]))
                        self.assertTrue(0 <= run["final_test_acc"] <= 1)
                self.assertFalse(output.with_name(output.name + ".tmp").exists())
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
