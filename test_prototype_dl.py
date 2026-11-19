"""Equation, gradient, and synthetic training checks for Li et al. (AAAI 2018)."""
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

from train_deep_lda_covariances import Encoder
from train_prototype_dl import PrototypeDL, PrototypeHead, main, pixel_targets, prototype_loss


class PrototypeDLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_distance_classifier_and_gradients_match_direct_equations(self):
        torch.manual_seed(17)
        head = PrototypeHead(3, 4, 5).double()
        self.assertIsNone(head.classifier.bias)
        for parameter in head.parameters():
            self.assertTrue(((parameter >= 0) & (parameter <= 1)).all())
        z = torch.randn(7, 4, dtype=torch.double, requires_grad=True)
        labels = torch.arange(7) % 3
        # Independent broadcast implementation of Eq. (1).
        distances = (z[:, None] - head.prototypes[None]).square().sum(-1)
        expected_logits = distances @ head.classifier.weight.T
        logits = head(z)
        torch.testing.assert_close(head.squared_distances(z), distances)
        torch.testing.assert_close(logits, expected_logits)
        actual_distances = head.squared_distances(z)
        actual = (F.cross_entropy(logits, labels) + 0.05 * actual_distances.min(0).values.mean()
                  + 0.05 * actual_distances.min(1).values.mean())
        expected = (F.cross_entropy(expected_logits, labels) + 0.05 * distances.min(0).values.mean()
                    + 0.05 * distances.min(1).values.mean())
        inputs = (z, head.prototypes, head.classifier.weight)
        grads = torch.autograd.grad(actual, inputs, retain_graph=True)
        reference_grads = torch.autograd.grad(expected, inputs)
        for grad, reference in zip(grads, reference_grads):
            torch.testing.assert_close(grad, reference)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm().item(), 0)

    def test_four_loss_terms_and_reductions(self):
        logits = torch.tensor([[2., -1.], [0., 1.], [1., 1.]])
        labels = torch.tensor([0, 1, 0])
        reconstruction = torch.arange(1., 7.).reshape(3, 1, 1, 2)
        target = torch.zeros_like(reconstruction)
        distances = torch.tensor([[1., 9.], [4., 0.], [16., 25.]])
        loss, terms = prototype_loss(logits, labels, reconstruction, target, distances,
                                     lambda_reconstruction=0.2, lambda_1=0.3, lambda_2=0.4)
        classification = -logits.log_softmax(1)[torch.arange(3), labels].mean()
        torch.testing.assert_close(terms["classification_loss"], classification)
        torch.testing.assert_close(terms["reconstruction_loss"], torch.tensor(91 / 3))
        torch.testing.assert_close(terms["prototype_to_data_loss"], torch.tensor(0.5))
        torch.testing.assert_close(terms["data_to_prototype_loss"], torch.tensor(17 / 3))
        torch.testing.assert_close(loss, classification + 0.2 * (91 / 3) + 0.3 * 0.5 + 0.4 * (17 / 3))

    def test_pixel_targets_reverse_loader_normalization(self):
        pixels = torch.rand(2, 3, 32, 32)
        for name, mean, std in [
            ("CIFAR10", (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ("CIFAR100", (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]:
            normalized = (pixels - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
            torch.testing.assert_close(pixel_targets(normalized, name), pixels)
        grayscale = torch.rand(2, 1, 28, 28)
        self.assertIs(pixel_targets(grayscale, "FashionMNIST"), grayscale)

    def test_exact_baseline_encoder_and_joint_optimization(self):
        torch.manual_seed(1338)
        reference = Encoder(9, 1)
        torch.manual_seed(1338)
        model = PrototypeDL(10, 9, 1, 28)
        self.assertIs(type(model.encoder), Encoder)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.encoder.state_dict()[name], value, rtol=0, atol=0)
        x = torch.rand(2, 1, 28, 28)
        labels = torch.tensor([0, 1])
        logits, reconstruction, distances = model.training_outputs(x)
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertTrue(((reconstruction >= 0) & (reconstruction <= 1)).all())
        loss, _ = prototype_loss(logits, labels, reconstruction, x, distances)
        before = {name: value.clone() for name, value in model.named_parameters()}
        opt = torch.optim.Adam(model.parameters())
        loss.backward()
        for module in (model.encoder, model.decoder, model.head.classifier):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in module.parameters()))
            self.assertGreater(sum(p.grad.norm().item() for p in module.parameters()), 0)
        self.assertGreater(model.head.prototypes.grad.norm().item(), 0)
        opt.step()
        for prefix in ("encoder.", "decoder.", "head.prototypes", "head.classifier."):
            self.assertTrue(any(not torch.equal(before[name], value)
                                for name, value in model.named_parameters() if name.startswith(prefix)))
        model.eval()
        with torch.no_grad(), patch.object(model.decoder, "forward", side_effect=AssertionError):
            self.assertEqual(model(x).shape, (2, 10))

    def test_end_to_end_all_datasets_with_synthetic_images(self):
        def fake_loaders(name, data_root, batch_size, test_batch_size):
            channels, side = (1, 28) if name == "FashionMNIST" else (3, 32)
            pixels = torch.rand(3, channels, side, side)
            # Invert pixel_targets to reproduce each loader's normalized input.
            offset = pixel_targets(torch.zeros_like(pixels), name)
            scale = pixel_targets(torch.ones_like(pixels), name) - offset
            ds = TensorDataset((pixels - offset) / scale, torch.arange(3))
            return DataLoader(ds, batch_size=2), DataLoader(ds, batch_size=3), channels

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "results.json"
            argv = ["train_prototype_dl.py", "--epochs", "2", "--runs", "2", "--output", str(output),
                    "--num-prototypes", "7", "--lr", "0.0001", "--lambda-1", "0.1"]
            with patch("sys.argv", argv), patch("train_prototype_dl.build_loaders", fake_loaders), \
                 patch("torch.cuda.is_available", return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads(output.read_text())
            self.assertEqual(set(result["results"]), {"FashionMNIST", "CIFAR10", "CIFAR100"})
            self.assertEqual(result["meta"]["num_prototypes"], 7)
            self.assertEqual(result["meta"]["optimizer"]["lr"], 0.0001)
            self.assertEqual(result["meta"]["lambda_1"], 0.1)
            self.assertEqual([cfg["D"] for cfg in result["meta"]["datasets"]], [9, 9, 99])
            for dataset, entries in result["results"].items():
                runs = entries["prototype_dl"]
                self.assertEqual([run["seed"] for run in runs], [1338, 1339])
                self.assertEqual(result["summary"][dataset]["completed_runs"], 2)
                finals = [run["final_test_acc"] for run in runs]
                self.assertEqual(result["summary"][dataset]["final_test_mean"], sum(finals) / 2)
                for run in runs:
                    self.assertEqual(len(run["epoch_metrics"]), 2)
                    for record in run["epoch_metrics"]:
                        self.assertTrue(all(math.isfinite(value) for value in record.values()))
                        expected = (record["classification_loss"] + 0.05 * record["reconstruction_loss"]
                                    + 0.1 * record["prototype_to_data_loss"]
                                    + 0.05 * record["data_to_prototype_loss"])
                        self.assertAlmostEqual(record["train_loss"], expected, places=4)
                    self.assertTrue(0 <= run["final_test_acc"] <= 1)
            self.assertFalse(output.with_name(output.name + ".tmp").exists())

    def test_invalid_cli_configuration(self):
        for flag, value in [("--epochs", "0"), ("--runs", "0"), ("--batch-size", "0"),
                            ("--test-batch-size", "0"), ("--num-prototypes", "0"),
                            ("--lr", "0"), ("--lr", "nan"), ("--lambda-1", "-1"),
                            ("--lambda-2", "inf"), ("--lambda-reconstruction", "nan")]:
            with self.subTest(flag=flag, value=value), \
                 patch("sys.argv", ["train_prototype_dl.py", flag, value]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
