import contextlib
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from FaceFlow.data import PairSampler, split_identities
from FaceFlow.flow import flow_loss, sample_u, uniform_cosine_theta
from FaceFlow.geometry import compose, isotropic_direction, unit
from FaceFlow.model import FaceFlow
from train import get_args
from sample import main as sample_main


class CaptureField(torch.nn.Module):
    def forward(self, u, t, a, theta):
        self.theta = theta.detach().clone()
        return torch.zeros_like(u)


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_public_package_and_model_name(self):
        from FaceFlow import FaceFlow as exported_model
        self.assertIs(exported_model, FaceFlow)
        self.assertEqual(FaceFlow.__name__, 'FaceFlow')
        self.assertEqual(FaceFlow.__module__, 'FaceFlow.model')

    def test_angle_bins_are_equal_width_and_balanced(self):
        theta = np.linspace(0.02, 1.45, 1001)
        cosine = np.cos(theta).astype(np.float32)
        index = np.arange(len(theta), dtype=np.int32)
        sampler = PairSampler(index, index, cosine, n_bins=20)
        np.testing.assert_allclose(np.diff(sampler.edges), np.diff(sampler.edges)[0])
        membership = np.empty(len(theta), dtype=int)
        for k, rows in enumerate(sampler.bins):
            membership[rows.numpy()] = k
        a, _ = sampler.sample(200)
        np.testing.assert_array_equal(np.bincount(membership[a.numpy()]), np.full(20, 10))
        self.assertGreater(np.unique(np.round(np.diff(np.cos(sampler.edges)), 5)).size, 1)

    def test_sampler_handles_remainders_and_constant_angle(self):
        index = np.arange(5, dtype=np.int32)
        sampler = PairSampler(index, index, np.full(5, 0.5, dtype=np.float32))
        self.assertEqual(len(sampler.bins), 1)
        self.assertEqual(len(sampler.sample(3)[0]), 3)
        with self.assertRaises(ValueError):
            sampler.sample(0)

    def test_loss_uses_the_observed_pair_angle(self):
        torch.manual_seed(4)
        anchor = unit(torch.randn(8, 32, dtype=torch.float64))
        theta = torch.linspace(0.2, 1.2, 8, dtype=torch.float64).view(-1, 1)
        target = compose(anchor, theta, isotropic_direction(anchor))
        net = CaptureField()
        loss, _ = flow_loss(net, anchor, target)
        torch.testing.assert_close(net.theta, theta)
        self.assertTrue(torch.isfinite(loss))

    def test_angle_override_and_weighted_sampler_are_not_exposed(self):
        self.assertNotIn('theta_override', inspect.signature(flow_loss).parameters)
        self.assertNotIn('weights', inspect.signature(PairSampler).parameters)

    def test_uniform_cosine_sampling(self):
        generator = torch.Generator().manual_seed(11)
        theta = uniform_cosine_theta(10000, 0.5, generator=generator)
        expected = 0.5 + 0.5 * torch.rand(10000, 1, generator=torch.Generator().manual_seed(11))
        torch.testing.assert_close(theta.cos(), expected)
        self.assertLess(abs(theta.cos().mean().item() - 0.75), 0.005)

    def test_zero_angle_limit_and_invalid_bounds(self):
        self.assertTrue(torch.equal(uniform_cosine_theta(10, 1), torch.zeros(10, 1)))
        for lb in [-1, 0, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                uniform_cosine_theta(10, lb)

    def test_zero_field_preserves_the_paired_initial_direction(self):
        torch.manual_seed(7)
        anchor = unit(torch.randn(8, 32))
        theta = torch.full((8, 1), 0.5)
        net = FaceFlow(d=32, w=16, depth=1).eval()
        actual = sample_u(net, anchor, theta, generator=torch.Generator().manual_seed(3))
        expected = isotropic_direction(anchor, torch.Generator().manual_seed(3))
        torch.testing.assert_close(actual, expected)

    def test_training_defaults(self):
        with patch.object(sys, 'argv', ['train.py', '--emb', 'emb.npy', '--labels', 'labels.npy', '--out', 'training-output']):
            args = get_args()
        for key, value in {'steps': 100000, 'batch': 4096, 'width': 768, 'depth': 8,
                           'mlp_ratio': 4, 'theta_bins': 20, 'min_cos': 0.1,
                           'max_pairs_per_id': 2000, 'val_frac': 0.05, 'warmup': 2000,
                           'lr': 1e-4, 'wd': 0.01, 'clip': 1.0, 'ema': 0.999,
                           'bf16': True}.items():
            self.assertEqual(getattr(args, key), value)

    def test_legacy_theta_distribution_option_is_rejected(self):
        with patch.object(sys, 'argv', ['train.py', '--emb', 'e', '--labels', 'l', '--theta-dist', 'x']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            get_args()

    def test_identity_split_is_disjoint(self):
        labels = np.repeat(np.arange(100), 2)
        train, heldout = split_identities(labels, 0.05, 1337)
        self.assertEqual(len(np.unique(labels[heldout])), 5)
        self.assertEqual(np.intersect1d(labels[train], labels[heldout]).size, 0)

    def test_generator_and_recognition_config(self):
        with (ROOT / 'generator_configs/generator.yaml').open() as handle:
            generator = yaml.safe_load(handle)['sampling']
        self.assertTrue(generator['two_stage'])
        self.assertFalse(generator['stage2_only'])
        self.assertEqual(generator['sample_config']['MSE_th'], 0.005)
        self.assertEqual(generator['sample_config']['cfg_scale'], 1.0)
        with (ROOT / 'generator_configs/downstream.json').open() as handle:
            fr = json.load(handle)
        self.assertEqual(fr['loss'], 'CosFace')
        self.assertEqual(fr['epochs'], 34)
        self.assertEqual(fr['global_batch_size'], 512)
        self.assertEqual(fr['lr_milestones'], [22, 28, 32])

    def test_sampling_entry_point_uses_ema_and_preserves_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = root / 'bank.npy'
            reference = unit(torch.randn(2, 512)).numpy()
            np.save(bank, reference)
            net = FaceFlow(d=512, w=16, depth=1)
            checkpoint = root / 'zero.pt'
            online = {name: value + 0.1 for name, value in net.state_dict().items()}
            torch.save({'args': {'width': 16, 'depth': 1, 'mlp_ratio': 4},
                        'ema': net.state_dict(), 'model': online, 'step': 0}, checkpoint)
            for run_name in ['first', 'repeat']:
                argv = ['sample.py', '--bank', str(bank), '--output', str(root / run_name),
                        '--checkpoint', str(checkpoint), '--identities', '2', '--samples', '3',
                        '--batch', '3', '--device', 'cpu']
                with patch.object(sys, 'argv', argv):
                    sample_main()
            contexts = np.load(root / 'first/contexts.npy')
            np.testing.assert_array_equal(contexts, np.load(root / 'repeat/contexts.npy'))
            theta = uniform_cosine_theta(6, 0.5, generator=torch.Generator().manual_seed(1337))
            np.testing.assert_allclose(np.load(root / 'first/angles_degrees.npy'),
                                       torch.rad2deg(theta).numpy().reshape(2, 3), atol=1e-5)
            anchors = unit(torch.from_numpy(np.load(root / 'first/reference_embeddings.npy'))).repeat_interleave(3, 0)
            direction_rng = torch.Generator().manual_seed(1337)
            expected = []
            for start in range(0, 6, 3):
                batch_anchor = anchors[start:start + 3]
                expected.append(compose(batch_anchor, theta[start:start + 3],
                                        isotropic_direction(batch_anchor, direction_rng)))
            np.testing.assert_allclose(contexts.reshape(6, 512), torch.cat(expected).numpy(), atol=2e-7)
            np.testing.assert_allclose(np.linalg.norm(contexts, axis=-1), 1, atol=2e-6)
            np.testing.assert_allclose((contexts.reshape(6, 512) * anchors.numpy()).sum(-1),
                                       theta.cos().numpy().ravel(), atol=2e-6)
            manifest = json.loads((root / 'first/manifest.json').read_text())
            self.assertEqual(manifest['method'], 'FaceFlow')
            self.assertEqual(manifest['angle_law'], 'uniform_cosine')
            self.assertEqual(manifest['ode_steps'], 16)
            self.assertEqual(manifest['weights'], 'ema')
            self.assertNotIn('fixed_theta_degrees', manifest)

    def test_sampling_requires_a_checkpoint(self):
        with patch.object(sys, 'argv', ['sample.py', '--bank', 'bank.npy', '--output', 'output']), \
                contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as failure:
            sample_main()
        self.assertEqual(failure.exception.code, 2)
        self.assertIn('--checkpoint', errors.getvalue())

    def test_removed_experiment_options_are_rejected(self):
        for option in [['--method', 'idperturb'], ['--fixed-cos', '0.6'], ['--attribute-angle']]:
            arguments = ['sample.py', '--bank', 'bank.npy', '--checkpoint', 'model.pt',
                         '--output', 'output', *option]
            with self.subTest(option=option), patch.object(sys, 'argv', arguments), \
                    contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as failure:
                sample_main()
            self.assertEqual(failure.exception.code, 2)
            self.assertIn('unrecognized arguments', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
