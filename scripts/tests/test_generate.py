import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class GenerateTests(unittest.TestCase):
    def test_required_paths(self):
        result = subprocess.run(['bash', str(ROOT / 'scripts/generate.sh')],
                                env={'PATH': '/usr/bin:/bin'}, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('UIFACE_GENERATOR_ROOT', result.stderr)

    def test_config_and_paths_are_forwarded(self):
        with tempfile.TemporaryDirectory(prefix='faceflow test ') as directory:
            root = Path(directory)
            (root / 'sample_precomputed.py').write_text(
                'import json\nimport sys\nprint(json.dumps(sys.argv[1:]))\n',
                encoding='utf-8')
            environment = {
                'PATH': '/usr/bin:/bin', 'PYTHON': sys.executable,
                'UIFACE_GENERATOR_ROOT': str(root),
                'DIFFUSION_CHECKPOINT': str(root / 'model.pt'),
                'DIFFUSION_CONFIG': str(root / 'model.yaml'),
                'VQ_ENCODER': str(root / 'encoder.pt'),
                'VQ_DECODER': str(root / 'decoder.pt'),
                'CONTEXT_DIR': str(root / 'contexts'),
                'IMAGE_OUTPUT': str(root / 'images'), 'SEED': '42',
            }
            command = ['bash', str(ROOT / 'scripts/generate.sh')]
            result = subprocess.run(command, cwd=root, env=environment,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = json.loads(result.stdout)
            config_path = Path(arguments[arguments.index('--config-path') + 1])
            config_name = arguments[arguments.index('--config-name') + 1]
            self.assertEqual(config_name, 'generator')
            self.assertTrue((config_path / f'{config_name}.yaml').is_file())
            self.assertIn(f'checkpoint.path={root / "model.pt"}', arguments)
            self.assertIn(f'sampling.precomputed_contexts_file={root / "contexts/contexts.npy"}', arguments)
            self.assertIn(f'sampling.context_ids_file={root / "contexts/identity_ids.npy"}', arguments)
            self.assertIn('sampling.seed=42', arguments)
            (root / 'images').mkdir()
            result = subprocess.run(command, cwd=root, env=environment,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
