import argparse
import ast
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.path_args import data_path, path_string
from sample import main as sample_main
from train import get_args


class CliTests(unittest.TestCase):
    def test_empty_strings_are_not_current_directory_paths(self):
        for value in ['', ' ', '\t\n']:
            for converter in [data_path, path_string]:
                with self.subTest(value=value, converter=converter.__name__):
                    with self.assertRaises(argparse.ArgumentTypeError):
                        converter(value)
        self.assertEqual(data_path('explicit-directory'), Path('explicit-directory'))

    def test_training_requires_explicit_output(self):
        with patch.object(sys, 'argv', ['train.py', '--emb', 'emb.npy', '--labels', 'labels.npy']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            get_args()
        self.assertEqual(failure.exception.code, 2)

    def test_training_rejects_each_empty_path(self):
        for option in ['--emb', '--labels', '--out']:
            arguments = ['train.py', '--emb', 'emb.npy', '--labels', 'labels.npy', '--out', 'output']
            arguments[arguments.index(option) + 1] = ''
            with self.subTest(option=option), patch.object(sys, 'argv', arguments), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                get_args()
            self.assertEqual(failure.exception.code, 2)

    def test_sampler_rejects_blank_bank_before_loading(self):
        with patch.object(sys, 'argv', ['sample.py', '--bank', '', '--output', 'output']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            sample_main()
        self.assertEqual(failure.exception.code, 2)

    def test_path_defaults_are_unset(self):
        paths = [ROOT / name for name in ['train.py', 'sample.py', 'extract_uiface_fr_embeddings.py']]
        path_arguments = 0
        for source in paths:
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != 'add_argument':
                    continue
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                converter = keywords.get('type')
                if not isinstance(converter, ast.Name):
                    continue
                self.assertNotEqual(converter.id, 'Path', str(source.relative_to(ROOT)))
                if converter.id not in {'data_path', 'path_string'}:
                    continue
                path_arguments += 1
                default = keywords.get('default')
                self.assertIsInstance(default, ast.Constant, str(source.relative_to(ROOT)))
                self.assertIsNone(default.value, str(source.relative_to(ROOT)))
        self.assertEqual(path_arguments, 10)

    def test_yaml_resource_paths_are_empty(self):
        config = yaml.safe_load((ROOT / 'generator_configs/generator.yaml').read_text())
        values = [config['checkpoint']['path'], config['diffusion_cfg_path'],
                  config['VQEncoder_path'], config['VQDecoder_path'],
                  config['sampling']['precomputed_contexts_file'],
                  config['sampling']['context_ids_file'], config['sampling']['save_dir']]
        self.assertEqual(values, [''] * 7)

    def test_environment_path_template_is_empty(self):
        entries = (ROOT / 'paths.example.env').read_text().splitlines()
        self.assertEqual(len(entries), 16)
        for entry in entries:
            self.assertEqual(entry.split('=', 1)[1], '""')

if __name__ == '__main__':
    unittest.main()
