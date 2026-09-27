from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.check_package import FILES, check_package


class PackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('', encoding='utf-8')

    def test_clean_package(self):
        self.assertEqual(check_package(self.root), [])

    def test_missing_source(self):
        (self.root / 'train.py').unlink()
        self.assertIn('Missing file: train.py', check_package(self.root))

    def test_extra_data_is_rejected_without_reading_it(self):
        (self.root / 'weights.pt').write_bytes(b'\xff\x00')
        self.assertIn('Unexpected file: weights.pt', check_package(self.root))

    def test_local_files_are_rejected_only_in_the_checked_directory(self):
        for name in ['.venv', '.git', '__pycache__']:
            (self.root / name).mkdir()
        for name in ['paths.env', '.DS_Store']:
            (self.root / name).write_bytes(b'\xff')
        errors = check_package(self.root)
        for name in ['.venv', '.git', '__pycache__', 'paths.env', '.DS_Store']:
            self.assertIn(f'Local file or directory: {name}', errors)
        with tempfile.TemporaryDirectory() as directory:
            clean = Path(directory)
            for name in FILES:
                path = clean / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('', encoding='utf-8')
            self.assertEqual(check_package(clean), [])

    def test_broken_symlink(self):
        (self.root / 'link').symlink_to(self.root / 'missing')
        self.assertIn('Symlink: link', check_package(self.root))

    def test_local_paths_and_credentials(self):
        examples = [
            ('/' + 'Users/' + 'someone/data', 'Local filesystem path'),
            ('person' + '@' + 'example.org', 'Contact details or credentials'),
            ('-----BEGIN ' + 'PRIVATE KEY-----', 'Contact details or credentials'),
        ]
        for text, message in examples:
            with self.subTest(message=message):
                (self.root / 'README.md').write_text(text, encoding='utf-8')
                self.assertIn(f'{message}: README.md', check_package(self.root))

    def test_non_text_source(self):
        (self.root / 'README.md').write_bytes(b'\xff')
        self.assertIn('Cannot read as UTF-8: README.md', check_package(self.root))

    def test_missing_directory(self):
        missing = self.root / 'missing'
        self.assertEqual(check_package(missing), [f'Not a directory: {missing}'])

    def test_command_requires_an_explicit_directory(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_package.py')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('directory', result.stderr)

    def test_command_rejects_blank_directory(self):
        for directory in ['', ' ', '\t\n']:
            with self.subTest(directory=directory):
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_package.py'), directory],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('directory must not be empty', result.stderr)


if __name__ == '__main__':
    unittest.main()
