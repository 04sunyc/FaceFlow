"""Check an exported source directory before sharing it."""

import argparse
from pathlib import Path
import re


FILES = {
    '.gitignore', 'README.md', 'requirements.txt', 'paths.example.env',
    'train.py', 'sample.py', 'extract_uiface_fr_embeddings.py', 'tests.py',
    'FaceFlow/__init__.py', 'FaceFlow/data.py', 'FaceFlow/flow.py',
    'FaceFlow/geometry.py', 'FaceFlow/model.py',
    'generator_configs/downstream.json', 'generator_configs/generator.yaml',
    'scripts/generate.sh', 'scripts/path_args.py', 'scripts/check_package.py',
    'scripts/tests/test_flow.py', 'scripts/tests/test_cli.py',
    'scripts/tests/test_generate.py', 'scripts/tests/test_package.py',
}


def check_package(root):
    if not root.is_dir():
        return [f'Not a directory: {root}']
    errors = []
    found = set()
    blocked = {'.git', '.hg', '.svn', '.DS_Store', '__MACOSX', '__pycache__',
               '.pytest_cache', '.venv', '.idea', '.vscode', '.env', 'paths.env',
               'third_party'}
    prefixes = ('/' + 'home/', '/' + 'Users/', '/' + 'mnt/',
                '/' + 'scratch/', '/' + 'Volumes/')
    patterns = [
        r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
        r'(?m)^\s*__(?:author|email|maintainer)__\s*=',
        r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----',
        r'(?i)\b[A-Z]:[\\/]Users[\\/]',
    ]
    for source in sorted(root.rglob('*')):
        relative = source.relative_to(root)
        name = relative.as_posix()
        if source.is_symlink():
            errors.append(f'Symlink: {name}')
            continue
        if blocked.intersection(relative.parts) or any(part.startswith('._') for part in relative.parts):
            errors.append(f'Local file or directory: {name}')
            continue
        if not source.is_file():
            continue
        found.add(name)
        if name not in FILES:
            errors.append(f'Unexpected file: {name}')
            continue
        try:
            text = source.read_text(encoding='utf-8')
        except (OSError, UnicodeError):
            errors.append(f'Cannot read as UTF-8: {name}')
            continue
        if any(prefix in text for prefix in prefixes):
            errors.append(f'Local filesystem path: {name}')
        if any(re.search(pattern, text) for pattern in patterns):
            errors.append(f'Contact details or credentials: {name}')
    errors.extend(f'Missing file: {name}' for name in sorted(FILES - found))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='exported source directory')
    args = parser.parse_args()
    if not args.directory.strip():
        parser.error('directory must not be empty')
    errors = check_package(Path(args.directory))
    if errors:
        for error in errors:
            print(error)
        return 1
    print('Source package checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
