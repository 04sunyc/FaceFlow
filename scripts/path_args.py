"""Path argument converters that reject empty or whitespace-only values."""

import argparse
from pathlib import Path


def path_string(value):
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("path is empty; provide an explicit filesystem path")
    return value


def data_path(value):
    return Path(path_string(value))
