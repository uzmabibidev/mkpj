"""mkpj — one-command Python project generator."""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__", "main"]


def main(argv=None) -> int:
    from .cli import main as _main

    return _main(argv)
