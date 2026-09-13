"""Command line interface for loopty.

The placeholder release ships a single entry point that reports what loopty is
and that it does not do anything yet.
"""

from __future__ import annotations

from loopty import __version__

REPO_URL = "https://github.com/xywei/loopty"


def main() -> int:
    """Print the placeholder banner for loopty."""
    print("loopty")
    print(f"version: {__version__}")
    print("work in progress: placeholder release")
    print(REPO_URL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
