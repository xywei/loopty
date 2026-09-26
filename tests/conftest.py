"""Make ``tests/`` importable, so that ``import hand_terms`` says what it means.

Several test modules share the hand-written terms in :mod:`hand_terms`, and
import it by bare name. That works by accident under pytest's default
``rootdir`` insertion, and stops working the moment ``tests/`` gains an
``__init__.py``, is collected through a different ``importmode``, or is run from
another directory. Putting the directory on ``sys.path`` here makes the
dependency explicit and local to the test suite rather than a property of how
pytest happened to be invoked.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture(autouse=True)
def _silence_loopys_own_deprecations() -> Iterator[None]:
    """Let a deprecation from loopty fail the run, and only from loopty.

    The same two exemptions as ``filterwarnings`` in ``pyproject.toml``, applied
    again here because a ``-W error::DeprecationWarning`` on the command line
    takes precedence over the ini file and would otherwise turn loopy's own
    warnings into failures of loopty's tests. A filter installed inside the test
    is installed last, so it wins, and it wins over nothing else: every other
    deprecation, including any that loopty causes, is still an error.

    Both exemptions are described in ``docs/loopy-notes.md``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'GCCToolchain.copy' is deprecated",
            category=DeprecationWarning,
        )
        # Fires only on a cold code-generation cache (loopy's persistent dict
        # under the user cache directory): a warm machine never reaches
        # simplify_pw_aff, a fresh CI runner always does.
        warnings.filterwarnings(
            "ignore",
            message="Aff.is_equal with implicit conversion",
            category=DeprecationWarning,
        )
        yield
