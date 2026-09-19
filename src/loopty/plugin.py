"""loopty's plugin surface: the four objects lanky discovers by entry point.

lanky never imports loopty. It reads the entry-point groups ``lanky.theories``,
``lanky.oracles``, ``lanky.executors`` and ``lanky.verbs``, and every one of
loopty's entries points into this module. Keeping them here, rather than at the
class definitions, means the import path in ``pyproject.toml`` does not move when
a class does.

* :class:`KernelTheory` turns decorated kernels into facts.
* :class:`IslOracle` decides the Presburger ones, with witnesses.
* :class:`LoopyExecutor` runs kernels and schedules through loopy.
* :class:`RunVerb` is the ``run`` subcommand of both commands.
"""

from __future__ import annotations

from loopty.cli import RunVerb
from loopty.executor import LoopyExecutor
from loopty.kernel import KernelTheory
from loopty.oracle import IslOracle

__all__ = ["IslOracle", "KernelTheory", "LoopyExecutor", "RunVerb"]
