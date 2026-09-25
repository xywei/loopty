"""loopty: loopy, with types.

loopty is loop + ty, for types: a typed polyhedral layer over loopy
(https://github.com/inducer/loopy). Kernels are decorated Python functions. Their
bodies run natively under plain ``python`` as the reference implementation and
are traced under loopty to build a typed term; nothing is parsed, because
annotations are evaluated and bodies are executed once against a generic point.

Types are isl objects. A statement's type is its iteration domain (an isl set)
together with its read, write, and accumulation footprints (isl maps), so type
checking is emptiness, subset, and bijectivity, and dependences are derived by
flow analysis rather than declared. Index types include ragged, dependent shapes,
so CSR-style data is a dependent sum and disjointness and in-bounds facts come
from the shape rather than from offset arithmetic. Loop transformations are casts
along bijections: any transformation may be applied, and a small isl checker
either accepts it or rejects it with a concrete pair of statement instances.
loopy generates the code (C, OpenCL, CUDA).

loopty is the first plugin for its sister project lanky (a Python-hosted proof
language over Lean 4; https://github.com/xywei/lanky): loopty's typing rules emit
facts into lanky's ledger, its isl oracle decides the Presburger ones, and the
residual obligations become lanky theorems.

Status: in development, and honest about it; the README's status list is the
long form. A decorated kernel runs natively on numpy, traces to a term, states
its obligations as facts that ``lanky check`` prints with the oracle that
decided each one, lowers through loopy, and runs on the C target with its
result compared against the Python body at the tolerance its types state. Each
schedule step is checked as a cast and refused with a witness pair when it
would reorder a dependence, and asked separately whether the target can build
it at all. What is not here: a dependent sum deeper than two axes, the use of a
postcondition as a hypothesis, and any execution on a device from a development
machine.

``import loopty`` imports nothing but this module. Each name below comes from its
own module the first time it is used, so the import costs no loopy and no islpy,
and a file that defines kernels and runs them natively never imports loopy.
islpy arrives with the first name that needs the tracer, such as ``kernel``.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0.dev0"

#: The module each top-level name is defined in, imported on first use.
_EXPORTS = {
    "Access": "loopty.term",
    "Arr": "loopty.arr",
    "ArrType": "loopty.term",
    "Fin": "loopty.idx",
    "IllegalCast": "loopty.schedule",
    "IslOracle": "loopty.oracle",
    "Kernel": "loopty.kernel",
    "Program": "loopty.kernel",
    "Reduction": "loopty.term",
    "Schedule": "loopty.schedule",
    "Stmt": "loopty.term",
    "Term": "loopty.term",
    "TraceError": "loopty.trace",
    "UnbuildableSchedule": "loopty.schedule",
    "facts_for": "loopty.typing",
    "kernel": "loopty.kernel",
    "program": "loopty.kernel",
    "reduce_sum": "loopty.reduction",
    "trace": "loopty.trace",
    "when": "loopty.trace",
}

#: Compatibility aliases, each to the exported name it stands for. ``sum`` is
#: the development API's name for ``reduce_sum``; new kernel code should use
#: ``reduce_sum``, which makes the operation's Loopty ownership explicit. It is
#: not in ``__all__``, so that ``from loopty import *`` leaves the builtin
#: ``sum`` alone.
_ALIASES = {"sum": "reduce_sum"}

__all__ = [
    "Access",
    "ArrType",
    "Arr",
    "Fin",
    "IllegalCast",
    "IslOracle",
    "Kernel",
    "Program",
    "Reduction",
    "Schedule",
    "Stmt",
    "Term",
    "TraceError",
    "UnbuildableSchedule",
    "__version__",
    "facts_for",
    "kernel",
    "program",
    "reduce_sum",
    "trace",
    "when",
]


def __getattr__(name: str) -> Any:
    """Import a top-level name from its module the first time it is asked for."""
    exported = _ALIASES.get(name, name)
    module = _EXPORTS.get(exported)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), exported)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, *_ALIASES})


class _Package(ModuleType):
    """The ``loopty`` module, which keeps ``kernel`` and ``trace`` the functions.

    Both names are also submodules, and importing a submodule binds it on its
    package under its own name. That used to happen before this module bound
    the functions, which it did eagerly. Now the submodule can be imported
    first, and lanky's plugin discovery does exactly that before it imports a
    kernel file, so ``from loopty import kernel`` would give the module and
    ``@kernel`` would fail. That one binding is dropped here, and the name is
    looked up through :func:`__getattr__` as every other one is.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in _EXPORTS
            and isinstance(value, ModuleType)
            and value.__name__ == f"{__name__}.{name}"
        ):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Package

if TYPE_CHECKING:
    from loopty.arr import Arr
    from loopty.idx import Fin
    from loopty.kernel import Kernel, Program, kernel, program
    from loopty.oracle import IslOracle
    from loopty.reduction import reduce_sum
    from loopty.schedule import IllegalCast, Schedule, UnbuildableSchedule
    from loopty.term import Access, ArrType, Reduction, Stmt, Term
    from loopty.trace import TraceError, trace, when
    from loopty.typing import facts_for

    sum = reduce_sum
