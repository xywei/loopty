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
"""

from __future__ import annotations

from loopty.arr import Arr
from loopty.idx import Fin
from loopty.kernel import Kernel, Program, kernel, program
from loopty.oracle import IslOracle
from loopty.reduction import reduce_sum
from loopty.schedule import IllegalCast, Schedule, UnbuildableSchedule
from loopty.term import Access, ArrType, Reduction, Stmt, Term
from loopty.trace import TraceError, trace, when
from loopty.typing import facts_for

__version__ = "0.1.0.dev0"

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
    "sum",
    "trace",
    "when",
]

# Compatibility alias for the development API. New kernel code should use
# ``reduce_sum``, which makes the operation's Loopty ownership explicit.
sum = reduce_sum
