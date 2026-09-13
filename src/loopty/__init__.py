"""loopty: loopy, with types.

loopty is loop + ty, for types: a typed polyhedral layer over loopy
(https://github.com/inducer/loopy). Kernels are decorated Python functions whose
bodies run natively under plain ``python`` as the reference implementation and
are traced under loopty to build a typed term. Types are isl objects: a
statement's type is its iteration domain (an isl set) and its read, write, and
accumulation footprints (isl maps); dependences are derived by isl flow
analysis, not declared; index types include ragged, dependent shapes (CSR-style
data as dependent sums) so disjointness and in-bounds facts come from the shape
rather than from offset arithmetic; loop transformations are checked as casts
along bijections, with a concrete witness on failure; loopy generates the code
(OpenCL, CUDA, C). loopty is the first plugin for its sister project lanky (a
Python-hosted proof language over Lean 4; https://github.com/xywei/lanky):
loopty's typing rules emit facts into lanky's ledger, its isl oracle decides the
Presburger ones, and residual obligations become lanky theorems.

Status: work in progress; this is a placeholder release to reserve the name.
Nothing works yet.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
