"""PREPARED, NOT IMPLEMENTED: loopty's plugin surface for lanky.

loopty is the first plugin for lanky (https://github.com/xywei/lanky), a
Python-hosted proof language over Lean 4. When the plugin interfaces land,
loopty will register the following with the lanky host:

* :class:`KernelTheory` -- the ``@kernel`` and ``@program`` decorators, whose
  typing rules emit facts into lanky's ledger.
* :class:`IslOracle` -- decides the Presburger facts, with witnesses; trusted in
  development mode, re-derivable by Lean's ``omega``.
* :class:`LoopyExecutor` -- runs kernels through loopy.
* :class:`RunVerb` -- the ``loopty run`` / ``lanky run`` subcommand.

Every name below is a placeholder: it documents the intended role and has no
behaviour.
"""

from __future__ import annotations

# NOTE: the runtime dependency on `lanky` will be added when the plugin
# interfaces land. This placeholder release has no dependencies on purpose, so
# that reserving the name on PyPI does not pull anything into a user's
# environment.


class KernelTheory:
    """The theory loopty contributes to lanky's ledger.

    Will provide the ``@kernel`` and ``@program`` decorators. A decorated
    function's body runs natively under plain ``python`` as the reference
    implementation and is traced under loopty to build a typed term; the typing
    rules for that term emit facts (domains, footprints, dependences,
    transformation casts) into lanky's ledger.
    """

    ...


class IslOracle:
    """The decision procedure for the Presburger fragment.

    Will answer emptiness, subset, and bijectivity questions about isl sets and
    maps, returning a concrete witness when a check fails. Trusted in
    development mode; the facts it discharges are re-derivable by Lean's
    ``omega``, and anything outside its fragment becomes a residual lanky
    theorem.
    """

    ...


class LoopyExecutor:
    """The executor that runs typed kernels through loopy.

    Will lower a checked loopty term to a loopy kernel, generate code for the
    requested target (OpenCL, CUDA, C), and run it -- with the plain ``python``
    execution of the same body available as the differential-test oracle.
    """

    ...


class RunVerb:
    """The ``run`` subcommand loopty registers with lanky.

    Will back both ``loopty run`` and ``lanky run`` for loopty targets:
    elaborate the kernel, check its obligations, generate code, execute, and
    compare against the reference run.
    """

    ...


__all__ = ["IslOracle", "KernelTheory", "LoopyExecutor", "RunVerb"]
