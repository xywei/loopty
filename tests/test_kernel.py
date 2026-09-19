"""The ``@kernel`` and ``@program`` decorators: inert, registering, traceable."""

from __future__ import annotations

import numpy as np
from lanky.ledger import Status
from lanky.plugins import registry
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, kernel, program
from loopty.kernel import Kernel, KernelTheory, Program


@kernel
def scale(a: Real, x: Arr[Fin[n], Real]):  # noqa: F821
    """Multiply every entry of ``x`` by ``a``."""
    for i in x.dom:
        x[i] = a * x[i]


@kernel
def scan(cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]) -> all(  # noqa: F821
    off[r + 1] == off[r] + cnt[r] for r in Fin[n]  # noqa: F821
):
    """Exclusive prefix sum."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@program
def both(cnt, off, a):
    """Lay out the rows and then scale the offsets, as a composition."""
    scan(cnt, off)
    scale(a, off)


def test_the_decorator_is_inert_and_the_body_still_runs() -> None:
    x = Arr.from_numpy(np.array([1.0, 2.0, 3.0]))
    scale(2.0, x)
    assert list(x.numpy()) == [2.0, 4.0, 6.0]


def test_a_decorated_kernel_is_registered_in_import_order() -> None:
    objects = [obj for obj in registry.objects if isinstance(obj, Kernel | Program)]
    names = [obj.__name__ for obj in objects]
    assert names.index("scale") < names.index("scan") < names.index("both")


def test_the_term_is_traced_once_and_kept() -> None:
    assert scale.term is scale.term
    assert scale.term.name == "scale"
    assert scale.term.params[0][0] == "a"


def test_the_return_annotation_is_the_postcondition() -> None:
    assert scan.term.post is not None
    assert scale.term.post is None


def test_the_theory_owns_kernels_and_nothing_else() -> None:
    theory = KernelTheory()
    assert theory.name == "kernel"
    assert theory.facts(object()) == ()
    kinds = {fact.kind for fact in theory.facts(scan)}
    assert {"in-bounds", "disjoint-writes", "ordering", "postcondition"} <= kinds


def test_a_body_that_cannot_be_traced_is_reported_as_a_fact() -> None:
    @kernel
    def branchy(u: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            if i > 0:
                u[i] = 1.0

    facts = branchy.facts()
    assert len(facts) == 1
    assert facts[0].status is Status.REFUTED
    assert facts[0].kind == "trace"
    assert "when" in facts[0].provenance["error"]


def test_a_program_runs_natively_and_records_its_callees_claims() -> None:
    cnt = Arr.from_numpy(np.array([1, 2, 3], dtype=np.int64))
    off = Arr.zeros(4, dtype=np.int64)
    both(cnt, off, 2)
    assert list(off.numpy()) == [0, 2, 6, 12]

    assert {callee.__name__ for callee in both.callees()} == {"scan", "scale"}
    facts = both.facts()
    assert [fact.kind for fact in facts] == ["postcondition-in-scope"]
    assert facts[0].status is Status.ASSUMED
    assert facts[0].owner.endswith("both")


def test_a_kernel_without_a_when_block_gets_its_arguments_untouched() -> None:
    assert not scale.guards_writes
    given = Arr.zeros(2)
    seen = []

    @kernel
    def peek(x: Arr[Fin[n], Real]):  # noqa: F821
        seen.append(x)
        for i in x.dom:
            x[i] = 0.0

    peek(given)
    assert seen[0] is given


def test_the_kernel_keeps_the_function_name_docstring_and_source_line() -> None:
    assert scale.__name__ == "scale"
    assert scale.__doc__ is not None and "Multiply" in scale.__doc__
    assert scale.where.startswith("test_kernel.py:")
    assert scale.qualname == "scale"


def test_a_body_that_needs_dom_reads_a_bare_numpy_array_as_a_dense_array() -> None:
    # A body that iterates ``.dom`` used to fail on an ndarray with
    # "AttributeError: 'ndarray' object has no attribute 'dom'". The array is
    # now wrapped, sharing its buffer, so the writes land in the caller's array.
    given = np.array([1.0, 2.0])
    scale(2.0, given)
    assert list(given) == [2.0, 4.0]


def test_wrapping_a_numpy_argument_does_not_copy_it() -> None:
    seen = []

    @kernel
    def peek_raw(x: Arr[Fin[n], Real]):  # noqa: F821
        seen.append(x)
        for i in x.dom:
            x[i] = 0.0

    given = np.ones(3)
    peek_raw(given)
    assert isinstance(seen[0], Arr)
    assert seen[0].numpy() is given
    assert list(given) == [0.0, 0.0, 0.0]
