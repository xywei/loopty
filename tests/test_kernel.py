"""The ``@kernel`` and ``@program`` decorators: inert, registering, traceable."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from lanky.ledger import Ledger, Status
from lanky.plugins import registry
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, kernel, program, when
from loopty.kernel import Kernel, KernelTheory, Program

KERNELS = Path(__file__).parent / "kernels"

#: A module that re-exports the guard under another name, so that a body can
#: reach it as an attribute without the identifier ``when`` appearing anywhere.
guards = ModuleType("guards")
guards.mask = when


def shift_guarded(v, i, u, size) -> None:
    """Write ``u[i + 1]`` into ``v[i]``, under a guard the *caller* never names."""
    with when(i + 1 < size):
        v[i] = u[i + 1]


class _Shifter:
    """A helper reached as a method rather than as a plain global function."""

    def shift(self, v, i, u, size) -> None:
        """The same guarded write, as a method."""
        shift_guarded(v, i, u, size)


shifter = _Shifter()

#: The same helper as a *bound method* sitting in a global, which is a shape the
#: static walk does follow.
bound_shift = shifter.shift


def fill(x, value) -> None:
    """A helper that asks an argument for its domain: the body never says ``dom``."""
    for i in x.dom:
        x[i] = value


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
    # A closed claim refuted at no assignment in particular: lanky prints the
    # reason of such a fact under its REFUTED line, so the fix reaches the
    # terminal and not only the JSON ledger.
    assert facts[0].provenance["counterexample"] == {}
    assert facts[0].provenance["reason"] == facts[0].provenance["error"]


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


def test_a_programs_restatement_rests_on_the_callees_own_fact() -> None:
    """``rests_on`` names the id the callee's postcondition fact really has.

    It used to be a ``from`` entry in the provenance, which lanky could not
    read, so the ledger showed the restatement as a free-standing assumption.
    """
    (restated,) = both.facts()
    (post,) = [fact for fact in scan.facts() if fact.kind == "postcondition"]
    assert restated.rests_on == (post.id,)
    assert restated.provenance == {"callee": scan.qualname}

    ledger = Ledger([*scan.facts(), restated])
    assert ledger.support(restated).under == (post.id,)
    assert ledger.support(restated).effective is Status.ASSUMED
    # the kernel owns several facts, so the one meant is named by its id
    lines = ledger.render().splitlines()
    (row,) = [line for line in lines if "after scan(...)" in line]
    assert row.startswith(f"assumed under {post.id}  ")


def test_the_ledger_of_a_file_says_what_its_program_rests_on() -> None:
    """Checked with its callees, a program's restatements name their facts."""
    from lanky.check import check_path

    ledger = check_path(KERNELS / "spmv_min.py")
    (restated,) = [fact for fact in ledger if fact.kind == "postcondition-in-scope"]
    (post,) = [fact for fact in ledger if fact.kind == "postcondition"]
    assert restated.rests_on == (post.id,)
    assert ledger.support(restated).under == (post.id,)
    data = json.loads(ledger.to_json())
    (row,) = [row for row in data if row["kind"] == "postcondition-in-scope"]
    assert row["rests_on"] == [post.id]
    assert row["under"] == [post.id]
    assert row["effective"] == "assumed"


CALLEE = '''
from __future__ import annotations

from lanky.prelude import Nat

from loopty import Arr, Fin, kernel


@kernel
def scan(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """The offsets, in a file of their own."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]
'''

CALLER = """
from __future__ import annotations

from loopty_test_callee import scan

from loopty import program


@program
def solve(cnt, off):
    \"\"\"Runs a kernel of another file.\"\"\"
    scan(cnt, off)
"""


def test_a_callee_of_another_file_is_an_id_this_ledger_does_not_hold(
    tmp_path, capsys
) -> None:
    """The callee's fact is in its own file's ledger, and lanky says so.

    The restatement still rests on the id the callee's fact has there, which
    this ledger counts as an assumption and names under the table, without
    failing the check.
    """
    import sys

    from lanky import cli

    (tmp_path / "loopty_test_callee.py").write_text(CALLEE, encoding="utf-8")
    caller = tmp_path / "caller.py"
    caller.write_text(CALLER, encoding="utf-8")
    try:
        assert cli.main(["check", str(caller)]) == 0
    finally:
        sys.modules.pop("loopty_test_callee", None)
    printed = capsys.readouterr().out.splitlines()
    (row,) = [line for line in printed if "after scan(...)" in line]
    assert row.startswith("assumed under scan:postcondition  ")
    unresolved = "rests on scan:postcondition, which this ledger does not hold"
    assert any(
        line.startswith("UNRESOLVED solve at caller.py:") and line.endswith(unresolved)
        for line in printed
    )


def test_a_guard_is_found_under_an_aliased_import() -> None:
    # ``guards_writes`` used to be ``"when" in co_names``, so importing the
    # guard under another name ran the native body unmasked: the write under a
    # false condition was performed, and ``python file.py`` computed something
    # the lowered kernel does not.
    from loopty import when as guard

    @kernel
    def aliased(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            with guard(i + 1 < u.dom.size):
                v[i] = u[i + 1]

    assert aliased.guards_writes
    out = Arr.zeros(3)
    aliased(Arr.from_numpy(np.array([1.0, 2.0, 3.0])), out)
    assert list(out.numpy()) == [2.0, 3.0, 0.0]


def test_a_guard_is_found_through_a_module_attribute() -> None:
    import loopty

    @kernel
    def qualified(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            with loopty.when(i + 1 < u.dom.size):
                v[i] = u[i + 1]

    assert qualified.guards_writes
    out = Arr.zeros(3)
    qualified(Arr.from_numpy(np.array([1.0, 2.0, 3.0])), out)
    assert list(out.numpy()) == [2.0, 3.0, 0.0]


def test_a_guard_is_found_through_a_renamed_module_attribute() -> None:
    # Neither the name ``when`` nor a global bound to it appears: the body
    # reaches the guard as ``guards.mask``. Identity is what finds it.
    @kernel
    def renamed(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            with guards.mask(i + 1 < u.dom.size):
                v[i] = u[i + 1]

    assert "when" not in renamed.fn.__code__.co_names
    assert renamed.guards_writes
    out = Arr.zeros(3)
    renamed(Arr.from_numpy(np.array([1.0, 2.0, 3.0])), out)
    assert list(out.numpy()) == [2.0, 3.0, 0.0]


def test_every_native_run_sees_a_masking_view_that_shares_the_buffer() -> None:
    # The contract used to be "an unguarded kernel is called with exactly the
    # objects it was given", which made masking depend on a static guess about
    # the body. It is now "every array argument is a masking view", because no
    # inspection can decide whether a helper opens a guard. The view shares the
    # buffer, so the results are still written in place.
    given = Arr.from_numpy(np.array([3.0, 4.0]))
    seen = []

    @kernel
    def peek(x: Arr[Fin[n], Real]):  # noqa: F821
        seen.append(x)
        for i in x.dom:
            x[i] = 2.0 * x[i]

    peek(given)
    assert not scale.guards_writes
    assert seen[0] is not given
    assert isinstance(seen[0], Arr)
    assert seen[0].numpy() is given.numpy()
    assert list(given.numpy()) == [6.0, 8.0]


def test_a_guard_opened_by_a_global_helper_masks_the_native_write() -> None:
    # The body names no guard at all: ``shift_guarded`` opens it. The guard
    # detection used to compare the helper with ``when`` and never look inside
    # it, so the native run performed the guarded write unmasked and computed
    # something the lowered kernel does not.
    @kernel
    def via_helper(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            shift_guarded(v, i, u, u.dom.size)

    assert "when" not in via_helper.fn.__code__.co_names
    assert via_helper.guards_writes
    out = Arr.zeros(3)
    via_helper(Arr.from_numpy(np.array([1.0, 2.0, 3.0])), out)
    assert list(out.numpy()) == [2.0, 3.0, 0.0]


def test_a_guard_opened_through_a_bound_method_masks_the_native_write() -> None:
    @kernel
    def via_method(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            bound_shift(v, i, u, u.dom.size)

    assert via_method.guards_writes
    out = Arr.zeros(3)
    via_method(Arr.from_numpy(np.array([1.0, 2.0, 3.0])), out)
    assert list(out.numpy()) == [2.0, 3.0, 0.0]


def test_a_guard_no_static_walk_can_see_still_masks_the_native_write() -> None:
    # The helper is reached as an attribute of an ordinary object, which
    # ``opens_a_guard`` deliberately does not follow (reading attributes off an
    # arbitrary object can run a property). The flag is therefore False and the
    # write is masked anyway, which is the point of masking every run.
    @kernel
    def via_attribute(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            shifter.shift(v, i, u, u.dom.size)

    assert not via_attribute.guards_writes
    out = Arr.zeros(3)
    via_attribute(Arr.from_numpy(np.array([1.0, 2.0, 3.0])), out)
    assert list(out.numpy()) == [2.0, 3.0, 0.0]


def test_a_helper_that_iterates_dom_accepts_a_bare_numpy_argument() -> None:
    # The ``.dom`` a body needs may be asked for by a helper, and the top-level
    # ``co_names`` check did not see it, so a kernel called with a bare ndarray
    # failed with "'ndarray' object has no attribute 'dom'". Every argument is
    # wrapped now, so the question does not arise.
    @kernel
    def fill_all(x: Arr[Fin[n], Real]):  # noqa: F821
        fill(x, 7.0)

    assert "dom" not in fill_all.fn.__code__.co_names
    given = np.zeros(3)
    fill_all(given)
    assert list(given) == [7.0, 7.0, 7.0]


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


def test_a_native_run_does_not_wrap_a_negative_index() -> None:
    # ``x[i - 1]`` at ``i = 0`` is refuted by the typing rules and reads in front
    # of the buffer in generated C. numpy would read the last cell instead, so
    # the reference run computed a value that neither the ledger nor the
    # compiled code agrees with.
    @kernel
    def lag(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = x[i - 1]

    with pytest.raises(IndexError, match="is negative"):
        lag(np.array([1.0, 2.0, 3.0]), np.zeros(3))


def test_a_negative_read_under_a_false_guard_still_answers_zero() -> None:
    # The boundary guard of a stencil evaluates ``x[i - 1]`` at ``i = 0`` too;
    # the masked read answers zero there and the write is dropped, as it is for
    # a read past the other end.
    @kernel
    def guarded_lag(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(i > 0):
                y[i] = x[i - 1]

    y = np.full(3, -1.0)
    guarded_lag(np.array([1.0, 2.0, 3.0]), y)
    assert list(y) == [-1.0, 1.0, 2.0]
