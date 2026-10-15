"""A postcondition decided from what the statements store (#113), and cited theorems.

Where no cell of an array is stored twice, or read before its store, what each
statement stores is what the kernel leaves (:mod:`loopty.dataflow`): its value
equation. The postcondition is decided under them, a claim about reals with
each equation an atom, and a theorem the kernel cites is applied to it. The
first half needs no Lean; the second runs it, and skips without it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from lanky import theorem
from lanky.ledger import Ledger, Status
from lanky.plugins import registry
from lanky.prelude import Fn, Int, Nat, Real
from lanky.terms import render

from loopty import Arr, Fin, kernel
from loopty.dataflow import APPLIED, KIND, stores
from loopty.schedule import IllegalCast

pytest.importorskip("loopy")

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "tests" / "kernels"
EXAMPLES = ROOT / "examples"


def load(path: Path) -> Any:
    """Import a file of kernels as a module of its own."""
    name = f"dataflow_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


# imported with what it registers released again, so that the registry other
# test modules read is the one they made
with registry.collecting():
    kernels = load(KERNELS / "dataflow.py")


def facts(kernel: Any, kind: str) -> list[Any]:
    return [fact for fact in kernel.facts() if fact.kind == kind]


# {{{ decided


def test_the_scans_postcondition_is_decided_from_what_it_stores() -> None:
    """``off[r + 1] == off[r] + cnt[r]`` is what ``S1`` stores at ``r``: decided by isl.

    It used to be ``tested`` on the native runs, which it still is, as
    ``native`` in its provenance; it rests on the fact that no cell of
    ``off`` is stored twice or read before its store, which comes before it.
    """
    (post,) = facts(kernels.scan, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert post.provenance["native"] == "tested"
    (single,) = facts(kernels.scan, KIND)
    assert (single.status, single.decided_by) == (Status.DECIDED, "isl")
    assert (
        single.statement == "no cell of off is stored twice, or read before its store"
    )
    assert post.rests_on == (single.id,)
    assert single.id == f"{KIND}:dataflow.scan@18:off"
    ids = [fact.id for fact in kernels.scan.facts()]
    assert ids.index(single.id) == ids.index(post.id) - 1
    assert post.provenance["used"] == [
        f"what {stmt.id} stores in off at {stmt.where}"
        for stmt in kernels.scan.term.stmts
    ]


def test_the_value_equations_are_what_each_statement_stores_where_it_runs() -> None:
    found = stores(kernels.scan.term, "scan")
    texts = [render(equation.claim) for equation in found.equations]
    assert texts[0] == "off[0] == 0"
    assert texts[1].startswith("forall r in Int where ")
    assert texts[1].endswith(". off[r + 1] == off[r] + cnt[r]")
    assert "0 <= r and r < n" in texts[1]


def test_the_isl_oracle_decides_the_postcondition_again() -> None:
    from loopty.oracle import IslOracle

    (post,) = facts(kernels.scan, "postcondition")
    (single,) = facts(kernels.scan, KIND)
    oracle = IslOracle()
    for fact in (post, single):
        handed = fact.with_status(Status.ASSUMED)
        assert oracle.establish(handed).status is Status.DECIDED


def test_a_postcondition_about_reals_is_decided_with_each_equation_an_atom() -> None:
    """``R[a + 2, b] == -R[a, b + 2]`` is what ``S1`` stores at ``(a + 2, b)``."""
    (post,) = facts(kernels.triangle, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert len(post.provenance["used"]) == 2


def test_a_statement_without_loops_is_stated_for_lean_as_storing_once() -> None:
    """One instance, so no two that differ: a part, its hypothesis ``False``."""
    (fact,) = facts(kernels.one_cell, KIND)
    assert fact.status is Status.DECIDED
    obligation = fact.term.lean_obligation()
    assert not obligation.declined
    (part,) = obligation.parts
    assert "two instances of S0 storing into lim" in part.label


# }}}


# {{{ not decided, and why


def test_a_cell_stored_twice_leaves_the_postcondition_tested() -> None:
    (post,) = facts(kernels.twice, "postcondition")
    assert post.status is Status.TESTED
    assert "S0 and S1 store into one cell of y" in post.provenance["undecided"]
    assert not facts(kernels.twice, KIND)


def test_a_read_before_its_store_leaves_the_postcondition_tested() -> None:
    """True at every input, and not decided: what ``S0`` read was stored after it."""
    (post,) = facts(kernels.from_the_right, "postcondition")
    assert post.status is Status.TESTED
    assert (
        "S0 stores a cell of y that S0 reads before it" in post.provenance["undecided"]
    )


@pytest.mark.parametrize("name", ["in_place", "halves"])
def test_a_cell_the_storing_instance_reads_first_is_read_before_its_store(
    name: str,
) -> None:
    """The instance that reads a cell and stores it reads the value from before.

    So its store is no equation about what the kernel leaves: ``x == 2x -
    1000005`` would decide a claim that is false where ``x`` holds 1000004 on
    entry. ``halves`` reads its own cell at ``i = 0`` only.
    """
    k = getattr(kernels, name)
    (post,) = facts(k, "postcondition")
    assert post.status is Status.TESTED
    array = "x" if name == "in_place" else "y"
    assert (
        f"S0 stores a cell of {array} that S0 reads before it"
        in post.provenance["undecided"]
    )
    assert not facts(k, KIND)
    assert not stores(k.term, name).equations


def test_a_cell_no_statement_stores_leaves_the_claim_about_it_undecided() -> None:
    """``y[i] == 1`` past ``i = 3`` is what ``y`` held, which no equation says."""
    from loopty.dataflow import decide_postcondition

    (post,) = facts(kernels.first_three, "postcondition")
    assert post.status is not Status.DECIDED
    term = kernels.first_three.term
    found = stores(term, "first_three")
    assert found.equations  # y is stored once where it is stored
    assert isinstance(decide_postcondition(term, found.equations), str)


def test_a_read_before_the_statement_that_stores_it_runs_leaves_it_undecided() -> None:
    """The scan's loop reads ``off[0]``, which ``S1`` stores after it."""
    (post,) = facts(kernels.scan_backwards, "postcondition")
    assert post.status is not Status.DECIDED
    assert not facts(kernels.scan_backwards, KIND)
    assert not stores(kernels.scan_backwards.term, "scan_backwards").equations


def test_a_recurrence_with_its_sign_lost_is_no_atom_it_stores() -> None:
    """Refuted by the native runs, never decided: another equation than the stored."""
    (post,) = facts(kernels.lost_sign, "postcondition")
    assert post.status is Status.REFUTED
    from loopty.dataflow import decide_postcondition

    term = kernels.lost_sign.term
    found = stores(term, "lost_sign")
    assert isinstance(decide_postcondition(term, found.equations), str)


# }}}


# {{{ updates: a cell holds what it held, plus every update


def _update_reason(stmt: str, cell: str) -> str:
    return (
        f"{stmt} updates {cell} in place, so a cell of y holds what it held on "
        "entry plus every update, not what one statement stores"
    )


@pytest.mark.parametrize(
    ("name", "stmt", "cell"),
    [
        ("bumped", "S0", "y[i]"),
        ("scattered", "S0", "y[cell[c, k]]"),
        ("zeroed_then_added", "S1", "y[i_0]"),  # its second loop's i
    ],
)
def test_an_updated_array_has_no_equation_and_its_claim_is_left_to_the_runs(
    name: str, stmt: str, cell: str
) -> None:
    """``y[i] += v`` stores ``y[i] + v``: the cell's value on entry, plus each update.

    So no one statement says what the kernel leaves, and the claim is left to
    the native runs, with why. Read as an equation, ``bumped``'s ``y[i] ==
    y[i] + 5`` holds nowhere and would decide its claim, which is false where
    a cell held 999998; ``zeroed_then_added``'s is true, and is not decided
    either.
    """
    k = getattr(kernels, name)
    (post,) = facts(k, "postcondition")
    assert post.status is not Status.DECIDED
    assert post.status is not Status.REFUTED
    assert _update_reason(stmt, cell) in post.provenance["undecided"]
    assert not facts(k, KIND)
    found = stores(k.term, name)
    assert not found.equations
    assert found.why == {"y": _update_reason(stmt, cell)}


def test_an_update_beside_an_array_stored_once_leaves_that_arrays_claim_decided() -> (
    None
):
    """``z`` is stored once per cell and reads nothing of ``y``: decided from it."""
    (post,) = facts(kernels.doubled_beside, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    (single,) = facts(kernels.doubled_beside, KIND)
    assert single.provenance["array"] == "z"
    assert post.rests_on == (single.id,)
    found = stores(kernels.doubled_beside.term, "doubled_beside")
    assert found.why == {"y": _update_reason("S1", "y[i]")}


# }}}


# {{{ a cited theorem


def test_a_conjunct_the_cited_theorem_gives_is_decided_under_it() -> None:
    """The monotonicity is ``offsets_increase`` at ``off`` and ``cnt``.

    Its hypotheses are the conjuncts decided before.
    """
    (post,) = facts(kernels.scan_monotone, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert kernels.offsets_increase.fact_id in post.rests_on
    assert any(
        source.startswith("offsets_increase at ") for source in post.provenance["used"]
    )
    # its conclusion is a conjunct already: no fact of its own
    assert not facts(kernels.scan_monotone, APPLIED)


def test_a_cited_theorem_gives_its_conclusion_from_the_postcondition() -> None:
    (applied,) = facts(kernels.scan_cited, APPLIED)
    assert (applied.status, applied.decided_by) == (Status.DECIDED, "theorem")
    assert applied.statement == (
        "by offsets_increase: forall a in Fin(n + 1), b in Fin(n + 1) where a <= b. "
        "off[a] <= off[b]"
    )
    (post,) = facts(kernels.scan_cited, "postcondition")
    assert applied.rests_on == (kernels.offsets_increase.fact_id, post.id)
    assert applied.provenance["bindings"] == "n = n, cnt = cnt, off = off"
    ids = [fact.id for fact in kernels.scan_cited.facts()]
    assert ids.index(applied.id) == ids.index(post.id) + 1


def test_the_flagships_theorem_gives_r_is_d_for_every_d_c_holds() -> None:
    """``on_the_triangle``'s copies and recurrence are ``reconstruct``'s postcondition.

    The PDE and what ``c`` stores of ``D`` are no conjunct of it, so the
    claim is for every ``D`` they hold of: ``R`` is ``D`` on the triangle.
    """
    (applied,) = facts(kernels.reconstruct, APPLIED)
    claim = applied.term.application.claim
    assert [var.name for var, _ in claim.binders] == ["D"]
    assert render(claim.guard).startswith(
        "(forall a in Nat, b in Nat. D(a + 2)(b) + D(a)(b + 2) == 0) and "
    )
    assert "c[a, b] == D(a)(b)" in render(claim.guard)
    assert render(claim.body) == (
        "forall a in Fin(p + 1), b in Fin(p + 1) where a + b <= p. R[a, b] == D(a)(b)"
    )
    assert applied.provenance["bindings"] == "p = p, c = c, R = R"
    assert applied.provenance["matched"] == [
        render(part) for part in kernels.reconstruct.term.post.children
    ]


def test_a_kernel_computing_another_recurrence_gets_nothing_from_its_theorem() -> None:
    """It claims the theorem's recurrence: refuted by its runs, never applied."""
    (post,) = facts(kernels.wrong_coefficient, "postcondition")
    assert post.status is Status.REFUTED
    assert not facts(kernels.wrong_coefficient, APPLIED)
    from loopty.dataflow import decide_postcondition

    term = kernels.wrong_coefficient.term
    found = stores(term, "wrong_coefficient")
    decision = decide_postcondition(
        term, found.equations, theorems=kernels.wrong_coefficient.uses
    )
    assert isinstance(decision, str)


def test_a_kernel_off_by_one_gets_nothing_from_the_theorem() -> None:
    """Its postcondition is what it stores; the theorem's recurrence is not that.

    Matched alone, the copies would bind the theorem's ``D`` to ``c`` and its
    ``R`` to every table at all, under a recurrence the kernel does not
    compute: every hypothesis about an array the kernel writes has to be a
    conjunct of its postcondition.
    """
    (post,) = facts(kernels.off_by_one, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert not facts(kernels.off_by_one, APPLIED)


@pytest.mark.parametrize("name", ["scaled", "summed"])
def test_a_conjunct_that_rounds_is_no_hypothesis_of_a_theorem_about_reals(
    name: str,
) -> None:
    """``R[i] == c[i] * 0.1`` holds of the floats and not of them as reals.

    Decided as the kernel computes, it stays decided; a theorem about the
    reals concludes ``R[i] * 10 == c[i]`` from it, which is false at ``c[i] =
    3.0``, and is not applied.
    """
    k = getattr(kernels, name)
    (post,) = facts(k, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert not facts(k, APPLIED)
    assert 3.0 * 0.1 * 10 != 3.0


def test_a_conjunct_reads_the_same_as_reals_where_nothing_in_it_rounds() -> None:
    from lanky.terms import conjuncts

    from loopty.dataflow import exact

    assert all(
        exact(part, {"c", "R"}) for part in conjuncts(kernels.triangle.term.post)
    )
    assert not exact(kernels.scaled.term.post, {"c", "R"})
    assert not exact(kernels.summed.term.post, {"a", "b", "s"})
    # over the integers, every comparison is exact
    assert all(exact(part, set()) for part in conjuncts(kernels.scan.term.post))


def test_a_theorems_integer_is_never_taken_at_a_real() -> None:
    """``integral_fill``'s ``k`` is an integer: taken at ``fill``'s real ``s`` it says
    ``R[i] * 2 != 1``, false at ``s = 0.5``. The match is refused (Codex).
    """
    (post,) = facts(kernels.fill, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert not facts(kernels.fill, APPLIED)


def test_a_conclusion_that_rounds_decides_no_conjunct_and_is_no_row() -> None:
    """``(R[i] + 1) - R[i] == 1`` holds as reals and not of floats at ``2**53`` (Codex).

    Given by ``plus_one`` from the exact copy, it decides no conjunct the
    floats are claimed of, and is no ``by-theorem`` row.
    """
    (post,) = facts(kernels.round_trip, "postcondition")
    assert post.status is not Status.DECIDED
    assert (2.0**53 + 1) - 2.0**53 != 1
    assert not facts(kernels.ones, APPLIED)


def test_a_theorems_integer_decides_nothing_at_a_real() -> None:
    """``integral_half``'s ``k`` at a real ``s`` would decide ``R[i] != 0.5``."""
    (post,) = facts(kernels.fill_half, "postcondition")
    assert post.status is not Status.DECIDED
    assert kernels.integral_half.fact_id not in post.rests_on


def test_a_theorems_real_is_taken_at_a_kernels_real_scalar() -> None:
    """``constant_table``'s ``k`` is a real, and ``fill_real``'s ``s`` is one (#247).

    The cells hold ``s`` exactly, so ``R[i] == s`` reads the same as reals,
    and the theorem gives that any two cells agree; claimed, that conjunct
    is decided by the theorem's instance. ``s + 0.1`` rounds, and the
    theorem's ``k`` does not stand for it.
    """
    (applied,) = facts(kernels.fill_real, APPLIED)
    assert applied.statement == (
        "by constant_table: forall i in Fin(n), j in Fin(n). R[i] == R[j]"
    )
    assert applied.provenance["bindings"] == "n = n, k = s, R = R"
    (post,) = facts(kernels.fill_agrees, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert kernels.constant_table.fact_id in post.rests_on
    assert not facts(kernels.fill_shifted, APPLIED)


def test_numpys_floats_are_reals_to_a_theorem_about_reals() -> None:
    (applied,) = facts(kernels.copy64, APPLIED)
    assert applied.statement == "by copied: forall i in Fin(n). -R[i] == -c[i]"


def test_a_theorem_that_does_not_apply_gives_nothing() -> None:
    assert not facts(kernels.scan_unrelated, APPLIED)
    # over Nat: the cells the scan writes are not known to be naturals
    assert not facts(kernels.scan_naturals, APPLIED)
    (post,) = facts(kernels.scan_naturals, "postcondition")
    assert kernels.naturals_increase.fact_id not in post.rests_on


def test_uses_takes_theorems_and_refuses_anything_else() -> None:
    from loopty import kernel

    with pytest.raises(TypeError, match="uses= of the kernel names theorems"):
        kernel(uses=["postcondition:spmv.scan@69"])
    with pytest.raises(TypeError, match="the theorems a kernel cites go in uses="):
        kernel(kernels.offsets_increase)


def test_the_ledger_counts_the_theorem_in_what_the_application_is_worth() -> None:
    applied = facts(kernels.scan_cited, APPLIED)[0]
    ledger = Ledger([*kernels.scan_cited.facts()])
    # the theorem is not in this ledger: it is what the application is under
    assert ledger.support(applied).under == (kernels.offsets_increase.fact_id,)


class Sized:
    """Kernels of a size ``p`` citing theorems over an ``n`` (#239).

    Made here, rather than in ``kernels/dataflow.py``, so that the coverage
    report over the test kernels (``docs/lean-checks.md``) stays as it is.
    """

    def __init__(self) -> None:
        @theorem
        def sized(
            n: Nat,
            c: Fn[Nat, Real],
            D: Fn[Fin[n], Real],  # noqa: F821
            R: Fn[Nat, Real],
            copies: all(R(i) == c(i) for i in Fin[n]),  # noqa: F821
            stored: all(c(i) == D(i) for i in Fin[n]),  # noqa: F821
        ) -> all(R(i) == D(i) for i in Fin[n]):  # noqa: F821
            """A copy of what holds ``D``'s entries is ``D``."""

        @kernel(uses=[sized])
        def copy_p(
            c: Arr[Fin[p], Real],  # noqa: F821
            R: Arr[Fin[p], Real],  # noqa: F821
        ) -> all(R[i] == c[i] for i in Fin[p]):  # noqa: F821
            """The issue's kernel: its size is ``p``, and it has no ``n``."""
            for i in R.dom:
                R[i] = c[i]

        @kernel(uses=[sized])
        def copy_pn(
            c: Arr[Fin[p], Real],  # noqa: F821
            R: Arr[Fin[p], Real],  # noqa: F821
            s: Arr[Fin[n], Real],  # noqa: F821
        ) -> all(R[i] == c[i] for i in Fin[p]):  # noqa: F821
            """Has an ``n`` of its own, ``s``'s length, which is not ``p``."""
            for i in R.dom:
                R[i] = c[i]
            for j in s.dom:
                s[j] = 0.0

        @theorem
        def sized_late(
            n: Nat,
            D: Fn[Fin[m], Real],  # noqa: F821
            m: Nat,
            c: Fn[Nat, Real],
            R: Fn[Nat, Real],
            copies: all(R(i) == c(i) for i in Fin[n]),  # noqa: F821
            stored: all(c(i) == D(i) for i in Fin[n]),  # noqa: F821
            wide: n <= m,  # noqa: F821
        ) -> all(R(i) == D(i) for i in Fin[n]):  # noqa: F821
            """``sized`` with ``D`` over a size of its own, named after ``D``."""

        @kernel(uses=[sized_late])
        def copy_m(
            c: Arr[Fin[p], Real],  # noqa: F821
            R: Arr[Fin[p], Real],  # noqa: F821
            s: Arr[Fin[m], Real],  # noqa: F821
        ) -> all(R[i] == c[i] for i in Fin[p]):  # noqa: F821
            """Has an ``m`` of its own, which the theorem's ``m`` is not."""
            for i in R.dom:
                R[i] = c[i]
            for j in s.dom:
                s[j] = 0.0

        @theorem
        def negated(
            n: Nat,
            c: Fn[Nat, Real],
            R: Fn[Nat, Real],
            h: all(R[i] == c[i] for i in Fin[n]),  # noqa: F821
        ) -> all(-R[i] == -c[i] for i in Fin[n]):  # noqa: F821
            """A copy's sign changes as what it copies does; families subscripted."""

        @kernel(uses=[negated])
        def copy_ab(
            a: Arr[Fin[p], Real],  # noqa: F821
            b: Arr[Fin[p], Real],  # noqa: F821
        ) -> all(b[i] == a[i] for i in Fin[p]):  # noqa: F821
            """Copies ``a`` into ``b``, names the theorem's families are not."""
            for i in b.dom:
                b[i] = a[i]

        @theorem
        def squared(
            n: Nat,
            m: Int,
            c: Fn[Nat, Int],
            R: Fn[Nat, Int],
            h1: all(c(i) == m for i in Fin[n]),  # noqa: F821
            h2: all(R(i) == m**2 for i in Fin[n]),  # noqa: F821
        ) -> all(R(i) >= c(i) for i in Fin[n]):  # noqa: F821
            """An integer's square is at least the integer."""

        @kernel(uses=[squared])
        def square_of_m(
            k: Int,
            m: Int,
            c: Arr[Fin[p], Int],  # noqa: F821
            R: Arr[Fin[p], Int],  # noqa: F821
        ) -> all(c[i] == k for i in Fin[p]) & all(  # noqa: F821
            R[i] == m**2 for i in Fin[p]  # noqa: F821
        ):
            """``c`` holds ``k`` and ``R`` the square of another integer, ``m``."""
            for i in R.dom:
                c[i] = k
                R[i] = m**2

        @kernel(uses=[squared])
        def square_of_k(
            k: Int,
            c: Arr[Fin[p], Int],  # noqa: F821
            R: Arr[Fin[p], Int],  # noqa: F821
        ) -> all(c[i] == k for i in Fin[p]) & all(  # noqa: F821
            R[i] == k**2 for i in Fin[p]  # noqa: F821
        ):
            """``R`` holds the square of what ``c`` holds."""
            for i in R.dom:
                c[i] = k
                R[i] = k**2

        self.sized, self.copy_p, self.copy_pn = sized, copy_p, copy_pn
        self.sized_late, self.copy_m = sized_late, copy_m
        self.copy_ab = copy_ab
        self.square_of_m, self.square_of_k = square_of_m, square_of_k


@pytest.fixture(scope="module")
def sized() -> Sized:
    with registry.collecting():
        return Sized()


def test_a_family_nothing_bound_ranges_over_the_kernels_size(sized: Sized) -> None:
    """``sized``'s ``D`` ranges over ``Fin[n]``, and its ``n`` is the kernel's ``p``.

    So the claim is for every ``D`` over ``Fin(p)``. At ``Fin(n)`` it named a
    size ``copy_p`` does not have, and in ``copy_pn``, whose ``n`` is the
    length of another array, a size that is not ``p`` (#239).
    """
    for k in (sized.copy_p, sized.copy_pn):
        (applied,) = facts(k, APPLIED)
        assert (applied.status, applied.decided_by) == (Status.DECIDED, "theorem")
        claim = applied.term.application.claim
        assert [(var.name, str(sort)) for var, sort in claim.binders] == [
            ("D", "Fn[Fin(p), Real]")
        ]
        assert render(claim) == (
            "forall D in Fn[Fin(p), Real] where forall i in Fin(p). c[i] == D(i). "
            "forall i in Fin(p). R[i] == D(i)"
        )
        assert applied.provenance["bindings"] == "n = p, c = c, R = R"


def test_a_family_sized_by_a_variable_nothing_bound_comes_after_it(
    sized: Sized,
) -> None:
    """``sized_late``'s ``D`` ranges over ``Fin[m]``, and nothing binds its ``m``.

    ``m`` is a binder of the claim, named apart from ``copy_m``'s own ``m``,
    and bound before ``D``, whose sort names it: before, ``D`` ranged over
    ``Fin(m)`` of the kernel's ``m``, the length of ``s`` (#239).
    """
    (applied,) = facts(sized.copy_m, APPLIED)
    claim = applied.term.application.claim
    assert [(var.name, str(sort)) for var, sort in claim.binders] == [
        ("m_0", "Nat"),
        ("D", "Fn[Fin(m_0), Real]"),
    ]
    assert render(claim.guard).endswith("p <= m_0")
    assert applied.statement.startswith("by sized_late, for every m_0, D with ")


def test_a_family_applied_by_subscript_is_the_array_it_matched(sized: Sized) -> None:
    """A theorem's ``R[i]`` is ``R`` applied, as ``R(i)`` is: ``copy_ab``'s ``b[i]``."""
    (applied,) = facts(sized.copy_ab, APPLIED)
    assert applied.statement == "by negated: forall i in Fin(p). -b[i] == -a[i]"
    assert applied.provenance["bindings"] == "n = p, c = a, R = b"


def test_a_theorems_variable_in_a_power_is_never_the_kernels_of_its_spelling(
    sized: Sized,
) -> None:
    """``squared``'s ``m`` is the ``k`` that ``c`` holds, so ``h2`` is about ``k**2``.

    ``square_of_m``'s ``R`` holds its own ``m**2``, which is not that: read as
    the kernel's ``m``, the theorem gave ``R[i] >= c[i]``, false at ``k = 5``
    and ``m = 0`` (#239). Where ``R`` holds ``k**2``, it is given.
    """
    assert all(
        not fact.statement.endswith("R[i] >= c[i]")
        for fact in facts(sized.square_of_m, APPLIED)
    )
    (applied,) = facts(sized.square_of_k, APPLIED)
    assert applied.statement == "by squared: forall i in Fin(p). R[i] >= c[i]"


# }}}


# {{{ lanky check lists the casts of a file's schedules (#191)


def test_lanky_check_lists_the_casts_of_the_files_schedules_once(capsys: Any) -> None:
    """Two schedules that share their first step share its facts: one claim each."""
    from lanky import cli
    from lanky.check import check_path

    ledger = check_path(KERNELS / "dataflow.py")
    casts = [fact for fact in ledger if fact.id.startswith("cast:")]
    assert casts
    assert all(
        fact.status is Status.DECIDED and fact.decided_by == "isl" for fact in casts
    )
    first = [
        fact
        for fact in casts
        if "split('r', 2" in fact.id and "split('r_outer'" not in fact.id
    ]
    assert len(first) == 2  # bijective and monotone, once
    assert not [fact for fact in ledger if fact.provenance.get("duplicate_claims")]
    assert (
        cli.main(["check", str(KERNELS / "dataflow.py")]) == 1
    )  # the refuted postcondition
    out = capsys.readouterr().out
    assert "DUPLICATE" not in out
    assert "split('r', 2" in out or "split(r, 2" in out


# }}}


# {{{ a layout and a window: the postcondition is the term's either way


def test_a_postcondition_read_through_a_layout_is_decided() -> None:
    """``x[back_to_front(i)]`` is ``x[3 - i]`` in the term, and the claim says so."""
    (post,) = facts(kernels.reversed_copy, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    (single,) = facts(kernels.reversed_copy, KIND)
    assert post.rests_on == (single.id,)


def test_a_window_leaves_the_producers_postcondition_decided_from_its_term() -> None:
    """``contract('f')`` is the program's schedule; ``doubled``'s claim, its term's."""
    assert kernels.windowed_differences.facts()
    (post,) = facts(kernels.doubled, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    (restated,) = [
        fact
        for fact in kernels.differences.facts()
        if fact.kind == "postcondition-in-scope"
    ]
    assert restated.rests_on == (post.id,)


def test_lanky_check_lists_a_layouts_facts_and_a_windows_casts_once() -> None:
    """The layout's three facts, the window's casts under the program, each once."""
    from lanky.check import check_path

    ledger = list(check_path(KERNELS / "dataflow.py"))
    ids = [fact.id for fact in ledger]
    assert len(ids) == len(set(ids))
    kinds = {
        fact.kind: fact.status
        for fact in ledger
        if fact.owner == "back_to_front" and fact.kind.startswith("layout-")
    }
    assert kinds == {
        "layout-in-bounds": Status.DECIDED,
        "layout-one-for-one": Status.DECIDED,
        "layout-onto": Status.DECIDED,
    }
    posts = {fact.owner: fact for fact in ledger if fact.kind == "postcondition"}
    for owner in ("reversed_copy", "doubled"):
        assert (posts[owner].status, posts[owner].decided_by) == (
            Status.DECIDED,
            "isl",
        )
    (clobber,) = [fact for fact in ledger if fact.kind == "no-clobber"]
    assert clobber.owner == "differences"
    assert (clobber.status, clobber.decided_by) == (Status.DECIDED, "isl")
    assert clobber.id in {fact.id for fact in kernels.windowed_differences.facts()}
    window = [fact for fact in ledger if ".contract('f'" in fact.id]
    assert window
    assert all(fact.owner == "differences" for fact in window)


def test_a_substitutions_and_a_windows_definedness_read_apart_in_the_ledger() -> None:
    """Both steps need every cell of ``f`` stored before it is read: two rows.

    ``examples/fusion.py`` substitutes ``f`` in one schedule and keeps it in a
    window in another, after the same fusion. Each step's fact leads with the
    step, so the two rows of ``lanky check`` read apart.
    """
    from lanky.check import check_path

    ledger = list(check_path(EXAMPLES / "fusion.py"))
    rows = [
        fact.statement
        for fact in ledger
        if fact.kind == "definedness" and fact.statement.startswith("for ")
    ]
    assert len(rows) == len(set(rows)) == 2
    assert any(row.startswith("for substitute('f'), every cell of f") for row in rows)
    assert any(row.startswith("for contract('f'") for row in rows)


# }}}


# {{{ with Lean


def _without_lean(reason: str) -> None:
    if os.environ.get("LANKY_LEAN_TEST_REQUIRED"):
        pytest.fail(f"LANKY_LEAN_TEST_REQUIRED is set, but {reason}", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def lean(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A Lean oracle with an open session and a proof cache of its own."""
    from lanky.oracles.lean import LeanOracle, ProofCache

    if os.environ.get("LANKY_LEAN_DISABLE"):
        _without_lean("the Lean oracle is disabled by LANKY_LEAN_DISABLE")
    oracle = LeanOracle(timeout=float(os.environ.get("LANKY_LEAN_TEST_TIMEOUT", "120")))
    available, reason = oracle.availability()
    if not available:
        _without_lean(f"no Lean oracle here: {reason}")
    if not oracle.session.start():
        _without_lean(f"the Lean REPL could not be built: {oracle.session.error}")
    oracle.proofs = ProofCache(str(tmp_path_factory.mktemp("proofs")))
    yield oracle
    for session in oracle._pool:
        session.close()
    oracle.session.close()


@pytest.mark.lean
def test_lean_proves_the_scans_postcondition_and_what_it_rests_on(lean: Any) -> None:
    for kind in (KIND, "postcondition"):
        (fact,) = facts(kernels.scan, kind)
        proved = lean.establish(fact)
        assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
        assert proved.decided_by == "lean"


@pytest.mark.lean
def test_lean_proves_a_statement_without_loops_stores_its_cell_once(lean: Any) -> None:
    for kind in (KIND, "postcondition"):
        (fact,) = facts(kernels.one_cell, kind)
        proved = lean.establish(fact)
        assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")


@pytest.mark.lean
def test_lean_does_not_prove_a_read_by_the_storing_instance_is_after_its_store(
    lean: Any,
) -> None:
    """Stated for Lean as isl was asked: the store has to run before the read.

    ``in_place``'s instance reads ``x[i + 1 - 1]``, its own cell, before it
    stores it; isl finds that and states no fact. Asked of Lean regardless,
    as a printer that read "not after" for "before" would ask it, the part
    for that read is false at the instance itself, and is not proved.
    """
    from functools import partial

    import islpy as isl
    from lanky.ledger import Fact

    from loopty import flow
    from loopty.obligations import Stated, single_assignment_obligation
    from loopty.oracle import Empty

    term = kernels.in_place.term
    writers = tuple(stmt for stmt in term.stmts if stmt.assignee.array == "x")
    places = tuple(
        (stmt, tuple(indices), tuple(inames), domain)
        for stmt in term.stmts
        for name, indices, kind, inames, domain in flow.statement_accesses(stmt, term)
        if name == "x" and kind == "read"
    )
    question = Empty(
        isl.UnionSet("{ }"),
        description="a single-assignment fact isl did not decide",
        lean=Stated(partial(single_assignment_obligation, term, "x", writers, places)),
    )
    fact = Fact(
        id=f"{KIND}:in_place:x",
        kind=KIND,
        statement="no cell of x is stored twice, or read before its store",
        term=question,
        status=Status.DECIDED,
        decided_by="isl",
    )
    result = lean.establish(fact)
    assert result.status is not Status.PROVED
    assert result.decided_by != "lean"


@pytest.mark.lean
def test_lean_checks_that_the_cited_theorem_applies(lean: Any) -> None:
    (applied,) = facts(kernels.scan_cited, APPLIED)
    proved = lean.establish(applied)
    assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
    assert "apply h" in proved.provenance["lean_source"]
    assert kernels.offsets_increase.fact_id in proved.rests_on


@pytest.mark.lean
def test_without_mathlib_a_claim_about_reals_is_left_to_isl_saying_why(
    lean: Any,
) -> None:
    if lean._mathlib():
        pytest.skip("Mathlib is imported here")
    (post,) = facts(kernels.triangle, "postcondition")
    result = lean.establish(post)
    assert result.status is Status.DECIDED
    assert "real numbers" in result.provenance["lean_declined"]


@pytest.mark.lean
def test_with_mathlib_the_reconstruction_and_the_theorem_it_cites_are_proved(
    lean: Any,
) -> None:
    """Over reals: the postcondition by ``linarith``, the application by ``apply``."""
    if not lean._mathlib():
        pytest.skip("Mathlib is not imported here (LANKY_LEAN_MATHLIB)")
    for kind in (KIND, "postcondition"):
        (fact,) = facts(kernels.triangle, kind)
        proved = lean.establish(fact)
        assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
    (applied,) = facts(kernels.reconstruct, APPLIED)
    proved = lean.establish(applied)
    assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
    assert "apply h" in proved.provenance["lean_source"]
    assert kernels.on_the_triangle.fact_id in proved.rests_on


@pytest.mark.lean
def test_with_mathlib_a_theorem_taken_at_a_real_scalar_is_proved(lean: Any) -> None:
    """``constant_table`` applied at ``k = s``, ``s`` a real of Lean's (#247)."""
    if not lean._mathlib():
        pytest.skip("Mathlib is not imported here (LANKY_LEAN_MATHLIB)")
    (applied,) = facts(kernels.fill_real, APPLIED)
    proved = lean.establish(applied)
    assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
    assert "(s : ℝ)" in proved.provenance["lean_source"]
    assert kernels.constant_table.fact_id in proved.rests_on


@pytest.mark.lean
def test_with_mathlib_a_family_over_the_kernels_size_is_proved(
    lean: Any, sized: Sized
) -> None:
    """``sized`` applied at ``n = p``, with ``D`` over ``Fin(p)`` (#239)."""
    if not lean._mathlib():
        pytest.skip("Mathlib is not imported here (LANKY_LEAN_MATHLIB)")
    for k in (sized.copy_p, sized.copy_pn):
        (applied,) = facts(k, APPLIED)
        proved = lean.establish(applied)
        assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
        assert "apply h" in proved.provenance["lean_source"]
        assert sized.sized.fact_id in proved.rests_on


@pytest.mark.lean
def test_a_wrong_application_is_no_proof(lean: Any) -> None:
    """The claim turned round is no consequence of the theorem, and is not proved."""
    import dataclasses

    from loopty.dataflow import Applied

    (applied,) = facts(kernels.scan_cited, APPLIED)
    # the claim with its order turned round, which the theorem does not give
    flipped = applied.term.application.claim
    from lanky.terms import Comparison, Forall

    body = Comparison(flipped.body.right, "<=", flipped.body.left)
    wrong = dataclasses.replace(
        applied.term.application, claim=Forall(flipped.binders, body, flipped.guard)
    )
    term = Applied(applied.term.term, wrong, applied.term.given, applied.term.owner)
    result = lean.establish(dataclasses.replace(applied, term=term))
    assert result.status is Status.DECIDED


# }}}


def test_the_example_reads_top_to_bottom_and_refuses_the_interchange() -> None:
    example = load(EXAMPLES / "taylor_reconstruction.py")
    with pytest.raises(IllegalCast, match="interchange"):
        example.rejected_interchange()
    (applied,) = facts(example.reconstruct, APPLIED)
    assert applied.statement == (
        "by on_the_triangle, for every D with pde and stored: forall a in "
        "Fin(p + 1), b in Fin(p + 1) where a + b <= p. R[a, b] == D(a)(b)"
    )
