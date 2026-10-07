"""Facts that travel: a program's calls assume what earlier calls established.

#65 refused a program in which an earlier call writes an array whose cells a
later call's contract checks. Those checks are the later call's requirements,
and they are now decided under the hypotheses that hold at the call, or
checked between the calls when nothing decides them (``loopty.compose``,
``loopty.hypotheses``). The programs are in ``kernels/travel.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import islpy as isl
import numpy as np
import pytest
from lanky import theorem
from lanky.ledger import Ledger, Status
from lanky.prelude import Fin, Fn, Nat, Real
from lanky.terms import (
    Add,
    Comparison,
    Forall,
    LogicalAnd,
    LogicalNot,
    Product,
    Subscript,
    Var,
    render,
)

from loopty import Arr, Schedule, kernel, program, reduce_sum
from loopty.hypotheses import discharge, substitute, theorem_instances
from loopty.interpret import CheckFailed, interpret
from loopty.oracle import IslOracle
from loopty.term import Hypothesis

pytest.importorskip("loopy")

from loopty.executor import LoopyExecutor, emit_code  # noqa: E402

KERNELS = Path(__file__).parent / "kernels"

travel = importlib.import_module("kernels.travel")
wrong = importlib.import_module("kernels.travel_wrong")


def _ledger():
    from lanky.check import check_path

    return check_path(KERNELS / "travel.py")


_LEDGER: list[Ledger] = []


def ledger() -> Ledger:
    """The ledger of ``kernels/travel.py``, checked once."""
    if not _LEDGER:
        _LEDGER.append(_ledger())
    return _LEDGER[0]


def facts_of(owner: str, kind: str) -> list:
    return [fact for fact in ledger() if fact.owner == owner and fact.kind == kind]


def decided(fact):
    """``fact`` as the isl oracle leaves it."""
    oracle = IslOracle()
    return oracle.establish(fact) if oracle.can_establish(fact) else fact


# {{{ the engine


off, cnt, perm = Var("off"), Var("cnt"), Var("perm")
n, q, r, a0 = Var("n"), Var("q"), Var("r"), Var("a0")


def _scan_post():
    return LogicalAnd(
        (
            Comparison(Subscript(off, 0), "==", 0),
            Forall(
                ((r, Fin[n]),),
                Comparison(
                    Subscript(off, r + 1),
                    "==",
                    Subscript(off, r) + Subscript(cnt, r),
                ),
            ),
        )
    )


def _layout_goal():
    """``off`` holds the offsets the counts give, at ``q`` in ``0..n``."""
    return LogicalAnd(
        (
            Comparison(q, ">", 0) | Comparison(Subscript(off, 0), "==", 0),
            Comparison(q, "==", 0)
            | Comparison(
                Subscript(off, q),
                "==",
                Subscript(off, q - 1) + Subscript(cnt, q - 1),
            ),
        )
    )


LAYOUT = isl.Set("[n] -> { [q] : 0 <= q <= n }")


def test_a_universal_hypothesis_is_used_at_the_cells_the_claim_reads() -> None:
    # off[q] == off[q - 1] + cnt[q - 1] is the scan's recurrence at r = q - 1,
    # which the cell off[q] of the claim and off[r + 1] of the recurrence give.
    post = Hypothesis(_scan_post(), "scan", ("postcondition:x",), frozenset({"off"}))
    found = discharge(
        LAYOUT, _layout_goal(), [post], integral={"off", "cnt"}, known={"n"},
        nonneg={"n"},
    )
    assert found.decided
    assert found.used == (post,)
    assert found.question.obj.is_empty()
    # Without it, the cells are anything, and the room it leaves is named.
    alone = discharge(
        LAYOUT, _layout_goal(), [], integral={"off", "cnt"}, known={"n"},
        nonneg={"n"},
    )
    assert not alone.decided
    assert alone.question is None
    assert "off[" in alone.reason


def test_a_hypothesis_a_fact_does_not_need_is_not_what_it_rests_on() -> None:
    post = Hypothesis(_scan_post(), "scan", ("postcondition:x",))
    idle = Hypothesis(
        Forall(((a0, Fin[n]),), Comparison(Subscript(cnt, a0), ">=", 0)),
        "the type of cnt",
        ("idle",),
    )
    found = discharge(
        LAYOUT, _layout_goal(), [idle, post], integral={"off", "cnt"},
        known={"n"}, nonneg={"n"},
    )
    assert found.decided
    assert found.used == (post,)


def test_what_isl_cannot_state_assumes_nothing() -> None:
    # A product of two cells is no affine constraint. As a hypothesis it is
    # read as true: had it been read as false, this one, which no array
    # satisfies, would have made every claim vacuously decided.
    impossible = Hypothesis(
        Forall(
            ((a0, Fin[n]),),
            LogicalNot(
                Comparison(
                    Product((Subscript(perm, a0), Subscript(perm, a0))), ">=", 0
                )
            ),
        ),
        "a claim isl cannot state",
    )
    domain = isl.Set("[n] -> { [a0] : 0 <= a0 < n }")
    goal = Comparison(Subscript(perm, a0), "<", n)
    found = discharge(
        domain, goal, [impossible], integral={"perm"}, known={"n"}, nonneg={"n"}
    )
    assert not found.decided


def test_two_cells_written_alike_to_some_depth_are_two_parameters() -> None:
    # pymbolic's repr abbreviates past a depth, so off[q + 1] and off[r + 1]
    # used to be one key and one parameter, asserting the two cells equal.
    domain = isl.Set("[n] -> { [q, r] : 0 <= q < n and 0 <= r < n }")
    known = Hypothesis(
        Comparison(Subscript(off, Add((q, 1))), "==", 5), "a cell is five"
    )
    goal = Comparison(Subscript(off, Add((r, 1))), "==", 5)
    found = discharge(domain, goal, [known], integral={"off"}, known={"n"})
    assert not found.decided


def test_an_instance_holds_only_where_its_binder_is_a_point() -> None:
    # The recurrence says nothing about off[0], so the claim off[0] == 1 is
    # not decided by it, though it would be at r = -1.
    post = Hypothesis(
        Forall(
            ((r, Fin[n]),),
            Comparison(Subscript(off, r + 1), "==", Subscript(off, r) + 1),
        ),
        "steps",
    )
    domain = isl.Set("[n] -> { [q] : q = 0 }")
    goal = Comparison(Subscript(off, q), "==", Subscript(off, q - 1) + 1)
    found = discharge(domain, goal, [post], integral={"off"}, known={"n"})
    assert not found.decided


def test_substitution_renames_a_binder_it_would_capture() -> None:
    claim = Forall(((q, Fin[n]),), Comparison(Subscript(off, q), "<=", r))
    out = substitute(claim, {"r": q + 1})
    (binder, _sort), = out.binders
    assert binder.name != "q"
    assert render(out) == f"forall {binder.name} in Fin(n). off[{binder.name}] <= q + 1"


def test_a_theorem_is_instantiated_at_the_arrays_its_hypotheses_match() -> None:
    post = Hypothesis(_scan_post(), "scan", ("postcondition:x",), frozenset({"off"}))
    seen = []

    def applicable(array, domain, codomain):
        seen.append((array, str(domain), str(codomain)))
        return None

    found, reasons = theorem_instances(
        travel.scan_monotone, [post], applicable, lambda expr: True
    )
    assert reasons == []
    (instance,) = found
    assert render(instance.claim) == (
        "forall a in Fin(n + 1), b in Fin(n + 1) where a <= b. off[a] <= off[b]"
    )
    assert instance.rests_on == (travel.scan_monotone.fact_id, "postcondition:x")
    assert ("off", "Fin(n + 1)", "Nat") in seen and ("cnt", "Fin(n)", "Nat") in seen

    # A binding the families' sorts refuse is no instance, and says why.
    found, reasons = theorem_instances(
        travel.scan_monotone, [post], lambda *_: "off is written", lambda e: True
    )
    assert found == [] and reasons == ["off is written"]


def test_a_theorems_family_is_checked_against_its_sorts_in_the_programs_names() -> None:
    # bounded's f ranges over Fin[m], and its hypothesis binds m to the k of
    # the program's perm: the family's codomain is Fin(k) there, not Fin(m),
    # which would compare equal with an element sort the program calls m.
    @theorem
    def bounded(
        n: Nat,  # noqa: F821
        m: Nat,  # noqa: F821
        f: Fn[Fin[n], Fin[m]],  # noqa: F821
        h: all((f(i) >= 0) & (f(i) < m) for i in Fin[n]),  # noqa: F821
    ) -> all(f(i) <= m for i in Fin[n]):  # noqa: F821
        """A point of Fin[m] is at most m."""

    entry = Hypothesis(
        Forall(
            ((a0, Fin[n]),),
            LogicalAnd(
                (
                    Comparison(0, "<=", Subscript(perm, a0)),
                    Comparison(Subscript(perm, a0), "<", Var("k")),
                )
            ),
        ),
        "the type of perm",
    )
    seen = []

    def applicable(array, domain, codomain):
        seen.append((array, str(domain), str(codomain)))
        return None

    found, reasons = theorem_instances(bounded, [entry], applicable, lambda e: True)
    assert reasons == []
    assert seen == [("perm", "Fin(n)", "Fin(k)")]
    (instance,) = found
    assert render(instance.claim) == "forall i in Fin(n). perm[i] <= k"


# }}}


# {{{ a permutation, computed and then used


def test_a_kernels_postcondition_is_tested_against_its_native_runs() -> None:
    (post,) = facts_of("number", "postcondition")
    assert post.status is Status.TESTED
    assert post.decided_by == "native"
    assert post.provenance["compared"] >= 1
    # Its term is the claim about a call, which no oracle takes for closed.
    assert str(post.term) == "forall i in Fin(n). perm[i] == n - 1 - i"


def test_a_false_postcondition_is_refuted_by_a_native_run() -> None:
    @kernel
    def counts_up(perm: Arr[Fin[n], Fin[n]]) -> all(perm[i] == i for i in Fin[n]):  # noqa: F821
        for i in perm.dom:
            perm[i] = perm.dom.size - 1 - i

    (post,) = [fact for fact in counts_up.facts() if fact.kind == "postcondition"]
    assert post.status is Status.REFUTED
    assert "leaves forall i in Fin(n). perm[i] == i false" in post.provenance["reason"]


def test_gathers_requirement_is_decided_under_numbers_postcondition() -> None:
    # #65's permuted program: number writes perm, gather reads x[perm[i]].
    (requirement,) = facts_of("permuted", "requirement")
    assert requirement.statement.startswith(
        "the elements of perm are points of Fin(n) where gather is called"
    )
    (restated,) = facts_of("permuted", "postcondition-in-scope")
    # The postcondition is tested on number's body, and the program runs its
    # term: the requirement rests on the two computing alike, too.
    (faithful,) = facts_of("number", "trace-faithful")
    assert requirement.rests_on == (restated.id, faithful.id)
    (used,) = requirement.provenance["used"]
    assert used.startswith("the postcondition of number, after number at travel.py:")
    assert requirement.status is Status.DECIDED
    assert requirement.decided_by == "isl"
    support = ledger().support(requirement)
    assert support.effective is Status.TESTED
    assert support.under == ()
    # Nothing is checked: the compiled program is the two loops.
    assert travel.permuted.term.checks == ()


def test_the_permuted_program_runs_compiled_as_it_does_natively() -> None:
    inputs = travel.example_inputs()["permuted"]
    fact = LoopyExecutor().differential(
        travel.permuted, Schedule(travel.permuted), inputs
    )
    assert fact.status is Status.TESTED, fact.provenance
    (faithful,) = facts_of("permuted", "trace-faithful")
    assert faithful.status is Status.TESTED


def test_an_undecided_requirement_is_checked_where_the_native_call_refuses() -> None:
    term = travel.permuted_up.term
    (requirement,) = term.requirements
    assert not requirement.decided
    assert "nothing that held at the call says" in requirement.reason
    ((flag, message),) = term.checks
    code = emit_code(travel.permuted_up)
    assert f"{flag}[0] = 1" in code and f"if ({flag}[0] == 0)" in code

    # The ledger keeps it assumed, and says why.
    (fact,) = facts_of("permuted_up", "requirement")
    assert fact.status is Status.ASSUMED
    assert fact.statement.endswith("(checked when it runs)")
    assert fact.provenance["checked"].endswith(message)

    # Natively gather's contract refuses perm; compiled, the check does.
    def inputs():
        return {
            "perm": Arr.zeros(4, dtype=np.int64),
            "x": Arr.from_numpy(np.arange(4.0)),
            "y": Arr.zeros(4),
        }

    with pytest.raises(ValueError, match=r"perm\[3\] is 4"):
        travel.permuted_up(**inputs())
    with pytest.raises(ValueError, match="stops before gather"):
        LoopyExecutor().run(travel.permuted_up, **inputs())
    # So does the term, interpreted.
    with pytest.raises(CheckFailed, match="stops before gather"):
        interpret(term, inputs())


# }}}


# {{{ offsets a scan computes


def test_the_layout_requirement_is_decided_under_scans_postcondition() -> None:
    (requirement,) = facts_of("through", "requirement")
    assert requirement.statement.startswith(
        "off holds the offsets the counts in cnt give the rows of val"
    )
    assert requirement.status is Status.DECIDED
    (restated,) = facts_of("through", "postcondition-in-scope")
    # The program cites scan_monotone, and the requirement does not need it:
    # it is the contract's equality, which the postcondition states.
    (faithful,) = facts_of("scan", "trace-faithful")
    assert requirement.rests_on == (restated.id, faithful.id)
    assert travel.through.uses == (travel.scan_monotone,)
    assert not any("scan_monotone" in h for h in requirement.provenance["used"])
    assert ledger().support(requirement).effective is Status.TESTED
    # And the layout fact of the program rests on it.
    (layout,) = facts_of("through", "layout")
    assert requirement.id in layout.rests_on
    assert "deferred" in layout.provenance


def test_offsets_written_before_they_are_read_are_not_checked_on_entry() -> None:
    # The scan computes off, so what the caller passes for it is overwritten
    # before anything reads a row through it; the requirement checks it there.
    assert travel.through.term.deferred_offsets == ("off",)
    inputs = travel.example_inputs()["through"]
    fact = LoopyExecutor().differential(
        travel.through, Schedule(travel.through), inputs
    )
    assert fact.status is Status.TESTED, fact.provenance


def test_a_write_after_the_scan_retires_its_postcondition() -> None:
    # bump writes off after scan, so scan's claim about off no longer holds
    # where rowsums reads through it, and the requirement is checked.
    term = travel.bumped.term
    (requirement,) = term.requirements
    assert not requirement.decided
    assert "the postcondition of scan" not in [h.source for h in requirement.offered]
    (layout,) = facts_of("bumped", "layout")
    assert layout.status is Status.ASSUMED

    def inputs():
        return travel.csr([2, 0, 3])

    with pytest.raises(ValueError, match="the offsets argument off holds"):
        travel.bumped(**inputs())
    with pytest.raises(ValueError, match="does not hold the offsets the counts"):
        LoopyExecutor().run(travel.bumped, **inputs())


# }}}


# {{{ a theorem the program cites


def test_a_requirement_only_the_cited_theorem_decides() -> None:
    # 0 = off[0] <= off[i] <= off[n] <= n needs the offsets monotone.
    (requirement,) = facts_of("picked", "requirement")
    assert requirement.status is Status.DECIDED
    assert travel.scan_monotone_int.fact_id in requirement.rests_on
    assert any(
        source.startswith("scan_monotone_int at")
        for source in requirement.provenance["used"]
    )
    support = ledger().support(requirement)
    assert support.effective is Status.TESTED
    # Without the theorem, the same program is checked between the calls.
    (alone,) = facts_of("picked_alone", "requirement")
    assert alone.status is Status.ASSUMED
    assert travel.picked_alone.term.checks
    fact = LoopyExecutor().differential(
        travel.picked, Schedule(travel.picked), travel.example_inputs()["picked"]
    )
    assert fact.status is Status.TESTED, fact.provenance
    # The theorem passes the property tester, so the compiled program skips
    # the check on its strength.
    assert travel.picked.term.checks == ()


def test_a_theorem_over_naturals_is_no_hypothesis_about_written_offsets() -> None:
    # scan_monotone ranges over Nat offsets, and nothing establishes that the
    # cells scan wrote are naturals: the binding is refused, with the reason.
    @program(uses=[travel.scan_monotone])
    def picked_nat(cnt, off, x, y):
        travel.scan_unit(cnt, off)
        travel.pick(off, x, y)

    (requirement,) = picked_nat.term.requirements
    assert not requirement.decided
    assert "nothing establishes that its cells are points of Nat" in (
        requirement.reason
    )


def test_uses_takes_theorems_and_refuses_an_id() -> None:
    with pytest.raises(TypeError, match="states nothing a program can instantiate"):

        @program(uses=["theorem:travel.scan_monotone@73"])
        def by_id(cnt, off):
            travel.scan(cnt, off)

    with pytest.raises(TypeError, match="uses=None names no theorem"):

        @program(uses=None)
        def by_none(cnt, off):
            travel.scan(cnt, off)

    assert travel.through.uses == (travel.scan_monotone,)


# }}}


# {{{ flat CSR after the scan


def test_a_flat_access_is_in_bounds_where_the_scan_was_called() -> None:
    # Alone, val[off[r] + j] is assumed: nothing in weigh_flat lays the rows
    # out inside val.
    (alone,) = [
        fact
        for fact in facts_of("weigh_flat", "in-bounds")
        if fact.statement.startswith("val[off[r] + j]")
    ]
    assert alone.status is Status.ASSUMED
    # After the scan, off[r] + j < off[r] + cnt[r] == off[r + 1] <= nnz.
    (scoped,) = facts_of("flat", "in-bounds")
    assert scoped.statement.startswith("val[off[r_0] + j] is in bounds")
    assert scoped.status is Status.DECIDED
    used = scoped.provenance["used"]
    assert any(source.startswith("the postcondition of scan_csr") for source in used)
    assert any(source.startswith("the requirement on off") for source in used)
    # The bound off[r + 1] <= nnz is off's element type, which the program
    # checks where weigh_flat is called: the fact is decided under that.
    (element,) = [
        fact
        for fact in facts_of("flat", "requirement")
        if fact.provenance["requirement"] == "element"
    ]
    assert element.status is Status.ASSUMED
    assert ledger().support(scoped).under == (element.id,)
    fact = LoopyExecutor().differential(
        travel.flat, Schedule(travel.flat), travel.example_inputs()["flat"]
    )
    assert fact.status is Status.TESTED, fact.provenance


def test_a_call_that_writes_the_offsets_assumes_nothing_of_them() -> None:
    # What held when the call started holds of off only until the call
    # writes it, so a flat read after the write is not decided by it.
    @kernel
    def shift_then_read(
        cnt: Arr[Fin[n], Nat],  # noqa: F821
        off: Arr[Fin[n + 1], Fin[nnz + 1]],  # noqa: F821
        wt: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
        val: Arr[Fin[nnz], Real],  # noqa: F821
        y: Arr[Fin[n], Real],  # noqa: F821
    ):
        for r in y.dom:
            y[r] = reduce_sum(wt[r, j] * val[off[r] + j] for j in wt.dom[r])
            off[r] = off[r + 1]

    @program
    def shifted(cnt, off, wt, val, y):
        travel.scan_csr(cnt, off)
        shift_then_read(cnt, off, wt, val, y)

    assert [fact for fact in shifted.facts() if fact.kind == "in-bounds"] == []
    (scope,) = [s for s in shifted.term.scopes if s.call == "shift_then_read"]
    assert all("off" not in h.mentions for h in scope.hypotheses)


# }}}


def test_the_bound_of_an_element_sort_is_unified_like_an_axis() -> None:
    # gather_m declares perm's elements over Fin[m], and m is x's length;
    # number's perm says Fin[n], so m is n, as an axis would be.
    @kernel
    def gather_m(
        perm: Arr[Fin[k], Fin[m]],  # noqa: F821
        x: Arr[Fin[m], Real],  # noqa: F821
        y: Arr[Fin[k], Real],  # noqa: F821
    ):
        for i in y.dom:
            y[i] = x[perm[i]]

    @program
    def unified(perm, x, y):
        travel.number(perm)
        gather_m(perm, x, y)

    types = dict(unified.term.params)
    assert str(types["x"].axes[0]) == "n"
    (requirement,) = unified.term.requirements
    assert requirement.decided


def test_a_failed_check_in_the_term_is_a_disagreement_where_the_body_runs() -> None:
    from loopty.faithful import _compare

    @kernel
    def keep(perm: Arr[Fin[n], Fin[n]]):  # noqa: F821
        for i in perm.dom:
            perm[i] = perm[i]

    @program
    def kept(perm, x, y):
        keep(perm)
        travel.gather(perm, x, y)

    # permuted_up's term against kept's body, by hand: the term's check
    # fails on what number_up writes, and the body, which keeps perm as it
    # is, runs. A compiled program that refuses what the native one takes is
    # a disagreement, not an input to skip.
    term = travel.permuted_up.term
    inputs = {
        "perm": Arr.from_numpy(np.array([0, 1, 2, 3])),
        "x": Arr.from_numpy(np.arange(4.0)),
        "y": Arr.zeros(4),
    }
    kind, (counterexample, reason) = _compare(kept, term, "hand", inputs)
    assert kind == "differ"
    assert "stops at a checked point the native run passes" in reason


# {{{ what does not follow is never decided


def test_hypotheses_that_contradict_each_other_decide_nothing() -> None:
    # off[q] == 0 and off[q] == 1 hold of no cell, so every claim follows
    # from the two: a decision under them would be vacuous.
    zero = Hypothesis(
        Forall(((r, Fin[n + 1]),), Comparison(Subscript(off, r), "==", 0)), "zero"
    )
    one = Hypothesis(
        Forall(((r, Fin[n + 1]),), Comparison(Subscript(off, r), "==", 1)), "one"
    )
    goal = Comparison(Subscript(off, q), "==", 7)
    found = discharge(
        LAYOUT, goal, [zero, one], integral={"off"}, known={"n"}, nonneg={"n"}
    )
    assert not found.decided
    assert found.question is None
    assert found.contradicting == (zero, one)
    assert "contradict each other" in found.reason
    # A claim about a domain with no points is decided, with nothing used.
    nowhere = isl.Set("[n] -> { [q] : 0 <= q < 0 }")
    found = discharge(
        nowhere, goal, [zero, one], integral={"off"}, known={"n"}, nonneg={"n"}
    )
    assert found.decided and found.used == ()


@pytest.mark.parametrize(
    "prog",
    [
        wrong.past_the_end_then_gather,
        wrong.the_other_then_gather,
        wrong.all_but_the_last_then_gather,
        wrong.all_but_the_first_then_gather,
        wrong.either_then_gather,
        wrong.doubled_then_gather,
        wrong.squared_then_gather,
        wrong.at_some_then_gather,
        wrong.both_then_gather,
    ],
    ids=lambda prog: prog.__name__,
)
def test_a_postcondition_short_of_the_requirement_leaves_it_checked(prog) -> None:
    (requirement,) = prog.term.requirements
    assert not requirement.decided
    assert prog.term.checks
    assert requirement.reason


def test_a_false_postcondition_is_named_rather_than_used() -> None:
    # perm[0] == 0 and perm[0] == 1: a contradiction would decide gather's
    # requirement vacuously, and the compiled program would read x[perm[i]]
    # unchecked. It is checked, and the reason names the postcondition.
    (requirement,) = wrong.both_then_gather.term.requirements
    assert not requirement.decided
    assert requirement.reason.startswith("no cells satisfy the postcondition of both")
    assert "decided vacuously" in requirement.reason


def test_what_isl_cannot_state_is_named_in_the_reason() -> None:
    (requirement,) = wrong.squared_then_gather.term.requirements
    assert "isl cannot state perm[" in requirement.reason
    assert "(of the postcondition of squared, after squared at" in requirement.reason
    (requirement,) = wrong.at_some_then_gather.term.requirements
    assert "isl cannot state exists i in Fin(n). perm[i] == 0" in requirement.reason


def test_a_scan_with_the_wrong_step_is_checked_three_ways() -> None:
    (requirement,) = wrong.gapped.term.requirements
    assert requirement.kind == "layout" and not requirement.decided
    # The room the hypotheses leave names the cells the requirement reads,
    # not every cell an instance of the recurrence reached.
    assert "off[q]" in requirement.reason
    assert "off[q + 2]" not in requirement.reason

    def inputs():
        return travel.csr([2, 0, 3])

    with pytest.raises(ValueError, match="the offsets argument off holds"):
        wrong.gapped(**inputs())
    with pytest.raises(ValueError, match="does not hold the offsets the counts"):
        LoopyExecutor().run(wrong.gapped, **inputs())
    with pytest.raises(CheckFailed, match="does not hold the offsets the counts"):
        interpret(wrong.gapped.term, inputs())


def test_a_flat_access_is_not_decided_under_a_contradicted_layout() -> None:
    # weigh reads wt through off, so its layout requirement says off[r + 1]
    # == off[r] + cnt[r], and the gapped scan says off[r + 1] == off[r] +
    # cnt[r] + 1: together they hold nowhere, and would decide val[off[r] +
    # j] in bounds vacuously.
    assert [f for f in wrong.gapped_flat.facts() if f.kind == "in-bounds"] == []


def test_a_write_to_an_array_a_binders_sort_reads_retires_the_claim() -> None:
    # clear_some says perm[i] == 0 for i < lim[0], with lim[0] == 1, and
    # every_cell then makes lim[0] == n. Read after that write, the claim
    # would say every cell of perm is 0, where clear_some put n in all but
    # the first: gather's requirement was decided, and the compiled program
    # read x[n]. The write to lim retires the claim, and the requirement is
    # checked.
    (requirement,) = [
        r for r in wrong.cleared.term.requirements if r.call == "gather"
    ]
    assert not requirement.decided
    assert all(
        not h.source.startswith("the postcondition of clear_some")
        for h in requirement.offered
    )

    def inputs():
        return {
            "perm": Arr.zeros(4, dtype=np.int64),
            "lim": Arr.zeros(1, dtype=np.int64),
            "x": Arr.from_numpy(np.arange(4.0)),
            "y": Arr.zeros(4),
        }

    with pytest.raises(ValueError, match=r"perm\[1\] is 4"):
        wrong.cleared(**inputs())
    with pytest.raises(ValueError, match="stops before gather"):
        LoopyExecutor().run(wrong.cleared, **inputs())


def test_a_name_a_postcondition_has_no_value_for_means_nothing_in_a_program() -> None:
    # capped says perm[i] < k with k neither a parameter nor a size of its
    # own, so natively k has no value and the claim is never evaluated. In
    # the program, k is the length of x, which touch names so; read as that,
    # the claim decided gather's requirement, and the compiled program read
    # past x where the native one was refused.
    (requirement,) = wrong.capped_then_gather.term.requirements
    assert not requirement.decided

    def inputs():
        return {
            "perm": Arr.zeros(4, dtype=np.int64),
            "x": Arr.from_numpy(np.arange(4.0)),
            "y": Arr.zeros(4),
        }

    with pytest.raises(ValueError, match=r"perm\[0\] is 4"):
        wrong.capped_then_gather(**inputs())
    with pytest.raises(ValueError, match="stops before gather"):
        LoopyExecutor().run(wrong.capped_then_gather, **inputs())


def test_a_hypothesis_never_speaks_of_the_claims_own_binders() -> None:
    # scan_at_q's q is free, and the layout requirement is a claim over a
    # binder the composition also calls q: read as that binder, one row's
    # equation would have been every row's.
    (requirement,) = wrong.scanned_at_q.term.requirements
    assert requirement.kind == "layout" and not requirement.decided
    q_, off_, cnt_ = Var("q"), Var("off"), Var("cnt")
    one_row = Hypothesis(
        LogicalAnd(
            (
                Comparison(Subscript(off_, 0), "==", 0),
                Comparison(
                    Subscript(off_, q_),
                    "==",
                    Subscript(off_, q_ - 1) + Subscript(cnt_, q_ - 1),
                ),
            )
        ),
        "one row nobody named",
    )
    found = discharge(
        LAYOUT, _layout_goal(), [one_row], integral={"off", "cnt"}, known={"n"},
        nonneg={"n"},
    )
    assert not found.decided


def test_a_size_only_an_element_sort_names_is_renamed_apart() -> None:
    # Both calls of scan_flat name their buffer nnz, which no axis of
    # scan_flat is as long as. Taken as one name, the two buffers would be
    # one size, and the compiled program would refuse val2 of another length.
    types = dict(wrong.two_buffers.term.params)
    first, second = types["val"].axes[0], types["val2"].axes[0]
    assert render(first) != render(second)
    assert render(types["off"].dtype.bound) == f"{render(first)} + 1"
    assert render(types["off2"].dtype.bound) == f"{render(second)} + 1"

    def matrix(counts, seed):
        rng = np.random.default_rng(seed)
        data = travel.csr(counts, seed)
        data["wt"] = data.pop("val")
        data["val"] = Arr.from_numpy(rng.normal(size=sum(counts)))
        return data

    inputs = {
        **matrix([2, 0, 3], 0),
        **{f"{name}2": value for name, value in matrix([1, 1], 1).items()},
    }
    fact = LoopyExecutor().differential(
        wrong.two_buffers, Schedule(wrong.two_buffers), inputs
    )
    assert fact.status is Status.TESTED, fact.provenance


# }}}


# {{{ what no run bears out is checked all the same


def _permutation_inputs():
    return {
        "perm": Arr.zeros(4, dtype=np.int64),
        "x": Arr.from_numpy(np.arange(4.0)),
        "y": Arr.zeros(4),
    }


def test_a_requirement_decided_under_a_refuted_postcondition_is_checked() -> None:
    # #115: liar says perm[i] == n - 1 - i and writes i + 1. Its runs refute
    # that, and gather's requirement, decided under it, skipped its check:
    # the compiled program read x[4], where the native gather is refused.
    (post,) = [fact for fact in wrong.liar.facts() if fact.kind == "postcondition"]
    assert post.status is Status.REFUTED
    term = wrong.lied_to.term
    (requirement,) = term.requirements
    assert requirement.decided
    assert requirement.flag is not None
    assert "checked all the same, since the postcondition of liar is refuted" in (
        requirement.reason
    )
    assert term.checks == ((requirement.flag, requirement.message),)

    with pytest.raises(ValueError, match=r"perm\[3\] is 4"):
        wrong.lied_to(**_permutation_inputs())
    with pytest.raises(ValueError, match="stops before gather") as compiled:
        LoopyExecutor().run(wrong.lied_to, **_permutation_inputs())
    assert "the postcondition of liar is refuted" in str(compiled.value)
    with pytest.raises(CheckFailed, match="stops before gather"):
        interpret(term, _permutation_inputs())

    # The ledger keeps what isl decided, worth what it rests on, and says
    # that the compiled program checks it.
    (fact,) = [f for f in wrong.lied_to.facts() if f.kind == "requirement"]
    assert fact.statement.endswith("(checked when it runs)")
    assert "stops before gather" in fact.provenance["checked"]
    assert decided(fact).status is Status.DECIDED
    (restated,) = [
        f for f in wrong.lied_to.facts() if f.kind == "postcondition-in-scope"
    ]
    assert restated.id in fact.rests_on


def test_a_postcondition_of_a_body_its_term_does_not_compute_is_checked() -> None:
    # two_faced's postcondition holds of its body, and its term, which the
    # compiled program runs, counts up to n: gather's requirement, decided
    # under the postcondition, skipped its check, and the compiled program
    # read x[n] where the native one ran.
    facts = {fact.kind: fact for fact in wrong.two_faced.facts()}
    assert facts["postcondition"].status is Status.TESTED
    assert facts["trace-faithful"].status is Status.REFUTED
    term = wrong.faced.term
    (requirement,) = term.requirements
    assert requirement.decided and requirement.flag is not None
    assert (
        "that the term of two_faced computes what its body computes, on which "
        "its postcondition was tested, is refuted"
    ) in requirement.reason
    # The fact rests on the callee's trace-faithful fact, as well as on the
    # restatement of its postcondition.
    (fact,) = [f for f in wrong.faced.facts() if f.kind == "requirement"]
    assert facts["trace-faithful"].id in fact.rests_on

    wrong.faced(**_permutation_inputs())
    with pytest.raises(ValueError, match="stops before gather"):
        LoopyExecutor().run(wrong.faced, **_permutation_inputs())


def test_a_theorems_binder_does_not_capture_the_size_it_is_instantiated_at() -> None:
    # bounded says f(a) <= n for its own binder a. Instantiated at n = a, a
    # size of the program spelled like that binder, it said perm[a] <= a of
    # the binder, which is false of up_to_a's cells, and decided gather's
    # requirement under a true theorem and a true postcondition: the
    # compiled program read x[4].
    (requirement,) = wrong.captured.term.requirements
    (instance,) = [
        h for h in requirement.offered if h.source.startswith("bounded at")
    ]
    said = render(instance.claim)
    assert said.startswith("forall a_0 in Fin(a). perm[a_0] <= a and")
    assert not requirement.decided
    assert requirement.flag is not None

    with pytest.raises(ValueError, match=r"perm\[3\] is 4"):
        wrong.captured(**_permutation_inputs())
    with pytest.raises(ValueError, match="stops before gather"):
        LoopyExecutor().run(wrong.captured, **_permutation_inputs())


def test_a_postcondition_is_no_hypothesis_where_its_contract_went_unchecked() -> None:
    # clamp's postcondition holds of every run its contract lets in, where
    # src holds naturals. below_zero leaves -4 in src, which clamp's contract
    # refuses natively and nothing checks compiled; the postcondition then
    # decided gather's requirement, and the compiled program read x[-4].
    (post,) = [fact for fact in wrong.clamp.facts() if fact.kind == "postcondition"]
    assert post.status is Status.TESTED
    (requirement,) = wrong.clamped.term.requirements
    assert not requirement.decided
    assert not any(
        h.source.startswith("the postcondition of clamp") for h in requirement.offered
    )
    assert "the postcondition of clamp after clamp at" in requirement.reason
    assert "elements of src are naturals, and below_zero at" in requirement.reason

    inputs = {**_permutation_inputs(), "src": Arr.zeros(4, dtype=np.int64)}
    with pytest.raises(ValueError, match=r"src\[0\] is -4"):
        wrong.clamped(**{name: value.copy() for name, value in inputs.items()})
    with pytest.raises(ValueError, match="stops before gather"):
        LoopyExecutor().run(
            wrong.clamped, **{name: value.copy() for name, value in inputs.items()}
        )


def test_a_requirement_decided_under_an_axiom_is_checked_all_the_same() -> None:
    term = wrong.picked_on_a_citation.term
    (requirement,) = term.requirements
    assert requirement.decided
    assert any(
        h.source.startswith("scan_monotone_cited at") for h in requirement.used
    )
    assert "the axiom scan_monotone_cited is assumed on its citation" in (
        requirement.reason
    )
    assert term.checks
    inputs = {
        "cnt": Arr.from_numpy(np.array([1, 0, 1], dtype=np.int64)),
        "off": Arr.zeros(4, dtype=np.int64),
        "x": Arr.from_numpy(np.arange(4.0)),
        "y": Arr.zeros(4),
    }
    fact = LoopyExecutor().differential(
        wrong.picked_on_a_citation, Schedule(wrong.picked_on_a_citation), inputs
    )
    assert fact.status is Status.TESTED, fact.provenance


# }}}
