"""Three probes written to break the core, kept because they did not.

Each one aims at a different half of the claim loopty makes. The first at the
schedule checker: a tag that removes the order from a loop that carries a
dependence has to be refused, and refused with the pair of instances that would
race, not with a shrug. The second at the typing rules: a strided access is the
smallest thing isl has to be able to decide *and* to refute, and getting the
second half wrong is invisible, because an unrefuted obligation looks exactly
like a discharged one until the program runs off the end of an array. The third
at the ragged path end to end: a row with no entries is the case every CSR
implementation gets wrong once, and it has to come out the same from the Python
body, the term, the lowering, and the compiled code.
"""

from __future__ import annotations

import numpy as np
import pytest
from lanky.ledger import Status
from lanky.prelude import Nat, Real

import hand_terms as ht
from loopty import Arr, Fin, facts_for, kernel
from loopty import sum as reduce_sum
from loopty.executor import LoopyExecutor
from loopty.lower import lower_generic
from loopty.oracle import IslOracle
from loopty.schedule import IllegalCast, Schedule


def _run(obj, **arguments):
    """Run on the C target, or skip if the local toolchain cannot be used."""
    try:
        return LoopyExecutor().run(obj, **arguments)
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        if "compil" in str(exc).lower() or isinstance(exc, OSError):
            pytest.skip(f"the C toolchain path is unusable here: {exc}")
        raise


# {{{ probe one: a parallel tag on a loop that carries a dependence


def test_tagging_a_dependence_carrying_iname_parallel_is_refuted_with_a_witness():
    """``g.0`` on the time loop of a Jacobi stencil is a race, and is refused.

    A parallel tag is not a hint: it says the instances of that loop may run in
    any order, or at once. The time loop of ``u[t+1, i] = (u[t, i-1] +
    u[t, i+1]) / 2`` carries a flow dependence from every step to the next, so
    the claim is false, and the checker has to produce the counterexample rather
    than merely disagree.
    """
    schedule = Schedule(ht.jacobi_term(), sizes={"nt": 8, "nx": 8})
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(t="g.0")
    error = caught.value

    fact = error.fact
    assert fact.kind == "monotone"
    assert fact.status is Status.REFUTED
    assert fact.decided_by == "isl"
    assert fact.owner == "jacobi"

    # The witness is two statement instances and the sizes they were read at,
    # not a boolean. The source writes the cell the sink reads, and the tagged
    # order puts them in no order at all.
    (source_id, source_coords), (sink_id, sink_coords), params = error.witness
    assert source_id == sink_id == "S0"
    assert set(source_coords) == set(sink_coords) == {"t", "i"}
    assert source_coords["t"] + 1 == sink_coords["t"]
    assert params == {"nt": 8, "nx": 8}

    message = str(error)
    assert message.startswith("tag(t='g.0') illegal: instance S0[")
    assert "writes u[" in message and "read by S0[" in message
    assert message.endswith("scheduled earlier (at nt=8, nx=8, as hinted)")

    # The refusal is a refusal: nothing was committed to the schedule it was
    # asked of, which is what lets a caller try a different transformation.
    assert schedule.tags == {}
    assert schedule.facts() == ()


# }}}


# {{{ probe two: strided accesses into an axis of twice the size


@kernel
def pairs(u: Arr[Fin[2 * n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
    """``v[i] = u[2i] + u[2i+1]``: both halves of a pair, both in bounds."""
    for i in v.dom:
        v[i] = u[2 * i] + u[2 * i + 1]


@kernel
def pairs_off_by_one(u: Arr[Fin[2 * n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
    """``v[i] = u[2i+2]``: in bounds everywhere except the last row."""
    for i in v.dom:
        v[i] = u[2 * i + 2]


def _in_bounds(kernel_object, array: str) -> list:
    """The in-bounds facts about one array, as the isl oracle leaves them.

    The typing rules state the obligation and leave it ``ASSUMED``; deciding it
    is the oracle's job, exactly as ``lanky check`` arranges it. Asking the
    oracle here is the point of the probe: a rule that states an obligation
    nobody can decide would pass a test that only looked at the rules.
    """
    term = kernel_object.trace()
    oracle = IslOracle()
    out = []
    for fact in facts_for(term, owner=term.name):
        if fact.kind != "in-bounds" or f"{array}[" not in fact.statement:
            continue
        out.append(oracle.establish(fact) if oracle.can_establish(fact) else fact)
    return out


def test_a_strided_access_into_an_axis_of_twice_the_size_is_decided() -> None:
    """``u[2i]`` and ``u[2i+1]`` for ``i < n`` into ``Fin[2n]``: isl decides both.

    This is the smallest access that is neither the identity nor a constant
    offset, and the one that needs the axis size to be related to the loop
    bound rather than merely compared with it. Both are in bounds, which is why
    the refutation below has to use ``2i + 2``: ``2i + 1`` is the last cell of
    the array at ``i = n - 1``, not one past it, and a checker that refuted it
    would be wrong.
    """
    facts = _in_bounds(pairs, "u")
    assert len(facts) == 2, [fact.statement for fact in facts]
    assert {fact.status for fact in facts} == {Status.DECIDED}
    assert {fact.decided_by for fact in facts} == {"isl"}
    assert {fact.statement.split(" is ")[0] for fact in facts} == {
        "u[2*i]",
        "u[2*i + 1]",
    }


def test_a_strided_access_one_pair_too_far_is_refuted_with_a_witness() -> None:
    """``u[2i+2]`` leaves ``Fin[2n]`` at ``i = n - 1``, and isl says where."""
    facts = _in_bounds(pairs_off_by_one, "u")
    assert len(facts) == 1, [fact.statement for fact in facts]
    fact = facts[0]
    assert fact.status is Status.REFUTED
    assert fact.decided_by == "isl"
    # The witness is the cell that is not there, at sizes that make it real: at
    # ``n`` rows the array has ``2n`` cells numbered up to ``2n - 1``, and the
    # last iteration asks for ``2n``. isl chooses the smallest such ``n``.
    params = fact.provenance["witness_params"]
    (cell,) = fact.provenance["witness"]
    assert cell == 2 * params["n"], (cell, params)
    assert fact.provenance["witness_text"] == f"[a0={cell}] at [n={params['n']}]"


def test_the_two_strided_kernels_disagree_only_about_the_last_row() -> None:
    """Both compile; only one of them is the program the type allows.

    Running them is what makes the point that the refutation is about the
    program and not about loopy: the off-by-one kernel builds, runs, and reads
    one cell past the array, which is exactly the class of bug the in-bounds
    obligation exists to catch before it happens.
    """
    u = np.arange(6, dtype=np.float64)
    v = np.zeros(3)
    out = _run(pairs.trace(), u=Arr.from_numpy(u), v=Arr.from_numpy(v))
    assert list(out["v"]) == [1.0, 5.0, 9.0]


# }}}


# {{{ probe three: a ragged kernel whose counts include zero rows


@kernel
def rowsums(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``y[r]`` is the sum of row ``r``, over however many entries it has."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


#: Leading, interior and trailing empty rows, and one row of every other size.
COUNTS = [0, 3, 0, 1, 0, 2, 0]
VALUES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
WANT = [0.0, 7.0, 0.0, 8.0, 0.0, 48.0, 0.0]


def _ragged_inputs() -> dict:
    return {
        "cnt": Arr.from_numpy(np.array(COUNTS, dtype=np.int64)),
        "val": Arr.ragged(COUNTS, values=VALUES),
        "y": Arr.zeros(len(COUNTS)),
    }


def test_an_empty_row_sums_to_zero_in_the_python_body() -> None:
    """The reference run: an empty fiber iterates no times and leaves zero."""
    arrays = _ragged_inputs()
    rowsums(**arrays)
    assert list(arrays["y"].numpy()) == WANT


def test_an_empty_row_leaves_the_term_and_the_lowering_intact() -> None:
    """The term says the bound is the row's count; nothing assumes it positive.

    A zero row is not a special case in the term at all, which is the point: the
    reduction's domain is ``0 <= j < cnt[r]`` with the count reflected as an isl
    parameter, and isl answers for every value of a parameter including zero. If
    anything in the path had quietly assumed a non-empty row, it would be here,
    in a domain constraint or in the offsets the lowering derives.
    """
    term = rowsums.trace()
    assert len(term.stmts) == 1
    (stmt,) = term.stmts
    assert stmt.inames == ("r",)

    lowering = lower_generic(term, "c")
    # The flat buffer is addressed through an offsets argument, which is where a
    # row of length zero has to be harmless: off[r] == off[r + 1].
    assert lowering.ragged == {"val": "off_cnt"}
    assert "val" in lowering.array_args and "off_cnt" in lowering.array_args
    assert lowering.outputs == ("y",)

    facts = facts_for(term, owner="rowsums")
    assert facts
    assert not [fact for fact in facts if fact.status is Status.REFUTED]


def test_an_empty_row_gives_the_same_answer_on_the_c_target() -> None:
    """The compiled run, the Python body and numpy agree, zero rows included."""
    arrays = _ragged_inputs()
    out = _run(rowsums.trace(), **arrays)
    assert list(np.asarray(out["y"])) == WANT

    native = _ragged_inputs()
    rowsums(**native)
    assert list(np.asarray(out["y"])) == list(native["y"].numpy())


def test_a_matrix_of_nothing_but_empty_rows_still_runs() -> None:
    """Every row empty: the flat buffer has no cells at all.

    loopy's C target cannot be handed a zero-length array (it tries to make a
    null pointer by calling the pointer type on ``0.0``), so the executor pads
    such an argument with a cell it can never reach and restores the original
    afterwards. That workaround is why this runs; see ``docs/loopy-notes.md``.
    """
    counts = [0, 0, 0]
    arrays = {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "val": Arr.ragged(counts, values=[]),
        "y": Arr.zeros(3),
    }
    out = _run(rowsums.trace(), **arrays)
    assert list(np.asarray(out["y"])) == [0.0, 0.0, 0.0]
    # The empty argument came back empty, not as the one-cell pad.
    assert np.asarray(out.get("val", arrays["val"].numpy())).size == 0


def test_a_kernel_with_no_rows_at_all_fails_on_the_c_target() -> None:
    """The fully degenerate case is loopy's limit, and is recorded as such.

    With zero rows, *every* argument is empty, including the ones whose length
    is how loopy infers ``n``. The executor's pad deliberately does not touch
    those (lengthening one would make loopy infer the wrong size), so the call
    reaches loopy's own inability to pass an empty array and fails there. This
    test pins that down rather than leaving it to be discovered: a shape with
    no rows is not something loopty can run on the C target today, and the
    failure is loopy's, not a silently wrong answer.
    """
    arrays = {
        "cnt": Arr.from_numpy(np.array([], dtype=np.int64)),
        "val": Arr.ragged([], values=[]),
        "y": Arr.zeros(0),
    }
    with pytest.raises(TypeError, match="expected c_"):
        LoopyExecutor().run(rowsums.trace(), **arrays)


# }}}


# {{{ what the reflected ragged bound cannot know


@kernel
def flat_csr(
    off: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[nnz], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The same product against flat storage: ``val`` indexed through ``off``."""
    for r in y.dom:
        y[r] = val[off[r]]


def test_the_flat_csr_access_is_assumed_and_not_decided() -> None:
    """``val[off[r]]`` is honestly unknown, and says so rather than passing.

    Deciding it needs the scan's postcondition, and no oracle here can take a
    hypothesis; see :mod:`loopty.flow`. The rule therefore states the obligation
    and leaves it ``ASSUMED`` with a reason. This test exists so that a future
    widening of the reflected parameter, which would make isl *able* to answer,
    cannot turn "nobody knows" into "decided" without someone noticing.
    """
    facts = _in_bounds(flat_csr, "val")
    assert len(facts) == 1, [fact.statement for fact in facts]
    fact = facts[0]
    assert fact.status is Status.ASSUMED
    assert not fact.decided_by
    assert "not quasi-affine" in fact.provenance["reason"]

    # The ragged spelling of the same access is decided, because there the
    # raggedness is in the type rather than in the arithmetic.
    ragged = _in_bounds(rowsums, "val")
    assert [fact.status for fact in ragged] == [Status.DECIDED]


# }}}
