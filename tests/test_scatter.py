"""Updates, ``y[cell[c, k]] += v``, and the atomic step that runs them at once.

PyOP2's assembly loop adds each cell's values into the nodes of the cell, and
two cells that share a node add into one cell of ``y``. Such a statement is an
update: it adds to the cell it writes a value that reads nothing else of the
array, so two of its instances' updates of one cell give the same sum in
either order, to the class of the sum. The kernel owes no distinct writes for
it, a schedule may reorder its updates, and two of them may run at the same
time only once ``Schedule.atomic`` makes them atomic. Where the cell is read
from an array, which isl cannot follow, a refusal names two instances that
update one cell on the example inputs, not a pair of isl's that need not.
"""

from __future__ import annotations

import numpy as np
import pytest
from lanky.ledger import Status
from lanky.prelude import Int, Real

from lines import where
from loopty import Arr, Fin, Schedule, kernel, program, reduce_sum, when
from loopty.executor import LoopyExecutor, emit_code
from loopty.flow import dependences
from loopty.interpret import interpret
from loopty.oracle import IslOracle
from loopty.schedule import IllegalCast

pytest.importorskip("loopy")


# {{{ kernels


@kernel
def scatter(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    yl: Arr[Fin[nc], Fin[3], Real],  # noqa: F821
    y: Arr[Fin[nv], Real],  # noqa: F821
):
    """Each cell's three values added into its nodes, as PyOP2 assembles."""
    for c in yl.dom:
        for k in yl.dom[c]:
            y[cell[c, k]] += yl[c, k]


@kernel
def overwrite(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    yl: Arr[Fin[nc], Fin[3], Real],  # noqa: F821
    y: Arr[Fin[nv], Real],  # noqa: F821
):
    """The same loop with ``=`` for ``+=``: a node keeps one cell's value."""
    for c in yl.dom:
        for k in yl.dom[c]:
            y[cell[c, k]] = yl[c, k]


@kernel
def diagonal(
    p: Arr[Fin[n], Fin[m]],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Fin[m], Real],  # noqa: F821
):
    """One cell of each row, chosen by ``p``: rows apart, whatever ``p`` holds."""
    for i in x.dom:
        y[i, p[i]] = x[i]


@kernel
def count(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    m: Arr[Fin[nv], Int],  # noqa: F821
):
    """How many cells meet at each node: an update of integers, exact."""
    for c in cell.dom:
        for k in cell.dom[c]:
            m[cell[c, k]] += 1


@kernel
def rowsum(v: Arr[Fin[n], Fin[m], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Each row's entries added into its cell of ``y``, one at a time."""
    for r in v.dom:
        for j in v.dom[r]:
            y[r] += v[r, j]


@kernel
def axpy(a: Real, x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """An update of a cell of its own at every instance."""
    for i in y.dom:
        y[i] += a * x[i]


@kernel
def doubled(y: Arr[Fin[n], Real]):  # noqa: F821
    """A read-modify-write that is no sum."""
    for i in y.dom:
        y[i] = 2.0 * y[i]


@kernel
def from_the_first(y: Arr[Fin[n], Real]):  # noqa: F821
    """A sum that reads another cell of the array it adds into."""
    for i in y.dom:
        y[i] += y[0]


@kernel
def taken_from(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """The cell subtracted from a value, not a value from the cell."""
    for i in y.dom:
        y[i] = x[i] - y[i]


@kernel
def guarded(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """An addition under a guard that reads the array it adds into."""
    for i in y.dom:
        with when(y[i] > 0.0):
            y[i] += x[i]


@kernel
def subtracted(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    yl: Arr[Fin[nc], Fin[3], Real],  # noqa: F821
    y: Arr[Fin[nv], Real],  # noqa: F821
):
    """Each cell's values taken from its nodes: an update too."""
    for c in yl.dom:
        for k in yl.dom[c]:
            y[cell[c, k]] -= yl[c, k]


@kernel
def zeroed_then_assembled(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    yl: Arr[Fin[nc], Fin[3], Real],  # noqa: F821
    y: Arr[Fin[nv], Real],  # noqa: F821
    total: Arr[Fin[1], Real],  # noqa: F821
):
    """``y`` zeroed, assembled, and summed: a write, updates and a read of it."""
    for i in y.dom:
        y[i] = 0.0
    for c in yl.dom:
        for k in yl.dom[c]:
            y[cell[c, k]] += yl[c, k]
    for z in total.dom:
        total[z] = reduce_sum(y[i] for i in y.dom)


@kernel
def assembled_then_scaled(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    yl: Arr[Fin[nc], Fin[3], Real],  # noqa: F821
    y: Arr[Fin[nv], Real],  # noqa: F821
):
    """``y`` assembled, then each node scaled by the sum of them: no update."""
    for c in yl.dom:
        for k in yl.dom[c]:
            y[cell[c, k]] += yl[c, k]
    for i in y.dom:
        y[i] = y[i] * reduce_sum(y[j] for j in y.dom)


@kernel
def moved(
    p: Arr[Fin[n], Fin[n]],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``x`` written into ``y`` through ``p``."""
    for i in x.dom:
        y[p[i]] = x[i]


@kernel
def scaled(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    yl: Arr[Fin[nc], Fin[3], Real],  # noqa: F821
    y: Arr[Fin[nv], Real],  # noqa: F821
):
    """Each node multiplied by its cells' values: no sum, so no update."""
    for c in yl.dom:
        for k in yl.dom[c]:
            y[cell[c, k]] = y[cell[c, k]] * yl[c, k]


@kernel
def reflected(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    yl: Arr[Fin[nc], Fin[3], Real],  # noqa: F821
    y: Arr[Fin[nv], Real],  # noqa: F821
):
    """Each node taken from its cells' values: two in turn do not commute."""
    for c in yl.dom:
        for k in yl.dom[c]:
            y[cell[c, k]] = yl[c, k] - y[cell[c, k]]


@kernel
def halves(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    m: Arr[Fin[nv], Int],  # noqa: F821
):
    """Half of each cell added into integers: every partial sum truncated."""
    for c in cell.dom:
        for k in cell.dom[c]:
            m[cell[c, k]] += 0.5


@kernel
def read_back(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    yl: Arr[Fin[nc], Fin[3], Real],  # noqa: F821
    y: Arr[Fin[nv], Real],  # noqa: F821
    z: Arr[Fin[nc], Fin[3], Real],  # noqa: F821
):
    """An assembly that reads each node back as it goes."""
    for c in yl.dom:
        for k in yl.dom[c]:
            y[cell[c, k]] += yl[c, k]
            z[c, k] = y[cell[c, k]]


@kernel
def two_updates(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    a: Arr[Fin[nc], Real],  # noqa: F821
    b: Arr[Fin[nc], Real],  # noqa: F821
    y: Arr[Fin[nv], Real],  # noqa: F821
):
    """Two statements that add into one array: their updates commute too."""
    for c in a.dom:
        y[cell[c, 0]] += a[c]
        y[cell[c, 1]] += b[c]


@program
def assembled_twice(cell, yl, y):
    """Two assemblies into one array."""
    scatter(cell, yl, y)
    scatter(cell, yl, y)


# }}}


def mesh() -> np.ndarray:
    """PyOP2's two triangles, which share the nodes 1 and 3."""
    return np.array([[0, 1, 3], [2, 3, 1]], dtype=np.int64)


def example_inputs() -> dict:
    """The inputs every fact of these kernels is tried on first."""
    cells = Arr.from_numpy(mesh())
    values = np.arange(1.0, 7.0).reshape(2, 3)
    assembly = {
        "cell": cells,
        "yl": Arr.from_numpy(values),
        "y": Arr.zeros(4),
    }
    return {
        "scatter": assembly,
        "overwrite": dict(assembly),
        "subtracted": dict(assembly),
        "diagonal": {
            "p": Arr.from_numpy(np.array([1, 1, 0])),
            "x": Arr.from_numpy(np.array([1.0, 2.0, 3.0])),
            "y": Arr.zeros((3, 2)),
        },
        "count": {"cell": cells, "m": Arr.from_numpy(np.zeros(4, dtype=np.int64))},
        "rowsum": {"v": Arr.from_numpy(np.ones((3, 4))), "y": Arr.zeros(3)},
        "scaled": dict(assembly),
        "reflected": dict(assembly),
        "halves": {"cell": cells, "m": Arr.from_numpy(np.zeros(4, dtype=np.int64))},
        "read_back": {**assembly, "z": Arr.zeros((2, 3))},
        "assembled_then_scaled": dict(assembly),
        # the last of 12 000 writes is the only one to a cell written before
        "moved": {
            "p": Arr.from_numpy(np.append(np.arange(11_999), 0)),
            "x": Arr.from_numpy(np.ones(12_000)),
            "y": Arr.zeros(12_000),
        },
        # node 3 is the second node of cell 0 and the first of cell 1
        "two_updates": {
            "cell": Arr.from_numpy(np.array([[0, 3, 1], [3, 2, 1]])),
            "a": Arr.from_numpy(np.ones(2)),
            "b": Arr.from_numpy(np.ones(2)),
            "y": Arr.zeros(4),
        },
    }


def inputs(name: str) -> dict:
    """A fresh copy of one kernel's example inputs."""
    return example_inputs()[name]


def facts_of(kern, kind: str) -> list:
    """The facts of one kind, as ``lanky check`` settles them with isl."""
    oracle = IslOracle()
    out = []
    for fact in kern.facts():
        if fact.kind != kind:
            continue
        if fact.status is Status.ASSUMED and oracle.can_establish(fact):
            fact = oracle.establish(fact) or fact
        out.append(fact)
    return out


# {{{ the statement


@pytest.mark.parametrize(
    ("kern", "exactness"),
    [
        (scatter, "approx"),
        (subtracted, "approx"),
        (rowsum, "approx"),
        (axpy, "approx"),
        (count, "exact"),
        (doubled, None),
        (from_the_first, None),
        (taken_from, None),
        (guarded, None),
    ],
)
def test_an_update_is_traced_with_the_class_of_its_sum(kern, exactness) -> None:
    (stmt,) = kern.term.stmts
    assert stmt.kind == "accumulate"
    assert stmt.exactness == exactness


def test_a_program_keeps_the_class_of_its_calls_updates() -> None:
    classes = {stmt.id: stmt.exactness for stmt in assembled_twice.term.stmts}
    assert classes == {"scatter.S0": "approx", "scatter@2.S0": "approx"}


def test_two_updates_of_one_cell_are_no_dependence() -> None:
    # Either order gives the sum: the source order owes them nothing.
    assert dependences(rowsum.term).is_empty()
    # A sum that reads another cell of its array keeps its dependences.
    assert not dependences(from_the_first.term).is_empty()


# }}}


# {{{ the kernel's facts


def test_an_assembly_owes_the_class_of_its_sum_and_no_distinct_writes() -> None:
    assert facts_of(scatter, "disjoint-writes") == []
    (sums,) = facts_of(scatter, "exactness")
    assert sums.statement == "the accumulation into y[cell[c, k]] over c, k is approx"
    assert sums.status is Status.DECIDED and sums.decided_by == "type"
    (rows,) = facts_of(rowsum, "exactness")
    assert rows.statement == "the accumulation into y[r] over j is approx"
    assert facts_of(rowsum, "disjoint-writes") == []


def test_an_update_of_a_cell_of_its_own_owes_distinct_writes() -> None:
    (disjoint,) = facts_of(axpy, "disjoint-writes")
    assert disjoint.status is Status.DECIDED and disjoint.decided_by == "isl"
    assert facts_of(axpy, "exactness") == []


def test_a_write_through_a_map_is_refuted_with_two_cells_of_the_mesh() -> None:
    # isl takes y[cell[c, k]] to reach any cell, and its pair need not share
    # one; the refutation is two cells of the mesh that write one node.
    (disjoint,) = facts_of(overwrite, "disjoint-writes")
    assert disjoint.status is Status.REFUTED
    assert disjoint.decided_by == "native"
    assert disjoint.provenance["witness"] == "S0[c=0, k=2] and S0[c=1, k=1]"
    # the witness keeps the ids; the reason names the statement as written
    assert disjoint.provenance["reason"] == (
        f"y[cell[c, k]] = yl[c, k] ({where(overwrite, '] = yl')}) at c=0, k=2 and "
        "at c=1, k=1 both write y[3] on example_inputs() (nc=2, nv=4), where "
        "cell[0, 2] = 3 and cell[1, 1] = 3, so the later write overwrites the "
        "earlier one"
    )


def test_a_write_through_a_map_is_refuted_past_ten_thousand_writes() -> None:
    # Every write of the statement is looked at: the only collision is the
    # last instance's, past the cap a refusal's search for a pair keeps.
    (disjoint,) = facts_of(moved, "disjoint-writes")
    assert disjoint.status is Status.REFUTED
    assert disjoint.provenance["witness"] == "S0[i=0] and S0[i=11999]"
    assert "on example_inputs() (n=12000)" in disjoint.provenance["reason"]


def test_a_write_through_a_map_that_keeps_rows_apart_is_tested() -> None:
    (disjoint,) = facts_of(diagonal, "disjoint-writes")
    assert disjoint.status is Status.TESTED
    assert disjoint.decided_by == "native"
    assert disjoint.provenance["inputs"][0] == "example_inputs()"
    assert "reason" not in disjoint.provenance


def test_the_interpreter_logs_the_cells_each_instance_touches() -> None:
    from loopty.interpret import Touch

    log: list[Touch] = []
    interpret(scatter.term, inputs("scatter"), log=log)
    writes = [(t.coords, t.cell, t.value) for t in log if t.kind == "write"]
    assert writes[:3] == [((0, 0), (0,), 1.0), ((0, 1), (1,), 2.0), ((0, 2), (3,), 3.0)]
    assert ((1, 1), (3,), 3.0 + 5.0) in writes
    reads = {(t.coords, t.array, t.cell, t.value) for t in log if t.kind == "read"}
    assert ((1, 1), "cell", (1, 1), 3) in reads


# }}}


# {{{ schedules


def test_the_cells_split_onto_work_items_are_refused_with_a_shared_node() -> None:
    schedule = Schedule(scatter).split(scatter.c, 32)
    with pytest.raises(IllegalCast) as caught:
        schedule.tag({scatter.c.outer: "g.0", scatter.c.inner: "l.0"})
    assert str(caught.value) == (
        "tag({scatter.c.outer: 'g.0', scatter.c.inner: 'l.0'}) illegal: "
        f"y[cell[c, k]] += yl[c, k] ({where(scatter, '+=')}) at c=0, k=2 and at "
        "c=1, k=1 both update y[3] on example_inputs() (nc=2, nv=4), where "
        "cell[0, 2] = 3 and cell[1, 1] = 3, and the loop scatter.c.inner on l.0 "
        "runs them on two work items, so one can overwrite what the other added; "
        "make the updates atomic, with atomic(scatter.y) before this step"
    )
    fact = caught.value.fact
    assert fact.kind == "disjoint-writes" and fact.status is Status.REFUTED
    assert fact.decided_by == "native"
    (first, _), (second, _), sizes = caught.value.witness
    assert first == second == "S0" and sizes == {"nc": 2, "nv": 4}


def test_a_schedules_own_example_inputs_are_read_first() -> None:
    # Two triangles that share node 0, given to the schedule: its refusal is
    # read off them, before the module's example_inputs().
    cells = Arr.from_numpy(np.array([[0, 1, 2], [3, 4, 0]]))
    schedule = (
        Schedule(scatter)
        .example(cell=cells, yl=Arr.zeros((2, 3)), y=Arr.zeros(5))
        .split(scatter.c, 32)
    )
    with pytest.raises(IllegalCast) as caught:
        schedule.tag({scatter.c.outer: "g.0", scatter.c.inner: "l.0"})
    assert (
        f"({where(scatter, '+=')}) at c=0, k=0 and at c=1, k=2 both update y[0] on "
        "the schedule's example inputs (nc=2, nv=5), where cell[0, 0] = 0 and "
        "cell[1, 2] = 0"
    ) in str(caught.value)


def test_atomic_updates_may_run_on_two_work_items() -> None:
    schedule = (
        Schedule(scatter)
        .atomic(scatter.y)
        .split(scatter.c, 32)
        .tag({scatter.c.outer: "g.0", scatter.c.inner: "l.0"})
    )
    assert schedule.atomics == frozenset({"y"})
    statements = [fact.statement for fact in schedule.facts()]
    assert (
        "atomic updates of scatter.y run at the same time after "
        "tag({scatter.c.outer: 'g.0', scatter.c.inner: 'l.0'}), so their sum is "
        "reassociated and compared at 'reassoc'"
    ) in statements
    # atomic moves nothing, and records no cast of its own
    assert not any(statement.startswith("atomic(") for statement in statements)
    assert "y" in schedule.reassociated
    # The C target has no work items: the schedule is legal and not built.
    buildable, reason = schedule.buildable
    assert not buildable and "the C target has none" in reason


def test_atomic_updates_compile_to_a_compare_and_swap_on_opencl(plain_opencl) -> None:
    schedule = (
        Schedule(scatter, target="opencl")
        .atomic(scatter.y)
        .split(scatter.c, 32)
        .tag({scatter.c.outer: "g.0", scatter.c.inner: "l.0"})
    )
    assert schedule.buildable == (True, "")
    code = emit_code(schedule)
    assert "__global volatile double *__restrict__ y" in code
    assert "atom_cmpxchg" in code
    assert "loopy_new_val = loopy_old_val + yl[" in code


@kernel
def both_ends(
    a: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n + 1], Real],  # noqa: F821
):
    """Each segment's value onto both its ends, as a face's flux onto its nodes."""
    for p in a.dom:
        y[p] += a[p]
        y[p + 1] += a[p]


def test_two_statements_updating_one_cell_need_no_barrier_once_atomic(
    plain_opencl,
) -> None:
    # S0 at p = 1 and S1 at p = 0 both update y[1]: on two work items they
    # race, and once atomic the two need no order. The lowering's order of
    # S1 after S0 has to go with it, or loopy asks for a global barrier
    # between them and generates nothing (MissingBarrierError).
    with pytest.raises(IllegalCast, match="both update y"):
        Schedule(both_ends).split(both_ends.p, 32).tag(
            {both_ends.p.outer: "g.0", both_ends.p.inner: "l.0"}
        )
    schedule = (
        Schedule(both_ends, target="opencl")
        .atomic(both_ends.y)
        .split(both_ends.p, 32)
        .tag({both_ends.p.outer: "g.0", both_ends.p.inner: "l.0"})
    )
    assert schedule.buildable == (True, "")
    insns = {insn.id: insn for insn in schedule.kernel.default_entrypoint.instructions}
    assert not insns["S1"].depends_on
    assert ("S0", "any") in insns["S1"].no_sync_with
    code = emit_code(schedule)
    assert code.count("atom_cmpxchg") == 2


def test_atomic_on_c_changes_no_code_and_agrees() -> None:
    plain = Schedule(scatter).split(scatter.c, 2)
    schedule = (
        Schedule(scatter)
        .atomic(scatter.y)
        .split(scatter.c, 2)
        .tag(scatter.c.inner, "ilp")
    )
    assert "atom" not in emit_code(schedule)
    assert emit_code(Schedule(scatter).atomic(scatter.y)) == emit_code(
        Schedule(scatter)
    )
    fact = LoopyExecutor().differential(scatter, schedule, inputs("scatter"))
    assert fact.status is Status.TESTED, fact.provenance
    assert fact.provenance["outputs"]["y"]["exactness"] == "approx"
    assert LoopyExecutor().differential(scatter, plain, inputs("scatter")).status is (
        Status.TESTED
    )


def test_updates_in_no_order_are_refused_until_atomic_on_c() -> None:
    # ilp leaves the two cells of a pair of rows in no order; the pair isl
    # picks is a real one here, since y[r] is a cell isl reads.
    with pytest.raises(IllegalCast) as caught:
        Schedule(rowsum).tag(rowsum.j, "ilp")
    message = str(caught.value)
    assert message.startswith(
        f"tag(rowsum.j, 'ilp') illegal: y[r] += v[r, j] ({where(rowsum, '+=')}) "
        "at r="
    )
    assert "both update y[" in message
    assert "the loop rowsum.j (ilp) leaves them in no order" in message
    assert caught.value.fact.decided_by == "isl"
    Schedule(rowsum).atomic(rowsum.y).tag(rowsum.j, "ilp")


def test_updates_reordered_in_one_thread_are_a_reassociation() -> None:
    # The entries of each row are added last first: the same sum, in another
    # order, which the source's order of updates no longer constrains.
    reversed_rows = Schedule(rowsum).affine(
        "[m] -> { [r, j] -> [a = r, k = m - 1 - j] }"
    )
    assert "y" in reversed_rows.reassociated
    fact = LoopyExecutor().differential(rowsum, reversed_rows, inputs("rowsum"))
    assert fact.status is Status.TESTED, fact.provenance


def test_atomic_refuses_an_exact_accumulation_and_what_is_not_updated() -> None:
    with pytest.raises(IllegalCast) as caught:
        Schedule(count).atomic(count.m)
    assert str(caught.value) == (
        "atomic(count.m) illegal: the accumulation into count.m is exact, and "
        "updates that run at the same time are applied in an order the source "
        "did not write, which reassociates it"
    )
    assert caught.value.fact.kind == "exactness"
    with pytest.raises(ValueError) as caught:
        Schedule(overwrite).atomic(overwrite.y)
    assert str(caught.value) == (
        "atomic(overwrite.y): overwrite does not update overwrite.y: S0 writes "
        "y[cell[c, k]] = "
        "yl[c, k], which replaces the cell's value. An update adds to the cell "
        "it writes a value that reads nothing else of y, y[...] += v, and "
        "atomic makes two such updates safe to run at the same time"
    )
    with pytest.raises(ValueError, match="scatter does not write scatter.yl"):
        Schedule(scatter).atomic(scatter.yl)


def test_an_exact_update_keeps_its_order_and_names_cells_of_the_mesh() -> None:
    with pytest.raises(IllegalCast) as caught:
        Schedule(count).tag(count.c, "g.0")
    message = str(caught.value)
    assert f"m[cell[c, k]] += 1 ({where(count, '+=')}) at c=0, k=2 writes m[3]" in (
        message
    )
    assert "also written by the same statement at c=1, k=1" in message
    assert (
        "on example_inputs() (nc=2, nv=4), where cell[0, 2] = 3 and cell[1, 1] = 3"
    ) in message


def test_a_write_through_a_map_on_two_work_items_names_cells_of_the_mesh() -> None:
    with pytest.raises(IllegalCast) as caught:
        Schedule(overwrite).tag(overwrite.c, "g.0")
    assert str(caught.value).startswith(
        "tag(overwrite.c, 'g.0') illegal: y[cell[c, k]] = yl[c, k] "
        f"({where(overwrite, '] = yl')}) at c=0, k=2 writes y[3], also written by "
        "the same statement at c=1, k=1, scheduled earlier on example_inputs() "
        "(nc=2, nv=4), where cell[0, 2] = 3 and cell[1, 1] = 3"
    )


def test_atomic_is_replayed_on_another_target(plain_opencl) -> None:
    schedule = Schedule(scatter).atomic(scatter.y).split(scatter.c, 4)
    again = schedule.retarget("opencl")
    assert again.atomics == frozenset({"y"})
    assert "atom_cmpxchg" in emit_code(again)
    assert again.history == ("atomic(scatter.y)", "split(scatter.c, 4)")


def test_atomic_updates_of_a_type_loopy_has_no_atomics_for_are_unbuildable(
    plain_opencl,
) -> None:
    @kernel
    def complex_scatter(
        cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
        yl: Arr[Fin[nc], Fin[3], np.complex128],  # noqa: F821
        y: Arr[Fin[nv], np.complex128],  # noqa: F821
    ):
        for c in yl.dom:
            for k in yl.dom[c]:
                y[cell[c, k]] += yl[c, k]

    schedule = Schedule(complex_scatter, target="opencl").atomic(complex_scatter.y)
    buildable, reason = schedule.buildable
    assert not buildable
    assert reason.startswith(
        "atomic(complex_scatter.y) asks for atomic updates of complex_scatter.y, "
        "which holds complex128"
    )


def test_every_access_to_an_atomic_array_is_atomic_on_opencl(plain_opencl) -> None:
    # loopy refuses a non-atomic access to an array declared for atomics: the
    # zeroing initializes y atomically, and the sum loads it so. (On work
    # items the zeroing and the sum, in no loop on the axis, would run on
    # every one of them, which is refused: a kernel has no barrier between
    # groups.)
    with pytest.raises(
        IllegalCast,
        match=rf"y.i. = 0.0 .{where(zeroed_then_assembled, '= 0.0')}. at i=.*: the "
        "first write is in no loop on g.0, so on every work item of it",
    ):
        Schedule(zeroed_then_assembled, target="opencl").atomic(
            zeroed_then_assembled.y
        ).split(zeroed_then_assembled.c, 32).tag(
            {zeroed_then_assembled.c.outer: "g.0", zeroed_then_assembled.c.inner: "l.0"}
        )
    schedule = (
        Schedule(zeroed_then_assembled, target="opencl")
        .atomic(zeroed_then_assembled.y)
        .split(zeroed_then_assembled.c, 4)
        .tag(zeroed_then_assembled.c.inner, "ilp")
    )
    code = emit_code(schedule)
    assert "__global volatile double *__restrict__ y" in code
    # The zeroing, a compare-and-swap of 0; the updates, four of them unrolled
    # by ilp; and the sum, reading y as the double it holds.
    assert code.count("atom_cmpxchg") == 1 + 4
    assert "loopy_new_val = (volatile double) (0.0);" in code
    assert "acc_i_0 = acc_i_0 + (double) (y[i_0]);" in code


def test_a_writer_that_reads_an_atomic_array_reads_it_as_its_dtype(
    plain_opencl,
) -> None:
    # The scaling writes y and reads it in a sum: its store initializes y
    # atomically, and its reads are of the double y holds, so that the sum's
    # accumulator is a plain double, as a read-only instruction's are.
    schedule = Schedule(assembled_then_scaled, target="opencl").atomic(
        assembled_then_scaled.y
    )
    assert schedule.buildable == (True, ""), schedule.buildable
    code = emit_code(schedule)
    assert "acc_j = acc_j + (double) (y[j]);" in code


@pytest.mark.parametrize(
    ("kern", "array", "why", "text"),
    [
        (
            scaled,
            "y",
            "S0 writes y[cell[c, k]] = y[cell[c, k]]*yl[c, k], which is not "
            "y[cell[c, k]] plus a value that reads nothing else of y",
            f"y[cell[c, k]] = y[cell[c, k]] * yl[c, k] ({where(scaled, '=')})",
        ),
        (
            reflected,
            "y",
            "S0 writes y[cell[c, k]] = yl[c, k] - y[cell[c, k]], which is not "
            "y[cell[c, k]] plus a value that reads nothing else of y",
            f"y[cell[c, k]] = yl[c, k] - y[cell[c, k]] ({where(reflected, '=')})",
        ),
        (
            halves,
            "m",
            "S0 writes m[cell[c, k]] = m[cell[c, k]] + 0.5, which adds a value "
            "that is not an integer to an integer array: each partial sum is "
            "truncated when it is stored, so the order of the updates changes "
            "the result",
            f"m[cell[c, k]] += 0.5 ({where(halves, '+=')})",
        ),
    ],
)
def test_what_does_not_commute_is_refused_atomic_and_keeps_its_order(
    kern, array, why, text
) -> None:
    (stmt,) = kern.term.stmts
    assert stmt.kind == "accumulate" and stmt.exactness is None
    with pytest.raises(ValueError) as caught:
        Schedule(kern).atomic(getattr(kern, array))
    assert f"{kern.__name__} does not update {kern.__name__}.{array}: {why}. " in str(
        caught.value
    )
    # On work items the two cells' writes of one node are a dependence, refused
    # with the cells of the mesh that share it, atomic or not.
    with pytest.raises(IllegalCast) as caught:
        Schedule(kern).split(kern.c, 32).tag({kern.c.outer: "g.0", kern.c.inner: "l.0"})
    assert str(caught.value).startswith(
        f"tag({{{kern.__name__}.c.outer: 'g.0', {kern.__name__}.c.inner: 'l.0'}}) "
        f"illegal: {text} at c=0, k=2 writes "
        f"{array}[3], also written by the same statement at c=1, k=1, scheduled "
        "earlier on example_inputs() (nc=2, nv=4), where cell[0, 2] = 3 and "
        "cell[1, 1] = 3"
    )


def test_a_read_beside_the_updates_is_refused_on_work_items_even_atomic() -> None:
    # Atomic makes two updates safe together, not a read of what one of them
    # is adding to: the pair names both subscripts the mesh read.
    with pytest.raises(IllegalCast) as caught:
        Schedule(read_back).atomic(read_back.y).split(read_back.c, 32).tag(
            {read_back.c.outer: "g.0", read_back.c.inner: "l.0"}
        )
    assert str(caught.value).startswith(
        "tag({read_back.c.outer: 'g.0', read_back.c.inner: 'l.0'}) illegal: "
        f"y[cell[c, k]] += yl[c, k] ({where(read_back, '+=')}) at c=0, k=2 writes "
        "y[3], read by z[c, k] = "
        f"y[cell[c, k]] ({where(read_back, 'z[c, k] =')}) at c=1, k=1, scheduled "
        "earlier on "
        "example_inputs() (nc=2, nv=4), where cell[0, 2] = 3 and cell[1, 1] = 3"
    )


def test_updates_of_two_statements_commute_and_race_on_work_items() -> None:
    assert dependences(two_updates.term).is_empty()
    with pytest.raises(IllegalCast) as caught:
        Schedule(two_updates).split(two_updates.c, 32).tag(
            {two_updates.c.outer: "g.0", two_updates.c.inner: "l.0"}
        )
    # S0 meets no other cell's S0 on the example, and the pair is not drawn:
    # S1 of cell 0 and S0 of cell 1 both add into node 3 there.
    assert str(caught.value).startswith(
        "tag({two_updates.c.outer: 'g.0', two_updates.c.inner: 'l.0'}) illegal: "
        "y[cell[c, 1]] += b[c] "
        f"({where(two_updates, '+= b[c]')}) at c=0 and y[cell[c, 0]] += a[c] "
        f"({where(two_updates, '+= a[c]')}) at c=1 both update y[3] on "
        "example_inputs() "
        "(nc=2, nv=4), where cell[0, 1] = 3 and cell[1, 0] = 3, and the loop "
        "two_updates.c.inner on l.0 runs them on two work items"
    )
    assert caught.value.fact.decided_by == "native"
    Schedule(two_updates).atomic(two_updates.y).split(two_updates.c, 32).tag(
        {two_updates.c.outer: "g.0", two_updates.c.inner: "l.0"}
    )


def test_a_refusal_names_the_first_pair_of_a_large_example() -> None:
    # A mesh of 1 800 triangles: the refusal reads the first two cells that
    # share a node off it, on two work items, without asking every pair.
    n = 30
    cells = []
    for i in range(n):
        for j in range(n):
            a, b = i * (n + 1) + j, (i + 1) * (n + 1) + j
            cells += [[a, b, b + 1], [a, b + 1, a + 1]]
    schedule = Schedule(scatter).example(
        cell=Arr.from_numpy(np.array(cells)),
        yl=Arr.zeros((len(cells), 3)),
        y=Arr.zeros((n + 1) ** 2),
    )
    with pytest.raises(IllegalCast) as caught:
        schedule.split(scatter.c, 32).tag(
            {scatter.c.outer: "g.0", scatter.c.inner: "l.0"}
        )
    assert str(caught.value).startswith(
        "tag({scatter.c.outer: 'g.0', scatter.c.inner: 'l.0'}) illegal: "
        f"y[cell[c, k]] += yl[c, k] ({where(scatter, '+=')}) at c=0, k=0 and at "
        "c=1, k=0 both update y[0] on the schedule's example inputs (nc=1800, "
        "nv=961), where cell[0, 0] = 0 and cell[1, 0] = 0, and the loop "
        "scatter.c.inner on l.0 runs them on two work items"
    )


def test_an_assumption_keeps_the_pair_to_the_inputs_it_admits() -> None:
    # The module's example has two cells; assumed a multiple of four, the
    # pair is read off an input that is one.
    with pytest.raises(IllegalCast) as caught:
        Schedule(scatter).assume(scatter.nc % 4 == 0).split(scatter.c, 4).tag(
            scatter.c.inner, "ilp"
        )
    message = str(caught.value)
    assert "example_inputs() (nc=2" not in message
    assert "both update y[" in message


def test_atomic_twice_is_one_step() -> None:
    once = Schedule(scatter).atomic(scatter.y)
    assert once.atomic(scatter.y) is once
    assert once.history == ("atomic(scatter.y)",)


def test_atomic_then_ilp_on_c_names_the_atomic_updates() -> None:
    schedule = (
        Schedule(scatter)
        .atomic(scatter.y)
        .split(scatter.c, 4)
        .tag(scatter.c.inner, "ilp")
    )
    rows = [(fact.kind, fact.statement) for fact in schedule.facts()]
    assert (
        "exactness",
        "atomic updates of scatter.y run at the same time after "
        "tag(scatter.c.inner, 'ilp'), so their sum is reassociated and compared "
        "at 'reassoc'",
    ) in rows
    assert [kind for kind, _ in rows].count("bijective") == 2  # split, tag


def test_an_ilp_tag_keeps_loopys_note_on_its_scheduler_out_of_a_run() -> None:
    # loopy schedules a kernel with an ilp loop with its first scheduler and
    # warns that it does, on every build. The tag is the schedule's own, so
    # the fallback is expected: the kernel silences the warning, and the code
    # is generated without it.
    import warnings

    schedule = (
        Schedule(scatter)
        .atomic(scatter.y)
        .split(scatter.c, 4)
        .tag(scatter.c.inner, "ilp")
    )
    silenced = schedule.kernel.default_entrypoint.silenced_warnings
    assert "v1_scheduler_fallback" in silenced
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit_code(schedule)
    assert not [w for w in caught if "slow scheduler" in str(w.message)]
    # a tag on a hardware axis leaves loopy's warnings as they were
    serial = Schedule(scatter).atomic(scatter.y).split(scatter.c, 32)
    tagged = serial.tag({scatter.c.outer: "g.0", scatter.c.inner: "l.0"})
    assert "v1_scheduler_fallback" not in (
        tagged.kernel.default_entrypoint.silenced_warnings
    )


@kernel
def twice_into(y: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """Twice each cell of y."""
    for i in y.dom:
        f[i] = 2.0 * y[i]


@kernel
def added_back(
    f: Arr[Fin[n], Real],  # noqa: F821
    node: Arr[Fin[n], Fin[n]],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """f added back into y, at the cells node names."""
    for i in f.dom:
        y[node[i]] += f[i]


@program
def twice_and_added_back(y, node):
    """y doubled into f, then f added back into y: updates of what f read."""
    f = Arr.zeros_like(y)
    twice_into(y, f)
    added_back(f, node, y)


def test_a_substitution_read_before_the_updates_names_isls_pair_as_isls() -> None:
    # Computed where it is read, 2 * y[i] would read y after the updates of
    # the instances before it. That read is one no run of an input makes,
    # so no input's pair is one of its pairs (a pair of the updates' own
    # reads would not be): the pair is isl's, and the message names the
    # subscript isl cannot follow, the update's.
    with pytest.raises(IllegalCast) as caught:
        Schedule(twice_and_added_back, sizes={"n": 8}).substitute(
            twice_and_added_back.f
        )
    assert str(caught.value) == (
        "substitute(twice_and_added_back.f) illegal: y[node[i]] += f[i] "
        f"({where(added_back, '+=')}) at i=1 reads y[1], overwritten by the same "
        "statement at i=0, scheduled earlier (at n=8, as hinted). The pair is "
        "isl's: the cell y[node[i]] is read from an array, which isl cannot "
        "follow, and substitute(twice_and_added_back.f) reads y where f was "
        "read, which no run of an input shows"
    )


def test_atomic_updates_run_on_an_opencl_device() -> None:
    # Needs pyopencl and a device, which CI has neither of.
    pytest.importorskip("pyopencl")
    schedule = (
        Schedule(scatter, target="opencl")
        .atomic(scatter.y)
        .split(scatter.c, 32)
        .tag({scatter.c.outer: "g.0", scatter.c.inner: "l.0"})
    )
    big = np.random.default_rng(0).integers(0, 9, size=(200, 3))
    arguments = {
        "cell": Arr.from_numpy(big),
        "yl": Arr.from_numpy(np.random.default_rng(1).random((200, 3))),
        "y": Arr.zeros(9),
    }
    fact = LoopyExecutor(target="opencl").differential(scatter, schedule, arguments)
    assert fact.status is Status.TESTED, fact.provenance
    zeroed = (
        Schedule(zeroed_then_assembled, target="opencl")
        .atomic(zeroed_then_assembled.y)
        .split(zeroed_then_assembled.c, 4)
        .tag(zeroed_then_assembled.c.inner, "ilp")
    )
    arguments = {**arguments, "y": Arr.zeros(9), "total": Arr.zeros(1)}
    fact = LoopyExecutor(target="opencl").differential(
        zeroed_then_assembled, zeroed, arguments
    )
    assert fact.status is Status.TESTED, fact.provenance
    # Two statements' updates of one array, on work items; and single precision.
    rng = np.random.default_rng(2)
    both = (
        Schedule(two_updates, target="opencl")
        .atomic(two_updates.y)
        .split(two_updates.c, 32)
        .tag({two_updates.c.outer: "g.0", two_updates.c.inner: "l.0"})
    )
    arguments = {
        "cell": Arr.from_numpy(big),
        "a": Arr.from_numpy(rng.random(200)),
        "b": Arr.from_numpy(rng.random(200)),
        "y": Arr.zeros(9),
    }
    fact = LoopyExecutor(target="opencl").differential(two_updates, both, arguments)
    assert fact.status is Status.TESTED, fact.provenance

    @kernel
    def scatter32(
        cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
        yl: Arr[Fin[nc], Fin[3], np.float32],  # noqa: F821
        y: Arr[Fin[nv], np.float32],  # noqa: F821
    ):
        for c in yl.dom:
            for k in yl.dom[c]:
                y[cell[c, k]] += yl[c, k]

    single = (
        Schedule(scatter32, target="opencl")
        .atomic(scatter32.y)
        .split(scatter32.c, 32)
        .tag({scatter32.c.outer: "g.0", scatter32.c.inner: "l.0"})
    )
    arguments = {
        "cell": Arr.from_numpy(big),
        "yl": Arr.from_numpy(rng.random((200, 3)).astype(np.float32)),
        "y": Arr.from_numpy(np.zeros(9, dtype=np.float32)),
    }
    fact = LoopyExecutor(target="opencl").differential(scatter32, single, arguments)
    assert fact.status is Status.TESTED, fact.provenance


# }}}
