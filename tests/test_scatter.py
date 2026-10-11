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
    assert disjoint.provenance["reason"] == (
        "S0[c=0, k=2] and S0[c=1, k=1] both write y[3] on example_inputs() "
        "(nc=2, nv=4), where cell[0, 2] = 3 and cell[1, 1] = 3, so the later "
        "write overwrites the earlier one"
    )


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
    schedule = Schedule(scatter).split("c", 32)
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(c_outer="g.0", c_inner="l.0")
    assert str(caught.value) == (
        "tag(c_outer='g.0', c_inner='l.0') illegal: S0[c=0, k=2] and S0[c=1, "
        "k=1] both update y[3] on example_inputs() (nc=2, nv=4), where cell[0, "
        "2] = 3 and cell[1, 1] = 3, and the loop c_inner on l.0 runs them on "
        "two work items, so one can overwrite what the other added; make the "
        "updates atomic, with atomic('y') before this step"
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
        .split("c", 32)
    )
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(c_outer="g.0", c_inner="l.0")
    assert (
        "S0[c=0, k=0] and S0[c=1, k=2] both update y[0] on the schedule's "
        "example inputs (nc=2, nv=5), where cell[0, 0] = 0 and cell[1, 2] = 0"
    ) in str(caught.value)


def test_atomic_updates_may_run_on_two_work_items() -> None:
    schedule = (
        Schedule(scatter).atomic("y").split("c", 32).tag(c_outer="g.0", c_inner="l.0")
    )
    assert schedule.atomics == frozenset({"y"})
    statements = [fact.statement for fact in schedule.facts()]
    assert (
        "the accumulation into y is reassociated by tag(c_outer='g.0', "
        "c_inner='l.0'), so its result is compared at 'reassoc'"
    ) in statements
    assert "y" in schedule.reassociated
    # The C target has no work items: the schedule is legal and not built.
    buildable, reason = schedule.buildable
    assert not buildable and "the C target has none" in reason


def test_atomic_updates_compile_to_a_compare_and_swap_on_opencl(plain_opencl) -> None:
    schedule = (
        Schedule(scatter, target="opencl")
        .atomic("y")
        .split("c", 32)
        .tag(c_outer="g.0", c_inner="l.0")
    )
    assert schedule.buildable == (True, "")
    code = emit_code(schedule)
    assert "__global volatile double *__restrict__ y" in code
    assert "atom_cmpxchg" in code
    assert "loopy_new_val = loopy_old_val + yl[" in code


def test_atomic_on_c_changes_no_code_and_agrees() -> None:
    plain = Schedule(scatter).split("c", 2)
    schedule = Schedule(scatter).atomic("y").split("c", 2).tag(c_inner="ilp")
    assert "atom" not in emit_code(schedule)
    assert emit_code(Schedule(scatter).atomic("y")) == emit_code(Schedule(scatter))
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
        Schedule(rowsum).tag(j="ilp")
    message = str(caught.value)
    assert message.startswith("tag(j='ilp') illegal: S0[r=")
    assert "both update y[" in message
    assert "the loop j (ilp) leaves them in no order" in message
    assert caught.value.fact.decided_by == "isl"
    Schedule(rowsum).atomic("y").tag(j="ilp")


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
        Schedule(count).atomic("m")
    assert str(caught.value) == (
        "atomic('m') illegal: the accumulation into m is exact, and updates that "
        "run at the same time are applied in an order the source did not write, "
        "which reassociates it"
    )
    assert caught.value.fact.kind == "exactness"
    with pytest.raises(ValueError, match="overwrite does not update y: S0 writes it"):
        Schedule(overwrite).atomic("y")
    with pytest.raises(ValueError, match="scatter does not write yl"):
        Schedule(scatter).atomic("yl")


def test_an_exact_update_keeps_its_order_and_names_cells_of_the_mesh() -> None:
    with pytest.raises(IllegalCast) as caught:
        Schedule(count).tag(c="g.0")
    message = str(caught.value)
    assert "writes m[3] also written by" in message
    assert (
        "on example_inputs() (nc=2, nv=4), where cell[0, 2] = 3 and cell[1, 1] = 3"
    ) in message


def test_a_write_through_a_map_on_two_work_items_names_cells_of_the_mesh() -> None:
    with pytest.raises(IllegalCast) as caught:
        Schedule(overwrite).tag(c="g.0")
    assert str(caught.value).startswith(
        "tag(c='g.0') illegal: instance S0[c=0, k=2] writes y[3] also written by "
        "S0[c=1, k=1] scheduled earlier on example_inputs() (nc=2, nv=4), where "
        "cell[0, 2] = 3 and cell[1, 1] = 3"
    )


def test_atomic_is_replayed_on_another_target(plain_opencl) -> None:
    schedule = Schedule(scatter).atomic("y").split("c", 4)
    again = schedule.retarget("opencl")
    assert again.atomics == frozenset({"y"})
    assert "atom_cmpxchg" in emit_code(again)
    assert again.history == ("atomic('y')", "split(c, 4)")


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

    schedule = Schedule(complex_scatter, target="opencl").atomic("y")
    buildable, reason = schedule.buildable
    assert not buildable
    assert reason.startswith(
        "atomic('y') asks for atomic updates of y, which holds complex128"
    )


def test_every_access_to_an_atomic_array_is_atomic_on_opencl(plain_opencl) -> None:
    # loopy refuses a non-atomic access to an array declared for atomics: the
    # zeroing initializes y atomically, and the sum loads it so. (On work
    # items the zeroing and the sum, in no loop on the axis, would run on
    # every one of them, which is refused: a kernel has no barrier between
    # groups.)
    with pytest.raises(IllegalCast, match="S0 runs in no loop on g.0"):
        Schedule(zeroed_then_assembled, target="opencl").atomic("y").split(
            "c", 32
        ).tag(c_outer="g.0", c_inner="l.0")
    schedule = (
        Schedule(zeroed_then_assembled, target="opencl")
        .atomic("y")
        .split("c", 4)
        .tag(c_inner="ilp")
    )
    code = emit_code(schedule)
    assert "__global volatile double *__restrict__ y" in code
    # The zeroing, a compare-and-swap of 0; the updates, four of them unrolled
    # by ilp; and the sum, reading y as the double it holds.
    assert code.count("atom_cmpxchg") == 1 + 4
    assert "loopy_new_val = (volatile double) (0.0);" in code
    assert "acc_i_0 = acc_i_0 + (double) (y[i_0]);" in code


def test_atomic_updates_run_on_an_opencl_device() -> None:
    # Needs pyopencl and a device, which CI has neither of.
    pytest.importorskip("pyopencl")
    schedule = (
        Schedule(scatter, target="opencl")
        .atomic("y")
        .split("c", 32)
        .tag(c_outer="g.0", c_inner="l.0")
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
        .atomic("y")
        .split("c", 4)
        .tag(c_inner="ilp")
    )
    arguments = {**arguments, "y": Arr.zeros(9), "total": Arr.zeros(1)}
    fact = LoopyExecutor(target="opencl").differential(
        zeroed_then_assembled, zeroed, arguments
    )
    assert fact.status is Status.TESTED, fact.provenance


# }}}
