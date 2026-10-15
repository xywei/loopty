"""Dependences between work items, refused as casts (#63).

A loop on a hardware axis is the launch grid: loopy runs every work item at
its own value of the loop, outside every other loop of the kernel, and orders
no two work items through global memory. The casts used to drop such a loop
from the order and let the loops around it order the rest, so a dependence
between two work items passed them. On a device that is a race loopy does
not see, because its barrier check asks only about pairs of instructions and
a stencil's dependence is its statement's own, or a global barrier loopy asks
for when it generates the code. ``buildable`` passed both.

Each case is asked of the casts and of loopy itself, whose code generation is
run on the kernel tagged directly wherever the cast now refuses, with loopy's
plain OpenCL target standing in for a device (see the ``plain_opencl``
fixture) and its caches off.
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import Any

import islpy as isl
import pymbolic.primitives as prim
import pytest
from lanky.prelude import Nat, Real

import hand_terms as ht
from lines import where
from loopty import Arr, Fin, kernel, program, reduce_sum, when
from loopty.handles import scope
from loopty.isl_reading import declining
from loopty.schedule import IllegalCast, Schedule
from loopty.term import Access, Stmt, Term

pytest.importorskip("loopy")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@functools.cache
def example(name: str) -> Any:
    """A demo, imported in this process."""
    from lanky.check import import_path

    return import_path(str(EXAMPLES / f"{name}.py"))


def loopy_says(lp, kernel_: Any, tags: dict[str, str]) -> str:
    """The device code loopy generates for ``kernel_`` tagged so, or its error.

    The error is its type and its message. Caches off, so that code generated
    once is generated again rather than found. loopy warns that it cannot nest
    a loop it was asked to nest once that loop is on an axis, which is what
    the tag asks for here, and the warning is silenced.
    """
    with warnings.catch_warnings(), lp.CacheMode(False), declining():
        warnings.simplefilter("ignore", lp.diagnostic.LoopyWarning)
        warnings.filterwarnings(
            "ignore", message="Cannot enforce the constraint", category=UserWarning
        )
        try:
            return lp.generate_code_v2(lp.tag_inames(kernel_, tags)).device_code()
        except Exception as exc:  # noqa: BLE001 - loopy raises anything
            return f"{type(exc).__name__}: {exc}"


def on_loops(schedule: Schedule, tags: dict[str, str]) -> Schedule:
    """``schedule`` with each loop of ``tags``, named by its binder, so tagged.

    The kernel's own loops, which ``loopy_says`` is given by the same names.
    """
    names = scope(schedule)
    return schedule.tag({getattr(names, loop): tag for loop, tag in tags.items()})


def casts(schedule: Schedule) -> list[str]:
    """The statuses of a schedule's cast facts, every kind but ``buildable``."""
    return [fact.status.value for fact in schedule.facts() if fact.kind != "buildable"]


GLOBAL_BARRIER = "requires synchronization by a global barrier"


# {{{ the rows of the issue


@pytest.mark.parametrize("axis", ["g.0", "l.0"])
def test_a_stencil_step_on_two_work_items_is_refused(plain_opencl, axis) -> None:
    # Each work item reads at step t + 1 what its neighbour wrote at step t.
    # The dependence is the statement's own, across iterations of t, and
    # loopy's barrier check asks only about two instructions, so it generates
    # the code with no barrier at all.
    jacobi = example("stencil_skew").jacobi
    schedule = Schedule(jacobi, target="opencl", sizes={"nt": 16, "nx": 16})
    code = loopy_says(plain_opencl, schedule.kernel, {"i": axis})
    assert ("get_group_id" if axis == "g.0" else "get_local_id") in code, code
    assert "barrier" not in code

    with pytest.raises(IllegalCast) as caught:
        schedule.tag(jacobi.i, axis)
    message = str(caught.value)
    # the statement as the demo writes it, its line, and the pair's coordinates
    assert message.startswith(
        f"tag(jacobi.i, {axis!r}) illegal: u[t + 1, i] = (u[t, i - 1] + "
        "u[t, i + 1]) / 2 "
        f"({where(jacobi, 'u[t + 1, i] =')}) at t="
    )
    assert "writes u[" in message and ", read by the same statement at t=" in message
    assert "on another work item (at nt=16, nx=16, as hinted)" in message
    assert f"the loop jacobi.i on {axis} runs them on work items" in message
    assert message.endswith(
        "nothing in a kernel orders two work items through global memory"
    )
    (source_id, source), (sink_id, sink), params = caught.value.witness
    assert source_id == sink_id == "S0"
    assert sink["t"] == source["t"] + 1
    assert abs(sink["i"] - source["i"]) == 1
    assert params == {"nt": 16, "nx": 16}
    fact = caught.value.fact
    assert (fact.kind, fact.status.value) == ("monotone", "refuted")
    assert fact.provenance["reason"] == message
    assert fact.provenance["detail"] == (
        f"a raw dependence on u joins two work items of {axis}"
    )


@pytest.mark.parametrize("axis", ["g.0", "l.0"])
def test_the_acoustic_pair_on_two_work_items_is_refused(plain_opencl, axis) -> None:
    # S1 reads the velocity S0 wrote one point to the left in the same step.
    # loopy sees that one, between two instructions, and asks for a global
    # barrier; buildable used to say the code could be generated.
    acoustic = example("wavefront_acoustic").acoustic
    schedule = Schedule(acoustic, target="opencl", sizes={"nt": 16, "nx": 32})
    said = loopy_says(plain_opencl, schedule.kernel, {"i": axis})
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said

    with pytest.raises(IllegalCast) as caught:
        schedule.tag(acoustic.i, axis)
    message = str(caught.value)
    # too long to show whole, each statement keeps what it writes into
    assert "velocity[t + 1, i] = ... (wavefront_acoustic.py:" in message
    assert "writes velocity[" in message
    assert ", read by pressure[t + 1, i] = ... (wavefront_acoustic.py:" in message
    assert (
        "on another work item (at nt=16, nx=32, as hinted): the loop acoustic.i "
        f"on {axis}"
        in message
    )
    (source_id, source), (sink_id, sink), _ = caught.value.witness
    assert (source_id, sink_id) == ("S0", "S1")
    assert sink["t"] == source["t"] and sink["i"] == source["i"] + 1


@kernel
def reversed_copy(x: Arr[Fin[8], Real], y: Arr[Fin[8], Real], z: Arr[Fin[8], Real]):
    """``y`` in one loop, then ``z`` from ``y`` read backwards in the next."""
    for i in y.dom:
        y[i] = x[i] + 1.0
    for k in z.dom:
        z[k] = y[7 - k]


@pytest.mark.parametrize("axis", ["g.0", "l.0"])
def test_two_loops_that_pass_data_along_one_axis_are_refused(plain_opencl, axis):
    schedule = Schedule(reversed_copy, target="opencl")
    said = loopy_says(plain_opencl, schedule.kernel, {"i": axis, "k": axis})
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said

    with pytest.raises(IllegalCast) as caught:
        schedule.tag({reversed_copy.i: axis, reversed_copy.k: axis})
    message = str(caught.value)
    assert (
        f"the loops reversed_copy.i and reversed_copy.k on {axis} run them on "
        "work items"
    ) in message
    (source_id, source), (sink_id, sink), _ = caught.value.witness
    assert (source_id, sink_id) == ("S0", "S1")
    assert source["i"] == 7 - sink["k"] != sink["k"]

    # #55's remedy for a statement off the axis, a loop of it on the axis as
    # well, leads here. With the axis on one loop only, the other statement
    # runs on every work item, and the cast says so before buildable is asked.
    said = loopy_says(plain_opencl, schedule.kernel, {"i": axis})
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(reversed_copy.i, axis)
    assert f"the read is in no loop on {axis}, so on every work item of it" in str(
        caught.value
    )


def test_rows_whose_dependences_stay_in_their_row_are_unchanged(plain_opencl):
    # spmv's rows on a group axis: nothing one row computes is read by another.
    schedule = example("spmv").rows_parallel()
    assert casts(schedule) == ["decided"] * 2
    assert schedule.buildable == (True, "")
    (_, monotone) = schedule.facts()
    assert monotone.statement == (
        "the order after tag(spmv.r, 'g.0') runs every dependence of spmv "
        "forward, within one work item"
    )


@pytest.mark.parametrize("split", [False, True])
def test_targets_whose_terms_stay_with_their_target_are_unchanged(
    plain_opencl, split
) -> None:
    # p2p's targets across the grid: each writes the terms of its own list
    # and sums them into its own potential, so nothing crosses a work item.
    p2p = example("p2p").p2p
    schedule = Schedule(p2p, target="opencl")
    tags = {"t": "g.0"}
    handles = {p2p.t: "g.0"}
    if split:
        schedule = schedule.split(p2p.t, 4)
        tags = {"t_outer": "g.0", "t_inner": "l.0"}
        handles = {p2p.t.outer: "g.0", p2p.t.inner: "l.0"}
    code = loopy_says(plain_opencl, schedule.kernel, tags)
    assert "get_group_id" in code, code
    tagged = schedule.tag(handles)
    assert set(casts(tagged)) == {"decided"}
    assert tagged.buildable == (True, "")


# }}}


# {{{ where a work item is


def offset_loops_term(shift: int) -> Term:
    """``y[i] = x[i]`` over ``1 <= i < n``, then ``z[k] = y[k + shift]``.

    The second loop runs over ``0 <= k < n - 1``, so the two loops start at
    different values, and loopy counts each one's work items from its own
    start. ``order`` puts the two statements in two loops one after the
    other, as the tracer would record them.
    """
    first = Stmt(
        id="S0",
        inames=("i",),
        domain=isl.Set("[n] -> { [i] : 1 <= i < n }"),
        assignee=Access("y", (prim.Variable("i"),)),
        expr=ht.S("x", prim.Variable("i")),
        kind="assign",
        guard=None,
        where="test_work_items.py:offset_loops",
        order=(0, 0),
    )
    second = Stmt(
        id="S1",
        inames=("k",),
        domain=isl.Set("[n] -> { [k] : 0 <= k < n - 1 }"),
        assignee=Access("z", (prim.Variable("k"),)),
        expr=ht.S("y", prim.Variable("k") + shift),
        kind="assign",
        guard=None,
        where="test_work_items.py:offset_loops",
        order=(1, 0),
    )
    n = prim.Variable("n")
    return Term(
        name="offset_loops",
        params=(("x", ht.dense(n)), ("y", ht.dense(n)), ("z", ht.dense(n))),
        sizes=("n",),
        stmts=(first, second),
        post=None,
    )


def test_a_work_item_is_counted_from_where_loopy_starts_the_loop(plain_opencl):
    # loopy runs i as 1 + the work item and k as 0 + the work item. So y[k + 1]
    # is read by the work item that wrote it, and y[k] by the one after it,
    # although i and k have the same value there.
    tags = {"i": "g.0", "k": "g.0"}
    same = Schedule(offset_loops_term(1), target="opencl", sizes={"n": 8})
    code = loopy_says(plain_opencl, same.kernel, tags)
    assert "get_group_id" in code, code
    tagged = on_loops(same, tags)
    assert casts(tagged) == ["decided"] * 2
    assert tagged.buildable == (True, "")

    next_one = Schedule(offset_loops_term(0), target="opencl", sizes={"n": 8})
    said = loopy_says(plain_opencl, next_one.kernel, tags)
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said
    with pytest.raises(IllegalCast) as caught:
        on_loops(next_one, tags)
    (_, source), (_, sink), _ = caught.value.witness
    assert source["i"] == sink["k"]
    first, second = source["i"] - 1, sink["k"]
    assert f"run them on work items {first} and {second} of it" in str(caught.value)


@kernel
def broadcast(x: Arr[Fin[8], Real], s: Arr[Fin[1], Real], y: Arr[Fin[8], Real]):
    """A cell set once, then read at every iteration of a loop."""
    s[0] = x[0]
    for i in y.dom:
        y[i] = s[0] * x[i]


def test_a_statement_with_no_loop_on_the_axis_runs_on_every_work_item(plain_opencl):
    # loopy asks for a global barrier between S0 and S1, and would refuse S0
    # for running on no loop of the axis besides (#55); the remedy it names
    # for that runs S0 on every work item, all of them writing s[0] while
    # others read it. The cast refuses first, and says why.
    schedule = Schedule(broadcast, target="opencl")
    said = loopy_says(plain_opencl, schedule.kernel, {"i": "g.0"})
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(broadcast.i, "g.0")
    message = str(caught.value)
    # a statement in no loop has one instance, which needs no coordinates
    assert message.startswith(
        "tag(broadcast.i, 'g.0') illegal: s[0] = x[0] "
        f"({where(broadcast, 's[0] =')}) writes s[0]"
    )
    assert f"read by y[i] = s[0] * x[i] ({where(broadcast, 'y[i] =')}) at i=" in (
        message
    )
    assert "the write is in no loop on g.0, so on every work item of it" in message


@kernel
def halved_columns(u: Arr[Fin[nt], Fin[nx], Real]):  # noqa: F821
    """Each column halved down the time levels; nothing leaves its column."""
    steps = u.dom
    for t in steps:
        for i in u.dom[t]:
            with when(t + 1 < steps.size):
                u[t + 1, i] = u[t, i] / 2


def test_a_skew_that_moves_a_column_across_work_items_is_refused() -> None:
    # The work item is asked after every step, not only after a tag: skewing
    # i by t puts step t of column i on work item i + t.
    sizes = {"nt": 4, "nx": 4}
    tagged = Schedule(halved_columns, sizes=sizes).tag(halved_columns.i, "l.0")
    assert casts(tagged) == ["decided"] * 2
    with pytest.raises(IllegalCast) as caught:
        tagged.skew(halved_columns.i, by=halved_columns.t)
    message = str(caught.value)
    assert message.startswith(
        "skew(halved_columns.i, by=halved_columns.t) illegal: u[t + 1, i] = "
        "u[t, i] / 2 "
        f"({where(halved_columns, 'u[t + 1, i] =')}) at t="
    )
    (_, source), (_, sink), _ = caught.value.witness
    assert sink["t"] == source["t"] + 1 and sink["i"] == source["i"]
    first, second = source["i"] + source["t"], sink["i"] + sink["t"]
    assert (
        f"the loop halved_columns.i on l.0 runs them on work items {first} and "
        f"{second}"
    ) in message


@pytest.mark.parametrize("tag", ["ilp", "vec"])
def test_a_loop_that_runs_in_one_work_item_loses_only_its_order(tag) -> None:
    # loopy runs ilp and vec around each instruction inside one work item, so
    # dropping them from the order is all there is to it (#57).
    jacobi = example("stencil_skew").jacobi
    tagged = Schedule(jacobi, sizes={"nt": 16, "nx": 16}).tag(jacobi.i, tag)
    assert casts(tagged) == ["decided"] * 2
    assert not tagged.facts()[1].statement.endswith("within one work item")


def test_without_a_kernel_two_loops_never_share_a_work_item() -> None:
    # A schedule with no kernel has no start to read off a loop (see
    # Schedule.kernel), so each start is a parameter of its own: two instances
    # of one loop share a work item when they agree on it, and two of
    # different loops never do.
    from loopty.schedule import _apart, _Layout, _work_items

    layout = _Layout(stmt_ids=("S0", "S1"), coords={"S0": ("i",), "S1": ("k",)})
    work = _work_items(layout, {"i": "g.0", "k": "g.0"}, None, {"n"})
    assert work is not None
    assert work.axes == ("g.0",) and work.known == frozenset()
    assert work.loops == {"S0": {"g.0": "i"}, "S1": {"g.0": "k"}}
    instances = isl.Set("[n] -> { [s, x0] : 0 <= s <= 1 and 0 <= x0 < n }")
    identity = isl.Map.identity(instances.get_space().map_from_set())
    apart = _apart(identity.intersect_domain(instances), work.coordinates["g.0"])

    def pair(a: tuple[int, int], b: tuple[int, int]) -> isl.Map:
        return isl.Map(
            f"[n] -> {{ [{a[0]}, {a[1]}] -> [{b[0]}, {b[1]}] : n = 8 }}"
        )

    assert pair((0, 3), (0, 3)).intersect(apart).is_empty()
    assert not pair((0, 3), (0, 4)).intersect(apart).is_empty()
    assert not pair((0, 3), (1, 3)).intersect(apart).is_empty()
    assert _work_items(layout, {"i": "ilp", "k": "vec"}, None, {"n"}) is None

    # A schedule left with no kernel by a map the rewrite cannot write: the
    # rows on a group axis keep their accumulation in their own work item.
    unwritten = Schedule(fiber_sums, sizes={"n": 4}).affine(
        "{ [r, j] -> [q, k] : q = r and k = j + r }"
    )
    assert unwritten.kernel is None
    tagged = unwritten.tag(fiber_sums.loops.q, "g.0")
    assert casts(tagged) == ["decided"] * 4
    assert tagged.facts()[-1].statement.endswith("within one work item")


@kernel
def fiber_sums(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Row sums written as a loop over the fiber, not as a reduction."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + val[r, j]


# }}}


# {{{ local sums stay allowed


@kernel
def two_row_sums(
    a: Arr[Fin[8], Fin[8], Real],
    b: Arr[Fin[8], Fin[8], Real],
    y: Arr[Fin[8], Real],
    z: Arr[Fin[8], Real],
):
    """Two sums on a local axis in one row loop, the second reading the first."""
    for i in y.dom:
        y[i] = reduce_sum(a[i, j] for j in a.dom[i])
        z[i] = y[i] + reduce_sum(b[i, k] for k in b.dom[i])


@kernel
def row_sums_read_backwards(
    a: Arr[Fin[8], Fin[8], Real],
    b: Arr[Fin[8], Fin[8], Real],
    y: Arr[Fin[8], Real],
    z: Arr[Fin[8], Real],
):
    """The same in two loops, the second reading the first's rows backwards."""
    for i in y.dom:
        y[i] = reduce_sum(a[i, j] for j in a.dom[i])
    for p in z.dom:
        z[p] = y[7 - p] + reduce_sum(b[p, k] for k in b.dom[p])


def test_local_sums_stay_allowed(plain_opencl) -> None:
    # A sum on a local axis happens inside one instance: loopy sums the parts
    # on every work item of the axis, combines them through local memory with
    # barriers, and stores the result from one work item, the first. Every
    # statement summed on the axis runs its own instruction there, so the
    # second reads, outside its sum, what the first wrote on that work item.
    sums = {"j": "l.0", "k": "l.0"}
    for fn, rows in ((two_row_sums, {"i": "g.0"}), (row_sums_read_backwards, {})):
        schedule = Schedule(fn, target="opencl")
        tagged = on_loops(schedule, {**rows, **sums})
        assert casts(tagged) == ["decided"] * 4
        assert tagged.facts()[1].statement.endswith("within one work item")
        assert tagged.buildable == (True, "")
        code = loopy_says(plain_opencl, schedule.kernel, {**rows, **sums})
        assert "barrier(CLK_LOCAL_MEM_FENCE)" in code, code
        assert code.count("if (lid(0) == 0)") == 2, code


def test_rows_on_a_group_axis_do_not_share_their_local_sums(plain_opencl):
    # The sums are allowed; the second loop reading another row's sum, in
    # another group, is not.
    tags = {"i": "g.0", "p": "g.0", "j": "l.0", "k": "l.0"}
    schedule = Schedule(row_sums_read_backwards, target="opencl")
    said = loopy_says(plain_opencl, schedule.kernel, tags)
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said
    with pytest.raises(
        IllegalCast,
        match="the loops row_sums_read_backwards.i and row_sums_read_backwards.p "
        "on g.0 run them",
    ):
        on_loops(schedule, tags)


@kernel
def sum_of_a_sum(a: Arr[Fin[8], Real], b: Arr[Fin[8], Real], s: Arr[Fin[2], Real]):
    """A sum, then a sum whose body reads it."""
    s[0] = reduce_sum(a[j] for j in a.dom)
    s[1] = reduce_sum(s[0] * b[k] for k in b.dom)


@kernel
def sum_after_a_sum(a: Arr[Fin[8], Real], b: Arr[Fin[8], Real], s: Arr[Fin[2], Real]):
    """A sum, then a statement that adds it to a sum of its own."""
    s[0] = reduce_sum(a[j] for j in a.dom)
    s[1] = s[0] + reduce_sum(b[k] for k in b.dom)


@kernel
def row_sum_in_a_row_sum(
    a: Arr[Fin[8], Fin[8], Real],
    b: Arr[Fin[8], Fin[8], Real],
    y: Arr[Fin[8], Real],
    z: Arr[Fin[8], Real],
):
    """Two row sums, the second's body reading the first's."""
    for i in y.dom:
        y[i] = reduce_sum(a[i, j] for j in a.dom[i])
        z[i] = reduce_sum(b[i, k] * y[i] for k in b.dom[i])


@kernel
def running_sums(a: Arr[Fin[nt], Fin[8], Real]):  # noqa: F821
    """Each step's last cell is the sum of the step before."""
    steps = a.dom
    for t in steps:
        with when(t + 1 < steps.size):
            a[t + 1, 7] = reduce_sum(a[t, k] for k in a.dom[t])


@pytest.mark.parametrize(
    ("fn", "tags", "sizes", "pair"),
    [
        (sum_of_a_sum, {"j": "l.0", "k": "l.0"}, None, ("S0", "S1")),
        (
            row_sum_in_a_row_sum,
            {"i": "g.0", "j": "l.0", "k": "l.0"},
            None,
            ("S0", "S1"),
        ),
        (running_sums, {"k": "l.0"}, {"nt": 4}, ("S0", "S0")),
    ],
    ids=["a-sum-of-a-sum", "a-row-sum-in-a-row-sum", "a-sum-at-the-next-step"],
)
def test_a_sum_that_reads_what_a_sum_stored_is_refused(
    plain_opencl, fn, tags, sizes, pair
) -> None:
    # The first statement stores its result on one work item; the body of
    # the second sum reads it on every work item of the axis. loopy asks for
    # a global barrier between the two, and the casts used to leave an axis
    # only sums are on unasked, so buildable said the code could be generated.
    schedule = Schedule(fn, target="opencl", sizes=sizes)
    said = loopy_says(plain_opencl, schedule.kernel, tags)
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said

    with pytest.raises(IllegalCast) as caught:
        on_loops(schedule, tags)
    message = str(caught.value)
    assert (
        "the write is on one work item of l.0, once its sum on l.0 is done; the "
        "read is in a sum, on every work item of l.0, and nothing"
    ) in message, message
    (source_id, _), (sink_id, _), _ = caught.value.witness
    assert (source_id, sink_id) == pair
    fact = caught.value.fact
    assert (fact.kind, fact.status.value) == ("monotone", "refuted")
    assert fact.provenance["detail"].endswith("joins two work items of l.0")


@kernel
def total_into_its_own_array(a: Arr[Fin[8], Real]):
    """The last cell set to the sum of the array, itself among them."""
    a[7] = reduce_sum(a[k] for k in a.dom)


def test_a_sum_over_the_cell_its_statement_writes_is_refused(plain_opencl):
    # One instance reads a[7] in its sum and writes it, so no two instances
    # depend on each other. On a device the read is on work item 7 and the
    # write on the first, and loopy asks for a global barrier between them.
    schedule = Schedule(total_into_its_own_array, target="opencl")
    said = loopy_says(plain_opencl, schedule.kernel, {"k": "l.0"})
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(total_into_its_own_array.k, "l.0")
    assert str(caught.value) == (
        "tag(total_into_its_own_array.k, 'l.0') illegal: a[7] = "
        "reduce_sum(a[k] for k in a.dom) "
        f"({where(total_into_its_own_array, 'a[7] =')}) reads a[7], overwritten "
        "by the same statement on another work item: the read is in a sum, on "
        "every work item of l.0; the write is on one work item of l.0, once its "
        "sum on l.0 is done, and nothing in a kernel orders two work items "
        "through global memory"
    )
    assert caught.value.fact.provenance["detail"] == (
        "a war dependence on a joins two work items of l.0"
    )
    # A row's sum into another array is a sum like any other.
    tags = {"j": "l.0", "k": "l.0"}
    assert casts(on_loops(Schedule(two_row_sums, target="opencl"), tags)) == [
        "decided"
    ] * 4


@kernel
def sums_beside_a_loop(
    x: Arr[Fin[8], Real], s: Arr[Fin[2], Real], y: Arr[Fin[8], Real]
):
    """Two sums on a local axis that a loop between them is on as well."""
    s[0] = reduce_sum(x[j] for j in x.dom)
    for i in y.dom:
        y[i] = x[i] * 2.0
    s[1] = s[0] + reduce_sum(x[k] for k in x.dom)


def test_a_sum_result_read_by_another_sums_statement_stays_allowed(plain_opencl):
    # Both statements store from the work item their sums leave the result
    # on, the first one, and the second reads s[0] there, outside its sum.
    # loopy generates both kernels. With a loop of another statement on the
    # axis, the statements around the sums used to count as running on every
    # work item, and the second kernel was refused.
    for fn, tags in (
        (sum_after_a_sum, {"j": "l.0", "k": "l.0"}),
        (sums_beside_a_loop, {"j": "l.0", "i": "l.0", "k": "l.0"}),
    ):
        schedule = Schedule(fn, target="opencl")
        code = loopy_says(plain_opencl, schedule.kernel, tags)
        assert "s[1] = s[0] + acc_k[0];" in code, code
        tagged = on_loops(schedule, tags)
        assert casts(tagged)[:2] == ["decided"] * 2
        assert tagged.buildable == (True, "")


def test_each_access_says_which_instruction_of_its_statement_makes_it() -> None:
    # The statement's own instruction, a sum's body, and the length of a
    # ragged row read where the row's loop starts: on a device three
    # instructions, which need not run on one work item.
    from loopty.schedule import _accesses

    term = Schedule(row_sum_in_a_row_sum).term
    accesses = _accesses(term.stmts[1], term)
    parts = {(kind, array, part) for kind, array, _, part in accesses}
    assert parts == {("write", "z", ""), ("read", "b", "sum"), ("read", "y", "sum")}
    # p2p's first statement runs in the loop over a target's list, whose
    # length is read once per target, outside that loop.
    term = Schedule(example("p2p").p2p).term
    accesses = _accesses(term.stmts[0], term)
    parts = {(kind, array, part) for kind, array, _, part in accesses}
    assert ("write", "term", "") in parts and ("read", "cnt", "row") in parts


# }}}


# {{{ a program's zeros on a device (#255, #261)


@kernel
def doubled(x: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """Each cell twice ``x``'s."""
    for i in f.dom:
        f[i] = 2.0 * x[i]


@kernel
def plus_one(f: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Each cell one more than ``f``'s."""
    for i in y.dom:
        y[i] = f[i] + 1.0


@program
def twice_plus_one(x, y):
    """``f`` an array of the program's own, zeroed where it is made."""
    f = Arr.zeros_like(x)
    doubled(x, f)
    plus_one(f, y)


def lp_quiet(lp: Any) -> Any:
    """loopy's code generation for a loopty kernel, its caches off."""
    import contextlib

    stack = contextlib.ExitStack()
    stack.enter_context(lp.CacheMode(False))
    stack.enter_context(declining())
    return stack


def test_a_programs_zeros_run_on_the_work_item_that_stores_each_cell(plain_opencl):
    # The program zeroes f in a statement of its own, in a loop the file
    # never names (i_1). Tagged with the calls' loop alone, every work item
    # would zero every cell, those another work item stores included: loopy
    # asks for a global barrier there. The zeros go on the loop's axis
    # instead, since doubled stores f along it, and each work item zeroes
    # the cell it stores, before it stores it.
    fused = Schedule(twice_plus_one, target="opencl").fuse(
        twice_plus_one.doubled.i, twice_plus_one.plus_one.i
    )
    (zeros,) = [stmt for stmt in fused.term.stmts if stmt.id == "f.zeros"]
    (loop,) = zeros.inames
    said = loopy_says(plain_opencl, fused.kernel, {"i": "g.0"})
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said
    tagged = fused.tag(twice_plus_one.doubled.i, "g.0")
    assert repr(tagged).endswith(".tag(twice_plus_one.doubled.i, 'g.0')")
    assert casts(tagged) == ["decided"] * 4
    assert tagged.buildable == (True, "")
    assert tagged.tags == {"i": "g.0", loop: "g.0"}
    with lp_quiet(plain_opencl):
        code = plain_opencl.generate_code_v2(tagged.kernel).device_code()
    body = code[code.index("{", code.index("__kernel")) :]
    assert body.index("f[gid(0)] = (double) (0.0);") < body.index(
        "f[gid(0)] = 2.0 * x[gid(0)];"
    ), code


def test_zeros_a_split_loop_stores_are_refused_saying_what_drops_them(plain_opencl):
    # Split, doubled stores f[4*i_outer + i_inner], along no loop of its own,
    # so the zeros stay on every work item, and the refusal says why and
    # which steps drop them.
    fused = Schedule(twice_plus_one, target="opencl").fuse(
        twice_plus_one.doubled.i, twice_plus_one.plus_one.i
    )
    with pytest.raises(IllegalCast) as caught:
        loop = twice_plus_one.doubled.i
        fused.split(loop, 4).tag({loop.outer: "g.0", loop.inner: "l.0"})
    message = str(caught.value)
    # the zeros, which no line of the file stores, by the array they zero
    made = where(twice_plus_one, "f = Arr.zeros_like")
    assert message.startswith(
        "tag({twice_plus_one.doubled.i.outer: 'g.0', "
        "twice_plus_one.doubled.i.inner: 'l.0'}) illegal: the zeros of "
        f"twice_plus_one.f ({made}) at f[0] writes f[0]"
    )
    assert (
        "the first write is in no loop on g.0, so on every work item of it"
    ) in message
    assert message.endswith(
        "nothing in a kernel orders two work items through global memory. The "
        "zeros twice_plus_one.f is made with run on the work item that stores "
        "each cell only where every statement that stores twice_plus_one.f "
        "stores it along a loop of its own on g.0, and none does here; keep "
        "twice_plus_one.f in a window (contract) or compute it where it is read "
        "(substitute), which drops them"
    )
    # substituted, f has no zeros, and the split is legal
    computed = (
        Schedule(twice_plus_one, target="opencl")
        .substitute(twice_plus_one.f)
        .split(twice_plus_one.plus_one.i, 4)
        .tag(
            {
                twice_plus_one.plus_one.i.outer: "g.0",
                twice_plus_one.plus_one.i.inner: "l.0",
            }
        )
    )
    assert set(casts(computed)) == {"decided"}


def test_a_call_that_runs_on_every_work_item_is_named_before_the_zeros(
    plain_opencl,
):
    # plus_one's loop is not tagged, so it runs on every work item and reads
    # what doubled stored on one: that is the refusal, not the zeros, which
    # follow doubled's loop.
    with pytest.raises(IllegalCast) as caught:
        Schedule(twice_plus_one, target="opencl").tag(
            twice_plus_one.doubled.i, "g.0"
        )
    message = str(caught.value)
    reading = where(plus_one, "y[i] = f[i] + 1.0")
    assert f"read by y[i] = f[i] + 1.0 ({reading}) at i=" in message
    assert "the read is in no loop on g.0" in message
    assert "zeros" not in message


def test_one_launch_of_the_hexahedra_passes_is_refused_for_its_reads(plain_opencl):
    # Every node of an element on a work item of its own, in one launch: the
    # pass along r reads a row the metric stored on other work items. That is
    # the refusal now, where the zeros of w on every work item used to be.
    hexahedra = example("hexahedra")
    grad = hexahedra.weak_grad
    calls = (grad.metric, grad.along_r, grad.along_s, grad.gradient_along_t)
    nodes = {"e": "g.0", "k": "l.2", "j": "l.1", "i": "l.0"}
    every = {
        getattr(call, loop): axis for call in calls for loop, axis in nodes.items()
    }
    with pytest.raises(IllegalCast) as caught:
        Schedule(grad, target="opencl").tag(every)
    message = str(caught.value)
    along_r = where(hexahedra.along_r, "v[q, e, k, j, i] = reduce_sum(")
    assert " writes w[" in message
    assert f" read by v[q, e, k, j, i] = reduce_sum(...) ({along_r}) at q=" in message
    assert (
        "the loops weak_grad.metric.i and weak_grad.along_r.i on l.0 run them on "
        "work items"
    ) in message
    assert "zeros" not in message
    # Each element's passes on one work item, in one launch, are legal, the
    # zeros of w, v and vs on the element's work group.
    tagged = Schedule(grad, target="opencl").tag({call.e: "g.0" for call in calls})
    assert set(casts(tagged)) == {"decided"}
    assert tagged.buildable == (True, "")


def test_sumpys_gpu_schedule_of_l2p_names_only_the_files_loops() -> None:
    # sumpy's own schedule for a GPU, each box a work group, as the file
    # writes it: no generated loop names, each box's cells of C and R zeroed
    # on its work item, and the fusions' facts those of fused, on C.
    l2p = example("l2p")
    schedule = l2p.on_a_gpu
    assert schedule.target == "opencl"
    assert repr(schedule).endswith(
        ".fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box)"
        ".tag(l2p.unpack.box, 'g.0')"
    )
    assert set(casts(schedule)) == {"decided"}
    assert schedule.buildable == (True, "")
    zeros = {
        stmt.id: stmt.inames[0] for stmt in schedule.term.stmts if ".zeros" in stmt.id
    }
    assert set(zeros) == {"C.zeros", "R.zeros"}
    assert {schedule.tags[loop] for loop in zeros.values()} == {"g.0"}
    fused = {fact.id for fact in l2p.fused.facts()}
    mine = [fact for fact in schedule.facts() if fact.id not in fused]
    assert [fact.statement for fact in mine] == [
        "tag(l2p.unpack.box, 'g.0') renames the instances of l2p one for one",
        "the order after tag(l2p.unpack.box, 'g.0') runs every dependence of l2p "
        "forward, within one work item",
    ]


# }}}
