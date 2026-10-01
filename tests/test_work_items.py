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
from lanky.prelude import Real

import hand_terms as ht
from loopty import Arr, Fin, kernel, reduce_sum, when
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
    with warnings.catch_warnings(), lp.CacheMode(False):
        warnings.simplefilter("ignore", lp.diagnostic.LoopyWarning)
        warnings.filterwarnings(
            "ignore", message="Cannot enforce the constraint", category=UserWarning
        )
        try:
            return lp.generate_code_v2(lp.tag_inames(kernel_, tags)).device_code()
        except Exception as exc:  # noqa: BLE001 - loopy raises anything
            return f"{type(exc).__name__}: {exc}"


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
        schedule.tag(i=axis)
    message = str(caught.value)
    assert message.startswith(f"tag(i={axis!r}) illegal: instance S0[t=")
    assert "writes u[" in message and "read by S0[" in message
    assert "on another work item (at nt=16, nx=16, as hinted)" in message
    assert f"the loop i on {axis} runs them on work items" in message
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
        schedule.tag(i=axis)
    message = str(caught.value)
    assert "writes velocity[" in message and "read by S1[" in message
    assert (
        f"on another work item (at nt=16, nx=32, as hinted): the loop i on {axis}"
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
        schedule.tag(i=axis, k=axis)
    message = str(caught.value)
    assert f"the loops i and k on {axis} run them on work items" in message
    (source_id, source), (sink_id, sink), _ = caught.value.witness
    assert (source_id, sink_id) == ("S0", "S1")
    assert source["i"] == 7 - sink["k"] != sink["k"]

    # #55's remedy for a statement off the axis, a loop of it on the axis as
    # well, leads here. With the axis on one loop only, the other statement
    # runs on every work item, and the cast says so before buildable is asked.
    said = loopy_says(plain_opencl, schedule.kernel, {"i": axis})
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(i=axis)
    assert f"S1 runs in no loop on {axis}, so on every work item of it" in str(
        caught.value
    )


def test_rows_whose_dependences_stay_in_their_row_are_unchanged(plain_opencl):
    # spmv's rows on a group axis: nothing one row computes is read by another.
    schedule = example("spmv").rows_parallel()
    assert casts(schedule) == ["decided"] * 2
    assert schedule.buildable == (True, "")
    (_, monotone) = schedule.facts()
    assert monotone.statement == (
        "the order after tag(r='g.0') runs every dependence of spmv forward, "
        "within one work item"
    )


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
    tagged = same.tag(**tags)
    assert casts(tagged) == ["decided"] * 2
    assert tagged.buildable == (True, "")

    next_one = Schedule(offset_loops_term(0), target="opencl", sizes={"n": 8})
    said = loopy_says(plain_opencl, next_one.kernel, tags)
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said
    with pytest.raises(IllegalCast) as caught:
        next_one.tag(**tags)
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
        schedule.tag(i="g.0")
    message = str(caught.value)
    assert message.startswith("tag(i='g.0') illegal: instance S0[] writes s[0]")
    assert "read by S1[i=" in message
    assert "S0 runs in no loop on g.0, so on every work item of it" in message


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
    tagged = Schedule(halved_columns, sizes=sizes).tag(i="l.0")
    assert casts(tagged) == ["decided"] * 2
    with pytest.raises(IllegalCast) as caught:
        tagged.skew("i", by="t")
    message = str(caught.value)
    assert message.startswith("skew(i, by='t') illegal: instance S0[t=")
    (_, source), (_, sink), _ = caught.value.witness
    assert sink["t"] == source["t"] + 1 and sink["i"] == source["i"]
    first, second = source["i"] + source["t"], sink["i"] + sink["t"]
    assert f"the loop i on l.0 runs them on work items {first} and {second}" in message


@pytest.mark.parametrize("tag", ["ilp", "vec"])
def test_a_loop_that_runs_in_one_work_item_loses_only_its_order(tag) -> None:
    # loopy runs ilp and vec around each instruction inside one work item, so
    # dropping them from the order is all there is to it (#57).
    jacobi = example("stencil_skew").jacobi
    tagged = Schedule(jacobi, sizes={"nt": 16, "nx": 16}).tag(i=tag)
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
    # A sum on a local axis happens inside one instance, and every work item
    # of the group runs the statement around it; loopy synchronizes the
    # partial sums through local memory, with barriers. An axis only sums are
    # on is not asked about.
    sums = {"j": "l.0", "k": "l.0"}
    for fn, rows in ((two_row_sums, {"i": "g.0"}), (row_sums_read_backwards, {})):
        schedule = Schedule(fn, target="opencl")
        tagged = schedule.tag(**rows, **sums)
        assert casts(tagged) == ["decided"] * 4
        assert tagged.buildable == (True, "")
        assert "barrier(CLK_LOCAL_MEM_FENCE)" in loopy_says(
            plain_opencl, schedule.kernel, {**rows, **sums}
        )


def test_rows_on_a_group_axis_do_not_share_their_local_sums(plain_opencl):
    # The sums are allowed; the second loop reading another row's sum, in
    # another group, is not.
    tags = {"i": "g.0", "p": "g.0", "j": "l.0", "k": "l.0"}
    schedule = Schedule(row_sums_read_backwards, target="opencl")
    said = loopy_says(plain_opencl, schedule.kernel, tags)
    assert said.startswith("MissingBarrierError") and GLOBAL_BARRIER in said, said
    with pytest.raises(IllegalCast, match="the loops i and p on g.0 run them"):
        schedule.tag(**tags)


# }}}
