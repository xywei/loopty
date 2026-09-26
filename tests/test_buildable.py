"""What the target can build, measured against loopy's own code generation.

A schedule's ``buildable`` fact is a claim about loopy: that code generation
for the schedule's kernel will not stop. Each case below is asked both ways,
of the schedule and of loopy itself, with loopy's caches off so that code it
generated once is generated again rather than found. The cases are the ones
the reviews of the wave C1 batch measured (#47, #48, #55) and the rest of the
matrix they came from: hardware axes on the C target, unrolled and vectorized
loops without a numeric length, loopy's rules for sharing and numbering
hardware axes, temporaries a vectorized loop cannot hold, and concurrent loops
in a domain whose extent is read out of an array.

The OpenCL cases are generated for loopy's plain OpenCL target, which stands in
for the pyopencl one (see the ``plain_opencl`` fixture), since code generation
is all that is asked. Note 11 of ``docs/loopy-notes.md`` has the same table in
words.
"""

from __future__ import annotations

import pytest
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, kernel
from loopty import sum as reduce_sum
from loopty.schedule import Schedule

pytest.importorskip("loopy")


# {{{ kernels


@kernel
def copy_n(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A copy over a size that is free."""
    for i in y.dom:
        y[i] = x[i]


@kernel
def copy8(x: Arr[Fin[8], Real], y: Arr[Fin[8], Real]):
    """A copy over a size known when the code is generated."""
    for i in y.dom:
        y[i] = x[i]


@kernel
def copy2_8(a: Arr[Fin[8], Fin[8], Real], b: Arr[Fin[8], Fin[8], Real]):
    """A two-level copy: one statement, two loops."""
    for i in b.dom:
        for j in b.dom[i]:
            b[i, j] = a[i, j]


@kernel
def total_n(a: Arr[Fin[n], Real], s: Arr[Fin[1], Real]):  # noqa: F821
    """A sum over a size that is free."""
    s[0] = reduce_sum(a[i] for i in a.dom)


@kernel
def rowsum8(a: Arr[Fin[8], Fin[8], Real], y: Arr[Fin[8], Real]):
    """Row sums: a statement loop around a sum."""
    for i in y.dom:
        y[i] = reduce_sum(a[i, j] for j in a.dom[i])


@kernel
def two_sums8(a: Arr[Fin[8], Real], b: Arr[Fin[8], Real], s: Arr[Fin[1], Real]):
    """Two sums side by side in one statement."""
    s[0] = reduce_sum(a[i] for i in a.dom) + reduce_sum(b[j] for j in b.dom)


@kernel
def lower_total8(a: Arr[Fin[8], Fin[8], Real], s: Arr[Fin[1], Real]):
    """A sum nested in a sum, over a triangle."""
    s[0] = reduce_sum(reduce_sum(a[i, j] for j in Fin[i + 1]) for i in a.dom)


@kernel
def triangle8(a: Arr[Fin[8], Fin[8], Real], y: Arr[Fin[8], Real]):
    """Each row summed up to the diagonal: at most 8 terms."""
    for i in y.dom:
        y[i] = reduce_sum(a[i, j] for j in Fin[i + 1])


@kernel
def two_loops8(x: Arr[Fin[8], Real], y: Arr[Fin[8], Real], z: Arr[Fin[8], Real]):
    """Two statements in two loops of their own."""
    for i in y.dom:
        y[i] = x[i]
    for k in z.dom:
        z[k] = x[k]


@kernel
def outer_and_inner8(
    a: Arr[Fin[8], Fin[8], Real],
    y: Arr[Fin[8], Real],
    b: Arr[Fin[8], Fin[8], Real],
):
    """A statement of the outer loop beside the inner loop."""
    for i in y.dom:
        y[i] = 1.0
        for j in b.dom[i]:
            b[i, j] = a[i, j]


@kernel
def ragged_copy(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    w: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
):
    """A copy over the fibers of a ragged array."""
    for r in w.dom:
        for j in val.dom[r]:
            w[r, j] = val[r, j]


@kernel
def between8(
    x: Arr[Fin[8], Real],
    cnt: Arr[Fin[8], Nat],
    val: Arr[Fin[8], Fin[cnt], Real],  # noqa: F821
    z: Arr[Fin[8], Fin[8], Real],
):
    """A dense loop between a ragged row and its fiber."""
    for r in z.dom:
        for i in x.dom:
            for j in val.dom[r]:
                z[r, i] = z[r, i] + x[i] * val[r, j]


@kernel
def row_sums_by_column8(
    x: Arr[Fin[8], Real],
    cnt: Arr[Fin[8], Nat],
    val: Arr[Fin[8], Fin[cnt], Real],  # noqa: F821
    z: Arr[Fin[8], Fin[8], Real],
):
    """A ragged sum inside a dense loop inside its row."""
    for r in z.dom:
        for i in x.dom:
            z[r, i] = x[i] * reduce_sum(val[r, j] for j in val.dom[r])


# }}}


def loopy_says(lp, schedule) -> str | None:
    """What loopy's code generation says to the schedule's kernel.

    ``None`` when it generates the code, and otherwise the exception's type
    and message. With loopy's caches off, so that code generated once for the
    same kernel is generated again.

    loopy's code generation for a ``vec`` loop warns through its own
    deprecated ``loopy.diagnostic.warn``, which this suite would turn into an
    error; it is loopy's, like the two in note 5 of ``docs/loopy-notes.md``,
    and is silenced here only.
    """
    import warnings

    with warnings.catch_warnings(), lp.CacheMode(False):
        warnings.simplefilter("ignore", lp.diagnostic.LoopyWarning)
        warnings.filterwarnings(
            "ignore",
            message="This function is deprecated and will go away",
            category=DeprecationWarning,
        )
        try:
            lp.generate_code_v2(schedule.kernel).device_code()
        except Exception as exc:  # noqa: BLE001 - loopy raises anything
            return f"{type(exc).__name__}: {exc}"
    return None


def scheduled(fn, target: str, steps) -> Schedule:
    """``Schedule(fn, target)`` with each ``(method, kwargs)`` step applied."""
    schedule = Schedule(fn, target=target)
    for method, args, kwargs in steps:
        schedule = getattr(schedule, method)(*args, **kwargs)
    return schedule


def tag(**tags):
    return ("tag", (), tags)


def split(iname, factor, **names):
    return ("split", (iname, factor), names)


#: ``(kernel, target, steps, part of loopty's reason, part of loopy's error)``,
#: with ``None`` for both when the schedule builds.
CASES = {
    # {{{ the C target has no hardware axes (#47)
    "c-local-axis": (
        copy8, "c", [tag(i="l.0")],
        "the tag i='l.0' puts a loop on a hardware axis, and the C target has none",
        "plain C does not have local hw axes",
    ),
    "c-group-axis": (
        copy8, "c", [tag(i="g.0")],
        "Retarget to opencl, or leave the loop sequential",
        "plain C does not have group hw axes",
    ),
    "c-local-reduction": (
        rowsum8, "c", [tag(j="l.0")],
        "the tag j='l.0' puts a loop on a hardware axis",
        "NotImplementedError",
    ),
    "c-vector-accumulator": (
        rowsum8, "c", [tag(i="vec")],
        "the accumulator of the sum in statement S0, as a vector along it; the "
        "C target has no vector types",
        "does not understand axis tag",
    ),
    "opencl-local-axis": (copy8, "opencl", [tag(i="l.0")], None, None),
    "opencl-vector-accumulator": (rowsum8, "opencl", [tag(i="vec")], None, None),
    "c-vector-copy": (copy8, "c", [tag(i="vec")], None, None),
    # }}}
    # {{{ an unrolled loop needs a numeric length (#48)
    "unr-symbolic": (
        copy_n, "opencl", [tag(i="unr")],
        "the extent of i is at most n, which no number bounds",
        "unbounded optimum",
    ),
    "unr-symbolic-c": (
        copy_n, "c", [tag(i="unr")],
        "loopy unrolls a loop tagged unr",
        "unbounded optimum",
    ),
    "ilp-symbolic": (
        copy_n, "opencl", [tag(i="ilp")],
        "loopy unrolls a loop tagged ilp",
        "unbounded optimum",
    ),
    "vec-symbolic": (
        copy_n, "opencl", [tag(i="vec")],
        "loopy vectorizes a loop tagged vec",
        "unbounded optimum",
    ),
    "unr-symbolic-reduction": (
        total_n, "opencl", [tag(i="unr")],
        "the loop i is tagged unr",
        "unbounded optimum",
    ),
    "unr-ragged-fiber": (
        ragged_copy, "opencl", [tag(j="unr")],
        "the extent of j is read out of an array (a ragged fiber)",
        "unbounded optimum",
    ),
    "unr-split": (
        copy_n, "opencl", [split("i", 4, inner="ii"), tag(ii="unr")], None, None
    ),
    "unr-fixed": (copy8, "c", [tag(i="unr")], None, None),
    "unr-triangle": (triangle8, "opencl", [tag(j="unr")], None, None),
    # }}}
    # {{{ loopy's numbering of hardware axes (#55)
    "local-axis-gap": (
        copy8, "opencl", [tag(i="l.1")],
        "the loop i is on the local axis l.1, and no loop of the kernel is on l.0",
        "local axis 0 unused",
    ),
    "group-axis-gap": (
        copy8, "opencl", [tag(i="g.1")],
        "no loop of the kernel is on g.0",
        "global axis 0 unused",
    ),
    "gap-across-statements": (
        two_loops8, "opencl", [tag(i="l.1", k="l.1")],
        "no loop of the kernel is on l.0",
        "local axis 0 unused",
    ),
    # }}}
    # {{{ one loop of an instruction per axis (#55)
    "two-loops-one-local-axis": (
        copy2_8, "opencl", [tag(i="l.0", j="l.0")],
        "statement S0 runs in two loops tagged l.0, i and j",
        "has multiple inames tagged 'l.0'",
    ),
    "two-loops-one-group-axis": (
        copy2_8, "opencl", [tag(i="g.0", j="g.0")],
        "statement S0 runs in two loops tagged g.0, i and j",
        "has multiple inames tagged 'g.0'",
    ),
    "two-loops-vectorized": (
        copy2_8, "opencl", [tag(i="vec", j="vec")],
        "loopy vectorizes one loop of an instruction at most",
        "has multiple inames tagged 'vec'",
    ),
    "a-statement-loop-and-its-sum-on-one-axis": (
        rowsum8, "opencl", [tag(i="l.0", j="l.0")],
        "statement S0 runs in two loops tagged l.0, i and j",
        "has multiple inames tagged 'l.0'",
    ),
    "two-sums-on-one-axis": (
        two_sums8, "opencl", [tag(i="l.0", j="l.0")],
        "statement S0 runs in two loops tagged l.0, i and j",
        "has multiple inames tagged 'l.0'",
    ),
    "a-statement-loop-and-its-sum-on-two-axes": (
        rowsum8, "opencl", [tag(i="g.0", j="l.0")], None, None
    ),
    # }}}
    # {{{ every instruction on every axis of the kernel
    "a-statement-off-the-group-axis": (
        two_loops8, "opencl", [tag(i="g.0")],
        "statement S1 runs in no loop on g.0, the group axis i is on",
        "does not use all group hw axes",
    ),
    "both-statements-on-the-group-axis": (
        two_loops8, "opencl", [tag(i="g.0", k="g.0")], None, None
    ),
    "a-sum-beside-a-local-sum": (
        two_sums8, "opencl", [tag(i="l.0")],
        "the sum over j in statement S0 runs in no loop on l.0",
        "does not use all local hw axes",
    ),
    "an-outer-statement-beside-a-local-loop": (
        outer_and_inner8, "opencl", [tag(j="l.0")],
        "statement S0 runs in no loop on l.0, the local axis j is on",
        "does not use all local hw axes",
    ),
    "a-row-length-off-the-local-axis": (
        row_sums_by_column8, "opencl", [tag(i="l.0")],
        "which reads the length of a ragged row, runs in no loop on l.0",
        "does not use all local hw axes",
    ),
    "an-outer-sum-on-a-local-axis": (
        lower_total8, "opencl", [tag(i="l.0")], None, None
    ),
    # }}}
    # {{{ an axis loopy is asked to choose
    "automatic-local-axis": (
        copy8, "opencl", [tag(i="l.auto")],
        "the tag l.auto on i asks loopy to choose a local axis",
        "automatically-assigned local axes",
    ),
    # }}}
    # {{{ temporaries a vectorized loop cannot hold
    "a-local-sum-in-a-vectorized-loop": (
        rowsum8, "opencl", [tag(i="vec", j="l.0")],
        "the sum over j in statement S0 runs on a local axis inside the loop i, "
        "which is tagged vec",
        "TypeError",
    ),
    "a-row-length-in-a-vectorized-loop": (
        between8, "opencl", [tag(r="vec")],
        "the length of a ragged row is read inside the loop r, which is tagged vec",
        "TypeError",
    ),
    # }}}
    # {{{ a concurrent loop in a domain whose extent is read out of an array
    "a-vectorized-ragged-fiber": (
        ragged_copy, "opencl", [tag(j="vec")],
        "the tag j='vec' makes a loop inside a ragged fiber concurrent",
        "data-dependent parameter",
    ),
    "a-local-loop-beside-a-ragged-fiber": (
        between8, "opencl", [tag(i="l.0")],
        "loopy defines i in one domain with the length of a ragged row, beside "
        "the ragged fiber j",
        "data-dependent parameter",
    ),
    # }}}
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_buildable_says_what_loopy_does(plain_opencl, case) -> None:
    fn, target, steps, reason, error = CASES[case]
    schedule = scheduled(fn, target, steps)
    ok, said = schedule.buildable
    loopy = loopy_says(plain_opencl, schedule)
    if reason is None:
        assert (ok, said) == (True, ""), said
        assert loopy is None, loopy
        return
    assert not ok
    assert reason in said, said
    assert loopy is not None, "loopy generated the code"
    assert error in loopy, loopy
    (fact,) = [fact for fact in schedule.facts() if fact.kind == "buildable"]
    assert fact.status.value == "refuted"
    assert fact.provenance["reason"] == said


# {{{ what each refusal is for


def test_retargeting_a_device_schedule_to_c_refuses_its_axes(plain_opencl) -> None:
    # ``loopty run --target c`` on a file written for a device replays its
    # steps against C, and used to come out buildable.
    device = Schedule(copy8, target="opencl").tag(i="g.0")
    assert device.buildable == (True, "")
    on_c = device.retarget("c")
    ok, reason = on_c.buildable
    assert not ok
    assert "the C target has none" in reason


def test_the_c_target_is_named_only_when_nothing_else_stands_in_the_way() -> None:
    # A limit that holds on every target comes first: retargeting would only
    # trade the one refusal for the next. The design's spmv device schedule
    # on C keeps its ragged-fiber reason, and carries one buildable fact.
    import hand_terms as ht

    rows = Schedule(ht.spmv_term()).tag(r="g.0")
    assert "the C target has none" in rows.buildable[1]
    device = rows.split("j", 32, inner="j_in", outer="j_out").tag(j_in="l.0")
    assert "ragged fiber" in device.buildable[1]
    (fact,) = [fact for fact in device.facts() if fact.kind == "buildable"]
    assert fact.id.endswith(".tag(j_in='l.0'):buildable")


def test_ilp_on_a_nested_sum_is_refused_as_ilp_not_as_a_hardware_axis() -> None:
    # The check for a hardware axis on a nested reduction counted ilp as one.
    # loopy unrolls an ilp loop; what stands in the way is the privatization
    # that makes the build depend on the string hash seed (note 11).
    ok, reason = Schedule(lower_total8).tag(j="ilp").buildable
    assert not ok
    assert "hardware axis" not in reason
    assert "on an ilp axis" in reason and "Tag it unr instead" in reason
    assert Schedule(lower_total8).tag(j="unr").buildable == (True, "")


def test_a_later_step_puts_an_unrolled_loop_right_by_splitting_it() -> None:
    # The question is asked of the schedule as it stands: tagging the inner
    # half after a split is the remedy the reason names.
    whole = Schedule(copy_n).tag(i="unr")
    assert not whole.buildable[0]
    assert "split('i', 4)" in whole.buildable[1]
    halves = Schedule(copy_n).split("i", 4, inner="ii").tag(ii="unr")
    assert halves.buildable == (True, "")
    assert not [fact for fact in halves.facts() if fact.kind == "buildable"]


# }}}
