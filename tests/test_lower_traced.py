"""Lowering and scheduling terms that came from the tracer, not from hand.

The rest of this wave's tests build terms by hand so that the two halves of the
compiler can be developed apart. This one joins them: a decorated kernel is
traced, the term is lowered, the compiled code is compared with the kernel's own
Python body, and the stencil is put through the rejection and the fix. If the
tracer is not landed yet, the whole module skips rather than failing, because the
two waves are written at the same time.
"""

from __future__ import annotations

import islpy as isl
import numpy as np
import pytest
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, kernel, when
from loopty import sum as reduce_sum

pytest.importorskip("loopy")

kernels = pytest.importorskip("kernels")


def traced(name: str, module: str):
    """A decorated kernel from ``tests/kernels``, or a skip with the reason."""
    import importlib

    try:
        loaded = importlib.import_module(f"kernels.{module}")
        kernel = getattr(loaded, name)
        kernel.trace()
    except Exception as exc:  # pragma: no cover - depends on the tracer's state
        pytest.skip(f"tracing {name} is not available yet: {type(exc).__name__}: {exc}")
    return kernel


def run(obj, **arguments):
    from loopty.executor import LoopyExecutor

    try:
        return LoopyExecutor().run(obj, **arguments)
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        if "compil" in str(exc).lower() or isinstance(exc, OSError):
            pytest.skip(f"the C toolchain path is unusable here: {exc}")
        raise


def test_a_traced_dense_kernel_lowers_runs_and_agrees() -> None:
    from loopty.arr import Arr
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule

    axpy = traced("axpy", "axpy")
    # The kernel's Python body iterates ``y.dom``, so the reference run is given
    # runtime arrays; the compiled run unwraps them to numpy itself.
    arrays = {
        "a": 2.0,
        "x": Arr.from_numpy(np.arange(4, dtype=np.float64)),
        "y": Arr.zeros(4),
    }
    schedule = Schedule(axpy)
    fact = LoopyExecutor().differential(axpy, schedule, arrays)
    assert fact.status.value == "tested", fact.provenance


def test_a_traced_ragged_kernel_computes_the_sparse_product() -> None:
    spmv = traced("spmv", "spmv_min")
    module = __import__("kernels.spmv_min", fromlist=["example"])
    cnt, col, val, x, y, _off = module.example()

    out = run(spmv.trace(), cnt=cnt, col=col, val=val, x=x, y=y)

    want = np.zeros(3)
    offsets = val.offsets
    flat_val, flat_col, x_values = val.numpy(), col.numpy(), x.numpy()
    for row in range(3):
        for a in range(offsets[row], offsets[row + 1]):
            want[row] += flat_val[a] * x_values[flat_col[a]]
    assert np.allclose(out["y"], want)


def test_a_traced_reduction_is_split_tagged_and_reassociated() -> None:
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule, UnbuildableSchedule

    spmv = traced("spmv", "spmv_min")
    schedule = (
        Schedule(spmv, target="c")
        .split("j", 2, inner="j_in", outer="j_out")
        .tag(j_in="l.0")
        .realize("y", tree=True)
    )
    assert schedule.reassociated == frozenset({"y"})
    # Every cast is legal: the instances are untouched and nothing is reordered.
    casts = [fact for fact in schedule.facts() if fact.kind != "buildable"]
    assert all(fact.status.value == "decided" for fact in casts)

    # And yet loopy cannot generate it: ``j_in`` is a hardware axis inside a
    # ragged fiber. The schedule says so instead of letting code generation
    # throw, and refuses when something asks it for code.
    ok, reason = schedule.buildable
    assert not ok
    assert "ragged fiber" in reason
    buildable = [fact for fact in schedule.facts() if fact.kind == "buildable"]
    assert [fact.status.value for fact in buildable] == ["refuted"]
    assert buildable[0].decided_by == "loopy-target"
    with pytest.raises(UnbuildableSchedule, match="ragged fiber"):
        LoopyExecutor().run(schedule)


def test_the_traced_stencil_refuses_a_tiling_and_accepts_a_skewed_one() -> None:
    from loopty.schedule import IllegalCast, Schedule

    jacobi = traced("jacobi", "stencil")
    schedule = Schedule(jacobi, sizes={"nt": 16, "nx": 16})

    with pytest.raises(IllegalCast) as caught:
        schedule.tile("t", "i", 8, 8)
    assert "scheduled earlier" in str(caught.value)

    tiled = schedule.skew("i", by="t").tile("t", "i", 8, 8)

    u = np.zeros((6, 6))
    u[0] = np.arange(6.0)
    want = u.copy()
    for t in range(5):
        for i in range(1, 5):
            want[t + 1, i] = (want[t, i - 1] + want[t, i + 1]) / 2
    out = run(tiled, u=u.copy())
    assert np.allclose(out["u"], want)


# {{{ two reductions that reuse one binder name


@kernel
def two_sums(
    a: Arr[Fin[2], Real],  # noqa: F821
    b: Arr[Fin[4], Real],  # noqa: F821
    y: Arr[Fin[2], Real],  # noqa: F821
):
    """Two reductions, both written over ``j``, over rows of different length."""
    y[0] = reduce_sum(a[j] for j in a.dom)
    y[1] = reduce_sum(b[j] for j in b.dom)


def test_two_reductions_over_one_binder_keep_their_own_domains() -> None:
    # loopy gives an iname one domain, and lowering used to hand both
    # reductions the union of theirs: the sum over ``a`` became a sum over four
    # points, reading two cells past the end of a two-cell array. A statement in
    # that position gets its own domain back as a predicate, which a reduction
    # cannot carry, so the second binder becomes an iname of its own instead.
    from loopty.lower import lower_generic

    term = two_sums.trace()
    lowering = lower_generic(term, "c")
    inames = set(lowering.kernel.default_entrypoint.inames)
    assert "j" in inames, inames
    assert len(inames) == 2, inames
    extents = {
        tuple(domain.get_var_names(isl.dim_type.set)): domain.to_set()
        .count_val()
        .to_python()
        for domain in lowering.kernel.default_entrypoint.domains
    }
    assert extents[("j",)] == 2, extents
    assert 4 in extents.values(), extents

    a = np.array([1.0, 2.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    out = run(term, a=a, b=b, y=np.zeros(2))
    assert np.allclose(out["y"], [a.sum(), b.sum()])


def test_a_reduction_whose_binder_is_unambiguous_keeps_the_name() -> None:
    # Renaming unconditionally would rename the inames the demos and
    # ``Schedule.split("j", ...)`` refer to. A binder is renamed only when the
    # same name is already bound to a different domain.
    spmv = traced("spmv", "spmv_min")
    lowering = __import__(
        "loopty.lower", fromlist=["lower_generic"]
    ).lower_generic(spmv.trace(), "c")
    assert "j" in lowering.kernel.default_entrypoint.inames


# }}}


# {{{ a size spelled like the parameter a ragged bound reflects to


@kernel
def shadowed_bound(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    w: Arr[Fin[nl_cnt_r], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A ragged row sum, in a kernel whose vector size is called ``nl_cnt_r``."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * w[0] for j in val.dom[r])


def test_a_shadowed_reflected_bound_still_lowers_as_a_ragged_bound() -> None:
    # The bound ``cnt[r]`` cannot be called ``nl_cnt_r`` here, because a size
    # already is. Lowering has to follow the name the trace actually allocated:
    # recognizing ragged bounds by their spelling alone would declare this one
    # as a size argument nobody passes.
    from loopty.arr import Arr as RuntimeArr

    term = shadowed_bound.trace()
    (name,) = [symbol for symbol, _ in term.reflected]
    assert name != "nl_cnt_r"

    lowering = __import__(
        "loopty.lower", fromlist=["lower_generic"]
    ).lower_generic(term, "c")
    assert name not in lowering.value_args, lowering.value_args
    assert lowering.ragged == {"val": "off_cnt"}

    counts = [2, 0, 3]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = run(
        term,
        cnt=RuntimeArr.from_numpy(np.array(counts, dtype=np.int64)),
        val=RuntimeArr.ragged(counts, values=values),
        w=np.array([3.0, 0.0]),
        y=np.zeros(3),
    )
    assert np.allclose(out["y"], [3.0 * 3.0, 0.0, 3.0 * 12.0])


# }}}


# {{{ a kernel that writes the offsets it reads through


@kernel
def scan_then_row_sums(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A scan of the counts, then row sums of ``val``, whose flat index is ``off``."""
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + val[r, j]


@kernel
def row_sums_then_scan(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The same two loops the other way round."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + val[r, j]
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@pytest.mark.parametrize(
    "which", [scan_then_row_sums, row_sums_then_scan], ids=lambda k: k.__name__
)
def test_reading_through_offsets_is_ordered_against_writing_them(which) -> None:
    # ``val[r, j]`` is ``val[off[r] + j]`` once lowered, a read of ``off`` that
    # belongs to the layout and not to the body. loopy's single-writer
    # heuristic used to order it after the scan wherever the scan was, so the
    # second kernel read its rows after rewriting their offsets. With every
    # instruction's dependences final the edge has to come from the access
    # collector, which lists that read, or loopy refuses the kernel with
    # VariableAccessNotOrdered; it now follows the body in both kernels.
    from loopty.arr import Arr as RuntimeArr
    from loopty.lower import lower_generic

    term = which.trace()
    lowering = lower_generic(term, "c")
    assert lowering.ragged == {"val": "off"}
    insns = {insn.id: insn for insn in lowering.kernel.default_entrypoint.instructions}
    assert "S0" in insns["S1"].depends_on
    assert "S1" not in insns["S0"].depends_on

    counts = [2, 0, 3, 1]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    out = run(
        term,
        cnt=RuntimeArr.from_numpy(np.array(counts, dtype=np.int64)),
        off=RuntimeArr.from_numpy(np.array([0, 2, 2, 5, 6], dtype=np.int64)),
        val=RuntimeArr.ragged(counts, values=values),
        y=np.zeros(4),
    )
    assert np.allclose(out["y"], [3.0, 0.0, 12.0, 6.0])
    assert np.array_equal(out["off"], [0, 2, 2, 5, 6])


# }}}


# {{{ a reduction nested in another one


@kernel
def nested_total(a: Arr[Fin[n], Fin[m], Real], s: Arr[Fin[1], Real]):  # noqa: F821
    """A double sum over a dense matrix."""
    s[0] = reduce_sum(reduce_sum(a[i, j] for j in a.dom[i]) for i in a.dom)


@kernel
def lower_total(a: Arr[Fin[n], Fin[n], Real], s: Arr[Fin[1], Real]):  # noqa: F821
    """A double sum over the lower triangle, whose inner bound is the outer binder."""
    s[0] = reduce_sum(reduce_sum(a[i, j] for j in Fin[i + 1]) for i in a.dom)


@kernel
def two_totals(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    b: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    s: Arr[Fin[2], Real],
):
    """Two double sums whose inner binders share a name under different outer ones."""
    s[0] = reduce_sum(reduce_sum(a[i, j] for j in a.dom[i]) for i in a.dom)
    s[1] = reduce_sum(reduce_sum(b[k, j] for j in b.dom[k]) for k in b.dom)


def test_a_nested_reduction_still_lowers_and_runs() -> None:
    # The inner domain now names the outer binder and its bound as parameters.
    # The lowering already treats a reduction domain's extra names that way, so
    # the dense and the triangular double sum compute what they did before.
    a = np.arange(12.0).reshape(3, 4)
    out = run(nested_total.trace(), a=a, s=np.zeros(1))
    assert np.allclose(out["s"], [a.sum()])
    t = np.arange(9.0).reshape(3, 3)
    out = run(lower_total.trace(), a=t, s=np.zeros(1))
    assert np.allclose(out["s"], [np.tril(t).sum()])


def test_inner_binders_under_different_outer_binders_get_their_own_inames() -> None:
    # Without the outer binder the two inner domains were the same set, so both
    # inner sums shared the iname ``j`` while nested in ``i`` and in ``k``, and
    # loopy found no loop nest to schedule. With it they differ, and the second
    # is given an iname of its own as any two different reduction domains are.
    a = np.arange(12.0).reshape(3, 4)
    out = run(two_totals.trace(), a=a, b=2.0 * a, s=np.zeros(2))
    assert np.allclose(out["s"], [a.sum(), 2.0 * a.sum()])


@kernel
def two_totals_one_spelling(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    b: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    s: Arr[Fin[2], Real],
):
    """Two double sums that bind ``i`` and ``j`` at the same two levels."""
    s[0] = reduce_sum(reduce_sum(a[i, j] for j in a.dom[i]) for i in a.dom)
    s[1] = reduce_sum(reduce_sum(b[i, j] for j in b.dom[i]) for i in b.dom)


@kernel
def sum_then_loop(
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    s: Arr[Fin[1], Real],
):
    """A sum over ``j``, then a loop over ``j`` that reads the sum."""
    s[0] = reduce_sum(x[j] for j in x.dom)
    for j in x.dom:
        y[j] = x[j] + s[0]


@kernel
def pair_and_single(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    b: Arr[Fin[m], Real],  # noqa: F821
    s: Arr[Fin[1], Real],
):
    """One statement with a sum over ``i, j`` and a sum over ``j`` alone."""
    s[0] = reduce_sum(a[i, j] for i in a.dom for j in a.dom[i]) + reduce_sum(
        b[j] for j in b.dom
    )


def test_two_statements_with_the_same_nested_binders_lower_and_run() -> None:
    # loopy realizes a reduction as a loop inside its instruction, and two
    # instructions that reduce over one iname share that loop. ``s[1]`` is
    # ordered after ``s[0]``, so its sums had to run inside loops that must
    # finish before it starts, and loopy stopped with a CycleError. The second
    # statement's binders get inames of their own, and the inner domain names
    # the renamed outer binder, not the first statement's ``i``.
    from loopty.lower import lower_generic

    a = np.arange(12.0).reshape(3, 4)
    out = run(two_totals_one_spelling.trace(), a=a, b=2.0 * a, s=np.zeros(2))
    assert np.allclose(out["s"], [a.sum(), 2.0 * a.sum()])

    lowering = lower_generic(two_totals_one_spelling.trace(), "c")
    assert lowering.reduction_inames == {
        "S0:0": ("i",),
        "S0:1": ("j",),
        "S1:0": ("i_0",),
        "S1:1": ("j_0",),
    }
    params = {
        tuple(domain.get_var_names(isl.dim_type.set)): set(
            domain.get_var_names(isl.dim_type.param)
        )
        for domain in lowering.kernel.default_entrypoint.domains
    }
    assert "i_0" in params[("j_0",)] and "i" not in params[("j_0",)], params


def test_a_sum_and_a_later_loop_over_the_same_name_lower_and_run() -> None:
    # The same collision between a reduction binder and another statement's
    # loop variable: the loop over ``j`` reads the sum, and the sum's own loop
    # was that loop.
    x = np.arange(4.0)
    out = run(sum_then_loop.trace(), x=x, y=np.zeros(4), s=np.zeros(1))
    assert np.allclose(out["y"], x + x.sum())


def test_a_binder_pair_and_a_single_binder_in_one_statement_lower_and_run() -> None:
    # ``j`` bound as the second of a pair and then alone used to be kept for
    # both, and loopy refused the second domain for redefining ``j``.
    a = np.arange(12.0).reshape(3, 4)
    b = np.arange(4.0)
    out = run(pair_and_single.trace(), a=a, b=b, s=np.zeros(1))
    assert np.allclose(out["s"], [a.sum() + b.sum()])


@kernel
def approx_then_exact(
    a: Arr[Fin[n], Real],  # noqa: F821
    c: Arr[Fin[n], Nat],  # noqa: F821
    s: Arr[Fin[1], Real],
    t: Arr[Fin[1], Nat],
):
    """Two sums over ``j``, the first ``approx`` and the second ``exact``."""
    s[0] = reduce_sum(a[j] for j in a.dom)
    t[0] = reduce_sum(c[j] for j in c.dom)


def test_a_schedule_names_a_renamed_reduction_by_its_own_iname() -> None:
    # The second sum is ``j_0`` in the kernel, and a schedule step reaches it
    # under that name, with its own exactness: parallelizing it would
    # reassociate an exact accumulation. ``j`` is the first sum's alone.
    from loopty.schedule import IllegalCast, Schedule

    schedule = Schedule(approx_then_exact)
    with pytest.raises(IllegalCast, match="exact"):
        schedule.split("j_0", 2, inner="k_in", outer="k_out").tag(k_in="l.0")
    tagged = schedule.split("j", 2, inner="j_in", outer="j_out").tag(j_in="l.0")
    assert tagged.reassociated == frozenset({"s"})


@kernel
def ragged_row_sums_twice(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """Two ragged row sums over ``j``, in two loops one after the other."""
    for q in val.dom:
        y[q] = reduce_sum(val[q, j] for j in val.dom[q])
    for p in val.dom:
        z[p] = reduce_sum(val[p, j] * val[p, j] for j in val.dom[p])


def test_a_renamed_ragged_reduction_nests_under_its_own_row_loop() -> None:
    # The second sum is ``j_0``, and its row length is computed in the loop
    # over ``p``; its domain has to follow that loop's, as the first one's
    # follows ``q``'s, or loopy fixes the nesting up through a call islpy
    # deprecates.
    from loopty.lower import lower_generic

    counts = [2, 0, 3]
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = run(
        ragged_row_sums_twice.trace(),
        cnt=np.array(counts),
        val=Arr.ragged(counts, values=values),
        y=np.zeros(3),
        z=np.zeros(3),
    )
    assert np.allclose(out["y"], [3.0, 0.0, 12.0])
    assert np.allclose(out["z"], [5.0, 0.0, 50.0])
    names = [
        tuple(domain.get_var_names(isl.dim_type.set))
        for domain in lower_generic(
            ragged_row_sums_twice.trace(), "c"
        ).kernel.default_entrypoint.domains
    ]
    assert names.index(("j",)) == names.index(("q",)) + 1, names
    assert names.index(("j_0",)) == names.index(("p",)) + 1, names


def test_a_renamed_ragged_reduction_is_still_a_ragged_fiber() -> None:
    # The second row sum is ``j_0`` in the kernel, and its bound is a row length
    # just as the first one's is. A hardware axis on it is the ragged-fiber case
    # under the name the step used; if the renamed sum were looked up under its
    # written name, ``j_0`` would pass for a loop with a known extent.
    from loopty.lower import lower_generic
    from loopty.schedule import Schedule, data_dependent_inames

    term = ragged_row_sums_twice.trace()
    renamed = lower_generic(term, "c").reduction_inames
    assert {"j", "j_0"} <= data_dependent_inames(term, renamed)

    schedule = (
        Schedule(ragged_row_sums_twice)
        .split("j_0", 2, inner="k_in", outer="k_out")
        .tag(k_in="l.0")
    )
    ok, reason = schedule.buildable
    assert not ok
    assert "ragged fiber" in reason and "k_in" in reason


@kernel
def ragged_total(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    s: Arr[Fin[2], Real],
):
    """A double sum whose inner bound is the row length the outer binder selects.

    The second statement reads ``cnt``, which the first one does only through
    the row length it cannot compute; without it the lowering would refuse
    ``cnt`` as a parameter the body never touches.
    """
    s[0] = reduce_sum(reduce_sum(val[q, j] for j in val.dom[q]) for q in val.dom)
    s[1] = reduce_sum(cnt[r] for r in cnt.dom)


@kernel
def ragged_total_by_rows(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    s: Arr[Fin[1], Real],
):
    """The same sum with the outer reduction written as an accumulation loop."""
    for q in val.dom:
        s[0] += reduce_sum(val[q, j] for j in val.dom[q])


@kernel
def ragged_total_by_cells(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    rows: Arr[Fin[n], Real],  # noqa: F821
    s: Arr[Fin[1], Real],
):
    """A ragged double sum with each row's sum kept in a cell indexed by the row."""
    for q in val.dom:
        rows[q] = reduce_sum(val[q, j] for j in val.dom[q])
    s[0] = reduce_sum(rows[p] for p in rows.dom)


def test_a_ragged_bound_over_an_outer_reduction_binder_is_refused_by_name() -> None:
    # The analysis decides this term (``val[q, j]`` is in bounds). The lowering
    # cannot build it: ``cnt[q]`` is computed inside the loop over ``q``, and a
    # reduction binder has no loop another instruction can run in. It used to
    # get as far as the run and fail there with loopy's "value argument
    # 'nl_cnt_q' was not given".
    from loopty.lower import LoweringError

    counts = [2, 0, 3]
    with pytest.raises(LoweringError, match=r"bounded by cnt\[q\].*binder of the"):
        run(
            ragged_total.trace(),
            cnt=np.array(counts),
            val=Arr.ragged(counts, values=[1.0, 2.0, 3.0, 4.0, 5.0]),
            s=np.zeros(2),
        )


def test_the_nestings_the_refusal_suggests_lower_and_run() -> None:
    counts = [2, 0, 3]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = run(
        ragged_total_by_rows.trace(),
        cnt=np.array(counts),
        val=Arr.ragged(counts, values=values),
        s=np.zeros(1),
    )
    assert np.allclose(out["s"], [15.0])
    out = run(
        ragged_total_by_cells.trace(),
        cnt=np.array(counts),
        val=Arr.ragged(counts, values=values),
        rows=np.zeros(3),
        s=np.zeros(1),
    )
    assert np.allclose(out["rows"], [3.0, 0.0, 12.0])
    assert np.allclose(out["s"], [15.0])


def test_a_ragged_domain_follows_the_loop_it_is_nested_in() -> None:
    # loopy reads the nesting of domains off their order. The row-length domain
    # of ``j`` used to come after the domain of the second loop, ``p``, so
    # loopy made it a root and fixed the loop up through a call islpy
    # deprecates, and the run failed on that DeprecationWarning.
    from loopty.lower import lower_generic

    counts = [2, 0, 3]
    out = run(
        ragged_total_by_cells.trace(),
        cnt=np.array(counts),
        val=Arr.ragged(counts, values=[1.0, 2.0, 3.0, 4.0, 5.0]),
        rows=np.zeros(3),
        s=np.zeros(1),
    )
    assert np.allclose(out["s"], [15.0])
    lowered = lower_generic(ragged_total_by_cells.trace(), "c").kernel
    names = [
        tuple(domain.get_var_names(isl.dim_type.set))
        for domain in lowered.default_entrypoint.domains
    ]
    assert names.index(("j",)) == names.index(("q",)) + 1


# }}}


# {{{ statements at two depths of one loop


@kernel
def after_inner(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """A statement inside the loop over ``j``, and one after it, in ``r``'s."""
    for r in y.dom:
        for j in a.dom[r]:
            y[r] = y[r] + a[r, j]
        z[r] = 1.0


@kernel
def before_inner(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """The same two statements the other way round."""
    for r in y.dom:
        z[r] = 1.0
        for j in a.dom[r]:
            y[r] = y[r] + a[r, j]


@kernel
def guarded_inner(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """The statement inside the inner loop skips the first row; the other not."""
    for r in y.dom:
        for j in a.dom[r]:
            with when(r > 0):
                y[r] = y[r] + a[r, j]
        z[r] = 1.0


@kernel
def side_by_side(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    b: Arr[Fin[n], Fin[p], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """Two inner loops of different extents in one outer loop, nothing between."""
    for r in y.dom:
        for j in a.dom[r]:
            y[r] = y[r] + a[r, j]
        for k in b.dom[r]:
            z[r] = z[r] + b[r, k]


@kernel
def three_depths(
    a: Arr[Fin[n], Fin[m], Fin[p], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Fin[m], Real],  # noqa: F821
):
    """A statement at each depth of a triple nest."""
    for r in y.dom:
        y[r] = 0.0
        for j in z.dom[r]:
            z[r, j] = 0.0
            for k in a.dom[r, j]:
                z[r, j] = z[r, j] + a[r, j, k]
            y[r] = y[r] + z[r, j]


@kernel
def rows_and_lengths(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
    c: Arr[Fin[n], Nat],  # noqa: F821
):
    """A ragged and a dense inner loop in one row loop, and the row's length."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + val[r, j]
        for k in a.dom[r]:
            z[r] = z[r] + a[r, k]
        c[r] = cnt[r]


def native(fn, **arguments):
    """The body's own run, on copies, as runtime arrays; the outputs as numpy."""
    copies = {
        name: Arr.from_numpy(np.array(value))
        if isinstance(value, np.ndarray)
        else value
        for name, value in arguments.items()
    }
    fn(**copies)
    return {
        name: value.numpy() if isinstance(value, Arr) else value
        for name, value in copies.items()
    }


def test_statements_at_two_depths_of_one_loop_lower_and_run() -> None:
    # loopy defined ``r`` twice, once in the domain of the statement inside the
    # loop over ``j`` and once in the other's, and refused the second with a
    # bare RuntimeError that the executor and ``Schedule`` passed on.
    a = np.arange(12.0).reshape(3, 4)
    for fn in (after_inner, before_inner):
        arguments = {"a": a, "y": np.zeros(3), "z": np.zeros(3)}
        want = native(fn, **arguments)
        out = run(fn.trace(), **arguments)
        assert np.allclose(out["y"], a.sum(axis=1)), fn.__name__
        assert np.array_equal(out["z"], np.ones(3)), fn.__name__
        assert np.allclose(out["y"], want["y"]), fn.__name__


def test_a_guard_inside_the_inner_loop_narrows_only_its_statement() -> None:
    # The outer stretch of the guarded statement is ``1 <= r < n``, the other
    # statement's is ``0 <= r < n``, and the loop runs over the second with the
    # first put back as a predicate on the guarded statement alone.
    a = np.arange(12.0).reshape(3, 4)
    arguments = {"a": a, "y": np.zeros(3), "z": np.zeros(3)}
    want = native(guarded_inner, **arguments)
    out = run(guarded_inner.trace(), **arguments)
    assert np.allclose(out["y"], [0.0, *a.sum(axis=1)[1:]])
    assert np.array_equal(out["z"], np.ones(3))
    assert np.allclose(out["y"], want["y"])


def test_statements_at_two_depths_agree_with_the_body_under_a_schedule() -> None:
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule

    square = Arr.from_numpy(np.arange(12.0).reshape(3, 4))
    for fn in (after_inner, before_inner):
        arrays = {"a": square, "y": Arr.zeros(3), "z": Arr.zeros(3)}
        schedule = Schedule(fn).split("r", 2)
        fact = LoopyExecutor().differential(fn, schedule, arrays)
        assert fact.status.value == "tested", (fn.__name__, fact.provenance)


def test_the_inner_loop_is_its_own_domain_nested_in_the_outer_one() -> None:
    from loopty.lower import lower_generic

    domains = lower_generic(after_inner.trace(), "c").kernel.default_entrypoint.domains
    shapes = {
        tuple(domain.get_var_names(isl.dim_type.set)): set(
            domain.get_var_names(isl.dim_type.param)
        )
        for domain in domains
    }
    assert set(shapes) == {("r",), ("j",)}
    assert "r" in shapes[("j",)]
    # The loop over ``r`` is the rows: its domain keeps nothing of ``j``'s,
    # ``m >= 1`` included.
    (outer,) = [
        domain
        for domain in domains
        if domain.get_var_names(isl.dim_type.set) == ["r"]
    ]
    outer = outer.to_set() if isinstance(outer, isl.BasicSet) else outer
    rows = isl.Set("[n] -> { [r] : 0 <= r < n }").align_params(outer.get_space())
    assert outer.align_params(rows.get_space()).is_equal(rows), outer


def test_two_inner_loops_of_different_extents_lower_and_run() -> None:
    # Projected out of their statements' domains, the two loops over ``r`` were
    # ``m >= 1`` and ``p >= 1`` apart, a union loopy cannot take as one loop.
    a = np.arange(12.0).reshape(3, 4)
    b = np.arange(6.0).reshape(3, 2)
    out = run(side_by_side.trace(), a=a, b=b, y=np.zeros(3), z=np.zeros(3))
    assert np.allclose(out["y"], a.sum(axis=1))
    assert np.allclose(out["z"], b.sum(axis=1))


def test_a_statement_at_every_depth_of_a_triple_nest_lowers_and_runs() -> None:
    a = np.arange(3.0 * 4 * 2).reshape(3, 4, 2)
    arguments = {"a": a, "y": np.full(3, 7.0), "z": np.full((3, 4), 7.0)}
    want = native(three_depths, **arguments)
    out = run(three_depths.trace(), **arguments)
    assert np.allclose(out["z"], a.sum(axis=2))
    assert np.allclose(out["y"], a.sum(axis=(1, 2)))
    assert np.allclose(out["z"], want["z"])
    assert np.allclose(out["y"], want["y"])


def test_a_ragged_and_a_dense_inner_loop_share_their_row_loop() -> None:
    counts = [2, 0, 3]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = np.arange(6.0).reshape(3, 2)
    out = run(
        rows_and_lengths.trace(),
        cnt=np.array(counts),
        val=Arr.ragged(counts, values=values),
        a=a,
        y=np.zeros(3),
        z=np.zeros(3),
        c=np.zeros(3, dtype=np.int64),
    )
    assert np.allclose(out["y"], [3.0, 0.0, 12.0])
    assert np.allclose(out["z"], a.sum(axis=1))
    assert list(out["c"]) == counts


@kernel
def total_after_inner(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """A statement after the inner loop that reads what the loop accumulates."""
    for r in y.dom:
        for j in a.dom[r]:
            y[r] = y[r] + a[r, j]
        z[r] = z[r] + y[r]


@kernel
def copy_before_inner(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """A statement before the inner loop that reads what the loop then updates."""
    for r in y.dom:
        z[r] = y[r]
        for j in a.dom[r]:
            y[r] = y[r] + a[r, j]


@kernel
def total_in_a_later_loop(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """The same total, read by a loop of its own after the nest."""
    for r in y.dom:
        for j in a.dom[r]:
            y[r] = y[r] + a[r, j]
    for q in z.dom:
        z[q] = z[q] + y[q]


@kernel
def total_after_ragged(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """A statement after a ragged inner loop that reads what the loop sums."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + val[r, j]
        z[r] = z[r] + y[r]


def test_a_statement_that_reads_the_inner_loop_stays_outside_it() -> None:
    # loopy adds to an instruction whose loops are not final the loops of the
    # instructions that write what it reads, less those the writer's subscripts
    # name: the statement reading ``y[r]`` went into the loop over ``j``. It ran
    # once per ``j``, which gave ``z`` the sum of the partial sums, and the
    # copy before the loop saw ``y[r]`` after all but the last update.
    a = np.arange(12.0).reshape(3, 4)
    totals = 1.0 + a.sum(axis=1)
    for fn, z in (
        (total_after_inner, 10.0 + totals),
        (copy_before_inner, np.ones(3)),
    ):
        arguments = {"a": a, "y": np.ones(3), "z": np.full(3, 10.0)}
        want = native(fn, **arguments)
        out = run(fn.trace(), **arguments)
        assert np.allclose(out["y"], totals), fn.__name__
        assert np.allclose(out["z"], z), (fn.__name__, out["z"])
        assert np.allclose(out["z"], want["z"]), fn.__name__


def test_a_loop_that_reads_an_earlier_nest_stays_out_of_its_inner_loop() -> None:
    # The same inference, on shapes that lowered before statements at two
    # depths did: a later loop of its own, and a ragged inner loop.
    a = np.arange(12.0).reshape(3, 4)
    arguments = {"a": a, "y": np.ones(3), "z": np.full(3, 10.0)}
    out = run(total_in_a_later_loop.trace(), **arguments)
    assert np.allclose(out["z"], 11.0 + a.sum(axis=1))

    counts = [2, 0, 3]
    out = run(
        total_after_ragged.trace(),
        cnt=np.array(counts),
        val=Arr.ragged(counts, values=[1.0, 2.0, 3.0, 4.0, 5.0]),
        y=np.ones(3),
        z=np.full(3, 10.0),
    )
    assert np.allclose(out["z"], [14.0, 11.0, 23.0])


def test_one_name_for_two_different_loops_is_refused_by_name() -> None:
    # A term built by hand can use ``j`` inside ``r`` in one statement and on
    # its own in another, which no cut can make one loop. loopy refused it with
    # a RuntimeError about a generated domain.
    import dataclasses

    import pymbolic.primitives as prim

    from loopty.lower import LoweringError, lower_generic
    from loopty.term import Access

    term = after_inner.trace()
    inner, after = term.stmts
    alone = dataclasses.replace(
        after,
        inames=("j",),
        domain=isl.Set("[m] -> { [j] : 0 <= j < m }"),
        loop_domain=None,
        assignee=Access("z", (prim.Variable("j"),)),
    )
    reused = dataclasses.replace(term, stmts=(inner, alone))
    with pytest.raises(LoweringError, match="loop variable j for two different"):
        lower_generic(reused, "c")


# }}}


# {{{ a bound affine in an outer binder, and hardware axes on nested reductions


@kernel
def lower_total8(a: Arr[Fin[8], Fin[8], Real], s: Arr[Fin[1], Real]):
    """The lower-triangle double sum at a size known when the code is generated."""
    s[0] = reduce_sum(reduce_sum(a[i, j] for j in Fin[i + 1]) for i in a.dom)


def test_an_inner_bound_affine_in_the_outer_binder_is_not_a_ragged_fiber() -> None:
    # ``j < i + 1`` names ``i`` as a parameter of the inner domain, and every
    # parameter that was not a size used to count as data.
    from loopty.schedule import Schedule, data_dependent_inames

    assert data_dependent_inames(lower_total.trace()) == frozenset()
    schedule = Schedule(lower_total).split("j", 2, inner="ji", outer="jo").tag(
        ji="l.0"
    )
    ok, reason = schedule.buildable
    assert not ok
    assert "ragged fiber" not in reason
    assert "nested in the reduction over i" in reason and "ji" in reason


def test_a_ragged_fiber_inside_a_statement_loop_is_still_one() -> None:
    from loopty.schedule import data_dependent_inames

    assert data_dependent_inames(rows_and_lengths.trace()) == frozenset({"j"})


@kernel
def ragged_inside_a_sum(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    b: Arr[Fin[p], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A ragged sum nested in a dense one, and the other way round."""
    for r in y.dom:
        y[r] = reduce_sum(
            reduce_sum(val[r, j] * b[k] for j in val.dom[r]) for k in b.dom
        ) + reduce_sum(
            reduce_sum(val[r, i] * b[q] for q in b.dom) for i in val.dom[r]
        )


def test_a_ragged_fiber_nested_in_another_sum_is_still_one() -> None:
    # The enclosing binder is no longer data; the row length beside it still
    # is, and the ragged reason comes first.
    from loopty.schedule import Schedule, data_dependent_inames

    assert data_dependent_inames(ragged_inside_a_sum.trace()) == frozenset(
        {"j", "i"}
    )
    ok, reason = Schedule(ragged_inside_a_sum).tag(j="l.0").buildable
    assert not ok
    assert "ragged fiber" in reason


def test_the_nested_axis_loopty_refuses_is_one_loopy_cannot_build(monkeypatch):
    # Measured, not guessed: loopy's plain OpenCL target stands in for the
    # pyopencl one, which cannot be built without pyopencl, and code generation
    # is all that is asked of it.
    import warnings

    lp = pytest.importorskip("loopy")
    from loopty import lower
    from loopty.schedule import Schedule

    plain = lower.target_for
    monkeypatch.setattr(
        lower,
        "target_for",
        lambda target="c": lp.OpenCLTarget() if target == "opencl" else plain(target),
    )
    outer = Schedule(lower_total8, target="opencl").tag(i="l.0")
    assert outer.buildable == (True, "")
    with warnings.catch_warnings():
        # Every work item stores the finished sum into ``s[0]``, the same
        # value, and loopy says so the first time it generates the code.
        warnings.simplefilter("ignore", lp.diagnostic.WriteRaceConditionWarning)
        code = lp.generate_code_v2(outer.kernel).device_code()
    assert "get_local_id" in code

    inner = Schedule(lower_total8, target="opencl").tag(j="l.0")
    ok, reason = inner.buildable
    assert not ok
    assert "nested in the reduction over i" in reason
    with pytest.raises(Exception, match="does not use all local hw axes"):
        lp.generate_code_v2(inner.kernel)


# }}}


# {{{ the other limits loopy has on reductions (#35)


@kernel
def total8(a: Arr[Fin[8], Real], s: Arr[Fin[1], Real]):
    """A sum over a size known when the code is generated."""
    s[0] = reduce_sum(a[i] for i in a.dom)


@kernel
def triangle_rows(a: Arr[Fin[n], Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Each row summed up to the diagonal: the extent of j is at most n."""
    for i in y.dom:
        y[i] = reduce_sum(a[i, j] for j in Fin[i + 1])


@kernel
def triangle_rows8(a: Arr[Fin[8], Fin[8], Real], y: Arr[Fin[8], Real]):
    """The same at a size known when the code is generated."""
    for i in y.dom:
        y[i] = reduce_sum(a[i, j] for j in Fin[i + 1])


@kernel
def rows_of_eight(a: Arr[Fin[n], Fin[8], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Rows of a fixed length, and any number of them."""
    for i in y.dom:
        y[i] = reduce_sum(a[i, j] for j in a.dom[i])


@pytest.fixture
def plain_opencl(monkeypatch):
    """loopy's plain OpenCL target, as the nested-axis test above uses it.

    It stands in for the pyopencl one, which cannot be built without
    pyopencl, and code generation is all that is asked of it.
    """
    lp = pytest.importorskip("loopy")
    from loopty import lower

    plain = lower.target_for
    monkeypatch.setattr(
        lower,
        "target_for",
        lambda target="c": lp.OpenCLTarget() if target == "opencl" else plain(target),
    )
    return lp


def generated(lp, schedule) -> str:
    """The device code loopy generates for ``schedule``."""
    import warnings

    with warnings.catch_warnings():
        # A work item per cell of a sum stores one finished value, and an
        # unrolled loop sends loopy to its older scheduler; both say so.
        warnings.simplefilter("ignore", lp.diagnostic.LoopyWarning)
        return lp.generate_code_v2(schedule.kernel).device_code()


@pytest.mark.parametrize("tag", ["g.0", "ilp.seq", "vec"])
def test_a_reduction_on_a_concurrent_axis_that_is_not_local_is_refused(
    plain_opencl, tag
) -> None:
    from loopty.schedule import Schedule

    lp = plain_opencl
    schedule = Schedule(total8, target="opencl").tag(i=tag)
    ok, reason = schedule.buildable
    assert not ok
    assert f"runs over i={tag!r}" in reason and "on a local axis (l.*)" in reason
    with pytest.raises(Exception, match="only form of parallelism supported"):
        generated(lp, schedule)


def test_a_reduction_across_two_local_axes_is_refused(plain_opencl) -> None:
    from loopty.schedule import Schedule

    lp = plain_opencl
    schedule = (
        Schedule(total8, target="opencl")
        .split("i", 2, inner="ii", outer="io")
        .tag(ii="l.0", io="l.1")
    )
    ok, reason = schedule.buildable
    assert not ok
    assert "runs over ii, io on 2 local axes" in reason
    with pytest.raises(Exception, match="more than one parallel iname"):
        generated(lp, schedule)


def test_an_unrolled_reduction_loop_is_a_sequence_as_loopy_reads_it(
    plain_opencl,
) -> None:
    # loopy unrolls an ``ilp`` loop and sums over it in order. Beside a local
    # axis that is a reduction partly in parallel, which loopy refuses and the
    # check used to pass; beside an untagged loop it is all in sequence, which
    # loopy builds and the check used to refuse.
    from loopty.schedule import Schedule

    lp = plain_opencl
    split = Schedule(total8, target="opencl").split("i", 2, inner="ii", outer="io")
    mixed = split.tag(ii="l.0", io="ilp")
    ok, reason = mixed.buildable
    assert not ok
    assert "over ii in parallel and io in sequence (loopy unrolls io" in reason
    with pytest.raises(Exception, match="both parallel and sequential"):
        generated(lp, mixed)

    unrolled = split.tag(ii="ilp")
    assert unrolled.buildable == (True, "")
    assert generated(lp, unrolled)


def test_a_local_reduction_needs_a_numeric_maximum_of_its_extent(plain_opencl):
    from loopty.schedule import Schedule

    lp = plain_opencl
    symbolic = Schedule(triangle_rows, target="opencl").tag(j="l.0")
    ok, reason = symbolic.buildable
    assert not ok
    assert "runs over j on a local axis" in reason
    assert "the extent of j is at most n" in reason
    with pytest.raises(Exception, match="a numeric maximum was not found"):
        generated(lp, symbolic)

    fixed = Schedule(triangle_rows8, target="opencl").tag(j="l.0")
    assert fixed.buildable == (True, "")
    assert "get_local_id" in generated(lp, fixed)


def test_a_local_loop_around_a_local_reduction_needs_one_too(plain_opencl):
    # loopy sizes the reduction's array in local memory by every local axis
    # the statement runs on, and n is not a number.
    from loopty.schedule import Schedule

    lp = plain_opencl
    around = Schedule(rows_of_eight, target="opencl").tag(i="l.1", j="l.0")
    ok, reason = around.buildable
    assert not ok
    assert "inside the loop i, which is on a local axis too" in reason
    assert "the extent of i is at most n" in reason
    with pytest.raises(Exception, match="a numeric maximum was not found"):
        generated(lp, around)

    grouped = Schedule(rows_of_eight, target="opencl").tag(i="g.0", j="l.0")
    assert grouped.buildable == (True, "")
    assert "get_group_id" in generated(lp, grouped)


# }}}


# {{{ one Reduction object in two statements


@kernel
def two_sums_of_one_array(
    x: Arr[Fin[n], Real],  # noqa: F821
    s: Arr[Fin[2], Real],
):
    """The same sum twice, which the tracer records as two Reduction objects."""
    s[0] = reduce_sum(x[j] for j in x.dom)
    s[1] = reduce_sum(x[j] for j in x.dom)


def test_a_reduction_object_two_statements_share_is_planned_in_each() -> None:
    # The tracer builds a Reduction per statement; a term built by hand may
    # share one. The plan was keyed by the object alone, so the second
    # statement's overwrote the first's, both reduced over one iname, and loopy
    # stopped with a CycleError.
    import dataclasses

    from loopty.lower import lower_generic

    term = two_sums_of_one_array.trace()
    first, second = term.stmts
    shared = dataclasses.replace(
        term, stmts=(first, dataclasses.replace(second, expr=first.expr))
    )
    assert lower_generic(shared, "c").reduction_inames == {
        "S0:0": ("j",),
        "S1:0": ("j_0",),
    }
    out = run(shared, x=np.arange(4.0), s=np.zeros(2))
    assert np.allclose(out["s"], [6.0, 6.0])


# }}}


# {{{ names the generated code cannot use


@kernel
def long_extent(x: Arr[Fin[long], Real]):  # noqa: F821
    """A size spelled like a C type: loopy declares it ``int32_t const long``."""
    for i in x.dom:
        x[i] = 1.0


@kernel
def double_counter(x: Arr[Fin[n], Real]):  # noqa: F821
    """A loop variable spelled like a C type: ``for (int32_t double = 0; ...)``."""
    for double in x.dom:
        x[double] = 1.0


@kernel
def int_binder(
    x: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A reduction variable spelled like a C type."""
    for r in y.dom:
        y[r] = reduce_sum(x[r, int] for int in x.dom[r])


@kernel
def keyword_parameter(long: Arr[Fin[n], Real]):  # noqa: F821
    """A parameter spelled like a C type, which was refused already."""
    for i in long.dom:
        long[i] = 1.0


def test_a_size_spelled_like_a_keyword_is_refused() -> None:
    # Only parameters used to be checked, so this reached the C compiler and
    # failed there, about generated code the user never wrote.
    from loopty.lower import LoweringError, lower_generic

    with pytest.raises(LoweringError, match=r"sizes long\b"):
        lower_generic(long_extent.trace(), "c")


def test_a_loop_or_reduction_variable_spelled_like_a_keyword_is_refused() -> None:
    from loopty.lower import LoweringError, lower_generic

    with pytest.raises(LoweringError, match=r"loop variables double\b"):
        lower_generic(double_counter.trace(), "c")
    with pytest.raises(LoweringError, match=r"reduction variables int\b"):
        lower_generic(int_binder.trace(), "c")


def test_a_parameter_spelled_like_a_keyword_is_still_refused() -> None:
    from loopty.lower import LoweringError, lower_generic

    with pytest.raises(LoweringError, match=r"parameters long\b"):
        lower_generic(keyword_parameter.trace(), "c")


@kernel
def underscored_names(x: Arr[Fin[_Complex], Real]):  # noqa: F821
    """A size and a loop variable spelled like C's own underscored keywords."""
    for _Bool in x.dom:
        x[_Bool] = 1.0


@kernel
def double_underscored(x: Arr[Fin[n], Real]):  # noqa: F821
    """A loop variable spelled like an OpenCL C qualifier."""
    for __global in x.dom:
        x[__global] = 1.0


def test_names_c_reserves_by_their_spelling_are_refused() -> None:
    # C reserves every name that starts with an underscore and a capital letter
    # or with two underscores, which is where ``_Bool``, ``_Complex`` and
    # ``_Generic`` live, and OpenCL C's ``__global``. Only the unprefixed
    # spellings were listed, so these passed the check and failed in the
    # compiler.
    from loopty.lower import LoweringError, is_reserved, lower_generic

    refused = r"sizes _Complex\b.*loop variables _Bool\b"
    with pytest.raises(LoweringError, match=refused):
        lower_generic(underscored_names.trace(), "c")
    with pytest.raises(LoweringError, match=r"loop variables __global\b"):
        lower_generic(double_underscored.trace(), "c")
    for name in ("_Generic", "_Static_assert", "_Thread_local", "__kernel", "__x"):
        assert is_reserved(name)
    for name in ("_", "_x", "_x1", "x_", "Bool", "generic_"):
        assert not is_reserved(name)


def test_a_kernel_named_like_an_underscored_keyword_is_renamed_with_a_prefix() -> None:
    from loopty.lower import _kernel_name

    # A suffix leaves ``_Generic_knl`` in the reserved space; a prefix does not.
    assert _kernel_name("_Generic", []) == "k_Generic"
    assert _kernel_name("double", []) == "double_knl"
    assert _kernel_name("axpy", []) == "axpy"


# }}}


# {{{ sorts that are free names


@kernel
def scaled_by_float(
    a: float,
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``a: float`` under postponed annotations: lanky gives ``Var("float")``."""
    for i in y.dom:
        y[i] = a * x[i]


@kernel
def int_elements(c: Arr[Fin[n], int], y: Arr[Fin[n], Real]):  # noqa: F821
    """An element sort spelled with the builtin ``int``."""
    for i in y.dom:
        y[i] = 2.0 * c[i]


def test_a_sort_that_is_a_free_name_is_refused_with_the_sort_to_write() -> None:
    # The sort used to reach the lowering as ``Var("float")``: no numpy dtype,
    # not an integral sort, and read as an ``exact`` index type by the ledger.
    from loopty.lower import LoweringError, lower_generic
    from loopty.trace import TraceError

    with pytest.raises(TraceError, match=r"a: float.*write Real.*np\.float64"):
        scaled_by_float.trace()
    with pytest.raises(TraceError, match=r"the elements of c as int.*write Nat"):
        int_elements.trace()
    # ``lanky check`` reports the refusal as the kernel's one fact.
    (fact,) = scaled_by_float.facts()
    assert fact.status.value == "refuted"
    assert "write Real" in fact.provenance["error"]
    # A hand-built term with such a sort is refused by the lowering, by name.
    from lanky.terms import Var

    from loopty.term import Term

    term = Term(
        name="bad", params=(("a", Var("float")),), sizes=(), stmts=(), post=None
    )
    with pytest.raises(LoweringError, match=r"a: float.*free name"):
        lower_generic(term, "c")
    # The native run needs no sort, and still runs.
    y = np.zeros(3)
    scaled_by_float(2.0, np.array([1.0, 2.0, 3.0]), y)
    assert np.array_equal(y, [2.0, 4.0, 6.0])


# }}}


def test_an_equality_guard_and_an_equality_condition_lower_and_run() -> None:
    # isl spells equality with one '='. The guard used to be handed to it as
    # 'i == 0' and tracing stopped on isl's syntax error, for a statement guard
    # and a reduction condition alike.
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule

    @kernel
    def first(y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(i == 0):
                y[i] = 1.0

    @kernel
    def diagonal(a: Arr[Fin[n], Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = reduce_sum(a[i, j] for j in a.dom[i] if j == i)

    (stmt,) = first.term.stmts
    assert stmt.domain.is_equal(isl.Set("[n] -> { [i = 0] : n > 0 }"))
    square = Arr.from_numpy(np.arange(16.0).reshape(4, 4))
    for k, arguments in (
        (first, {"y": Arr.zeros(4)}),
        (diagonal, {"a": square, "y": Arr.zeros(4)}),
    ):
        fact = LoopyExecutor().differential(k, Schedule(k, target="c"), arguments)
        assert fact.status.value == "tested", fact.provenance
    y = Arr.zeros(4)
    run(diagonal, a=square, y=y)
    assert list(y.numpy()) == [0.0, 5.0, 10.0, 15.0]


def test_a_guard_against_a_real_scalar_agrees_on_the_c_target() -> None:
    # The guard used to be a domain constraint, with a an integer parameter of
    # the compiled kernel, and at a = 2.5 the compiled run disagreed with the
    # native one by 1.0 at a cell. It is now a predicate on the real a.
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule

    @kernel
    def below(a: Real, y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(i < a):
                y[i] = 1.0

    fact = LoopyExecutor().differential(
        below, Schedule(below, target="c"), {"a": 2.5, "y": Arr.zeros(5)}
    )
    assert fact.status.value == "tested", fact.provenance
    y = Arr.zeros(5)
    run(below, a=2.5, y=y)
    assert list(y.numpy()) == [1.0, 1.0, 1.0, 0.0, 0.0]
    (stmt,) = below.term.stmts
    assert "a" not in stmt.domain.get_var_names(isl.dim_type.param)
