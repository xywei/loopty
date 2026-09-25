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

from loopty import Arr, Fin, kernel
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
