"""A program's term: its kernel calls composed in call order, and lowered as one.

``@program`` used to run natively and restate its callees' postconditions,
with no term and no lowering. Its term is now what its body does against
placeholders (:mod:`loopty.compose`): the callees' statements in call order,
in the program's names, with an array the body makes as a temporary. That
term lowers into one loopy kernel, and the compiled program is compared with
the native one, as a kernel is.
"""

from __future__ import annotations

import islpy as isl
import loopy as lp
import numpy as np
import pytest
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, Schedule, TraceError, kernel, program, reduce_sum, when
from loopty.executor import LoopyExecutor
from loopty.flow import dependences
from loopty.lower import lower_generic
from loopty.term import declared_layout

# {{{ the kernels


@kernel
def scale(a: Real, x: Arr[Fin[n], Real]):  # noqa: F821
    """Multiply every entry of ``x`` by ``a``."""
    for i in x.dom:
        x[i] = a * x[i]


@kernel
def shift(a: Nat, x: Arr[Fin[n], Nat]):  # noqa: F821
    """Add ``a`` to every entry of ``x``."""
    for i in x.dom:
        x[i] = x[i] + a


@kernel
def scan(cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]):  # noqa: F821
    """Exclusive prefix sum of the counts."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@kernel
def spmv(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    col: Arr[Fin[n], Fin[cnt], Fin[m]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A sparse product that reads its rows through their own offsets."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])


@kernel
def spmv_declared(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A row sum that reads its rows through the offsets it declares."""
    for r in y.dom:
        y[r] = x[r] + reduce_sum(val[r, j] for j in val.dom[r])


@kernel
def flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """Pointwise Burgers flux."""
    for j in u.dom:
        f[j] = 0.5 * u[j] * u[j]


@kernel
def divergence(f: Arr[Fin[n], Real], rhs: Arr[Fin[n], Real]):  # noqa: F821
    """Centred divergence of the flux, inside the boundary."""
    for i in rhs.dom:
        with when((i > 0) & (i + 1 < rhs.dom.size)):
            rhs[i] = -(f[i + 1] - f[i - 1]) / 2


@kernel
def accumulate(x: Arr[Fin[n], Real], t: Arr[Fin[n], Real.exact]):  # noqa: F821
    """Add ``x`` into ``t``, which is compared bit for bit."""
    for i in x.dom:
        t[i] += x[i]


@kernel
def copy(t: Arr[Fin[n], Real.exact], y: Arr[Fin[n], Real.exact]):  # noqa: F821
    """Copy ``t`` into ``y``."""
    for i in y.dom:
        y[i] = t[i]


@kernel
def pair(x: Arr[Fin[k], Real], y: Arr[Fin[k], Real]):  # noqa: F821
    """Two arrays of one length."""
    for i in x.dom:
        y[i] = x[i]


# }}}


# {{{ the programs


@program
def solve(cnt, col, val, x, y, off):
    """Lay the rows out, then multiply: two calls with no edge between them."""
    scan(cnt, off)
    spmv(cnt, col, val, x, y)


@program
def both(cnt, off, a):
    """Lay out the rows and then shift the offsets."""
    scan(cnt, off)
    shift(a, off)


@program
def burgers(u, rhs):
    """The flux into an array the program makes, and its divergence."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)


@program
def twice(x):
    """One kernel, called twice."""
    scale(2.0, x)
    scale(3.0, x)


@program
def outer(u, rhs):
    """A program that calls a program."""
    burgers(u, rhs)


def burgers_inputs(size: int = 8) -> dict:
    u = np.sin(np.linspace(0.0, 2.0 * np.pi, size, endpoint=False))
    return {"u": Arr.from_numpy(u), "rhs": Arr.zeros(size)}


def csr_inputs() -> dict:
    counts = [2, 0, 3]
    return {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "col": Arr.ragged(counts, values=[0, 2, 1, 2, 3], dtype=np.int64),
        "val": Arr.ragged(counts, values=[1.0, 2.0, 3.0, 4.0, 5.0]),
        "x": Arr.from_numpy(np.array([1.0, 10.0, 100.0, 1000.0])),
        "y": Arr.zeros(3),
        "off": Arr.zeros(4, dtype=np.int64),
    }


def _same_set(left: isl.Set, right: isl.Set) -> bool:
    first = left.align_params(right.get_space())
    second = right.align_params(first.get_space())
    return bool(first.is_equal(second))


# }}}


# {{{ the term


def test_a_programs_term_is_its_calls_in_call_order() -> None:
    term = solve.term
    assert term is solve.term
    assert term.name == "solve"
    assert term.param_names == ("cnt", "col", "val", "x", "y", "off")
    assert [stmt.id for stmt in term.stmts] == ["scan.S0", "scan.S1", "spmv.S0"]
    # The statements are in sequence: every one is in a top-level block after
    # the one before, so the source order runs them in call order.
    assert [stmt.order[0] for stmt in term.stmts] == [0, 1, 2]
    # Each call keeps its own file and line, and the term has the program's.
    assert term.stmts[0].where.startswith("test_program.py:")
    assert term.where == solve.where
    assert term.post is None


def test_loops_of_two_calls_get_names_of_their_own() -> None:
    # scan's loop and spmv's are both written ``for r``; two loops are two
    # names, the way the tracer names a second ``for r`` ``r_0``.
    term = solve.term
    assert term.stmt("scan.S1").inames == ("r",)
    assert term.stmt("spmv.S0").inames == ("r_0",)
    # The row length spmv reflects follows its loop.
    assert dict(term.reflected)["nl_cnt_r_0"].index.name == "r_0"


def test_sizes_are_unified_through_the_arrays_passed() -> None:
    # shift's n is scan's n + 1, because both are handed off.
    term = both.term
    assert term.sizes == ("n",)
    types = dict(term.params)
    assert str(types["off"].axes[0]) == "n + 1"
    assert types["a"] is Nat
    loop = term.stmt("shift.S0").domain
    assert _same_set(loop, isl.Set("[n] -> { [i] : 0 <= i <= n }"))
    inputs = {
        "cnt": Arr.from_numpy(np.array([2, 0, 3], dtype=np.int64)),
        "off": Arr.zeros(4, dtype=np.int64),
        "a": 1,
    }
    fact = LoopyExecutor().differential(both, Schedule(both), inputs)
    assert fact.status.value == "tested", fact.provenance


def test_two_element_sorts_for_one_array_are_refused() -> None:
    # The lowered program declares off once; scan says it holds naturals and
    # scale says reals, and choosing either would change what one of them does.
    @program
    def mixed(cnt, off, a):
        scan(cnt, off)
        scale(a, off)

    with pytest.raises(TraceError, match="declares the elements of x as Real"):
        mixed.trace()


def test_sizes_that_cannot_agree_are_refused() -> None:
    @program
    def mismatched(cnt, off):
        scan(cnt, off)
        pair(cnt, off)

    with pytest.raises(TraceError, match="nothing says the two agree"):
        mismatched.trace()


def test_a_repeated_call_gets_statements_and_loops_of_its_own() -> None:
    term = twice.term
    assert [stmt.id for stmt in term.stmts] == ["scale.S0", "scale@2.S0"]
    assert [stmt.inames for stmt in term.stmts] == [("i",), ("i_0",)]
    # The literal scalars are substituted, one per call.
    assert "2.0" in str(term.stmts[0].expr)
    assert "3.0" in str(term.stmts[1].expr)
    assert term.param_names == ("x",)


def test_a_program_called_by_a_program_is_inlined() -> None:
    term = outer.term
    assert [stmt.id for stmt in term.stmts] == ["f.zeros", "flux.S0", "divergence.S0"]
    assert [name for name, _ in term.temporaries] == ["f"]


def test_a_kernel_called_by_keyword_is_recorded_by_parameter() -> None:
    @program
    def by_keyword(u, rhs):
        f = Arr.zeros_like(u)
        flux(f=f, u=u)
        divergence(rhs=rhs, f=f)

    assert [stmt.id for stmt in by_keyword.term.stmts] == [
        "f.zeros",
        "flux.S0",
        "divergence.S0",
    ]


# }}}


# {{{ what the program makes


def test_an_array_the_program_makes_is_a_temporary() -> None:
    term = burgers.term
    assert term.param_names == ("u", "rhs")
    ((name, typ),) = term.temporaries
    assert name == "f"
    assert typ.dtype is Real
    # Zeroed where the body made it, before the first call.
    zeros = term.stmts[0]
    assert zeros.id == "f.zeros"
    assert zeros.order[0] == 0
    assert zeros.assignee.array == "f"
    assert zeros.where.startswith("test_program.py:")


def test_the_lowering_declares_the_temporary_and_nobody_passes_it() -> None:
    lowering = lower_generic(burgers.term)
    entry = lowering.kernel.default_entrypoint
    assert lowering.temporaries == ("f",)
    assert "f" not in {arg.name for arg in entry.args}
    assert lowering.outputs == ("rhs",)
    temporary = entry.temporary_variables["f"]
    assert temporary.address_space == lp.AddressSpace.PRIVATE
    # A Real temporary is as approximate as a Real argument: nothing pins
    # contraction off for it.
    assert lowering.contraction


def test_an_exact_temporary_is_as_exact_as_an_exact_argument() -> None:
    # The accumulation into t is exact because t's sort is Real.exact, and t
    # is a temporary; a reduction tree over it is refused as it would be over
    # an argument of that sort, and contraction is pinned off.
    from loopty import IllegalCast

    @program
    def exact_sum(x, y):
        t = Arr.zeros_like(x)
        accumulate(x, t)
        copy(t, y)

    assert not lower_generic(exact_sum.term).contraction
    with pytest.raises(IllegalCast, match="the accumulation into t is exact"):
        Schedule(exact_sum).realize("t", tree=True)


def test_the_interpreter_runs_a_programs_term() -> None:
    # The term is a meaning of its own: interpreted with numpy's arithmetic,
    # a temporary allocated at the arguments' sizes, it computes what the
    # program computes natively. solve's rows are read through their own
    # offsets, as spmv reads them, and not through the zeroed off.
    from loopty.interpret import interpret

    for prog, make in ((burgers, burgers_inputs), (solve, csr_inputs)):
        native, interpreted = make(), make()
        prog(**native)
        interpret(prog.term, interpreted)
        for name, value in native.items():
            if isinstance(value, Arr):
                assert np.array_equal(interpreted[name].numpy(), value.numpy()), name


def test_the_array_made_is_shaped_like_what_it_was_made_like() -> None:
    @program
    def made_short(cnt, off, a):
        scan(cnt, off)
        other = Arr.zeros_like(cnt)
        shift(a, other)
        shift(a, off)

    # scale gives ``other`` a size of its own, and being made like cnt makes
    # that size scan's n; off is n + 1 long, so the second call's loop is.
    term = made_short.term
    assert term.sizes == ("n",)
    types = dict(term.temporaries)
    assert str(types["other"].axes[0]) == "n"
    first = term.stmt("shift.S0").domain
    second = term.stmt("shift@2.S0").domain
    assert _same_set(first, isl.Set("[n] -> { [i] : 0 <= i < n }"))
    assert _same_set(second, isl.Set("[n] -> { [i] : 0 <= i <= n }"))


def test_an_array_made_like_a_ragged_one_is_refused() -> None:
    @program
    def ragged_temporary(cnt, col, val, x, y):
        v = Arr.zeros_like(val)
        spmv(cnt, col, val, x, y)
        spmv(cnt, col, v, x, y)

    with pytest.raises(TraceError, match="as a ragged array"):
        ragged_temporary.trace()


def test_zeros_like_is_zeros_with_the_layout_of_its_argument() -> None:
    dense = Arr.zeros_like(Arr.from_numpy(np.array([1.0, 2.0, 3.0])))
    assert list(dense.numpy()) == [0.0, 0.0, 0.0]
    ragged = Arr.ragged([2, 0, 1], values=[1.0, 2.0, 3.0])
    copy = Arr.zeros_like(ragged, dtype=np.int64)
    assert copy.is_ragged and list(copy.offsets) == [0, 2, 2, 3]
    assert copy.offsets is not ragged.offsets
    assert copy.numpy().dtype == np.int64
    assert list(Arr.zeros_like(np.ones((2, 2))).numpy().ravel()) == [0.0] * 4


# }}}


# {{{ one kernel, run


def test_the_compiled_program_agrees_with_the_native_one() -> None:
    fact = LoopyExecutor().differential(burgers, Schedule(burgers), burgers_inputs())
    assert fact.status.value == "tested", fact.provenance
    assert fact.id == "agreement:burgers[c]"
    assert fact.where == burgers.where
    assert set(fact.provenance["outputs"]) == {"rhs"}


def test_a_compiled_program_computes_the_composition() -> None:
    inputs = burgers_inputs()
    u = inputs["u"].numpy().copy()
    LoopyExecutor().run(burgers, **inputs)
    f = 0.5 * u * u
    want = np.zeros_like(u)
    want[1:-1] = -(f[2:] - f[:-2]) / 2
    assert np.allclose(inputs["rhs"].numpy(), want)


def test_dependences_are_derived_across_kernels() -> None:
    # divergence reads the f that flux writes, which is one array of the
    # program's term, so the edge is in the footprints and the lowering orders
    # the two instructions by it; the zeroing is ordered before flux's write.
    term = burgers.term
    assert not dependences(term).is_empty()
    entry = lower_generic(term).kernel.default_entrypoint
    assert "flux_S0" in entry.id_to_insn["divergence_S0"].depends_on
    assert "f_zeros" in entry.id_to_insn["flux_S0"].depends_on
    # And no edge where nothing is shared but a read: scan and spmv both read
    # cnt, and neither writes anything the other touches.
    entry = lower_generic(solve.term).kernel.default_entrypoint
    assert not entry.id_to_insn["spmv_S0"].depends_on & {"scan_S0", "scan_S1"}


def test_a_program_parameter_called_off_is_not_taken_for_the_offsets() -> None:
    # spmv declares no offsets, so it reads its rows through their own; a
    # program parameter of the name ``off`` is scan's output, and indexing
    # through it would make the program's contract refuse a zeroed ``off``.
    term = solve.term
    assert term.offsets == (("cnt", None),)
    assert declared_layout(term.params, term.offsets) == {
        "col": ("cnt", None),
        "val": ("cnt", None),
    }
    lowering = lower_generic(term)
    assert lowering.ragged == {"col": "off_cnt", "val": "off_cnt"}
    fact = LoopyExecutor().differential(solve, Schedule(solve), csr_inputs())
    assert fact.status.value == "tested", fact.provenance
    assert set(fact.provenance["outputs"]) == {"y", "off"}


def test_offsets_a_kernel_declares_are_read_through_what_the_program_passed() -> None:
    @program
    def rows(cnt, offs, val, x, y):
        spmv_declared(cnt, offs, val, x, y)

    term = rows.term
    assert term.offsets == (("cnt", "offs"),)
    assert lower_generic(term).ragged == {"val": "offs"}
    inputs = csr_inputs()
    arguments = {
        "cnt": inputs["cnt"],
        "offs": Arr.from_numpy(inputs["val"].offsets.copy()),
        "val": inputs["val"],
        "x": Arr.from_numpy(np.array([1.0, 2.0, 3.0])),
        "y": Arr.zeros(3),
    }
    fact = LoopyExecutor().differential(rows, Schedule(rows), arguments)
    assert fact.status.value == "tested", fact.provenance


def test_calls_reading_one_family_through_two_offsets_are_refused() -> None:
    @program
    def two_layouts(cnt, col, val, x, y, off):
        spmv(cnt, col, val, x, y)
        spmv_declared(cnt, off, val, x, y)

    with pytest.raises(TraceError, match="its own offsets"):
        two_layouts.trace()


# }}}


# {{{ what a program's body may not do


@pytest.mark.parametrize(
    ("body", "said"),
    [
        (lambda u, rhs: u[0], "reads a cell of its parameter u"),
        (lambda u, rhs: rhs.dom, "asks for .dom of its parameter rhs"),
        (lambda u, rhs: u * 2.0, "computes with its parameter u"),
        (lambda u, rhs: bool(u), "branches on its parameter u"),
        (lambda u, rhs: list(range(u)), "uses as an integer"),
        (lambda u, rhs: np.sum(u), "hands numpy"),
    ],
)
def test_a_body_that_touches_an_argument_is_refused(body, said) -> None:
    def touches(u, rhs):
        body(u, rhs)
        flux(u, rhs)

    touches.__name__ = "touches"
    with pytest.raises(TraceError, match=said) as refused:
        program(touches).trace()
    assert "Do the work in a kernel" in str(refused.value)


def test_an_array_from_outside_the_program_is_refused() -> None:
    outside = Arr.zeros(4)

    @program
    def borrows(u):
        flux(u, outside)

    with pytest.raises(TraceError, match="neither a parameter of borrows"):
        borrows.trace()


def test_one_array_for_two_parameters_of_a_call_is_refused() -> None:
    @program
    def aliased(u):
        flux(u, u)

    with pytest.raises(TraceError, match="may not share storage"):
        aliased.trace()


def test_an_array_passed_as_a_scalar_is_refused() -> None:
    @program
    def confused(a, x):
        scale(a, x)
        scale(x, a)

    with pytest.raises(TraceError, match="scalar parameter"):
        confused.trace()


def test_a_parameter_no_kernel_is_given_is_refused() -> None:
    @program
    def idle(u, rhs, spare):
        burgers(u, rhs)

    with pytest.raises(TraceError, match="never passes its parameter spare"):
        idle.trace()


def test_a_callee_that_cannot_be_traced_is_named() -> None:
    @kernel
    def branchy(u: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            if i > 0:
                u[i] = 1.0

    @program
    def calls_branchy(u):
        branchy(u)

    with pytest.raises(TraceError, match="calls branchy at .*cannot be traced"):
        calls_branchy.trace()


def test_the_native_run_is_not_refused() -> None:
    # A program runs natively whether or not it has a term.
    x = Arr.from_numpy(np.array([1.0, 2.0]))

    @program
    def natively(x):
        scale(2.0, x)
        x.numpy()[0] = 7.0

    natively(x)
    assert list(x.numpy()) == [7.0, 4.0]
    with pytest.raises(TraceError):
        natively.trace()


# }}}
