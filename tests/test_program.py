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
from lanky.prelude import Bool, Int, Nat, Real

from loopty import (
    Arr,
    Fin,
    Schedule,
    TraceError,
    Where,
    kernel,
    program,
    reduce_sum,
    when,
)
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


@kernel
def poke(i: Fin[n], x: Arr[Fin[n], Real]):  # noqa: F821
    """Set one cell, whose index is in bounds by its type."""
    x[i] = 1.0


@kernel
def fill(m: Nat, x: Arr[Fin[m], Real]):  # noqa: F821
    """Number the cells of an array a scalar sizes."""
    for i in x.dom:
        x[i] = 1.0 + i


@kernel
def poke_offsets(i: Fin[n], off: Arr[Fin[n + 1], Nat]):  # noqa: F821
    """Set one cell, whose index is bounded only through an ``n + 1`` axis."""
    off[i] = 1


@kernel
def number(perm: Arr[Fin[n], Fin[n]]):  # noqa: F821
    """Write a permutation, off by one at the end: ``perm[n - 1]`` is ``n``."""
    for i in perm.dom:
        perm[i] = i + 1


@kernel
def gather(
    perm: Arr[Fin[n], Fin[n]],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``y[i] = x[perm[i]]``, in bounds by the element type of ``perm``."""
    for i in y.dom:
        y[i] = x[perm[i]]


@kernel
def lengthen(cnt: Arr[Fin[n], Nat]):  # noqa: F821
    """Make every row longer than its storage."""
    for r in cnt.dom:
        cnt[r] = cnt[r] + 5


@kernel
def move(off: Arr[Fin[n + 1], Nat]):  # noqa: F821
    """Move every row start past where its row is stored."""
    for r in off.dom:
        off[r] = off[r] + 3


@kernel
def bump_two(x: Arr[Fin[n + 2], Real]):  # noqa: F821
    """Add one to an array of at least two cells."""
    for i in x.dom:
        x[i] = x[i] + 1.0


@kernel
def bump_one(x: Arr[Fin[n + 1], Real]):  # noqa: F821
    """Add one to an array of at least one cell."""
    for i in x.dom:
        x[i] = x[i] + 1.0


@kernel
def count_up(f: Arr[Fin[n], np.float32]):  # noqa: F821
    """``1e8 + i``, which a float32 cell rounds to a multiple of eight."""
    for i in f.dom:
        f[i] = 1e8 + i


@kernel
def count_down(
    u: Arr[Fin[n], np.float32],  # noqa: F821
    f: Arr[Fin[n], np.float32],  # noqa: F821
    y: Arr[Fin[n], np.float32],  # noqa: F821
):
    """``f - 1e8``, plus ``u``."""
    for i in y.dom:
        y[i] = f[i] - 1e8 + u[i]


@kernel
def clear(off: Arr[Fin[n], Nat]) -> all(off[r] == 0 for r in Fin[n]):  # noqa: F821
    """Zero every cell, and say so."""
    for r in off.dom:
        off[r] = 0


@kernel
def halve(c: Arr[Fin[n], Nat], f: Arr[Fin[n], Real]):  # noqa: F821
    """Half of every count, which is a real."""
    for i in c.dom:
        f[i] = 0.5 * c[i]


@kernel
def rotate(u: Arr[Fin[n], Real], f: Arr[Fin[n], np.complex128]):  # noqa: F821
    """A quarter turn of every entry, which is imaginary."""
    for i in u.dom:
        f[i] = 1j * u[i]


@kernel
def square(
    f: Arr[Fin[n], np.complex128],  # noqa: F821
    g: Arr[Fin[n], np.complex128],  # noqa: F821
):
    """The square of every entry: ``-u * u`` after ``rotate``."""
    for i in f.dom:
        g[i] = f[i] * f[i]


@kernel
def mark(u: Arr[Fin[n], Real], b: Arr[Fin[n], Bool]):  # noqa: F821
    """Which entries are above one."""
    for i in u.dom:
        b[i] = u[i] > 1.0


@kernel
def keep_unmarked(
    b: Arr[Fin[n], Bool],  # noqa: F821
    u: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Copy the entries ``mark`` did not mark."""
    for i in u.dom:
        with when(~b[i]):
            y[i] = u[i]


@kernel
def truncate(u: Arr[Fin[n], Real], c: Arr[Fin[n], Nat]):  # noqa: F821
    """Every entry as a natural number, which drops its fraction."""
    for i in u.dom:
        c[i] = u[i]


@kernel
def count(c: Arr[Fin[n], Nat], y: Arr[Fin[n], Real]):  # noqa: F821
    """Every count, as a real."""
    for i in c.dom:
        y[i] = 1.0 * c[i]


@kernel
def triangle(
    x: Arr[Fin[n], Real],  # noqa: F821
    f: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real],  # noqa: F821
):
    """A value per pair of the lower triangle."""
    for i in f.dom:
        for j in f.dom[i]:
            f[i, j] = x[i] * x[j]


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


@program
def clears(off):
    """A program with one callee that states a postcondition."""
    clear(off)


@program
def wraps_clears(off):
    """A program whose only call is to a program."""
    clears(off)


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


def test_two_shifted_sizes_are_unified() -> None:
    # bump_two's x is n + 2 long and bump_one's is n + 1: bump_one's n is the
    # program's n + 1, which no bare size of either call says on its own.
    @program
    def bumps(x):
        bump_two(x)
        bump_one(x)

    term = bumps.term
    assert term.sizes == ("n",)
    assert str(dict(term.params)["x"].axes[0]) == "n + 2"
    loop = term.stmt("bump_one.S0").domain
    assert _same_set(loop, isl.Set("[n] -> { [i] : 0 <= i < n + 2 }"))
    x = Arr.from_numpy(np.array([1.0, 2.0, 3.0]))
    fact = LoopyExecutor().differential(bumps, Schedule(bumps), {"x": x})
    assert fact.status.value == "tested", fact.provenance

    # The other way round, the size of the first call is the one that goes:
    # its n is the second call's n + 1, and never the other way round, which
    # would make the second call's n negative for an x of one cell.
    @program
    def bumps_back(x):
        bump_one(x)
        bump_two(x)

    ((size,),) = {bumps_back.term.sizes}
    assert str(dict(bumps_back.term.params)["x"].axes[0]) == f"{size} + 2"
    fact = LoopyExecutor().differential(bumps_back, Schedule(bumps_back), {"x": x})
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
    # And the compiled program agrees with the native one bit for bit.
    x = Arr.from_numpy(np.array([0.1, 0.2, 0.3]))
    fact = LoopyExecutor().differential(
        exact_sum, Schedule(exact_sum), {"x": x, "y": Arr.zeros(3)}
    )
    assert fact.status.value == "tested", fact.provenance
    assert fact.provenance["outputs"]["y"]["tolerance"] == 0


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


def test_the_interpreter_stores_a_temporary_as_both_runs_do() -> None:
    # f is float32 natively and compiled, so 1e8 + 1 is rounded to 1e8 in it,
    # and y comes out u. A float64 f in the interpreter kept the 1.
    from loopty.interpret import interpret

    @program
    def round_trip(u, y):
        f = Arr.zeros_like(u)
        count_up(f)
        count_down(u, f, y)

    def make() -> dict:
        return {
            "u": Arr.from_numpy(np.array([1.0, 3.0], dtype=np.float32)),
            "y": Arr.zeros(2, dtype=np.float32),
        }

    native, interpreted = make(), make()
    round_trip(**native)
    interpret(round_trip.term, interpreted)
    assert list(native["y"].numpy()) == [1.0, 3.0]
    assert np.array_equal(interpreted["y"].numpy(), native["y"].numpy())
    fact = LoopyExecutor().differential(round_trip, Schedule(round_trip), make())
    assert fact.status.value == "tested", fact.provenance


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
    assert fact.id == f"agreement:{burgers.definition}:[c]"
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


@pytest.mark.parametrize(
    ("body", "said"),
    [
        # Natively, poke's contract refuses 7 for a Fin[3]; compiled, the 7 is
        # no argument, and would be written to x[7].
        (lambda x: poke(7, x), "depends on n, which is known only when"),
        (lambda x: poke(1.0, x), r"not an integer.*pass int\(1.0\)"),
        (lambda x: shift(-1, x), "has to satisfy 0 <= a"),
        (lambda x: shift(True, x), "not an integer"),
    ],
)
def test_a_number_for_a_scalar_is_checked_against_its_sort(body, said) -> None:
    def literal(x):
        body(x)

    literal.__name__ = "literal"
    with pytest.raises(TraceError, match=said):
        program(literal).trace()


def test_a_number_the_sort_allows_is_substituted() -> None:
    @program
    def shifted(x):
        shift(2, x)

    x = Arr.from_numpy(np.array([1, 2, 3], dtype=np.int64))
    fact = LoopyExecutor().differential(shifted, Schedule(shifted), {"x": x})
    assert fact.status.value == "tested", fact.provenance


def test_an_index_the_program_is_passed_is_checked_by_its_contract() -> None:
    @program
    def pokes(i, x):
        poke(i, x)

    assert str(dict(pokes.term.params)["i"].bound) == "n"
    with pytest.raises(ValueError, match="0 <= i < 3"):
        LoopyExecutor().run(pokes, i=7, x=Arr.zeros(3))
    x = Arr.zeros(3)
    LoopyExecutor().run(pokes, i=2, x=x)
    assert list(x.numpy()) == [0.0, 0.0, 1.0]


def test_a_scalar_that_sizes_an_array_sizes_it_in_the_programs_names() -> None:
    # fill's x is m long, and m is its scalar: in the program, x is as long as
    # whatever the program passed for m, as the loop over it is. Left in the
    # callee's name, x's length was nobody's, and a k longer than x ran the
    # loop past the end of it.
    @program
    def numbered(k, x):
        fill(k, x)

    term = numbered.term
    assert str(dict(term.params)["x"].axes[0]) == "k"
    fact = LoopyExecutor().differential(
        numbered, Schedule(numbered), {"k": 4, "x": Arr.zeros(4)}
    )
    assert fact.status.value == "tested", fact.provenance
    with pytest.raises(ValueError, match="shape mismatch"):
        LoopyExecutor().run(numbered, k=5, x=Arr.zeros(4))

    @program
    def five(x):
        fill(5, x)

    ((_, typ),) = five.term.params
    assert typ.axes == (5,)
    x = Arr.zeros(5)
    LoopyExecutor().run(five, x=x)
    assert list(x.numpy()) == [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(ValueError, match="shape mismatch"):
        LoopyExecutor().run(five, x=Arr.zeros(4))


def test_a_program_that_returns_something_is_refused() -> None:
    # Natively it returns the array it made; compiled, it is one kernel, which
    # returns what it writes into its parameters.
    @program
    def made_and_returned(u):
        f = Arr.zeros_like(u)
        flux(u, f)
        return f

    with pytest.raises(TraceError, match="returns <f of the program"):
        made_and_returned.trace()

    # Called by a program, it hands the caller an array to pass on.
    @program
    def passes_it_on(u, rhs):
        divergence(made_and_returned(u), rhs)

    term = passes_it_on.term
    assert [name for name, _ in term.temporaries] == ["f"]
    fact = LoopyExecutor().differential(
        passes_it_on, Schedule(passes_it_on), burgers_inputs()
    )
    assert fact.status.value == "tested", fact.provenance


def test_an_index_bounded_through_an_offset_axis_is_checked() -> None:
    # i: Fin[n] beside off: Arr[Fin[n + 1]] alone. The program's sizes are
    # lanky's variables, as a kernel's are, so the contract solves n from
    # off's four cells and holds i below 3, and the lowering can say so.
    @program
    def pokes_offsets(i, off):
        poke_offsets(i, off)

    with pytest.raises(ValueError, match="0 <= i < 3"):
        LoopyExecutor().run(pokes_offsets, i=100, off=Arr.zeros(4, dtype=np.int64))
    off = Arr.zeros(4, dtype=np.int64)
    LoopyExecutor().run(pokes_offsets, i=2, off=off)
    assert list(off.numpy()) == [0, 0, 1, 0]


def test_a_number_is_checked_against_the_programs_sizes() -> None:
    # fill makes x five cells long, so poke's n is 5 there, and 3 is a point
    # of Fin[5] and 7 is not.
    @program
    def fills_and_pokes(x):
        fill(5, x)
        poke(3, x)

    x = Arr.zeros(5)
    LoopyExecutor().run(fills_and_pokes, x=x)
    assert list(x.numpy()) == [1.0, 2.0, 3.0, 1.0, 5.0]

    @program
    def pokes_past(x):
        fill(5, x)
        poke(7, x)

    with pytest.raises(TraceError, match="has to satisfy 0 <= i < 5"):
        pokes_past.trace()


@pytest.mark.parametrize(
    ("body", "said"),
    [
        # number writes n into perm[n - 1], and gather reads x[perm[i]] in
        # bounds by type; natively gather's contract refuses perm.
        (
            lambda perm, cnt, col, val, x, y, off: (
                number(perm),
                gather(perm, x, y),
            ),
            "number at .* writes first, and the elements of perm are declared",
        ),
        # lengthen makes the rows longer than val stores them.
        (
            lambda perm, cnt, col, val, x, y, off: (
                lengthen(cnt),
                spmv(cnt, col, val, x, y),
            ),
            "cnt is the row lengths of col",
        ),
        # move shifts the row starts spmv_declared reads val through.
        (
            lambda perm, cnt, col, val, x, y, off: (
                move(off),
                spmv_declared(cnt, off, val, x, y),
            ),
            "off is the offsets of the rows of val",
        ),
    ],
)
def test_contract_data_an_earlier_call_writes_is_refused(body, said) -> None:
    def writes_then_reads(perm, cnt, col, val, x, y, off):
        body(perm, cnt, col, val, x, y, off)

    writes_then_reads.__name__ = "writes_then_reads"
    with pytest.raises(TraceError, match=said) as refused:
        program(writes_then_reads).trace()
    assert "nothing would check" in str(refused.value)


def test_an_index_array_the_program_makes_is_refused() -> None:
    # Zeros are no point of Fin[0], and nothing checks them against n.
    @program
    def gathers(x, y):
        perm = Arr.zeros_like(y)
        gather(perm, x, y)

    with pytest.raises(TraceError, match="the Arr.zeros_like at .* writes first"):
        gathers.trace()


def test_the_native_program_stops_where_the_refused_term_would_not() -> None:
    # Natively gather's contract refuses the perm number wrote. The term would
    # run gather on it unchecked and read x[4], which is why it is refused.
    @program
    def permuted(perm, x, y):
        number(perm)
        gather(perm, x, y)

    perm = Arr.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError, match=r"perm\[3\] is 4"):
        permuted(perm, Arr.zeros(4), Arr.zeros(4))
    with pytest.raises(TraceError, match="nothing would check"):
        permuted.trace()


def test_a_natural_array_an_earlier_call_writes_is_passed_on() -> None:
    # No fact rests on a Nat cell being non-negative, and off is no layout of
    # shift's: the both program of the unification test stands.
    assert [stmt.id for stmt in both.term.stmts] == ["scan.S0", "scan.S1", "shift.S0"]


def test_a_default_is_refused() -> None:
    @program
    def scaled(x, a=2.0):
        scale(a, x)

    with pytest.raises(TraceError, match="gives its parameter a the default 2.0"):
        scaled.trace()
    # The native run uses it.
    x = Arr.from_numpy(np.array([1.0, 2.0]))
    scaled(x)
    assert list(x.numpy()) == [2.0, 4.0]


def test_a_temporary_of_reals_is_stored_as_the_compiled_one_is() -> None:
    # zeros_like(c) is natively an array of c's dtype. For an integer c that
    # truncates halve's halves, which the compiled temporary, a Real stored as
    # float64, keeps: the compiled run refuses the call, naming the dtype.
    @program
    def halves(c, y):
        f = Arr.zeros_like(c)
        halve(c, f)
        pair(f, y)

    assert halves.term.temporaries_like == (("f", "c"),)
    counts = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="c is stored as int64"):
        LoopyExecutor().run(halves, c=counts.copy(), y=Arr.zeros(3))
    inputs = {"c": Arr.from_numpy(counts.copy()), "y": Arr.zeros(3)}
    fix = "Give that Arr.zeros_like dtype=float64, or pass c as float64"
    with pytest.raises(ValueError, match=fix):
        LoopyExecutor().differential(halves, Schedule(halves), inputs)
    # A c the contract accepts as whole floats makes a float64 f natively too.
    fact = LoopyExecutor().differential(
        halves,
        Schedule(halves),
        {"c": Arr.from_numpy(counts.astype(np.float64)), "y": Arr.zeros(3)},
    )
    assert fact.status.value == "tested", fact.provenance

    # So does a Real u stored as float32, whose f would round what the
    # compiled f keeps.
    with pytest.raises(ValueError, match="u is stored as float32"):
        LoopyExecutor().run(
            burgers, u=np.zeros(4, dtype=np.float32), rhs=Arr.zeros(4)
        )

    # A dtype given is the native array's, and checked when the term is built.
    @program
    def real_halves(c, y):
        f = Arr.zeros_like(c, dtype=np.float64)
        halve(c, f)
        pair(f, y)

    assert real_halves.term.temporaries_like == ()
    fact = LoopyExecutor().differential(real_halves, Schedule(real_halves), inputs)
    assert fact.status.value == "tested", fact.provenance
    y = Arr.zeros(3)
    real_halves(inputs["c"], y)
    assert list(y.numpy()) == [0.5, 1.0, 1.5]

    for dtype in (np.int64, np.float32):

        @program
        def other_dtype(c, y):
            f = Arr.zeros_like(c, dtype=dtype)
            halve(c, f)
            pair(f, y)

        with pytest.raises(TraceError, match="Pass Arr.zeros_like dtype=float64"):
            other_dtype.trace()


@pytest.mark.parametrize(
    ("sort", "storage", "holds", "refused"),
    [
        (Real, np.float64, [np.float64], [np.float32, np.int64, np.complex128]),
        (Real.exact, np.float64, [np.float64], [np.float32]),
        (np.float32, np.float32, [np.float32], [np.float64]),
        (np.complex128, np.complex128, [np.complex128], [np.float64, np.complex64]),
        (complex, np.complex128, [np.complex128], [np.float64]),
        (Bool, np.bool_, [np.bool_], [np.int8, np.float64]),
        (Nat, np.int64, [np.int64, np.int32], [np.int16, np.uint64, np.float64]),
        (Fin[4], np.int64, [np.int64, np.int32], [np.float64, np.bool_]),
        (np.int32, np.int32, [np.int32], [np.int64]),
    ],
)
def test_every_sort_has_a_native_storage(sort, storage, holds, refused) -> None:
    # What a native array has to be stored as to hold what the compiled
    # temporary of the sort holds.
    from loopty.contract import holds_natively, native_storage

    assert native_storage(sort) == np.dtype(storage)
    for dtype in holds:
        assert holds_natively(sort, dtype), dtype
    for dtype in refused:
        assert not holds_natively(sort, dtype), dtype


@pytest.mark.parametrize(
    "sort",
    [
        Real,
        Real.exact,
        float,
        complex,
        np.float32,
        np.complex128,
        np.complex64,
        np.int32,
        Nat,
        Int,
        int,
        Fin[4],
        Bool,
        bool,
    ],
)
def test_every_sort_that_lowers_has_a_native_storage(sort) -> None:
    # The lowering and the storage check know the same sorts, so no temporary
    # lowers unchecked. The compiled storage holds what the native one does,
    # but for Bool, whose compiled byte holds a truth value as a bool does.
    from loopty.contract import holds_natively, native_storage
    from loopty.lower import numpy_dtype

    compiled = numpy_dtype(sort)
    assert native_storage(sort) is not None
    if sort is Bool or sort is bool:
        assert (compiled.kind, compiled.itemsize) == ("i", 1)
    else:
        assert holds_natively(sort, compiled), compiled


def test_a_refused_storage_names_only_the_fixes_that_keep_the_parameter() -> None:
    # Passing the parameter in the temporary's dtype is named as a fix only
    # when that dtype holds the parameter's own sort too: a real u passed as
    # a bool or an integer to make b or c of it would be another u.
    @program
    def unmarked(u, y):
        b = Arr.zeros_like(u)
        mark(u, b)
        keep_unmarked(b, u, y)

    @program
    def counted(u, y):
        c = Arr.zeros_like(u)
        truncate(u, c)
        count(c, y)

    @program
    def rotated(u, g):
        f = Arr.zeros_like(u)
        rotate(u, f)
        square(f, g)

    u = Arr.from_numpy(np.array([0.5, 2.0]))
    cases = [
        (unmarked, {"u": u, "y": Arr.zeros(2)}, "bool", None),
        (counted, {"u": u, "y": Arr.zeros(2)}, "int64", None),
        (
            rotated,
            {"u": u, "g": Arr.zeros(2, dtype=np.complex128)},
            "complex128",
            "complex128",
        ),
    ]
    for prog, inputs, given, passed in cases:
        with pytest.raises(ValueError) as refused:
            LoopyExecutor().run(prog, **inputs)
        message = str(refused.value)
        assert f"Give that Arr.zeros_like dtype={given}" in message, message
        assert ("or pass u as" in message) == (passed is not None), message
        if passed is not None:
            assert f"or pass u as {passed}" in message, message
        assert "<class" not in message, message

    @program
    def given_reals(u, g):
        f = Arr.zeros_like(u, dtype=np.float64)
        rotate(u, f)
        square(f, g)

    with pytest.raises(TraceError, match="declares its elements complex128,"):
        given_reals.trace()


def test_a_complex_temporary_is_stored_as_the_compiled_one_is() -> None:
    # zeros_like(u) of a real u is real natively, so rotate's quarter turn
    # cannot be stored in it, where the compiled temporary, complex, keeps it:
    # the compiled run refuses the call, naming the dtype.
    @program
    def rotated(u, g):
        f = Arr.zeros_like(u)
        rotate(u, f)
        square(f, g)

    def make() -> dict:
        return {
            "u": Arr.from_numpy(np.array([1.0, 2.0, 3.0])),
            "g": Arr.zeros(3, dtype=np.complex128),
        }

    assert rotated.term.temporaries_like == (("f", "u"),)
    with pytest.raises(ValueError, match="u is stored as float64"):
        LoopyExecutor().run(rotated, **make())
    with pytest.raises(ValueError, match="Give that Arr.zeros_like dtype=complex128"):
        LoopyExecutor().differential(rotated, Schedule(rotated), make())

    @program
    def rotated_given(u, g):
        f = Arr.zeros_like(u, dtype=np.complex128)
        rotate(u, f)
        square(f, g)

    assert rotated_given.term.temporaries_like == ()
    fact = LoopyExecutor().differential(
        rotated_given, Schedule(rotated_given), make()
    )
    assert fact.status.value == "tested", fact.provenance
    native = make()
    rotated_given(**native)
    assert list(native["g"].numpy()) == [-1.0, -4.0, -9.0]

    for dtype in (np.float64, np.complex64):

        @program
        def other_dtype(u, g):
            f = Arr.zeros_like(u, dtype=dtype)
            rotate(u, f)
            square(f, g)

        with pytest.raises(TraceError, match="Pass Arr.zeros_like dtype=complex128"):
            other_dtype.trace()


def test_a_temporary_of_truth_values_is_a_bool_natively() -> None:
    # Compiled, b is a byte. Natively only a bool array holds a truth value
    # with ~ logical on it: ~ on a float refuses, and on an integer is
    # bitwise, which when refuses. So b is made a bool, or refused: natively
    # by mark, which writes it, and compiled by the program.
    @program
    def unmarked(u, y):
        b = Arr.zeros_like(u)
        mark(u, b)
        keep_unmarked(b, u, y)

    def make() -> dict:
        return {"u": Arr.from_numpy(np.array([0.5, 2.0, 1.0])), "y": Arr.zeros(3)}

    assert unmarked.term.temporaries_like == (("b", "u"),)
    with pytest.raises(ValueError, match="b is stored as float64.*Pass b as bool"):
        unmarked(**make())
    with pytest.raises(ValueError, match="Give that Arr.zeros_like dtype=bool"):
        LoopyExecutor().run(unmarked, **make())

    @program
    def unmarked_given(u, y):
        b = Arr.zeros_like(u, dtype=bool)
        mark(u, b)
        keep_unmarked(b, u, y)

    native = make()
    unmarked_given(**native)
    assert list(native["y"].numpy()) == [0.5, 0.0, 1.0]
    fact = LoopyExecutor().differential(
        unmarked_given, Schedule(unmarked_given), make()
    )
    assert fact.status.value == "tested", fact.provenance

    @program
    def unmarked_bytes(u, y):
        b = Arr.zeros_like(u, dtype=np.int8)
        mark(u, b)
        keep_unmarked(b, u, y)

    with pytest.raises(TraceError, match="Pass Arr.zeros_like dtype=bool"):
        unmarked_bytes.trace()


def test_a_temporary_of_naturals_is_an_integer_natively() -> None:
    # A real u would keep the fraction truncate drops compiled. Any signed
    # integer of 32 bits or more holds a Nat as the compiled one does.
    @program
    def counted(u, y):
        c = Arr.zeros_like(u)
        truncate(u, c)
        count(c, y)

    def make() -> dict:
        return {"u": Arr.from_numpy(np.array([1.5, 2.0, 0.25])), "y": Arr.zeros(3)}

    assert counted.term.temporaries_like == (("c", "u"),)
    with pytest.raises(ValueError, match="a signed integer of 32 bits or more"):
        LoopyExecutor().run(counted, **make())

    for dtype in (np.int64, np.int32):

        @program
        def counted_given(u, y):
            c = Arr.zeros_like(u, dtype=dtype)
            truncate(u, c)
            count(c, y)

        native = make()
        counted_given(**native)
        assert list(native["y"].numpy()) == [1.0, 2.0, 0.0]
        fact = LoopyExecutor().differential(
            counted_given, Schedule(counted_given), make()
        )
        assert fact.status.value == "tested", fact.provenance

    for dtype in (np.float64, np.int16):

        @program
        def counted_other(u, y):
            c = Arr.zeros_like(u, dtype=dtype)
            truncate(u, c)
            count(c, y)

        with pytest.raises(TraceError, match="Pass Arr.zeros_like dtype=int64"):
            counted_other.trace()


def test_the_interpreter_stores_a_temporary_of_any_sort_as_natively() -> None:
    # A complex temporary allocated as reals dropped rotate's quarter turn, and
    # a Bool one as reals refused ~.
    from loopty.interpret import interpret

    @program
    def rotated(u, g):
        f = Arr.zeros_like(u, dtype=np.complex128)
        rotate(u, f)
        square(f, g)

    @program
    def unmarked(u, y):
        b = Arr.zeros_like(u, dtype=bool)
        mark(u, b)
        keep_unmarked(b, u, y)

    def rotated_inputs() -> dict:
        u = Arr.from_numpy(np.array([1.0, 2.0]))
        return {"u": u, "g": Arr.zeros(2, dtype=np.complex128)}

    def unmarked_inputs() -> dict:
        return {"u": Arr.from_numpy(np.array([0.5, 2.0])), "y": Arr.zeros(2)}

    for prog, make in ((rotated, rotated_inputs), (unmarked, unmarked_inputs)):
        native, interpreted = make(), make()
        prog(**native)
        interpret(prog.term, interpreted)
        for name, value in native.items():
            assert np.array_equal(interpreted[name].numpy(), value.numpy()), name


def test_a_callee_with_an_array_over_a_domain_is_refused() -> None:
    # The domain's sizes are not unified across calls, so the term would not
    # know f's cells. Natively the program runs, and zeros_like keeps f's
    # domain and storage.
    @program
    def pairs_of(x, f):
        triangle(x, f)

    with pytest.raises(TraceError, match="is an array over Where"):
        pairs_of.trace()
    domain = triangle.arg_types["f"].domain
    x = Arr.from_numpy(np.array([1.0, 2.0, 3.0]))
    for storage in ("box", "packed"):
        f = Arr.zeros(domain, n=3, storage=storage)
        pairs_of(x, f)
        assert list(f.cells()) == [2.0, 3.0, 6.0]
        made = Arr.zeros_like(f)
        assert made.domain is not None and made.storage == storage
        assert made.numpy().shape == f.numpy().shape
        assert list(made.cells()) == [0.0, 0.0, 0.0]


def test_an_array_made_like_a_made_array_is_laid_out_as_the_first() -> None:
    # g is made like f, which is made like u and given to no kernel: g is u's
    # length, as it is natively, and so is y, which pair makes g's.
    @program
    def chained(u, rhs, y):
        f = Arr.zeros_like(u)
        g = Arr.zeros_like(f)
        flux(u, rhs)
        pair(g, y)

    term = chained.term
    assert term.sizes == ("n",)
    assert str(dict(term.params)["y"].axes[0]) == "n"
    with pytest.raises(ValueError, match="shape mismatch"):
        LoopyExecutor().run(chained, u=Arr.zeros(3), rhs=Arr.zeros(3), y=Arr.zeros(4))


def test_a_program_that_returns_a_parameter_is_not_refused() -> None:
    # The compiled program writes y, which is what the native one returns.
    @program
    def returns_rhs(u, rhs):
        burgers(u, rhs)
        return rhs

    assert [name for name, _ in returns_rhs.term.params] == ["u", "rhs"]


def test_a_program_that_writes_no_parameter_is_refused() -> None:
    @program
    def keeps_it(u):
        f = Arr.zeros_like(u)
        flux(u, f)

    with pytest.raises(TraceError, match="writes none of its parameters"):
        keeps_it.trace()


def test_a_program_restates_the_postconditions_of_a_program_it_calls() -> None:
    # wraps_clears's term is clear's statements, recorded in place through
    # clears, and so is its restatement of clear's postcondition.
    from loopty.typing import postcondition_id

    assert [stmt.id for stmt in wraps_clears.term.stmts] == ["clear.S0"]
    assert wraps_clears.callees() == (clear,)
    (fact,) = restatements(wraps_clears)
    assert fact.statement.startswith("after clear(...) in wraps_clears")
    assert fact.rests_on == (
        postcondition_id(clear.qualname, module=clear.module, line=clear.line),
    )
    assert [f.statement for f in restatements(clears)] == [
        fact.statement.replace("wraps_clears", "clears")
    ]


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


# {{{ maps per statement of a program (#79)


def test_a_map_per_statement_names_a_program_statement_as_isl_spells_it() -> None:
    # ``flux.S0`` is not an isl tuple name, and ``flux_S0``, loopy's id of its
    # instruction, was refused as not a statement of the program. It names
    # the one statement it spells.
    schedule = Schedule(burgers).affine(
        "[n] -> { flux_S0[j] -> [jj] : jj = n - 1 - j }"
    )
    assert "jj" in schedule.order
    assert {fact.kind for fact in schedule.facts()} == {"bijective", "monotone"}
    assert {fact.status.value for fact in schedule.facts()} == {"decided"}
    fact = LoopyExecutor().differential(burgers, schedule, burgers_inputs())
    assert fact.status.value == "tested", fact.provenance


def test_a_repeated_call_s_statement_is_named_with_its_count() -> None:
    # ``scale@2.S0`` is spelled ``scale_2_S0``, and only its own loop moves.
    stmt = twice.term.stmt("scale@2.S0")
    (loop,) = stmt.inames
    schedule = Schedule(twice).affine(f"{{ scale_2_S0[{loop}] -> [k] : k = {loop} }}")
    assert schedule.order == ("i", "k")
    x = Arr.from_numpy(np.arange(4.0))
    fact = LoopyExecutor().differential(twice, schedule, {"x": x})
    assert fact.status.value == "tested", fact.provenance


def test_a_program_statement_named_by_its_id_says_how_to_spell_it() -> None:
    with pytest.raises(ValueError) as caught:
        Schedule(burgers).affine("{ flux.S0[j] -> [jj] : jj = j }")
    message = str(caught.value)
    assert "is not an isl map" in message
    assert "spelled _: flux.S0 as flux_S0" in message


# }}}


# {{{ a layout a call rewrites (#51)


@kernel
def clear_next(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Sum a row, then clear the length of the next one."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])
        with when(r + 1 < y.dom.size):
            cnt[r + 1] = 0


@program
def clears_next(cnt, val, y):
    """One call that rewrites the counts its rows are read through."""
    clear_next(cnt, val, y)


def test_a_program_whose_call_rewrites_a_layout_has_its_layout_fact() -> None:
    # A schedule of the program rests its monotone casts on the program's
    # layout fact, which the program's facts carry, as a kernel's do.
    from lanky.ledger import fact_id

    (layout,) = [fact for fact in clears_next.facts() if fact.kind == "layout"]
    assert layout.id == fact_id(
        "layout",
        clears_next.qualname,
        module=clears_next.module,
        line=clears_next.line,
        detail="cnt",
    )
    (loop,) = clears_next.term.stmts[0].inames
    schedule = Schedule(clears_next).split(loop, 2)
    (monotone,) = [fact for fact in schedule.facts() if fact.kind == "monotone"]
    assert monotone.rests_on == (layout.id,)
    assert not [fact for fact in burgers.facts() if fact.kind == "layout"]


# }}}


# {{{ the faithfulness fact


def restatements(prog) -> list:
    """The restatements of a program's callee postconditions, among its facts."""
    return [fact for fact in prog.facts() if fact.kind == "postcondition-in-scope"]


@kernel
def plus_one(x: Arr[Fin[n], Real]):  # noqa: F821
    """Add one to every entry."""
    for i in x.dom:
        x[i] = x[i] + 1.0


@kernel
def doubles(x: Arr[Fin[n], Real]):  # noqa: F821
    """Double every entry."""
    for i in x.dom:
        x[i] = 2.0 * x[i]


@program
def probing(x):
    """The issue's program: the second call only when ``x`` is an array."""
    plus_one(x)
    if isinstance(x, Arr):  # False for the placeholder, True natively
        doubles(x)


def faithful_of(prog):
    """The ``trace-faithful`` fact of a program, which is its last fact."""
    from lanky.ledger import fact_id

    fact = prog.facts()[-1]
    assert fact.kind == "trace-faithful"
    assert fact.id == fact_id(
        "trace-faithful", prog.qualname, module=prog.module, line=prog.line
    )
    assert fact.owner == prog.qualname
    assert fact.where == prog.where
    return fact


def test_a_probe_that_changes_the_term_refutes_the_faithfulness_fact() -> None:
    # The term is plus_one alone, and the body runs both kernels, so the two
    # meanings disagree at the first cell of the first sample (#66). It used
    # to be caught only by a differential run on a file with example inputs,
    # and a schedule of the program decided its casts against a term the body
    # does not compute.
    from lanky.ledger import Status

    assert [stmt.id for stmt in probing.term.stmts] == ["plus_one.S0"]
    fact = faithful_of(probing)
    assert fact.status is Status.REFUTED, fact.provenance
    assert fact.decided_by == "interpreter"
    counterexample = fact.provenance["counterexample"]
    assert counterexample["input"].startswith("sample 1 (")
    assert counterexample["cell"] == "x[0]"
    assert counterexample["body"] == 2 * counterexample["term"]
    assert "does not compute what the body computes" in fact.provenance["reason"]
    assert set(fact.provenance["arguments"]) == {"x"}


@pytest.mark.parametrize(
    "prog", [solve, burgers, twice, outer], ids=lambda p: p.__name__
)
def test_a_faithful_program_is_tested_on_every_sample(prog) -> None:
    # The samples are drawn from the types the term gives the program's
    # parameters, which are its callees'; a temporary is the program's own.
    from lanky.ledger import Status

    from loopty.faithful import SAMPLES

    fact = faithful_of(prog)
    assert fact.status is Status.TESTED, fact.provenance
    assert fact.provenance["compared"] == SAMPLES


def test_a_program_whose_term_cannot_be_built_is_assumed() -> None:
    # Nothing to interpret: the fact says why, and the program still runs.
    from lanky.ledger import Status

    @program
    def touches(x):
        scale(2.0, x)
        x.numpy()[0] = 7.0

    fact = faithful_of(touches)
    assert fact.status is Status.ASSUMED
    reason = fact.provenance["reason"]
    assert reason.startswith("the term of touches cannot be built: TraceError: ")


PROBING = """
from __future__ import annotations

from lanky.prelude import Real

from loopty import Arr, Fin, kernel, program


@kernel
def plus_one(x: Arr[Fin[n], Real]):
    for i in x.dom:
        x[i] = x[i] + 1.0


@kernel
def doubles(x: Arr[Fin[n], Real]):
    for i in x.dom:
        x[i] = 2.0 * x[i]


@program
def probing(x):
    plus_one(x)
    if isinstance(x, Arr):
        doubles(x)
"""


def test_check_refutes_a_program_whose_term_is_not_its_body(tmp_path, capsys) -> None:
    # ``lanky check`` used to say nothing about the program at all.
    from lanky.cli import main as lanky_main

    path = tmp_path / "probing.py"
    path.write_text(PROBING, encoding="utf-8")
    code = lanky_main(["check", str(path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "REFUTED probing at probing.py:" in out
    assert "the traced term computes what the body computes" in out


# }}}
