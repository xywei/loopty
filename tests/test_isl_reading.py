"""What loopy reads into isl: subscripts and guards on the loops.

loopy reads a subscript into isl for its bounds check and to simplify it in
code generation, and a guard that names only loop variables, sizes and
scalars for its bounds check. Its reader raised on a conversion, which kept
every subscript in 32 bits (#129) and refused a substitution that carried a
store's conversion into a subscript (#145, in ``test_fusion.py``), and it read
a constant by its integer part, and a real scalar as an integer, which made
its bounds check of a guarded access wrong (#137). :mod:`loopty.isl_reading`
makes it decline all three, and the lowering widens a subscript loopy does
not read as affine with no division as it widens any other integer
arithmetic.

It does so only inside ``isl_reading.declining()``, where loopty builds,
transforms, checks, generates and runs its kernels: another user of loopy in
the process reads with loopy's own readings, and loopy checks and generates
its kernels as it does without loopty.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading

import islpy as isl
import loopy as lp
import loopy.kernel.instruction as instruction
import numpy as np
import pymbolic.primitives as prim
import pytest
from lanky.prelude import Int, Real
from loopy.diagnostic import ExpressionToAffineConversionError, LoopyIndexError
from loopy.symbolic import PwAffEvaluationMapper, TypeCast, guarded_aff_from_expr
from pymbolic.mapper import UnsupportedExpressionError

from loopty import Arr, Fin, Schedule, isl_reading, kernel, when
from loopty.executor import LoopyExecutor, emit_code
from loopty.isl_reading import (
    ReadingsInactive,
    active,
    affine_form,
    declining,
    read_as_affine,
)


def agrees(kern, make) -> None:
    """The compiled run agrees with the native one, bit for bit and by its fact."""
    native = make()
    kern(**native)
    compiled = make()
    LoopyExecutor().run(kern, **compiled)
    for name, value in native.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, compiled[name]), name
    fact = LoopyExecutor().differential(kern, Schedule(kern), make())
    assert fact.status.value == "tested", fact.provenance


# {{{ the reader


def test_the_reader_declines_a_conversion_and_a_constant_not_an_integer() -> None:
    i = prim.Variable("i")
    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=["i"])
    # loopy's reader raised UnsupportedExpressionError on a cast, which
    # nothing caught, read 0.5 as 0, and read a float that is an integer as
    # the integer, though an expression with one is computed in floating
    # point, which rounds: np.float32(1.0) * i is 2**24 at i = 2**24 + 1.
    for declined in (
        TypeCast(np.dtype(np.int64), i),
        np.float64(0.5) * i,
        i * 0.5,
        np.float64(2.0) * i,
        i * 2.0,
        np.float32(1.0) * i,
        complex(1, 0) * i,
        np.float64(np.inf) + i,
    ):
        with declining(), pytest.raises(ExpressionToAffineConversionError):
            guarded_aff_from_expr(space, declined)
        assert not read_as_affine(declined)
    # An integer and a 64-bit integer literal are read as the integers they
    # are.
    for read, coefficient in (
        (2 * i, 2),
        (np.int64(100_000) * i, 100_000),
    ):
        with declining():
            aff = guarded_aff_from_expr(space, read)
        assert aff.get_coefficient_val(isl.dim_type.in_, 0).to_python() == coefficient
        assert read_as_affine(read)
    assert not read_as_affine(i * i)
    assert not read_as_affine(prim.Subscript(prim.Variable("col"), (i,)))


# }}}


# {{{ where the readings apply


def loopys_own() -> None:
    """loopy's own readings are in place: none of loopty's is installed."""
    assert vars(PwAffEvaluationMapper)["map_constant"].__module__ == "loopy.symbolic"
    assert "map_type_cast" not in vars(PwAffEvaluationMapper)
    assert instruction.get_insn_domain.__module__ == "loopy.kernel.instruction"
    assert not active()
    assert isl_reading._OPEN == 0


def loopys_own_reading_of(space: isl.Space, i: prim.Variable) -> None:
    """The reader reads as loopy does: 0.5 as 0, and a cast it cannot read."""
    zero = guarded_aff_from_expr(space, i * 0.5)
    assert zero.is_cst() and zero.get_constant_val().to_python() == 0
    with pytest.raises(UnsupportedExpressionError):
        guarded_aff_from_expr(space, TypeCast(np.dtype(np.int64), i))


def test_running_a_kernel_leaves_loopys_own_readings_in_place() -> None:
    # Importing loopty and loopty.lower installed the readings in loopy for
    # the whole process, so every other loopy user in it (sumpy, pytential,
    # Volumential) had its kernels read with them, and loopy's bounds check
    # could refuse a kernel of theirs it accepted without loopty.
    import loopty.lower  # noqa: F401

    loopys_own()
    agrees(
        index_square,
        lambda: {"x": np.arange(9, dtype=np.float64), "y": np.zeros(9)},
    )
    emit_code(Schedule(index_square).split("i", 2))
    loopys_own()
    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=["i"])
    loopys_own_reading_of(space, prim.Variable("i"))


def test_importing_loopty_and_running_a_kernel_installs_nothing(tmp_path) -> None:
    # In a fresh process, so that what loopy had is known before loopty is
    # imported: the same objects are in place after loopty, loopty.lower and
    # a run of a kernel through the executor, and a kernel of loopy's own
    # with a guard loopy reads still generates.
    script = tmp_path / "fresh.py"
    script.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import loopy.kernel.instruction as instruction
            import numpy as np
            from loopy.symbolic import PwAffEvaluationMapper

            def own():
                return (
                    vars(PwAffEvaluationMapper).get("map_constant"),
                    vars(PwAffEvaluationMapper).get("map_type_cast"),
                    instruction.get_insn_domain,
                )

            before = own()
            import loopty
            import loopty.lower
            assert own() == before, "importing loopty changed loopy"

            from lanky.prelude import Real
            from loopty import Arr, Fin, kernel
            from loopty.executor import LoopyExecutor

            @kernel
            def index_square(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
                for i in x.dom:
                    y[i] = x[(i * i) % x.dom.size]

            x = np.arange(9, dtype=np.float64)
            out = LoopyExecutor().run(index_square, x=x, y=np.zeros(9))
            assert list(out["y"]) == [float((i * i) % 9) for i in range(9)]
            after = own()
            assert all(a is b for a, b in zip(after, before)), "a run changed loopy"
            print("loopy's own")
            """
        )
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "loopy's own"


def loopys_kernel(guard: str, value_args: dict) -> object:
    """A kernel of loopy's own: ``y[i] = x[2 * i]`` under ``guard``, ``x`` of ``n``."""
    return lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        [
            lp.Assignment(
                "y[i]",
                "x[2 * i]",
                within_inames=frozenset({"i"}),
                predicates=frozenset({lp.symbolic.parse(guard)}),
            )
        ],
        [
            lp.GlobalArg("x", np.float64, shape=("n",)),
            lp.GlobalArg("y", np.float64, shape=("n",)),
            lp.ValueArg("n", np.int32),
            *(lp.ValueArg(name, dtype) for name, dtype in value_args.items()),
        ],
        target=lp.ExecutableCTarget(),
        lang_version=(2018, 2),
    )


@pytest.mark.parametrize(
    ("guard", "value_args"),
    [
        # loopy reads 2.0 as 2, and checks x[2 * i] where 2 * i < n.
        ("i * 2.0 < n", {}),
        # loopy reads a double scalar as an integer parameter, and checks
        # x[2 * i] where 2 * i < a <= n.
        ("2 * i < a and a <= n", {"a": np.float64}),
    ],
)
def test_loopys_own_kernel_is_read_as_loopy_reads_it(guard, value_args) -> None:
    # loopy's own reading of each guard narrows its bounds check to the
    # cells x has, and loopy generates the kernel. loopty's readings decline
    # the guard, under which loopy refuses it (#148): only inside
    # declining(), before and after which loopy reads it as its own again.
    with lp.CacheMode(False):
        code = lp.generate_code_v2(loopys_kernel(guard, value_args)).device_code()
        assert "y[i] = x[2 * i];" in code
        with declining(), pytest.raises(LoopyIndexError, match="could not establish"):
            lp.generate_code_v2(loopys_kernel(guard, value_args))
        loopys_own()
        again = lp.generate_code_v2(loopys_kernel(guard, value_args))
        assert again.device_code() == code


def test_the_readings_apply_inside_and_nested_contexts_restore() -> None:
    i = prim.Variable("i")
    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=["i"])
    loopys_own()
    with declining():
        assert active()
        with pytest.raises(ExpressionToAffineConversionError):
            guarded_aff_from_expr(space, i * 0.5)
        with declining():
            assert isl_reading._OPEN == 2
            with pytest.raises(ExpressionToAffineConversionError):
                guarded_aff_from_expr(space, TypeCast(np.dtype(np.int64), i))
        # The inner one left the readings to the outer.
        assert active() and isl_reading._OPEN == 1
        with pytest.raises(ExpressionToAffineConversionError):
            guarded_aff_from_expr(space, i * 2.0)
        assert guarded_aff_from_expr(space, 2 * i) is not None
    loopys_own()
    loopys_own_reading_of(space, i)
    # An exception leaves through the context, which restores loopy's own.
    with pytest.raises(RuntimeError, match="raised inside"), declining():
        with declining():
            raise RuntimeError("raised inside")
    loopys_own()


def test_a_thread_outside_reads_with_loopys_own() -> None:
    # The readings are installed in loopy while any thread is inside one, and
    # apply in that thread only: another reads with loopy's own meanwhile.
    i = prim.Variable("i")
    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=["i"])
    inside = threading.Event()
    read = threading.Event()
    seen: list = []

    def outside() -> None:
        inside.wait(timeout=60)
        try:
            loopys_own_reading_of(space, i)
            seen.append(active())
        except BaseException as exc:  # noqa: BLE001 - reported below
            seen.append(exc)
        read.set()

    thread = threading.Thread(target=outside)
    thread.start()
    with declining():
        assert "map_type_cast" in vars(PwAffEvaluationMapper)
        inside.set()
        assert read.wait(timeout=60)
        with pytest.raises(ExpressionToAffineConversionError):
            guarded_aff_from_expr(space, i * 0.5)
    thread.join()
    assert seen == [False]
    loopys_own()

    # Threads entering and leaving at once leave loopy's own behind.
    def nested() -> None:
        for _ in range(50):
            with declining(), declining():
                assert active()
                assert read_as_affine(2 * i) and not read_as_affine(i * 0.5)
            assert not active()

    threads = [threading.Thread(target=nested) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    loopys_own()


def test_loopy_checks_loopty_kernels_only_with_its_readings() -> None:
    # loopty's targets refuse to have loopy check or generate their kernels
    # outside declining(): a cast in a subscript is what loopy's own reader
    # raises on, and code made outside would be served from loopy's cache to
    # a run inside, without the bounds check that refuses x[i + 4] under
    # when(i * 0.5 >= 1). Inside, loopy refuses that check.
    with lp.CacheMode(False):
        for source in (index_square, half_guard_past):
            schedule = Schedule(source)
            with pytest.raises(ReadingsInactive, match="outside"):
                lp.generate_code_v2(schedule.kernel)
            with pytest.raises(ReadingsInactive, match="outside"):
                schedule.kernel.executor()(**inputs()())
        with declining():
            assert "loopty_mod_int64" in lp.generate_code_v2(
                Schedule(index_square).kernel
            ).device_code()
            with pytest.raises(LoopyIndexError, match="could not establish"):
                lp.generate_code_v2(Schedule(half_guard_past).kernel)
            preprocessed = lp.preprocess_program(Schedule(half_guard_past).kernel)
        # Preprocessed inside, the kernel passes loopy's own bounds check
        # outside, which reads the guard as false everywhere, and is refused
        # before its code is generated.
        with pytest.raises(ReadingsInactive, match="outside"):
            lp.generate_code_v2(preprocessed)


# }}}


# {{{ a subscript computed in 64 bits (#129)


@kernel
def index_square(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A cell found by the square of the loop variable."""
    for i in x.dom:
        y[i] = x[(i * i) % x.dom.size]


@kernel
def index_scaled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A cell found by a multiple of the loop variable."""
    for i in x.dom:
        y[i] = x[(i * 7919) % x.dom.size]


@kernel
def col_scaled(
    col: Arr[Fin[n], Fin[m]],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A cell found by a multiple of an index array's entry."""
    for i in y.dom:
        y[i] = x[(col[i] * 7919) % x.dom.size]


def test_a_subscript_is_computed_in_64_bits() -> None:
    # i * i leaves 32 bits at i = 46341, and i * 7919 and col[i] * 7919 at
    # 271_183. Natively they are Python's and numpy's 64-bit integers; the
    # compiled run computed them in 32 bits, read wrong cells, and could die
    # of a read out of bounds. Each subscript is widened as any other integer
    # arithmetic is, and loopy, whose isl reader declines the cast, writes it
    # as it is written.
    rows = 46_342
    agrees(
        index_square,
        lambda: {"x": np.arange(rows, dtype=np.float64), "y": np.zeros(rows)},
    )
    assert "y[i] = x[loopty_mod_int64((int64_t) (i) * i, (int64_t) (n))]" in (
        emit_code(index_square)
    )
    rows = 271_184
    agrees(
        index_scaled,
        lambda: {"x": np.arange(rows, dtype=np.float64), "y": np.zeros(rows)},
    )
    assert "x[loopty_mod_int64((int64_t) (i) * 7919, (int64_t) (n))]" in (
        emit_code(index_scaled)
    )
    m = 300_000

    def entries() -> dict:
        return {
            "col": np.array([271_183, 3, 299_999]),
            "x": np.arange(m, dtype=np.float64),
            "y": np.zeros(3),
        }

    native = entries()
    col_scaled(**native)
    assert native["y"][0] == 271_183 * 7919 % m
    agrees(col_scaled, entries)
    assert "x[loopty_mod_int64((int64_t) (col[i]) * 7919, (int64_t) (m))]" in (
        emit_code(col_scaled)
    )


@kernel
def near_half(x: Arr[Fin[m], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A cell found by a floor division of a multiple of the loop variable."""
    for i in y.dom:
        y[i] = x[(i * 499_999) // 1_000_000 + 2200]


@kernel
def near_half_mod(x: Arr[Fin[m], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A cell found by a remainder of a multiple of the loop variable."""
    for i in y.dom:
        y[i] = x[(i * 499_999) % 1_000_000 + 1_000_000]


def test_an_affine_subscript_with_a_division_is_computed_in_64_bits() -> None:
    # loopy reads these as affine, and wrote isl's form in its 32-bit index
    # type, x[2200 + (499999 * i) / 1000000]: 499999 * i leaves 32 bits at
    # i = 4295, and the division of the wrapped value named cell 53 there,
    # where numpy reads cell 4347; without the 2200 it named a negative cell.
    # isl keeps a multiple below half the divisor as it is, so this happens
    # for small arrays. A division in isl's form makes the subscript one
    # computed in 64 bits, as any other.
    rows = 4400
    agrees(
        near_half,
        lambda: {"x": np.arange(rows, dtype=np.float64), "y": np.zeros(rows)},
    )
    assert (
        "x[loopty_floor_div_int64((int64_t) (i) * 499999, (int64_t) (1000000))"
        " + 2200]" in emit_code(near_half)
    )
    agrees(
        near_half_mod,
        lambda: {"x": np.arange(2_000_000, dtype=np.float64), "y": np.zeros(rows)},
    )
    assert (
        "x[loopty_mod_int64((int64_t) (i) * 499999, (int64_t) (1000000))"
        " + 1000000]" in emit_code(near_half_mod)
    )
    i = prim.Variable("i")
    form = affine_form(prim.FloorDiv(i * 499_999, 1_000_000))
    assert form is not None and form.dim(isl.dim_type.div) == 1


@kernel
def doubled_index(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Every other cell, while there is one."""
    for i in y.dom:
        with when(2 * i < x.dom.size):
            y[i] = x[i * 2]


@kernel
def scaled_mod_seven(x: Arr[Fin[7], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A cell of seven found by a multiple of the loop variable."""
    for i in y.dom:
        y[i] = x[(i * 7919) % 7]


def test_an_affine_subscript_with_no_division_is_left_as_it_was() -> None:
    # loopy simplifies a subscript it reads as affine to the expression isl
    # gives back, in its 32-bit index type, so a widening there would only
    # make it unreadable, and its bounds unchecked: i * 2 is cast nowhere,
    # and C computes its sums and products modulo 2**32 under -fwrapv, which
    # gives the cell an in-bounds subscript names. isl writes (i * 7919) % 7
    # with a division, 2 * i + -7 * ((2 * i) / 7), which is computed in 64
    # bits instead, as the subscripts above are.
    code = emit_code(doubled_index)
    assert "y[i] = x[2 * i];" in code
    agrees(doubled_index, lambda: {"x": np.arange(9.0), "y": np.zeros(9)})
    assert "y[i] = x[loopty_mod_int64((int64_t) (i) * 7919, (int64_t) (7))];" in (
        emit_code(scaled_mod_seven)
    )
    rows = 271_190
    agrees(scaled_mod_seven, lambda: {"x": np.arange(7.0), "y": np.zeros(rows)})


# }}}


# {{{ a non-integer literal in a guard on the loops (#137)


@kernel
def half_guard(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """The cells four on, for the first four."""
    for i in y.dom:
        with when(i * 0.5 < 2):
            y[i] = x[i + 4]


@kernel
def half_guard_past(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """The cells four on, from the third on: past the end."""
    for i in y.dom:
        with when(i * 0.5 >= 1):
            y[i] = x[i + 4]


@kernel
def below_one_and_a_half(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """The last cell and the one past it."""
    for i in y.dom:
        with when(i < 1.5):
            y[i] = x[i + y.dom.size - 1]


@kernel
def twice_under(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Every other cell, with the bound written in double."""
    for i in y.dom:
        with when(i * 2.0 < x.dom.size):
            y[i] = x[2 * i]


@kernel
def rounded_away(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A hundred cells on, where a double sum past 2**53 rounds the one away."""
    for i in y.dom:
        with when(i * 4503599627370496.0 + 1.0 <= i * 4503599627370496.0):
            y[i] = x[i + 100]


@kernel
def between_reals(a: Real, x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A hundred cells on, at the loop variable strictly between a - 1 and a."""
    for i in y.dom:
        with when((i < a) & (i > a - 1)):
            y[i] = x[i + 100]


@kernel
def below_an_integer(
    k: Int,
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The cells of x below k, where k is no more than its size."""
    for i in y.dom:
        with when((i < k) & (k <= x.dom.size)):
            y[i] = x[i]


def inputs(size: int = 8):
    return lambda: {"x": np.arange(size, dtype=np.float64), "y": np.zeros(size)}


def test_a_guard_with_a_non_integer_literal_is_not_read_by_its_integer_part() -> None:
    # loopy read 0.5 as 0. when(i * 0.5 >= 1) read as false everywhere, so
    # its bounds check passed x[i + 4] without looking, and the compiled run
    # read past the end of x where the native one is refused. when(i < 1.5)
    # read as i < 1, which let x[i + n - 1] through at i = 1. Neither guard
    # is read into isl now, so loopy checks the access at every point of the
    # loop, and refuses it.
    for past in (half_guard_past, below_one_and_a_half):
        with pytest.raises(IndexError):
            past(**inputs()())
        with pytest.raises(LoopyIndexError, match="could not establish"):
            LoopyExecutor().run(past, **inputs()())
    # when(i * 0.5 < 2) read as true everywhere, which loopy refused x[i + 4]
    # under; it is refused for every i it checks now, as an access under any
    # guard loopy cannot read is (a product of loop variables, an array's
    # entry), though the native run is in bounds.
    native = inputs()()
    half_guard(**native)
    assert list(native["y"][:5]) == [4.0, 5.0, 6.0, 7.0, 0.0]
    with pytest.raises(LoopyIndexError, match=r"4 <= i0 <= 3 \+ n"):
        LoopyExecutor().run(half_guard, **inputs()())


def test_a_guard_computed_in_floating_point_is_not_read() -> None:
    # A float that is an integer was read as the integer, and a Real scalar
    # as an integer parameter, though the guard is computed in double. Read
    # so, the first guard held nowhere, and the second for no integer a, so
    # loopy's bounds check passed x[i + 100] under each without looking, and
    # the compiled run read past the end of x where the native one is
    # refused: i * 2**52 + 1.0 rounds to i * 2**52 from i = 2, and a = 1.5
    # puts i = 1 strictly between. Neither is read now.
    for past, arguments in (
        (rounded_away, {}),
        (between_reals, {"a": 1.5}),
    ):
        with pytest.raises(IndexError):
            past(**inputs()(), **arguments)
        with pytest.raises(LoopyIndexError, match="could not establish"):
            LoopyExecutor().run(past, **inputs()(), **arguments)
    # So under a guard in double, one in bounds is refused too (#148): the
    # same guard in integers, doubled_index's, compiles.
    native = inputs(9)()
    twice_under(**native)
    assert list(native["y"][:6]) == [0.0, 2.0, 4.0, 6.0, 8.0, 0.0]
    with pytest.raises(LoopyIndexError, match="could not establish"):
        LoopyExecutor().run(twice_under, **inputs(9)())
    # A guard on an integer scalar is read as it was, and narrows the check.
    agrees(
        below_an_integer,
        lambda: {"k": 3, "x": np.arange(4.0), "y": np.zeros(6)},
    )


# }}}
