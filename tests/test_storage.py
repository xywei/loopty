"""What a cell is stored as, in both runs: literals, arguments and truth values.

The compiled run stores every element sort in the dtype the lowering gives it
(:func:`loopty.lower.numpy_dtype`) and the native run in the dtype numpy gives
it, and the two have to hold the same values or the contract has to refuse the
call. Three ways they did not: a real literal lowered in the integer type of
the cell it is stored into (#73), an argument whose dtype does not hold its
sort (#77), and a value that is not a truth value stored into an array of
``Bool`` (#78).
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Bool, Int, Nat, Real

from loopty import Arr, Fin, Schedule, kernel
from loopty.executor import LoopyExecutor, emit_code


def agrees(kern, make) -> None:
    """The scheduled run of ``kern`` agrees with its native run on ``make()``."""
    fact = LoopyExecutor().differential(kern, Schedule(kern), make())
    assert fact.status.value == "tested", fact.provenance


# {{{ a real literal in an integer cell (#73)


@kernel
def halve_nat(u: Arr[Fin[n], Real], c: Arr[Fin[n], Nat]):  # noqa: F821
    """Half of every entry, as a natural number: the fraction goes at the store."""
    for i in u.dom:
        c[i] = u[i] * 0.5


@kernel
def less_half(u: Arr[Fin[n], Real], k: Arr[Fin[n], Int]):  # noqa: F821
    """Every entry less a half, as an integer, which truncates towards zero."""
    for i in u.dom:
        k[i] = u[i] - 0.5


@kernel
def halve_int(k: Arr[Fin[n], Int], h: Arr[Fin[n], Int]):  # noqa: F821
    """Half of every integer, truncated: ``0.5`` beside an integer is a double."""
    for i in k.dom:
        h[i] = k[i] * 0.5


@kernel
def above_half(k: Arr[Fin[n], Int], b: Arr[Fin[n], Bool]):  # noqa: F821
    """Whether half of an integer is above ``2 ** 24``, compared in double."""
    for i in k.dom:
        b[i] = k[i] * 0.5 > 16777216.0


@kernel
def turn(u: Arr[Fin[n], Real], f: Arr[Fin[n], np.complex128]):  # noqa: F821
    """A turn by a complex number single precision does not hold."""
    for i in u.dom:
        f[i] = (0.1 + 0.2j) * u[i]


def test_a_real_literal_stored_into_naturals_is_not_lowered_as_its_integer_part():
    # loopy wrote the bare 0.5 in the type of the integer assignee, as 0.
    def make() -> dict:
        return {
            "u": np.array([1.0, 2.0, 3.0, 4.0]),
            "c": np.zeros(4, dtype=np.int64),
        }

    native = make()
    halve_nat(**native)
    assert list(native["c"]) == [0, 1, 1, 2]
    out = LoopyExecutor().run(halve_nat, **make())
    assert list(out["c"]) == [0, 1, 1, 2]
    assert "u[i] * 0.5" in emit_code(halve_nat)
    agrees(halve_nat, make)


def test_a_real_literal_stored_into_integers_truncates_as_the_native_store_does():
    def make() -> dict:
        return {
            "u": np.array([0.0, 1.0, -1.0, 2.75]),
            "k": np.zeros(4, dtype=np.int64),
        }

    out = LoopyExecutor().run(less_half, **make())
    assert list(out["k"]) == [0, 0, -1, 2]
    agrees(less_half, make)


def test_a_real_literal_beside_an_integer_is_double_precision():
    # loopy took 0.5 for a float32, which an integer joined into a float32, so
    # half of 2 ** 25 + 1 lost its half: 0 into an integer, and not above 2 **
    # 24 into a Bool. numpy computes both in double.
    def integers() -> dict:
        return {
            "k": np.array([33554433, 3], dtype=np.int64),
            "h": np.zeros(2, dtype=np.int64),
        }

    def truths() -> dict:
        return {
            "k": np.array([33554433, 3], dtype=np.int64),
            "b": np.zeros(2, dtype=bool),
        }

    assert list(LoopyExecutor().run(halve_int, **integers())["h"]) == [16777216, 1]
    assert list(LoopyExecutor().run(above_half, **truths())["b"]) == [1, 0]
    agrees(halve_int, integers)
    agrees(above_half, truths)


def test_a_complex_literal_is_double_precision():
    # loopy refuses to guess the precision of a complex constant that single
    # precision does not hold, so this did not lower at all.
    def make() -> dict:
        return {
            "u": np.array([1.0, -2.0]),
            "f": np.zeros(2, dtype=np.complex128),
        }

    out = LoopyExecutor().run(turn, **make())
    assert np.array_equal(out["f"], (0.1 + 0.2j) * np.array([1.0, -2.0]))
    agrees(turn, make)


# }}}
