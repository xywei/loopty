"""Arithmetic in both runs: what C computes and what numpy computes.

The native run computes with numpy's arithmetic and the compiled run with C's,
and the two type an operation differently: numpy by NEP 50, where a Python
number takes the dtype of what stands beside it, C by its usual arithmetic
conversions (:mod:`loopty.promotion`). Five ways the runs disagreed: an
integer quotient (#82), a connective of integers (#83), a power (#84), a
literal beside a ``float32`` (#91), and an integral value outside 32 bits
(#92). Each kernel here either agrees with its native run bit for bit, or is
refused, by the trace or the contract, with the fix named.
"""

from __future__ import annotations

import numpy as np
import pytest
from lanky.prelude import Int, Nat

from loopty import Arr, Fin, Schedule, kernel
from loopty.contract import INTEGRAL_RANGE
from loopty.executor import LoopyExecutor
from loopty.lower import numpy_dtype


def agrees(kern, make) -> None:
    """The compiled run of ``kern`` agrees bit for bit with its native run."""
    native = make()
    kern(**native)
    compiled = make()
    LoopyExecutor().run(kern, **compiled)
    for name, value in native.items():
        if isinstance(value, np.ndarray):
            assert value.dtype == compiled[name].dtype
            assert np.array_equal(value, compiled[name]), (name, value, compiled[name])
    fact = LoopyExecutor().differential(kern, Schedule(kern), make())
    assert fact.status.value == "tested", fact.provenance


# {{{ an integral value outside 32 bits (#92)


@kernel
def plus_one(c: Arr[Fin[n], Nat], d: Arr[Fin[n], Nat]):  # noqa: F821
    """Every entry plus one."""
    for i in c.dom:
        d[i] = c[i] + 1


@kernel
def shifted(c: Arr[Fin[n], Int], s: Int, d: Arr[Fin[n], Int]):  # noqa: F821
    """Every entry plus a scalar."""
    for i in c.dom:
        d[i] = c[i] + s


def test_an_integral_value_outside_32_bits_is_refused():
    # 2**32 + 5 ran natively as it was and was 5 compiled.
    low, high = INTEGRAL_RANGE
    assert (low, high) == (-(2**31), 2**31)
    info = np.iinfo(numpy_dtype(Nat))
    assert (int(info.min), int(info.max) + 1) == INTEGRAL_RANGE

    def make(c) -> dict:
        return {"c": c, "d": np.zeros(len(c), np.int64)}

    message = r"c\[0\] is 4294967301, which is outside -2147483648 <= v < 2147483648"
    for run in (
        lambda data: plus_one(**data),
        lambda data: LoopyExecutor().run(plus_one, **data),
        lambda data: LoopyExecutor().differential(plus_one, Schedule(plus_one), data),
    ):
        with pytest.raises(ValueError, match=message):
            run(make(np.array([2**32 + 5, 1])))
    with pytest.raises(
        ValueError, match="declare the elements of c as a numpy integer"
    ):
        plus_one(**make(np.array([2**32 + 5, 1])))
    # A uint64 entry was read natively through an int64 copy, as a negative
    # number, and a float-stored one is a whole number outside the range.
    with pytest.raises(ValueError, match=r"c\[1\] is 9223372036854775813, which is"):
        plus_one(**make(np.array([1, 2**63 + 5], np.uint64)))
    with pytest.raises(
        ValueError, match=r"c\[0\] is 1099511627776.0, which is outside"
    ):
        plus_one(**make(np.array([2.0**40, 1.0])))
    agrees(plus_one, lambda: make(np.array([high - 2, 0])))

    def scalar(s) -> dict:
        return {"c": np.array([low, 5]), "s": s, "d": np.zeros(2, np.int64)}

    for s in (high, np.int64(2**40), low - 1):
        with pytest.raises(
            ValueError, match=f"the argument s is {s}, which is outside"
        ):
            shifted(**scalar(s))
        with pytest.raises(ValueError, match="declare s as a numpy integer"):
            LoopyExecutor().run(shifted, **scalar(s))
    agrees(shifted, lambda: scalar(7))


# }}}
