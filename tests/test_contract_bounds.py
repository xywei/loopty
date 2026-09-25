"""The edges of the argument contract: booleans, non-numbers, affine bounds, complex.

Each of these is a value that used to slip past :mod:`loopty.contract` and
reach the compiled kernel as an index nobody declared.
"""

from __future__ import annotations

import numpy as np
import pytest
from lanky.prelude import Real
from lanky.terms import Var

from loopty import Arr, Fin, kernel
from loopty.contract import (
    element_types,
    resolve_sizes,
    scalar_parameters,
    sort_bound,
)
from loopty.executor import LoopyExecutor


@kernel
def pick_next(
    i: Fin[n + 1],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n + 1], Real],  # noqa: F821
):
    """``y[i] = 1 + x[0]``: a scalar whose index type is affine in a size.

    ``x`` is read, and not only there to determine ``n``: an array parameter the
    body never touches cannot be lowered, see
    :func:`test_an_unused_array_parameter_is_refused`.
    """
    y[i] = x[0] + 1.0


def arguments(index) -> dict:
    # ``x`` has three cells, so ``n = 3`` and ``i: Fin[n + 1]`` allows ``0 <= i < 4``.
    return {"i": index, "x": np.zeros(3), "y": np.zeros(4)}


def test_a_boolean_is_not_a_point_of_an_index_type() -> None:
    types = pick_next.arg_types
    with pytest.raises(ValueError, match=r"the argument i is True, a boolean"):
        scalar_parameters(types, arguments(True))
    with pytest.raises(ValueError, match=r"a boolean"):
        scalar_parameters(types, arguments(np.bool_(False)))


def test_a_non_number_is_refused_rather_than_skipped() -> None:
    with pytest.raises(ValueError, match=r"is not a number"):
        scalar_parameters(pick_next.arg_types, arguments("0"))
    with pytest.raises(ValueError, match=r"is not a number"):
        scalar_parameters(pick_next.arg_types, arguments(np.zeros(2)))


def test_a_zero_dimensional_array_is_a_number() -> None:
    scalar_parameters(pick_next.arg_types, arguments(np.array(2)))


def test_an_affine_fin_bound_is_enforced_against_the_resolved_sizes() -> None:
    types = pick_next.arg_types
    for index in (0, 3, np.int64(3)):
        scalar_parameters(types, arguments(index))
    with pytest.raises(ValueError, match=r"the argument i is -1"):
        scalar_parameters(types, arguments(-1))
    with pytest.raises(ValueError, match=r"the argument i is 4"):
        scalar_parameters(types, arguments(4))


def test_an_unresolved_fin_bound_keeps_its_floor() -> None:
    sort = Fin[Var("n") + 1]
    assert sort_bound(sort, {}) == (0, None)
    assert sort_bound(sort, {"n": 3}) == (0, 4)


def test_an_affine_scalar_bound_lowers_and_runs() -> None:
    out = LoopyExecutor().run(pick_next.trace(), **arguments(3))
    assert out["y"][3] == 1.0
    assert not out["y"][:3].any()


@kernel
def never_reads_x(
    i: Fin[n + 1],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n + 1], Real],  # noqa: F821
):
    y[i] = 1.0


def test_an_unused_array_parameter_is_refused() -> None:
    # loopy's C target lists only the arrays the body touches in the device
    # function's signature but passes every argument from the host wrapper, so
    # an array the body never reads or writes shifts every later argument into
    # the wrong register: the run above returned zeros and corrupted the heap.
    # Refusing the term is the honest answer until loopy changes.
    from loopty.lower import LoweringError, lower

    with pytest.raises(LoweringError, match=r"x.*never (read|written|touched)"):
        lower(never_reads_x.trace())


@kernel
def gather(
    col: Arr[Fin[n], Fin[m]],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for i in y.dom:
        y[i] = x[col[i]]


def test_a_complex_index_array_is_checked_for_whole_real_entries() -> None:
    types = gather.arg_types
    good = {
        "col": np.array([0, 1, 0], dtype=complex),
        "x": np.zeros(2),
        "y": np.zeros(3),
    }
    element_types(types, good)
    with pytest.raises(ValueError, match=r"finite whole number"):
        element_types(types, dict(good, col=np.array([0, 1.5 + 0j, 0])))
    with pytest.raises(ValueError, match=r"imaginary part"):
        element_types(types, dict(good, col=np.array([0, 1j, 0])))


@kernel
def bump(
    i: Fin[n + 1],  # noqa: F821
    y: Arr[Fin[n + 1], Real],  # noqa: F821
):
    """``n`` occurs only as ``n + 1``: nothing but this expression determines it."""
    y[i] = y[i] + 1.0


def test_a_size_that_only_occurs_in_an_affine_axis_is_solved_for() -> None:
    types = bump.arg_types
    assert resolve_sizes(types, {"i": 0, "y": np.zeros(4)})["n"] == 3
    for index in (0, 3):
        scalar_parameters(types, {"i": index, "y": np.zeros(4)})
    with pytest.raises(ValueError, match=r"the argument i is 4"):
        scalar_parameters(types, {"i": 4, "y": np.zeros(4)})
    with pytest.raises(ValueError, match=r"the argument i is -1"):
        scalar_parameters(types, {"i": -1, "y": np.zeros(4)})


def test_a_bare_axis_wins_over_a_solved_one() -> None:
    # ``x`` says ``n = 3`` directly; ``y`` of length 4 agrees, and even a
    # disagreeing ``y`` would not override the bare axis.
    types = pick_next.arg_types
    assert resolve_sizes(types, arguments(0))["n"] == 3
    assert resolve_sizes(types, {"i": 0, "x": np.zeros(3), "y": np.zeros(9)})["n"] == 3


@kernel
def square_pick(
    i: Fin[n * n],  # noqa: F821
    y: Arr[Fin[n * n], Real],  # noqa: F821
):
    """``n`` occurs only as ``n * n``, which is not linear in ``n``."""
    y[i] = y[i] + 1.0


def test_a_size_under_a_non_linear_axis_is_not_solved_for() -> None:
    # Evaluated at 0 and at 1, ``n * n`` looks like the line ``n``, so nine cells
    # used to resolve ``n`` to 9 and bound ``i: Fin[n * n]`` by 81: ``i = 9``
    # was accepted and indexed past the end of ``y``.
    types = square_pick.arg_types
    assert "n" not in resolve_sizes(types, {"i": 0, "y": np.zeros(9)})
    # The scalar is measured against the axis written the same way instead.
    for index in (0, 8):
        scalar_parameters(types, {"i": index, "y": np.zeros(9)})
    with pytest.raises(ValueError, match=r"the argument i is 9"):
        scalar_parameters(types, {"i": 9, "y": np.zeros(9)})


def test_a_floor_divided_axis_is_not_solved_for() -> None:
    # ``(n + 1) // 2`` is affine to isl, but three cells are ``n = 5`` or
    # ``n = 6``; the two-point slope used to answer 3, whose axis has two.
    from loopty.term import ArrType

    halves = (Var("n") + 1) // 2
    types = {"y": ArrType(axes=(halves,), dtype=Real, ragged=(False,))}
    assert resolve_sizes(types, {"y": np.zeros(3)}) == {}
    # A linear axis is still solved, as before.
    types = {"y": ArrType(axes=(2 * Var("n") + 1,), dtype=Real, ragged=(False,))}
    assert resolve_sizes(types, {"y": np.zeros(7)}) == {"n": 3}
