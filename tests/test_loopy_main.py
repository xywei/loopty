"""loopty on loopy main, whose domains are namedisl sets (loopy #1023).

loopy's main branch keeps a kernel's domains and assumptions as namedisl
objects, and loopty keeps its own on islpy; :mod:`loopty.named` is where the
two meet. These tests pin that crossing, and the two things loopty tells loopy
main that it did not have to tell loopy 2025.2. ``docs/loopy-main.md`` has the
long form.
"""

from __future__ import annotations

import islpy as isl
import namedisl as nisl
import numpy as np
import pymbolic.primitives as prim
import pytest
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, Schedule, kernel
from loopty.executor import LoopyExecutor, emit_code
from loopty.lower import lower
from loopty.named import (
    aff_from_expr,
    as_isl,
    domains_of,
    is_named,
    to_named,
    to_named_set,
    with_domains,
)

pytest.importorskip("loopy")


@kernel
def scan(cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]):  # noqa: F821
    """The exclusive prefix sum, whose first store is outside every loop."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@kernel
def copy(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A copy over a size that is free."""
    for i in y.dom:
        y[i] = x[i]


# {{{ the crossing


def test_an_isl_object_crosses_to_namedisl_and_back_as_it_was() -> None:
    # Its dimensions keep their order and names both ways: [j, i], not the
    # sorted order namedisl aligns two objects in.
    triangle = isl.Set("[n] -> { [j, i] : 0 <= i < j < n }")
    named = to_named(triangle)
    assert isinstance(named, nisl.Set) and is_named(named)
    back = as_isl(named)
    assert isinstance(back, isl.Set)
    assert back.get_var_names(isl.dim_type.set) == ["j", "i"]
    assert back.is_equal(triangle)
    access = isl.Map("[n] -> { [i] -> [c] : c = i + 1 and 0 <= i < n }")
    assert as_isl(to_named(access)).is_equal(access)


def test_an_object_already_on_its_side_comes_back_as_it_is() -> None:
    named = nisl.make_set("[n] -> { [i] : 0 <= i < n }")
    assert to_named(named) is named
    plain = isl.Set("[n] -> { [i] : 0 <= i < n }")
    assert as_isl(plain) is plain
    assert as_isl(5) == 5


def test_a_basic_set_becomes_the_set_loopy_keeps() -> None:
    # loopy main keeps every domain as a namedisl Set, and refuses a basic
    # set for the assumptions of a kernel it makes.
    basic = isl.BasicSet("[n] -> { : n >= 0 }")
    assert isinstance(to_named(basic), nisl.BasicSet)
    assert isinstance(to_named_set(basic), nisl.Set)


def test_loopys_reader_reads_over_an_isl_space() -> None:
    # loopy main reads an expression over a mapping from names to their
    # affine expressions; loopty asks it over an isl space, and gets isl.
    space = isl.Set("[n] -> { [a, b] }").get_space()
    a, b, n = (prim.Variable(name) for name in "abn")
    aff = aff_from_expr(space, 2 * b - a + n)
    assert isinstance(aff, isl.Aff)
    assert aff.get_domain_space().is_equal(space)
    coefficients = [
        aff.get_coefficient_val(isl.dim_type.in_, k).to_python() for k in range(2)
    ]
    assert coefficients == [-1, 2]
    assert aff.get_coefficient_val(isl.dim_type.param, 0).to_python() == 1


def test_a_kernels_domains_are_read_and_written_as_islpy() -> None:
    entry = lower(copy.trace()).default_entrypoint
    (domain,) = domains_of(entry)
    assert isinstance(domain, isl.Set)
    assert domain.is_equal(isl.Set("[n] -> { [i] : 0 <= i < n }"))
    again = with_domains(entry, [domain])
    assert all(isinstance(kept, nisl.Set) for kept in again.domains)
    assert domains_of(again)[0].is_equal(domain)


# }}}


# {{{ what loopty tells loopy main


def test_loopy_is_told_that_sizes_are_not_negative() -> None:
    # loopy main checks every access over its instruction's domain with the
    # kernel's assumptions, which brings in every size, so off[0] outside
    # every loop is checked, and needs n >= 0. The contract refuses an
    # argument that would make a size negative, so loopy is told so, always,
    # where loopy 2025.2 was told only beside what a scalar's sort says.
    # (namedisl keeps a set of parameters alone as a set of no dimensions.)
    for term in (scan.trace(), copy.trace()):
        assumptions = as_isl(lower(term).default_entrypoint.assumptions)
        assert assumptions.is_equal(isl.Set("[n] -> { [] : n >= 0 }"))


def test_an_access_outside_every_loop_builds_and_runs() -> None:
    counts = np.array([2, 0, 3, 1], dtype=np.int64)
    out = LoopyExecutor().run(scan, cnt=counts, off=np.zeros(5, dtype=np.int64))
    assert list(out["off"]) == [0, 2, 2, 5, 6]
    native = np.zeros(5, dtype=np.int64)
    scan(cnt=counts, off=native)
    assert np.array_equal(out["off"], native)


def test_a_split_loop_is_counted_with_a_division_of_a_size_known_not_negative() -> (
    None
):
    # With n >= 0 known, the number of outer iterations, (3 + n) / 4, is a
    # division of a non-negative number, which C's division rounds the way
    # loopy means: loopy writes it so, where it called its floor-division
    # helper before.
    code = emit_code(Schedule(copy).split("i", 4))
    assert "i_outer <= -1 + (3 + n) / 4;" in code, code
    assert "loopy_floor_div_pos_b" not in code


# }}}
