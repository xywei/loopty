"""A program's arrays over polyhedral domains: its callees', and its own (#74).

A callee may take an array over a ``Where``, ``Sigma`` or union domain, and
the program unifies the domain's sizes across its calls as it unifies the
axes of a dense array: two calls given one array have to agree on its points,
as isl sets in the program's sizes. An array the program makes may be over a
domain too, made like an argument over one, ``Arr.zeros_like(L)``, or of a
declared shape, ``Arr.zeros(TRIANGLE, n=x.dom.size)``, whose sizes are read
off the program's arrays. It is a temporary of the lowered kernel, stored
boxed, zeroed where the program makes it, and its edges have definedness
facts as a dense one's do.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from lanky.ledger import Status
from lanky.prelude import Fin, Real

from loopty import Arr, Schedule, TraceError, kernel, program
from loopty.executor import LoopyExecutor
from loopty.faithful import KIND
from loopty.oracle import IslOracle

K = importlib.import_module("kernels.program_domains")
TRIANGLE, fill, rowsum, diagonal = K.TRIANGLE, K.fill, K.rowsum, K.diagonal


def _inputs(count: int = 5, *, with_l: bool = False) -> dict:
    x = np.arange(1.0, count + 1.0)
    out = {"x": Arr.from_numpy(x), "e": Arr.zeros(count)}
    if with_l:
        out["L"] = Arr.zeros(TRIANGLE, n=count)
    return out


def _expected(count: int = 5) -> np.ndarray:
    x = np.arange(1.0, count + 1.0)
    return np.array([x[p] * x[:p].sum() for p in range(count)])


def _settled(facts):
    oracle = IslOracle()
    return [
        (oracle.establish(fact) or fact) if oracle.can_establish(fact) else fact
        for fact in facts
    ]


def test_a_domains_sizes_are_unified_across_calls() -> None:
    through = K.through
    term = through.term
    params = dict(term.params)
    assert str(params["L"].domain) == "Where[i: Fin(n), j: Fin(n), j < i]"
    # rowsum's m is the program's n, through the triangle it is given
    assert term.sizes == ("n",)
    data = _inputs(with_l=True)
    through(**data)
    np.testing.assert_allclose(data["e"].numpy(), _expected())
    for schedule in (Schedule(through), Schedule(through).pack("L")):
        fact = LoopyExecutor().differential(through, schedule, _inputs(with_l=True))
        assert fact.status is Status.TESTED, fact.provenance


def test_two_calls_that_see_other_points_are_refused() -> None:
    @program
    def mismatched(x, L, e):
        fill(x, L)
        diagonal(L, e)

    with pytest.raises(TraceError, match="nothing says the two have the same points"):
        mismatched.term


def test_an_array_made_like_one_over_a_domain_is_over_it() -> None:
    like = K.like
    (name, typ), *_ = like.term.temporaries
    assert name == "f" and str(typ.domain) == str(TRIANGLE)
    zeros = like.term.stmt("f.zeros")
    assert zeros.domain.get_var_names(3) == ["i_0", "i_1"]
    fact = LoopyExecutor().differential(like, Schedule(like), _inputs(with_l=True))
    assert fact.status is Status.TESTED


def test_an_array_of_a_declared_domain_takes_its_sizes_from_the_programs() -> None:
    declared = K.declared
    term = declared.term
    (name, typ), *_ = term.temporaries
    assert name == "f" and str(typ.domain) == str(TRIANGLE)
    data = _inputs()
    declared(**data)
    np.testing.assert_allclose(data["e"].numpy(), _expected())
    fact = LoopyExecutor().differential(declared, Schedule(declared), _inputs())
    assert fact.status is Status.TESTED
    facts = _settled(declared.facts())
    (defined,) = [f for f in facts if f.kind == "definedness"]
    assert defined.status is Status.DECIDED
    assert (
        defined.statement == "every cell of f that rowsum reads, fill stored before it"
    )
    (faithful,) = [f for f in facts if f.kind == KIND]
    assert faithful.status is Status.TESTED
    fused = Schedule(declared).fuse("fill", "rowsum", loops=1)
    assert all(f.status is Status.DECIDED for f in fused.facts())
    fact = LoopyExecutor().differential(declared, fused, _inputs())
    assert fact.status is Status.TESTED


def test_an_array_of_a_declared_dense_shape() -> None:
    @kernel
    def double(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for p in y.dom:
            y[p] = 2.0 * x[p]

    @program
    def twice(x, e):
        y = Arr.zeros((x.dom.size,))
        double(x, y)
        double(y, e)

    data = _inputs()
    twice(**data)
    np.testing.assert_allclose(data["e"].numpy(), 4.0 * np.arange(1.0, 6.0))
    fact = LoopyExecutor().differential(twice, Schedule(twice), _inputs())
    assert fact.status is Status.TESTED


def test_a_declared_shape_names_its_sizes_and_reads_them_off_arrays() -> None:
    @program
    def unsized(x, e):
        f = Arr.zeros(TRIANGLE)
        fill(x, f)
        rowsum(f, e)

    with pytest.raises(TraceError, match="gives no n: give each size"):
        unsized.term

    @program
    def packed(x, e):
        f = Arr.zeros(TRIANGLE, n=x.dom.size, storage="packed")
        fill(x, f)
        rowsum(f, e)

    with pytest.raises(TraceError, match="stored boxed"):
        packed.term

    @program
    def wrong_size(x, e):
        f = Arr.zeros(TRIANGLE, n=x.dom.size + 1)
        fill(x, f)
        rowsum(f, e)

    with pytest.raises(TraceError, match="nothing says the two have the same points"):
        wrong_size.term


def test_a_body_may_read_sizes_and_not_loop_over_them() -> None:
    @program
    def looping(x, e):
        for _ in x.dom:
            rowsum(x, e)

    with pytest.raises(TraceError, match="iterates over the domain of its parameter x"):
        looping.term
