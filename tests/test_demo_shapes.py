"""Shapes sumpy's L2P and grudge's passes are written in, each run three ways.

Five constructs these kernels lean on were believed to work and had never
been run: a ``Where`` array beside a ragged family in one kernel, a ``Where``
of three binders, ``**`` with a loop variable as the exponent inside a sum of
two binders and inside a sum nested in another, arrays of four dense axes, and
``when(a + b == 0)``. Each is here, in the form a loop-y L2P and grudge's
sum-factorized passes write it: run natively and against numpy, every
in-bounds fact decided by isl, the trace faithful, and the compiled run on
the C target compared with the native one.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from lanky.ledger import Status
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, Schedule, Where, kernel, reduce_sum
from loopty.executor import LoopyExecutor
from loopty.faithful import KIND
from loopty.oracle import IslOracle

pytest.importorskip("loopy")

#: Boxes, and the order of the triangle ``a + b < Q``.
NB, Q = 3, 4


# {{{ kernels


@kernel
def reconstruct(
    coeffs: Arr[Fin[nb], Fin[2 * Q], Real],  # noqa: F821
    R: Arr[Where[i : Fin[nb], a : Fin[Q], b : Fin[Q], a + b < Q], Real],  # noqa: F821
):
    """An L2P's reconstruction of sumpy's stored derivatives: three binders."""
    for i in R.dom:
        for a in R.dom[i]:
            for b in R.dom[i, a]:
                if a + b == 0:
                    R[i, a, b] = coeffs[i, 0]
                if (a < 2) & (a + b >= 1):
                    R[i, a, b] = coeffs[i, 2 * (a + b) - 1 + a]
                if a >= 2:
                    R[i, a, b] = -R[i, a - 2, b + 2]


@kernel
def evaluate(
    R: Arr[Where[i : Fin[nb], a : Fin[Q], b : Fin[Q], a + b < Q], Real],  # noqa: F821
    cnt: Arr[Fin[nb], Nat],  # noqa: F821
    tx: Arr[Fin[nb], Fin[cnt], Real],  # noqa: F821
    ty: Arr[Fin[nb], Fin[cnt], Real],  # noqa: F821
    pot: Arr[Fin[nb], Fin[cnt], Real],  # noqa: F821
    inv_fact: Arr[Fin[Q], Real],
):
    """An L2P's evaluation: the triangle beside the ragged targets."""
    for i in pot.dom:
        for t in pot.dom[i]:
            x = tx[i, t]
            y = ty[i, t]
            pot[i, t] = reduce_sum(
                R[i, a, b] * x**a * y**b * inv_fact[a] * inv_fact[b]
                for a in R.dom[i]
                for b in R.dom[i, a]
            )


@kernel
def nested_powers(
    x: Arr[Fin[n], Real],  # noqa: F821
    c: Arr[Fin[q], Fin[q], Real],  # noqa: F821
    out: Arr[Fin[n], Real],  # noqa: F821
):
    """A loop variable as an exponent, in a sum nested in a sum."""
    for t in out.dom:
        out[t] = reduce_sum(
            reduce_sum(c[a, b] * x[t] ** a * x[t] ** (a - b) for b in Fin[a + 1])
            for a in c.dom
        )


@kernel
def along_r(
    F: Arr[Fin[m], Fin[m], Real],  # noqa: F821
    u: Arr[Fin[ne], Fin[m], Fin[m], Fin[m], Real],  # noqa: F821
    out: Arr[Fin[ne], Fin[m], Fin[m], Fin[m], Real],  # noqa: F821
):
    """One pass of a sum-factorized operator on hexahedra: four dense axes."""
    for e in out.dom:
        for a in out.dom[e]:
            for j in out.dom[e, a]:
                for k in out.dom[e, a, j]:
                    out[e, a, j, k] = reduce_sum(
                        F[a, s] * u[e, s, j, k] for s in F.dom[a]
                    )


# }}}


TRIANGLE = reconstruct.arg_types["R"].domain


def triangle_cells() -> list[tuple[int, int, int]]:
    return [
        (i, a, b) for i in range(NB) for a in range(Q) for b in range(Q) if a + b < Q
    ]


def decided(kern) -> None:
    """Every in-bounds and disjoint-writes fact of ``kern`` is decided by isl."""
    oracle = IslOracle()
    for fact in kern.facts():
        if fact.kind in ("in-bounds", "disjoint-writes"):
            assert oracle.establish(fact).status is Status.DECIDED, fact.statement
    (faithful,) = [fact for fact in kern.facts() if fact.kind == KIND]
    assert faithful.status is Status.TESTED, faithful.provenance


def agrees(kern, schedule, inputs: dict) -> None:
    fact = LoopyExecutor().differential(kern, schedule, inputs)
    assert fact.status is Status.TESTED, fact.provenance


def test_a_where_of_three_binders_under_an_equality_guard() -> None:
    rng = np.random.default_rng(1)
    coeffs = rng.normal(size=(NB, 2 * Q))
    R = Arr.zeros(TRIANGLE, nb=NB)
    reconstruct(Arr.from_numpy(coeffs), R)
    cells = dict(zip(triangle_cells(), R.cells(), strict=True))
    for i, a, b in triangle_cells():
        if a + b == 0:
            want = coeffs[i, 0]
        elif a < 2:
            want = coeffs[i, 2 * (a + b) - 1 + a]
        else:
            want = -cells[i, a - 2, b + 2]
        assert cells[i, a, b] == want
    decided(reconstruct)
    agrees(
        reconstruct,
        Schedule(reconstruct),
        {"coeffs": Arr.from_numpy(coeffs), "R": Arr.zeros(TRIANGLE, nb=NB)},
    )


def evaluate_inputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    counts = np.array([2, 0, 3], dtype=np.int64)
    total = int(counts.sum())
    return {
        "R": Arr.from_cells(TRIANGLE, rng.normal(size=len(triangle_cells())), nb=NB),
        "cnt": Arr.from_numpy(counts),
        "tx": Arr.ragged(counts, values=rng.random(total)),
        "ty": Arr.ragged(counts, values=rng.random(total)),
        "pot": Arr.ragged(counts),
        "inv_fact": Arr.from_numpy(1.0 / np.array([1.0, 1.0, 2.0, 6.0])),
    }


def test_a_where_array_beside_a_ragged_family_with_powers_of_binders() -> None:
    inputs = evaluate_inputs()
    evaluate(**inputs)
    cells = dict(zip(triangle_cells(), inputs["R"].cells(), strict=True))
    offsets = inputs["tx"].offsets
    tx, ty = inputs["tx"].numpy(), inputs["ty"].numpy()
    inv = inputs["inv_fact"].numpy()
    for i in range(NB):
        for t in range(offsets[i], offsets[i + 1]):
            want = sum(
                cells[i, a, b] * tx[t] ** a * ty[t] ** b * inv[a] * inv[b]
                for a in range(Q)
                for b in range(Q - a)
            )
            assert np.isclose(inputs["pot"].numpy()[t], want)
    # The sum is one reduction over the triangle, not terms unrolled by the
    # trace, so its binders are the exponents.
    (stmt,) = evaluate.term.stmts
    assert stmt.expr.inames == ("a", "b")
    decided(evaluate)
    agrees(evaluate, Schedule(evaluate).split(evaluate.i, 2), evaluate_inputs(1))


def test_a_loop_variable_exponent_in_a_sum_nested_in_a_sum() -> None:
    rng = np.random.default_rng(2)
    x, c = rng.random(5) + 0.5, rng.normal(size=(Q, Q))
    out = Arr.zeros(5)
    nested_powers(Arr.from_numpy(x), Arr.from_numpy(c), out)
    want = [
        sum(c[a, b] * xt**a * xt ** (a - b) for a in range(Q) for b in range(a + 1))
        for xt in x
    ]
    assert np.allclose(out.numpy(), want)
    decided(nested_powers)
    inputs = {"x": Arr.from_numpy(x), "c": Arr.from_numpy(c), "out": Arr.zeros(5)}
    agrees(nested_powers, Schedule(nested_powers), inputs)


def test_four_dense_axes_run_without_loopys_parameter_warnings() -> None:
    rng = np.random.default_rng(3)
    F, u = rng.normal(size=(3, 3)), rng.normal(size=(2, 3, 3, 3))
    out = Arr.from_numpy(np.zeros((2, 3, 3, 3)))
    along_r(Arr.from_numpy(F), Arr.from_numpy(u), out)
    assert np.allclose(out.numpy(), np.einsum("as,esjk->eajk", F, u))
    decided(along_r)
    # loopy's invoker warned that it could not find m from u's strides, m * m,
    # on the first run of the kernel, though it found it from the shape. With
    # loopy's caches off, every run is a first one.
    import loopy as lp
    from loopy.diagnostic import ParameterFinderWarning

    with lp.CacheMode(False), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LoopyExecutor().run(
            Schedule(along_r).split(along_r.e, 2),
            F=F,
            u=u,
            out=np.zeros((2, 3, 3, 3)),
        )
    assert not [w for w in caught if issubclass(w.category, ParameterFinderWarning)]
    agrees(
        along_r,
        Schedule(along_r).split(along_r.e, 2),
        {
            "F": Arr.from_numpy(F),
            "u": Arr.from_numpy(u),
            "out": Arr.from_numpy(np.zeros((2, 3, 3, 3))),
        },
    )
