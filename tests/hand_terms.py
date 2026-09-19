"""Hand-built terms, the fixtures the lowering and schedule tests run on.

The tracer builds terms from Python bodies; these are the same terms written out
by hand. Writing them by hand is deliberate. It pins the Term IR down as a
contract rather than as whatever the tracer happens to emit, it lets lowering and
scheduling be tested without the tracer, and it keeps the four shapes that matter
in one readable place: a dense elementwise kernel, a two-dimensional permutation,
a stencil with a loop-carried dependence in time, and a ragged reduction.

Sizes are symbolic pymbolic variables, as they are in a real trace; the test
supplies concrete values only when it runs the compiled code.
"""

from __future__ import annotations

import islpy as isl
import numpy as np
import pymbolic.primitives as prim

from loopty.term import Access, ArrType, Reduction, Stmt, Term

V = prim.Variable


def S(name: str, *indices: object) -> prim.Subscript:
    """``name[indices]``: an array reference inside a statement's expression.

    :class:`~loopty.term.Access` is the dataclass a statement's *assignee* is,
    and a pymbolic ``Subscript`` is what the same reference looks like inside an
    expression tree, where it has to take part in arithmetic. Lowering accepts
    either.
    """
    return prim.Subscript(prim.Variable(name), tuple(indices))

REAL = np.dtype(np.float64)
INT = np.dtype(np.int32)


def dense(*axes: object, dtype: object = REAL) -> ArrType:
    """A dense array type with the given axis sizes."""
    return ArrType(axes=tuple(axes), dtype=dtype, ragged=(False,) * len(axes))


def axpy_term() -> Term:
    """``z[i] = a * x[i] + y[i]`` over ``Fin[n]``, with a scalar parameter."""
    domain = isl.Set("[n] -> { [i] : 0 <= i < n }")
    stmt = Stmt(
        id="S0",
        inames=("i",),
        domain=domain,
        assignee=Access("z", (V("i"),)),
        expr=V("a") * S("x", V("i")) + S("y", V("i")),
        kind="assign",
        guard=None,
        where="hand_terms.py:axpy",
    )
    return Term(
        name="axpy",
        params=(
            ("a", np.dtype(np.float64)),
            ("x", dense(V("n"))),
            ("y", dense(V("n"))),
            ("z", dense(V("n"))),
        ),
        sizes=("n",),
        stmts=(stmt,),
        post=None,
    )


def transpose_term() -> Term:
    """``b[j, i] = a[i, j]``: a two-dimensional permutation of index space."""
    domain = isl.Set("[n, m] -> { [i, j] : 0 <= i < n and 0 <= j < m }")
    stmt = Stmt(
        id="S0",
        inames=("i", "j"),
        domain=domain,
        assignee=Access("b", (V("j"), V("i"))),
        expr=S("a", V("i"), V("j")),
        kind="assign",
        guard=None,
        where="hand_terms.py:transpose",
    )
    return Term(
        name="transpose",
        params=(("a", dense(V("n"), V("m"))), ("b", dense(V("m"), V("n")))),
        sizes=("n", "m"),
        stmts=(stmt,),
        post=None,
    )


def jacobi_term() -> Term:
    """1D Jacobi in time: ``u[t+1, i] = (u[t, i-1] + u[t, i+1]) / 2``.

    The dependence ``(t, i) -> (t+1, i-1)`` is what makes a rectangular tiling of
    ``(t, i)`` illegal and a skew by ``t`` the fix, which is the whole point of
    the stencil demo.
    """
    domain = isl.Set("[nt, nx] -> { [t, i] : 0 <= t < nt - 1 and 1 <= i < nx - 1 }")
    stmt = Stmt(
        id="S0",
        inames=("t", "i"),
        domain=domain,
        assignee=Access("u", (V("t") + 1, V("i"))),
        expr=(S("u", V("t"), V("i") - 1) + S("u", V("t"), V("i") + 1)) / 2,
        kind="assign",
        guard=None,
        where="hand_terms.py:jacobi",
    )
    return Term(
        name="jacobi",
        params=(("u", dense(V("nt"), V("nx"))),),
        sizes=("nt", "nx"),
        stmts=(stmt,),
        post=None,
    )


def spmv_term(exactness: str = "reassoc") -> Term:
    """Ragged CSR product, the reduction form.

    ``val`` and ``col`` are ragged in their second axis, whose bound at row ``r``
    is ``cnt[r]``; storage is flat and indexed through ``off``. The reduction's
    domain has the row as a leading dimension, so the bound can depend on it, and
    lowering turns that dimension into the parameter ``cnt_r``.

    ``exactness`` is the accumulation's floating-point contract, the tracer's
    choice for a ``Real`` reduction being ``reassoc``. Pass ``"exact"`` to see a
    reduction tree refused.
    """
    row_domain = isl.Set("[n] -> { [r] : 0 <= r < n }")
    reduction_domain = isl.Set(
        "[n, cnt_r] -> { [r, j] : 0 <= r < n and 0 <= j < cnt_r }"
    )
    body = S("val", V("r"), V("j")) * S("x", S("col", V("r"), V("j")))
    reduction = Reduction(
        op="sum",
        inames=("j",),
        domain=reduction_domain,
        body=body,
        exactness=exactness,
    )
    stmt = Stmt(
        id="S0",
        inames=("r",),
        domain=row_domain,
        assignee=Access("y", (V("r"),)),
        expr=reduction,
        kind="assign",
        guard=None,
        where="hand_terms.py:spmv",
    )
    ragged_real = ArrType(axes=(V("n"), V("cnt")), dtype=REAL, ragged=(False, True))
    ragged_index = ArrType(axes=(V("n"), V("cnt")), dtype=INT, ragged=(False, True))
    return Term(
        name="spmv",
        params=(
            ("off", dense(V("n") + 1, dtype=INT)),
            ("col", ragged_index),
            ("val", ragged_real),
            ("x", dense(V("m"))),
            ("y", dense(V("n"))),
        ),
        sizes=("n", "m"),
        stmts=(stmt,),
        post=None,
    )


def spmv_accumulate_term() -> Term:
    """The same product written as an accumulation over a two-deep loop nest.

    ``y[r] = y[r] + val[r, j] * x[col[r, j]]`` with ``j`` an ordinary iname
    rather than a reduction iname. This is the shape a schedule reassociates:
    splitting ``j`` and running the pieces in parallel is only legal once the
    accumulation is marked ``reassoc``.

    ``expr`` is the whole right-hand side, ``y[r] + ...``, which is what
    ``kind="accumulate"`` means in :class:`~loopty.term.Stmt`; lowering refuses
    a term that records the increment alone.
    """
    domain = isl.Set("[n, cnt_r] -> { [r, j] : 0 <= r < n and 0 <= j < cnt_r }")
    stmt = Stmt(
        id="S0",
        inames=("r", "j"),
        domain=domain,
        assignee=Access("y", (V("r"),)),
        expr=S("y", V("r"))
        + S("val", V("r"), V("j")) * S("x", S("col", V("r"), V("j"))),
        kind="accumulate",
        guard=None,
        where="hand_terms.py:spmv_accumulate",
    )
    ragged_real = ArrType(axes=(V("n"), V("cnt")), dtype=REAL, ragged=(False, True))
    ragged_index = ArrType(axes=(V("n"), V("cnt")), dtype=INT, ragged=(False, True))
    return Term(
        name="spmv_acc",
        params=(
            ("off", dense(V("n") + 1, dtype=INT)),
            ("col", ragged_index),
            ("val", ragged_real),
            ("x", dense(V("m"))),
            ("y", dense(V("n"))),
        ),
        sizes=("n", "m"),
        stmts=(stmt,),
        post=None,
    )


def csr_example() -> dict:
    """A tiny CSR matrix and a vector, as numpy arrays keyed by parameter name.

    Three rows with two, zero and three stored entries, which exercises the empty
    fiber that a ragged type has to allow.
    """
    off = np.array([0, 2, 2, 5], dtype=np.int32)
    col = np.array([0, 1, 0, 2, 3], dtype=np.int32)
    val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    y = np.zeros(3)
    return {"off": off, "col": col, "val": val, "x": x, "y": y}


def csr_reference(off: np.ndarray, col: np.ndarray, val: np.ndarray, x: np.ndarray):
    """The numpy reference for :func:`csr_example`."""
    out = np.zeros(len(off) - 1)
    for r in range(len(off) - 1):
        for a in range(off[r], off[r + 1]):
            out[r] += val[a] * x[col[a]]
    return out


def jacobi_reference(u: np.ndarray) -> np.ndarray:
    """The numpy reference for :func:`jacobi_term`, run in place on a copy."""
    out = u.copy()
    nt, nx = out.shape
    for t in range(nt - 1):
        for i in range(1, nx - 1):
            out[t + 1, i] = (out[t, i - 1] + out[t, i + 1]) / 2
    return out
