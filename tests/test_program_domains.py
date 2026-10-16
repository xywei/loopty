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
from lanky.terms import Var

from lines import where
from loopty import Arr, Schedule, TraceError, kernel, program, reduce_sum
from loopty.executor import LoopyExecutor
from loopty.faithful import KIND
from loopty.oracle import IslOracle
from loopty.schedule import IllegalCast

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
    for schedule in (Schedule(through), Schedule(through).pack(through.L)):
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
    fused = Schedule(declared).fuse(declared.fill.i, declared.rowsum.p)
    assert all(f.status is Status.DECIDED for f in fused.facts())
    fact = LoopyExecutor().differential(declared, fused, _inputs())
    assert fact.status is Status.TESTED


def test_a_window_of_an_array_over_a_domain_keeps_one_row_of_it() -> None:
    # A placement takes the domain's binders, and the cells it places are the
    # domain's points (#244): fused on their rows, fill and rowsum keep one
    # row of the triangle, n cells, found along the order or given.
    fused = Schedule(K.declared).fuse(K.declared.fill.i, K.declared.rowsum.p)
    windowed = fused.contract(K.declared.f)
    window = windowed.windows["f"]
    assert repr(window.placement) == "lambda i, j: j"
    assert [str(extent) for extent in window.extents] == ["n"]
    # the window rests on the fact the schedule is made with, that no read
    # sees the zeros f is made with (#290)
    made = windowed.facts()[0]
    assert (made.kind, made.status) == ("definedness", Status.DECIDED)
    assert [(f.kind, f.status) for f in windowed.facts()[-1:]] == [
        ("no-clobber", Status.DECIDED),
    ]
    assert windowed.facts()[-1].rests_on == (made.id,)
    assert windowed.facts()[-1].statement == (
        "no read of declared.f sees its slot overwritten after "
        "contract(declared.f, lambda i, j: j): declared.f kept in n cells, "
        "f[i, j] in slot j"
    )
    fact = LoopyExecutor().differential(K.declared, windowed, _inputs())
    assert fact.status is Status.TESTED
    # one statement stores f, so computing it where it is read is the other way
    substituted = fused.substitute(K.declared.f)
    fact = LoopyExecutor().differential(K.declared, substituted, _inputs())
    assert fact.status is Status.TESTED
    # one cell for the row is too few, and the refusal names the window
    with pytest.raises(IllegalCast) as caught:
        fused.contract(K.declared.f, lambda i, j: 0)
    rowsum = where(K.rowsum, "e[p] = reduce_sum(")
    fill = where(K.fill, "f[i, j] = x[i] * x[j]")
    assert str(caught.value) == (
        "contract(declared.f, lambda i, j: 0) illegal: e[p] = reduce_sum(f[p, q] "
        "for q "
        f"in f.dom[p]) ({rowsum}) at p=2 reads f[2, 0], which f[i, j] = x[i] * "
        f"x[j] ({fill}) at i=2, j=0 stored in slot 0, after the same statement at "
        "i=2, j=1 stored f[2, 1] there (at n=3, isl's choice); "
        "contract(declared.f) finds the smallest window that keeps every read, "
        "lambda i, j: j"
    )
    # before the fusion every row is read after the last is stored
    with pytest.raises(ValueError, match="no window is smaller than declared.f"):
        Schedule(K.declared).contract(K.declared.f)


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


def test_a_declared_length_the_sizes_make_negative_is_refused_both_ways() -> None:
    # numpy refuses Arr.zeros((-1,)) when the program makes it, and the
    # compiled program would declare the array with that length.
    @kernel
    def head(x: Arr[Fin[n], Real], y: Arr[Fin[k], Real]):  # noqa: F821
        for p in y.dom:
            if p + 3 < x.dom.size:
                y[p] = x[p + 3]

    @kernel
    def total(y: Arr[Fin[k], Real], e: Arr[Fin[1], Real]):  # noqa: F821
        for p in e.dom:
            e[p] = reduce_sum(y[q] for q in y.dom)

    @program
    def tail(x, e):
        y = Arr.zeros((x.dom.size - 3,))
        head(x, y)
        total(y, e)

    def inputs(count: int) -> dict:
        return {"x": Arr.from_numpy(np.arange(1.0, count + 1.0)), "e": Arr.zeros(1)}

    with pytest.raises(ValueError, match="negative dimensions"):
        tail(**inputs(2))
    with pytest.raises(ValueError) as caught:
        LoopyExecutor().run(tail, **inputs(2))
    assert "with axis 0 of length" in str(caught.value)
    assert "which is -1 at" in str(caught.value)
    assert "natively numpy refuses a negative length" in str(caught.value)
    fact = LoopyExecutor().differential(tail, Schedule(tail), inputs(5))
    assert fact.status is Status.TESTED


@kernel
def column_sums(x: Arr[Fin[n], Fin[m], Real], y: Arr[Fin[m], Real]):  # noqa: F821
    """Each column of ``x`` summed."""
    for j in y.dom:
        y[j] = reduce_sum(x[i, j] for i in x.dom)


@kernel
def doubled(y: Arr[Fin[m], Real], e: Arr[Fin[m], Real]):  # noqa: F821
    """``2 y``."""
    for j in e.dom:
        e[j] = 2.0 * y[j]


def test_a_fiber_a_size_is_read_off_has_to_exist_as_natively() -> None:
    # #250: x.dom[2].size is the length of x's second axis where x has a row
    # 2, and natively x.dom[2] raises IndexError where it has none. The
    # composer kept the axis and not the index, so the compiled program
    # ran on two rows.
    @program
    def third_row(x, e):
        y = Arr.zeros((x.dom[2].size,))
        column_sums(x, y)
        doubled(y, e)

    def make(rows: int) -> dict:
        return {"x": Arr.from_numpy(np.ones((rows, 4))), "e": Arr.zeros(4)}

    (fiber,) = third_row.term.fibers
    assert (fiber.spelled, fiber.made, fiber.index) == ("x.dom[2]", "y", (2,))
    fact = LoopyExecutor().differential(third_row, Schedule(third_row), make(3))
    assert fact.status is Status.TESTED
    with pytest.raises(IndexError, match="index 2 out of range"):
        third_row(**make(2))
    for rows in (2, 0):
        with pytest.raises(IndexError) as caught:
            LoopyExecutor().run(third_row, **make(rows))
        message = str(caught.value)
        assert "with x.dom[2].size, and axis 0 of x has no index 2" in message
        assert f"which is {rows} at n = {rows}" in message
        assert "Natively x.dom[2] raises IndexError" in message

    # Out of range at every size, the term is refused.
    @kernel
    def three_rows(x: Arr[Fin[3], Fin[m], Real], y: Arr[Fin[m], Real]):  # noqa: F821
        for j in y.dom:
            y[j] = x[0, j] + x[2, j]

    @program
    def fifth_row(x, e):
        y = Arr.zeros((x.dom[5].size,))
        three_rows(x, y)
        doubled(y, e)

    with pytest.raises(TraceError, match=r"axis 0 of x has 3 cells .*no fiber"):
        fifth_row.term

    @program
    def last_row(x, e):
        y = Arr.zeros((x.dom[-1].size,))
        column_sums(x, y)
        doubled(y, e)

    with pytest.raises(IndexError):
        last_row(**make(3))
    with pytest.raises(TraceError, match="x.dom.-1., a fiber at a negative index"):
        last_row.term


@kernel
def two_halves(x: Arr[Fin[n], Real], u: Arr[Fin[n] + Fin[n], Real]):  # noqa: F821
    """``x`` and ``2 x``, the two pieces of a union, each of ``x``'s length."""
    for i in x.dom:
        u[0, i] = x[i]
        u[1, i] = 2.0 * x[i]


@kernel
def both_halves(u: Arr[Fin[m] + Fin[m], Real], e: Arr[Fin[1], Real]):  # noqa: F821
    """Every cell of both pieces summed: only ``u`` says what ``m`` is."""
    for k in e.dom:
        e[k] = reduce_sum(u[0, i] + u[1, i] for i in u.dom[0])


def test_a_union_of_other_sizes_names_them_natively_too() -> None:
    # #249: the program unifies m with n piece by piece, and natively the
    # second kernel's contract read no size off a union, so it could not tell
    # what m is and refused the call.
    @program
    def halves(x, u, e):
        two_halves(x, u)
        both_halves(u, e)

    def make() -> dict:
        return {
            "x": Arr.from_numpy(np.arange(1.0, 4.0)),
            "u": Arr.zeros(two_halves.arg_types["u"].domain, n=3),
            "e": Arr.zeros(1),
        }

    alone = make()
    both_halves(alone["u"], alone["e"])
    native = make()
    halves(**native)
    np.testing.assert_allclose(native["e"].numpy(), [18.0])
    compiled = make()
    LoopyExecutor().run(halves, **compiled)
    np.testing.assert_allclose(compiled["e"].numpy(), [18.0])
    fact = LoopyExecutor().differential(halves, Schedule(halves), make())
    assert fact.status is Status.TESTED
    # Pieces of other lengths are other points, and still refused.
    other = Arr.zeros(Fin[Var("a")] + Fin[Var("b")], a=3, b=4)
    with pytest.raises(ValueError, match="the argument u is over"):
        both_halves(other, Arr.zeros(1))


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
