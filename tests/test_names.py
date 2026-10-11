"""The names of isl dimensions: the check loopty makes, and how it renames.

loopty builds its isl objects mostly from text, and isl's reader breaks a name
silently: a tuple variable spelled like a parameter is read as that parameter
and comes back unnamed, as a map's dimension named like a declared size did
(#110). :func:`loopty.names.check_names` checks namedisl's invariants where
loopty makes such an object (every dimension named, no two alike, none named
like a parameter), and :func:`loopty.names.rename` is the one way loopty
renames dimensions and replaces parameters, all at once, which composing,
lowering and the native run share.
"""

from __future__ import annotations

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
import pytest

from loopty import Arr, Fin, kernel, reduce_sum
from loopty.compose import rename_set
from loopty.flow import domain_set
from loopty.idx import Layout, to_set
from loopty.names import NamingError, check_names, rename
from loopty.term import Access, Reduction, Stmt
from loopty.trace import TraceError

V = prim.Variable


def _points(obj: isl.Set, **values: int) -> list[tuple[int, ...]]:
    """The points of ``obj`` with its parameters fixed at ``values``."""
    for name, value in values.items():
        position = obj.find_dim_by_name(isl.dim_type.param, name)
        obj = obj.fix_val(isl.dim_type.param, position, value)
    points: list[tuple[int, ...]] = []
    obj.foreach_point(
        lambda point: points.append(
            tuple(
                point.get_coordinate_val(isl.dim_type.set, k).to_python()
                for k in range(obj.dim(isl.dim_type.set))
            )
        )
    )
    return sorted(points)


# {{{ the check


def test_a_dimension_isl_read_as_a_parameter_is_refused() -> None:
    # isl reads the tuple variable k as the parameter k: one point per k, of
    # a dimension with no name.
    read = isl.Set("[k] -> { [k] : 0 <= k < 5 }")
    assert read.get_dim_name(isl.dim_type.set, 0) is None
    with pytest.raises(NamingError, match="set dimension 0 has no name"):
        check_names(read, "a loop over k")


def test_two_dimensions_of_one_name_are_refused() -> None:
    square = isl.Set("[n] -> { [i, j] : 0 <= i < n and 0 <= j < n }")
    assert check_names(square) is square
    with pytest.raises(NamingError, match=r"i names more than one dimension"):
        check_names(square.set_dim_name(isl.dim_type.set, 1, "i"))
    with pytest.raises(
        NamingError,
        match=r"n names more than one dimension \(parameter, set dimension\)",
    ):
        check_names(square.set_dim_name(isl.dim_type.set, 1, "n"))


def test_a_positional_tuple_is_left_out() -> None:
    # The cells an access reaches are index expressions with no names.
    access = isl.Map("[n] -> { [i] -> [i + 1] : 0 <= i < n }")
    with pytest.raises(NamingError, match="out dimension 0 has no name"):
        check_names(access)
    assert check_names(access, positional=(isl.dim_type.out,)) is access


def test_a_dimension_isl_took_a_parameter_for_is_refused() -> None:
    # Given in the text as a tuple variable, j is read as the variable in
    # 0 <= j < j too: a well-named set with no points, and the size j gone.
    # Only the caller knows it meant j as a parameter.
    read = isl.Set("{ [j] : 0 <= j < j }")
    assert check_names(read) is read
    with pytest.raises(NamingError, match="j names a dimension and is meant as a"):
        check_names(read, "a sum over Fin[j]", meant={"j"})
    with pytest.raises(NamingError, match="the loops over r, i"):
        domain_set(("r", "i"), (V("i"), V("m")))
    with pytest.raises(NamingError, match="the loops over j"):
        domain_set(("j",), (V("j"),))
    # A bound may name the loops outside it.
    triangle = domain_set(("i", "j"), (V("n"), V("i")))
    assert triangle.get_var_names(isl.dim_type.param) == ["n"]


def test_a_statement_or_a_sum_is_refused_a_domain_that_breaks_them() -> None:
    read = isl.Set("[k] -> { [k] : 0 <= k < 5 }")
    with pytest.raises(NamingError, match="the domain of statement S0"):
        Stmt(
            id="S0",
            inames=("k",),
            domain=read,
            assignee=Access("y", (V("k"),)),
            expr=0.0,
            kind="assign",
            guard=None,
            where="hand.py:1",
        )
    good = isl.Set("[n] -> { [i] : 0 <= i < n }")
    with pytest.raises(NamingError, match="the loop domain of the sum"):
        Reduction(
            op="sum",
            inames=("i",),
            domain=good,
            body=V("i"),
            exactness="exact",
            loop_domain=read,
        )


# }}}


# {{{ renaming


def test_a_renaming_is_simultaneous() -> None:
    # One name after the other, a -> b would meet the b still there.
    ordered = isl.Set("[n] -> { [a, b] : 0 <= a < b < n }")
    swapped = rename(ordered, {"a": "b", "b": "a"})
    assert swapped.get_var_names(isl.dim_type.set) == ["b", "a"]
    assert swapped.is_equal(isl.Set("[n] -> { [b, a] : 0 <= b < a < n }"))
    sizes = isl.Set("[n, m] -> { [i] : 0 <= i < n and i < m }")
    assert rename(sizes, {"n": "m", "m": "n"}).is_equal(
        isl.Set("[m, n] -> { [i] : 0 <= i < m and i < n }")
    )
    # A placeholder is named clear of the dimensions the set has already.
    taken = isl.Set("[n] -> { [__loopty_moved_0, i] : 0 <= i < n }")
    moved = rename(taken, {"i": "j"})
    assert moved.get_var_names(isl.dim_type.set) == ["__loopty_moved_0", "j"]


def test_a_parameter_is_replaced_by_an_expression_or_a_value() -> None:
    rows = isl.Set("[n, m] -> { [i] : 0 <= i < n }")
    assert rename(rows, exprs={"n": V("m") + 1}).is_equal(
        isl.Set("[m] -> { [i] : 0 <= i <= m }")
    )
    fixed = rename(rows, exprs={"n": 4})
    assert fixed.get_var_names(isl.dim_type.param) == ["m"]
    assert _points(fixed, m=0) == [(0,), (1,), (2,), (3,)]
    # Renamed to a parameter it has, it is that parameter.
    assert rename(rows, {"n": "m"}).is_equal(isl.Set("[m] -> { [i] : 0 <= i < m }"))


def test_a_dimension_renamed_like_a_parameter_is_refused() -> None:
    rows = isl.Set("[n] -> { [i] : 0 <= i < n }")
    with pytest.raises(NamingError, match="a callee's domain"):
        rename_set(rows, {"i": "n"}, {})


# }}}


# {{{ where loopty names dimensions


def test_the_index_tuples_of_a_size_named_like_a_dimension() -> None:
    # to_set named its dimensions i0, i1, ...: over Fin[i0] that was
    # 0 <= i0 < i0, the parameter read for the dimension, and no points.
    i0 = V("i0")
    cells = to_set((Fin[i0], Fin[2]))
    assert cells.get_var_names(isl.dim_type.set) == ["i_0", "i_1"]
    assert _points(cells, i0=2) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    layout = Layout((Fin[i0], Fin[3])).to_map()
    assert layout.domain().get_var_names(isl.dim_type.set) == ["i_0", "i_1"]
    assert _points(layout.domain(), i0=1) == [(0, 0), (0, 1), (0, 2)]


@kernel
def nested_totals(
    a: Arr[Fin[n], Fin[m], np.float64],  # noqa: F821
    s: Arr[Fin[n], np.float64],  # noqa: F821
    t: Arr[Fin[1], np.float64],  # noqa: F821
):
    """Sums inside sums: binders as parameters of the inner domain."""
    for r in a.dom:
        s[r] = reduce_sum(reduce_sum(a[q, c] for c in a.dom[q]) for q in a.dom)
    t[0] = reduce_sum(reduce_sum(a[q, c] for c in a.dom[q] if c <= q) for q in a.dom)


def test_a_kernel_with_nested_sums_keeps_every_name_apart() -> None:
    # Every domain of the term, each sum's included, is checked when it is
    # recorded. The second statement's sums get binders of their own in the
    # kernel, q_0 and c_0 (note 8), and the inner sum's domain, whose
    # parameter q is the outer binder, follows it through the same rename,
    # which checks what it returns.
    from loopty.executor import LoopyExecutor, emit_code

    term = nested_totals.trace()
    for stmt in term.stmts:
        check_names(stmt.domain)
    a = np.arange(6.0).reshape(2, 3)
    out = LoopyExecutor().run(nested_totals, a=a, s=np.zeros(2), t=np.zeros(1))
    assert list(out["s"]) == [15.0, 15.0]
    assert list(out["t"]) == [7.0]
    assert "c_0 <= ((-1 + m >= 0 && -1 * m + q_0 >= 0) ? -1 + m : q_0)" in (
        emit_code(nested_totals)
    )


# }}}


# {{{ a binder named like a size


@kernel
def summed_by_its_size(
    x: Arr[Fin[j], np.float64],  # noqa: F821
    y: Arr[Fin[1], np.float64],  # noqa: F821
):
    """A sum by ``j`` over ``Fin[j]``."""
    y[0] = reduce_sum(x[j] for j in x.dom)


@kernel
def summed_past_its_size(
    x: Arr[Fin[j], np.float64],  # noqa: F821
    y: Arr[Fin[1], np.float64],  # noqa: F821
):
    """A sum by ``j`` over ``Fin[j]`` that reads past ``x``."""
    y[0] = reduce_sum(x[j + 5] for j in x.dom)


@kernel
def summed_by_the_rows(
    a: Arr[Fin[i], Fin[m], np.float64],  # noqa: F821
    s: Arr[Fin[i], np.float64],  # noqa: F821
):
    """A sum by ``i`` inside a loop bounded by the size ``i``."""
    for r in a.dom:
        s[r] = reduce_sum(a[r, i] for i in a.dom[r])


@kernel
def summed_after_a_loop(
    x: Arr[Fin[n], np.float64],  # noqa: F821
    y: Arr[Fin[n], np.float64],  # noqa: F821
):
    """A sum by the name of a loop that has ended."""
    for i in x.dom:
        y[i] = 2.0 * x[i]
    y[0] = reduce_sum(x[i] for i in x.dom)


def test_a_sum_by_a_binder_named_like_a_size_is_refused() -> None:
    # The domain of a sum by j over Fin[j] was { [j] : 0 <= j < j }: no
    # points, so the compiled sum was 0 where numpy's was not, and the
    # in-bounds fact of x[j + 5] was decided over the empty domain. A sum by
    # i inside a loop bounded by the size i took the loop's bound for its
    # binder, and loopy refused to build it. Each is refused at the trace now,
    # with the fix named.
    for kern in (summed_by_its_size, summed_past_its_size, summed_by_the_rows):
        with pytest.raises(TraceError, match="named like a size") as caught:
            kern.trace()
        assert str(caught.value).endswith("Rename the binder")
    # A binder may take the name of a loop that has ended.
    from loopty.executor import LoopyExecutor

    x = np.arange(4.0)
    out = LoopyExecutor().run(summed_after_a_loop, x=x, y=np.zeros(4))
    assert list(out["y"]) == [6.0, 2.0, 4.0, 6.0]


# }}}


# {{{ the points the compiled run's refusal asks isl about (#153)


def test_the_points_a_refusal_asks_about_are_checked() -> None:
    # loopty.refusal states to isl the points of a statement's loops where
    # loop arithmetic leaves a numpy integer's type, reading the sizes the
    # arithmetic names as parameters. A loop named like one of them would
    # take it for itself, as j did over Fin[j], and the size would be gone:
    # n - 3 over a loop n would be asked at the loop's n.
    from loopty.refusal import _leaves

    term = summed_after_a_loop.trace()
    uint32 = np.dtype(np.uint32)
    loops = isl.Set("[n] -> { [i] : 0 <= i < n }")
    found = _leaves(V("i") - 3, uint32, loops, {"n": 4}, term, ())
    assert found is not None and found.endswith("at i = 0")
    assert _leaves(V("n") - 3, uint32, loops, {"n": 4}, term, ()) is None
    misnamed = isl.Set("{ [n] : 0 <= n < 4 }")
    with pytest.raises(NamingError, match="n names a dimension and is meant as a"):
        _leaves(V("n") - 3, uint32, misnamed, {"n": 4}, term, ())


# }}}
