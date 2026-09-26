"""Array arguments over polyhedral domains: written, iterated, checked, lowered.

``Where[...]`` cuts a box by constraints, ``Sigma[...]`` is a sum with affine
fibers, and ``Fin[n] + Fin[m]`` is a union of pieces (loopty #12). Each kind is
annotated and iterated, refused at the boundary when an argument is over other
points, decided in bounds over its exact set rather than the box around it, and
lowered and run on the C target in both layouts, agreeing with the native run.
The kernels are in ``tests/kernels/domains.py``.
"""

from __future__ import annotations

from pathlib import Path

import islpy as isl
import numpy as np
import pytest
from lanky.check import import_path
from lanky.ledger import Status
from lanky.prelude import Fin, Real, SumType
from lanky.terms import Var, evaluate_annotations

from loopty import Arr, Schedule, Sigma, Where, reduce_sum
from loopty import typing as rules
from loopty.domain import Polyhedron, Union, index_domain
from loopty.faithful import KIND
from loopty.interpret import interpret
from loopty.oracle import IslOracle
from loopty.trace import TraceError, array_type, reductions_in, trace

KERNELS = Path(__file__).parent / "kernels"
K = import_path(str(KERNELS / "domains.py"))

i, j, n, m = Var("i"), Var("j"), Var("n"), Var("m")

#: The lower triangle, written in this module as a kernel file writes it.
TRIANGLE = Where[i : Fin[n], j : Fin[n], j < i]


def settled(facts):
    """The facts, each decided by the isl oracle when it takes it."""
    oracle = IslOracle()
    return [
        (oracle.establish(fact) or fact) if oracle.can_establish(fact) else fact
        for fact in facts
    ]


def isl_set(text: str) -> isl.Set:
    return isl.Set(text)


# {{{ the written forms


def test_where_takes_binders_and_then_the_constraints_that_cut_their_box() -> None:
    assert isinstance(TRIANGLE, Polyhedron)
    assert TRIANGLE.names == ("i", "j")
    assert TRIANGLE.ndim == 2
    assert TRIANGLE.size_names() == {"n"}
    assert str(TRIANGLE) == "Where[i: Fin(n), j: Fin(n), j < i]"
    band = Where[i : Fin[n], j : Fin[n], (i - j <= 1) & (j - i <= 1)]
    # A conjunction is kept as its comparisons.
    assert len(band.constraints) == 2


def test_the_annotation_is_evaluated_with_its_binders_invented() -> None:
    annotation = evaluate_annotations(K.pairs.fn)["f"]
    arrtype = array_type(annotation, {"f": annotation}, "f")
    assert arrtype.axes == ()
    assert arrtype.ndim == 2
    assert arrtype.domain == TRIANGLE


def test_sigma_takes_named_binders_and_an_unnamed_last_fiber() -> None:
    total = Sigma[i : Fin[n], Fin[i + 1]]
    assert total.ndim == 2
    assert str(total) == "Sigma[i: Fin(n), Fin(i + 1)]"
    # The fiber gets a name nothing in the domain uses.
    assert total.names[1] not in {"i", "n"}
    assert total.isl_set().is_equal(
        isl_set("[n] -> { [a0, a1] : 0 <= a1 <= a0 < n }")
    )


def test_a_sum_of_index_types_is_a_union_of_pieces() -> None:
    both = Fin[n] + Fin[m]
    assert isinstance(both, SumType)
    union = index_domain(both)
    assert isinstance(union, Union)
    assert union.ndim == 2
    assert str(union) == "Fin(n) + Fin(m)"
    # Each piece assumes its own sizes non-negative, as a loop nest does.
    assert union.isl_set().is_equal(
        isl_set("[n, m] -> { [0, a1] : 0 <= a1 < n; [1, a1] : 0 <= a1 < m }")
    )
    # A Where or a Sigma is a piece too, on either side of the '+'.
    mixed = index_domain(Fin[n] + Where[i : Fin[m], i >= 1])
    assert isinstance(mixed, Union) and len(mixed.pieces) == 2
    assert index_domain(Where[i : Fin[m], i >= 1] + Fin[n]).pieces[1].spelling == (
        "Fin"
    )


@pytest.mark.parametrize(
    ("build", "words"),
    [
        (lambda: Where[i : Real, j : Fin[n]], "ranges over Fin"),
        (lambda: Where[i : Fin[n], j : Fin[n], i != j], "!="),
        (lambda: Where[i : Fin[n], j : Fin[n], (i < j) | (j < i)], "disjunction"),
        (lambda: Where[i : Fin[n], j : Fin[n], i * j < n], "quasi-affine"),
        (lambda: Where[i : Fin[n], i < n, j : Fin[n]], "after a constraint"),
        (lambda: Where[i : Fin[n], i : Fin[n]], "twice"),
        (lambda: Where[i : Fin[j], j : Fin[n]], "bound at or after"),
        (lambda: Where[i : Fin[n], j : Fin[i * i]], "not linear"),
        (lambda: Where[i : Fin[n], j : Fin[n], j + 1], "not a constraint"),
        (lambda: Where[3 : Fin[n]], "not a name"),
        (lambda: Sigma[Fin[n], i : Fin[n]], "only the last part"),
        (lambda: Sigma[i : Fin[n], i < n], "neither"),
        (lambda: index_domain(Fin[n] + Where[i : Fin[n], j : Fin[n]]), "axes"),
    ],
)
def test_a_domain_isl_cannot_state_exactly_is_refused(build, words) -> None:
    with pytest.raises(TypeError, match=words):
        build()


def test_a_domain_beside_other_axes_is_refused_when_the_type_is_read() -> None:
    spec = Arr[TRIANGLE, Fin[n], Real]
    with pytest.raises(TraceError, match="whole index set"):
        array_type(spec, {"f": spec}, "f")


# }}}


# {{{ runtime arrays


def test_an_array_over_the_triangle_is_iterated_binder_by_binder() -> None:
    f = Arr.zeros(TRIANGLE, n=4)
    assert list(f.dom) == [0, 1, 2, 3]
    assert [list(f.dom[r]) for r in f.dom] == [[], [0], [0, 1], [0, 1, 2]]
    # ``size`` is the binder's bound, as it is in a trace; ``len`` the points.
    assert f.dom[2].size == 4 and len(f.dom[2]) == 2
    assert f.sizes == {"n": 4}
    assert f.domain.count == 6


def test_a_cell_of_the_box_outside_the_domain_is_not_a_cell() -> None:
    for storage in ("box", "packed"):
        f = Arr.zeros(TRIANGLE, n=4, storage=storage)
        f[2, 1] = 5.0
        assert f[2, 1] == 5.0
        with pytest.raises(IndexError, match="not a point"):
            f[1, 1]
        with pytest.raises(IndexError, match="not a point"):
            f[1, 1] = 1.0
        with pytest.raises(IndexError, match="one integer for each"):
            f[1]


def test_both_layouts_hold_the_same_cells_in_one_order() -> None:
    values = np.arange(6.0)
    boxed = Arr.from_cells(TRIANGLE, values, n=4)
    packed = Arr.from_cells(TRIANGLE, values, n=4, storage="packed")
    assert boxed.numpy().shape == (4, 4)
    assert packed.numpy().shape == (6,)
    assert np.array_equal(boxed.cells(), values)
    assert np.array_equal(packed.cells(), values)
    assert boxed[3, 1] == packed[3, 1] == 4.0
    assert np.array_equal(packed.stored("box"), boxed.numpy())
    assert np.array_equal(boxed.stored("packed"), packed.numpy())


def test_the_packed_table_keeps_a_row_where_it_starts_less_its_first_column() -> (
    None
):
    band = K.band_product.arg_types["b"].domain
    b = Arr.from_cells(band, np.arange(13.0), n=5, storage="packed")
    # Rows [0, 1], [0, 2], [1, 3], [2, 4], [3, 4] start at 0, 2, 5, 8, 11.
    assert b.table().tolist() == [0, 2, 4, 6, 8]
    assert b[2, 1] == 5.0 and b[4, 4] == 12.0
    upper = K.upper.arg_types["g"].domain
    g = Arr.from_cells(upper, np.arange(6.0), n=4, storage="packed")
    # The first row starts at column 1, so its entry is one before the buffer,
    # and the last row is empty, so its entry is where it would start.
    assert g.table().tolist() == [-1, 1, 2, 6]
    assert g[0, 1] == 0.0 and g[2, 3] == 5.0


def test_a_domain_whose_rows_skip_columns_cannot_be_packed() -> None:
    domain = K.even_columns.arg_types["s"].domain
    assert not domain.rows_are_intervals()
    assert TRIANGLE.rows_are_intervals()
    with pytest.raises(ValueError, match="not an interval"):
        Arr.zeros(domain, n=4, storage="packed")
    boxed = Arr.zeros(domain, n=4)
    assert [list(boxed.dom[0])] == [[0, 2]]


def test_an_array_over_a_union_is_indexed_by_piece_then_point() -> None:
    u = Arr.from_cells(Fin[2] + Fin[3], np.arange(5.0))
    assert list(u.dom) == [0, 1]
    assert list(u.dom[1]) == [0, 1, 2]
    assert u[0, 1] == 1.0 and u[1, 2] == 4.0
    with pytest.raises(IndexError):
        u[0, 2]
    with pytest.raises(IndexError, match="piece 2"):
        u.dom[2]


def test_the_sizes_of_a_domain_have_to_be_given() -> None:
    with pytest.raises(ValueError, match="names n"):
        Arr.zeros(TRIANGLE)
    with pytest.raises(ValueError, match="stored"):
        Arr.zeros(TRIANGLE, n=3, storage="rows")
    with pytest.raises(TypeError, match="not a domain"):
        Arr.zeros(Fin[3], n=3)


# }}}


# {{{ the term and its facts


def test_a_statement_in_a_fiber_loop_is_over_the_exact_domain() -> None:
    stmt = K.pairs.term.stmts[0]
    assert stmt.inames == ("i", "j")
    assert stmt.domain.is_equal(isl_set("[n] -> { [i, j] : 0 <= j < i < n }"))


def test_every_in_bounds_obligation_is_decided_over_the_exact_triangle() -> None:
    facts = settled(rules.facts_for(K.pairs.term, owner="pairs"))
    in_bounds = [fact for fact in facts if fact.kind == "in-bounds"]
    assert in_bounds
    assert all(fact.status is Status.DECIDED for fact in in_bounds)
    assert {fact.decided_by for fact in in_bounds} == {"isl"}
    # The cells f[k, p] is compared with are the triangle's, not the box's.
    column = next(f for f in in_bounds if "f[k, p]" in f.statement)
    cells = column.term.large
    assert cells.is_equal(isl_set("[n] -> { [a0, a1] : 0 <= a1 < a0 < n }"))


def test_an_access_in_the_box_but_outside_the_domain_is_refuted() -> None:
    """``f[i, i]`` is a cell of the ``n x n`` box and not of the triangle.

    The facts are stated over the domain and not over its bounding box: the
    same access against the box would be decided, and against the domain it
    is refuted, with the witness on the diagonal.
    """
    facts = settled(rules.facts_for(K.diagonal.term, owner="diagonal"))
    fact = next(f for f in facts if "f[i, i]" in f.statement)
    assert fact.status is Status.REFUTED
    assert fact.decided_by == "isl"
    witness = fact.provenance["witness"]
    assert witness[0] == witness[1]
    reached = fact.term.small
    box = isl_set("[n] -> { [a0, a1] : 0 <= a0 < n and 0 <= a1 < n }")
    reached, box = reached.align_params(box.get_space()), box.align_params(
        reached.get_space()
    )
    assert reached.is_subset(box)


def test_a_fiber_at_a_point_outside_the_domain_is_empty_in_both_runs() -> None:
    term = K.later_rows.term
    inner = term.stmts[1]
    assert inner.domain.is_equal(
        isl_set("[n] -> { [i, j] : 2 <= i < n and 0 <= j < i }")
    )
    fact = K.later_rows.facts()[-1]
    assert fact.kind == KIND
    assert fact.status is Status.TESTED, fact.provenance


def test_the_statement_order_is_kept_at_two_depths_of_one_fiber_loop() -> None:
    x = Arr.from_numpy(np.arange(5.0))
    domain = K.later_rows.arg_types["h"].domain
    for storage in ("box", "packed"):
        h = Arr.zeros(domain, n=5, storage=storage)
        y = Arr.zeros(Fin[5])
        K.later_rows(x, h, y)
        assert h[4, 3] == 3.0 and h[2, 0] == 0.0
        assert np.array_equal(y.numpy(), x.numpy())


def test_the_interpreter_runs_a_term_over_domains() -> None:
    n_ = 4
    x = Arr.from_numpy(np.arange(1.0, n_ + 1))
    f = Arr.zeros(TRIANGLE, n=n_, storage="packed")
    e = Arr.zeros(Fin[n_])
    interpret(K.pairs.term, {"x": x, "f": f, "e": e})
    want = np.array([x.numpy().sum() * v - v * v for v in x.numpy()])
    assert np.allclose(e.numpy(), want)
    assert f[3, 2] == 12.0


@pytest.mark.parametrize(
    "name",
    [
        "pairs",
        "symmetric_product",
        "band_product",
        "upper",
        "two_pieces",
        "every_piece",
        "later_rows",
        "offset_rows",
        "total",
    ],
)
def test_the_trace_of_a_kernel_over_a_domain_is_faithful(name: str) -> None:
    fact = getattr(K, name).facts()[-1]
    assert fact.kind == KIND
    assert fact.status is Status.TESTED, fact.provenance


def test_a_piece_of_a_union_is_chosen_by_a_python_integer() -> None:
    def symbolic_piece(u: Arr[Fin[n] + Fin[m], Real], w: Arr[Fin[2], Real]):  # noqa: F821
        for p in w.dom:
            for k in u.dom[0]:
                u[p, k] = 1.0

    def missing_piece(u: Arr[Fin[n] + Fin[m], Real]):  # noqa: F821
        for k in u.dom[0]:
            u[2, k] = 1.0

    def sum_of_pieces(u: Arr[Fin[n] + Fin[m], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        for r in y.dom:
            y[r] = reduce_sum(u[p, 0] for p in u.dom)

    for body, words in (
        (symbolic_piece, "Python integer"),
        (missing_piece, "has 2 pieces"),
        (sum_of_pieces, "Reduce over each piece"),
    ):
        with pytest.raises(TraceError, match=words):
            trace(body, evaluate_annotations(body))


def test_a_reduction_over_a_fiber_of_its_own_binder_is_over_the_triangle() -> None:
    (reduction,) = reductions_in(K.total.term.stmts[0].expr)
    assert reduction.inames == ("i", "j")
    assert reduction.domain.is_equal(
        isl_set("[n] -> { [r, i, j] : r = 0 and 0 <= j < i < n }")
    )


def test_a_fiber_taken_at_the_variable_of_a_closed_loop_is_refused() -> None:
    def escaped(
        x: Arr[Fin[n], Real],  # noqa: F821
        f: Arr[Where[i : Fin[n], j : Fin[n], j < i], Real],  # noqa: F821
    ):
        for r in x.dom:
            x[r] = 0.0
        for c in f.dom[r]:
            f[c + 1, c] = 1.0

    with pytest.raises(TraceError, match="outside that loop"):
        trace(escaped, evaluate_annotations(escaped))


def test_a_fiber_at_a_point_isl_cannot_state_is_refused() -> None:
    def indirect(
        p: Arr[Fin[n], Fin[n]],  # noqa: F821
        f: Arr[Where[i : Fin[n], j : Fin[n], j < i], Real],  # noqa: F821
    ):
        for r in p.dom:
            for c in f.dom[p[r]]:
                f[p[r], c] = 0.0

    with pytest.raises(TraceError, match="not quasi-affine"):
        trace(indirect, evaluate_annotations(indirect))


# }}}


# {{{ the boundary


def test_an_argument_over_other_points_is_refused_natively_and_compiled() -> None:
    from loopty.executor import LoopyExecutor

    x = Arr.from_numpy(np.arange(5.0))
    e = Arr.zeros(Fin[5])
    short = Arr.zeros(TRIANGLE, n=4)
    with pytest.raises(ValueError, match="its type says"):
        K.pairs(x, short, e)
    with pytest.raises(ValueError, match="its type says"):
        LoopyExecutor().run(K.pairs, x, short, e)
    with pytest.raises(ValueError, match="array over a domain"):
        K.pairs(x, np.zeros((5, 5)), e)
    diagonal = Where[i : Fin[n], j : Fin[n], j <= i]
    with pytest.raises(ValueError, match="is a point of the argument's"):
        K.pairs(x, Arr.zeros(diagonal, n=5), e)
    strict = Where[i : Fin[n], j : Fin[n], j < i - 1]
    with pytest.raises(ValueError, match="is a point of the declared domain"):
        K.pairs(x, Arr.zeros(strict, n=5), e)
    # And an array over a domain is not a box: x has no domain.
    with pytest.raises(ValueError, match="has no domain"):
        K.pairs(Arr.zeros(TRIANGLE, n=5), Arr.zeros(TRIANGLE, n=5), e)


def test_an_argument_over_the_same_points_written_otherwise_is_accepted() -> None:
    """The points are compared, not the spelling: ``Sigma[a: Fin[n], Fin[a]]``."""
    a = Var("a")
    same = Sigma[a : Fin[n], Fin[a]]
    x = Arr.from_numpy(np.arange(1.0, 5.0))
    f = Arr.zeros(same, n=4, storage="packed")
    e = Arr.zeros(Fin[4])
    K.pairs(x, f, e)
    assert f[3, 1] == 8.0


def test_a_scalar_the_domain_names_is_a_size_of_the_call() -> None:
    domain = K.offset_rows.arg_types["w"].domain
    x = Arr.from_numpy(np.arange(5.0))
    w = Arr.zeros(domain, n=5, k=2)
    K.offset_rows(2, x, w)
    assert w.cells().tolist() == [3.0, 4.0, 5.0]
    with pytest.raises(ValueError, match="its type says"):
        K.offset_rows(1, x, w)


# }}}


# {{{ lowering and the C target


def _arguments(name: str, storage: str, seed: int = 0) -> dict:
    """Inputs for one of the kernels, its arrays over domains stored so."""
    rng = np.random.default_rng(seed)
    kernel = getattr(K, name)
    sizes = {"n": 5, "m": 3, "k": 2}
    out: dict = {}
    for param, typ in kernel.arg_types.items():
        if not hasattr(typ, "domain"):
            out[param] = sizes[param]
            continue
        if typ.domain is not None:
            needed = {size: sizes[size] for size in typ.domain.size_names()}
            count = typ.domain.fixed(needed).count
            out[param] = Arr.from_cells(
                typ.domain, rng.normal(size=count), storage=storage, **needed
            )
            continue
        axis = typ.axes[0]
        extent = axis if isinstance(axis, int) else sizes[axis.name]
        out[param] = Arr.from_numpy(rng.normal(size=extent))
    return out


RUNS = [
    ("pairs", ("f",)),
    ("symmetric_product", ("a",)),
    ("band_product", ("b",)),
    ("upper", ("g",)),
    ("two_pieces", ("u",)),
    ("every_piece", ("u", "v")),
    ("later_rows", ("h",)),
    ("offset_rows", ("w",)),
    ("total", ("f",)),
]


@pytest.mark.parametrize(("name", "packable"), RUNS)
@pytest.mark.parametrize("storage", ["box", "packed"])
@pytest.mark.parametrize("pack", [False, True])
def test_each_kind_runs_on_the_c_target_and_agrees_with_the_native_run(
    name: str, packable: tuple[str, ...], storage: str, pack: bool
) -> None:
    from loopty.executor import LoopyExecutor

    kernel = getattr(K, name)
    schedule = Schedule(kernel)
    if pack:
        schedule = schedule.pack(*packable)
    fact = LoopyExecutor().differential(kernel, schedule, _arguments(name, storage))
    assert fact.status is Status.TESTED, fact.provenance


def test_the_run_writes_back_into_the_callers_layout_and_answers_cells() -> None:
    from loopty.executor import LoopyExecutor

    x = Arr.from_numpy(np.arange(1.0, 5.0))
    f = Arr.zeros(TRIANGLE, n=4)
    e = Arr.zeros(Fin[4])
    out = LoopyExecutor().run(Schedule(K.pairs).pack("f"), x=x, f=f, e=e)
    assert f.storage == "box" and f[3, 2] == 12.0
    assert np.array_equal(out["f"], f.cells())


def test_the_packed_triangle_is_read_through_its_table_of_row_starts() -> None:
    from loopty.executor import emit_code

    boxed = emit_code(Schedule(K.pairs))
    packed = emit_code(Schedule(K.pairs).pack("f"))
    assert "off_f" not in boxed
    assert "f[off_f[i] + j]" in packed
    assert "f[off_f[k] + p]" in packed


def test_pack_is_a_step_of_the_schedule() -> None:
    schedule = Schedule(K.pairs).split("p", 2).pack("f")
    assert schedule.history[-1] == "pack(f)"
    assert schedule.key.endswith(".pack('f')")
    assert repr(schedule).endswith(".pack(f)")
    assert schedule.lowering.storage == {"f": "packed"}
    assert schedule.lowering.tables == {"f": "off_f"}
    assert Schedule(K.pairs).lowering.storage == {"f": "box"}
    with pytest.raises(ValueError, match="not an array over"):
        Schedule(K.pairs).pack("x")


def test_packing_a_domain_whose_rows_skip_columns_is_refused_when_lowered() -> None:
    from loopty.lower import LoweringError, lower_generic

    with pytest.raises(LoweringError, match="not an interval"):
        Schedule(K.even_columns).pack("s")
    with pytest.raises(LoweringError, match="no layout to choose"):
        lower_generic(K.pairs.term, layouts={"x": "packed"})


# }}}
