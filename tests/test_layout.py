"""Checked layout maps: a declared storage index, and loopy's dim_tags.

A layout is a Python function from a cell's indices to its address, with
annotations that say which cells and which buffer (:mod:`loopty.layout`).
Each claims three facts, decided by isl here: every cell inside the buffer,
no two at one address, a cell at every address; and two layouts of the same
cells may be claimed to be one. ``tests/test_lean_checks.py`` proves them
where Lean is installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from lanky.ledger import Status

from loopty import Fin, IslOracle, Where, layout
from loopty.layout import LayoutError

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path):
    name = f"layout_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


EXAMPLE = load(ROOT / "examples" / "storage_layouts.py")


def decided(obj) -> list[tuple[str, str, str]]:
    oracle = IslOracle()
    out = []
    for fact in obj.facts():
        result = oracle.establish(fact) or fact
        out.append((fact.kind, result.status.value, fact.statement))
    return out


def test_sumpys_storage_index_is_one_for_one_onto_its_buffer() -> None:
    assert decided(EXAMPLE.storage_index) == [
        (
            "layout-in-bounds",
            "decided",
            "storage_index stores every cell at an address of Fin(9)",
        ),
        (
            "layout-one-for-one",
            "decided",
            "storage_index stores no two cells at one address",
        ),
        (
            "layout-onto",
            "decided",
            "storage_index stores a cell at every address of Fin(9)",
        ),
    ]
    assert [EXAMPLE.derivatives((a, s - a)) for s in (1, 2) for a in (0, 1)] == [
        1,
        2,
        3,
        4,
    ]


def test_grudges_dim_tags_store_every_cell_where_the_dof_array_does() -> None:
    assert [status for _, status, _ in decided(EXAMPLE.dof_array)] == ["decided"] * 3
    assert decided(EXAMPLE.as_dim_tags) == [
        (
            "layout-same",
            "decided",
            "dof_array stores every cell where dim_tags N3,N0,N1,N2 does",
        )
    ]
    assert EXAMPLE.as_dim_tags.facts()[0].where == "storage_layouts.py:101"


def test_a_wrong_nesting_is_refuted_with_the_cell_it_moves() -> None:
    (fact,) = EXAMPLE.dof_array.same_as("N3,N2,N1,N0").facts()
    result = IslOracle().establish(fact)
    assert result.status is Status.REFUTED
    assert result.provenance["reason"] == (
        "dof_array stores (e=0, a=1, b=0, c=2) at 19, and dim_tags N3,N2,N1,N0 "
        "at 11 (at ne=1)"
    )


def test_a_layout_that_folds_two_cells_or_leaves_an_address_is_refuted() -> None:
    def folded(i: Fin[4]) -> Fin[2]:  # noqa: F821
        return i // 2

    def boxed(i: Fin[3]) -> Fin[4]:  # noqa: F821
        return i

    assert [status for _, status, _ in decided(layout(folded))] == [
        "decided",
        "refuted",
        "decided",
    ]
    assert [status for _, status, _ in decided(layout(boxed))] == [
        "decided",
        "decided",
        "refuted",
    ]


def reasons(obj) -> list[tuple[str, str | None]]:
    oracle = IslOracle()
    out = []
    for fact in obj.facts():
        result = oracle.establish(fact) or fact
        out.append((result.status.value, result.provenance.get("reason")))
    return out


def test_a_refuted_layout_names_the_cells_and_the_address() -> None:
    def fold(a: Fin[4], b: Fin[4]) -> Fin[16]:  # noqa: F821
        return a + 2 * b

    def wide(a: Fin[4], b: Fin[5]) -> Fin[16]:  # noqa: F821
        return 4 * a + b

    def even(i: Fin[n]) -> Fin[2 * n]:  # noqa: F821
        return 2 * i

    assert reasons(layout(fold)) == [
        ("decided", None),
        ("refuted", "fold stores (a=1, b=3) and (a=3, b=2) both at 7"),
        ("refuted", "fold stores no cell at 15 of Fin(16)"),
    ]
    assert reasons(layout(wide))[:2] == [
        ("refuted", "wide stores (a=3, b=4) at 16, outside Fin(16)"),
        ("refuted", "wide stores (a=2, b=4) and (a=3, b=0) both at 12"),
    ]
    assert reasons(layout(even))[2] == (
        "refuted",
        "even stores no cell at 1 of Fin(2*n) (at n=1)",
    )


def test_two_pieces_at_one_address_are_refuted_with_both_cells() -> None:
    def low(cell: Where[i: Fin[4], i < 2]) -> Fin[4]:  # noqa: F821
        (i,) = cell
        return i

    def high(cell: Where[i: Fin[4], i >= 2]) -> Fin[4]:  # noqa: F821
        (i,) = cell
        return i - 1

    assert reasons(layout(low, high))[1:] == [
        ("refuted", "low stores (i=1) and (i=2) both at 1"),
        ("refuted", "low stores no cell at 3 of Fin(4)"),
    ]


def test_a_quasi_affine_layout_is_decided_one_for_one_or_refuted() -> None:
    def interleave(i: Fin[8]) -> Fin[8]:  # noqa: F821
        return (i % 2) * 4 + i // 2

    def halves(i: Fin[8]) -> Fin[8]:  # noqa: F821
        return (i % 2) * 4 + i // 4

    assert [status for status, _ in reasons(layout(interleave))] == ["decided"] * 3
    assert reasons(layout(halves))[1] == (
        "refuted",
        "halves stores (i=2) and (i=0) both at 0",
    )


def test_a_size_in_a_layouts_body_is_said_to_be_one() -> None:
    def backwards(i: Fin[n]) -> Fin[n]:  # noqa: F821
        return n - 1 - i  # noqa: F821

    with pytest.raises(LayoutError) as caught:
        layout(backwards)
    assert str(caught.value) == (
        "backwards: the address reads the size n, which a layout cannot know: it "
        "is called with a cell's indices alone, in a kernel too; make the sizes "
        "it computes with module constants, N = 8 and Fin[N] for Fin[n]"
    )


def test_a_kernel_reads_through_the_layout_it_declares() -> None:
    inputs = EXAMPLE.example_inputs()
    EXAMPLE.unflatten(**inputs["unflatten"])
    assert EXAMPLE.dof_array(1, 2, 0, 1) == 27 + 2 + 9
    with pytest.raises(LayoutError, match="call the one whose cells you are at"):
        EXAMPLE.storage_index(0, 0)


def test_lanky_check_collects_the_layouts_of_a_file() -> None:
    from lanky.check import check_path

    ledger = check_path(ROOT / "examples" / "storage_layouts.py")
    kinds = sorted(fact.kind for fact in ledger if fact.kind.startswith("layout"))
    assert kinds == sorted(
        ["layout-in-bounds", "layout-one-for-one", "layout-onto"] * 2
        + ["layout-same"]
    )
    assert all(
        fact.status is Status.DECIDED
        for fact in ledger
        if fact.kind.startswith("layout")
    )


@pytest.mark.parametrize(
    ("fn", "words"),
    [
        (lambda: _no_return(), "the return annotation says which addresses"),
        (lambda: _squared(), "is not a sum of the indices times numbers"),
        (lambda: _unannotated(), "each parameter is an index of the cell"),
        (lambda: _two_ranks(), "cells of different ranks"),
        (lambda: _overlapping(), "store one cell at two addresses"),
        (
            lambda: EXAMPLE.dof_array.same_as("N3,N0,N1"),
            "give 3 levels for the 4 axes",
        ),
        (lambda: EXAMPLE.dof_array.same_as("N3,M0,N1,N2"), "'M0' is no nesting"),
        (
            lambda: EXAMPLE.dof_array.same_as("N0,N1,N2,N3"),
            "the axis e of dof_array has the extent ne, which is not a number",
        ),
    ],
)
def test_a_layout_is_written_as_a_function_of_its_cells(fn, words) -> None:
    with pytest.raises(LayoutError, match=words):
        fn()


def _no_return():
    def flat(i: Fin[4]):  # noqa: F821
        return i

    return layout(flat)


def _squared():
    def square(i: Fin[4]) -> Fin[16]:  # noqa: F821
        return i * i

    return layout(square)


def _unannotated():
    def plain(i) -> Fin[4]:  # noqa: F821
        return i

    return layout(plain)


def _two_ranks():
    def one(i: Fin[2]) -> Fin[6]:  # noqa: F821
        return i

    def two(i: Fin[2], j: Fin[2]) -> Fin[6]:  # noqa: F821
        return 2 + 2 * i + j

    return layout(one, two)


def _overlapping():
    def first(cell: Where[i: Fin[3], i < 2]) -> Fin[4]:  # noqa: F821
        (i,) = cell
        return i

    def second(cell: Where[i: Fin[3], i >= 1]) -> Fin[4]:  # noqa: F821
        (i,) = cell
        return i + 1

    return layout(first, second)


def test_the_statements_lean_is_given_without_lean() -> None:
    pytest.importorskip("lanky.obligations", reason="no statements for Lean here")
    bounds, injective, onto = EXAMPLE.storage_index.facts()
    assert [p.name for p in bounds.term.lean_obligation().parts] == [
        "in_bounds_0",
        "in_bounds_1",
    ]
    assert [p.name for p in injective.term.lean_obligation().parts] == [
        "one_for_one_0_0",
        "one_for_one_0_1",
        "one_for_one_1_1",
    ]
    (part,) = onto.term.lean_obligation().parts
    # isl's inverse: the centre at 0, and a derivative where the address is
    # odd or even, with // 2 and % 2
    assert {piece.disjunct for piece in part.hints.witness} == {0, 1}
    (same,) = EXAMPLE.as_dim_tags.facts()
    assert [p.name for p in same.term.lean_obligation().parts] == ["same_0_0"]
