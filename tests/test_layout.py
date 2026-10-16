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

#: A layout at every size that pads its rows: at n = 1 it stores every row at 0.
PADDED = [
    ("tested", None),
    ("refuted", "padded stores (i=0, j=1) and (i=0, j=0) both at 0 (at m=2, n=1)"),
    ("refuted", "padded stores no cell at 1 of Fin(n*m) (at m=2, n=1)"),
]

#: The DOFArray with the element innermost, which makes ne a stride.
NESTING = (
    "dof_array stores (e=1, a=1, b=2, c=2) at 52, and dim_tags N0,N1,N2,N3 at 51 "
    "(at ne=2)"
)


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
    assert EXAMPLE.as_dim_tags.facts()[0].where == "storage_layouts.py:99"


def test_a_nesting_with_a_size_inside_is_refuted_at_a_size() -> None:
    # e innermost makes ne a stride, so isl asks the claim at ne from 1 to 6
    (fact,) = EXAMPLE.dof_array.same_as("N0,N1,N2,N3").facts()
    result = IslOracle().establish(fact)
    assert result.status is Status.REFUTED
    assert result.provenance["reason"] == NESTING


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


def test_a_size_in_a_layouts_body_is_the_size_and_a_call_gives_it() -> None:
    def backwards(i: Fin[n]) -> Fin[n]:  # noqa: F821
        return n - 1 - i  # noqa: F821

    checked = layout(backwards)
    # n - 1 - i is affine, so isl decides it for every n
    assert [status for status, _ in reasons(checked)] == ["decided"] * 3
    assert checked(1, n=4) == 2
    with pytest.raises(LayoutError) as caught:
        checked(1)
    assert str(caught.value) == (
        "backwards computes with the size n, which a call gives by name: "
        "backwards(i, n=...)"
    )


def dof_array(e: Fin[ne], i: Fin[n], j: Fin[n], k: Fin[n]) -> Fin[ne * n**3]:  # noqa: F821
    """grudge's DOFArray at every order, as ``examples/hexahedra.py`` declares it."""
    return n**3 * e + (i + n * j + n**2 * k)  # noqa: F821


def test_a_layout_at_every_size_is_tested_by_isl_at_a_few() -> None:
    every = layout(dof_array)
    assert every.polynomial and every.sampled() == ("n",)
    oracle = IslOracle()
    for fact in every.facts():
        result = oracle.establish(fact)
        assert result.status is Status.TESTED, fact.statement
        assert result.provenance["detail"] == (
            "decided by isl at n = 1 to 6; n multiplies an index in the address "
            "of dof_array, which isl cannot state for every n; Lean proves it "
            "for every n"
        )
    assert [fact.statement for fact in every.facts()] == [
        "dof_array stores every cell at an address of Fin(ne*n**3)",
        "dof_array stores no two cells at one address",
        "dof_array stores a cell at every address of Fin(ne*n**3)",
    ]
    # loopy's dim_tags at every order too, the strides products of extents
    (same,) = every.same_as("N3,N0,N1,N2").facts()
    assert oracle.establish(same).status is Status.TESTED
    # at a size, the layout is one isl states
    assert [status for status, _ in reasons(every.at(n=3))] == ["decided"] * 3
    assert every(1, 2, 0, 1, n=3) == 27 + 2 + 9
    # a dim_tags layout computes its own address, from its strides (Codex)
    assert every.same_as("N3,N0,N1,N2").second(1, 2, 0, 1, n=3) == 27 + 2 + 9
    assert every.same_as("N3,N2,N1,N0").second(0, 1, 0, 2, n=3) == 9 + 2
    with pytest.raises(LayoutError, match=r"isl cannot state; .*dof_array.at\(n=4\)"):
        every.relation()


def test_products_that_cancel_leave_a_layout_isl_states() -> None:
    # Codex: n * i - n * i + i was taken for a layout at every size, which
    # sampled no size and then asked isl a question it cannot state
    def cancelling(i: Fin[n]) -> Fin[n]:  # noqa: F821
        return n * i - n * i + i  # noqa: F821

    checked = layout(cancelling)
    assert not checked.polynomial
    assert [status for status, _ in reasons(checked)] == ["decided"] * 3
    assert checked(2, n=3) == 2
    # traced, the body's term is the same address multiplied out
    from lanky.terms import Var

    assert checked(Var("i"), n=Var("n")) is not None


def test_a_layout_at_every_size_is_refuted_at_the_size_isl_finds() -> None:
    def padded(i: Fin[n], j: Fin[m]) -> Fin[n * m]:  # noqa: F821
        return i + (n - 1) * j  # noqa: F821

    assert [result for result in reasons(layout(padded))] == PADDED


#: Module constants a layout computes with, directly and through a helper.
SHIFT = 1
SCALE = 1


def scaled(i):
    """A helper a layout calls."""
    return SCALE * i


def test_a_layout_called_after_what_it_computes_with_changed_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The facts are about the address as it was when the layout was
    # declared; a kernel that read through it after SHIFT changed, or a
    # helper's SCALE, would read cells the facts say nothing about.
    def shifted(i: Fin[4]) -> Fin[4]:  # noqa: F821
        return (i + SHIFT) % 4

    def through(i: Fin[4]) -> Fin[4]:  # noqa: F821
        return scaled(i)

    checked, helped = layout(shifted), layout(through)
    assert (checked(2), helped(2)) == (3, 2)
    assert [status for status, _ in reasons(checked)] == ["decided"] * 3
    monkeypatch.setitem(globals(), "SHIFT", 2)
    monkeypatch.setitem(globals(), "SCALE", 2)
    with pytest.raises(LayoutError) as caught:
        checked(2)
    assert str(caught.value) == (
        "shifted now computes the address 0 for (i=2), where its facts say 3: "
        "something it computes with besides the indices changed after it was "
        "declared; declare the layout after the numbers it reads are set, and do "
        "not change them"
    )
    with pytest.raises(LayoutError, match="through now computes the address 4"):
        helped(2)
    # traced, at symbolic indices, the function's term is compared
    from lanky.terms import Var

    with pytest.raises(LayoutError, match=r"2\*i, where its facts say 1\*i"):
        helped(Var("i"))


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
