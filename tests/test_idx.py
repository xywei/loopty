"""Index types to isl: sets, normalization, layouts, delinearization."""

from __future__ import annotations

import islpy as isl
import pymbolic.primitives as prim
import pytest

from loopty.idx import (
    Fin,
    Layout,
    NonAffineLayout,
    RaggedLayout,
    Reflections,
    axis_size,
    delinearize,
    is_affine,
    linearize,
    normalize,
    size_params,
    strides_of,
    to_set,
)

n = prim.Variable("n")
m = prim.Variable("m")


def test_the_same_term_always_reflects_to_the_same_parameter() -> None:
    table = Reflections()
    first = table.symbol(prim.Subscript(prim.Variable("cnt"), prim.Variable("r")))
    again = table.symbol(prim.Subscript(prim.Variable("cnt"), prim.Variable("r")))
    assert first == "nl_cnt_r"
    assert again == first
    assert table.get(first) is not None


def test_two_terms_that_spell_the_same_get_two_parameters() -> None:
    # ``cnt[r]`` and ``cnt*r`` both read ``nl_cnt_r`` once every non-word run
    # has become an underscore. Giving them one parameter asserts they are
    # equal, which is a thing isl would then happily reason from.
    table = Reflections()
    subscript = table.symbol(
        prim.Subscript(prim.Variable("cnt"), prim.Variable("r"))
    )
    product = table.symbol(prim.Product((prim.Variable("cnt"), prim.Variable("r"))))
    assert subscript == "nl_cnt_r"
    assert product != subscript
    assert product.startswith("nl_cnt_r")


def test_a_reflected_parameter_keeps_clear_of_a_name_already_in_use() -> None:
    table = Reflections(["nl_cnt_r"])
    name = table.symbol(prim.Subscript(prim.Variable("cnt"), prim.Variable("r")))
    assert name != "nl_cnt_r"
    assert "nl_cnt_r" not in table


def test_a_set_reflects_through_the_table_it_is_given() -> None:
    table = Reflections(["nl_n_m"])
    domain = to_set((Fin[n * m],), reflections=table)
    (param,) = domain.get_var_names(isl.dim_type.param)
    assert param != "nl_n_m"
    assert param in table


def test_axis_size_accepts_fin_int_and_term() -> None:
    assert axis_size(Fin[7]) == 7
    assert axis_size(5) == 5
    assert axis_size(n + 1) == n + 1


def test_to_set_concrete_counts_points() -> None:
    domain = to_set((Fin[3], Fin[4]))
    assert domain.count_val().to_python() == 12
    assert not domain.is_empty()


def test_to_set_with_parameters() -> None:
    domain = to_set((Fin[n], Fin[n + 1]))
    assert domain.get_var_names(isl.dim_type.param) == ["n"]
    instance = domain.intersect_params(isl.Set("[n] -> { : n = 2 }"))
    assert instance.count_val().to_python() == 2 * 3


def test_to_set_extra_params_land_in_the_space() -> None:
    domain = to_set((Fin[n],), params=("cnt_r",))
    assert set(domain.get_var_names(isl.dim_type.param)) == {"n", "cnt_r"}


def test_to_set_reflects_a_nonaffine_bound_as_a_parameter() -> None:
    # isl cannot multiply two unknowns, so the axis size n*m becomes one fresh
    # parameter. That widens the set, which keeps in-bounds proofs sound.
    domain = to_set((Fin[n * m],))
    params = domain.get_var_names(isl.dim_type.param)
    assert len(params) == 1
    assert params[0].startswith("nl_")


def test_to_set_empty_shape_is_a_single_point() -> None:
    assert to_set(()).count_val().to_python() == 1


def test_normalize_splits_a_product_axis() -> None:
    assert normalize((Fin[n * m],)) == (Fin[n], Fin[m])
    # and the split is what makes the size expressible: two honest axes.
    assert to_set(normalize((Fin[n * m],))).get_var_names(isl.dim_type.param) == [
        "m",
        "n",
    ]


def test_normalize_leaves_literals_and_folds_constants() -> None:
    assert normalize((Fin[6],)) == (Fin[6],)
    assert normalize((Fin[2 * n],)) == (Fin[2], Fin[n])


def test_is_affine() -> None:
    assert is_affine(n + 1)
    assert is_affine(3 * n - 2)
    assert is_affine(prim.FloorDiv(n, 4))
    assert not is_affine(n * m)
    assert not is_affine(prim.Subscript(prim.Variable("cnt"), n))


def test_size_params_collects_free_names() -> None:
    assert size_params([n + m, 4, 2 * n]) == ("m", "n")


def test_strides_row_and_column_major() -> None:
    assert strides_of((Fin[3], Fin[4], Fin[5]), "C") == (20, 5, 1)
    assert strides_of((Fin[3], Fin[4], Fin[5]), "F") == (1, 3, 12)
    with pytest.raises(ValueError, match="order"):
        strides_of((Fin[3],), "Z")


def test_linearize_and_delinearize_round_trip() -> None:
    shape = (Fin[3], Fin[4])
    for order in ("C", "F"):
        for i in range(3):
            for j in range(4):
                flat = linearize((i, j), shape, order)
                back = delinearize(flat, shape, order)
                assert tuple(int(b) for b in back) == (i, j)


def test_layout_map_is_a_bijection_onto_the_addresses() -> None:
    layout = Layout((Fin[3], Fin[4]))
    a_map = layout.to_map()
    assert a_map.is_bijective()
    assert a_map.range().count_val().to_python() == 12


def test_column_major_layout_differs_from_row_major() -> None:
    row = Layout((Fin[3], Fin[4]), "C")
    column = Layout((Fin[3], Fin[4]), "F")
    assert row.strides == (4, 1)
    assert column.strides == (1, 3)
    assert not row.to_map().is_equal(column.to_map())
    # Both cover the same addresses: a layout change is a permutation.
    assert row.to_map().range().is_equal(column.to_map().range())


def test_symbolic_strides_have_no_isl_map() -> None:
    layout = Layout((Fin[n], Fin[m]))
    assert layout.strides == (m, 1)
    with pytest.raises(NonAffineLayout, match="symbolic strides"):
        layout.to_map()


def test_ragged_layout_offsets_counts_and_map() -> None:
    ragged = RaggedLayout((0, 2, 2, 5))
    assert ragged.nrows == 3
    assert ragged.counts == (2, 0, 3)
    assert ragged.total == 5
    assert ragged.to_set().count_val().to_python() == 5
    a_map = ragged.to_map()
    assert a_map.is_bijective()
    assert a_map.range().count_val().to_python() == 5


def test_ragged_layout_flat_index_is_off_r_plus_j() -> None:
    ragged = RaggedLayout((0, 2, 5))
    r = prim.Variable("r")
    j = prim.Variable("j")
    assert str(ragged.flat_index(r, j)) == "off[r] + j"


def test_ragged_layout_rejects_decreasing_offsets() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        RaggedLayout((0, 3, 1))
