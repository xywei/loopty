"""Handles: a schedule names its loops and arrays by scope (loopty 0.4.0).

A handle is a path from what is scheduled, ``l2p.reconstruct.i``, made from
the traced kernel or program (:mod:`loopty.handles`). It is resolved through
what each step before it did to the loops, every step takes handles and
refuses strings with the handle to write, and everything a schedule prints
names loops and arrays by these paths.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest
from lanky.prelude import Real

from lines import where
from loopty import Arr, Fin, Schedule, kernel, program, reduce_sum, scope, when
from loopty.executor import LoopyExecutor
from loopty.handles import Array, Call, Condition, Loop, ShadowedName, Size
from loopty.schedule import IllegalCast

# {{{ the kernels and programs


@kernel
def flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """The pointwise Burgers flux."""
    for j in u.dom:
        f[j] = 0.5 * u[j] * u[j]


@kernel
def divergence(f: Arr[Fin[n], Real], rhs: Arr[Fin[n], Real]):  # noqa: F821
    """The centred divergence of the flux, at the interior points."""
    for i in rhs.dom:
        with when((i > 0) & (i + 1 < rhs.dom.size)):
            rhs[i] = -(f[i + 1] - f[i - 1]) / 2


@kernel
def scale(f: Arr[Fin[n], Real], g: Arr[Fin[n], Real]):  # noqa: F821
    """Twice every cell."""
    for k in f.dom:
        g[k] = 2.0 * f[k]


@program
def burgers(u, rhs):
    """The flux into an array the program makes, and its divergence."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)


@program
def three(u, rhs, h):
    """The flux, its divergence and the flux doubled: three calls."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)
    scale(f, h)


@program
def twice(u, g, h):
    """``scale`` called twice: two calls of one kernel."""
    f = Arr.zeros_like(u)
    flux(u, f)
    scale(f, g)
    scale(g, h)


@kernel
def rows(a: Arr[Fin[m], Fin[n], Real], s: Arr[Fin[m], Real]):  # noqa: F821
    """Each row summed, then halved."""
    for r in s.dom:
        s[r] = reduce_sum(a[r, c] for c in a.dom[r])
    for r in s.dom:
        s[r] = s[r] / 2


@kernel
def doubled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Each cell doubled."""
    for i in x.dom:
        y[i] = 2.0 * x[i]


@kernel
def doubled_rows(a: Arr[Fin[m], Fin[n], Real], b: Arr[Fin[m], Fin[n], Real]):  # noqa: F821
    """Each row doubled, a call per row."""
    for r in a.dom:
        doubled(a[r], b[r])


@kernel
def stencil(u: Arr[Fin[nt], Fin[nx], Real]):  # noqa: F821
    """A Jacobi sweep."""
    steps = u.dom
    for t in steps:
        row = u.dom[t]
        for i in row:
            with when((t + 1 < steps.size) & (i > 0) & (i + 1 < row.size)):
                u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2


def velocity(size: int) -> np.ndarray:
    x = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
    return np.sin(x) + 0.2 * np.sin(3.0 * x)


# }}}


# {{{ what a handle is


def test_a_handle_is_a_path_from_the_kernel_with_the_line_it_names() -> None:
    loop = burgers.divergence.i
    assert isinstance(loop, Loop) and repr(loop) == "burgers.divergence.i"
    assert loop.where == where(divergence, "for i in rhs.dom")
    array = burgers.f
    assert isinstance(array, Array) and repr(array) == "burgers.f"
    assert array.name == "f"
    call = burgers.divergence
    assert isinstance(call, Call) and repr(call) == "burgers.divergence"
    assert call.where == where(burgers, "divergence(f, rhs)")
    # a call's array is the array the call was passed
    assert burgers.divergence.f.name == "f"
    assert repr(burgers.divergence.rhs) == "burgers.divergence.rhs"
    # a kernel's own loops, arrays and sizes
    assert repr(stencil.t) == "stencil.t"
    assert repr(stencil.u) == "stencil.u"
    assert isinstance(stencil.nt, Size) and repr(stencil.nt) == "stencil.nt"


def test_the_long_forms_say_what_kind_of_name_they_are() -> None:
    assert burgers.calls.divergence.loops.i == burgers.divergence.i
    assert burgers.arrays.f == burgers.f
    assert stencil.loops.t == stencil.t
    # a size's == builds a condition, so it is compared by its path
    assert repr(stencil.sizes.nx) == repr(stencil.nx) == "stencil.nx"


def test_a_kernel_called_inside_a_kernels_loop_is_a_scope_of_it() -> None:
    inner = doubled_rows.doubled.i
    assert repr(inner) == "doubled_rows.doubled.i"
    split = Schedule(doubled_rows).split(inner, 2)
    assert split.nest == (
        "doubled_rows.r",
        "doubled_rows.doubled.i.outer",
        "doubled_rows.doubled.i.inner",
    )
    a = np.arange(12.0).reshape(3, 4)
    fact = LoopyExecutor().differential(
        doubled_rows, split, {"a": Arr.from_numpy(a), "b": Arr.zeros((3, 4))}
    )
    assert fact.status.value == "tested"


def test_a_name_a_scope_does_not_have_is_refused_with_what_it_has() -> None:
    with pytest.raises(AttributeError) as caught:
        burgers.divergence.k  # noqa: B018
    assert str(caught.value) == (
        "burgers.divergence has no loop, array, call or size named k; its loops "
        "are i; its arrays f and rhs"
    )
    # a program has no loops of its own: its calls' are named
    with pytest.raises(AttributeError) as caught:
        burgers.j  # noqa: B018
    assert str(caught.value).endswith(
        "Its calls' loops of that name are burgers.flux.j"
    )


def test_the_names_of_a_scope_complete() -> None:
    assert {"flux", "divergence", "f", "u", "rhs"} <= set(dir(burgers))
    assert {"i", "f", "rhs"} <= set(dir(burgers.divergence))
    assert {"t", "i", "u", "nt", "nx"} <= set(dir(stencil))
    assert list(burgers.loops) == []
    assert list(burgers.divergence.loops) == ["i"]


# }}}


# {{{ one name for several


def test_a_kernel_called_twice_is_two_calls_named_by_their_order() -> None:
    with pytest.raises(AttributeError) as caught:
        twice.scale.k  # noqa: B018
    first, second = twice.scale
    assert str(caught.value) == (
        f"twice.scale names 2 calls, twice.scale[0] at {first.where} and "
        f"twice.scale[1] at {second.where}; name one"
    )
    assert repr(twice.scale[0].k) == "twice.scale[0].k"
    assert repr(twice.scale[1].g) == "twice.scale[1].g"
    assert twice.scale[1].f.name == "g"  # what the second call was passed
    fused = Schedule(twice).fuse(twice.flux.j, twice.scale[0].k, twice.scale[1].k)
    assert fused.nest[-1] == "twice.flux.j"
    # the ambiguous name is refused where a step is given it too
    with pytest.raises(ValueError, match="twice.scale names 2 calls"):
        Schedule(twice).substitute(twice.scale)


def test_two_loops_of_one_binder_are_named_by_their_order() -> None:
    first, second = rows.r
    assert (repr(first), repr(second)) == ("rows.r[0]", "rows.r[1]")
    with pytest.raises(ValueError, match=r"rows.r names 2 loops, rows.r\[0\] at"):
        Schedule(rows).split(rows.r, 2)
    split = Schedule(rows).split(second, 2)
    assert split.nest[-2:] == ("rows.r[1].outer", "rows.r[1].inner")


# }}}


# {{{ names that clash


def test_a_loop_named_like_the_api_is_reached_by_its_long_form() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @kernel
        def clash(x: Arr[Fin[n], Real], term: Arr[Fin[n], Real]):  # noqa: F821
            for facts in x.dom:
                term[facts] = x[facts]

    said = [str(w.message) for w in caught if issubclass(w.category, ShadowedName)]
    assert said == [
        f"the array term of clash ({clash.where}) has the name of Kernel.term, so "
        "clash.term is not the array; a schedule names it clash.arrays.term",
        f"the loop facts of clash ({clash.where}) has the name of Kernel.facts, so "
        "clash.facts is not the loop; a schedule names it clash.loops.facts",
    ]
    loop = clash.loops.facts
    assert repr(loop) == "clash.loops.facts"
    split = Schedule(clash).split(loop, 2)
    assert split.history == ("split(clash.loops.facts, 2)",)
    assert split.nest == ("clash.loops.facts.outer", "clash.loops.facts.inner")
    # the short form is the API's, and a step given it says what to write
    with pytest.raises(TypeError, match="split\\(\\) takes a loop handle such as"):
        Schedule(clash).split(clash.facts, 2)


def test_a_loop_and_an_array_of_one_name_are_refused_for_the_short_form() -> None:
    with pytest.raises(AttributeError) as caught:
        _loop_and_array().i  # noqa: B018
    assert str(caught.value) == (
        "loop_and_array.i names a loop and an array; write "
        "loop_and_array.loops.i or loop_and_array.arrays.i"
    )
    assert repr(_loop_and_array().loops.i) == "loop_and_array.loops.i"


def _loop_and_array():
    """A term whose loop and whose array are both called ``i``."""
    from lanky.terms import evaluate_annotations

    from loopty.trace import trace

    def loop_and_array(
        i: Arr[Fin[n], Real],  # noqa: F821
        y: Arr[Fin[n], Real],  # noqa: F821
    ):
        for k in y.dom:
            y[k] = i[k]

    term = trace(loop_and_array, evaluate_annotations(loop_and_array))
    # rename the loop to the array's name, as a term written by hand may
    import dataclasses


    (binder,) = term.binders
    renamed = dataclasses.replace(binder, binder="i")
    return scope(dataclasses.replace(term, binders=(renamed,)))


# }}}


# {{{ every step takes handles, and refuses strings


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (
            lambda s: s.split("j", 2),
            "split() takes a loop handle such as burgers.flux.j, not the string 'j'",
        ),
        (
            lambda s: s.tag("j", "g.0"),
            "tag() takes a loop handle such as burgers.flux.j, not the string 'j'",
        ),
        (
            lambda s: s.on_groups("i"),
            "tag() takes a loop handle such as burgers.divergence.i, not the string "
            "'i'",
        ),
        (
            lambda s: s.fuse("flux", "divergence"),
            "fuse() takes a loop handle such as burgers.flux.j, not the string "
            "'flux'",
        ),
        (
            lambda s: s.interchange("j"),
            "interchange() takes a loop handle such as burgers.flux.j, not the "
            "string 'j'",
        ),
        (
            lambda s: s.contract("f"),
            "contract() takes an array handle such as burgers.f, not the string 'f'",
        ),
        (
            lambda s: s.substitute("f"),
            "substitute() takes an array handle such as burgers.f, not the string "
            "'f'",
        ),
        (
            lambda s: s.contract(burgers.flux.j),
            "contract() takes an array handle such as burgers.f, not the loop "
            "burgers.flux.j",
        ),
        (
            lambda s: s.split(burgers.f, 2),
            "split() takes a loop handle such as burgers.flux.j, not the array "
            "burgers.f",
        ),
        (
            lambda s: s.assume("n > 2"),
            "assume() takes a condition of the sizes, written with their handles, "
            "such as burgers.n > 2, not the string 'n > 2'",
        ),
    ],
)
def test_a_string_is_refused_with_the_handle_to_write(step, message: str) -> None:
    with pytest.raises(TypeError) as caught:
        step(Schedule(burgers))
    assert str(caught.value) == message


def test_a_handle_of_another_kernel_is_refused_with_the_path_to_write() -> None:
    with pytest.raises(ValueError) as caught:
        Schedule(burgers).split(divergence.i, 2)
    assert str(caught.value) == (
        "split(divergence.i, 2): divergence.i names something of divergence, and "
        "this schedule is of burgers: a handle is a path from what is scheduled; "
        "write burgers.divergence.i"
    )


def test_a_handle_of_a_namesake_says_which_kernel_is_which() -> None:
    # two kernels of one name, as a factory makes them or a file loaded twice
    def make() -> object:
        @kernel
        def twin(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
            for i in x.dom:
                y[i] = x[i]

        return twin

    first, second = make(), make()
    with pytest.raises(ValueError) as caught:
        Schedule(first).split(second.i, 2)
    defined = f"defined at {first.where}"
    assert (
        f"twin.i names something of another twin ({defined}), and this schedule "
        f"is of the twin ({defined})"
    ) in str(caught.value)


def test_a_fusion_of_three_calls_is_one_step_and_names_them_after() -> None:
    fused = Schedule(three, sizes={"n": 16}).fuse(
        three.flux.j, three.divergence.i, three.scale.k, shift=1
    )
    assert fused.history == (
        "fuse(three.flux.j, three.divergence.i, three.scale.k, shift=1)",
    )
    # the program's f, which no read sees zeroed (#290), and the fusion's two
    assert [fact.status.value for fact in fused.facts()] == ["decided"] * 3
    # one loop now, by any of the three
    for loop in (three.flux.j, three.divergence.i, three.scale.k):
        assert fused.split(loop, 4).nest[-2:] == (
            "three.flux.j.outer",
            "three.flux.j.inner",
        )
    inputs = {
        "u": Arr.from_numpy(velocity(16)),
        "rhs": Arr.zeros(16),
        "h": Arr.zeros(16),
    }
    fact = LoopyExecutor().differential(three, fused, inputs)
    assert fact.status.value == "tested"


def test_tag_and_on_groups_and_on_items_are_one_step() -> None:
    split = Schedule(doubled, target="opencl").split(doubled.i, 4)
    tagged = split.tag({doubled.i.outer: "g.0", doubled.i.inner: "l.0"})
    by_words = split.on_groups(doubled.i.outer).on_items(doubled.i.inner)
    assert tagged.tags == by_words.tags == {"i_outer": "g.0", "i_inner": "l.0"}
    assert by_words.history[-2:] == (
        "tag(doubled.i.outer, 'g.0')",
        "tag(doubled.i.inner, 'l.0')",
    )
    assert (
        split.on_groups(doubled.i.outer, axis=1).history[-1]
        == split.tag(doubled.i.outer, "g.1").history[-1]
    )
    with pytest.raises(ValueError, match="an axis is a whole number from 0"):
        split.on_items(doubled.i.inner, axis=-1)
    # each step's facts are at the line that asked for it, through on_groups too
    line = f"test_handles.py:{sys._getframe().f_lineno + 1}"
    grouped = split.on_groups(doubled.i.outer)
    assert {fact.where for fact in grouped.facts()[-2:]} == {line}


def test_a_condition_of_sizes_is_written_with_their_handles() -> None:
    condition = (stencil.nx % 4 == 0) & (stencil.nt >= 2)
    assert isinstance(condition, Condition)
    assert repr(condition) == "(stencil.nx % 4 == 0) & (stencil.nt >= 2)"
    assert condition.text == "nx % 4 == 0 and nt >= 2"
    assumed = Schedule(stencil).assume(condition).split(stencil.i, 4)
    (assumption, *_casts) = assumed.facts()
    assert assumption.statement == (
        "(stencil.nx % 4 == 0) & (stencil.nt >= 2), which stencil is scheduled "
        "assuming"
    )
    with pytest.raises(TypeError, match="has no truth value"):
        bool(stencil.nt >= 2)
    with pytest.raises(ValueError, match="this schedule is of burgers"):
        Schedule(burgers).assume(stencil.nt >= 2)


# }}}


# {{{ a handle through the steps


def test_a_handle_to_a_loop_a_step_replaced_is_refused_with_the_step() -> None:
    split = Schedule(stencil).split(stencil.i, 4)
    with pytest.raises(ValueError) as caught:
        split.split(stencil.i, 2)
    assert str(caught.value) == (
        "split(stencil.i, 2): split(stencil.i, 4) replaced stencil.i by "
        "stencil.i.outer and stencil.i.inner"
    )
    tiled = Schedule(stencil).skew(stencil.i, by=stencil.t).tile(
        stencil.t, stencil.i, 4, 4
    )
    assert tiled.nest == (
        "stencil.t.outer",
        "stencil.i.outer",
        "stencil.t.inner",
        "stencil.i.inner",
    )
    with pytest.raises(
        ValueError, match="tile\\(stencil.t, stencil.i, 4, 4\\) replaced"
    ):
        tiled.tag(stencil.t, "unr")


def test_a_handle_to_a_loop_a_substitution_took_out_is_refused_with_it() -> None:
    computed = Schedule(burgers).substitute(burgers.f)
    with pytest.raises(ValueError) as caught:
        computed.tag(burgers.flux.j, "g.0")
    assert str(caught.value) == (
        "tag(burgers.flux.j, 'g.0'): burgers.flux.j is no loop after "
        "substitute(burgers.f): no statement that still runs runs in it"
    )


def test_a_map_names_the_loops_it_makes_and_replaces_the_ones_it_takes() -> None:
    diamond = Schedule(stencil).affine("{ [t, i] -> [a, b] : a = t + i and b = t - i }")
    assert diamond.nest == ("stencil.loops.a", "stencil.loops.b")
    assert diamond.loops.a == stencil.loops.a
    with pytest.raises(ValueError) as caught:
        diamond.tag(stencil.t, "unr")
    assert str(caught.value) == (
        "tag(stencil.t, 'unr'): affine({ [t, i] -> [a = t + i, b = t - i] }) "
        "replaced stencil.t by stencil.loops.a and stencil.loops.b"
    )
    with pytest.raises(AttributeError, match="has no loop c a map of affine"):
        diamond.loops.c  # noqa: B018
    with pytest.raises(ValueError, match="no map of affine\\(...\\) in"):
        Schedule(stencil).tag(stencil.loops.a, "unr")


def test_a_handle_of_a_fused_loop_is_the_loop_by_any_of_its_handles() -> None:
    fused = Schedule(burgers).fuse(burgers.flux.j, burgers.divergence.i, shift=1)
    assert fused.nest[-1] == "burgers.flux.j"
    by_flux = fused.split(burgers.flux.j, 2)
    by_divergence = fused.split(burgers.divergence.i, 2)
    assert by_flux.order == by_divergence.order
    assert by_divergence.history[-1] == "split(burgers.divergence.i, 2)"
    assert by_divergence.nest[-2:] == ("burgers.flux.j.outer", "burgers.flux.j.inner")


# }}}


# {{{ what is printed


def test_a_refusal_names_loops_by_their_paths_and_no_loopy_names() -> None:
    with pytest.raises(IllegalCast) as caught:
        Schedule(three, sizes={"n": 16}).fuse(three.flux.j, three.divergence.i)
    message = str(caught.value)
    assert message.startswith("fuse(three.flux.j, three.divergence.i) illegal: ")
    assert message.endswith(
        "fuse(three.flux.j, three.divergence.i, shift=1) runs every dependence "
        "between them forward"
    )
    assert "_0" not in message and "_1" not in message
    fused = Schedule(three).fuse(three.flux.j, three.divergence.i, shift=1)
    with pytest.raises(IllegalCast) as caught:
        fused.tag(three.flux.j, "g.0")
    assert "the loop three.flux.j on g.0" in str(caught.value)
    assert "_0" not in str(caught.value)


def test_the_steps_and_the_facts_name_loops_and_arrays_by_their_paths() -> None:
    schedule = (
        Schedule(burgers, sizes={"n": 16})
        .fuse(burgers.flux.j, burgers.divergence.i, shift=1)
        .contract(burgers.f)
    )
    assert repr(schedule) == (
        "Schedule(burgers, target='c')"
        ".fuse(burgers.flux.j, burgers.divergence.i, shift=1)"
        ".contract(burgers.f, lambda j: j % 3)"
    )
    statements = [fact.statement for fact in schedule.facts()]
    assert statements[-1] == (
        "no read of burgers.f sees its slot overwritten after contract(burgers.f, "
        "lambda j: j % 3): burgers.f kept in 3 cells, f[j] in slot j % 3"
    )
    # the window rests on the fact the schedule was made with (#290)
    assert statements[0] == (
        "every cell of burgers.f that burgers reads, burgers.flux has stored by "
        "the time it is read, so burgers runs without the zeros burgers.f is "
        "made with"
    )
    assert schedule.key == (
        "burgers[c].fuse(burgers.flux.j, burgers.divergence.i, shift=1)"
        ".contract(burgers.f)"
    )


# }}}
