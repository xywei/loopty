"""Window contraction: an array a program makes, kept in the cells live at once (#132).

``Schedule.contract`` keeps an array in a window, each cell in a slot its
placement gives, and it is legal when every read is of a cell the producer
stored before it (``definedness``) and no read sees its slot overwritten
(``no-clobber``), both decided by isl over the schedule's order. From then
on the pairs of accesses to one slot are dependences of the schedule, so a
later step that breaks the window is refused with its pair. These tests need
no Lean; ``tests/test_lean_checks.py`` proves the same facts where it is
installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from lanky.ledger import Status
from lanky.prelude import Real

from lines import where
from loopty import Arr, Fin, Schedule, kernel, program, reduce_sum, when
from loopty.executor import LoopyExecutor, emit_code
from loopty.schedule import IllegalCast
from loopty.window import Placement, WindowError

lp = pytest.importorskip("loopy")

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path):
    """Import an example or a test kernel's file as a module of its own."""
    name = f"window_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


WINDOWS = load(ROOT / "tests" / "kernels" / "windows.py")


@kernel
def interior_flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """The flux at the interior points only."""
    for j in u.dom:
        with when((j > 0) & (j + 1 < u.dom.size)):
            f[j] = 0.5 * u[j] * u[j]


@kernel
def shifted(f: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Every cell, plus one."""
    for i in y.dom:
        y[i] = f[i] + 1.0


@program
def padded(u, y):
    """A consumer that reads the boundary cells the producer leaves zero."""
    f = Arr.zeros_like(u)
    interior_flux(u, f)
    shifted(f, y)


@program
def through(u, f, rhs):
    """The producer's array is a parameter of the program."""
    WINDOWS.flux(u, f)
    WINDOWS.divergence(f, rhs)


def velocity(n: int = 16) -> np.ndarray:
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.sin(x) + 0.2 * np.sin(3.0 * x)


def agrees(prog, schedule, inputs: dict, target: str = "c") -> None:
    fact = LoopyExecutor(target=target).differential(prog, schedule, inputs)
    assert fact.status is Status.TESTED, fact.provenance


def burgers_inputs(n: int = 16) -> dict:
    return {"u": Arr.from_numpy(velocity(n)), "rhs": Arr.zeros(n)}


def facts(schedule: Schedule) -> list[tuple[str, str]]:
    return [(fact.kind, fact.status.value) for fact in schedule.facts()]


# {{{ the window, found and given


def test_the_window_after_the_fusion_is_three_cells_found_along_the_loop() -> None:
    windowed = WINDOWS.windowed
    window = windowed.windows["f"]
    assert repr(window.placement) == "lambda j: j % 3"
    assert window.extents == (3,)
    assert windowed.history[-1] == "contract(burgers.f, lambda j: j % 3)"
    # the definedness the window needs is the fact the schedule is made with,
    # without the zeros of f (#290)
    assert facts(windowed)[0] == ("definedness", "decided")
    assert facts(windowed)[-1:] == [("no-clobber", "decided")]
    (no_clobber,) = [f for f in windowed.facts() if f.kind == "no-clobber"]
    assert no_clobber.statement == (
        "no read of burgers.f sees its slot overwritten after contract(burgers.f, "
        "lambda j: j % 3): burgers.f kept in 3 cells, f[j] in slot j % 3"
    )
    (definedness,) = [f for f in windowed.facts() if f.kind == "definedness"]
    assert no_clobber.rests_on == (definedness.id,)


def test_the_window_is_a_temporary_of_three_cells_and_the_zeros_are_gone() -> None:
    code = emit_code(WINDOWS.windowed)
    assert "double f_window[3];" in code
    assert "f_window[j + -3 * (j / 3)] = 0.5 * u[j] * u[j];" in code
    assert "double *__restrict__ f" not in code
    for n in (1, 2, 3, 16):
        agrees(WINDOWS.burgers, WINDOWS.windowed, burgers_inputs(n))


def test_a_placement_the_author_gives_is_checked_as_given() -> None:
    four = WINDOWS.fused.contract(WINDOWS.burgers.f, lambda j: j % 4)
    assert four.windows["f"].extents == (4,)
    assert four.history[-1] == "contract(burgers.f, lambda j: j % 4)"
    assert facts(four)[-1] == ("no-clobber", "decided")
    agrees(WINDOWS.burgers, four, burgers_inputs())
    # the same placement as the one found is the same schedule
    three = WINDOWS.fused.contract(WINDOWS.burgers.f, lambda j: j % 3)
    placement = Placement(("j",), (WINDOWS.windowed.windows["f"].placement.slots[0],))
    assert three.key == WINDOWS.fused.contract(WINDOWS.burgers.f, placement).key


def test_a_window_too_small_is_refused_with_the_read_and_the_overwrite() -> None:
    with pytest.raises(IllegalCast) as caught:
        WINDOWS.fused.contract(WINDOWS.burgers.f, lambda j: j % 2)
    # the reader, the producer and the producer's later instance
    assert str(caught.value) == (
        "contract(burgers.f, lambda j: j % 2) illegal: rhs[i] = -(f[i + 1] - "
        f"f[i - 1]) / 2 ({where(WINDOWS.divergence, 'rhs[i] =')}) at i=14 reads "
        f"f[13], which f[j] = 0.5 * u[j] * u[j] ({where(WINDOWS.flux, 'f[j] =')}) "
        "at j=13 stored in slot 1, after the same statement at j=15 stored f[15] "
        "there (at n=16, as hinted); contract(burgers.f) finds the smallest "
        "window that keeps every read, lambda j: j % 3"
    )
    fact = caught.value.fact
    assert (fact.kind, fact.status) == ("no-clobber", Status.REFUTED)
    (reader, _), (writer, _), (other, _), params = fact.provenance["witness"]
    assert (reader, writer, other, params) == (
        "divergence.S0",
        "flux.S0",
        "flux.S0",
        {"n": 16},
    )


def test_a_shift_register_keeps_every_access_at_a_constant_index() -> None:
    code = emit_code(WINDOWS.registers)
    assert "f_window[0] = f_window[1];" in code
    assert "f_window[1] = f_window[2];" in code
    assert "f_window[2] = 0.5 * u[j] * u[j];" in code
    assert "(f_window[2] + -1.0 * f_window[0])" in code
    assert "%" not in code.split("{", 1)[1] and "/ 3" not in code
    assert WINDOWS.registers.history[-1] == (
        "contract(burgers.f, lambda j: j % 3, registers=True)"
    )
    for n in (1, 2, 3, 16):
        agrees(WINDOWS.burgers, WINDOWS.registers, burgers_inputs(n))


def test_a_recurrence_along_the_loop_is_a_shift_register_too() -> None:
    # running.S1 reads f[j - 1] as well as storing f[j]: only its store is
    # the newest register, and its read is the one before
    schedule = WINDOWS.running_registers
    assert repr(schedule.windows["f"].placement) == "lambda j: j % 2"
    code = emit_code(schedule)
    assert "f_window[0] = f_window[1];" in code
    assert "f_window[1] = f_window[0] + u[j];" in code
    assert "%" not in code.split("{", 1)[1]
    for n in (1, 2, 16):
        agrees(
            WINDOWS.running_differences,
            schedule,
            {"u": Arr.from_numpy(velocity(n)), "y": Arr.zeros(n)},
        )


def test_registers_where_the_window_is_not_one_along_a_loop_are_refused() -> None:
    with pytest.raises(ValueError, match="the accesses span 3 cells of f along j"):
        WINDOWS.fused.contract(WINDOWS.burgers.f, lambda j: j % 4, registers=True)


def test_the_window_of_a_sum_factorized_pass_is_one_row() -> None:
    example = load(ROOT / "examples" / "sum_factorization.py")
    contracted = example.fused.contract(example.weak_derivative.v)
    window = contracted.windows["v"]
    assert repr(window.placement) == "lambda e, a, b: b"
    assert [str(extent) for extent in window.extents] == ["n"]
    assert "double v_window[n];" in emit_code(contracted)
    agrees(example.weak_derivative, contracted, example.inputs())


# }}}


# {{{ what it is checked against


def test_a_window_before_any_fusion_is_refused_as_no_smaller() -> None:
    with pytest.raises(ValueError) as caught:
        Schedule(WINDOWS.burgers, sizes={"n": 16}).contract(WINDOWS.burgers.f)
    assert str(caught.value) == (
        "contract(burgers.f): every cell of burgers.f is still to be read while "
        "the others are stored, in the order as it stands, so no window is "
        "smaller than burgers.f; fuse its readers with burgers.flux first, so "
        "that fewer cells are live at once"
    )


def test_a_read_of_the_zeros_refuses_the_window() -> None:
    fused = Schedule(padded, sizes={"n": 8}).fuse(
        padded.interior_flux.j, padded.shifted.i
    )
    with pytest.raises(IllegalCast) as caught:
        fused.contract(padded.f)
    fact = caught.value.fact
    assert (fact.kind, fact.status) == ("definedness", Status.REFUTED)
    assert "and a window would not" in str(caught.value)


def test_only_an_array_the_program_makes_is_kept_in_a_window() -> None:
    fused = Schedule(through, sizes={"n": 8}).fuse(
        through.flux.j, through.divergence.i, shift=1
    )
    with pytest.raises(WindowError, match="f is a parameter of through"):
        fused.contract(through.f)
    with pytest.raises(ValueError, match="kept in a window already"):
        WINDOWS.windowed.contract(WINDOWS.burgers.f)
    with pytest.raises(ValueError, match="kept in a window, lambda j: j % 3"):
        WINDOWS.windowed.substitute(WINDOWS.burgers.f)
    with pytest.raises(ValueError, match="computed where it is read already"):
        WINDOWS.fused.substitute(WINDOWS.burgers.f).contract(WINDOWS.burgers.f)


@pytest.mark.parametrize(
    ("placement", "error", "words"),
    [
        (lambda j, k: j, TypeError, "one index per axis of f, 1 of them"),
        ("j % 3", TypeError, "a placement is a function"),
        (lambda j: j * j, WindowError, "the slot j*j is not a sum of the indices"),
        (lambda j: j % 3 - 1, WindowError, "is below zero for some cells"),
        (lambda j: j // 2, WindowError, "grows with the sizes"),
    ],
)
def test_a_placement_is_quasi_affine_in_the_indices(placement, error, words) -> None:
    with pytest.raises(error, match=words.replace("*", r"\*")):
        WINDOWS.fused.contract(WINDOWS.burgers.f, placement)


def test_a_tag_before_the_window_is_refused_and_one_after_is_checked() -> None:
    # A window is taken over an order, so a loop that runs its instances in
    # none (ilp, here on the row of the first pass) refuses it. A tag after
    # it is checked against its slots: a row of n cells is one for every
    # work item, on C as once retargeted, so two elements on two work groups
    # store into one slot of it, and the group axis is refused with the pair.
    example = load(ROOT / "examples" / "sum_factorization.py")
    fused = example.fused
    program = example.weak_derivative
    row = program.along_r.b
    assert ("monotone", "decided") in facts(fused.tag(row, "ilp"))[-3:]
    with pytest.raises(
        ValueError, match="weak_derivative.along_r.b on ilp runs its instances in no"
    ):
        fused.tag(row, "ilp").contract(program.v)
    assert not fused.contract(program.v).windows["v"].private
    with pytest.raises(IllegalCast) as caught:
        fused.contract(program.v).tag(program.along_r.e, "g.0")
    assert (
        "stores v[0, 0, 0] in slot 0 of the window lambda e, a, b: b, which "
        "the same statement at e=1, a=0, b=0"
    ) in str(caught.value)
    assert str(caught.value).endswith(
        "v is kept in a window over the order this tag drops, one for every "
        "work item since its size is not a number (in global memory on "
        "OpenCL); a window of a number of cells is each work item's own"
    )


def test_a_window_of_a_size_on_opencl_is_shared_and_a_group_axis_refused() -> None:
    # A row of n cells is kept in global memory on OpenCL, one for every work
    # item, so two elements on two work groups store into one slot of it.
    pytest.importorskip("pyopencl")
    example = load(ROOT / "examples" / "sum_factorization.py")
    program = example.weak_derivative
    on_device = example.fused.retarget("opencl").contract(program.v)
    assert not on_device.windows["v"].private
    with pytest.raises(IllegalCast) as caught:
        on_device.tag(program.along_r.e, "g.0")
    message = str(caught.value)
    assert (
        "stores v[0, 0, 0] in slot 0 of the window lambda e, a, b: b, which "
        "along_r.S0[e=1, a=0, b=0]"
    ) in message
    assert message.endswith(
        "; v is kept in a window over the order this tag drops, one for every "
        "work item since its size is not a number (in global memory on "
        "OpenCL); a window of a number of cells is each work item's own"
    )


@kernel
def totals(f: Arr[Fin[n], Real], y: Arr[Fin[m], Real]):  # noqa: F821
    """Every cell of y, the sum of f."""
    for t in y.dom:
        y[t] = reduce_sum(f[s] for s in f.dom)


@program
def summed(u, y):
    """The flux, then sums of it."""
    f = Arr.zeros_like(u)
    WINDOWS.flux(u, f)
    totals(f, y)


def test_a_sum_run_outside_a_loop_refuses_the_window() -> None:
    # prioritize runs the sum's loop outside t, which then orders none of the
    # instances of totals, so no window is taken over it
    totals = summed.totals
    schedule = Schedule(summed, sizes={"n": 4, "m": 4}).prioritize(totals.s, totals.t)
    with pytest.raises(ValueError, match="a sum runs outside summed.totals.t"):
        schedule.contract(summed.f)


def test_a_step_after_the_window_keeps_the_order_of_every_slot() -> None:
    assert facts(WINDOWS.split_after)[-2:] == [
        ("bijective", "decided"),
        ("monotone", "decided"),
    ]
    agrees(WINDOWS.burgers, WINDOWS.split_after, burgers_inputs())
    assert WINDOWS.rejected_after_window() == (
        "affine({ flux_S0[j] -> [k = j]; divergence_S0[j] -> [k = 1 + j] }) "
        "illegal: rhs[i] = -(f[i + 1] - f[i - 1]) / 2 "
        f"({where(WINDOWS.divergence, 'rhs[i] =')}) at i=1 reads f[0] from slot "
        "0 of the window lambda j: j % 3, which f[j] = 0.5 * u[j] * u[j] "
        f"({where(WINDOWS.flux, 'f[j] =')}) at j=3 would now overwrite with "
        "f[3] first (at n=16, as hinted); a window keeps the order it was "
        "taken in, so take it after this step"
    )
    # the same step before the window takes a window of four
    later = WINDOWS.fused.affine(
        "{ flux_S0[j] -> [k] : k = j; divergence_S0[j] -> [k] : k = j + 1 }"
    ).contract(WINDOWS.burgers.f)
    assert repr(later.windows["f"].placement) == "lambda j: j % 4"


def test_a_window_is_replayed_with_its_schedule() -> None:
    again = Schedule(WINDOWS.registers)
    assert again.key == WINDOWS.registers.key
    assert again.history == WINDOWS.registers.history
    assert emit_code(again) == emit_code(WINDOWS.registers)


def test_a_window_on_opencl_is_a_shift_register_unless_asked_otherwise() -> None:
    pytest.importorskip("pyopencl")
    on_device = WINDOWS.windowed.retarget("opencl")
    code = emit_code(on_device)
    assert "f_window[0] = f_window[1];" in code
    assert "f_window[j + -3 * (j / 3)]" in emit_code(
        Schedule(WINDOWS.burgers, target="opencl", sizes={"n": 16})
        .fuse(WINDOWS.burgers.flux.j, WINDOWS.burgers.divergence.i, shift=1)
        .contract(WINDOWS.burgers.f, registers=False)
    )
    # back on the C target the window is addressed modulo again
    assert "f_window[j + -3 * (j / 3)]" in emit_code(on_device.retarget("c"))


def test_a_shift_register_inside_an_outer_loop_shifts_in_each_of_its_steps() -> None:
    # the window's loop j runs inside the rows' loop e, and so do its shifts:
    # a shift in a j loop of its own would make loopy nest e inside j
    schedule = WINDOWS.row_registers
    assert repr(schedule.windows["f"].placement) == "lambda e, j: j % 3"
    assert schedule.buildable == (True, ""), schedule.buildable
    code = emit_code(schedule)
    assert "f_window[0] = f_window[1];" in code
    assert code.count("for (") == 2
    agrees(WINDOWS.rows, schedule, row_inputs())


# }}}


# {{{ the zeros no read sees (#290)


@kernel
def halving(x: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """``f[i]`` from ``f[i // 2]``, which at ``i = 0`` is ``f[0]`` itself."""
    for i in f.dom:
        f[i] = x[i] + f[i // 2]


@kernel
def copied(f: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Every cell as it is."""
    for i in y.dom:
        y[i] = f[i]


@program
def halved(x, y):
    """``halving`` reads ``f[0]`` before it stores it: the zero ``f`` is made with."""
    f = Arr.zeros_like(x)
    halving(x, f)
    copied(f, y)


@kernel
def lagging(
    x: Arr[Fin[n], Real],  # noqa: F821
    w: Arr[Fin[k], Real],  # noqa: F821
    f: Arr[Fin[n], Real],  # noqa: F821
):
    """The first ``k`` cells of ``x``, then each cell twice the one ``k`` before."""
    for i in f.dom:
        with when(i < w.dom.size):
            f[i] = x[i] + 0.0 * w[i]
        with when(i >= w.dom.size):
            f[i] = 2.0 * f[i - w.dom.size]


@program
def lag(x, w, y):
    """``f[i - k]`` was stored before ``f[i]`` is, where ``k`` is at least one."""
    f = Arr.zeros_like(x)
    lagging(x, w, f)
    copied(f, y)


def zeroed(schedule: Schedule, array: str) -> bool:
    """Whether the schedule's kernel zeroes the array where the program makes it."""
    insn = schedule.lowering.insn_ids[f"{array}.zeros"]
    return insn in {i.id for i in schedule.kernel.default_entrypoint.instructions}


def test_a_schedule_runs_without_the_zeros_no_read_sees() -> None:
    # Every cell divergence reads, flux stored before it, so the zeros burgers
    # makes f with are work and nothing else: a schedule drops them before
    # any step, on the fact a window and a substitution ask, which is its
    # first, at the line that makes f.
    schedule = Schedule(WINDOWS.burgers, sizes={"n": 16})
    assert not zeroed(schedule, "f")
    (made,) = schedule.facts()
    assert (made.kind, made.status, made.decided_by) == (
        "definedness",
        Status.DECIDED,
        "isl",
    )
    assert made.statement == (
        "every cell of burgers.f that burgers reads, burgers.flux has stored by "
        "the time it is read, so burgers runs without the zeros burgers.f is "
        "made with"
    )
    assert made.where == where(WINDOWS.burgers, "f = Arr.zeros_like(u)")
    assert made.id.endswith(":zeros:f") and made.provenance["array"] == "f"
    # the same on every target, and after every step
    device = Schedule(WINDOWS.burgers, target="opencl")
    assert [fact.id for fact in device.facts()] == [made.id]
    assert WINDOWS.fused.facts()[0].id == made.id
    agrees(WINDOWS.burgers, schedule, burgers_inputs())
    agrees(WINDOWS.burgers, WINDOWS.fused, burgers_inputs())


def test_a_window_and_a_substitution_rest_on_the_fact_the_schedule_is_made_with():
    made = WINDOWS.fused.facts()[0]
    for schedule in (WINDOWS.windowed, WINDOWS.fused.substitute(WINDOWS.burgers.f)):
        defined = [fact for fact in schedule.facts() if fact.kind == "definedness"]
        assert defined == [made]
    (no_clobber,) = [f for f in WINDOWS.windowed.facts() if f.kind == "no-clobber"]
    assert no_clobber.rests_on == (made.id,)


def test_the_zeros_a_read_sees_are_kept() -> None:
    # shifted reads f[0] and f[n - 1], which interior_flux leaves as made: the
    # zeros are the program's boundary, and every schedule keeps them.
    schedule = Schedule(padded, sizes={"n": 8})
    assert zeroed(schedule, "f")
    assert schedule.facts() == ()
    fused = schedule.fuse(padded.interior_flux.j, padded.shifted.i)
    assert zeroed(fused, "f")
    agrees(padded, fused, {"u": Arr.from_numpy(velocity(8)), "y": Arr.zeros(8)})


def test_a_read_of_the_cell_its_own_instance_stores_sees_the_zeros() -> None:
    # halving reads f[i // 2], which an earlier instance stored, except at
    # i = 0, where it reads f[0] before it stores it: the zero. No dependence
    # pairs an instance with itself, and a window that dropped the zeros was
    # taken (and the definedness fact decided) all the same.
    assert zeroed(Schedule(halved), "f")
    halving_line = where(halving, "f[i] = x[i] + f[i // 2]")
    with pytest.raises(IllegalCast) as caught:
        Schedule(halved).contract(halved.f, lambda i: i)
    assert str(caught.value) == (
        f"contract(halved.f, lambda i: i) illegal: f[i] = x[i] + f[i // 2] "
        f"({halving_line}) at i=0 reads f[0], the cell it stores, before it "
        "stores it (at n=1, isl's choice), so the read sees the zeros the array "
        "was made with, and a window would not"
    )
    fact = caught.value.fact
    assert (fact.kind, fact.status) == ("definedness", Status.REFUTED)


def test_a_read_before_a_later_store_is_asked_where_the_sizes_are_as_assumed():
    # f[i - k] for i >= k reads a cell stored k steps before, where k >= 1;
    # where k < 0 it reads one stored after it. Assumed k >= 1, the window
    # is taken; the read after the store used to be looked for outside the
    # assumption, and its witness was empty (a ValueError).
    assumed = Schedule(lag).assume(lag.k >= 1)
    windowed = assumed.contract(lag.f, lambda i: i)
    defined = [fact for fact in windowed.facts() if fact.kind == "definedness"]
    assert [fact.status for fact in defined] == [Status.DECIDED]
    x = np.arange(1.0, 9.0)
    agrees(lag, windowed, {"x": x, "w": np.zeros(3), "y": np.zeros(8)})
    # assumed k >= 0 only, f[i - 0] is f[i] itself, at k = 0
    lagged_line = where(lagging, "f[i] = 2.0 * f[i - w.dom.size]")
    with pytest.raises(IllegalCast) as caught:
        Schedule(lag).assume(lag.k >= 0).contract(lag.f, lambda i: i)
    assert str(caught.value).startswith(
        "contract(lag.f, lambda i: i) illegal: f[i] = 2.0 * f[i - w.dom.size] "
        f"({lagged_line}) at i=0 reads f[0], the cell it stores, before it "
        "stores it (at k=0, "
    ), str(caught.value)


# }}}


# {{{ the review's questions: a window one short, a loop run backwards, a
# second reader, a parallel axis, a substitution through it


def row_inputs(m: int = 3, n: int = 8) -> dict:
    u = np.stack([velocity(n) * (1 + e) for e in range(m)])
    return {"u": Arr.from_numpy(u), "rhs": Arr.zeros((m, n))}


def two_reader_inputs(n: int = 16) -> dict:
    return {**burgers_inputs(n), "z": Arr.zeros(n)}


def run(schedule: Schedule, inputs: dict, target: str = "c") -> dict:
    copies = {name: Arr.from_numpy(arr.numpy().copy()) for name, arr in inputs.items()}
    return LoopyExecutor(target=target).run(schedule, **copies)


def bit_for_bit(first: Schedule, second: Schedule, inputs: dict) -> None:
    """The two schedules' runs, on their target, give the same bits."""
    target = first.target
    one, other = run(first, inputs, target), run(second, inputs, target)
    assert one.keys() == other.keys()
    for name in one:
        a, b = np.asarray(one[name]), np.asarray(other[name])
        assert a.shape == b.shape
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), name


def windows_and_arrays(target: str) -> list[tuple[Schedule, Schedule, dict]]:
    """Each window beside the same schedule with the array stored in full."""
    def on(schedule: Schedule) -> Schedule:
        return schedule.retarget(target)

    example = load(ROOT / "examples" / "sum_factorization.py")
    fused = on(WINDOWS.fused)
    out = [
        (fused, on(WINDOWS.windowed), burgers_inputs(n)) for n in (1, 2, 3, 16)
    ]
    out += [
        (fused, on(WINDOWS.registers), burgers_inputs(n)) for n in (1, 2, 3, 16)
    ]
    out += [
        (fused, fused.contract(WINDOWS.burgers.f, registers=False), burgers_inputs()),
        (fused, fused.contract(WINDOWS.burgers.f, lambda j: j % 4), burgers_inputs()),
        (fused, on(WINDOWS.split_after), burgers_inputs()),
        (
            on(WINDOWS.fused.affine(WINDOWS.REVERSED)),
            on(WINDOWS.reversed_window),
            burgers_inputs(),
        ),
        (
            on(Schedule(WINDOWS.two_readers, sizes=WINDOWS.SIZES)).fuse(
                WINDOWS.two_readers.flux.j,
                WINDOWS.two_readers.divergence_and_lag.i,
                shift=1,
            ),
            on(WINDOWS.deeper),
            two_reader_inputs(),
        ),
        (
            on(Schedule(WINDOWS.rows, sizes={"m": 3, "n": 8})).fuse(
                WINDOWS.rows.row_flux.j, WINDOWS.rows.row_divergence.i, shift=(0, 1)
            ),
            on(WINDOWS.row_registers),
            row_inputs(),
        ),
        (on(example.fused), on(example.contracted), example.inputs()),
        (on(WINDOWS.boxed), on(WINDOWS.one_triangle), triangle_inputs()),
    ]
    if target == "opencl":
        # each box a work group, and its triangle the work item's own
        out.append(
            (
                on(WINDOWS.boxed),
                on(WINDOWS.one_triangle).tag(WINDOWS.triangles.triangle.i, "g.0"),
                triangle_inputs(),
            )
        )
    return out


def test_a_window_computes_the_bits_the_array_does_on_c() -> None:
    for whole, windowed, inputs in windows_and_arrays("c"):
        bit_for_bit(whole, windowed, inputs)


def test_a_window_computes_the_bits_the_array_does_on_opencl() -> None:
    pytest.importorskip("pyopencl")
    for whole, windowed, inputs in windows_and_arrays("opencl"):
        bit_for_bit(whole, windowed, inputs)


def test_a_window_one_short_is_refused_with_the_window_that_fits() -> None:
    example = load(ROOT / "examples" / "sum_factorization.py")
    program = example.weak_derivative
    sized = Schedule(program, sizes={"n": 4, "ne": 3}).fuse(
        program.along_r.a, program.along_s.a
    )
    assert (
        repr(sized.contract(example.weak_derivative.v).windows["v"].placement)
        == "lambda e, a, b: b"
    )
    for short in (lambda e, a, b: b % 3, lambda e, a, b: 0, lambda e, a, b: a):
        with pytest.raises(IllegalCast) as caught:
            sized.contract(example.weak_derivative.v, short)
        message = str(caught.value)
        # the read is in a sum, and the cell it reads is the one stored
        assert "m_0" not in message
        assert message.endswith(
            "; contract(weak_derivative.v) finds the smallest window that keeps "
            "every read, lambda e, a, b: b"
        )
    with pytest.raises(IllegalCast) as caught:
        sized.contract(example.weak_derivative.v, lambda e, a, b: b % 3)
    # each pass's statement, shortened, at the names the source gives its loops
    assert (
        f"w[e, a, b] = reduce_sum(...) ({where(example.along_s, 'w[e, a, b] =')}) "
        "at e=0, a=0, b=0 reads v[0, 0, 0], which v[e, a, b] = reduce_sum(...) "
        f"({where(example.along_r, 'v[e, a, b] =')}) at e=0, a=0, b=0 stored in "
        "slot 0, after the same statement at e=0, a=0, b=3 stored v[0, 0, 3] "
        "there"
    ) in str(caught.value)


def test_a_second_reader_further_back_widens_the_window() -> None:
    assert repr(WINDOWS.deeper.windows["f"].placement) == "lambda j: j % 4"
    readers = WINDOWS.two_readers
    fused = Schedule(readers, sizes=WINDOWS.SIZES).fuse(
        readers.flux.j, readers.divergence_and_lag.i, shift=1
    )
    with pytest.raises(IllegalCast) as caught:
        fused.contract(readers.f, lambda j: j % 3)
    assert str(caught.value).startswith(
        "contract(two_readers.f, lambda j: j % 3) illegal: z[i] = f[i - 2] + "
        "0.25 * f[i] "
        f"({where(WINDOWS.divergence_and_lag, 'z[i] =')}) at i=14 reads f[12]"
    )
    assert str(caught.value).endswith("keeps every read, lambda j: j % 4")


@kernel
def doubled(f: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Every cell, twice."""
    for k in y.dom:
        y[k] = 2.0 * f[k]


@program
def read_after_the_loop(u, rhs, y):
    """The flux, its divergence, and a third call that reads every cell after."""
    f = Arr.zeros_like(u)
    WINDOWS.flux(u, f)
    WINDOWS.divergence(f, rhs)
    doubled(f, y)


def test_a_reader_after_the_fused_loop_leaves_no_window() -> None:
    fused = Schedule(read_after_the_loop, sizes={"n": 16}).fuse(
        read_after_the_loop.flux.j, read_after_the_loop.divergence.i, shift=1
    )
    with pytest.raises(
        ValueError, match="no window is smaller than read_after_the_loop.f"
    ):
        fused.contract(read_after_the_loop.f)
    with pytest.raises(IllegalCast) as caught:
        fused.contract(read_after_the_loop.f, lambda j: j % 3)
    assert f"y[k] = 2.0 * f[k] ({where(doubled, 'y[k] =')}) at k=0 reads f[0]" in str(
        caught.value
    )
    assert str(caught.value).endswith(
        "; no window of read_after_the_loop.f is smaller than read_after_the_loop.f in "
        "the order as it stands"
    )


def test_a_loop_run_backwards_takes_its_window_and_a_window_refuses_it() -> None:
    assert repr(WINDOWS.reversed_window.windows["f"].placement) == "lambda j: j % 3"
    agrees(WINDOWS.burgers, WINDOWS.reversed_window, burgers_inputs())
    with pytest.raises(IllegalCast) as caught:
        WINDOWS.windowed.affine(WINDOWS.REVERSED)
    assert str(caught.value).startswith(
        "affine({ flux_S0[j] -> [k = -j]; divergence_S0[j] -> [k = 2 - j] }) "
        f"illegal: f[j] = 0.5 * u[j] * u[j] ({where(WINDOWS.flux, 'f[j] =')}) at "
        "j=0 stores f[0] in slot 0 of the window lambda j: j % 3, which the same "
        "statement at j=3 would now overwrite with f[3] first"
    )
    # a shift register shifts along a loop over increasing cells
    with pytest.raises(ValueError, match="flux.S0 stores f at no index"):
        WINDOWS.fused.affine(WINDOWS.REVERSED).contract(
            WINDOWS.burgers.f, registers=True
        )


@pytest.mark.parametrize(
    ("target", "tag"),
    [("c", "ilp"), ("opencl", "l.0"), ("opencl", "g.0"), ("opencl", "vec")],
)
def test_the_loop_of_a_window_on_a_parallel_axis_is_refused(target, tag) -> None:
    # On a hardware axis the window is each work item's own, and the read of
    # f[j - 2] is on another work item than its store; ilp and vec run the
    # loop in one work item, in no order.
    if target == "opencl":
        pytest.importorskip("pyopencl")
    on_device = Schedule(WINDOWS.burgers, target=target, sizes={"n": 16}).fuse(
        WINDOWS.burgers.flux.j, WINDOWS.burgers.divergence.i, shift=1
    )
    with pytest.raises(IllegalCast) as caught:
        on_device.contract(WINDOWS.burgers.f).tag(WINDOWS.burgers.flux.j, tag)
    assert str(caught.value).endswith(
        "each work item keeps a window of burgers.f of its own, which no other "
        "work item sees"
        if tag in ("l.0", "g.0")
        else "; burgers.f is kept in a window over the order this tag drops"
    )
    # so is the inner loop of a split of it
    with pytest.raises(IllegalCast):
        on_device.contract(WINDOWS.burgers.f).split(WINDOWS.burgers.flux.j, 4).tag(
            WINDOWS.burgers.flux.j.inner, tag
        )


@kernel
def stored_and_shifted(
    u: Arr[Fin[n], Real],  # noqa: F821
    f: Arr[Fin[n], Real],  # noqa: F821
    g: Arr[Fin[n], Real],  # noqa: F821
    h: Arr[Fin[n], Real],  # noqa: F821
):
    """The flux, g computed from it, and h from it too."""
    for j in u.dom:
        f[j] = 0.5 * u[j] * u[j]
        g[j] = f[j] + 1.0
        h[j] = 2.0 * f[j]


@kernel
def lagged(g: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """g three cells back."""
    for k in y.dom:
        with when(k >= 3):
            y[k] = g[k - 3]


@program
def through_g(u, h, y):
    """g, computed from the flux, read three steps after it is stored."""
    f = Arr.zeros_like(u)
    g = Arr.zeros_like(u)
    stored_and_shifted(u, f, g, h)
    lagged(g, y)


def test_a_window_and_a_substitution_through_it_are_refused_together() -> None:
    # g computed where it is read reads f three steps after f is stored,
    # which a window of one cell, taken over the reads in the term, has
    # overwritten (#229): either order is refused
    fused = Schedule(through_g, sizes={"n": 16}).fuse(
        through_g.stored_and_shifted.j, through_g.lagged.k
    )
    windowed = fused.contract(through_g.f)
    assert repr(windowed.windows["f"].placement) == "lambda j: 0"
    with pytest.raises(ValueError) as caught:
        windowed.substitute(through_g.g)
    assert str(caught.value) == (
        "substitute(through_g.g): through_g.stored_and_shifted computes "
        "through_g.g from through_g.f, which is kept in a window, lambda j: 0, so "
        "computing through_g.g where it is read would read through_g.f where its "
        "slot may hold another cell; keep through_g.f in a window or compute "
        "through_g.g where it is read, not both"
    )
    with pytest.raises(ValueError) as caught:
        fused.substitute(through_g.g).contract(through_g.f)
    assert str(caught.value) == (
        "contract(through_g.f): substitute(through_g.g) computes through_g.g "
        "from through_g.f where it is read, so through_g.f is read there too, "
        "which a window is not taken over; keep through_g.f in a window or "
        "compute through_g.g where it is read, not both"
    )


# }}}


# {{{ a triangle: over a domain, two writers, a recurrence, each work item's own


def triangle_inputs(boxes: int = 3) -> dict:
    rng = np.random.default_rng(0)
    return {
        "c": Arr.from_numpy(rng.random((boxes, 2, WINDOWS.Q))),
        "s": Arr.zeros(boxes),
    }


def test_a_window_of_a_triangle_its_recurrence_reads_keeps_one_box() -> None:
    # R is over Where[i, a, b, a + b <= P] (#244), stored by a copy of two
    # rows and by the recurrence that reads the row two up (#213): fused on
    # the boxes, one box's triangle is live at a time, in the 5 x 5 box it
    # spans
    windowed = WINDOWS.one_triangle
    window = windowed.windows["R"]
    assert repr(window.placement) == "lambda i, a, b: (a, b)"
    assert window.extents == (5, 5) and window.private
    assert [stmt.id for stmt in window.producers] == ["triangle.S0", "triangle.S1"]
    assert [stmt.id for stmt in window.readers] == ["triangle.S1", "triangle_sums.S0"]
    assert [(f.kind, f.status, f.statement) for f in windowed.facts()[:1]] == [
        (
            "definedness",
            Status.DECIDED,
            "every cell of triangles.R that triangles reads, triangles.triangle "
            "has stored by the time it is read, so triangles runs without the "
            "zeros triangles.R is made with",
        ),
    ]
    assert [(f.kind, f.status, f.statement) for f in windowed.facts()[-1:]] == [
        (
            "no-clobber",
            Status.DECIDED,
            "no read of triangles.R sees its slot overwritten after "
            "contract(triangles.R, lambda i, a, b: (a, b)): triangles.R kept in 25 "
            "cells, R[i, a, b] in slot (a, b)",
        ),
    ]
    assert "double R_window[5 * 5];" in emit_code(windowed)
    agrees(WINDOWS.triangles, windowed, triangle_inputs())


def test_a_window_the_recurrence_reads_past_is_refused_with_the_read() -> None:
    copy = where(WINDOWS.triangle, "R[i, a, b] = c[i, a, b]")
    recurrence = where(WINDOWS.triangle, "R[i, a, b] = -R[i, a - 2, b + 2]")
    assert WINDOWS.rejected_row() == (
        "contract(triangles.R, lambda i, a, b: b) illegal: R[i, a, b] = "
        "-R[i, a - 2, b + "
        f"2] ({recurrence}) at i=2, a=3, b=0 reads R[2, 1, 2], which R[i, a, b] = "
        f"c[i, a, b] ({copy}) at i=2, a=1, b=2 stored in slot 2, after R[i, a, b] "
        f"= -R[i, a - 2, b + 2] ({recurrence}) at i=2, a=2, b=2 stored R[2, 2, 2] "
        "there (at nb=3, as hinted); contract(triangles.R) finds the smallest "
        "window that keeps every read, lambda i, a, b: (a, b)"
    )
    # before the fusion every box's triangle is read after the last is stored
    with pytest.raises(ValueError) as caught:
        Schedule(WINDOWS.triangles, sizes={"nb": 3}).contract(WINDOWS.triangles.R)
    assert str(caught.value).endswith(
        "fuse its readers with triangles.triangle first, so that fewer cells are "
        "live at once"
    )
    with pytest.raises(IllegalCast, match=r"stored in slot \(2, 2\), after"):
        Schedule(WINDOWS.triangles, sizes={"nb": 3}).contract(
            WINDOWS.triangles.R, lambda i, a, b: (a, b)
        )


def test_each_work_item_keeps_its_own_window_of_a_number_of_cells() -> None:
    # each box a work group keeps its own triangle (#214), so two boxes'
    # stores into one slot need no order
    box = WINDOWS.triangles.triangle.i
    on_groups = WINDOWS.one_triangle.tag(box, "g.0")
    assert facts(on_groups)[-3:] == [
        ("bijective", "decided"),
        ("monotone", "decided"),
        ("buildable", "refuted"),  # C has no work groups
    ]
    # but a read on another work item than its store does not see it
    with pytest.raises(IllegalCast) as caught:
        WINDOWS.one_triangle.tag({box: "g.0", WINDOWS.triangles.triangle.b: "l.0"})
    copy = where(WINDOWS.triangle, "R[i, a, b] = c[i, a, b]")
    recurrence = where(WINDOWS.triangle, "R[i, a, b] = -R[i, a - 2, b + 2]")
    assert str(caught.value) == (
        "tag({triangles.triangle.i: 'g.0', triangles.triangle.b: 'l.0'}) "
        f"illegal: R[i, a, b] = c[i, a, b] ({copy}) at i=0, a=0, b=4 writes "
        "R[0, 0, 4], read by R[i, a, b] = -R[i, a - 2, b + 2] "
        f"({recurrence}) at i=0, a=2, b=2 on another work item (at nb=3, as "
        "hinted): the loop triangles.triangle.b on l.0 runs them on work items 4 "
        "and 2 of it, and each work item keeps a window of triangles.R of its "
        "own, which no other work item sees"
    )


# }}}


# {{{ a window and updates


@kernel
def added_flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """The flux added into the zeros of f: an update of each cell."""
    for j in u.dom:
        f[j] += 0.5 * u[j] * u[j]


@kernel
def scattered_flux(
    u: Arr[Fin[n], Real],  # noqa: F821
    node: Arr[Fin[n], Fin[n]],  # noqa: F821
    f: Arr[Fin[n], Real],  # noqa: F821
):
    """The flux added into the cell each point names: a scatter-add."""
    for j in u.dom:
        f[node[j]] += 0.5 * u[j] * u[j]


@program
def added(u, rhs):
    """The flux added into f, then its divergence."""
    f = Arr.zeros_like(u)
    added_flux(u, f)
    WINDOWS.divergence(f, rhs)


@program
def scattered(u, node, rhs):
    """The flux scattered into f, then its divergence."""
    f = Arr.zeros_like(u)
    scattered_flux(u, node, f)
    WINDOWS.divergence(f, rhs)


def test_a_window_of_an_array_that_is_updated_is_refused_with_the_fix() -> None:
    # An update adds to the zeros, which a window drops, and the slot of
    # f[j] holds f[j - 3] when f[j] is first updated: the window would add
    # to it. So it is refused, found or given, with the two fixes.
    fused = Schedule(added, sizes={"n": 16}).fuse(
        added.added_flux.j, added.divergence.i, shift=1
    )
    for placement in (None, lambda j: j % 3):
        with pytest.raises(WindowError) as caught:
            fused.contract(added.f, placement)
        assert str(caught.value).endswith(
            "added.added_flux updates f[j] in place, starting from the zeros f was "
            "made with, and a window keeps a value stored once, without them; "
            "store each cell once, f[j] = ... (a reduce_sum where the updates run "
            "over a loop), or keep f whole"
        )
    # through a map, which cell an update adds into is read from an array
    with pytest.raises(WindowError) as caught:
        Schedule(scattered, sizes={"n": 16}).contract(scattered.f)
    assert str(caught.value) == (
        "contract(scattered.f): scattered.scattered_flux updates f[node[j]] in "
        "place, starting from the zeros f was made with, and a window keeps a "
        "value stored once, without them; keep f whole"
    )


def assembled_inputs(n: int = 16, m: int = 4) -> dict:
    """The flux's points, each added into node ``i % m``."""
    return {
        "u": Arr.from_numpy(velocity(n)),
        "node": Arr.from_numpy(np.arange(n) % m),
        "y": Arr.zeros(m),
    }


def test_an_update_may_read_a_window_which_keeps_its_slots_in_order() -> None:
    fused = Schedule(WINDOWS.burgers_assembled, sizes={"n": 16, "m": 4}).fuse(
        WINDOWS.burgers_assembled.flux.j,
        WINDOWS.burgers_assembled.assembled_divergence.i,
        shift=1,
    )
    schedule = WINDOWS.assembled
    assert repr(schedule.windows["f"].placement) == "lambda j: j % 3"
    assert schedule.atomics == frozenset({"y"})
    assert ("no-clobber", "decided") in facts(schedule)
    assert facts(schedule)[-2:] == [("bijective", "decided"), ("monotone", "decided")]
    # atomic before the window takes the same window
    before = (
        fused.atomic(WINDOWS.burgers_assembled.y)
        .contract(WINDOWS.burgers_assembled.f)
        .windows["f"]
    )
    assert (repr(before.placement), before.extents) == ("lambda j: j % 3", (3,))
    # the updates run in the source's order on C: the same bits as the array
    # stored in full
    whole = run(fused, assembled_inputs())["y"]
    assert np.array_equal(np.asarray(run(schedule, assembled_inputs())["y"]), whole)
    agrees(WINDOWS.burgers_assembled, schedule, assembled_inputs())
    # ilp leaves the updates of a node in no order, which atomic allows, and
    # the window's slots in no order, which nothing does
    with pytest.raises(IllegalCast) as caught:
        fused.tag(WINDOWS.burgers_assembled.flux.j, "ilp")
    assert "both update y[" in str(caught.value)
    assert "with atomic(burgers_assembled.y) before this step" in str(caught.value)
    program = WINDOWS.burgers_assembled
    fused.atomic(program.y).tag(program.flux.j, "ilp")
    for windowed in (
        fused.contract(program.f),
        fused.contract(program.f).atomic(program.y),
    ):
        with pytest.raises(IllegalCast) as caught:
            windowed.tag(program.flux.j, "ilp")
        assert str(caught.value).endswith(
            "; burgers_assembled.f is kept in a window over the order this tag drops"
        )


def test_a_window_and_atomic_updates_compile_together_on_opencl(plain_opencl) -> None:
    schedule = (
        Schedule(WINDOWS.burgers_assembled, target="opencl", sizes={"n": 16, "m": 4})
        .fuse(
            WINDOWS.burgers_assembled.flux.j,
            WINDOWS.burgers_assembled.assembled_divergence.i,
            shift=1,
        )
        .contract(WINDOWS.burgers_assembled.f)
        .atomic(WINDOWS.burgers_assembled.y)
    )
    assert schedule.buildable == (True, ""), schedule.buildable
    code = emit_code(schedule)
    assert "f_window[0] = f_window[1];" in code
    assert "atom_cmpxchg" in code


# }}}


# {{{ the statements Lean is given, without Lean


def test_the_no_clobber_fact_is_stated_from_the_description() -> None:
    (fact,) = [f for f in WINDOWS.windowed.facts() if f.kind == "no-clobber"]
    obligation = fact.term.lean_obligation()
    assert not obligation.declined
    assert [part.name for part in obligation.parts] == ["read_0", "read_1"]
    from lanky.terms import render

    assert "% 3" in render(obligation.parts[1].term)


def test_the_casts_after_a_window_state_its_slots() -> None:
    from loopty.obligations import cast_obligation

    monotone = WINDOWS.split_after.facts()[-1]
    names = [part.name for part in cast_obligation(monotone.term).parts]
    assert any(name.startswith("window_") for name in names)
    assert not any(
        part.name.startswith("dependence_") and "f.zeros" in part.label
        for part in cast_obligation(monotone.term).parts
    )


def test_the_casts_after_a_window_read_by_updates_order_no_two_updates() -> None:
    from loopty.obligations import cast_obligation

    monotone = WINDOWS.assembled.facts()[-1]
    assert monotone.kind == "monotone"
    parts = cast_obligation(monotone.term).parts
    assert any(part.name.startswith("window_") for part in parts)
    # two updates of a node need no order, and are no class
    assert not any("y[" in part.label for part in parts)


# }}}
