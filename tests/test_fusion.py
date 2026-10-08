"""Fusing a program's calls, and the storage of what passes between them (#13).

A program's term runs its calls one after the other (:mod:`loopty.compose`),
and lowers to one kernel whose loops follow one another. Three things are
built on that here:

* a ``definedness`` fact for each array the program makes and each call that
  reads it after another call wrote it, decided by isl: the cells it reads
  are cells stored before it, so the zeros the array was made with reach
  none of its reads;
* :meth:`Schedule.fuse`, a map per statement that runs the consumer's loops
  in the producer's, checked as every cast is, and refused with the pair of
  instances that it would run backwards;
* :meth:`Schedule.substitute`, which computes such an array where it is
  read instead of storing it, through loopy's ``assignment_to_subst``, and
  checks every later step against the dependences of the program as it then
  runs.
"""

from __future__ import annotations

import islpy as isl
import numpy as np
import pytest
from lanky.ledger import Status
from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, kernel, program, reduce_sum, when
from loopty.executor import LoopyExecutor
from loopty.oracle import IslOracle
from loopty.schedule import IllegalCast

# {{{ the kernels and programs


@kernel
def flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """Pointwise Burgers flux."""
    for j in u.dom:
        f[j] = 0.5 * u[j] * u[j]


@kernel
def divergence(f: Arr[Fin[n], Real], rhs: Arr[Fin[n], Real]):  # noqa: F821
    """Centred divergence of the flux, inside the boundary."""
    for i in rhs.dom:
        with when((i > 0) & (i + 1 < rhs.dom.size)):
            rhs[i] = -(f[i + 1] - f[i - 1]) / 2


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


@kernel
def doubled(f: Arr[Fin[n], Real], g: Arr[Fin[n], Real]):  # noqa: F821
    """Every cell, twice."""
    for k in g.dom:
        g[k] = 2.0 * f[k]


@kernel
def front(f: Arr[Fin[n], Real], h: Arr[Fin[n], Real]):  # noqa: F821
    """The front half of ``f``."""
    for x in h.dom:
        with when(2 * x < h.dom.size):
            h[x] = f[x]


@kernel
def back(h: Arr[Fin[n], Real], rhs: Arr[Fin[n], Real]):  # noqa: F821
    """The back half of ``h``, plus one."""
    for i in rhs.dom:
        with when(2 * i >= rhs.dom.size):
            rhs[i] = h[i] + 1.0


@kernel
def bump(u: Arr[Fin[n], Real]):  # noqa: F821
    """Add one to every cell, in place."""
    for k in u.dom:
        u[k] = u[k] + 1.0


@kernel
def update(f: Arr[Fin[n], Real], u: Arr[Fin[n], Real]):  # noqa: F821
    """One explicit step of Burgers' equation, in place."""
    for i in u.dom:
        with when((i > 0) & (i + 1 < u.dom.size)):
            u[i] = u[i] - (f[i + 1] - f[i - 1]) / 2


@kernel
def halved_flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """The flux of each pair of cells, stored twice."""
    for j in u.dom:
        f[j // 2] = 0.5 * u[j] * u[j]


@kernel
def totals(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """The sum of every cell, in every cell."""
    for j in f.dom:
        f[j] = reduce_sum(u[k] for k in u.dom)


@kernel
def gather(
    idx: Arr[Fin[n], Fin[n]],  # noqa: F821
    f: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Each cell of ``f`` that ``idx`` names."""
    for i in y.dom:
        y[i] = f[idx[i]]


@kernel
def square2(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    g: Arr[Fin[n], Fin[m], Real],  # noqa: F821
):
    """Every cell squared."""
    for i in a.dom:
        for j in a.dom[i]:
            g[i, j] = a[i, j] * a[i, j]


@kernel
def down(
    g: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    b: Arr[Fin[n], Fin[m], Real],  # noqa: F821
):
    """Each cell less the one below it."""
    for p in b.dom:
        for q in b.dom[p]:
            with when(p + 1 < b.dom.size):
                b[p, q] = g[p + 1, q] - g[p, q]


@kernel
def row_total(
    g: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """The sum of each row."""
    for w in z.dom:
        z[w] = reduce_sum(g[w, q] for q in g.dom[w])


@kernel
def rows_then_cells(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    s: Arr[Fin[n], Real],  # noqa: F821
    g: Arr[Fin[n], Fin[m], Real],  # noqa: F821
):
    """A statement per row, and one per cell of the row."""
    for r in s.dom:
        s[r] = 1.0 + r
        for c in a.dom[r]:
            g[r, c] = a[r, c] + 1.0


@kernel
def row_sum(s: Arr[Fin[n], Real], z: Arr[Fin[n], Real]):  # noqa: F821
    """Each row's value, doubled."""
    for w in z.dom:
        z[w] = s[w] + s[w]


@program
def burgers(u, rhs):
    """The flux into an array the program makes, and its divergence."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)


@program
def padded(u, y):
    """A consumer that reads the boundary cells the producer leaves zero."""
    f = Arr.zeros_like(u)
    interior_flux(u, f)
    shifted(f, y)


@program
def early(u, rhs, y):
    """A read of the array before the producer stores it."""
    f = Arr.zeros_like(u)
    shifted(f, y)
    flux(u, f)
    divergence(f, rhs)


@program
def chained(u, rhs):
    """Two arrays the program makes, the second computed from the first."""
    f = Arr.zeros_like(u)
    g = Arr.zeros_like(u)
    flux(u, f)
    doubled(f, g)
    divergence(g, rhs)


@program
def between(u, rhs):
    """A call between the two fused that the second reads only by array."""
    f = Arr.zeros_like(u)
    h = Arr.zeros_like(u)
    flux(u, f)
    front(f, h)
    back(h, rhs)


@program
def bumped(u, rhs):
    """The producer's input written between the producer and the consumer."""
    f = Arr.zeros_like(u)
    flux(u, f)
    bump(u)
    divergence(f, rhs)


@program
def bumped_after(u, rhs):
    """The producer's input written after the consumer has read the array."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)
    bump(u)


@program
def stepped(u):
    """The consumer writes in place what the producer read."""
    f = Arr.zeros_like(u)
    flux(u, f)
    update(f, u)


@program
def twice(u, rhs):
    """Two calls store the same array."""
    f = Arr.zeros_like(u)
    flux(u, f)
    flux(u, f)
    divergence(f, rhs)


@program
def halved(u, rhs):
    """A producer that stores each cell of half the array twice."""
    f = Arr.zeros_like(u)
    halved_flux(u, f)
    divergence(f, rhs)


@program
def summed(u, rhs):
    """A producer that stores a sum."""
    f = Arr.zeros_like(u)
    totals(u, f)
    divergence(f, rhs)


@program
def unread(u, rhs):
    """An array the program makes and nothing reads."""
    f = Arr.zeros_like(u)
    flux(u, f)
    flux(u, rhs)


@program
def through(u, f, rhs):
    """The producer's array is a parameter of the program."""
    flux(u, f)
    divergence(f, rhs)


@program
def gathered(u, idx, y):
    """The array read through an index array."""
    f = Arr.zeros_like(u)
    flux(u, f)
    gather(idx, f, y)


@program
def gathered_inside(u, idx, y):
    """The array read through an index array, and stored inside only."""
    f = Arr.zeros_like(u)
    interior_flux(u, f)
    gather(idx, f, y)


@program
def squares(a, b):
    """Two nests of two loops, the second reading the row below."""
    g = Arr.zeros_like(a)
    square2(a, g)
    down(g, b)


@program
def squared_rows(a, z):
    """Every cell squared, then each row summed."""
    g = Arr.zeros_like(a)
    square2(a, g)
    row_total(g, z)


@program
def rows(a, z, g):
    """A producer with statements at two depths, then a consumer of one."""
    s = Arr.zeros_like(z)
    rows_then_cells(a, s, g)
    row_sum(s, z)


def velocity(size: int) -> np.ndarray:
    return np.sin(np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)) + 0.3


def burgers_inputs(size: int = 8) -> dict:
    return {"u": Arr.from_numpy(velocity(size)), "rhs": Arr.zeros(size)}


def agrees(prog, schedule, inputs: dict) -> None:
    fact = LoopyExecutor().differential(prog, schedule, inputs)
    assert fact.status is Status.TESTED, fact.provenance


def statements(schedule: Schedule) -> list[tuple[str, str]]:
    return [(fact.kind, fact.status.value) for fact in schedule.facts()]


# }}}


# {{{ definedness


def definedness(prog) -> list:
    return [fact for fact in prog.facts() if fact.kind == "definedness"]


def test_the_consumer_reads_only_cells_the_producer_stored() -> None:
    (fact,) = definedness(burgers)
    assert fact.statement == (
        "every cell of f that divergence reads, flux stored before it"
    )
    assert fact.status is Status.ASSUMED
    decided = IslOracle().establish(fact)
    assert decided.status is Status.DECIDED
    assert decided.decided_by == "isl"
    assert fact.id.startswith("definedness:")
    assert fact.id.endswith(":f:divergence")


def test_a_consumer_that_reads_the_zeros_is_told_so_and_not_refuted() -> None:
    # Reading the zeros the program made the array with is no error, so the
    # fact says what happens, decided, with a cell that shows it.
    (fact,) = definedness(padded)
    assert fact.status is Status.DECIDED
    assert fact.decided_by == "isl"
    assert fact.statement.startswith(
        "shifted reads cells of f that interior_flux did not store, such as f["
    )
    cell = fact.provenance["witness"]
    size = fact.provenance["witness_params"]["n"]
    assert cell[0] in (0, size - 1)
    assert "zeros" in fact.provenance["reads"]


def test_a_read_through_an_index_array_is_decided_when_every_cell_is_stored() -> None:
    (fact,) = definedness(gathered)
    assert IslOracle().establish(fact).status is Status.DECIDED
    (unknown,) = definedness(gathered_inside)
    assert unknown.status is Status.ASSUMED
    assert unknown.term is None
    assert "not affine" in unknown.provenance["reason"]


def test_a_read_before_any_call_stored_the_array_is_no_edge() -> None:
    # shifted reads the zeros before flux runs: no call stored f before it,
    # so there is no edge to ask about. divergence's edge is decided.
    (fact,) = definedness(early)
    assert fact.provenance["call"] == "divergence"


def test_an_edge_through_a_parameter_has_no_definedness_fact() -> None:
    # The caller passes f and sees it: it is not the program's to store or
    # not, and what its cells hold before flux runs is the caller's.
    assert definedness(through) == []


# }}}


# {{{ fusion


def test_a_fusion_that_runs_a_dependence_backwards_is_refused_with_the_pair() -> None:
    schedule = Schedule(burgers, sizes={"n": 16})
    with pytest.raises(IllegalCast) as caught:
        schedule.fuse("flux", "divergence")
    assert caught.value.fact.kind == "monotone"
    assert caught.value.fact.status is Status.REFUTED
    (source, source_at), (sink, sink_at), sizes = caught.value.witness
    assert (source, sink) == ("flux.S0", "divergence.S0")
    assert source_at["j"] == sink_at["i"] + 1
    assert sizes == {"n": 16}
    message = str(caught.value)
    assert message.startswith("fuse(flux, divergence) illegal: instance flux.S0[")
    # The refusal names the least shift that is accepted, which it checked.
    assert message.endswith(
        "fuse('flux', 'divergence', shift=1) runs every dependence between "
        "them forward"
    )
    assert caught.value.fact.provenance["reason"] == message


def test_the_fused_loop_is_one_loop_and_agrees_with_the_native_run() -> None:
    fused = Schedule(burgers, sizes={"n": 16}).fuse("flux", "divergence", shift=1)
    assert fused.history == ("fuse(flux, divergence, shift=1)",)
    assert fused.key == "burgers[c].fuse('flux', 'divergence', shift=1)"
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    loop = fused.term.stmt("flux.S0").inames[0]
    assert fused._layout.coords["divergence.S0"] == (loop,)
    entry = fused.kernel.default_entrypoint
    insns = {insn.id: insn for insn in entry.instructions}
    assert insns["flux_S0"].within_inames == insns["divergence_S0"].within_inames
    owners = [d for d in entry.domains if loop in d.get_var_names(isl.dim_type.set)]
    assert len(owners) == 1
    for size in (1, 2, 3, 5, 16):
        agrees(burgers, fused, burgers_inputs(size))


def test_a_fusion_is_a_map_per_statement_and_affine_takes_it_as_well() -> None:
    term = burgers.term
    j = term.stmt("flux.S0").inames[0]
    i = term.stmt("divergence.S0").inames[0]
    by_hand = Schedule(burgers).affine(
        f"{{ flux_S0[{j}] -> [k] : k = {j}; divergence_S0[{i}] -> [k] : k = {i} + 1 }}"
    )
    assert statements(by_hand) == [("bijective", "decided"), ("monotone", "decided")]
    assert by_hand.order[-1] == "k"
    agrees(burgers, by_hand, burgers_inputs(9))
    with pytest.raises(ValueError, match="their maps have to make the same new ones"):
        Schedule(burgers).affine(
            f"{{ flux_S0[{j}] -> [k] : k = {j}; divergence_S0[{i}] -> [m] : m = {i} }}"
        )


def test_a_fusion_is_replayed_when_retargeted() -> None:
    fused = Schedule(burgers).fuse("flux", "divergence", shift=1)
    again = fused.retarget("c-source")
    assert again.history == fused.history
    assert statements(again) == statements(fused)


def test_a_fusion_in_place_runs_the_writes_after_the_reads() -> None:
    # update writes u[i] in place, which flux reads at j = i: run one step
    # behind, flux has read it by then.
    fused = Schedule(stepped).fuse("flux", "update", shift=1)
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    for size in (1, 3, 8):
        agrees(stepped, fused, {"u": Arr.from_numpy(velocity(size))})


def test_two_loops_fuse_with_a_shift_each() -> None:
    schedule = Schedule(squares, sizes={"n": 6, "m": 5})
    with pytest.raises(IllegalCast) as caught:
        schedule.fuse("square2", "down")
    assert str(caught.value).endswith(
        "fuse('square2', 'down', shift=(1, 0)) runs every dependence between "
        "them forward"
    )
    fused = schedule.fuse("square2", "down", shift=(1, 0))
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    for rows, cols in ((1, 1), (2, 3), (6, 5)):
        a = np.arange(rows * cols, dtype=float).reshape(rows, cols) / 7.0
        agrees(
            squares,
            fused,
            {"a": Arr.from_numpy(a), "b": Arr.from_numpy(np.zeros((rows, cols)))},
        )
    with pytest.raises(ValueError, match="one shift per loop"):
        schedule.fuse("square2", "down", shift=1)


def test_a_statement_deeper_in_the_producer_moves_with_its_row() -> None:
    # rows_then_cells has a statement in r and one in r and c; the domain of
    # c is nested in r's, and moves along the producer's map with it.
    fused = Schedule(rows).fuse("rows_then_cells", "row_sum")
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    assert fused.buildable == (True, "")
    for size, width in ((1, 1), (3, 2), (5, 4)):
        a = np.arange(size * width, dtype=float).reshape(size, width) + 1.0
        agrees(
            rows,
            fused,
            {
                "a": Arr.from_numpy(a),
                "z": Arr.zeros(size),
                "g": Arr.from_numpy(np.zeros((size, width))),
            },
        )


def test_the_outer_loop_of_a_nest_fuses_with_a_loop_of_one_level() -> None:
    # square2's two loops are one domain of the kernel, since no statement
    # leaves the nest; the fusion takes the outer one and cuts the domain
    # after it, as the lowering cuts a nest a statement leaves. Each row is
    # squared and then summed, in one loop over the rows.
    fused = Schedule(squared_rows).fuse("square2", "row_total")
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    assert fused.buildable == (True, "")
    loop = squared_rows.term.stmt("square2.S0").inames[0]
    entry = fused.kernel.default_entrypoint
    insns = {insn.id: insn for insn in entry.instructions}
    assert loop in insns["row_total_S0"].within_inames
    for rows, cols in ((1, 1), (3, 4), (5, 2)):
        a = np.arange(rows * cols, dtype=float).reshape(rows, cols) / 3.0
        agrees(squared_rows, fused, {"a": Arr.from_numpy(a), "z": Arr.zeros(rows)})
    # And with the squares computed where the sum reads them, no g at all.
    substituted = fused.substitute("g")
    assert "g" not in substituted.kernel.default_entrypoint.temporary_variables
    a = np.arange(12, dtype=float).reshape(3, 4)
    agrees(squared_rows, substituted, {"a": Arr.from_numpy(a), "z": Arr.zeros(3)})


def test_a_call_between_the_fused_runs_after_their_loop() -> None:
    # front reads the f flux stores, so it runs after the fused loop; back
    # reads h, which front writes, but at cells front does not write. The
    # lowering's instruction dependencies are by array, and loopy found no
    # order for the three (a CycleError) until those were cut back to the
    # dependences the casts were checked against.
    fused = Schedule(between).fuse("flux", "back")
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    entry = fused.kernel.default_entrypoint
    insns = {insn.id: insn for insn in entry.instructions}
    assert "front_S0" not in insns["back_S0"].depends_on
    assert "flux_S0" in insns["front_S0"].depends_on
    for size in (1, 2, 5, 8):
        inputs = {"u": Arr.from_numpy(velocity(size)), "rhs": Arr.zeros(size)}
        agrees(between, fused, inputs)


@pytest.mark.parametrize(
    ("producer", "consumer", "shift", "error", "message"),
    [
        ("nothing", "divergence", 0, ValueError, "names no statement of burgers"),
        ("divergence", "flux", 0, ValueError, "flux comes before divergence"),
        ("flux", "flux.S0", 0, ValueError, "is named on both sides"),
        ("flux", "divergence", True, TypeError, "a shift is a whole number"),
        ("flux", "divergence", (1, 0), ValueError, "takes 1 whole number"),
    ],
)
def test_a_fusion_names_two_sides_in_order(
    producer: str, consumer: str, shift, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        Schedule(burgers).fuse(producer, consumer, shift=shift)


def test_a_statement_s_map_moves_every_loop_of_the_step_it_runs_in() -> None:
    # rows_then_cells.S1 runs in the row loop, which S0's map takes, and in
    # its cell loop: a map of the cell loop alone would leave it in a row
    # loop the step replaces.
    term = rows.term
    row = term.stmt("rows_then_cells.S0").inames[0]
    cell = term.stmt("rows_then_cells.S1").inames[1]
    with pytest.raises(ValueError, match=f"S1 runs in {row} as well"):
        Schedule(rows).affine(
            f"{{ rows_then_cells_S0[{row}] -> [q] : q = {row}; "
            f"rows_then_cells_S1[{cell}] -> [q] : q = {cell} }}"
        )


def test_statements_of_one_loop_cannot_be_fused_into_it() -> None:
    fused = Schedule(burgers).fuse("flux", "divergence", shift=1)
    with pytest.raises(ValueError, match="already run in"):
        fused.fuse("flux", "divergence")


def test_a_parallel_loop_after_fusion_is_refused_for_the_edge_it_carries() -> None:
    # In the fused loop, divergence at j reads f[j - 2], which flux stored two
    # steps earlier: on a hardware axis those are two work items. The C
    # target has no hardware axes, but the casts are asked before the target
    # is.
    fused = Schedule(burgers).fuse("flux", "divergence", shift=1)
    loop = fused.order[-1]
    with pytest.raises(IllegalCast, match="on another work item"):
        fused.tag(**{loop: "g.0"})


# }}}


# {{{ substitution


def test_a_substituted_array_is_computed_where_it_is_read() -> None:
    schedule = Schedule(burgers, sizes={"n": 16}).substitute("f")
    assert schedule.substituted == ("f",)
    assert statements(schedule) == [
        ("definedness", "decided"),
        ("bijective", "decided"),
        ("monotone", "decided"),
    ]
    entry = schedule.kernel.default_entrypoint
    assert "f" not in entry.temporary_variables
    assert {insn.id for insn in entry.instructions} == {"divergence_S0"}
    assert entry.substitutions
    for size in (1, 2, 3, 16):
        agrees(burgers, schedule, burgers_inputs(size))


def test_a_fused_array_is_substituted_into_the_fused_loop() -> None:
    schedule = (
        Schedule(burgers).fuse("flux", "divergence", shift=1).substitute("f")
    )
    assert schedule.history == (
        "fuse(flux, divergence, shift=1)",
        "substitute('f')",
    )
    assert "f" not in schedule.kernel.default_entrypoint.temporary_variables
    for size in (1, 4, 11):
        agrees(burgers, schedule, burgers_inputs(size))
    again = schedule.retarget("c-source")
    assert again.history == schedule.history


def test_a_substitution_drops_the_dependences_through_the_array() -> None:
    # Fused and substituted, nothing passes from one step of the loop to the
    # next, so it may go on a hardware axis; fused alone, it may not (above).
    schedule = Schedule(burgers).fuse("flux", "divergence", shift=1).substitute("f")
    tagged = schedule.tag(**{schedule.order[-1]: "g.0"})
    monotone = [fact for fact in tagged.facts() if fact.kind == "monotone"][-1]
    assert monotone.status is Status.DECIDED
    assert "within one work item" in monotone.statement
    # The target is the C target, which has no hardware axes.
    assert not tagged.buildable[0]


def test_a_read_of_a_cell_the_producer_does_not_store_is_refused() -> None:
    with pytest.raises(IllegalCast) as caught:
        Schedule(padded).substitute("f")
    fact = caught.value.fact
    assert (fact.kind, fact.status) == ("definedness", Status.REFUTED)
    message = str(caught.value)
    assert message.startswith("substitute('f') illegal: shifted.S0 reads f[")
    assert "which interior_flux.S0 does not store" in message
    assert "would see the zeros" in message


def test_a_read_before_the_producer_stores_the_cell_is_refused() -> None:
    with pytest.raises(IllegalCast) as caught:
        Schedule(early, sizes={"n": 4}).substitute("f")
    assert caught.value.fact.kind == "definedness"
    message = str(caught.value)
    assert "instance shifted.S0[" in message
    assert "before flux.S0[" in message
    assert "stores it (at n=4, as hinted)" in message


def test_a_write_between_the_producer_and_a_read_is_refused() -> None:
    # bump writes u after flux read it and before divergence reads f: the
    # value computed again at the read would be the bumped one.
    with pytest.raises(IllegalCast) as caught:
        Schedule(bumped, sizes={"n": 8}).substitute("f")
    assert caught.value.fact.kind == "monotone"
    message = str(caught.value)
    assert message.startswith("substitute('f') illegal: instance divergence.S0[")
    assert " reads u[" in message
    assert "overwritten by bump.S0[" in message


def test_a_later_write_of_what_the_producer_read_waits_for_the_reads() -> None:
    # bump writes u after divergence has read f; computed again at the
    # read, f reads u, so bump has to wait for divergence, which nothing in
    # the lowered kernel said: bump depended on flux, which is gone.
    schedule = Schedule(bumped_after).substitute("f")
    insns = {
        insn.id: insn for insn in schedule.kernel.default_entrypoint.instructions
    }
    assert "divergence_S0" in insns["bump_S0"].depends_on
    for size in (1, 3, 8):
        agrees(bumped_after, schedule, burgers_inputs(size))
    # Fused one step behind the divergence, bump writes u[k] in the step in
    # which the divergence reads it to compute f[k]: the divergence has to
    # come first in the step, and the kernel says so.
    fused = Schedule(bumped_after).fuse("divergence", "bump", shift=1)
    with pytest.raises(IllegalCast):
        Schedule(bumped_after).fuse("divergence", "bump").substitute("f")
    substituted = fused.substitute("f")
    insns = {
        insn.id: insn for insn in substituted.kernel.default_entrypoint.instructions
    }
    assert "divergence_S0" in insns["bump_S0"].depends_on
    for size in (1, 2, 6):
        agrees(bumped_after, substituted, burgers_inputs(size))


def test_an_in_place_consumer_cannot_have_the_array_computed_again() -> None:
    # update writes u[i - 1] one step before it reads f[i - 1], which is
    # 0.5 * u[i - 1]**2 of the u before the update.
    with pytest.raises(IllegalCast, match="overwritten by update.S0"):
        Schedule(stepped).substitute("f")
    # Stored, the same program fuses and runs.
    agrees(
        stepped,
        Schedule(stepped).fuse("flux", "update", shift=1),
        {"u": Arr.from_numpy(velocity(6))},
    )


def test_a_read_through_an_index_array_is_substituted() -> None:
    schedule = Schedule(gathered).substitute("f")
    for size in (1, 4, 7):
        idx = np.random.default_rng(size).permutation(size).astype(np.int64)
        agrees(
            gathered,
            schedule,
            {
                "u": Arr.from_numpy(velocity(size)),
                "idx": Arr.from_numpy(idx),
                "y": Arr.zeros(size),
            },
        )


@pytest.mark.parametrize(
    ("prog", "array", "message"),
    [
        (burgers, "u", "u is a parameter of burgers, which its caller passes"),
        (burgers, "g", "g is no array of burgers"),
        (twice, "f", "f is written by flux.S0, flux@2.S0"),
        (halved, "f", r"stores f\[j // 2\]|at its own loop variables"),
        (summed, "f", "stores a sum"),
        (unread, "f", "nothing in unread reads f"),
    ],
)
def test_what_a_substitution_cannot_take_is_named(
    prog, array: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Schedule(prog).substitute(array)


def test_a_substitution_through_two_arrays_is_refused_either_way() -> None:
    # The dependences carried over are those of what the producer reads, and
    # through two arrays that is what the first producer reads, which the
    # second step does not see: refused, whichever comes first.
    with pytest.raises(ValueError, match="doubled.S0 reads f, which is computed"):
        Schedule(chained).substitute("f").substitute("g")
    with pytest.raises(ValueError, match="doubled.S0 reads f and runs no more"):
        Schedule(chained).substitute("g").substitute("f")
    for array in ("f", "g"):
        agrees(chained, Schedule(chained).substitute(array), burgers_inputs(7))


def test_a_statement_taken_out_leaves_no_loop_to_name() -> None:
    schedule = Schedule(burgers).substitute("f")
    term = burgers.term
    gone = {term.stmt("flux.S0").inames[0], term.stmt("f.zeros").inames[0]}
    assert not gone & set(schedule.order)
    assert schedule.order == term.stmt("divergence.S0").inames
    with pytest.raises(ValueError, match="is not an iname"):
        schedule.tag(**{min(gone): "g.0"})
    with pytest.raises(ValueError, match="runs no more"):
        schedule.fuse("flux", "divergence")
    loop = schedule.order[0]
    split = schedule.split(loop, 4)
    assert [fact.status.value for fact in split.facts()][-2:] == ["decided"] * 2
    agrees(burgers, split, burgers_inputs(10))


def test_an_array_is_substituted_once() -> None:
    schedule = Schedule(burgers).substitute("f")
    with pytest.raises(ValueError, match="substituted already"):
        schedule.substitute("f")


# }}}
