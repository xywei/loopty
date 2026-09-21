"""Point to point: the near field of a fast multipole method, as a ragged kernel.

This is the shape the user's boxtree and sumpy code has. Points are sorted into
boxes; each target box has an interaction list of neighbouring source boxes; the
direct, or point-to-point, part of the method sums the Coulomb kernel over the
sources in those boxes, and everything further away is handled by expansions.
The structure is a two-level dependent sum: target box, then its source boxes,
then the sources in each of them.

loopty lowers one ragged level, so the two levels are flattened where they come
from, in the data: :func:`interaction_lists` walks box by box and produces, for
each target point, the list of source indices it interacts with. That list is
the dependent fiber the kernel iterates, and its length differs per target, so
the term's inner bound is the count of that row and isl carries it as a
parameter. Flattening the list is not a workaround for the demo's sake: it is
what a near-field kernel is handed on a device, one list per target.

Two things are on show beyond the ragged shape.

*The self-interaction is a guard, not an ``if``.* A target's own box is in its
interaction list, so the list contains the target itself, and the Coulomb kernel
is singular there. ``with when(lst[t, j] != t)`` records the condition: under
tracing it becomes the statement's predicate, and under plain ``python`` it
masks the write. Masking is on the write and not on the arithmetic, so the self
pair is still evaluated natively and produces an infinity that is thrown away.
That is why the contribution is zeroed first by its own statement, and why
:func:`coulomb` tells numpy not to warn about the division it is about to throw
away.

*The indirection is in bounds by type.* ``lst: Arr[Fin[n], Fin[cnt], Fin[n]]``
says the entries of the list are points of ``Fin[n]``, which is exactly the
index type of the coordinate arrays, so ``x[lst[t, j]]`` is discharged by the
typing rule with no call to isl.

Run this file three ways.

``python examples/p2p.py``
    Builds the boxes, runs the kernel natively, and checks it against a direct
    sum over the same lists.

``lanky check examples/p2p.py``
    Prints the ledger of the kernel's obligations.

``loopty run examples/p2p.py``
    Compiles it for the C target and compares with the native run.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Nat, Real
from pymbolic.primitives import Call, Variable

from loopty import Arr, Fin, Schedule, kernel, reduce_sum, when

#: A four by four grid of boxes over the unit square, and this many points in
#: it. Tiny, and still ragged: the lists run from a handful to a few dozen.
BOXES, POINTS = 4, 16


def coulomb(r2: float) -> float:
    """``1 / r`` from the squared distance, symbolically or numerically.

    Two modes, because the body runs both ways. On numbers this is numpy; on a
    term it builds a call to ``sqrt``, which loopy resolves against the target's
    own library. loopty has no elementary-function surface of its own yet, so a
    kernel that needs one writes the call, as here.
    """
    if isinstance(r2, int | float | np.floating):
        # The masked self pair reaches this with r2 == 0 (see the module
        # docstring): the infinity is discarded by the mask, so the warning
        # would be about a number nobody reads.
        with np.errstate(divide="ignore"):
            return 1.0 / np.sqrt(r2)
    return 1 / Call(Variable("sqrt"), (r2,))


@kernel
def p2p(
    cnt: Arr[Fin[n], Nat],
    lst: Arr[Fin[n], Fin[cnt], Fin[n]],
    x: Arr[Fin[n], Real],
    y: Arr[Fin[n], Real],
    q: Arr[Fin[n], Real],
    term: Arr[Fin[n], Fin[cnt], Real],
    pot: Arr[Fin[n], Real],
):
    """The near-field potential at every target, over its interaction list.

    ``term`` is the per-pair contribution, written out rather than accumulated:
    a guarded contribution has to be zero where the guard is false, and a
    separate statement saying so is both what the native mask needs and what
    lets the sum over the row stay a reduction instead of a loop-carried
    accumulation.
    """
    for t in pot.dom:
        for j in lst.dom[t]:
            term[t, j] = 0.0
            with when(lst[t, j] != t):
                dx = x[t] - x[lst[t, j]]
                dy = y[t] - y[lst[t, j]]
                term[t, j] = q[lst[t, j]] * coulomb(dx * dx + dy * dy)
        pot[t] = reduce_sum(term[t, j] for j in lst.dom[t])


# {{{ the boxes and their interaction lists


def interaction_lists(
    xs: np.ndarray, ys: np.ndarray, boxes: int = BOXES
) -> list[list[int]]:
    """For each point, the sources in the neighbouring boxes of its own box.

    The two levels of the sum are here: a target box has a list of neighbouring
    source boxes, and each source box has its points. Walking them in that order
    and concatenating is the flattening the kernel is handed. Points of the same
    box share a list, which is why a real implementation loops over boxes and
    this one over targets: the flattened form costs storage and buys a loop nest
    that is one ragged level deep.
    """
    index = [
        (min(int(px * boxes), boxes - 1), min(int(py * boxes), boxes - 1))
        for px, py in zip(xs, ys, strict=True)
    ]
    members: dict[tuple[int, int], list[int]] = {}
    for point, box in enumerate(index):
        members.setdefault(box, []).append(point)

    lists: list[list[int]] = []
    for bx, by in index:
        neighbours = [
            (bx + dx, by + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if 0 <= bx + dx < boxes and 0 <= by + dy < boxes
        ]
        sources: list[int] = []
        for box in sorted(neighbours):
            sources.extend(members.get(box, ()))
        lists.append(sources)
    return lists


def scene(points: int = POINTS, seed: int = 0) -> dict:
    """Random points with charges, and the ragged arrays the kernel takes."""
    rng = np.random.default_rng(seed)
    xs, ys = rng.random(points), rng.random(points)
    charges = rng.normal(size=points)
    lists = interaction_lists(xs, ys)
    counts = [len(entry) for entry in lists]
    flat = [source for entry in lists for source in entry]
    return {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "lst": Arr.ragged(counts, values=flat, dtype=np.int64),
        "x": Arr.from_numpy(xs),
        "y": Arr.from_numpy(ys),
        "q": Arr.from_numpy(charges),
        "term": Arr.ragged(counts),
        "pot": Arr.zeros(points),
    }


def direct(data: dict) -> np.ndarray:
    """The same near-field sum written out in numpy, as the thing to agree with."""
    xs, ys, charges = data["x"].numpy(), data["y"].numpy(), data["q"].numpy()
    offsets, flat = data["lst"].offsets, data["lst"].numpy()
    out = np.zeros(len(xs))
    for target in range(len(xs)):
        for a in range(offsets[target], offsets[target + 1]):
            source = int(flat[a])
            if source == target:
                continue
            r2 = (xs[target] - xs[source]) ** 2 + (ys[target] - ys[source]) ** 2
            out[target] += charges[source] / np.sqrt(r2)
    return out


# }}}


#: Split the targets and sum each row as a tree: the first step is a pure
#: reindexing, the second is a reassociation, and both are checked.
split_targets = (
    Schedule(p2p, target="c")
    .split("t", 4, inner="t_in", outer="t_out")
    .realize("pot", tree=True)
    .example(**scene())
)


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives the kernel."""
    return scene()


def main() -> int:
    """Run the near field natively, then compiled, and compare both."""
    from loopty.executor import LoopyExecutor

    data = scene()
    counts = data["cnt"].numpy()
    p2p(
        data["cnt"],
        data["lst"],
        data["x"],
        data["y"],
        data["q"],
        data["term"],
        data["pot"],
    )
    print(f"{POINTS} points in a {BOXES} by {BOXES} grid of boxes")
    print(f"interaction lists of {counts.min()} to {counts.max()} sources")
    print("potential =", np.array2string(data["pot"].numpy(), precision=3))
    wanted = direct(data)
    print("direct    =", np.array2string(wanted, precision=3))
    native_ok = bool(np.allclose(data["pot"].numpy(), wanted))
    print(f"the kernel agrees with the direct sum: {native_ok}")

    print()
    print(f"schedule: {split_targets!r}")
    for fact in split_targets.facts():
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")

    print()
    fact = LoopyExecutor().differential(p2p, split_targets, scene())
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )
    return 0 if native_ok and fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
