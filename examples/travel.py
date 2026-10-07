"""Facts that travel: a program's calls assume what earlier calls established.

A kernel's requirements on its inputs are its argument types. ``gather``
reads ``x[perm[i]]``, in bounds by the element type of ``perm``, and natively
its contract checks that every cell of ``perm`` is a point of ``Fin[n]`` when
it is called. A program is one compiled call, whose contract checks its
arguments when it starts, so where an earlier call wrote ``perm`` the check
is an obligation of the program: decided by isl under what the earlier call
established, or made by the compiled program between the two calls.

Run this file three ways.

``python examples/travel.py``
    Runs each program natively, says how each requirement was met, compares
    each compiled program with its native run, and prints the code of the
    program whose requirement is checked.

``lanky check examples/travel.py``
    The kernels' facts, their postconditions tested against their native
    runs, and each program's requirements: ``decided`` by isl and resting on
    the postcondition they used, or ``assumed`` and checked when it runs.

``loopty run examples/travel.py``
    Every kernel alone, and every program as one kernel, compiled and
    compared with the native run.

What is worth reading here
--------------------------

*A postcondition is a hypothesis.* ``number`` says what it writes,
``perm[i] == n - 1 - i``, and the requirement of ``gather`` that follows it
is decided under that: a cell ``n - 1 - i`` with ``0 <= i < n`` is a point of
``Fin[n]``. The fact rests on ``number``'s postcondition, which is ``tested``
against ``number``'s native runs, so it is worth a test.

*So is a theorem the program cites.* ``through`` scans the counts into the
offsets that ``rowsums`` reads the rows of ``val`` through. ``rowsums``'s
contract compares them with the offsets the counts give, which is ``scan``'s
postcondition verbatim, so isl decides it from that alone, and the fact does
not rest on ``scan_monotone``, which the program cites with ``uses=``. The
offsets ``through`` is passed are overwritten before anything reads a row
through them, so its contract does not compare them with ``val`` on entry.

*Where nothing decides it, the program checks it.* ``number_quiet`` writes
the same permutation and says nothing about it. The requirement of the
``gather`` after it is a checked point: the compiled program checks every
cell of ``perm`` between the calls, runs ``gather`` only if they all passed,
and stops with an error where the native ``gather`` would be refused.

*A callee's fact can be decided where it is called.* ``weigh`` reads a flat
buffer, ``val[off[r] + j]``, which is in bounds only for offsets that lay the
rows out inside ``val``: alone, its in-bounds fact is ``assumed``. In
``flat`` it is called after ``scan_flat``, and there ``off[r] + j < off[r] +
cnt[r] == off[r + 1] <= nnz``, through the cells ``off[r]``, ``off[r + 1]``
and ``cnt[r]``: the scan's postcondition gives the equality, and the element
type of ``off`` the bound. Nothing says the scan's offsets stay below ``nnz``,
so that type is a checked point of ``flat``, and the in-bounds fact is
decided under the postcondition and the check.
"""

from __future__ import annotations

import numpy as np
from lanky import theorem
from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, Schedule, kernel, program, reduce_sum

# {{{ a permutation, computed and then used


@kernel
def number(perm: Arr[Fin[n], Fin[n]]) -> all(perm[i] == n - 1 - i for i in Fin[n]):
    """Reverse the cells, and say so."""
    for i in perm.dom:
        perm[i] = perm.dom.size - 1 - i


@kernel
def number_quiet(perm: Arr[Fin[n], Fin[n]]):
    """Reverse the cells, and say nothing about it."""
    for i in perm.dom:
        perm[i] = perm.dom.size - 1 - i


@kernel
def gather(
    perm: Arr[Fin[n], Fin[n]], x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]
):
    """``y[i] = x[perm[i]]``, in bounds by the element type of ``perm``."""
    for i in y.dom:
        y[i] = x[perm[i]]


@program
def permuted(perm, x, y):
    """gather's requirement on perm is number's postcondition."""
    number(perm)
    gather(perm, x, y)


@program
def checked(perm, x, y):
    """Nothing says what number_quiet writes, so the program checks it."""
    number_quiet(perm)
    gather(perm, x, y)


# }}}


# {{{ offsets a scan computes, and rows read through them


@kernel
def scan(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """Exclusive prefix sum of the counts: where each row starts."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@theorem
def scan_monotone(
    n: Nat,
    cnt: Fn[Fin[n], Nat],
    off: Fn[Fin[n + 1], Nat],
    h0: off(0) == 0,
    hs: all(off(r + 1) == off(r) + cnt(r) for r in Fin[n]),
) -> all(off(a) <= off(b) for a in Fin[n + 1] for b in Fin[n + 1] if a <= b):
    """The offsets a scan produces are monotone."""


@kernel
def rowsums(
    cnt: Arr[Fin[n], Nat],
    off: Arr[Fin[n + 1], Nat],
    val: Arr[Fin[n], Fin[cnt], Real],
    y: Arr[Fin[n], Real],
):
    """Sum each row of ``val``, read through the offsets it declares."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


@program(uses=[scan_monotone])
def through(cnt, off, val, y):
    """Scan the counts into the offsets, then read the rows through them."""
    scan(cnt, off)
    rowsums(cnt, off, val, y)


# }}}


# {{{ a flat buffer read through the offsets a scan computes


@kernel
def scan_flat(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Fin[nnz + 1]]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """The same scan, into offsets that are points of a buffer of ``nnz`` cells."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@kernel
def weigh(
    cnt: Arr[Fin[n], Nat],
    off: Arr[Fin[n + 1], Fin[nnz + 1]],
    wt: Arr[Fin[n], Fin[cnt], Real],
    val: Arr[Fin[nnz], Real],
    y: Arr[Fin[n], Real],
):
    """Weigh the entries of each row, read from the flat buffer ``val``.

    ``val[off[r] + j]`` is in bounds only for offsets that lay the rows out
    inside ``val``, which nothing in this kernel says: alone, the fact is
    ``assumed``.
    """
    for r in y.dom:
        y[r] = reduce_sum(wt[r, j] * val[off[r] + j] for j in wt.dom[r])


@program
def flat(cnt, off, wt, val, y):
    """After the scan, the flat access is in bounds."""
    scan_flat(cnt, off)
    weigh(cnt, off, wt, val, y)


# }}}


def matrix(counts: list[int], seed: int = 0) -> dict:
    """Rows of these lengths, the offsets to compute, and a result."""
    rng = np.random.default_rng(seed)
    return {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "off": Arr.zeros(len(counts) + 1, dtype=np.int64),
        "val": Arr.ragged(counts, values=rng.normal(size=sum(counts))),
        "y": Arr.zeros(len(counts)),
    }


def permutation(size: int = 5) -> dict:
    """An array to number, values to gather, and a result."""
    return {
        "perm": Arr.zeros(size, dtype=np.int64),
        "x": Arr.from_numpy(np.arange(10.0, 10.0 + size)),
        "y": Arr.zeros(size),
    }


def flat_matrix(counts: list[int], seed: int = 0) -> dict:
    """Rows of these lengths: weights by row, and a flat buffer of entries."""
    rng = np.random.default_rng(seed)
    data = matrix(counts, seed)
    data["wt"] = data.pop("val")
    data["val"] = Arr.from_numpy(rng.normal(size=sum(counts)))
    return data


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives each kernel in this file, and each program."""
    counts = [2, 0, 3, 1]
    data = matrix(counts)
    scanned = matrix(counts)
    scanned["off"] = Arr.from_numpy(np.array([0, 2, 2, 5, 6], dtype=np.int64))
    weighed = flat_matrix(counts)
    weighed["off"] = Arr.from_numpy(np.array([0, 2, 2, 5, 6], dtype=np.int64))
    numbered = permutation()
    numbered["perm"] = Arr.from_numpy(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    return {
        "number": {"perm": permutation()["perm"]},
        "number_quiet": {"perm": permutation()["perm"]},
        "gather": numbered,
        "permuted": permutation(),
        "checked": permutation(),
        "scan": {"cnt": data["cnt"], "off": data["off"]},
        "rowsums": scanned,
        "through": matrix(counts),
        "scan_flat": {"cnt": data["cnt"], "off": data["off"]},
        "weigh": weighed,
        "flat": flat_matrix(counts),
    }


def main() -> int:
    """Run each program natively, say how its requirements were met, compare."""
    from lanky.terms import render

    from loopty.executor import LoopyExecutor, emit_code

    ok = True
    for prog in (permuted, through, checked, flat):
        inputs = example_inputs()[prog.__name__]
        prog(**{name: value.copy() for name, value in inputs.items()})
        print(f"{prog.__name__}:")
        for requirement in prog.term.requirements:
            if requirement.decided:
                how = "decided under " + ", ".join(h.source for h in requirement.used)
            else:
                how = "checked when it runs"
            print(f"  {requirement.statement}")
            print(f"    {how}")
        for fact in prog.facts():
            if fact.kind == "in-bounds":
                used = ", ".join(fact.provenance["used"])
                print(f"  {fact.statement}")
                print(f"    decided under {used}")
        fact = LoopyExecutor().differential(prog, Schedule(prog), inputs)
        ok = ok and fact.status.value == "tested"
        for name, detail in fact.provenance["outputs"].items():
            print(
                f"  {name}: difference {detail['difference']:.3g} within "
                f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
                f"{fact.status.value}"
            )
    post = [fact for fact in number.facts() if fact.kind == "postcondition"]
    print()
    print(f"number's postcondition, {render(number.term.post)}: {post[0].status.value}")
    print()
    print(emit_code(checked))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
