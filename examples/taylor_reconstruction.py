"""sumpy's compressed Taylor reconstruction: theorem, kernel and schedule in one file.

One file, three layers, and one ledger. It is the seed of the demo's flagship,
sumpy's L2P for the 2-D Laplace kernel.

``python examples/taylor_reconstruction.py``
    Reconstructs the Taylor coefficients of ``log r`` on the triangle ``a + b
    <= p`` from the ``2p + 1`` sumpy stores, natively, compares each with the
    derivative it stands for, and runs the schedule compiled.

``lanky check examples/taylor_reconstruction.py``
    The ledger, in one table: the theorem, proved by Lean with Mathlib for
    every ``p``; the kernel's facts, its postcondition decided from what its
    statements store; the theorem applied to the postcondition, which is what
    the kernel's output is; and the schedule's casts.

``loopty run examples/taylor_reconstruction.py``
    The kernel, scheduled, compiled for the C target and compared with the
    native run.

The mathematics. A Taylor expansion of a kernel ``G`` needs its derivatives
``D(a, b) = d^(a+b) G / dx^a dy^b`` with ``a + b`` up to the order ``p``. The
2-D Laplace kernel is harmonic, so ``D(a + 2, b) == -D(a, b + 2)``, and a
derivative with ``a >= 2`` follows from those with ``a < 2``: sumpy stores
those, ``2p + 1`` of the ``(p + 1)(p + 2)/2``, and reconstructs the rest by that
recurrence (``LinearPDEBasedExpansionTermsWrangler``). The kernel below is
that reconstruction, written as loops over the triangle.

What links the layers:

- ``on_the_triangle`` is the theorem: a table ``R`` that copies a table ``c``
  where ``a < 2`` and follows the recurrence inside the triangle is ``D``
  there, for any ``D`` that satisfies the PDE and whose entries ``c`` holds
  where ``a < 2``. It is lanky's sumpy demonstration's claim on the triangle
  of every order, with the stored entries named.
- ``reconstruct`` is the kernel. Its postcondition is what it stores: the
  copies, and the recurrence. That is decided from its term, since every
  cell of ``R`` is stored once and read only after its store
  (``loopty.dataflow``), and with Lean, proved.
- ``@kernel(uses=[on_the_triangle])`` applies the theorem to the
  postcondition: its copies and recurrence are two of the theorem's
  hypotheses, so ``R[a, b] == D(a)(b)`` on the triangle for every ``D``
  that the other two, ``pde`` and ``stored``, hold of. The row ``by
  on_the_triangle, for every D with pde and stored`` rests on the
  theorem's proof and on the postcondition, and Lean checks that the
  theorem applies.
- ``cpu`` schedules the kernel, and each step is a cast, checked; its casts
  are rows of the same ledger. Running ``b`` outside ``a`` is refused, with
  the pair of instances it would reorder: ``R[a, b]`` reads ``R[a - 2, b +
  2]``, stored in an earlier row.
"""

from __future__ import annotations

import math

from lanky import theorem
from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, Schedule, Where, kernel

#: The order. sumpy's 2-D benchmarks use 10 to 20; small here, so that every
#: command runs in seconds.
ORDER = 6

#: The point the derivatives are taken at: the target less the source, ``x + iy``.
POINT = complex(0.6, 0.8)


# {{{ the theorem


@theorem
def on_the_triangle(
    p: Nat,
    c: Fn[Nat, Fn[Nat, Real]],
    D: Fn[Nat, Fn[Nat, Real]],
    R: Fn[Nat, Fn[Nat, Real]],
    pde: all(D(a + 2)(b) + D(a)(b + 2) == 0 for a in Nat for b in Nat),
    stored: all(
        c(a)(b) == D(a)(b)
        for a in Fin[p + 1]
        for b in Fin[p + 1]
        if a < 2
        if a + b <= p
    ),
    copies: all(
        R(a)(b) == c(a)(b)
        for a in Fin[p + 1]
        for b in Fin[p + 1]
        if a < 2
        if a + b <= p
    ),
    recurrence: all(
        R(a + 2)(b) == -R(a)(b + 2)
        for a in Fin[p + 1]
        for b in Fin[p + 1]
        if a + b + 2 <= p
    ),
) -> all(R(a)(b) == D(a)(b) for a in Fin[p + 1] for b in Fin[p + 1] if a + b <= p):
    """What ``c`` stores of a harmonic ``D``, reconstructed by the recurrence, is ``D``.

    By induction on ``a``: below 2 the copy is the stored entry, which is
    ``D``'s; at ``a + 2`` the recurrence, the induction hypothesis two ``y``'s
    up and the PDE make it ``D``'s too. Lean proves it for every ``p``, with
    Mathlib, Python having found which hypotheses to combine where.
    """


# }}}


# {{{ the kernel


@kernel(uses=[on_the_triangle])
def reconstruct(
    c: Arr[Where[a : Fin[2], b : Fin[p + 1], a + b <= p], Real],
    R: Arr[Where[a : Fin[p + 1], b : Fin[p + 1], a + b <= p], Real],
) -> all(
    R[a, b] == c[a, b] for a in Fin[p + 1] for b in Fin[p + 1] if a < 2 if a + b <= p
) & all(
    R[a + 2, b] == -R[a, b + 2]
    for a in Fin[p + 1]
    for b in Fin[p + 1]
    if a + b + 2 <= p
):
    """Every coefficient on the triangle, from the ``2p + 1`` stored ones.

    ``G_xx = -G_yy``: a coefficient with two more ``x``'s is minus the one
    with two more ``y``'s, which an earlier row of the triangle stored.
    """
    for a, b in R.dom:
        if a < 2:
            R[a, b] = c[a, b]
        else:
            R[a, b] = -R[a - 2, b + 2]


#: The domains of the stored coefficients and of the triangle.
STORED = reconstruct.arg_types["c"].domain
TRIANGLE = reconstruct.arg_types["R"].domain


# }}}


# {{{ the derivatives, and the inputs


def derivative(a: int, b: int, z: complex = POINT) -> float:
    """``d^(a+b) log|z| / dx^a dy^b``, in closed form.

    ``log|z|`` is the real part of ``log z``, whose ``n``-th derivative in ``z``
    is ``(-1)^(n-1) (n-1)! / z^n``, and ``d/dy`` is ``i d/dz`` on an analytic
    function.
    """
    n = a + b
    if n == 0:
        return math.log(abs(z))
    value = (1j) ** b * (-1) ** (n - 1) * math.factorial(n - 1) / z**n
    return float(value.real)


def stored(order: int = ORDER) -> Arr:
    """What sumpy stores: the derivatives with ``a < 2``, over ``STORED``."""
    cells = [(a, b) for a in range(2) for b in range(order + 1) if a + b <= order]
    return Arr.from_cells(STORED, [derivative(a, b) for a, b in cells], p=order)


def example_inputs() -> dict:
    """The stored coefficients, and a triangle of zeros to reconstruct into."""
    return {"c": stored(), "R": Arr.zeros(TRIANGLE, p=ORDER)}


# }}}


# {{{ the schedule

#: Rows of the triangle two at a time: a cast, checked, and run compiled.
cpu = (
    Schedule(reconstruct, target="c")
    .split(reconstruct.a, 2)
    .example(**example_inputs())
)


def rejected_interchange() -> Schedule:
    """Columns outside rows: refused, with the pair of instances it reorders."""
    return Schedule(reconstruct, target="c").interchange(reconstruct.b, reconstruct.a)


# }}}


def main() -> int:
    """Reconstruct natively, compare with the derivatives, and run compiled."""
    from loopty.executor import LoopyExecutor
    from loopty.schedule import IllegalCast

    data = example_inputs()
    reconstruct(**data)
    cells = [
        (a, b) for a in range(ORDER + 1) for b in range(ORDER + 1) if a + b <= ORDER
    ]
    got = data["R"].cells()
    worst = max(
        abs(value - derivative(a, b)) / max(1.0, abs(derivative(a, b)))
        for (a, b), value in zip(cells, got, strict=True)
    )
    print(
        f"order {ORDER}: {len(cells)} coefficients on the triangle from the "
        f"{2 * ORDER + 1} stored"
    )
    print(f"largest difference from the derivative: {worst:.1e}")
    print(
        f"at z = {POINT}: d^6 log r / dx^6 = {derivative(6, 0):.6f}, "
        f"reconstructed {got[cells.index((6, 0))]:.6f}"
    )
    print()
    print(f"schedule: {cpu!r}")
    for fact in cpu.facts():
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")
    try:
        rejected_interchange()
    except IllegalCast as exc:
        print(f"interchange(b, a) -> IllegalCast: {exc}")
    print()
    fact = LoopyExecutor().differential(reconstruct, cpu, example_inputs())
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )
    exact = math.isclose(worst, 0.0, abs_tol=1e-9)
    return 0 if exact and fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
