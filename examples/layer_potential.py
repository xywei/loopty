"""pytential's QBX on its own curve: sumpy's tile checked, its series proved.

pytential evaluates a layer potential at the points of its own curve by QBX: for
each target ``z`` on the curve it takes a center ``c`` just off it, expands the
kernel about ``c`` and sums the expansion at ``z``. With ``fmm_order=False`` and
no derivative at the target, that is sumpy's ``LayerPotential``, the direct sum
over every source, with a line-Taylor expansion: the kernel along the line from
the center through the target, ``log |alpha + tau * beta|`` with ``alpha = c -
y`` and ``beta = z - c``, expanded in ``tau`` to the QBX order and summed at
``tau = 1``. Read ``alpha`` and ``beta`` as complex numbers and that is::

    log|alpha| + Re( c(1) + ... + c(5) ),   c(k) = -(-beta / alpha)**k / k

One file holds the three layers of it:

- **The algorithm.** ``line_taylor`` is the theorem that the ``c(k)`` are the
  Taylor coefficients of ``log(alpha + tau * beta)`` for every order, lanky's
  ``examples/line_taylor.py``: a table that satisfies what the differential
  equation ``(alpha + tau * beta) f'(tau) == beta`` says about its
  coefficients is the closed form ``coefficient`` writes, at the ratio
  ``-beta / alpha``. The kernels add ``coefficient(ratio, k)``, the same
  function, with the ratio computed as ``-beta * conj(alpha) / |alpha|**2``.
- **The kernels and their transforms.** ``single_layer`` and ``double_layer``
  are pytential's ``S`` and ``D`` as sumpy computes them, over complex points.
  ``sumpy_tile`` is sumpy's CPU schedule, as ``get_optimized_kernel`` writes
  it, and ``tiled`` the same tile on the C target. sumpy silences loopy's
  ``write_race(write_lpot*)`` warning on this kernel; here the ledger has the
  fact that stands in its place, ``distinct instances of S0 write distinct
  cells of out``, decided by isl and, with Lean, proved.
- **The accuracy.** pytential's own identity test on its own curve: Green's
  identity ``S(du/dn) - D(u) - u/2 == 0`` on the five-armed starfish, and the
  jump relations its integral equations rest on, each a ``tested`` row,
  measured with the compiled kernels at 30 to 90 panels.

Run this file three ways.

``python examples/layer_potential.py``
    Evaluates both layer potentials natively and compiled on the starfish,
    prints the facts of sumpy's tile and its run on the C target, and measures
    Green's identity and the two jumps as the panels shrink.

``lanky check examples/layer_potential.py``
    The ledger. The theorem reads ``proved`` by Lean where Mathlib is
    installed, and ``assumed`` without it, since the property tester draws no
    table over ``Nat``. Each kernel's accesses are in bounds and its writes
    distinct, decided by isl (proved by Lean where it is installed), its trace
    faithful to its body, and the three accuracy rows ``tested``.

``loopty run examples/layer_potential.py``
    The tiled kernel compiled on the C target and compared with the native
    run; sumpy's own tile names the device axes C cannot build, and runs with
    ``--target opencl``.

What it mirrors, at the commits the demonstration pins
------------------------------------------------------

- sumpy's ``LayerPotential.get_kernel``, whose result is ``knl_scaling *
  simul_reduce(sum, isrc, pair_result)`` with the warning silenced
  (https://github.com/inducer/sumpy/blob/13a48a0260eec8cf7ac67f1faca701205d9096ae/sumpy/qbx.py#L248-L306),
  and its CPU schedule, ``get_optimized_kernel(is_cpu=True)``
  (https://github.com/inducer/sumpy/blob/13a48a0260eec8cf7ac67f1faca701205d9096ae/sumpy/qbx.py#L208-L240).
- The expansion, ``LineTaylorLocalExpansion``
  (https://github.com/inducer/sumpy/blob/13a48a0260eec8cf7ac67f1faca701205d9096ae/sumpy/expansion/local.py#L116-L185).
- pytential's direct self-interaction, which calls it with the stage-1 nodes
  as targets, the upsampled nodes as sources and the centers on the side the
  operator asks for
  (https://github.com/inducer/pytential/blob/8700a2918085572ae86ace15a0beb2f106458bb2/pytential/qbx/__init__.py#L800-L829
  and #L866-L973), its expansion radii, half the panel size on a curve
  (https://github.com/inducer/pytential/blob/8700a2918085572ae86ace15a0beb2f106458bb2/pytential/symbolic/primitives.py#L1301-L1396),
  and its identity test, ``GreenExpr`` on the starfish at QBX order 5, target
  order 8 and sources upsampled by 4, which asks for an order above ``5 -
  1.6`` (https://github.com/inducer/pytential/blob/8700a2918085572ae86ace15a0beb2f106458bb2/test/test_layer_pot_identity.py#L87-L99
  and #L277-L411).
- meshmode's ``NArmedStarfish(5, 0.25)``
  (https://github.com/inducer/meshmode/blob/bbaf4ab4ecfb3b1cfcdac79165d4ca61fe0de64a/meshmode/mesh/generation.py#L310-L374).

What differs: sumpy computes the expansion in real arithmetic, straight-line
code sympy writes for each order; here it is the same series in complex
arithmetic, one loop over the order that the compiler writes out. The curve
here is the parametrization itself at Gauss-Legendre points, where meshmode
interpolates it by polynomials and pytential refines its panels first. The
agreement with sumpy's own kernel, on pytential's own geometry, is measured
by ``benchmarks/qbx_layer_potential.py``.
"""

from __future__ import annotations

import numpy as np
from lanky import theorem
from lanky.prelude import Complex, Fn, Nat, Real

from loopty import Arr, Fin, Schedule, kernel, reduce_sum, tested
from loopty.math import conj, log, real

#: pytential's QBX order in its identity test.
ORDER = 5

#: The Laplace kernel's constant: G(x, y) = -log|x - y| / (2 pi).
SCALE = -1 / (2 * np.pi)


# {{{ the algorithm: the line-Taylor coefficients, for every order


def coefficient(ratio, k):
    """The ``k``-th Taylor coefficient of ``log(alpha + tau * beta)``, ``k >= 1``.

    ``ratio`` is ``-beta / alpha``.
    """
    return -(ratio**k) / k


@theorem
def line_taylor(
    alpha: Complex,
    beta: Complex,
    c: Fn[Nat, Complex],
    away: alpha != 0,
    first: alpha * c(1) == beta,
    ode: all((n + 1) * alpha * c(n + 1) + n * beta * c(n) == 0 for n in Nat if n >= 1),
) -> all(c(n) == coefficient(-beta / alpha, n) for n in Nat if n >= 1):
    """The line-Taylor coefficients of ``log(alpha + tau * beta)``, for every order.

    ``f(tau) = log(alpha + tau * beta)`` solves ``(alpha + tau * beta) * f'(tau)
    == beta``. Comparing the powers of ``tau`` in it gives ``first`` and
    ``ode`` (L. V. Ahlfors, Complex Analysis, 3rd ed., 1979, chs. 2 and 5;
    lanky's ``examples/line_taylor.py`` states the equation as an axiom and
    checks sumpy's own coefficients against the closed form through order 6).
    The induction is two orders at a time; Python finds the linear combination
    of the hypotheses each step needs, and Lean checks it with Mathlib.
    """


def single(alpha, beta):
    """``log|alpha + beta|``: ``log|alpha|`` and the series through ``ORDER``.

    With ``r2 = |alpha|**2``, ``log|alpha|`` is ``log(r2) / 2``, and the
    ratio ``-beta / alpha`` is ``-beta * conj(alpha) / r2``, a complex
    division multiplied out by the conjugate, as sumpy's real arithmetic has
    it: one real division where a complex division takes three, and a branch.
    """
    r2 = real(alpha * conj(alpha))
    ratio = -beta * conj(alpha) * (1 / r2)
    series = reduce_sum(coefficient(ratio, k) for k in Fin[ORDER + 1] if k >= 1)
    return log(r2) / 2 + real(series)


def double(alpha, beta, normal):
    """:func:`single` differentiated along ``normal`` at the source, by the same series.

    The source is ``c - alpha``, so this is ``Re(normal * d/dy)`` of the
    series in ``y``: ``-Re(normal / alpha * (1 + ratio + ... +
    ratio**ORDER))``, the line-Taylor series of ``Re(normal / (y - x))``
    itself, divided out by the conjugate as :func:`single` does.
    """
    r2 = real(alpha * conj(alpha))
    ratio = -beta * conj(alpha) * (1 / r2)
    powers = reduce_sum(ratio**k for k in Fin[ORDER + 1])
    return -real(normal * conj(alpha) * (1 / r2) * powers)


# }}}


# {{{ the kernels: pytential's S and D, as sumpy's LayerPotential computes them


@kernel
def single_layer(
    z: Arr[Fin[nt], np.complex128],
    c: Arr[Fin[nt], np.complex128],
    y: Arr[Fin[ns], np.complex128],
    w: Arr[Fin[ns], Real],
    out: Arr[Fin[nt], Real],
):
    """``S`` at each target ``z[t]``, through the expansion about its center ``c[t]``.

    ``y`` holds the sources and ``w`` the quadrature weight times the density
    at each, as pytential hands sumpy its strengths.
    """
    for t in out.dom:
        out[t] = SCALE * reduce_sum(
            w[s] * single(c[t] - y[s], z[t] - c[t]) for s in y.dom
        )


@kernel
def double_layer(
    z: Arr[Fin[nt], np.complex128],
    c: Arr[Fin[nt], np.complex128],
    y: Arr[Fin[ns], np.complex128],
    n: Arr[Fin[ns], np.complex128],
    w: Arr[Fin[ns], Real],
    out: Arr[Fin[nt], Real],
):
    """``D`` at each target: ``S``'s kernel differentiated along ``n[s]``."""
    for t in out.dom:
        out[t] = SCALE * reduce_sum(
            w[s] * double(c[t] - y[s], z[t] - c[t], n[s]) for s in y.dom
        )


# }}}


# {{{ the curve: meshmode's five-armed starfish, discretized as pytential does


def _panels(panels: int, nodes: int) -> tuple:
    """``nodes`` Gauss-Legendre points a panel: points, normals, weights, panel sizes.

    The curve is ``NArmedStarfish(5, 0.25)``, ``r = 1 + sin(5 theta) / 4``,
    with the panels equal in its parameter. The weights are the rule's times
    the speed, the curve's length element, and the size of a panel is its
    parameter length times the largest speed on it.
    """
    x, w = np.polynomial.legendre.leggauss(nodes)
    h = 1 / panels
    t = (np.arange(panels)[:, None] * h + h * (x + 1) / 2).ravel()
    theta = 2 * np.pi * t
    radius = 1 + 0.25 * np.sin(5 * theta)
    point = radius * np.exp(1j * theta)
    tangent = (2 * np.pi * 1.25 * np.cos(5 * theta) + 2j * np.pi * radius) * np.exp(
        1j * theta
    )
    speed = np.abs(tangent)
    size = np.repeat(h * speed.reshape(panels, nodes).max(axis=1), nodes)
    return point, -1j * tangent / speed, np.tile(w * h / 2, panels) * speed, size


def starfish(panels: int, source_panels: int | None = None) -> dict:
    """pytential's discretization of the starfish at ``panels`` panels.

    The targets are the stage-1 nodes, 9 a panel (target order 8); the
    sources are the upsampled nodes, 17 a panel (order 4 * 8), on the same
    panels unless ``source_panels`` says how many; each target's centers are
    half its panel's size inside and outside along the normal.
    """
    z, normal, _, size = _panels(panels, 9)
    y, ny, wy, _ = _panels(source_panels or panels, 17)
    return {
        "z": z,
        "normal": normal,
        "inside": z - size / 2 * normal,
        "outside": z + size / 2 * normal,
        "size": size,
        "y": y,
        "ny": ny,
        "wy": wy,
    }


def arguments(curve: dict, side: str, density: np.ndarray) -> dict:
    """A kernel's arguments: the curve's targets, centers on ``side``, and a density."""
    nt = curve["z"].size
    return {
        "z": Arr.from_numpy(curve["z"]),
        "c": Arr.from_numpy(curve[side]),
        "y": Arr.from_numpy(curve["y"]),
        "n": Arr.from_numpy(curve["ny"]),
        "w": Arr.from_numpy(curve["wy"] * density),
        "out": Arr.zeros(nt),
    }


#: A harmonic function inside the starfish, pytential's: ``log|x - x0|``,
#: ``x0 = (3, 1)``.
SOURCE = 3 + 1j


def harmonic(x: np.ndarray) -> np.ndarray:
    """``u(x) = log|x - x0|``."""
    return np.log(np.abs(x - SOURCE))


def normal_derivative(x: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """``du/dn = n . (x - x0) / |x - x0|**2``."""
    d = x - SOURCE
    return (d.real * normal.real + d.imag * normal.imag) / np.abs(d) ** 2


def density(x: np.ndarray) -> np.ndarray:
    """A smooth density for the jumps: ``cos(3 theta)``."""
    return np.cos(3 * np.angle(x))


def example_inputs() -> dict:
    """Four panels, small enough to run natively: 36 targets, 68 sources."""
    curve = starfish(4)
    sigma = density(curve["y"])
    single_args = arguments(curve, "inside", sigma)
    del single_args["n"]
    return {
        "single_layer": single_args,
        "double_layer": arguments(curve, "inside", sigma),
    }


# }}}


# {{{ the schedules


def sumpy_tile(target: str = "opencl") -> Schedule:
    """sumpy's CPU schedule, as ``get_optimized_kernel(is_cpu=True)`` writes it.

    16 targets to a work group, a work item each, the sources 256 at a time,
    and the sources' loop outside the targets'. It is a schedule for a device,
    PoCL's CPU device being the one ``is_cpu`` is written for; on the C
    target its casts are decided as well, and it does not build, which is
    why it is a function: a file's schedules are run by ``loopty run`` and
    listed by ``lanky check``, and this one runs where pyopencl does
    (``benchmarks/qbx_layer_potential.py``).
    """
    return (
        Schedule(single_layer, target=target)
        .split("t", 16)
        .tag(t_outer="g.0", t_inner="l.0")
        .split("s", 256)
        .prioritize("s_outer", "t_inner")
    )


#: The same tile on the C target, the 16 targets an ``ilp`` loop, which gives
#: each an accumulator of its own.
tiled = (
    Schedule(single_layer, target="c")
    .split("t", 16)
    .tag(t_inner="ilp")
    .split("s", 256)
    .prioritize("s_outer", "t_inner")
    .example(**example_inputs()["single_layer"])
)

#: The double layer on the same tile, which the accuracy rows run as well.
tiled_double = (
    Schedule(double_layer, target="c")
    .split("t", 16)
    .tag(t_inner="ilp")
    .split("s", 256)
    .prioritize("s_outer", "t_inner")
    .example(**example_inputs()["double_layer"])
)


# }}}


# {{{ the accuracy: pytential's identity and the jump relations, measured

#: The panels the accuracy rows are measured at, pytential's starfish resolutions.
PANELS = (30, 50, 70, 90)

#: The order the errors have to fall at: pytential's test asks for ``ORDER - 1.6``.
EXPECTED = ORDER - 1.6


def _layer(
    schedule: Schedule, curve: dict, side: str, values: np.ndarray
) -> np.ndarray:
    """One layer potential of ``values`` at the sources, compiled and run."""
    from loopty.executor import LoopyExecutor

    args = arguments(curve, side, values)
    if schedule is tiled:
        del args["n"]
    return LoopyExecutor().run(schedule, **args)["out"]


def residuals(panels: int) -> dict[str, float]:
    """The largest error of Green's identity and of each jump at ``panels`` panels."""
    curve = starfish(panels)
    z, y = curve["z"], curve["y"]
    dn_u = normal_derivative(y, curve["ny"])
    u, sigma = harmonic(y), density(y)
    green = (
        _layer(tiled, curve, "inside", dn_u)
        - (
            _layer(tiled_double, curve, "inside", u)
            + _layer(tiled_double, curve, "outside", u)
        )
        / 2
        - harmonic(z) / 2
    )
    jump_s = _layer(tiled, curve, "outside", sigma) - _layer(
        tiled, curve, "inside", sigma
    )
    jump_d = (
        _layer(tiled_double, curve, "outside", sigma)
        - _layer(tiled_double, curve, "inside", sigma)
        - density(z)
    )
    return {
        "green": float(np.abs(green).max()),
        "jump_S": float(np.abs(jump_s).max()),
        "jump_D": float(np.abs(jump_d).max()),
    }


_MEASURED: dict[str, list[float]] = {}


def measured() -> dict[str, list[float]]:
    """Each row's errors at :data:`PANELS`, measured once."""
    if not _MEASURED:
        rows = [residuals(panels) for panels in PANELS]
        for name in rows[0]:
            _MEASURED[name] = [row[name] for row in rows]
    return _MEASURED


def order(errors: list[float]) -> float:
    """The order the errors fall at as the panels shrink: a least-squares slope."""
    sizes = [starfish(panels)["size"].max() for panels in PANELS]
    return float(np.polyfit(np.log(sizes), np.log(errors), 1)[0])


def _verdict(name: str) -> tuple[bool, str]:
    errors = measured()[name]
    rate = order(errors)
    shown = ", ".join(f"{error:.1e}" for error in errors)
    return rate > EXPECTED, f"errors {shown}: order {rate:.2f}"


@tested
def green_identity():
    """S(du/dn) - D(u) = u/2, Green's identity, at order 5 - 1.6 or faster."""
    return _verdict("green")


@tested
def jump_S():
    """S(+) - S(-) = 0: S is continuous, at order 5 - 1.6 or faster."""
    return _verdict("jump_S")


@tested
def jump_D():
    """D(+) - D(-) = sigma: D jumps by the density, at order 5 - 1.6 or faster."""
    return _verdict("jump_D")


# }}}


def main() -> int:
    """Run both layers, show sumpy's tile, and measure the identity and the jumps."""
    from loopty.executor import LoopyExecutor
    from loopty.oracle import IslOracle

    curve = starfish(30)
    print(
        f"the starfish at 30 panels: {curve['z'].size} targets, "
        f"{curve['y'].size} sources, QBX order {ORDER}"
    )
    ok = True
    for kern, schedule in ((single_layer, tiled), (double_layer, tiled_double)):
        fact = LoopyExecutor().differential(kern, schedule, schedule.examples)
        detail = fact.provenance["outputs"]["out"]
        print(
            f"  {kern.__name__}: compiled agrees with native to the "
            f"{detail['exactness']} tolerance: {detail['agree']} -> {fact.status.value}"
        )
        ok = ok and fact.status.value == "tested"

    print()
    tile = sumpy_tile("c")
    print(f"sumpy's tile: {tile!r}")
    writes = [fact for fact in single_layer.facts() if fact.kind == "disjoint-writes"]
    oracle = IslOracle()
    for fact in [*tile.facts(), *writes]:
        if fact.kind == "buildable":
            continue
        if fact.status.value == "assumed":
            fact = oracle.establish(fact) or fact
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")
    print(f"  on C: {tiled!r}")
    print()
    print(f"accuracy at {', '.join(map(str, PANELS))} panels:")
    for claim in (green_identity, jump_S, jump_D):
        held, what = claim()
        print(f"  {claim.__name__:15} {what} -> {'tested' if held else 'REFUTED'}")
        ok = ok and held
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
