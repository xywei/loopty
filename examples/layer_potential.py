"""pytential's QBX on its own curve: sumpy's tile checked, its series proved.

pytential evaluates a layer potential at the points of its own curve by QBX: for
each target ``z`` on the curve it takes a center ``c`` just off it, expands the
kernel about ``c`` and sums the expansion at ``z``. With ``fmm_order=False`` and
no derivative at the target, that is sumpy's ``LayerPotential``, the direct sum
over every source, with a line-Taylor expansion: the kernel along the line from
the center through the target, ``log |alpha + tau * beta|`` with ``alpha = c -
y`` and ``beta = z - c``, expanded in ``tau`` to the QBX order and summed at
``tau = 1``. Read ``alpha`` and ``beta`` as complex numbers and that is::

    log|alpha| + Re( a(1) + ... + a(5) ),   a(k) = -(-beta / alpha)**k / k

One file holds the three layers of it:

- **The algorithm.** ``line_taylor`` is the theorem behind the coefficients,
  for every order, lanky's ``examples/line_taylor.py``: a table ``a`` that
  satisfies what the differential equation ``(alpha + tau * beta) f'(tau) ==
  beta`` says about its coefficients is the closed form ``coefficient``
  writes, at the ratio ``-beta / alpha``. That the Taylor coefficients of
  ``log(alpha + tau * beta)`` satisfy it is cited, not proved. The kernels
  say the math as written, ``log|alpha|`` and ``coefficient(-beta / alpha,
  k)``, so ``single_layer`` cites the theorem directly, and its ledger has
  the row that links them: ``single_layer adds line_taylor's a(n) for n in
  1..5``, about the term it evaluates read over the complex numbers, not
  about the floats it produces.
- **The kernels and their transforms.** ``single_layer`` and ``double_layer``
  are pytential's ``S`` and ``D`` over complex points (lanky's ``Complex``,
  stored as ``complex128``). ``as_sumpy_computes`` is the single layer in
  sumpy's arithmetic: each division by ``alpha`` through its conjugate, one
  real division where a complex one takes three (``conjugate_form``), and
  ``log|alpha|`` with no square root (``log_of_the_modulus``), two math steps
  of the schedule, each a theorem. Lean proves each step exact over the
  complex numbers wherever ``c[t] - y[s] != 0``, which is tested on the
  example inputs; the compiled run is compared with the native one, and at
  inputs of extreme magnitude the row notes where ``alpha * conj(alpha)``
  underflows or overflows and ``log|alpha|`` does not, as sumpy's arithmetic
  does. ``sumpy_tile()`` is sumpy's CPU schedule on that arithmetic, as
  ``get_optimized_kernel`` writes it, and ``tiled`` the same tile on the C
  target. sumpy silences loopy's ``write_race(write_lpot*)`` warning on this
  kernel; here the ledger has the fact that stands in its place, ``distinct
  instances of S0 write distinct cells of out``, decided by isl and, with
  Lean, proved.
- **The agreement and the accuracy.** ``sumpy_single_layer`` and
  ``sumpy_double_layer`` are what sumpy's own ``LayerPotential`` returns on
  the same inputs, called as pytential calls it, and each kernel's
  ``upstream`` row compares its native run with them, within ``1e-12`` of
  the largest value, which is round-off between two orders of summation.
  pytential's identity test case, Green's identity ``S(du/dn) - D(u) - u/2
  == 0`` on the five-armed starfish, and the jump relations its integral
  equations rest on are measured with the compiled kernels at 30 to 90
  panels, each a ``tested`` row decided by the run.

Run this file three ways.

``python examples/layer_potential.py``
    Evaluates both layer potentials natively and compiled on the starfish,
    prints the rows of sumpy's arithmetic, the facts of sumpy's tile and its
    run on the C target, and measures Green's identity and the two jumps as
    the panels shrink.

``lanky check examples/layer_potential.py``
    The ledger. ``line_taylor`` reads ``proved`` by Lean where Mathlib is
    installed, and ``assumed`` without it, since the property tester draws no
    table over ``Nat``; ``conjugate_form`` reads ``proved`` with Mathlib and
    ``tested`` without, and ``log_of_the_modulus`` reads ``tested``, which
    lanky's search does not prove alone. The row that ``single_layer`` adds
    the theorem's ``a(n)`` is ``decided`` by ``theorem``, and proved by Lean
    with Mathlib. Each math step of ``as_sumpy_computes`` is a row, proved
    by Lean with Mathlib and tested on drawn inputs without it; the
    condition ``c[t] - y[s] != 0`` is tested on the example inputs; and the
    compiled schedule's agreement with the native kernel is a row. Each
    kernel's accesses are in bounds and its writes distinct, and each cast
    of its tile on C keeps every dependence, decided by isl (proved by Lean
    where it is installed); its trace is faithful to its body, its
    agreement with sumpy is ``tested`` by ``upstream`` within ``1e-12`` of
    the largest value, and the three accuracy rows are ``tested`` by their
    run.

``loopty run examples/layer_potential.py``
    Both kernels in sumpy's arithmetic and on their tile, compiled on the C
    target and compared with the native run. ``sumpy_tile()``, with its work
    items, runs on a device in ``benchmarks/qbx_layer_potential.py``.

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
  1.6`` (https://github.com/inducer/pytential/blob/8700a2918085572ae86ace15a0beb2f106458bb2/test/test_layer_pot_identity.py#L87-L99,
  #L258-L268 and #L277-L411). pytential runs that case with its FMM; here it
  is the direct evaluation, ``fmm_order=False``, as pytential's test runs the
  starfish's Calderon case.
- meshmode's ``NArmedStarfish(5, 0.25)``
  (https://github.com/inducer/meshmode/blob/bbaf4ab4ecfb3b1cfcdac79165d4ca61fe0de64a/meshmode/mesh/generation.py#L310-L374).

What differs: sumpy computes the expansion in real arithmetic, straight-line
code sympy writes for each order; here it is the same series in complex
arithmetic, one sum over the order that the trace writes out, with each
division multiplied out by the conjugate by a math step. The curve here is the
parametrization itself at Gauss-Legendre points, where meshmode interpolates
it by polynomials and pytential refines its panels first, so the accuracy
rows measure this discretization and not pytential's. What sumpy's kernel
returns on this file's own inputs is recorded in
``layer_potential_sumpy.json`` with the commits it ran at, since sumpy needs a
newer loopy than loopty's; ``benchmarks/qbx_layer_potential.py`` records it,
and compares the kernels with sumpy on the arguments pytential itself hands
it, on its refined starfish.

What loopy does with sumpy's tile: ``prioritize_loops([isrc_outer,
itgt_inner])`` cannot be kept with the targets on a hardware axis, which
loopy says, and it then nests the inner source loop outside the outer one,
so that each work item adds its sources 256 apart. The schedule here states
the order its facts were checked for, and its code keeps it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lanky import theorem
from lanky.prelude import Complex, Fn, Nat, Real

from loopty import Arr, Fin, Schedule, kernel, reduce_sum, tested, upstream
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
    a: Fn[Nat, Complex],
    nonzero: alpha != 0,
    first_term: alpha * a[1] == beta,
    ode: all((n + 1) * alpha * a[n + 1] + n * beta * a[n] == 0 for n in Nat if n >= 1),
) -> all(a[n] == coefficient(-beta / alpha, n) for n in Nat if n >= 1):
    """The line-Taylor coefficients of ``log(alpha + tau * beta)``, at every order.

    ``f(tau) = log(alpha + tau * beta)`` solves ``(alpha + tau * beta) * f'(tau)
    == beta``; comparing the powers of ``tau`` in it gives ``first_term`` and
    ``ode`` (L. V. Ahlfors, Complex Analysis, 3rd ed., 1979, chs. 2 and 5;
    lanky's ``examples/line_taylor.py`` states the equation as an axiom and
    checks sumpy's own coefficients against the closed form through order 6).
    The induction is two orders at a time; Python finds the linear combination
    of the hypotheses each step needs, and Lean checks it with Mathlib.
    """


@theorem
def conjugate_form(x: Complex, alpha: Complex, nonzero: alpha != 0) -> (
    x / alpha == x * conj(alpha) * (1 / real(alpha * conj(alpha)))
):
    """Division by alpha through its conjugate.

    alpha times its conjugate is ``|alpha|**2``, a real number, so dividing by
    alpha is multiplying by ``conj(alpha)`` and by the reciprocal of
    ``|alpha|**2``: one real division where a complex one takes three.
    """


@theorem
def log_of_the_modulus(alpha: Complex, nonzero: alpha != 0) -> (
    log(abs(alpha)) == log(real(alpha * conj(alpha))) / 2
):
    """``log|alpha|`` with no square root: half the log of alpha times its conjugate.

    ``log|alpha|`` is half the log of ``|alpha|**2`` (the log of a power), and
    ``|alpha|**2`` is alpha times its conjugate (the modulus squared).
    """


def single(alpha, beta):
    """``log|alpha + beta|``: ``log|alpha|``, and the series in ``-beta / alpha``."""
    return log(abs(alpha)) + real(
        reduce_sum(coefficient(-beta / alpha, k) for k in Fin[ORDER + 1] if k >= 1)
    )


def double(alpha, beta, normal):
    """:func:`single` differentiated along ``normal`` at the source, by the same series.

    The source is ``c - alpha``, so this is ``Re(normal * d/dy)`` of the
    series in ``y``: ``-Re(normal / alpha * (1 + ratio + ... +
    ratio**ORDER))`` with ``ratio = -beta / alpha``, the line-Taylor series of
    ``Re(normal / (y - x))`` itself.
    """
    return -real(
        normal / alpha * reduce_sum((-beta / alpha) ** k for k in Fin[ORDER + 1])
    )


# }}}


# {{{ the kernels: pytential's S and D, as the math says them


@kernel(uses=[line_taylor])
def single_layer(
    z: Arr[Fin[nt], Complex],
    c: Arr[Fin[nt], Complex],
    y: Arr[Fin[ns], Complex],
    w: Arr[Fin[ns], Real],
    out: Arr[Fin[nt], Real],
):
    """``S`` at each target ``z[t]``, through the expansion about its center ``c[t]``.

    ``y`` holds the sources and ``w`` the quadrature weight times the density
    at each, as pytential hands sumpy its strengths. It cites ``line_taylor``
    for the terms of its series.
    """
    for t in out.dom:
        out[t] = SCALE * reduce_sum(
            w[s] * single(c[t] - y[s], z[t] - c[t]) for s in y.dom
        )


@kernel
def double_layer(
    z: Arr[Fin[nt], Complex],
    c: Arr[Fin[nt], Complex],
    y: Arr[Fin[ns], Complex],
    n: Arr[Fin[ns], Complex],
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

    The targets are the stage-1 nodes, 9 a panel (meshmode's group of order
    8); the sources are the upsampled nodes, 33 a panel (its quadrature group
    of order 4 * 8, Gauss-Legendre nodes that integrate degree 65), on the
    same panels unless ``source_panels`` says how many; each target's centers
    are half its panel's size inside and outside along the normal.
    """
    z, normal, _, size = _panels(panels, 9)
    y, ny, wy, _ = _panels(source_panels or panels, 33)
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


def arguments(
    curve: dict, side: str, density: np.ndarray, normals: bool = False
) -> dict:
    """A kernel's arguments: the curve's targets, centers on ``side``, a density.

    With ``normals``, the sources' normals too, which ``double_layer`` takes.
    """
    args = {
        "z": Arr.from_numpy(curve["z"]),
        "c": Arr.from_numpy(curve[side]),
        "y": Arr.from_numpy(curve["y"]),
        "w": Arr.from_numpy(curve["wy"] * density),
        "out": Arr.zeros(curve["z"].size),
    }
    if normals:
        args["n"] = Arr.from_numpy(curve["ny"])
    return args


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
    """Four panels, small enough to run natively: 36 targets, 132 sources."""
    curve = starfish(4)
    sigma = density(curve["y"])
    return {
        "single_layer": arguments(curve, "inside", sigma),
        "double_layer": arguments(curve, "inside", sigma, normals=True),
    }


# }}}


# {{{ the schedules

#: As sumpy computes it: divisions through the conjugate, log|alpha| with no root.
as_sumpy_computes = (
    Schedule(single_layer, target="c")
    .rewrite(single_layer.out, by=(conjugate_form, log_of_the_modulus))
    .example(**example_inputs()["single_layer"])
)


def sumpy_tile(target: str = "opencl") -> Schedule:
    """sumpy's CPU schedule, as ``get_optimized_kernel(is_cpu=True)`` writes it.

    sumpy's arithmetic first (:data:`as_sumpy_computes`), then 16 targets to a
    work group, a work item each, the sources 256 at a time, and the sources'
    loop outside the targets'. It is a schedule for a device, PoCL's CPU
    device being the one ``is_cpu`` is written for; on the C target its casts
    are decided as well, and it does not build, which is why it is a
    function: a file's schedules are run by ``loopty run`` and listed by
    ``lanky check``, and this one runs where pyopencl does
    (``benchmarks/qbx_layer_potential.py``).
    """
    return (
        Schedule(single_layer, target=target)
        .rewrite(single_layer.out, by=(conjugate_form, log_of_the_modulus))
        .split(single_layer.t, 16)
        .tag({single_layer.t.outer: "g.0", single_layer.t.inner: "l.0"})
        .split(single_layer.s, 256)
        .prioritize(single_layer.s.outer, single_layer.t.inner)
    )


#: sumpy's CPU tile, on that arithmetic, on the C target: the 16 targets an
#: ``ilp`` loop, which gives each an accumulator of its own.
tiled = (
    as_sumpy_computes
    .split(single_layer.t, 16)
    .tag(single_layer.t.inner, "ilp")
    .split(single_layer.s, 256)
    .prioritize(single_layer.s.outer, single_layer.t.inner)
)

#: The double layer on the same tile, divided through the conjugate, which the
#: accuracy rows run as well.
tiled_double = (
    Schedule(double_layer, target="c")
    .rewrite(double_layer.out, by=conjugate_form)
    .split(double_layer.t, 16)
    .tag(double_layer.t.inner, "ilp")
    .split(double_layer.s, 256)
    .prioritize(double_layer.s.outer, double_layer.t.inner)
    .example(**example_inputs()["double_layer"])
)


# }}}


# {{{ the agreement with sumpy's own kernel

#: What sumpy's ``LayerPotential`` returns on :func:`example_inputs`, called as
#: pytential calls it (``benchmarks/qbx_layer_potential.py record``).
RECORDED = Path(__file__).with_name("layer_potential_sumpy.json")


def recorded(layer: str) -> list[float]:
    """What sumpy returned for ``single_layer`` or ``double_layer``."""
    return json.loads(RECORDED.read_text(encoding="utf-8"))[layer]


@upstream(single_layer, within=1e-12)
def sumpy_single_layer(z, c, y, w, out):
    """sumpy's own LayerPotential at 13a48a0 for S, as pytential calls it, recorded."""
    return {"out": recorded("single_layer")}


@upstream(double_layer, within=1e-12)
def sumpy_double_layer(z, c, y, n, w, out):
    """sumpy's own LayerPotential at 13a48a0 for D, as pytential calls it, recorded."""
    return {"out": recorded("double_layer")}


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

    args = arguments(curve, side, values, normals=schedule is tiled_double)
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
    from loopty.obligations import prove
    from loopty.oracle import IslOracle

    curve = starfish(4)
    print(
        f"pytential's starfish, QBX order {ORDER}, at 4 panels: "
        f"{curve['z'].size} targets, {curve['y'].size} sources"
    )
    ok = True
    for kern, schedule in ((single_layer, tiled), (double_layer, tiled_double)):
        fact = LoopyExecutor().differential(kern, schedule, schedule.examples)
        detail = fact.provenance["outputs"]["out"]
        print(
            f"  {kern.__name__}: compiled on C agrees with native to the "
            f"{detail['exactness']} tolerance: {detail['agree']} -> {fact.status.value}"
        )
        ok = ok and fact.status.value == "tested"

    for agreement in (sumpy_single_layer, sumpy_double_layer):
        # the verdict and the allowance the row states, not the difference,
        # whose last digits are the platform's
        (fact,) = agreement.facts()
        detail = fact.provenance.get("outputs", {}).get("out")
        measured = (
            f"within {detail['within']:g} of the largest value"
            if detail and "within" in detail
            else f"to the {detail['exactness']} tolerance"
            if detail
            else fact.provenance.get("reason", "")
        )
        print(
            f"  {agreement.kernel.__name__}: agrees with sumpy's own kernel, as "
            f"pytential calls it, {measured} -> {fact.status.value}"
        )
        ok = ok and fact.status.value == "tested"

    # the theorem and the kernel, in one row
    (series,) = [fact for fact in single_layer.facts() if fact.kind == "by-theorem"]
    print(f"  {series.statement} -> {series.status.value}")
    ok = ok and series.status.value in ("decided", "proved")

    print()
    print(f"as sumpy computes it: {'.'.join(as_sumpy_computes.history)}")
    math = set()
    for fact in as_sumpy_computes.facts():
        math.add(fact.id)
        # Lean, where Mathlib is, proves each step exact over the complex numbers
        fact = prove(fact) if fact.kind == "rewrite" else fact
        print(f"  {fact.status.value:8} {fact.decided_by or '-':11} {fact.statement}")
        ok = ok and fact.status.value != "refuted"

    print()
    tile = sumpy_tile("c")
    casts = [step for step in tile.history if not step.startswith("rewrite(")]
    print(f"sumpy's tile, get_optimized_kernel(is_cpu=True): {'.'.join(casts)}")
    writes = [fact for fact in single_layer.facts() if fact.kind == "disjoint-writes"]
    oracle = IslOracle()
    for fact in [*tile.facts(), *writes]:
        if fact.kind == "buildable" or fact.id in math:
            # the arithmetic's rows, said above
            continue
        # Lean, where it is installed, proves each cast, that no dependence
        # joins two work items included, and the writes; isl decides the rest
        fact = prove(fact)
        if fact.status.value == "assumed":
            fact = oracle.establish(fact) or fact
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")
    print(f"  on C, an ilp loop for the work items: {'.'.join(tiled.history)}")
    print()
    print(f"accuracy at {', '.join(map(str, PANELS))} panels:")
    for claim in (green_identity, jump_S, jump_D):
        held, what = claim()
        print(f"  {claim.__name__:15} {what} -> {'tested' if held else 'REFUTED'}")
        ok = ok and held
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
