"""Compose two loopty kernels, then let loopy erase the intermediate.

The native application is intentionally ordinary::

    flux(u, f)
    divergence(f, rhs)

`@program` can already express that composition to Python and to lanky's
ledger, but loopty does not yet lower a Program.  This demo asks the next
question directly at the loopy layer:

1. lower both typed kernels independently;
2. fuse their domains and instruction streams with `loopy.fuse_kernels`;
3. state the producer -> consumer edge for `f`;
4. turn the producer assignment into a substitution rule with
   `loopy.assignment_to_subst`.

The interesting wart is also the design result.  Independently lowered kernels
disagree about the shared argument's interface: the producer marks `f` as an
output and the consumer marks it as input-only.  Loopy's fuser quite reasonably
requires matching declarations.  `_link_as_inout` is a deliberately local
adapter that makes them agree.  A first-class loopty Program lowering should own
that job and, better, make program-local producer/consumer arrays temporaries
rather than public arguments.

This is an example first and infrastructure second: if this pattern stays
useful, the primitive to add is composition with typed data-flow edges, not a
special Burgers-equation helper.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, kernel, program, when

N = 32


@kernel
def flux(
    u: Arr[Fin[n], Real],  # noqa: F821
    f: Arr[Fin[n], Real],  # noqa: F821
):
    """Pointwise Burgers flux, kept separate so composition has a producer."""
    for j in u.dom:
        f[j] = 0.5 * u[j] * u[j]


@kernel
def divergence(
    f: Arr[Fin[n], Real],  # noqa: F821
    rhs: Arr[Fin[n], Real],  # noqa: F821
):
    """Centered divergence of the flux."""
    for i in rhs.dom:
        with when((i > 0) & (i + 1 < rhs.dom.size)):
            rhs[i] = -(f[i + 1] - f[i - 1]) / 2


@program
def burgers_rhs(u, f, rhs):
    """The application-level composition, executable natively today."""
    flux(u, f)
    divergence(f, rhs)


def sample(n: int = N) -> dict:
    """Smooth nontrivial input and zeroed producer/output buffers."""
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    u = np.sin(x) + 0.2 * np.sin(3.0 * x)
    return {
        "u": Arr.from_numpy(u),
        "f": Arr.zeros(n),
        "rhs": Arr.zeros(n),
    }


def reference(u: np.ndarray) -> np.ndarray:
    """The fused mathematical expression, without materializing flux in a loop."""
    f = 0.5 * u * u
    rhs = np.zeros_like(u)
    rhs[1:-1] = -(f[2:] - f[:-2]) / 2
    return rhs


def example_inputs() -> dict:
    """Inputs for the two kernels when `loopty run` checks them individually."""
    data = sample()
    u = data["u"].numpy().copy()
    f_values = 0.5 * u * u
    return {
        "flux": {
            "u": Arr.from_numpy(u.copy()),
            "f": Arr.zeros(len(u)),
        },
        "divergence": {
            "f": Arr.from_numpy(f_values),
            "rhs": Arr.zeros(len(u)),
        },
    }


def _lowered_pair():
    from loopty.lower import lower_generic

    return (
        lower_generic(flux.term, target="c").kernel,
        lower_generic(divergence.term, target="c").kernel,
    )


def direct_fusion_error() -> str | None:
    """Why independently lowered interfaces are not quite compositional yet."""
    import loopy as lp
    from loopy.diagnostic import LoopyError

    producer, consumer = _lowered_pair()
    try:
        lp.fuse_kernels(
            [producer, consumer],
            data_flow=[("f", 0, 1)],
        )
    except LoopyError as exc:
        return str(exc)
    return None


def _link_as_inout(tunit, name: str):
    """Make one shared argument declaration agree across independently lowered TUs.

    This is intentionally *not* a public loopty primitive.  The application
    example is showing why Program lowering needs to understand producer /
    consumer edges and internal storage.
    """
    entry = tunit.default_entrypoint
    args = [
        (
            arg.copy(is_input=True, is_output=True)
            if getattr(arg, "name", None) == name
            else arg
        )
        for arg in entry.args
    ]
    return tunit.with_kernel(entry.copy(args=args))


def fused_rhs(*, inline_intermediate: bool = True):
    """Return a loopy TranslationUnit for the composed application."""
    import loopy as lp

    producer, consumer = _lowered_pair()
    producer = _link_as_inout(producer, "f")
    consumer = _link_as_inout(consumer, "f")
    fused = lp.fuse_kernels(
        [producer, consumer],
        data_flow=[("f", 0, 1)],
    )
    if inline_intermediate:
        fused = lp.assignment_to_subst(fused, "f")
    return fused


def execute_fused(u: np.ndarray) -> dict[str, np.ndarray]:
    """Compile and execute the inlined fusion, returning its named outputs."""
    tunit = fused_rhs(inline_intermediate=True)
    entry = tunit.default_entrypoint
    buffers = {
        "u": np.asarray(u).copy(),
        "f": np.zeros_like(u),
        "rhs": np.zeros_like(u),
    }
    call = {}
    for arg in entry.args:
        name = arg.name
        if name == "n":
            call[name] = np.int32(len(u))
        elif name in buffers:
            call[name] = buffers[name]
        else:
            raise AssertionError(f"unexpected fused argument {name!r}")

    _event, results = tunit.executor()(**call)
    if not isinstance(results, tuple):
        results = (results,)
    output_names = [
        arg.name for arg in entry.args if bool(getattr(arg, "is_output", False))
    ]
    return dict(zip(output_names, results, strict=True))


def main() -> int:
    """Run the program natively and the fused/inlined loopy form."""
    data = sample()
    u = data["u"].numpy().copy()
    burgers_rhs(data["u"], data["f"], data["rhs"])
    expected = reference(u)
    native_ok = np.allclose(data["rhs"].numpy(), expected)

    print(f"native @program agrees with direct expression: {native_ok}")
    error = direct_fusion_error()
    if error is None:
        print("direct fusion: current loopy accepts the two lowered interfaces")
    else:
        print("direct fusion before linking the shared interface:")
        print(f"  {error}")

    fused = fused_rhs(inline_intermediate=False)
    fused_entry = fused.default_entrypoint
    print()
    print(
        "after fuse: "
        f"{len(fused_entry.instructions)} instructions, "
        f"args={[arg.name for arg in fused_entry.args]}"
    )

    inlined = fused_rhs(inline_intermediate=True)
    inlined_entry = inlined.default_entrypoint
    print(
        "after assignment_to_subst('f'): "
        f"{len(inlined_entry.instructions)} instructions, "
        f"args={[arg.name for arg in inlined_entry.args]}, "
        f"substitutions={list(inlined_entry.substitutions)}"
    )

    result = execute_fused(u)
    got = result["rhs"]
    fused_ok = np.allclose(got, expected)
    print(f"fused/inlined generated C agrees: {fused_ok}")
    print(
        "application pressure: Program.lower/compose should infer the f edge, "
        "internalize f, fuse, and optionally substitute it."
    )
    return 0 if native_ok and fused_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
