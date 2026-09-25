"""Floating-point contraction, and why an ``exact`` output pins it off.

A fused multiply-add rounds ``a * b + c`` once. The native run of a kernel body
is Python and numpy arithmetic, which rounds after the multiplication and again
after the addition, so the two can differ in the last bit, and an ``exact``
output is compared bit for bit. The lowering therefore asks the compiler not to
contract whenever a kernel has an ``exact`` output: ``-ffp-contract=off`` on
the C target, and the ``FP_CONTRACT`` pragma in the source, which is how OpenCL
C spells it.

loopy compiles C with ``gcc -std=c99 -O3 -fPIC``. GCC does not contract in a
standard mode, and x86-64 has no FMA instruction without ``-march``, so a stock
build contracts nothing on such a machine. clang does contract by default, and
arm64 has FMA in its baseline, which is where the pin earns its keep. The last
test makes a compiler here behave that way and shows the difference, on
hardware that has FMA; elsewhere it skips.
"""

from __future__ import annotations

import numpy as np
import pytest
from lanky.prelude import Real

from loopty import Arr, Fin, kernel

lp = pytest.importorskip("loopy")


@kernel
def fused_exact(
    a: Arr[Fin[n], Real.exact],  # noqa: F821
    b: Arr[Fin[n], Real.exact],  # noqa: F821
    c: Arr[Fin[n], Real.exact],  # noqa: F821
    y: Arr[Fin[n], Real.exact],  # noqa: F821
):
    """``a * b + c``, with every array of the ``exact`` class."""
    for i in y.dom:
        y[i] = a[i] * b[i] + c[i]


@kernel
def fused_approx(
    a: Arr[Fin[n], Real],  # noqa: F821
    b: Arr[Fin[n], Real],  # noqa: F821
    c: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The same arithmetic over ``Real``, which is ``approx``."""
    for i in y.dom:
        y[i] = a[i] * b[i] + c[i]


def inputs() -> dict[str, np.ndarray]:
    """Arrays whose product is not a double.

    ``(1 + 2**-30) * (1 - 2**-30)`` is ``1 - 2**-60``, which rounds to 1.0, so
    rounding the product and then adding -1 gives 0.0, and the fused operation
    gives ``-2**-60``.
    """
    eps = 2.0**-30
    return {
        "a": np.full(4, 1.0 + eps),
        "b": np.full(4, 1.0 - eps),
        "c": np.full(4, -1.0),
        "y": np.zeros(4),
    }


def compiled(kernel_: object, build_options: list[str]) -> np.ndarray:
    """``y`` from the lowered kernel built with exactly these extra C flags."""
    from loopty.lower import lower_generic

    translation_unit = lp.set_options(
        lower_generic(kernel_.trace(), "c").kernel, build_options=build_options
    )
    arrays = inputs()
    try:
        _event, (y,) = translation_unit.executor()(
            a=arrays["a"], b=arrays["b"], c=arrays["c"], y=arrays["y"]
        )
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        pytest.skip(f"the C toolchain cannot build this here: {exc}")
    return y


def test_an_exact_output_pins_contraction_off() -> None:
    from loopty.lower import NO_CONTRACTION_FLAG, lower_generic

    lowering = lower_generic(fused_exact.trace(), "c")
    assert not lowering.contraction
    entry = lowering.kernel.default_entrypoint
    assert NO_CONTRACTION_FLAG in entry.options.build_options
    code = lp.generate_code_v2(lowering.kernel).device_code()
    assert "#pragma STDC FP_CONTRACT OFF" in code


def test_an_approx_output_leaves_contraction_to_the_compiler() -> None:
    from loopty.lower import NO_CONTRACTION_FLAG, lower_generic

    lowering = lower_generic(fused_approx.trace(), "c")
    assert lowering.contraction
    entry = lowering.kernel.default_entrypoint
    assert NO_CONTRACTION_FLAG not in (entry.options.build_options or ())
    assert "FP_CONTRACT" not in lp.generate_code_v2(lowering.kernel).device_code()


def test_the_opencl_source_carries_the_opencl_pragma(monkeypatch) -> None:
    # OpenCL C has no build option for contraction, so the source says it.
    # loopy's plain OpenCL target stands in for the pyopencl one, which cannot
    # be built without pyopencl and is never imported here.
    from loopty import lower

    plain = lower.target_for
    monkeypatch.setattr(
        lower,
        "target_for",
        lambda target="c": lp.OpenCLTarget() if target == "opencl" else plain(target),
    )
    lowering = lower.lower_generic(fused_exact.trace(), "opencl")
    code = lp.generate_code_v2(lowering.kernel).device_code()
    assert "#pragma OPENCL FP_CONTRACT OFF" in code
    assert not lowering.kernel.default_entrypoint.options.build_options


def test_an_exact_kernel_agrees_with_its_native_run_bit_for_bit() -> None:
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule

    arrays = inputs()
    native = {name: value.copy() for name, value in arrays.items()}
    fused_exact(**native)
    assert np.array_equal(native["y"], np.zeros(4))
    fact = LoopyExecutor().differential(fused_exact, Schedule(fused_exact), arrays)
    assert fact.status.value == "tested", fact.provenance
    assert fact.provenance["outputs"]["y"]["exactness"] == "exact"


def test_on_hardware_with_fma_the_pin_is_what_keeps_the_bits() -> None:
    # A compiler that contracts, as clang on arm64 does with no flags at all:
    # GCC told to contract, and to use the instructions this machine has.
    contracting = ["-march=native", "-ffp-contract=fast"]
    native = {name: value.copy() for name, value in inputs().items()}
    fused_exact(**native)

    loose = compiled(fused_approx, contracting)
    if np.array_equal(loose, native["y"]):
        pytest.skip(
            "this machine did not fuse a * b + c even when told to, so it has "
            "no FMA to show the difference with"
        )
    assert np.array_equal(loose, np.full(4, -(2.0**-60)))

    # The same compiler, handed the lowering's own options after its own, as
    # loopy appends them to the toolchain's flags: the pin has the last word.
    from loopty.lower import lower_generic

    lowering = lower_generic(fused_exact.trace(), "c")
    own = list(lowering.kernel.default_entrypoint.options.build_options or ())
    pinned = compiled(fused_exact, [*contracting, *own])
    assert np.array_equal(pinned, native["y"])
