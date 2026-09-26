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

The source ``loopty run --emit-code`` prints travels without those flags, and
GCC ignores the standard pragma, so an ``exact`` kernel's C also carries GCC's
own pragma. The tests at the end compile that source by hand, the way a reader
of it would, in a GNU dialect in which GCC contracts by default.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess

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


def compiled(
    kernel_: object, build_options: list[str], may_skip: bool = True
) -> np.ndarray:
    """``y`` from the lowered kernel built with exactly these extra C flags.

    A toolchain that refuses the flags skips the test, unless ``may_skip`` is
    false: once one build with them has succeeded, a second one that fails is a
    failure, not a property of the machine.
    """
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
        if not may_skip:
            raise
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

    # A schedule transforms that kernel, and the executor builds what the
    # schedule holds: the pin has to survive the steps.
    from loopty.schedule import Schedule

    scheduled = Schedule(fused_exact).split("i", 2).kernel
    assert NO_CONTRACTION_FLAG in scheduled.default_entrypoint.options.build_options
    code = lp.generate_code_v2(scheduled).device_code()
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
    pinned = compiled(fused_exact, [*contracting, *own], may_skip=False)
    assert np.array_equal(pinned, native["y"])


# {{{ the emitted source, compiled by hand

#: A GNU dialect, in which GCC contracts by default, told that the machine has
#: FMA. Only the assembly is asked for, so the machine need not have it.
BY_HAND = ["-std=gnu99", "-O2", "-mfma"]


def assembly(source: str, tmp_path, flags: list[str]) -> str:
    """What ``gcc`` makes of ``source``, or a skip when it cannot be asked."""
    gcc = shutil.which("gcc")
    if gcc is None:  # pragma: no cover - depends on the local toolchain
        pytest.skip("no gcc here to compile the emitted source with")
    path = tmp_path / "emitted.c"
    path.write_text(source, encoding="utf-8")
    done = subprocess.run(
        [gcc, *flags, "-S", "-o", "-", str(path)], capture_output=True, text=True
    )
    if done.returncode:  # pragma: no cover - not an x86-64 toolchain, say
        pytest.skip(f"gcc {' '.join(flags)} cannot compile here: {done.stderr}")
    return done.stdout


def fused(code: str) -> bool:
    """Does x86-64 assembly use a fused multiply-add instruction?"""
    return any(op in code for op in ("vfmadd", "vfmsub", "vfnmadd", "vfnmsub"))


def test_the_emitted_source_of_an_exact_kernel_keeps_gcc_from_contracting(
    tmp_path,
) -> None:
    from loopty.executor import emit_code
    from loopty.lower import GCC_NO_CONTRACTION_PRAGMA

    exact = emit_code(fused_exact)
    assert "#pragma STDC FP_CONTRACT OFF" in exact
    assert GCC_NO_CONTRACTION_PRAGMA in exact
    loose = emit_code(fused_approx)
    assert "GCC optimize" not in loose

    if not fused(assembly(loose, tmp_path, BY_HAND)):
        pytest.skip(f"gcc {' '.join(BY_HAND)} does not contract a * b + c here")
    assert not fused(assembly(exact, tmp_path, BY_HAND))


def run_by_hand(source: str, name: str, tmp_path, flags: list[str]) -> np.ndarray:
    """``y`` from ``source`` built by hand into a shared library and called."""
    gcc = shutil.which("gcc")
    if gcc is None:  # pragma: no cover - depends on the local toolchain
        pytest.skip("no gcc here to compile the emitted source with")
    # The argument order the call below relies on.
    assert f"void {name}(int32_t const n, double const *__restrict__ a," in source
    path = tmp_path / f"{name}.c"
    library = tmp_path / f"{name}.so"
    path.write_text(source, encoding="utf-8")
    done = subprocess.run(
        [gcc, *flags, "-shared", "-fPIC", "-o", str(library), str(path)],
        capture_output=True,
        text=True,
    )
    if done.returncode:  # pragma: no cover - depends on the local toolchain
        pytest.skip(f"gcc {' '.join(flags)} cannot build here: {done.stderr}")
    function = getattr(ctypes.CDLL(str(library)), name)
    double_p = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [ctypes.c_int32, double_p, double_p, double_p, double_p]
    function.restype = None
    arrays = inputs()
    function(
        4, *(arrays[key].ctypes.data_as(double_p) for key in ("a", "b", "c", "y"))
    )
    return arrays["y"]


def test_on_hardware_with_fma_the_emitted_source_keeps_the_bits(tmp_path) -> None:
    # The same source, built for this machine in a GNU dialect: on hardware
    # with FMA, GCC fuses the approx kernel and, told by the pragma, not the
    # exact one.
    from loopty.executor import emit_code

    here = ["-std=gnu99", "-O2", "-march=native"]
    native = {name: value.copy() for name, value in inputs().items()}
    fused_exact(**native)
    loose = run_by_hand(emit_code(fused_approx), "fused_approx", tmp_path, here)
    if np.array_equal(loose, native["y"]):
        pytest.skip(
            "gcc -march=native did not fuse a * b + c here, so this machine has "
            "no FMA to show the difference with"
        )
    assert np.array_equal(loose, np.full(4, -(2.0**-60)))
    pinned = run_by_hand(emit_code(fused_exact), "fused_exact", tmp_path, here)
    assert np.array_equal(pinned, native["y"])


# }}}
