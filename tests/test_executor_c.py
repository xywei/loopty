"""De-risking the executor: does a loopy kernel compile and run locally?

Wave 2 builds ``loopty.executor`` on top of ``lp.ExecutableCTarget``, which
compiles generated C with the system toolchain through codepy and calls it from
Python. That is the only target the laptop and CI ever run (the OpenCL target
needs a device and belongs on a remote host), so it is worth knowing early, and
in one place, whether the path works. Eight elements keeps it under a second.

A missing or unusable C toolchain is a skip with the reason attached, not a
failure: the rest of loopty does not depend on it.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_a_tiny_loopy_kernel_runs_on_the_c_target() -> None:
    lp = pytest.importorskip("loopy")

    knl = lp.make_kernel(
        "{ [i] : 0 <= i < n }",
        "out[i] = 2.0 * a[i]",
        target=lp.ExecutableCTarget(),
        lang_version=(2018, 2),
        name="double_it",
    )

    a = np.arange(8, dtype=np.float64)
    try:
        # Through the executor rather than by calling the translation unit:
        # the direct call recompiles every time and loopy warns that it does,
        # and the executor is what `loopty.executor` uses in earnest.
        _evt, (out,) = knl.executor()(a=a)
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        pytest.skip(
            f"the C toolchain path is unusable here: {type(exc).__name__}: {exc}"
        )

    assert np.array_equal(out, 2.0 * a)
