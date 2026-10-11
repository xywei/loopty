"""The timing driver of ``examples/hexahedra.py``, without a device.

``benchmarks/hexahedra.py`` times loopty's launches beside the same kernels
in plain loopy on a device, which CI does not have. What can be checked
without one is that every variant it times is a schedule loopty accepts and
loopy can generate code for, and that the plain loopy kernels it compares
with have the same loops on the same axes.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest
from lanky.prelude import FnType

pytest.importorskip("loopy")

#: ``examples/hexahedra.py`` writes its theorem with families of several
#: indices, which are lanky main's and not the 0.2.0 release's.
pytestmark = pytest.mark.skipif(
    not hasattr(FnType, "indices"),
    reason="the demo's theorems index families at several indices (lanky main)",
)

DRIVER = Path(__file__).resolve().parent.parent / "benchmarks" / "hexahedra.py"


def _driver() -> dict[str, Any]:
    return runpy.run_path(str(DRIVER), run_name="hexahedra_driver")


def test_the_operators_move_thirteen_doubles_per_node() -> None:
    driver = _driver()
    for op in driver["OPERATORS"].values():
        assert driver["required_bytes"](op, 1000) == 8 * 13 * 1000


def test_every_shape_names_its_axes() -> None:
    driver = _driver()
    loops = {"e": "e", "k": "k", "j": "j", "i": "i", "q": "q"}
    names = set()
    for shape in driver["SHAPES"]:
        names.add(shape.name)
        splits, tags, sequential = shape.steps(loops)
        assert "l.0" in tags.values() and "g.0" in tags.values(), shape
        assert not set(sequential) & set(tags), shape
    assert len(names) == len(driver["SHAPES"])


def _axes(kernel: Any) -> set[str]:
    """The hardware axes a kernel's loops are on, as loopy writes them."""
    from loopy.kernel.data import HardwareConcurrentTag

    entry = kernel.default_entrypoint
    return {
        str(tag)
        for iname in entry.inames.values()
        for tag in iname.tags
        if isinstance(tag, HardwareConcurrentTag)
    }


@pytest.mark.parametrize("op", ["weak_grad", "weak_div"])
def test_every_variant_is_accepted_and_builds(plain_opencl, op: str) -> None:
    driver = _driver()
    demo = driver["load_demo"]()
    for shape in driver["SHAPES"]:
        mine = driver["loopty_launches"](demo, op, shape, order=3, ne=8)
        theirs = driver["plain_launches"](op, shape, order=3, ne=8)
        assert len(mine) == len(theirs) == 3
        for launch in mine:
            assert launch.loopty
            assert {fact.status.value for fact in launch.facts} == {"decided"}
        for ours, plain in zip(mine, theirs):
            assert _axes(ours.kernel) == _axes(plain.kernel), shape.name
