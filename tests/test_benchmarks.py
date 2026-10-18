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

pytest.importorskip("loopy")

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


def test_the_report_matches_copies_by_mesh_and_device_and_drops_wrong_rows(
    tmp_path, capsys
) -> None:
    import csv

    driver = _driver()
    device = {"device": "a GPU", "driver": "1"}
    copy = tmp_path / "copy.csv"
    with copy.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, ["operator", "order", "elements", "device", "driver", "bandwidth"]
        )
        writer.writeheader()
        # 104 bytes a node in a millisecond: 100% of the roof below
        writer.writerow(
            {
                "operator": "weak_grad",
                "order": 3,
                "elements": 2,
                **device,
                "bandwidth": 104 * 512 / 1e-3,
            }
        )
    rows = tmp_path / "rows.csv"
    with rows.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, driver["FIELDS"], extrasaction="ignore")
        writer.writeheader()
        base = {"operator": "weak_grad", "order": 3, "ndofs": 512, "median_s": 1e-3}
        writer.writerow(
            {**base, "variant": "right", "elements": 2, "agree": "1e-16", **device}
        )
        writer.writerow(
            {**base, "variant": "other mesh", "elements": 4, "agree": "1e-16", **device}
        )
        writer.writerow(
            {**base, "variant": "wrong", "elements": 2, "agree": "0.5", **device}
        )
    driver["main"](["report", str(copy), str(rows)])
    out = capsys.readouterr().out
    assert "left out, it disagrees with numpy: wrong" in out
    assert "| right | 0.00051 (100%) |" in out
    assert "| other mesh | 0.00051 |" in out
    assert "| wrong |" not in out
    # the other mesh is another measurement, in a table of its own
    assert out.count("weak_grad: GDOFs/s") == 2
    assert "4**3 hexahedra on a GPU" in out


def test_the_c_mode_times_the_three_programs_with_one_set_of_flags(
    tmp_path, capsys
) -> None:
    # The weak gradient's passes one after another, fused, and fused and
    # contracted, each checked against numpy before it is timed, all built
    # with loopty's flags and the same extra ones.
    import csv

    driver = _driver()
    paths = []
    for flags in ("-O2", "-O1"):
        out = tmp_path / f"c{flags}.csv"
        paths.append(str(out))
        driver["main"](
            [
                "c",
                "--elements",
                "2",
                "--orders",
                "3",
                "--warmup",
                "0",
                "--reps",
                "1",
                "--batch",
                "1",
                f"--cflags={flags}",
                "--csv",
                str(out),
            ]
        )
        with out.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["variant"] for row in rows] == [
            "loopty on C, passes one after another",
            "loopty on C, fused",
            "loopty on C, fused and contracted",
        ]
        assert all(float(row["agree"]) <= driver["AGREEMENT"] for row in rows)
        assert {row["flags"] for row in rows} == {f"-std=c99 -O3 -fPIC -fwrapv {flags}"}
        assert {row["platform"] for row in rows} == {"C target, one thread"}
    capsys.readouterr()
    # two sets of flags are two measurements, each in a table of its own
    driver["main"](["report", *paths])
    out = capsys.readouterr().out
    assert out.count("weak_grad: GDOFs/s") == 2
    assert "built with -std=c99 -O3 -fPIC -fwrapv -O1" in out
    assert out.count("| loopty on C, fused and contracted | ") == 2


# {{{ the advection right-hand side

ADVECTION = Path(__file__).resolve().parent.parent / "benchmarks" / "advection.py"


def _advection() -> dict[str, Any]:
    return runpy.run_path(str(ADVECTION), run_name="advection_driver")


def test_the_advection_bytes_are_those_of_each_geometry() -> None:
    driver = _advection()
    n = 4
    # u, rhs, 1/J and V at a node, sJ v.n at a face node
    assert driver["required_bytes"](3, 10) == 8 * 10 * (6 * n**3 + 6 * n**2)
    # and from the metric: nine metric terms at a node, the normal and sJ at
    # a face node
    assert driver["required_bytes"](3, 10, "one launch, from the metric") == (
        8 * 10 * (12 * n**3 + 24 * n**2)
    )


def test_every_advection_shape_names_its_axes() -> None:
    driver = _advection()
    for shape in driver["SHAPES"]:
        roles = (("nodes", "ekji"), ("faces", "efab"), ("face nodes", "eab"))
        for role, loops in roles:
            splits, tags, sequential = shape.steps(role, tuple(loops))
            assert "g.0" in tags.values() and "l.0" in tags.values(), shape
            assert not set(sequential) & set(tags), shape


@pytest.mark.parametrize(
    "split",
    [
        "three launches",
        "volume, surface",
        "fluxes, volume",
        "one launch",
        "three launches, the GLL lift on the face nodes, atomic",
        "one launch, from the metric",
    ],
)
def test_every_advection_variant_is_accepted_and_builds(plain_opencl, split) -> None:
    driver = _advection()
    demo = driver["load_demo"]()
    assert split in driver["SPLITS"]
    for shape in driver["SHAPES"]:
        mine = driver["loopty_launches"](demo, split, shape, order=3, ne=8)
        theirs = driver["plain_launches"](split, shape, order=3, ne=8)
        assert len(mine) == len(theirs) == len(driver["SPLITS"][split])
        for launch in mine:
            assert launch.loopty
            assert {fact.status.value for fact in launch.facts} == {"decided"}
        for ours, plain in zip(mine, theirs):
            assert _axes(ours.kernel) == _axes(plain.kernel), shape.name
            assert set(ours.args) == set(plain.args), shape.name


def test_the_advection_reference_is_the_demos_program() -> None:
    import numpy as np

    driver = _advection()
    demo = driver["load_demo"]()
    host = driver["host_arrays"](demo, 3, 2)
    data = demo["inputs"](2, 3)
    demo["advection_rhs"](**data)
    expected = data["rhs"].numpy()
    got = driver["reference"](host, np.arange(8))
    assert np.abs(got - expected).max() <= 1e-13 * np.abs(expected).max()
    downwind = driver["reference"](host, np.arange(8), downwind=True)
    assert np.abs(downwind - expected).max() > 1e-3 * np.abs(expected).max()


# }}}


# {{{ the QBX layer potential

QBX = Path(__file__).resolve().parent.parent / "benchmarks" / "qbx_layer_potential.py"


def test_the_qbx_kernels_are_timed_in_sumpys_arithmetic() -> None:
    """Each layer under the math steps of the example's own schedules.

    The driver imports the example afresh for each kernel it builds, so the
    single layer is told from the double by its name: one told apart by
    identity across two imports lost ``log_of_the_modulus`` (Codex, #341).
    """
    pytest.importorskip("lanky.known")
    driver = runpy.run_path(str(QBX), run_name="qbx_driver")
    names = {}
    for layer in ("single", "double"):
        (rewritten,) = driver["arithmetic"](driver["example"](), layer, "c").rewrites
        names[layer] = [step.identity.name for step in rewritten.steps]
    assert names == {
        "single": ["conjugate_form", "log_of_the_modulus"],
        "double": ["conjugate_form"],
    }


# }}}
