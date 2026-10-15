"""The demos, run the three ways the project promises they run.

Each file in ``examples/`` is exercised as a subprocess under plain ``python``,
under ``lanky check``, and under ``loopty run``, because that is what a reader
will do with it and because the three paths share almost nothing: the first is
numpy alone, the second is the typing rules and the oracles, the third is loopy
and a C compiler. The ledgers are read back as JSON rather than scraped from the
rendered table, so an assertion here is about a fact's status and its decider
and not about column widths.

The commands are cached per file, so each subprocess runs once however many
assertions read it. Sizes in the demos are tiny on purpose; the whole module is
a few seconds once the compiler cache is warm.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import islpy as isl
import numpy as np
import pytest
from lanky import terms as lanky_terms
from lanky.prelude import FnType

pytest.importorskip("loopy")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

#: Every demo. A demo whose ledger has an ``assumed`` fact is not a failure: a
#: postcondition nobody can decide yet is exactly what the ledger is for. A
#: ``refuted`` one is, and every demo here is meant to come out clean.
NAMES = (
    "spmv",
    "stencil_skew",
    "wavefront_acoustic",
    "reshape_layouts",
    "p2p",
    "pairs",
    "composition",
    "travel",
    "fusion",
    "quadrature",
    "layer_potential",
    "sum_factorization",
    "calls",
    "assembly",
    "storage_layouts",
    "l2p",
    "hexahedra",
    "advection",
)

#: The families with several indices, ``Fn[Nat, Nat, Real]`` applied as
#: ``D[a, b]``, that ``examples/l2p.py`` writes its theorems with, and
#: ``Fn[Fin[n], Fin[n], Real]`` applied as ``Fr[i, a]``, that
#: ``examples/hexahedra.py`` and ``examples/advection.py`` write their theorems
#: with, are lanky main's and not the 0.2.0 release's, which the released-lanky
#: job installs.
SEVERAL_INDICES = hasattr(FnType, "indices")

#: The demos that need them.
NEEDS_SEVERAL_INDICES = {"l2p", "hexahedra", "advection"}

needs_several_indices = pytest.mark.skipif(
    not SEVERAL_INDICES,
    reason="the demo's theorems index families at several indices (lanky main)",
)

#: ``examples/hexahedra.py``'s operators claim over ``x in Fin[3]`` beside
#: ``e in Fin[ne]``, which lanky reads as two binders since lanky#144.
LITERAL_BINDERS = hasattr(lanky_terms, "literal_binders")

#: The demos that need that.
NEEDS_LITERAL_BINDERS = {"hexahedra"}

needs_literal_binders = pytest.mark.skipif(
    not LITERAL_BINDERS,
    reason="the demo's claims put a literal domain beside a symbolic one (lanky main)",
)

#: The parts of a complex number, ``real``, ``imag`` and ``conj``, that
#: ``examples/layer_potential.py``'s identity is written with, are lanky
#: main's and not the 0.2.0 release's.
PARTS = "conj" in getattr(lanky_terms, "ELEMENTARY_FUNCTIONS", ())

#: The demos that need that.
NEEDS_PARTS = {"layer_potential"}

needs_parts = pytest.mark.skipif(
    not PARTS, reason="the demo's identity takes complex numbers apart (lanky main)"
)

#: Every demo, as a parameter: those that need lanky main skip without it.
DEMOS = [
    pytest.param(
        name,
        marks=[
            *([needs_several_indices] if name in NEEDS_SEVERAL_INDICES else []),
            *([needs_literal_binders] if name in NEEDS_LITERAL_BINDERS else []),
            *([needs_parts] if name in NEEDS_PARTS else []),
        ],
    )
    for name in NAMES
]

_RESULTS: dict[tuple[str, str], tuple[Any, list[dict]]] = {}
_MODULES: dict[str, Any] = {}


def _path(name: str) -> Path:
    return EXAMPLES / f"{name}.py"


def _invoke(name: str, verb: str) -> tuple[Any, list[dict]]:
    """Run one command on one demo, once, and return the process and its facts."""
    key = (name, verb)
    if key in _RESULTS:
        return _RESULTS[key]

    with tempfile.TemporaryDirectory() as directory:
        ledger = Path(directory) / "ledger.json"
        if verb == "python":
            command = [sys.executable, str(_path(name))]
        elif verb == "check":
            command = [
                sys.executable,
                "-m",
                "lanky.cli",
                "check",
                str(_path(name)),
                "--json",
                str(ledger),
            ]
        elif verb == "run":
            command = [
                sys.executable,
                "-m",
                "loopty.cli",
                "run",
                str(_path(name)),
                "--json",
                str(ledger),
            ]
        else:  # pragma: no cover - a typo in a test
            raise ValueError(verb)
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        facts = json.loads(ledger.read_text()) if ledger.exists() else []

    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 and (
        "compil" in output.lower() or "cc1" in output or "gcc" in output.lower()
    ):  # pragma: no cover - depends on the local toolchain
        pytest.skip(f"the C toolchain path is unusable here:\n{output[-2000:]}")
    _RESULTS[key] = (result, facts)
    return _RESULTS[key]


def _module(name: str) -> Any:
    """Import a demo in this process, for the parts a ledger does not show."""
    if name not in _MODULES:
        from lanky.check import import_path

        _MODULES[name] = import_path(str(_path(name)))
    return _MODULES[name]


def _statuses(facts: list[dict], **fields: Any) -> list[str]:
    """The statuses of the facts matching every given field."""
    return [
        fact["status"]
        for fact in facts
        if all(fact.get(key) == value for key, value in fields.items())
    ]


# {{{ every demo, the three commands


@pytest.mark.parametrize("name", DEMOS)
def test_a_demo_runs_under_plain_python(name: str) -> None:
    result, _ = _invoke(name, "python")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip(), "a demo that prints nothing demonstrates nothing"


@pytest.mark.parametrize("name", DEMOS)
def test_a_demo_states_obligations_and_none_is_refuted(name: str) -> None:
    result, facts = _invoke(name, "check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert facts, "a kernel file with no obligations is not checking anything"
    assert "refuted" not in {fact["status"] for fact in facts}
    # Every fact that was settled says who settled it, which is the whole point
    # of the ledger: a status with no decider is a status nobody can audit.
    for fact in facts:
        if fact["status"] != "assumed":
            assert fact["decided_by"], fact


#: The kernels of each demo, each of which owes one faithfulness fact.
#: The kernels and the programs of each demo, each of which has a
#: ``trace-faithful`` fact.
KERNELS = {
    "spmv": {"scan", "spmv", "solve"},
    "stencil_skew": {"jacobi"},
    "wavefront_acoustic": {"acoustic"},
    "reshape_layouts": {"rows_of", "cols_of", "transpose"},
    "p2p": {"p2p"},
    "pairs": {"pairs"},
    "composition": {"flux", "divergence", "burgers_rhs"},
    "fusion": {"flux", "divergence", "burgers_rhs"},
    "quadrature": {"mass_action"},
    "layer_potential": {"single_layer", "double_layer"},
    "sum_factorization": {"along_r", "along_s", "weak_derivative"},
    "calls": {"dot", "matvec", "csr", "integrate", "element_mass", "mass_matrices"},
    "assembly": {"mass_ffc", "mass_action"},
    "storage_layouts": {"stored_table", "unflatten"},
    "l2p": {"unpack", "reconstruct", "evaluate", "l2p"},
    "hexahedra": {
        "metric",
        "contravariant",
        "along_r",
        "along_s",
        "gradient_along_t",
        "divergence_along_t",
        "weak_grad",
        "weak_div",
        "metric_along_r",
        "contravariant_along_r",
    },
    "advection": {
        "reference_velocity",
        "normal_velocity",
        "upwind",
        "volume",
        "lift",
        "lift_on_faces",
        "advection_rhs",
        "folded_rhs",
        "surface",
        "volume_and_lift",
    },
    "travel": {
        "number",
        "number_quiet",
        "gather",
        "permuted",
        "checked",
        "scan",
        "rowsums",
        "through",
        "scan_flat",
        "weigh",
        "flat",
    },
}


#: The kernels whose drawn inputs the body may refuse. ``weigh`` reads a flat
#: buffer of ``nnz`` cells through offsets of ``Fin[nnz + 1]``, and counts
#: drawn from their type can sum past ``nnz``: its contract refuses the
#: offsets they give, and an input the body refuses says nothing either way.
REFUSES_DRAWS = {("travel", "weigh"), ("travel", "flat")}


@pytest.mark.parametrize("name", DEMOS)
def test_every_kernel_of_a_demo_computes_what_its_body_computes(name: str) -> None:
    # The traced term, interpreted, against the native body: on the demo's
    # own example inputs first, then on inputs drawn from the declared types.
    # A program's term is composed from its calls, and it has the fact too.
    _result, facts = _invoke(name, "check")
    faithful = [fact for fact in facts if fact["kind"] == "trace-faithful"]
    assert {fact["owner"] for fact in faithful} == KERNELS[name]
    for fact in faithful:
        assert fact["status"] == "tested", fact
        assert fact["decided_by"] == "interpreter"
        inputs = fact["provenance"]["inputs"]
        assert inputs[0] == {"input": "example_inputs()", "outcome": "agreed"}
        outcomes = [entry["outcome"] for entry in inputs]
        if (name, fact["owner"]) in REFUSES_DRAWS:
            assert len(outcomes) == 4
            assert all(
                outcome == "agreed" or outcome.startswith("skipped: the body raised")
                for outcome in outcomes
            ), outcomes
            continue
        assert outcomes == ["agreed"] * 4


@pytest.mark.parametrize("name", DEMOS)
def test_a_demo_compiles_and_agrees_with_the_native_run(name: str) -> None:
    result, facts = _invoke(name, "run")
    assert result.returncode == 0, result.stdout + result.stderr
    agreements = [fact for fact in facts if fact["kind"] == "agreement"]
    assert agreements, "nothing was compiled and compared"
    for fact in agreements:
        assert fact["status"] == "tested", fact
        assert fact["decided_by"] == "loopy"
        for output, detail in fact["provenance"]["outputs"].items():
            assert detail["agree"], (output, detail)
            assert detail["difference"] <= detail["tolerance"], (output, detail)


# }}}


# {{{ spmv


def test_the_spmv_ledger_has_both_deciders_and_the_accumulations_class() -> None:
    _result, facts = _invoke("spmv", "check")
    deciders = {fact["decided_by"] for fact in facts}
    assert "isl" in deciders, "no affine obligation was decided"
    # The indirection through the column array is the one obligation isl cannot
    # decide, and the one the index type discharges outright.
    by_type = [fact for fact in facts if fact["decided_by"] == "type"]
    assert any("x[col[r, j]]" in fact["statement"] for fact in by_type), by_type

    # The class is read off what is summed: ``val`` and ``x`` are ``Real``, so
    # the sum is ``approx``. ``reassoc`` is not something a trace may assume;
    # it is what a schedule lowers an accumulation to when it reorders one, and
    # that fact is emitted by the schedule, below.
    # lanky check lists the casts of the file's schedule too (#191), among
    # them the reassociation realize() makes; the kernel's own is approx
    exactness = [
        fact
        for fact in facts
        if fact["kind"] == "exactness" and not fact["id"].startswith("cast:")
    ]
    assert exactness, "the accumulation's exactness class was never stated"
    assert all("is approx" in fact["statement"] for fact in exactness), exactness
    assert all(fact["status"] == "decided" for fact in exactness)


def test_realizing_the_spmv_accumulation_records_the_reassociation() -> None:
    _result, facts = _invoke("spmv", "run")
    reassociated = [
        fact
        for fact in facts
        if fact["kind"] == "exactness" and "reassociated" in fact["statement"]
    ]
    assert reassociated, [fact["statement"] for fact in facts]
    assert all(fact["status"] == "decided" for fact in reassociated)


def test_the_scan_postcondition_and_its_theorem_are_in_the_ledger() -> None:
    _result, facts = _invoke("spmv", "check")
    post = _statuses(facts, kind="postcondition", owner="scan")
    # Decided from the term (#113): every cell of off is stored once, and read
    # after its store, so what each statement stores is what the kernel leaves;
    # proved where Lean is installed.
    assert post and set(post) <= {"decided", "proved"}, post

    theorem = _statuses(facts, owner="scan_monotone")
    assert theorem and set(theorem) <= {"tested", "proved", "certified"}, theorem


def test_the_spmv_demo_prints_the_agreement_it_found() -> None:
    result, _ = _invoke("spmv", "python")
    assert "-> tested" in result.stdout
    assert "device schedule" in result.stdout


def test_the_device_schedule_of_spmv_is_legal_and_not_buildable() -> None:
    # Every cast is decided: the transformation preserves the meaning of the
    # program. What it does not survive is code generation, on a device as much
    # as on C, which the schedule now reports instead of leaving to loopy.
    from loopty.executor import LoopyExecutor
    from loopty.schedule import UnbuildableSchedule

    module = _module("spmv")
    schedule = module.device_schedule()
    assert schedule.tags == {"r": "g.0", "j_in": "l.0"}
    assert schedule.reassociated == frozenset({"y"})

    casts = [fact for fact in schedule.facts() if fact.kind != "buildable"]
    assert casts and all(fact.status.value == "decided" for fact in casts)

    buildable = [fact for fact in schedule.facts() if fact.kind == "buildable"]
    assert [fact.status.value for fact in buildable] == ["refuted"]
    assert buildable[0].decided_by == "loopy-target"
    ok, reason = schedule.buildable
    assert not ok and "ragged fiber" in reason

    # And asking it to run says so, rather than throwing from inside loopy.
    with pytest.raises(UnbuildableSchedule, match="ragged fiber"):
        LoopyExecutor().run(schedule, **module.example_inputs()["spmv"])


def test_one_row_per_group_is_the_schedule_that_does_build(plain_opencl) -> None:
    # The same product with only the rows parallel: the ragged loop stays
    # sequential, so nothing asks for a hardware axis inside it. It builds for
    # a device (loopy's plain OpenCL target stands in for one here), and not
    # for C, which has no hardware axes at all (#47).
    from loopty.isl_reading import declining

    module = _module("spmv")
    schedule = module.rows_parallel()
    assert schedule.target == "opencl"
    assert schedule.buildable == (True, "")
    assert [fact.status.value for fact in schedule.facts()] == ["decided"] * 2
    with declining():
        code = plain_opencl.generate_code_v2(schedule.kernel).device_code()
    assert "get_group_id" in code

    on_c = module.rows_parallel(target="c")
    ok, reason = on_c.buildable
    assert not ok
    assert reason.startswith("the tag r='g.0' puts a loop on a hardware axis")
    assert "Retarget to opencl" in reason


# }}}


# {{{ the stencil


def test_the_rectangular_tiling_is_refuted_with_a_witness_pair() -> None:
    module = _module("stencil_skew")
    message, witness = module.rejected_tiling()
    assert "illegal" in message
    assert "scheduled earlier" in message

    (source_id, source), (sink_id, sink), params = witness
    assert source_id == sink_id
    # The two instances are one time step apart and one space step apart: the
    # (1, 1) and (1, -1) dependences a rectangular tile cuts.
    assert sink["t"] == source["t"] + 1
    assert abs(sink["i"] - source["i"]) == 1
    assert params["nt"] > 0 and params["nx"] > 0


def test_the_refused_tiling_leaves_a_refuted_fact_behind() -> None:
    module = _module("stencil_skew")
    from loopty.schedule import IllegalCast, Schedule

    schedule = Schedule(module.jacobi, target="c", sizes={"nt": 16, "nx": 16})
    with pytest.raises(IllegalCast) as refused:
        schedule.tile("t", "i", 8, 8)
    assert refused.value.fact.status.value == "refuted"
    assert refused.value.fact.kind == "monotone"
    assert refused.value.fact.provenance["witness"] == refused.value.witness
    assert "not ordered forward" in refused.value.fact.provenance["detail"]
    # The schedule the cast was asked of is untouched.
    assert schedule.history == ()


def test_skewing_first_makes_the_same_tiling_decided() -> None:
    module = _module("stencil_skew")
    schedule = module.skewed()
    assert schedule.history == ("skew(i, by='t')", "tile(t,i,8,8)")
    assert [fact.status.value for fact in schedule.facts()] == ["decided"] * 4
    assert schedule.order == ("t_outer", "i_outer", "t_inner", "i_inner")


def test_the_stencil_demo_prints_the_rejection_then_the_agreement() -> None:
    result, _ = _invoke("stencil_skew", "python")
    assert "IllegalCast" in result.stdout
    assert "witness:" in result.stdout
    assert "-> tested" in result.stdout
    assert "matches the hand-written sweep: True" in result.stdout


def test_the_stencil_in_diamond_coordinates_agrees_with_the_native_run() -> None:
    # The README's answer for the stencil, against the traced kernel's own
    # body rather than a numpy reference: the diamond, and rectangles in it,
    # which are the diamonds of diamond tiling. The approx tolerance allows
    # rounding; the difference is none.
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule

    module = _module("stencil_skew")
    diamond = Schedule(module.jacobi, sizes={"nt": 16, "nx": 16}).affine(
        "{ [t, i] -> [a, b] : a = t + i and b = t - i }"
    )
    for schedule in (diamond, diamond.tile("a", "b", 4, 4)):
        assert [f.status.value for f in schedule.facts()] == ["decided"] * len(
            schedule.facts()
        )
        fact = LoopyExecutor().differential(
            module.jacobi, schedule, module.example_inputs()
        )
        assert fact.status.value == "tested", schedule
        assert fact.provenance["outputs"]["u"]["difference"] == 0.0, schedule


# }}}


# {{{ the coupled wavefront stencil


def test_the_wave_example_has_cross_instruction_time_dependence() -> None:
    module = _module("wavefront_acoustic")
    assert [stmt.id for stmt in module.acoustic.term.stmts] == ["S0", "S1"]

    message, witness = module.rejected_tiling()
    assert "illegal" in message
    assert "writes pressure[" in message
    (source_id, source), (sink_id, sink), params = witness

    # Unlike stencil_skew, the cut dependence crosses the two instructions:
    # pressure from S1 at the previous time level feeds velocity in S0. It is
    # the only dependence with a negative space distance, (1, -1).
    assert source_id != sink_id
    assert source_id == "S1"
    assert sink_id == "S0"
    assert sink["t"] == source["t"] + 1
    assert source["i"] == sink["i"] + 1
    assert params["nt"] > 0 and params["nx"] > 0


@pytest.mark.parametrize(
    ("sizes", "tile", "hinted"),
    [
        ((16, 32), (2, 2), True),
        ((9, 33), (3, 5), True),
        ((64, 64), (4, 16), True),
        ((5, 7), (4, 8), False),
    ],
)
def test_every_rectangular_wave_tile_cuts_the_same_dependence(
    sizes: tuple[int, int], tile: tuple[int, int], hinted: bool
) -> None:
    # The pair the demo prints is not an accident of where isl happened to
    # look. Whatever the tile and the sizes, the witness is S1 and then S0, one
    # step later in time and one back in space, because that is the only
    # dependence a rectangle cuts. The last case has no space-tile boundary at
    # the hinted sizes, so isl chooses the sizes as well, and the pair is the
    # same.
    from loopty.schedule import IllegalCast, Schedule

    module = _module("wavefront_acoustic")
    nt, nx = sizes
    schedule = Schedule(module.acoustic, target="c", sizes={"nt": nt, "nx": nx})
    with pytest.raises(IllegalCast) as refused:
        schedule.tile("t", "i", *tile)
    assert ("as hinted" in str(refused.value)) is hinted
    (source_id, source), (sink_id, sink), _params = refused.value.witness
    assert (source_id, sink_id) == ("S1", "S0")
    assert (sink["t"] - source["t"], sink["i"] - source["i"]) == (1, -1)


def test_skewing_the_coupled_wave_makes_the_tile_legal() -> None:
    module = _module("wavefront_acoustic")
    schedule = module.wavefront_schedule()
    assert schedule.history == ("skew(i, by='t')", "tile(t,i,4,8)")
    assert [fact.status.value for fact in schedule.facts()] == ["decided"] * 4
    assert schedule.order == ("t_outer", "i_outer", "t_inner", "i_inner")


def test_the_wave_schedules_emit_one_text_whatever_the_hash_seed(tmp_path) -> None:
    # loopy's assumptions of a kernel it was given none took the sizes from a
    # frozenset, so the order isl writes them in, in every bound of the
    # wavefront and diamond schedules, followed the process's hash seed:
    # seeds 0 and 2 printed -4 + nx + nt and -4 + nt + nx (#125). Each seed
    # runs in a process of its own with no code cache to serve another's.
    script = (
        "from lanky.check import import_path\n"
        "from loopty.executor import emit_code\n"
        f"module = import_path({str(_path('wavefront_acoustic'))!r})\n"
        "for make in (module.wavefront_schedule, module.diamond_schedule,\n"
        "             module.offset_diamond_schedule):\n"
        "    print(emit_code(make()))\n"
    )
    texts = []
    for seed in ("0", "2"):
        cache = tmp_path / f"cache{seed}"
        environment = {
            **os.environ,
            "PYTHONHASHSEED": seed,
            "XDG_CACHE_HOME": str(cache),
            "LOOPY_NO_CACHE": "1",
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            env=environment,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        texts.append(result.stdout)
    assert "-4 + nt + nx" in texts[0]
    assert texts[0] == texts[1]


def test_the_wave_demo_prints_the_rejections_then_every_agreement() -> None:
    # Three schedules, the wavefront block, the diamond and the diamond
    # tiling, and two fields each.
    result, _ = _invoke("wavefront_acoustic", "python")
    assert result.stdout.count("IllegalCast") == 3
    assert "witness: S1" in result.stdout
    assert "pressure: difference" in result.stdout
    assert "velocity: difference" in result.stdout
    assert result.stdout.count("-> tested") == 6
    assert "[a, b_step]" in result.stdout
    assert "S1[t, i] -> [a = t + i, b = 1 + t - i] }).tile(a,b,4,4)" in result.stdout
    assert "matches the hand-written recurrence: True" in result.stdout


def test_loopty_run_keeps_the_facts_of_every_wave_schedule() -> None:
    # The block, the diamond and the diamond tiling are three schedules of one
    # kernel. Their facts used to share ids, so the diamond was left out of
    # the file's schedules.
    _result, facts = _invoke("wavefront_acoustic", "run")
    agreements = [fact for fact in facts if fact["kind"] == "agreement"]
    assert [fact["provenance"]["schedule"] for fact in agreements] == [
        ["skew(i, by='t')", "tile(t,i,4,8)"],
        ["affine({ [t, i] -> [a = t + i, b = t - i] })"],
        [
            "affine({ S0[t, i] -> [a = t + i, b = t - i]; "
            "S1[t, i] -> [a = t + i, b = 1 + t - i] })",
            "tile(a,b,4,4)",
        ],
    ]
    assert {fact["status"] for fact in agreements} == {"tested"}
    casts = [fact for fact in facts if fact["kind"] in ("bijective", "monotone")]
    assert len(casts) == 10
    assert len({fact["id"] for fact in facts}) == len(facts)


def test_the_space_first_diamond_runs_the_cross_statement_dependence_backwards():
    # a = i + t and b = i - t: S1 at (t, i) feeds S0 at (t + 1, i - 1), which
    # has the same a and a b two lower, so the order runs it the wrong way.
    module = _module("wavefront_acoustic")
    message, witness = module.rejected_diamond()
    assert message.startswith("affine(")
    assert "writes pressure[" in message
    (source_id, source), (sink_id, sink), _params = witness
    assert (source_id, sink_id) == ("S1", "S0")
    assert (sink["t"] - source["t"], sink["i"] - source["i"]) == (1, -1)


def test_the_time_first_diamond_is_accepted_and_agrees_bit_for_bit() -> None:
    # The question the spike asked: loopy, given the image with its holes,
    # generates code that computes what the native body computes. The
    # differential compares at the approx tolerance; the hand recurrence is
    # compared exactly, at sizes of both parities.
    from loopty.executor import LoopyExecutor

    module = _module("wavefront_acoustic")
    schedule = module.diamond_schedule()
    assert schedule.order == ("a", "b")
    assert [fact.status.value for fact in schedule.facts()] == ["decided"] * 2
    for nt, nx in [(3, 4), (5, 7), (9, 33), (16, 32)]:
        data = module.initial(nt, nx)
        pressure = data["pressure"].numpy().copy()
        velocity = data["velocity"].numpy().copy()
        out = LoopyExecutor().run(
            module.diamond_schedule(nt, nx),
            pressure=pressure.copy(),
            velocity=velocity.copy(),
            courant=module.COURANT,
        )
        want_pressure, want_velocity = module.reference(pressure, velocity)
        assert np.array_equal(out["pressure"], want_pressure), (nt, nx)
        assert np.array_equal(out["velocity"], want_velocity), (nt, nx)


def test_the_wave_diamond_compiles_to_loops_with_no_parity_test() -> None:
    # The image of the diamond is the points where a + b is even. loopy looped
    # over every b and tested the parity inside the innermost loop,
    # "if (-b - a + 2 * ((b + a) / 2) == 0)", so half of its iterations did
    # nothing. The kernel now counts b's steps, b = 2*b_step - a: its domain
    # has no holes, the code has no parity test, and it still agrees with the
    # hand recurrence bit for bit, at sizes of both parities (the test above).
    from loopty.executor import emit_code

    module = _module("wavefront_acoustic")
    for nt, nx in [(3, 4), (5, 7), (16, 32)]:
        schedule = module.diamond_schedule(nt, nx)
        assert schedule.strides == {"b": "2*b_step - a"}
        (domain,) = schedule.kernel.default_entrypoint.domains
        assert domain.get_var_names(isl.dim_type.set) == ["a", "b_step"]
        assert domain.dim(isl.dim_type.div) == 0
        assert "== 0" not in emit_code(schedule)


def test_tiling_the_wave_diamond_cuts_the_same_step_dependence() -> None:
    # What the docstring predicted: S0 at (t, i) writes the velocity S1 at
    # (t, i + 1) reads, a distance of (0, 1), which t - i runs backwards. A
    # diamond tiling of this pair needs a time offset between the statements.
    module = _module("wavefront_acoustic")
    message, witness = module.rejected_diamond_tiling()
    assert message.startswith("tile(a,b,4,4) illegal")
    assert "writes velocity[" in message
    (source_id, source), (sink_id, sink), _params = witness
    assert (source_id, sink_id) == ("S0", "S1")
    assert (sink["t"] - source["t"], sink["i"] - source["i"]) == (0, 1)


def test_the_offset_diamond_tiles_the_pair_and_agrees_bit_for_bit() -> None:
    # The time offset the refusal above asks for, as a map per statement: S1
    # sits half a step after S0 along the diamond, every dependence becomes
    # (0, 1), (1, 0) or (1, 1) in (a, b), and the rectangles are legal. affine
    # refused a map per statement before (#46).
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule

    module = _module("wavefront_acoustic")
    schedule = module.offset_diamond_schedule()
    assert schedule.order == ("a_outer", "b_outer", "a_inner", "b_inner")
    assert [fact.status.value for fact in schedule.facts()] == ["decided"] * 4
    # Against the native run, the fields differ by nothing at all, though the
    # fact only asks for the approx tolerance.
    fact = LoopyExecutor().differential(module.acoustic, schedule, module.initial())
    assert fact.status.value == "tested"
    outputs = fact.provenance["outputs"]
    assert {name: outputs[name]["difference"] for name in outputs} == {
        "pressure": 0,
        "velocity": 0,
    }
    # The two images are the points where a + b is even and those where it is
    # odd: together, every point of the loops, each of them one statement's.
    untiled = Schedule(module.acoustic, sizes={"nt": 16, "nx": 32}).affine(
        module.OFFSET_DIAMOND
    )
    (domain,) = untiled.kernel.default_entrypoint.domains
    assert domain.dim(isl.dim_type.div) == 0
    for nt, nx in [(2, 3), (3, 4), (5, 7), (9, 33), (16, 32), (17, 9)]:
        rng = np.random.default_rng(100 * nt + nx)
        pressure = rng.standard_normal((nt, nx))
        velocity = rng.standard_normal((nt, nx))
        want_pressure, want_velocity = module.reference(pressure, velocity)
        for built in (module.offset_diamond_schedule(nt, nx), untiled):
            out = LoopyExecutor().run(
                built,
                pressure=pressure.copy(),
                velocity=velocity.copy(),
                courant=module.COURANT,
            )
            assert np.array_equal(out["pressure"], want_pressure), (nt, nx)
            assert np.array_equal(out["velocity"], want_velocity), (nt, nx)


# }}}


# {{{ reshape and layouts


def test_the_layout_questions_are_all_decided_by_isl() -> None:
    module = _module("reshape_layouts")
    ledger = module.layout_ledger()
    statuses = {fact.status.value for fact in ledger}
    assert statuses == {"decided"}
    assert {fact.decided_by for fact in ledger} == {"isl"}


def test_the_product_index_type_normalizes_into_two_axes() -> None:
    from pymbolic.primitives import Variable

    from loopty.idx import Fin, normalize

    axes = normalize((Fin[Variable("n") * Variable("m")],))
    assert [str(axis) for axis in axes] == ["Fin(n)", "Fin(m)"]


def test_a_linearized_access_is_in_bounds_by_isl() -> None:
    _result, facts = _invoke("reshape_layouts", "check")
    linearized = [
        fact
        for fact in facts
        if fact["owner"] == "rows_of" and "flat[" in fact["statement"]
    ]
    assert linearized, [fact["statement"] for fact in facts]
    for fact in linearized:
        assert fact["status"] == "decided"
        assert fact["decided_by"] == "isl"


def test_the_two_layouts_read_the_same_buffer_differently() -> None:
    result, _ = _invoke("reshape_layouts", "python")
    assert "read row-major:" in result.stdout
    assert "read column-major:" in result.stdout


# }}}


# {{{ the point-to-point stretch demo


def test_the_interaction_list_indirection_is_in_bounds_by_type() -> None:
    _result, facts = _invoke("p2p", "check")
    by_type = [fact for fact in facts if fact["decided_by"] == "type"]
    assert any("lst[t, j] : Fin(n)" in fact["statement"] for fact in by_type), by_type


def test_the_guarded_self_interaction_is_left_out_of_the_sum() -> None:
    module = _module("p2p")
    data = module.scene()
    module.p2p(
        data["cnt"],
        data["lst"],
        data["x"],
        data["y"],
        data["q"],
        data["term"],
        data["pot"],
    )

    # The direct sum skips the self pair with a Python ``if``; the kernel skips
    # it with ``when``, which masks the write. The two have to agree, or the
    # mask is not doing what the guard says.
    assert np.allclose(data["pot"].numpy(), module.direct(data))
    lists = data["lst"].numpy()
    offsets = data["lst"].offsets
    assert any(
        int(lists[a]) == target
        for target in range(len(offsets) - 1)
        for a in range(offsets[target], offsets[target + 1])
    ), "the demo is pointless unless some list contains its own target"


# }}}


# {{{ the pair interactions over the triangle


def test_every_in_bounds_obligation_of_the_pairs_is_decided_by_isl() -> None:
    # Over the exact triangle: the row f[p, j], and the column f[k, p] read
    # under k > p, which only the domain, and not its box, makes a point.
    _result, facts = _invoke("pairs", "check")
    in_bounds = [fact for fact in facts if fact["kind"] == "in-bounds"]
    assert in_bounds
    assert {fact["status"] for fact in in_bounds} == {"decided"}
    assert {fact["decided_by"] for fact in in_bounds} == {"isl"}
    statements = {fact["statement"] for fact in in_bounds}
    assert "f[k, p] is in bounds for every instance of S1" in statements


def test_the_pairs_run_in_both_layouts_and_both_agree() -> None:
    result, facts = _invoke("pairs", "run")
    assert result.returncode == 0, result.stdout + result.stderr
    agreements = sorted(
        fact["id"] for fact in facts if fact["kind"] == "agreement"
    )
    # Named by the kernel's definition, then the target and the steps.
    (definition,) = {fact_id.split(":")[1] for fact_id in agreements}
    assert definition.startswith("pairs.pairs@")
    assert agreements == [
        f"agreement:{definition}:[c]",
        f"agreement:{definition}:[c].pack('f')",
    ]


def test_the_packed_pairs_keep_the_triangle_and_no_more() -> None:
    module = _module("pairs")
    boxed = module.scene()["f"]
    packed = module.scene(storage="packed")["f"]
    particles = module.PARTICLES
    assert boxed.numpy().size == particles * particles
    assert packed.numpy().size == particles * (particles - 1) // 2
    data = module.scene(storage="packed")
    module.pairs(**data)
    assert np.allclose(data["e"].numpy(), module.dense(data))


# }}}


# {{{ programs


def test_loopty_run_compiles_each_program_as_one_kernel() -> None:
    # A program is run like a kernel: its term lowered, compiled, and compared
    # with its native run, in the ledger as an agreement of its own.
    for name, program in (("spmv", "solve"), ("composition", "burgers_rhs")):
        result, facts = _invoke(name, "run")
        assert f"{program}: Schedule({program}, target='c')" in result.stdout
        (fact,) = [
            fact
            for fact in facts
            if fact["kind"] == "agreement" and fact["owner"] == program
        ]
        assert fact["status"] == "tested", fact


def test_the_composed_program_keeps_its_intermediate_to_itself() -> None:
    module = _module("composition")
    term = module.burgers_rhs.term
    assert term.param_names == ("u", "rhs")
    assert [name for name, _ in term.temporaries] == ["f"]
    assert [stmt.id for stmt in term.stmts] == [
        "f.zeros",
        "flux.S0",
        "divergence.S0",
    ]
    result, _ = _invoke("composition", "python")
    # The executor allocates it on the heap and passes it after the program's
    # own arguments, where it was a variable-length array on the stack (#62).
    assert "double *__restrict__ f)" in result.stdout
    assert "double f[n];" not in result.stdout
    assert "-> tested" in result.stdout


def test_the_composed_program_reads_only_what_its_producer_stored() -> None:
    _result, facts = _invoke("composition", "check")
    (fact,) = [fact for fact in facts if fact["kind"] == "definedness"]
    assert fact["status"] == "decided"
    assert fact["decided_by"] == "isl"
    assert fact["statement"] == (
        "every cell of f that divergence reads, flux stored before it"
    )


# }}}


# {{{ fusion (#13)


def test_the_fusion_demo_decides_the_fusion_the_substitution_and_the_window() -> None:
    result, facts = _invoke("fusion", "run")
    assert result.returncode == 0, result.stdout + result.stderr
    casts = [
        (fact["kind"], fact["status"], fact["decided_by"])
        for fact in facts
        if fact["owner"] == "burgers_rhs" and fact["kind"] != "agreement"
    ]
    assert casts == [
        ("bijective", "decided", "isl"),
        ("monotone", "decided", "isl"),
        ("definedness", "decided", "isl"),
        ("bijective", "decided", "isl"),
        ("monotone", "decided", "isl"),
        ("definedness", "decided", "isl"),
        ("no-clobber", "decided", "isl"),
    ]
    agreements = [
        fact
        for fact in facts
        if fact["owner"] == "burgers_rhs" and fact["kind"] == "agreement"
    ]
    assert [fact["status"] for fact in agreements] == ["tested"] * 3


def test_the_fusion_demo_prints_the_refusals_and_three_kernels() -> None:
    result, _ = _invoke("fusion", "python")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "refused: fuse(flux, divergence) illegal: instance flux.S0[j=2] writes "
        "f[2] read by divergence.S0[i=1] scheduled earlier (at n=16, as hinted); "
        "fuse('flux', 'divergence', shift=1) runs every dependence between them "
        "forward"
    ) in result.stdout
    assert (
        "refused: contract('f', lambda j: j % 2) illegal: instance "
        "divergence.S0[i=14] reads f[13], which flux.S0[j=13] stored in slot 1, "
        "after flux.S0[j=15] stored f[15] there (at n=16, as hinted)"
    ) in result.stdout
    fused, substituted, windowed = result.stdout.split("accepted: ")[1:]
    # Fused, f is stored, and the executor allocates it (#62); substituted,
    # it is gone; kept in a window, it is three cells on the stack (#132).
    assert "double *__restrict__ f)" in fused
    assert "__restrict__ f" not in substituted
    assert "double f[" not in substituted
    assert "0.5 * u[j] * u[j]" in substituted
    assert "double f_window[3];" in windowed
    assert "__restrict__ f" not in windowed
    assert fused.count("for (") == 2
    assert substituted.count("for (") == 1
    assert windowed.count("for (") == 1
    assert result.stdout.count("-> tested") == 3


def test_the_fusion_demo_states_the_definedness_of_its_intermediate() -> None:
    _result, facts = _invoke("fusion", "check")
    # the substitution's own definedness, a cast of the file's schedule, is
    # listed too (#191)
    (fact,) = [
        fact
        for fact in facts
        if fact["kind"] == "definedness" and not fact["id"].startswith("cast:")
    ]
    assert (fact["status"], fact["decided_by"]) == ("decided", "isl")
    assert fact["owner"] == "burgers_rhs"


# }}}


# {{{ storage layouts


def test_the_storage_layouts_demo_decides_every_layout_fact() -> None:
    result, facts = _invoke("storage_layouts", "check")
    assert result.returncode == 0, result.stdout + result.stderr
    layouts = [
        (fact["owner"], fact["kind"], fact["status"])
        for fact in facts
        if fact["kind"].startswith("layout")
    ]
    assert layouts == [
        ("storage_index", "layout-in-bounds", "decided"),
        ("storage_index", "layout-one-for-one", "decided"),
        ("storage_index", "layout-onto", "decided"),
        ("dof_array", "layout-in-bounds", "decided"),
        ("dof_array", "layout-one-for-one", "decided"),
        ("dof_array", "layout-onto", "decided"),
        ("as_dim_tags", "layout-same", "decided"),
    ]


def test_the_storage_layouts_demo_refutes_the_wrong_nesting() -> None:
    result, _ = _invoke("storage_layouts", "python")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "(1, 3) -> 8" in result.stdout
    assert "is modepy's reshape: True" in result.stdout
    assert (
        "refuted: dof_array stores every cell where dim_tags N3,N2,N1,N0 does"
    ) in result.stdout


# }}}


# {{{ hexahedra


@needs_several_indices
@needs_literal_binders
def test_the_hexahedra_demo_decides_its_layouts_and_its_programs() -> None:
    result, facts = _invoke("hexahedra", "check")
    assert result.returncode == 0, result.stdout + result.stderr
    layouts = [
        (fact["owner"], fact["kind"], fact["status"])
        for fact in facts
        if fact["kind"].startswith("layout")
    ]
    # at every order: isl decides each at n from 1 to 6, and without Lean
    # that is what the fact is, tested
    assert layouts == [
        ("dof_array", "layout-in-bounds", "tested"),
        ("dof_array", "layout-one-for-one", "tested"),
        ("dof_array", "layout-onto", "tested"),
        ("as_dim_tags", "layout-same", "tested"),
        ("row_major", "layout-in-bounds", "tested"),
        ("row_major", "layout-one-for-one", "tested"),
        ("row_major", "layout-onto", "tested"),
        ("as_row_major", "layout-same", "tested"),
    ]
    # every intermediate a program makes is read where its producer stored it
    defined = [fact for fact in facts if fact["kind"] == "definedness"]
    assert {fact["owner"] for fact in defined} == {
        "weak_grad",
        "weak_div",
        "metric_along_r",
        "contravariant_along_r",
    }
    assert all(fact["status"] == "decided" for fact in defined), defined
    # each operator states what it leaves, the dense Kronecker operators:
    # tested on its runs, and proved by Lean where Mathlib is
    posts = {
        fact["owner"]: fact for fact in facts if fact["kind"] == "postcondition"
    }
    assert set(posts) == {"weak_grad", "weak_div"}
    for post in posts.values():
        assert (post["status"], post["decided_by"]) in (
            ("tested", "native"),
            ("proved", "lean"),
        ), post
    assert posts["weak_grad"]["statement"].startswith(
        "forall x in Fin(3), e in Fin(ne), k in Fin(n), j in Fin(n), i in Fin(n). "
        "grad[x, e, k, j, i] == sum(Ft[3*x + d, k, c]*Fs[3*x + d, j, b]"
    )
    # what each call stores, once per cell, which the proof rests on
    stored = [
        (fact["owner"], fact["statement"].split(" stores")[0])
        for fact in facts
        if fact["kind"] == "single-assignment"
    ]
    assert stored == [
        ("weak_grad", call)
        for call in ("metric", "along_r", "along_s", "gradient_along_t")
    ] + [
        ("weak_div", call)
        for call in ("contravariant", "along_r", "along_s", "divergence_along_t")
    ]
    # nothing in the file is left to an assumption
    assert "assumed" not in {fact["status"] for fact in facts}


@needs_several_indices
@needs_literal_binders
def test_the_hexahedra_layouts_are_loopys_strides_and_modepys_reshape() -> None:
    # The layout facts are about loopty's reading of the dim_tags and of the
    # DOFArray. This ties each reading to the code it stands for: the strides
    # loopy itself computes for N3,N0,N1,N2, modepy's F-order reshape of an
    # element's row, and numpy's row-major u[e, k, j, i], at several orders.
    import loopy as lp

    module = _module("hexahedra")
    ne = 3
    for n in (2, module.N, 5):
        kernel = lp.make_kernel(
            f"{{[e, a, b, c]: 0 <= e < {ne} and 0 <= a, b, c < {n}}}",
            "out[e, a, b, c] = 1",
            [
                lp.GlobalArg(
                    "out", np.float64, shape=(ne, n, n, n), dim_tags="N3,N0,N1,N2"
                )
            ],
            target=lp.CTarget(),
            lang_version=(2018, 2),
        )
        tags = kernel.default_entrypoint.arg_dict["out"].dim_tags
        strides = [tag.stride for tag in tags]
        address = " + ".join(f"{stride}*c{axis}" for axis, stride in enumerate(strides))
        loopys = isl.Map(
            f"[ne] -> {{ [c0, c1, c2, c3] -> [{address}] : "
            f"0 <= c0 < ne and 0 <= c1, c2, c3 < {n} }}"
        )
        ours = module.as_dim_tags.second.at(n=n).relation()
        assert loopys.is_equal(ours), strides
        dof_array = np.arange(ne * n**3).reshape(ne, n**3)
        row_major = np.arange(ne * n**3).reshape(ne, n, n, n)
        for e, i, j, k in np.ndindex(ne, n, n, n):
            stored = module.dof_array(e, i, j, k, n=n)
            assert stored == dof_array[e].reshape(n, n, n, order="F")[i, j, k]
            assert stored == row_major[e, k, j, i] == module.row_major(e, i, j, k, n=n)


@needs_several_indices
@needs_literal_binders
def test_the_hexahedra_theorem_holds_at_exact_matrices() -> None:
    from fractions import Fraction

    module = _module("hexahedra")
    rng = np.random.default_rng(0)
    n = 3

    def exact(*shape: int) -> Any:
        return np.vectorize(Fraction)(rng.integers(-4, 5, size=shape)).tolist()

    Fr, Fs, Ft, u = exact(n, n), exact(n, n), exact(n, n), exact(n, n, n)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                claim = module.sum_factorization(
                    n=n, Fr=Fr, Fs=Fs, Ft=Ft, u=u, k=k, j=j, i=i
                )
                assert claim.holds, (k, j, i)


@needs_several_indices
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
def test_the_hexahedra_factors_are_the_stiffness_and_the_mass(order: int) -> None:
    # sum_j S^T[i, j] is the integral of the i-th basis function's derivative,
    # its value at 1 minus its value at -1; the mass integrates 1 to 2
    module = _module("hexahedra")
    stiffness_t, mass = module.factors(order)
    ends = np.zeros(order + 1)
    ends[0], ends[-1] = -1.0, 1.0
    assert np.allclose(stiffness_t.sum(axis=1), ends, atol=1e-12)
    assert np.isclose(mass.sum(), 2.0)
    assert np.allclose(mass, mass.T)
    assert np.all(np.linalg.eigvalsh(mass) > 0)


@needs_several_indices
@needs_literal_binders
def test_the_hexahedra_demo_refuses_the_deeper_fusion_and_keeps_the_windows() -> None:
    result, _ = _invoke("hexahedra", "python")
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "the passes are the dense Kronecker operators: True" in out
    assert "refused: fuse(along_r, along_s) illegal: instance along_r.S0" in out
    assert "fuse('along_r', 'along_s', loops=3) runs every dependence" in out
    assert "w kept in n cells" in out
    assert "v kept in n*n cells" in out
    assert "grad: difference" in out and "-> tested" in out
    assert out.count("all decided by isl") == 3


@needs_several_indices
@needs_literal_binders
def test_the_hexahedra_launches_are_legal_and_build_on_opencl(plain_opencl) -> None:
    module = _module("hexahedra")
    for operator in ("grad", "div"):
        launches = module.passes_on_a_device(operator)
        assert len(launches) == 3
        for schedule in launches:
            on_device = schedule.retarget("opencl")
            assert on_device.buildable == (True, ""), on_device.buildable
            statuses = {fact.status.value for fact in on_device.facts()}
            assert statuses == {"decided"}, [
                (fact.statement, fact.status.value) for fact in on_device.facts()
            ]


# }}}


# {{{ advection


@needs_several_indices
def test_the_advection_demo_decides_every_access_and_every_launch() -> None:
    result, facts = _invoke("advection", "check")
    assert result.returncode == 0, result.stdout + result.stderr
    # the face gathers: the neighbor's node by type, this element's by type
    # on the node and by isl on the element
    gathers = {
        fact["statement"]: (fact["status"], fact["decided_by"])
        for fact in facts
        if fact["owner"] == "upwind" and fact["statement"].startswith("u[")
    }
    assert gathers == {
        "u[e, inside[f, a, b, 0], inside[f, a, b, 1], inside[f, a, b, 2]] is in "
        "bounds for every instance of S0 (inside[f, a, b, 0] : Fin(n), inside[f, "
        "a, b, 1] : Fin(n), inside[f, a, b, 2] : Fin(n), by type)": ("decided", "isl"),
        "u[nbr[e, f], outside[f, a, b, 0], outside[f, a, b, 1], outside[f, a, b, "
        "2]] is in bounds by type (nbr[e, f] : Fin(ne), outside[f, a, b, 0] : "
        "Fin(n), outside[f, a, b, 1] : Fin(n), outside[f, a, b, 2] : Fin(n))": (
            "decided",
            "type",
        ),
    }
    # every array a program makes is read where its producer stored it
    defined = [fact for fact in facts if fact["kind"] == "definedness"]
    assert {fact["owner"] for fact in defined} == {
        "advection_rhs",
        "folded_rhs",
        "surface",
    }
    assert all(fact["status"] == "decided" for fact in defined), defined
    assert "assumed" not in {fact["status"] for fact in facts}


@needs_several_indices
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
def test_the_advection_tables_are_grudges_operators_with_the_inverse_mass_in(
    order: int,
) -> None:
    # D and the lift's columns against grudge's operators as dense matrices,
    # the inverse mass applied last, on random fields: the rearrangement the
    # two theorems state, at the tables' values
    module = _module("advection")
    stiffness_t, mass = module.factors(order)
    inverse = np.linalg.inv(mass)
    D, L = module.tables(order)
    n = order + 1
    rng = np.random.default_rng(order)
    w = rng.normal(size=(n, n, n))  # [t, s, r]
    weak = np.einsum("ct,bs,ar,tsr->cba", mass, mass, stiffness_t, w)
    pushed = np.einsum("kc,jb,ia,cba->kji", inverse, inverse, inverse, weak)
    assert np.allclose(pushed, np.einsum("ia,kja->kji", D, w), atol=1e-11)
    g = rng.normal(size=(n, n))  # a face's sJ * flux, [t, s]
    for side, end in enumerate((np.eye(n)[0], np.eye(n)[-1])):
        face = np.einsum("a,ct,bs,ts->cba", end, mass, mass, g)
        lifted = np.einsum("kc,jb,ia,cba->kji", inverse, inverse, inverse, face)
        assert np.allclose(lifted, np.einsum("i,kj->kji", L[side], g), atol=1e-11)


@needs_several_indices
@pytest.mark.parametrize("order", [1, 3, 6])
def test_the_gll_tables_are_the_quadratures_and_conserve(order: int) -> None:
    module = _module("advection")
    _nodes, weights = module.gll(order)
    D, L = module.tables(order, mass="gll")
    n = order + 1
    # the lift is a face node's own weight, and nothing else
    assert np.allclose(L[0], np.eye(n)[0] / weights[0])
    assert np.allclose(L[1], np.eye(n)[-1] / weights[-1])
    # W D is S^T with the GLL quadrature: its columns sum to the ends, the
    # integral of each basis function's derivative
    ends = np.zeros(n)
    ends[0], ends[-1] = -1.0, 1.0
    assert np.allclose((weights[:, None] * D).sum(axis=1), ends, atol=1e-11)
    assert np.isclose(weights.sum(), 2.0)


@needs_several_indices
def test_the_lift_onto_the_face_nodes_is_the_lift_with_the_gll_tables() -> None:
    from loopty import Arr

    module = _module("advection")
    inputs = module.example_inputs()["lift_on_faces"]
    _D, L = module.tables(mass="gll")
    dense = {
        name: Arr.from_numpy(inputs[name].numpy().copy())
        for name in ("invJ", "flux", "rhs")
    }
    module.lift(Arr.from_numpy(L), **dense)
    module.lift_on_faces(**inputs)
    expected = dense["rhs"].numpy()
    gap = np.abs(inputs["rhs"].numpy() - expected).max()
    assert gap <= 1e-14 * np.abs(expected).max()


@needs_several_indices
def test_the_face_maps_are_modepys_faces_and_the_box_is_periodic() -> None:
    module = _module("advection")
    n = module.N
    inside, outside = module.face_maps()
    # modepy's faces of the hypercube: t, r, s, each at -1 and then at +1
    assert module.FACES == ((2, -1), (2, 1), (0, -1), (0, 1), (1, -1), (1, 1))
    for f, (axis, side) in enumerate(module.FACES):
        position = 2 - axis  # (k, j, i)
        here, there = (0, n - 1) if side < 0 else (n - 1, 0)
        assert (inside[f, ..., position] == here).all()
        assert (outside[f, ..., position] == there).all()
        others = [p for p in range(3) if p != position]
        assert (inside[f][..., others] == outside[f][..., others]).all()
        # the face's nodes, each once
        assert len({tuple(node) for node in inside[f].reshape(-1, 3)}) == n * n
    nbr = module.box(3)
    for f in range(6):
        across = nbr[:, f]
        assert (nbr[across, f ^ 1] == np.arange(27)).all()


@needs_several_indices
def test_the_advection_right_hand_side_conserves_on_the_periodic_box() -> None:
    # integrated with the mass, the right-hand side of a periodic problem is
    # zero: what leaves an element through a face enters its neighbor
    module = _module("advection")
    data = module.inputs(elements=3)
    module.advection_rhs(**data)
    _nodes, weights = module.gll(module.ORDER)
    J = 1 / data["invJ"].numpy()
    rhs = data["rhs"].numpy()
    total = np.einsum("ekji,k,j,i->", J * rhs, weights, weights, weights)
    scale = np.einsum("ekji,k,j,i->", J * np.abs(rhs), weights, weights, weights)
    assert abs(total) <= 1e-13 * scale


@needs_several_indices
def test_the_advection_demo_refuses_the_deeper_fusion_and_runs_every_split() -> None:
    result, _ = _invoke("advection", "python")
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "inverse mass and all, to 1e-13 of the largest value: True" in out
    assert "M^-1 M is the identity: True" in out
    assert "the transpose of the face gather on these maps: True" in out
    assert "refused: fuse(upwind, volume) illegal: instance upwind.S0" in out
    assert "fuse('upwind', 'volume', loops=1) runs every dependence" in out
    assert "flux kept in lambda" in out and "-> tested" in out
    # libParanumal's lift onto the face nodes races on an edge without its
    # barriers, and is legal once the updates are atomic
    assert "refused: tag(e='g.0', a='l.1', b='l.0') illegal: S" in out
    assert "both update rhs[" in out and "atomic('rhs')" in out
    assert "accepted: atomic('rhs').tag(" in out
    for split in _module("advection").SPLITS:
        assert f"  {split}: " in out, split
    assert "all decided" in out and "refuted" not in out


@needs_several_indices
def test_the_advection_launches_are_legal_and_build_on_opencl(plain_opencl) -> None:
    module = _module("advection")
    for split in module.SPLITS:
        for schedule in module.launches(split):
            on_device = schedule.retarget("opencl")
            assert on_device.buildable == (True, ""), (split, on_device.buildable)
            statuses = {fact.status.value for fact in on_device.facts()}
            assert statuses == {"decided"}, [
                (fact.statement, fact.status.value) for fact in on_device.facts()
            ]


# }}}


# {{{ travel


def test_the_travel_demo_decides_its_requirements_or_checks_them() -> None:
    _result, facts = _invoke("travel", "check")
    requirements = {
        (fact["owner"], fact["provenance"]["requirement"]): fact
        for fact in facts
        if fact["kind"] == "requirement"
    }
    assert set(requirements) == {
        ("permuted", "element"),
        ("checked", "element"),
        ("through", "layout"),
        ("flat", "layout"),
        ("flat", "element"),
    }
    # gather's requirement after number is number's postcondition, and the
    # rows read through a scan's offsets are the scan's.
    for key in (("permuted", "element"), ("through", "layout"), ("flat", "layout")):
        requirement = requirements[key]
        assert requirement["status"] == "decided"
        assert requirement["effective"] == "tested"
        restated, faithful = requirement["rests_on"]
        assert restated.startswith(f"postcondition-in-scope:travel.{key[0]}@")
        assert faithful.startswith("trace-faithful:travel.")
    # Nothing says what number_quiet writes, or that the scan's offsets stay
    # below nnz, so the program checks them.
    for key in (("checked", "element"), ("flat", "element")):
        assert requirements[key]["status"] == "assumed"
        assert "checked" in requirements[key]["provenance"]
    post = [
        fact
        for fact in facts
        if fact["kind"] == "postcondition"
        and fact["owner"] in ("number", "scan", "scan_flat")
    ]
    # decided from the term, and tested on the native runs all the same (#113)
    assert [fact["status"] for fact in post] == ["decided", "decided", "decided"]
    assert [fact["provenance"]["native"] for fact in post] == ["tested"] * 3


def test_the_travel_demo_decides_a_flat_access_where_the_scan_was_called() -> None:
    _result, facts = _invoke("travel", "check")
    (alone,) = [
        fact
        for fact in facts
        if fact["owner"] == "weigh"
        and fact["kind"] == "in-bounds"
        and fact["statement"].startswith("val[off[r] + j]")
    ]
    assert alone["status"] == "assumed"
    (scoped,) = [
        fact
        for fact in facts
        if fact["owner"] == "flat" and fact["kind"] == "in-bounds"
    ]
    assert scoped["status"] == "decided"
    assert scoped["decided_by"] == "isl"
    # Under the scan's postcondition, which is tested, and the element type of
    # off, which the program checks: that is the assumption it is decided under.
    (element,) = [
        fact
        for fact in facts
        if fact["owner"] == "flat"
        and fact["kind"] == "requirement"
        and fact["provenance"]["requirement"] == "element"
    ]
    assert scoped["under"] == [element["id"]]


def test_the_travel_demo_prints_the_check_it_compiles() -> None:
    result, _ = _invoke("travel", "python")
    assert "checked when it runs" in result.stdout
    assert "gather_perm_ok[0] = (int64_t) (1)" in result.stdout


# }}}


# {{{ calls.py


def test_the_calls_demo_owes_what_each_call_owes_and_compiles_its_rules() -> None:
    """A row in bounds, a gather by type, the postcondition after each call."""
    _result, facts = _invoke("calls", "check")
    statements = {fact["statement"]: fact for fact in facts}
    row = next(s for s in statements if s.startswith("A[i], the row passed to dot"))
    assert statements[row]["status"] == "decided"
    gathers = [s for s in statements if s.startswith("the cells X[cell[c]] picks")]
    assert [statements[s]["decided_by"] for s in gathers] == ["type"]
    restated = [f for f in facts if f["kind"] == "postcondition-in-scope"]
    assert [(f["owner"], f["decided_by"]) for f in restated] == [
        ("mass_matrices", "call")
    ]
    result, facts = _invoke("calls", "run")
    assert "integrate: takes the function f" in result.stdout
    owners = {f["owner"] for f in facts if f["kind"] == "agreement"}
    assert owners == {"dot", "matvec", "csr", "element_mass", "mass_matrices"}


# }}}


# {{{ assembly


def test_the_assembly_demo_names_the_node_two_cells_share() -> None:
    result, _ = _invoke("assembly", "python")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "PyOP2's two triangles: the cells' matrices sum to expected_matrix: True"
    ) in result.stdout
    assert "and y is that matrix times u: True" in result.stdout
    assert "y is the closed form: True" in result.stdout
    assert "W3 and FE0 are the rule to 3.3e-16: True" in result.stdout
    assert (
        "mass_ffc's postcondition, its matrix |J| (1 + delta_jk) / 24 to the "
        "approx tolerance -> tested"
    ) in result.stdout
    assert (
        "IllegalCast: tag(c_outer='g.0', c_inner='l.0') illegal: S1[c=0, a=2] and "
        "S1[c=1, a=1] both update y[3] on example_inputs() (nc=2, nv=4), where "
        "cell[0, 2] = 3 and cell[1, 1] = 3"
    ) in result.stdout
    accepted = result.stdout.split("accepted: ")[1]
    assert "atomic updates of y run at the same time after tag(" in accepted
    assert "atomic('y') renames" not in accepted
    assert "-> tested" in result.stdout


def test_the_assembly_owes_its_sum_and_not_distinct_writes_of_y() -> None:
    _result, facts = _invoke("assembly", "check")
    disjoint = {
        (fact["owner"], fact["provenance"]["array"])
        for fact in facts
        if fact["kind"] == "disjoint-writes"
    }
    assert disjoint == {("mass_ffc", "A"), ("mass_action", "Ae")}
    (update,) = [
        fact
        for fact in facts
        # the kernel's own; the reassociation a schedule records has no op
        if fact["kind"] == "exactness"
        and fact["provenance"].get("op") == "update"
        and fact["owner"] == "mass_action"
        and "over c, a" in fact["statement"]
    ]
    assert update["statement"] == (
        "the accumulation into y[cell[c, a]] over c, a is approx"
    )
    assert (update["status"], update["decided_by"]) == ("decided", "type")


def test_the_element_mass_is_mass_ffcs_postcondition_to_the_approx_tolerance() -> None:
    """``|J| (1 + delta_jk) / 24`` from FFC's 15-digit tables: to round-off (#254).

    Tested on the runs at the approx tolerance; with Mathlib, proved over the
    reals from what the kernel stores, resting on that and on its trace.
    """
    _result, facts = _invoke("assembly", "check")
    owned = [fact for fact in facts if fact["owner"] == "mass_ffc"]
    (post,) = [fact for fact in owned if fact["kind"] == "postcondition"]
    assert post["statement"].startswith(
        "(A[0, 0] == abs((x[1, 0] - x[0, 0])*(x[2, 1] - x[0, 1]) - (x[2, 0] - "
        "x[0, 0])*(x[1, 1] - x[0, 1])) / 12 and "
    )
    assert post["statement"].endswith(") / 24), to the approx tolerance")
    assert (post["status"], post["decided_by"]) in (
        ("tested", "native"),
        ("proved", "lean"),
    )
    (stored,) = [fact for fact in owned if fact["kind"] == "single-assignment"]
    (faithful,) = [fact for fact in owned if fact["kind"] == "trace-faithful"]
    assert set(post["rests_on"]) >= {stored["id"], faithful["id"]}


def test_the_assembly_calls_mass_ffc_per_cell_and_has_its_element_mass() -> None:
    """The D3 form: the element kernel inlined per cell, its postcondition carried."""
    _result, facts = _invoke("assembly", "check")
    owned = [fact for fact in facts if fact["owner"] == "mass_action"]
    (after,) = [
        fact for fact in owned if fact["statement"].startswith("after each call")
    ]
    assert "mass_ffc(coords[cell[c]], Ae[c])" in after["statement"]
    assert "Ae[c, 0, 0] == abs((coords[cell[c, 1], 0] - " in after["statement"]
    assert after["statement"].endswith(", to the approx tolerance")
    assert after["status"] == "decided"
    # every read of the corners through the cell map is decided, none assumed
    corners = [fact for fact in owned if fact["statement"].startswith("coords[")]
    assert len(corners) == 6
    assert {fact["status"] for fact in corners} == {"decided"}
    # nothing is assumed but the schedule's assumption, nc a multiple of four
    assumed = [fact for fact in facts if fact["status"] == "assumed"]
    assert [fact["kind"] for fact in assumed] in ([], ["assumption"])


def test_the_assembly_theorems_are_in_its_ledger() -> None:
    """The rule, the element's mass and the scatter's total: proved with Mathlib."""
    _result, facts = _invoke("assembly", "check")
    claims = {
        fact["owner"]: fact["status"] for fact in facts if fact["kind"] == "theorem"
    }
    assert set(claims) == {"exact_to_degree_two", "element_mass", "conservation"}
    assert set(claims.values()) <= {"tested", "decided", "proved"}


# }}}


# {{{ l2p


@needs_several_indices
def test_the_l2p_demo_applies_its_theorem_to_what_reconstruct_stores() -> None:
    """The postcondition decided from the stores, and the theorem applied to it."""
    _result, facts = _invoke("l2p", "check")
    owned = {(fact["owner"], fact["kind"]): fact for fact in facts}
    post = owned[("reconstruct", "postcondition")]
    assert (post["status"], post["decided_by"]) == ("decided", "isl")
    applied = [fact for fact in facts if fact["kind"] == "by-theorem"]
    assert [(fact["owner"], fact["decided_by"]) for fact in applied] == [
        ("reconstruct", "theorem")
    ]
    assert applied[0]["statement"].startswith(
        "by on_every_triangle, for every D with pde and stored: "
    )
    assert owned[("reconstruct", "single-assignment")]["status"] == "decided"
    # the table the theorem's hypothesis is about holds sumpy's coefficients
    # in sumpy's storage order, decided from what unpack stores
    stored = owned[("unpack", "postcondition")]
    assert (stored["status"], stored["decided_by"]) == ("decided", "isl")
    assert "C[i, a, b] == coeffs[i, 2*(a + b) - 1 + a]" in stored["statement"]
    # sumpy's storage index is one for one onto its 2P + 1 addresses
    layout = [fact for fact in facts if fact["owner"] == "storage_index"]
    assert len(layout) == 3
    assert {(fact["status"], fact["decided_by"]) for fact in layout} == {
        ("decided", "isl")
    }
    # what each call reads, the call before it stored
    defined = [
        fact["statement"]
        for fact in facts
        if fact["owner"] == "l2p" and fact["kind"] == "definedness"
    ]
    assert defined == [
        "every cell of C that reconstruct reads, unpack stored before it",
        "every cell of R that evaluate reads, reconstruct stored before it",
        # and what each window keeps, the zeros the program made it with gone
        "for contract('R'), every cell of R that l2p reads, reconstruct.S0 or "
        "reconstruct.S1 has stored by the time it is read",
        "for contract('C'), every cell of C that l2p reads, unpack.S0 or unpack.S1 "
        "has stored by the time it is read",
    ]


@needs_several_indices
def test_the_l2p_demo_agrees_with_sumpys_own_kernel() -> None:
    """The row the recording of sumpy's E2PFromSingleBox gives, tested."""
    _result, facts = _invoke("l2p", "check")
    (row,) = [fact for fact in facts if fact["kind"] == "upstream"]
    assert (row["status"], row["decided_by"], row["owner"]) == (
        "tested",
        "upstream",
        "l2p",
    )
    assert row["statement"] == (
        "the native run of l2p agrees with sumpy_l2p in pot, on its example inputs"
    )
    assert row["provenance"]["uncompared"] == []
    detail = row["provenance"]["outputs"]["pot"]
    assert detail["agree"] and detail["exactness"] == "approx"
    # round-off, far inside the class's allowance
    assert detail["difference"] < 1e-14 * detail["largest"]


@needs_several_indices
def test_the_l2p_demo_refuses_the_interchange_and_fuses_box_by_box() -> None:
    result, _ = _invoke("l2p", "python")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "largest error against the charges' potential: " in result.stdout
    refusals = [
        line.split("refused: ")[1]
        for line in result.stdout.splitlines()
        if line.startswith("refused: ")
    ]
    assert refusals[0].startswith("interchange(b, a) illegal: instance S0[i=")
    assert " writes R[" in refusals[0] and " read by S1[i=" in refusals[0]
    module = _module("l2p")
    steps = [fact.statement for fact in module.fused.facts()]
    assert steps == [
        "fuse(unpack, reconstruct, loops=1) renames the instances of l2p one for one",
        "the order after fuse(unpack, reconstruct, loops=1) runs every dependence "
        "of l2p forward",
        "fuse(reconstruct, evaluate, loops=1) renames the instances of l2p one for "
        "one",
        "the order after fuse(reconstruct, evaluate, loops=1) runs every dependence "
        "of l2p forward",
    ]
    assert {fact.status.value for fact in module.fused.facts()} == {"decided"}
    assert "pot: difference 0 within" in result.stdout
    # sumpy's own GPU schedule, each box a work group, with the file's names:
    # the tag's two facts, the fusions' being fused's, above
    device = result.stdout.split("on a GPU, each box a work group, as sumpy's: ")[1]
    assert device.splitlines()[1:] == [
        "  decided  isl  tag(i='g.0') renames the instances of l2p one for one",
        "  decided  isl  the order after tag(i='g.0') runs every dependence of l2p "
        "forward, within one work item",
    ]


@needs_several_indices
def test_the_l2p_demo_keeps_one_triangle_per_box_and_refuses_one_row() -> None:
    # One box's triangle of R, over a domain (#244), stored by a copy and by
    # the recurrence that reads it (#213): the window is found along the
    # fused order, and one row of it is refused with the read it would break.
    module = _module("l2p")
    windows = module.contracted.windows
    assert {name: repr(window.placement) for name, window in windows.items()} == {
        "R": "lambda i, a, b: (a, b)",
        "C": "lambda i, a, b: (a, b)",
    }
    assert windows["R"].extents == (11, 11) and windows["C"].extents == (2, 11)
    added = module.contracted.facts()[len(module.fused.facts()) :]
    assert [(fact.kind, fact.status.value) for fact in added] == [
        ("definedness", "decided"),
        ("no-clobber", "decided"),
    ] * 2
    assert [fact.statement for fact in added][:2] == [
        "for contract('R'), every cell of R that l2p reads, reconstruct.S0 or "
        "reconstruct.S1 has stored by the time it is read",
        "no read of R sees its slot overwritten after contract('R', lambda i, a, "
        "b: (a, b)): R kept in 121 cells, R[i, a, b] in slot (a, b)",
    ]
    result, _ = _invoke("l2p", "python")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "refused: contract('R', lambda i, a, b: b) illegal: instance "
        "reconstruct.S1[i_0=0, a_0=9, b_0=0] reads R[0, 7, 2], which "
        "reconstruct.S1[i_0=0, a_0=7, b_0=2] stored in slot 2, after "
        "reconstruct.S1[i_0=0, a_0=8, b_0=2] stored R[0, 8, 2] there (at nb=1, "
        "isl's choice); contract('R') finds the smallest window that keeps every "
        "read, lambda i, a, b: (a, b)\n"
    ) in result.stdout
    assert (
        "on a device, each box a work group keeping its windows: tag(i='g.0'): 2 "
        "facts, all decided by isl\n"
    ) in result.stdout


# {{{ the QBX layer potential


@needs_parts
def test_the_qbx_demo_states_the_fact_sumpy_silences_for_both_layers() -> None:
    # sumpy silences loopy's write_race(write_lpot*) on its LayerPotential;
    # the fact in its place is that each target's cell is written once.
    _result, facts = _invoke("layer_potential", "check")
    for owner in ("single_layer", "double_layer"):
        (writes,) = [
            fact
            for fact in facts
            if fact["owner"] == owner and fact["kind"] == "disjoint-writes"
        ]
        assert writes["statement"] == (
            "distinct instances of S0 write distinct cells of out"
        )
        assert writes["status"] in ("decided", "proved"), writes
        assert writes["decided_by"] in ("isl", "lean"), writes


@needs_parts
def test_the_qbx_demo_agrees_with_what_sumpy_returned_at_the_pinned_commit() -> None:
    _result, facts = _invoke("layer_potential", "check")
    rows = {fact["owner"]: fact for fact in facts if fact["kind"] == "upstream"}
    assert set(rows) == {"single_layer", "double_layer"}
    for owner, row in rows.items():
        assert (row["status"], row["decided_by"]) == ("tested", "upstream"), row
        # round-off of two orders of summation and two arithmetics, real and
        # complex, far inside the class's allowance: the row states its own
        assert row["statement"] == (
            f"the native run of {owner} agrees with sumpy_{owner} in out within "
            "1e-12 of the largest value, on its example inputs"
        )
        assert row["provenance"]["uncompared"] == []
        detail = row["provenance"]["outputs"]["out"]
        assert detail["agree"] and detail["exactness"] == "approx", detail
        assert detail["within"] == 1e-12
        assert detail["tolerance"] == 1e-12 * detail["largest"], detail
        assert detail["difference"] <= detail["tolerance"], detail
    # the recording is of the sumpy the docstring cites, on loopy main
    module = _module("layer_potential")
    recorded = json.loads(module.RECORDED.read_text(encoding="utf-8"))
    assert recorded["sumpy"] == "13a48a0260eec8cf7ac67f1faca701205d9096ae"
    assert recorded["sumpy"] in module.__doc__
    assert recorded["versions"]["loopy"].startswith("24a2da6")
    size = module.example_inputs()["single_layer"]["z"].numpy().size
    assert len(recorded["single_layer"]) == len(recorded["double_layer"]) == size


@needs_parts
def test_the_qbx_demo_measures_pytentials_identity_and_the_jumps() -> None:
    _result, facts = _invoke("layer_potential", "check")
    rows = {fact["owner"]: fact for fact in facts if fact["kind"] == "tested"}
    assert set(rows) == {"green_identity", "jump_S", "jump_D"}
    for fact in rows.values():
        assert (fact["status"], fact["decided_by"]) == ("tested", "run"), fact
        order = float(fact["provenance"]["measured"].rsplit("order ", 1)[1])
        # pytential's test asks for the QBX order less 1.6
        assert order > 5 - 1.6, fact


@needs_parts
def test_the_line_taylor_theorem_is_in_the_qbx_ledger_and_not_refuted() -> None:
    _result, facts = _invoke("layer_potential", "check")
    statuses = {
        fact["status"] for fact in facts if fact["owner"] == "line_taylor"
    }
    assert statuses and "refuted" not in statuses
    # without Mathlib the tester draws no table over Nat, and says so
    assert statuses <= {"assumed", "tested", "proved"}


@needs_parts
def test_the_qbx_ledger_says_single_layer_adds_line_taylors_terms() -> None:
    """The theorem and the kernel in one row: its terms are line_taylor's c(n) (#252).

    The kernel's ratio is by the conjugate, and ``by_the_conjugate`` says it
    is the theorem's ``-beta / alpha``: a theorem of its own, tested by the
    property tester and, with Mathlib, proved.
    """
    _result, facts = _invoke("layer_potential", "check")
    (identity,) = [
        fact
        for fact in facts
        if fact["owner"] == "by_the_conjugate" and fact["kind"] == "theorem"
    ]
    assert identity["status"] in ("tested", "proved"), identity
    (row,) = [fact for fact in facts if fact["kind"] == "by-theorem"]
    assert row["owner"] == "single_layer"
    assert row["statement"] == (
        "single_layer adds line_taylor's c(n) for n in 1..5, for every c with "
        "away, first and ode, read over the complex numbers"
    )
    assert row["status"] in ("decided", "proved"), row
    assert row["decided_by"] in ("theorem", "lean"), row
    provenance = row["provenance"]
    assert provenance["bindings"] == "alpha = c[t] - y[s], beta = z[t] - c[t]"
    assert provenance["orders"] == [1, 2, 3, 4, 5]
    module = _module("layer_potential")
    assert set(row["rests_on"]) >= {
        module.line_taylor.fact_id,
        module.by_the_conjugate.fact_id,
    }
    # where the statement is that adds them
    line = int(row["where"].rsplit(":", 1)[1])
    source = _path("layer_potential").read_text(encoding="utf-8").splitlines()
    assert source[line - 1].strip() == "out[t] = SCALE * reduce_sum("
    assert "single(c[t] - y[s], z[t] - c[t])" in source[line]


@needs_parts
def test_the_kernels_ratio_is_the_theorems() -> None:
    # The kernels add coefficient(ratio, k) with the ratio multiplied out by
    # the conjugate; the theorem's is -beta / alpha. They agree to rounding,
    # and the series is log|alpha + beta| to the order's truncation.
    module = _module("layer_potential")
    rng = np.random.default_rng(5)
    for alpha, beta in zip(
        rng.normal(size=20) + 1j * rng.normal(size=20),
        0.3 * (rng.normal(size=20) + 1j * rng.normal(size=20)),
        strict=True,
    ):
        ratio = module.ratio(alpha, beta)
        assert abs(ratio - (-beta / alpha)) <= 1e-15 * abs(beta / alpha) * 4
        closed = sum(
            module.coefficient(-beta / alpha, k) for k in range(1, module.ORDER + 1)
        )
        series = module.single(alpha, beta)
        assert abs(series - (np.log(abs(alpha)) + closed.real)) <= 1e-14
        truncation = abs(beta / alpha) ** (module.ORDER + 1) / (1 - abs(beta / alpha))
        if abs(beta / alpha) < 0.5:
            assert abs(series - np.log(abs(alpha + beta))) <= truncation


@needs_parts
def test_sumpys_tile_is_legal_and_c_cannot_build_its_axes() -> None:
    module = _module("layer_potential")
    tile = module.sumpy_tile("c")
    casts = [fact for fact in tile.facts() if fact.kind != "buildable"]
    assert len(casts) == 8
    assert all(fact.status.value == "decided" for fact in casts)
    buildable, reason = tile.buildable
    assert not buildable
    assert "puts a loop on a hardware axis, and the C target has none" in reason


@needs_parts
def test_the_qbx_demo_prints_the_tile_the_fact_and_the_rows() -> None:
    result, _ = _invoke("layer_potential", "python")
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert (
        "decided  isl  prioritize(s_outer, t_inner) renames the instances of "
        "single_layer one for one"
    ) in out
    assert "decided  isl  distinct instances of S0 write distinct cells of out" in out
    # the tile is shown as sumpy writes it, without the target its casts were
    # decided on, which cannot build its hardware axes
    assert (
        "sumpy's tile, get_optimized_kernel(is_cpu=True): split(t, 16)"
        ".tag(t_outer='g.0', t_inner='l.0').split(s, 256)"
        ".prioritize(s_outer, t_inner)"
    ) in out
    lines = [line for line in out.splitlines() if "sumpy's own kernel" in line]
    assert [line.split(":")[0].strip() for line in lines] == [
        "single_layer",
        "double_layer",
    ]
    for line in lines:
        assert line.endswith(", within 1e-12 of the largest value -> tested"), line
    assert (
        "  single_layer adds line_taylor's c(n) for n in 1..5, for every c with "
        "away, first and ode, read over the complex numbers -> decided"
    ) in out.splitlines()
    for name in ("green_identity", "jump_S", "jump_D"):
        (line,) = [line for line in out.splitlines() if line.strip().startswith(name)]
        assert line.endswith("-> tested"), line


# }}}
