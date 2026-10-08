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
)

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


@pytest.mark.parametrize("name", NAMES)
def test_a_demo_runs_under_plain_python(name: str) -> None:
    result, _ = _invoke(name, "python")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip(), "a demo that prints nothing demonstrates nothing"


@pytest.mark.parametrize("name", NAMES)
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


@pytest.mark.parametrize("name", NAMES)
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


@pytest.mark.parametrize("name", NAMES)
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
    exactness = [fact for fact in facts if fact["kind"] == "exactness"]
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
    # Nobody can decide the recurrence from the term alone; whether it is
    # assumed, property-tested or proved depends on which oracles are installed,
    # and all three are honest answers.
    assert post and set(post) <= {"assumed", "tested", "proved"}, post

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
    assert "double f[n];" in result.stdout
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


def test_the_fusion_demo_decides_the_fusion_and_the_substitution() -> None:
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
    ]
    agreements = [
        fact
        for fact in facts
        if fact["owner"] == "burgers_rhs" and fact["kind"] == "agreement"
    ]
    assert [fact["status"] for fact in agreements] == ["tested", "tested"]


def test_the_fusion_demo_prints_the_refusal_and_two_kernels() -> None:
    result, _ = _invoke("fusion", "python")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "refused: fuse(flux, divergence) illegal: instance flux.S0[j=2] writes "
        "f[2] read by divergence.S0[i=1] scheduled earlier (at n=16, as hinted); "
        "fuse('flux', 'divergence', shift=1) runs every dependence between them "
        "forward"
    ) in result.stdout
    fused, substituted = result.stdout.split("accepted: ")[1:]
    assert "double f[n];" in fused
    assert "double f[n];" not in substituted
    assert "0.5 * u[j] * u[j]" in substituted
    assert fused.count("for (") == 2
    assert substituted.count("for (") == 1
    assert result.stdout.count("-> tested") == 2


def test_the_fusion_demo_states_the_definedness_of_its_intermediate() -> None:
    _result, facts = _invoke("fusion", "check")
    (fact,) = [fact for fact in facts if fact["kind"] == "definedness"]
    assert (fact["status"], fact["decided_by"]) == ("decided", "isl")
    assert fact["owner"] == "burgers_rhs"


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
    assert [fact["status"] for fact in post] == ["tested", "tested", "tested"]


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
