"""The ``loopty`` command: run a file, and delegate checking to lanky.

``loopty run FILE`` is the third of the design's three commands, and what it
has to get right is the join: import a file, find what is runnable in it, find
the inputs the file offers, run each schedule, and put the comparison in the
ledger with everything else. The fixture below is written out as a file rather
than built in process because that is what the command actually does.
"""

from __future__ import annotations

import json

import pytest

from loopty.cli import RunVerb, build_parser, main

pytest.importorskip("loopy")

FIXTURE = '''
"""A file of the shape ``loopty run`` expects."""

import islpy as isl
import numpy as np
import pymbolic.primitives as prim

from loopty.schedule import Schedule
from loopty.term import Access, ArrType, Stmt, Term

V = prim.Variable


def scale_term() -> Term:
    domain = isl.Set("[n] -> { [i] : 0 <= i < n }")
    stmt = Stmt(
        id="S0",
        inames=("i",),
        domain=domain,
        assignee=Access("y", (V("i"),)),
        expr=2.0 * prim.Subscript(V("x"), (V("i"),)),
        kind="assign",
        guard=None,
        where="fixture.py:1",
    )
    real = np.dtype(np.float64)
    return Term(
        name="scale",
        params=(
            ("x", ArrType(axes=(V("n"),), dtype=real, ragged=(False,))),
            ("y", ArrType(axes=(V("n"),), dtype=real, ragged=(False,))),
        ),
        sizes=("n",),
        stmts=(stmt,),
        post=None,
    )


class Scale:
    """A stand-in for a decorated kernel: callable, and carrying a term."""

    term = scale_term()
    __name__ = "scale"

    def __call__(self, x, y):
        y[...] = 2.0 * x

    def trace(self):
        return self.term


scale = Scale()
sched = Schedule(scale).split("i", 4)


def example_inputs():
    return {"x": np.arange(8, dtype=np.float64), "y": np.zeros(8)}
'''


def write_fixture(tmp_path, body: str = FIXTURE):
    path = tmp_path / "fixture.py"
    path.write_text(body, encoding="utf-8")
    return path


def refutation_block(out: str, prefix: str) -> tuple[list[str], int]:
    """The lines of ``out``, and the index of the one ``REFUTED {prefix}...`` line."""
    lines = out.splitlines()
    (header,) = [
        k for k, line in enumerate(lines) if line.startswith(f"REFUTED {prefix}")
    ]
    return lines, header


def test_version_is_printed_without_a_verb(capsys) -> None:
    assert main(["--version"]) == 0
    assert "loopty" in capsys.readouterr().out


def test_no_verb_prints_help(capsys) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out


def test_run_executes_every_schedule_and_reports_agreement(tmp_path, capsys) -> None:
    path = write_fixture(tmp_path)
    code = main(["run", str(path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "scale" in out
    assert "difference" in out
    assert "tested" in out


def test_run_writes_the_ledger_as_json(tmp_path) -> None:
    path = write_fixture(tmp_path)
    out_path = tmp_path / "ledger.json"
    assert main(["run", str(path), "--json", str(out_path)]) == 0
    facts = json.loads(out_path.read_text(encoding="utf-8"))
    kinds = {fact["kind"] for fact in facts}
    assert {"bijective", "monotone", "agreement"} <= kinds


def test_emit_code_prints_the_generated_source(tmp_path, capsys) -> None:
    path = write_fixture(tmp_path)
    assert main(["run", str(path), "--emit-code"]) == 0
    assert "void scale" in capsys.readouterr().out


def test_a_file_with_no_inputs_says_so_rather_than_guessing(tmp_path, capsys) -> None:
    body = FIXTURE.replace("def example_inputs()", "def unused_inputs()")
    path = write_fixture(tmp_path, body)
    assert main(["run", str(path)]) == 0
    assert "no example inputs" in capsys.readouterr().out


def test_inputs_may_be_recorded_on_the_schedule(tmp_path, capsys) -> None:
    body = FIXTURE.replace(
        "sched = Schedule(scale).split(\"i\", 4)",
        "sched = Schedule(scale).split(\"i\", 4).example("
        "x=np.arange(8, dtype=np.float64), y=np.zeros(8))",
    ).replace("def example_inputs()", "def unused_inputs()")
    path = write_fixture(tmp_path, body)
    assert main(["run", str(path)]) == 0
    assert "difference" in capsys.readouterr().out


def test_the_run_verb_is_shaped_the_way_lanky_asks(tmp_path) -> None:
    verb = RunVerb()
    assert verb.name == "run"
    assert verb.help
    parser = build_parser()
    args = parser.parse_args(["run", str(write_fixture(tmp_path))])
    # ``--target`` defaults to nothing: each schedule keeps the target it was
    # written for, and only an explicit flag retargets it.
    assert args.target is None
    assert parser.parse_args(["run", "f.py", "--target", "opencl"]).target == "opencl"
    assert hasattr(verb, "run") and callable(verb.run)
    # lanky calls ``run``; the console script used to call the verb itself.
    assert verb.__call__ == verb.run


def test_check_is_an_alias_of_lanky_check(tmp_path, capsys) -> None:
    path = tmp_path / "empty.py"
    path.write_text("x = 1\n", encoding="utf-8")
    assert main(["check", str(path)]) == 0
    # Nothing in the file claims anything, so lanky prints an empty ledger; the
    # point of the test is that the verb reaches lanky at all.
    assert "ledger" in capsys.readouterr().out.lower()


DECORATED = '''
"""A file written the way a user writes one: a decorated kernel and a schedule."""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, kernel
from loopty.schedule import Schedule


@kernel
def scale(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
    """Twice x, into y."""
    for i in y.dom:
        y[i] = 2.0 * x[i]


sched = Schedule(scale).split("i", 4)


def example_inputs():
    return {"x": Arr.from_numpy(np.arange(8, dtype=np.float64)), "y": Arr.zeros(8)}
'''


def test_run_on_a_decorated_kernel_file(tmp_path, capsys) -> None:
    # The end of the design's third command: a file a person would write,
    # imported, traced, lowered, run, and compared with its own Python body.
    try:
        from loopty import kernel  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the tracer's state
        pytest.skip("the kernel decorator is not landed yet")
    path = write_fixture(tmp_path, DECORATED)
    code = main(["run", str(path)])
    out = capsys.readouterr().out
    if code != 0:  # pragma: no cover - depends on the tracer's state
        pytest.skip(f"tracing is not ready for this file yet:\n{out}")
    assert "scale" in out
    assert "tested" in out


def test_target_retargets_every_schedule_rather_than_ignoring_the_flag(
    tmp_path, capsys
) -> None:
    """``--target opencl`` must not quietly run a C-pinned schedule on C.

    The file pins its schedule to the default target. Asking for a device run
    used to leave that schedule exactly where it was and still report success,
    so the ledger claimed a device run that never happened. Now the schedule is
    rebuilt for the target, which re-checks every cast; on a machine with no
    pyopencl that rebuild fails, and the failure is reported by name and turns
    into a non-zero exit code.
    """
    path = write_fixture(tmp_path)
    code = main(["run", str(path), "--target", "opencl"])
    out = capsys.readouterr().out
    assert code == 1
    assert "cannot retarget scale" in out or "cannot schedule scale" in out
    # And it says why, rather than "something went wrong".
    assert "pyopencl" in out or "opencl" in out
    # Nothing was run on the C target under the name of the device target.
    assert "difference" not in out


def test_without_target_a_schedule_keeps_the_one_it_was_written_for(
    tmp_path, capsys
) -> None:
    path = write_fixture(tmp_path)
    assert main(["run", str(path)]) == 0
    assert "target='c'" in capsys.readouterr().out


CARRIED = '''
"""A kernel that carries a running sum through a Python name."""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, kernel


@kernel
def running_sum(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):
    s = 0.0
    for i in x.dom:
        s = s + x[i]
    y[0] = s


def example_inputs():
    return {"x": Arr.from_numpy(np.arange(4.0)), "y": Arr.zeros(1)}
'''


def test_check_refuses_a_loop_carried_name_and_prints_the_fix(
    tmp_path, capsys
) -> None:
    """``lanky check`` exits 1 with the message, not a ledger about a wrong term.

    The body used to trace to ``y[0] = 0.0 + x[i]`` and the ledger decided
    facts about that. Now tracing refuses it, the kernel's one fact is the
    refuted ``trace`` fact, and the message that names the fix is printed under
    it rather than left in the JSON: the line right under ``REFUTED`` is the
    error, with no counterexample line and no ``no witness recorded``.
    """
    from lanky.cli import main as lanky_main

    path = write_fixture(tmp_path, CARRIED)
    out_path = tmp_path / "ledger.json"
    code = lanky_main(["check", str(path), "--json", str(out_path)])
    out = capsys.readouterr().out
    assert code == 1
    lines, header = refutation_block(out, "running_sum at ")
    assert lines[header + 1].startswith("  TraceError: ")
    assert "counterexample" not in out
    assert "no witness recorded" not in out
    assert "carries 's'" in out
    assert "reduce_sum" in out
    assert "indexed cell" in out
    facts = json.loads(out_path.read_text(encoding="utf-8"))
    assert [(fact["kind"], fact["status"]) for fact in facts] == [
        ("trace", "refuted")
    ]
    assert "counterexample" not in facts[0]["provenance"]
    assert facts[0]["provenance"]["reason"].startswith("TraceError: ")


OUT_OF_BOUNDS = '''
"""A read one past the end, and a loop that writes one cell over and over."""

from __future__ import annotations

from lanky.prelude import Real

from loopty import Arr, Fin, kernel


@kernel
def shift(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):
    for i in u.dom:
        v[i] = u[i + 1]


@kernel
def collide(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
    for i in x.dom:
        y[0] = x[i]
'''


def test_check_prints_the_witness_of_an_isl_refutation_under_its_line(
    tmp_path, capsys
) -> None:
    """What isl refuted is explained under the ``REFUTED`` line, in words.

    The isl oracle recorded the cell that escapes, and the two instances that
    write one cell, as ``witness`` and ``witness_text``, which lanky does not
    print, and no ``reason``, which it does. So both lines came out bare. Now
    the reason names the question and the labelled witness.
    """
    from lanky.cli import main as lanky_main

    path = write_fixture(tmp_path, OUT_OF_BOUNDS)
    out_path = tmp_path / "ledger.json"
    code = lanky_main(["check", str(path), "--json", str(out_path)])
    out = capsys.readouterr().out
    assert code == 1
    lines, header = refutation_block(out, "shift at ")
    assert lines[header].endswith("u[i + 1] is in bounds for every instance of S0")
    facts = json.loads(out_path.read_text(encoding="utf-8"))
    refuted = {fact["owner"]: fact for fact in facts if fact["status"] == "refuted"}
    assert [f["owner"] for f in facts if f["status"] == "refuted"] == [
        "shift",
        "collide",
    ]
    escapes = refuted["shift"]
    reason = escapes["provenance"]["reason"]
    assert reason.startswith("cells u[i + 1] reaches are cells u has, except [a0=")
    assert escapes["provenance"]["witness_text"] in reason
    assert f"  {reason}" in lines[header + 1 : header + 3]

    lines, header = refutation_block(out, "collide at ")
    assert lines[header].endswith("distinct instances of S0 write distinct cells of y")
    reason = refuted["collide"]["provenance"]["reason"]
    assert " is one of the pairs of S0 instances writing the same cell" in reason
    assert reason.startswith("[s=0, d0=")
    assert f"  {reason}" in lines[header + 1 : header + 3]


def test_run_reports_a_body_that_reads_past_the_end_instead_of_a_traceback(
    tmp_path, capsys
) -> None:
    # The native run of ``shift`` raises IndexError on its example input. The
    # command reported the errors it expected by name and stopped with a
    # traceback on this one.
    body = OUT_OF_BOUNDS + (
        "\n\ndef example_inputs():\n"
        "    import numpy as np\n\n"
        "    return {\n"
        '        "shift": {"u": np.arange(4.0), "v": np.zeros(4)},\n'
        '        "collide": {"x": np.arange(4.0), "y": np.zeros(4)},\n'
        "    }\n"
    )
    path = write_fixture(tmp_path, body)
    code = main(["run", str(path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "  IndexError: " in out
    # The other kernel in the file still ran, and agreed.
    assert "  y: difference 0 " in out


def test_run_reports_a_loop_carried_name_instead_of_a_traceback(
    tmp_path, capsys
) -> None:
    path = write_fixture(tmp_path, CARRIED)
    code = main(["run", str(path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "cannot schedule running_sum" in out
    assert "TraceError" in out
    assert "reduce_sum" in out
    assert "difference" not in out


def test_run_prints_what_refuted_a_run_under_its_line(tmp_path, capsys) -> None:
    """``loopty run`` prints the block ``lanky check`` prints under ``REFUTED``.

    The compiled run follows the term, which doubles ``x``; the Python body the
    fixture compares it with triples it. The agreement fact is refuted, and
    what is printed under its line is lanky's own
    :func:`lanky.cli.refutation_lines` of the fact: its reason, naming the
    output that disagreed. ``loopty run`` used to print the bare line.
    """
    from types import SimpleNamespace

    from lanky.cli import refutation_lines

    body = FIXTURE.replace("y[...] = 2.0 * x", "y[...] = 3.0 * x")
    path = write_fixture(tmp_path, body)
    out_path = tmp_path / "ledger.json"
    code = main(["run", str(path), "--json", str(out_path)])
    out = capsys.readouterr().out
    assert code == 1
    lines, header = refutation_block(out, "scale at fixture.py:1: ")
    assert lines[header].endswith(
        "the scheduled run of scale agrees with the native run to the accuracy "
        "its types state"
    )
    assert lines[header - 1] == ""

    facts = json.loads(out_path.read_text(encoding="utf-8"))
    (fact,) = [fact for fact in facts if fact["kind"] == "agreement"]
    # The cell nearest to failing is the one whose difference is the largest
    # multiple of its own allowance: 7 against 21, allowed 1e-6 * (21 + 1).
    assert fact["provenance"]["reason"] == (
        "y differs from the native run: difference 7, allowed 2.2e-05 (approx)"
    )
    # ``refutation_lines`` reads nothing but the provenance.
    expected = refutation_lines(SimpleNamespace(provenance=fact["provenance"]))
    assert lines[header + 1 : header + 1 + len(expected)] == [
        f"  {line}" for line in expected
    ]
    assert "no witness recorded" not in out


UNBUILDABLE = '''
"""A ragged row sum with a hardware axis inside the row: legal, not buildable."""

from __future__ import annotations

from lanky.prelude import Nat, Real

from loopty import Arr, Fin, kernel, reduce_sum
from loopty.schedule import Schedule


@kernel
def rowsum(
    cnt: Arr[Fin[n], Nat], val: Arr[Fin[n], Fin[cnt], Real], y: Arr[Fin[n], Real]
):
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


sched = Schedule(rowsum).split("j", 32, inner="j_in", outer="j_out").tag(
    j_in="l.0"
)
'''


def test_run_prints_why_a_schedule_cannot_be_built_under_its_line(
    tmp_path, capsys
) -> None:
    """The refuted ``buildable`` fact's reason is printed under its line.

    The limit was printed where the schedule is reported, and the fact carried
    it as ``detail`` alone, so the ``REFUTED`` line at the bottom had nothing
    under it.
    """
    path = write_fixture(tmp_path, UNBUILDABLE)
    code = main(["run", str(path)])
    out = capsys.readouterr().out
    assert code == 1
    (limit,) = [
        line.split("not buildable for the c target: ", 1)[1]
        for line in out.splitlines()
        if "not buildable for the c target: " in line
    ]
    assert "ragged fiber" in limit
    lines, header = refutation_block(out, "rowsum at fixture.py:")
    assert lines[header].endswith(
        "c code can be generated for rowsum after tag(j_in='l.0')"
    )
    assert lines[header + 1] == f"  {limit}"
    assert "no witness recorded" not in out


# {{{ two schedules of one kernel (#36)


def test_every_schedule_of_one_kernel_keeps_its_facts_in_the_ledger(tmp_path) -> None:
    """Three schedules of ``scale``: two different ones, and the first again.

    The facts were named by the kernel and the position of the step, so the
    second schedule's facts replaced the first's, and one agreement fact was
    left for three runs. Now each schedule's facts are its own; the third,
    which is the first again, shares its cast facts, which are the same
    claims, and keeps its own agreement, since it ran on inputs of its own.
    """
    body = FIXTURE + (
        '\nother = Schedule(scale).split("i", 2)\n'
        'again = Schedule(scale).split("i", 4).example('
        "x=np.ones(8), y=np.zeros(8))\n"
    )
    path = write_fixture(tmp_path, body)
    out_path = tmp_path / "ledger.json"
    assert main(["run", str(path), "--json", str(out_path)]) == 0
    facts = json.loads(out_path.read_text(encoding="utf-8"))
    kinds = [fact["kind"] for fact in facts]
    assert kinds.count("bijective") == kinds.count("monotone") == 2
    agreements = [fact["id"] for fact in facts if fact["kind"] == "agreement"]
    four = "agreement:scale[c].split('i', 4, inner='i_inner', outer='i_outer')"
    two = "agreement:scale[c].split('i', 2, inner='i_inner', outer='i_outer')"
    assert agreements == [four, two, f"{four}#2"]
    assert len({fact["id"] for fact in facts}) == len(facts) == 7


# }}}


# {{{ a refutation over a domain a guard left wide (#40)


CLIPPED = '''
"""A guard that compares with a Real scalar, which isl cannot state."""

from __future__ import annotations

from lanky.prelude import Real

from loopty import Arr, Fin, kernel, when


@kernel
def clipped(a: Real, x: Arr[Fin[m], Real], y: Arr[Fin[n], Real]):
    for i in y.dom:
        with when((i < a) & (a < x.dom.size)):
            y[i] = x[i]


@kernel
def stated(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):
    for i in u.dom:
        with when(i + 1 <= u.dom.size):
            v[i] = u[i + 1]
'''


def test_check_says_a_witness_may_be_one_the_guard_masks(tmp_path, capsys) -> None:
    """The conjuncts a domain leaves out are printed under the ``REFUTED`` line.

    ``x[i]`` is refuted at an ``i >= m``, which the guard masks for every real
    ``a``. The provenance said so under ``unnarrowed``, and the screen did
    not, so the refutation read as an out-of-bounds read. ``stated`` is
    refuted over a domain its guard narrowed whole, and says nothing more.
    """
    from lanky.cli import main as lanky_main

    path = write_fixture(tmp_path, CLIPPED)
    out_path = tmp_path / "ledger.json"
    code = lanky_main(["check", str(path), "--json", str(out_path)])
    out = capsys.readouterr().out
    assert code == 1
    lines, header = refutation_block(out, "clipped at ")
    assert lines[header].endswith("x[i] is in bounds for every instance of S0")
    block = lines[header + 1 : header + 6]
    assert block[0].startswith("  witness: ")
    assert block[1].startswith("  cells x[i] reaches are cells x has, except")
    assert block[2].startswith("  the domain is wider than the instances that write")
    assert block[3] == (
        "    i < a, which compares with the scalar a of sort Real, which is not "
        "a loop variable, a size or a scalar of an integral sort (Nat, Int, "
        "Fin[...]), and isl would read every name of a constraint as an integer"
    )
    assert block[4].startswith("    a < m, which compares with the scalar a")

    lines, header = refutation_block(out, "stated at ")
    assert lines[header].endswith("u[i + 1] is in bounds for every instance of S0")
    assert lines[header + 2].startswith("  cells u[i + 1] reaches are cells u has")
    assert "the domain is wider" not in "\n".join(lines[header:])

    facts = json.loads(out_path.read_text(encoding="utf-8"))
    refuted = [fact for fact in facts if fact["status"] == "refuted"]
    assert [fact["owner"] for fact in refuted] == ["clipped", "stated"]
    # A decided fact over the same wide domain carries its note in the
    # provenance only, and needs nothing on the screen.
    decided = [
        fact
        for fact in facts
        if fact["owner"] == "clipped" and fact["status"] == "decided"
    ]
    assert any("unnarrowed" in fact["provenance"] for fact in decided)


# }}}
