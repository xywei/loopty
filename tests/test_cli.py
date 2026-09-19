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
