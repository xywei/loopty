"""Where the C target builds, and that a run leaves nothing there (#335).

loopy 2025.2's ``CCompiler`` makes a ``tmp_loopy*`` directory in ``$TMPDIR``
with each C target and never removes it, and loopty made a target with each
kernel it lowers. Every :class:`loopty.lower.InProcessCTarget` of a process
builds with one compiler now, whose directory the process removes when it
exits (:class:`loopty.lower._ProcessCompiler`, note 34 in
``docs/loopy-notes.md``).
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from loopy.target.c.c_execution import CCompiler

from loopty.lower import (
    InProcessCTarget,
    _process_compiler,
    _ProcessCompiler,
    target_for,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _left(directory: Path) -> list[str]:
    """The build directories loopy's compiler left in ``directory``."""
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.name.startswith("tmp_loopy")
    )


def _run(command: list[str], tmp_path: Path) -> tuple[Path, str]:
    """Run ``command`` with ``$TMPDIR`` an empty directory, and return it."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "TMPDIR": str(scratch)},
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return scratch, output


def test_the_c_targets_of_a_process_build_in_one_directory() -> None:
    one, other = InProcessCTarget(), target_for("c")
    assert one.compiler is other.compiler is _process_compiler()
    assert os.path.isdir(one.compiler.tempdir)


def test_a_pickled_target_builds_with_the_compiler_of_its_reader() -> None:
    target = InProcessCTarget()
    pickled = pickle.dumps(target)
    assert target.compiler.tempdir.encode() not in pickled
    read = pickle.loads(pickled)
    assert read.compiler is _process_compiler()
    assert read == target and hash(read) == hash(target)


def test_a_target_pickled_with_its_compiler_drops_it() -> None:
    """As loopty 0.4.0 pickled one into loopy's caches, with a directory now gone."""
    gone = CCompiler.__new__(CCompiler)
    gone.tempdir = "/nonexistent/tmp_loopy_of_a_process_that_exited"
    read = InProcessCTarget.__new__(InProcessCTarget)
    read.__setstate__({"fortran_abi": False, "compiler": gone})
    assert read.compiler is _process_compiler()


def test_a_child_builds_in_a_directory_of_its_own_once_its_parents_is_gone() -> None:
    compiler = _ProcessCompiler()
    parents = compiler.tempdir
    compiler._remove()
    assert not os.path.exists(parents)
    # As the compiler is in a process forked from the one that made it.
    compiler._owner = os.getppid()
    library = compiler.build("answer", "int answer(void) { return 42; }\n")
    assert library.answer() == 42
    assert compiler.tempdir != parents and os.path.isdir(compiler.tempdir)
    assert compiler._owner == os.getpid()
    compiler._remove()
    assert not os.path.exists(compiler.tempdir)


_FORKED = """
import os
import sys

from loopty.lower import InProcessCTarget

compiler = InProcessCTarget().compiler
child = os.fork()
if child == 0:
    assert compiler.build("child", "int child(void) { return 1; }\\n").child() == 1
    sys.exit(0)
_, status = os.waitpid(child, 0)
assert os.waitstatus_to_exitcode(status) == 0
assert os.path.isdir(compiler.tempdir), "the child removed its parent's directory"
assert compiler.build("parent", "int parent(void) { return 2; }\\n").parent() == 2
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no os.fork here")
def test_only_the_process_that_made_the_directory_removes_it(tmp_path: Path) -> None:
    scratch, _ = _run([sys.executable, "-c", _FORKED], tmp_path)
    assert _left(scratch) == []


@pytest.mark.parametrize("verb", ["python", "check", "run"])
def test_an_example_leaves_no_build_directory(verb: str, tmp_path: Path) -> None:
    """``examples/l2p.py`` left two, one and four before (#335)."""
    example = str(EXAMPLES / "l2p.py")
    command = {
        "python": [sys.executable, example],
        "check": [sys.executable, "-m", "lanky.cli", "check", example],
        "run": [sys.executable, "-m", "loopty.cli", "run", example],
    }[verb]
    scratch, output = _run(command, tmp_path)
    assert _left(scratch) == [], output[-2000:]
