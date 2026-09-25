"""The script that keeps the console blocks of ``examples/README.md`` true.

``scripts/refresh_example_outputs.py`` is what stands behind the README's claim
that every block is real output, so it is tested against a document of its own
rather than against the README: a command that exits non-zero has to fail the
refresh, in both modes, and leave its block alone.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "refresh_example_outputs.py"
)


def _script():
    spec = importlib.util.spec_from_file_location("refresh_example_outputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # The script defines a dataclass, and ``dataclasses`` looks the defining
    # module up in ``sys.modules`` while it processes the class.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _document(path: Path, command: str, body: str) -> Path:
    path.write_text(
        f"# a demo\n\n```console\n$ {command}\n{body}\n```\n", encoding="utf-8"
    )
    return path


#: A command that prints a line and then fails, as a demo with a broken import
#: halfway down would.
FAILING = shlex.join(
    [sys.executable, "-c", "print('partial'); raise SystemExit(3)"]
)


def test_a_failing_command_is_not_pasted_into_the_document(tmp_path) -> None:
    document = _document(tmp_path / "doc.md", FAILING, "the old output")
    before = document.read_text(encoding="utf-8")
    assert _script().main([str(document)]) == 1
    assert document.read_text(encoding="utf-8") == before


def test_a_failing_command_is_never_current_under_check(tmp_path) -> None:
    # The block already holds exactly what the command prints, so the text alone
    # would pass; the exit status is what says the document is out of date.
    document = _document(tmp_path / "doc.md", FAILING, "partial")
    assert _script().main(["--check", str(document)]) == 1


def test_a_command_that_succeeds_is_still_refreshed(tmp_path) -> None:
    command = shlex.join([sys.executable, "-c", "print('fresh')"])
    document = _document(tmp_path / "doc.md", command, "stale")
    script = _script()
    assert script.main(["--check", str(document)]) == 1
    assert script.main([str(document)]) == 0
    assert "\nfresh\n" in document.read_text(encoding="utf-8")
    assert script.main(["--check", str(document)]) == 0
