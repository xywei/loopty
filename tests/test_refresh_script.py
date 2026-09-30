"""The script that keeps the console blocks of the documentation true.

``scripts/refresh_example_outputs.py`` is what stands behind the claim that
every block is real output, so it is tested against documents of its own rather
than against the README: a command that exits non-zero has to fail the refresh,
in both modes, and leave its block alone, and an excerpt has to keep only lines
of what its command prints.
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


#: A command that prints a small ledger, as ``lanky check`` would.
LEDGER = shlex.join(
    [
        sys.executable,
        "-c",
        "print('decided  isl  demo.py:12  a[i] is in bounds'); "
        "print('decided  isl  demo.py:12  b[i] is in bounds'); "
        "print('tested   run  demo.py:40  the run agrees'); "
        "print(); print('3 facts: 2 decided, 1 tested')",
    ]
)


def _excerpt(path: Path, *kept: str) -> Path:
    return _document(path, LEDGER, "\n".join(kept))


def test_an_excerpt_whose_lines_are_all_in_the_output_is_current(tmp_path) -> None:
    document = _excerpt(
        tmp_path / "doc.md",
        "decided  isl  demo.py:12  b[i] is in bounds",
        "...",
        "3 facts: 2 decided, 1 tested",
    )
    before = document.read_text(encoding="utf-8")
    assert _script().main(["--check", str(document)]) == 0
    assert _script().main([str(document)]) == 0
    assert document.read_text(encoding="utf-8") == before


def test_an_excerpt_follows_a_moved_line_and_keeps_its_elisions(tmp_path) -> None:
    # The line number and the count moved; the rows kept and the elision stay.
    document = _excerpt(
        tmp_path / "doc.md",
        "decided  isl  demo.py:11  b[i] is in bounds",
        "...",
        "2 facts: 2 decided, 1 tested",
    )
    script = _script()
    assert script.main(["--check", str(document)]) == 1
    assert script.main([str(document)]) == 0
    body = document.read_text(encoding="utf-8").split("\n")[4:-2]
    assert body == [
        "decided  isl  demo.py:12  b[i] is in bounds",
        "...",
        "3 facts: 2 decided, 1 tested",
    ]
    assert script.main(["--check", str(document)]) == 0


def test_an_excerpt_line_that_is_not_in_the_output_fails(tmp_path) -> None:
    for kept in (
        ("decided  isl  demo.py:12  c[i] is in bounds", "..."),
        # In the output, but not in this order.
        (
            "tested   run  demo.py:40  the run agrees",
            "...",
            "decided  isl  demo.py:12  a[i] is in bounds",
        ),
    ):
        document = _excerpt(tmp_path / "doc.md", *kept)
        before = document.read_text(encoding="utf-8")
        assert _script().main(["--check", str(document)]) == 1
        assert _script().main([str(document)]) == 1
        assert document.read_text(encoding="utf-8") == before


def test_the_top_level_readme_ledgers_are_checked_excerpts() -> None:
    # Their rows say they are verbatim, and they were kept so by hand.
    script = _script()
    assert "README.md" in script.DOCUMENTS
    readme = SCRIPT.parent.parent / "README.md"
    blocks = script.blocks_of(readme.read_text(encoding="utf-8").split("\n"))
    ledger = [block for block in blocks if block.command.startswith("lanky check")]
    assert [block.command for block in ledger] == [
        "lanky check examples/spmv.py",
        "lanky check examples/pairs.py",
    ]
    assert all(block.elided for block in ledger)


def test_ci_checks_the_transcripts() -> None:
    workflow = SCRIPT.parent.parent / ".github" / "workflows" / "ci.yml"
    assert "refresh_example_outputs.py --check" in workflow.read_text(encoding="utf-8")
