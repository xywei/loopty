"""Smoke tests: the package imports, re-exports, and the command runs."""

from __future__ import annotations

import subprocess
import sys

import loopty
from loopty import cli


def test_version() -> None:
    assert loopty.__version__ == "0.1.0.dev0"


def test_top_level_names_are_re_exported() -> None:
    for name in (
        "Arr",
        "Fin",
        "IllegalCast",
        "Kernel",
        "Schedule",
        "Term",
        "kernel",
        "program",
        "sum",
        "when",
    ):
        assert hasattr(loopty, name), name


def test_cli_without_a_verb_prints_help(capsys) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "loopty" in out
    assert "run" in out
    assert "check" in out


def test_cli_version(capsys) -> None:
    assert cli.main(["--version"]) == 0
    assert loopty.__version__ in capsys.readouterr().out


def test_plugin_module_exports_the_four_entry_points() -> None:
    from loopty import plugin

    assert plugin.IslOracle().trust_class() == "decision-procedure"
    assert plugin.RunVerb().name == "run"
    assert plugin.KernelTheory.name == "kernel"
    assert plugin.LoopyExecutor.name == "loopy"


def test_star_import_leaves_the_builtin_sum_alone() -> None:
    # ``loopty.sum`` is the compatibility alias of ``reduce_sum``; exporting it
    # made ``from loopty import *`` shadow the builtin in the importing module.
    namespace: dict = {}
    exec("from loopty import *", namespace)
    assert "sum" not in namespace
    assert namespace["reduce_sum"] is loopty.reduce_sum
    assert loopty.sum is loopty.reduce_sum


def heavy_modules_after(code: str) -> set[str]:
    """Which of loopy, islpy and pyopencl a fresh interpreter has after ``code``."""
    probe = (
        "import sys\n"
        "print(' '.join(m for m in ('loopy', 'islpy', 'pyopencl') "
        "if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", f"{code}\n{probe}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.split())


def test_importing_loopty_imports_neither_loopy_nor_islpy() -> None:
    assert heavy_modules_after("import loopty") == set()


def test_a_kernel_module_does_not_import_loopy() -> None:
    # What a file of kernels imports; loopy is for lowering, and a native run
    # never lowers. islpy comes with the tracer.
    code = "from loopty import Arr, Fin, kernel, reduce_sum, when"
    assert "loopy" not in heavy_modules_after(code)


def test_discovering_the_plugins_does_not_import_loopy() -> None:
    # lanky loads every entry point for every command it runs.
    code = "from lanky.plugins import registry\nregistry.load_entry_points()"
    assert "loopy" not in heavy_modules_after(code)


def test_kernel_and_trace_are_the_functions_after_their_modules_load() -> None:
    # Both are also submodules, and plugin discovery imports them before any
    # kernel file does; the submodule must not take the name's place.
    code = (
        "import types\n"
        "import loopty.plugin\n"
        "import loopty\n"
        "from loopty import kernel, trace\n"
        "assert not isinstance(kernel, types.ModuleType), kernel\n"
        "assert not isinstance(trace, types.ModuleType), trace\n"
        "assert loopty.kernel is kernel and loopty.trace is trace\n"
    )
    assert heavy_modules_after(code) <= {"islpy"}
