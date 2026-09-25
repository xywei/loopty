"""Smoke tests: the package imports, re-exports, and the command runs."""

from __future__ import annotations

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

