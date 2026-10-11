"""Claims settled by running them: an accuracy row in the ledger, and no more.

Some of what a file claims is about numbers no decision procedure reads: that
a quadrature converges at its order, that pytential's Green identity holds on a
curve to the accuracy its expansions promise. Such a claim is checked by
running it, and the ledger should show it as that, beside the facts isl
decided and Lean proved, rather than leave it to a print statement::

    @tested
    def green_identity():
        \"\"\"S(du/dn) - D(u) - u/2 falls as the panels shrink, at QBX order 5.\"\"\"
        orders = ...
        return min(orders) > 3.4, f"orders {orders}"

The function takes nothing and returns whether the claim held, alone or with
a line saying what was measured. The docstring's first line is the claim, and
it is what ``lanky check`` prints. The row is ``tested``, decided by ``run``,
when the function returns true, and ``refuted`` when it returns false or
raises, with what it measured or the error as the reason. A run is evidence
about the inputs it ran on and never a proof, so nothing here is offered to
an oracle, and no claim made this way reads more than ``tested``.

Calling the decorated function runs it, as ``python file.py`` does, and
returns what the function returns. ``lanky check`` runs it once, for the row.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import numpy as np
from lanky.check import module_name
from lanky.ledger import Fact, Status, fact_id
from lanky.plugins import registry

__all__ = ["Tested", "tested"]

#: The kind of the fact a :class:`Tested` claim makes.
KIND = "tested"


class Tested:
    """A claim checked by running a function: ``tested``, or ``refuted``.

    Attributes:
        fn: The function, which takes no arguments and returns a ``bool``, or
            a ``bool`` and a string saying what it measured.
        statement: The claim, the first line of the docstring.
    """

    def __init__(self, fn: Any) -> None:
        if not callable(fn) or getattr(fn, "__code__", None) is None:
            raise TypeError(f"@tested decorates a function, and was given {fn!r}")
        doc = (fn.__doc__ or "").strip()
        if not doc:
            raise TypeError(
                f"@tested {fn.__qualname__} has no docstring: its first line is "
                "the claim the ledger prints, so write the claim there"
            )
        self.fn = fn
        functools.update_wrapper(self, fn)
        code = fn.__code__
        self.path = code.co_filename
        self.line = code.co_firstlineno
        self.where = f"{os.path.basename(code.co_filename)}:{code.co_firstlineno}"
        self.qualname = getattr(fn, "__qualname__", fn.__name__)
        self.module = (
            module_name(code.co_filename) or getattr(fn, "__module__", "") or ""
        )
        self.statement = doc.splitlines()[0].strip().rstrip(".")
        self._facts: tuple[Fact, ...] | None = None

    def __call__(self) -> Any:
        """Run the claim, and return what the function returns."""
        return self.fn()

    def outcome(self) -> tuple[bool, str]:
        """Run the claim: whether it held, and what was measured, or the error.

        A ``bool`` alone (numpy's too) is read as the verdict with nothing
        measured; a pair as the verdict and the measurement. Anything else is
        refused, since a number or a string returned by mistake would read as
        true.
        """
        try:
            result = self.fn()
        except Exception as exc:  # noqa: BLE001 - the error is the row's reason
            return False, f"the run raised {type(exc).__name__}: {exc}"
        if isinstance(result, tuple) and len(result) == 2:
            verdict, measured = result
        else:
            verdict, measured = result, ""
        if isinstance(verdict, np.bool_):
            verdict = bool(verdict)
        if not isinstance(verdict, bool):
            return False, (
                f"{self.qualname} returned {type(verdict).__name__}, not a bool "
                "or a bool and a string: return whether the claim held"
            )
        return verdict, str(measured)

    def fact_id(self) -> str:
        """The id of the claim's one fact."""
        return fact_id(KIND, self.qualname, module=self.module, line=self.line)

    def facts(self) -> tuple[Fact, ...]:
        """The claim's row: ``tested`` by the run, or ``refuted`` with the reason."""
        if self._facts is not None:
            return self._facts
        held, measured = self.outcome()
        provenance: dict[str, Any] = {"measured": measured}
        if not held:
            provenance["reason"] = measured or "the run returned False"
        self._facts = (
            Fact(
                id=self.fact_id(),
                kind=KIND,
                statement=self.statement,
                term=None,
                status=Status.TESTED if held else Status.REFUTED,
                decided_by="run",
                provenance=provenance,
                where=self.where,
                owner=self.qualname,
            ),
        )
        return self._facts

    def __repr__(self) -> str:
        return f"<tested {self.__name__} at {self.where}: {self.statement}>"


def tested(fn: Any) -> Tested:
    """Decorate ``fn`` as a claim checked by running it, and register it."""
    from loopty.kernel import ensure_registered

    ensure_registered()
    return registry.register_object(Tested(fn))
