"""A statement's source text, as a message names the statement.

A refusal that names a pair of instances names each by the statement it is an
instance of: ``R[i, a, b] = -R[i, a - 2, b + 2] (l2p.py:269) at i=0, a=3,
b=7``, where the term alone could only say ``S1[i=0, a=3, b=7]``
(:func:`instances_text`). The text is read once, when the statement is
traced, from the file the body was written in (:func:`statement_text`), and
kept on the statement (:attr:`loopty.term.Stmt.text`).

The text is the statement as written, on one line. One longer than
:data:`LIMIT` characters keeps what it writes into and shortens the calls on
its right, ``y[cell[c, a]] += reduce_sum(...)``; one that is still too long,
its whole right side, ``u[t + 1, i] = ...``. The line of the message beside it
leads to the rest.
"""

from __future__ import annotations

import ast
import copy
import itertools
import linecache
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loopty.term import Stmt

__all__ = [
    "LIMIT",
    "frame_statement_text",
    "instance_text",
    "instances_text",
    "statement_text",
]

#: How long a statement's text may be before it is shortened.
LIMIT = 60

#: Each file's statements by their span, parsed once per version of its lines.
_PARSED: dict[str, tuple[list[str], list[ast.stmt]]] = {}


def frame_statement_text(frame: Any) -> str:
    """The text of the statement ``frame`` is executing; see :func:`statement_text`.

    The statement is found by the position of the instruction the frame is
    at, so that of two statements on one line (``a[i] = 0; b[i] = 1``) it is
    the one that is running.
    """
    code = frame.f_code
    # one entry per two-byte code unit, the instruction's at f_lasti // 2
    positions = next(
        itertools.islice(code.co_positions(), frame.f_lasti // 2, None),
        (frame.f_lineno, None, None, None),
    )
    line, _, column, _ = positions
    return statement_text(code.co_filename, line or frame.f_lineno, column)


def statement_text(filename: str, line: int, column: int | None = None) -> str:
    """The innermost statement at ``line`` of ``filename``, on one line.

    At ``column`` of the line when it is given; otherwise the innermost
    statement that starts on the line. Shortened as the module says.
    Empty where the source cannot be read: a body made by ``exec``, or a
    file that has changed since it was imported so that it no longer
    parses.
    """
    statements = _statements(filename)
    if not statements:
        return ""
    found = None
    for node in statements:
        if not _contains(node, line, column):
            continue
        if found is None or _within(node, found):
            found = node
    if found is None:
        return ""
    return _text(found, linecache.getlines(filename))


def _statements(filename: str) -> list[ast.stmt]:
    """Every statement of ``filename`` that a message can name, parsed once."""
    lines = linecache.getlines(filename)
    if not lines:
        return []
    cached = _PARSED.get(filename)
    if cached is not None and cached[0] is lines:
        return cached[1]
    try:
        tree = ast.parse("".join(lines), filename)
    except (SyntaxError, ValueError):
        statements: list[ast.stmt] = []
    else:
        statements = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign | ast.Return)
        ]
    _PARSED[filename] = (lines, statements)
    return statements


def _contains(node: ast.stmt, line: int, column: int | None) -> bool:
    """Whether ``node`` spans the position ``line:column``, or starts on ``line``."""
    end = node.end_lineno or node.lineno
    if column is None:
        return node.lineno == line
    if not node.lineno <= line <= end:
        return False
    if line == node.lineno and column < node.col_offset:
        return False
    return not (
        line == end
        and node.end_col_offset is not None
        and column >= node.end_col_offset
    )


def _within(inner: ast.stmt, outer: ast.stmt) -> bool:
    """Whether ``inner`` starts no earlier than ``outer``: the innermost wins."""
    return (inner.lineno, inner.col_offset) >= (outer.lineno, outer.col_offset)


def _text(node: ast.stmt, lines: list[str]) -> str:
    """``node`` on one line, shortened when it is longer than :data:`LIMIT`."""
    source = "".join(lines)
    segment = ast.get_source_segment(source, node) or ""
    whole = segment.strip() if "\n" not in segment.strip() else _unparse(node)
    if whole and len(whole) <= LIMIT:
        return whole
    value = getattr(node, "value", None)
    if value is None:
        return whole
    shortened = _unparse(_with_value(node, _ElidedCalls().visit(_copy(value))))
    if len(shortened) <= LIMIT:
        return shortened
    return _unparse(_with_value(node, ast.Constant(Ellipsis)))


class _ElidedCalls(ast.NodeTransformer):
    """Every call's arguments as ``...``: ``reduce_sum(...)``."""

    def visit_Call(self, node: ast.Call) -> ast.Call:  # noqa: N802 - ast's name
        if node.args or node.keywords:
            node.args = [ast.Constant(Ellipsis)]
            node.keywords = []
        node.func = self.visit(node.func)
        return node


class _LoneGenerator(ast.NodeTransformer):
    """A call of one generator spelled as Python spells it, ``f(x for x in y)``.

    :func:`ast.unparse` writes ``f((x for x in y))``; the generator's own
    text, without its parentheses, stands in for it as a name would.
    """

    def visit_Call(self, node: ast.Call) -> ast.Call:  # noqa: N802 - ast's name
        self.generic_visit(node)
        if (
            len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.GeneratorExp)
        ):
            inner = ast.unparse(node.args[0])
            if inner.startswith("(") and inner.endswith(")"):
                node.args = [ast.Name(id=inner[1:-1])]
        return node


def _unparse(node: ast.AST) -> str:
    """:func:`ast.unparse` of a copy of ``node``, a lone generator unparenthesized."""
    return ast.unparse(_LoneGenerator().visit(_copy(node)))


def _copy(node: ast.AST) -> Any:
    """A deep copy of ``node`` that the transformers may change."""
    return copy.deepcopy(node)


def _with_value(node: ast.stmt, value: ast.expr) -> ast.stmt:
    """``node`` with its right side ``value``: what it writes into, kept."""
    copied = _copy(node)
    copied.value = value
    return copied


# {{{ instances, as a message names them


def instance_text(stmt_id: str, coords: Mapping[str, int]) -> str:
    """``S0[t=0, i=8]``: an instance by its statement's id, as a witness records it."""
    inner = ", ".join(f"{name}={value}" for name, value in coords.items())
    return f"{stmt_id}[{inner}]"


def instances_text(
    stmts: Mapping[str, Stmt],
    *instances: tuple[str, Mapping[str, int]],
    called: Any = None,
    arrays: Any = None,
    checked: Any = None,
) -> list[str]:
    """Instances as one message names them, in the order it names them.

    Each is the statement as the source writes it, the line it is on, and
    the instance's coordinates by the names the source gives its loops:
    ``R[i, a, b] = C[i, a, b] (l2p.py:267) at i=0, a=1, b=9``. An instance
    of the statement named just before it is ``the same statement at i=0,
    a=3, b=7``, and two statements of one text and one line, the same
    kernel's statement in two of a program's calls, are told apart by the
    call: ``(hexahedra.py:211, along_r@2)``. A statement with no source
    text, of a term written by hand or one loopty adds, is named by its id,
    ``S1[i=0, a=3, b=7]``, as the witness in a fact's provenance names it;
    but the zeros a program makes an array with, which no line of the
    source stores, are named by the array and the cell, ``the zeros of l2p.R
    (l2p.py:316) at R[0, 1, 2]``.

    ``called`` names the call a statement of a program's term is in
    (``l2p.reconstruct``, :mod:`loopty.handles`), and ``arrays`` an array,
    each by its path; the label and the name otherwise. ``checked`` gives
    the array a statement of a program's checked point checks, which no line
    stores either: it is named as ``called`` names it, ``the check of p.idx
    before p.gather (travel.py:106) at idx[3]``.
    """
    if called is None:

        def called(stmt_id: str) -> str:
            return stmt_id.rsplit(".", 1)[0]

    if arrays is None:

        def arrays(name: str) -> str:
            return name

    places: dict[tuple[str, str], set[str]] = {}
    for stmt_id, _ in instances:
        stmt = stmts.get(stmt_id)
        if stmt is not None and stmt.text:
            places.setdefault((stmt.text, stmt.where), set()).add(stmt_id)
    out = []
    previous = None
    for stmt_id, coords in instances:
        stmt = stmts.get(stmt_id)
        if stmt is not None and not stmt.text and stmt.id.endswith(".zeros"):
            array = stmt.assignee.array
            cell = ", ".join(str(coords.get(name, name)) for name in stmt.inames)
            out.append(
                f"the zeros of {arrays(array)} ({stmt.where}) at {array}[{cell}]"
            )
            previous = stmt_id
            continue
        array = None if checked is None or stmt is None else checked(stmt_id)
        if array is not None and not stmt.text:
            cell = ", ".join(str(coords.get(name, name)) for name in stmt.inames)
            at = f" at {array}[{cell}]" if cell else ""
            if stmt_id == previous:
                out.append(f"the same check{at}")
            else:
                out.append(f"{called(stmt_id)} ({stmt.where}){at}")
            previous = stmt_id
            continue
        if stmt is None or not stmt.text:
            out.append(instance_text(stmt_id, coords))
            previous = stmt_id
            continue
        at = _at_coords(stmt, coords)
        if stmt_id == previous:
            out.append(f"the same statement{at}")
            continue
        where = stmt.where
        if len(places[stmt.text, stmt.where]) > 1:
            where = f"{where}, {called(stmt_id)}"
        out.append(f"{stmt.text} ({where}){at}")
        previous = stmt_id
    return out


def _at_coords(stmt: Stmt, coords: Mapping[str, int]) -> str:
    """`` at i=0, a=3, b=7``: an instance's coordinates, by the source's names.

    Empty for a statement in no loop, whose one instance needs none.
    """
    spelled = dict(zip(stmt.inames, stmt.spelled, strict=False))
    inner = ", ".join(
        f"{spelled.get(name, name)}={value}" for name, value in coords.items()
    )
    return f" at {inner}" if inner else ""


# }}}
