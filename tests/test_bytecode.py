"""What a body's bytecode says about the statement running it (:mod:`loopty.bytecode`).

The reading has to hold on every CPython loopty runs on, 3.12 to 3.14, whose
instruction layouts differ. A layout this interpreter does not make is pinned
here as another interpreter lists it, transcribed from ``dis``.
"""

from __future__ import annotations

import dis
from types import SimpleNamespace

import numpy as np

from loopty import Arr, bytecode
from loopty.bytecode import for_targets, loops_back

#: Statements enough that a ``for``'s body is longer than 255 instructions.
LONG = 40


def _long_loop() -> object:
    """``for a, b in t.dom:`` around :data:`LONG` statements, as a function."""
    body = "\n".join(f"        t[a, b] = t[a, b] + {k}.0" for k in range(LONG))
    namespace: dict[str, object] = {}
    exec(f"def fill(t):\n    for a, b in t.dom:\n{body}\n", namespace)
    return namespace["fill"]


def test_a_for_around_a_long_body_names_its_targets_past_extended_arg() -> None:
    # The FOR_ITER that jumps past a body that long carries the high bits of
    # its argument in an EXTENDED_ARG before it, between it and the GET_ITER
    # that asks the domain for an iterator. examples/advection.py's lift on
    # the faces is that long on 3.14, where 'for a, b in flux.dom[e, 0]:'
    # handed out single indices and the unpacking failed.
    fill = _long_loop()
    listing = list(dis.get_instructions(fill.__code__))
    (asks,) = [ins for ins in listing if ins.opname == "GET_ITER"]
    after = listing[listing.index(asks) + 1]
    assert after.opname == "EXTENDED_ARG", [ins.opname for ins in listing[:12]]
    assert for_targets(fill.__code__, asks.offset) == ("a", "b")
    (loops,) = [ins for ins in listing if ins.opname == "FOR_ITER"]
    assert for_targets(fill.__code__, loops.offset) == ("a", "b")
    t = Arr.from_numpy(np.zeros((2, 3)))
    fill(t)
    assert np.array_equal(t.numpy(), np.full((2, 3), sum(range(LONG))))


def _listing(rows: list[tuple]) -> tuple[tuple, dict[int, int]]:
    """Instructions as ``dis`` lists them, from ``(offset, opname, argval, line)``."""
    listing = tuple(
        SimpleNamespace(
            offset=offset,
            opname=opname,
            argval=argval,
            arg=None,
            positions=SimpleNamespace(lineno=line),
        )
        for offset, opname, argval, line in rows
    )
    return listing, {ins.offset: index for index, ins in enumerate(listing)}


#: 3.14's ``for i in u.dom:`` around ``while u[i] < 1.0: u[i] = u[i] + 1.0``,
#: lines counted from the ``def``. The test is at the top of the loop, and
#: its pass ends with the jump back to it (96, to 32).
WHILE_IN_A_FOR = [
    (2, "LOAD_FAST_BORROW", "u", 1),
    (4, "LOAD_ATTR", "dom", 1),
    (24, "GET_ITER", None, 1),
    (26, "FOR_ITER", 100, 1),
    (30, "STORE_FAST", "i", 1),
    (32, "LOAD_FAST_BORROW_LOAD_FAST_BORROW", ("u", "i"), 2),
    (34, "BINARY_OP", 26, 2),
    (46, "LOAD_CONST", 1.0, 2),
    (48, "COMPARE_OP", "<", 2),
    (52, "POP_JUMP_IF_TRUE", 62, 2),
    (56, "NOT_TAKEN", None, 2),
    (58, "JUMP_BACKWARD", 26, 2),
    (62, "LOAD_FAST_BORROW_LOAD_FAST_BORROW", ("u", "i"), 3),
    (64, "BINARY_OP", 26, 3),
    (76, "LOAD_CONST", 1.0, 3),
    (78, "BINARY_OP", 0, 3),
    (90, "LOAD_FAST_BORROW_LOAD_FAST_BORROW", ("u", "i"), 3),
    (92, "STORE_SUBSCR", None, 3),
    (96, "JUMP_BACKWARD", 32, 3),
    (100, "END_FOR", None, 1),
    (102, "POP_ITER", None, 1),
]

#: The same with ``if`` for ``while``: the block jumps back to the FOR_ITER.
IF_IN_A_FOR = [
    *WHILE_IN_A_FOR[:-3],
    (96, "JUMP_BACKWARD", 26, 3),
    *WHILE_IN_A_FOR[-2:],
]

#: 3.14's ``k = 0`` and ``while k < 3:`` around ``if u[i] < 1.0: u[i] = u[i]
#: + 1.0`` and ``k += 1``: the pass jumps back to the while's own test, on
#: its own line, not to the if's.
IF_IN_A_PLAIN_WHILE = [
    (2, "LOAD_SMALL_INT", 0, 1),
    (4, "STORE_FAST", "k", 1),
    (6, "LOAD_FAST_BORROW", "k", 2),
    (8, "LOAD_SMALL_INT", 3, 2),
    (10, "COMPARE_OP", "<", 2),
    (14, "POP_JUMP_IF_FALSE", 102, 2),
    (18, "NOT_TAKEN", None, 2),
    (20, "LOAD_FAST_BORROW_LOAD_FAST_BORROW", ("u", "i"), 3),
    (22, "BINARY_OP", 26, 3),
    (34, "LOAD_CONST", 1.0, 3),
    (36, "COMPARE_OP", "<", 3),
    (40, "POP_JUMP_IF_FALSE", 80, 3),
    (44, "NOT_TAKEN", None, 3),
    (46, "LOAD_FAST_BORROW_LOAD_FAST_BORROW", ("u", "i"), 4),
    (48, "BINARY_OP", 26, 4),
    (60, "LOAD_CONST", 1.0, 4),
    (62, "BINARY_OP", 0, 4),
    (74, "LOAD_FAST_BORROW_LOAD_FAST_BORROW", ("u", "i"), 4),
    (76, "STORE_SUBSCR", None, 4),
    (80, "LOAD_FAST_BORROW", "k", 5),
    (82, "LOAD_SMALL_INT", 1, 5),
    (84, "BINARY_OP", 13, 5),
    (96, "STORE_FAST", "k", 5),
    (98, "JUMP_BACKWARD", 6, 5),
    (102, "LOAD_CONST", None, 2),
    (104, "RETURN_VALUE", None, 2),
]


def test_a_while_tested_at_the_top_of_its_loop_loops_back(monkeypatch) -> None:
    # 3.14 tests a while's condition once per pass, at the top, where 3.12
    # and 3.13 test it at the bottom too; a while inside a for read as an if
    # there, and the trace asked it until its bound of asks, not refusing it
    # as a while.
    code = object()
    for rows, asked, expected in (
        (WHILE_IN_A_FOR, 48, True),
        (IF_IN_A_FOR, 48, False),
        (IF_IN_A_PLAIN_WHILE, 36, False),
    ):
        listed = _listing(rows)
        monkeypatch.setattr(bytecode, "_instructions", lambda _, listed=listed: listed)
        assert loops_back(code, asked) is expected, rows[-3:]
