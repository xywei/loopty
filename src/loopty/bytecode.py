"""What a body's bytecode says about the statement running it.

A kernel body is run, not parsed, and two questions about the statement that
is running are answered by the instructions around the one a frame is at.

* **The names a ``for`` binds.** ``for box, a, b in R.dom:`` asks its domain
  for an iterator and unpacks each point into three names, and the iterator
  is the one to hand it points of three loops; ``for box in R.dom:`` is the
  one loop (:func:`for_targets`). The names are those of the source, which
  keeps the term's loop variables the source's.
* **Whether a truth value closes a loop.** ``while a < 2:`` asks ``a < 2``
  for its truth again at each pass, and the pass ends with a jump back
  (:func:`loops_back`), which is how a ``while`` on a traced condition is
  told from an ``if``.

The instruction set is CPython's, of 3.12, 3.13 and 3.14; an instruction
layout none has is no answer, ``None`` or ``False``, and the caller falls back
on what it did before.
"""

from __future__ import annotations

import dis
import functools
from typing import Any

__all__ = ["for_targets", "loops_back"]

#: Instructions that do nothing here: ``NOT_TAKEN``, which 3.14 puts after a
#: conditional jump, ``NOP``, and ``EXTENDED_ARG``, the high bits of the next
#: instruction's argument, which a jump over a long loop body needs: ``GET_ITER``
#: is followed by ``EXTENDED_ARG`` and then ``FOR_ITER`` where the body is
#: longer than 255 instructions.
_INERT = ("NOT_TAKEN", "NOP", "EXTENDED_ARG")

#: The instructions that end a statement, or start a loop's next pass, which
#: the evaluation of a condition has none of.
_STATEMENT_ENDS = (
    *("STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF", "STORE_ATTR"),
    *("STORE_SUBSCR", "STORE_FAST_STORE_FAST", "STORE_FAST_LOAD_FAST"),
    *("POP_TOP", "FOR_ITER", "END_FOR", "POP_ITER", "JUMP_FORWARD"),
    *("JUMP_BACKWARD", "RETURN_VALUE"),
)

#: The instructions that store one name, the target of a ``for`` or a part
#: of it.
_STORES = ("STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF")


@functools.lru_cache(maxsize=256)
def _instructions(code: Any) -> tuple[tuple[Any, ...], dict[int, int]]:
    """The instructions of a code object, and an offset-to-index table."""
    listing = tuple(dis.get_instructions(code))
    index_of = {instruction.offset: index for index, instruction in enumerate(listing)}
    return listing, index_of


def _next(listing: tuple[Any, ...], position: int) -> int:
    """The index of the first instruction from ``position`` on that does something."""
    while position < len(listing) and listing[position].opname in _INERT:
        position += 1
    return position


@functools.lru_cache(maxsize=1024)
def for_targets(code: Any, offset: int) -> tuple[str, ...] | None:
    """The names the ``for`` at ``offset`` of ``code`` binds, or ``None``.

    ``offset`` is the instruction a frame is at when it asks a domain for an
    iterator (``GET_ITER``) or for a point (``FOR_ITER``). The answer is one
    name for ``for i in x.dom``, and the names in order for ``for box, a, b
    in R.dom``; an entry is ``""`` where the target is not a plain name
    (``for box, (a, b) in ...``, ``for self.i in ...``), which no loop nest
    is. ``None`` means the frame is not starting a ``for`` this can read,
    such as a call ``list(x.dom)`` or ``zip(x.dom, y.dom)``.

    A generator expression asks for its outermost iterator in the frame
    that makes it, ``GET_ITER`` and then the call of the generator's code,
    and unpacks inside that code: its first ``for`` is read there.
    """
    listing, index_of = _instructions(code)
    position = index_of.get(offset)
    if position is None:
        return None
    current = listing[position]
    if current.opname == "GET_ITER":
        after = _next(listing, position + 1)
        if after >= len(listing):
            return None
        if listing[after].opname == "FOR_ITER":
            return _targets_from(listing, after + 1)
        if listing[after].opname == "CALL" and listing[after].arg == 0:
            generator = _generator_code(listing, position)
            return None if generator is None else _first_targets(generator)
        return None
    if current.opname == "FOR_ITER":
        return _targets_from(listing, position + 1)
    return None


def _targets_from(listing: tuple[Any, ...], position: int) -> tuple[str, ...] | None:
    """The names stored by the instructions at ``position``, a ``for``'s target."""
    position = _next(listing, position)
    if position >= len(listing):
        return None
    first = listing[position]
    if first.opname.startswith("STORE_"):
        name = first.argval
        if isinstance(name, tuple) and name:
            # 3.13 fuses a store with the instruction after it on one line:
            # STORE_FAST_LOAD_FAST ('i', 's') stores i first.
            name = name[0]
        return (name,) if isinstance(name, str) and name.isidentifier() else ("",)
    if first.opname != "UNPACK_SEQUENCE":
        return None
    count = first.arg
    names: list[str] = []
    position += 1
    while len(names) < count and position < len(listing):
        instruction = listing[position]
        if instruction.opname in _STORES:
            names.append(instruction.argval)
        elif instruction.opname == "STORE_FAST_STORE_FAST":
            names.extend(instruction.argval)
        elif instruction.opname == "STORE_FAST_LOAD_FAST":
            names.append(instruction.argval[0])
        else:
            # A nested unpacking, an attribute or a subscript: no plain name.
            names.append("")
            break
        position += 1
    names.extend([""] * (count - len(names)))
    return tuple(names[:count])


def _generator_code(listing: tuple[Any, ...], position: int) -> Any:
    """The code of the generator expression made just before ``position``.

    ``(f(a, b) for a, b in R.dom)`` loads the generator's code, makes a
    function of it, evaluates ``R.dom`` and asks it for an iterator, and
    calls the function with that iterator.
    """
    for index in range(position - 1, 0, -1):
        if listing[index].opname != "MAKE_FUNCTION":
            continue
        loaded = listing[index - 1]
        value = loaded.argval
        named = getattr(value, "co_name", "")
        if loaded.opname == "LOAD_CONST" and named == "<genexpr>":
            return value
        return None
    return None


def _first_targets(code: Any) -> tuple[str, ...] | None:
    """The names the first ``for`` of a generator's code binds."""
    listing, _ = _instructions(code)
    for position, instruction in enumerate(listing):
        if instruction.opname == "FOR_ITER":
            return _targets_from(listing, position + 1)
    return None


def loops_back(code: Any, offset: int) -> bool:
    """Whether the truth value asked for at ``offset`` decides whether a loop goes on.

    3.12 and 3.13 test a ``while``'s condition at its start and again at the
    end of each pass, and the second test is a conditional jump out of the
    loop followed by the jump back to its start: when the condition holds,
    control falls through to ``JUMP_BACKWARD``. 3.14 tests it once per pass,
    at the loop's start, and each pass ends with the jump back to that test
    (:func:`_tested_at_the_top`). An ``if`` is followed by its own block, or,
    last in a ``for`` loop's body, by the jump back to the ``FOR_ITER`` that
    takes the next point, which is no ``while``. The truth value is asked for
    by the jump itself (3.12), or by the ``TO_BOOL`` or the comparison just
    before it (3.13 and 3.14).
    """
    listing, index_of = _instructions(code)
    position = index_of.get(offset)
    if position is None:
        return False
    asked = position
    if not listing[position].opname.startswith("POP_JUMP_IF"):
        position = _next(listing, position + 1)
        if position >= len(listing) or not listing[position].opname.startswith(
            "POP_JUMP_IF"
        ):
            return False
    jump = listing[position]
    after = _next(listing, position + 1)
    if after < len(listing):
        back = listing[after]
        target = jump.argval if isinstance(jump.argval, int) else -1
        if back.opname == "JUMP_BACKWARD" and target > back.offset:
            start = index_of.get(back.argval)
            if start is None or listing[start].opname != "FOR_ITER":
                return True
    return _tested_at_the_top(listing, index_of, asked, position)


def _tested_at_the_top(
    listing: tuple[Any, ...], index_of: dict[int, int], asked: int, jump: int
) -> bool:
    """Whether a pass of a loop ends with a jump back to the condition at ``asked``.

    3.14's ``while`` inside a ``for``: the test at the top, whose jump leads
    into the body when the condition holds, and the body's last instruction
    jumps back to the test's first::

        L4: LOAD_FAST_BORROW_LOAD_FAST_BORROW (u, i); BINARY_OP ([])
            LOAD_CONST (1.0); COMPARE_OP (bool(<))   # asked here
            POP_JUMP_IF_TRUE L5; NOT_TAKEN; JUMP_BACKWARD (to the FOR_ITER)
        L5: ...; JUMP_BACKWARD L4

    The jump back lands on the condition's own line, with nothing between it
    and the test that ends a statement, and not on a ``FOR_ITER``: an ``if``
    in a ``for`` is jumped back to from its ``FOR_ITER``, and an ``if`` in a
    ``while`` whose test is plain Python from that test, on its own line.
    """
    line = _line(listing[asked])
    if line is None:
        return False
    first = asked
    while first > 0 and _line(listing[first - 1]) in (line, None):
        if listing[first - 1].opname in _STATEMENT_ENDS:
            break
        first -= 1
    span = range(listing[first].offset, listing[asked].offset + 1)
    for instruction in listing[jump + 1 :]:
        if instruction.opname != "JUMP_BACKWARD" or instruction.argval not in span:
            continue
        start = index_of.get(instruction.argval)
        if start is not None and listing[start].opname != "FOR_ITER":
            return True
    return False


def _line(instruction: Any) -> int | None:
    """The source line an instruction is on, if it is on one."""
    positions = getattr(instruction, "positions", None)
    return getattr(positions, "lineno", None)
