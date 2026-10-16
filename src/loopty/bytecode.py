"""What a body's bytecode says about the statement running it.

A kernel body is run, not parsed, and two questions about the statement that
is running are answered by the instructions around the one a frame is at.

* **The names a ``for`` binds.** ``for box, a, b in R.dom:`` asks its domain
  for an iterator and unpacks each point into three names, and the iterator
  is the one to hand it points of three loops; ``for box in R.dom:`` is the
  one loop (:func:`for_targets`). The names are those of the source, which
  keeps the term's loop variables the source's.
* **Whether a truth value closes a loop.** ``while a < 2:`` asks ``a < 2``
  for its truth again at the end of each pass, and the jump that takes the
  answer leads back to the loop's start (:func:`loops_back`), which is how a
  ``while`` on a traced condition is told from an ``if``.

The instruction set is CPython's, of 3.12 and 3.13; an instruction layout
neither has is no answer, ``None`` or ``False``, and the caller falls back on
what it did before.
"""

from __future__ import annotations

import dis
import functools
from typing import Any

__all__ = ["for_targets", "loops_back"]

#: Instructions that do nothing, which 3.14 puts after a conditional jump.
_INERT = ("NOT_TAKEN", "NOP")

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

    A ``while`` tests its condition at its start and again at the end of
    each pass, and the second test is a conditional jump out of the loop
    followed by the jump back to its start: when the condition holds,
    control falls through to ``JUMP_BACKWARD``. An ``if`` is followed by its
    own block, or, last in a ``for`` loop's body, by the jump back to the
    ``FOR_ITER`` that takes the next point, which is no ``while``. The truth
    value is asked for by the jump itself (3.12), or by the ``TO_BOOL`` or the
    comparison just before it (3.13).
    """
    listing, index_of = _instructions(code)
    position = index_of.get(offset)
    if position is None:
        return False
    if not listing[position].opname.startswith("POP_JUMP_IF"):
        position = _next(listing, position + 1)
        if position >= len(listing) or not listing[position].opname.startswith(
            "POP_JUMP_IF"
        ):
            return False
    jump = listing[position]
    after = _next(listing, position + 1)
    if after >= len(listing):
        return False
    back = listing[after]
    target = jump.argval if isinstance(jump.argval, int) else -1
    if back.opname != "JUMP_BACKWARD" or target <= back.offset:
        return False
    start = index_of.get(back.argval)
    return start is None or listing[start].opname != "FOR_ITER"
