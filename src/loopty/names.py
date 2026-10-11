"""The names of isl dimensions: the invariant loopty keeps, and how they change.

loopty builds its isl objects mostly from text and refers to their dimensions
by name: a statement's domain over its loop variables, with the sizes as
parameters, a sum's domain over its binders, the cells of an index type, an
access relation from the loops to the cells. isl itself is positional, and
its reader breaks a name silently. A tuple variable spelled like a parameter
is read as that parameter, so the dimension comes back unnamed and equal to
it: ``[k] -> { [k] : 0 <= k < 5 }`` is one point per ``k``, and a map's
dimension named like a declared size was read so (#110). Two dimensions of
one name are two variables isl prints apart (``[i, i']``), and anything that
finds a dimension by its name then finds the first.

namedisl, the isl in which names matter that loopy's next release builds on
(``docs/loopy-notes.md``, and the reading of it in the project's notes),
keeps three invariants, and loopty keeps them too wherever it names
dimensions:

* every parameter, set, ``in`` and ``out`` dimension has a name;
* no two dimensions of an object share one;
* so no set, ``in`` or ``out`` dimension is named like a parameter.

:func:`check_names` checks them, and raises :class:`NamingError`, where such
an object is made: a statement's and a sum's domain when the term records it
(:class:`loopty.term.Stmt`, :class:`loopty.term.Reduction`), the cells of an
index type (:func:`loopty.idx.to_set`), an access relation
(:func:`loopty.flow.access_relation`), and whatever :func:`rename` returns.
A breach is loopty's own bug, not the kernel's, and the error says what the
object was and which names clash.

What is positional by design is left out, named in ``positional``: the cells
an access reaches are ``x[i + 1]``'s index expressions, which have no names,
and the padded instance space and its logical time (:mod:`loopty.flow`) are
positions, the statement's index among them, so that dependences and
schedules are plain maps. A dimension of such a tuple may be unnamed, and
its name, if it has one, is no variable of the object's.

:func:`rename` is the one way loopty renames dimensions and replaces
parameters, all at once: composing a program renames each callee's loop
variables and sizes apart and replaces a size by the caller's expression
(:mod:`loopty.compose`), the lowering renames the binders of a nested sum
(:mod:`loopty.lower`), and the native run fixes a parameter at the value it
reads (:mod:`loopty.interpret`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from typing import Any

import islpy as isl
import numpy as np
import pymbolic.primitives as prim

__all__ = [
    "NamingError",
    "check_names",
    "rename",
]


class NamingError(ValueError):
    """An isl object loopty built breaks the invariants of :func:`check_names`."""


def _kinds(obj: Any) -> tuple[tuple[Any, str], ...]:
    """The dimension types of ``obj`` besides its parameters, and their names.

    isl's set and out types are one type, ``isl.dim_type.set`` and
    ``isl.dim_type.out`` alike, so the object says which it is.
    """
    if isinstance(obj, isl.Map | isl.BasicMap):
        return ((isl.dim_type.in_, "in dimension"), (isl.dim_type.out, "out dimension"))
    return ((isl.dim_type.set, "set dimension"),)


def check_names(
    obj: Any, what: str = "", positional: Collection[Any] = ()
) -> Any:
    """``obj``, once every dimension of it has a name of its own; see the module.

    A parameter and each set dimension (``in`` and ``out`` of a map) has a
    name, and no two share one, so none is named like a parameter. A
    dimension type in ``positional`` is left out: its dimensions may be
    unnamed, and their names are no variables. ``what`` says what ``obj`` is,
    for the message of the :class:`NamingError` raised otherwise. Returns
    ``obj``, so a constructor can return what it checked.
    """
    kinds = ((isl.dim_type.param, "parameter"), *_kinds(obj))
    seen: list[tuple[str, str]] = []
    unnamed: list[str] = []
    for kind, shown in kinds:
        if kind in positional and kind != isl.dim_type.param:
            continue
        for position in range(obj.dim(kind)):
            name = obj.get_dim_name(kind, position)
            if name is None:
                unnamed.append(f"{shown} {position}")
            else:
                seen.append((name, shown))
    counts = Counter(name for name, _kind in seen)
    clashes = sorted(name for name, count in counts.items() if count > 1)
    if not unnamed and not clashes:
        return obj
    about = f"{what}, " if what else ""
    problems = []
    if unnamed:
        problems.append(
            f"the {', '.join(unnamed)} {'has' if len(unnamed) == 1 else 'have'} "
            "no name (isl reads a tuple variable spelled like a parameter as "
            "that parameter, and names the dimension nothing)"
        )
    for name in clashes:
        kinds_of = sorted({shown for seen_name, shown in seen if seen_name == name})
        problems.append(
            f"{name} names more than one dimension ({', '.join(kinds_of)})"
        )
    raise NamingError(
        f"{about}{obj}: {'; and '.join(problems)}. Every dimension loopty builds "
        "has a name of its own, which no parameter has (loopty.names); this "
        "one does not, which is a bug of loopty's: please report it with the "
        "kernel"
    )


def rename(
    obj: Any,
    names: Mapping[str, str] | None = None,
    exprs: Mapping[str, Any] | None = None,
    what: str = "a set renamed",
) -> Any:
    """``obj`` with dimensions renamed and parameters replaced, all at once.

    ``names`` maps a dimension's name, a parameter's or a set dimension's, to
    its new one, and ``exprs`` a parameter's to what it stands for: an
    integer, or an affine expression of other parameters. Every renaming
    happens at once: ``a -> b`` beside ``b -> c`` gives no dimension the name
    ``b`` twice, as renaming one after the other would. A parameter renamed to
    a name the object already has as a parameter is that parameter: the two
    are equated, as replacing it by that parameter would. A parameter given an
    expression is replaced: ``obj`` is intersected with ``p = expression`` and
    ``p`` projected out, which is exact, since an equality determines it; an
    integer is fixed and projected out without isl's reader. A set dimension
    renamed to a name a parameter has, or a name another dimension keeps, is
    a clash, which :func:`check_names` refuses. ``obj`` is a set; the result
    is checked, as ``what``.
    """
    names = dict(names or {})
    exprs = dict(exprs or {})
    moved: list[tuple[str, Any, str]] = []
    for kind in (isl.dim_type.param, *(kind for kind, _shown in _kinds(obj))):
        for position in range(obj.dim(kind)):
            original = obj.get_dim_name(kind, position)
            if original in names or original in exprs:
                placeholder = f"__loopty_moved_{len(moved)}"
                obj = obj.set_dim_name(kind, position, placeholder)
                moved.append((placeholder, kind, original))
    replaced: list[tuple[str, Any]] = []
    for placeholder, kind, original in moved:
        if original not in names:
            replaced.append((placeholder, exprs[original]))
            continue
        target = names[original]
        present = set(obj.get_var_names(isl.dim_type.param))
        if target in present and kind == isl.dim_type.param:
            replaced.append((placeholder, prim.Variable(target)))
            continue
        position = obj.find_dim_by_name(kind, placeholder)
        obj = obj.set_dim_name(kind, position, target)
    for placeholder, value in replaced:
        obj = _replace(obj, placeholder, value)
    return check_names(obj, what)


def _replace(obj: Any, name: str, value: Any) -> Any:
    """``obj`` with its parameter ``name`` replaced by ``value``; see :func:`rename`."""
    position = obj.find_dim_by_name(isl.dim_type.param, name)
    if isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_):
        obj = obj.fix_val(isl.dim_type.param, position, int(value))
        return obj.project_out(isl.dim_type.param, position, 1)
    from loopty.flow import free_names

    text = _isl_text(value)
    params = [name, *sorted(free_names(text) - {name})]
    kind = isl.BasicSet if isinstance(obj, isl.BasicSet) else isl.Set
    constraint = kind(f"[{', '.join(params)}] -> {{ : {name} = {text} }}")
    obj = obj.align_params(constraint.get_space())
    constraint = constraint.align_params(obj.get_space())
    obj = obj.intersect_params(constraint)
    position = obj.find_dim_by_name(isl.dim_type.param, name)
    return obj.project_out(isl.dim_type.param, position, 1)


def _isl_text(value: Any) -> str:
    """``value`` in isl's syntax, or a refusal: a domain holds integers only."""
    from loopty.compose import _isl_text as text

    return text(value)
