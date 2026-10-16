"""Where loopty's isl objects meet loopy's: islpy on one side, namedisl on the other.

loopy's main branch keeps a kernel's domains and assumptions, and takes and
gives every isl object of its interface, as namedisl objects (loopy #1023):
an isl object paired with the names of its dimensions, which namedisl's
operations align by name, sorting them, and which may not be unnamed or
shared. A raw islpy object passed to ``lp.assume`` is deprecated, and most of
loopy's helpers (``set_to_cond_expr``, ``pw_aff_to_expr``, ``aff_from_expr``)
take only namedisl's.

loopty stays on islpy. Its analysis, :mod:`loopty.flow`, :mod:`loopty.typing`,
:mod:`loopty.hypotheses`, :mod:`loopty.oracle` and :mod:`loopty.domain`, builds
isl objects from text, reasons by position where it means to (the padded
instance space and the statement index in it), and wants a mismatch of
dimensions to fail rather than to broadcast, which is what aligning by name
does. Its objects already keep namedisl's invariants (:mod:`loopty.names`), so
each one converts to namedisl without loss, and back.

This module is the one place the two meet, so that loopty's own modules never
import namedisl:

* :func:`to_named` gives loopy an isl object loopty built: a domain, the
  kernel's assumptions, a set loopy writes as a condition;
* :func:`as_isl` takes back an object loopy keeps or computes, a domain or a
  loop's bounds, as islpy;
* :func:`domains_of` reads a kernel's domains, and :func:`with_domains` gives it
  new ones, in loopy's order;
* :func:`aff_from_expr` is loopy's reader of an affine expression, asked over
  an islpy space, as loopty's rewrite of the loops along a map asks it.

A namedisl object handed to :func:`to_named`, and an islpy object handed to
:func:`as_isl`, come back as they are, so a caller need not know which side an
object it was given is on.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import islpy as isl
import namedisl as nisl

__all__ = [
    "aff_from_expr",
    "as_isl",
    "domains_of",
    "is_named",
    "to_named",
    "to_named_set",
    "with_domains",
]

#: Every namedisl type loopy's interface takes or gives.
_NAMED = (
    nisl.Aff,
    nisl.BasicMap,
    nisl.BasicSet,
    nisl.Constraint,
    nisl.Map,
    nisl.MultiAff,
    nisl.PwAff,
    nisl.PwMultiAff,
    nisl.PwQPolynomial,
    nisl.QPolynomial,
    nisl.Set,
)


def is_named(obj: Any) -> bool:
    """Whether ``obj`` is a namedisl object, which loopy main keeps and takes."""
    return isinstance(obj, _NAMED)


def to_named(obj: Any) -> Any:
    """The namedisl object over ``obj``, an isl object loopty built.

    Its dimensions keep their order and their names; namedisl refuses an
    object with an unnamed dimension or two of one name, which loopty's
    objects never have (:func:`loopty.names.check_names`). A namedisl object
    comes back as it is.
    """
    if is_named(obj):
        return obj
    return nisl.to_named(obj)


def to_named_set(obj: Any) -> nisl.Set:
    """:func:`to_named`, as a set: what loopy wants a domain and assumptions to be.

    loopy main keeps every domain as a ``namedisl.Set``, and refuses a basic
    set for the assumptions of a kernel it makes.
    """
    named = to_named(obj)
    if isinstance(named, nisl.BasicSet):
        return named.as_set()
    return named


def as_isl(obj: Any) -> Any:
    """The islpy object under ``obj``, an object loopy keeps or computes.

    Its dimensions carry their names, in the order namedisl keeps them in. An
    islpy object, or anything that is no namedisl object (an ``int`` bound,
    say), comes back as it is.
    """
    if is_named(obj):
        return obj.as_isl()
    return obj


def domains_of(entry: Any) -> list[isl.Set]:
    """The domains of a loopy kernel (not a translation unit), as islpy sets.

    In loopy's order, which is the one loopy reads their nesting from.
    """
    return [as_isl(domain) for domain in entry.domains]


def with_domains(entry: Any, new: Iterable[Any], **changes: Any) -> Any:
    """``entry`` over the domains ``new``, isl or namedisl, and ``changes``.

    ``entry.copy(domains=..., **changes)``, with each domain made the
    ``namedisl.Set`` loopy keeps.
    """
    return entry.copy(domains=[to_named_set(domain) for domain in new], **changes)


def aff_from_expr(space: isl.Space, expr: Any) -> isl.Aff:
    """loopy's reading of ``expr`` as one affine expression over ``space``.

    loopy main reads an expression over a mapping from each name to its
    affine expression, where loopy 2025.2 took the space itself; this asks it
    over the names of ``space``, in its order, and gives the expression back
    as islpy, over ``space``. Raises what loopy's ``aff_from_expr`` raises.
    """
    from loopy.symbolic import aff_from_expr as loopy_aff_from_expr

    over = to_named(isl.Set.universe(space))
    return as_isl(loopy_aff_from_expr(over.var_affs, expr))
