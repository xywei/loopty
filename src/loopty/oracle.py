"""The isl oracle: a decision procedure with witnesses.

Types are isl objects, so type checking is emptiness, subset, bijectivity, and
one more question that belongs to scheduling: is the new execution order monotone
on the dependence relation? All four are decidable in Presburger arithmetic, and
isl decides them. That is the oracle's trust class: ``decision-procedure``, below
a Lean kernel proof and above a property test, and re-derivable later by ``omega``
because the same statements are expressible in Lean.

What makes the verdicts usable is the witness. A failed check returns a concrete
point of the offending set, or a concrete pair of statement instances, so a
rejected schedule can say *which* two instances it would reorder rather than
merely refusing. Every primitive here returns a :class:`Verdict` carrying that
witness.

The fact-level entry points (``can_establish``, ``establish``) dispatch on the
four question types below. A typing rule states an obligation by putting one of
them in a fact's ``term``; the oracle answers it and writes back ``DECIDED`` or
``REFUTED``, and a refuted fact carries the witness in its provenance, labelled
with the names of the coordinates so that the ledger can print ``S0[r=4]``
rather than ``(0, 4)``. A fact whose term is not one of them is declined, which
leaves it to a weaker oracle (the property tester) or to ``ASSUMED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import islpy as isl
from lanky.ledger import Status

__all__ = [
    "TRUST_CLASS",
    "Bijective",
    "Empty",
    "IslOracle",
    "IslQuestion",
    "Monotone",
    "Subset",
    "Verdict",
    "decide",
    "is_bijective",
    "is_empty",
    "is_monotone",
    "is_subset",
    "label_witness",
    "sample_pair",
    "sample_parameters",
    "sample_point",
]

#: How much a verdict from this oracle should be trusted. isl is a complete
#: decision procedure for the fragment, but it is not a proof-checking kernel.
TRUST_CLASS = "decision-procedure"


@dataclass(frozen=True)
class Verdict:
    """The answer to one decidable question, with a witness when it fails.

    ``witness`` is a concrete counterexample: a point of an index space, or a
    pair of statement instances. ``parameters`` is the size valuation isl chose
    at the same time, which a parametric question needs: the cell ``a0=2`` is
    out of bounds only for some ``n``, and without the ``n`` the witness reads as
    if it were in bounds. ``detail`` is one line naming what went wrong, for the
    ledger's provenance record.
    """

    ok: bool
    witness: Any = None
    detail: str = ""
    parameters: dict[str, int] | None = None

    def __bool__(self) -> bool:
        return self.ok


def _sample_full(a_set: isl.Set) -> tuple[tuple[int, ...], dict[str, int]] | None:
    """Sample ``a_set`` once and read back both its point and its parameters.

    Sampling once matters: a parametric set has a point only *at* a parameter
    valuation, and reading the two from separate samples could report a cell and
    a size that do not belong together.
    """
    if a_set.is_empty():
        return None
    point = a_set.sample_point()
    n_dims = a_set.dim(isl.dim_type.set)
    coordinates = tuple(
        point.get_coordinate_val(isl.dim_type.set, k).to_python() for k in range(n_dims)
    )
    parameters = {}
    for k in range(a_set.dim(isl.dim_type.param)):
        name = a_set.get_dim_name(isl.dim_type.param, k) or f"p{k}"
        parameters[name] = point.get_coordinate_val(
            isl.dim_type.param, k
        ).to_python()
    return coordinates, parameters


def sample_point(a_set: isl.Set) -> tuple[int, ...] | None:
    """One concrete point of ``a_set``, or ``None`` if it is empty.

    isl's ``sample`` returns a basic set containing a single point; the
    coordinates are read out as Python ints so that a witness can be printed and
    compared without isl on the other end.
    """
    sampled = _sample_full(a_set)
    return None if sampled is None else sampled[0]


def sample_parameters(a_set: isl.Set | isl.Map) -> dict[str, int]:
    """The parameter valuation at which ``a_set`` has the point ``sample_point``
    returns.

    Empty for a set with no parameters, and for an empty set, which has no
    witness to place.
    """
    if isinstance(a_set, isl.Map):
        a_set = a_set.wrap()
    sampled = _sample_full(a_set)
    return {} if sampled is None else sampled[1]


def sample_pair(a_map: isl.Map) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """One concrete ``(source, target)`` pair of ``a_map``, or ``None``.

    The map is wrapped into a set of pairs, sampled, and split back at the
    domain dimension, which is how a dependence violation becomes a pair of
    statement instances.
    """
    sampled = _sample_pair_full(a_map)
    return None if sampled is None else sampled[0]


def _sample_pair_full(
    a_map: isl.Map,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], dict[str, int]] | None:
    """:func:`sample_pair` and the parameters it holds at, from one sample."""
    if a_map.is_empty():
        return None
    n_in = a_map.dim(isl.dim_type.in_)
    sampled = _sample_full(a_map.wrap())
    if sampled is None:  # pragma: no cover - is_empty already ruled this out
        return None
    coordinates, parameters = sampled
    return (coordinates[:n_in], coordinates[n_in:]), parameters


def _witness(obj: isl.Set | isl.Map) -> tuple[Any, dict[str, int]]:
    """A counterexample and the parameter valuation it is one at, sampled once.

    Every refutation below reports a point (or a pair of instances) together
    with the sizes it was read off at, and the two only belong together if they
    come from the same sample: asking isl twice leaves it free to answer with
    two different points, and a cell reported outside an array at a size where
    it is inside is a counterexample to nothing. A map is sampled as a pair.
    """
    if isinstance(obj, isl.Map):
        sampled = _sample_pair_full(obj)
    else:
        sampled = _sample_full(obj)
    if sampled is None:
        return None, {}
    return sampled


def _at_text(parameters: dict[str, int] | None) -> str:
    """`` at [n=1]``, or the empty string when there is nothing to say."""
    if not parameters:
        return ""
    return " at [" + ", ".join(f"{k}={v}" for k, v in parameters.items()) + "]"


def _identity_like(a_map: isl.Map) -> isl.Map:
    """The identity map on ``a_map``'s domain space."""
    return isl.Map.identity(a_map.get_space().domain().map_from_set())


def is_empty(a_set: isl.Set | isl.Map) -> bool:
    """Is the set (or map) empty? The base question every other one reduces to."""
    return bool(a_set.is_empty())


def is_subset(small: isl.Set, large: isl.Set) -> Verdict:
    """Is every point of ``small`` a point of ``large``?

    This is the in-bounds check: ``small`` is the set of addresses an access can
    produce, ``large`` is the set the array has. The witness is a point that the
    access reaches and the array does not.
    """
    outside = small.subtract(large)
    if outside.is_empty():
        return Verdict(True, None, "subset")
    point, parameters = _witness(outside)
    at = _at_text(parameters)
    return Verdict(
        False,
        point,
        f"point {point} is in the first set but not in the second{at}",
        parameters,
    )


def is_bijective(a_map: isl.Map) -> Verdict:
    """Is ``a_map`` a bijection between its domain and its range?

    This is the obligation of a reindexing cast: split, tile, interchange, and
    skew are admissible exactly when they rename statement instances one for one.
    The witness distinguishes the two failure modes. Not injective: two domain
    instances that would collapse onto one. Not single-valued: one instance sent
    to two places.
    """
    if a_map.is_bijective():
        return Verdict(True, None, "bijective")
    if not a_map.is_injective():
        collisions = a_map.apply_range(a_map.reverse()).subtract(_identity_like(a_map))
        pair, parameters = _witness(collisions)
        return Verdict(
            False,
            pair,
            f"not injective: instances {pair[0]} and {pair[1]} share an image"
            f"{_at_text(parameters)}"
            if pair
            else "not injective",
            parameters,
        )
    inverse = a_map.reverse()
    splits = inverse.apply_range(inverse.reverse()).subtract(_identity_like(inverse))
    pair, parameters = _witness(splits)
    return Verdict(
        False,
        pair,
        f"not single-valued: one instance reaches both {pair[0]} and {pair[1]}"
        f"{_at_text(parameters)}"
        if pair
        else "not single-valued",
        parameters,
    )


def is_monotone(schedule: isl.Map, deps: isl.Map) -> Verdict:
    """Does ``schedule`` order every dependence forward in time?

    ``schedule`` maps statement instances to logical time (the loop nest, one
    dimension per level); ``deps`` maps a source instance to a sink instance that
    must run after it. The schedule is legal when, for every dependence, the
    source's time is lexicographically strictly before the sink's.

    The check runs in time space and the witness is pulled back to instance
    space: the violating times are mapped back through the inverse schedule and
    intersected with the dependences, so the pair reported is a real source and
    sink of the program, which is what the tutorial's rejected tiling prints.

    Self-dependences are dropped first: an instance never has to run before
    itself. Parallel (``g.*``, ``l.*``) inames are unordered and must be removed
    from the schedule by the caller before asking, because this question is about
    a sequential order.
    """
    deps = deps.subtract(_identity_like(deps))
    if deps.is_empty():
        return Verdict(True, None, "no dependences to violate")

    timed = deps.apply_domain(schedule).apply_range(schedule)
    before = isl.Map.lex_lt(timed.get_space().domain())
    violating_times = timed.subtract(before)
    if violating_times.is_empty():
        return Verdict(True, None, "schedule is monotone on the dependences")

    inverse = schedule.reverse()
    bad = violating_times.apply_domain(inverse).apply_range(inverse).intersect(deps)
    sampled = bad if not bad.is_empty() else violating_times
    pair, parameters = _witness(sampled)
    return Verdict(
        False,
        pair,
        f"dependence {pair[0]} -> {pair[1]} is not ordered forward by the schedule"
        f"{_at_text(parameters)}"
        if pair
        else "the schedule reorders a dependence",
        parameters,
    )


# {{{ the questions a typing rule may ask


@dataclass(frozen=True)
class Empty:
    """Is this set (or map) empty?

    The shape of a disjointness obligation: the set is the *bad* case, the pairs
    of instances that write the same cell, so emptiness is the good news and a
    sample point is the counterexample.
    """

    obj: Any
    description: str = ""
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Subset:
    """Is every point of ``small`` a point of ``large``?

    The shape of an in-bounds obligation: ``small`` is the set of cells an
    access reaches, ``large`` is the set of cells the array has.
    """

    small: Any
    large: Any
    description: str = ""
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Bijective:
    """Is this map a bijection?

    The shape of a reindexing cast: split, tile, interchange and skew are
    admissible exactly when they rename statement instances one for one.
    """

    map: Any
    description: str = ""
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Monotone:
    """Does ``schedule`` run every dependence of ``deps`` forward in time?

    The shape of a legality obligation, asked of the source order by the typing
    rules and of every proposed schedule by :mod:`loopty.schedule`.
    """

    schedule: Any
    deps: Any
    description: str = ""
    labels: tuple[str, ...] = ()


#: The questions :class:`IslOracle` answers.
IslQuestion = Empty | Subset | Bijective | Monotone


def label_witness(witness: Any, labels: tuple[str, ...]) -> str:
    """Render a witness with the names of its coordinates.

    A witness is a tuple of integers, or a pair of them; the labels are the
    dimension names of the space it lives in, which is what turns ``(0, 4, 0)``
    into ``s=0, d0=4, d1=0`` and makes a rejected schedule readable.
    """
    if witness is None:
        return ""
    if (
        isinstance(witness, tuple)
        and len(witness) == 2
        and all(isinstance(half, tuple) for half in witness)
    ):
        return " -> ".join(label_witness(half, labels) for half in witness)
    if not labels or len(labels) != len(witness):
        return "(" + ", ".join(str(c) for c in witness) + ")"
    pairs = ", ".join(f"{n}={v}" for n, v in zip(labels, witness, strict=True))
    return f"[{pairs}]"


def decide(question: IslQuestion) -> Verdict:
    """Answer one question with the matching decision primitive."""
    if isinstance(question, Subset):
        return is_subset(question.small, question.large)
    if isinstance(question, Bijective):
        return is_bijective(question.map)
    if isinstance(question, Monotone):
        return is_monotone(question.schedule, question.deps)
    if isinstance(question, Empty):
        obj = question.obj
        if obj.is_empty():
            return Verdict(True, None, "empty")
        witness, parameters = _witness(obj)
        text = label_witness(witness, question.labels)
        return Verdict(
            False,
            witness,
            f"{text} is in the set{_at_text(parameters)}",
            parameters,
        )
    raise TypeError(f"not an isl question: {question!r}")


# }}}


class IslOracle:
    """lanky's oracle for the Presburger fragment.

    Registered under the ``lanky.oracles`` entry-point group. The primitives
    above are the whole decision procedure; this class is the adapter that
    lanky's ledger talks to. The typing rules in :mod:`loopty.typing` state
    their obligations as :data:`IslQuestion` terms, and this class answers them,
    attaching the witness of a failure to the ``REFUTED`` fact's provenance so
    that the ledger can say which cell or which pair of statement instances went
    wrong.
    """

    name = "isl"

    #: Fact kinds this oracle recognizes even before looking at the term.
    KINDS = ("empty", "subset", "bijective", "monotone")

    def trust_class(self) -> str:
        """``"decision-procedure"``: complete for the fragment, not a kernel."""
        return TRUST_CLASS

    def can_establish(self, fact: Any) -> bool:
        """Can this oracle decide ``fact``?

        True for a fact whose term is one of the four questions, and for the
        four kind names as a courtesy to a caller that has not built the term
        yet. Anything else -- a postcondition with multiplication in it, a claim
        about floating point -- belongs to a stronger or a weaker oracle, and
        saying so here is what lets lanky try them strongest first.
        """
        if isinstance(getattr(fact, "term", None), IslQuestion):
            return True
        return getattr(fact, "kind", None) in self.KINDS

    def establish(self, fact: Any) -> Any:
        """Decide ``fact`` and return it with a status and a witness.

        Returns ``None`` to decline a fact whose term is not an isl question,
        which is how a fact merely *named* like one is passed on to the next
        oracle rather than silently failed.
        """
        question = getattr(fact, "term", None)
        if not isinstance(question, IslQuestion):
            return None
        verdict = decide(question)
        if verdict.ok:
            return fact.with_status(
                Status.DECIDED,
                self.name,
                detail=verdict.detail,
                question=type(question).__name__.lower(),
            )
        return fact.with_status(
            Status.REFUTED,
            self.name,
            witness=verdict.witness,
            witness_text=label_witness(verdict.witness, question.labels)
            + _at_text(verdict.parameters),
            witness_params=dict(verdict.parameters or {}),
            detail=verdict.detail,
            question=type(question).__name__.lower(),
        )

    # The primitives, also exposed as methods so that a plugin holding only the
    # oracle object can ask the questions directly.
    is_empty = staticmethod(is_empty)
    is_subset = staticmethod(is_subset)
    is_bijective = staticmethod(is_bijective)
    is_monotone = staticmethod(is_monotone)
    sample_point = staticmethod(sample_point)
    sample_pair = staticmethod(sample_pair)
    decide = staticmethod(decide)
