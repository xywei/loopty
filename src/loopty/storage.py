"""Experimental scoped storage plans; lowering support is deliberately explicit.

A Stage is a residence of a logical value, not a new source array. Plans are
bound to a schedule snapshot and validate handles when constructed. The current
released backend realizes the single-window subset; richer requests can be
inspected but cannot silently degrade to contraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loopty.handles import Call
from loopty.window import Placement, placement_of


class UnsupportedStoragePlan(ValueError):
    """A valid request whose lowering is not implemented on this backend."""


@dataclass(frozen=True, eq=False)
class Stage:
    name: str
    value: Any
    uses: tuple[Call, ...] | None
    at: tuple[Any, ...]
    per: tuple[Any, ...]
    memory: str
    reuse_over: tuple[Any, ...]
    source: Stage | None
    placement: Placement | None
    registers: bool | None
    _owner: object = field(repr=False)

    def __repr__(self) -> str:
        return f"{self.value!r}.storage.{self.name}"


class StoragePlan:
    """A builder scoped to one value and one unchanged schedule snapshot.

    Constructing or inspecting a plan never changes the schedule. Applying a
    supported plan returns another schedule; an invalid or unsupported plan
    leaves both the source and the builder available for correction.
    """

    def __init__(self, schedule: Any, value: Any):
        self._array = schedule._array(value, "storage")
        self._schedule = schedule
        self._value = value
        self._owner = object()
        self._stages: list[Stage] = []
        self._routes: dict[Call, Stage] = {}

    @property
    def stages(self) -> tuple[Stage, ...]:
        return tuple(self._stages)

    def _member(self, stage: Stage) -> None:
        if not isinstance(stage, Stage) or stage._owner is not self._owner:
            raise ValueError("the storage stage belongs to another plan")
        if not any(stage is member for member in self._stages):
            raise ValueError("the storage stage is not a declared member of this plan")

    def _uses(self, uses: Any) -> tuple[Call, ...] | None:
        if uses is None:
            return None
        found = (uses,) if isinstance(uses, Call) else tuple(uses)
        if not found:
            raise ValueError("uses must name a consumer, or be None for all consumers")
        for use in found:
            if not isinstance(use, Call):
                raise TypeError(
                    "uses takes scoped call handles, not names or internal IDs"
                )
            self._schedule._own(use, "storage uses")
        if len(set(found)) != len(found):
            raise ValueError("a consumer was selected more than once")
        return found

    def _coordinates(self, handles: Any, label: str) -> tuple[tuple, frozenset[str]]:
        handles = tuple(handles)
        resolved = [self._schedule._loop(h, f"storage {label}") for h in handles]
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"storage {label} names the same resolved loop twice")
        return handles, frozenset(resolved)

    def stage(
        self,
        *,
        name: str,
        uses: Any = None,
        at: tuple = (),
        per: tuple = (),
        memory: str = "auto",
        reuse_over: tuple = (),
        source: Stage | None = None,
        placement: Any = None,
        registers: bool | None = None,
    ) -> Stage:
        """Declare a stage; memory and sharing do not imply backend support.

        ``uses`` requests a footprint. :meth:`serve` selects the stage supplying
        those uses; an intermediate stage may instead supply another stage.
        ``auto`` retains current contraction's target-dependent memory choice.
        """
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("a stage name must be a Python identifier")
        if any(s.name == name for s in self._stages):
            raise ValueError(f"storage stage {name!r} is already declared")
        if memory not in {"auto", "global", "local", "private"}:
            raise ValueError("memory must be auto, global, local, or private")
        if registers is not None and not isinstance(registers, bool):
            raise TypeError("registers must be True, False, or None")
        if source is not None:
            self._member(source)
        at, contexts = self._coordinates(at, "at")
        per, owners = self._coordinates(per, "per")
        reuse_over, reuse = self._coordinates(reuse_over, "reuse_over")
        if not owners <= contexts or not reuse <= contexts:
            raise ValueError("per and reuse_over must name coordinates in at")
        if owners & reuse:
            raise ValueError(
                "a coordinate cannot both distinguish owners and reuse storage"
            )
        uses = self._uses(uses)
        typ = self._schedule.term.array_types[self._array]
        rank = len(typ.axes) if typ.domain is None else typ.domain.ndim
        placed = (
            None
            if placement is None
            else placement_of(placement, rank, self._array, f"storage stage {name}")
        )
        stage = Stage(
            name,
            self._value,
            uses,
            at,
            per,
            memory,
            reuse_over,
            source,
            placed,
            registers,
            self._owner,
        )
        self._stages.append(stage)
        return stage

    def serve(self, uses: Any, *, from_: Stage) -> None:
        self._member(from_)
        selected = self._uses(uses)
        if selected is None:
            raise ValueError("serve names explicit consumer calls")
        if from_.uses is not None and not set(selected) <= set(from_.uses):
            raise ValueError("the stage did not request a footprint for this consumer")
        if any(
            use in self._routes and self._routes[use] is not from_ for use in selected
        ):
            raise ValueError("a consumer already has a different supplying stage")
        self._routes.update(dict.fromkeys(selected, from_))

    def inspect(self) -> dict[str, Any]:
        """Requested controls only: no inferred footprint or proof is fabricated."""
        return {
            "value": repr(self._value),
            "stages": [
                {
                    "name": s.name,
                    "memory": s.memory,
                    "source": None if s.source is None else s.source.name,
                    "uses": None if s.uses is None else [repr(u) for u in s.uses],
                    "at": [repr(h) for h in s.at],
                    "per": [repr(h) for h in s.per],
                    "reuse_over": [repr(h) for h in s.reuse_over],
                    "placement": None if s.placement is None else str(s.placement),
                }
                for s in self._stages
            ],
            "routes": {repr(use): stage.name for use, stage in self._routes.items()},
            "lowering_limit": self._limit(),
        }

    def _limit(self) -> str | None:
        if len(self._stages) != 1:
            return "the released backend currently lowers exactly one window stage"
        stage = self._stages[0]
        if (
            stage.uses is not None
            or self._routes
            or stage.source is not None
            or stage.at
            or stage.per
            or stage.reuse_over
        ):
            return (
                "explicit consumer routing, fill contexts and storage owners "
                "need a richer backend"
            )
        if stage.memory != "auto":
            return "explicit memory placement is not yet implemented for storage plans"
        return None

    def apply(self, schedule: Any) -> Any:
        if schedule is not self._schedule:
            raise ValueError(
                "storage plan belongs to a different schedule snapshot; rebuild it"
            )
        reason = self._limit()
        if reason is not None:
            raise UnsupportedStoragePlan(reason)
        stage = self._stages[0]
        return schedule.contract(
            self._value, stage.placement, registers=stage.registers
        )
