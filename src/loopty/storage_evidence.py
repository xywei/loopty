"""Retained storage-phase checkpoints, distinct from final-code correctness."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from lanky.ledger import Fact, Status


@dataclass(frozen=True)
class StorageCheckpoint:
    """Actual objects at one window rewrite and its originating schedule snapshot.

    The binding to a schedule records where this phase was captured. It does
    not verify the dependency ordering or code generation performed afterward.
    Historical checkpoints remain available, but cannot establish a claim about
    a new snapshot without replaying the transformation.
    """

    request: Any
    result: Any
    source: Any
    prerequisites: tuple[str, ...]
    identifier: str
    where: str
    owner: str
    binding: object | None = None
    snapshot_key: str = ""

    def bind(self, schedule: Any) -> StorageCheckpoint:
        from loopy.tools import LoopyKeyBuilder

        return replace(
            self,
            binding=schedule._storage_binding,
            snapshot_key=LoopyKeyBuilder()(schedule._kernel),
        )

    def fact(self, schedule: Any) -> Fact:
        from loopy.tools import LoopyKeyBuilder

        record = self.result.evidence
        key = LoopyKeyBuilder()
        return Fact(
            id=self.identifier,
            kind="storage-correspondence",
            statement=(
                f"the retained checkpoint is valid evidence for the window rewrite of "
                f"{self.request.array} preserving accesses and values "
                "before dependency ordering"
            ),
            term=StorageCheck(self, schedule),
            provenance={
                "scope": "window rewrite before dependency ordering",
                "predicate": "retained checkpoint validity",
                "target": self.request.target,
                "input_key": key(self.request.kernel),
                "output_key": key(self.result.kernel),
                "snapshot_key": self.snapshot_key,
                "schema": None if record is None else record.schema,
                "versions": None if record is None else list(record.versions),
                "label": f"storage rewrite of {self.request.array}",
            },
            where=self.where,
            owner=self.owner,
            rests_on=self.prerequisites,
        )


@dataclass(frozen=True)
class StorageCheck:
    checkpoint: StorageCheckpoint
    schedule: Any

    def __repr__(self) -> str:
        return (
            f"StorageCheck({self.checkpoint.request.array!r}, "
            f"snapshot={self.checkpoint.snapshot_key!r}, phase='window rewrite')"
        )


class StorageOracle:
    """Recheck checkpoint structure; neither compilation nor hashes prove it."""

    name = "loopty-storage"

    def trust_class(self) -> str:
        return "decision-procedure"

    def can_establish(self, fact: Fact) -> bool:
        return isinstance(fact.term, StorageCheck)

    def establish(self, fact: Fact) -> Fact:
        if not self.can_establish(fact):
            return fact
        from loopy.tools import LoopyKeyBuilder

        from loopty.backend_evidence import (
            InvalidStorageEvidence,
            WindowEvidence,
            _versions,
        )

        check = fact.term
        checkpoint = check.checkpoint
        schedule = check.schedule
        if (
            checkpoint.binding is not schedule._storage_binding
            or checkpoint.snapshot_key != LoopyKeyBuilder()(schedule._kernel)
        ):
            # The historical check may still be valid. Its applicability to
            # this schedule is unknown, not a refutation of program semantics.
            return fact.with_status(
                Status.ASSUMED,
                self.name,
                declined="storage checkpoint belongs to a different schedule snapshot",
                stale=True,
            )
        result = checkpoint.result
        if result.evidence is None:
            return fact.with_status(
                Status.ASSUMED,
                self.name,
                declined=result.evidence_limit
                or result.reason
                or "storage evidence unavailable",
            )
        if (
            not isinstance(result.evidence, WindowEvidence)
            or result.evidence.schema != 1
        ):
            return fact.with_status(
                Status.ASSUMED,
                self.name,
                declined="unsupported storage evidence schema",
            )
        if result.evidence.versions != _versions():
            return fact.with_status(
                Status.ASSUMED,
                self.name,
                declined="storage evidence was captured with different tool versions",
            )
        try:
            result.evidence.verify(checkpoint.request, result.kernel)
        except InvalidStorageEvidence as exc:
            return fact.with_status(Status.REFUTED, self.name, reason=str(exc))
        # Unexpected exceptions propagate to lanky's dispatcher, which keeps
        # them as raised errors rather than manufacturing a counterexample.
        return fact.with_status(
            Status.DECIDED,
            self.name,
            trust_class=self.trust_class(),
            detail="actual retained input/output structure rechecked",
        )
