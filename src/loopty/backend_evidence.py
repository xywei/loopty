"""Check a storage rewrite against the actual input and output kernels.

This checks correspondence, not lifetime legality: the latter is independently
stated by Schedule before lowering. The first supported fragment is modulo
windows. Register shifts require another correspondence rule and are explicitly
outside this check. No backend verdict or fingerprint is treated as a proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import loopy as lp
import pymbolic.primitives as p
from loopy.symbolic import IdentityMapper
from loopy.tools import LoopyKeyBuilder


class InvalidStorageEvidence(ValueError):
    """The supplied output does not implement the requested storage rewrite."""


def _versions() -> tuple[str, ...]:
    return tuple(version(package) for package in ("loopy", "islpy", "pymbolic"))


def _request_key(request: Any, name: str, space: Any) -> str:
    return LoopyKeyBuilder()(
        (
            request.array,
            request.removed,
            request.placement.names,
            request.placement.slots,
            request.extents,
            request.target,
            request.registers,
            request.private,
            request.producers,
            request.nest,
            request.strides,
            name,
            space,
        )
    )


@dataclass(frozen=True)
class WindowEvidence:
    """A scoped record of a checked modulo rewrite, not a proof certificate.

    ``verify`` checks correspondence again; identities reject stale replay but
    never replace the check. Scheduling after this phase needs its own evidence.
    """

    input_key: str
    output_key: str
    request_key: str
    versions: tuple[str, ...]
    storage_name: str
    address_space: Any
    schema: int = 1

    def verify(self, request: Any, output: Any) -> None:
        if self.schema != 1 or self.versions != _versions():
            raise InvalidStorageEvidence(
                "storage evidence schema or checker inputs changed"
            )
        key = LoopyKeyBuilder()
        if (
            key(request.kernel) != self.input_key
            or key(output) != self.output_key
            or _request_key(request, self.storage_name, self.address_space)
            != self.request_key
        ):
            raise InvalidStorageEvidence(
                "storage evidence belongs to a different request or kernel"
            )
        _check_modulo(request, output, self.storage_name, self.address_space)


def check_modulo_window(
    request: Any, output: Any, name: str, space: Any
) -> WindowEvidence:
    """Check every retained instruction and all unchanged kernel properties."""
    _check_modulo(request, output, name, space)
    key = LoopyKeyBuilder()
    return WindowEvidence(
        key(request.kernel),
        key(output),
        _request_key(request, name, space),
        _versions(),
        name,
        space,
    )


def _check_modulo(request: Any, output: Any, name: str, space: Any) -> None:
    if output is None:
        raise InvalidStorageEvidence("storage output is missing")
    source = request.kernel
    before = source.default_entrypoint
    if request.array not in before.temporary_variables:
        raise InvalidStorageEvidence("the storage request does not name a temporary")
    if name in before.all_variable_names():
        raise InvalidStorageEvidence("the new storage name is not fresh")
    missing = set(request.removed) - set(before.id_to_insn)
    if missing:
        raise InvalidStorageEvidence("the request removes unknown instructions")
    # Removal is licensed by the source-level definedness fact, not by this
    # syntactic check. Preserve loopy's dependency repair for removed nodes.
    if request.removed:
        loops = set().union(
            *(before.id_to_insn[i].within_inames for i in request.removed)
        )
        source = lp.remove_instructions(source, set(request.removed))
        source = lp.remove_unused_inames(source, loops)
    before = source.default_entrypoint

    class ExpectedAccess(IdentityMapper):
        def map_subscript(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
            # Reconstruct from the input access, not from any relation the
            # lowerer supplies. All arithmetic outside these accesses remains.
            if expr.aggregate == p.Variable(request.array):
                indices = expr.index if isinstance(expr.index, tuple) else (expr.index,)
                slots = request.placement.at(tuple(self.rec(i) for i in indices))
                return p.Subscript(
                    p.Variable(name), slots[0] if len(slots) == 1 else slots
                )
            return super().map_subscript(expr, *args, **kwargs)

    mapper = ExpectedAccess()
    instructions = [
        instruction.with_transformed_expressions(mapper)
        for instruction in before.instructions
    ]
    temps = dict(before.temporary_variables)
    old = temps.pop(request.array)
    temps[name] = lp.TemporaryVariable(
        name,
        old.dtype,
        shape=request.extents,
        address_space=space,
    )
    expected = source.with_kernel(
        before.copy(
            instructions=instructions,
            temporary_variables=temps,
        )
    )
    # Compare structure as well as identity: an omitted instruction, changed
    # guard/domain/assumption, wrong dtype, new dependency or altered expression
    # must not be hidden behind a matching access summary.
    if expected != output:
        raise InvalidStorageEvidence(
            "the output does not preserve the requested accesses and values "
            f"of {request.array}"
        )
