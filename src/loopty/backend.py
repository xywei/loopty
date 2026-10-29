"""Internal loopy operations, with explicit requests and support limits.

Schedule owns the source obligations. This adapter owns storage realization;
additional operations can accept their own request types without exposing
backend-specific constructors to the user. Capability absence is different
from a failed check: failures never select a weaker implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import loopy as lp

from loopty import window as win


@dataclass(frozen=True)
class WindowRequest:
    """An already checked source window to realize on the supported backend."""

    kernel: Any
    array: str
    removed: tuple[str, ...]
    placement: win.Placement
    extents: tuple[Any, ...]
    private: bool
    target: str
    registers: bool | None
    producers: tuple[str, ...]
    shown: Mapping[str, str]
    strides: frozenset[str]
    nest: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class StorageResult:
    kernel: Any
    reason: str | None = None
    evidence: Any = None
    evidence_limit: str | None = None


def lower_window(request: WindowRequest) -> StorageResult:
    """Realize a window using the existing released-loopy implementation.

    Preserve automatic register selection and target buildability reporting.
    No optional compute import or version-dependent default dispatch occurs.
    """
    entry = request.kernel.default_entrypoint
    name = win.window_name(request.array, set(entry.all_variable_names()))
    space = (
        lp.AddressSpace.GLOBAL
        if request.target == "opencl" and not request.private
        else lp.AddressSpace.PRIVATE
    )
    register = None
    if request.registers or (request.registers is None and request.target == "opencl"):
        plan = win.register_plan(
            request.kernel,
            request.array,
            request.producers,
            request.placement,
            request.removed,
            request.shown,
        )
        if not isinstance(plan, str) and plan[0] in request.strides:
            plan = (
                f"the loop {plan[0]} steps over a lattice, and a shift "
                "register shifts once per step of a loop over every value"
            )
        if isinstance(plan, str):
            if request.registers:
                raise ValueError(f"{request.description}: {plan}")
        else:
            register = plan
    outer = (
        request.nest[: request.nest.index(register[0])]
        if register is not None and register[0] in request.nest
        else ()
    )
    kernel, reason = win.windowed_kernel(
        request.kernel,
        request.array,
        request.removed,
        request.placement,
        request.extents,
        name,
        space,
        register,
        outer,
    )
    if reason is not None:
        return StorageResult(kernel, reason)
    if register is not None:
        return StorageResult(
            kernel,
            evidence_limit="register-shift correspondence is not checked",
        )
    from loopty.backend_evidence import check_modulo_window

    evidence = check_modulo_window(request, kernel, name, space)
    return StorageResult(kernel, evidence=evidence)
