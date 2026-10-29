"""The storage adapter checks actual output, including deliberately corrupted code."""

from dataclasses import replace

import loopy as lp
import numpy as np
import pytest
from loopy.tools import LoopyKeyBuilder

from loopty.backend import WindowRequest, lower_window
from loopty.backend_evidence import InvalidStorageEvidence
from loopty.window import placement_of


def request():
    kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 8 }",
        "f[i] = 2*a[i] {id=fill}\ny[i] = f[i] {id=read,dep=fill}",
        [
            lp.GlobalArg("a", np.float64, shape=(8,)),
            lp.GlobalArg("y", np.float64, shape=(8,)),
            lp.TemporaryVariable("f", np.float64, shape=(8,)),
        ],
        target=lp.ExecutableCTarget(),
        lang_version=(2018, 2),
    )
    return WindowRequest(
        kernel,
        "f",
        (),
        placement_of(lambda i: i % 3, 1, "f", "window"),
        (3,),
        True,
        "c",
        False,
        ("fill",),
        {},
        frozenset(),
        ("i",),
        "window",
    )


def test_correspondence_is_bound_to_the_kernel_and_placement():
    req = request()
    result = lower_window(req)
    assert result.reason is None
    result.evidence.verify(req, result.kernel)
    with pytest.raises(InvalidStorageEvidence, match="different request"):
        result.evidence.verify(replace(req, extents=(4,)), result.kernel)
    with pytest.raises(InvalidStorageEvidence, match="different request"):
        result.evidence.verify(
            replace(req, kernel=lp.assume(req.kernel, "1 = 0")),
            result.kernel,
        )
    with pytest.raises(InvalidStorageEvidence, match="schema"):
        replace(result.evidence, schema=2).verify(req, result.kernel)


@pytest.mark.parametrize("mutation", ["read", "producer", "missing", "dtype", "domain"])
def test_even_a_rebound_digest_cannot_hide_a_corrupted_rewrite(mutation):
    req = request()
    result = lower_window(req)
    entry = result.kernel.default_entrypoint
    if mutation in {"read", "producer"}:
        target = "read" if mutation == "read" else "fill"
        entry = entry.copy(
            instructions=[
                insn.copy(expression=insn.expression + 1) if insn.id == target else insn
                for insn in entry.instructions
            ]
        )
    elif mutation == "missing":
        entry = entry.copy(
            instructions=[i for i in entry.instructions if i.id != "read"]
        )
    elif mutation == "dtype":
        temps = dict(entry.temporary_variables)
        name = result.evidence.storage_name
        temps[name] = temps[name].copy(dtype=np.float32)
        entry = entry.copy(temporary_variables=temps)
    else:
        entry = entry.copy(assumptions=entry.assumptions.empty(entry.assumptions.space))
    bad = result.kernel.with_kernel(entry)
    with pytest.raises(InvalidStorageEvidence, match="different request"):
        result.evidence.verify(req, bad)
    forged = replace(result.evidence, output_key=LoopyKeyBuilder()(bad))
    with pytest.raises(InvalidStorageEvidence, match="does not preserve"):
        forged.verify(req, bad)


def test_an_unsupported_register_check_never_claims_correspondence():
    result = lower_window(
        replace(
            request(),
            registers=True,
            extents=(1,),
            placement=placement_of(lambda i: i % 1, 1, "f", "window"),
        )
    )
    assert result.kernel is not None
    assert result.evidence is None
    assert result.evidence_limit == "register-shift correspondence is not checked"
