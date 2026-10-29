# Experimental storage plans

`contract` remains the convenient operation for an intermediate's live window.
`Schedule.storage(value)` builds a request scoped to a logical value and a
schedule snapshot. A stage is a residence of that value, with its own requested
memory space, fill context and owners. Constructing a plan does not change the
schedule; `schedule.materialize(plan)` returns a new schedule when supported.

For the fused Burgers example in `tests/kernels/windows.py`:

```python
plan = fused.storage(burgers.f)
window = plan.stage(name="window", placement=lambda j: j % 3, registers=False)
windowed = fused.materialize(plan)
```

This uses the same legality checks, lowering, numeric behavior and history as
`fused.contract(burgers.f, lambda j: j % 3, registers=False)`. Two slots still
fail with the source read and overwrite. A failed application preserves the
source schedule. A plan must be rebuilt after changing its schedule snapshot;
this prototype does not yet transport explicit stages through transformations.

The builder also represents multiple stages and explicit consumer routing:

```python
plan = fused.storage(burgers.f)
tile = plan.stage(name="tile", uses=burgers.divergence, memory="local")
fragment = plan.stage(
    name="fragment", source=tile, uses=burgers.divergence, memory="private"
)
plan.serve(burgers.divergence, from_=fragment)
request = plan.inspect()
```

`uses` requests a footprint; `serve` explicitly selects the supplier. Stage
declaration order never selects a consumer's supplier. A source stage must
belong to the same plan. `at`, `per` and `reuse_over` accept scoped loop handles:
owner and reuse coordinates must belong to the fill context and cannot overlap.

**The multistage request above is inspectable but cannot yet be lowered.**
Inspection reports requested controls and a `lowering_limit`, not an inferred
footprint or a proof of legality. Current released loopy supports only the
single-window subset of this interface: no explicit consumer selection,
routing, fill contexts, owners, reuse axes or memory choice. Applying any of
those requests raises `UnsupportedStoragePlan`; it does not silently call
`contract` and discard the extra controls. `memory="private"` is not a promise
of register allocation. `registers` retains contraction's existing meaning.

The remaining work includes inferred footprints, value versions, ownership and
coverage checks, multistage lowering, synchronization, and evidence transport
after fusion, splitting, retagging and retargeting. This is an experimental
surface for that work, not a completed richer materialization interface.

# Backend correspondence

Window lowering now takes an internal `WindowRequest` and returns a
`StorageResult`. The current implementation remains the default, with no new
dependency or automatic upstream-version dispatch.

For modulo windows, a separate checker reconstructs each retained instruction
from the actual input and requested placement and compares the actual output.
It checks the retained arithmetic, guards, domains, assumptions, dtype and
kernel properties. Its record binds the input, output, request and dependency
versions; replay rechecks structure, not just hashes. Corrupting an expression,
removing an instruction or changing a dtype fails even with a rebound digest.

This check is deliberately narrow. Source-level definedness licenses removal
of initialization, and no-clobber establishes lifetime legality. Register-shift
correspondence is explicitly unsupported. Later dependency ordering and loopy
scheduling need separate checks. The result is currently consumed at the
lowering boundary, not retained as a new ledger fact or certified artifact.
Consequently this does not complete source-to-generated-code assurance.
