# Changelog

All notable changes to loopty are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[PEP 440](https://peps.python.org/pep-0440/).

## [0.1.0.dev0] - 2026-09-18

The first release in which something works. A decorated kernel runs natively on
numpy, traces to a typed term, emits its obligations into lanky's ledger, lowers
through loopy, and runs on the C target. An illegal transformation is rejected
with a pair of statement instances.

### Added

- **Index types** (`loopty.idx`). `to_set(shape, params)` builds the isl set of a
  shape, with a non-affine axis size reflected into a fresh parameter, which
  widens soundly. `normalize` realises `Fin[a*b]` as `Fin[a] x Fin[b]`.
  `Layout` is the affine map of a dense row- or column-major array and
  `RaggedLayout(offsets)` the map of a dependent-sum axis; `linearize` and
  `delinearize` are the index arithmetic.
- **Runtime arrays** (`loopty.arr`). `Arr` over numpy, dense and ragged, with
  `Arr.zeros`, `Arr.from_numpy` and `Arr.ragged(counts, values)`; `.dom` is the
  iterable domain and `.dom[i]` the fiber over a row, with that row's extent;
  `.type` is the array's index type and `.numpy()` the buffer.
- **The Term IR** (`loopty.term`). `Access`, `Reduction`, `Stmt`, `ArrType` and
  `Term`: the contract between tracing and lowering. A statement carries its
  inames, its isl domain, its assignee, its expression, its guard and its source
  location.
- **Tracing** (`loopty.trace`). The body runs once against symbolic arrays.
  Subscripting builds an access, assignment records a statement, iterating a
  symbolic `.dom` opens a loop level whose iname is the name the `for` statement
  wrote, `loopty.sum` becomes a reduction, and `with when(cond):` records a guard
  that is also intersected into the statement's domain when it is affine. A
  Python `if` on a computed value raises `TraceError` naming `when` as the fix.
- **Footprints and dependences** (`loopty.flow`). Every statement instance is a
  point of one padded, statement-tagged space, so footprints, schedules and
  dependences are plain isl maps. Time is the 2d+1 vector of the loop nest. A
  ragged bound is reflected as an isl parameter, which is sound because isl
  answers for every value of it.
- **Typing rules** (`loopty.typing`). In-bounds per access, decided by isl when
  the index is affine and *by type* when the index's element type is the axis's
  index type, which is what discharges `x[col[r, j]]` with no proof at all.
  Write disjointness, an ordering obligation for the source schedule, the
  exactness class of each reduction, and the postcondition left to weaker
  oracles.
- **The isl oracle** (`loopty.oracle`). `Empty`, `Subset`, `Bijective` and
  `Monotone` question terms and the decision primitives behind them. A refuted
  fact carries a concrete witness: a point, or a pair of statement instances
  pulled back from time space.
- **Kernels** (`loopty.kernel`). `@kernel` is inert and registering: it runs
  natively, caches its term, and its `KernelTheory` turns it into facts under
  `lanky check`. `@program` sequences kernel calls and restates each callee's
  postcondition as a fact in scope.
- **Lowering** (`loopty.lower`). A term becomes an `lp.make_kernel`: domains from
  the statements, one assignment each with data-derived dependences, reductions
  as `lp.Reduction`, and a ragged axis as a flat buffer plus an offsets argument
  indexed `off[r] + j`, which is a genuine CSR loop.
- **Transformations as casts** (`loopty.schedule`). `Schedule(kernel).tag`,
  `.split`, `.interchange`, `.prioritize`, `.tile`, `.skew` and `.realize`. Each
  states its reindexing as an isl map, checks it is a bijection on statement
  instances, and checks the new execution order is monotone on the dependence
  relation. Failure raises `IllegalCast` carrying the rendered witness and the
  refuted fact, and the message names the sizes the witness was read off at.
  `.facts()` yields the cast facts for the ledger.
- **A target-capability check** (`loopty.schedule`). Legal and buildable are
  different questions, and a step that passes the first can still fail the
  second. A parallel tag inside a loop whose bound comes from an array (a ragged
  fiber), or a reduction split across parallel and sequential inames, produces a
  `refuted` fact of kind `buildable` decided by `loopy-target` with the limit in
  words, and `UnbuildableSchedule` is raised as soon as anything asks the
  schedule for code. Both limits were measured on real devices; see
  `docs/device-runs.md`. The design's own spmv device schedule is the case.
- **`Schedule.retarget(target)`**. The same transformations replayed against
  another loopy target, with every cast checked again and the buildability
  question re-asked, rather than relabelled.
- **Execution** (`loopty.executor`). `LoopyExecutor` runs a kernel, a schedule or
  a term through `lp.ExecutableCTarget` on numpy or `Arr` arguments, and
  `differential()` compares the compiled run against the Python body at the
  tolerance the exactness class states, returning a fact.
- **Commands.** `loopty run FILE [--target c|opencl] [--emit-code] [--json OUT]`
  and `loopty check FILE` (lanky's checker). `RunVerb` is registered under
  `lanky.verbs`, so `lanky run FILE` works too. `--target` retargets every
  schedule in the file and reports by name any that cannot be retargeted;
  without it each schedule keeps the target it was written for.
- **Four demos** (`examples/`) and `scripts/refresh_example_outputs.py`, which
  re-runs the commands pasted into `examples/README.md` and rewrites their
  output, so the document cannot drift from the code in silence.
- **Documentation.** `docs/quickstart.md`, `docs/device-runs.md` with the
  transcripts under `docs/device-runs/`, and `docs/loopy-notes.md`: the loopy
  and islpy interactions that cost debugging time, each with its local
  workaround and the reason it is local.
- **The argument contract** (`loopty.contract`). What a call owes a term, in one
  place and asked by every entry point that runs a kernel: distinct array
  parameters are distinct storage, a ragged argument agrees with its counts
  family, and an element of a refined sort is one. Each is an assumption a
  typing rule makes about the call rather than about the term, so none of them
  can be established inside the type system, and a violation raises `ValueError`
  naming the argument.
- **Plugin surface** (`loopty.plugin`). `KernelTheory`, `IslOracle`,
  `LoopyExecutor` and `RunVerb`, exported through the four `lanky.*` entry-point
  groups. lanky never imports loopty; it finds these and asks each what it can do.

### Fixed

- The generated C function is renamed when the kernel's name is a C or OpenCL C
  keyword, or collides with an argument name (`def double(...)` used to emit
  `void double(...)`, which no compiler accepts). A *parameter* with such a name
  is refused instead, because renaming one would break every call.
- A CSR matrix whose rows are all empty runs on the C target. loopy tries to
  pass a null pointer for a zero-length array by calling the pointer type on
  `0.0`, which raises `TypeError: expected c_double instead of float`, so such a
  matrix could not be run at all. `executor` pads the flat buffer with one cell
  and restores the original afterwards, which is sound because no index into an
  empty array is in bounds. A kernel called with no rows at all still cannot
  run; see `docs/loopy-notes.md` for why the same trick does not apply there.
- A kernel body that iterates `.dom` accepts a plain `ndarray`: it is wrapped in
  `Arr`, sharing the buffer, instead of failing with `AttributeError: 'ndarray'
  object has no attribute 'dom'`.
- `KernelTheory` is registered once rather than twice. Importing
  `loopty.kernel` no longer registers it, because lanky's entry point already
  does; registration happens on demand instead.
- `Arr` refuses ragged offsets that do not start at zero. Only the differences
  and the last entry were checked, so `[-1, 2]` was accepted and gave row 0 a
  flat slice numpy wraps round to the end of the buffer and generated C reads in
  front of.
- A `when` guard is found by the identity of the guard object rather than by the
  name `when` appearing in the body. `from loopty import when as guard` and a
  renamed module attribute used to run the native body unmasked, so `python
  file.py` computed something the lowered kernel does not.
- A `break` or a `return` inside a traced loop raises `TraceError` naming `when`
  as the fix. The loop level is closed when the iterator raises `StopIteration`,
  which an early exit skips, so every statement after it used to be recorded
  under a loop variable the body had already left.
- `differential()` requires an explicit `reference` to cover exactly the outputs
  of the lowering. A reference naming one of several outputs used to bypass the
  native run and leave the rest unchecked under a `tested` fact.
- Two statements over the same iname with different, unguarded domains keep
  their own domains. loopy gives an iname one domain, so the two share the
  union; each statement now carries the gist of its own domain as an instruction
  predicate, instead of a statement written over `0 <= i < 2` executing over
  four points. A statement whose domain is empty gets a condition nothing
  satisfies rather than running over the whole union.
- Only size and count parameters are assumed non-negative in a statement domain.
  A guard on a signed scalar (`with when(a < 0)` with `a : Int`) used to make the
  domain empty and discharge every obligation over it vacuously.
- The exactness class consulted by `tag` is that of the reduction the tagged
  iname belongs to, not the first reduction found writing that output; with two
  reductions into one array the wrong contract used to be read. `realize` joins
  all of them and takes the strictest.
- A ragged argument is checked against the counts array its type names, ragged
  arguments sharing a counts family against each other, and explicit offsets
  against both. The generated loop is bounded by `cnt[r]` while the flattened
  access goes through the offsets, so a disagreement made compiled C read past
  a row while the native run followed the `Arr`'s own counts.
- Distinct array parameters may not share storage, and the executor and the
  native run both refuse a call in which two of them do. Dependences are
  computed per array name, so a kernel reading `x[i - 1]` and writing `y[i]`
  may tag `i` parallel and then race when called with `x is y`; `differential`
  copied each argument separately and destroyed the alias before either run
  could see it.
- An argument whose element sort is `Fin[m]` is checked to hold points of
  `Fin[m]` at the executor boundary and on the native run. The typing rule marks
  `x[col[r, j]]` decided *by type* from that declaration, so a `col` entry of
  `-1` or of `m` used to reach generated C as an address outside `x`; it is now
  a `ValueError` naming the first offending cell and its value.
- Each reduction keeps its own iteration domain. Reduction domains were merged
  by iname like statement domains, but a reduction cannot carry the narrowing
  predicate that gives a statement its domain back, so two reductions over `j`
  with bounds 2 and 4 both summed over four points. A binder is renamed to a
  fresh iname only when the same name is already bound to a different domain, so
  the name the source wrote survives wherever it is unambiguous.
- Reflected parameter names are allocated rather than derived. `nl_cnt_r` was
  spelled from the term by replacing non-word runs with underscores, which is
  not injective (`cnt[r]` and `cnt*r` spell the same) and can collide with a
  size the kernel declares, silently asserting two unknowns equal. One
  `Reflections` table per term keys the parameter on the term, keeps the
  readable spelling when it is free and suffixes it when it is not, and is
  shared by every isl set built about that term; what it allocated travels on
  `Term.reflected`, which is how lowering recognizes a ragged bound whose name
  had to move.
- Every native run wraps its array arguments in the masking views, so the
  contract is now "a body sees a view sharing the caller's buffer" rather than
  "an unguarded kernel sees the objects it was given". Whether a body opens a
  `when` block used to be decided by inspecting that body, so a kernel calling a
  helper that opens the guard performed the guarded write and `python file.py`
  computed something the lowered kernel does not; the same inspection missed a
  helper that asks an argument for its `.dom`, which failed with
  `AttributeError: 'ndarray' object has no attribute 'dom'`. No inspection can
  decide either question, so neither is asked; `opens_a_guard` still answers it
  as well as a static walk can, following function-valued globals and closure
  cells, but it reports rather than decides. The demos' native runs are within
  a few percent of what they were.

### Changed

- **The accumulate convention is part of the Term IR.** `Stmt.kind ==
  "accumulate"` now *means* that `expr` is the complete right-hand side and
  reads the assignee cell, so `y[r] += t` is
  `Stmt(assignee=y[r], expr=y[r] + t)`. Lowering checks the invariant and
  refuses a term that breaks it, in place of a heuristic that inspected the
  expression and guessed.
- **The differential tolerance is per element.** `exact` is bitwise; `reassoc`
  and `approx` ask that `|got - want| <= eps_class * (|want| + 1)` at every cell.
  It used to be one tolerance for the whole output, scaled by that output's
  1-norm, so a large output bought a large allowance for each of its cells (a
  million ones gave a tolerance of 1.0). The `difference ... within ...` line
  now names the element that came closest to its own allowance, so the numbers
  printed by the demos are smaller than they were.
- **A reduction's exactness class is derived rather than fixed.** It comes from
  the element sorts of what the reduction sums, joined so that the weakest wins:
  a sum of `Nat` is `exact`, a sum of `Real` is `approx`. `reassoc` is no longer
  something a trace assumes; it is what a schedule lowers an accumulation to
  when it reorders one, and `realize(var, tree=True)` over an `exact`
  accumulation is refused.
- `islpy` is pinned below 2026. loopy 2025.2 calls `Aff.is_equal` during code
  generation for a tiled loop nest and `BasicMap.is_bijective` in `map_domain`,
  and islpy 2026 removed both. Drop the ceiling once a loopy release supports
  islpy 2026.
- `lanky>=0.1.0.dev0` is a dependency, resolved from a sibling checkout by
  `[tool.uv.sources]` during development.

### Notes

- A kernel file needs `from __future__ import annotations` and a ruff `F821`
  per-file ignore, because a size such as `n` in `Arr[Fin[n], Real]` is a
  symbolic variable lanky invents while evaluating the annotation.
- An axis after the first is ragged when its size is a bare name that is also an
  array parameter of the same kernel: `val: Arr[Fin[n], Fin[cnt], Real]` next to
  `cnt: Arr[Fin[n], Nat]`. A symbolic inner size naming no parameter is read as
  an ordinary uniform size.
- The OpenCL target is written but never exercised on a development machine.
  Device runs happen elsewhere and are reported in `docs/device-runs.md`.
- Ragged bounds are reflected into isl as one parameter per distinct bound term,
  so `cnt[r]` and `cnt[r + 1]` are unrelated. An access against flat storage,
  `val[off[r] + j]`, is therefore reported `assumed` with the reason in its
  provenance, never `decided`; the ragged spelling `val[r, j]` is decided. See
  the module docstring of `loopty/flow.py`.
- The test suite treats `DeprecationWarning` as an error. Three exemptions are
  loopy's own and are listed in `pyproject.toml` and `tests/conftest.py`, with
  the reasons in `docs/loopy-notes.md`.
