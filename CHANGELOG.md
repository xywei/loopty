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
  wrote, `loopty.reduce_sum` becomes a reduction, and `with when(cond):` records a guard
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
- **A fifth demo**, `examples/wavefront_acoustic.py`: two coupled statements in
  one acoustic-wave nest, whose rectangular tiling is refused with a witness
  that crosses them (`S1` at `(t, i + 1)` before `S0` at `(t + 1, i)`), and the
  skew that makes the same tiling a legal wavefront block. Its section in
  `examples/README.md` has the three transcripts, generated like the others.
  `--bench` times the untiled and blocked kernels; nothing runs it but a reader.
- **Documentation.** `docs/quickstart.md`, `docs/device-runs.md` with the
  transcripts under `docs/device-runs/`, and `docs/loopy-notes.md`: the loopy
  and islpy interactions that cost debugging time, each with its local
  workaround and the reason it is local.
- **The argument contract** (`loopty.contract`). What a call owes a term, in one
  place and asked by every entry point that runs a kernel: distinct array
  parameters are distinct storage, a ragged argument agrees with its counts
  family, and a value of a refined sort is one, array element and scalar
  argument alike. Each is an assumption a typing rule makes about the call
  rather than about the term, so none of them can be established inside the type
  system, and a violation raises `ValueError` naming the argument.
- **Plugin surface** (`loopty.plugin`). `KernelTheory`, `IslOracle`,
  `LoopyExecutor` and `RunVerb`, exported through the four `lanky.*` entry-point
  groups. lanky never imports loopty; it finds these and asks each what it can do.

### Fixed

- A guard's reads are stated over the loop nest *before* the guard narrowed it
  (`Stmt.loop_domain`), because `when` evaluates its whole condition at every
  point and only masks the write: `when((i + 1 < n) & (flag[i + 1] != 0))`
  reads `flag[n]` at `i = n - 1`, and stating that read over the narrowed
  domain used to prove it in bounds by the very condition that does not
  protect it.
- A boolean or a non-number passed for a scalar of a refined sort (`i: Fin[n]`)
  is refused instead of silently skipping the check; a zero-dimensional numeric
  array counts as a number.
- An expression-valued `Fin` bound on a scalar (`i: Fin[n + 1]`) is evaluated
  against the resolved sizes; when a size is unknown the value is still
  required to be non-negative.
- A complex array declared with an integral element sort is checked for finite,
  whole, real entries before the cast into the compiled kernel's integer dtype.
- `resolve_sizes` solves an axis written as an affine expression in one name
  (`Fin[n + 1]`) for that name when no bare axis determines it, so a scalar
  `i: Fin[n + 1]` is range-checked even when `n` occurs nowhere else.
- A guard's read of the cell an accumulation writes keeps its own in-bounds
  fact: the `acc` footprint covers the right-hand side's read over the narrowed
  domain, not the guard's eager read over the loop nest.
- The lowering states `0 <= i < n + 1` for a scalar declared `Fin[n + 1]`, not
  only for a bare `Fin[n]`, so such a kernel lowers.
- A term with an array parameter the body never reads or writes is refused by
  the lowering with a `LoweringError`. loopy's C target lists only the arrays
  the body touches in the device signature and passes every argument from the
  host wrapper, so such a parameter shifted every later argument into the wrong
  register: the compiled run returned zeros and corrupted the heap. See
  `docs/loopy-notes.md`, note 1.

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
- A scalar parameter of a refined sort is checked at the call boundary. A
  kernel taking `i: Fin[n]` and writing `x[i]` has that access decided *by
  type*, exactly as an indirection through a column array is, but the contract
  skipped every non-array parameter, so `run(..., i=-1)` reached generated C as
  an address in front of `x`. `Fin[b]` is checked against the sizes the call's
  arrays determine, `Nat` for non-negativity and `Int` for being whole, at the
  executor and on the native run.
- A value of a refined integer sort has to be a finite whole number, which is a
  separate question from being in range and is asked first. A range test is two
  comparisons, and `nan` fails both, so it used to pass; `1.5` passed honestly
  and was then truncated to the index `1` by the cast into the compiled
  kernel's integer dtype while the native run kept the float. An integer dtype
  passes without a test and an *integer-valued* float array is accepted, because
  being a point of `Fin[m]` is a property of the value and not of its storage
  and that cast is exact on it; `1.5`, `inf` and `nan` are refused, naming the
  cell.
- An array read inside an assignee's subscripts is an access. For
  `y[col[i + 1]] = v` the write was recorded and `col[i + 1]` was not, so the
  write was discharged in bounds by `col`'s element type while nothing asked
  whether the kernel read past the end of `col` to find the cell.
- An array read inside a `when` guard is an access. `with when(flag[i] != 0)`
  reads `flag[i]`, and the reference lives only in `stmt.guard`: no in-bounds
  obligation was stated for it, and no dependence was seen on an earlier
  statement writing `flag`, so a parallel tag that lets the predicate observe
  the overwritten value was accepted.
- Those accesses are collected in one place. `flow.statement_accesses` now
  returns everything a statement touches — assignee, right-hand side, reduction
  bodies, assignee subscripts and guard — and `loopty.typing`,
  `loopty.flow.footprints`, `loopty.schedule` and `loopty.lower` all read it
  instead of walking the statement again themselves. The four had drifted apart,
  which is how one omission could be three different bugs.
- A kernel with an integral scalar parameter lowers. `i: Fin[n]` *declares*
  `0 <= i < n`, which loopy has no way to learn about a value argument, so
  `x[i]` failed its bounds check ("could not establish ... is a subset of ...")
  for the legal call as readily as for the illegal one. The declaration is
  passed to loopy as an assumption, which is sound because the contract now
  refuses any argument it is false of.
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
- The `reduce_sum` import in `examples/spmv.py`, `examples/p2p.py` and
  `tests/test_trace.py` is merged into the `from loopty import ...` line it
  belongs to. It had been kept on a line of its own so that the source
  locations in the example transcripts would not move, which failed ruff's
  import sorting (`I001`); ruff runs first in CI, so the tests did not run at
  all. The transcripts in `examples/README.md` are regenerated by
  `scripts/refresh_example_outputs.py`, and each ledger that repeats them,
  the abridged one in `README.md` and the two in `docs/quickstart.md`,
  follows them: every `spmv.py` location moves up one line and nothing else
  changes.
- Two statements that feed each other across an iteration of an enclosing loop
  no longer lower to a dependency cycle. `lower_generic` marks each
  instruction's `depends_on` final, so loopy's single-writer heuristic cannot
  add an edge from a statement to a later one that writes what it reads: that
  order is the loop's, and the edge made the first run fail with
  `DependencyCycleFound`. The coupled wave demo was the first kernel to have the
  shape; see note 7 in `docs/loopy-notes.md`. The heuristic had also been what
  ordered a read through a ragged array's offsets against a statement that
  writes them, when that statement was their only writer, and not always in the
  body's direction. Every statement that touches a ragged array now counts as
  reading its offsets, so a kernel that computes its offsets and then reads
  through them is not refused with `VariableAccessNotOrdered`.
  The instruction that computes a ragged row's length is ordered the same way,
  where the first statement that needs it runs. Left to the heuristic, it
  waited for a statement that rewrites the offsets even when that statement
  came after the ragged loop, and the three made a cycle. The length is
  computed once, so a statement that needs it after the offsets (or counts) it
  was computed from have been rewritten is refused with a `LoweringError`,
  instead of running over the old row length.
- `Arr` refuses a negative index into a dense array with an `IndexError`. It
  handed the key straight to numpy, which reads `x[-1]` as the last cell, while
  `Fin[n]` has no negative points and generated C reads `x[-1]` in front of the
  buffer, so a native run could compute a value neither the ledger nor the
  compiled kernel agrees with. A read under a false `when` still answers zero,
  but a read taken before its guard opens (`v = x[i - 1]`, then
  `with when(i > 0):`) now fails at `i = 0` on a native run, as `x[i + 1]`
  already did at the other end; it belongs inside the guard.
- `LoopyExecutor.run` writes its results back into every `ndarray` output, not
  only into an `Arr`. An output that is a strided view, or whose dtype is not
  the lowered one, is copied on the way into loopy, and the results used to stay
  in that copy.
- `scripts/refresh_example_outputs.py` treats a command that exits non-zero as
  a failure in both modes: the block is left as it was and the script exits 1.
  It used to paste the partial output of a broken demo into the document, and
  `--check` called a block current as long as its text matched.
- A reduction's exactness class is that of the value it sums, not only of the
  arrays it reads. `a * x[j]` with `a : Real` over an integer `x`, `0.1 * j` and
  `x[j] / 2` were all `exact`, so a schedule refused to reassociate them and the
  ledger stated an exactness that did not hold. A Python `float` sort, as a
  hand-built term may give one, is `approx` as well; lanky reads anything that
  is not one of its sorts as an index type.
- Sizes, loop variables and reduction variables spelled like a C or OpenCL C
  keyword are refused by the lowering, as parameters already were.
  `Arr[Fin[long], Real]` and `for double in x.dom` passed the check and then
  failed in the C compiler, on generated code. So are the names C reserves by
  their spelling, those that start with an underscore and a capital letter or
  with two underscores: `_Bool`, `_Complex`, `_Generic` and OpenCL C's
  `__global` were not in the list. A kernel with such a name is renamed with a
  `k` prefix, because no suffix takes a name out of that space.
- The native run reads an integer-valued float array of an integral element
  sort as integers when the body only reads it. The contract accepts
  `col = [1.0, 0.0]` for `Fin[m]` and the compiled run casts it, but numpy
  refuses a float as an index, so the input could not be tested
  differentially. An array the body writes is passed as it is, so that its
  writes land in the caller's buffer. The contract refuses a float-stored
  entry outside the range of `int64`: `1e20` is a whole number, a `Nat` array
  has no upper end to refuse it by, and the conversion used to hand the body an
  unrelated integer.
- The isl oracle reads a witness and the sizes it holds at from one sample.
  They came from two, which leaves isl free to report a cell outside an array
  at a size where it is inside.
- A reduction nested in another one is constrained by the outer binder and the
  outer generator's condition. In `reduce_sum(reduce_sum(a[i, j] for j in
  a.dom[i]) for i in a.dom)` the inner domain left `i` unconstrained, so
  `a[i, j]` was refuted at `i = -1`. A nested binder that reuses the outer
  one's name raises `TraceError`. Two statements whose nested sums both bind
  `j`, under outer binders of different names, now lower: the two inner
  domains used to be the same set, so they shared one iname nested in two
  loops, and loopy found no loop nest to schedule.
- `resolve_sizes` solves only an axis that is linear in its one name. The
  solution is read off two evaluations, so `Fin[n * n]` of nine cells gave
  `n = 9`, and `Fin[(n + 1) // 2]` an `n` whose axis is too short. A `Fin` bound
  that cannot be evaluated is measured against an axis written the same way
  (`contract.axis_extents`), so `i : Fin[n * n]` is still bounded by the nine
  cells it indexes.
- A reduction nested in another one whose bound is read off the outer binder
  and is not affine, as in `reduce_sum(reduce_sum(val[q, j] for j in
  val.dom[q]) for q in val.dom)`, is refused by the lowering with a
  `LoweringError` that names the statement and the bound and says what to
  write instead: the outer reduction as a loop that accumulates into the
  output, or each inner sum kept in a cell indexed by the row. It used to fail
  at run time with loopy's "value argument 'nl_cnt_q' was not given". A row
  length is computed inside the loop over its row, and a reduction binder has
  no loop another instruction can run in.
- The domains of a lowered kernel are ordered so that each follows the domain
  whose loop variables it names. loopy reads the nesting off that order, and a
  ragged loop followed by a second loop put the row-length domain after the
  second loop's: loopy made it a top-level domain and got the loop right only
  through a call islpy deprecates and says will stop working in 2026.
- An integral scalar has to be passed as an integer. `i = 1.0` for `i: Fin[n]`
  passed the contract as a finite whole number, and neither run could use it:
  the native run raised numpy's "only integers ... are valid indices" and the
  compiled run "'float' object cannot be interpreted as an integer". It is a
  `ValueError` naming the argument now, and says `int(i)`.
- A sort that is a free name is refused when the kernel is traced, with a
  `TraceError` that says what to write instead: `Real` or `np.float64` for
  `float`, `Nat`, `Int`, `Fin[n]` or `np.int64` for `int`. Under `from
  __future__ import annotations` lanky's scope invents `float` and `int` like
  any other name it does not know, so `a: float` gave the sort `Var("float")`,
  which has no numpy dtype, is not an integral sort, and was called an `exact`
  index type in the ledger. A hand-built term with such a sort is refused by
  the lowering. The native run needs no sort and still runs.
- State that a Python name carries from one loop iteration to the next is
  refused with a `TraceError` naming the name and the two fixes, instead of
  tracing to a wrong term. `s = 0.0; for i in x.dom: s = s + x[i]` followed by
  `y[0] = s` used to trace to one statement, `y[0] = 0.0 + x[i]`, with no loop
  around it and `i` free, while the native run summed the array. Two checks
  catch the idiom. A statement whose right-hand side, guard, assignee indices,
  loop bounds or reduction bounds mention the variable of a loop it is not
  inside is refused where it is recorded. And the locals of the frame running
  a `for` (the kernel body or a helper it calls) are compared when the loop
  opens and when it closes: a name bound before the loop and bound to a
  different value after one iteration is loop-carried, whether the value is a
  term or a plain Python number (a counter `k = k + 1` used as an index), and
  `s += x[i]`, tuple unpacking and `del s` count. The loop's own target, a
  per-iteration temporary first bound inside the loop, a rebinding to the same
  object or an equal value, and a name whose old value already mentions a
  closed loop's variable (a `for` target reused by a later loop) are left
  alone. State kept outside a plain name is compared the same way. A global
  the frame's code rebinds with `global G` counts like a local, with the same
  exemptions (a `for` target stored as a global is not state). So does a list,
  dict or set reachable from the frame's locals, or held by a global its code
  names, whose contents change across one iteration: `state = [0]` followed
  by `state[0] += 1` and `y[i] = state[0]` in the loop used to trace to
  `y[i] = 1`. Elements are compared by identity or structurally, never with
  `==`, and the message names the container and the cell that changed. A
  container first created inside the loop is scratch and is left alone; one
  created before the loop and reused as scratch is refused, with a message
  that says to create it inside the loop. A name first bound inside the loop
  is a temporary only while every iteration binds it, so code running a traced
  loop may not ask which names are bound: the builtins `locals()`,
  `globals()` and `vars()`, and an `except` clause naming `NameError` or
  `UnboundLocalError`, are refused when the loop opens (`if "s" not in
  locals(): s = 0` used to trace to one iteration's value). A local or global
  of the same name, and an attribute, are left alone. The fixes are
  `reduce_sum(...)` for an accumulation and an indexed cell
  (`s[i + 1] = s[i] + x[i]`, as `scan` in `examples/spmv.py` does) otherwise,
  spelled with the loop's own target and domain. A message names a loop by its
  `for` target and line, and adds the iname when a reused target made the two
  differ. Both checks are trace-time only; plain `python` runs the body as
  written. A change nested below a container's own elements
  (`state[0][0] += 1`), an attribute, and a global that only a helper defined
  outside the body rebinds or changes are not seen yet.
- A loop whose target is spelled like a size or a parameter of the kernel gets
  an iname of its own. `for k in x.dom` over `x: Arr[Fin[k], Real]` used to make
  the size and the iname one isl dimension, so the loop's domain was
  `0 <= k < k`, which is empty, and every statement in it was checked over no
  points at all. The iname is now `k_0`, as for a target reused by a second
  loop.
- A loop written on one line keeps its source name under Python 3.13, which
  fuses the store of the `for` target with the load after it
  (`STORE_FAST_LOAD_FAST`). The target was read as unknown, so
  `for i in x.dom: y[i] = x[i]` traced over an iname `i0` on 3.13 and `i` on
  3.12.
- `lanky check` prints the error of a kernel that cannot be traced under its
  `REFUTED` line: the `trace` fact carries it as its `reason`, next to an empty
  `counterexample`, which is lanky's form for a closed claim refuted at no
  assignment in particular. The fix a `TraceError` names used to reach only the
  JSON ledger. `loopty run` reports such a kernel as one it cannot schedule,
  naming the error, and exits 1, where it used to stop with a traceback from
  the search for kernels.
- The schedule checker and the typing rules see the reads a ragged access makes
  through its offsets. `val[r, j]` is `val[off[r] + j]` once lowered, and row
  `r` ends at `off[r + 1]`, but neither read is in the body, so only the
  lowering knew of them: a cast's legality check saw no dependence between a
  statement that writes the offsets and one that indexes through them.
  `tag(r="l.0")` on a loop that sums row `r` and then stores `off[r + 1]` was
  accepted, and would run row `r + 1` before the offset it starts at is
  stored; it is refused now, with that pair as the witness.
  `flow.statement_accesses` lists `off[r]` and `off[r + 1]` after every ragged
  access, read or written, whenever the kernel declares the offsets as a
  parameter (the names lowering looks for, now `loopty.term.OFFSETS_CANDIDATES`
  and `loopty.term.declared_offsets`), and takes the term to know. Both
  dependence relations (`flow.dependences` and the schedule checker's own), the
  in-bounds rule and the lowering's instruction order read that list; the
  lowering's own addition of the offsets is gone. The two reads are in-bounds
  obligations of their own: decided by isl for offsets of `n + 1` cells,
  refuted with a witness for offsets declared a cell short. Offsets a kernel
  does not declare are an argument lowering adds, which nothing in the body can
  write and which the row index keeps in bounds, so they are not listed, and no
  example's ledger changes.

### Changed

- **Executor options are separate from kernel arguments.** Every argument of
  `LoopyExecutor.run` is an argument of the kernel, and the target is chosen by
  the schedule (`Schedule(kernel, target="opencl")`) or by the executor
  (`LoopyExecutor(target="opencl")`). `run` used to pop `target=` from its
  keywords as the backend, so a kernel parameter called `target` could not be
  passed by keyword: the call failed on numpy's "truth value of an array ... is
  ambiguous". `differential` no longer takes extra keywords either, which it
  passed on to `run`.
- **Reductions have a Loopty-owned frontend.** `loopty.reduce_sum` is the public
  kernel API; Lanky still supplies generator binder capture internally, while
  tracing immediately converts the captured node into an ISL-backed
  `loopty.term.Reduction`. `loopty.sum` remains as a compatibility alias for
  the development API, but examples and documentation use `reduce_sum`.
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
