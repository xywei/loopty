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
  fiber), a hardware axis on a reduction nested in another, or a reduction split
  across parallel and sequential inames, produces a `refuted` fact of kind
  `buildable` decided by `loopy-target` with the limit in words, and
  `UnbuildableSchedule` is raised as soon as anything asks the schedule for
  code. The first and the last were measured on real devices, see
  `docs/device-runs.md`, and the second in loopy's code generation, see note 11
  in `docs/loopy-notes.md`. The design's own spmv device schedule is the case.
- **`Schedule.retarget(target)`**. The same transformations replayed against
  another loopy target, with every cast checked again and the buildability
  question re-asked, rather than relabelled.
- **`Schedule.affine(map)`**. A cast along any injective affine map, given as
  an isl map, or its text, from loops of the kernel to the loops that replace
  them: `affine("{ [t, i] -> [a, b] : a = t + i and b = t - i }")`. It is
  checked as every cast is: the map has to be defined on every statement
  instance and one for one there (a `bijective` fact, refuted with the
  instance it misses or the two it merges), and the new order has to run every
  dependence forward (a `monotone` fact, refuted with the pair of instances and
  the array cell between them). The new loops take the places of the old ones
  in the loop order. The map need not be unimodular. loopy's `map_domain` and
  `affine_map_inames` both refuse the diamond, whose image is only the points
  of equal parity, so loopty rewrites the kernel from the same isl map: the
  domain becomes its image, with the parity as an existentially quantified
  constraint, a domain nested in the mapped loops follows with the new loops as
  its parameters, and each old loop variable becomes the quasi-affine inverse
  isl gives, `floor((a + b)/2)`. That answers the question the spike asked:
  loopy 2025.2 generates correct code for a non-unimodular image, bit for bit
  on the stencil (against its reference, untiled and tiled in diamond
  coordinates) and on the acoustic pair (against the native run), with the
  parity tested inside the innermost loop rather than stepped over. A map the rewrite cannot write for loopy (loops no one domain
  defines, an image that is not one basic set, a piecewise inverse) is a
  `refuted` `buildable` fact, and the schedule has no kernel from then on. A
  map moves every statement in its loops alike; a map per statement is
  refused. `examples/wavefront_acoustic.py` tries the diamond three ways: with
  space first it is refused with a witness, with time first it is accepted and
  runs, and tiling it is refused, because the pair needs a time offset between
  its statements. Note 13 in `docs/loopy-notes.md` has the details, and
  `loopty.oracle.is_bijection_on` is the totality-and-bijectivity question the
  first fact asks.
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
- **The term interpreter** (`loopty.interpret`). `interpret(term, arguments)`
  runs a traced term on concrete arguments, in place, without loopy: every
  instance of every statement, over the points of its isl domain at the sizes
  the arguments determine, in the order of the source schedule (statement by
  statement in source order within each iteration), with a guard evaluated at
  each instance. Expressions are evaluated node by node with Python's
  operators on the numpy scalars read from the arrays, which is the arithmetic
  the native run does, and a reduction is summed in the order `reduce_sum`
  sums natively, lexicographically over its binders with Python's `sum`. A
  ragged bound is read from its counts array once the loop variables it names
  have values. What it has no meaning for is an `InterpretError` naming it, not
  a guess: a call with no numpy counterpart, a domain it cannot enumerate, and
  a loop bound reading an array the same kernel writes.
- **The faithfulness fact** (`loopty.faithful`). Every kernel's ledger ends
  with a fact of kind `trace-faithful`, "the traced term computes what the body
  computes". The native body and the interpreted term run on copies of the
  same inputs, and every array argument is compared afterwards, bit for bit
  when its exactness class is `exact` (two NaNs agree whatever their sign and
  payload, which IEEE 754 leaves open) and within the class's tolerance
  otherwise. The inputs are the module's `example_inputs()`, the ones
  `loopty run` reads, and three drawn from the declared types from a fixed
  seed, with every size at least 2 so that a loop runs more than one iteration,
  a ragged axis laid out by the counts array it names, and a point of `Fin[m]`
  below `m`. The fact is `tested` by `interpreter` when the runs agree,
  `refuted` at the first input that disagrees with a counterexample naming the
  input, the first differing cell and both values (the drawn arguments are in
  the provenance), and `assumed` with the reason when no input runs natively or
  the interpreter cannot read the term. An input with more statement instances
  and reduction terms than `MAX_INSTANCES` is skipped before the native run,
  and a domain is judged by its bounding box before its points are collected,
  so an input far past the limit costs nothing. The fact is what catches state a body keeps where tracing does not look: a change
  nested below a container's elements (`acc[0][0] += 1`), an attribute of an
  object an attribute holds, a `deque`, a loop over a generator that wraps a
  domain, and a `dir()` or frame probe all trace to one iteration's value and
  are refuted. The demos' ledgers carry
  one more row per kernel, and their transcripts are regenerated.

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
- An operation on a whole symbolic array is refused with the loop nest that
  does it one cell at a time (`for i in y.dom: y[i] = ...`). `y[:] = 0.0` used
  to be recorded as one statement whose index was a slice, `u[t] = ...` of a
  two-axis `u` as a statement on a row, `x * 2` failed with "unsupported
  operand", `np.sum(x)` handed the symbolic array back, and `for v in x`
  walked it forever through Python's old sequence protocol, since a symbolic
  array answers any index. A slice, an `...`, a list or an array of indices,
  fewer indices than the array has axes, arithmetic and comparison operators,
  iteration, `.numpy()`, and numpy's ufuncs and functions of a symbolic array
  are each a `TraceError` now, and so is a fiber over a slice (`u.dom[1:]`).
- A reduction's `if` clause that isl cannot state is refused rather than
  dropped. The clause is a constraint of the reduction's domain, and a
  reduction keeps its condition nowhere else, so
  `reduce_sum(x[j] for j in x.dom if x[j] > 0)` traced to the sum of every
  `x[j]`, and `if j != i` to a sum that included the diagonal. The message
  says to split a `!=` in two (`<` and `>`), and otherwise to write each term
  to an indexed cell under `with when(condition):` and sum the cells, as the
  near-field demo does. An affine clause (`if j < i`) is a constraint as
  before.
- An equality in a guard or a reduction condition (`with when(i == 0):`,
  `if j == i`) is handed to isl as `=`, which is how isl spells it. It was
  handed over as `==`, and tracing stopped on isl's syntax error.
- A body's effects outside its array parameters are refused once it has been
  traced. Tracing runs the body once, so such an effect happens once in the
  trace and once per call natively, and the compiled kernel never has it. The
  state the body's code reaches by name (module globals it names, closure
  cells, default values, and the same for every helper of the kernel author's
  that it calls, eight levels deep) is copied before the trace, one level into
  it as the loop snapshot is: a list, dict or set shallowly, a numpy array
  cell by cell (a write into one changes no output, so comparing outputs could
  not see it), and an object's attributes together with the lists, dicts, sets
  and arrays they hold. A change is a
  `TraceError` naming the state and both values, so `obj.count += 1`,
  `self.s = self.s + x[i]`, `LOG.append(x[0])`, a global rebound outside any
  loop, and a global that only a helper defined outside the body changes are
  all refused now. State the body creates for itself is scratch, and a `for`
  target stored as a global is left alone, as it is across an iteration. So are
  the attributes of an object of a library's type (a logger fills a cache on
  its first `debug` call; a `types.SimpleNamespace` is the author's), and the
  value a `functools.cached_property` stores the first time the body reads it.
  A call that prints, reads input, opens a file, or draws a random number from
  `random` or from a numpy generator is refused too, with its line: the calls
  a body makes are seen through `sys.monitoring` while it is traced, and a
  call from library code (loopty, lanky, numpy, pymbolic, islpy, loopy, the
  standard library, site-packages) is never counted as the body's. Plain
  `python` runs the body as written.
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
  `REFUTED` line: the `trace` fact carries it as its `reason`, and no
  `counterexample`, because it is refuted at no assignment in particular (an
  empty one, there only to get the reason printed, is gone now that lanky
  prints a reason without one). The fix a `TraceError` names used to reach only
  the JSON ledger. `loopty run` reports such a kernel as one it cannot
  schedule, naming the error, and exits 1, where it used to stop with a
  traceback from the search for kernels.
- A refuted cast fact carries its explanation as `reason`, which lanky prints
  under its `REFUTED` line: the message of the `IllegalCast` it is raised with,
  or, for a `buildable` fact, the limit the target hits, which is also
  `UnbuildableSchedule.reason`. It had it as `detail` alone, which only the
  JSON ledger shows, so the block under the line read `no witness recorded`
  for a fact with no witness (`exactness`, `buildable`) and was empty for a
  `bijective` or `monotone` fact with one. `detail` stays, in the oracle's
  words, and `witness` is recorded whenever isl gives one.
- A fact the isl oracle refutes (an access out of bounds, two instances
  writing one cell) carries a `reason` that names the question and the
  labelled witness at its sizes, which lanky prints under its `REFUTED` line:
  `cells u[i + 1] reaches are cells u has, except [a0=1] at [n=1]`. Nothing was
  printed under the line, because lanky counts the oracle's `witness` as what
  explains a refutation and prints neither it nor `witness_text`.
- What refuted a fact is printed under its `REFUTED` line by `loopty run` as by
  `lanky check`. `loopty run` printed the bare line; it now prints lanky's own
  block (`lanky.cli.refutation_lines`) under each one, and the line itself as
  lanky does, `REFUTED owner at where: statement`, after a blank line. A
  refuted `agreement` fact names each output that disagreed, and by how much
  against its allowance (or its shape and the native run's, when the two
  differ), as its `reason`, where the block used to read `no witness
  recorded`.
- `loopty run` reports a body that raises `IndexError` or an `ArithmeticError`
  on its example inputs (a read past the end, a division by zero) by name, as
  it reports the other errors a run stops with, goes on to the file's other
  kernels, and exits 1. It stopped with a traceback.
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
- An access listed over more than one domain is one in-bounds fact, about the
  cells it reaches over all of them. The fact's id names the access and not the
  domain, and the ledger keeps one fact per id, so the later of two facts
  replaced the earlier: `y[r] = x[r - 1] + reduce_sum(x[r - 1] for q in
  Fin[r])` was reported in bounds from the read inside the sum, which runs only
  for `r >= 1`, and the refutation of the direct read of `x[-1]` was lost. The
  offsets a ragged access reads through made this easier to reach: `off[r - 1]`
  read directly, and again through `val[r - 1, j]` inside such a sum, collided
  the same way. No example's ledger changes: where p2p lists a read twice, the
  two facts agreed.
- Two statements whose sums bind the same name lower and run. loopy realizes a
  reduction as a loop inside its instruction and an iname is one loop, so when
  the second statement depends on the first, its sum had to run inside a loop
  that must finish before it starts: two statements whose nested sums both bind
  `i` and `j`, or a sum over `j` followed by a loop over `j` that reads it,
  failed at the first run with a `CycleError`. A reduction now keeps its
  binders only when no other statement has them, as loop variables or as the
  binders of an earlier sum, and is lowered under fresh inames (`j_0`)
  otherwise. In one statement a name is shared only by sums over the same
  domain, so `j` bound as the second of a pair and then alone no longer makes
  loopy refuse the kernel for defining `j` twice. A nested reduction's domain
  follows its renamed outer binder, which it used to name by the written name,
  tying the inner loop to the other statement's outer one.
  `Lowering.reduction_inames` lists the inames each reduction ends up with, and
  `Schedule` addresses a reduction by them, so `split("j_0", 2)` reaches the
  second sum and a parallel tag on it is judged by that sum's exactness rather
  than by the last sum written over `j`. See note 8 in `docs/loopy-notes.md`.
- `a.dom[r, i]` is `a.dom[r][i]`, while tracing and on a runtime array, as
  `a[r, i]` is a cell. The tracer took the tuple for one index and gave the
  domain of axis 1 whatever its length, so a loop over the third axis of a
  three-axis array ran over the second, and a native run refused the tuple.
  `a.dom[()]` is refused in both.
- A kernel with an `exact` output is compiled with floating-point contraction
  off: `-ffp-contract=off` in its build options on the C target, and
  `#pragma STDC FP_CONTRACT OFF` (or the OpenCL pragma) in the source. A fused
  multiply-add rounds `a * b + c` once where the native run rounds twice, and
  an `exact` output is compared bit for bit, so a compiler that contracts
  (clang by default, on arm64 where FMA is in the baseline) could refute a
  correct kernel. loopy's own `gcc -std=c99 -O3 -fPIC` does not contract on
  x86-64, which is why nothing had shown it; `tests/test_contraction.py` makes
  a compiler contract on hardware with FMA and shows the pin keeping the bits.
  `Lowering.contraction` records the choice, and note 9 in
  `docs/loopy-notes.md` has the flags.
- The transcripts in `README.md` and `docs/quickstart.md` are regenerated and
  checked by `scripts/refresh_example_outputs.py`, as those in
  `examples/README.md` were, and CI runs it with `--check`. A console block
  with a `...` line is an excerpt: the lines it keeps have to be lines of the
  output, in order, verbatim, and a refresh follows a moved line number or a
  wider column by the shape of the line and fails on a line that is gone. The
  abridged ledger in `README.md`, whose rows say they are verbatim, was kept
  so by hand, and its rule of dashes was not.
- A `when` guard that compares with a `Real` scalar no longer narrows the
  statement's isl domain. isl reads every name of a constraint as an integer,
  so `with when(i < a):` with `a: Real` made `a` an integer parameter of the
  domain, and at `a = 2.5` the compiled kernel disagreed with the native run
  (the differential test was refuted by 1.0 at a cell), while the faithfulness
  fact was left `assumed` because the interpreter would not fix a parameter at
  a value that is not an integer. A comparison is a constraint only when every
  name in it is a loop variable, a size, or a scalar of an integral sort
  (`Nat`, `Int`, `Fin[...]`); any other conjunct is left to the statement's
  guard predicate, evaluated at run time, and the domain over-approximates the
  instances that write. `Stmt.unnarrowed` lists the conjuncts a domain leaves
  out (this one, a data guard, a `!=`, a guard the trace already found
  `False`), each with the reason, and the in-bounds, disjointness and ordering
  facts stated over such a domain carry the list under `unnarrowed` in their
  provenance: proved, they hold for the instances that write too; refuted, the
  witness may be an instance the guard masks. A reduction condition that
  compares with a `Real` scalar is refused, as a condition its domain cannot
  state already was. `with when(i < a):` with `a: Nat` narrows the domain as
  before.
- A `when` guard whose value is an integer rather than a truth value is
  refused with a `TraceError` that names the fix, on a native run as well as
  under tracing. `~` on a Python bool is bitwise (`~True` is `-2`, `~False` is
  `-1`, both true), so `with when(~(i > 0)):` on a loop variable wrote every
  cell natively while the trace recorded `not (i > 0)` and the compiled kernel
  wrote one; `&` or `|` with an integer operand is bitwise in the same way. The
  fix is the complement written as a comparison (`i <= 0`), and an explicit
  comparison (`k != 0`) for an integer. A data comparison is a numpy `bool_`,
  on which `~` is logical, and is not affected, and natively a guard nested
  under a false one is not asked, since nothing under it is written (a read out
  of range there answers the integer 0). The faithfulness fact counts a
  `TraceError` from the native run as a disagreement, not as an input the body
  refuses, so such a kernel is `refuted` with the refusal as its reason instead
  of `assumed` for want of an input that ran.
- Which code is a library's, for the trace-time refusals of hidden state, is
  decided by module. The kernel's own module and the top-level package holding
  it (below a namespace package, the first regular package, which is the
  author's alone) are never a library's, so a kernel installed into
  site-packages by a non-editable install has its `print()` refused, the
  helpers of its package followed and their module state copied, and the
  objects of its classes compared, as a kernel in a source tree does. The
  package is read off the module's `__package__` too, so a kernel file that
  `lanky check` imports by path under a name of its own, or that `python -m`
  runs as `__main__`, keeps the package it sits in. loopty's dependencies are
  machinery whatever directory they come from, and so is the standard library,
  by name where its code is where the standard library is (a `colorsys.py` of
  the author's next to the kernel is the author's). Another installed package
  is a library's for a kernel outside it, and its call locations are passed
  over rather than disabled for good, so that a kernel of its own traced later
  is watched.
- The outside-state snapshot reads an object's slots along with its
  `__dict__`, and a ragged `Arr`'s offsets along with its values. An object of
  the kernel author's class with `__slots__` was not copied at all, so
  `state.count += 1` on a global or closure-held one traced, and a write into
  the offsets of a global ragged array went unseen. Every slot named along the
  class's MRO is read (a private one by its mangled name, an unset one as
  unbound, a base's slot that a subclass declares again as `Base.x`, and a
  `__dict__` entry a slot's name hides as `__dict__['x']`), and the offsets
  are copied as a second buffer, named `rows.offsets` in the message.
- A kernel with statements at two depths of one loop lowers and runs: `z[r] =
  1.0` after a dense loop over `j` that writes `y[r]`, or before it, and two
  inner loops side by side in one outer loop. loopy defines each iname in one
  domain, each statement contributed its domain over every loop around it, and
  loopy refused the second domain that defined `r` with a bare `RuntimeError`
  that the executor, `Schedule` and `loopty run` passed on. A statement's
  domain is now cut after every loop at which another statement leaves its
  nest, as a ragged one already was at its row, and an outer stretch drops the
  constraints of the loops inside it rather than projecting them out, so two
  inner loops over different extents do not leave the loop over `r` a union
  that is not convex. Every example lowers to the code it lowered to before.
  One name for two different loops, which only a term built by hand can have,
  is refused with a `LoweringError` naming the loop. See note 10 in
  `docs/loopy-notes.md`.
- A statement that reads what an inner loop writes stays outside that loop.
  loopy adds to an instruction whose loops are not final the loops of every
  instruction that writes what it reads, less those the writer's subscripts
  name, so `z[r] = z[r] + y[r]` after the loop over `j` that accumulates
  `y[r]` ran once per `j`, and a copy of `y[r]` before that loop saw all but
  the last update. A later loop that reads a nest's result, and a statement
  after a ragged inner loop, computed the wrong values the same way before
  this batch; the differential test refuted them, and `LoopyExecutor.run`
  returned them. The loops of every instruction the lowering writes are now
  final. See note 12 in `docs/loopy-notes.md`.
- An inner reduction bounded by an expression affine in an outer reduction's
  binder (`reduce_sum(a[i, j] for j in Fin[i + 1])` inside a sum over `i`) is a
  triangle, not a ragged fiber. The inner domain names the binder as a
  parameter, and `data_dependent_inames` counted every parameter that was not a
  size as data read out of an array; an enclosing binder, or a loop of the
  statement, is not. A parallel tag on a reduction nested in another is
  refused as unbuildable with its own reason, the limit loopy actually has
  (the enclosing reduction's accumulator is set outside the inner loop, by
  instructions that do not run on its axis), where it used to be refused only
  by accident, as a ragged fiber, and not at all when the inner bound was a
  size. See note 11 in `docs/loopy-notes.md`, which also lists three limits
  the check does not know yet.
- A term built by hand that holds one `Reduction` object in two statements
  lowers and runs. The lowering planned a reduction's inames by the object's
  identity, so the second statement's plan replaced the first's, both
  instructions reduced over one iname, and loopy stopped with a `CycleError`.
  A plan now belongs to a reduction in a statement, as if each statement had
  its own copy.
- The differential test judges each cell by `loopty.tolerance.disagreement`,
  the comparison the faithfulness fact makes. An infinity both runs computed
  agrees, where `|inf - inf|` was NaN and the search for the worst cell then
  raised `ValueError` out of `differential`; a NaN both runs computed in an
  `exact` output agrees, where it was refuted; and `exact` compares bits, so
  `-0.0` against `0.0` is a difference, as the docstring always said. The
  `difference ... within ...` line reports the closest finite cell, or `inf`
  for a disagreeing cell that is not finite, and outputs of different integer
  widths are compared in the type both promote to.
- An expected value that is not finite has no allowance in
  `loopty.tolerance.disagreement`: `eps_class * (|inf| + 1)` is infinite, so a
  finite value, or the infinity of the other sign, agreed with an expected
  infinity under `approx` and `reassoc`. This is the comparison of the
  faithfulness fact as well as of the differential test.
- The in-bounds fact of a read of the offsets a ragged access is flattened
  through says which access it serves: `off[r + 1], the end of row r that
  val[r, j] is flattened through, is in bounds ...`, with `layout` in its
  provenance saying the same, and `read directly and as ...` when the kernel
  also reads it itself. The source never writes those reads, so a refuted one
  used to name a read nobody could find in the kernel. No example declares its
  offsets, so no transcript changes.
- The C of a kernel with an `exact` output carries GCC's own pragma,
  `#pragma GCC optimize ("fp-contract=off")` behind a guard that keeps it from
  other compilers, beside the standard one GCC ignores. The build loopty runs
  was pinned by `-ffp-contract=off`; the source `loopty run --emit-code` prints
  was not, and GCC in a GNU dialect contracts by default whenever `-march` gives
  it an FMA instruction. `tests/test_contraction.py` compiles the emitted
  source by hand to show it, and on hardware with FMA runs it. See note 9 in
  `docs/loopy-notes.md`.
- `tile(second, first, ...)`, with the loop that comes second in the nest
  named first, is a tiling like the other: `skew("i", by="t").tile("i", "t",
  4, 4)` and a transpose's `tile("j", "i", 2, 2)` used to be refused as "not
  single-valued", because the second split was written against the position
  the first split had already moved.
- A statement in one of two tiled loops and not the other, such as the clear
  of a row before a loop over its ragged fiber, is split by its own loop and
  checked, as `split_iname` splits it in the kernel. The two splits of a tile
  are one map now, and a map applies to a statement in only some of its loops
  when it is that statement's part side by side with the rest; a skew or a
  diamond mixes its loops, and a statement in only one of them is refused
  with a `ValueError` naming it.
- A split of a reduction loop into a name the kernel already uses (a loop, a
  size, an array) is refused with a `ValueError`, like the split of any other
  loop, where it reached isl's "non-unique var name" from inside loopy; so is
  a new loop named like an argument the lowering adds, such as a ragged
  array's offsets.
- A skewed loop keeps its tag in the kernel. The skew went through
  `lp.map_domain` and back, which dropped it: a loop the schedule checked as
  a local axis ran one iteration at a time.
- The target-capability check knows the other reductions loopy 2025.2 will not
  realize, and asks about the ones it knew the way loopy does. A reduction on a
  group axis (or on `ilp.seq` or `vec`), a reduction split with both halves on
  local axes, and a reduction on a local axis whose extent has no numeric
  maximum (`reduce_sum(a[i, j] for j in Fin[i + 1])` with `j` on `l.0` and `n`
  free), or inside a statement loop on a local axis that has none, passed
  `buildable` and then failed in code generation; each is a `refuted`
  `buildable` fact now, with its cause in words. A reduction's loops are
  classified with loopy's own tag classes, as `realize_reduction` classifies
  them, so an `ilp` loop, which loopy unrolls, is a sequence, and split with
  its other half on a local axis it is refused, as loopy refuses it. A
  reduction over an `ilp` loop is refused on its own account as well: loopy
  privatizes the accumulator along the loop and then refuses the instruction
  that initializes it, under some string hash seeds and not others. Split
  with its other half untagged it used to be refused as partly parallel,
  which it is not, and a reduction over one `ilp` loop was not refused at
  all; `unr` unrolls the sum in order and builds. The extent is asked of the
  loop's bounds with `static_max_of_pw_aff(..., constants_only=True)`, as
  loopy asks it. The tests generate the code with loopy's plain OpenCL target
  and see loopy's own error for each; note 11 of `docs/loopy-notes.md` has the
  table.
- A tile or an interchange that orders a loop outside a loop loopy nests it
  inside is a `refuted` `buildable` fact. loopy nests a ragged fiber's domain
  inside its row, the loops below a statement at a shallower depth inside the
  loops around them, and a statement loop inside the row of a ragged
  reduction in its body, and when the loop priority disagrees it drops the
  priority and runs a nest of its own choosing, which the cast facts did not
  check: `tile("r", "j", 2, 2)` on the ragged recurrence `w[r + 1, j] = w[r,
  j] + val[r, j]` was decided and compiled to code that ran the dependence
  backwards, and so was `tile("r", "k", 2, 2)` on `w[r + 1, k] = w[r, k] +
  reduce_sum(val[r, j] for j in val.dom[r])`. The nesting is read after every
  step with loopy's own `find_loop_nest_around_map`, and the reason names the
  two loops, why loopy nests one in the other, and the interchange that puts
  them right. Buildability is now asked of the schedule as it stands rather
  than kept once lost, so that interchange makes the tiled schedule buildable
  again, and the schedule then carries no `buildable` fact; a schedule's one
  `buildable` fact is about its current reason. A kernel `affine` could not
  write stays unwritten. See note 6 in `docs/loopy-notes.md`.
- Two schedules of one kernel in one file keep their own facts in the ledger.
  A cast fact's id named the kernel and the position of the step
  (`cast:spmv:0:bijective`), and an agreement fact's the kernel alone, so the
  second schedule's facts replaced the first's in `loopty run`'s ledger. The
  ids now name the schedule: `Schedule.key` is the kernel, the target and
  every step with every argument it was given
  (`spmv[c].split('j', 2, inner='j_in', outer='j_out')`), a cast fact's id is
  `cast:`, the key up to its step, and its kind, and an agreement's is
  `agreement:` and the whole key. Two schedules that begin alike share the
  facts about those steps, which are the same claims, and a schedule run twice
  from one file, on two sets of inputs, keeps both agreements (`#2`). Two
  exactness facts of one step, when one tag reassociates two accumulations,
  are told apart by the array. `examples/wavefront_acoustic.py` runs the
  diamond under `loopty run` beside the wavefront block, which it left to
  `python` for this reason, and its transcript is regenerated.
- A fact the isl oracle refutes over a domain a guard left wide says so in its
  reason, which `lanky check` and `loopty run` print under its `REFUTED` line:
  the domain is wider than the instances that write, so the witness may be an
  instance the guard masks, and each conjunct left out is named with the
  reason it was left out. It was in the provenance and the JSON ledger only,
  so a refuted in-bounds fact read as an out-of-bounds read. A fact refuted
  over a domain its guard narrowed whole, and a decided one, say nothing more.
- `Schedule.tag` refuses a name that is neither a loop nor the loop of a
  reduction, and a tag loopy cannot read, with a `ValueError`, before anything
  else. loopy's `tag_inames` was the only check, and it is not asked once a
  step has left the schedule with no kernel, so such a tag was accepted there
  with a decided `bijective` and `monotone` fact; on a schedule with a kernel
  the unknown name was loopy's `LoopyError`.
- `loopty run` reports whatever a run raises by its type and message, counts
  the kernel as failed, goes on to the file's other kernels, and exits 1. Only
  the errors it expected were reported (`IndexError`, `ArithmeticError`,
  `ValueError` and a few more); a body's `KeyError` ended the command with a
  traceback. So is an error in the file's `example_inputs()`, or in code
  generation under `--emit-code`, which were not inside what a run reports at
  all. A native `TraceError` in `LoopyExecutor.differential` is not
  raised but returned, as a `refuted` agreement fact with the refusal as its
  reason and no outputs, the way the faithfulness fact counts it, so
  `loopty run` prints it under the fact's `REFUTED` line.

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
- **`import loopty` imports neither loopy nor islpy.** Each top-level name is
  imported from its module the first time it is used, and the executor imports
  the lowering when it first lowers, so a file of kernels imports no loopy, and
  neither does lanky's plugin discovery, which loads loopty's entry points for
  every command. Measured once, `import loopty` went from about 160 ms to
  about 1 ms, and importing the names a kernel file uses, or loading the
  plugins, from about 160 ms to about 100 ms, which is numpy, pymbolic, lanky
  and islpy. `kernel` and `trace` stay the functions after their submodules are
  imported.
- `loopty.sum` is no longer in `__all__`, so `from loopty import *` leaves the
  builtin `sum` alone. The alias itself remains.
- The per-class tolerances (`TOLERANCE`, `TOLERANCE_FLOOR`) and the class an
  output is compared at live in `loopty.tolerance`, which the differential
  test and the faithfulness fact both read, so that the fact needs no loopy.
  `loopty.executor` still exports `TOLERANCE`, and `exactness_of_output`.
- `islpy` is pinned below 2026. loopy 2025.2 calls `Aff.is_equal` during code
  generation for a tiled loop nest, and islpy 2026 removed it. (loopy's
  `map_domain` calls `BasicMap.is_bijective`, which islpy 2026 removed too, but
  loopty no longer calls `map_domain`.) Drop the ceiling once a loopy release
  supports islpy 2026.
- **`skew`, `split` and `tile` are affine maps.** Each states its reindexing as
  an isl map from the loops it replaces to the loops that replace them, and one
  builder turns that map into the step the checker asks about; each used to
  write its constraints in the checker's padded coordinates by hand. `skew` is
  `affine` with a map that keeps both names, and its kernel goes through the
  same rewrite rather than `lp.map_domain`; `split` and `tile` still split the
  kernel with loopy's `split_iname`. A `skew` or `tile` of a loop with itself,
  and a new loop that would share its name with another loop, a size or an
  array, are refused with a `ValueError` naming the problem, where they used to
  fail from inside isl or loopy.
- `BasicMap.is_bijective with implicit conversion` is no longer exempt from the
  suite's deprecation errors: it was raised by `lp.map_domain`, which nothing
  in loopty calls now. The one test that calls it, to pin that loopy refuses
  the diamond, silences it locally.
- `lanky>=0.1.0.dev0` is a dependency, resolved from a sibling checkout by
  `[tool.uv.sources]` during development.
- **A program's restatement of a callee's postcondition rests on the callee's
  fact.** `Program.facts` pointed at the callee's postcondition with a `from`
  entry in the provenance, which lanky had no way to read, so the ledger
  showed each restatement as an assumption standing on its own. It now sets
  lanky's `Fact.rests_on` to the id of that fact, built by the new
  `loopty.typing.postcondition_id`, which the kernel's own postcondition fact
  uses too, so the two cannot drift apart. The ledger names the callee's fact
  beside the restatement, as in `assumed under scan:postcondition`, counts it
  in what the restatement is worth, and `lanky check --json` carries
  `rests_on`, `effective` and `under`. The `from` entry is gone; `callee`
  stays. The `lanky check` transcripts of `examples/spmv.py` show the new row,
  and this needs the lanky that has `Fact.rests_on`.

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
- The test suite treats `DeprecationWarning` as an error. Two exemptions are
  loopy's own and are listed in `pyproject.toml` and `tests/conftest.py`, with
  the reasons in `docs/loopy-notes.md`.
