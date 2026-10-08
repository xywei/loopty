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
  coordinates) and on the acoustic pair (against the native run). A map the
  rewrite cannot write for loopy (loops no one domain defines, an image that
  is not one basic set, a piecewise inverse) is a `refuted` `buildable` fact,
  and the schedule has no kernel from then on. `examples/wavefront_acoustic.py`
  tries the diamond four ways: with space first it is refused with a witness,
  with time first it is accepted and runs, tiling it is refused, because the
  pair needs an offset between its statements, and tiling it with that offset,
  a map per statement (below), is accepted and runs. Note 13 in
  `docs/loopy-notes.md` has the details, and `loopty.oracle.is_bijection_on`
  is the totality-and-bijectivity question the first fact asks.
- **A loop over a lattice counts its steps** (`Schedule.strides`). loopy
  loops over the bounding box of a domain with an existentially quantified
  constraint and tests the constraint inside the innermost loop, so the
  diamond's loop over `b` tested the parity of `a + b` at every `b`
  (`if (-b - a + 2 * ((b + a) / 2) == 0)`) and did nothing at half of them.
  Once a step has set the nest, the kernel code is generated from asks isl
  for the stride of each loop given the loops outside it, and replaces a loop
  that has one by a counter of its steps, `b = 2*b_step - a`, in the domain,
  whose preimage has no holes left, and in every instruction. The counter and
  the loop increase together for fixed outer loops, so the instances and their
  order are the ones the checker approved, and isl confirms for each loop that
  the new domain maps back onto the old one. It is done on the kernel code is
  made from, not on the one steps transform, so a later tile still splits the
  loop the checker knows, and the loop counted after a tile of the diamond is
  `b_inner`. `Schedule.strides` names each loop replaced and its expression,
  `{"b": "2*b_step - a"}`. The diamond on the stencil and on the acoustic pair
  compiles with no parity test and still agrees bit for bit at sizes of both
  parities, and a guard that narrows a loop to a congruence (`when(i % 2 ==
  0)`) is stepped over too. A loop with a tag, the loop of a reduction, a loop
  another domain names, and a loop whose offset involves a loop not around all
  its instructions keep loopy's test. Every other schedule of the examples
  generates the code it did (#45).
- **A map per statement in `Schedule.affine`.** A union map whose input tuples
  name statements moves each statement by its own map:
  `affine("{ S0[t, i] -> [a, b] : a = t + i and b = t - i; S1[t, i] -> [a, b] :
  a = t + i and b = t - i + 1 }")` puts the pressure update of the acoustic
  pair half a step after the velocity update along the diamond, which is the
  offset a diamond tiling of the pair needs, and `.tile("a", "b", 4, 4)` after
  it is accepted where the tiling of the plain diamond is refused. The
  `bijective` and `monotone` facts are asked of the maps together, over the
  dependences between the statements as well as within each: `S1` put before
  the `S0` whose velocity it reads is refused with that pair as the witness.
  The statements of a loop keep sharing its loops in the kernel, since loopy
  gives them one domain: the new loops run over the union of the images (its
  polyhedral hull when the union is not one basic set), each instruction is
  predicated on its own image and reads its old loops back from its own
  inverse, and the instruction that computes a ragged row's length moves with
  the statement whose fiber it bounds. Maps that are all one map build that
  map's kernel. Every statement in the loops the maps name has to run in all
  of them and be given a map, the maps have to take the same loops to the
  same new ones, and a map may not name its output tuple, or a statement the
  kernel does not have, and a statement may not be given two maps; each is a
  `ValueError` naming the statement. Two statements of one ragged fiber moved
  by different maps are a `refuted` `buildable` fact, since the fiber is one
  loopy domain, and so are two fibers of one row, whose length one instruction
  computes for both. The diamond tiling of
  `examples/wavefront_acoustic.py` agrees with the native run bit for bit, and
  `loopty run` compiles it as a third schedule of the kernel (#46).
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
- **Array arguments over polyhedral domains** (`loopty.domain`, #12). The
  decision's slice binders: `Where[i: Fin[n], j: Fin[n], j < i]` is a box cut
  by constraints (the lower triangle; a band is `(i - j <= 1) & (j - i <= 1)`),
  `Sigma[i: Fin[n], Fin[i + 1]]` a sum whose fibers are affine in the binders
  before them, and `Fin[n] + Fin[m]`, lanky's `SumType`, a union of pieces
  whose points are `(p, x)`. Python reads `i: Fin[n]` in a subscript as a
  slice and lanky's annotation scope invents `i`, so nothing is parsed. A
  constraint is a conjunction of comparisons of quasi-affine terms, which isl
  states exactly; `!=`, `|`, a non-affine term and a non-linear bound are
  refused where the annotation is evaluated, since a domain wider than it was
  written would decide a read outside it in bounds. Such an array has its
  domain in `ArrType.domain` and no axes of its own. `.dom` runs binder by
  binder: `L.dom[i]` over the points the constraints on the first two axes
  allow at `i`, the traced loop carrying those constraints as the native one
  applies them, and a fiber at a point outside the domain is empty in both. A
  union's pieces are walked by number, which a trace runs as the Python loop
  it is; a piece is chosen by a Python integer (anything else, a piece that is
  not there, and a reduction over the pieces are `TraceError`s naming the
  fix), and so is a fiber taken at a point isl cannot state. In-bounds
  obligations are stated over the exact set (`flow.cell_set`), so `L[i, i]` is
  refuted although the box around the triangle has the cell. The contract
  refuses, at every entry point, an argument whose points are not the declared
  domain's at the sizes the call determines, compared as sets and not as
  spellings, and a plain `ndarray`. Two layouts store the same array: the box
  of the binders, which a single domain lowers to as an array of that shape
  and a union to pieces one after another at bases that are sizes, and packed
  rows, the domain's cells in lexicographic order read as `L[off_L[i] + j]`
  through a table of row starts less their first column, which the executor
  computes from the domain and passes as it passes a ragged array's offsets.
  A domain whose rows skip columns cannot be packed and is refused. At run
  time, `Arr.zeros(domain, n=..., storage=...)` and `Arr.from_cells(domain,
  values, ...)` build an array over a domain (a kernel's own is
  `kernel.arg_types[name].domain`); it is indexed at the domain's points and
  refuses any other cell, and `Arr.cells()` reads it in one order whatever its
  layout, which is what the contract, the differential test and the
  faithfulness fact compare. A run is over the declared domain whatever the
  argument's spelling: natively, an argument written otherwise is copied into
  an array over the declared domain for the call, since `L.dom[i].size` is
  the binder's bound and `L.dom` runs binder by binder, and a compiled kernel
  addresses the declared domain's layout at the call's sizes, the executor
  copying an argument into it and back when its storage differs, or its box
  does (the strict triangle written `Sigma[a: Fin[n], Fin[a]]` has the
  declared points in an `n x (n - 1)` box). The executor also passes the
  sizes the call determines to a kernel whose flat buffers give loopy none
  (note 15 in `docs/loopy-notes.md`).
  A box extent that can be negative at some size, `n - 1` at `n = 0`, is
  neither a shape nor part of where a piece starts, since the domain is empty
  there and its box has no cells: isl decides which extents are never
  negative, a single domain with another one is a flat buffer, and a piece
  after one starts at a value argument the executor computes. A size a
  binder's bound runs up to is a non-negative integer, as the isl set and
  the boxes assume, and `Arr.zeros` and the contract refuse any other. The
  faithfulness fact draws such arrays from the declared domain. This needs
  the lanky that has `SumType`.
- **`Schedule.pack(*arrays)`.** Store arrays over a domain packed. Not a cast:
  no instance moves and no fact is emitted, since a layout says where a cell
  is kept and every fact is about the cells. The schedule is lowered again
  with the arrays packed and every step so far replayed, so each is checked
  against the kernel that stores them so, and the step is in the schedule's
  key, its history and what `retarget` replays. `lower_generic` and `lower`
  take the layouts as `layouts=`, and `Lowering.storage` and
  `Lowering.tables` record them.
- **A sixth demo**, `examples/pairs.py`: symmetric pair interactions over the
  lower triangle, one statement writing each pair once and a second summing a
  particle's row and column (`f[k, p]` under `k > p`). Every in-bounds
  obligation is decided by isl over the exact triangle, and `loopty run`
  compiles it boxed and packed, each agreeing with the native run. Its section
  in `examples/README.md` and the excerpts in `README.md` and
  `docs/quickstart.md` are generated by the refresh script.
- **A program's term, and one loopy kernel for it** (`loopty.compose`).
  `Program.term` is what the body does run once against placeholders: each
  kernel call is recorded instead of run, with its arguments by parameter,
  and the callees' terms follow one another in call order, in the program's
  names, with no AST pass. A callee's arrays and scalars become what the
  program passed; a number passed for a scalar is substituted, and checked
  against the scalar's sort when the term is built, since no contract sees it
  afterwards (a `Fin[n]` whose `n` is known only at the call refuses a
  number). Its sizes are unified through the arrays, a program array having
  the type of the first call that passes it and every later call having to
  agree, so that `scale`'s `n` is `scan`'s `n + 1` when both are handed `off`,
  and `n + 1` against `n + 2` makes one size the other plus one; a scalar
  that sizes an array sizes it in the program's names too, and two sizes
  nothing shows equal are refused. Its loops, reduction binders and reflected
  parameters get names no earlier call has; its statements are named after the
  call (`scan.S1`, and `step@2.S0` in a second call of `step`) and keep their
  own `file:line`. A program called by a program is recorded in place, and
  its kernels' postconditions are restated in the calling program's scope
  too (`Program.callees` follows the programs a body names).
  Dependences across kernels are not declared: an array one call writes and a
  later call reads is one array of the term, so the edge is in the
  footprints, the lowering orders the instructions by it, and a
  `Schedule(program)` checks casts against it. The term lowers into one
  kernel whose loops run in call order, which `LoopyExecutor.run` and
  `.differential` accept like a kernel's, and `loopty run` compiles every
  program of a file and compares it with its native run in an agreement fact
  of its own, placed at the program. Fusion is not done. A body that reads,
  writes, computes with, iterates over or branches on an argument, or hands
  it to numpy, is refused with a `TraceError` naming the fix, and so are a
  loop whose trip count is an argument, an array from outside the program,
  one array for two parameters of a call, an array given for a scalar, a
  parameter no kernel is given, a parameter with a default, two element sorts
  for one array, two calls reading one ragged family through different
  offsets, a body that returns an array it made, a program that writes none
  of its parameters, and a callee with an array over a `Where`, `Sigma` or
  union domain, whose sizes are not unified across calls. The compiled
  program's contract checks its arguments once, when it starts, so an array
  whose cells a callee's contract checks and its in-bounds facts rest on (an
  element sort `Fin[m]`, the counts or the offsets of a ragged family it
  reads) is refused when an earlier call wrote it or the program made it:
  `perm[i] = i + 1` in one kernel would be an address past the end of `x` in
  the next one's `x[perm[i]]`, where the native run is refused by that
  kernel's contract.
- **An array a program makes is a temporary** (`Arr.zeros_like`,
  `Term.temporaries`). `Arr.zeros_like(u)` is zeros laid out as `u` natively,
  and inside a program being traced it makes a placeholder, named after the
  variable it is stored to, which the program's term keeps as a temporary:
  typed by the kernels it is passed to, shaped like `u`, and zeroed by a
  statement of its own (`f.zeros`) where the body made it. The lowering
  declares it as a loopy temporary, not an argument, so the intermediate
  between two kernels is nobody's argument and has one declaration: private on
  C, which is a variable-length array on the stack of the call, and global on
  OpenCL, because loopy's C host code never allocates a global temporary (note
  16 in `docs/loopy-notes.md`). `Lowering.temporaries` names them, and a
  temporary's element sort counts in the exactness class and the contraction
  pin as a parameter's does. A ragged one is refused, and natively
  `Arr.zeros_like` of an array over a domain is over that domain, in its
  storage. A temporary of any sort has to be stored natively in a dtype that
  holds what the compiled one holds (`contract.native_storage`), or one run
  truncates, rounds or drops what the other keeps: `float64` for `Real`, the
  dtype itself for a numpy one (`np.float32` rounds, `np.complex128` keeps an
  imaginary part that a real array drops), `bool` for `Bool`, whose compiled
  byte holds a truth value as a bool does and on which natively only a bool
  has a logical `~`, and a signed integer of 32 bits or more for `Nat`, `Int`
  and `Fin[m]`. A `dtype` given to `Arr.zeros_like` is checked when the term
  is built, and one left to the parameter it copies when the compiled program
  runs (`Term.temporaries_like`, `contract.inherited_storage`): an integer or
  a `float32` `u` is refused for a real `f = Arr.zeros_like(u)`, a real one
  for a complex or a `Bool` `f`, and a real one for a natural `f` (#71). The
  refusal names the dtype to give `Arr.zeros_like`, and passing `u` in that
  dtype only when it holds `u`'s own sort too, so a real `u` is never told to
  be passed as a bool. The interpreter stores a temporary as the native run
  has to, and Python's `complex` lowers as `complex128`, as `float` lowers as
  `float64`, so the lowering and the storage check know the same sorts.
- **A term may state its offsets** (`Term.offsets`, `Term.offsets_of`). The
  offsets a counts family's rows are read through were always read off the
  names of the term's parameters, which is right for a kernel and wrong for a
  program, whose parameters are named by the program: `solve(cnt, col, val, x,
  y, off)` would have indexed `spmv`'s rows through `scan`'s output `off`,
  which `spmv` never declared, and its contract would have refused a zeroed
  `off`. A program's term states the family's offsets as each call's kernel
  reads them, in the program's names, or `None` for the array's own; the
  lowering, the access collector, the interpreter and the sampled inputs of
  the faithfulness fact all ask `Term.offsets_of`, and an added offsets
  argument avoids every name the term has (`off_cnt_`). `Term.where` places a
  program's facts at the program, and `Term.array_types` is the parameters'
  and temporaries' types together.
- **A seventh demo**, `examples/composition.py`: a Burgers flux and its
  divergence, composed by a program through an array it makes, which runs
  natively, prints its term and the one kernel loopy generates for it (with
  `double f[n];` declared inside), and agrees compiled. `examples/spmv.py`
  gives its `solve` program example inputs, so `loopty run` compiles it too;
  the transcripts are regenerated.
- **A program has a `trace-faithful` fact** (#66), the last of
  `Program.facts()`, as a kernel's is the last of its own: the program's term,
  interpreted, against its body, run natively, on the module's example inputs
  for it and on inputs drawn from the types the term gives its parameters.
  The term is built from what the body does with placeholders, and a body can
  look at an argument in a way no placeholder sees: `if isinstance(x, Arr):
  scale2(x)` left `scale2` out of the term, which only a differential run on
  a file with example inputs caught, while `lanky check` said nothing about
  the program and a `Schedule` of it decided its casts against a term the
  body does not compute. Such a program is now refuted, with the input and
  the first differing cell, and `lanky check` exits 1. A program whose term
  cannot be built has the fact `assumed`, with the composition's refusal as
  its reason (`loopty.faithful.no_term_fact`). The spmv and composition demos
  have one row more each, `solve`'s and `burgers_rhs`'s, both `tested`, and
  `Program.facts()` is computed once, as a kernel's is.
- **Facts travel between a program's calls** (#65, #13). A kernel's
  requirements on its inputs are its argument types, and two of them are
  about what an array's cells hold: an element of a `Fin[m]` sort is a point
  of it, and the offsets a ragged family is read through are the ones its
  counts give. #58 refused a program in which an earlier call writes such an
  array and a later call is passed it, since the compiled program's contract
  checks its arguments only when it starts. The requirement is now an
  obligation of the program (`Term.requirements`, `loopty.term.Requirement`),
  decided by isl under the hypotheses that held at the call: the earlier
  callees' postconditions that nothing has written over since, the zeros an
  `Arr.zeros_like` starts an array with, the types the program's contract
  checks of what nothing has written yet, and the theorems the program cites
  with `@program(uses=[scan_monotone])`. `loopty.hypotheses` asks the
  question: every cell a claim or a hypothesis reads is an isl parameter of
  its own (`off[q]`, `cnt[q - 1]`, keyed structurally after putting the index
  in a canonical affine form), a universal hypothesis is instantiated at the
  cells the claim reads, for three rounds, and a part isl cannot state is
  dropped toward the safe side, in negation normal form. A theorem is
  instantiated at the arrays its hypotheses match, and only where each
  array's cells are points of the family's sort when the call is made. A
  decided requirement is a `requirement` fact whose term is the empty set isl
  answered, resting on the facts of the hypotheses it used and on nothing it
  did not: #65's `permuted` program has `gather`'s requirement decided under
  `number`'s postcondition, worth `tested`, and a scan whose offsets a later
  call reads rows through has the layout requirement decided under `scan`'s.
  Where the hypotheses do not decide it, the requirement is a checked point:
  the lowered program has a statement between the two calls that sets a
  one-cell flag where a cell fails, every later statement is guarded by the
  flag, and the executor and the interpreter raise the requirement's message
  (`loopty.interpret.CheckFailed`), so the compiled program stops where the
  native one is refused; the fact stays `assumed`, and says why. Offsets an
  earlier call writes before any call reads rows through them are no longer
  compared with the rows' own offsets when the compiled program starts
  (`Term.deferred_offsets`), as natively the call that writes them does not
  read through them either. A count an earlier call wrote is still refused.
- **A callee's fact can be decided where it is called**
  (`loopty.typing.scoped_in_bounds_facts`, `Term.scopes`). A flat access,
  `val[off[r] + j]`, is `assumed` in its kernel's ledger; in a program that
  calls the kernel after the scan that wrote `off`, it is decided under the
  scan's postcondition, through the cells `off[r]`, `off[r + 1]` and `cnt[r]`,
  as a fact of the program's resting on what it used.
- **A kernel's postcondition is tested** (`loopty.faithful.postcondition_fact`).
  It is evaluated at what every native run of the `trace-faithful` fact left
  in the arguments, on the module's example inputs and the drawn ones (those
  after an input the term differs at run natively too, since that settles
  the comparison and not the postcondition), and is
  `tested` by `native` when it held after each, `refuted` with the input
  after which it did not, and `assumed`, with the reason, when nothing ran or
  it could not be evaluated after some run (it reads a cell the run's arrays
  do not have, say). Its term is a `loopty.typing.AfterCall`, which no oracle
  takes for a closed proposition. A program's restatement of it is `decided`
  by the call, so it is worth what the postcondition is: `scan`'s in
  `examples/spmv.py` reads `tested` where both rows read `assumed`. Deciding
  a postcondition from the term, by exact dataflow, is not done yet.
- **An eighth demo**, `examples/travel.py`: #65's permuted program, a scan
  whose offsets a later call reads rows through, a permutation checked when
  the compiled program runs, and a flat access in bounds where it follows the
  scan. `examples/README.md`, the README and the quickstart describe it, and
  the spmv transcripts are regenerated.
- **Definedness of what passes between a program's calls** (#13,
  `loopty.flow.definedness`, `loopty.typing.definedness_facts`). An array a
  program makes with `Arr.zeros_like`, written by one call and read by a
  later one, is an internal edge of the program, and the program has a
  `definedness` fact for each call that reads it after another wrote it:
  every cell the call reads, a call before it stored, so that the zeros the
  array was made with reach none of its reads but through a call that adds
  to a cell. The fact's term is the isl subset question between the two
  sets of cells, for the isl oracle. A call that reads a cell no call before
  it stored, and that it does not store itself, reads the zeros there, and
  the fact is `refuted` by isl with that cell, as a kernel that writes one
  cell twice has its `disjoint-writes` fact refuted: the program runs as
  written, but the edge does not carry what is read, and the reason names
  the fix, a producer that stores those cells too. A read or a write isl
  cannot list (an index that is not affine, a guard it cannot state), or a
  read of a cell the reading call stores itself, before the read or after
  it, leaves the fact `assumed`, with the reason. The composition demo has
  one row more, `decided`.
- **Fusion as a checked cast** (#13). `Schedule.affine` takes maps per
  statement whose statements run in different loops, each taking its own
  loops to the same new ones, which is a fusion: `{ flux_S0[j] -> [j];
  divergence_S0[i] -> [j] : j = i + 1 }` runs a program's two calls in one
  loop, checked as every map per statement is, on the dependences between
  the calls as well as within each. `Schedule.fuse(producer, consumer,
  shift=0)` builds that map from two statements, or two calls by their
  labels, and their loops, outermost first, one shift per loop. A fusion
  that runs a dependence backwards is refused with the pair of instances and
  the cell, and the message names the least shift the checker accepts, when
  a number per loop gives one. A label names the call's own statements and
  not the checked points a program puts before it, so a fusion that would
  run the call before such a check is refused for the flag the check sets
  and the call reads; the check fuses with the producer instead, and stays
  before the reads. The kernel rewrite replaces the two loops'
  domains by one, the union of the images, each statement predicated on its
  own and given its own inverse, and nests the domains again; the union of
  two loops that run to two sizes, `n` and `m`, is bounded by `n + m`, its
  hull where the sizes are not negative, which they never are. A nest the
  lowering wrote as one domain, `{ [i, j] }`, is cut after the loops the
  maps take when they are its outer ones, so the rows of a two-loop
  producer fuse with a one-loop consumer of each row; other loops that share
  a domain with a loop no map takes leave the kernel unbuildable, with the
  reason. After a fusion, and after a substitution, the statements'
  instruction dependencies are drawn again from the dependences the casts
  are checked against, in the order the schedule puts the statements, and
  two that touch one variable and that no dependence joins are marked as
  needing no order (`no_sync_with`): the lowering draws them by array, in
  term order, and a call between the fused ones that the second reads only
  at cells it never writes left loopy no order (a `CycleError`; note 21 of
  `docs/loopy-notes.md`). Maps per statement that take a loop in common still have to
  take the same loops, and every statement in a loop some map takes has to
  be given one that takes all of them.
- **Storing an intermediate, or not** (#13). `Schedule.substitute(array)`
  computes an array the program makes where it is read: the one statement
  that writes it, pointwise, becomes a substitution rule through loopy's
  `assignment_to_subst`, after the statement that zeroes the array is
  dropped, and the temporary and the loops left empty go with it. Each read
  gets the value converted to the array's element type, as the store
  converted it (a `float32` intermediate rounds it); where that conversion
  would sit in a subscript, which loopy cannot simplify through, the kernel
  is left unwritten with the reason. It is
  refused with a `ValueError` for a parameter, an array two statements
  write, or a producer that is not pointwise, and with an `IllegalCast` when
  a read is of a cell the producer does not store, or stores after the read
  (the `definedness` fact of the step), or when something writes what the
  producer read between its run and a read of what it stored (the
  `monotone` fact). The dependences of the dropped statements go, those of
  the producer's reads are carried over to the reads that replace them, and
  every later step is checked against the result, so a fused loop that
  carried the array from step to step may take a hardware axis once it is
  substituted; the kernel's instruction dependencies are drawn again from
  them, so a later write of what the producer read waits for the reads that
  compute it again. Contraction to the cells live at once is not done.
- **A ninth demo**, `examples/fusion.py`: the Burgers flux and divergence of
  the closed #6, rebuilt on these. The fusion without a shift is refused with
  its pair, the fusion one step behind is decided and compiled to one loop,
  and the substituted schedule stores no flux; both runs agree with the
  native one. `examples/README.md`, the README and the composition demo
  describe it, and the composition transcripts are regenerated.

### Fixed

- The cells of a dense array are stated over dimensions none of whose names
  is a size: over `x: Arr[Fin[a0], Real]` they were `0 <= a0 < a0`, which has
  no points, and `x[i]` was refuted. The dimensions of a domain's set get the
  same care (`loopty.domain.dimension_names`).
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
- The schedule checker and the typing rules see the read a ragged loop's bound
  makes. `for j in val.dom[r]` and `reduce_sum(... for j in val.dom[r])` run
  to `cnt[r]`, which lowering assigns once per row, but the read was only a
  reflected parameter of a domain and in no statement's accesses:
  `tag(r="l.0")` on a loop that sums row `r` and then clears `cnt[r + 1]` was
  accepted, and could sum row `r + 1` on either side of the change to its
  length; it is refused now, with that pair as the witness.
  `flow.layout_reads` lists the read, `cnt[r]`, or `off[r]` and `off[r + 1]`
  in a term built by hand whose counts are not a parameter, on every
  statement the bound bounds, over the loop nest up to the row, since the
  bound is read once per row whatever the loops inside it or a guard do. The
  fiber of another row, `val.dom[r - 1]` or `val.dom[p[i]]`, which lowering
  cannot assign, has its length read where its loop starts, with whatever the
  row expression reads. `flow.statement_accesses` lists these reads after the
  source's accesses, so both dependence relations, the in-bounds rule and the
  lowering's instruction order read them. Each is an in-bounds fact that says
  what it serves, `cnt[r], the length of row r that bounds the loop over j,
  is in bounds ...`, which is where counts declared a cell short, and the
  length of row `-1`, are refuted. The recognition of a ragged bound
  parameter moved from the lowering next to the collector, as
  `flow.ragged_bound_params`; the spellings `COUNT_PARAM`,
  `COUNT_PARAM_REFLECTED` and `count_param_names` live in `loopty.term`, and
  are still importable from `loopty.lower`. The `lanky check` transcripts of
  `examples/spmv.py` have one more row, and the ledger of `examples/p2p.py`
  three more.
- `off[r + 1]` is listed only where the lowered code reads it: where a row's
  length is computed from the offsets. The flat index of `val[r, j]` reads
  `off[r]` alone, so a kernel whose counts are a parameter, which a traced
  kernel's always are, never reads where a row ends. Storing `off[r]` before
  summing row `r` through it was refused a parallel tag over that read
  (`S1[r=2] reads off[3] overwritten by S0[r=3]`), and is accepted again. The
  in-bounds facts follow the reads: offsets of `n` cells beside the counts are
  decided for the row starts the code reads, and a call with them is still
  refused by the contract, since they are not the layout of `n` rows.
- A kernel that writes its counts or its offsets means one thing however it
  runs. The native run and the term interpreter read a ragged array through
  the counts and offsets the kernel declares, as the kernel has left them
  (`Arr.through`, `loopty.term.declared_layout`), which is what the lowered
  kernel does with the arguments it is handed. They followed the array's own
  offsets instead: row sums followed by `off[r + 1] = ends[r]` computed
  `[3, 3, 15]` natively and `[3, 2, 9]` compiled, and only the differential
  run noticed. Every read through the declared layout is checked against the
  flat buffer, since rewritten offsets can point anywhere. The contract checks
  a declared offsets argument against the ragged argument on the native path
  too, as it did on the compiled one, so the two layouts agree when a run
  starts, and the faithfulness fact's samples pass the drawn array's offsets
  for such a parameter, where they drew them at random and would now be
  refused. One case still had two meanings, and lowering now refuses it: a
  loop inside row `r` that both rewrites `cnt[r]` and holds a statement
  bounded by it. The body reads the length where the loop over the fiber
  starts, once per iteration of that loop, and the lowered kernel once per
  row, so from the loop's second iteration on the two summed rows of
  different lengths; with the rewrite first in the body, loopy could not
  schedule the kernel at all.
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
  reason, which `lanky check` (and `loopty check`) prints under its `REFUTED`
  line: the domain is wider than the instances that write, so the witness may
  be an instance the guard masks, and each conjunct left out is named with the
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
- A hardware axis on the C target is a `refuted` `buildable` fact (#47): loopy's
  C code runs in one thread, and code generation stopped with "plain C does not
  have local hw axes" (or group) for a schedule the check had passed, including
  one retargeted to C by `loopty run --target c`. So is a `vec` loop that keeps
  a sum's accumulator in it, which C has no vector types for. The target's own
  limit is asked after every other, since it is the one `retarget("opencl")`
  removes; the spmv demo's device schedule, written for `"c"`, keeps its
  ragged-fiber reason, and its transcript is unchanged. The tests that tagged
  loops on `"c"` for the casts' sake assert the refuted fact beside the decided
  casts, or run on loopy's plain OpenCL target, now a shared `plain_opencl`
  fixture.
- An `unr`, `ilp` or `vec` loop whose length is not a number when the code is
  generated is a `refuted` `buildable` fact (#48). loopy writes such a loop
  out, and a loop over `Fin[n]` with `n` free failed inside isl with
  "unbounded optimum", naming no loop. The length is asked as loopy asks it,
  of the loop's bounds with every size and every other loop projected out, so
  a triangle inside a fixed extent builds, and the reason names the split that
  makes a loop of fixed length. The check for a hardware axis on a nested
  reduction no longer counts `ilp` as one; such a loop is refused for the
  privatization note 11 describes instead.
- The check knows loopy's rules for hardware axes (#55): the axes of a kind are
  numbered from 0 with none left out (`l.1` alone was "local axis 0 unused"),
  an instruction has one loop per axis, `vec` included, with a sum's loop
  counted as one of its statement's ("instruction 'S0' has multiple inames
  tagged 'l.0'"), and every instruction runs on every group and local axis the
  kernel uses, a sum's accumulator and a ragged row's length among them ("does
  not use all local hw axes"). `l.auto`, which loopy assigns only inside its
  own transforms, is refused too. Each is read off the kernel's own
  instructions and tags after every step.
- The same comparison with loopy's code generation, over every tag and pair of
  tags on a set of small kernels, found three more, now refused: a concurrent
  loop (`ilp` and `vec` as well as a hardware axis) on a ragged fiber, or on a
  dense loop the lowering defines in one domain with a fiber, which loopy
  refuses in any domain with a data-dependent parameter; a `vec` loop in which
  a ragged row's length is read, or around a sum on a local axis, which failed
  with a `TypeError` inside loopy. `tests/test_buildable.py` asks each case of
  the schedule and of loopy, with loopy's caches off; note 14 of
  `docs/loopy-notes.md` has the table.
- An `ilp` or `ilp.seq` loop in which a ragged row's length is read is a
  `refuted` `buildable` fact. loopy generated its code, and the code was wrong:
  it gives a temporary written inside an `ilp` loop an array along the loop,
  and the loop over the row's fiber still read the length by its name, so it
  compared its variable with the array's address. The C run read past the rows
  and crashed. It had passed the check, since nothing concurrent sits in the
  fiber's domain; `unr` builds and runs.
- A ragged fiber inside a dense loop of its row lowers beside a statement of
  that loop (#53). The statement in the fiber was cut after its row, where the
  row's length is assigned, and the one beside the fiber was not, so the row
  loop came out in two domains and the kernel was refused for using `r` for
  two loops, which it does not. Every cut is now passed on to the statements
  that share the loops up to it, and the kernel lowers and agrees with its
  body. A loop the lowering still cannot define once is refused as a limit of
  the lowering, not blamed on its name, when every statement that has it has
  the same loops around it.
- The agreement fact of a kernel or a term run by `LoopyExecutor.differential`
  records the target the run was made on (#56). It read the target off the
  object, which a kernel does not carry, so a run on OpenCL was recorded, and
  named, as `[c]`. `agreement` takes the target as a keyword, defaulting to the
  schedule's.
- A loop tagged `vec` carries no order, as one tagged `ilp` does not (#57).
  loopy runs such a loop around each instruction of its body separately, so
  two statements of the loop no longer interleave, and `vec` kept its place in
  the order the checker asks about: `tag(i="vec")` on a loop whose second
  statement feeds the first statement of the next iteration was a decided
  cast, and the compiled run disagreed with the body. It is refused with the
  witness `ilp` gets, and a `vec` tag on a reduction's loop asks the
  accumulation's permission to be reassociated, as `ilp` does.
- The assumption loopy is given for the sorts of the scalar parameters says
  the sizes are non-negative too. An assumption makes every parameter of the
  kernel part of the domain loopy checks an access over, and loopy checks an
  access only when that domain names everything the array's shape does, so a
  `Nat` scalar beside `off[0] = ...` of `off: Arr[Fin[n + 1], Nat]` outside
  any loop had loopy refuse `off[0]` for `n = -1` with a `LoopyIndexError`.
- The contract refuses an argument too short for its type at any size: an
  `off` of no cells for `Arr[Fin[n + 1], Nat]` stands for `n = -1`, which
  every fact about the kernel excludes, and which loopy, reading `n` off the
  shape, used as the index of `off[n]` (`contract.sizes_not_negative`).
- A program's restatement of a callee's postcondition rests on that callee's
  fact, and never on a kernel of the program's own file that shares its name
  (#42). Kernel fact ids were keyed by the qualified name alone
  (`scan:postcondition`), so in a file that defines a `scan` and whose
  program calls `from helpers_scan import scan as helper_scan`, the
  restatement of the helper's postcondition rested on an id this file's
  ledger resolved to the local `scan`, a different statement, and no
  `UNRESOLVED` line said that the real fact is in the helper's ledger. A
  program that called both kernels got one restatement where it owes two,
  the second replacing the first. Every fact id of a kernel and a program is
  now keyed by definition (see Changed), so the restatement names the
  helper's fact by the id it has in the helper's own ledger, and each callee
  gets a restatement of its own.
- The term interpreter runs a kernel whose loop bound reads an array the
  kernel writes (#52). It enumerated every instance of every statement before
  it ran any, so it refused such a kernel ("which instances run depends on
  when the bound is read"), and the `trace-faithful` fact of, say, a kernel
  that clears the next row's count after summing a row stayed `assumed`
  while its differential run was `tested`. It now walks the loop tree as the
  body runs it, and enumerates a loop when it reaches it: the bound of a loop
  over `val.dom[r]` is read where that loop starts, each time it starts, as
  the native `for` reads it, and a reduction's where it is summed, while
  the bounds of the loops around a reduction, which its domain repeats, keep
  the values their loops were read at. So such a kernel's fact is `tested` or
  `refuted` like any other. A bound is one parameter of a domain, read once,
  so one that bounds two nested loops of one statement (`for k in
  val.dom[r]` inside `for j in val.dom[r]`), or a loop and a sum inside it
  (`reduce_sum(val[r, k] for k in val.dom[r])` inside `for j in
  val.dom[r]`), and reads an array the kernel writes is refused, and the fact
  stays `assumed`, since the body reads it where each loop or sum starts. A
  domain no array bounds is still checked against the work limit before
  anything runs, by its bounding box.
- The facts of a schedule are keyed by the kernel's definition, as the
  kernel's own facts are (#75). Their ids started with `Schedule.key`, which
  names the kernel by its name, so in a file that defines a `double` and
  imports another as `helper_double`, `Schedule(double).split("i", 2)` and
  `Schedule(helper_double).split("i", 2)` shared their ids: `loopty run`
  kept the helper's cast facts in place of the local kernel's, which could
  hide a refuted fact of one behind a decided fact of the other, and the
  helper's agreement survived only through the `#2` a schedule run twice
  gets. The ids now go through `lanky.ledger.fact_id`, with the kernel's
  definition and then the target and the steps
  (`Schedule.fact_id`): `cast:spmv.spmv@102:[c].split('j', 2, inner='j_in',
  outer='j_out'):bijective`, and `agreement:spmv.spmv@102:[c]` for a kernel
  run without a schedule. `Schedule.key` stays the readable call text, and a
  term scheduled with no kernel behind it is named by its name,
  `cast:transpose:[c]...`, and a schedule of a schedule by the kernel behind
  it. `loopty.schedule.definition_of` gives what a fact names an object by.
- A map per statement in `Schedule.affine` can name a program's statement
  (#79). A program's statements are named after their calls, `flux.S0` and
  `step@2.S0`, which isl cannot read as tuple names, so `{ flux.S0[j] -> [jj]
  : jj = j }` was a syntax error, and `flux_S0`, loopy's id of the
  instruction, was refused as not a statement of the program. A tuple name
  that is no statement's id is now read as the id spelled with every
  character other than a letter, a digit or an underscore written `_`, which
  is the instruction's id and so names one statement, and isl's refusal of a
  map that names a statement by such an id says how to spell it.
- An index read from an array the kernel writes is no longer in bounds by
  its element sort alone (#64). `x[perm[j]]` with `perm: Arr[Fin[n], Fin[n]]`
  was decided by type, which rests on the contract's check of every cell of
  `perm` when the kernel is called, and nothing checked what the kernel then
  wrote into `perm`: after `perm[i] = i + 1` the native run raised
  `IndexError` at `x[4]`, the compiled one read past the end of `x`, and the
  ledger said decided. Every write into an array of a `Fin[m]` element sort
  now owes a fact of kind `element-sort`, that the value written is a point
  of `Fin[m]` (`loopty.typing.element_sort_facts`): isl decides it for a
  quasi-affine value and refutes it with the instance that writes outside
  (`[i=0] ... at [n=1]` here), the type decides it for a value read from an
  array of the same sort, and it is assumed otherwise. A fact decided by type
  through an array the kernel writes rests on the element-sort facts of the
  writes into it, so the ledger shows `x[perm[j]]` decided under the write's
  fact and worth what that fact is worth, refuted here, and `lanky check`
  exits 1. A kernel that only reads its index arrays, as every demo does,
  has no such fact.
- A kernel that rewrites the layout of a ragged array no longer has that
  array's in-bounds and disjoint-writes facts decided outright (#51). They
  are decided against the length of a row and over `[r, j]`, which holds of
  the flat buffer while every row lies inside it and apart from the others,
  as the contract checks when the call starts; `off[r] = s[r]` can move a row
  past the end of the buffer, and `off[r] = 0` every row onto the same cells,
  where the ledger decided "distinct instances of S1 write distinct cells of
  val" and a parallel `r` was accepted. Such a kernel now has one `layout`
  fact per counts family whose counts or declared offsets it writes,
  `assumed` with the reason (`loopty.typing.layout_facts`), and the in-bounds
  and disjoint-writes facts of the family's ragged arrays rest on it, as does
  a fact decided by type through an index read from one of them
  (`x[col[r, j]]`, whose index is read from a cell of a moved row), and the
  `monotone` casts of a schedule of the kernel or of a program whose call
  rewrites it: the ledger shows them decided under the layout, and worth an
  assumption. `lanky check` lists the layout fact among the kernel's facts,
  and `loopty run` beside the casts that rest on it. A kernel that only reads
  its layout, as every demo does, has no such fact.
- A real literal stored into an integer array keeps its fraction until the
  store (#73). The lowering handed loopy a Python `float` as it was, and loopy
  writes an untyped constant in the type of the expression around it, which
  on the right-hand side is the assignee's: `c[i] = u[i] * 0.5` into a `Nat`
  `c` was generated as `c[i] = (int32_t) (u[i] * 0)`, so `[1, 2, 3, 4]` gave
  `[0, 0, 0, 0]` compiled and `[0, 1, 1, 2]` natively. loopy also took `0.5`
  for a `float32`, so half of an integer was computed in single precision.
  A `float` is lowered as `np.float64` and a `complex` as `np.complex128`,
  which loopy writes as they are; a complex literal single precision does
  not hold, which loopy refused to type, lowers too. Note 17 in
  `docs/loopy-notes.md`.
- An array argument stored in a dtype that does not hold its element sort no
  longer computes one thing natively and another compiled (#77). The compiled
  run converts every array into the dtype its sort is lowered as, and the
  native run computed in the dtype it was given: an integer `x` for a `Real`
  parameter that the kernel halves and doubles came back `[2, 4]` natively and
  `[3., 5.]` compiled, and a complex `x` lost its imaginary part compiled
  only. An array the kernel writes has to be stored as its sort is natively
  (`contract.written_storage`, the rule `contract.native_storage` states for a
  program's temporary), and is refused on every entry point otherwise, naming
  the dtype to pass: an integer or `float32` one for `Real`, a float one for
  `Nat`, a byte for `Bool`. An array the kernel only reads is read by the
  native run, and by the interpreter, through a copy in that dtype
  (`contract.read_storage`, which generalizes the integer copy of a
  float-stored index array), so an integer `x` read as reals no longer
  overflows natively at `x * x`. The contract refuses a complex entry with an
  imaginary part for any sort that is not complex, as it did for an integral
  one, and an entry of `Bool` stored as a number that is not `0` or `1`, which
  the compiled byte would hold as it is. The executor drops a zero imaginary
  part before the cast, which no longer warns. A scalar argument is passed by
  value, so the native run and the interpreter convert it into the dtype of
  its sort whatever the kernel does (`contract.native_scalar`):
  `np.int64(2**32)` for a `Real` no longer overflows at `a * a`, `np.int8(100)`
  for a `Nat` no longer wraps at `a + a`, and a `Bool` is a numpy bool, so
  `~flag` of Python's `True` is `False` there as it is compiled, where it was
  `-2`. The contract asks a scalar what it asks an entry: a `Bool` given as a
  number is `0` or `1` (`2` was stored as the byte `2` compiled), and a
  complex one of a sort that is not complex has no imaginary part. The
  executor passes a truth value to compiled code as an `int`, since `ctypes`
  refused a numpy bool for the byte `Bool` is lowered as.
- A store into an array of `Bool` of what is not a truth value is a
  `TraceError` naming the fix (#78). Natively the array is a numpy bool, which
  stores `0.5` as `True`, and compiled a byte, into which C converts `0.5` as
  `0` and `2.0` as `2`. A comparison, a connective whose operands are truth
  values, a read of an array or a scalar of truth values, and `True` and
  `False` are stored; `b[i] = u[i]` is refused with `b[i] = u[i] != 0` as the
  fix, an integer constant with the truth value numpy makes of it, and
  `k[i] & 1` of an integer `k`, which the trace reads as `and` and numpy
  computes bitwise, with its operand named. Natively an integer stored into a
  bool array is refused as `when` refuses an integer guard, since that is
  what `~(i > 0)` of a loop variable is there, `-2` or `-1`, which a bool
  array stored as `True` at every point.
- Each operation is computed in the type numpy computes it in natively
  (#82, #91). numpy types an operation by NEP 50, where a Python number takes
  the dtype of what stands beside it, and C by its usual arithmetic
  conversions, and the two disagreed: `k[i] / 2 * 2` of an integer `k` was
  C's integer division compiled, `2` at `k = 3` where the native run stores
  `3`; `x[i] * 0.1 + 0.3` of a `float32` `x` was computed in double compiled,
  a bit away from numpy's single precision, which the `approx` class hid; and
  `x[i] / k[i]` of a `float32` `x` and an integer `k` was single precision
  compiled and double natively. `loopty.promotion` reads both types off the
  term, numpy's by doing each operation in numpy on a sample of its
  operands' types, and the lowering converts an operand where they differ: a
  literal is written in numpy's dtype (`0.10000000149011612f`, and
  `complex64` parts beside a `float32`), anything else is cast
  (`(double) (k[i]) / 2`). Nothing changes for an operation the two type
  alike. Note 19 in `docs/loopy-notes.md`.
- A power compiles on the C target, and is computed with `pow`, as numpy
  computes it (#84). Any power but `x ** 0`, `1` and `2` failed: an integer
  exponent calls a power loopy defines in a preamble whose signature names
  `int32_t` before loopy includes `stdint.h`, and a floating one calls `pow`,
  for which loopy includes no `math.h` (note 2). A term with such a power
  gets both headers in a preamble that sorts first, and `complex.h` too when
  it has complex values, since the power of a complex base names
  `double complex` there. A floating power is given a floating exponent,
  since loopy's integer power multiplies repeatedly and rounds at every step,
  and `x ** 3` differed from numpy's in the last bit at about one `x` in four.
- A connective of an operand that is not a truth value is a `TraceError`
  wherever it is (#83), as it was in a store into `Bool` only (#78): the
  trace reads `&`, `|` and `~` as `and`, `or` and `not`, and natively they
  are bitwise on an integer, so `(k[i] & 1) * x[i]` was `0` natively at
  `k[i] = 2` and `x[i]` compiled. A stored value, a guard, a sum's body, an
  index, the index of the cell written and a comparison are all asked, and
  the fix is named: `k[i] != 0`, or `k % 2` for `k & 1`. `~` of a comparison
  of Python numbers, `~(i > 0)` of a loop variable, is bitwise natively too,
  `-2` or `-1`, which the native run refuses in a guard and a `Bool` cell
  (#25, #78); used as a number anywhere else, as in `x[i] * ~(i > 0)`, it is
  now a `TraceError` naming the complement, `i <= 0`.
- An entry or a scalar of `Nat`, `Int` or `Fin[m]` outside the 32-bit range
  the compiled run stores it in is refused on every entry point (#92), naming
  the range (`contract.INTEGRAL_RANGE`) and the fix: a value inside it, or a
  numpy integer sort, which both runs store as it is. `2**32 + 5` ran
  natively as it was and was `5` compiled, and a `uint64` entry from `2**63`
  on was read natively through an `int64` copy as a negative number. The
  dtype a sort is compiled in is `contract.compiled_storage`, which
  `lower.numpy_dtype` now reads. A result outside 32 bits is a stated limit.
- The `monotone` cast refuses a dependence between instances on two work
  items of a hardware axis (#63). It dropped a loop on `g.*` or `l.*` from the
  order it checked, as it drops `ilp` and `vec`, and let the loops around it
  order the rest; loopy runs the axis as the launch grid, outside every loop,
  and nothing in a kernel orders two work items through global memory. So
  `jacobi` in `examples/stencil_skew.py` with `i` on `g.0` or `l.0` was a
  `decided` cast and a buildable schedule, and loopy generated it with no
  barrier, each work item reading what its neighbour wrote a step before; the
  acoustic pair of `examples/wavefront_acoustic.py`, and two loops on one axis
  where the second reads the first's cells in another order, were the same,
  and loopy refused them with `MissingBarrierError`. Each is refused now, with
  the dependence and the two work items: `tag(i='g.0') illegal: instance
  S0[t=0, i=1] writes u[1, 1] read by S0[t=1, i=2] on another work item ...:
  the loop i on g.0 runs them on work items 0 and 1 of it`. An instance's work
  item is the value of its loop on the axis counted from where loopy starts
  that loop (`get_hw_axis_base_for_codegen`), so two loops on one axis that
  start at different values are compared as loopy runs them. A statement with
  no loop on an axis that another statement's loop is on runs on every work
  item of it, so a dependence to or from it is refused too: the scan of
  `off` before the rows that read it, with the rows on `l.0`, used to be a
  legal cast. A sum on a local axis is not a loop of its statement, and stays
  allowed, since loopy synchronizes its partial sums. Its body reads on every
  work item of the axis, though, and its statement stores the result from
  one, so a sum whose body reads what another sum's statement stored, at the
  same step or a later one, or the cell its own statement writes, is refused
  too, where loopy asks for a global barrier and `buildable` used to pass;
  two statements may still pass a sum's result between them outside their
  sums. `ilp` and `vec` lose only their order, as before; spmv's rows and
  p2p's targets on `g.0` are unchanged. The check
  is asked after every step, so a skew that moves a tagged loop's dependences
  onto two work items is refused as well. A monotone fact about a schedule
  with a loop on a hardware axis says "within one work item", and the
  transcripts are regenerated.
- `Schedule.split` and `Schedule.tile` refuse a loop that carries a tag, with
  a `ValueError` naming the fix, before loopy is asked (#93): `split(r, 2): r
  carries the tag 'g.0'; split it before tagging the loops it makes`. loopy
  refuses to split a loop with any tag but `for`, so with a kernel the
  refusal was loopy's `LoopyError`, which named no fix, and it splits a loop
  tagged `for` into two with no tag. Without a kernel (after an `affine` step
  the kernel rewrite could not write) nothing refused. Either way the tag
  stayed in `Schedule.tags` on a loop the schedule no longer has, and the
  checker read the loops that replaced it as untagged: sequential in the
  order, and on no hardware axis, so a monotone fact no longer said "within
  one work item". Every tag is refused, `for` included, and a sum's loop as
  well, as `affine` refuses a map over a tagged loop: a tag goes on the loops
  a step makes, after it.
- `loopty run` refuses two kernels' claims of one fact id, as `lanky check`
  does since lanky's #52 (#95). Two kernels one definition makes, as a
  factory does each time it is called, share every id, and the run built its
  ledger with `Ledger.add`, which replaces a fact of the same id: a file
  scheduling both with the same steps kept one set of facts for the two,
  the second kernel's in place of the first's, a refuted one as readily as a
  decided one, and exited 0. The first kernel's fact is kept now, the other
  kernel's claim is recorded on it, by its statement, as `duplicate_claims`
  in its provenance (the key `lanky check` records it under, so `--json`
  carries it), and a `DUPLICATE` block under the table names the kernel,
  each id, and the claims of it in the table and not, and says how to give
  each kernel an id of its own; the command exits 1. The run has decided
  every claim by then, so one not in the table that was refuted, such as
  the other kernel's run, is said to be, with what explains it under it,
  since no row and no `REFUTED` block shows it. The fix named is a
  definition or a `__qualname__` of the kernel's own, or, for a term
  scheduled with no kernel behind it, which is named by its name, a name of
  its own. One kernel scheduled several times is not refused, nor a
  schedule of a schedule of it: its schedules that share their first steps
  share the facts about them, and a schedule run twice keeps both agreement
  facts, the second with `#2` after its id.
- A guard that reads only scalars is kept in the compiled kernel (#90).
  `when(flag)` of a `Bool` scalar, or `when(a > 0.5)` of a `Real` one, around
  every statement of a kernel was lowered as an instruction predicate naming
  no loop variable, which loopy hoists out of the device function into host
  code around its call, and `lp.ExecutableCTarget` generates that host code
  and never runs it. So the compiled run wrote every cell whatever the scalar
  said, where the native run wrote none: the differential test refuted such
  a kernel and the `trace-faithful` fact did not. A guard on one cell of an
  array (`when(x[0] > 0.5)`) was dropped the same way. Target `c` is now
  `loopty.lower.InProcessCTarget`, whose host code cannot hold a condition, so
  the guard is emitted around the loop in the function that runs. On the
  PyOpenCL target the hoisted guard wrapped the launch and the event the host
  code returns, so a false guard raised `UnboundLocalError`; target `opencl`
  is now `loopty.lower.InKernelOpenCLTarget`, which keeps the guard in the
  kernel the same way. See note 18 in `docs/loopy-notes.md`.
- A whole array stored into a cell is refused while tracing (#85).
  `y[i] = u` traced, with the symbolic array itself as the statement's
  right-hand side, where natively numpy refuses to store a sequence in a
  cell. It is now a `TraceError` naming the loop nest to write, as `u` used
  whole on the right of an operator is, and a domain (`y[i] = u.dom`), a
  list, tuple, set or dict, a numpy array with an axis, an array the body
  holds, and a generator (with `reduce_sum(...)` named as the fix) are refused
  the same way. Natively numpy refuses each of them for a number, and stores
  into a `Bool` cell the truth value of the whole (`[False]` is `True`),
  which the message says.
- Two statements whose ids spell one instruction id are refused by the
  lowering, naming both (#88). A statement's instruction is named by its id
  with every character other than a letter, a digit or an underscore written
  `_`, so a hand-built term with statements `a.S0` and `a@S0` failed inside
  `lp.make_kernel` with loopy's "duplicate instruction id: 'a_S0'", which
  names neither. A statement whose id spells the instruction that computes a
  row length (`nl_cnt_r_init`) is refused the same way. A traced kernel's ids
  and a program's call labels cannot collide.
- The term interpreter reads a bound that bounds two loops of one
  statement, or a loop and a sum inside it, where each of them starts
  (#87). A bound is one parameter of a domain, and the interpreter read it
  once, so it refused such a bound when it reads an array the kernel writes
  (`for k in val.dom[r]` inside `for j in val.dom[r]`, and `reduce_sum(val[r,
  k] for k in val.dom[r])` inside the loop over `j`, each followed by
  `cnt[r] = 1`), and the `trace-faithful` fact stayed `assumed`. Such a
  bound now gets a parameter of its own for each loop inside the outermost
  one it bounds, read where that loop starts, and one for the statement's
  sums, read where each sum starts, while the constraints a sum repeats
  from the loops around it keep the readings of those loops
  (`interpret._Run.read_apart`). So the fact of such a kernel is `tested` or
  `refuted` like any other; the lowering still refuses a statement that
  would see a length rewritten after it was computed.
- The `layout` fact of a kernel that writes its offsets is decided where
  the writes say what they do (#86). It was `assumed` whatever the kernel
  wrote. For a family whose rows are as long as a counts array the kernel
  reads and does not write, every write to the offsets is now either one
  that lays the row out as the counts do, `off[q] = off[q - 1] + cnt[q -
  1]` at `1 <= q <= n` (the value the contract checked there, so by
  induction the offsets stay as they were), or a value of the loop variables
  and the sizes alone, which is the start the counts give its row only for
  `off[0] = 0`. The fact is then the isl question whether any instance
  writes another such start: decided when none does, so a kernel that
  rescans its offsets has its facts worth what they are, and refuted with
  the instance otherwise, `[r=1]` at `n = 2` for the issue's kernel that
  sets every start to 0, where counts of 1 in rows 0 and 1 put both rows on
  cell 0. `lanky check` exits 1 on it, and `loopty run` asks isl about the
  layout fact a cast rests on as `lanky check` does. A start read from
  another array, a count written, a write under a guard isl cannot state, or
  counts with more than one axis leave the fact `assumed`, and its reason
  names the write.
- A name only the bound of a `Fin` sort mentions is a size, and not
  negative, as an axis extent is (`flow.size_names`). `nnz` in
  `off: Arr[Fin[n + 1], Fin[nnz + 1]]` is the length of the buffer the
  offsets point into, which no axis of a scan has to be, and the
  `element-sort` fact of `off[0] = 0` was refuted at `nnz = -1`.
- In a program, such a name is renamed apart as a callee's other sizes are
  (`loopty.compose`). It kept its own spelling, so two calls of a scan that
  each name their buffer `nnz` gave two unrelated buffers one size, and the
  compiled program refused a second matrix of another length with a shape
  mismatch, where the native one ran.
- Hypotheses that contradict each other decide nothing
  (`loopty.hypotheses.discharge`). A postcondition no run can satisfy, or a
  postcondition and a requirement that hold together nowhere, left no point
  of the claim's domain, and every requirement and every flat access under
  them read `decided`, vacuously; with a false postcondition the compiled
  program then skipped the check the native callee refuses on. Such a
  requirement is now checked when the program runs, and its reason names the
  hypotheses. A reason also names what isl could not state of a hypothesis
  or of the claim, which was read as saying nothing, and the values only of
  the cells the claim reads, not of every cell an instance reached.
- A call that writes an array a postcondition names only in a binder's sort
  retires the postcondition (`loopty.hypotheses.mentioned`).
  `all(perm[i] == 0 for i in Fin[lim[0]])` stayed a hypothesis after a later
  call made `lim[0]` larger, and then said that cells nobody cleared were 0:
  a requirement on `perm` read `decided` under two true postconditions, and
  the compiled program read past `x` where the native one was refused.
- A name a postcondition leaves free that is neither a parameter nor a size
  of its kernel means nothing in a program (`loopty.compose`): natively it
  has no value, and the claim is never evaluated. It kept its spelling, and
  was read as whatever the program called so, a size another kernel named
  `k` say, or the binder `q` of a layout requirement; one row's equation was
  then every row's. A hypothesis is also never read as speaking of the
  claim's own binders (`loopty.hypotheses.discharge`): a name of one it
  leaves free is renamed apart.
- A theorem a program cites has a family's codomain checked in the
  program's names, as its domain is (`loopty.hypotheses.theorem_instances`):
  `f: Fn[Fin[n], Fin[m]]` was compared with an array's element sort as
  `Fin(m)`, the theorem's own `m`, whatever its hypotheses had bound `m` to.
- A requirement decided on the strength of a fact that is not at least
  `tested` keeps its checked point (#115): a postcondition its kernel's native
  runs refute or that nothing tested, a cited theorem the property tester
  does not pass, or an axiom. The compiled program skipped the check, so a
  postcondition that says `perm[i] == n - 1 - i` of a kernel that writes
  `i + 1` had it read `x[n]`, where the native `gather` is refused. The fact
  stays `decided`, worth what it rests on, and its statement and provenance
  say that the program checks it when it runs.
- A postcondition used as a hypothesis rests on its callee's `trace-faithful`
  fact as well as on the postcondition (`loopty.compose`). It is tested on
  the callee's body, and the compiled program runs the callee's term: a
  kernel whose term the trace made another kernel (an `isinstance` taken the
  other way) kept a postcondition true of its body, and the compiled program
  read `x[n]` where the native one ran. The requirement is worth no more than
  that fact, and is checked when it is not at least `tested`.
- A cited theorem is instantiated without capture
  (`loopty.hypotheses.theorem_instances`). A binder of its goal spelled like
  the program size a variable of it matched took that size's place:
  `all(f(a) <= n for a in Fin[n])` at `n = a` said `perm[a] <= a` of the
  binder, which is false of cells the theorem's hypotheses describe, and
  decided a requirement under a true theorem and a true postcondition; the
  compiled program read `x[n]`. The binder is renamed apart.
- A callee's postcondition is no hypothesis after a call whose contract
  nothing checks in the program (`loopty.compose`). It is tested on the runs
  its contract lets in, and a `Nat` element sort of an array an earlier call
  wrote is checked natively by the callee's contract and by nothing in the
  compiled program: an earlier call leaving `-4` in it made a postcondition
  false that the compiled program then skipped a check on, and read `x[-4]`.
  The requirement after it is checked, and its reason says why the
  postcondition was not used.

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
- `lanky>=0.1.0.dev1` is a dependency, resolved from a sibling checkout by
  `[tool.uv.sources]` during development (#37). Every lanky commit used to be
  `0.1.0.dev0`, the placeholder that reserved the name included, so the floor
  admitted a lanky with none of what loopty imports. lanky's version now moves
  to the next `0.1.0.devN` whenever an interface loopty uses changes, and this
  floor is raised with it; 0.1.0.dev1 is the lanky whose ids are keyed by the
  path of the defining file (`lanky.check.module_name`), which loopty's ids
  now use. In between, loopty follows lanky's `main`, which the README's
  install section says.
- **Every fact of a kernel or a program is keyed by its definition** (#42).
  The ids go through `lanky.ledger.fact_id(kind, owner, module, line)`, as a
  theorem's do: the kind, then the module the file's path gives it under its
  source root (`lanky.check.module_name`, not the name it was imported
  under), the qualified name and the line, then what the fact is about.
  In `examples/spmv.py`, `scan:postcondition` is now
  `postcondition:spmv.scan@69` and `spmv:in-bounds:S0:write:y[r]` is
  `in-bounds:spmv.spmv@102:S0:write:y[r]`; a kernel that cannot be traced
  is `trace:<module>.<name>@<line>` rather than `kernel:<name>:traced`. A
  program's restatement is
  `postcondition-in-scope:spmv.solve@115:spmv.scan@69`, naming the callee by
  its definition too. The one id is the same in the kernel's own file's
  ledger and in the `rests_on` of a program of another file that calls it.
  `Kernel` and `Program` carry `module` and `definition`, the typing rules
  and `faithfulness_fact` take `module=` and `line=`, and
  `loopty.typing.postcondition_id(owner, module=, line=)` builds the
  postcondition's id for both sides. Every id `lanky check` prints and
  writes with `--json` for a kernel file changes this way, and the `lanky
  check` transcripts show the new ids. A schedule's cast and agreement facts
  are keyed by definition too, since #75 (see Fixed).
- **A program's restatement of a callee's postcondition rests on the callee's
  fact.** `Program.facts` pointed at the callee's postcondition with a `from`
  entry in the provenance, which lanky had no way to read, so the ledger
  showed each restatement as an assumption standing on its own. It now sets
  lanky's `Fact.rests_on` to the id of that fact, built by the new
  `loopty.typing.postcondition_id`, which the kernel's own postcondition fact
  uses too, so the two cannot drift apart. The ledger names the callee's fact
  beside the restatement, as in `assumed under postcondition:spmv.scan@69`
  (an id keyed by definition, above), counts it in what the restatement is
  worth, and `lanky check --json` carries `rests_on`, `effective` and
  `under`. The `from` entry is gone; `callee`
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
  `val[off[r] + j]`, is therefore reported `assumed` in its kernel's ledger,
  with the reason in its provenance; the ragged spelling `val[r, j]` is
  decided, and so is the flat one in a program that calls the kernel after
  the scan that wrote `off`. See the module docstring of `loopty/flow.py`.
- The test suite treats `DeprecationWarning` as an error. Two exemptions are
  loopy's own and are listed in `pyproject.toml` and `tests/conftest.py`, with
  the reasons in `docs/loopy-notes.md`.
