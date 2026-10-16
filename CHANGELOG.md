# Changelog

All notable changes to loopty are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

A schedule names what it moves by where the file defines it. Every step
takes handles, paths from the kernel or program it schedules, and no step
takes a string any more: `l2p.R` is the table the program `l2p` makes,
`l2p.unpack.box` the loop over boxes of the kernel `unpack` where `l2p`
calls it, `reconstruct.b` a loop of `reconstruct` itself, and
`mass_action.nc` one of its sizes. A handle is checked when it is written,
against the traced kernel, and each step resolves it through what the steps
before it did, so a loop a split replaced, a fusion merged or a substitution
took out is refused with that step. Everything a schedule prints names loops
and arrays by these paths: its repr, its steps in the facts' statements and
the ledger, the refusals and the fixes they suggest, and `loopty run`.

A kernel body branches with Python's own `if`, `elif` and `else`, `and`,
`or`, `not` and conditional expressions, and a loop nest is one line,
`for box, a, b in R.dom:`. The trace runs the body once per path through its
branches, answering each traced condition from the path, and joins the runs
into one term; `with when(...)` is gone. The flagship kernel reads as a
numpy user writes it:

```python
for box, a, b in R.dom:
    if a < 2:
        R[box, a, b] = C[box, a, b]
    else:
        R[box, a, b] = -R[box, a - 2, b + 2]
```

### Breaking

- **Every step takes handles, and a string is a `TypeError`** that writes
  the handle for it, computed from the schedule: `contract('R')` is refused
  with "contract() takes an array handle such as l2p.R, not the string
  'R'". 0.3's other spellings are refused the same way, with the step to
  write: `tag(i="g.0")` and loopy's `tag("i:g.0")` with `tag(k.i, 'g.0')`,
  `fuse("unpack", "reconstruct", loops=1)` with `fuse(l2p.unpack.box,
  l2p.reconstruct.box)`, and `split(k.j, 32, inner="j_in")` with the
  halves' handles. The step's text is part of a fact's id, so the ids of every
  schedule's facts change; the kernels' and programs' own facts keep
  theirs. The migration, step by step, with `k` a kernel and `p` a program:

  | 0.3 | 0.4 |
  |---|---|
  | `tag(i="g.0")` | `tag(k.i, "g.0")`, or `on_groups(k.i)` |
  | `tag(i="l.0")` | `tag(k.i, "l.0")`, or `on_items(k.i)` |
  | `tag(i_outer="g.0", i_inner="l.0")` | `tag({k.i.outer: "g.0", k.i.inner: "l.0"})`, or `on_groups(k.i.outer).on_items(k.i.inner)` |
  | `tag(q_inner="ilp")` | `tag(k.q.inner, "ilp")` |
  | `split("i", 4)` | `split(k.i, 4)`; its loops are `k.i.outer` and `k.i.inner` |
  | `split("j", 32, inner="j_in", outer="j_out")` | `split(k.j, 32)`: the halves are named by the handle, `inner=` and `outer=` are gone, and loopy's names for them are `j_inner` and `j_outer` |
  | `interchange("b", "a")` | `interchange(k.b, k.a)` |
  | `interchange("i_out", "j", "i_in")` | `interchange(k.i.outer, k.j, k.i.inner)` |
  | `prioritize("s_outer", "t_inner")` | `prioritize(k.s.outer, k.t.inner)` |
  | `tile("t", "i", 8, 8)` | `tile(k.t, k.i, 8, 8)`; the tiles are `k.t.outer`, `k.i.outer`, `k.t.inner` and `k.i.inner` |
  | `skew("i", by="t")` | `skew(k.i, by=k.t)` |
  | `affine("{ [t, i] -> [a, b] : ... }")` | unchanged; the loops the map makes, which no line of the file binds, are `k.loops.a` and `k.loops.b` |
  | `fuse("flux", "divergence", shift=1)` | `fuse(p.flux.j, p.divergence.i, shift=1)`: a loop of each call, not the call |
  | `fuse("unpack", "reconstruct", loops=1).fuse("reconstruct", "evaluate", loops=1)` | `fuse(p.unpack.box, p.reconstruct.box, p.evaluate.box)`: one step for any number of calls |
  | `fuse("along_r", "along_s", loops=2)` | `fuse(k.along_r.a, k.along_s.a)`: the loops are fused from the outermost down to the one named, at the same depth in each call; `loops=` is gone |
  | `fuse("reverse_quiet", "gather.check.idx")`, a program's checked point by its statement's id | `fuse(p.reverse_quiet.i, p.gather.checks.idx)`: the check's loop is the call's, by the parameter it checks |
  | `substitute("f")` | `substitute(p.f)` |
  | `contract("R")`, `contract("v", lambda e, a, b: b)` | `contract(p.R)`, `contract(k.v, lambda e, a, b: b)` |
  | `realize("y", tree=True)` | `realize(k.y, tree=True)` |
  | `pack("f")` | `pack(k.f)` |
  | `atomic("y")` | `atomic(k.y)` |
  | `assume("nc % 4 == 0")`, `assume("nc >= 1")` | `assume(k.nc % 4 == 0)`, `assume(k.nc >= 1)`: a condition of the sizes' handles, several joined with `&` |
  | `retarget("opencl")`, `example(...)` | unchanged |

- **A kernel parameter or loop with the name of an attribute of `Kernel` or
  `Program`** (`term`, `facts`, `sizes`, ...) is reached by its long form,
  `k.arrays.term` or `k.loops.sizes`, and its definition warns with
  `ShadowedName`, naming the long form. `examples/p2p.py`'s `term` is
  `contribution` for this reason.
- **`with when(condition):` is removed.** Using it raises a `TraceError`,
  natively and traced, under any name it is imported as, that shows the
  `if` to write in its place, read off the source, and an `else` where the
  next block is the complementary `when`. The native run takes the branch
  each point takes, where `when` ran the block and masked its writes, so a
  read the condition protects is not made where it fails: the masked
  branch read `R[box, a - 2, b + 2]` at `a < 2`, a negative index numpy
  wraps. `loopy` is unchanged: a statement still carries its path's
  conditions as predicates.

  | 0.3 | 0.4 |
  |---|---|
  | `with when(a < 2):` ... `with when(a >= 2):` ... | `if a < 2:` ... `else:` ... |
  | `with when((i > 0) & (i + 1 < n)):` | `if i > 0 and i + 1 < n:`, or `if (i > 0) & (i + 1 < n):` |
  | `with when(~(i > 0)):` | `if not (i > 0):`, or `if i <= 0:`; `if ~(i > 0):` is refused, since `~` of a Python bool is `-2` or `-1`, which Python's `if` takes as true |
  | `with when(x[i] > 0): f[i] = a[i]` and `with when(x[i] <= 0): f[i] = b[i]` | `f[i] = a[i] if x[i] > 0 else b[i]`, one statement with a select |
  | `for i in R.dom: for a in R.dom[i]: for b in R.dom[i, a]:` | unchanged, or `for i, a, b in R.dom:` |

- **A condition on a loop variable whose loop has closed** (`if i > 0:`
  after `for i in x.dom:`) is refused as a write that mentions it is, and
  **`~` of a Python bool in a condition** is refused by the trace, naming the
  comparison and `not`: natively it is an integer Python's `if` takes as
  true, which nothing else refuses.

### Added

- **Branches** (`loopty.trace`). A traced condition's truth value is
  answered by the tracer (`Tracer.decide`), which runs the body again until
  every condition it asked has been answered both ways: one run per path.
  Each statement carries the conditions of its path as its guard, narrowing
  its isl domain where isl states them and evaluated at run time otherwise,
  as `when`'s guard was, and each condition's reads are stated where the
  conditions before it held (`Stmt.path`): `if i > 0 and x[i - 1] > 0:`
  reads `x[i - 1]` only where `i > 0`. The runs are joined: a statement
  after an `if`/`else` is recorded once, its guards `g and c` and `g and not
  c` merged to `g`; a statement whose value two paths choose between under
  a condition on data is one statement whose value is a select
  (`loopty.trace.Select`, pymbolic's `If`, `?:` in C), the value select the
  advection flux lacked (#266); a path isl shows to be empty (`a < 2` and
  then `a == 3`) records nothing. One run answers a condition once, as
  `x and y` asks `x` twice when it is false. The runs are capped at
  `loopty.trace.MAX_RUNS`, 256, refused past it with the conditions and what
  to write instead; a `while` on a traced condition is refused, since its
  paths have no bound; and a branch that raises is refused with its path,
  `on the branch where a >= 2`. A condition the trace can evaluate, a module
  constant, is Python's own.
- **One-line loop nests.** `for box, a, b in R.dom:` is the three loops `for
  box in R.dom`, `for a in R.dom[box]` and `for b in R.dom[box, a]`, natively
  and traced, with those binder names and their scoped handles
  (`reconstruct.box`, `reconstruct.a`, `reconstruct.b`).
- **A loop whose statements' domains do not merge into one set**, the two
  points of `if i == 0 or i == n - 1:`, runs over their convex hull,
  each statement predicated back to its own points.

- **Handles** (`loopty.handles`). A kernel's and a program's attributes are
  its loops, arrays, sizes and calls, made from the traced body: `for` and
  `sum` binders, array parameters, the arrays a program makes, calls, and
  calls inside a kernel's loop, each with the line that defines it. The long
  forms, `k.loops.i`, `k.arrays.f`, `k.calls.flux` and `k.sizes.n`, say
  what kind of name they are, and a short name that is two kinds at once is
  refused with both long forms. A kernel called twice is `p.scale[0]` and
  `p.scale[1]`, and a second `for i` in one body `k.i[1]`; the bare name is
  refused, listing both with their lines. A split's halves are `.outer` and
  `.inner`. The loop of a program's checked point, which no line binds, is
  its call's, by the parameter it checks: `p.gather.checks.idx`, which a
  fusion runs in the loop that writes the array, so that each cell is
  checked as it is written (#306). `loopty.scope(term)` gives the same for a
  term written by hand, and `dir` lists a scope's names.
- **`Schedule.on_groups(loop, axis=0)` and `Schedule.on_items(loop,
  axis=0)`**, `tag` with `"g.<axis>"` and `"l.<axis>"` in words, and printed
  as they are written, `on_groups(l2p.unpack.box)`; `tag` takes a dictionary
  of handles for several tags in one step.
- **`Schedule.loops` and `Schedule.nest`**: the schedule's loops as handles,
  and the loop nest by their paths, `l2p.unpack.box` or `jacobi.t.outer`.
- **The term records where its names come from** (`loopty.term.Binder`,
  `loopty.term.Callsite`): each binder's name, its line and the calls it is
  in, and each call's labels, kernel, line and arrays.
- **A device schedule's code where pyopencl is not installed** (#291).
  `loopty run --emit-code` and `loopty.executor.emit_code` print the OpenCL
  C of a schedule for `"opencl"` on `CheckedOpenCLTarget`, the code
  `InKernelOpenCLTarget` generates where pyopencl is, numpy's arithmetic and
  pyopencl's complex numbers included. Running it still needs pyopencl, and
  says so.

### Changed

- **A schedule runs without the zeros no read sees** (#290). Where isl
  decides that every cell a program's calls read of an array it makes, an
  earlier call stored, the definedness `substitute` and `contract` ask, the
  schedule drops the zeros the program makes the array with, on every
  target and before any step. The fact is the schedule's first, at the line
  that makes the array (`every cell of burgers_rhs.f that burgers_rhs reads,
  burgers_rhs.flux has stored by the time it is read, so burgers_rhs runs
  without the zeros burgers_rhs.f is made with`), and a window or a
  substitution rests on it instead of a definedness fact of its own. So a
  split loop that stores such an array is legal on a device (#289):
  `examples/l2p.py`'s `on_a_gpu` zeroes nothing, and the C target no longer
  zeroes what the executor allocates as zeros. Zeros a read sees, padding at a
  boundary, are kept, and a tag refused for them says which read sees them.
- The definedness of a window or a substitution is refused for a read of
  the cell its own instance stores, `f[i] = x[i] + f[i // 2]` at `i = 0`,
  which no dependence pairs with its store, and asks whether a read comes
  before a later store under what `assume` says of the sizes.

- Every printed loop, array and statement of a schedule is its path. The
  zeros a program makes an array with are `the zeros of l2p.R`, with the
  line that makes it, and not `R.zeros[i_1=0]`, a loop the program made for
  them is `l2p.R.zeros[0]`, by its axis, a statement of a call is the call,
  `l2p.reconstruct`, where its id was printed, and a program's checked
  point is `the check of p.idx before p.gather`, which sets `the check's
  flag`. A refusal's suggestion is a call the person can paste:
  `fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) runs every
  dependence between them forward`. A schedule the C target cannot build
  names its loops on hardware axes the same way: `the loop spmv.r on g.0 is
  on a hardware axis, and the C target has none`.
- `examples/l2p.py` names its boxes `box` where it named them `i`, and its
  three calls fuse in one step, so `fused` has two cast facts where it had
  four.
- **A sum's operands that read none of its binders are computed once,
  before the sum** (#270), in the code a kernel or a schedule generates, as
  a loopy author writes `<> x = ...` before it. `x` and `y` of
  `examples/l2p.py`'s `evaluate`, Python locals the trace reads into the
  summand, were divided out at every term of the triangle; `loopty run
  --emit-code` now shows them assigned once per target, `invariant_0 = (tx[...]
  + -1.0 * cx[box]) / rscale`, and the sum reading `invariant_0`. The loop
  form of `contracted` on C, one core, at order 10: 33.0 ms before, 29.7 ms
  after, plain loopy's own loop 29.2 ms. The potentials are the same bits:
  only a whole operand moves, or the leading operands of a real sum or
  product, of the type it is computed in where it stands, under the
  statement's guard, and one that reads an array only where its cells are
  in bounds at every instance or the sum has a term at every instance (note
  33 in `docs/loopy-notes.md`). No fact changes: the steps transform and
  check the kernel as it is lowered, and the temporaries are in the code
  generated after them (`loopty.hoist`).
- **Two layouts at a number are stated for Lean as written (#275).** At a
  module constant, `N = 3`, Python folds `N**3 * e + (a + N * b + N**2 *
  c)` to `27*e + a + 3*b + 9*c` before loopty reads it, and `dim_tags`
  built the same sum from its strides, so `--show-lean as_dim_tags` in
  `examples/storage_layouts.py` read `27 * e + a + 3 * b + 9 * c = 27 * e +
  a + 3 * b + 9 * c`: true, and the arithmetic Python's. A layout's body is
  now run once more with each module constant it reads a symbol, and the
  number put back unfolded, and `dim_tags` gives each stride as the product
  of the extents nested inside it, so Lean is asked `3 ^ 3 * e + a + 3 * b
  + 3 ^ 2 * c = ((3 * 3) * 3) * e + a + 3 * b + (3 * 3) * c`, and proves
  it. isl reads the address Python computed, as before; a body that
  branches on a constant keeps the folded address alone.
- **The flagship's agreement with sumpy says how close it is (#278).**
  `sumpy_l2p` is `@upstream(l2p, within=1e-14)`, and its row reads `the
  native run of l2p agrees with sumpy_l2p in pot within 1e-14 of the largest
  value, on its example inputs`, where it said only that the two agree to
  the `approx` class, a millionth. On the example inputs the two differ by
  8.3e-17 where the largest potential is 0.217.
- A ledger row, a reason and a refusal about one statement name it as a
  refused pair names its instances, by its text, shortened as they shorten
  it, and its line, where they named it by its id (#274): `y[r] is in bounds
  for every instance of y[r] = reduce_sum(...) (spmv.py:114)`, not `... of
  S0`; `every cell of off that off[r + 1] = off[r] + cnt[r] (spmv.py:83)
  reads was written before the read, by off[0] = 0 (spmv.py:81) or itself`,
  not `every cell of off that S1 reads, S0 or S1 wrote before the read`;
  the definedness reasons, `y[i] = f[i] + 1.0 (padding.py:19) reads the
  zeros at f[0] and f[n - 1]`, not `shifted.S0 reads ...`; and atomic's,
  the window refusals, a program's flat access and loopy's guard refusals
  the same way. A statement loopty adds is named by what it is: `the zeros
  of l2p.R (l2p.py:317)`, and `the check of perm before travel.gather
  (travel.py:106)` with its flag `the check's flag`. A program's access is
  spelled with the source's loop names, `val[off[r] + j]` and not
  `val[off[r_0] + j]`, and so is the cell of an update it refuses. Fact
  ids, the witnesses in a fact's provenance and a map per statement still
  name a statement by its id, which is how such a map names it; its
  refusals give both, `S0 (y[r] = 0.0, rows.py:12)`.
  `loopty.source.Named` and `loopty.handles.statement_names` do the naming.
- The parts of a cast, and of a kernel's ordering and definedness facts,
  that `--verbose` and Lean show name their statements the same way, and
  spell each cell with the loops and sums the source names (#307):
  `R[box, a, b] written again by R[box, a, b] = -R[box, a - 2, b + 2]
  (l2p.py:273) after R[box, a, b] is written by R[box, a, b] = C[box, a, b]
  (l2p.py:271)`, not `R[i_0, a_0, b_0] written again by reconstruct.S1
  ...`; zeros a read sees, which a schedule keeps, write `a cell of f`, and
  a statement after them `the same cell`; and a sum's
  binder as written, `R[box, a, b]` and not `R[box, a_1, b_1]`. So does the
  reason a cast through the part of a tile isl computed is not offered to
  Lean, and the rule of a theorem's series row.
- **`IllegalCast.instances`** is the witness's pair as the message names
  it, `('u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2 (stencil_skew.py:61)
  at t=0, i=8', 'the same statement at t=1, i=7')`; `witness` keeps the
  ids. `examples/stencil_skew.py` and `examples/wavefront_acoustic.py`
  print it, `witness: ... runs before the same statement at t=1, i=7, where
  nt=16, nx=16`, where they printed the tuple, and their `rejected_*`
  functions return the refusal, not its message and witness. The wave demo
  lists the ids its map per statement names, each with its statement, and
  `examples/calls.py` and `examples/composition.py` show a statement's text
  where they showed its id.

### Fixed

- `realize(k.y, tree=True)` records one fact, that the accumulation into
  `k.y` is reassociated, and no `bijective` or `monotone` fact: the step
  moves no instance and changes no order, so the two said nothing about it,
  as `atomic`'s did not (#203). `examples/spmv.py`'s ledger is two rows
  shorter, and so is `examples/p2p.py`'s.
- A program fact's term in the JSON ledger (`lanky check --json`, `loopty
  run --json`) is one text whatever the hash seed: a hypothesis prints the
  arrays it mentions sorted, where a frozenset printed them in the order of
  the process's hash seed, `{'C', 'R'}` in one run of `examples/l2p.py` and
  `{'R', 'C'}` in another (#298).
- The test suite passes with the `opencl` extra installed. The two tests
  that ask that nothing imports pyopencl ask it where pyopencl is not
  installed, since `import loopy` imports it where it is; the test of
  `loopty run --target opencl` on a machine without pyopencl hides it (#179).
  The refusal of a group axis over a window kept in global memory is asked
  without pyopencl too, on loopy's plain OpenCL target, so that CI sees it,
  and with the refusal's wording since #276 (#299).
- **A program may pass one array to two parameters a kernel only reads
  (#221).** `outer_product(u, u, z)` runs natively, since reads do not race,
  and the program's term was refused when it was built. The term now names
  `u` once, and the two parameters' sizes are unified, so `a: Arr[Fin[n],
  Real]` and `b: Arr[Fin[m], Real]` make `n` and `m` the one length of `u`.
  A kernel that writes either is still refused, now saying which it writes;
  and what its contract checks of an array an earlier call wrote is checked
  once, however many parameters it is given for.
- **A union with sizes of other names (#249).** A kernel over `Fin[m] +
  Fin[m]` given an array built over `Fin[n] + Fin[n]` read no size off a
  union natively, so its contract could not tell what `m` is and refused
  the call, while the program that passes it on composed. Each piece is now
  solved for its sizes as a `Where` is.
- **A program's `x.dom[i].size` checks that the fiber exists (#250).** The
  composer kept the axis and not the index, so `x.dom[99].size`, and
  `x.dom[0].size` of an `x` with no rows, were read as the length of `x`'s
  second axis, where natively `x.dom[99]` raises `IndexError`. The term
  keeps the fiber (`loopty.term.Fiber`), an index out of range at every size
  is refused when the term is built, and the compiled run is refused at
  sizes that leave the fiber out, with `IndexError`, as natively.
- **A refused call after a checked point keeps what ran before it (#176).**
  The compiled program ran every call, the refused one and those after it
  included, and wrote back what they computed before it raised the
  refusal, where natively the program stops inside the refused call. The
  run is now of the program stopped at that call by a flag of its own
  (`loopty.compose.stopped_at`), so the caller's arrays hold what the calls
  before it wrote, as natively.
- **A layout called in a traced kernel is checked by the address the call
  returned (#236).** At symbolic indices the check called the layout's
  function a second time, on fresh variables, and compared that term with
  the stated one, so a function that answers differently when called again
  passed, and the kernel read through a subscript the layout's facts say
  nothing about. The term the call returned, which the trace records, is
  now compared with the stated address at the call's own indices and
  sizes: the same term, the same polynomial multiplied out, or the same
  quasi-affine function as isl reads it.
- **`@upstream` runs on the comparison's own file's `example_inputs()`
  (#262).** It read the kernel's module's, so a comparison declared in a
  file of its own, importing the kernel (`from l2p import l2p` in
  `l2p_vs_sumpy.py`), was `assumed` where the kernel's file has none, or
  compared on inputs the comparison file did not mean. The comparison's
  file comes first, then the kernel's, and the row's provenance says whose
  were used (`inputs`: `example_inputs() of l2p.py`).
- **A theorem a kernel cites that gives it nothing says why (#234).** It
  had no row and no reason anywhere, so a recurrence off by one, a guard
  off by one or a conjunct about reals that rounds looked the same as a
  theorem never cited, and an exception of the matcher was taken for a
  theorem that does not apply. Such a theorem now has a row of kind
  `by-theorem`, `assumed`, `on_the_triangle, which off_by_one cites,
  applies to it`, and `lanky check` says why under the table, in the
  kernel's terms: the hypothesis about an array the kernel writes that no
  conjunct of the postcondition is, put in the kernel's names, with the
  nearest conjunct (`no conjunct of the postcondition of off_by_one is its
  hypothesis recurrence, ... R[a + 2, b] == -R[a, b + 2]; the nearest is
  ... R[a + 2, b] == -R[a, b + 1]`); the conjunct about reals that would be
  a hypothesis and rounds; the binding no array or size can stand for
  (`its integer k would be s, which is no integer of fill's`); a conclusion
  that rounds; a refuted postcondition; or the exception. A theorem that
  decides a conjunct of the postcondition, or gives a row, has none.

## [0.3.0] - 2026-10-15

The third release, on lanky 0.3.0. Five demos, each written after a kernel
of sumpy, pytential, grudge or firedrake: sumpy's L2P (`examples/l2p.py`),
grudge's weak gradient and divergence on hexahedra (`examples/hexahedra.py`)
and its advection right-hand side (`examples/advection.py`), pytential's
direct QBX (`examples/layer_potential.py`) and PyOP2's mass assembly
(`examples/assembly.py`). Each states its facts, cites the theorems it
computes, and agrees with the upstream code it mirrors (`@upstream`). isl
decides each fact, and Lean proves it again from loopty's own description of
the kernel, with the hints isl found. New in the language and the schedule:
theorems a kernel cites (`uses=`), a program's postcondition, constant
tables, `loopty.math`, kernels called inside a loop, a program's arrays over
a domain, updates and `Schedule.atomic`, window contraction, checked layout
maps at every size, fusion on chosen outer loops and `Schedule.assume`. The
demos write their theorems with lanky 0.3.0's families of several indices
and complex parts, so loopty requires it.

### Added

- **What fusing and contracting S1's passes buys on a CPU**,
  `benchmarks/hexahedra.py c`: the weak gradient's three programs of
  `examples/hexahedra.py`, the passes one after another, `fused`, and
  `contracted`, on the C target, each checked against numpy and then timed
  in turn on one thread. Every program is built with loopty's flags and the
  same `--cflags`, which the CSV records, and `report` tables the rows as it
  does the device's.
- **sumpy's GPU schedule in the flagship's ledger** (`examples/l2p.py`):
  `on_a_gpu = fused.retarget("opencl").tag(i="g.0")`, `E2PFromSingleBox`'s
  own, each box a work group, written with the file's names. Its tag's two
  facts are rows of `lanky check` (the fusions' are `fused`'s), the python
  run lists them, and `loopty run` runs it on a device where pyopencl is
  installed and compares it with the native run; on a GPU and on a CPU
  through PoCL the two agreed exactly, up to 4,096 boxes. The same with each
  box's windows, `on_a_device()`, is checked by isl in the python run; Lean
  does not state a cast on windows each work item keeps of its own yet
  (#295).
- **A schedule for a device, checked where pyopencl is not installed**
  (`loopty.lower.CheckedOpenCLTarget`). A schedule for `"opencl"` is built
  on loopy's plain OpenCL target there, which has the same kernel and
  imports nothing, so a file keeps its device schedules at module level and
  `python`, `lanky check` and `loopty run` read them on any machine.
  Generating or running their code is refused with the reason
  (`ImportError: generating and running OpenCL code needs pyopencl, ...`),
  and `loopty run` says `not run here: running on a device needs pyopencl,
  which is not installed; its facts are checked all the same`, which is no
  failure. `--target opencl` without pyopencl is refused by name, as before.
- **A window of an array over a domain, of several writers and of a
  recurrence** (`Schedule.contract`, #244, #213). A placement over an array
  over a domain takes the domain's binders, and the window is the box its
  slots span: sumpy's L2P, fused box by box, keeps one box's triangle,
  `contract("R")` finds `lambda i, a, b: (a, b)`, 121 cells for the 66 of a
  triangle of order 10, as plain loopy's temporary of the box. The array may
  be stored by several statements, and read by one of them: the copy of
  the stored rows and the recurrence `R[i, a, b] = -R[i, a - 2, b + 2]`. The
  `definedness` fact asks that every read is of a cell one of them stored
  before it, and the `no-clobber` fact is asked of every pair of them,
  decided by isl and proved by Lean; a window too small, one row of the
  triangle, is refused with the read the recurrence makes, the cell and the
  instance that overwrote its slot. A window of a number of cells is each
  work item's own, as loopy keeps a private temporary, so two of its
  accesses on two work items need no order, and a read on a work item its
  cell's store is not on is refused with the pair; the boxes on `g.0`,
  sumpy's own schedule, each keep their triangle. A window of a size,
  `f_window[n]`, is still one for every work item. A recurrence along a
  loop, `f[j] = f[j - 1] + u[j]`, is a shift register too
  (`registers=True`, the default on OpenCL): its store is the newest
  register and its read the one before. `examples/l2p.py` now
  runs `contracted`, its fused program with the triangle and the table kept
  one box at a time, refuses the row with its read, and checks sumpy's
  schedule on a device; `benchmarks/l2p.py --variant windowed` (and
  `unrolled_windowed`) times it.
- **A layout at every size** (`loopty.layout`, #263). A layout's body may
  compute with a size its annotations name, so grudge's DOFArray is one
  layout for every order, `n**3 * e + (i + n * j + n**2 * k)` over
  `Fin[ne] x Fin[n]**3`, and so is loopy's `dim_tags` `N3,N0,N1,N2` of it,
  each stride the product of the extents nested inside it. isl cannot state
  `n * j`, so it decides each fact at `n` from 1 to 6 and the fact reads
  `tested`, saying why; Lean proves it for every `n`, given the steps
  `omega` lacks (lanky's products, quotients and multiplying out): the
  products that bound each digit's term, a case split on the first digit
  two cells differ in, and an address's digits by division as the cell it
  stores. Two layouts the same is a polynomial identity Lean multiplies
  out. All in core Lean. A refuted one names the size isl found it at,
  `padded stores (i=0, j=1) and (i=0, j=0) both at 0 (at m=2, n=1)`. A call
  gives the size by name, `dof_array(e, i, j, k, n=4)`.
  `examples/hexahedra.py` declares its three layouts at a symbolic order, so
  its claim that the DOFArray, #418's `dim_tags` and the kernels' row-major
  arrays are one layout holds for every order, where it held at order 3 and
  `as_dim_tags` read `x = x` in Lean.
- **A device cast proved whole** (#188). A cast that puts a loop on a
  hardware axis claims besides its order that no dependence joins two work
  items, which was stated for isl only, so the cast stayed `decided` where
  Lean is installed. Lean is now asked it too: for each dependence class,
  that the two instances run on one work item of every axis, each one's loop
  counted from where loopy starts it, and for a statement's sum that reads
  what its own instruction writes, that the two never touch one cell where
  they run apart. S1's three device launches, sumpy's tile in S2, Kulkarni's
  split in S3 and the advection right-hand side's launches of every split
  read `proved` by Lean, and the four demo drivers offer their casts to Lean
  where it is installed (`python examples/hexahedra.py`: `along_s: tag(...):
  2 facts, all proved by lean`).
- **A program's postcondition** (`loopty.unfolding`,
  `loopty.dataflow.program_stores`). A `@program`'s return annotation is a
  claim about what it leaves in its parameters, written in their names and
  the sizes its kernels name, and it is a row of the ledger, keyed as a
  kernel's is:
  - `tested` on the program's native runs, an equation that reads an output
    compared at that output's class, `approx` for reals, as the compiled run
    is compared with the native one: the calls sum in an order of their own,
    and agree with the claim as reals and as floats to rounding. A run that
    leaves it false refutes it, naming the cell and the two values;
  - `decided` by isl where what the calls store decides it, as a kernel's
    postcondition is: each call that stores no cell of an array twice, and
    reads none before its store, has a `single-assignment` fact of the
    program's, and each of its statements is an equation about what the
    program leaves, until a later call writes an array it names;
  - otherwise stated for Lean, a part of Mathlib's for a claim about reals:
    what the calls store unfolded into the claim a pass at a time, each
    theorem the program cites (`@program(uses=[...])`) rewritten where one of
    its sides is, its families at the sections of the arrays they stand
    for, `Fr (3 * x + 0)` and `w (3 * x + 0) e`, then the sums over a literal
    range written out and `ring`. Python finds the steps, and Lean checks
    them; a wrong one is no proof. Proved, the fact rests on the
    `single-assignment` facts and the theorems it used.
  In `examples/hexahedra.py`, `weak_grad` and `weak_div` claim that `grad`
  and `div` are the dense Kronecker operators of the three directions
  applied to the field times its metric terms, citing `sum_factorization`,
  and with Mathlib both rows read `proved lean`: the theorem proved for
  every order now says something about the programs that compute it.
  A claim that names anything but the program's parameters and sizes is
  `assumed`, and says what it names.
- **A theorem tied to the terms a kernel adds** (`loopty.summands`, #252). A
  cited theorem that concludes `all(c(n) == E for n in Nat if ...)` gives a
  `by-theorem` row where a sum in the kernel's term adds `E` at whole
  numbers `n`, with the theorem's variables bound to the kernel's
  expressions of the sorts it takes them at (a theorem about reals says
  nothing of complex numbers), and an identity the kernel cites, `L == R`,
  rewrites `R` in `E` into the kernel's `L` first; the row names the
  identity's premises that are not the theorem's. `examples/layer_potential.py`'s `single_layer` cites
  `line_taylor` and the new `by_the_conjugate`, `ratio(alpha, beta) == -beta
  / alpha`, the kernels' ratio by the conjugate, and its ledger reads
  `single_layer adds line_taylor's c(n) for n in 1..5, for every c with away,
  first and ode, read over the complex numbers`: `decided` by `theorem`,
  resting on both theorems, and proved by Lean with Mathlib, the identity
  rewritten and the theorem applied a term at a time. A theorem's annotation
  that calls `loopty.math`'s `conj`, `real`, `exp` and the rest, itself or
  through a helper the kernel calls too, reads them as lanky's functions
  (needs lanky with `real`, `imag` and `conj`), and a kernel's postcondition
  keeps the calls its trace makes (`loopty.math.as_traced`, `as_stated`).
- **A postcondition's equations of reals at their class** (`loopty.reals`,
  #254). An equation that rounds is compared after each native run within
  the tolerance of the class of what it reads, as the compiled run is with
  the native one, `|a - b| <= 1e-6 (|b| + 1)` for `approx`, and the
  statement says so (`..., to the approx tolerance`); one that rounds
  nothing, one of integers and one of an `exact` class are compared bit for
  bit as before. Where its cells are stored once, by statements whose sums
  run over concrete bounds, Lean proves it with Mathlib over the reals from
  what they store, the tables at the values they hold, resting on the
  `single-assignment` facts and on the `trace-faithful` fact.
  `examples/assembly.py`'s `mass_ffc` now promises the element's mass,
  `|J| (1 + delta_jk) / 24`, in place of a symmetric matrix: `tested` on its
  runs, `proved` with Mathlib, and restated of every cell after its call.
- **`@upstream(kernel, within=1e-12)`** (#268): the agreement with the code a
  kernel mirrors, compared within the stated fraction of the largest value
  of each output computed in floating point, and the row says so (`... in
  out within 1e-12 of the largest value, on its example inputs`); an `exact`
  output is still compared bit for bit. `examples/layer_potential.py`'s two
  rows state it, where the bound was in a test only.
- **Claims settled by running them** (`loopty.tested`): `@tested` on a
  function that takes nothing and returns whether its claim held, alone or
  with what it measured, puts the claim in the ledger as a `tested` row,
  decided by `run`, the docstring's first line its statement; `refuted` with
  the reason when it returns false, raises, or returns something that is not
  a verdict. A run is evidence about its inputs, so nothing here reads more
  than `tested`. `examples/layer_potential.py` measures pytential's Green
  identity and the two jump relations this way.
- **S2 of the demonstration** (`examples/layer_potential.py`): pytential's
  QBX on its own curve. The line-Taylor theorem, for every order, in the same
  file as the kernels that add its closed form, `coefficient(ratio, k)`;
  pytential's single and double layer on the five-armed starfish over complex
  points; sumpy's CPU tile (`sumpy_tile()`), its casts decided, with the
  write sumpy silences (`write_race(write_lpot*)`) the fact `distinct
  instances of S0 write distinct cells of out`, proved by Lean where it is
  installed; each kernel's agreement with what sumpy's own `LayerPotential`
  returned on the example's inputs (recorded at sumpy's commit,
  `layer_potential_sumpy.json`), an `upstream` row; and, as tested rows,
  pytential's identity test case and the jump relations, the errors falling
  faster than the QBX order less 1.6.
  `benchmarks/qbx_layer_potential.py` times the kernel against plain loopy
  and sumpy's own on one OpenCL device, records sumpy's results, and runs
  the kernels on the arguments pytential itself hands sumpy.
- **S1 of the demonstration** (`examples/hexahedra.py`, #253): grudge
  #418's sum-factorized weak gradient and weak divergence on hexahedra, the
  kernels written for `n` nodes per axis, so that their facts hold at every
  order. The metric product (`metric`, or `contravariant`, which sums the
  metric terms first, three operator applications where grudge makes
  nine), a pass per reference axis (`along_r`, `along_s`) and the last pass
  with the three directions summed (`gradient_along_t`,
  `divergence_along_t`), chained by the programs `weak_grad` and
  `weak_div`. The Kronecker theorem the passes rest on,
  `sum_factorization`, for every `n`, proved by Lean with Mathlib and tested
  without it; the DOFArray's layout, at order 3, the same as #418's
  `dim_tags` `N3,N0,N1,N2` and as the kernels' row-major arrays, each a
  fact. The passes fused as deep as each pair allows, one loop deeper
  refused with its pair and the fusion of fewer loops named, and what
  passes between them kept a row (`w`) and a slab (`v`) at a time; on a
  device, three launches, every node on a work item
  (`passes_on_a_device`). `benchmarks/hexahedra.py` times the variants
  against the same kernels in plain loopy and against grudge on one device,
  and compares them with #418's own `weak_local_grad` and `weak_local_div`
  (`agree`): on a warped mesh at orders 3 to 6 they agree to `3e-15` to
  `5e-15`, natively, compiled, fused and contracted, and as the device
  launches.
- **The demo's flagship, `examples/l2p.py`**: sumpy's L2P for the 2-D
  Laplace kernel (`E2PFromSingleBox` with the compressed Taylor expansion),
  in one file and one ledger. The claim sumpy's reconstruction rests on,
  for every order (lanky#86), and its form box by box on the triangle `a + b
  <= p`, both proved by Lean with Mathlib; sumpy's storage index as a
  checked layout; `unpack`, whose postcondition, decided from what it
  stores, is sumpy's storage order, `reconstruct`, whose postcondition is
  decided from what it stores and is the theorem's hypotheses, so the
  theorem applies, and `evaluate`, composed by the program `l2p` and fused
  box by box; the interchange of the triangle's loops refused with isl's
  pair, and each box's triangle kept in a window (above). The program agrees with sumpy's own kernel on the file's inputs to
  round-off (`upstream`, below), and with the charges' potential to `4e-13`
  at order 10.
- **Agreement with the code a kernel mirrors**, `@upstream(kernel)`
  (`loopty.upstream`). The decorated function computes, on the file's
  example inputs, what the code the kernel is written after computes there;
  `lanky check` runs the kernel natively on the same inputs and compares,
  output by output, at the accuracy the kernel's types claim, a dense
  output in its own shape. The row is `tested` by `upstream`, `refuted`
  with the cell and both values (or both shapes), or `assumed` with why
  where the function cannot run: its package is not installed, or its
  recording is not there.
- **`loopty.math.ipow(x, n)`**, `x` to an integer power multiplied out as
  loopy multiplies one, by squaring, rounded at every product, in both runs:
  natively by the same loop (`loopty.math.integer_power`), compiled as
  loopy's `loopy_pow_<base>_<exponent>`. The two runs agree bit for bit, in
  single precision too. An integer base or a real power is refused, natively
  from the values and traced from the operands' types, and so is the
  smallest power of the type the compiled kernel holds the power in
  (`-2**31`, `-2**63`), whose negation loopy's helper cannot take. `x ** n` of a real `x` stays numpy's power, the C
  library's `pow` in both runs; at the flagship's Taylor sum `ipow` is an
  order of magnitude faster on the C target (`benchmarks/l2p.py`).
- **`benchmarks/l2p.py`**, which times sumpy's own `E2PFromSingleBox`, the
  same algorithm written directly in plain loopy (loopy #970's L2P
  microkernel with the Laplace step put back, and variants that make
  loopty's choices), and loopty's schedules of `examples/l2p.py`, on one
  tree and one device, and compares their potentials with each other and
  with the direct sum. On the C target plain loopy is built with the flags
  loopty builds its kernels with (`-fwrapv`; `--no-wrapv` for loopy's
  alone), and each record names the flags it was built with. It also
  records sumpy's potential on the example's inputs, for the `upstream`
  row.
- **The S3 demo: firedrake's PyOP2 assembly loop** (`examples/assembly.py`).
  FFC's P1 mass kernel of firedrake's PyOP2 tests, `mass_ffc`, with its
  15-digit tables as constant tables, is a kernel of its own, called inside
  the loop over the cells as PyOP2 calls it, `mass_ffc(coords[cell[c]],
  Ae[c])`, and the cell's matrix applied to `u` is added into the nodes,
  `y[cell[c, a]] += ...`. The three identities the loop rests on are lanky
  theorems in the same file, proved where Mathlib is installed: the 3-point
  rule integrates every quadratic exactly, a cell's mass matrix is `|J| (1 +
  delta_jk) / 24`, and the scatter keeps the total on every mesh. PyOP2's GPU
  split is refused with the node its two test triangles share and accepted
  once the updates are atomic, and both PyOP2 branches' schedules take the
  assumptions their transforms make (`assume("nc >= 1")`, `assume("nc % 4
  == 0")`). `benchmarks/assembly.py` times loopty's schedules against a
  plain-loopy transcription of PyOP2's wrapper with Kulkarni's and Sun et
  al.'s transforms applied, and with `--nested` against the transcription
  given loopty's nesting of a cell's loops under the cell loop, which is
  what decides where loopy puts the `ilp` loop of Sun et al.'s split
  (interleaving the four cells, or running each whole); and
  `benchmarks/assembly_agreement.py` compares
  the example with firedrake's own assembly and PyOP2's own `par_loop`.

- **The advection right-hand side on hexahedra** (`examples/advection.py`),
  the stretch of the S1 demo: grudge #418's `WeakAdvectionOperator` with the
  upwind flux, from grudge's own geometry, as kernels for every order. The
  flux reads the neighbor's value through the face maps, in bounds by type;
  the inverse mass is pushed inward, as grudge #362 rewrites it, and the
  same kernels take the GLL quadrature's tables. Three lanky theorems, proved
  where Mathlib is installed: the two rearrangements, and that the lift onto
  the face nodes is the transpose of the face gather, so it keeps the
  element's total. The kernels fuse on the element, one loop deeper is
  refused with its pair, and the fluxes are kept an element at a time;
  libParanumal's lift onto the face nodes without its barriers is refused
  with the edge node two faces update, and accepted atomic. Five splits on a
  device (libParanumal's volume and surface, the fluxes first, three
  launches, one, one from the metric terms), each launch a checked schedule;
  `benchmarks/advection.py` times them in five shapes beside the same
  launches in plain loopy and grudge's own right-hand side, and compares
  them with grudge's on a warped periodic mesh.

- **A kernel's postcondition decided from what its statements store**
  (`loopty.dataflow`, #113). Where no cell of an array a kernel writes is
  stored twice, or read before its store, a `single-assignment` fact says
  so, decided by isl, and what each statement stores is what the kernel
  leaves: its value equation, `off[r + 1] == off[r] + cnt[r]` at every `r`
  of `S1`'s loop. isl decides the postcondition under the equations, each
  taken at the cells the claim reads, so `scan`'s reads `decided isl`,
  resting on the `single-assignment` fact of `off`, and with Lean `proved
  lean`, where it read `tested native`. An equation of reals isl cannot read
  is an atom, one per equation as written, its cells canonical and its sides
  in either order (`discharge(..., atoms=True)`), so `R[a + 2, b] == -R[a, b
  + 2]` is decided as what the recurrence stores at `(a + 2, b)`; Lean
  states it with Mathlib's reals and proves it by `linarith`. The native
  runs still test the postcondition (`native` in its provenance), and one
  that refutes it refutes it; one the equations do not decide keeps the
  runs' status, with why (`undecided`): a cell stored twice, a read before
  its store (the instance that stores a cell and reads it, however its
  index is spelled, `x[i] = 2 * x[i + 1 - 1] - 5`, reads it first), a write
  under a guard isl cannot state, an update (`y[i] += v`, or a scatter-add
  through a map, `y[cell[c, k]] += v`), whose cells hold what they held on
  entry plus every update: `S0 updates y[cell[c, k]] in place, so a cell of
  y holds what it held on entry plus every update, not what one statement
  stores`.
- **Theorems a kernel cites**, `@kernel(uses=[on_the_triangle])`. A theorem
  whose hypotheses are conjuncts of the postcondition, up to the names of
  their binders, with a family `R(a)(b)` matching the cell `R[a, b]`, gives
  its conclusion at the kernel's arrays: a `by-theorem` fact, `decided` by
  `theorem` and resting on the theorem and the postcondition, whose
  unbound variables stay universal and whose unmatched hypotheses are its
  own, named in the row. For `examples/taylor_reconstruction.py` it reads
  `by on_the_triangle, for every D with pde and stored: ... R[a, b] ==
  D(a)(b)`. Lean checks the application, the theorem as it was proved
  applied with the arrays loopty matched, so a wrong match is no proof. A
  theorem whose conclusion is a conjunct of the postcondition decides that
  conjunct instead. A match is made only where it says something of what
  the kernel computes: every hypothesis about an array the kernel writes is
  a conjunct of the postcondition (a kernel whose recurrence is off by one
  gets nothing), and a conjunct about reals is matched only where nothing
  in it rounds, copies and sign changes, since the theorem reads it as
  reals. A family into `Nat` or a `Fin` sort stands for an array the kernel
  does not write; the cells it writes take a theorem over `Int`. The
  theorem's variables are put in wherever they are, every sort included: a
  `D: Fn[Fin[n], Real]` nothing binds, with `n` matched to the kernel's `p`,
  ranges over `Fin[p]` (#239), and `m` in `m**2` is the `m` the match bound,
  never a kernel name of the same spelling. A family is applied as `R(a)(b)`,
  `R(a, b)` or `R[a, b]`, as lanky reads them.

  ```python
  @kernel(uses=[on_the_triangle])
  def reconstruct(
      c: Arr[Where[a : Fin[2], b : Fin[p + 1], a + b <= p], Real],
      R: Arr[Where[a : Fin[p + 1], b : Fin[p + 1], a + b <= p], Real],
  ) -> copies & recurrence:  # the two conjuncts, written out in the example
      ...
  ```

  ```console
  $ lanky check examples/taylor_reconstruction.py
  ...
  decided under on_the_triangle  assumed    theorem      taylor_reconstruction.py:116  reconstruct      by on_the_triangle, for every D with pde and stored: forall a in Fin(...
  ...
  ```
- **`lanky check` lists a file's schedules** (#191). The casts of every
  schedule a file holds at module level are rows of its ledger, under the
  kernel or program they schedule, decided by isl as each step was taken
  and proved by Lean where it is installed; two schedules that share their
  first steps share those facts, which are listed once.
- `examples/taylor_reconstruction.py`: sumpy's compressed Taylor
  reconstruction from the theorem to the kernel to the schedule, in one
  file and one ledger, the seed of the demo's flagship.
- A matched theorem's family of families applied one index at a time,
  `R(a)(b)`, stands for an array indexed at once, `R[a, b]`, for a program's
  cited theorems too.
- **Window contraction** (`Schedule.contract`,
  [#132](https://github.com/xywei/loopty/issues/132)): an array a program
  makes, kept in the cells live at once. Fused one step behind, the Burgers
  divergence reads `f[j]` and `f[j - 2]` at step `j`, so

  ```python
  fused = Schedule(burgers_rhs).fuse("flux", "divergence", shift=1)
  fused.contract("f")                  # lambda j: j % 3, double f_window[3]
  ```

  keeps three cells and computes each flux once, where `substitute("f")`
  computes it twice. A placement is a Python function from a cell's indices
  to its slot, quasi-affine, given (`contract("f", lambda j: j % 3)`) or
  found, axis by axis, outermost first, along the order as it stands (one
  row of the sum-factorized passes' `v`, `lambda e, a, b: b`). The step is
  legal when every read is of a cell the producer stored before it (the
  strict `definedness` fact a substitution asks) and no read sees its slot
  overwritten (a new `no-clobber` fact, decided by isl over the schedule's
  order, refuted with the read, the cell it reads and the instance that
  overwrote its slot, `contract('f', lambda j: j % 2) illegal: instance
  divergence.S0[i=14] reads f[13], which flux.S0[j=13] stored in slot 1,
  after flux.S0[j=15] stored f[15] there`). From then on every pair of
  accesses to one slot is a dependence of the schedule, in the order they
  ran, so a later step that breaks the window, or puts its slots on two work
  items, is refused with its pair. Two lowerings: modulo addressing,
  `f_window[j % 3]`, and a shift register (`registers=True`, the default on
  OpenCL) whose every access has a constant index, the window shifted down
  by one at each step of its loop, which a device keeps in registers. The
  zeros the program made the array with are dropped, as a substitution
  drops them. Lean states `no-clobber` from loopty's description: three
  instances through every step's map, the placement with `%` by a number,
  and the time vectors of the order. `examples/fusion.py` and
  `examples/sum_factorization.py` show it; an array its producer reads, or
  that two statements store, and a window per work item are not done (#213,
  #214). A refused window names the fix, `contract('f') finds the smallest
  window that keeps every read, lambda j: j % 3`, and a step that breaks one
  the instance that would now overwrite a slot first. A window and a
  substitution of an array computed from it are refused together, in either
  order (#229): the substitution moves the window's reads. Every window of the
  tests computes the bits the array stored in full does, on C and on an
  OpenCL device, and a shift register inside an outer loop shifts in each of
  its steps.
- **Checked layout maps** (`loopty.layout`): a layout is a Python function
  from a cell's indices to its address, whose annotations say which cells
  (one `Fin` per axis, or one parameter over a `Where`) and which buffer
  (the return annotation), and `layout(first, second)` is a layout of
  pieces:

  ```python
  @layout
  def dof_array(e: Fin[ne], a: Fin[N], b: Fin[N], c: Fin[N]) -> Fin[ne * N**3]:
      return N**3 * e + (a + N * b + N**2 * c)

  as_dim_tags = dof_array.same_as("N3,N0,N1,N2")   # loopy's dim_tags
  ```

  `lanky check` asks three facts of each layout, every cell at an address of
  its buffer, no two cells at one address, and a cell at every address, and
  one of a `same_as` claim, every cell stored where the other layout, or
  loopy's `dim_tags`, stores it; isl decides each and Lean proves it, the
  third with isl's inverse of the layout, in pieces with `//` and `%`, as
  its witness. A layout of one piece is called as its function, so a kernel
  reads through it; a call whose address is not the one the facts are about
  (a constant or a helper the function computes with changed since it was
  declared) is refused, with both addresses.
  `examples/storage_layouts.py` checks sumpy's
  compressed-Taylor storage index in two pieces, and grudge #418's
  `N3,N0,N1,N2` against modepy's F-order reshape of the DOFArray; the
  reshape in C order, `N3,N2,N1,N0`, is refuted with a cell, in words:
  `dof_array stores (e=0, a=1, b=0, c=2) at 19, and dim_tags N3,N2,N1,N0 at
  11`; so are two cells at one address and an address with none. A buffer of
  several axes, and an argument that names its layout, are not done (#216).
- **A kernel called inside another kernel's loop** (`loopty.calls`). A
  kernel body calls a kernel as Python calls a function. A kernel whose
  return annotation is a sort is a rule: it returns its value and writes
  none of its parameters, and a call of it is that value; a kernel that
  writes its parameters is called as a statement and writes through what it
  is given:

  ```python
  @kernel
  def dot(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real]) -> Real:
      return reduce_sum(a[k] * b[k] for k in a.dom)

  @kernel
  def matvec(A: Arr[Fin[m], Fin[n], Real], x: Arr[Fin[n], Real],
             y: Arr[Fin[m], Real]):
      for i in y.dom:
          y[i] = dot(A[i], x)
  ```

  `A[i]` is the row of `A` and `x[col[r]]` the cells of `x` a row of
  indices picks, as in numpy, so a sparse product is `y[r] = dot(val[r],
  x[col[r]])`; a row of a ragged array, of a table, or a whole array over a
  domain may be passed too. A parameter of a function sort, `f: Fn[Fin[3],
  Real]`, takes a `lambda` or another rule, so that a quadrature rule is
  applied to an integrand, `integrate(lambda q: PHI[q, j] * PHI[q, k])`.
  Traced, the call is inlined: the callee's body runs again in the caller's
  trace against views of the caller's arrays, its loops and sums nested in
  the loops around the call, a binder named like a name of the caller's
  given one of its own (`k_0`), which is loopy's substitution rule with a
  domain of its own (loopy #1044) expanded where it is used, and every fact
  of the caller is about the expansion. The shapes are unified at the call,
  size by size, and a call whose sizes nothing says agree is refused, naming
  the axis. What the callee's contract checks of each call natively, the
  caller owes at every point it calls, as facts isl decides and Lean states
  from loopty's own description of the call: each row passed in bounds
  (`A[i], the row passed to dot as a, is in bounds wherever dot is called`),
  the cells a gather picks in bounds by type, an integral scalar in its sort
  (kind `argument`), two rows of one array apart where the callee writes
  one (kind `distinct-arguments`); and after each call the callee's
  postcondition, restated of what the call passed and resting on the
  callee's own fact (kind `postcondition-in-scope`). An argument is
  evaluated once, when the call is made, as in Python: every cell read for
  the value of a scalar argument, or for the indices of a row (`p[i]` of
  `A[p[i]]`), is owed in bounds at every point of the call, whether the
  callee uses the parameter or not, and a call whose
  callee writes an array an argument reads (`scale(A[i], A[i, 0])`) is
  refused, naming both; so is a function given whose default or captured
  name holds a value read from such an array (`lambda q, saved=x[0]:
  saved`), which natively is the value when the function was made. A
  scalar argument is converted into the sort the callee declares, as its
  contract converts it natively (#232): a double given for `s:
  np.float32` is `np.float32(x[i])` in the caller's term, which the
  interpreter computes as numpy does and the lowering writes as C's cast,
  and a loop variable given for `Int` is a numpy `int64` beside a
  `float32`, while an index or a comparison of it reads as the caller's
  own, `A[i, i]`. What the contract refuses is refused where it is passed:
  a real or a truth value for an integral sort, a number for `Bool`, a
  complex value for a real sort (naming `real(z[i])`), a literal numpy
  does not convert (`300` for `np.int8`), and, when the compiled run is
  asked for, loop arithmetic a narrow integer does not hold where the call
  is made. A callee's positional-only parameters are passed in order
  (#233). A rule may be applied to a function that applies it again, `integrate(lambda q: integrate(lambda p: ...))`; a kernel that
  calls itself, directly or through the functions it is passed, is
  refused. A rule on its own is the kernel that
  writes its value into a one-cell array, `result[0]`, which `loopty run`
  compiles and compares with the native call; one that takes a function is
  compiled where it is called. Natively, a call inside a `when` whose guard
  is false is not made, and a rule's value there is `0`, so the value of a
  rule called inside a `when` block is refused where it is used after the
  block. `examples/calls.py` has `dot` in a matrix-vector
  and a sparse product, and firedrake's P1 quadrature rule applied per
  element. A callee's update, `y[cell[k]] += yl[k]`, is an update of the
  caller, of the class of its sum and owing no distinct writes, as written
  inline: a step that runs two at once is refused with a pair of the
  caller's loop and the callee's read off the example inputs (`S0[c=0, k=2]
  and S0[c=1, k=1] both update y[3]`), and `atomic('y')` on the caller
  lifts it. An integer a callee is given for an `Int`, `np.int64(c)`,
  keeps such an update exact. A window (`Schedule.contract`) is taken over
  the statements of the kernels a program's kernels call, and its
  `no-clobber` fact is about them; a callee reads through a declared
  layout, which is checked against its facts where it is called.
- **A program's arrays over a domain** (#74). A program's callee may take
  an array over a `Where`, `Sigma` or union domain: the domain's sizes are
  unified across the calls, binder by binder, and two calls given one array
  have to agree on its points, compared as isl sets. A program may make an
  array over a domain, like an argument over one, `Arr.zeros_like(L)`, or
  of a declared shape, whose sizes it reads off its arrays as it does
  natively:

  ```python
  @program
  def pair_sums(x, e):
      f = Arr.zeros(TRIANGLE, n=x.dom.size)
      fill(x, f)
      rowsum(f, e)
  ```

  It is a temporary of the lowered kernel, stored boxed and zeroed over its
  points where the program makes it, its edges have definedness facts as a
  dense temporary's do, and a schedule fuses through it; `Schedule.pack`
  stores a program's argument over a domain packed. A declared dense shape
  is `Arr.zeros((x.dom.size, 3))`, and a run whose sizes make one of its
  lengths negative is refused compiled, as numpy refuses it natively.
  `Schedule.contract` may keep a dense one in a window; one over a domain is
  refused there, naming the fix, keep it whole or `substitute` it
  ([#244](https://github.com/xywei/loopty/issues/244)).

- **Constant tables** (`loopty.table`): quadrature weights, a basis
  tabulated at the quadrature points, a differentiation matrix, declared
  next to the kernel as the numpy array they are, `W = table([1 / 6, 1 / 6,
  1 / 6])`, and read by the body as an array is, `W[q]`, `W.dom`. Natively a
  table is a read-only copy of its values. Traced, it is recorded in the
  term (`Term.tables`) under the name the body reads it by, and each read is
  an access whose in-bounds fact isl decides over the table's shape. Lowered,
  it is a read-only loopy temporary with the values as its initializer, which
  loopy writes once with every digit, `static double W[3] = {
  0.16666666666666666, ... };` on the C target and `__constant` on OpenCL, so
  both runs read the same numbers; the interpreter reads them too. Complex,
  single precision and integer tables are read in their own dtype. A table
  read in a helper of the body is the kernel's, named as it is declared, a
  program holds one table for every call that reads it, and a table named
  like a parameter or a size of the kernel is named apart, `y_table`.
  Writing one is a `TypeError` in both runs. Natively a table is read as an
  array is: an index is never counted from the end, and a read out of range
  under a false `when` answers zero. A table holds finite numbers (#193), in
  a type the compiled kernel has (not `float16` or `longdouble`),
  is no argument (an array of its values is,
  `Arr.from_numpy(W.values.copy())`), and is read in a statement's values,
  not in a loop's bound (#194); each of these is refused with the fix named.
  A compiled run refuses a read-only array for a parameter it writes, such
  as a table's `W.values`, which it wrote into before failing. On OpenCL a
  complex table is an array of the kernel's own, since pyopencl writes its
  values as calls, which a `__constant` array cannot be initialized by.
- **`loopty.math`**: `sqrt`, `exp`, `log`, `sin`, `cos`, `real`, `imag` and
  `conj` for kernel bodies. Each is numpy's natively and a call the
  trace records, which the interpreter evaluates with the same numpy function
  and the lowering writes as the target's library function, `csqrt` and
  `creal` of a complex number on the C target and pyopencl's `cdouble_sqrt`
  and `cdouble_real` on OpenCL, in the type numpy computes it in: `sqrt` of
  an integer is a double (`sqrt((double) (k[i]))`), where loopy refused it on
  the C target and took it for a `float` on OpenCL, and `real`, `imag` and
  `conj` of a number that is not complex are the number, `0` and the number,
  where loopy wrote `creal` of a double. A function numpy computes in half
  precision, `sqrt` of an `int8`, is a `TraceError` naming `sqrt(1.0 *
  k[i])`. pytential's line-Taylor single layer over complex points,
  `real(log(c[t] - y[s]) - ...)`, runs, checks and compiles as any kernel
  (`examples/layer_potential.py`). A postcondition may call them too, `->
  all(y[i] == sqrt(x[i]) for i in Fin[n] if x[i] >= 0)`, and is tested after
  each native run with numpy's, where it was left `assumed`, not evaluated.
  `examples/p2p.py` calls `loopty.math.sqrt` where it built the call by hand.
  A value of one compares and combines as any traced value does,
  `when(sqrt(x[i]) > 1.5)`. numpy's own functions, Python's `math` and
  `x[i].real` take numbers: a body that calls them on a traced value is
  refused at its line, with the `loopty.math` function to write (#195),
  where it failed far from the body (numpy's "loop of ufunc does not support
  argument 0 of type Subscript", pymbolic's `UnknownVariableError`).
- **A sum's loop ordered outside a loop of its statement**
  (`Schedule.prioritize("s_outer", "t_inner")`). sumpy's CPU tile of
  a layer potential splits the targets by 16 onto a group and its work
  items, splits the sources by 256, and runs the sources' outer loop
  outside the work items' loop; `prioritize` and `interchange` refused the
  sum's loop as no loop of the kernel. A loop of a sum named before loops
  of its statement now runs outside them, each sum still adding its terms
  in its own order, so nothing is reassociated; a loop the sum runs outside
  of orders none of the instances in it, so a dependence between two of
  them is refused with its pair, as one across an `ilp` loop is. loopy runs
  the order only where it keeps an accumulator per instance, on a hardware
  axis or an `ilp` loop (`acc[16]` on the C target); on a loop that runs in
  sequence it keeps one and drops the order, and the schedule's `buildable`
  fact is refuted with that reason and the tag that fixes it. Two loops of
  one sum are never reordered, and an inner one never runs outside a loop
  its outer one does not.
- **Fusion on chosen outer loops** (`Schedule.fuse(..., loops=1)`). A
  fusion took every loop both sides have, so two passes over elements, the
  second of which reads every cell of the element the first wrote (a
  sum-factorized operator, one reference axis per pass), could not be fused
  at all: fused on the element and the row, the second pass reads a cell of
  the element before the first stores it. `loops` fuses only that many of
  the outer loops and leaves the loops inside them one after the other, the
  producer's first, checked as every fusion is. A refused fusion now names
  each fusion it checked that is accepted: the least shift at the same
  depth, as before, and the deepest fusion of fewer loops, `fuse('along',
  'along@2', loops=1)`.

- **isl decides, Lean checks** (`loopty.description`, `loopty.obligations`).
  Every fact isl decides about a kernel, a program or a schedule's step is
  also stated as lanky terms for lanky's Lean oracle, which proves it with
  `omega`; where Lean is installed such a fact reads `proved` by `lean`, and
  where it is not, nothing changes. The statements come from loopty's own
  description of the kernel and never from a set isl computed: every set
  loopty builds keeps the text it was written from, on the set itself
  (`description_of`), with each call's renaming, and every schedule step a
  record of its maps as written (`Schedule._records`, `loopty.schedule.Step`).
  In bounds, distinct writes, element sorts, the source order, a cast's
  bijectivity and its legality (one part per dependence class), and what a
  program's call may assume (requirements and flat accesses under the
  hypotheses isl used) are stated. isl's work is used as hints Lean checks:
  its functional form of a step's map as the witness that the step is total,
  its `lexmax` pieces as the witness of a definedness, the instances it took
  of a universal hypothesis, and its counterexample to a refuted fact, which
  Lean proves false. A refuted fact is offered to Lean only as that
  counterexample, and one with none Lean can check (a refused bijection, say)
  is not offered at all, so that a proof never reads `proved` where isl
  refused. A map given to `Schedule.affine` as isl writes it (`2 t + i`,
  `floor(i / 4)`) is read back too. Facts loopty states for isl alone (a
  rewritten layout, the casts after a substitution, a cast with a hardware
  axis) say so. `docs/lean-checks.md` explains it, and
  `scripts/lean_coverage.py` reports what Lean proves of every example and
  test kernel.
- `loopty run` asks Lean about each fact isl decided that it reports (the
  casts, a layout's facts), names what Lean did not prove under the table
  (with `--verbose`, also what Lean was not asked about, and why), and prints
  the Lean with `--show-lean [TEXT]`; `loopty check --show-lean` passes it to
  `lanky check`.
- **Definedness inside one kernel.** An array a kernel reads only at cells
  an earlier instance of its own wrote, as `scan` reads `off[r]`, gets a
  `definedness` fact where isl shows it, stated for Lean as a ∀∃ with isl's
  last writer as the witness. A kernel that reads what an array held when it
  was called gets none, and a write under a guard isl cannot state
  (`with when(x[i] > 0)`) counts for nothing, since its domain is wider than
  where it stores. A program's definedness of an edge is stated the same
  way.
- **Mixed subscripts.** An access whose indirect axes are indexed by a value
  of their own sort and whose other axes are affine, `X[cell[c, k], d]`, is in
  bounds by type on the first and decided by isl on the rest, where it was
  `assumed`; Lean states the whole access with the type as a hypothesis.
- **`Schedule.assume(condition)`**: a condition of the sizes, `"n % 4 == 0"`,
  taken as given, as loopy's `assume`. loopy is told, every later cast is
  decided and stated where it holds and rests on an `assumption` fact, and a
  call whose sizes break it is refused before anything is compiled.
- **Updates, a scatter-add: `y[cell[c, k]] += yl[c, k]`.** A statement
  that adds to the cell it writes a value that reads nothing else of the
  array is an update (`Stmt.exactness`). Its sum has the class of what it
  adds, as a reduction's has: `approx` for `Real`, `exact` for integers. Two
  updates of one cell give the same sum in either order, so they are no
  dependence of each other (`loopty.flow.dependences`), and a schedule may
  reorder them, which it records as a reassociation, as `realize` does. An
  update whose instances can add to one cell owes the class of its sum, `the
  accumulation into y[cell[c, k]] over c, k is approx`, instead of distinct
  writes, which were refuted before: a correct assembly loop or row sum
  written with `+=` now checks clean. An update of a cell of its own at
  every instance, `y[i] += a * x[i]`, still owes distinct writes.
- **`Schedule.atomic("y")`.** Two updates of one cell that a step runs at
  the same time, on two work items or under `ilp` or `vec`, can lose one of
  the two values. Such a step is refused with the pair and the fix:
  `tag(c_outer='g.0', c_inner='l.0') illegal: S1[c=0, k=2] and S1[c=1, k=1]
  both update y[3] on example_inputs() (nc=2, nv=4), where cell[0, 2] = 3
  and cell[1, 1] = 3, ...; make the updates atomic, with atomic('y') before
  this step`. After `atomic("y")` the step is accepted, it records `atomic
  updates of y run at the same time after tag(...)`, and `y` is compared at
  `reassoc`. On OpenCL each update is loopy's `AtomicUpdate`, a
  compare-and-swap loop, and every other access to `y` is atomic too, as
  loopy requires. loopy has atomic updates of `int32`, `int64`, `float32`
  and `float64` only; any other dtype is a `buildable` fact refuted. On the
  C target, which runs in one thread, the code is unchanged. `atomic`
  refuses an `exact` accumulation, as `realize` refuses a tree of one
  (#200), and an array nothing updates, naming what each statement that
  writes it does instead. `examples/assembly.py` is PyOP2's P1 mass action:
  its GPU split refused with the node PyOP2's two triangles share, accepted
  with atomics, and run on the PoCL CPU device (`docs/device-runs.md`).
  An update may read a window (`Schedule.contract`), `y[node[i]] += f[i +
  1] - f[i - 1]`, and `atomic` does not lift the order of the window's
  slots; a window of an array that is updated is refused, since the
  updates add to the zeros a window drops, with the fix: `contract('f'):
  added_flux.S0 updates f[j] in place, ...; store each cell once, f[j] =
  ... (a reduce_sum where the updates run over a loop), or keep f whole`.
- **Refusals name a pair from the data.** isl cannot follow a cell read
  from an array, `y[cell[c, k]]`: it takes it to be any cell, so the pair it
  picks need not share one. A refused cast through such a cell now names two
  instances that touch one cell on an input, with what the subscripts read
  there. The inputs are tried in turn: the schedule's own example inputs,
  the module's `example_inputs()`, then drawn ones. Where none has such a
  pair, the message says the pair is isl's, naming the subscript isl cannot
  follow; so it does for a read a substitution moves (`substitute('f')
  reads y where f was read, which no run of an input shows`), which no
  input's run makes. `loopty.interpret.interpret(..., log=...)` records the
  cells each instance touches.

### Changed

- **The README opens with a front page**: what loopty is to someone who
  writes loopy kernels, the flagship `examples/l2p.py` in its three layers
  with its refusals and six rows of its ledger with Lean and Mathlib, the
  five demos in a table (what each is written after, what its ledger has,
  how it compares), how to run them, and where the limits are.
  The reference below it is as it was, the spmv walk-through under its own
  heading and the paragraphs the front page now says shortened. Run it and
  Install install the 0.3.0 releases from PyPI and clone the `v0.3.0` tag
  for the examples.
- **Requires lanky 0.3.0** (`lanky>=0.3.0`, was `>=0.2.0`). Its Lean oracle
  proves a plugin's parts with the solver's hints, which is how loopty
  states each fact isl decides, and its families take several indices,
  `D[a, b]`, and complex numbers apart, which the demos' theorems are written
  with. The tests that skipped on lanky 0.2.0 (the demos `l2p`, `hexahedra`,
  `advection` and `layer_potential`, the benchmark driver's test, the
  statements for Lean, rewrites by an identity) no longer skip, so CI's job
  against the released lanky runs them. `[tool.uv.sources]` still resolves
  lanky from `../lanky` in a checkout, and the built distributions ask for it
  from PyPI.
- **The interpreter finds a sum's points once, not at every instance**
  (lanky #125). The faithful check runs each kernel's term through
  `loopty.interpret`, and a reduction asked isl for its points at every
  statement instance: it fixed the loop variables, the domain's parameters
  and enumerated. Now a reduction's domain is enumerated once per run at
  the values of its parameters, its points kept by the loop variables each
  instance fixes, and where it is the loops' points times the sum's (a sum
  over `Fin[n]` in a loop nest), as its two factors. Each instance reads its
  own points, in the same order, and the work limit is checked on them by
  their bounding box as before. The rule for a node is found once per type.
  `lanky check examples/hexahedra.py` spends 7 s in the interpreter where it
  spent 31, and takes 12 s of CPU where it took 38 without Lean; every
  example's ledger, JSON included, is what it was.
- **`loopty run` says what a row lacks to be proved**, with lanky's
  `WITHOUT LEAN` and `WITHOUT MATHLIB` lines under its table, and sizes the
  table's statement column to the terminal, as `lanky check` does, where the
  lanky it runs with has them. The transcripts show each statement whole,
  wrapped under its row.
- A refusal names each instance of its pair by the statement as the source
  writes it, the line it is on, and its coordinates by the names the source
  gives its loops (#260). The flagship's interchange reads `R[i, a, b] =
  C[i, a, b] (l2p.py:276) at i=0, a=1, b=9 writes R[0, 1, 9], read by R[i,
  a, b] = -R[i, a - 2, b + 2] (l2p.py:278) at i=0, a=3, b=7, scheduled
  earlier`, where it read `instance S0[i=0, a=1, b=9] writes R[0, 1, 9]
  read by S1[i=0, a=3, b=7] scheduled earlier`; S3's race, `y[cell[c, a]]
  += reduce_sum(...) (assembly.py:269) at c=0, a=2 and at c=1, a=1 both
  update y[3]`. A long statement keeps what it writes into and shortens the
  calls on its right, `reduce_sum(...)`; an instance of the statement just
  named is `the same statement at t=1, i=7`. The same holds for every
  refusal that names a pair (of `tile`, `interchange`, `skew`, `affine`,
  `fuse`, `tag`, `contract`, `substitute`), and for a disjoint-writes fact
  refuted on an input, and the fix each names is kept. A pair on two work
  items says where each access runs by what it does, `the write is in a
  sum, on every work item of l.0`, where it named the statement a second
  time. A fact's `witness` keeps the ids, `S0[t=0, i=8]`, and so does a
  term written by hand, which has no source.
- A schedule's facts are at the line of the step that made them, the
  `.split(...)` or the `.assume(...)` of a chain, and its run at the line
  the schedule is finished, its last step (#259). They were all at the
  kernel's first statement, which for S3's `mass_action`, whose first
  statement is the one of `mass_ffc` it calls, was a line in `mass_ffc`.
  The run of a kernel the file does not schedule is at the kernel's
  definition.
- A sum over a `Fin` of a number, which the trace adds up term by term, has
  the facts of its terms' accesses stated once, over its binder, as the
  source writes the access (#181, #223): S1's `contravariant` reads
  `G[3*x + d, e, k, j, i] is in bounds for every x in Fin(3) and every
  instance of S0`, where it listed `G[0 + d, ...]`, `G[3 + d, ...]` and `G[6 +
  d, ...]`, and the three sums over `c` of `gradient_along_t` are one
  exactness row, `for every d in Fin(3)`. The term, and the code, still
  add the terms one by one. S1's ledger is 80 facts where it was 96, with
  the same verdicts.
- **A cast's fact id leaves out the target.** What a step means is the same
  on every target, so the id of a cast's facts is the kernel's definition and
  the steps (`cast:spmv.spmv@102:split('j', 2, ...):bijective`), and a
  schedule retargeted (`fused.retarget("opencl").tag(i="g.0")`) shares the
  facts of the steps it replays with the one it came from: a ledger lists
  each fusion once. A `buildable` fact and a run's agreement keep the
  target (`agreement:...:[c].split(...)`).
- **A substitution through an array computed from inputs alone.**
  `Schedule.substitute` computes an array where it is read from what its
  producer read, and refused one whose producer reads an array an earlier
  substitution computes where it is read, since the dependences of that
  array's own reads were carried to a statement that no longer runs. Where
  that array is computed from arrays nothing in the program writes (its
  inputs), there are none, and the chain is substituted: grudge's `v.n`
  from the normals, read by the upwind flux, read by the lift
  (`examples/advection.py`). A chain through an array something writes is
  refused as before.

- `contract` of a program's array over a domain names `substitute` as the
  other fix only where one statement stores the array, which is what a
  substitution computes again; the flagship's triangles, which two
  statements store, are refused with `keep it whole`.

- A kernel's postcondition that its statements' stores decide reads
  `decided isl` where it read `tested native`, after a new `single-assignment`
  row, and the restatements of it in programs are worth `decided`; the
  ledger of a file with schedules at module level lists their casts, a
  window's `no-clobber` and an atomic step's reassociation among them. The
  `definedness` fact a substitution or a window asks of a schedule leads
  with its step, `for substitute('f'), every cell of f that burgers_rhs
  reads, flux.S0 has stored by the time it is read`, so that the two rows of
  `examples/fusion.py`, one per step after one fusion, read apart. The
  transcripts in the documentation are regenerated.
- `loopty.hypotheses.discharge` names a universal hypothesis's binders apart
  from the claim's own names before taking it at the claim's cells: an
  instance at the claim's `r` of a hypothesis over `r` was left out, which
  lost a verdict.
- Where Lean is installed, a rewritten layout's facts are proved
  ([#186](https://github.com/xywei/loopty/issues/186)): one decided by the
  row starts is stated as the induction step, each write of the offsets
  storing the value its cell holds where they start at 0 and step by the
  counts, as the contract checks on entry; one decided by induction over the
  run, as each write's question under the order before it. And a
  substitution's `definedness` fact and the casts after a substitution are
  proved ([#187](https://github.com/xywei/loopty/issues/187)): the reads a
  substitution computes again are stated as reads of what the producer read,
  at the instance that stored the cell, ordered against every write of it,
  and the statements it took out are in no class. So every fact isl decides
  about the examples is now stated for Lean, and the coverage report in
  `docs/lean-checks.md` is rerun. A claim decided under hypotheses is given
  up to 32 of isl's instances of them, where it was given 16.
- `substitute` refuses an array kept in a window, and `contract` one
  computed where it is read.
- The coverage report in `docs/lean-checks.md` covers the demos' examples
  (`l2p.py`, the flagship, `assembly.py`, `hexahedra.py`,
  `layer_potential.py`, `advection.py`, `quadrature.py`,
  `sum_factorization.py`), `storage_layouts.py`, `taylor_reconstruction.py`,
  the calls in loops (`examples/calls.py`, `tests/kernels/calls.py`,
  `tests/kernels/program_domains.py`), the upstream test kernels
  (`tests/kernels/upstream.py`), the test kernels of programs'
  postconditions, of reals at their class and of series
  (`tests/kernels/program_posts.py`, `reals.py`, `series.py`), and the
  windows, updates, layouts, postconditions decided from what kernels store
  and cited theorems above: of 1584 facts isl decides in 34 files, Lean
  proves 1521 and confirms the counterexample of 37 of the 38 refuted ones;
  the other, two updates of one cell at the same time in `advection.py`, is
  stated for isl only. The 25 it leaves decided are claims about reals,
  which core Lean has no words for; with Mathlib it proves them too, 1546 in
  all, and none stays decided. `examples/layer_potential.py` has 32, all
  proved with Mathlib. `examples/l2p.py` has 48: 46 proved with Mathlib, and
  its refused interchange and window of one row, whose negations Lean
  proves.
- A ledger gains a `definedness` row for each array a kernel reads only where
  it wrote it before (`scan`'s `off`, `p2p`'s `term`, `pairs`' `f`), and an
  access with mixed subscripts is decided where it was assumed; the
  transcripts in the documentation are regenerated.
- The questions isl answers (`Empty`, `Subset`, `Bijective`, `Monotone`)
  carry a `lean` field, how the fact is stated for Lean, and a schedule's
  cast facts have a `Cast` as their term where they had none (a
  substitution's definedness fact an `IslOnly`, which says why Lean is not
  asked).
- The tests run with lanky's Lean oracle off, as CI runs them, but the ones
  marked `lean` (`tests/test_lean_checks.py`), so that the suite passes where
  Lean is installed and proves what the other tests read as `decided isl`.
  Those hold the printer to its safeguards: non-vacuity of every part of
  every example and test kernel, arrays given as tables; a mutation matrix of
  the stencil's subscripts, guard and schedules, where Lean has to fail on
  exactly the dependence classes isl finds reversed; and a differential of
  every domain and every in-bounds goal's cells against isl's membership.
- A cell set's dimensions are named apart from the loops of the access it is
  built for, so that a ragged row's bound written in a loop called `a0`
  reads back as that loop.
- A step that runs two updates of one cell at the same time is refused with
  a `disjoint-writes` fact of the step, where a `monotone` fact named the
  pair before. That fact is stated for isl only, with the reason
  (`docs/lean-checks.md`, #201). A `monotone` cast is stated for Lean without
  the pairs of updates of one cell, which need no order; a cast with no pair
  left is one trivial part.
- Fewer indices than an array has axes, `A[i]`, are a row of it in a kernel
  body, and an array of indices, `u[cell[c]]`, the cells it picks: a view a
  called kernel takes, natively the row as an `Arr` sharing the buffer and
  the gathered cells as a read-only copy, where both were refused as an
  operation on the whole array. Used as a value, a view is still refused,
  with the loop nest to write. A row of the cells a gather picks,
  `u[cell][c]`, is `u[cell[c]]`, as numpy reads it. A table's row, `PHI[q]`,
  reads natively as a read-only array, and as zeros out of range under a
  false guard, as a cell does.
- In a program's body, `x.dom.size` (and `x.dom[0].size`) is the size of an
  axis of the argument, for the arrays it makes; iterating `x.dom` is still
  refused, as a loop whose trip count is an argument. A kernel's contract
  reads a size of its declared domain off the bounds of an argument's
  binders, where the argument was built over a domain of other names.
- A kernel that returns a value without declaring its sort, `-> Real`, is
  refused when traced, naming the sort to declare; its value was dropped by
  the compiled kernel.
- Two array parameters a kernel only reads may share storage, `dot(x, x)`
  or the diagonal of a Gram matrix: reads do not race, and both runs read
  the same values. The contract still refuses two that share storage where
  the kernel writes either; a native call traces the body to know which
  it writes, only when two of its arguments share storage.
- A schedule step that names no loop of the kernel lists its loops, and
  names a called kernel's loop that was renamed when it was inlined:
  `fill's loop n, inlined at ..., is n_0 here`.

### Fixed

- **An `ilp` loop around a sum runs its lanes in one loop of the sum
  (#264).** A sum's domain named every loop around its statement, and loopy
  opens a sum's loop inside every loop its domain names, so
  `split("q", 3).tag(q_inner="ilp")` around `reduce_sum(... for b in
  Fs.dom[q, j])` gave each lane a loop over `b` of its own, and the lanes
  ran one after another. The domain now names only the loops its bounds
  read: over a box, the loop over `b` holds the three lanes' updates, as
  plain loopy writes it, while a triangle's `j < i + 1` keeps its sum in the
  loop it reads, and a ragged row's length keeps every loop. The facts are
  the term's and do not change. The launches of S1's "fields three at a
  time" shape are now plain loopy's code up to loop names, and every other
  shape's code is what it was; sumpy's tile on C in
  `examples/layer_potential.py` runs the sources' loop once around its 16
  targets, where it ran it once per target. Note 32 in
  `docs/loopy-notes.md` has the details.
- **A program's zeros on a device** (#255, #261). A program zeroes each array
  it makes in a statement of its own, in loops the lowering names (`i_2`,
  `i_5`), which ran on every work item: `fused.tag(i="g.0")` was refused at
  `C.zeros`, and the schedule that was timed tagged `i_2` and `i_5`, names the
  file never wrote. The zeros' loop along an axis of the array now goes on
  the axis of the loop that every statement storing the array stores it
  along, so each work item zeroes the cells it stores, before it stores them,
  and the work-item check asks about the zeros as about any statement.
  Where a crossing is a call's and the zeros', the call's is named. Zeros
  that cannot follow (the array stored along a split loop, or along loops on
  two axes) stay on every work item, and the refusal says why and that
  `contract` or `substitute` drops them (#289). grudge's weak gradient in one launch,
  every node on a work item of its own, is now refused for the row the pass
  along `r` reads from other work items, not for the zeros of `w`.
- On OpenCL, a complex number to a natural power below 100 is the products
  numpy's power by an integer takes (`z * ((z * z) * (z * z))` for 5), never
  pyopencl's `cdouble_powr`. loopy writes only the powers by 2, 3 and 4 as
  products, and by 1 or 5, which a written-out series has, `exp(n log z)`:
  a logarithm and an exponential for each, and not exact for `z**1`.
- A schedule's `ilp` or `vec` tag no longer has loopy warn, on every build,
  that it falls back to its first scheduler, which it does for every kernel
  with such a loop: the tag silences `v1_scheduler_fallback` on the kernel
  it makes. The warning opened `python examples/assembly.py` and
  `python examples/layer_potential.py` on a fresh checkout.
- An array of several axes read only through an index array,
  `coords[cell[c, k], d]` with `nv` in no loop bound, is declared with its
  strides, `stride:2,stride:1`, so that loopy indexes its rows; it was
  declared without a shape, and loopy refused the read.
- An access with mixed subscripts that a statement reads twice over one
  domain, `x[0, 0]` read for two entries of an inlined Jacobian, is one
  place, decided by isl; it was `assumed`.
- A schedule's assumption is named `assume('nc % 4 == 0')` in the status of
  the casts that rest on it (with lanky's `label`), where its whole id was.
- **Two statements' atomic updates of one cell asked loopy for a global
  barrier.** `y[p] += a[p]` and `y[p + 1] += a[p]` with `p` on `l.0` race
  on `y[p + 1]`, and after `atomic("y")` the schedule is legal and its
  `buildable` fact holds, but the lowered kernel kept the source's order of
  the two, so loopy refused to generate it (`MissingBarrierError`). Once an
  array's updates are atomic, the order is drawn again from the dependences
  the casts are checked against, and two updates no dependence joins need
  no order (loopy's `no_sync_with`): libParanumal's lift onto the face nodes
  (`examples/advection.py`).

- A write through a cell read from an array, `y[perm[i]] = x[i]`, no longer
  has its distinct writes refuted by isl with two instances that write no
  common cell. The fact is decided on the kernel's native runs instead. It is
  refuted with two instances that write one cell on an input, the example
  inputs first: `S0[c=0, k=2] and S0[c=1, k=1] both write y[3] on
  example_inputs() (nc=2, nv=4), where cell[0, 2] = 3 and cell[1, 1] = 3`.
  Where no input has such a pair, as for `y[i, p[i]]`, it is `tested`.
- The first run of a kernel with an array of four axes over one size, or a
  triangle boxed over two, no longer prints loopy's `ParameterFinderWarning:
  Unable to generate code to automatically find 'm' from 'u': nonlinear
  expression`, a warning about an equation of the strides loopy's invoker
  could not solve, while it found the size from the shape (note 28 in
  `docs/loopy-notes.md`). Five constructs kernels written like sumpy's L2P
  and grudge's sum-factorized passes need, a
  `Where` array beside a ragged family, a `Where` of three binders, `**` with
  a loop variable as the exponent in a sum of two binders and in a sum
  nested in another, four dense axes and `when(a + b == 0)`, were run three
  ways and work; `tests/test_demo_shapes.py` keeps them so.
- A split sum adds its terms in order. `split("s", 256)` of a sum's loop
  left loopy to choose the order of the two loops it makes, and loopy ran
  the inner one outside, `for s_inner ... for s_outer`, adding every 256th
  term first: another sum in floating point, `2**53` where the native run
  adds `1 + 1 + 2**53` to `2**53 + 2`. Every schedule now gives loopy a sum's
  loops in order, outermost first, as it gives it each statement's nest.
- `fuse` after a `split` or a `tile` of the producer's outer loop is no
  longer refused as unbuildable. The producer's domain is cut after the fused
  loops by name, not by position: loopy 2025.2 puts a split loop's two after
  the others, `[j, i_outer, i_inner]`, so
  `Schedule(squared_rows).split(i, 2).fuse("square2", "row_total")` read
  `buildable` refuted though both casts were decided
  ([#192](https://github.com/xywei/loopty/issues/192)).

## [0.2.0] - 2026-10-11

The second release, on lanky 0.2.0, which checks each theorem's reading of
its annotations and gives it a row of its own in the ledger, and whose `abs`
of a term compares (lanky#94); loopty requires it, so that the ledgers its
documentation shows are the ones an install prints. The rest is two waves of
fixes where the compiled run computed otherwise than numpy, or did not build:
subscripts, storage and names (#164: an index sum in a dividing subscript is
64 bits wide, a narrow subscript stays in its type, complex `==` builds on
OpenCL, a program's arrays are off the C stack, and every isl dimension
loopty names has a name of its own), and arithmetic and messages (#169: a
sum is computed as the body wrote it, `abs` of a real lowers in a guard, the
compiled run is refused before it starts where the native run is refused,
and loopy's refusal of a guard it cannot read names the fix). CI also tests
loopty against the released lanky (#160). The README's Known limits section
lists what is still open.

### Added

- **The names of isl dimensions** (`loopty.names`, #161). Every isl object
  loopty names the dimensions of keeps namedisl's invariants: each
  dimension has a name, no two share one, and none is named like a
  parameter. isl's reader breaks them silently, reading a tuple variable
  spelled like a parameter as that parameter (#110). `check_names` checks
  them where such an object is made, a statement's and a sum's domain when
  the term records it, the loops of a nest, an access relation (its cells
  are positional), the index tuples of a shape, a dense layout's map and a
  polyhedral domain's points, and raises `NamingError`, a bug of loopty's,
  naming the object and the clash. `rename` is the one way loopty renames
  dimensions and replaces parameters, all at once: composing a program
  (`compose.rename_set`), the binders of a nested sum in the lowering, and
  the native run fixing a parameter at the value it reads use it.

### Changed

- **Requires lanky 0.2.0** (`lanky>=0.2.0`, was `>=0.1.0`). Its ledger has
  a reading row under each theorem (the next entry), which the `lanky check`
  transcripts in the README, the quickstart and the examples' README show,
  and its `abs` of a term is lanky's `Abs`, so a comparison of it traces,
  `when(abs(k[i]) > 2)` and `127 == abs(c[i])` (lanky#94); the test of that
  (#165) no longer skips on an older lanky. `[tool.uv.sources]` still
  resolves lanky from `../lanky` in a checkout, and the built distributions
  ask for it from PyPI.
- **A theorem in a kernel file has its reading under it** (lanky #91, #155).
  lanky now checks each claim's reading of its annotations: the annotations
  are run again as plain Python at drawn values, the term is evaluated at the
  same values, and the two have to agree. Each theorem gets a second fact,
  `tested python ... the term computes what the annotations compute`, on
  which its pass or proof rests. The `lanky check` transcripts of
  `examples/spmv.py` and `examples/travel.py`, each of which has a
  `scan_monotone`, show the new row, and are regenerated with
  `scripts/refresh_example_outputs.py`; `examples/README.md` says what the
  row is. A kernel's and a program's facts are loopty's, and have no such
  row: their trace has its own check, `trace-faithful`. (This entry was
  filed under 0.1.0 by mistake; it was merged after that release.)
- CI also tests loopty as `pip install loopty` gets it: a second job installs
  it without `[tool.uv.sources]`, so with lanky from PyPI, and runs the suite
  and the worked example's `lanky check` and `loopty run` (#147).
- loopy's refusal of an access under a guard its bounds check does not read,
  a product of loop variables, a guard computed in floating point or an
  array's entry, names the statement, where the body wrote it, the guard,
  and a guard over integers that keeps the access in bounds, which loopy
  reads, with a size as the body reads it: `with when(i <= y.dom.size - 5):`
  for `x[i + 4]` under `when(i * 0.5 < 2)` (#148). It is
  `loopty.refusal.GuardNotRead`, a `LoopyIndexError` still, with loopy's
  message at its end. A guard that calls a function no statement's value
  calls, which loopy never resolves, is refused with a `LoweringError`
  naming the fix, where loopy failed with `KeyError` and the function's
  name.
- An array a program makes is off the C stack (#62). On the C target loopy
  declared it a variable-length array on the stack of the call, `double
  f[n];`, so the thread's stack bounded how large an intermediate could be,
  and a program whose intermediate was larger, eight megabytes on a thread
  with four, died with no diagnostic. `LoopyExecutor` now allocates it on
  the heap, zeroed, and passes it after the program's own arguments,
  `double *__restrict__ f`, and `emit_code` and `loopty run --emit-code`
  print it so (`lower.heap_temporaries`). The term and its schedules keep it
  a temporary, which `Schedule.substitute` computes away; an empty one stays
  where it takes no room, since loopy's C invoker cannot pass an empty
  array. The OpenCL target's global temporary is unchanged.

### Fixed

- A sum is computed as the body wrote it (#151). pymbolic builds `a - b` as
  `a + -1 * b` and `u - 1` as `u + -1`, the same nodes `a + -b` and `u + -1`
  build, and the trace, the plan and the interpreter read every such sum as
  the difference, which numpy computes otherwise: `c[i] + -a[i]` of an `int8`
  `a` is `c[i] + -128` at `a[i] = -128` natively and was `c[i] + 128`
  compiled, and `c[i] + -u[i]` of a `uint8` `u` adds `256 - u[i]` natively
  and subtracted `u[i]`, at every `u[i]` but zero. The trace builds every `-`
  the body writes as a `loopty.term.Difference`, a sum whose last term is
  subtracted, and any other sum is read as written. So `u[i] + -1` of an
  unsigned `u`, `k[i] - 2**31` and `k[i] + 2**31` of an `int32` `k`, `u[i] -
  -7`, `c[i] + -b[i]` and `b[i] - True` of a truth value `b`, which numpy
  refuses at every point and the compiled run computed as the other
  spelling, are `TraceError`s naming the fix, and `k[i] + -(2**31)` and
  `c[i] - -32768` of an `int16` `c` are computed in the type numpy computes
  them in. A numpy scalar subtracted is kept as written, where pymbolic negates it
  in its own type: `k[i] - np.uint8(3)` was `k[i] + 253` compiled. The
  smallest `int32` and `int64` are written as C writes `INT_MIN`,
  `(-2147483647 - 1)`: `-2147483648` is the negation of a `long` in C, and
  `k[i] + -2147483648` of an `int32` `k` was computed in 64 bits, never
  wrapping round as numpy does; `-9223372036854775808` is the negation of a
  constant no signed type holds, which GCC reads in 128 bits with a warning
  and clang as an `unsigned long`. That is on the C target, built with
  `-fwrapv`; OpenCL C has none, and its compilers fold `k[i] + (-2147483647 -
  1)` into `k[i] | (-2147483647 - 1)`, so there the literal is left as loopy
  writes it. A real less an integer keeps the sign of zero: `x[i] - 0` and
  `x[i] - i`, `-0.0` natively at `x[i] = -0.0` and `i = 0`, were `0.0`
  compiled, since pymbolic builds them as `x[i] + 0` and `x[i] + -1 * i`; the
  zero is written `-0.0`, and `i` is converted into the real's type before it
  is negated. A truth value subtracted, `k[i] - True`, which pymbolic takes as
  no operand, is a `TraceError` naming the integer, `k[i] - 1`. `k[i] - 2**63`
  and `b[i] - 2**63` of a truth value `b`, which numpy refuses in `int64`, are
  refused by the trace, naming a real, `2.0 ** 63`.
- `abs` of a real in a guard lowers (#154): loopy's type inference visits a
  statement's value and never its guard, so `abs`, which only the guard
  called, was never resolved, and loopy failed with `KeyError: 'abs'`. In a
  guard it is written `x < 0 ? -x : (x == 0 ? 0 : x)`, which is numpy's
  `abs(-0.0)` too, `0.0`.
- Loop arithmetic beside a numpy integer whose type does not hold it at a
  point of the loops, `u[i] + (i - 3)` of a `uint32` `u` at `i = 0`, `a[i] *
  (i * 100)` of an `int8` `a` from `i = 2`, and a loop variable stored into an
  `int8` array from `i = 128`, is refused by numpy natively there, and the
  compiled run computed it in C's type (#153). Where the call's sizes reach
  such a point, isl finds it, and the compiled run is refused before it
  starts (`loopty.refusal.RefusedNatively`, an `OverflowError`), naming the
  point and `(i - 3) * np.int64(1)` as the fix. Arithmetic isl cannot read,
  or a ragged row's loop, is left to the native refusal (#166). A sum's body
  is asked about where the generator's `if` clause holds, which skips it
  natively: `reduce_sum(a[j] + j for j in a.dom if j < 100)` of an `int8`
  `a` is computed at any size. A program's call after a checked point is
  refused once the run has shown that the points before it passed, whatever
  a point after it says; where one before it failed, that point's error is
  raised, as natively. A literal stored into an integer array that does not
  hold it, `y[i] = 300` of a `uint8` `y`, is a `TraceError`, as #141's
  literals are; under a guard, which masks the store, it is refused where the
  call's sizes reach a point of the guard, and `with when(i < 0): y[i] = 300`
  stores nothing in either run.
- A regression test for #152, which #156 fixed: a literal beside a loop
  variable under a double power in a single-precision statement is written as
  an integer, `(double) (i + 16777216)`, where loopy wrote `1.0f` and C
  rounded the loop variable past `2**24`.
- A sum of loop variables and sizes in a dividing subscript is computed in
  64 bits (#149). The plan leaves such a sum in 32 bits, as loopy computes a
  loop bound, and a remainder or a floor division in a subscript used the
  wrapped value once the sum passed `2**31`: `x[(i + k) % m]` at `k = 2**31 -
  1` read cell `(-2**31) % 3` at `i = 1`, where numpy reads `2**31 % 3`, and
  loopy wrote `x[(i + n) // 2]`, which it reads as affine with a division,
  as `x[(i + n) / 2]`. In a subscript loopy does not read as affine with no
  division the sum's first operand is widened now,
  `x[loopty_mod_int64((int64_t) (i) + k, (int64_t) (m))]`, and `abs` of one
  takes its sign from bit 63. The same sum in a value or a guard on the
  loops is still 32 bits wide (#162).
- A subscript over a narrow numpy integer type is computed in it, as numpy
  computes it (#157). The conversion back into `int8`, `uint8`, `int16` or
  `uint16` that an operation in such a type gets elsewhere (#122) was left
  out of a subscript, so `x[s + i]` of a `uint8` `s`, which loopy reads as
  affine, read cell 256 where numpy takes `250 + 6` round to 0, and
  `x[abs(s) % 7]` of an `int8` `-128` read cell 2, where numpy's `abs` keeps
  `-128`, whose remainder is 5. It is kept now, `x[(uint8_t) (s + i)]`, and
  loopy's reader declines the cast, so the subscript is written as it is.
- `==` and `!=` of complex values build on the OpenCL target (#158).
  pyopencl's complex types are structs, which OpenCL C does not compare, and
  loopy wrote `z[i] == w[i]` as it is, so the kernel did not build. They are
  compared part by part, as numpy compares, in the complex type of the two,
  a real operand with a zero imaginary part: `cdouble_real(z[i]) ==
  cdouble_real(w[i]) && cdouble_imag(z[i]) == cdouble_imag(w[i])`. The C
  targets compare C99's complex types with C's operators, as before.
- The index tuples of a shape whose size is named `i0`, `idx.to_set((Fin[i0],
  ...))`, had no points: their dimensions were named `i0, i1, ...` whatever
  the sizes are called, and isl read the dimension `i0` as the size, `0 <= i0
  < i0`. They are named clear of the sizes now, as a polyhedral domain's
  already were (#161).
- A sum by a binder named like a size or an integer parameter of the kernel,
  `reduce_sum(x[j] for j in x.dom)` over `x: Arr[Fin[j], Real]`, is refused
  at the trace, naming the fix (#161). Its domain was `{ [j] : 0 <= j < j }`,
  a set with a well-named dimension and no points, which the name check
  could not see: the compiled sum was `0` where numpy's was not, the
  in-bounds fact of `x[j + 5]` was decided over the empty domain, and a sum
  by `i` inside a loop bounded by a size `i` did not build. A loop variable
  so named was already given a name of its own. `flow.domain_set` and a
  polyhedral domain's points now also refuse a dimension named like a size
  their bounds meant as a parameter (`check_names(..., meant=...)`).

## [0.1.0] - 2026-10-09

The first release on PyPI, where only the 0.0.1 placeholder that reserved the
name was before, and the first on a released lanky, 0.1.0. A kernel is a
Python function over typed arrays, dense, ragged or over a polyhedral domain.
It runs natively on numpy, traces to a typed term whose obligations
`lanky check` prints in one ledger, decided by isl or by type where they can
be, and lowers through loopy onto the C target, where `loopty run` compares
the compiled run with the native one. A program's facts travel from one call to
the next, and every schedule step, a fusion of two calls included, is a cast
checked before it is applied. The README's Known limits section says what a
user is most likely to meet: index arithmetic that stays 32 bits wide in a
loop bound, an affine subscript and a few sums (#149, #157), integers past 64
bits (#139), a guard loopy's bounds check cannot read (#148), what
substitution and fusion do not do yet (#132, #133), and the OpenCL target,
which CI never runs.

The entries below are every change since that placeholder. 0.1.0.dev0, the
version on `main` from 2026-09-18, when the first kernel ran, was never
published.

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
  every cell the call reads is one a call before it stored, or one of the
  zeros the array was made with, which the zeroing stored when the program
  made it. The fact's term is the isl subset question between the two
  sets of cells, for the isl oracle. Where the call reads zeros no call
  stored, the statement says so (`... or one of the zeros f was made
  with`) and the reason lists the cells (`shifted.S0 reads the zeros at
  f[0] and f[n - 1]`), so zero padding at a boundary is read as the
  program wrote it and `lanky check` passes; where it reads none, the
  statement is the stricter one, every cell read was stored by a call
  before it. Only a read of a cell that neither a call nor the zeroing
  stored is `refuted`, with the cell, which isl never shows for an array
  `Arr.zeros_like` made. A read isl cannot list (an index that is not
  affine, a guard it cannot state) that may reach a cell no call stored
  leaves the fact `assumed`, with the reason; a cell the reading call, or
  a write isl cannot list, may have stored first is a zero or that value,
  and the reason says so. The composition demo has one row more,
  `decided`. A substitution keeps the stricter claim (below).
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
  at cells it never writes left loopy no order (a `CycleError`; note 22 of
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
  is left unwritten with the reason, as for an index array computed with a
  conversion in it (`(t[j] + 1) % n` of a 32-bit `Fin[m]` entry, computed
  in 64 bits), or a `Fin[m]` entry stored in a 64-bit cell (#145). It is
  refused with a `ValueError` for a parameter, an array two statements
  write, or a producer that is not pointwise, and with an `IllegalCast` when
  a read is of a cell the producer does not store, or stores after the read
  (the `definedness` fact of the step, refuted with the cell: computed where
  it is read, the array has no zeros, so a program that reads its zero
  padding is not substituted), or when something writes what the
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
- **A rewritten layout decided by induction over the run, or refuted by a
  native run** (#103). The `layout` fact of a kernel that writes a count, a
  start read from another array, or a start under a guard isl cannot state
  was left `assumed`, and every fact resting on it worth an assumption. The
  rows in order (each starting no earlier than the row before it ends, every
  row inside the buffer, no count negative) hold when the call starts, where
  the contract checks the offsets against the counts and the buffer, and isl
  now asks of every statement that writes the offsets or the counts whether
  the write keeps them so wherever they held before it
  (`loopty.typing.layout_facts`, through `loopty.hypotheses.discharge`): the
  cells the claim reads are parameters, the order before the write is the
  hypothesis about them, the statement's guard is the claim's antecedent,
  and the buffer's length is a parameter of its own. `cnt[r] = 0`,
  `cnt[r] // 2`, and a start moved within the room its row leaves are
  decided, and the reads and casts resting on the fact are worth `decided`.
  A family with no counts array is asked of its offsets alone. A kernel that
  writes the offsets and an array of the family, or a row's length inside a
  loop over the row, is not asked, since what rests on the layout needs more
  than the order there: the disjoint writes and dependences of a ragged
  array tell its cells apart as `[r, j]`, one cell of the buffer each only
  while no row moves (`cnt[r] = 0; off[r + 1] = off[r] + cnt[r]` writes
  `val[0, 0]` and `val[1, 0]` on one cell), and a loop over a row runs to the
  length it read when it started, which the in-bounds facts of its entries
  are stated against (`cnt[r] = 0` inside `for j in val.dom[r]`). Where isl
  cannot show a write keeps the order, the fact stays `assumed` with both
  reasons, and a kernel's native runs may refute it
  (`loopty.faithful.layout_fact`): one that reads a row off its buffer
  through the layout it has written, `off[r + 1] = off[r] + cnt[r] + 1`
  moving the last row past the end, or one that leaves two rows on one
  cell, `off[r] = s[r]`. Such a read is a `loopty.arr.LayoutError`, an
  `IndexError` that names the array read, and the runs the body raised in
  are kept for it (`loopty.faithful.Raised`). `loopty run` lists the
  kernel's own run-tried fact under a cast that rests on it. A program's
  layout facts are not tried on runs.
- **A `nat` requirement** (#119). A kernel's contract checks that the cells
  of a `Nat` argument are not negative, and in a program nothing checked
  that of an array an earlier call wrote: the compiled program ran a call
  the native one refuses, and #118 withheld the callee's postcondition after
  it. It is now a requirement of the program like a `Fin` element sort,
  decided under the hypotheses that held at the call or checked between the
  calls, so the compiled program stops where the native callee is refused,
  and the postcondition is a hypothesis after the call again. Offsets with a
  layout requirement owe none, since the layout makes each a sum of counts
  that are naturals at the call. Offsets a scan computes and a later call
  declares `Nat` without reading rows through them gain a checked point:
  that they are naturals follows from the scan's recurrence only by
  induction.
- **`refused_claims` in `loopty run --json`** (#99). Each claim of an id that
  another kernel claimed first, which `loopty run` decided or ran all the
  same, is kept on the fact in the table with its status and its
  counterexample, witness or reason, beside lanky's `duplicate_claims`,
  which keeps its statement alone; a consumer of the JSON can tell that the
  other claim was refuted, and why.

### Fixed

- A subscript loopy does not read as affine is computed in 64 bits compiled,
  as the rest of the integer arithmetic is (#129). `x[(i * i) % n]` was
  `x[loopty_mod_int32(i * i, n)]`, which wrapped round at `i = 46341` and
  read a wrong cell, or none, and `x[(col[i] * 7919) % m]` read a wrong cell
  at `col[i] = 271183`. The lowering had left every subscript in 32 bits,
  since loopy reads a subscript into isl and its reader raised on the cast
  that widens an operand anywhere else. `loopty.isl_reading` makes the reader
  decline a cast, as it declines a call, and loopy generates such a
  subscript as written: `x[loopty_mod_int64((int64_t) (i) * i, (int64_t)
  (n))]`. So is a sum of a 32-bit entry and a loop variable, as `travel.py`
  reads its flat buffer, `val[(int64_t) (off[r]) + j]`. A subscript loopy
  reads as affine without the widening, and isl writes with no division, is
  left without it (`x[2 * i]`): loopy replaces it by the affine expression
  isl gives back, in its 32-bit index type, which `-fwrapv` makes right for
  sums and products, and checks its bounds, which a cast would stop. One
  with a division is widened: isl keeps a multiple below half the divisor,
  and `x[(i * 499999) // 1000000]` was `x[(499999 * i) / 1000000]`, which
  named a negative cell from `i = 4295` over a few thousand cells. A sum of
  loop variables and sizes, which the plan leaves in 32 bits, stays so in a
  division too, past `2**30` (#149; note 23 of `docs/loopy-notes.md`). The
  reader declines so only while loopty builds, transforms, checks, generates
  or runs one of its kernels, inside `isl_reading.declining()` and in that
  thread alone: nothing is installed in loopy when loopty is imported, and
  the last context to exit puts loopy's own readings back, so another user
  of loopy in the process (sumpy, pytential) has its kernels read, checked
  and generated as without loopty. loopty enters it in
  `lower.lower_generic`, in building a `Schedule` and in each of its public
  methods, in `LoopyExecutor.run` and in `emit_code`.
- loopy no longer reads a non-integer literal in a guard on the loops by its
  integer part (#137). Its bounds check reads such a guard into isl, and read
  `0.5` as `0`: `when(i * 0.5 >= 1)` as false everywhere, so the check passed
  `x[i + 4]` under it without looking, and the compiled run read past the end
  of `x` where the native one is refused; `when(i < 1.5)` as `i < 1`, which
  let `x[i + n - 1]` through at `i = 1`. A float that is an integer was read
  as the integer, though the guard is computed in floating point, which
  rounds: `when(i * 2.0**52 + 1.0 <= i * 2.0**52)` read as false everywhere,
  and holds from `i = 2`. And a `Real` scalar was an integer parameter of
  the domain the guards are read over: `when((i < a) & (i > a - 1))` holds
  for no integer `a`, and at `i = 1` for `a = 1.5`. The reader declines a
  constant whose type is not an integer's, and a value argument whose dtype
  is not an integer's is no parameter, so loopy reads no guard computed in
  floating point, checks the access at every point of its loop, and refuses
  each. Under such a guard it also refuses an access the guard keeps in
  bounds, `when(i * 0.5 < 2)` over `x[i + 4]` and `when(i * 2.0 < n)` over
  `x[2 * i]`, as it does under a product of loop variables (#148). A guard on
  the loops that numpy computes in single precision, `(i + 1) ** -1` beside a
  `float32` scalar, is written with the cast instead of refused.
- `Schedule.substitute` computes an index array where it is read in a
  subscript again (#145). Since #101 and #128 such a value holds a
  conversion, the store's or its own (`(t[j] + 1) % n` of a 32-bit
  `Fin[n]` entry, computed in 64 bits), and loopy failed on it in code
  generation, so the substitution was decided and its kernel left unwritten.
  loopy's isl reader declines the cast now, and a read in a subscript keeps
  the value's conversions, the store's narrowing included (`y[(int16_t)
  (t[i])]`). A store's widening keeps the value and is the last thing done
  to it, so a read in a subscript reads the value inside it (`x[t[i]]`, and
  `x[n - 1 - i]`, which loopy reads as affine and checks); a read used as a
  value keeps it, `q[i] * q[i]` too after `x[q[i]]`. Each substituted kernel
  agrees bit for bit with the one that stores the array.
- The emitted code of a schedule no longer depends on the process's hash
  seed (#125). loopy builds the assumptions of a kernel given none over a
  `frozenset` of its parameters, and isl wrote the sizes in every bound in
  that order: `wavefront_acoustic.py --emit-code` printed `-4 + nt + nx` or
  `-4 + nx + nt` by `PYTHONHASHSEED`, with a cold cache. The lowering gives
  `lp.make_kernel` the universe over the parameters in the order the domains
  name them first (note 24 of `docs/loopy-notes.md`).
- `Schedule(schedule)` starts from the schedule it is given (#135). It read
  the schedule for its term alone, so `Schedule(Schedule(double).split("i",
  2))` was a schedule of `double` with no step, a split after it split the
  original loop, `loopty run` compared it with `double` and found them in
  agreement, and `Schedule(Schedule(double, target="opencl"))` was a C
  schedule. It is the schedule built again from its kernel, its steps
  replayed and each cast checked again, on its target, with its sizes and
  its example inputs, unless a target or sizes are given (the default
  target is now `None`, the given schedule's or `"c"`).
- The in-bounds fact of a read against a row length a write in its loop left
  behind is `assumed` (#144). A loop over a row reads the row's length once,
  when it starts, and the facts of the row's entries were decided against
  that reading: with `cnt[r] = 0` inside `for j in val.dom[r]`, the native
  run refuses `val[r, 1]`, and its fact was decided, worth assumed only
  through the kernel's layout fact, which anything that decided the layout
  another way would have made decided. A read in a statement that shares the
  innermost loop the length bounds with a write that can reach the cell the
  length is read from is now stated with the reason and left `assumed`; a
  sum or an inner loop over the row reads the length again where it starts,
  and a write of another row's length changes no reading of this row's, so
  their reads stay decided.
- An `int32` array of `Fin[m]` that a kernel writes is computed with in 64
  bits natively too (#121): the native run read an entry back as an
  `np.int32`, and `p[i] * p[i]` wrapped round at `p[i] = 46341`, where the
  compiled run computes in 64 bits. Its elements are read as `int64` through
  the masking view (`loopty.trace.read_elements_as`), and by the
  interpreter, and the array is written as the caller gave it, so a
  permutation may still be computed into an `int32` array.
- An operation numpy computes in a numpy integer type is computed in it
  compiled too (#122). C computes an integer narrower than `int` as an `int`:
  `a[i] * a[i] // 2` of an `np.int8` `a` was `8` natively at `a[i] = 100` and
  `-120` compiled; the result is converted back into the narrow type
  (`(int8_t) (a[i] * a[i])`), which wraps round as numpy does, and loopty's
  targets write that conversion where loopy would leave it out. C computes an
  unsigned integer beside a signed one of as many bits as unsigned: `u[i] +
  k[i]` of a `uint32` and an `int32` was `4294967295` compiled at `0 + -1`,
  and `u[i] < k[i]` true; the operands are converted into the type numpy
  computes in, and a comparison of integers compares a negative operand's
  sign first where C would compare it unsigned (`k[i] >= 0 && u[i] < k[i]`),
  as numpy compares exactly, `uint64` against `int64` included. A
  difference, which pymbolic builds as `a + -1 * b`, is computed in the
  difference's type, where the term negates `b` in its own type first:
  `k[i] - u[i]` of an `int32` `k` and a `uint32` `u` was computed in `uint32`
  compiled, and `c[i] - a[i]` of an `int8` `a` at `-128`, which C negated in
  `int`, keeps its value now that a narrow result is converted back. A `uint32`
  literal is written `3u`, where loopy wrote `3ul`, a `uint8` or `uint16` one
  `3`, where loopy wrote `3u`, which took an `int8` beside it round into an
  `unsigned int` (`a[i] + np.uint16(3)` was `4294967294` at `a[i] = -5`),
  and an operation loopy
  computes by a function in a wider type than numpy (`u[i] << 3`, by loopy's
  `int64` one) is converted back. `loopty.promotion` types C's operators by
  C's conversions (`_c_result`) and the functions by loopy's
  (`_loopy_result`). Note 23 in `docs/loopy-notes.md`.
- Integer literals in integer arithmetic are written as integers: loopy wrote
  a literal in the type of the place its operation stands in, so `k[i] + 3`
  of an `Int` stored into a real was `k[i] + 3.0`, computed in double, which
  never wraps round at `2**63 - 1` where numpy does, `-1 * k[i]` was
  `-1.0 * k[i]`, and `k[i] ^ 3` was `k[i] ^ 3.0`, which C refuses. So are
  those of a comparison of integers, which loopy wrote in the type it infers
  for their difference, a double for a `uint64` beside a signed integer:
  `u[i] == 9007199254740993` was `u[i] == 9007199254740992.0`, true at
  `2**53`. Any other literal is written in the type of the operands that are
  no Python number, as numpy computes it, and not in loopy's type for the
  operation, which takes an integer literal of 32 bits or more for an
  `int64`, and a `float32` beside one for a double: `x[i] * 2**62` of a
  `float32` `x` was multiplied in double, where numpy's product is
  infinite at `3e38`.
- pymbolic builds `c[i] - -32768` as `c[i] + 32768`, which numpy refuses
  beside an `int16` `c`; the plan computes it as the difference numpy
  computes, in `int16`, where the compiled run added in `int`, and the
  interpreter subtracts a literal only where numpy refuses to add it, as
  `u[i] + -1` of an unsigned `u`: it read `a[i] + -128` of an `int8` `a`
  as `a[i] - 128`, which numpy refuses.
- A sum or a product nested after the first operand of another is printed in
  brackets: pymbolic prints them flat, and C computes from the left, so
  `1.0 * (k[i] * j[i])` of two integers was multiplied in double, where
  numpy multiplies the integers first, and `x[i] + (y[i] + z[i])` was
  rounded otherwise (`lower._CText`).
- `abs` of an integer lowers, and computes numpy's (#123): loopy resolves
  `abs` as C's and refused an integer (`abs does not support type float32`).
  It is written without a branch, `(k ^ s) - s` with `s = k >> 63`, which is
  the smallest `int64` at the smallest `int64` under `-fwrapv`, as numpy's
  is, and converted back into a type narrower than `int`; of a truth value
  or an unsigned integer it is the operand. Not as `k < 0 ? -1 * k : k`:
  GCC reads that as its own `abs`, which it takes to be non-negative whatever
  `-fwrapv` says, and folded `1 >= abs(c[i])` to false at the smallest
  `int32`, at `-O0` too; loopy sums a reduction in a branch of an `If` only
  where the branch's condition holds of the partial sum; and loopy's bounds
  check reads an `If`'s condition into isl, whose reader raised on the cast
  that widens `i * i` in `abs(i * i - m)` (note 25). `abs` of a loop
  variable is a Python int's, natively and to the plan.
- A parameter, a size or a loop variable named like a macro a header the
  generated code includes defines, or like a function the kernel calls, is
  refused, naming what the code means by it (#124): `I` with complex values
  was `complex.h`'s imaginary unit in the parameter's declaration, and gcc
  failed; so would `NAN`, `INFINITY` and the `stdint.h` limits, OpenCL C's
  macros (`M_PI`, `INT_MAX`), the work-item functions OpenCL code calls on a
  parallel loop, and `pow` in a kernel with a power or `floor` beside a call
  of `floor`. A function the kernel never calls is a name it may use. Built
  on an OpenCL device, `NULL`, `SCHAR_MAX`, the extension macros
  (`cl_khr_fp64`), `pipe`, the image types, and `barrier` beside a sum on a
  local axis, which loopy syncs with `barrier()`, failed too, and are refused
  (`ATOMIC_FLAG_INIT`, `MAX_WORK_DIM` and the other macros OpenCL C or PoCL
  define). A size, an integral scalar or a reduction binder named like one
  of isl's keywords (`max`, `min`, `floor`, `mod`, read whatever their case)
  failed inside the trace with `isl_set_read_from_str failed: syntax error`,
  and is refused naming them; a loop variable of such a name is renamed in
  the term (`max_0`). A size that names something the module or Python already
  defines (`Fin[abs]`, lanky's `abs`) was that object, and lowering failed
  on it as a foreign object; it is a `TraceError` asking for another name.
- `-b[i]` of a truth value is a `TraceError` naming `~b[i]` for `not` and
  `-(1 * b[i])` for the integer (#130): numpy refuses it at every point, and
  the compiled run stored `-1`. pymbolic builds it as `-1 * b[i]`, so that is
  refused too; `1 - b[i]`, which is built as `1 + -1 * b[i]`, is not.
- A kernel named like an OpenCL C built-in function is renamed in the
  generated code (#131): `get_global_id`, `clamp`, `select`, `mad`, the
  `convert_`, `as_`, `vload`, `atomic_`, `work_group_` and `native_` families
  and the others of its specification, on every target, as loopy's and the C
  library's names are (`lower.is_library_name`). A name of a family, which
  takes in any ending (`atomic_add`), gets a `knl_` prefix instead of the
  suffix. Run on an OpenCL device, a kernel named `select`, `mad`, `sign`,
  `as_float` or `M_PI` failed to build or to be found in the program, and so
  did one named like a macro OpenCL C defines (`NULL`, `ATOMIC_VAR_INIT`) or
  a type it declares (`size_t`, `event_t`, `memory_order`), which are renamed
  too, as is `main`, which OpenCL C refuses as a kernel's name.
- An integer literal past 64 bits lowers where numpy computes it in a type
  that holds it (#140): `x[i] * 2**70` failed inside loopy's type inference
  (`integer constant too large`), and is written as a double now, as numpy
  converts it; beside a `uint64` one up to `2**64 - 1` is a `uint64`, and a
  comparison with one is exact, as numpy's is: one past that, or below
  `-2**63`, is compared with an integer as `2.0 ** 65` with its sign, where
  its own double rounded onto values a `uint64` or an `int64` reach
  (`u[i] < 2**64` was false at `2**64 - 1`). Beside a real or a complex
  value it is written in that value's type, into which numpy takes it:
  `f[i] < 2**70 + 2**46` of a `float32` `f` compared with a double, and was
  false at `2**70`, where numpy compares with the `float32` `2**70`. A guard
  on the loops or the scalars that compares with one, `when(i < 2**70)`, is
  not stated to isl, which wrote it into a loop bound loopy failed to type,
  and the guard's predicate compares it. Elsewhere, `1.0 * (i + 2**64)`,
  which Python computes exactly, or a store of it, is a `TraceError` naming
  `2.0 ** 64`, and one no double holds (`x[i] * 3**700`), which numpy
  refuses to convert, a `TraceError` saying so.
- An integer literal beside a numpy integer whose type does not hold it is a
  `TraceError` naming `np.int64(-1)` (#141): numpy refuses `u[i] // -1` of a
  `uint64` at every point (NEP 50), and the compiled run computed it in
  double; so it refuses `(b[i] // c[i]) * 200` of two truth values, which is
  an `int8` natively (#122). A literal after the first operand of a sum is
  refused only where its negation is not held either, since `u[i] - 1` is
  built as `u[i] + -1`; one first, `-7 + u[i]` or `2**31 + k[i]` of an
  `int32` `k`, as it stands. A negation is not refused, nor a comparison,
  which numpy and the compiled run decide exactly. `^`, `<<` or `>>` of a
  `uint64` and a signed integer, which numpy refuses, since only a double
  holds both, is a `TraceError` too.

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
- An integer result outside 32 bits is computed in 64 compiled too, as numpy
  computes it (#101): `c[i] * c[i] // 1024` of an `Int` at `c[i] = 2**20`
  stored `2**30` natively and `0` compiled. `Nat`, `Int` and Python's `int`
  are stored in 64 bits compiled (`contract.INTEGRAL_STORAGE`), and the
  contract's range for them rises to 64 bits (`contract.INTEGRAL_RANGE`);
  `Fin[m]` stays 32 bits wide (`contract.INDEX_STORAGE`, `INDEX_RANGE`),
  which its bound holds, so an index array stays compact. An operation that
  can leave its operands' range, a sum, a product, a power or a left shift,
  of a narrower operand, a `Fin[m]` entry or a loop variable, is computed in
  64 bits (`(int64_t) (col[i]) * col[i]`), but in a subscript, which loopy
  gives no way to widen (#129), and in a sum of loop variables, sizes and
  literals that total less than `2**30`, which is index arithmetic. An
  `np.uint64` beside a Python int or a loop variable is computed in `uint64`,
  as numpy computes it, where loopy typed it in double, lost the low bits of
  `u[i] % 3` and failed to lower `u[i] << 3`. The C target builds every
  kernel with `-fwrapv` (`lower.WRAP_FLAG`), so a signed overflow wraps round
  as numpy's does, where GCC folded `x[i] + 1 > x[i]` to true at `2**63 - 1`.
  The sum of a `Bool` array, a count natively, was
  accumulated in a byte compiled and wrapped round at 128; its body is cast.
  The native run reads an integral array stored in fewer bits through an
  `int64` copy, an `int32` one of `Fin[m]` included. Note 19.
- A scalar argument has one native meaning however it is passed (#102): a
  Python number is converted into a numpy scalar of its sort's dtype, as a
  numpy scalar of another dtype was, so `x[i] * a` of a `float32` `x` is a
  double natively for `a=0.7` as for `a=np.float64(0.7)`, and for a Python
  `int` beside it as for `np.int64`, which is what the compiled run computes.
  `~` of a comparison of a scalar is then a numpy bool's, logical, and the
  trace no longer refuses it as a Python bool's.
- Floating `%` and `//` lower, and compute what numpy computes (#104): loopy
  raised `NotImplementedError` from its code generator. Integer `//` and `%`
  by zero give numpy's `0` compiled too, where C's division killed the
  process with `SIGFPE`, and so does the smallest `int64` by `-1`, numpy's
  smallest `int64` and `0` (#105); loopy's own floor division also overflowed
  near the ends of the range. `<<` and `>>` lower as numpy computes them, `0`
  (or `-1`) for a shift past the width or by a negative amount, and `^` as
  C's, which is numpy's (#107); each failed with an empty
  `NotImplementedError`. The operations are functions defined as numpy's
  loops define them (`loopty.operations`), which loopty's targets call
  wherever `//`, `%`, `<<` or `>>` is, a guard included; index arithmetic by
  a positive constant stays loopy's. Target `c-source` is
  `lower.SourceCTarget`. The definitions, the plan of conversions and the
  lowering are hashed into loopty's targets (`operations.CODE_DIGEST`), so
  that loopy's code cache, whose key does not tell a literal `3` from
  `np.uint64(3)`, serves no code another version of them generated. A loop
  variable divided by a literal zero, or shifted by a negative literal, is a
  `TraceError`, since Python refuses it natively, and so is `//` or `%` of a
  complex value, which numpy refuses, and a loop variable shifted left by a
  literal of 64 or more, which Python shifts exactly and the compiled run to
  `0` (`i * 2.0 ** 64` is named). Note 20 in `docs/loopy-notes.md`.
- A comparison of an `^`, or of another comparison, is printed with C's
  precedence: loopy printed by Python's, in which `^` binds more tightly than
  a comparison and C's does not, so `k[i] < (k[i] ^ 1)` was
  `k[i] < k[i] ^ 1`, true at every `k` compiled, and `x[i] < (k[i] == 1)`
  was `(x[i] < k[i]) == 1` (`lower._CText`).
- A comparison of an `^`, a `<<` or a `>>` is traced as one, now that lanky
  builds its own terms for the three (`lanky.terms.BitwiseXor`, `LeftShift`
  and `RightShift`), whose comparisons are propositions: `when((k[i] ^ 1) ==
  0)` was traced as `when(False)` and `(k[i] << 1) != 4` as `True`, since
  pymbolic's `==` of its own nodes compares them as expressions, and
  `(i << 2) > 5` raised a `TypeError`.
- A sum of truth values is a `TraceError` naming the fix (#106): numpy's `+`
  of two bools is `or`, so `(b[i] + b[i]) * 1.0` was `1.0` natively and
  `2.0` compiled, with the `trace-faithful` fact `tested`. The fix named is
  `b[i] | c[i]` for `or`, or `1 * b[i] + c[i]` for a count. A difference of
  truth values, which numpy refuses, is one too, naming `b[i] ^ c[i]` or
  `1 * b[i] - c[i]`.
- A kernel named like a function loopy or the C library knows is renamed in
  the generated code with the `_knl` suffix, as a C keyword is (#108):
  `floor` failed inside loopy with `KeyError: 'floor'`, and `cpow` over
  complex arrays with a power in gcc. The names are every function loopy
  resolves on the C and OpenCL targets, what `math.h` and `complex.h`
  declare with their `f` and `l` forms, what they and `stdint.h` define as a
  macro or a type, and loopy's and loopty's helpers (`lower.is_library_name`).
- An integer element or scalar to a negative integer power is a `TraceError`
  naming `1 / k[i] ** 2`, a real (#109): numpy refuses it at every point, and
  the compiled run stored `1`, `-1` or `0`. A loop variable to one is a
  Python int's power natively, a real, and is computed so compiled. `^`, `<<`
  and `>>` of a real are a `TraceError` too, since numpy refuses them and C
  cannot compile them.
- A guard on loop variables that numpy computes in another type compiles
  (`with when(i ** 0.5 > 1.5)`): loopy reads a guard naming no array into
  isl, and its reader failed on the cast the operation planning of #82 put
  there. Such a guard is lowered with no cast in it: an operand numpy
  computes in double or in 64 bits is multiplied by `1.0` or by a 64-bit `1`
  instead, so `when(s * a > t)` of a `float32` `a` is double precision, and
  `when(i * i < m)` does not wrap round at `i = 46341`. An operand numpy
  rounds to single precision there, `(i + 1) ** -1` beside a `float32`
  scalar, has no such product, and the guard is refused, naming `Real` for
  the scalar.
- A helper's name, `loopty_mod_int64` or `loopy_pow_int64_int32`, is refused
  for a parameter, a size or a loop variable, as a C keyword is: a parameter
  of that name shadowed the helper the kernel calls, and the C did not
  compile.
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
- A schedule of an access indexed by a size, `x[x.dom.size - 1]`, is built
  (#110): the map of the pairs of instances that touch one cell named the
  size without declaring it, and isl read the text as a syntax error naming
  neither the kernel nor the access. A write of the last offset beside a
  ragged read failed the same way. The map declares the sizes, and its
  dimensions have primed names, which no size can have: isl reads a
  dimension named like a declared size as the size, and a kernel over
  `Fin[s]` lost every dependence that map gives.
- A ragged type nothing builds is refused where it is traced (#112): a fiber
  after two dense axes, a dense axis after the fiber, or counts of two axes.
  `Arr.ragged` builds none of them and lowering indexes none, but tracing and
  the typing rules accepted them, and the ledger stated facts about a row's
  start read through the inner index alone.
- `loopty run` compares a schedule of a schedule with its kernel (#98):
  `Schedule(Schedule(double)).split("i", 2)` was only executed, since its
  source is a schedule and not callable, and recorded no agreement, and
  `double` ran through the identity schedule as if nothing scheduled it. The
  nested run's agreement is kept under `#2` as a second run of one schedule.
- A written bound is read where each loop and sum starts whatever a guard or
  a clause makes of the domain (#111). isl states `for j / for k over one
  row` with `when(k == j)` as `k = j, 0 <= j < nl_cnt_r`, with no bound of
  `k` left, so the interpreter gave the loop over `k` no reading of its own,
  ran it past its row's new length, and refuted the faithful trace by an
  `IndexError` where `when(k <= j)` was tested. The readings are told apart
  over the statement's `loop_domain`, and over a sum's new `loop_domain`
  (`loopty.term.Reduction`), its domain before the statement's guard and its
  clause narrowed it, and each reading is read where its own loop starts.
  Lowering computes a row length once per row and refuses a rewrite that a
  loop inside the row would read stale, but counted only the first loop the
  length bounds: `cnt[r] = 1` between the loop over `j` and the loop over
  `k` inside it was lowered, and the compiled run disagreed with the native
  one. The innermost loop counts now, over the nest before a guard narrowed
  it, and `loopty.flow` states a sum's bound reads over its `loop_domain`, so
  a tag across a read the clause hid is refused.
- A checked point reads a value written outside 32 bits as written (#128).
  The compiled program stored `Fin` cells in 32 bits, so `perm[i] = 2**32`
  was `0` after the store, a point of `Fin(n)`, where the native contract of
  the next call refuses `2**32`; the program ran the call. An integral array
  a checked point reads (`Term.checked_arrays`) is stored in 64 bits, in the
  lowered program (`loopty.contract.array_storage`) and in the types
  `loopty.promotion` gives its elements: a `Fin[m]` one is widened, and one
  of `Nat` or `Int` is stored so already (#101). The native run computes in the
  array it is given, and one of 32 bits narrows what is written into it
  where the compiled program does not, so the compiled run refuses such an
  array passed in fewer than 64 bits, `Fin[m]`'s `int32` included, or the
  parameter a temporary of it is made like
  (`loopty.contract.checked_storage`), and composing refuses a temporary of
  it given a narrower `dtype`.

### Changed

- loopy refuses to preprocess or generate code for a kernel on one of
  loopty's targets outside `loopty.isl_reading.declining()`, with
  `isl_reading.ReadingsInactive`. Such a kernel is written for loopty's
  readings of its subscripts and guards (#129, #137), and loopy's code
  cache, whose key does not say which readings made an entry, would serve
  code generated with loopy's own to a run of loopty's, past the bounds
  check that refuses it. `LoopyExecutor` and `emit_code` enter the context;
  a caller who hands `schedule.kernel` to `lp.generate_code_v2` or to its
  `executor()` itself does it inside `with declining():`. Code loopy finds in
  its cache was generated inside, and is served as before.

- **Packaged for the first release.** The version is 0.1.0, the classifiers
  say `Development Status :: 3 - Alpha`, and the project's URLs name the
  quickstart, this file and the issue tracker. The dependencies are what an
  install from PyPI has to get: lanky 0.1.0 or later (the floor was
  0.1.0.dev1, met only by a checkout of lanky), numpy 2 or later, for NEP 50's
  promotion, which the compiled run follows, loopy 2025.2, the one release
  loopty is tested with and whose internals it builds on, and islpy from
  2025.2.5 and below 2026, as before; the `opencl` extra asks for the
  pyopencl loopy's own target asks for. `[tool.uv.sources]` still resolves
  lanky from a sibling checkout for development, and a built distribution
  carries the requirement on PyPI instead. The sdist holds the package, the
  tests, the examples, the documentation and the script that checks its
  transcripts, named in `pyproject.toml` rather than every file git does not
  ignore, and hatchling 1.27 or later builds it, for the license metadata.
  The publish workflow stops on a tag that is not the package's version in
  `pyproject.toml` and `loopty.__version__`, builds without
  `[tool.uv.sources]`, checks that it built one sdist and one wheel of that
  version, and runs `twine check` on them. It builds in a job that can only
  read the repository, and uploads from a second job that alone may ask for
  the token PyPI trusts and runs no checkout or build. The README
  installs from PyPI with `pip`, says what the install pulls in and needs,
  where the project stands, has a Known limits section, and links the
  documentation by full addresses, so that the links work on PyPI as well.
- **`Nat` and `Int` are 64 bits wide compiled** (#101). An array of them the
  kernel writes, or a program's temporary of them, has to be an `int64` one
  natively, where a signed integer of 32 bits or more was accepted: an
  `int32` one wraps round where the compiled one does not. `Fin[m]` keeps the
  old rule. The C code of a kernel over them declares `int64_t`.
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
