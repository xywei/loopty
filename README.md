# loopty

loopy, with types.

A loopty kernel is a Python function. Run it under plain `python` and it computes
on numpy. Trace it and you get a typed term whose obligations go into a ledger.
Transform it and every step is checked before it is applied.

```python
@kernel
def spmv(
    cnt: Arr[Fin[n], Nat],
    col: Arr[Fin[n], Fin[cnt], Fin[m]],
    val: Arr[Fin[n], Fin[cnt], Real],
    x: Arr[Fin[m], Real],
    y: Arr[Fin[n], Real],
):
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])
```

`reduce_sum` is Loopty's reduction frontend: it binds `j` over the symbolic
fiber and traces the operation as an ISL-backed `loopty.term.Reduction`.

`val: Arr[Fin[n], Fin[cnt], Real]` is a ragged shape: for each of the `n` rows,
`cnt[r]` entries. The second axis names the counts array, which is what makes it
a dependent sum rather than a rectangle, and `val.dom[r]` is the fiber over that
row. Nothing in the kernel mentions offsets or flat storage; that is layout, and
layout is the compiler's business.

Three commands, all of them working today:

```console
$ python examples/spmv.py            # runs natively on numpy; the theorem is tested
$ lanky check examples/spmv.py       # the ledger: every obligation and who decided it
$ loopty run examples/spmv.py        # lowers through loopy, compiles, runs, compares
```

and the ledger `lanky check` prints, abridged to the rows discussed here (each
row verbatim, and checked by `scripts/refresh_example_outputs.py`):

```console
$ lanky check examples/spmv.py
STATUS                            BY             WHERE        OWNER          STATEMENT
--------------------------------  -------------  -----------  -------------  ------------------------------------------------------------------------
decided                           isl            spmv.py:79   scan           off[0] is in bounds for every instance of S0
decided                           isl            spmv.py:81   scan           off[r + 1] is in bounds for every instance of S1
decided                           isl            spmv.py:79   scan           distinct instances of S0 write distinct cells of off
assumed                           -              spmv.py:69   scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
tested                            property-test  spmv.py:84   scan_monotone  n : Nat, cnt : Fn[Fin(n), Nat], off : Fn[Fin(n + 1), Nat] | off(0) ==...
decided                           isl            spmv.py:112  spmv           y[r] is in bounds for every instance of S0
decided                           type           spmv.py:112  spmv           x[col[r, j]] is in bounds by type (col[r, j] : Fin(m))
decided                           isl            spmv.py:112  spmv           distinct instances of S0 write distinct cells of y
decided                           type           spmv.py:112  spmv           the accumulation into y[r] over j is approx
tested                            interpreter    spmv.py:102  spmv           the traced term computes what the body computes
assumed under scan:postcondition  -              spmv.py:115  solve          after scan(...) in solve: off[0] == 0 and (forall r in Fin(n). off[r ...
...
19 facts: 2 assumed, 14 decided, 3 tested
```

Look at the `x[col[r, j]]` row, and at what decided it. That indirection is the
one obligation in a sparse product a polyhedral checker cannot settle on its own,
and here it was settled by the *type*: the entries of `col` are points of
`Fin[m]` and `x` has `m` cells, so the shape of the data discharges it and isl
is never called.

And look at the `interpreter` row, the one fact that is about the trace rather
than about the term. Every other row is a claim about what tracing recorded; this
one checks that the record is the body. The term is run by an interpreter of
its own, with numpy semantics, and compared with the body run natively, on the
file's example inputs and on inputs drawn from the declared types. A body that
kept state where tracing does not look would be refuted here, with the input
and the first cell that differs.

The last row belongs to `solve`, the `@program` that runs `scan` and then
`spmv`. It restates `scan`'s postcondition in the program's scope, and rests on
`scan`'s own postcondition fact, which the row names:
`assumed under scan:postcondition`. Nothing has established that fact yet, and
the restatement is worth no more than it.

And a transformation is a cast, checked before it is applied:

```console
$ python examples/stencil_skew.py
...
Schedule(jacobi).tile('t', 'i', 8, 8) ->
  IllegalCast: tile(t,i,8,8) illegal: instance S0[t=0, i=8] writes u[1, 8] read by S0[t=1, i=7] scheduled earlier (at nt=16, nx=16, as hinted)
...
accepted: Schedule(jacobi, target='c').skew(i, by='t').tile(t,i,8,8)
  loop nest: t_outer i_outer t_inner i_inner
  decided  isl  skew(i, by='t') renames the instances of jacobi one for one
  decided  isl  the order after skew(i, by='t') runs every dependence of jacobi forward
  decided  isl  tile(t,i,8,8) renames the instances of jacobi one for one
  decided  isl  the order after tile(t,i,8,8) runs every dependence of jacobi forward
...
```

The rejection names two real instances of the kernel, not an empty set or a
failed pattern match. It is the pair a person would find by hand, at sizes the
message states, because which violating pair isl picks depends on them.

## What nothing else does

- **The ragged shape is a type, and it is the *same* type isl reasons about.** A
  CSR matrix is a dependent sum, `Arr[Fin[n], Fin[cnt], Real]`. In-bounds and
  disjointness come from the shape, so a fiber loop is checked rather than
  trusted. Systems that model sparsity as offset arithmetic have to prove what
  this states.
- **An indirection can be in bounds by type.** `col: Arr[..., Fin[m]]` makes
  `x[col[r, j]]` sound with no proof obligation at all. This is where the
  polyhedral model normally gives up and inserts a runtime check or a
  "trust me". The type is what discharges it, so the executor is what enforces
  the type: every argument is checked against its declared element sort on the
  way in, and a `col` entry of `-1` or of `m` is a `ValueError` naming the cell
  rather than an address outside `x`.
- **Transformations are casts with witnesses.** Every `split`, `tile`,
  `interchange`, `skew` and `realize` states its reindexing as an isl map, is
  checked for bijectivity on statement instances, and is checked for monotonicity
  on the dependence relation. A failure prints two instances and the array cell
  between them, before any code is generated.
- **Reassociation is visible in the type.** Splitting an accumulation and summing
  the pieces is not free on floating point. A trace reads the accumulation's
  class off what it sums, `realize("y", tree=True)` is what lowers it to
  `reassoc`, and the differential comparison against the native run widens its
  tolerance because of that fact and not because someone chose a number. The
  tolerance is per element and local: `exact` is bitwise, and `reassoc` and
  `approx` ask that `|got - want| <= eps_class * (|want| + 1)` at every cell, so
  it is the accuracy claimed for that cell and does not grow with the size of
  the output. Over an `exact` accumulation the same cast is refused, and a
  kernel with an `exact` output is compiled with floating-point contraction
  off, so a fused multiply-add cannot change its last bit.
- **Legal and buildable are different questions, and both are answered.** A
  transformation can preserve the meaning of a program and still be one the
  backend cannot generate. Every accepted step is asked whether the target can
  build it, and a failure is a `refuted` fact of kind `buildable` decided by
  `loopy-target`, with the limit in words as its reason, rather than a
  `LoopyError` thrown from inside code generation several steps later.
- **The reference implementation is the kernel.** The same body runs on numpy
  under plain `python` and traces to the term loopy compiles, so the differential
  test compares a program with itself rather than with a second implementation.
  That the trace *is* the body is checked too, not assumed: every kernel's
  ledger has a `trace-faithful` fact, the traced term interpreted and compared
  with the native run, bit for bit when the output is `exact`.
- **It plugs into a proof host.** loopty registers a theory, an isl oracle, an
  executor and a `run` verb with [lanky](https://github.com/xywei/lanky), so a
  residual obligation an oracle cannot decide is an ordinary theorem a person can
  prove, in the same ledger.

## Status

This is `0.1.0.dev0`, a development release. The spmv and stencil stories work
end to end; the edges are sharp.

**Works.**

- `@kernel` and `@program`: inert, registering, running natively on numpy.
- Tracing a body to a typed term: accesses, statements, reductions, ragged
  fibers, `when` guards, source locations, and a `TraceError` that names the fix
  when a Python `if` is used on a computed value, when a Python name, a
  global, or a list, dict or set carries state from one loop iteration to the
  next, when an array is used whole (`y[:] = ...`, `x * 2`, a numpy function of
  it), when a reduction's `if` clause is not a bound isl can state, when the
  trace changes Python state outside the arrays (a global, a closure variable,
  an object's attribute or slot, a list, dict, set or numpy array they hold, a
  ragged array's offsets, or the same in a helper the body calls), and when the
  body prints, reads input, opens a file or draws a random number. The
  kernel's own module and package count as its code wherever they are
  installed, site-packages included.
- `when` guards narrow a statement's isl domain where isl can state them: an
  affine comparison of loop variables, sizes and scalars of an integral sort.
  A guard that reads an array, compares with `!=`, or compares with a `Real`
  scalar is evaluated at run time only, the statement lists it in
  `Stmt.unnarrowed`, and the facts stated over its domain say so in their
  provenance. A guard has to be a truth value, and one whose value is an
  integer is a `TraceError`, on a native run and under tracing alike. Natively
  that is what `~(i > 0)` is (`~` on a Python bool is bitwise), while the trace
  records `not (i > 0)`, so such a kernel traces and its `trace-faithful` fact
  is refuted by the native refusal, which names the fix.
- The faithfulness fact. For each kernel, the traced term is run by an
  interpreter (`loopty.interpret`: statement by statement in source order over
  each statement's isl domain, expressions evaluated with numpy's arithmetic,
  reductions summed in the order `reduce_sum` sums natively) and compared with
  the native run, on the file's `example_inputs()` and on three inputs drawn
  from the declared types. It is a `trace-faithful` fact, `tested` on
  agreement and `refuted` with the input and the first differing cell, which
  is where state hidden past every check above shows up.
- Typing rules and the ledger: in-bounds by isl or by type, write disjointness,
  ordering, reduction exactness, postconditions. A ragged access's reads of the
  offsets it is flattened through, when the kernel declares them, are accesses
  like any other: in-bounds obligations, and dependences every cast is checked
  against.
- `IslOracle`: `Empty`, `Subset`, `Bijective`, `Monotone`, each refutation with a
  witness.
- `Schedule`: `tag`, `split`, `interchange`, `prioritize`, `tile`, `skew`,
  `realize`, each checked as a cast, each emitting its fact; `retarget`, which
  replays every step against another loopy target and re-checks it.
- The target-capability check: a parallel tag inside a data-dependent (ragged)
  loop bound, a hardware axis on a reduction nested in another, or a reduction
  split across parallel and sequential inames, is reported as a `refuted`
  `buildable` fact and raises `UnbuildableSchedule` when something asks for
  code.
- Lowering to loopy, including a ragged axis as a flat buffer plus offsets, and
  running on `lp.ExecutableCTarget`. Every argument of `LoopyExecutor.run` is
  an argument of the kernel; the target is chosen by the schedule
  (`Schedule(kernel, target="opencl")`) or by the executor
  (`LoopyExecutor(target="opencl")`).
- The argument contract, enforced at every entry point that runs a kernel
  (compiled, differential and native): two distinct array parameters may not
  share storage, a ragged argument has to agree with the counts array its type
  names and with any offsets passed alongside it, and a value of a refined sort
  such as `Fin[m]` has to be one — an array element and a scalar argument
  alike, and being one means being a finite whole number in range, not merely
  passing two comparisons. These are the assumptions the typing
  rules make about a *call* rather than about the term, and a violation is a
  `ValueError` naming the argument. Distinct parameters being disjoint storage
  is the load-bearing one: dependences are computed per array name, so a kernel
  reading `x[i - 1]` and writing `y[i]` may legally run `i` in parallel, and
  the same kernel called with `x is y` is a race that the differential test
  cannot see, because it copies each argument separately.
- `loopty run FILE [--target c|opencl] [--emit-code] [--json OUT]`,
  `loopty check FILE`, and `lanky run FILE` through the entry point.
  `--target` retargets every schedule in the file, re-checking its casts, and
  says so by name when one cannot be retargeted; without it each schedule keeps
  the target it was written for. A refuted fact is repeated under the ledger
  with what explains it, the block `lanky check` prints (a compiled run that
  disagrees names the outputs and by how much), and the command exits 1.

**Partial.**

- Ragged bounds are reflected into isl as one parameter per distinct bound term
  (allocated once per kernel, so the same `cnt[r]` is one parameter everywhere
  and two different bounds are never given one name), so `cnt[r]` and
  `cnt[r + 1]` are unrelated to isl. Nothing knows that counts are
  non-negative, that they sum to the offsets, or that `off` is monotone, so the
  scan's recurrence is not usable by the decision procedure. The visible
  consequence: an access against flat storage, `val[off[r] + j]`, is reported
  **`assumed`** with the reason in its provenance, never `decided`. The ragged
  spelling `val[r, j]` over `0 <= j < cnt[r]`, which is what the tracer and the
  demos produce, *is* decided. The monotone-offsets formulation is the
  documented next step; see the module docstring of `loopty/flow.py`.
- `@program` restates a callee's postcondition as a fact in scope, which rests
  on the callee's own fact (lanky's `rests_on`, so the row reads
  `assumed under scan:postcondition`), but no rule consumes postconditions as
  hypotheses yet, so "facts travel" is bookkeeping. A callee imported from
  another file has its fact in that file's ledger, so `lanky check` counts the
  id as an assumption and names it in an `UNRESOLVED` line.
- Only a two-axis (row, fiber) ragged array lowers. A deeper dependent sum
  raises.
- A reduction nested in another one cannot take its bound from the outer
  binder when that bound is not affine:
  `reduce_sum(reduce_sum(val[q, j] for j in val.dom[q]) for q in val.dom)` is
  decided by the analysis and refused by the lowering with a `LoweringError`,
  because a row length is computed inside the loop over its row and a reduction
  binder has no such loop. Writing the outer reduction as a loop that
  accumulates into the output, or keeping each row's sum in a cell indexed by
  the row, lowers and runs. An affine inner bound (`Fin[i + 1]`) lowers as it
  is.
- An index expression isl cannot express widens the footprint to the whole
  array. That is sound, but it can reject a legal schedule.
- `realize(var, tree=True)` checks and marks the reassociation; the reduction
  tree itself comes from splitting and tagging the reduction iname, which is
  checked separately and not verified on the C target.
- The accumulation convention: a traced `y[r] += ...` under a parallel iname is
  reported as a disjointness refutation, which is the conservative reading. The
  `reassoc` fact is what should license it and nothing consumes that yet.
- The target-capability check knows two limits of loopy 2025.2 and no others, so
  it is a list rather than a model of what the backend can do. A schedule it
  passes can still fail in code generation for a reason nobody has met yet.
- A kernel called with a zero-length *shape-bearing* argument (a matrix with no
  rows at all) cannot run on the C target: loopy cannot pass an empty array, and
  the workaround that rescues the empty flat buffer of a ragged axis cannot be
  applied to an argument loopy reads a size from. A matrix whose rows are all
  empty does run. See `docs/loopy-notes.md`.
- Array *shapes* are not checked at the executor boundary, though element types,
  ragged layouts and aliasing now are. Lowering has to declare some arrays with
  `shape=None` (see `docs/loopy-notes.md`), so a wrongly sized array is
  undefined behaviour rather than an error.
- A reduction binder keeps its name in the generated kernel only where nothing
  else has it. loopy gives an iname one domain, a reduction cannot carry a
  predicate, and two statements cannot share a reduction's loop, so a reduction
  whose binder another statement already uses (as a loop variable or a binder),
  or that its own statement binds over another domain, is lowered under a
  fresh iname (`j_0`), and the binders nested in it follow. The name in the
  term is unchanged, and so is every message, but a `Schedule` step names such
  a reduction by its iname in the kernel (`split("j_0", 2)`), which
  `Lowering.reduction_inames` lists.
- The trace-time refusals of hidden state look one level below a name: into
  the containers and the objects it holds (their `__dict__` and their slots),
  the buffers of the arrays it holds (both of a ragged one), and the containers
  those objects hold. `acc[0][0] += 1`, `holder.inner.s = ...`, a `deque`, a
  loop over a generator that wraps a domain, and a `dir()` probe trace without
  an error, and the `trace-faithful` fact is what refutes them. That fact is a
  test, not a proof: it compares the runs on the inputs it tries, so hidden
  state no such input exercises goes unseen. It stays `assumed`, with the
  reason, when no input runs natively, when the term calls a function the
  interpreter has no numpy counterpart for, or when a loop bound reads an array
  the same kernel writes.

**Not yet.**

- The OpenCL target is never executed on a development machine: nothing here
  imports pyopencl, and the test suite asserts it. Device runs happen on real
  hardware elsewhere and are reported under
  [docs/device-runs/](docs/device-runs/).
- CUDA, and targets beyond C and OpenCL.
- The two-level FMM point-to-point shape, which is a dependent sum of dependent
  sums and needs more than the two ragged axes lowering handles.
- Anything on PyPI above the 0.0.1 placeholder.

## Install

```sh
uv add loopty
```

```sh
pip install loopty
```

loopty pulls [lanky](https://github.com/xywei/lanky), numpy, islpy, loopy and
pymbolic. Device execution is an extra, and it is never installed on a laptop:

```sh
uv add "loopty[opencl]"
```

lanky's Lean oracle is an extra of *lanky*, so a ledger that says `proved lean`
needs `uv add "lanky[lean]"` in the same environment. Without it the same
theorem reads `tested property-test`, and every other row is unchanged.

For work on loopty itself, lanky is resolved from a sibling checkout:

```sh
git clone https://github.com/xywei/lanky.git     # next to this one
uv sync --group dev
uv run pytest -q
uv run ruff check .
```

A kernel file needs `from __future__ import annotations` and a ruff `F821`
per-file ignore, because a size such as `n` in `Arr[Fin[n], Real]` is a symbolic
variable lanky invents while evaluating the annotation. The same scope invents
`float` and `int`, so a sort is written `Real`, `Nat`, `Int`, `Fin[n]` or a
numpy type such as `np.float64`; a sort that is a free name is refused when the
kernel is traced.

## Architecture

**Types are isl objects.** A statement's type is its iteration domain, an isl
set; its effects are its read, write and accumulation footprints, isl maps.
Dependences are derived from the footprints, not declared. Every statement
instance is a point of one padded, statement-tagged space, so schedules and
dependences are plain maps and the checker is a handful of isl calls.

**Index types carry shape.** `Fin[n]` is an index type, `Fin[a*b]` normalizes to
`Fin[a] x Fin[b]`, and a ragged axis is a dependent sum whose bound is another
array's entry. `Layout` and `RaggedLayout` are the maps from index space to
storage.

**Evaluate annotations, trace bodies.** No Python parser and no AST pass.
Annotations are evaluated with lanky's scope, and the body is run once against
symbolic arrays. That is why the native run and the compiled run are the same
program. Tracing is the only thing that gives a body its meaning, and whether
the trace kept that meaning is a checked fact rather than an assumption: the
term is interpreted on its own and compared with the native run.

**Transformations are casts.** Each states a reindexing map, which isl checks for
bijectivity, and a new execution order, which isl checks for monotonicity on the
dependence relation. Failure is an `IllegalCast` carrying the witness and the
refuted fact, whose reason is the exception's message. Casts that change
floating-point semantics mark the result's exactness class instead of being
refused.

**loopty is a lanky plugin.** It registers a *theory* (`KernelTheory`, which
turns a kernel into facts), an *oracle* (`IslOracle`, trust class
`decision-procedure`), an *executor* (`LoopyExecutor`, trust class `test`) and a
*verb* (`run`), through the `lanky.theories`, `lanky.oracles`, `lanky.executors`
and `lanky.verbs` entry-point groups. lanky never imports loopty; it finds these
and asks each what it can do. Obligations loopty cannot decide stay `ASSUMED` in
the ledger, or become theorems a person proves.

## Name

loopty is loop + ty, for types: loops, typed. It follows `loopy`, `sumpy`, and
`pytato` in the naming tradition of the loopy ecosystem.

## Documentation

- [docs/quickstart.md](docs/quickstart.md): the two demos end to end, with the
  output the commands actually print.
- [examples/README.md](examples/README.md): all five demos, with every console
  block regenerated by `scripts/refresh_example_outputs.py`.
- [docs/device-runs.md](docs/device-runs.md) and
  [docs/device-runs/](docs/device-runs/): the demos run on real OpenCL devices,
  with the commands and the transcripts.
- [docs/loopy-notes.md](docs/loopy-notes.md): the loopy and islpy interactions
  that cost debugging time, each with its local workaround and why it is local.
- [CHANGELOG.md](CHANGELOG.md).
- [lanky](https://github.com/xywei/lanky): the proof host loopty plugs into, and
  where the ledger, the statuses and the oracle protocol are defined.

## AI disclosure

This project is developed with substantial assistance from AI coding agents
(Anthropic's Claude, via Claude Code). Design, direction, and review are by
Xiaoyu Wei. Generated text and code are reviewed before release, but readers
should assume AI involvement throughout.

## License

MIT. See [LICENSE](LICENSE).
