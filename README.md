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

and the ledger `lanky check` prints (abridged: nine of its seventeen rows,
each row verbatim):

```text
STATUS   BY             WHERE        OWNER          STATEMENT
-------  -------------  -----------  -------------  ---------------------------------------------
decided  isl            spmv.py:80   scan           off[0] is in bounds for every instance of S0
decided  isl            spmv.py:82   scan           off[r + 1] is in bounds for every instance of S1
decided  isl            spmv.py:80   scan           distinct instances of S0 write distinct cells of off
assumed  -              spmv.py:70   scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
tested   property-test  spmv.py:85   scan_monotone  n : Nat, cnt : Fn[Fin(n), Nat], off : Fn[Fin(n + 1), Nat] | off(0) ==...
decided  isl            spmv.py:113  spmv           y[r] is in bounds for every instance of S0
decided  type           spmv.py:113  spmv           x[col[r, j]] is in bounds by type (col[r, j] : Fin(m))
decided  isl            spmv.py:113  spmv           distinct instances of S0 write distinct cells of y
decided  type           spmv.py:113  spmv           the accumulation into y[r] over j is approx
...
17 facts: 2 assumed, 14 decided, 1 tested
```

Look at the `x[col[r, j]]` row, and at what decided it. That indirection is the
one obligation in a sparse product a polyhedral checker cannot settle on its own,
and here it was settled by the *type*: the entries of `col` are points of
`Fin[m]` and `x` has `m` cells, so the shape of the data discharges it and isl
is never called.

And a transformation is a cast, checked before it is applied:

```console
$ python examples/stencil_skew.py
Schedule(jacobi).tile('t', 'i', 8, 8) ->
  IllegalCast: tile(t,i,8,8) illegal: instance S0[t=0, i=8] writes u[1, 8] read by S0[t=1, i=7] scheduled earlier (at nt=16, nx=16, as hinted)

accepted: Schedule(jacobi, target='c').skew(i, by='t').tile(t,i,8,8)
  loop nest: t_outer i_outer t_inner i_inner
  decided  isl  skew(i, by='t') renames the instances of jacobi one for one
  decided  isl  the order after skew(i, by='t') runs every dependence of jacobi forward
  decided  isl  tile(t,i,8,8) renames the instances of jacobi one for one
  decided  isl  the order after tile(t,i,8,8) runs every dependence of jacobi forward
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
  "trust me".
- **Transformations are casts with witnesses.** Every `split`, `tile`,
  `interchange`, `skew` and `realize` states its reindexing as an isl map, is
  checked for bijectivity on statement instances, and is checked for monotonicity
  on the dependence relation. A failure prints two instances and the array cell
  between them, before any code is generated.
- **Reassociation is visible in the type.** Splitting an accumulation and summing
  the pieces is not free on floating point. A trace reads the accumulation's
  class off what it sums, `realize("y", tree=True)` is what lowers it to
  `reassoc`, and the differential comparison against the native run widens its
  tolerance because of that fact and not because someone chose a number. Over an
  `exact` accumulation the same cast is refused.
- **Legal and buildable are different questions, and both are answered.** A
  transformation can preserve the meaning of a program and still be one the
  backend cannot generate. Every accepted step is asked whether the target can
  build it, and a failure is a `refuted` fact of kind `buildable` decided by
  `loopy-target`, with the limit in words, rather than a `LoopyError` thrown
  from inside code generation several steps later.
- **The reference implementation is the kernel.** The same body runs on numpy
  under plain `python` and traces to the term loopy compiles, so the differential
  test compares a program with itself rather than with a second implementation.
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
  when a Python `if` is used on a computed value.
- Typing rules and the ledger: in-bounds by isl or by type, write disjointness,
  ordering, reduction exactness, postconditions.
- `IslOracle`: `Empty`, `Subset`, `Bijective`, `Monotone`, each refutation with a
  witness.
- `Schedule`: `tag`, `split`, `interchange`, `prioritize`, `tile`, `skew`,
  `realize`, each checked as a cast, each emitting its fact; `retarget`, which
  replays every step against another loopy target and re-checks it.
- The target-capability check: a parallel tag inside a data-dependent (ragged)
  loop bound, or a reduction split across parallel and sequential inames, is
  reported as a `refuted` `buildable` fact and raises `UnbuildableSchedule` when
  something asks for code.
- Lowering to loopy, including a ragged axis as a flat buffer plus offsets, and
  running on `lp.ExecutableCTarget`.
- `loopty run FILE [--target c|opencl] [--emit-code] [--json OUT]`,
  `loopty check FILE`, and `lanky run FILE` through the entry point.
  `--target` retargets every schedule in the file, re-checking its casts, and
  says so by name when one cannot be retargeted; without it each schedule keeps
  the target it was written for.

**Partial.**

- Ragged bounds are reflected into isl as one parameter per occurrence, so
  `cnt[r]` and `cnt[r + 1]` are unrelated to isl. Nothing knows that counts are
  non-negative, that they sum to the offsets, or that `off` is monotone, so the
  scan's recurrence is not usable by the decision procedure. The visible
  consequence: an access against flat storage, `val[off[r] + j]`, is reported
  **`assumed`** with the reason in its provenance, never `decided`. The ragged
  spelling `val[r, j]` over `0 <= j < cnt[r]`, which is what the tracer and the
  demos produce, *is* decided. The monotone-offsets formulation is the
  documented next step; see the module docstring of `loopty/flow.py`.
- `@program` restates a callee's postcondition as a fact in scope, but no rule
  consumes postconditions as hypotheses yet, so "facts travel" is bookkeeping.
- Only a two-axis (row, fiber) ragged array lowers. A deeper dependent sum
  raises.
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
- Array *shapes* are not checked at the executor boundary. Lowering has to
  declare some arrays with `shape=None` (see `docs/loopy-notes.md`), so a
  wrongly sized array is undefined behaviour rather than an error.

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
variable lanky invents while evaluating the annotation.

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
program.

**Transformations are casts.** Each states a reindexing map, which isl checks for
bijectivity, and a new execution order, which isl checks for monotonicity on the
dependence relation. Failure is an `IllegalCast` carrying the witness and the
refuted fact. Casts that change floating-point semantics mark the result's
exactness class instead of being refused.

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
- [examples/README.md](examples/README.md): all four demos, with every console
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
