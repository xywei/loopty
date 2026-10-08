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
STATUS   EFFECTIVE  BY             WHERE        OWNER          STATEMENT
-------  ---------  -------------  -----------  -------------  ------------------------------------------------------------------------
decided  decided    isl            spmv.py:79   scan           off[0] is in bounds for every instance of S0
decided  decided    isl            spmv.py:81   scan           off[r + 1] is in bounds for every instance of S1
decided  decided    isl            spmv.py:79   scan           distinct instances of S0 write distinct cells of off
tested   tested     native         spmv.py:69   scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
tested   tested     property-test  spmv.py:84   scan_monotone  n : Nat, cnt : Fn[Fin(n), Nat], off : Fn[Fin(n + 1), Nat] | off(0) ==...
decided  decided    isl            spmv.py:112  spmv           y[r] is in bounds for every instance of S0
decided  decided    type           spmv.py:112  spmv           x[col[r, j]] is in bounds by type (col[r, j] : Fin(m))
decided  decided    isl            spmv.py:112  spmv           distinct instances of S0 write distinct cells of y
decided  decided    type           spmv.py:112  spmv           the accumulation into y[r] over j is approx
tested   tested     interpreter    spmv.py:102  spmv           the traced term computes what the body computes
decided  tested     call           spmv.py:115  solve          after scan(...) in solve: off[0] == 0 and (forall r in Fin(n). off[r ...
...
21 facts: 16 decided, 5 tested
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

`scan`'s postcondition is `tested` by `native`: it is evaluated at what every
native run of `scan` left in its arguments, on the file's example inputs and on
the drawn ones, and held after each. The last row belongs to `solve`, the
`@program` that runs `scan` and then `spmv`. It restates `scan`'s postcondition
in the program's scope, `decided` by the call, and rests on `scan`'s own
postcondition fact, `postcondition:spmv.scan@69`, so it is worth what that fact
is worth: the `EFFECTIVE` column reads `tested`. The id is the kind, then
`scan`'s definition, the module the file's path gives it, its name and its
line, so a `scan` imported from another file, or defined twice, is never taken
for this one. `loopty run` compiles `solve` too, as one kernel: its term is the
two kernels' statements in call order, and the compiled program is compared
with the program run natively.

A postcondition is also what a later call may assume. A kernel's requirements
on its inputs are its argument types, and where an earlier call of a program
wrote the array, the requirement is decided under what held when the call was
made, or checked by the compiled program between the two calls;
`examples/travel.py` shows both.

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

And an array's index set need not be a box. One value per pair of particles is
an array over the lower triangle, and the triangle is its type:

```python
@kernel
def pairs(
    x: Arr[Fin[n], Real],
    y: Arr[Fin[n], Real],
    q: Arr[Fin[n], Real],
    f: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real],
    e: Arr[Fin[n], Real],
):
    for i in f.dom:
        for j in f.dom[i]:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            f[i, j] = q[i] * q[j] / (1.0 + dx * dx + dy * dy)
    for p in e.dom:
        e[p] = reduce_sum(f[p, j] for j in f.dom[p]) + reduce_sum(
            f[k, p] for k in e.dom if k > p
        )
```

`Where` takes binders, written as slices, and then the constraints that cut
their box; `Sigma[i: Fin[n], Fin[i + 1]]` is a sum with affine fibers, and
`Fin[n] + Fin[m]` a union of pieces. Every in-bounds obligation is decided over
the exact set:

```console
$ lanky check examples/pairs.py
STATUS   BY           WHERE        OWNER  STATEMENT
-------  -----------  -----------  -----  ------------------------------------------------------
decided  isl          pairs.py:78  pairs  f[i, j] is in bounds for every instance of S0
...
decided  isl          pairs.py:80  pairs  f[p, j] is in bounds for every instance of S1
decided  isl          pairs.py:80  pairs  f[k, p] is in bounds for every instance of S1
...
16 facts: 15 decided, 1 tested
```

`f[k, p]` is read under `k > p`, which makes `(k, p)` a point of the triangle,
while `f[p, p]` would be refused although the `n x n` box around the triangle
has the cell. How `f` is stored is a separate choice that changes no fact:
`Schedule(pairs)` keeps it in that box, and `Schedule(pairs).pack("f")` keeps
its cells and no others, row after row, read as `f[off_f[i] + j]` through a
table of row starts. Both compiled runs agree with the native one.

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
  `interchange`, `skew` and `realize` states its reindexing as an isl map, and
  `affine` takes the map from you, any injective affine one, the diamond
  `(t, i) -> (t + i, t - i)` included, or a map per statement. Each is checked
  for bijectivity on statement instances and for monotonicity on the dependence
  relation. A failure prints two instances and the array cell between them,
  before any code is generated.
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
  That the trace *is* the body is checked too, not assumed: every kernel and
  every program has a `trace-faithful` fact, the traced term interpreted and
  compared with the native run, bit for bit when the output is `exact`.
- **The index set of an argument is a type, and not a box.** An array over the
  lower triangle, a band, a sum with affine fibers or a union of pieces is
  written as that set, `Arr[Where[i: Fin[n], j: Fin[n], j < i], Real]`, and its
  in-bounds obligations are decided over the set itself, so a cell of the
  bounding box outside it is refused. Where the cells are kept is a layout,
  boxed or packed, chosen by a schedule step, and changes no fact.
- **It plugs into a proof host.** loopty registers a theory, an isl oracle, an
  executor and a `run` verb with [lanky](https://github.com/xywei/lanky), so a
  residual obligation an oracle cannot decide is an ordinary theorem a person can
  prove, in the same ledger.

## Status

This is `0.1.0.dev0`, a development release. The spmv and stencil stories work
end to end; the edges are sharp.

**Works.**

- `@kernel` and `@program`: inert, registering, running natively on numpy.
- A program's term, and one kernel for it. `Program.term` is what the body
  does run once against placeholders: every kernel call is recorded instead
  of run, and the callees' terms follow one another in call order, in the
  program's names. Sizes are unified through the arrays passed (`scale`'s `n`
  is `scan`'s `n + 1` when both are handed `off`), each call's loops get names
  of their own, and its statements are named after it (`scan.S1`,
  `step@2.S0`). An array the body makes with `Arr.zeros_like(u)` is a
  temporary of the term, zeroed where it was made, and a loopy temporary of
  the lowered kernel, so the intermediate between two kernels is nobody's
  argument. The edges between the calls are not declared: an array one call
  writes and a later one reads is one array of the term, so the dependence is
  in the footprints, and it orders the instructions of the one loopy kernel
  the program lowers to. `LoopyExecutor().run(program, ...)` runs it,
  `Schedule(program)` schedules it, and `loopty run` compares every program in
  a file with its native run, as it does a kernel. A body that does anything
  to an argument but pass it to a kernel, or make an array like it, is refused
  with a `TraceError` naming the fix.
- Facts that travel between a program's calls (`loopty.hypotheses`). A
  kernel's requirements on its inputs are its argument types, and three of
  them are about what an array's cells hold: an element of a `Fin[m]` sort is
  a point of it, one of the `Nat` sort is not negative, and the offsets a
  ragged family is read through are the ones its counts give. Where an
  earlier call wrote the array, the requirement is
  a `requirement` fact of the program, decided by isl under what held at the
  call: the earlier callees' postconditions that nothing has written over
  since, the zeros an `Arr.zeros_like` starts an array with, the types the
  program's contract checks of what nothing has written yet, and the
  theorems the program cites (`@program(uses=[scan_monotone])`),
  instantiated at the cells the requirement reads. The fact rests on the
  facts it used, so `gather`'s requirement after `number` reads
  `decided` and is worth `tested`, as `number`'s postcondition is. Where the
  hypotheses do not decide it, the compiled program checks the cells between
  the two calls and stops there, with the message of the native refusal; the
  requirement stays `assumed`, and its fact says why. The same hypotheses
  decide a callee's in-bounds fact its own term leaves `assumed`, a flat
  `val[off[r] + j]` after the scan, as a fact of the program's.
  `examples/travel.py` shows all three.
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
  is refuted by the native refusal, which names the fix. What is stored into
  an array of `Bool` has to be a truth value too, a comparison, a connective
  of truth values or a `Bool` read, since natively the array is a bool and
  compiled a byte, into which C converts `0.5` as `0`: `b[i] = u[i]` is a
  `TraceError` naming `b[i] = u[i] != 0`, and an integer arriving at a bool
  array natively, `~(i > 0)` again, is refused there as it is by `when`.
  Every operand of `&`, `|` and `~` has to be a truth value, wherever the
  connective is, the index of the cell written included, since the trace and
  the compiled kernel read them as `and`, `or` and `not` and natively they are
  bitwise on an integer: `(k[i] & 1) * x[i]` is a `TraceError` naming
  `k[i] != 0`, and `k % 2` for `k & 1`. And `~(i > 0)` used as a number
  (`x[i] * ~(i > 0)`, in a sum's body or an index) is a `TraceError` naming
  `i <= 0`, since natively it is the `-2` or `-1` the native run refuses only
  where a truth value is asked for.
- The faithfulness fact. For each kernel and each program, the traced term is
  run by an interpreter (`loopty.interpret`: statement by statement in source
  order over each statement's isl domain, each loop enumerated when the run
  reaches it, so that a ragged loop runs to the length its row has when the
  loop starts, expressions evaluated with numpy's arithmetic, reductions
  summed in the order `reduce_sum` sums natively) and compared with the
  native run, on the file's `example_inputs()` and on three inputs drawn from
  the declared types. It is a `trace-faithful` fact, `tested` on agreement and
  `refuted` with the input and the first differing cell, which is where state
  hidden past every check above shows up, and, for a program, a call its body
  makes or skips by looking at an argument as no placeholder can be looked at
  (`isinstance(x, Arr)`).
- Typing rules and the ledger: in-bounds by isl or by type, write disjointness,
  ordering, reduction exactness, postconditions. A write into an array whose
  element sort is `Fin[m]` owes the fact that the value written is a point of
  `Fin[m]` (`element-sort`), decided by isl for a quasi-affine value, by type
  for a value read from an array of that sort, and assumed otherwise, and an
  index read from such an array is in bounds by type resting on those facts,
  so `perm[i] = i + 1` followed by `x[perm[j]]` is refuted where it was
  decided. The reads a ragged layout
  makes are accesses like any other, in-bounds obligations and dependences
  every cast is checked against: the start of the row a ragged access is
  flattened through (`off[r]`, when the kernel declares the offsets), and the
  length of the row a loop over a ragged fiber runs to (`cnt[r]`, which the
  lowered code reads once per row). Each is listed where the lowered code
  reads it and nowhere else, so a kernel that writes its counts or its offsets
  is ordered against exactly the rows that use them.
- One meaning for a kernel that writes its own layout. The native run and the
  term interpreter read a ragged array through the counts and offsets the
  kernel declares, as the kernel has left them, which is what the lowered code
  does with the arguments it is handed; the contract checks that they are the
  array's own layout when the run starts.
- `IslOracle`: `Empty`, `Subset`, `Bijective`, `Monotone`, each refutation with a
  witness.
- `Schedule`: `tag`, `split`, `interchange`, `prioritize`, `tile`, `skew`,
  `affine`, `realize`, each checked as a cast, each emitting its fact;
  `retarget`, which replays every step against another loopy target and
  re-checks it. A tag belongs to a loop, so `split`, `tile` and `affine`
  refuse a loop that carries one, and the loops they make are tagged after
  them. `affine(map)` takes an isl map from loops to the loops that
  replace them, refuses one that misses or merges an instance or runs a
  dependence backwards, and rewrites the kernel over the map's image; `skew` is
  that method with a particular map. A union map whose tuples name statements,
  `{ S0[t, i] -> [a, b] : ...; S1[t, i] -> [a, b] : ... }`, moves each
  statement by its own map, checked on the dependences between the statements
  as well as within each, which is the time offset a diamond tiling of two
  statements that feed each other needs. A program's statement, `flux.S0` or
  `step@2.S0`, which isl cannot read as a tuple name, is named with every
  other character spelled `_`, `flux_S0` or `step_2_S0`, the id of its
  instruction in the lowered kernel. A loop over an image with holes, such
  as the diamond's `b`, counts its steps (`b = 2*b_step - a`,
  `Schedule.strides`) instead of testing a parity at every `b`. A loop on a
  hardware axis (`g.*`, `l.*`) is the launch grid, outside every other loop,
  and nothing in a kernel orders two of its work items through global memory,
  so the `monotone` cast also refuses any dependence between instances on two
  work items, with the dependence and the two work items as the witness (the
  stencil's `jacobi` with `i` on `g.0`, the acoustic pair likewise). A
  statement with no loop on the axis runs on every work item of it. A sum on
  a local axis stays allowed, since loopy synchronizes its partial sums; its
  body reads on every work item, and its statement stores the result from
  one, so a sum whose body reads what another sum's statement stored is
  refused, as loopy refuses it for want of a global barrier.
- The target-capability check: a concurrent tag (a hardware axis, `ilp` or
  `vec`) inside a data-dependent (ragged) loop bound or its domain, a hardware
  axis on a reduction nested in another, a reduction loopy will not realize
  (partly in parallel and partly in sequence, across two local axes, on a
  group axis, or on a local axis whose extent has no numeric maximum), a
  hardware axis loopy will not assign (numbered past an unused one, shared by
  two loops of one statement, missing from an instruction the kernel runs
  beside it, or `l.auto`), an `unr`, `ilp` or `vec` loop whose length is not a
  number, a temporary loopy misreads once an `ilp` or `vec` loop has a copy of
  it per iteration (a ragged row's length), a loop ordered outside a loop
  loopy nests it inside, or a hardware axis on the C target, which has none,
  is reported as a `refuted` `buildable` fact and raises `UnbuildableSchedule`
  when something asks for code. It is asked of the schedule as it stands after
  every step, so an interchange can make a tiled ragged loop buildable again.
- Lowering to loopy, including a ragged axis as a flat buffer plus offsets, and
  running on `lp.ExecutableCTarget`. Every argument of `LoopyExecutor.run` is
  an argument of the kernel; the target is chosen by the schedule
  (`Schedule(kernel, target="opencl")`) or by the executor
  (`LoopyExecutor(target="opencl")`). Each operation is computed in the type
  numpy computes it in natively (`loopty.promotion`, by NEP 50), where C's
  conversions would pick another: a quotient of integers is a double, a Python
  float beside a `float32` is single precision (`0.1f`), a `float32` beside an
  integer array is a double, and a floating power calls `pow`, as numpy does.
  Any power compiles on the C target, of a complex base too. See note 19 in
  `docs/loopy-notes.md`.
- Array arguments over polyhedral domains (`loopty.domain`): `Where[...]`,
  binders written as slices and then the comparisons that cut their box,
  joined by `&`; `Sigma[...]`, binders and an unnamed last fiber affine in
  them; and a union of pieces, `Fin[n] + Fin[m]` (lanky's `SumType`). `.dom`
  runs binder by binder, `L.dom[i]` over the points the constraints allow at
  `i` (a fiber at a point outside the domain is empty), and a union's pieces
  by number, which a trace runs as the Python loop it is. The in-bounds
  obligations are decided over the exact set. An argument over other points
  is refused at every entry point, and so is a plain `ndarray`. Both layouts
  lower and run on the C target: the box of the binders, addressed by loopy,
  and packed rows through a table of row starts (`Schedule.pack`), which the
  executor computes from the domain and passes. `Arr.zeros(domain, n=...,
  storage=...)` and `Arr.from_cells` build such an array, and `Arr.cells()`
  reads it in one order whatever its layout. `examples/pairs.py` is the demo.
- The argument contract, enforced at every entry point that runs a kernel
  (compiled, differential and native): two distinct array parameters may not
  share storage, a ragged argument has to agree with the counts array its type
  names and with any offsets passed alongside it, an argument over a domain has
  to have the declared domain's points at the sizes of the call, and a value
  of a refined sort such as `Fin[m]` has to be one — an array element and a
  scalar argument alike, and being one means being a finite whole number in
  range, not merely passing two comparisons. A value of `Nat`, `Int` or
  `Fin[m]` is also inside the 32 bits the compiled run stores it in, since
  `2**32 + 5` ran natively as it was and was `5` compiled. An array the kernel
  writes has to be stored as its element sort is natively (`float64` for
  `Real`, `bool` for `Bool`, a signed integer of 32 bits or more for `Nat`,
  `Int` and `Fin[m]`, a numpy sort as itself), since an integer `x` for a
  `Real` parameter truncates every write the compiled run keeps; an array it
  only reads is read by the native run in that dtype, as the compiled run
  converts it, so an integer `x` no longer overflows natively where the
  compiled double does not. A complex
  entry of a sort that is not complex has no imaginary part, and an entry of
  `Bool` stored as a number is `0` or `1`. A scalar is asked the same, and is
  converted into the dtype of its sort in both runs, since it is passed by
  value: `np.int64(2**32)` for a `Real` is a double natively too, and a `Bool`
  a numpy bool, on which `~` is `not`. These are the assumptions the typing
  rules and the two runs make about a *call* rather than about the term, and a
  violation is a `ValueError` naming the argument. Distinct parameters being disjoint storage
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
  disagrees names the outputs and by how much), and the command exits 1. So
  it does when two kernels claim one fact id, as two kernels one definition
  makes (a factory) do: a `DUPLICATE` block names the kernel and its ids, as
  `lanky check` names them, and the `--json` ledger keeps each claim not in
  the table under `refused_claims`, with its status and what explains it. A
  schedule of a schedule, `Schedule(Schedule(k))`, is a schedule of `k`: its
  run is compared with `k`'s body, and `k` is not run again through the
  identity.

**Partial.**

- Ragged bounds are reflected into isl as one parameter per distinct bound term
  (allocated once per kernel, so the same `cnt[r]` is one parameter everywhere
  and two different bounds are never given one name), so `cnt[r]` and
  `cnt[r + 1]` are unrelated to isl. Within one kernel nothing knows that
  counts sum to the offsets, or that `off` is monotone, so an access against
  flat storage, `val[off[r] + j]`, is reported **`assumed`** in the kernel's
  own ledger, with the reason in its provenance. The ragged spelling
  `val[r, j]` over `0 <= j < cnt[r]`, which is what the tracer and the demos
  produce, *is* decided. In a program, the same access is decided where the
  kernel is called after the call that wrote `off`, under that call's
  postcondition: the cells `off[r]`, `off[r + 1]` and `cnt[r]` become isl
  parameters, and the scan's recurrence, instantiated at them, is an affine
  constraint between them (`loopty.hypotheses`, `examples/travel.py`). The
  fact is the program's, resting on what it used.
- A kernel's postcondition is `tested` against its native runs, on the
  file's example inputs and on drawn ones, never `decided`: deciding it from
  the term, by which statement last writes each cell, is not done yet.
  `@program` restates a callee's postcondition as a fact in scope, `decided` by
  the call and resting on the callee's own fact (lanky's `rests_on`), and
  offers it as a hypothesis to the calls after it, until a call writes an
  array it mentions. A callee imported from another file has its fact in that
  file's ledger, so `lanky check` counts the id as an assumption and names it
  in an `UNRESOLVED` line; the id is the one that ledger holds, since every
  fact of a kernel is keyed by the kernel's definition (`lanky.ledger.fact_id`
  over the module the file's path gives it), and a kernel of the program's
  own file with the callee's name has an id of its own. A theorem the program
  cites (`@program(uses=[scan_monotone])`) is used at the arrays its
  hypotheses match, and only where each array's cells are points of the
  family's sort when the call is made: `scan_monotone` is about offsets of
  `Nat`, and nothing says the cells a scan wrote are, so it takes a theorem
  over `Int` to say the offsets a scan computed are monotone. Only affine
  facts reach isl; a hypothesis isl cannot state is dropped, which assumes
  less, never more.
- A program lowers sequentially: its calls' loops run one after another, as
  the program runs them. Fusing them is a cast over the program's term that
  is not written yet, and so is deciding the storage of an intermediate. The
  compiled program is one call, so the contract checks its arguments when it
  starts and not at every call. What a callee's contract checks of the cells
  of an array an earlier call wrote, or the program made (an element sort
  `Fin[m]` or `Nat`, the offsets a ragged family is read through), is a
  `requirement` of the program: decided by isl under the hypotheses that held
  at the call, or, where they do not decide it, checked by the compiled
  program between the two calls, which then stops with the requirement's
  message where the native callee is refused (`examples/travel.py`). An
  integral array such a check reads is stored in 64 bits, as the native run
  holds it, so that a value written outside 32 bits is read as written and
  not as the store narrowed it. Offsets a scan computes and a later call
  declares `Nat` without reading rows through them are checked: that they
  are naturals follows from the scan's recurrence only by induction. The
  counts of a ragged family an earlier call wrote are still refused: its rows
  are laid out in a buffer the program is given, which no hypothesis about
  the program's arrays can speak of. The kernels an array is passed to have to declare the same
  element sort for it, and every call has to read a ragged family's rows
  through the same offsets; a loop in the body whose trip count is an
  argument (a host loop) is refused, and so are an array made like a ragged
  one and a callee with an array over a `Where`, `Sigma` or union domain. A
  temporary made like a parameter, `Arr.zeros_like(u)`, has `u`'s dtype
  natively, so the compiled program refuses a `u` whose dtype does not hold
  what the compiled temporary holds: `float64` for `Real`, the dtype itself
  for a numpy one such as `np.complex128`, `bool` for `Bool`, and a signed
  integer of 32 bits or more for `Nat`, `Int` and `Fin[m]`. On the C target a
  temporary is a variable-length array on the stack of the call, which bounds
  its size (note 16 in `docs/loopy-notes.md`); on OpenCL it is a global
  temporary, which is generated but, like every device path, not run from a
  development machine.
- Only a two-axis (row, fiber) ragged array, its rows counted by an array of
  one axis, is built and lowered; tracing refuses any other ragged type, a
  fiber after two dense axes, a dense axis after the fiber, or counts of two
  axes, before any fact is stated about it.
- Integers are 64 bits wide natively and 32 bits compiled. The contract keeps
  every integral argument inside 32 bits, but a result that leaves them
  (`c[i] * c[i]` at `c[i] = 2**20`) is a wider number natively and wraps
  compiled. A scalar of `Real` or of an integral sort is a Python number
  natively when the caller passes one, and so takes a `float32`'s precision
  beside one where a numpy scalar would not; an operation whose type depends
  on that is left as C types it. See note 19 in `docs/loopy-notes.md`.
- A polyhedral domain is an array's whole index set, so it cannot sit beside
  a dense axis (`Arr[Fin[k], Where[...], Real]` is refused; write the axis as
  a binder of the domain). The pieces of a union have the same number of axes,
  and a piece is chosen by a Python integer, never by a loop variable. A
  constraint is a conjunction of comparisons: `!=` and `|` are refused rather
  than widened (write a union instead), and the packed layout refuses a domain
  whose rows skip columns (a remainder in a constraint). An array over a
  domain is indexed, and its fibers taken, at quasi-affine expressions of loop
  variables and sizes; an indirect index such as `L[p[k], j]` is refused when
  traced. A size a binder's bound runs up to is never negative, so a scalar
  that makes one negative is refused. Natively, an array over a domain
  enumerates its points with isl when it is built and checks each access
  against them in Python. A run, native or compiled, is over the declared
  domain, its loops, its sizes and its layout, so an argument over the same
  points written otherwise, or stored the other way, is copied into it and
  back.
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
- The in-bounds fact of a ragged access `val[r, j]` is decided against the
  row's length, and the dependences and disjoint writes of `val` are decided
  over `[r, j]`. Both assume the offsets lay the rows out inside the flat
  buffer and apart from each other, which the contract checks when a run
  starts. A kernel that writes its counts or its offsets can break that during
  the run, so such a kernel has a `layout` fact for each counts family it
  rewrites, and the in-bounds and disjoint-writes facts of the family's
  ragged arrays rest on it, as do a fact decided by type through an index
  read from one of them (`x[col[r, j]]`) and the `monotone` casts of its
  schedules: the ledger shows them `decided under layout:...`. isl decides
  the layout fact when the kernel writes only the offsets, each either as
  the counts lay it out (`off[r + 1] = off[r] + cnt[r]`, which writes back
  what the contract checked) or as a value of the loop variables and the
  sizes, and refutes it with the instance that writes a start the counts can
  contradict (`off[r] = 0` at `r = 1`). Any other write, a count, a start
  read from another array, one under a guard isl cannot state, is decided by
  induction over the run when isl shows it keeps the rows in order (each
  starting no earlier than the row before it ends, inside the buffer)
  wherever they were before it: `cnt[r] = 0` does, and so does a start moved
  within the room its row leaves. One isl cannot show does leaves the fact
  `assumed`, and the facts on it worth an assumption, unless a native run of
  the kernel refutes it: one that reads a row off the buffer through the
  layout it has written (`off[r + 1] = off[r] + cnt[r] + 1`, which moves the
  last row past the end), or that leaves two rows on one cell
  (`off[r] = s[r]`). The order is more than the rows' staying apart, so a
  kernel that permutes its rows inside the buffer is left `assumed`; and it
  is less than some facts on the layout need, so the induction is not asked
  of a kernel that moves rows and writes the family's arrays, whose disjoint
  writes and dependences tell the cells apart as `[r, j]`, one cell each
  only while no row moves, nor of one that writes a row's length inside a
  loop over the row, which runs to the length it read when it started. A
  program's layout facts are not tried on runs.
- `Schedule.affine` and maps whose image has holes. The diamond
  `(t, i) -> (t + i, t - i)` reaches only the points of equal parity, and
  loopy's own `map_domain` refuses it, so loopty rewrites the kernel over the
  image itself, and the loop left with the holes counts its steps. The code is
  correct, bit for bit against the native run for the stencil and the acoustic
  pair, untiled and tiled in diamond coordinates. Statements moved by maps of
  their own keep sharing their loops, because loopy gives the statements of a
  loop one domain: the loops run over the union of the images, and each
  statement tests that a point is its own. For the acoustic pair every point
  is one statement's; for maps whose images leave holes between them, such as
  `S0` at `2t` and `S1` at `2t + 1` along the diamond, the loops run over the
  hull of the union and visit the holes. A map the rewrite cannot write for
  loopy, such as one over a row and the ragged fiber inside it, or maps that
  move two statements of one ragged fiber, or of two fibers of one row,
  differently, is a `refuted` `buildable` fact. See note 13 in `docs/loopy-notes.md`.
- `realize(var, tree=True)` checks and marks the reassociation; the reduction
  tree itself comes from splitting and tagging the reduction iname, which is
  checked separately and not verified on the C target.
- The accumulation convention: a traced `y[r] += ...` under a parallel iname is
  reported as a disjointness refutation, which is the conservative reading. The
  `reassoc` fact is what should license it and nothing consumes that yet.
- The target-capability check knows the limits of loopy 2025.2 listed in notes
  6, 11 and 14 of `docs/loopy-notes.md` and no others, so it is a list rather
  than a model of what the backend can do. A schedule it passes can still fail
  in code generation for a reason nobody has met yet.
- A statement with no loop on a hardware axis that a loop or a sum of another
  statement is on runs on every work item of it, so every dependence to or
  from it is refused, even where each work item reads back only what it
  wrote itself (every work item storing one value into one cell, say; loopy
  refuses such a statement for the axis it lacks in any case). The body of a
  sum on such an axis is taken to read on every work item of it, though each
  read happens on the work item of the sum's loop, and the work item a sum's
  statement stores its result from is taken to be one unknown work item, the
  same for every such statement, though loopy always uses the first. Both
  are coarser than the check could be, and refuse more. Without a kernel to
  read a loop's start off (after an `affine` step the kernel rewrite could
  not write), only two instances of one loop are taken to share a work item.
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
  interpreter has no numpy counterpart for, or when one ragged bound of a
  statement bounds two of its loops (a fiber loop inside another over the
  same row), or a loop and a sum inside it over the same row, and reads an
  array the kernel writes, which the body reads where each loop or sum starts
  and the interpreter reads once.

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

loopty needs lanky 0.1.0.dev1 or later, and follows lanky's `main` between
`devN` bumps. lanky's version moves to the next `0.1.0.devN` whenever an
interface loopty uses changes, and loopty's floor (`lanky>=0.1.0.dev1` in
`pyproject.toml`) is raised with it; in between, loopty is developed and
tested against lanky's `main`, and a lanky that satisfies the floor may still
lack something loopty's `main` uses.

For work on loopty itself, lanky is resolved from a sibling checkout:

```sh
git clone https://github.com/xywei/lanky.git     # next to this one
uv sync --group dev
uv run pytest -q
uv run ruff check .
```

CI clones lanky's branch of the same name as the one under test when there is
one, and lanky's `main` otherwise, so a change that needs both repositories is
tested as one.

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
array's entry. An array's index set may also be a polyhedral domain, an isl set
it is compared with exactly. `Layout` and `RaggedLayout` are the maps from index
space to storage, and a domain's box and its packed rows are two more.

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
- [examples/README.md](examples/README.md): all seven demos, with every console
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
