# The loopty demos

Thirteen files, each of which runs three ways. Every console block below is a
snapshot of real output, not prose about it, and it is produced mechanically:

```console
$ uv run python scripts/refresh_example_outputs.py          # rewrite the blocks
$ uv run python scripts/refresh_example_outputs.py --check  # exit 1 if any is stale
```

The script re-runs the command at the top of each block and replaces the rest of
the block with what it printed (standard output only; loopy's warnings go to
standard error and would change with the toolchain rather than with loopty).
Run it after changing a demo, or the line numbers, fact counts and tolerances
here quietly stop being true; CI runs it with `--check`. It keeps the
transcripts in `../README.md` and `../docs/quickstart.md` too, where a block
marked with `...` is an excerpt and the lines it keeps are checked verbatim.

| file | what it shows |
|---|---|
| `spmv.py` | a ragged sparse product: an indirection in bounds by type, a scan with a postcondition, the theorem that postcondition needs, and a schedule whose reassociation is recorded |
| `stencil_skew.py` | a rectangular tiling of a Jacobi stencil refused with a witness pair, and the skew that makes the same tiling legal |
| `wavefront_acoustic.py` | two coupled statements in one acoustic-wave nest: a rectangular tiling refused with a witness that crosses them, the skew that makes it a legal wavefront block, the diamond map, accepted and run, whose tiling is refused, and the diamond with a map per statement, whose tiling is accepted and run |
| `reshape_layouts.py` | `Fin[n * m]` as `Fin[n] x Fin[m]`, one buffer read in two layouts, and a transpose split and interchanged |
| `p2p.py` | the near field of a fast multipole method: a two-level interaction list flattened into one ragged level, with the self-interaction guarded by `when` |
| `pairs.py` | symmetric pair interactions over the lower triangle, an array argument over the domain `Where[i: Fin[n], j: Fin[n], j < i]`, decided in bounds over the exact triangle and run boxed and packed |
| `composition.py` | two kernels composed by a program and lowered as one loopy kernel, with the intermediate a temporary of it and the edge between the kernels found in the footprints |
| `travel.py` | facts that travel between a program's calls: a requirement decided under the postcondition of the call before, one checked by the compiled program between the calls, and a flat access in bounds where it follows the scan |
| `fusion.py` | the program of `composition.py` with its two loops fused into one, a fusion one step too early refused with its pair, and the intermediate computed where it is read instead of stored |
| `quadrature.py` | firedrake's P1 mass kernel cell by cell, its quadrature rule and basis held in constant tables, read-only arrays written into the compiled code, and its gathers through the cell map in bounds by type |
| `layer_potential.py` | pytential's QBX single layer over complex points with `real` and `log` from `loopty.math`, scheduled as sumpy's CPU tile, the sources' loop outside the targets', and the same order on a loop loopy cannot keep it on |
| `sum_factorization.py` | grudge's two passes of a sum-factorized operator, fused on the element and the row and no deeper: the deeper fusion refused with its pair, and the fusion of fewer loops it names |
| `calls.py` | kernels calling kernels inside their loops: a rule with a loop of its own (`dot`, loopy #1044) in a matrix-vector and a sparse product, and firedrake's P1 quadrature rule applied per element through a rule that takes the integrand (#1043), each call inlined with binders of its own and owing the caller what the callee's contract checks |

Run them with `uv run` from the repository root:

```console
$ uv run python examples/spmv.py           # numpy only; theorems as property tests
$ uv run lanky check examples/spmv.py      # the ledger: every obligation and its decider
$ uv run loopty run examples/spmv.py       # through loopy onto the C target, compared
```

`lanky check` exits non-zero if any fact is refuted, and `loopty run` exits
non-zero if a compiled run disagrees with the native one, so both are usable in
a test or a hook. Sizes are tiny everywhere: the point is the ledger, not the
throughput.

## spmv.py

The kernel is `y[r] = sum(val[r, j] * x[col[r, j]] for j in val.dom[r])` over a
dependent sum. Three things are worth following through the three outputs.

`x[col[r, j]]` is decided **by type**: the entries of `col` are points of
`Fin[m]` and `x` has `m` cells, so there is nothing for an oracle to do.

The accumulation into `y` is `approx`, because that is the class of what it
sums, and it *becomes* `reassoc` when `realize('y', tree=True)` reorders it.
The direction matters: a trace may not assume permission to reassociate, and a
schedule has to ask for it, which is why the widened tolerance at the end of the
run is traceable to a line of the schedule rather than to a default.

The design's device schedule, `tag(r='g.0').split('j', 32).tag(j_in='l.0')`,
is legal and **not buildable**, and the demo prints both halves of that. Every
cast is `decided`: nothing is reordered that carries a dependence. Code
generation is refused, on a device as much as on C, because loopy will not put a
hardware axis inside a loop whose bound comes from an array, and a CSR row is
exactly such a loop. (C has no hardware axes at all, and says so for the rows
alone; the reason printed is the limit a device has too.) That refusal is a
`refuted` fact of kind `buildable` decided by `loopy-target`, carried beside
the decided casts; `docs/device-runs.md` has the measurement it comes from and
`docs/loopy-notes.md` the details.

### python examples/spmv.py

```console
$ uv run python examples/spmv.py
counts  = [3 2 2 1 1 0]
offsets = [0 3 5 7 8 9 9]
y       = [ 0.4821 -0.8792 -0.0973 -0.5674  0.4291  0.    ]
dense   = [ 0.4821 -0.8792 -0.0973 -0.5674  0.4291  0.    ]

scan_monotone: n : Nat, cnt : Fn[Fin(n), Nat], off : Fn[Fin(n + 1), Nat] | off(0) == 0, forall r in Fin(n). off(r + 1) == off(r) + cnt(r) |- forall a in Fin(n + 1), b in Fin(n + 1) where a <= b. off(a) <= off(b)
  ok after 50 valid draws of 50

schedule: Schedule(spmv, target='c').split(j, 2).realize('y', tree=True)
  decided  isl  split(j, 2) renames the instances of spmv one for one
  decided  isl  the order after split(j, 2) runs every dependence of spmv forward
  decided  isl  realize('y', tree=True) renames the instances of spmv one for one
  decided  isl  the order after realize('y', tree=True) runs every dependence of spmv forward
  decided  isl  the accumulation into y is reassociated by realize('y', tree=True), so its result is compared at 'reassoc'
device schedule: Schedule(spmv, target='c').tag(r='g.0').split(j, 32).tag(j_in='l.0').realize('y', tree=True)
  decided  isl  tag(r='g.0') renames the instances of spmv one for one
  decided  isl  the order after tag(r='g.0') runs every dependence of spmv forward, within one work item
  decided  isl  split(j, 32) renames the instances of spmv one for one
  decided  isl  the order after split(j, 32) runs every dependence of spmv forward, within one work item
  decided  isl  tag(j_in='l.0') renames the instances of spmv one for one
  decided  isl  the order after tag(j_in='l.0') runs every dependence of spmv forward, within one work item
  refuted  loopy-target c code can be generated for spmv after tag(j_in='l.0')
  decided  isl  the accumulation into y is reassociated by tag(j_in='l.0'), so its result is compared at 'reassoc'
  decided  isl  realize('y', tree=True) renames the instances of spmv one for one
  decided  isl  the order after realize('y', tree=True) runs every dependence of spmv forward, within one work item
  reason: the parallel tag on j_in sits inside a loop whose bound comes from an array (a ragged fiber), and loopy will not put a hardware axis in a domain with a data-dependent parameter. Parallelize an enclosing loop with a size known at launch instead, such as the rows of a CSR product

  y: difference 0 within 1e-06 (approx) -> tested
```

### lanky check examples/spmv.py

The scan's postcondition is `tested` by `native`: it held after every native
run of `scan`, on this file's example inputs and on drawn ones, and the
restatement of it in `solve` is `decided` by the call and worth `tested`, as
the `EFFECTIVE` column shows, since it rests on that fact. Nothing decides the
recurrence from the term yet, and `lanky` says how it was established rather
than passing over it. The theorem beside it is `tested` here, and `proved` on a
machine with the Lean extra installed, and the row under it is lanky's check of
its reading: the theorem's annotations, run again as plain Python at drawn
values, compute what its term does, so the term the oracles are handed is the
claim written. The last row of each kernel, `tested` by
`interpreter`, is the one fact about the trace itself: the traced term, run by
loopty's interpreter, agrees with the body run natively, on this file's
`example_inputs()` and on three inputs drawn from the declared types. The
program `solve` has that fact too, about the term its two calls compose into.

```console
$ uv run lanky check examples/spmv.py
STATUS   EFFECTIVE  BY             WHERE        OWNER          STATEMENT
-------  ---------  -------------  -----------  -------------  ------------------------------------------------------------------------
decided  decided    isl            spmv.py:79   scan           off[0] is in bounds for every instance of S0
decided  decided    isl            spmv.py:81   scan           off[r + 1] is in bounds for every instance of S1
decided  decided    isl            spmv.py:81   scan           off[r] is in bounds for every instance of S1
decided  decided    isl            spmv.py:81   scan           cnt[r] is in bounds for every instance of S1
decided  decided    isl            spmv.py:79   scan           distinct instances of S0 write distinct cells of off
decided  decided    isl            spmv.py:81   scan           distinct instances of S1 write distinct cells of off
decided  decided    isl            spmv.py:81   scan           every cell of off that S1 reads, S0 or S1 wrote before the read
decided  decided    isl            spmv.py:69   scan           the source order runs every dependence forward in time
tested   tested     native         spmv.py:69   scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
tested   tested     interpreter    spmv.py:69   scan           the traced term computes what the body computes
tested   tested     property-test  spmv.py:84   scan_monotone  n : Nat, cnt : Fn[Fin(n), Nat], off : Fn[Fin(n + 1), Nat] | off(0) ==...
tested   tested     python         spmv.py:84   scan_monotone  the term computes what the annotations compute
decided  decided    isl            spmv.py:112  spmv           y[r] is in bounds for every instance of S0
decided  decided    isl            spmv.py:112  spmv           val[r, j] is in bounds for every instance of S0
decided  decided    type           spmv.py:112  spmv           x[col[r, j]] is in bounds by type (col[r, j] : Fin(m))
decided  decided    isl            spmv.py:112  spmv           col[r, j] is in bounds for every instance of S0
decided  decided    isl            spmv.py:112  spmv           cnt[r], the length of row r that bounds the loop over j, is in bounds...
decided  decided    isl            spmv.py:112  spmv           distinct instances of S0 write distinct cells of y
decided  decided    isl            spmv.py:102  spmv           the source order runs every dependence forward in time
decided  decided    type           spmv.py:112  spmv           the accumulation into y[r] over j is approx
tested   tested     interpreter    spmv.py:102  spmv           the traced term computes what the body computes
decided  tested     call           spmv.py:115  solve          after scan(...) in solve: off[0] == 0 and (forall r in Fin(n). off[r ...
tested   tested     interpreter    spmv.py:115  solve          the traced term computes what the body computes

23 facts: 17 decided, 6 tested
```

### loopty run examples/spmv.py

The scheduled `spmv` and the unscheduled `scan` are both compiled and compared
with their own Python bodies. `scan` is integer arithmetic, so its tolerance is
zero; `y` is compared at the accuracy its exactness class states. So is
`solve`, the program: its term is `scan`'s statements followed by `spmv`'s, in
call order, lowered as one kernel and compared with the program run natively,
both outputs at once. The two calls share only `cnt`, which both read, so
nothing orders one loop after the other; `composition.py` below has a program
whose second kernel reads what its first wrote.

```console
$ uv run loopty run examples/spmv.py
spmv: Schedule(spmv, target='c').split(j, 2).realize('y', tree=True)
  y: difference 0 within 1e-06 (approx) -> tested
scan: Schedule(scan, target='c')
  off: difference 0 within 0 (exact) -> tested
solve: Schedule(solve, target='c')
  y: difference 0 within 1e-06 (approx) -> tested
  off: difference 0 within 0 (exact) -> tested

STATUS   BY     WHERE        OWNER  STATEMENT
-------  -----  -----------  -----  ------------------------------------------------------------------------
decided  isl    spmv.py:112  spmv   split(j, 2) renames the instances of spmv one for one
decided  isl    spmv.py:112  spmv   the order after split(j, 2) runs every dependence of spmv forward
decided  isl    spmv.py:112  spmv   realize('y', tree=True) renames the instances of spmv one for one
decided  isl    spmv.py:112  spmv   the order after realize('y', tree=True) runs every dependence of spmv...
decided  isl    spmv.py:112  spmv   the accumulation into y is reassociated by realize('y', tree=True), s...
tested   loopy  spmv.py:112  spmv   the scheduled run of spmv agrees with the native run to the accuracy ...
tested   loopy  spmv.py:79   scan   the scheduled run of scan agrees with the native run to the accuracy ...
tested   loopy  spmv.py:115  solve  the scheduled run of solve agrees with the native run to the accuracy...

8 facts: 5 decided, 3 tested
```

## stencil_skew.py

`u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2`, tiled. The rejection is the
point: the checker names the two statement instances a rectangular tile would
run out of order, and the same tiling is accepted once the space axis is skewed
by the time axis.

### python examples/stencil_skew.py

```console
$ uv run python examples/stencil_skew.py
native, the 16 by 16 array (first six levels around the spike):
[[0.    0.    0.    1.    0.    0.    0.   ]
 [0.    0.    0.5   0.    0.5   0.    0.   ]
 [0.    0.25  0.    0.5   0.    0.25  0.   ]
 [0.125 0.    0.375 0.    0.375 0.    0.125]
 [0.    0.25  0.    0.375 0.    0.25  0.   ]
 [0.156 0.    0.312 0.    0.312 0.    0.156]]

Schedule(jacobi).tile('t', 'i', 8, 8) ->
  IllegalCast: tile(t,i,8,8) illegal: instance S0[t=0, i=8] writes u[1, 8] read by S0[t=1, i=7] scheduled earlier (at nt=16, nx=16, as hinted)
  witness: S0{'t': 0, 'i': 8} runs before S0{'t': 1, 'i': 7} at {'nt': 16, 'nx': 16}

accepted: Schedule(jacobi, target='c').skew(i, by='t').tile(t,i,8,8)
  loop nest: t_outer i_outer t_inner i_inner
  decided  isl  skew(i, by='t') renames the instances of jacobi one for one
  decided  isl  the order after skew(i, by='t') runs every dependence of jacobi forward
  decided  isl  tile(t,i,8,8) renames the instances of jacobi one for one
  decided  isl  the order after tile(t,i,8,8) runs every dependence of jacobi forward

  u: difference 0 within 1e-06 (approx) -> tested
  the native run matches the hand-written sweep: True
```

### lanky check examples/stencil_skew.py

Every access is in bounds because the `when` guard narrows the statement's
domain: `u[t + 1, i]` is written only where `t + 1 < nt`, and isl is asked about
the narrowed domain rather than the whole loop nest.

```console
$ uv run lanky check examples/stencil_skew.py
STATUS   BY           WHERE               OWNER   STATEMENT
-------  -----------  ------------------  ------  ------------------------------------------------------
decided  isl          stencil_skew.py:62  jacobi  u[t + 1, i] is in bounds for every instance of S0
decided  isl          stencil_skew.py:62  jacobi  u[t, i - 1] is in bounds for every instance of S0
decided  isl          stencil_skew.py:62  jacobi  u[t, i + 1] is in bounds for every instance of S0
decided  isl          stencil_skew.py:62  jacobi  distinct instances of S0 write distinct cells of u
decided  isl          stencil_skew.py:54  jacobi  the source order runs every dependence forward in time
tested   interpreter  stencil_skew.py:54  jacobi  the traced term computes what the body computes

6 facts: 5 decided, 1 tested
```

### loopty run examples/stencil_skew.py

```console
$ uv run loopty run examples/stencil_skew.py
jacobi: Schedule(jacobi, target='c').skew(i, by='t').tile(t,i,8,8)
  u: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE               OWNER   STATEMENT
-------  -----  ------------------  ------  ------------------------------------------------------------------------
decided  isl    stencil_skew.py:62  jacobi  skew(i, by='t') renames the instances of jacobi one for one
decided  isl    stencil_skew.py:62  jacobi  the order after skew(i, by='t') runs every dependence of jacobi forward
decided  isl    stencil_skew.py:62  jacobi  tile(t,i,8,8) renames the instances of jacobi one for one
decided  isl    stencil_skew.py:62  jacobi  the order after tile(t,i,8,8) runs every dependence of jacobi forward
tested   loopy  stencil_skew.py:62  jacobi  the scheduled run of jacobi agrees with the native run to the accurac...

5 facts: 4 decided, 1 tested
```

## wavefront_acoustic.py

A velocity update and a pressure update share one `(t, i)` loop nest. The
pressure statement `S1` reads the velocity that `S0` wrote in the same time
step, and the next step's velocity statement reads the pressure `S1` wrote. The
one dependence a rectangular tile runs backwards is therefore **S1 -> S0**, at
distance `(1, -1)`, and not a statement against itself as in the stencil.

The rectangular tile is rejected with that cross-statement witness, and
`skew("i", by="t").tile(...)` is accepted and runs. Geometrically this is a
wavefront temporal block: rectangular in `(t, i + t)`, a parallelogram in
`(t, i)`.

Then the diamond, `Schedule.affine("{ [t, i] -> [a, b] : a = t + i and b = t - i }")`.
The map is not unimodular: its image is only the points of equal parity. Written
with space first, `a = i + t` and `b = i - t`, it is refused, because `S0` at
`(t + 1, i - 1)` reads what `S1` wrote at `(t, i)` and the new order runs that
backwards. Written with time first it is accepted, and the question this demo
was extended to answer has its answer: loopy generates correct code over that
image, and both fields agree with the native run bit for bit. loopy's own
`map_domain` refuses the map, so loopty rewrites the kernel itself, and the
loop over `b` counts its steps, `b = 2*b_step - a`, which is why the printed
loopy domain is over `[a, b_step]` and has no parity left in it: loopy alone
loops over every `b` and tests the parity inside the innermost loop (note 13 in
`../docs/loopy-notes.md`). Tiling the diamond is refused: `S1` at `(t, i + 1)`
reads the velocity `S0` wrote at `(t, i)`, a distance of `(0, 1)` that the
`t - i` direction runs backwards.

A diamond tiling of this pair needs an offset between the two statements, and
`affine` takes a map per statement, with each statement named on its tuple:

```python
OFFSET_DIAMOND = (
    "{ S0[t, i] -> [a, b] : a = t + i and b = t - i; "
    "S1[t, i] -> [a, b] : a = t + i and b = t - i + 1 }"
)
```

`S1` now sits half a step after `S0` of the same `(t, i)`, which is where a
staggered scheme keeps its pressure, and every dependence of the pair is
`(0, 1)`, `(1, 0)` or `(1, 1)` in `(a, b)`. The same four by four tiling is then
accepted, the two casts are asked of both maps together, and the tiles agree
with the native run bit for bit. The statements still share the loops over `a`
and `b`, because loopy gives the statements of a loop one domain: `S0` runs at
the points where `a + b` is even and `S1` at those where it is odd, so every
point of the loops is one statement's. The module docstring has the details.

`uv run python examples/wavefront_acoustic.py --bench` times the untiled and the
wavefront-blocked compiled kernels at a larger size. There is no console block
for it: its numbers describe the machine that ran it, not loopty, and a speedup
is not something CI could hold anyone to.

### python examples/wavefront_acoustic.py

```console
$ uv run python examples/wavefront_acoustic.py
native pressure, 16 levels by 32 points (first five levels around the impulse):
[[0.    0.    0.    1.    0.    0.    0.   ]
 [0.    0.    0.062 0.875 0.062 0.    0.   ]
 [0.    0.004 0.172 0.648 0.172 0.004 0.   ]
 [0.    0.018 0.301 0.362 0.301 0.018 0.   ]
 [0.002 0.049 0.415 0.068 0.415 0.049 0.002]]
statements: S0 writes velocity, S1 writes pressure

Schedule(acoustic).tile('t', 'i', 4, 8) ->
  IllegalCast: tile(t,i,4,8) illegal: instance S1[t=0, i=8] writes pressure[1, 8] read by S0[t=1, i=7] scheduled earlier (at nt=16, nx=32, as hinted)
  witness: S1{'t': 0, 'i': 8} runs before S0{'t': 1, 'i': 7} at {'nt': 16, 'nx': 32}

accepted: Schedule(acoustic, target='c').skew(i, by='t').tile(t,i,4,8)
  loop nest: t_outer i_outer t_inner i_inner
  decided  isl  skew(i, by='t') renames the instances of acoustic one for one
  decided  isl  the order after skew(i, by='t') runs every dependence of acoustic forward
  decided  isl  tile(t,i,4,8) renames the instances of acoustic one for one
  decided  isl  the order after tile(t,i,4,8) runs every dependence of acoustic forward

  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested
  the native run matches the hand-written recurrence: True

Schedule(acoustic).affine('{ [t, i] -> [a, b] : a = i + t and b = i - t }') ->
  IllegalCast: affine({ [t, i] -> [a = t + i, b = -t + i] }) illegal: instance S1[t=0, i=2] writes pressure[1, 2] read by S0[t=1, i=1] scheduled earlier (at nt=16, nx=32, as hinted)

accepted: Schedule(acoustic, target='c').affine({ [t, i] -> [a = t + i, b = t - i] })
  loop nest: a b
  loopy domain: [nt, nx] -> { [a, b_step] : b_step >= 0 and 2 - nx + a <= b_step < a and b_step <= -2 + nt }
  decided  isl  affine({ [t, i] -> [a = t + i, b = t - i] }) renames the instances of acoustic one for one
  decided  isl  the order after affine({ [t, i] -> [a = t + i, b = t - i] }) runs every dependence of acoustic forward

  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested

Schedule(acoustic, target='c').affine({ [t, i] -> [a = t + i, b = t - i] }).tile('a', 'b', 4, 4) ->
  IllegalCast: tile(a,b,4,4) illegal: instance S0[t=7, i=15] writes velocity[8, 15] read by S1[t=7, i=16] scheduled earlier (at nt=16, nx=32, as hinted)

accepted: Schedule(acoustic, target='c').affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i] }).tile(a,b,4,4)
  loop nest: a_outer b_outer a_inner b_inner
  decided  isl  affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i] }) renames the instances of acoustic one for one
  decided  isl  the order after affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i] }) runs every dependence of acoustic forward
  decided  isl  tile(a,b,4,4) renames the instances of acoustic one for one
  decided  isl  the order after tile(a,b,4,4) runs every dependence of acoustic forward

  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested
```

### lanky check examples/wavefront_acoustic.py

Eight accesses, every one decided by isl over the domain the `when` guard
narrows: `velocity[t + 1, i - 1]` is in bounds because the guard keeps `i > 0`,
and both writes at `t + 1` because it keeps `t + 1 < nt`.

```console
$ uv run lanky check examples/wavefront_acoustic.py
STATUS   BY           WHERE                      OWNER     STATEMENT
-------  -----------  -------------------------  --------  ------------------------------------------------------------
decided  isl          wavefront_acoustic.py:111  acoustic  velocity[t + 1, i] is in bounds for every instance of S0
decided  isl          wavefront_acoustic.py:111  acoustic  velocity[t, i] is in bounds for every instance of S0
decided  isl          wavefront_acoustic.py:111  acoustic  pressure[t, i + 1] is in bounds for every instance of S0
decided  isl          wavefront_acoustic.py:111  acoustic  pressure[t, i] is in bounds for every instance of S0
decided  isl          wavefront_acoustic.py:114  acoustic  pressure[t + 1, i] is in bounds for every instance of S1
decided  isl          wavefront_acoustic.py:114  acoustic  pressure[t, i] is in bounds for every instance of S1
decided  isl          wavefront_acoustic.py:114  acoustic  velocity[t + 1, i] is in bounds for every instance of S1
decided  isl          wavefront_acoustic.py:114  acoustic  velocity[t + 1, i - 1] is in bounds for every instance of S1
decided  isl          wavefront_acoustic.py:111  acoustic  distinct instances of S0 write distinct cells of velocity
decided  isl          wavefront_acoustic.py:114  acoustic  distinct instances of S1 write distinct cells of pressure
decided  isl          wavefront_acoustic.py:99   acoustic  the source order runs every dependence forward in time
tested   interpreter  wavefront_acoustic.py:99   acoustic  the traced term computes what the body computes

12 facts: 11 decided, 1 tested
```

### loopty run examples/wavefront_acoustic.py

Three schedules of one kernel, the wavefront block, the diamond and the
diamond tiling, with two outputs each, all compared with the native run. Each
schedule keeps its own facts in the one ledger, because a fact's id names the
schedule it is about and not only the kernel.

```console
$ uv run loopty run examples/wavefront_acoustic.py
acoustic: Schedule(acoustic, target='c').skew(i, by='t').tile(t,i,4,8)
  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested
acoustic: Schedule(acoustic, target='c').affine({ [t, i] -> [a = t + i, b = t - i] })
  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested
acoustic: Schedule(acoustic, target='c').affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i] }).tile(a,b,4,4)
  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE                      OWNER     STATEMENT
-------  -----  -------------------------  --------  ------------------------------------------------------------------------
decided  isl    wavefront_acoustic.py:111  acoustic  skew(i, by='t') renames the instances of acoustic one for one
decided  isl    wavefront_acoustic.py:111  acoustic  the order after skew(i, by='t') runs every dependence of acoustic for...
decided  isl    wavefront_acoustic.py:111  acoustic  tile(t,i,4,8) renames the instances of acoustic one for one
decided  isl    wavefront_acoustic.py:111  acoustic  the order after tile(t,i,4,8) runs every dependence of acoustic forward
tested   loopy  wavefront_acoustic.py:111  acoustic  the scheduled run of acoustic agrees with the native run to the accur...
decided  isl    wavefront_acoustic.py:111  acoustic  affine({ [t, i] -> [a = t + i, b = t - i] }) renames the instances of...
decided  isl    wavefront_acoustic.py:111  acoustic  the order after affine({ [t, i] -> [a = t + i, b = t - i] }) runs eve...
tested   loopy  wavefront_acoustic.py:111  acoustic  the scheduled run of acoustic agrees with the native run to the accur...
decided  isl    wavefront_acoustic.py:111  acoustic  affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, ...
decided  isl    wavefront_acoustic.py:111  acoustic  the order after affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i]...
decided  isl    wavefront_acoustic.py:111  acoustic  tile(a,b,4,4) renames the instances of acoustic one for one
decided  isl    wavefront_acoustic.py:111  acoustic  the order after tile(a,b,4,4) runs every dependence of acoustic forward
tested   loopy  wavefront_acoustic.py:111  acoustic  the scheduled run of acoustic agrees with the native run to the accur...

13 facts: 10 decided, 3 tested
```

## pairs.py

`f: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real]` holds one value per pair of
particles, and nothing else: the triangle is the array's type, not a mask over
a matrix. One statement writes every pair once, over `f.dom` and `f.dom[i]`,
and a second sums for every particle its row, `f[p, j]` for `j < p`, and its
column, `f[k, p]` for `k > p`, which is the half of the symmetry the triangle
does not store.

Every in-bounds row of the ledger is `decided` by isl over the exact triangle.
The column read is the one to look at: `(k, p)` is a point of `f` because the
reduction's condition says `k > p`, and isl is asked about the triangle, not
about the `n x n` box around it, so `f[p, p]` would be refused although the box
has the cell (`tests/test_domains.py` pins that).

The two schedules differ in one step. `Schedule(pairs)` keeps `f` in the box of
its binders, 36 cells for 15 pairs; `Schedule(pairs).pack('f')` keeps the 15
cells row after row and reads `f[i, j]` as `f[off_f[i] + j]`, through the
table of row starts the run prints. `pack` is not a cast and adds no fact: a
layout says where a cell is kept, and every fact is about the cells. Both
compiled runs agree with the native one, which ran on the same layouts.

### python examples/pairs.py

```console
$ uv run python examples/pairs.py
6 particles, 15 pairs
f box: 36 cells for 15 pairs
f packed: 15 cells for 15 pairs, and a table of row starts [0, 0, 1, 3, 6, 10]
energies = [5.402 0.965 3.771 2.404 2.019 0.928]
dense    = [5.402 0.965 3.771 2.404 2.019 0.928]
both layouts agree with the dense reference: True

schedule: Schedule(pairs, target='c')
  f: difference 0 within 1.04e-06 (approx) -> tested
  e: difference 0 within 1.93e-06 (approx) -> tested

schedule: Schedule(pairs, target='c').pack(f)
  f: difference 0 within 1.04e-06 (approx) -> tested
  e: difference 0 within 1.93e-06 (approx) -> tested
```

### lanky check examples/pairs.py

```console
$ uv run lanky check examples/pairs.py
STATUS   BY           WHERE        OWNER  STATEMENT
-------  -----------  -----------  -----  -------------------------------------------------------
decided  isl          pairs.py:78  pairs  f[i, j] is in bounds for every instance of S0
decided  isl          pairs.py:78  pairs  q[i] is in bounds for every instance of S0
decided  isl          pairs.py:78  pairs  q[j] is in bounds for every instance of S0
decided  isl          pairs.py:78  pairs  x[i] is in bounds for every instance of S0
decided  isl          pairs.py:78  pairs  x[j] is in bounds for every instance of S0
decided  isl          pairs.py:78  pairs  y[i] is in bounds for every instance of S0
decided  isl          pairs.py:78  pairs  y[j] is in bounds for every instance of S0
decided  isl          pairs.py:80  pairs  e[p] is in bounds for every instance of S1
decided  isl          pairs.py:80  pairs  f[p, j] is in bounds for every instance of S1
decided  isl          pairs.py:80  pairs  f[k, p] is in bounds for every instance of S1
decided  isl          pairs.py:78  pairs  distinct instances of S0 write distinct cells of f
decided  isl          pairs.py:80  pairs  distinct instances of S1 write distinct cells of e
decided  isl          pairs.py:80  pairs  every cell of f that S1 reads, S0 wrote before the read
decided  isl          pairs.py:65  pairs  the source order runs every dependence forward in time
decided  type         pairs.py:80  pairs  the accumulation into e[p] over j is approx
decided  type         pairs.py:80  pairs  the accumulation into e[p] over k is approx
tested   interpreter  pairs.py:65  pairs  the traced term computes what the body computes

17 facts: 16 decided, 1 tested
```

### loopty run examples/pairs.py

```console
$ uv run loopty run examples/pairs.py
pairs: Schedule(pairs, target='c')
  f: difference 0 within 1.04e-06 (approx) -> tested
  e: difference 0 within 1.93e-06 (approx) -> tested
pairs: Schedule(pairs, target='c').pack(f)
  f: difference 0 within 1.04e-06 (approx) -> tested
  e: difference 0 within 1.93e-06 (approx) -> tested

STATUS  BY     WHERE        OWNER  STATEMENT
------  -----  -----------  -----  ------------------------------------------------------------------------
tested  loopy  pairs.py:78  pairs  the scheduled run of pairs agrees with the native run to the accuracy...
tested  loopy  pairs.py:78  pairs  the scheduled run of pairs agrees with the native run to the accuracy...

2 facts: 2 tested
```

## composition.py

A Burgers right-hand side in two kernels, `flux` and then `divergence`, and the
program that runs them, `burgers_rhs`. The flux goes into an array the program
makes with `Arr.zeros_like(u)`.

Lowered one kernel at a time, that array is an output of `flux` and an input of
`divergence`: a public argument of both, declared two ways. The program's term
is one term, the two kernels' statements in call order in the program's names,
and the array is one of its temporaries: declared inside the one kernel loopy
generates, zeroed where the program made it (`f.zeros`), and passed by nobody.
Nothing declares that `divergence` needs `flux`; the `f` one writes and the
other reads is one array of the term, so the dependence is in the footprints
and orders the two loops. The loops are not fused here: `fusion.py` fuses them,
a cast over this term checked against that dependence, and this run is the
reference it is compared with. The program also has a `definedness` fact:
every cell of `f` that `divergence` reads, `flux` stored before it, so the
zeros `f` was made with reach no read. A program that does read them, as
padding at a boundary, has the fact decided too, stated with the zeros, and
its reason lists the cells:
`shifted.S0 reads the zeros at f[0] and f[n - 1]`.

### python examples/composition.py

```console
$ uv run python examples/composition.py
native: the program agrees with the numpy slices: True

the term of burgers_rhs(u, rhs):
  temporary f: 1 axis, element Real
  f.zeros        over i_0 from composition.py:80
  flux.S0        over j   from composition.py:66
  divergence.S0  over i   from composition.py:74

#include <stdint.h>
#include <stdbool.h>

void burgers_rhs(int32_t const n, double const *__restrict__ u, double *__restrict__ rhs, double *__restrict__ f)
{
  for (int32_t i_0 = 0; i_0 <= -1 + n; ++i_0)
    f[i_0] = (double) (0.0);
  for (int32_t j = 0; j <= -1 + n; ++j)
    f[j] = 0.5 * u[j] * u[j];
  for (int32_t i = 1; i <= -2 + n; ++i)
    if (i > 0 && i + 1 < n)
      rhs[i] = (-1.0 * (f[1 + i] + -1.0 * f[-1 + i])) / 2.0;
}

  rhs: difference 0 within 1e-06 (approx) -> tested
```

### lanky check examples/composition.py

The two kernels' obligations, and the program's one fact: neither kernel
states a postcondition for it to restate, and its last row, `tested` by
`interpreter`, says that its term, the two calls composed, computes what its
body computes.

```console
$ uv run lanky check examples/composition.py
STATUS   BY           WHERE              OWNER        STATEMENT
-------  -----------  -----------------  -----------  ------------------------------------------------------------
decided  isl          composition.py:66  flux         f[j] is in bounds for every instance of S0
decided  isl          composition.py:66  flux         u[j] is in bounds for every instance of S0
decided  isl          composition.py:66  flux         distinct instances of S0 write distinct cells of f
decided  isl          composition.py:62  flux         the source order runs every dependence forward in time
tested   interpreter  composition.py:62  flux         the traced term computes what the body computes
decided  isl          composition.py:74  divergence   rhs[i] is in bounds for every instance of S0
decided  isl          composition.py:74  divergence   f[i + 1] is in bounds for every instance of S0
decided  isl          composition.py:74  divergence   f[i - 1] is in bounds for every instance of S0
decided  isl          composition.py:74  divergence   distinct instances of S0 write distinct cells of rhs
decided  isl          composition.py:69  divergence   the source order runs every dependence forward in time
tested   interpreter  composition.py:69  divergence   the traced term computes what the body computes
decided  isl          composition.py:82  burgers_rhs  every cell of f that divergence reads, flux stored before it
tested   interpreter  composition.py:77  burgers_rhs  the traced term computes what the body computes

13 facts: 10 decided, 3 tested
```

### loopty run examples/composition.py

Each kernel alone, and then the program as one kernel. On the C target the
executor allocates the program's array on the heap and passes it after the
program's own arguments; see note 16 in `../docs/loopy-notes.md`.

```console
$ uv run loopty run examples/composition.py
flux: Schedule(flux, target='c')
  f: difference 0 within 1e-06 (approx) -> tested
divergence: Schedule(divergence, target='c')
  rhs: difference 0 within 1e-06 (approx) -> tested
burgers_rhs: Schedule(burgers_rhs, target='c')
  rhs: difference 0 within 1e-06 (approx) -> tested

STATUS  BY     WHERE              OWNER        STATEMENT
------  -----  -----------------  -----------  ------------------------------------------------------------------------
tested  loopy  composition.py:66  flux         the scheduled run of flux agrees with the native run to the accuracy ...
tested  loopy  composition.py:74  divergence   the scheduled run of divergence agrees with the native run to the acc...
tested  loopy  composition.py:77  burgers_rhs  the scheduled run of burgers_rhs agrees with the native run to the ac...

3 facts: 3 tested
```

## travel.py

Facts that travel between the calls of a program. A kernel's requirements on
its inputs are its argument types: `gather` reads `x[perm[i]]`, in bounds by
the element type of `perm`, and natively its contract checks that every cell
of `perm` is a point of `Fin[n]` when it is called. A program is one compiled
call, whose contract checks its arguments when it starts, so where an earlier
call wrote `perm`, the check is a `requirement` of the program: decided by isl
under what held at the call, or made by the compiled program between the two
calls.

Four programs. In `permuted` (#65's), `number` writes a permutation and says
what it writes, and `gather`'s requirement is decided under that
postcondition. In `through`, `scan` computes the offsets `rowsums` reads its
rows through, and the layout requirement is `scan`'s postcondition verbatim;
the program cites `scan_monotone` with `uses=`, and the fact does not rest on
it, because the requirement does not need it. In `checked`, `number_quiet`
writes the same permutation and says nothing, so the compiled program checks
`perm` between the calls, and stores it in 64 bits, as the native run does,
so that the check reads what was written and not what a 32-bit store would
have narrowed it to. In `flat`, `weigh` reads a flat buffer,
`val[off[r] + j]`, after a scan: alone its in-bounds fact is `assumed`, and in
the program it is decided under the scan's postcondition and the element type
of `off`, which the program checks, since nothing says the scan's offsets stay
below `nnz`.

The compiled program skips a decided requirement's check only on the strength
of what a run has borne out: the postconditions here are `tested` against
their kernels' native runs, and so are the kernels' terms against their
bodies. A requirement decided under a postcondition its kernel's runs refute,
under one of a kernel whose term does not compute what its body does, or
under an axiom, stays `decided` in the ledger, worth what it rests on, and is
checked when the program runs all the same.

### python examples/travel.py

Each program natively, how each of its requirements was met, and its compiled
run against its native one; then the code of `checked`, whose check sets a
flag that guards the call after it.

```console
$ uv run python examples/travel.py
permuted:
  the elements of perm are points of Fin(n) where gather is called at travel.py:99, after number at travel.py:98 wrote perm
    decided under the postcondition of number, after number at travel.py:98
  perm: difference 0 within 0 (exact) -> tested
  y: difference 0 within 1.1e-05 (approx) -> tested
through:
  off holds the offsets the counts in cnt give the rows of val (off[0] == 0 and off[r + 1] == off[r] + cnt[r]) where rowsums is called at travel.py:152, after scan at travel.py:151 wrote off
    decided under the postcondition of scan, after scan at travel.py:151
  off: difference 0 within 0 (exact) -> tested
  y: difference 0 within 1e-06 (approx) -> tested
checked:
  the elements of perm are points of Fin(n) where gather is called at travel.py:106, after number_quiet at travel.py:105 wrote perm
    checked when it runs
  perm: difference 0 within 0 (exact) -> tested
  y: difference 0 within 1.1e-05 (approx) -> tested
flat:
  off holds the offsets the counts in cnt give the rows of wt (off[0] == 0 and off[r + 1] == off[r] + cnt[r]) where weigh is called at travel.py:193, after scan_flat at travel.py:192 wrote off
    decided under the postcondition of scan_flat, after scan_flat at travel.py:192
  the elements of off are points of Fin(nnz + 1) where weigh is called at travel.py:193, after scan_flat at travel.py:192 wrote off
    checked when it runs
  val[off[r_0] + j] is in bounds for every instance of weigh.S0, where weigh is called at travel.py:193
    decided under the postcondition of scan_flat, after scan_flat at travel.py:192, the requirement on off where weigh is called
  off: difference 0 within 0 (exact) -> tested
  y: difference 0 within 1e-06 (approx) -> tested

number's postcondition, forall i in Fin(n). perm[i] == n - 1 - i: tested

#pragma STDC FP_CONTRACT OFF
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize ("fp-contract=off")
#endif
#include <stdint.h>
#include <stdbool.h>

void checked(int32_t const n, int64_t *__restrict__ perm, double const *__restrict__ x, double *__restrict__ y, int64_t *__restrict__ gather_perm_ok)
{
  for (int32_t i = 0; i <= -1 + n; ++i)
    perm[i] = (int64_t) (n + -1 + -1 * i);
  for (int32_t i_1 = 0; i_1 <= -1 + n; ++i_1)
    if ((perm[i_1] < 0 || perm[i_1] >= n))
      gather_perm_ok[0] = (int64_t) (1);
  if (gather_perm_ok[0] == 0)
    for (int32_t i_0 = 0; i_0 <= -1 + n; ++i_0)
      y[i_0] = x[perm[i_0]];
}
```

### lanky check examples/travel.py

The postconditions are `tested` by `native`, and each requirement the
hypotheses decide is `decided` by `isl` and worth `tested`, what the
postcondition it rests on is worth. The two the compiled programs check are
`assumed`, and say why in their provenance. The flat access of `flat` is
`decided`, under the element requirement it used, which the `STATUS` column
names, so it is worth an assumption checked when the program runs.

```console
$ uv run lanky check examples/travel.py
STATUS                                                       EFFECTIVE  BY             WHERE          OWNER          STATEMENT
-----------------------------------------------------------  ---------  -------------  -------------  -------------  ------------------------------------------------------------------------
tested                                                       tested     native         travel.py:72   number         forall i in Fin(n). perm[i] == n - 1 - i
decided                                                      decided    type           travel.py:92   gather         x[perm[i]] is in bounds by type (perm[i] : Fin(n))
decided                                                      tested     call           travel.py:95   permuted       after number(...) in permuted: forall i in Fin(n). perm[i] == n - 1 - i
decided                                                      tested     isl            travel.py:99   permuted       the elements of perm are points of Fin(n) where gather is called at t...
assumed                                                      assumed    -              travel.py:106  checked        the elements of perm are points of Fin(n) where gather is called at t...
tested                                                       tested     native         travel.py:115  scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
decided                                                      tested     call           travel.py:148  through        after scan(...) in through: off[0] == 0 and (forall r in Fin(n). off[...
decided                                                      tested     isl            travel.py:152  through        off holds the offsets the counts in cnt give the rows of val (off[0] ...
assumed                                                      assumed    -              travel.py:186  weigh          val[off[r] + j] is in bounds
decided                                                      tested     call           travel.py:189  flat           after scan_flat(...) in flat: off[0] == 0 and (forall r in Fin(n). of...
decided                                                      tested     isl            travel.py:193  flat           off holds the offsets the counts in cnt give the rows of wt (off[0] =...
assumed                                                      assumed    -              travel.py:193  flat           the elements of off are points of Fin(nnz + 1) where weigh is called ...
decided under requirement:travel.flat@189:weigh:element:off  assumed    isl            travel.py:186  flat           val[off[r_0] + j] is in bounds for every instance of weigh.S0, where ...
...
73 facts: 4 assumed, 53 decided, 16 tested
```

### loopty run examples/travel.py

Every kernel alone, and every program as one kernel, compiled and compared
with its native run.

```console
$ uv run loopty run examples/travel.py
number: Schedule(number, target='c')
  perm: difference 0 within 0 (exact) -> tested
number_quiet: Schedule(number_quiet, target='c')
  perm: difference 0 within 0 (exact) -> tested
gather: Schedule(gather, target='c')
  y: difference 0 within 1.1e-05 (approx) -> tested
permuted: Schedule(permuted, target='c')
  perm: difference 0 within 0 (exact) -> tested
  y: difference 0 within 1.1e-05 (approx) -> tested
checked: Schedule(checked, target='c')
  perm: difference 0 within 0 (exact) -> tested
  y: difference 0 within 1.1e-05 (approx) -> tested
scan: Schedule(scan, target='c')
  off: difference 0 within 0 (exact) -> tested
rowsums: Schedule(rowsums, target='c')
  y: difference 0 within 1e-06 (approx) -> tested
through: Schedule(through, target='c')
  off: difference 0 within 0 (exact) -> tested
  y: difference 0 within 1e-06 (approx) -> tested
scan_flat: Schedule(scan_flat, target='c')
  off: difference 0 within 0 (exact) -> tested
weigh: Schedule(weigh, target='c')
  y: difference 0 within 1e-06 (approx) -> tested
flat: Schedule(flat, target='c')
  off: difference 0 within 0 (exact) -> tested
  y: difference 0 within 1e-06 (approx) -> tested

STATUS  BY     WHERE          OWNER         STATEMENT
------  -----  -------------  ------------  ------------------------------------------------------------------------
tested  loopy  travel.py:76   number        the scheduled run of number agrees with the native run to the accurac...
tested  loopy  travel.py:83   number_quiet  the scheduled run of number_quiet agrees with the native run to the a...
tested  loopy  travel.py:92   gather        the scheduled run of gather agrees with the native run to the accurac...
tested  loopy  travel.py:95   permuted      the scheduled run of permuted agrees with the native run to the accur...
tested  loopy  travel.py:102  checked       the scheduled run of checked agrees with the native run to the accura...
tested  loopy  travel.py:120  scan          the scheduled run of scan agrees with the native run to the accuracy ...
tested  loopy  travel.py:145  rowsums       the scheduled run of rowsums agrees with the native run to the accura...
tested  loopy  travel.py:148  through       the scheduled run of through agrees with the native run to the accura...
tested  loopy  travel.py:166  scan_flat     the scheduled run of scan_flat agrees with the native run to the accu...
tested  loopy  travel.py:186  weigh         the scheduled run of weigh agrees with the native run to the accuracy...
tested  loopy  travel.py:189  flat          the scheduled run of flat agrees with the native run to the accuracy ...

11 facts: 11 tested
```

## fusion.py

The program of `composition.py`, `burgers_rhs`, scheduled two steps further:
the Burgers flux and divergence of the closed #6, which fused the two kernels
with loopy's `fuse_kernels` and `assignment_to_subst` and checked neither.
Both steps are checked here before they are applied.

`fuse("flux", "divergence", shift=1)` runs `divergence`'s loop inside
`flux`'s, one step behind. It is `affine` with a map per statement, `{
flux_S0[j] -> [j]; divergence_S0[i] -> [j] : j = i + 1 }`, and isl decides it
on the dependences between the two calls: in each step of the one loop `flux`
stores `f[j]` first, and `divergence` reads `f[j]` and `f[j - 2]`, both
stored by then. Without the shift `divergence` at `i` would read `f[i + 1]` a
step before `flux` stores it, and that is the pair the refusal names, with
the shift that is accepted.

`substitute("f")` then stores no flux at all: `flux`'s statement becomes a
substitution rule, and `divergence` computes `0.5 * u[j] * u[j]` where it read
`f[j]`. The step is legal because every cell of `f` that is read, `flux`
stored before the read (the `definedness` fact), and nothing writes `u` in
between (the `monotone` fact, over the program as it now runs). The temporary
goes, and with it the loop that zeroed it.

### python examples/fusion.py

The refusal, then each accepted schedule with its facts, the code loopy
generates for it, and its compiled run against the native one.

```console
$ uv run python examples/fusion.py
native: the program agrees with the numpy slices: True

refused: fuse(flux, divergence) illegal: instance flux.S0[j=2] writes f[2] read by divergence.S0[i=1] scheduled earlier (at n=16, as hinted); fuse('flux', 'divergence', shift=1) runs every dependence between them forward

accepted: Schedule(burgers_rhs, target='c').fuse(flux, divergence, shift=1)
  decided  isl  fuse(flux, divergence, shift=1) renames the instances of burgers_rhs one for one
  decided  isl  the order after fuse(flux, divergence, shift=1) runs every dependence of burgers_rhs forward

#include <stdint.h>
#include <stdbool.h>

void burgers_rhs(int32_t const n, double const *__restrict__ u, double *__restrict__ rhs, double *__restrict__ f)
{
  for (int32_t i_0 = 0; i_0 <= -1 + n; ++i_0)
    f[i_0] = (double) (0.0);
  for (int32_t j = 0; j <= -1 + n; ++j)
  {
    f[j] = 0.5 * u[j] * u[j];
    if (-2 + j >= 0 && -1 + j > 0 && -1 + j + 1 < n)
      rhs[-1 + j] = (-1.0 * (f[j] + -1.0 * f[-2 + j])) / 2.0;
  }
}

  rhs: difference 0 within 1e-06 (approx) -> tested

accepted: Schedule(burgers_rhs, target='c').fuse(flux, divergence, shift=1).substitute('f')
  decided  isl  fuse(flux, divergence, shift=1) renames the instances of burgers_rhs one for one
  decided  isl  the order after fuse(flux, divergence, shift=1) runs every dependence of burgers_rhs forward
  decided  isl  every cell of f that burgers_rhs reads, flux.S0 has stored by the time it is read
  decided  isl  substitute('f') renames the instances of burgers_rhs one for one
  decided  isl  the order after substitute('f') runs every dependence of burgers_rhs forward

#include <stdint.h>
#include <stdbool.h>

void burgers_rhs(int32_t const n, double const *__restrict__ u, double *__restrict__ rhs)
{
  for (int32_t j = 0; j <= -1 + n; ++j)
    if (-2 + j >= 0 && -1 + j > 0 && -1 + j + 1 < n)
      rhs[-1 + j] = (-1.0 * (0.5 * u[j] * u[j] + -1.0 * (0.5 * u[-2 + j] * u[-2 + j]))) / 2.0;
}

  rhs: difference 0 within 1e-06 (approx) -> tested
```

### lanky check examples/fusion.py

The kernels' obligations, and the program's: the definedness of `f`, decided
by isl, and the program's `trace-faithful` fact. The schedules' facts are in
the ledger `loopty run` prints.

```console
$ uv run lanky check examples/fusion.py
STATUS   BY           WHERE         OWNER        STATEMENT
-------  -----------  ------------  -----------  ------------------------------------------------------------
decided  isl          fusion.py:71  flux         f[j] is in bounds for every instance of S0
decided  isl          fusion.py:71  flux         u[j] is in bounds for every instance of S0
decided  isl          fusion.py:71  flux         distinct instances of S0 write distinct cells of f
decided  isl          fusion.py:67  flux         the source order runs every dependence forward in time
tested   interpreter  fusion.py:67  flux         the traced term computes what the body computes
decided  isl          fusion.py:79  divergence   rhs[i] is in bounds for every instance of S0
decided  isl          fusion.py:79  divergence   f[i + 1] is in bounds for every instance of S0
decided  isl          fusion.py:79  divergence   f[i - 1] is in bounds for every instance of S0
decided  isl          fusion.py:79  divergence   distinct instances of S0 write distinct cells of rhs
decided  isl          fusion.py:74  divergence   the source order runs every dependence forward in time
tested   interpreter  fusion.py:74  divergence   the traced term computes what the body computes
decided  isl          fusion.py:87  burgers_rhs  every cell of f that divergence reads, flux stored before it
tested   interpreter  fusion.py:82  burgers_rhs  the traced term computes what the body computes

13 facts: 10 decided, 3 tested
```

### loopty run examples/fusion.py

Both schedules, compiled and compared with the native run, and each kernel
alone. The two schedules share their first step, and so its facts.

```console
$ uv run loopty run examples/fusion.py
burgers_rhs: Schedule(burgers_rhs, target='c').fuse(flux, divergence, shift=1)
  rhs: difference 0 within 1e-06 (approx) -> tested
burgers_rhs: Schedule(burgers_rhs, target='c').fuse(flux, divergence, shift=1).substitute('f')
  rhs: difference 0 within 1e-06 (approx) -> tested
flux: Schedule(flux, target='c')
  f: difference 0 within 1e-06 (approx) -> tested
divergence: Schedule(divergence, target='c')
  rhs: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE         OWNER        STATEMENT
-------  -----  ------------  -----------  ------------------------------------------------------------------------
decided  isl    fusion.py:85  burgers_rhs  fuse(flux, divergence, shift=1) renames the instances of burgers_rhs ...
decided  isl    fusion.py:85  burgers_rhs  the order after fuse(flux, divergence, shift=1) runs every dependence...
tested   loopy  fusion.py:82  burgers_rhs  the scheduled run of burgers_rhs agrees with the native run to the ac...
decided  isl    fusion.py:85  burgers_rhs  every cell of f that burgers_rhs reads, flux.S0 has stored by the tim...
decided  isl    fusion.py:85  burgers_rhs  substitute('f') renames the instances of burgers_rhs one for one
decided  isl    fusion.py:85  burgers_rhs  the order after substitute('f') runs every dependence of burgers_rhs ...
tested   loopy  fusion.py:82  burgers_rhs  the scheduled run of burgers_rhs agrees with the native run to the ac...
tested   loopy  fusion.py:71  flux         the scheduled run of flux agrees with the native run to the accuracy ...
tested   loopy  fusion.py:79  divergence   the scheduled run of divergence agrees with the native run to the acc...

9 facts: 5 decided, 4 tested
```

## quadrature.py

The element kernel FFC and TSFC generate for the P1 mass matrix, `mass_ffc` of
firedrake's PyOP2 tests, applied to a field on every cell of a mesh. Its
3-point rule and the basis at the rule's points are constant tables, declared
next to the kernel as the numpy arrays they are, `W = table([1 / 6, 1 / 6, 1 /
6])`, and read by the body as arrays are. Each read is in bounds over the
table's shape, decided by isl, and the compiled kernel holds the values with
every digit, `static double W[3]`, so both runs read the same numbers. The
gathers through the cell map, `u[cell[c, k]]`, are in bounds by type, since
the map's entries are points of `Fin[nv]`.

### python examples/quadrature.py

Both meshes against the element mass matrix in closed form, then the tables
as the compiled code holds them, and the compiled run of PyOP2's split of the
cells against the native one.

```console
$ uv run python examples/quadrature.py
PyOP2's two triangles: 2 cells, agrees with the closed form: True
a square: 32 cells, agrees with the closed form: True

static double PHI[3 * 3] = { 0.6666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.6666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.6666666666666666 };
static double W[3] = { 0.16666666666666666, 0.16666666666666666, 0.16666666666666666 };

schedule: Schedule(mass_action, target='c').split(c, 32)
  yl: difference 0 within 1.01e-06 (approx) -> tested
```

### lanky check examples/quadrature.py

```console
$ uv run lanky check examples/quadrature.py
STATUS   BY           WHERE             OWNER        STATEMENT
-------  -----------  ----------------  -----------  ---------------------------------------------------------
decided  isl          quadrature.py:78  mass_action  yl[c, j] is in bounds for every instance of S0
decided  type         quadrature.py:78  mass_action  X[cell[c, 1]] is in bounds by type (cell[c, 1] : Fin(nv))
decided  isl          quadrature.py:78  mass_action  cell[c, 1] is in bounds for every instance of S0
decided  type         quadrature.py:78  mass_action  X[cell[c, 0]] is in bounds by type (cell[c, 0] : Fin(nv))
decided  isl          quadrature.py:78  mass_action  cell[c, 0] is in bounds for every instance of S0
decided  type         quadrature.py:78  mass_action  Y[cell[c, 2]] is in bounds by type (cell[c, 2] : Fin(nv))
decided  isl          quadrature.py:78  mass_action  cell[c, 2] is in bounds for every instance of S0
decided  type         quadrature.py:78  mass_action  Y[cell[c, 0]] is in bounds by type (cell[c, 0] : Fin(nv))
decided  type         quadrature.py:78  mass_action  X[cell[c, 2]] is in bounds by type (cell[c, 2] : Fin(nv))
decided  type         quadrature.py:78  mass_action  Y[cell[c, 1]] is in bounds by type (cell[c, 1] : Fin(nv))
decided  isl          quadrature.py:78  mass_action  W[q] is in bounds for every instance of S0
decided  isl          quadrature.py:78  mass_action  PHI[q, j] is in bounds for every instance of S0
decided  isl          quadrature.py:78  mass_action  PHI[q, k] is in bounds for every instance of S0
decided  type         quadrature.py:78  mass_action  u[cell[c, k]] is in bounds by type (cell[c, k] : Fin(nv))
decided  isl          quadrature.py:78  mass_action  cell[c, k] is in bounds for every instance of S0
decided  isl          quadrature.py:78  mass_action  distinct instances of S0 write distinct cells of yl
decided  isl          quadrature.py:63  mass_action  the source order runs every dependence forward in time
decided  type         quadrature.py:78  mass_action  the accumulation into yl[c, j] over q, k is approx
tested   interpreter  quadrature.py:63  mass_action  the traced term computes what the body computes

19 facts: 18 decided, 1 tested
```

### loopty run examples/quadrature.py

```console
$ uv run loopty run examples/quadrature.py
mass_action: Schedule(mass_action, target='c').split(c, 32)
  yl: difference 0 within 1.01e-06 (approx) -> tested

STATUS   BY     WHERE             OWNER        STATEMENT
-------  -----  ----------------  -----------  ------------------------------------------------------------------------
decided  isl    quadrature.py:78  mass_action  split(c, 32) renames the instances of mass_action one for one
decided  isl    quadrature.py:78  mass_action  the order after split(c, 32) runs every dependence of mass_action for...
tested   loopy  quadrature.py:78  mass_action  the scheduled run of mass_action agrees with the native run to the ac...

3 facts: 2 decided, 1 tested
```

## layer_potential.py

The QBX single layer pytential evaluates at a target on a curve, through the
line-Taylor expansion of `log` about a center just off the curve, written over
complex points with `real` and `log` from `loopty.math`. On the unit circle,
for the density `cos(theta)`, the answer is `cos(theta) / 2`.

The schedule is sumpy's CPU tile, `LayerPotential.get_optimized_kernel`: the
targets split by 16, the sources by 256, and the sources' outer loop outside
the targets' inner loop, `.prioritize("s_outer", "t_inner")`. Each target's
sum still adds its sources in order. On the C target the 16 targets are an
`ilp` loop, which gives each an accumulator of its own; the same order on a
target loop in sequence is one loopy cannot keep, and the schedule's
`buildable` fact says so with the tag that fixes it.

### python examples/layer_potential.py

```console
$ uv run python examples/layer_potential.py
32 targets, 512 sources, order 4
largest error against cos(theta) / 2: 3.0e-10

schedule: Schedule(single_layer, target='c').split(t, 16).tag(t_inner='ilp').split(s, 256).prioritize(s_outer, t_inner)
  decided  isl  split(t, 16) renames the instances of single_layer one for one
  decided  isl  the order after split(t, 16) runs every dependence of single_layer forward
  decided  isl  tag(t_inner='ilp') renames the instances of single_layer one for one
  decided  isl  the order after tag(t_inner='ilp') runs every dependence of single_layer forward
  decided  isl  split(s, 256) renames the instances of single_layer one for one
  decided  isl  the order after split(s, 256) runs every dependence of single_layer forward
  decided  isl  prioritize(s_outer, t_inner) renames the instances of single_layer one for one
  decided  isl  the order after prioritize(s_outer, t_inner) runs every dependence of single_layer forward
  out: the compiled run agrees with the native run to the approx tolerance: True -> tested

schedule: Schedule(single_layer, target='c').split(t, 16).split(s, 256).prioritize(s_outer, t_inner)
  buildable: False
  reason: the loop s_outer of the sum over s_outer, s_inner in S0 is ordered outside t_inner, which runs in sequence: loopy keeps one accumulator for the sum, so it would run the sum inside every loop of S0 anyway, in another order than the one this schedule's facts were checked for. Tag t_inner 'ilp', tag(t_inner='ilp'), which gives each of its iterations an accumulator of its own, or on a device put it on a hardware axis, tag(t_inner='l.0')
```

### lanky check examples/layer_potential.py

```console
$ uv run lanky check examples/layer_potential.py
STATUS   BY           WHERE                  OWNER         STATEMENT
-------  -----------  ---------------------  ------------  ------------------------------------------------------
decided  isl          layer_potential.py:88  single_layer  out[t] is in bounds for every instance of S0
decided  isl          layer_potential.py:88  single_layer  w[s] is in bounds for every instance of S0
decided  isl          layer_potential.py:88  single_layer  c[t] is in bounds for every instance of S0
decided  isl          layer_potential.py:88  single_layer  y[s] is in bounds for every instance of S0
decided  isl          layer_potential.py:88  single_layer  z[t] is in bounds for every instance of S0
decided  isl          layer_potential.py:88  single_layer  distinct instances of S0 write distinct cells of out
decided  isl          layer_potential.py:78  single_layer  the source order runs every dependence forward in time
decided  type         layer_potential.py:88  single_layer  the accumulation into out[t] over s is approx
tested   interpreter  layer_potential.py:78  single_layer  the traced term computes what the body computes

9 facts: 8 decided, 1 tested
```

### loopty run examples/layer_potential.py

`uv run loopty run examples/layer_potential.py` compiles the tile on the C
target and compares it with the native run, as for every demo. Its transcript
is not kept here: the difference it prints is one of `log`'s last bits,
numpy's against the C library's, which can differ from machine to machine.

## sum_factorization.py

The weak derivative along `r` on quadrilaterals, as grudge applies it, a pass
per reference axis: the differentiation matrix along `r`, then the quadrature
weights along `s`, each pass a kernel over the elements and a program running
one after the other. Fused on every loop, the second pass would read a cell of
the row before the first stores it, and the refusal names that pair and the
fusion of fewer loops it checked, `fuse("along_r", "along_s", loops=2)`, which
fuses the element and the row and leaves the inner loops one after the other.
Each element's result is the Kronecker product of the two matrices applied to
it.

### python examples/sum_factorization.py

```console
$ uv run python examples/sum_factorization.py
native: each element is kron(D, M) applied to it: True

refused: fuse(along_r, along_s) illegal: instance along_r.S0[e=0, a=1, b=1] writes v[0, 1, 1] read by along_s.S0[e_0=0, a_0=1, b_0=0] scheduled earlier (at n=2, ne=1, isl's choice); fuse('along_r', 'along_s', loops=2) runs every dependence between them forward

accepted: Schedule(weak_derivative, target='c').fuse(along_r, along_s, loops=2)
  the passes' loops, outermost first: e, a, b, b_0
  decided  isl  fuse(along_r, along_s, loops=2) renames the instances of weak_derivative one for one
  decided  isl  the order after fuse(along_r, along_s, loops=2) runs every dependence of weak_derivative forward
  w: difference 0 within 1e-06 (approx) -> tested
```

### lanky check examples/sum_factorization.py

```console
$ uv run lanky check examples/sum_factorization.py
STATUS   BY           WHERE                    OWNER            STATEMENT
-------  -----------  -----------------------  ---------------  ------------------------------------------------------------
decided  isl          sum_factorization.py:74  along_r          v[e, a, b] is in bounds for every instance of S0
decided  isl          sum_factorization.py:74  along_r          F[a, m] is in bounds for every instance of S0
decided  isl          sum_factorization.py:74  along_r          u[e, m, b] is in bounds for every instance of S0
decided  isl          sum_factorization.py:74  along_r          distinct instances of S0 write distinct cells of v
decided  isl          sum_factorization.py:64  along_r          the source order runs every dependence forward in time
decided  type         sum_factorization.py:74  along_r          the accumulation into v[e, a, b] over m is approx
tested   interpreter  sum_factorization.py:64  along_r          the traced term computes what the body computes
decided  isl          sum_factorization.py:87  along_s          w[e, a, b] is in bounds for every instance of S0
decided  isl          sum_factorization.py:87  along_s          G[b, m] is in bounds for every instance of S0
decided  isl          sum_factorization.py:87  along_s          v[e, a, m] is in bounds for every instance of S0
decided  isl          sum_factorization.py:87  along_s          distinct instances of S0 write distinct cells of w
decided  isl          sum_factorization.py:77  along_s          the source order runs every dependence forward in time
decided  type         sum_factorization.py:87  along_s          the accumulation into w[e, a, b] over m is approx
tested   interpreter  sum_factorization.py:77  along_s          the traced term computes what the body computes
decided  isl          sum_factorization.py:95  weak_derivative  every cell of v that along_s reads, along_r stored before it
tested   interpreter  sum_factorization.py:90  weak_derivative  the traced term computes what the body computes

16 facts: 13 decided, 3 tested
```

### loopty run examples/sum_factorization.py

```console
$ uv run loopty run examples/sum_factorization.py
weak_derivative: Schedule(weak_derivative, target='c').fuse(along_r, along_s, loops=2)
  w: difference 0 within 1e-06 (approx) -> tested
along_r: Schedule(along_r, target='c')
  v: difference 0 within 1.02e-06 (approx) -> tested
along_s: Schedule(along_s, target='c')
  w: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE                    OWNER            STATEMENT
-------  -----  -----------------------  ---------------  ------------------------------------------------------------------------
decided  isl    sum_factorization.py:93  weak_derivative  fuse(along_r, along_s, loops=2) renames the instances of weak_derivat...
decided  isl    sum_factorization.py:93  weak_derivative  the order after fuse(along_r, along_s, loops=2) runs every dependence...
tested   loopy  sum_factorization.py:90  weak_derivative  the scheduled run of weak_derivative agrees with the native run to th...
tested   loopy  sum_factorization.py:74  along_r          the scheduled run of along_r agrees with the native run to the accura...
tested   loopy  sum_factorization.py:87  along_s          the scheduled run of along_s agrees with the native run to the accura...

5 facts: 2 decided, 3 tested
```

## calls.py

A kernel calls a kernel as Python calls a function. `dot` is a rule: its
return annotation is the sort of its value, `-> Real`, so a call of it is a
value, `y[i] = dot(A[i], x)`, with `A[i]` the row of `A` as in numpy, and a
sparse product is `y[r] = dot(val[r], x[col[r]])`, the row of a ragged array
dotted with the cells of `x` its columns pick. The P1 mass matrix is written
with the quadrature rule, `integrate(f)`, applied to a `lambda` of the
quadrature point, and the kernel over the mesh calls the element's kernel per
cell, `element_mass(X[cell[c]], Y[cell[c]], M[c])`, which writes through the
row `M[c]`.

Traced, a call is inlined: the callee's body runs again in the caller's
trace against views of the caller's arrays, so `dot`'s loop is a sum of the
caller's statement at every `i`, under a name of its own where the caller's
names would clash, and the compiled kernel is one loop nest. What the
callee's contract checks of each call natively, the caller owes at every
point it calls: the row `A[i]` in bounds, the cells `X[cell[c]]` picks in
bounds by type, and, after each call of `element_mass`, its postcondition
restated of `M[c]`, decided by the call and resting on the callee's own.

### python examples/calls.py

The kernels against numpy and the closed form, `matvec`'s term with `dot`
inlined and the loop nest loopy generates for it, a call given a vector of a
length nothing says is the row's, refused, and a schedule of `matvec`
compiled and compared with the native run.

```console
$ uv run python examples/calls.py
native: matvec agrees with A @ x: True
native: csr agrees with the dense product: True
native: each cell's matrix is |detJ| (1 + delta_jk) / 24: True

the term of matvec, with dot inlined at calls.py:99:
  S0 over i: y[i] = sum over k of A[i, k]*x[k]
  dot's a is A[i]
  dot's b is x

the loop nest loopy generates for it:
  for (int32_t i = 0; i <= -1 + m; ++i)
  {
    acc_k = (double) (0.0);
    for (int32_t k = 0; k <= -1 + n; ++k)
      acc_k = acc_k + A[n * i + k] * x[k];
    y[i] = acc_k;
  }

refused: matvec_by calls dot at calls.py:211: axis 0 of its b is n long there, and z, which it is given, has p; nothing says the two agree

schedule: Schedule(matvec, target='c').split(i, 4)
  y: difference 0 within 1.01e-06 (approx) -> tested
```

### lanky check examples/calls.py

```console
$ uv run lanky check examples/calls.py
STATUS   EFFECTIVE  BY           WHERE         OWNER          STATEMENT
-------  ---------  -----------  ------------  -------------  ------------------------------------------------------------------------
decided  decided    isl          calls.py:92   dot            a[k] is in bounds for every instance of S0
decided  decided    isl          calls.py:92   dot            b[k] is in bounds for every instance of S0
decided  decided    isl          calls.py:89   dot            the source order runs every dependence forward in time
decided  decided    type         calls.py:92   dot            the accumulation into the value dot returns over k is approx
tested   tested     interpreter  calls.py:89   dot            the traced term computes what the body computes
decided  decided    isl          calls.py:99   matvec         y[i] is in bounds for every instance of S0
decided  decided    isl          calls.py:99   matvec         A[i, k] is in bounds for every instance of S0
decided  decided    isl          calls.py:99   matvec         x[k] is in bounds for every instance of S0
decided  decided    isl          calls.py:99   matvec         distinct instances of S0 write distinct cells of y
decided  decided    isl          calls.py:95   matvec         the source order runs every dependence forward in time
decided  decided    type         calls.py:99   matvec         the accumulation into y[i] over k is approx
decided  decided    isl          calls.py:99   matvec         A[i], the row passed to dot as a, is in bounds wherever dot is called...
tested   tested     interpreter  calls.py:95   matvec         the traced term computes what the body computes
decided  decided    isl          calls.py:112  csr            y[r] is in bounds for every instance of S0
decided  decided    isl          calls.py:112  csr            val[r, k] is in bounds for every instance of S0
decided  decided    type         calls.py:112  csr            x[col[r, k]] is in bounds by type (col[r, k] : Fin(m))
decided  decided    isl          calls.py:112  csr            col[r, k] is in bounds for every instance of S0
decided  decided    isl          calls.py:112  csr            cnt[r], the length of row r that bounds the loop over k, is in bounds...
decided  decided    isl          calls.py:112  csr            distinct instances of S0 write distinct cells of y
decided  decided    isl          calls.py:102  csr            the source order runs every dependence forward in time
decided  decided    type         calls.py:112  csr            the accumulation into y[r] over k is approx
decided  decided    isl          calls.py:112  csr            col[r], the indices of the cells passed as b, is in bounds wherever d...
decided  decided    isl          calls.py:112  csr            val[r], the row passed to dot as a, is in bounds wherever dot is call...
decided  decided    type         calls.py:112  csr            the cells x[col[r]] picks, passed as b, are in bounds by type (the en...
tested   tested     interpreter  calls.py:102  csr            the traced term computes what the body computes
decided  decided    isl          calls.py:118  integrate      W[q] is in bounds for every instance of S0
decided  decided    isl          calls.py:115  integrate      the source order runs every dependence forward in time
decided  decided    type         calls.py:118  integrate      the accumulation into the value integrate returns over q is approx
tested   tested     interpreter  calls.py:115  integrate      the traced term computes what the body computes
decided  decided    isl          calls.py:129  element_mass   Me[j, k] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  element_mass   X[1] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  element_mass   X[0] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  element_mass   Y[2] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  element_mass   Y[0] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  element_mass   X[2] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  element_mass   Y[1] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  element_mass   W[q] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  element_mass   PHI[q, j] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  element_mass   PHI[q, k] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  element_mass   distinct instances of S0 write distinct cells of Me
decided  decided    isl          calls.py:121  element_mass   the source order runs every dependence forward in time
decided  decided    type         calls.py:129  element_mass   the accumulation into Me[j, k] over q is approx
tested   tested     native       calls.py:121  element_mass   Me[0, 0] == Me[0, 0] and Me[0, 1] == Me[1, 0] and Me[0, 2] == Me[2, 0...
tested   tested     interpreter  calls.py:121  element_mass   the traced term computes what the body computes
decided  decided    isl          calls.py:129  mass_matrices  M[c, j, k] is in bounds for every instance of S0
decided  decided    type         calls.py:129  mass_matrices  X[cell[c, 1]] is in bounds by type (cell[c, 1] : Fin(nv))
decided  decided    isl          calls.py:129  mass_matrices  cell[c, 1] is in bounds for every instance of S0
decided  decided    type         calls.py:129  mass_matrices  X[cell[c, 0]] is in bounds by type (cell[c, 0] : Fin(nv))
decided  decided    isl          calls.py:129  mass_matrices  cell[c, 0] is in bounds for every instance of S0
decided  decided    type         calls.py:129  mass_matrices  Y[cell[c, 2]] is in bounds by type (cell[c, 2] : Fin(nv))
decided  decided    isl          calls.py:129  mass_matrices  cell[c, 2] is in bounds for every instance of S0
decided  decided    type         calls.py:129  mass_matrices  Y[cell[c, 0]] is in bounds by type (cell[c, 0] : Fin(nv))
decided  decided    type         calls.py:129  mass_matrices  X[cell[c, 2]] is in bounds by type (cell[c, 2] : Fin(nv))
decided  decided    type         calls.py:129  mass_matrices  Y[cell[c, 1]] is in bounds by type (cell[c, 1] : Fin(nv))
decided  decided    isl          calls.py:129  mass_matrices  W[q] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  mass_matrices  PHI[q, j] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  mass_matrices  PHI[q, k] is in bounds for every instance of S0
decided  decided    isl          calls.py:129  mass_matrices  distinct instances of S0 write distinct cells of M
decided  decided    isl          calls.py:132  mass_matrices  the source order runs every dependence forward in time
decided  decided    type         calls.py:129  mass_matrices  the accumulation into M[c, j, k] over q is approx
decided  decided    isl          calls.py:141  mass_matrices  cell[c], the indices of the cells passed as X and Y, is in bounds whe...
decided  decided    type         calls.py:141  mass_matrices  the cells X[cell[c]] picks, passed as X, are in bounds by type (the e...
decided  decided    type         calls.py:141  mass_matrices  the cells Y[cell[c]] picks, passed as Y, are in bounds by type (the e...
decided  decided    isl          calls.py:141  mass_matrices  M[c], the row passed to element_mass as Me, is in bounds wherever ele...
decided  tested     call         calls.py:141  mass_matrices  after each call element_mass(X[cell[c]], Y[cell[c]], M[c]) at calls.p...
tested   tested     interpreter  calls.py:132  mass_matrices  the traced term computes what the body computes

66 facts: 59 decided, 7 tested
```

### loopty run examples/calls.py

`integrate` takes a function, so it is compiled where a kernel calls it; a
rule on its own writes its value into a one-cell array, `result`.

```console
$ uv run loopty run examples/calls.py
integrate: takes the function f, so it is compiled where a kernel calls it, inlined
matvec: Schedule(matvec, target='c').split(i, 4)
  y: difference 0 within 1.01e-06 (approx) -> tested
dot: Schedule(dot, target='c')
  result: difference 0 within 1.26e-06 (approx) -> tested
csr: Schedule(csr, target='c')
  y: difference 0 within 1e-06 (approx) -> tested
element_mass: Schedule(element_mass, target='c')
  Me: difference 0 within 1.08e-06 (approx) -> tested
mass_matrices: Schedule(mass_matrices, target='c')
  M: difference 0 within 1.02e-06 (approx) -> tested

STATUS   BY     WHERE         OWNER          STATEMENT
-------  -----  ------------  -------------  ------------------------------------------------------------------------
decided  isl    calls.py:99   matvec         split(i, 4) renames the instances of matvec one for one
decided  isl    calls.py:99   matvec         the order after split(i, 4) runs every dependence of matvec forward
tested   loopy  calls.py:99   matvec         the scheduled run of matvec agrees with the native run to the accurac...
tested   loopy  calls.py:92   dot            the scheduled run of dot agrees with the native run to the accuracy i...
tested   loopy  calls.py:112  csr            the scheduled run of csr agrees with the native run to the accuracy i...
tested   loopy  calls.py:129  element_mass   the scheduled run of element_mass agrees with the native run to the a...
tested   loopy  calls.py:129  mass_matrices  the scheduled run of mass_matrices agrees with the native run to the ...

7 facts: 2 decided, 5 tested
```

## reshape_layouts.py and p2p.py

Both run the same three ways and are documented in their own module docstrings.
`reshape_layouts.py` prints the two layout maps as isl objects and a small
ledger of what isl decided about them; `p2p.py` prints the near-field potential
of sixteen points in a four by four grid of boxes and checks it against a direct
sum over the same interaction lists.
