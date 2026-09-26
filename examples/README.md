# The loopty demos

Six files, each of which runs three ways. Every console block below is a
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
| `wavefront_acoustic.py` | two coupled statements in one acoustic-wave nest: a rectangular tiling refused with a witness that crosses them, the skew that makes it a legal wavefront block, and the diamond map, accepted and run, whose tiling is refused |
| `reshape_layouts.py` | `Fin[n * m]` as `Fin[n] x Fin[m]`, one buffer read in two layouts, and a transpose split and interchanged |
| `p2p.py` | the near field of a fast multipole method: a two-level interaction list flattened into one ragged level, with the self-interaction guarded by `when` |
| `composition.py` | two kernels composed by a program and lowered as one loopy kernel, with the intermediate a temporary of it and the edge between the kernels found in the footprints |

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
exactly such a loop. That refusal is a `refuted` fact of kind `buildable`
decided by `loopy-target`, carried beside the decided casts; `docs/device-runs.md`
has the measurement it comes from and `docs/loopy-notes.md` the details.

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
  decided  isl  the order after tag(r='g.0') runs every dependence of spmv forward
  decided  isl  split(j, 32) renames the instances of spmv one for one
  decided  isl  the order after split(j, 32) runs every dependence of spmv forward
  decided  isl  tag(j_in='l.0') renames the instances of spmv one for one
  decided  isl  the order after tag(j_in='l.0') runs every dependence of spmv forward
  refuted  loopy-target c code can be generated for spmv after tag(j_in='l.0')
  decided  isl  the accumulation into y is reassociated by tag(j_in='l.0'), so its result is compared at 'reassoc'
  decided  isl  realize('y', tree=True) renames the instances of spmv one for one
  decided  isl  the order after realize('y', tree=True) runs every dependence of spmv forward
  reason: the parallel tag on j_in sits inside a loop whose bound comes from an array (a ragged fiber), and loopy will not put a hardware axis in a domain with a data-dependent parameter. Parallelize an enclosing loop with a size known at launch instead, such as the rows of a CSR product

  y: difference 5.55e-17 within 1.48e-06 (approx) -> tested
```

### lanky check examples/spmv.py

The two `assumed` rows are the honest ones: nothing in the term decides the
scan's recurrence, and `lanky` says so rather than passing over it. The theorem
beside it is `tested` here, and `proved` on a machine with the Lean extra
installed. The last row of each kernel, `tested` by `interpreter`, is the one
fact about the trace itself: the traced term, run by loopty's interpreter,
agrees with the body run natively, on this file's `example_inputs()` and on
three inputs drawn from the declared types.

```console
$ uv run lanky check examples/spmv.py
STATUS                            BY             WHERE        OWNER          STATEMENT
--------------------------------  -------------  -----------  -------------  ------------------------------------------------------------------------
decided                           isl            spmv.py:79   scan           off[0] is in bounds for every instance of S0
decided                           isl            spmv.py:81   scan           off[r + 1] is in bounds for every instance of S1
decided                           isl            spmv.py:81   scan           off[r] is in bounds for every instance of S1
decided                           isl            spmv.py:81   scan           cnt[r] is in bounds for every instance of S1
decided                           isl            spmv.py:79   scan           distinct instances of S0 write distinct cells of off
decided                           isl            spmv.py:81   scan           distinct instances of S1 write distinct cells of off
decided                           isl            spmv.py:69   scan           the source order runs every dependence forward in time
assumed                           -              spmv.py:69   scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
tested                            interpreter    spmv.py:69   scan           the traced term computes what the body computes
tested                            property-test  spmv.py:84   scan_monotone  n : Nat, cnt : Fn[Fin(n), Nat], off : Fn[Fin(n + 1), Nat] | off(0) ==...
decided                           isl            spmv.py:112  spmv           y[r] is in bounds for every instance of S0
decided                           isl            spmv.py:112  spmv           val[r, j] is in bounds for every instance of S0
decided                           type           spmv.py:112  spmv           x[col[r, j]] is in bounds by type (col[r, j] : Fin(m))
decided                           isl            spmv.py:112  spmv           col[r, j] is in bounds for every instance of S0
decided                           isl            spmv.py:112  spmv           cnt[r], the length of row r that bounds the loop over j, is in bounds...
decided                           isl            spmv.py:112  spmv           distinct instances of S0 write distinct cells of y
decided                           isl            spmv.py:102  spmv           the source order runs every dependence forward in time
decided                           type           spmv.py:112  spmv           the accumulation into y[r] over j is approx
tested                            interpreter    spmv.py:102  spmv           the traced term computes what the body computes
assumed under scan:postcondition  -              spmv.py:115  solve          after scan(...) in solve: off[0] == 0 and (forall r in Fin(n). off[r ...

20 facts: 2 assumed, 15 decided, 3 tested
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
  y: difference 5.55e-17 within 1.48e-06 (approx) -> tested
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
The map is not unimodular: its image is only the points of equal parity, which
the printed loopy domain states as `(a + b) mod 2 = 0`. Written with space
first, `a = i + t` and `b = i - t`, it is refused, because `S0` at
`(t + 1, i - 1)` reads what `S1` wrote at `(t, i)` and the new order runs that
backwards. Written with time first it is accepted, and the question this demo
was extended to answer has its answer: loopy generates correct code over that
image, and both fields agree with the native run bit for bit. loopy's own
`map_domain` refuses the map, so loopty rewrites the kernel itself, and the
generated loop tests the parity inside the innermost loop rather than stepping
by two, so it is correct and not fast (note 13 in `../docs/loopy-notes.md`).
Tiling the diamond is refused: `S1` at `(t, i + 1)` reads the velocity `S0`
wrote at `(t, i)`, a distance of `(0, 1)` that the `t - i` direction runs
backwards. A diamond tiling of this pair needs an offset in time between the two
statements, and `affine` moves every statement in its loops alike. The module
docstring has the details.

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
  loopy domain: [nt, nx] -> { [a, b] : (a + b) mod 2 = 0 and b >= -a and 4 - 2nx + a <= b <= -2 + a and b <= -4 + 2nt - a }
  decided  isl  affine({ [t, i] -> [a = t + i, b = t - i] }) renames the instances of acoustic one for one
  decided  isl  the order after affine({ [t, i] -> [a = t + i, b = t - i] }) runs every dependence of acoustic forward

  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested

Schedule(acoustic, target='c').affine({ [t, i] -> [a = t + i, b = t - i] }).tile('a', 'b', 4, 4) ->
  IllegalCast: tile(a,b,4,4) illegal: instance S0[t=7, i=15] writes velocity[8, 15] read by S1[t=7, i=16] scheduled earlier (at nt=16, nx=32, as hinted)
```

### lanky check examples/wavefront_acoustic.py

Eight accesses, every one decided by isl over the domain the `when` guard
narrows: `velocity[t + 1, i - 1]` is in bounds because the guard keeps `i > 0`,
and both writes at `t + 1` because it keeps `t + 1 < nt`.

```console
$ uv run lanky check examples/wavefront_acoustic.py
STATUS   BY           WHERE                      OWNER     STATEMENT
-------  -----------  -------------------------  --------  ------------------------------------------------------------
decided  isl          wavefront_acoustic.py:102  acoustic  velocity[t + 1, i] is in bounds for every instance of S0
decided  isl          wavefront_acoustic.py:102  acoustic  velocity[t, i] is in bounds for every instance of S0
decided  isl          wavefront_acoustic.py:102  acoustic  pressure[t, i + 1] is in bounds for every instance of S0
decided  isl          wavefront_acoustic.py:102  acoustic  pressure[t, i] is in bounds for every instance of S0
decided  isl          wavefront_acoustic.py:105  acoustic  pressure[t + 1, i] is in bounds for every instance of S1
decided  isl          wavefront_acoustic.py:105  acoustic  pressure[t, i] is in bounds for every instance of S1
decided  isl          wavefront_acoustic.py:105  acoustic  velocity[t + 1, i] is in bounds for every instance of S1
decided  isl          wavefront_acoustic.py:105  acoustic  velocity[t + 1, i - 1] is in bounds for every instance of S1
decided  isl          wavefront_acoustic.py:102  acoustic  distinct instances of S0 write distinct cells of velocity
decided  isl          wavefront_acoustic.py:105  acoustic  distinct instances of S1 write distinct cells of pressure
decided  isl          wavefront_acoustic.py:90   acoustic  the source order runs every dependence forward in time
tested   interpreter  wavefront_acoustic.py:90   acoustic  the traced term computes what the body computes

12 facts: 11 decided, 1 tested
```

### loopty run examples/wavefront_acoustic.py

Two schedules of one kernel, the wavefront block and the diamond, with two
outputs each, all compared with the native run. Each schedule keeps its own
facts in the one ledger, because a fact's id names the schedule it is about
and not only the kernel.

```console
$ uv run loopty run examples/wavefront_acoustic.py
acoustic: Schedule(acoustic, target='c').skew(i, by='t').tile(t,i,4,8)
  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested
acoustic: Schedule(acoustic, target='c').affine({ [t, i] -> [a = t + i, b = t - i] })
  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE                      OWNER     STATEMENT
-------  -----  -------------------------  --------  ------------------------------------------------------------------------
decided  isl    wavefront_acoustic.py:102  acoustic  skew(i, by='t') renames the instances of acoustic one for one
decided  isl    wavefront_acoustic.py:102  acoustic  the order after skew(i, by='t') runs every dependence of acoustic for...
decided  isl    wavefront_acoustic.py:102  acoustic  tile(t,i,4,8) renames the instances of acoustic one for one
decided  isl    wavefront_acoustic.py:102  acoustic  the order after tile(t,i,4,8) runs every dependence of acoustic forward
tested   loopy  wavefront_acoustic.py:102  acoustic  the scheduled run of acoustic agrees with the native run to the accur...
decided  isl    wavefront_acoustic.py:102  acoustic  affine({ [t, i] -> [a = t + i, b = t - i] }) renames the instances of...
decided  isl    wavefront_acoustic.py:102  acoustic  the order after affine({ [t, i] -> [a = t + i, b = t - i] }) runs eve...
tested   loopy  wavefront_acoustic.py:102  acoustic  the scheduled run of acoustic agrees with the native run to the accur...

8 facts: 6 decided, 2 tested
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
and orders the two loops. The loops are not fused. That is a cast over this
term, still to come, and this run is what it will be checked against.

### python examples/composition.py

```console
$ uv run python examples/composition.py
native: the program agrees with the numpy slices: True

the term of burgers_rhs(u, rhs):
  temporary f: 1 axis, element Real
  f.zeros        over i_0 from composition.py:79
  flux.S0        over j   from composition.py:65
  divergence.S0  over i   from composition.py:73

#include <stdint.h>
#include <stdbool.h>

void burgers_rhs(int32_t const n, double const *__restrict__ u, double *__restrict__ rhs)
{
  double f[n];

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

The two kernels' obligations. The program adds no row: neither kernel states
a postcondition for it to restate.

```console
$ uv run lanky check examples/composition.py
STATUS   BY           WHERE              OWNER       STATEMENT
-------  -----------  -----------------  ----------  ------------------------------------------------------
decided  isl          composition.py:65  flux        f[j] is in bounds for every instance of S0
decided  isl          composition.py:65  flux        u[j] is in bounds for every instance of S0
decided  isl          composition.py:65  flux        distinct instances of S0 write distinct cells of f
decided  isl          composition.py:61  flux        the source order runs every dependence forward in time
tested   interpreter  composition.py:61  flux        the traced term computes what the body computes
decided  isl          composition.py:73  divergence  rhs[i] is in bounds for every instance of S0
decided  isl          composition.py:73  divergence  f[i + 1] is in bounds for every instance of S0
decided  isl          composition.py:73  divergence  f[i - 1] is in bounds for every instance of S0
decided  isl          composition.py:73  divergence  distinct instances of S0 write distinct cells of rhs
decided  isl          composition.py:68  divergence  the source order runs every dependence forward in time
tested   interpreter  composition.py:68  divergence  the traced term computes what the body computes

11 facts: 9 decided, 2 tested
```

### loopty run examples/composition.py

Each kernel alone, and then the program as one kernel. On the C target the
temporary is a variable-length array on the stack of the call; see note 14 in
`../docs/loopy-notes.md`.

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
tested  loopy  composition.py:65  flux         the scheduled run of flux agrees with the native run to the accuracy ...
tested  loopy  composition.py:73  divergence   the scheduled run of divergence agrees with the native run to the acc...
tested  loopy  composition.py:76  burgers_rhs  the scheduled run of burgers_rhs agrees with the native run to the ac...

3 facts: 3 tested
```

## reshape_layouts.py and p2p.py

Both run the same three ways and are documented in their own module docstrings.
`reshape_layouts.py` prints the two layout maps as isl objects and a small
ledger of what isl decided about them; `p2p.py` prints the near-field potential
of sixteen points in a four by four grid of boxes and checks it against a direct
sum over the same interaction lists.
