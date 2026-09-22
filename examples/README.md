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
here quietly stop being true.

| file | what it shows |
|---|---|
| `spmv.py` | a ragged sparse product: an indirection in bounds by type, a scan with a postcondition, the theorem that postcondition needs, and a schedule whose reassociation is recorded |
| `stencil_skew.py` | a rectangular tiling of a Jacobi stencil refused with a witness pair, and the skew that makes the same tiling legal |\n| `wavefront_acoustic.py` | two interdependent instructions in one acoustic-wave kernel, a cross-statement time dependence, its illegal rectangular tile, and the legal skewed wavefront block |\n| `composition_fusion.py` | an `@program` made from two typed kernels, Loopy fusion across their data-flow edge, and `assignment_to_subst` eliminating the intermediate array |
| `reshape_layouts.py` | `Fin[n * m]` as `Fin[n] x Fin[m]`, one buffer read in two layouts, and a transpose split and interchanged |
| `p2p.py` | the near field of a fast multipole method: a two-level interaction list flattened into one ragged level, with the self-interaction guarded by `when` |

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
installed.

```console
$ uv run lanky check examples/spmv.py
STATUS   BY             WHERE        OWNER          STATEMENT
-------  -------------  -----------  -------------  ------------------------------------------------------------------------
decided  isl            spmv.py:80   scan           off[0] is in bounds for every instance of S0
decided  isl            spmv.py:82   scan           off[r + 1] is in bounds for every instance of S1
decided  isl            spmv.py:82   scan           off[r] is in bounds for every instance of S1
decided  isl            spmv.py:82   scan           cnt[r] is in bounds for every instance of S1
decided  isl            spmv.py:80   scan           distinct instances of S0 write distinct cells of off
decided  isl            spmv.py:82   scan           distinct instances of S1 write distinct cells of off
decided  isl            spmv.py:70   scan           the source order runs every dependence forward in time
assumed  -              spmv.py:70   scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
tested   property-test  spmv.py:85   scan_monotone  n : Nat, cnt : Fn[Fin(n), Nat], off : Fn[Fin(n + 1), Nat] | off(0) ==...
decided  isl            spmv.py:113  spmv           y[r] is in bounds for every instance of S0
decided  isl            spmv.py:113  spmv           val[r, j] is in bounds for every instance of S0
decided  type           spmv.py:113  spmv           x[col[r, j]] is in bounds by type (col[r, j] : Fin(m))
decided  isl            spmv.py:113  spmv           col[r, j] is in bounds for every instance of S0
decided  isl            spmv.py:113  spmv           distinct instances of S0 write distinct cells of y
decided  isl            spmv.py:103  spmv           the source order runs every dependence forward in time
decided  type           spmv.py:113  spmv           the accumulation into y[r] over j is approx
assumed  -              spmv.py:116  solve          after scan(...) in solve: off[0] == 0 and (forall r in Fin(n). off[r ...

17 facts: 2 assumed, 14 decided, 1 tested
```

### loopty run examples/spmv.py

The scheduled `spmv` and the unscheduled `scan` are both compiled and compared
with their own Python bodies. `scan` is integer arithmetic, so its tolerance is
zero; `y` is compared at the accuracy its exactness class states.

```console
$ uv run loopty run examples/spmv.py
spmv: Schedule(spmv, target='c').split(j, 2).realize('y', tree=True)
  y: difference 5.55e-17 within 1.48e-06 (approx) -> tested
scan: Schedule(scan, target='c')
  off: difference 0 within 0 (exact) -> tested

STATUS   BY     WHERE        OWNER  STATEMENT
-------  -----  -----------  -----  ------------------------------------------------------------------------
decided  isl    spmv.py:113  spmv   split(j, 2) renames the instances of spmv one for one
decided  isl    spmv.py:113  spmv   the order after split(j, 2) runs every dependence of spmv forward
decided  isl    spmv.py:113  spmv   realize('y', tree=True) renames the instances of spmv one for one
decided  isl    spmv.py:113  spmv   the order after realize('y', tree=True) runs every dependence of spmv...
decided  isl    spmv.py:113  spmv   the accumulation into y is reassociated by realize('y', tree=True), s...
tested   loopy  spmv.py:113  spmv   the scheduled run of spmv agrees with the native run to the accuracy ...
tested   loopy  spmv.py:80   scan   the scheduled run of scan agrees with the native run to the accuracy ...

7 facts: 5 decided, 2 tested
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
STATUS   BY   WHERE               OWNER   STATEMENT
-------  ---  ------------------  ------  ------------------------------------------------------
decided  isl  stencil_skew.py:62  jacobi  u[t + 1, i] is in bounds for every instance of S0
decided  isl  stencil_skew.py:62  jacobi  u[t, i - 1] is in bounds for every instance of S0
decided  isl  stencil_skew.py:62  jacobi  u[t, i + 1] is in bounds for every instance of S0
decided  isl  stencil_skew.py:62  jacobi  distinct instances of S0 write distinct cells of u
decided  isl  stencil_skew.py:54  jacobi  the source order runs every dependence forward in time

5 facts: 5 decided
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

This example moves beyond a one-statement Jacobi recurrence. A velocity update
and a pressure update live in the same `(t, i)` loop nest. The pressure
instruction consumes the velocity instruction in the same time step, while the
next time step's velocity instruction consumes pressure produced by the other
instruction. The dependence that crosses an i-tile boundary is therefore
**S1 -> S0**, not S0 -> itself.

The ordinary rectangular tile is rejected with that cross-statement witness.
`skew("i", by="t").tile(...)` is accepted and executed. Geometrically this is a
wavefront/parallelogram temporal block: rectangular in `(t, i+t)`. The module
also has an optional `--bench` mode so the locality effect can be measured on a
real machine without turning a speedup into a CI invariant.

A full 1-D diamond is the next useful pressure test rather than something this
demo pretends to have already: it wants the two characteristic coordinates
`i+t` and `i-t`. That points at a multi-axis affine schedule primitive whose
checker reasons about the parity-constrained image of that map.

## composition_fusion.py

The application is `flux(u, f); divergence(f, rhs)`. It is already natural to
write natively as an `@program`, but `Program` does not lower yet. The example
therefore lowers the two kernels independently and uses Loopy's existing
composition machinery as an experiment:

1. `fuse_kernels(..., data_flow=[("f", 0, 1)])` coalesces domains and
   instructions and makes the producer/consumer edge explicit.
2. `assignment_to_subst(..., "f")` turns the pointwise producer into a
   substitution rule and removes the materialized intermediate.

The one local adapter in the example is the important API finding: independent
lowerings describe `f` differently because it is an output of the producer and
an input of the consumer, while Loopy fusion requires matching argument
declarations. A first-class `Program.lower()` / composition primitive should
own that interface reconciliation, make internal data-flow values temporaries,
and then choose fusion/substitution/inlining as lowering decisions.

## reshape_layouts.py and p2p.py

Both run the same three ways and are documented in their own module docstrings.
`reshape_layouts.py` prints the two layout maps as isl objects and a small
ledger of what isl decided about them; `p2p.py` prints the near-field potential
of sixteen points in a four by four grid of boxes and checks it against a direct
sum over the same interaction lists.
