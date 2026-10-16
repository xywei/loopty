# The loopty demos

Eighteen files, each of which runs three ways. Every console block below is a
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
| `fusion.py` | the program of `composition.py` with its two loops fused into one, a fusion one step too early refused with its pair, the intermediate computed where it is read instead of stored, and kept in the three cells live at once, a window of two refused with the read that sees its slot overwritten |
| `quadrature.py` | firedrake's P1 mass kernel cell by cell, its quadrature rule and basis held in constant tables, read-only arrays written into the compiled code, and its gathers through the cell map in bounds by type |
| `layer_potential.py` | pytential's QBX on its own curve: the line-Taylor theorem tied by a row to the terms the single layer adds, through the identity that its ratio by the conjugate is `-beta / alpha`, the single and double layer over complex points, sumpy's CPU tile checked and the write sumpy silences a fact, each kernel's agreement with sumpy's own as an `upstream` row within `1e-12` of the largest value, and pytential's Green identity and the jump relations as tested rows |
| `sum_factorization.py` | grudge's two passes of a sum-factorized operator, fused on the element and the row and no deeper: the deeper fusion refused with its pair, the fusion of fewer loops it names, and what passes between them kept one row at a time |
| `storage_layouts.py` | where cells are stored, as checked maps: sumpy's compressed-Taylor storage index in two pieces, one for one onto its buffer, and grudge's `dim_tags` `N3,N0,N1,N2` the same layout as modepy's reshape of the DOFArray, a wrong nesting refuted with a cell; two kernels read through them |
| `assembly.py` | firedrake's PyOP2 assembly loop with FFC's P1 mass kernel called per cell and the cell's matrix added into the nodes, `y[cell[c, a]] += ...`: the rule's exactness, the element's mass and the scatter's total as theorems, the element's mass as `mass_ffc`'s postcondition to the approx tolerance, an update that owes the class of its sum and not distinct writes, PyOP2's GPU split refused with the node its two triangles share, and accepted once the updates are atomic |
| `calls.py` | kernels calling kernels inside their loops: a rule with a loop of its own (`dot`, loopy #1044) in a matrix-vector and a sparse product, and firedrake's P1 quadrature rule applied per element through a rule that takes the integrand (#1043), each call inlined with binders of its own and owing the caller what the callee's contract checks |
| `taylor_reconstruction.py` | sumpy's compressed Taylor reconstruction in one file, theorem, kernel and schedule: the theorem proved for every order, the kernel's postcondition decided from what it stores, the theorem applied to it, and the schedule's casts, in one ledger |
| `l2p.py` | the demo's flagship, sumpy's L2P for the 2-D Laplace kernel (`E2PFromSingleBox`): the compressed Taylor theorem proved for every order and its form box by box, sumpy's storage index a checked layout, the kernel that reconstructs each box's triangle with the theorem applied to its postcondition, the interchange of the triangle's loops refused with its pair, the three calls fused box by box with each box's triangle and table in windows of its own, sumpy's own schedule for a GPU with each box a work group, and a row that compares the program with sumpy's own kernel, within `1e-14` of the largest value |
| `hexahedra.py` | grudge #418's weak gradient and weak divergence on hexahedra, the kernels for every order: the metric product, a pass per reference axis and the three directions summed, each a kernel and together a program; the Kronecker theorem the passes rest on, for every `n`; the DOFArray's layout at every order, the same as #418's `dim_tags` `N3,N0,N1,N2` and as the kernels' row-major arrays; the passes fused as deep as each pair allows, one loop deeper refused with its pair, and what passes between them kept a row and a slab at a time; and the three launches a device runs, every node on a work item, which `benchmarks/hexahedra.py` times |
| `advection.py` | grudge #418's weak advection right-hand side on hexahedra, from grudge's own geometry: the upwind flux at every face node, the neighbor's value gathered through the face maps and in bounds by type; the volume term and the lift with the inverse mass pushed inward, the rearrangement two theorems state for every order, and a third that the lift onto the face nodes is the face gather's transpose; the kernels fused on the element and one loop deeper refused with its pair, the fluxes kept an element at a time; every split a device runs (libParanumal's volume and surface, the fluxes first, three launches, one, one from the metric terms) checked before it runs; and libParanumal's lift onto the face nodes, refused without its barriers with the edge node two faces update, and accepted atomic. `benchmarks/advection.py` times them |

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
sums, and it *becomes* `reassoc` when `realize(spmv.y, tree=True)` reorders it.
The direction matters: a trace may not assume permission to reassociate, and a
schedule has to ask for it, which is why the widened tolerance at the end of the
run is traceable to a line of the schedule rather than to a default.

The design's device schedule,
`on_groups(spmv.r).split(spmv.j, 32).on_items(spmv.j.inner)`,
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

schedule: Schedule(spmv, target='c').split(spmv.j, 2).realize(spmv.y, tree=True)
  decided  isl  split(spmv.j, 2) renames the instances of spmv one for one
  decided  isl  the order after split(spmv.j, 2) runs every dependence of spmv forward
  decided  isl  the accumulation into spmv.y is reassociated by realize(spmv.y, tree=True), so its result is compared at 'reassoc'
device schedule: Schedule(spmv, target='c').on_groups(spmv.r).split(spmv.j, 32).on_items(spmv.j.inner).realize(spmv.y, tree=True)
  decided  isl  on_groups(spmv.r) renames the instances of spmv one for one
  decided  isl  the order after on_groups(spmv.r) runs every dependence of spmv forward, within one work item
  decided  isl  split(spmv.j, 32) renames the instances of spmv one for one
  decided  isl  the order after split(spmv.j, 32) runs every dependence of spmv forward, within one work item
  decided  isl  on_items(spmv.j.inner) renames the instances of spmv one for one
  decided  isl  the order after on_items(spmv.j.inner) runs every dependence of spmv forward, within one work item
  refuted  loopy-target c code can be generated for spmv after on_items(spmv.j.inner)
  decided  isl  the accumulation into spmv.y is reassociated by on_items(spmv.j.inner), so its result is compared at 'reassoc'
  reason: the parallel tag on spmv.j.inner sits inside a loop whose bound comes from an array (a ragged fiber), and loopy will not put a hardware axis in a domain with a data-dependent parameter. Parallelize an enclosing loop with a size known at launch instead, such as the rows of a CSR product

  y: difference 0 within 1e-06 (approx) -> tested
```

### lanky check examples/spmv.py

The scan's postcondition is `decided` by `isl`, from what its statements
store: no cell of `off` is stored twice, or read before its store, the row
above it, so each statement's store is what `off` holds when `scan` returns,
and the recurrence is what `S1` stores at `r`. It also held after every native
run of `scan`, on this file's example inputs and on drawn ones, which its
provenance keeps. The restatement of it in `solve` is `decided` by the call,
resting on it. The casts of the file's schedule, `rows_split`, are rows of the
ledger too, under the kernel they schedule. The theorem beside it is `tested`
here, and `proved` on a machine with the Lean extra installed, and the row
under it is lanky's check of its reading: the theorem's annotations, run again
as plain Python at drawn values, compute what its term does, so the term the
oracles are handed is the claim written. The last row of each kernel, `tested`
by `interpreter`, is the one fact about the trace itself: the traced term, run
by loopty's interpreter, agrees with the body run natively, on this file's
`example_inputs()` and on three inputs drawn from the declared types. The
program `solve` has that fact too, about the term its two calls compose into.

```console
$ uv run lanky check examples/spmv.py
STATUS   BY             WHERE        OWNER          STATEMENT
-------  -------------  -----------  -------------  ----------------------------------------------------------------------------------------------
decided  isl            spmv.py:81   scan           off[0] is in bounds for every instance of S0
decided  isl            spmv.py:83   scan           off[r + 1] is in bounds for every instance of S1
decided  isl            spmv.py:83   scan           off[r] is in bounds for every instance of S1
decided  isl            spmv.py:83   scan           cnt[r] is in bounds for every instance of S1
decided  isl            spmv.py:81   scan           distinct instances of S0 write distinct cells of off
decided  isl            spmv.py:83   scan           distinct instances of S1 write distinct cells of off
decided  isl            spmv.py:83   scan           every cell of off that S1 reads, S0 or S1 wrote before the read
decided  isl            spmv.py:71   scan           the source order runs every dependence forward in time
decided  isl            spmv.py:81   scan           no cell of off is stored twice, or read before its store
decided  isl            spmv.py:71   scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
tested   interpreter    spmv.py:71   scan           the traced term computes what the body computes
tested   property-test  spmv.py:86   scan_monotone  n : Nat, cnt : Fn[Fin(n), Nat], off : Fn[Fin(n + 1), Nat]
                                                    | h0: off(0) == 0
                                                    | hs: forall r in Fin(n). off(r + 1) == off(r) + cnt(r)
                                                    |- forall a in Fin(n + 1), b in Fin(n + 1) where a <= b. off(a) <= off(b)
tested   python         spmv.py:86   scan_monotone  the term computes what the annotations compute
decided  isl            spmv.py:114  spmv           y[r] is in bounds for every instance of S0
decided  isl            spmv.py:114  spmv           val[r, j] is in bounds for every instance of S0
decided  type           spmv.py:114  spmv           x[col[r, j]] is in bounds by type (col[r, j] : Fin(m))
decided  isl            spmv.py:114  spmv           col[r, j] is in bounds for every instance of S0
decided  isl            spmv.py:114  spmv           cnt[r], the length of row r that bounds the loop over j, is in bounds for every instance of S0
decided  isl            spmv.py:114  spmv           distinct instances of S0 write distinct cells of y
decided  isl            spmv.py:104  spmv           the source order runs every dependence forward in time
decided  type           spmv.py:114  spmv           the accumulation into y[r] over j is approx
tested   interpreter    spmv.py:104  spmv           the traced term computes what the body computes
decided  isl            spmv.py:195  spmv           split(spmv.j, 2) renames the instances of spmv one for one
decided  isl            spmv.py:195  spmv           the order after split(spmv.j, 2) runs every dependence of spmv forward
decided  isl            spmv.py:196  spmv           the accumulation into spmv.y is reassociated by realize(spmv.y, tree=True),
                                                      so its result is compared at 'reassoc'
decided  call           spmv.py:117  solve          after scan(...) in solve: off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
tested   interpreter    spmv.py:117  solve          the traced term computes what the body computes

27 facts: 22 decided, 5 tested

WITHOUT LEAN: scan_monotone and 18 more rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
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
spmv: Schedule(spmv, target='c').split(spmv.j, 2).realize(spmv.y, tree=True)
  y: difference 0 within 1e-06 (approx) -> tested
scan: Schedule(scan, target='c')
  off: difference 0 within 0 (exact) -> tested
solve: Schedule(solve, target='c')
  y: difference 0 within 1e-06 (approx) -> tested
  off: difference 0 within 0 (exact) -> tested

STATUS   BY     WHERE        OWNER  STATEMENT
-------  -----  -----------  -----  -------------------------------------------------------------------------------------
decided  isl    spmv.py:195  spmv   split(spmv.j, 2) renames the instances of spmv one for one
decided  isl    spmv.py:195  spmv   the order after split(spmv.j, 2) runs every dependence of spmv forward
decided  isl    spmv.py:196  spmv   the accumulation into spmv.y is reassociated by realize(spmv.y, tree=True),
                                      so its result is compared at 'reassoc'
tested   loopy  spmv.py:196  spmv   the scheduled run of spmv agrees with the native run to the accuracy its types state
tested   loopy  spmv.py:71   scan   the scheduled run of scan agrees with the native run to the accuracy its types state
tested   loopy  spmv.py:117  solve  the scheduled run of solve agrees with the native run to the accuracy its types state

6 facts: 3 decided, 3 tested

WITHOUT LEAN: 2 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
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

Schedule(jacobi).tile(jacobi.t, jacobi.i, 8, 8) ->
  IllegalCast: tile(jacobi.t, jacobi.i, 8, 8) illegal: u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2 (stencil_skew.py:62) at t=0, i=8 writes u[1, 8], read by the same statement at t=1, i=7, scheduled earlier (at nt=16, nx=16, as hinted)
  witness: S0{'t': 0, 'i': 8} runs before S0{'t': 1, 'i': 7} at {'nt': 16, 'nx': 16}

accepted: Schedule(jacobi, target='c').skew(jacobi.i, by=jacobi.t).tile(jacobi.t, jacobi.i, 8, 8)
  loop nest: jacobi.t.outer jacobi.i.outer jacobi.t.inner jacobi.i.inner
  decided  isl  skew(jacobi.i, by=jacobi.t) renames the instances of jacobi one for one
  decided  isl  the order after skew(jacobi.i, by=jacobi.t) runs every dependence of jacobi forward
  decided  isl  tile(jacobi.t, jacobi.i, 8, 8) renames the instances of jacobi one for one
  decided  isl  the order after tile(jacobi.t, jacobi.i, 8, 8) runs every dependence of jacobi forward

  u: difference 0 within 1e-06 (approx) -> tested
  the native run matches the hand-written sweep: True
```

### lanky check examples/stencil_skew.py

Every access is in bounds because the `when` guard narrows the statement's
domain: `u[t + 1, i]` is written only where `t + 1 < nt`, and isl is asked about
the narrowed domain rather than the whole loop nest.

```console
$ uv run lanky check examples/stencil_skew.py
STATUS   BY           WHERE                OWNER   STATEMENT
-------  -----------  -------------------  ------  --------------------------------------------------------------------------------------
decided  isl          stencil_skew.py:62   jacobi  u[t + 1, i] is in bounds for every instance of S0
decided  isl          stencil_skew.py:62   jacobi  u[t, i - 1] is in bounds for every instance of S0
decided  isl          stencil_skew.py:62   jacobi  u[t, i + 1] is in bounds for every instance of S0
decided  isl          stencil_skew.py:62   jacobi  distinct instances of S0 write distinct cells of u
decided  isl          stencil_skew.py:54   jacobi  the source order runs every dependence forward in time
tested   interpreter  stencil_skew.py:54   jacobi  the traced term computes what the body computes
decided  isl          stencil_skew.py:116  jacobi  skew(jacobi.i, by=jacobi.t) renames the instances of jacobi one for one
decided  isl          stencil_skew.py:116  jacobi  the order after skew(jacobi.i, by=jacobi.t) runs every dependence of jacobi forward
decided  isl          stencil_skew.py:117  jacobi  tile(jacobi.t, jacobi.i, 8, 8) renames the instances of jacobi one for one
decided  isl          stencil_skew.py:117  jacobi  the order after tile(jacobi.t, jacobi.i, 8, 8) runs every dependence of jacobi forward

10 facts: 9 decided, 1 tested

WITHOUT LEAN: 9 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

### loopty run examples/stencil_skew.py

```console
$ uv run loopty run examples/stencil_skew.py
jacobi: Schedule(jacobi, target='c').skew(jacobi.i, by=jacobi.t).tile(jacobi.t, jacobi.i, 8, 8)
  u: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE                OWNER   STATEMENT
-------  -----  -------------------  ------  --------------------------------------------------------------------------------------
decided  isl    stencil_skew.py:116  jacobi  skew(jacobi.i, by=jacobi.t) renames the instances of jacobi one for one
decided  isl    stencil_skew.py:116  jacobi  the order after skew(jacobi.i, by=jacobi.t) runs every dependence of jacobi forward
decided  isl    stencil_skew.py:117  jacobi  tile(jacobi.t, jacobi.i, 8, 8) renames the instances of jacobi one for one
decided  isl    stencil_skew.py:117  jacobi  the order after tile(jacobi.t, jacobi.i, 8, 8) runs every dependence of jacobi forward
tested   loopy  stencil_skew.py:117  jacobi  the scheduled run of jacobi agrees with the native run to the accuracy its types state

5 facts: 4 decided, 1 tested

WITHOUT LEAN: 4 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

## wavefront_acoustic.py

A velocity update and a pressure update share one `(t, i)` loop nest. The
pressure statement `S1` reads the velocity that `S0` wrote in the same time
step, and the next step's velocity statement reads the pressure `S1` wrote. The
one dependence a rectangular tile runs backwards is therefore **S1 -> S0**, at
distance `(1, -1)`, and not a statement against itself as in the stencil.

The rectangular tile is rejected with that cross-statement witness, and
`skew(acoustic.i, by=acoustic.t).tile(...)` is accepted and runs. Geometrically this is a
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

Schedule(acoustic).tile(acoustic.t, acoustic.i, 4, 8) ->
  IllegalCast: tile(acoustic.t, acoustic.i, 4, 8) illegal: pressure[t + 1, i] = ... (wavefront_acoustic.py:114) at t=0, i=8 writes pressure[1, 8], read by velocity[t + 1, i] = ... (wavefront_acoustic.py:111) at t=1, i=7, scheduled earlier (at nt=16, nx=32, as hinted)
  witness: S1{'t': 0, 'i': 8} runs before S0{'t': 1, 'i': 7} at {'nt': 16, 'nx': 32}

accepted: Schedule(acoustic, target='c').skew(acoustic.i, by=acoustic.t).tile(acoustic.t, acoustic.i, 4, 8)
  loop nest: acoustic.t.outer acoustic.i.outer acoustic.t.inner acoustic.i.inner
  decided  isl  skew(acoustic.i, by=acoustic.t) renames the instances of acoustic one for one
  decided  isl  the order after skew(acoustic.i, by=acoustic.t) runs every dependence of acoustic forward
  decided  isl  tile(acoustic.t, acoustic.i, 4, 8) renames the instances of acoustic one for one
  decided  isl  the order after tile(acoustic.t, acoustic.i, 4, 8) runs every dependence of acoustic forward

  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested
  the native run matches the hand-written recurrence: True

Schedule(acoustic).affine('{ [t, i] -> [a, b] : a = i + t and b = i - t }') ->
  IllegalCast: affine({ [t, i] -> [a = t + i, b = -t + i] }) illegal: pressure[t + 1, i] = ... (wavefront_acoustic.py:114) at t=0, i=2 writes pressure[1, 2], read by velocity[t + 1, i] = ... (wavefront_acoustic.py:111) at t=1, i=1, scheduled earlier (at nt=16, nx=32, as hinted)

accepted: Schedule(acoustic, target='c').affine({ [t, i] -> [a = t + i, b = t - i] })
  loop nest: acoustic.loops.a acoustic.loops.b
  loopy domain: [nt, nx] -> { [a, b_step] : b_step >= 0 and 2 - nx + a <= b_step < a and b_step <= -2 + nt }
  decided  isl  affine({ [t, i] -> [a = t + i, b = t - i] }) renames the instances of acoustic one for one
  decided  isl  the order after affine({ [t, i] -> [a = t + i, b = t - i] }) runs every dependence of acoustic forward

  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested

Schedule(acoustic, target='c').affine({ [t, i] -> [a = t + i, b = t - i] }).tile(acoustic.loops.a, acoustic.loops.b, 4, 4) ->
  IllegalCast: tile(acoustic.loops.a, acoustic.loops.b, 4, 4) illegal: velocity[t + 1, i] = ... (wavefront_acoustic.py:111) at t=7, i=15 writes velocity[8, 15], read by pressure[t + 1, i] = ... (wavefront_acoustic.py:114) at t=7, i=16, scheduled earlier (at nt=16, nx=32, as hinted)

accepted: Schedule(acoustic, target='c').affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i] }).tile(acoustic.loops.a, acoustic.loops.b, 4, 4)
  loop nest: acoustic.loops.a.outer acoustic.loops.b.outer acoustic.loops.a.inner acoustic.loops.b.inner
  decided  isl  affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i] }) renames the instances of acoustic one for one
  decided  isl  the order after affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i] }) runs every dependence of acoustic forward
  decided  isl  tile(acoustic.loops.a, acoustic.loops.b, 4, 4) renames the instances of acoustic one for one
  decided  isl  the order after tile(acoustic.loops.a, acoustic.loops.b, 4, 4) runs every dependence of acoustic forward

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
-------  -----------  -------------------------  --------  ---------------------------------------------------------------------------------------------------
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
decided  isl          wavefront_acoustic.py:167  acoustic  skew(acoustic.i, by=acoustic.t) renames the instances of acoustic one for one
decided  isl          wavefront_acoustic.py:167  acoustic  the order after skew(acoustic.i, by=acoustic.t) runs every dependence of acoustic forward
decided  isl          wavefront_acoustic.py:168  acoustic  tile(acoustic.t, acoustic.i, 4, 8) renames the instances of acoustic one for one
decided  isl          wavefront_acoustic.py:168  acoustic  the order after tile(acoustic.t, acoustic.i, 4, 8) runs every dependence of acoustic forward
decided  isl          wavefront_acoustic.py:195  acoustic  affine({ [t, i] -> [a = t + i, b = t - i] }) renames the instances of acoustic one for one
decided  isl          wavefront_acoustic.py:195  acoustic  the order after affine({ [t, i] -> [a = t + i, b = t - i] }) runs every dependence of
                                                             acoustic forward
decided  isl          wavefront_acoustic.py:230  acoustic  affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i] }) renames the
                                                             instances of acoustic one for one
decided  isl          wavefront_acoustic.py:230  acoustic  the order after affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i]
                                                                 }) runs every dependence of acoustic forward
decided  isl          wavefront_acoustic.py:233  acoustic  tile(acoustic.loops.a, acoustic.loops.b, 4, 4) renames the instances of acoustic one for one
decided  isl          wavefront_acoustic.py:233  acoustic  the order after tile(acoustic.loops.a, acoustic.loops.b, 4, 4) runs every dependence of
                                                             acoustic forward

22 facts: 21 decided, 1 tested

WITHOUT LEAN: 21 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

### loopty run examples/wavefront_acoustic.py

Three schedules of one kernel, the wavefront block, the diamond and the
diamond tiling, with two outputs each, all compared with the native run. Each
schedule keeps its own facts in the one ledger, because a fact's id names the
schedule it is about and not only the kernel.

```console
$ uv run loopty run examples/wavefront_acoustic.py
acoustic: Schedule(acoustic, target='c').skew(acoustic.i, by=acoustic.t).tile(acoustic.t, acoustic.i, 4, 8)
  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested
acoustic: Schedule(acoustic, target='c').affine({ [t, i] -> [a = t + i, b = t - i] })
  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested
acoustic: Schedule(acoustic, target='c').affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i] }).tile(acoustic.loops.a, acoustic.loops.b, 4, 4)
  pressure: difference 0 within 1e-06 (approx) -> tested
  velocity: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE                      OWNER     STATEMENT
-------  -----  -------------------------  --------  ---------------------------------------------------------------------------------------------------
decided  isl    wavefront_acoustic.py:167  acoustic  skew(acoustic.i, by=acoustic.t) renames the instances of acoustic one for one
decided  isl    wavefront_acoustic.py:167  acoustic  the order after skew(acoustic.i, by=acoustic.t) runs every dependence of acoustic forward
decided  isl    wavefront_acoustic.py:168  acoustic  tile(acoustic.t, acoustic.i, 4, 8) renames the instances of acoustic one for one
decided  isl    wavefront_acoustic.py:168  acoustic  the order after tile(acoustic.t, acoustic.i, 4, 8) runs every dependence of acoustic forward
tested   loopy  wavefront_acoustic.py:168  acoustic  the scheduled run of acoustic agrees with the native run to the accuracy its types state
decided  isl    wavefront_acoustic.py:195  acoustic  affine({ [t, i] -> [a = t + i, b = t - i] }) renames the instances of acoustic one for one
decided  isl    wavefront_acoustic.py:195  acoustic  the order after affine({ [t, i] -> [a = t + i, b = t - i] }) runs every dependence of
                                                       acoustic forward
tested   loopy  wavefront_acoustic.py:195  acoustic  the scheduled run of acoustic agrees with the native run to the accuracy its types state
decided  isl    wavefront_acoustic.py:230  acoustic  affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i] }) renames the
                                                       instances of acoustic one for one
decided  isl    wavefront_acoustic.py:230  acoustic  the order after affine({ S0[t, i] -> [a = t + i, b = t - i]; S1[t, i] -> [a = t + i, b = 1 + t - i]
                                                           }) runs every dependence of acoustic forward
decided  isl    wavefront_acoustic.py:233  acoustic  tile(acoustic.loops.a, acoustic.loops.b, 4, 4) renames the instances of acoustic one for one
decided  isl    wavefront_acoustic.py:233  acoustic  the order after tile(acoustic.loops.a, acoustic.loops.b, 4, 4) runs every dependence of
                                                       acoustic forward
tested   loopy  wavefront_acoustic.py:233  acoustic  the scheduled run of acoustic agrees with the native run to the accuracy its types state

13 facts: 10 decided, 3 tested

WITHOUT LEAN: 10 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
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
its binders, 36 cells for 15 pairs; `Schedule(pairs).pack(pairs.f)` keeps the 15
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

schedule: Schedule(pairs, target='c').pack(pairs.f)
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

WITHOUT LEAN: 14 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

### loopty run examples/pairs.py

```console
$ uv run loopty run examples/pairs.py
pairs: Schedule(pairs, target='c')
  f: difference 0 within 1.04e-06 (approx) -> tested
  e: difference 0 within 1.93e-06 (approx) -> tested
pairs: Schedule(pairs, target='c').pack(pairs.f)
  f: difference 0 within 1.04e-06 (approx) -> tested
  e: difference 0 within 1.93e-06 (approx) -> tested

STATUS  BY     WHERE         OWNER  STATEMENT
------  -----  ------------  -----  -------------------------------------------------------------------------------------
tested  loopy  pairs.py:111  pairs  the scheduled run of pairs agrees with the native run to the accuracy its types state
tested  loopy  pairs.py:112  pairs  the scheduled run of pairs agrees with the native run to the accuracy its types state

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

WITHOUT LEAN: 10 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
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
------  -----  -----------------  -----------  -------------------------------------------------------------------------------------------
tested  loopy  composition.py:62  flux         the scheduled run of flux agrees with the native run to the accuracy its types state
tested  loopy  composition.py:69  divergence   the scheduled run of divergence agrees with the native run to the accuracy its types state
tested  loopy  composition.py:77  burgers_rhs  the scheduled run of burgers_rhs agrees with the native run to the accuracy its types state

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
of what a run has borne out: the postconditions here are decided from what
their kernels store and tested against their native runs too, and the
kernels' terms are tested against their bodies. A requirement decided under a postcondition its kernel's runs refute,
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

number's postcondition, forall i in Fin(n). perm[i] == n - 1 - i: decided

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

The postconditions are `decided` by `isl`, from what each kernel stores, each
after the `single-assignment` fact it rests on, and each requirement the
hypotheses decide is `decided` by `isl` and worth `tested`, what the
`trace-faithful` fact of the kernel whose postcondition it used is worth. The two the compiled programs check are
`assumed`, and say why in their provenance. The flat access of `flat` is
`decided`, under the element requirement it used, which the `STATUS` column
names, so it is worth an assumption checked when the program runs.

```console
$ uv run lanky check examples/travel.py
STATUS                                                       EFFECTIVE  BY             WHERE          OWNER          STATEMENT
-----------------------------------------------------------  ---------  -------------  -------------  -------------  -------------------------------------------------------------------------------------------------
decided                                                      decided    isl            travel.py:72   number         forall i in Fin(n). perm[i] == n - 1 - i
decided                                                      decided    type           travel.py:92   gather         x[perm[i]] is in bounds by type (perm[i] : Fin(n))
decided                                                      decided    call           travel.py:95   permuted       after number(...) in permuted: forall i in Fin(n). perm[i] == n - 1 - i
decided                                                      tested     isl            travel.py:99   permuted       the elements of perm are points of Fin(n) where gather is called at travel.py:99,
                                                                                                                       after number at travel.py:98 wrote perm
assumed                                                      assumed    -              travel.py:106  checked        the elements of perm are points of Fin(n) where gather is called at travel.py:106,
                                                                                                                       after number_quiet at travel.py:105 wrote perm (checked when it runs)
decided                                                      decided    isl            travel.py:115  scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
decided                                                      decided    call           travel.py:148  through        after scan(...) in through: off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
decided                                                      tested     isl            travel.py:152  through        off holds the offsets the counts in cnt give the rows of val
                                                                                                                       (off[0] == 0 and off[r + 1] == off[r] + cnt[r]) where rowsums is called at travel.py:152,
                                                                                                                       after scan at travel.py:151 wrote off
assumed                                                      assumed    -              travel.py:186  weigh          val[off[r] + j] is in bounds
decided                                                      decided    call           travel.py:189  flat           after scan_flat(...) in flat: off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
decided                                                      tested     isl            travel.py:193  flat           off holds the offsets the counts in cnt give the rows of wt
                                                                                                                       (off[0] == 0 and off[r + 1] == off[r] + cnt[r]) where weigh is called at travel.py:193,
                                                                                                                       after scan_flat at travel.py:192 wrote off
assumed                                                      assumed    -              travel.py:193  flat           the elements of off are points of Fin(nnz + 1) where weigh is called at travel.py:193,
                                                                                                                       after scan_flat at travel.py:192 wrote off (checked when it runs)
decided under requirement:travel.flat@189:weigh:element:off  assumed    isl            travel.py:186  flat           val[off[r_0] + j] is in bounds for every instance of weigh.S0,
                                                                                                                       where weigh is called at travel.py:193
...
76 facts: 4 assumed, 59 decided, 13 tested
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
------  -----  -------------  ------------  --------------------------------------------------------------------------------------------
tested  loopy  travel.py:72   number        the scheduled run of number agrees with the native run to the accuracy its types state
tested  loopy  travel.py:79   number_quiet  the scheduled run of number_quiet agrees with the native run to the accuracy its types state
tested  loopy  travel.py:86   gather        the scheduled run of gather agrees with the native run to the accuracy its types state
tested  loopy  travel.py:95   permuted      the scheduled run of permuted agrees with the native run to the accuracy its types state
tested  loopy  travel.py:102  checked       the scheduled run of checked agrees with the native run to the accuracy its types state
tested  loopy  travel.py:115  scan          the scheduled run of scan agrees with the native run to the accuracy its types state
tested  loopy  travel.py:136  rowsums       the scheduled run of rowsums agrees with the native run to the accuracy its types state
tested  loopy  travel.py:148  through       the scheduled run of through agrees with the native run to the accuracy its types state
tested  loopy  travel.py:161  scan_flat     the scheduled run of scan_flat agrees with the native run to the accuracy its types state
tested  loopy  travel.py:171  weigh         the scheduled run of weigh agrees with the native run to the accuracy its types state
tested  loopy  travel.py:189  flat          the scheduled run of flat agrees with the native run to the accuracy its types state

11 facts: 11 tested
```

## fusion.py

The program of `composition.py`, `burgers_rhs`, scheduled two steps further:
the Burgers flux and divergence of the closed #6, which fused the two kernels
with loopy's `fuse_kernels` and `assignment_to_subst` and checked neither.
Both steps are checked here before they are applied.

`fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1)` runs
`divergence`'s loop over `i` inside `flux`'s over `j`, one step behind. It is
`affine` with a map per statement, `divergence` at `i` in the step `j = i +
1`, and isl decides it on the dependences between the two calls: in each step
of the one loop `flux` stores `f[j]` first, and `divergence` reads `f[j]` and
`f[j - 2]`, both stored by then. Without the shift `divergence` at `i` would
read `f[i + 1]` a step before `flux` stores it, and that is the pair the
refusal names, with the fusion that is accepted.

`substitute(burgers_rhs.f)` then stores no flux at all: `flux`'s statement becomes a
substitution rule, and `divergence` computes `0.5 * u[j] * u[j]` where it read
`f[j]`. The step is legal because every cell of `f` that is read, `flux`
stored before the read (the `definedness` fact), and nothing writes `u` in
between (the `monotone` fact, over the program as it now runs). The temporary
goes, and with it the loop that zeroed it.

`contract(burgers_rhs.f)` keeps the flux instead, but only the three cells live at once:
`divergence` reads `f[j]` and `f[j - 2]` at step `j`, so the window `lambda j:
j % 3` is found, and `f` is `double f_window[3]`, each flux computed once. It
is legal by the same `definedness` fact and by `no-clobber`: no read sees its
slot overwritten, since between `flux` storing `f[j - 2]` and the read of it,
the fluxes at `j - 1` and `j` store into the other two slots. A window of two
cells is refused with the read and the flux that overwrote its slot, and from
then on the slots are dependences a later step has to keep.

### python examples/fusion.py

The two refusals, then each accepted schedule with its facts, the code loopy
generates for it, and its compiled run against the native one.

```console
$ uv run python examples/fusion.py
native: the program agrees with the numpy slices: True

refused: fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i) illegal: f[j] = 0.5 * u[j] * u[j] (fusion.py:84) at j=2 writes f[2], read by rhs[i] = -(f[i + 1] - f[i - 1]) / 2 (fusion.py:92) at i=1, scheduled earlier (at n=16, as hinted); fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) runs every dependence between them forward

refused: contract(burgers_rhs.f, lambda j: j % 2) illegal: rhs[i] = -(f[i + 1] - f[i - 1]) / 2 (fusion.py:92) at i=14 reads f[13], which f[j] = 0.5 * u[j] * u[j] (fusion.py:84) at j=13 stored in slot 1, after the same statement at j=15 stored f[15] there (at n=16, as hinted); contract(burgers_rhs.f) finds the smallest window that keeps every read, lambda j: j % 3

accepted: Schedule(burgers_rhs, target='c').fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1)
  decided  isl  fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) renames the instances of burgers_rhs one for one
  decided  isl  the order after fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) runs every dependence of burgers_rhs forward

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

accepted: Schedule(burgers_rhs, target='c').fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1).substitute(burgers_rhs.f)
  decided  isl  fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) renames the instances of burgers_rhs one for one
  decided  isl  the order after fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) runs every dependence of burgers_rhs forward
  decided  isl  for substitute(burgers_rhs.f), every cell of burgers_rhs.f that burgers_rhs reads, burgers_rhs.flux has stored by the time it is read
  decided  isl  substitute(burgers_rhs.f) renames the instances of burgers_rhs one for one
  decided  isl  the order after substitute(burgers_rhs.f) runs every dependence of burgers_rhs forward

#include <stdint.h>
#include <stdbool.h>

void burgers_rhs(int32_t const n, double const *__restrict__ u, double *__restrict__ rhs)
{
  for (int32_t j = 0; j <= -1 + n; ++j)
    if (-2 + j >= 0 && -1 + j > 0 && -1 + j + 1 < n)
      rhs[-1 + j] = (-1.0 * (0.5 * u[j] * u[j] + -1.0 * (0.5 * u[-2 + j] * u[-2 + j]))) / 2.0;
}

  rhs: difference 0 within 1e-06 (approx) -> tested

accepted: Schedule(burgers_rhs, target='c').fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1).contract(burgers_rhs.f, lambda j: j % 3)
  decided  isl  fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) renames the instances of burgers_rhs one for one
  decided  isl  the order after fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) runs every dependence of burgers_rhs forward
  decided  isl  for contract(burgers_rhs.f), every cell of burgers_rhs.f that burgers_rhs reads, burgers_rhs.flux has stored by the time it is read
  decided  isl  no read of burgers_rhs.f sees its slot overwritten after contract(burgers_rhs.f, lambda j: j % 3): burgers_rhs.f kept in 3 cells, f[j] in slot j % 3

#include <stdint.h>
#include <stdbool.h>

void burgers_rhs(int32_t const n, double const *__restrict__ u, double *__restrict__ rhs)
{
  double f_window[3];

  for (int32_t j = 0; j <= -1 + n; ++j)
  {
    f_window[j + -3 * (j / 3)] = 0.5 * u[j] * u[j];
    if (-2 + j >= 0 && -1 + j > 0 && -1 + j + 1 < n)
      rhs[-1 + j] = (-1.0 * (f_window[j + -3 * (j / 3)] + -1.0 * f_window[1 + j + -3 * ((1 + j) / 3)])) / 2.0;
  }
}

  rhs: difference 0 within 1e-06 (approx) -> tested
```

### lanky check examples/fusion.py

The kernels' obligations, and the program's: the definedness of `f`, decided
by isl, and the program's `trace-faithful` fact. Then the casts of the file's
schedules, under the program, each fact once: the fusion's, shared by the
schedules that start with it, and what the substitution and the window each
need, every cell of `f` stored before it is read, and the window's
`no-clobber`.

```console
$ uv run lanky check examples/fusion.py
STATUS   BY           WHERE          OWNER        STATEMENT
-------  -----------  -------------  -----------  ----------------------------------------------------------------------------------------------------
decided  isl          fusion.py:84   flux         f[j] is in bounds for every instance of S0
decided  isl          fusion.py:84   flux         u[j] is in bounds for every instance of S0
decided  isl          fusion.py:84   flux         distinct instances of S0 write distinct cells of f
decided  isl          fusion.py:80   flux         the source order runs every dependence forward in time
tested   interpreter  fusion.py:80   flux         the traced term computes what the body computes
decided  isl          fusion.py:92   divergence   rhs[i] is in bounds for every instance of S0
decided  isl          fusion.py:92   divergence   f[i + 1] is in bounds for every instance of S0
decided  isl          fusion.py:92   divergence   f[i - 1] is in bounds for every instance of S0
decided  isl          fusion.py:92   divergence   distinct instances of S0 write distinct cells of rhs
decided  isl          fusion.py:87   divergence   the source order runs every dependence forward in time
tested   interpreter  fusion.py:87   divergence   the traced term computes what the body computes
decided  isl          fusion.py:100  burgers_rhs  every cell of f that divergence reads, flux stored before it
tested   interpreter  fusion.py:95   burgers_rhs  the traced term computes what the body computes
decided  isl          fusion.py:104  burgers_rhs  fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) renames the instances of
                                                    burgers_rhs one for one
decided  isl          fusion.py:104  burgers_rhs  the order after fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) runs every dependence of
                                                    burgers_rhs forward
decided  isl          fusion.py:109  burgers_rhs  for substitute(burgers_rhs.f), every cell of burgers_rhs.f that burgers_rhs reads,
                                                    burgers_rhs.flux has stored by the time it is read
decided  isl          fusion.py:109  burgers_rhs  substitute(burgers_rhs.f) renames the instances of burgers_rhs one for one
decided  isl          fusion.py:109  burgers_rhs  the order after substitute(burgers_rhs.f) runs every dependence of burgers_rhs forward
decided  isl          fusion.py:112  burgers_rhs  for contract(burgers_rhs.f), every cell of burgers_rhs.f that burgers_rhs reads,
                                                    burgers_rhs.flux has stored by the time it is read
decided  isl          fusion.py:112  burgers_rhs  no read of burgers_rhs.f sees its slot overwritten after contract(burgers_rhs.f, lambda j: j % 3):
                                                    burgers_rhs.f kept in 3 cells, f[j] in slot j % 3

20 facts: 17 decided, 3 tested

WITHOUT LEAN: 17 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

### loopty run examples/fusion.py

Both schedules, compiled and compared with the native run, and each kernel
alone. The two schedules share their first step, and so its facts.

```console
$ uv run loopty run examples/fusion.py
burgers_rhs: Schedule(burgers_rhs, target='c').fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1)
  rhs: difference 0 within 1e-06 (approx) -> tested
burgers_rhs: Schedule(burgers_rhs, target='c').fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1).substitute(burgers_rhs.f)
  rhs: difference 0 within 1e-06 (approx) -> tested
burgers_rhs: Schedule(burgers_rhs, target='c').fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1).contract(burgers_rhs.f, lambda j: j % 3)
  rhs: difference 0 within 1e-06 (approx) -> tested
flux: Schedule(flux, target='c')
  f: difference 0 within 1e-06 (approx) -> tested
divergence: Schedule(divergence, target='c')
  rhs: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE          OWNER        STATEMENT
-------  -----  -------------  -----------  ----------------------------------------------------------------------------------------------------
decided  isl    fusion.py:104  burgers_rhs  fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) renames the instances of
                                              burgers_rhs one for one
decided  isl    fusion.py:104  burgers_rhs  the order after fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i, shift=1) runs every dependence of
                                              burgers_rhs forward
tested   loopy  fusion.py:104  burgers_rhs  the scheduled run of burgers_rhs agrees with the native run to the accuracy its types state
decided  isl    fusion.py:109  burgers_rhs  for substitute(burgers_rhs.f), every cell of burgers_rhs.f that burgers_rhs reads,
                                              burgers_rhs.flux has stored by the time it is read
decided  isl    fusion.py:109  burgers_rhs  substitute(burgers_rhs.f) renames the instances of burgers_rhs one for one
decided  isl    fusion.py:109  burgers_rhs  the order after substitute(burgers_rhs.f) runs every dependence of burgers_rhs forward
tested   loopy  fusion.py:109  burgers_rhs  the scheduled run of burgers_rhs agrees with the native run to the accuracy its types state
decided  isl    fusion.py:112  burgers_rhs  for contract(burgers_rhs.f), every cell of burgers_rhs.f that burgers_rhs reads,
                                              burgers_rhs.flux has stored by the time it is read
decided  isl    fusion.py:112  burgers_rhs  no read of burgers_rhs.f sees its slot overwritten after contract(burgers_rhs.f, lambda j: j % 3):
                                              burgers_rhs.f kept in 3 cells, f[j] in slot j % 3
tested   loopy  fusion.py:112  burgers_rhs  the scheduled run of burgers_rhs agrees with the native run to the accuracy its types state
tested   loopy  fusion.py:80   flux         the scheduled run of flux agrees with the native run to the accuracy its types state
tested   loopy  fusion.py:87   divergence   the scheduled run of divergence agrees with the native run to the accuracy its types state

12 facts: 7 decided, 5 tested

WITHOUT LEAN: 7 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
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

schedule: Schedule(mass_action, target='c').split(mass_action.c, 32)
  yl: difference 0 within 1.01e-06 (approx) -> tested
```

### lanky check examples/quadrature.py

```console
$ uv run lanky check examples/quadrature.py
STATUS   BY           WHERE              OWNER        STATEMENT
-------  -----------  -----------------  -----------  -------------------------------------------------------------------------------------
decided  isl          quadrature.py:78   mass_action  yl[c, j] is in bounds for every instance of S0
decided  type         quadrature.py:78   mass_action  X[cell[c, 1]] is in bounds by type (cell[c, 1] : Fin(nv))
decided  isl          quadrature.py:78   mass_action  cell[c, 1] is in bounds for every instance of S0
decided  type         quadrature.py:78   mass_action  X[cell[c, 0]] is in bounds by type (cell[c, 0] : Fin(nv))
decided  isl          quadrature.py:78   mass_action  cell[c, 0] is in bounds for every instance of S0
decided  type         quadrature.py:78   mass_action  Y[cell[c, 2]] is in bounds by type (cell[c, 2] : Fin(nv))
decided  isl          quadrature.py:78   mass_action  cell[c, 2] is in bounds for every instance of S0
decided  type         quadrature.py:78   mass_action  Y[cell[c, 0]] is in bounds by type (cell[c, 0] : Fin(nv))
decided  type         quadrature.py:78   mass_action  X[cell[c, 2]] is in bounds by type (cell[c, 2] : Fin(nv))
decided  type         quadrature.py:78   mass_action  Y[cell[c, 1]] is in bounds by type (cell[c, 1] : Fin(nv))
decided  isl          quadrature.py:78   mass_action  W[q] is in bounds for every instance of S0
decided  isl          quadrature.py:78   mass_action  PHI[q, j] is in bounds for every instance of S0
decided  isl          quadrature.py:78   mass_action  PHI[q, k] is in bounds for every instance of S0
decided  type         quadrature.py:78   mass_action  u[cell[c, k]] is in bounds by type (cell[c, k] : Fin(nv))
decided  isl          quadrature.py:78   mass_action  cell[c, k] is in bounds for every instance of S0
decided  isl          quadrature.py:78   mass_action  distinct instances of S0 write distinct cells of yl
decided  isl          quadrature.py:63   mass_action  the source order runs every dependence forward in time
decided  type         quadrature.py:78   mass_action  the accumulation into yl[c, j] over q, k is approx
tested   interpreter  quadrature.py:63   mass_action  the traced term computes what the body computes
decided  isl          quadrature.py:138  mass_action  split(mass_action.c, 32) renames the instances of mass_action one for one
decided  isl          quadrature.py:138  mass_action  the order after split(mass_action.c, 32) runs every dependence of mass_action forward

21 facts: 20 decided, 1 tested

WITHOUT LEAN: 12 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

### loopty run examples/quadrature.py

```console
$ uv run loopty run examples/quadrature.py
mass_action: Schedule(mass_action, target='c').split(mass_action.c, 32)
  yl: difference 0 within 1.01e-06 (approx) -> tested

STATUS   BY     WHERE              OWNER        STATEMENT
-------  -----  -----------------  -----------  -------------------------------------------------------------------------------------------
decided  isl    quadrature.py:138  mass_action  split(mass_action.c, 32) renames the instances of mass_action one for one
decided  isl    quadrature.py:138  mass_action  the order after split(mass_action.c, 32) runs every dependence of mass_action forward
tested   loopy  quadrature.py:138  mass_action  the scheduled run of mass_action agrees with the native run to the accuracy its types state

3 facts: 2 decided, 1 tested

WITHOUT LEAN: 2 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

## layer_potential.py

pytential's QBX on its own curve, the demonstration of three layers in one
file. pytential evaluates a layer potential at the points of a curve by
expanding the kernel about a center just off the curve; with no FMM that is
sumpy's `LayerPotential` with a line-Taylor expansion, the kernel along the
line from the center through the target. Read the points as complex numbers,
with `alpha = c - y` and `beta = z - c`, and its terms are

```text
log|alpha| + Re( c(1) + ... + c(5) ),   c(k) = -(-beta / alpha)**k / k
```

- `line_taylor` is the theorem behind these terms, for every order: lanky's
  `examples/line_taylor.py`, here in the same file. A table that satisfies
  what `(alpha + tau * beta) f'(tau) == beta` says about its coefficients is
  `c(k)`; that the Taylor coefficients of `log(alpha + tau * beta)` satisfy
  that equation is cited, not proved. With Mathlib, Lean proves it; without,
  it is `assumed`, since the property tester draws no table over `Nat`. The
  kernels add `coefficient(ratio(alpha, beta), k)`, the same function the
  theorem's conclusion is written with, at the ratio computed by the
  conjugate, `-beta * conj(alpha) / |alpha|**2`. `by_the_conjugate` says
  that is `-beta / alpha`, and `single_layer` cites both: its row
  `single_layer adds line_taylor's c(n) for n in 1..5, for every c with
  away, first and ode, read over the complex numbers` ties the theorem to the
  terms the kernel evaluates, `decided` by `theorem` and, with Mathlib,
  proved by Lean, the identity rewritten and the theorem applied a term at a
  time. It is about the term read over the complex numbers, not about the
  floats the kernel produces.
- `single_layer` and `double_layer` are pytential's `S` and `D` on the
  five-armed starfish, as sumpy computes them, over complex points.
  `sumpy_tile()` is sumpy's CPU schedule, `get_optimized_kernel(is_cpu=True)`:
  16 targets to a work group, the sources 256 at a time, the sources' loop
  outside the targets'. Its casts are decided, and on the C target it does not
  build, since C has no hardware axes; `tiled` is the same tile on C, with an
  `ilp` loop for the work items. sumpy silences loopy's
  `write_race(write_lpot*)` on this kernel; the fact in its place is
  `distinct instances of S0 write distinct cells of out`, decided by isl and,
  with Lean, proved.
- `sumpy_single_layer` and `sumpy_double_layer` are what sumpy's own
  `LayerPotential` returned on the same inputs, called as pytential calls it,
  recorded with the commits it ran at in `layer_potential_sumpy.json` (sumpy
  needs a newer loopy than loopty's, so it is not run here). Each is
  `@upstream` of its kernel, as `examples/l2p.py`'s `sumpy_l2p` is, and its
  row, `tested` by `upstream`, compares the kernel's native run with it
  within `1e-12` of the largest value, which the row states: round-off
  between two orders of summation, far inside the approx allowance.
- `green_identity`, `jump_S` and `jump_D` are pytential's identity test case
  on the starfish and the jump relations, evaluated directly
  (`fmm_order=False`) with the compiled kernels at 30 to 90 panels of this
  file's own discretization: each a `tested` row, the order the errors fall at
  above the QBX order less 1.6, as pytential's test asks (`loopty.tested`).

`benchmarks/qbx_layer_potential.py` records sumpy's results, runs the kernels
on what pytential itself hands sumpy on its refined starfish, and times them
beside sumpy's own kernel and plain loopy.

### python examples/layer_potential.py

```console
$ uv run python examples/layer_potential.py
pytential's starfish, QBX order 5, at 4 panels: 36 targets, 132 sources
  single_layer: compiled on C agrees with native to the approx tolerance: True -> tested
  double_layer: compiled on C agrees with native to the approx tolerance: True -> tested
  single_layer: agrees with sumpy's own kernel, as pytential calls it, within 1e-12 of the largest value -> tested
  double_layer: agrees with sumpy's own kernel, as pytential calls it, within 1e-12 of the largest value -> tested
  single_layer adds line_taylor's c(n) for n in 1..5, for every c with away, first and ode, read over the complex numbers -> decided

sumpy's tile, get_optimized_kernel(is_cpu=True): split(single_layer.t, 16).tag({single_layer.t.outer: 'g.0', single_layer.t.inner: 'l.0'}).split(single_layer.s, 256).prioritize(single_layer.s.outer, single_layer.t.inner)
  decided  isl  split(single_layer.t, 16) renames the instances of single_layer one for one
  decided  isl  the order after split(single_layer.t, 16) runs every dependence of single_layer forward
  decided  isl  tag({single_layer.t.outer: 'g.0', single_layer.t.inner: 'l.0'}) renames the instances of single_layer one for one
  decided  isl  the order after tag({single_layer.t.outer: 'g.0', single_layer.t.inner: 'l.0'}) runs every dependence of single_layer forward, within one work item
  decided  isl  split(single_layer.s, 256) renames the instances of single_layer one for one
  decided  isl  the order after split(single_layer.s, 256) runs every dependence of single_layer forward, within one work item
  decided  isl  prioritize(single_layer.s.outer, single_layer.t.inner) renames the instances of single_layer one for one
  decided  isl  the order after prioritize(single_layer.s.outer, single_layer.t.inner) runs every dependence of single_layer forward, within one work item
  decided  isl  distinct instances of S0 write distinct cells of out
  on C, an ilp loop for the work items: split(single_layer.t, 16).tag(single_layer.t.inner, 'ilp').split(single_layer.s, 256).prioritize(single_layer.s.outer, single_layer.t.inner)

accuracy at 30, 50, 70, 90 panels:
  green_identity  errors 1.4e-04, 3.5e-05, 9.1e-06, 2.8e-06: order 3.51 -> tested
  jump_S          errors 5.4e-04, 9.4e-05, 2.3e-05, 7.2e-06: order 3.91 -> tested
  jump_D          errors 4.2e-03, 1.0e-03, 2.4e-04, 6.6e-05: order 3.75 -> tested
```

### lanky check examples/layer_potential.py

```console
$ uv run lanky check examples/layer_potential.py
STATUS                     EFFECTIVE  BY             WHERE                   OWNER             STATEMENT
-------------------------  ---------  -------------  ----------------------  ----------------  ------------------------------------------------------------------------------------------------
assumed                    assumed    -              layer_potential.py:145  line_taylor       alpha : Complex, beta : Complex, c : Fn[Nat, Complex]
                                                                                               | away: alpha != 0
                                                                                               | first: alpha*c(1) == beta
                                                                                               | ode: forall n in Nat where n >= 1. (n + 1)*alpha*c(n + 1) + n*beta*c(n) == 0
                                                                                               |- forall n in Nat where n >= 1. c(n) == -(-beta / alpha)**n / n
tested                     tested     python         layer_potential.py:145  line_taylor       the term computes what the annotations compute
tested                     tested     property-test  layer_potential.py:175  by_the_conjugate  alpha : Complex, beta : Complex
                                                                                               | away: alpha != 0
                                                                                               |- -beta*conj(alpha)*(1 / real(alpha*conj(alpha))) == -beta / alpha
tested                     tested     python         layer_potential.py:175  by_the_conjugate  the term computes what the annotations compute
decided                    decided    isl            layer_potential.py:237  single_layer      out[t] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:237  single_layer      w[s] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:237  single_layer      c[t] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:237  single_layer      y[s] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:237  single_layer      z[t] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:237  single_layer      distinct instances of S0 write distinct cells of out
decided                    decided    isl            layer_potential.py:221  single_layer      the source order runs every dependence forward in time
decided                    decided    type           layer_potential.py:237  single_layer      the accumulation into out[t] over s is approx
decided under line_taylor  assumed    theorem        layer_potential.py:237  single_layer      single_layer adds line_taylor's c(n) for n in 1..5, for every c with away, first
                                                                                                 and ode, read over the complex numbers
tested                     tested     interpreter    layer_potential.py:221  single_layer      the traced term computes what the body computes
decided                    decided    isl            layer_potential.py:389  single_layer      split(single_layer.t, 16) renames the instances of single_layer one for one
decided                    decided    isl            layer_potential.py:389  single_layer      the order after split(single_layer.t, 16) runs every dependence of single_layer forward
decided                    decided    isl            layer_potential.py:390  single_layer      tag(single_layer.t.inner, 'ilp') renames the instances of single_layer one for one
decided                    decided    isl            layer_potential.py:390  single_layer      the order after tag(single_layer.t.inner, 'ilp') runs every dependence of single_layer forward
decided                    decided    isl            layer_potential.py:391  single_layer      split(single_layer.s, 256) renames the instances of single_layer one for one
decided                    decided    isl            layer_potential.py:391  single_layer      the order after split(single_layer.s, 256) runs every dependence of single_layer forward
decided                    decided    isl            layer_potential.py:392  single_layer      prioritize(single_layer.s.outer, single_layer.t.inner) renames the instances of
                                                                                                 single_layer one for one
decided                    decided    isl            layer_potential.py:392  single_layer      the order after prioritize(single_layer.s.outer, single_layer.t.inner) runs every dependence of
                                                                                                 single_layer forward
decided                    decided    isl            layer_potential.py:253  double_layer      out[t] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:253  double_layer      w[s] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:253  double_layer      n[s] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:253  double_layer      c[t] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:253  double_layer      y[s] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:253  double_layer      z[t] is in bounds for every instance of S0
decided                    decided    isl            layer_potential.py:253  double_layer      distinct instances of S0 write distinct cells of out
decided                    decided    isl            layer_potential.py:242  double_layer      the source order runs every dependence forward in time
decided                    decided    type           layer_potential.py:253  double_layer      the accumulation into out[t] over s is approx
tested                     tested     interpreter    layer_potential.py:242  double_layer      the traced term computes what the body computes
decided                    decided    isl            layer_potential.py:399  double_layer      split(double_layer.t, 16) renames the instances of double_layer one for one
decided                    decided    isl            layer_potential.py:399  double_layer      the order after split(double_layer.t, 16) runs every dependence of double_layer forward
decided                    decided    isl            layer_potential.py:400  double_layer      tag(double_layer.t.inner, 'ilp') renames the instances of double_layer one for one
decided                    decided    isl            layer_potential.py:400  double_layer      the order after tag(double_layer.t.inner, 'ilp') runs every dependence of double_layer forward
decided                    decided    isl            layer_potential.py:401  double_layer      split(double_layer.s, 256) renames the instances of double_layer one for one
decided                    decided    isl            layer_potential.py:401  double_layer      the order after split(double_layer.s, 256) runs every dependence of double_layer forward
decided                    decided    isl            layer_potential.py:402  double_layer      prioritize(double_layer.s.outer, double_layer.t.inner) renames the instances of
                                                                                                 double_layer one for one
decided                    decided    isl            layer_potential.py:402  double_layer      the order after prioritize(double_layer.s.outer, double_layer.t.inner) runs every dependence of
                                                                                                 double_layer forward
tested                     tested     upstream       layer_potential.py:422  single_layer      the native run of single_layer agrees with sumpy_single_layer in out within 1e-12 of the largest
                                                                                                 value, on its example inputs
tested                     tested     upstream       layer_potential.py:428  double_layer      the native run of double_layer agrees with sumpy_double_layer in out within 1e-12 of the largest
                                                                                                 value, on its example inputs
tested                     tested     run            layer_potential.py:511  green_identity    S(du/dn) - D(u) = u/2, Green's identity, at order 5 - 1.6 or faster
tested                     tested     run            layer_potential.py:517  jump_S            S(+) - S(-) = 0: S is continuous, at order 5 - 1.6 or faster
tested                     tested     run            layer_potential.py:523  jump_D            D(+) - D(-) = sigma: D jumps by the density, at order 5 - 1.6 or faster

45 facts: 1 assumed, 34 decided, 10 tested

WITHOUT LEAN: line_taylor, by_the_conjugate and 32 more rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
WITHOUT MATHLIB: line_taylor, by_the_conjugate and 1 more row are not proved, since only Lean with Mathlib can state them. To have Lean prove them: python -m lanky.mathlib ~/mathlib (about 7 GB), then export LANKY_LEAN_MATHLIB=~/mathlib.
```

With Lean, the 31 facts isl decided read `proved` by `lean`, the write fact
sumpy silences and the tile's casts among them (`45 facts: 1 assumed, 3
decided, 31 proved, 10 tested`); the row that ties `line_taylor` to
`single_layer` is about complex numbers, and waits for Mathlib. With Mathlib
too, `line_taylor` reads `proved` by `lean`, by induction two orders at a time
with the linear combinations Python found, and is worth `tested`: what the
theorem rests on includes the reading of its annotations, which lanky checks
by running them (the second row). `by_the_conjugate` reads `proved`, and so
does the row, which rests on the two theorems.

### loopty run examples/layer_potential.py

`uv run loopty run examples/layer_potential.py` compiles `tiled` and
`tiled_double` on the C target and compares each with the native run, and
with Lean proves their casts. Its transcript is not kept here: the difference
it prints is in the last bits of `log`, numpy's against the C library's,
which can differ from machine to machine. `--target opencl` runs both on an
OpenCL device; sumpy's own tags are `sumpy_tile()`, which the benchmark runs.

## sum_factorization.py

The weak derivative along `r` on quadrilaterals, as grudge applies it, a pass
per reference axis: the differentiation matrix along `r`, then the quadrature
weights along `s`, each pass a kernel over the elements and a program running
one after the other. Fused on every loop, the second pass would read a cell of
the row before the first stores it, and the refusal names that pair and the
fusion of fewer loops it checked,
`fuse(weak_derivative.along_r.a, weak_derivative.along_s.a)`, which fuses the
element and the row, the loops outside `a` and `a` itself, and leaves the
inner loops one after the other.
Each element's result is the Kronecker product of the two matrices applied to
it. Fused so, the second pass reads only the row the first has just stored, so
`contract(weak_derivative.v)` keeps `v` one row at a time, `lambda e, a, b: b`, and the
program stores `n` cells where it stored `ne * n * n`.

### python examples/sum_factorization.py

```console
$ uv run python examples/sum_factorization.py
native: each element is kron(D, M) applied to it: True

refused: fuse(weak_derivative.along_r.b, weak_derivative.along_s.b) illegal: v[e, a, b] = reduce_sum(...) (sum_factorization.py:83) at e=0, a=1, b=1 writes v[0, 1, 1], read by w[e, a, b] = reduce_sum(...) (sum_factorization.py:96) at e=0, a=1, b=0, scheduled earlier (at n=2, ne=1, isl's choice); fuse(weak_derivative.along_r.a, weak_derivative.along_s.a) runs every dependence between them forward

accepted: Schedule(weak_derivative, target='c').fuse(weak_derivative.along_r.a, weak_derivative.along_s.a)
  the passes' loops, outermost first: weak_derivative.along_r.e, weak_derivative.along_r.a, weak_derivative.along_r.b, weak_derivative.along_s.b
  decided  isl  fuse(weak_derivative.along_r.a, weak_derivative.along_s.a) renames the instances of weak_derivative one for one
  decided  isl  the order after fuse(weak_derivative.along_r.a, weak_derivative.along_s.a) runs every dependence of weak_derivative forward
  w: difference 0 within 1e-06 (approx) -> tested

accepted: Schedule(weak_derivative, target='c').fuse(weak_derivative.along_r.a, weak_derivative.along_s.a).contract(weak_derivative.v, lambda e, a, b: b)
  v kept in n cells, lambda e, a, b: b
  decided  isl  for contract(weak_derivative.v), every cell of weak_derivative.v that weak_derivative reads, weak_derivative.along_r has stored by the time it is read
  decided  isl  no read of weak_derivative.v sees its slot overwritten after contract(weak_derivative.v, lambda e, a, b: b): weak_derivative.v kept in n cells, v[e, a, b] in slot b
  w: agrees with the native run -> tested
```

### lanky check examples/sum_factorization.py

```console
$ uv run lanky check examples/sum_factorization.py
STATUS   BY           WHERE                     OWNER            STATEMENT
-------  -----------  ------------------------  ---------------  ---------------------------------------------------------------------------------------------------
decided  isl          sum_factorization.py:83   along_r          v[e, a, b] is in bounds for every instance of S0
decided  isl          sum_factorization.py:83   along_r          F[a, m] is in bounds for every instance of S0
decided  isl          sum_factorization.py:83   along_r          u[e, m, b] is in bounds for every instance of S0
decided  isl          sum_factorization.py:83   along_r          distinct instances of S0 write distinct cells of v
decided  isl          sum_factorization.py:73   along_r          the source order runs every dependence forward in time
decided  type         sum_factorization.py:83   along_r          the accumulation into v[e, a, b] over m is approx
tested   interpreter  sum_factorization.py:73   along_r          the traced term computes what the body computes
decided  isl          sum_factorization.py:96   along_s          w[e, a, b] is in bounds for every instance of S0
decided  isl          sum_factorization.py:96   along_s          G[b, m] is in bounds for every instance of S0
decided  isl          sum_factorization.py:96   along_s          v[e, a, m] is in bounds for every instance of S0
decided  isl          sum_factorization.py:96   along_s          distinct instances of S0 write distinct cells of w
decided  isl          sum_factorization.py:86   along_s          the source order runs every dependence forward in time
decided  type         sum_factorization.py:96   along_s          the accumulation into w[e, a, b] over m is approx
tested   interpreter  sum_factorization.py:86   along_s          the traced term computes what the body computes
decided  isl          sum_factorization.py:104  weak_derivative  every cell of v that along_s reads, along_r stored before it
tested   interpreter  sum_factorization.py:99   weak_derivative  the traced term computes what the body computes
decided  isl          sum_factorization.py:108  weak_derivative  fuse(weak_derivative.along_r.a, weak_derivative.along_s.a) renames the instances of
                                                                   weak_derivative one for one
decided  isl          sum_factorization.py:108  weak_derivative  the order after fuse(weak_derivative.along_r.a, weak_derivative.along_s.a) runs every dependence of
                                                                   weak_derivative forward
decided  isl          sum_factorization.py:113  weak_derivative  for contract(weak_derivative.v), every cell of weak_derivative.v that weak_derivative reads,
                                                                   weak_derivative.along_r has stored by the time it is read
decided  isl          sum_factorization.py:113  weak_derivative  no read of weak_derivative.v sees its slot overwritten after
                                                                   contract(weak_derivative.v, lambda e, a, b: b):
                                                                   weak_derivative.v kept in n cells, v[e, a, b] in slot b

20 facts: 17 decided, 3 tested

WITHOUT LEAN: 15 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

### loopty run examples/sum_factorization.py

```console
$ uv run loopty run examples/sum_factorization.py
weak_derivative: Schedule(weak_derivative, target='c').fuse(weak_derivative.along_r.a, weak_derivative.along_s.a)
  w: difference 0 within 1e-06 (approx) -> tested
weak_derivative: Schedule(weak_derivative, target='c').fuse(weak_derivative.along_r.a, weak_derivative.along_s.a).contract(weak_derivative.v, lambda e, a, b: b)
  w: difference 0 within 1e-06 (approx) -> tested
along_r: Schedule(along_r, target='c')
  v: difference 0 within 1.02e-06 (approx) -> tested
along_s: Schedule(along_s, target='c')
  w: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE                     OWNER            STATEMENT
-------  -----  ------------------------  ---------------  ---------------------------------------------------------------------------------------------------
decided  isl    sum_factorization.py:108  weak_derivative  fuse(weak_derivative.along_r.a, weak_derivative.along_s.a) renames the instances of
                                                             weak_derivative one for one
decided  isl    sum_factorization.py:108  weak_derivative  the order after fuse(weak_derivative.along_r.a, weak_derivative.along_s.a) runs every dependence of
                                                             weak_derivative forward
tested   loopy  sum_factorization.py:108  weak_derivative  the scheduled run of weak_derivative agrees with the native run to the accuracy its types state
decided  isl    sum_factorization.py:113  weak_derivative  for contract(weak_derivative.v), every cell of weak_derivative.v that weak_derivative reads,
                                                             weak_derivative.along_r has stored by the time it is read
decided  isl    sum_factorization.py:113  weak_derivative  no read of weak_derivative.v sees its slot overwritten after
                                                             contract(weak_derivative.v, lambda e, a, b: b):
                                                             weak_derivative.v kept in n cells, v[e, a, b] in slot b
tested   loopy  sum_factorization.py:113  weak_derivative  the scheduled run of weak_derivative agrees with the native run to the accuracy its types state
tested   loopy  sum_factorization.py:73   along_r          the scheduled run of along_r agrees with the native run to the accuracy its types state
tested   loopy  sum_factorization.py:86   along_s          the scheduled run of along_s agrees with the native run to the accuracy its types state

8 facts: 4 decided, 4 tested

WITHOUT LEAN: 4 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

## storage_layouts.py

Two layouts from the codes the demos come from, each a Python function from a
cell's indices to its address, with its cells and its buffer in its
annotations. sumpy's storage index keeps the compressed derivatives of a 2-D
Laplace local expansion, `(0, 0)` at 0 and then two per order, which no one
formula gives, so it is a layout of two pieces, `layout(centre,
derivatives)`. grudge's DOFArray is `(ne, N**3)`, each element reshaped by
modepy in F order, and grudge #418 writes the same layout as loopy's
`dim_tags` `N3,N0,N1,N2`: `dof_array.same_as("N3,N0,N1,N2")` is that claim,
and `N3,N2,N1,N0`, the reshape in C order, is refuted with a cell. Each layout
stores every cell inside its buffer, no two at one address and a cell at every
address; with Lean installed, the last is proved with isl's inverse of the
layout as the witness, `//` and `%` by 2 for sumpy's. Two kernels read
through the layouts' own functions, `coeffs[i, derivatives((a, b))]`.

### python examples/storage_layouts.py

```console
$ uv run python examples/storage_layouts.py
sumpy's storage index, (a, b) -> address:
  (0, 0) -> 0, (0, 1) -> 1, (1, 0) -> 2, (0, 2) -> 3, (1, 1) -> 4, (0, 3) -> 5, (1, 2) -> 6, (0, 4) -> 7, (1, 3) -> 8
  box 0 read through it: R[0, 1, 3] = 8 (address 8)
grudge's DOFArray through dof_array is modepy's reshape: True
refuted: dof_array stores every cell where dim_tags N3,N2,N1,N0 does
  dof_array stores (e=0, a=1, b=0, c=2) at 19, and dim_tags N3,N2,N1,N0 at 11 (at ne=1)
```

### lanky check examples/storage_layouts.py

```console
$ uv run lanky check examples/storage_layouts.py
STATUS   BY           WHERE                   OWNER          STATEMENT
-------  -----------  ----------------------  -------------  ------------------------------------------------------------------
decided  isl          storage_layouts.py:56   storage_index  storage_index stores every cell at an address of Fin(9)
decided  isl          storage_layouts.py:56   storage_index  storage_index stores no two cells at one address
decided  isl          storage_layouts.py:56   storage_index  storage_index stores a cell at every address of Fin(9)
decided  isl          storage_layouts.py:83   stored_table   R[i, a, b] is in bounds for every instance of S0
decided  isl          storage_layouts.py:83   stored_table   coeffs[i, 0] is in bounds for every instance of S0
decided  isl          storage_layouts.py:85   stored_table   R[i, a, b] is in bounds for every instance of S1
decided  isl          storage_layouts.py:85   stored_table   coeffs[i, 2*(a + b) - 1 + a] is in bounds for every instance of S1
decided  isl          storage_layouts.py:83   stored_table   distinct instances of S0 write distinct cells of R
decided  isl          storage_layouts.py:85   stored_table   distinct instances of S1 write distinct cells of R
decided  isl          storage_layouts.py:73   stored_table   the source order runs every dependence forward in time
tested   interpreter  storage_layouts.py:73   stored_table   the traced term computes what the body computes
decided  isl          storage_layouts.py:94   dof_array      dof_array stores every cell at an address of Fin(ne*27)
decided  isl          storage_layouts.py:94   dof_array      dof_array stores no two cells at one address
decided  isl          storage_layouts.py:94   dof_array      dof_array stores a cell at every address of Fin(ne*27)
decided  isl          storage_layouts.py:101  as_dim_tags    dof_array stores every cell where dim_tags N3,N0,N1,N2 does
decided  isl          storage_layouts.py:114  unflatten      u[e, a, b, c] is in bounds for every instance of S0
decided  isl          storage_layouts.py:114  unflatten      flat[27*e + a + 3*b + 9*c] is in bounds for every instance of S0
decided  isl          storage_layouts.py:114  unflatten      distinct instances of S0 write distinct cells of u
decided  isl          storage_layouts.py:104  unflatten      the source order runs every dependence forward in time
tested   interpreter  storage_layouts.py:104  unflatten      the traced term computes what the body computes

20 facts: 18 decided, 2 tested

WITHOUT LEAN: 18 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

### loopty run examples/storage_layouts.py

```console
$ uv run loopty run examples/storage_layouts.py
stored_table: Schedule(stored_table, target='c')
  R: difference 0 within 1e-06 (approx) -> tested
unflatten: Schedule(unflatten, target='c')
  u: difference 0 within 1e-06 (approx) -> tested

STATUS  BY     WHERE                   OWNER         STATEMENT
------  -----  ----------------------  ------------  --------------------------------------------------------------------------------------------
tested  loopy  storage_layouts.py:73   stored_table  the scheduled run of stored_table agrees with the native run to the accuracy its types state
tested  loopy  storage_layouts.py:104  unflatten     the scheduled run of unflatten agrees with the native run to the accuracy its types state

2 facts: 2 tested
```

## hexahedra.py

grudge's weak gradient and divergence on hexahedra (grudge PR #418, as of
a-alveyblanc/grudge 4d103bb), written for `n` nodes per axis, so every fact
about the kernels and their schedules holds at every order; so do the
layouts, which isl tests at `n` from 1 to 6, since `n * j` is not linear,
and Lean proves for every `n`.
Each weak derivative is the field times the metric
terms `J dr_d/dx_x`, then a pass per reference axis with the factors
`make_stiffness_t_operator` builds, `S^T` on the derivative's axis and the
mass `M` on the others, then the three directions summed: the kernels
`metric` (or `contravariant`, for the divergence), `along_r`, `along_s` and
`gradient_along_t` (`divergence_along_t`), and the programs `weak_grad` and
`weak_div` that chain them. A node `(i, j, k)` of element `e`, `i` along `r`,
is `u[e, k, j, i]`: grudge's DOFArray is row-major with modepy's reshape of
each element in F order, which is #418's `dim_tags` `N3,N0,N1,N2`, and both
are the kernels' row-major arrays, three layouts that are one, each a fact.
The theorem `sum_factorization` is the reason the passes compute the dense
operator: `Ft (x) Fs (x) Fr` at a node is three sums of `n` terms, proved for
every `n` by Lean with Mathlib, and tested without it. The programs say so:
`weak_grad` claims that `grad` is the dense operator of each direction
applied to `u` times the direction's metric term, the three directions
summed, and `weak_div` the same of the flux. isl does not decide such a
claim, so Lean is asked: with Mathlib it proves each from what the four
calls store, a `single-assignment` row each, and the theorem, taken at each
direction's sections (`Fr (3 * x + 0)` for its `Fr`, `w (3 * x + 0) e` for
its `u`), and both rows read `proved lean`. Without Mathlib the claims are
`tested` on the programs' runs, compared at the class of `grad` and `div`,
`approx`, since the passes sum in another order than the dense operator.

The passes fuse as deep as each pair allows. `along_s` reads a whole column
of a `k`-slab `along_r` stored, so the two fuse on `(q, e, k)`, and fused on
every loop they are refused with the pair, the `j` that reads before the
store; the metric product fuses with `along_r` on every loop but the row's.
Fused so, `w` is read a row at a time and `v` a slab at a time, and
`contract` keeps them in `n` and `n * n` cells. On a device, a program is one
kernel and nothing orders two work items of it, so the passes are three
launches of their own, the element a work group and its nodes the work
items, each a schedule loopty checks, that no dependence joins two work
items included, which Lean proves where it is installed:
`passes_on_a_device`, which `benchmarks/hexahedra.py` times beside the same
kernels in plain loopy. Its `c` mode times `fused` and `contracted` against
the passes one after another on the C target, with the same flags.

### python examples/hexahedra.py

```console
$ uv run python examples/hexahedra.py
order 3, 2 hexahedra: the passes are the dense Kronecker operators: True (largest difference 5e-16 of the largest)

refused: fuse(weak_grad.along_r.i, weak_grad.along_s.i) illegal: v[q, e, k, j, i] = reduce_sum(...) (hexahedra.py:224) at q=8, e=0, k=1, j=1, i=1 writes v[8, 0, 1, 1, 1], read by vs[q, e, k, j, i] = reduce_sum(...) (hexahedra.py:241) at q=8, e=0, k=1, j=0, i=1, scheduled earlier (at n=2, ne=1, isl's choice); fuse(weak_grad.along_r.k, weak_grad.along_s.k) runs every dependence between them forward

accepted: Schedule(weak_grad, target='c').fuse(weak_grad.metric.j, weak_grad.along_r.j).fuse(weak_grad.along_r.k, weak_grad.along_s.k).contract(weak_grad.w, lambda q, e, k, j, i: i).contract(weak_grad.v, lambda q, e, k, j, i: (j, i))
  w kept in lambda q, e, k, j, i: i
  v kept in lambda q, e, k, j, i: (j, i)
  decided  isl  fuse(weak_grad.metric.j, weak_grad.along_r.j) renames the instances of weak_grad one for one
  decided  isl  the order after fuse(weak_grad.metric.j, weak_grad.along_r.j) runs every dependence of weak_grad forward
  decided  isl  fuse(weak_grad.along_r.k, weak_grad.along_s.k) renames the instances of weak_grad one for one
  decided  isl  the order after fuse(weak_grad.along_r.k, weak_grad.along_s.k) runs every dependence of weak_grad forward
  decided  isl  for contract(weak_grad.w), every cell of weak_grad.w that weak_grad reads, weak_grad.metric has stored by the time it is read
  decided  isl  no read of weak_grad.w sees its slot overwritten after contract(weak_grad.w, lambda q, e, k, j, i: i): weak_grad.w kept in n cells, w[q, e, k, j, i] in slot i
  decided  isl  for contract(weak_grad.v), every cell of weak_grad.v that weak_grad reads, weak_grad.along_r has stored by the time it is read
  decided  isl  no read of weak_grad.v sees its slot overwritten after contract(weak_grad.v, lambda q, e, k, j, i: (j, i)): weak_grad.v kept in n*n cells, v[q, e, k, j, i] in slot (j, i)
  grad: difference 0 within 1e-06 (approx) -> tested

on a device, the weak gradient as three launches, every node on a work item:
  metric_along_r: substitute(metric_along_r.w).tag({metric_along_r.along_r.e: 'g.0', metric_along_r.along_r.k: 'l.2', metric_along_r.along_r.j: 'l.1', metric_along_r.along_r.i: 'l.0'}): 5 facts, all decided by isl
  along_s: tag({along_s.e: 'g.0', along_s.k: 'l.2', along_s.j: 'l.1', along_s.i: 'l.0'}): 2 facts, all decided by isl
  gradient_along_t: tag({gradient_along_t.e: 'g.0', gradient_along_t.k: 'l.2', gradient_along_t.j: 'l.1', gradient_along_t.i: 'l.0'}): 2 facts, all decided by isl
```

### lanky check examples/hexahedra.py

```console
$ uv run lanky check examples/hexahedra.py
STATUS   BY             WHERE             OWNER                  STATEMENT
-------  -------------  ----------------  ---------------------  ---------------------------------------------------------------------------------------------------
tested   property-test  hexahedra.py:119  sum_factorization      n : Nat, Fr : Fn[Fin(n), Fin(n), Real], Fs : Fn[Fin(n), Fin(n), Real],
                                                                   Ft : Fn[Fin(n), Fin(n), Real], u : Fn[Fin(n), Fin(n), Fin(n), Real], k : Fin(n),
                                                                   j : Fin(n), i : Fin(n)
                                                                 |- sum(Ft[k, c]*Fs[j, b]*Fr[i, a]*u[c, b, a] for c in Fin(n), b in Fin(n), a in Fin(n))
                                                                     == sum(Ft[k, c]*sum(Fs[j, b]*sum(Fr[i, a]*u[c, b, a] for a in Fin(n)) for b in Fin(n))
                                                                       for c in Fin(n))
tested   python         hexahedra.py:119  sum_factorization      the term computes what the annotations compute
tested   isl            hexahedra.py:154  dof_array              dof_array stores every cell at an address of Fin(ne*n**3)
tested   isl            hexahedra.py:154  dof_array              dof_array stores no two cells at one address
tested   isl            hexahedra.py:154  dof_array              dof_array stores a cell at every address of Fin(ne*n**3)
tested   isl            hexahedra.py:161  as_dim_tags            dof_array stores every cell where dim_tags N3,N0,N1,N2 does
tested   isl            hexahedra.py:164  row_major              row_major stores every cell at an address of Fin(ne*n**3)
tested   isl            hexahedra.py:164  row_major              row_major stores no two cells at one address
tested   isl            hexahedra.py:164  row_major              row_major stores a cell at every address of Fin(ne*n**3)
tested   isl            hexahedra.py:171  as_row_major           dof_array stores every cell where row_major does
decided  isl            hexahedra.py:192  metric                 w[q, e, k, j, i] is in bounds for every instance of S0
decided  isl            hexahedra.py:192  metric                 G[q, e, k, j, i] is in bounds for every instance of S0
decided  isl            hexahedra.py:192  metric                 u[e, k, j, i] is in bounds for every instance of S0
decided  isl            hexahedra.py:192  metric                 distinct instances of S0 write distinct cells of w
decided  isl            hexahedra.py:180  metric                 the source order runs every dependence forward in time
tested   interpreter    hexahedra.py:180  metric                 the traced term computes what the body computes
decided  isl            hexahedra.py:207  contravariant          w[d, e, k, j, i] is in bounds for every instance of S0
decided  isl            hexahedra.py:207  contravariant          G[3*x + d, e, k, j, i] is in bounds for every x in Fin(3) and every instance of S0
decided  isl            hexahedra.py:207  contravariant          f[x, e, k, j, i] is in bounds for every x in Fin(3) and every instance of S0
decided  isl            hexahedra.py:207  contravariant          distinct instances of S0 write distinct cells of w
decided  isl            hexahedra.py:195  contravariant          the source order runs every dependence forward in time
tested   interpreter    hexahedra.py:195  contravariant          the traced term computes what the body computes
decided  isl            hexahedra.py:224  along_r                v[q, e, k, j, i] is in bounds for every instance of S0
decided  isl            hexahedra.py:224  along_r                Fr[q, i, a] is in bounds for every instance of S0
decided  isl            hexahedra.py:224  along_r                w[q, e, k, j, a] is in bounds for every instance of S0
decided  isl            hexahedra.py:224  along_r                distinct instances of S0 write distinct cells of v
decided  isl            hexahedra.py:212  along_r                the source order runs every dependence forward in time
decided  type           hexahedra.py:224  along_r                the accumulation into v[q, e, k, j, i] over a is approx
tested   interpreter    hexahedra.py:212  along_r                the traced term computes what the body computes
decided  isl            hexahedra.py:241  along_s                vs[q, e, k, j, i] is in bounds for every instance of S0
decided  isl            hexahedra.py:241  along_s                Fs[q, j, b] is in bounds for every instance of S0
decided  isl            hexahedra.py:241  along_s                v[q, e, k, b, i] is in bounds for every instance of S0
decided  isl            hexahedra.py:241  along_s                distinct instances of S0 write distinct cells of vs
decided  isl            hexahedra.py:229  along_s                the source order runs every dependence forward in time
decided  type           hexahedra.py:241  along_s                the accumulation into vs[q, e, k, j, i] over b is approx
tested   interpreter    hexahedra.py:229  along_s                the traced term computes what the body computes
decided  isl            hexahedra.py:258  gradient_along_t       grad[x, e, k, j, i] is in bounds for every instance of S0
decided  isl            hexahedra.py:258  gradient_along_t       Ft[3*x + d, k, c] is in bounds for every d in Fin(3) and every instance of S0
decided  isl            hexahedra.py:258  gradient_along_t       vs[3*x + d, e, c, j, i] is in bounds for every d in Fin(3) and every instance of S0
decided  isl            hexahedra.py:258  gradient_along_t       distinct instances of S0 write distinct cells of grad
decided  isl            hexahedra.py:246  gradient_along_t       the source order runs every dependence forward in time
decided  type           hexahedra.py:258  gradient_along_t       the accumulation into grad[x, e, k, j, i] over c is approx for every d in Fin(3)
tested   interpreter    hexahedra.py:246  gradient_along_t       the traced term computes what the body computes
decided  isl            hexahedra.py:278  divergence_along_t     div[e, k, j, i] is in bounds for every instance of S0
decided  isl            hexahedra.py:278  divergence_along_t     Ft[d, k, c] is in bounds for every d in Fin(3) and every instance of S0
decided  isl            hexahedra.py:278  divergence_along_t     vs[d, e, c, j, i] is in bounds for every d in Fin(3) and every instance of S0
decided  isl            hexahedra.py:278  divergence_along_t     distinct instances of S0 write distinct cells of div
decided  isl            hexahedra.py:267  divergence_along_t     the source order runs every dependence forward in time
decided  type           hexahedra.py:278  divergence_along_t     the accumulation into div[e, k, j, i] over c is approx for every d in Fin(3)
tested   interpreter    hexahedra.py:267  divergence_along_t     the traced term computes what the body computes
decided  isl            hexahedra.py:317  weak_grad              every cell of w that along_r reads, metric stored before it
decided  isl            hexahedra.py:318  weak_grad              every cell of v that along_s reads, along_r stored before it
decided  isl            hexahedra.py:319  weak_grad              every cell of vs that gradient_along_t reads, along_s stored before it
decided  isl            hexahedra.py:192  weak_grad              metric stores no cell of w twice, and reads none before its store
decided  isl            hexahedra.py:224  weak_grad              along_r stores no cell of v twice, and reads none before its store
decided  isl            hexahedra.py:241  weak_grad              along_s stores no cell of vs twice, and reads none before its store
decided  isl            hexahedra.py:258  weak_grad              gradient_along_t stores no cell of grad twice, and reads none before its store
tested   native         hexahedra.py:286  weak_grad              forall x in Fin(3), e in Fin(ne), k in Fin(n), j in Fin(n), i in Fin(n).
                                                                   grad[x, e, k, j, i]
                                                                   == sum(Ft[3*x + d, k, c]*Fs[3*x + d, j, b]*Fr[3*x + d, i, a]*G[3*x + d, e, c, b, a]*u[e, c, b, a]
                                                                     for d in Fin(3), c in Fin(n), b in Fin(n), a in Fin(n))
tested   interpreter    hexahedra.py:286  weak_grad              the traced term computes what the body computes
decided  isl            hexahedra.py:364  weak_grad              fuse(weak_grad.metric.j, weak_grad.along_r.j) renames the instances of weak_grad one for one
decided  isl            hexahedra.py:364  weak_grad              the order after fuse(weak_grad.metric.j, weak_grad.along_r.j) runs every dependence of
                                                                   weak_grad forward
decided  isl            hexahedra.py:365  weak_grad              fuse(weak_grad.along_r.k, weak_grad.along_s.k) renames the instances of weak_grad one for one
decided  isl            hexahedra.py:365  weak_grad              the order after fuse(weak_grad.along_r.k, weak_grad.along_s.k) runs every dependence of
                                                                   weak_grad forward
decided  isl            hexahedra.py:369  weak_grad              for contract(weak_grad.w), every cell of weak_grad.w that weak_grad reads,
                                                                   weak_grad.metric has stored by the time it is read
decided  isl            hexahedra.py:369  weak_grad              no read of weak_grad.w sees its slot overwritten after
                                                                   contract(weak_grad.w, lambda q, e, k, j, i: i):
                                                                   weak_grad.w kept in n cells, w[q, e, k, j, i] in slot i
decided  isl            hexahedra.py:369  weak_grad              for contract(weak_grad.v), every cell of weak_grad.v that weak_grad reads,
                                                                   weak_grad.along_r has stored by the time it is read
decided  isl            hexahedra.py:369  weak_grad              no read of weak_grad.v sees its slot overwritten after
                                                                   contract(weak_grad.v, lambda q, e, k, j, i: (j, i)):
                                                                   weak_grad.v kept in n*n cells, v[q, e, k, j, i] in slot (j, i)
decided  isl            hexahedra.py:350  weak_div               every cell of w that along_r reads, contravariant stored before it
decided  isl            hexahedra.py:351  weak_div               every cell of v that along_s reads, along_r stored before it
decided  isl            hexahedra.py:352  weak_div               every cell of vs that divergence_along_t reads, along_s stored before it
decided  isl            hexahedra.py:207  weak_div               contravariant stores no cell of w twice, and reads none before its store
decided  isl            hexahedra.py:224  weak_div               along_r stores no cell of v twice, and reads none before its store
decided  isl            hexahedra.py:241  weak_div               along_s stores no cell of vs twice, and reads none before its store
decided  isl            hexahedra.py:278  weak_div               divergence_along_t stores no cell of div twice, and reads none before its store
tested   native         hexahedra.py:322  weak_div               forall e in Fin(ne), k in Fin(n), j in Fin(n), i in Fin(n).
                                                                   div[e, k, j, i] ==
                                                                   sum(Ft[d, k, c]*Fs[d, j, b]*Fr[d, i, a]*sum(G[3*x + d, e, c, b, a]*f[x, e, c, b, a]
                                                                       for x in Fin(3)) for d in Fin(3), c in Fin(n), b in Fin(n), a in Fin(n))
tested   interpreter    hexahedra.py:322  weak_div               the traced term computes what the body computes
decided  isl            hexahedra.py:377  metric_along_r         every cell of w that along_r reads, metric stored before it
tested   interpreter    hexahedra.py:372  metric_along_r         the traced term computes what the body computes
decided  isl            hexahedra.py:385  contravariant_along_r  every cell of w that along_r reads, contravariant stored before it
tested   interpreter    hexahedra.py:380  contravariant_along_r  the traced term computes what the body computes

80 facts: 58 decided, 22 tested

WITHOUT LEAN: sum_factorization and 64 more rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
WITHOUT MATHLIB: sum_factorization and 2 more rows are not proved, since only Lean with Mathlib can state them. To have Lean prove them: python -m lanky.mathlib ~/mathlib (about 7 GB), then export LANKY_LEAN_MATHLIB=~/mathlib.
```

### loopty run examples/hexahedra.py

```console
$ uv run loopty run examples/hexahedra.py
weak_grad: Schedule(weak_grad, target='c').fuse(weak_grad.metric.j, weak_grad.along_r.j).fuse(weak_grad.along_r.k, weak_grad.along_s.k)
  grad: difference 0 within 1e-06 (approx) -> tested
weak_grad: Schedule(weak_grad, target='c').fuse(weak_grad.metric.j, weak_grad.along_r.j).fuse(weak_grad.along_r.k, weak_grad.along_s.k).contract(weak_grad.w, lambda q, e, k, j, i: i).contract(weak_grad.v, lambda q, e, k, j, i: (j, i))
  grad: difference 0 within 1e-06 (approx) -> tested
metric: Schedule(metric, target='c')
  w: difference 0 within 1e-06 (approx) -> tested
contravariant: Schedule(contravariant, target='c')
  w: difference 0 within 1e-06 (approx) -> tested
along_r: Schedule(along_r, target='c')
  v: difference 0 within 1e-06 (approx) -> tested
along_s: Schedule(along_s, target='c')
  vs: difference 0 within 1e-06 (approx) -> tested
gradient_along_t: Schedule(gradient_along_t, target='c')
  grad: difference 0 within 1e-06 (approx) -> tested
divergence_along_t: Schedule(divergence_along_t, target='c')
  div: difference 0 within 1.01e-06 (approx) -> tested
weak_div: Schedule(weak_div, target='c')
  div: difference 0 within 1e-06 (approx) -> tested
metric_along_r: Schedule(metric_along_r, target='c')
  v: difference 0 within 1e-06 (approx) -> tested
contravariant_along_r: Schedule(contravariant_along_r, target='c')
  v: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE             OWNER                  STATEMENT
-------  -----  ----------------  ---------------------  --------------------------------------------------------------------------------------------------
decided  isl    hexahedra.py:364  weak_grad              fuse(weak_grad.metric.j, weak_grad.along_r.j) renames the instances of weak_grad one for one
decided  isl    hexahedra.py:364  weak_grad              the order after fuse(weak_grad.metric.j, weak_grad.along_r.j) runs every dependence of
                                                           weak_grad forward
decided  isl    hexahedra.py:365  weak_grad              fuse(weak_grad.along_r.k, weak_grad.along_s.k) renames the instances of weak_grad one for one
decided  isl    hexahedra.py:365  weak_grad              the order after fuse(weak_grad.along_r.k, weak_grad.along_s.k) runs every dependence of
                                                           weak_grad forward
tested   loopy  hexahedra.py:365  weak_grad              the scheduled run of weak_grad agrees with the native run to the accuracy its types state
decided  isl    hexahedra.py:369  weak_grad              for contract(weak_grad.w), every cell of weak_grad.w that weak_grad reads,
                                                           weak_grad.metric has stored by the time it is read
decided  isl    hexahedra.py:369  weak_grad              no read of weak_grad.w sees its slot overwritten after
                                                           contract(weak_grad.w, lambda q, e, k, j, i: i):
                                                           weak_grad.w kept in n cells, w[q, e, k, j, i] in slot i
decided  isl    hexahedra.py:369  weak_grad              for contract(weak_grad.v), every cell of weak_grad.v that weak_grad reads,
                                                           weak_grad.along_r has stored by the time it is read
decided  isl    hexahedra.py:369  weak_grad              no read of weak_grad.v sees its slot overwritten after
                                                           contract(weak_grad.v, lambda q, e, k, j, i: (j, i)):
                                                           weak_grad.v kept in n*n cells, v[q, e, k, j, i] in slot (j, i)
tested   loopy  hexahedra.py:369  weak_grad              the scheduled run of weak_grad agrees with the native run to the accuracy its types state
tested   loopy  hexahedra.py:180  metric                 the scheduled run of metric agrees with the native run to the accuracy its types state
tested   loopy  hexahedra.py:195  contravariant          the scheduled run of contravariant agrees with the native run to the accuracy its types state
tested   loopy  hexahedra.py:212  along_r                the scheduled run of along_r agrees with the native run to the accuracy its types state
tested   loopy  hexahedra.py:229  along_s                the scheduled run of along_s agrees with the native run to the accuracy its types state
tested   loopy  hexahedra.py:246  gradient_along_t       the scheduled run of gradient_along_t agrees with the native run to the accuracy its types state
tested   loopy  hexahedra.py:267  divergence_along_t     the scheduled run of divergence_along_t agrees with the native run to the accuracy its types state
tested   loopy  hexahedra.py:322  weak_div               the scheduled run of weak_div agrees with the native run to the accuracy its types state
tested   loopy  hexahedra.py:372  metric_along_r         the scheduled run of metric_along_r agrees with the native run to the accuracy its types state
tested   loopy  hexahedra.py:380  contravariant_along_r  the scheduled run of contravariant_along_r agrees with the native run to the
                                                           accuracy its types state

19 facts: 8 decided, 11 tested

WITHOUT LEAN: 8 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```


## advection.py

grudge's weak advection right-hand side on hexahedra (grudge PR #418's
`WeakAdvectionOperator` with the upwind flux, as of a-alveyblanc/grudge
4d103bb), written for `n` nodes per axis on a periodic mesh. It reads
grudge's own geometry: the metric terms `J dr_d/dx_x` and `1 / J` at every
node, the normal and the area element `sJ` at every face node. The program
`advection_rhs` folds the velocity in (`reference_velocity`,
`normal_velocity`), computes the upwind flux at every face node (`upwind`),
the weak divergence (`volume`) and the lift (`lift`). The flux reads the
neighbor's value through the face maps, `u[nbr[e, f], outside[f, a, b, 0],
...]`, in bounds by type, and is written as libParanumal writes it, the
average and half the jump, which is grudge's `where(v.n > 0, ...)` with no
branch. The inverse mass is pushed inward, as grudge PR #362 rewrites it:
the volume term is one pass of `D = M^-1 S^T` along each axis and a face's
term the column `M^-1 e` of its end, and the two theorems
`inverse_mass_inward` and `lift_inward` state that rearrangement for every
order. A third, `lift_is_the_gather_transposed`, says that adding each face
node's value into the node it sits on, as `lift_on_faces` lifts with the
GLL quadrature's mass, is the transpose of the face gather the flux reads
through, for any face map and every order; with the field `1`, the lift
keeps the element's total. With Mathlib all three read `proved`.
`benchmarks/advection.py` compares the right-hand side with grudge's
own on a warped periodic mesh, where it agrees to a few units in the 15th
digit.

The lift at a node reads the fluxes of the six faces through it, so the flux
kernel fuses with the volume term and the lift on the element and no
deeper; the deeper fusion is refused with its pair and names the one that
is accepted, and fused so, the fluxes are kept one element at a time. On a
device every split is a set of launches each checked before it runs:
libParanumal's volume and surface kernels, the fluxes first, three launches,
one launch, and one launch from the metric terms, the velocity computed
where it is read, through two arrays computed where they are read.
libParanumal lifts onto the face nodes alone, with a barrier between pairs
of faces; `lift_on_faces` does the same with the GLL quadrature's mass, and
with a face's nodes on the work items it is refused with the edge node two
faces update, and accepted once the updates are atomic.

### python examples/advection.py

```console
$ uv run python examples/advection.py
order 3, 2**3 periodic hexahedra: the right-hand side is grudge's, inverse mass and all, to 1e-13 of the largest value: True (M^-1 M is the identity: True)
the lift onto the face nodes is the transpose of the face gather on these maps: True

refused: fuse(advection_rhs.upwind.b, advection_rhs.volume.i) illegal: flux[e, f, a, b] = upwind_flux(...) (advection.py:311) at e=0, f=1, a=0, b=0 writes flux[0, 1, 0, 0], read by rhs[e, k, j, i] = ... (advection.py:369) at e=0, k=0, j=0, i=0, scheduled earlier (at n=4, ne=8, as hinted); fuse(advection_rhs.upwind.f, advection_rhs.volume.k, shift=(0, 5)) runs every dependence between them forward

accepted: Schedule(advection_rhs, target='c').substitute(advection_rhs.V).substitute(advection_rhs.vn).fuse(advection_rhs.volume.i, advection_rhs.lift.i).fuse(advection_rhs.upwind.e, advection_rhs.volume.e).contract(advection_rhs.flux, lambda e, f, a, b: (f, a, b))
  flux kept in lambda e, f, a, b: (f, a, b)
  rhs: difference 0 within 1.01e-06 (approx) -> tested

libParanumal's lift onto the face nodes, a face's nodes on the work items:
  refused: tag({lift_on_faces.e: 'g.0', lift_on_faces.a: 'l.1', lift_on_faces.b: 'l.0'}) illegal: rhs[e, last, a, b] -= ... (advection.py:398) at e=0, a=1, b=0 and rhs[e, a, b, 0] -= ... (advection.py:400) at e=0, a=3, b=1 both update rhs[0, 3, 1, 0] (at n=4, ne=8, as hinted), and the loop lift_on_faces.b on l.0 runs them on two work items, so one can overwrite what the other added; make the updates atomic, with atomic(lift_on_faces.rhs) before this step
  accepted: atomic(lift_on_faces.rhs).tag({lift_on_faces.e: 'g.0', lift_on_faces.a: 'l.1', lift_on_faces.b: 'l.0'}): 3 facts, all decided

on a device, each split's launches, every node on a work item:
  volume, surface: volume, 1 step: 2 facts, all decided
  volume, surface: surface, 2 steps: 5 facts, all decided
  fluxes, volume: upwind, 1 step: 2 facts, all decided
  fluxes, volume: volume_and_lift, 2 steps: 4 facts, all decided
  three: upwind, 1 step: 2 facts, all decided
  three: volume, 1 step: 2 facts, all decided
  three: lift, 1 step: 2 facts, all decided
  one: folded_rhs, 3 steps: 7 facts, all decided
  one, from the metric: advection_rhs, 5 steps: 13 facts, all decided
```

### lanky check examples/advection.py

```console
$ uv run lanky check examples/advection.py
STATUS   BY             WHERE             OWNER                          STATEMENT
-------  -------------  ----------------  -----------------------------  ----------------------------------------------------------------------------------------------------
tested   property-test  advection.py:127  inverse_mass_inward            n : Nat, Mi : Fn[Fin(n), Fin(n), Real], S : Fn[Fin(n), Fin(n), Real], M : Fn[Fin(n), Fin(n), Real],
                                                                           w : Fn[Fin(n), Fin(n), Fin(n), Real], k : Fin(n), j : Fin(n), i : Fin(n)
                                                                         |- sum(Mi[k, c]*Mi[j, b]*Mi[i, a]*sum(M[c, t]*M[b, s]*S[a, r]*w[t, s, r] for t in Fin(n),
                                                                                 s in Fin(n), r in Fin(n)) for c in Fin(n), b in Fin(n), a in Fin(n))
                                                                             == sum(sum(Mi[k, c]*M[c, t] for c in Fin(n))*sum(Mi[j, b]*M[b, s]
                                                                                 for b in Fin(n))*sum(Mi[i, a]*S[a, r] for a in Fin(n))*w[t, s, r] for t in Fin(n),
                                                                               s in Fin(n), r in Fin(n))
tested   python         advection.py:127  inverse_mass_inward            the term computes what the annotations compute
tested   property-test  advection.py:169  lift_inward                    n : Nat, Mi : Fn[Fin(n), Fin(n), Real], M : Fn[Fin(n), Fin(n), Real], end : Fn[Fin(n), Real],
                                                                           g : Fn[Fin(n), Fin(n), Real], k : Fin(n), j : Fin(n), i : Fin(n)
                                                                         |- sum(Mi[k, c]*Mi[j, b]*Mi[i, a]*(end[a]*sum(M[c, t]*M[b, s]*g[t, s] for t in Fin(n), s in Fin(n)))
                                                                               for c in Fin(n), b in Fin(n), a in Fin(n))
                                                                             == sum(Mi[i, a]*end[a]
                                                                               for a in Fin(n))*sum(sum(Mi[k, c]*M[c, t] for c in Fin(n))*sum(Mi[j, b]*M[b, s]
                                                                                 for b in Fin(n))*g[t, s] for t in Fin(n), s in Fin(n))
tested   python         advection.py:169  lift_inward                    the term computes what the annotations compute
tested   property-test  advection.py:204  lift_is_the_gather_transposed  faces : Nat, n : Nat, nodes : Nat, node : Fn[Fin(faces), Fin(n), Fin(n), Fin(nodes)],
                                                                           g : Fn[Fin(faces), Fin(n), Fin(n), Real], u : Fn[Fin(nodes), Real]
                                                                         |- sum(sum(g[f, a, b] for f in Fin(faces), a in Fin(n), b in Fin(n) if node[f, a, b] == x)*u[x]
                                                                               for x in Fin(nodes))
                                                                             == sum(g[f, a, b]*u[node[f, a, b]] for f in Fin(faces), a in Fin(n), b in Fin(n))
tested   python         advection.py:204  lift_is_the_gather_transposed  the term computes what the annotations compute
decided  isl            advection.py:262  reference_velocity             V[d, e, k, j, i] is in bounds for every instance of S0
decided  isl            advection.py:262  reference_velocity             G[d, e, k, j, i] is in bounds for every instance of S0
decided  isl            advection.py:262  reference_velocity             v[0] is in bounds for every instance of S0
decided  isl            advection.py:262  reference_velocity             G[3 + d, e, k, j, i] is in bounds for every instance of S0
decided  isl            advection.py:262  reference_velocity             v[1] is in bounds for every instance of S0
decided  isl            advection.py:262  reference_velocity             G[6 + d, e, k, j, i] is in bounds for every instance of S0
decided  isl            advection.py:262  reference_velocity             v[2] is in bounds for every instance of S0
decided  isl            advection.py:262  reference_velocity             distinct instances of S0 write distinct cells of V
decided  isl            advection.py:246  reference_velocity             the source order runs every dependence forward in time
tested   interpreter    advection.py:246  reference_velocity             the traced term computes what the body computes
decided  isl            advection.py:281  normal_velocity                vn[e, f, a, b] is in bounds for every instance of S0
decided  isl            advection.py:281  normal_velocity                sJ[e, f, a, b] is in bounds for every instance of S0
decided  isl            advection.py:281  normal_velocity                normal[0, e, f, a, b] is in bounds for every instance of S0
decided  isl            advection.py:281  normal_velocity                v[0] is in bounds for every instance of S0
decided  isl            advection.py:281  normal_velocity                normal[1, e, f, a, b] is in bounds for every instance of S0
decided  isl            advection.py:281  normal_velocity                v[1] is in bounds for every instance of S0
decided  isl            advection.py:281  normal_velocity                normal[2, e, f, a, b] is in bounds for every instance of S0
decided  isl            advection.py:281  normal_velocity                v[2] is in bounds for every instance of S0
decided  isl            advection.py:281  normal_velocity                distinct instances of S0 write distinct cells of vn
decided  isl            advection.py:269  normal_velocity                the source order runs every dependence forward in time
tested   interpreter    advection.py:269  normal_velocity                the traced term computes what the body computes
decided  isl            advection.py:311  upwind                         flux[e, f, a, b] is in bounds for every instance of S0
decided  isl            advection.py:311  upwind                         vn[e, f, a, b] is in bounds for every instance of S0
decided  isl            advection.py:311  upwind                         u[e, inside[f, a, b, 0], inside[f, a, b, 1], inside[f, a, b, 2]] is in bounds
                                                                           for every instance of S0
                                                                           (inside[f, a, b, 0] : Fin(n), inside[f, a, b, 1] : Fin(n), inside[f, a, b, 2] : Fin(n), by type)
decided  isl            advection.py:311  upwind                         inside[f, a, b, 0] is in bounds for every instance of S0
decided  isl            advection.py:311  upwind                         inside[f, a, b, 1] is in bounds for every instance of S0
decided  isl            advection.py:311  upwind                         inside[f, a, b, 2] is in bounds for every instance of S0
decided  type           advection.py:311  upwind                         u[nbr[e, f], outside[f, a, b, 0], outside[f, a, b, 1], outside[f, a, b, 2]] is in bounds by type
                                                                           (nbr[e, f] : Fin(ne), outside[f, a, b, 0] : Fin(n), outside[f, a, b, 1] : Fin(n),
                                                                             outside[f, a, b, 2] : Fin(n))
decided  isl            advection.py:311  upwind                         nbr[e, f] is in bounds for every instance of S0
decided  isl            advection.py:311  upwind                         outside[f, a, b, 0] is in bounds for every instance of S0
decided  isl            advection.py:311  upwind                         outside[f, a, b, 1] is in bounds for every instance of S0
decided  isl            advection.py:311  upwind                         outside[f, a, b, 2] is in bounds for every instance of S0
decided  isl            advection.py:311  upwind                         distinct instances of S0 write distinct cells of flux
decided  isl            advection.py:297  upwind                         the source order runs every dependence forward in time
tested   interpreter    advection.py:297  upwind                         the traced term computes what the body computes
decided  isl            advection.py:341  volume                         rhs[e, k, j, i] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         invJ[e, k, j, i] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         D[i, a] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         V[0, e, k, j, a] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         u[e, k, j, a] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         D[j, b] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         V[1, e, k, b, i] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         u[e, k, b, i] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         D[k, c] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         V[2, e, c, j, i] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         u[e, c, j, i] is in bounds for every instance of S0
decided  isl            advection.py:341  volume                         distinct instances of S0 write distinct cells of rhs
decided  isl            advection.py:328  volume                         the source order runs every dependence forward in time
decided  type           advection.py:341  volume                         the accumulation into rhs[e, k, j, i] over a is approx
decided  type           advection.py:341  volume                         the accumulation into rhs[e, k, j, i] over b is approx
decided  type           advection.py:341  volume                         the accumulation into rhs[e, k, j, i] over c is approx
tested   interpreter    advection.py:328  volume                         the traced term computes what the body computes
decided  isl            advection.py:369  lift                           rhs[e, k, j, i] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           invJ[e, k, j, i] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           L[0, k] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           flux[e, 0, j, i] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           L[1, k] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           flux[e, 1, j, i] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           L[0, i] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           flux[e, 2, k, j] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           L[1, i] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           flux[e, 3, k, j] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           L[0, j] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           flux[e, 4, k, i] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           L[1, j] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           flux[e, 5, k, i] is in bounds for every instance of S0
decided  isl            advection.py:369  lift                           distinct instances of S0 write distinct cells of rhs
decided  isl            advection.py:357  lift                           the source order runs every dependence forward in time
tested   interpreter    advection.py:357  lift                           the traced term computes what the body computes
decided  isl            advection.py:397  lift_on_faces                  rhs[e, 0, a, b] is in bounds for every instance of S0
decided  isl            advection.py:397  lift_on_faces                  invJ[e, 0, a, b] is in bounds for every instance of S0
decided  isl            advection.py:397  lift_on_faces                  ends[0] is in bounds for every instance of S0
decided  isl            advection.py:397  lift_on_faces                  flux[e, 0, a, b] is in bounds for every instance of S0
decided  isl            advection.py:398  lift_on_faces                  rhs[e, n - 1, a, b] is in bounds for every instance of S1
decided  isl            advection.py:398  lift_on_faces                  invJ[e, n - 1, a, b] is in bounds for every instance of S1
decided  isl            advection.py:398  lift_on_faces                  ends[1] is in bounds for every instance of S1
decided  isl            advection.py:398  lift_on_faces                  flux[e, 1, a, b] is in bounds for every instance of S1
decided  isl            advection.py:400  lift_on_faces                  rhs[e, a, b, 0] is in bounds for every instance of S2
decided  isl            advection.py:400  lift_on_faces                  invJ[e, a, b, 0] is in bounds for every instance of S2
decided  isl            advection.py:400  lift_on_faces                  ends[0] is in bounds for every instance of S2
decided  isl            advection.py:400  lift_on_faces                  flux[e, 2, a, b] is in bounds for every instance of S2
decided  isl            advection.py:401  lift_on_faces                  rhs[e, a, b, n - 1] is in bounds for every instance of S3
decided  isl            advection.py:401  lift_on_faces                  invJ[e, a, b, n - 1] is in bounds for every instance of S3
decided  isl            advection.py:401  lift_on_faces                  ends[1] is in bounds for every instance of S3
decided  isl            advection.py:401  lift_on_faces                  flux[e, 3, a, b] is in bounds for every instance of S3
decided  isl            advection.py:403  lift_on_faces                  rhs[e, a, 0, b] is in bounds for every instance of S4
decided  isl            advection.py:403  lift_on_faces                  invJ[e, a, 0, b] is in bounds for every instance of S4
decided  isl            advection.py:403  lift_on_faces                  ends[0] is in bounds for every instance of S4
decided  isl            advection.py:403  lift_on_faces                  flux[e, 4, a, b] is in bounds for every instance of S4
decided  isl            advection.py:404  lift_on_faces                  rhs[e, a, n - 1, b] is in bounds for every instance of S5
decided  isl            advection.py:404  lift_on_faces                  invJ[e, a, n - 1, b] is in bounds for every instance of S5
decided  isl            advection.py:404  lift_on_faces                  ends[1] is in bounds for every instance of S5
decided  isl            advection.py:404  lift_on_faces                  flux[e, 5, a, b] is in bounds for every instance of S5
decided  isl            advection.py:397  lift_on_faces                  distinct instances of S0 write distinct cells of rhs
decided  isl            advection.py:398  lift_on_faces                  distinct instances of S1 write distinct cells of rhs
decided  isl            advection.py:400  lift_on_faces                  distinct instances of S2 write distinct cells of rhs
decided  isl            advection.py:401  lift_on_faces                  distinct instances of S3 write distinct cells of rhs
decided  isl            advection.py:403  lift_on_faces                  distinct instances of S4 write distinct cells of rhs
decided  isl            advection.py:404  lift_on_faces                  distinct instances of S5 write distinct cells of rhs
decided  isl            advection.py:380  lift_on_faces                  the source order runs every dependence forward in time
tested   interpreter    advection.py:380  lift_on_faces                  the traced term computes what the body computes
decided  isl            advection.py:417  advection_rhs                  every cell of V that volume reads, reference_velocity stored before it
decided  isl            advection.py:416  advection_rhs                  every cell of vn that upwind reads, normal_velocity stored before it
decided  isl            advection.py:418  advection_rhs                  every cell of flux that lift reads, upwind stored before it
tested   interpreter    advection.py:407  advection_rhs                  the traced term computes what the body computes
decided  isl            advection.py:439  advection_rhs                  for substitute(advection_rhs.V), every cell of advection_rhs.V that advection_rhs reads,
                                                                           advection_rhs.reference_velocity has stored by the time it is read
decided  isl            advection.py:439  advection_rhs                  substitute(advection_rhs.V) renames the instances of advection_rhs one for one
decided  isl            advection.py:439  advection_rhs                  the order after substitute(advection_rhs.V) runs every dependence of advection_rhs forward
decided  isl            advection.py:440  advection_rhs                  for substitute(advection_rhs.vn), every cell of advection_rhs.vn that advection_rhs reads,
                                                                           advection_rhs.normal_velocity has stored by the time it is read
decided  isl            advection.py:440  advection_rhs                  substitute(advection_rhs.vn) renames the instances of advection_rhs one for one
decided  isl            advection.py:440  advection_rhs                  the order after substitute(advection_rhs.vn) runs every dependence of advection_rhs forward
decided  isl            advection.py:441  advection_rhs                  fuse(advection_rhs.volume.i, advection_rhs.lift.i) renames the instances of
                                                                           advection_rhs one for one
decided  isl            advection.py:441  advection_rhs                  the order after fuse(advection_rhs.volume.i, advection_rhs.lift.i) runs every dependence of
                                                                           advection_rhs forward
decided  isl            advection.py:442  advection_rhs                  fuse(advection_rhs.upwind.e, advection_rhs.volume.e) renames the instances of
                                                                           advection_rhs one for one
decided  isl            advection.py:442  advection_rhs                  the order after fuse(advection_rhs.upwind.e, advection_rhs.volume.e) runs every dependence of
                                                                           advection_rhs forward
decided  isl            advection.py:446  advection_rhs                  for contract(advection_rhs.flux), every cell of advection_rhs.flux that advection_rhs reads,
                                                                           advection_rhs.upwind has stored by the time it is read
decided  isl            advection.py:446  advection_rhs                  no read of advection_rhs.flux sees its slot overwritten after
                                                                           contract(advection_rhs.flux, lambda e, f, a, b: (f, a, b)):
                                                                           advection_rhs.flux kept in 6*n*n cells, flux[e, f, a, b] in slot (f, a, b)
decided  isl            advection.py:427  folded_rhs                     every cell of flux that lift reads, upwind stored before it
tested   interpreter    advection.py:421  folded_rhs                     the traced term computes what the body computes
decided  isl            advection.py:454  surface                        every cell of flux that lift reads, upwind stored before it
tested   interpreter    advection.py:449  surface                        the traced term computes what the body computes
tested   interpreter    advection.py:457  volume_and_lift                the traced term computes what the body computes

128 facts: 112 decided, 16 tested

WITHOUT LEAN: inverse_mass_inward, lift_inward, lift_is_the_gather_transposed and 108 more rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
WITHOUT MATHLIB: inverse_mass_inward, lift_inward and lift_is_the_gather_transposed are not proved, since only Lean with Mathlib can state them. To have Lean prove them: python -m lanky.mathlib ~/mathlib (about 7 GB), then export LANKY_LEAN_MATHLIB=~/mathlib.
```

### loopty run examples/advection.py

```console
$ uv run loopty run examples/advection.py
advection_rhs: Schedule(advection_rhs, target='c').substitute(advection_rhs.V).substitute(advection_rhs.vn).fuse(advection_rhs.volume.i, advection_rhs.lift.i).fuse(advection_rhs.upwind.e, advection_rhs.volume.e)
  rhs: difference 0 within 1.01e-06 (approx) -> tested
advection_rhs: Schedule(advection_rhs, target='c').substitute(advection_rhs.V).substitute(advection_rhs.vn).fuse(advection_rhs.volume.i, advection_rhs.lift.i).fuse(advection_rhs.upwind.e, advection_rhs.volume.e).contract(advection_rhs.flux, lambda e, f, a, b: (f, a, b))
  rhs: difference 0 within 1.01e-06 (approx) -> tested
reference_velocity: Schedule(reference_velocity, target='c')
  V: difference 0 within 1.07e-06 (approx) -> tested
normal_velocity: Schedule(normal_velocity, target='c')
  vn: difference 0 within 1.07e-06 (approx) -> tested
upwind: Schedule(upwind, target='c')
  flux: difference 0 within 1e-06 (approx) -> tested
volume: Schedule(volume, target='c')
  rhs: difference 0 within 1e-06 (approx) -> tested
lift: Schedule(lift, target='c')
  rhs: difference 0 within 1.35e-06 (approx) -> tested
lift_on_faces: Schedule(lift_on_faces, target='c')
  rhs: difference 0 within 1.04e-06 (approx) -> tested
folded_rhs: Schedule(folded_rhs, target='c')
  rhs: difference 0 within 1.01e-06 (approx) -> tested
surface: Schedule(surface, target='c')
  rhs: difference 0 within 1.01e-06 (approx) -> tested
volume_and_lift: Schedule(volume_and_lift, target='c')
  rhs: difference 0 within 1.03e-06 (approx) -> tested

STATUS   BY     WHERE             OWNER               STATEMENT
-------  -----  ----------------  ------------------  --------------------------------------------------------------------------------------------------
decided  isl    advection.py:439  advection_rhs       for substitute(advection_rhs.V), every cell of advection_rhs.V that advection_rhs reads,
                                                        advection_rhs.reference_velocity has stored by the time it is read
decided  isl    advection.py:439  advection_rhs       substitute(advection_rhs.V) renames the instances of advection_rhs one for one
decided  isl    advection.py:439  advection_rhs       the order after substitute(advection_rhs.V) runs every dependence of advection_rhs forward
decided  isl    advection.py:440  advection_rhs       for substitute(advection_rhs.vn), every cell of advection_rhs.vn that advection_rhs reads,
                                                        advection_rhs.normal_velocity has stored by the time it is read
decided  isl    advection.py:440  advection_rhs       substitute(advection_rhs.vn) renames the instances of advection_rhs one for one
decided  isl    advection.py:440  advection_rhs       the order after substitute(advection_rhs.vn) runs every dependence of advection_rhs forward
decided  isl    advection.py:441  advection_rhs       fuse(advection_rhs.volume.i, advection_rhs.lift.i) renames the instances of
                                                        advection_rhs one for one
decided  isl    advection.py:441  advection_rhs       the order after fuse(advection_rhs.volume.i, advection_rhs.lift.i) runs every dependence of
                                                        advection_rhs forward
decided  isl    advection.py:442  advection_rhs       fuse(advection_rhs.upwind.e, advection_rhs.volume.e) renames the instances of
                                                        advection_rhs one for one
decided  isl    advection.py:442  advection_rhs       the order after fuse(advection_rhs.upwind.e, advection_rhs.volume.e) runs every dependence of
                                                        advection_rhs forward
tested   loopy  advection.py:442  advection_rhs       the scheduled run of advection_rhs agrees with the native run to the accuracy its types state
decided  isl    advection.py:446  advection_rhs       for contract(advection_rhs.flux), every cell of advection_rhs.flux that advection_rhs reads,
                                                        advection_rhs.upwind has stored by the time it is read
decided  isl    advection.py:446  advection_rhs       no read of advection_rhs.flux sees its slot overwritten after
                                                        contract(advection_rhs.flux, lambda e, f, a, b: (f, a, b)):
                                                        advection_rhs.flux kept in 6*n*n cells, flux[e, f, a, b] in slot (f, a, b)
tested   loopy  advection.py:446  advection_rhs       the scheduled run of advection_rhs agrees with the native run to the accuracy its types state
tested   loopy  advection.py:246  reference_velocity  the scheduled run of reference_velocity agrees with the native run to the accuracy its types state
tested   loopy  advection.py:269  normal_velocity     the scheduled run of normal_velocity agrees with the native run to the accuracy its types state
tested   loopy  advection.py:297  upwind              the scheduled run of upwind agrees with the native run to the accuracy its types state
tested   loopy  advection.py:328  volume              the scheduled run of volume agrees with the native run to the accuracy its types state
tested   loopy  advection.py:357  lift                the scheduled run of lift agrees with the native run to the accuracy its types state
tested   loopy  advection.py:380  lift_on_faces       the scheduled run of lift_on_faces agrees with the native run to the accuracy its types state
tested   loopy  advection.py:421  folded_rhs          the scheduled run of folded_rhs agrees with the native run to the accuracy its types state
tested   loopy  advection.py:449  surface             the scheduled run of surface agrees with the native run to the accuracy its types state
tested   loopy  advection.py:457  volume_and_lift     the scheduled run of volume_and_lift agrees with the native run to the accuracy its types state

23 facts: 12 decided, 11 tested

WITHOUT LEAN: 12 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
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

the term of matvec, with dot inlined at calls.py:100:
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

refused: matvec_by calls dot at calls.py:212: axis 0 of its b is n long there, and z, which it is given, has p; nothing says the two agree

schedule: Schedule(matvec, target='c').split(matvec.i, 4)
  y: difference 0 within 1.01e-06 (approx) -> tested
```

### lanky check examples/calls.py

```console
$ uv run lanky check examples/calls.py
STATUS   EFFECTIVE  BY           WHERE         OWNER          STATEMENT
-------  ---------  -----------  ------------  -------------  ----------------------------------------------------------------------------------------------------
decided  decided    isl          calls.py:93   dot            a[k] is in bounds for every instance of S0
decided  decided    isl          calls.py:93   dot            b[k] is in bounds for every instance of S0
decided  decided    isl          calls.py:90   dot            the source order runs every dependence forward in time
decided  decided    type         calls.py:93   dot            the accumulation into the value dot returns over k is approx
tested   tested     interpreter  calls.py:90   dot            the traced term computes what the body computes
decided  decided    isl          calls.py:100  matvec         y[i] is in bounds for every instance of S0
decided  decided    isl          calls.py:100  matvec         A[i, k] is in bounds for every instance of S0
decided  decided    isl          calls.py:100  matvec         x[k] is in bounds for every instance of S0
decided  decided    isl          calls.py:100  matvec         distinct instances of S0 write distinct cells of y
decided  decided    isl          calls.py:96   matvec         the source order runs every dependence forward in time
decided  decided    type         calls.py:100  matvec         the accumulation into y[i] over k is approx
decided  decided    isl          calls.py:100  matvec         A[i], the row passed to dot as a, is in bounds wherever dot is called at calls.py:100
tested   tested     interpreter  calls.py:96   matvec         the traced term computes what the body computes
decided  decided    isl          calls.py:146  matvec         split(matvec.i, 4) renames the instances of matvec one for one
decided  decided    isl          calls.py:146  matvec         the order after split(matvec.i, 4) runs every dependence of matvec forward
decided  decided    isl          calls.py:113  csr            y[r] is in bounds for every instance of S0
decided  decided    isl          calls.py:113  csr            val[r, k] is in bounds for every instance of S0
decided  decided    type         calls.py:113  csr            x[col[r, k]] is in bounds by type (col[r, k] : Fin(m))
decided  decided    isl          calls.py:113  csr            col[r, k] is in bounds for every instance of S0
decided  decided    isl          calls.py:113  csr            cnt[r], the length of row r that bounds the loop over k, is in bounds for every instance of S0
decided  decided    isl          calls.py:113  csr            distinct instances of S0 write distinct cells of y
decided  decided    isl          calls.py:103  csr            the source order runs every dependence forward in time
decided  decided    type         calls.py:113  csr            the accumulation into y[r] over k is approx
decided  decided    isl          calls.py:113  csr            col[r], the indices of the cells passed as b, is in bounds wherever dot is called at calls.py:113
decided  decided    isl          calls.py:113  csr            val[r], the row passed to dot as a, is in bounds wherever dot is called at calls.py:113
decided  decided    type         calls.py:113  csr            the cells x[col[r]] picks, passed as b, are in bounds by type (the entries of col : Fin(m))
tested   tested     interpreter  calls.py:103  csr            the traced term computes what the body computes
decided  decided    isl          calls.py:119  integrate      W[q] is in bounds for every instance of S0
decided  decided    isl          calls.py:116  integrate      the source order runs every dependence forward in time
decided  decided    type         calls.py:119  integrate      the accumulation into the value integrate returns over q is approx
tested   tested     interpreter  calls.py:116  integrate      the traced term computes what the body computes
decided  decided    isl          calls.py:130  element_mass   Me[j, k] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  element_mass   X[1] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  element_mass   X[0] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  element_mass   Y[2] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  element_mass   Y[0] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  element_mass   X[2] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  element_mass   Y[1] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  element_mass   W[q] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  element_mass   PHI[q, j] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  element_mass   PHI[q, k] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  element_mass   distinct instances of S0 write distinct cells of Me
decided  decided    isl          calls.py:122  element_mass   the source order runs every dependence forward in time
decided  decided    type         calls.py:130  element_mass   the accumulation into Me[j, k] over q is approx
tested   tested     native       calls.py:122  element_mass   Me[0, 0] == Me[0, 0] and Me[0, 1] == Me[1, 0] and Me[0, 2] == Me[2, 0] and Me[1, 0] == Me[0, 1]
                                                                and Me[1, 1] == Me[1, 1] and Me[1, 2] == Me[2, 1] and Me[2, 0] == Me[0, 2]
                                                                and Me[2, 1] == Me[1, 2] and Me[2, 2] == Me[2, 2]
tested   tested     interpreter  calls.py:122  element_mass   the traced term computes what the body computes
decided  decided    isl          calls.py:130  mass_matrices  M[c, j, k] is in bounds for every instance of S0
decided  decided    type         calls.py:130  mass_matrices  X[cell[c, 1]] is in bounds by type (cell[c, 1] : Fin(nv))
decided  decided    isl          calls.py:130  mass_matrices  cell[c, 1] is in bounds for every instance of S0
decided  decided    type         calls.py:130  mass_matrices  X[cell[c, 0]] is in bounds by type (cell[c, 0] : Fin(nv))
decided  decided    isl          calls.py:130  mass_matrices  cell[c, 0] is in bounds for every instance of S0
decided  decided    type         calls.py:130  mass_matrices  Y[cell[c, 2]] is in bounds by type (cell[c, 2] : Fin(nv))
decided  decided    isl          calls.py:130  mass_matrices  cell[c, 2] is in bounds for every instance of S0
decided  decided    type         calls.py:130  mass_matrices  Y[cell[c, 0]] is in bounds by type (cell[c, 0] : Fin(nv))
decided  decided    type         calls.py:130  mass_matrices  X[cell[c, 2]] is in bounds by type (cell[c, 2] : Fin(nv))
decided  decided    type         calls.py:130  mass_matrices  Y[cell[c, 1]] is in bounds by type (cell[c, 1] : Fin(nv))
decided  decided    isl          calls.py:130  mass_matrices  W[q] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  mass_matrices  PHI[q, j] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  mass_matrices  PHI[q, k] is in bounds for every instance of S0
decided  decided    isl          calls.py:130  mass_matrices  distinct instances of S0 write distinct cells of M
decided  decided    isl          calls.py:133  mass_matrices  the source order runs every dependence forward in time
decided  decided    type         calls.py:130  mass_matrices  the accumulation into M[c, j, k] over q is approx
decided  decided    isl          calls.py:142  mass_matrices  cell[c], the indices of the cells passed as X
                                                                and Y, is in bounds wherever element_mass is called at calls.py:142
decided  decided    type         calls.py:142  mass_matrices  the cells X[cell[c]] picks, passed as X, are in bounds by type (the entries of cell : Fin(nv))
decided  decided    type         calls.py:142  mass_matrices  the cells Y[cell[c]] picks, passed as Y, are in bounds by type (the entries of cell : Fin(nv))
decided  decided    isl          calls.py:142  mass_matrices  M[c], the row passed to element_mass as Me,
                                                                is in bounds wherever element_mass is called at calls.py:142
decided  tested     call         calls.py:142  mass_matrices  after each call element_mass(X[cell[c]], Y[cell[c]], M[c]) at calls.py:142: M[c, 0, 0] == M[c, 0, 0]
                                                                and M[c, 0, 1] == M[c, 1, 0] and M[c, 0, 2] == M[c, 2, 0] and M[c, 1, 0] == M[c, 0, 1]
                                                                and M[c, 1, 1] == M[c, 1, 1] and M[c, 1, 2] == M[c, 2, 1] and M[c, 2, 0] == M[c, 0, 2]
                                                                and M[c, 2, 1] == M[c, 1, 2] and M[c, 2, 2] == M[c, 2, 2]
tested   tested     interpreter  calls.py:133  mass_matrices  the traced term computes what the body computes

68 facts: 61 decided, 7 tested

WITHOUT LEAN: 44 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

### loopty run examples/calls.py

`integrate` takes a function, so it is compiled where a kernel calls it; a
rule on its own writes its value into a one-cell array, `result`.

```console
$ uv run loopty run examples/calls.py
integrate: takes the function f, so it is compiled where a kernel calls it, inlined
matvec: Schedule(matvec, target='c').split(matvec.i, 4)
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
-------  -----  ------------  -------------  ---------------------------------------------------------------------------------------------
decided  isl    calls.py:146  matvec         split(matvec.i, 4) renames the instances of matvec one for one
decided  isl    calls.py:146  matvec         the order after split(matvec.i, 4) runs every dependence of matvec forward
tested   loopy  calls.py:146  matvec         the scheduled run of matvec agrees with the native run to the accuracy its types state
tested   loopy  calls.py:90   dot            the scheduled run of dot agrees with the native run to the accuracy its types state
tested   loopy  calls.py:103  csr            the scheduled run of csr agrees with the native run to the accuracy its types state
tested   loopy  calls.py:122  element_mass   the scheduled run of element_mass agrees with the native run to the accuracy its types state
tested   loopy  calls.py:133  mass_matrices  the scheduled run of mass_matrices agrees with the native run to the accuracy its types state

7 facts: 2 decided, 5 tested

WITHOUT LEAN: 2 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

## reshape_layouts.py and p2p.py

Both run the same three ways and are documented in their own module docstrings.
`reshape_layouts.py` prints the two layout maps as isl objects and a small
ledger of what isl decided about them; `p2p.py` prints the near-field potential
of sixteen points in a four by four grid of boxes and checks it against a direct
sum over the same interaction lists.

## assembly.py

firedrake's PyOP2 test of the P1 mass kernel, `mass_ffc`, and the assembly
loop around it: per cell, the kernel called on the cell's corners, the rows of
`coords` the cell map picks, and the cell's matrix applied to `u` and added
into the nodes, `y[cell[c, a]] += ...`. The kernel is a kernel of its own,
inlined where it is called. The file's theorems are the mathematics the loop
rests on: the 3-point rule integrates every quadratic exactly, a cell's mass
matrix is `|J| (1 + delta_jk) / 24`, and the scatter keeps the total on every
mesh. With Mathlib all three are proved; the transcripts below have no Lean,
and read `tested`.

`mass_ffc` promises the element's mass in closed form, `|J| (1 + delta_jk) /
24`, which it computes from FFC's 15-digit tables, so to the last digits
only. Its postcondition is compared as its compiled run is with its native
one, at the approx tolerance, `|a - b| <= 1e-6 (|b| + 1)`, and its row says
so (`..., to the approx tolerance`): `tested` on its runs, and, with Mathlib,
`proved` by Lean over the reals from what the kernel stores, with each table
at the value it holds, resting on `no cell of A is stored twice, or read
before its store` and on the kernel's trace. After each call it is restated
of the cell's `Ae[c]`.

Two cells that share a node add into one cell of `y`. Such a statement is an
update: either order of two updates of one cell gives the same sum, to the
class of the sum, so the kernel owes that class and not distinct writes, and a
schedule may reorder the updates but not run two of one cell at the same time.
The cell an update writes is read from the cell map, which isl cannot follow,
so the refusal of PyOP2's GPU split names two updates that collide on the
example inputs, the node the two triangles share, and `atomic(mass_action.y)` is what
makes the split legal.

### python examples/assembly.py

The cells' matrices on PyOP2's two triangles against the test's
`expected_matrix`, `y` on a square against its closed form, FFC's tables
against the rule they round, PyOP2's split of the cells onto groups and work
items refused and then accepted with atomic updates (on the C target, which
has no work items, it is legal and cannot be built), and Sun et al.'s
schedule, the cells four at a time, compiled and compared.

```console
$ uv run python examples/assembly.py
PyOP2's two triangles: the cells' matrices sum to expected_matrix: True
  and y is that matrix times u: True
a square, 32 cells: y is the closed form: True
W3 and FE0 are the rule to 3.3e-16: True
mass_ffc's postcondition, its matrix |J| (1 + delta_jk) / 24 to the approx tolerance -> tested

Schedule(mass_action).split(mass_action.c, 32).tag({mass_action.c.outer: 'g.0', mass_action.c.inner: 'l.0'}) ->
  IllegalCast: tag({mass_action.c.outer: 'g.0', mass_action.c.inner: 'l.0'}) illegal: y[cell[c, a]] += reduce_sum(...) (assembly.py:270) at c=0, a=2 and at c=1, a=1 both update y[3] on example_inputs() (nc=2, nv=4), where cell[0, 2] = 3 and cell[1, 1] = 3, and the loop mass_action.c.inner on l.0 runs them on two work items, so one can overwrite what the other added; make the updates atomic, with atomic(mass_action.y) before this step

accepted: Schedule(mass_action, target='c').assume(mass_action.nc >= 1).atomic(mass_action.y).split(mass_action.c, 32).tag({mass_action.c.outer: 'g.0', mass_action.c.inner: 'l.0'})
  assumed  -    mass_action.nc >= 1, which mass_action is scheduled assuming
  decided  isl  split(mass_action.c, 32) renames the instances of mass_action one for one
  decided  isl  the order after split(mass_action.c, 32) runs every dependence of mass_action forward
  decided  isl  tag({mass_action.c.outer: 'g.0', mass_action.c.inner: 'l.0'}) renames the instances of mass_action one for one
  decided  isl  the order after tag({mass_action.c.outer: 'g.0', mass_action.c.inner: 'l.0'}) runs every dependence of mass_action forward, within one work item
  decided  isl  atomic updates of mass_action.y run at the same time after tag({mass_action.c.outer: 'g.0', mass_action.c.inner: 'l.0'}), so their sum is reassociated and compared at 'reassoc'
  buildable on c: False: the loops mass_action.c.outer on g.0 and mass_action.c.inner on l.0 are on hardware axes, and the C target has none

schedule: Schedule(mass_action, target='c').assume(mass_action.nc % 4 == 0).atomic(mass_action.y).split(mass_action.c, 4).tag(mass_action.c.inner, 'ilp')
  Ae: difference 0 within 1e-06 (approx) -> tested
  y: difference 0 within 1.01e-06 (approx) -> tested
```

### lanky check examples/assembly.py

```console
$ uv run lanky check examples/assembly.py
STATUS                                         EFFECTIVE  BY             WHERE            OWNER                STATEMENT
---------------------------------------------  ---------  -------------  ---------------  -------------------  ----------------------------------------------------------------------------------------------------
tested                                         tested     property-test  assembly.py:160  exact_to_degree_two  a : Real, b : Real, c : Real, d : Real, e : Real, f : Real
                                                                                                               |- (a + b*(1/6) + c*(1/6) + d*(1/6)*(1/6) + e*(1/6)*(1/6) + f*(1/6)*(1/6) + a + b*(1/6) + c*(2/3) +
                                                                                                                     d*(1/6)*(1/6) + e*(1/6)*(2/3) + f*(2/3)*(2/3) + a + b*(2/3) + c*(1/6) + d*(2/3)*(2/3) +
                                                                                                                     e*(2/3)*(1/6) + f*(1/6)*(1/6)) / 6 == a / 2 + b / 6 + c / 6 + d / 12 + e / 24 + f / 12
tested                                         tested     python         assembly.py:160  exact_to_degree_two  the term computes what the annotations compute
tested                                         tested     property-test  assembly.py:173  element_mass         J : Real, u : Fn[Fin(3), Real]
                                                                                                               |- abs(J)*((2/3*(u[0]*(2/3) + u[1]*(1/6) + u[2]*(1/6)) + 1/6*(u[0]*(1/6) + u[1]*(1/6) + u[2]*(2/3))
                                                                                                                       + 1/6*(u[0]*(1/6) + u[1]*(2/3) + u[2]*(1/6))) / 6)
                                                                                                                   == abs(J)*(u[0] + u[0] + u[1] + u[2]) / 24
                                                                                                                   and abs(J)*((1/6*(u[0]*(2/3) + u[1]*(1/6) + u[2]*(1/6)) +
                                                                                                                       1/6*(u[0]*(1/6) + u[1]*(1/6) + u[2]*(2/3)) + 2/3*(u[0]*(1/6) + u[1]*(2/3) + u[2]*(1/6))) /
                                                                                                                     6) == abs(J)*(u[1] + u[0] + u[1] + u[2]) / 24
                                                                                                                   and abs(J)*((1/6*(u[0]*(2/3) + u[1]*(1/6) + u[2]*(1/6)) +
                                                                                                                       2/3*(u[0]*(1/6) + u[1]*(1/6) + u[2]*(2/3)) + 1/6*(u[0]*(1/6) + u[1]*(2/3) + u[2]*(1/6))) /
                                                                                                                     6) == abs(J)*(u[2] + u[0] + u[1] + u[2]) / 24
tested                                         tested     python         assembly.py:173  element_mass         the term computes what the annotations compute
tested                                         tested     property-test  assembly.py:187  conservation         nc : Nat, nv : Nat, d : Nat, cell : Fn[Fin(nc), Fin(d), Fin(nv)], yl : Fn[Fin(nc), Fin(d), Real],
                                                                                                                 y : Fn[Fin(nv), Real]
                                                                                                               | scatter: forall i in Fin(nv). y[i]
                                                                                                                   == sum(yl[c][k] for c in Fin(nc), k in Fin(d) if cell[c][k] == i)
                                                                                                               |- sum(y[i] for i in Fin(nv)) == sum(yl[c][k] for c in Fin(nc), k in Fin(d))
tested                                         tested     python         assembly.py:187  conservation         the term computes what the annotations compute
decided                                        decided    isl            assembly.py:253  mass_ffc             A[j, k] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_ffc             FE0[ip, j] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_ffc             FE0[ip, k] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_ffc             W3[ip] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_ffc             x[1, 0] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_ffc             x[0, 0] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_ffc             x[2, 1] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_ffc             x[0, 1] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_ffc             x[2, 0] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_ffc             x[1, 1] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_ffc             distinct instances of S0 write distinct cells of A
decided                                        decided    isl            assembly.py:237  mass_ffc             the source order runs every dependence forward in time
decided                                        decided    type           assembly.py:253  mass_ffc             the accumulation into A[j, k] over ip is approx
decided                                        decided    isl            assembly.py:253  mass_ffc             no cell of A is stored twice, or read before its store
tested                                         tested     native         assembly.py:237  mass_ffc             (A[0, 0] == abs((x[1, 0] - x[0, 0])*(x[2, 1] - x[0, 1]) - (x[2, 0] - x[0, 0])*(x[1, 1] - x[0, 1])) /
                                                                                                                   12 and A[1, 1] ==
                                                                                                                   abs((x[1, 0] - x[0, 0])*(x[2, 1] - x[0, 1]) - (x[2, 0] - x[0, 0])*(x[1, 1] - x[0, 1])) / 12
                                                                                                                   and A[2, 2] == abs((x[1, 0] - x[0, 0])*(x[2, 1] - x[0, 1]) -
                                                                                                                     (x[2, 0] - x[0, 0])*(x[1, 1] - x[0, 1])) / 12)
                                                                                                                 and (A[0, 1] == abs((x[1, 0] - x[0, 0])*(x[2, 1] - x[0, 1]) -
                                                                                                                     (x[2, 0] - x[0, 0])*(x[1, 1] - x[0, 1])) / 24
                                                                                                                   and A[0, 2] == abs((x[1, 0] - x[0, 0])*(x[2, 1] - x[0, 1]) -
                                                                                                                     (x[2, 0] - x[0, 0])*(x[1, 1] - x[0, 1])) / 24
                                                                                                                   and A[1, 0] == abs((x[1, 0] - x[0, 0])*(x[2, 1] - x[0, 1]) -
                                                                                                                     (x[2, 0] - x[0, 0])*(x[1, 1] - x[0, 1])) / 24
                                                                                                                   and A[1, 2] == abs((x[1, 0] - x[0, 0])*(x[2, 1] - x[0, 1]) -
                                                                                                                     (x[2, 0] - x[0, 0])*(x[1, 1] - x[0, 1])) / 24
                                                                                                                   and A[2, 0] == abs((x[1, 0] - x[0, 0])*(x[2, 1] - x[0, 1]) -
                                                                                                                     (x[2, 0] - x[0, 0])*(x[1, 1] - x[0, 1])) / 24
                                                                                                                   and A[2, 1] == abs((x[1, 0] - x[0, 0])*(x[2, 1] - x[0, 1]) -
                                                                                                                     (x[2, 0] - x[0, 0])*(x[1, 1] - x[0, 1])) / 24), to the approx tolerance
tested                                         tested     interpreter    assembly.py:237  mass_ffc             the traced term computes what the body computes
decided                                        decided    isl            assembly.py:253  mass_action          Ae[c, j, k] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_action          FE0[ip, j] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_action          FE0[ip, k] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_action          W3[ip] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_action          coords[cell[c, 1], 0] is in bounds for every instance of S0 (cell[c, 1] : Fin(nv), by type)
decided                                        decided    isl            assembly.py:253  mass_action          cell[c, 1] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_action          coords[cell[c, 0], 0] is in bounds for every instance of S0 (cell[c, 0] : Fin(nv), by type)
decided                                        decided    isl            assembly.py:253  mass_action          cell[c, 0] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_action          coords[cell[c, 2], 1] is in bounds for every instance of S0 (cell[c, 2] : Fin(nv), by type)
decided                                        decided    isl            assembly.py:253  mass_action          cell[c, 2] is in bounds for every instance of S0
decided                                        decided    isl            assembly.py:253  mass_action          coords[cell[c, 0], 1] is in bounds for every instance of S0 (cell[c, 0] : Fin(nv), by type)
decided                                        decided    isl            assembly.py:253  mass_action          coords[cell[c, 2], 0] is in bounds for every instance of S0 (cell[c, 2] : Fin(nv), by type)
decided                                        decided    isl            assembly.py:253  mass_action          coords[cell[c, 1], 1] is in bounds for every instance of S0 (cell[c, 1] : Fin(nv), by type)
decided                                        decided    type           assembly.py:270  mass_action          y[cell[c, a]] is in bounds by type (cell[c, a] : Fin(nv))
decided                                        decided    isl            assembly.py:270  mass_action          cell[c, a] is in bounds for every instance of S1
decided                                        decided    isl            assembly.py:270  mass_action          Ae[c, a, b] is in bounds for every instance of S1
decided                                        decided    type           assembly.py:270  mass_action          u[cell[c, b]] is in bounds by type (cell[c, b] : Fin(nv))
decided                                        decided    isl            assembly.py:270  mass_action          cell[c, b] is in bounds for every instance of S1
decided                                        decided    isl            assembly.py:253  mass_action          distinct instances of S0 write distinct cells of Ae
decided                                        decided    isl            assembly.py:270  mass_action          every cell of Ae that S1 reads, S0 wrote before the read
decided                                        decided    isl            assembly.py:258  mass_action          the source order runs every dependence forward in time
decided                                        decided    type           assembly.py:253  mass_action          the accumulation into Ae[c, j, k] over ip is approx
decided                                        decided    type           assembly.py:270  mass_action          the accumulation into y[cell[c, a]] over c, a is approx
decided                                        decided    type           assembly.py:270  mass_action          the accumulation into y[cell[c, a]] over b is approx
decided                                        decided    isl            assembly.py:268  mass_action          cell[c], the indices of the cells passed as x,
                                                                                                                 is in bounds wherever mass_ffc is called at assembly.py:268
decided                                        decided    type           assembly.py:268  mass_action          the cells coords[cell[c]] picks, passed as x, are in bounds by type (the entries of cell : Fin(nv))
decided                                        decided    isl            assembly.py:268  mass_action          Ae[c], the row passed to mass_ffc as A, is in bounds wherever mass_ffc is called at assembly.py:268
decided                                        tested     call           assembly.py:268  mass_action          after each call mass_ffc(coords[cell[c]], Ae[c]) at assembly.py:268:
                                                                                                                 (Ae[c, 0, 0] == abs((coords[cell[c, 1], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 2], 1] -
                                                                                                                       coords[cell[c, 0], 1]) -
                                                                                                                     (coords[cell[c, 2], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 1], 1] -
                                                                                                                       coords[cell[c, 0], 1])) / 12 and Ae[c, 1, 1]
                                                                                                                   == abs((coords[cell[c, 1], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 2], 1] -
                                                                                                                       coords[cell[c, 0], 1]) -
                                                                                                                     (coords[cell[c, 2], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 1], 1] -
                                                                                                                       coords[cell[c, 0], 1])) / 12 and Ae[c, 2, 2]
                                                                                                                   == abs((coords[cell[c, 1], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 2], 1] -
                                                                                                                       coords[cell[c, 0], 1]) -
                                                                                                                     (coords[cell[c, 2], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 1], 1] -
                                                                                                                       coords[cell[c, 0], 1])) / 12) and
                                                                                                                 (Ae[c, 0, 1] == abs((coords[cell[c, 1], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 2], 1] -
                                                                                                                       coords[cell[c, 0], 1]) -
                                                                                                                     (coords[cell[c, 2], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 1], 1] -
                                                                                                                       coords[cell[c, 0], 1])) / 24 and Ae[c, 0, 2]
                                                                                                                   == abs((coords[cell[c, 1], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 2], 1] -
                                                                                                                       coords[cell[c, 0], 1]) -
                                                                                                                     (coords[cell[c, 2], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 1], 1] -
                                                                                                                       coords[cell[c, 0], 1])) / 24 and Ae[c, 1, 0]
                                                                                                                   == abs((coords[cell[c, 1], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 2], 1] -
                                                                                                                       coords[cell[c, 0], 1]) -
                                                                                                                     (coords[cell[c, 2], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 1], 1] -
                                                                                                                       coords[cell[c, 0], 1])) / 24 and Ae[c, 1, 2]
                                                                                                                   == abs((coords[cell[c, 1], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 2], 1] -
                                                                                                                       coords[cell[c, 0], 1]) -
                                                                                                                     (coords[cell[c, 2], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 1], 1] -
                                                                                                                       coords[cell[c, 0], 1])) / 24 and Ae[c, 2, 0]
                                                                                                                   == abs((coords[cell[c, 1], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 2], 1] -
                                                                                                                       coords[cell[c, 0], 1]) -
                                                                                                                     (coords[cell[c, 2], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 1], 1] -
                                                                                                                       coords[cell[c, 0], 1])) / 24 and Ae[c, 2, 1]
                                                                                                                   == abs((coords[cell[c, 1], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 2], 1] -
                                                                                                                       coords[cell[c, 0], 1]) -
                                                                                                                     (coords[cell[c, 2], 0] - coords[cell[c, 0], 0])*(coords[cell[c, 1], 1] -
                                                                                                                       coords[cell[c, 0], 1])) / 24), to the approx tolerance
tested                                         tested     interpreter    assembly.py:258  mass_action          the traced term computes what the body computes
assumed                                        assumed    -              assembly.py:388  mass_action          mass_action.nc % 4 == 0, which mass_action is scheduled assuming
decided under assume(mass_action.nc % 4 == 0)  assumed    isl            assembly.py:390  mass_action          split(mass_action.c, 4) renames the instances of mass_action one for one
decided under assume(mass_action.nc % 4 == 0)  assumed    isl            assembly.py:390  mass_action          the order after split(mass_action.c, 4) runs every dependence of mass_action forward
decided under assume(mass_action.nc % 4 == 0)  assumed    isl            assembly.py:391  mass_action          tag(mass_action.c.inner, 'ilp') renames the instances of mass_action one for one
decided under assume(mass_action.nc % 4 == 0)  assumed    isl            assembly.py:391  mass_action          the order after tag(mass_action.c.inner, 'ilp') runs every dependence of mass_action forward
decided                                        decided    isl            assembly.py:391  mass_action          atomic updates of mass_action.y run at the same time after tag(mass_action.c.inner, 'ilp'),
                                                                                                                 so their sum is reassociated and compared at 'reassoc'

57 facts: 1 assumed, 47 decided, 9 tested

WITHOUT LEAN: exact_to_degree_two, element_mass, conservation and 39 more rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
WITHOUT MATHLIB: exact_to_degree_two, element_mass, conservation and 1 more row are not proved, since only Lean with Mathlib can state them. To have Lean prove them: python -m lanky.mathlib ~/mathlib (about 7 GB), then export LANKY_LEAN_MATHLIB=~/mathlib.
```

### loopty run examples/assembly.py

```console
$ uv run loopty run examples/assembly.py
mass_action: Schedule(mass_action, target='c').assume(mass_action.nc % 4 == 0).atomic(mass_action.y).split(mass_action.c, 4).tag(mass_action.c.inner, 'ilp')
  Ae: difference 0 within 1e-06 (approx) -> tested
  y: difference 0 within 1.01e-06 (approx) -> tested
mass_ffc: Schedule(mass_ffc, target='c')
  A: difference 0 within 1.13e-06 (approx) -> tested

STATUS                                         EFFECTIVE  BY     WHERE            OWNER        STATEMENT
---------------------------------------------  ---------  -----  ---------------  -----------  --------------------------------------------------------------------------------------------
assumed                                        assumed    -      assembly.py:388  mass_action  mass_action.nc % 4 == 0, which mass_action is scheduled assuming
decided under assume(mass_action.nc % 4 == 0)  assumed    isl    assembly.py:390  mass_action  split(mass_action.c, 4) renames the instances of mass_action one for one
decided under assume(mass_action.nc % 4 == 0)  assumed    isl    assembly.py:390  mass_action  the order after split(mass_action.c, 4) runs every dependence of mass_action forward
decided under assume(mass_action.nc % 4 == 0)  assumed    isl    assembly.py:391  mass_action  tag(mass_action.c.inner, 'ilp') renames the instances of mass_action one for one
decided under assume(mass_action.nc % 4 == 0)  assumed    isl    assembly.py:391  mass_action  the order after tag(mass_action.c.inner, 'ilp') runs every dependence of mass_action forward
decided                                        decided    isl    assembly.py:391  mass_action  atomic updates of mass_action.y run at the same time after tag(mass_action.c.inner, 'ilp'),
                                                                                                 so their sum is reassociated and compared at 'reassoc'
tested                                         tested     loopy  assembly.py:391  mass_action  the scheduled run of mass_action agrees with the native run to the accuracy its types state
tested                                         tested     loopy  assembly.py:237  mass_ffc     the scheduled run of mass_ffc agrees with the native run to the accuracy its types state

8 facts: 1 assumed, 5 decided, 2 tested

WITHOUT LEAN: 4 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

## l2p.py

The demo's flagship: sumpy's L2P for the 2-D Laplace kernel, the evaluation of
each leaf box's local expansion at the targets the box owns, in one file and
one ledger, from the theorem about the expansion to the kernels that compute
it to the schedule that runs them. The file is read top to bottom, and its
docstring has the long form and the permalinks into sumpy at the commit it
mirrors.

- `compressed_taylor` is the claim sumpy's compressed Taylor expansion rests
  on (lanky#86): a table that agrees with the derivatives where sumpy stores
  them, `a < 2`, and follows `R[a + 2, b] == -R[a, b + 2]` everywhere else is
  the table of derivatives, for any table that satisfies Laplace's equation.
  `on_every_triangle` is its form box by box on the triangle `a + b <= p`,
  which a kernel of order `p` cites. Lean proves both for every order, with
  Mathlib; without it, the rows are `assumed`.
- `storage_index` is sumpy's `get_storage_index`, a layout of two pieces whose
  three facts say it stores the `2p + 1` derivatives one for one onto their
  addresses. `unpack` reads each box's coefficients through it into the
  table the theorem's `stored` is about, and its postcondition, decided from
  what it stores, is sumpy's storage order: `C[i, a, b] == coeffs[i, 2*(a +
  b) - 1 + a]` after the value at the centre.
- `reconstruct` fills each box's triangle. Its postcondition, the copies and
  the recurrence, is decided by isl from what its statements store, and it
  is the theorem's hypotheses about the kernel, so the theorem applies: the
  row `by on_every_triangle, for every D with pde and stored` says the
  triangle is the table of derivatives of any harmonic `D` whose entries
  the stored table holds. `evaluate` sums the Taylor polynomial at each
  target, and the program `l2p` runs the three, the stored table and the
  triangles arrays it makes, each read only where the call before it stored.
- Running `b` outside `a` is refused with the pair of instances it would
  reorder, each named by the statement it is an instance of, as the file
  writes it, and its line. `fused` runs the three calls box by box, as sumpy
  does, one fusion of the three calls' loops over boxes,
  `fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box)`, whose two
  facts isl decides and Lean proves. `contracted` then keeps only
  the box at hand's triangle and table, as sumpy keeps them in registers:
  `contract` finds the window `lambda box, a, b: (a, b)` along the fused
  order, the 11 x 11 box the triangle spans, and its two facts, that every
  read is of a cell the copy or the recurrence stored and that no read sees
  its slot overwritten, are decided by isl and proved by Lean. One row of
  the triangle, `lambda box, a, b: b`, is refused with the read of the
  recurrence it would break.
- `on_a_gpu` is sumpy's own schedule for a GPU, each box a work group:
  `fused.retarget("opencl").on_groups(l2p.unpack.box)`, with no loop the file did not
  write. The table and the triangles the program makes are zeroed on the
  work item of the box that stores them, and the tag's facts say that no
  dependence joins two work items. `loopty run` runs it on a device where
  pyopencl is installed, and says it ran none where it is not.
- `on_a_device()` is the same with each box's windows, `contracted` on
  `g.0`: each work item keeps its own windows, as loopy keeps a private
  temporary, so the boxes need no order. isl decides it in the python run;
  Lean does not state a cast on such windows yet (#295), so it is no row of
  the ledger.
- `sumpy_l2p` is sumpy's own `E2PFromSingleBox`, on this file's inputs: what
  it computed is recorded beside the file (`l2p_sumpy.json`, written by
  `benchmarks/l2p.py` in an environment with sumpy main and loopy main), and
  the `upstream` row compares the native run of `l2p` with it. They agree to
  round-off, and the row says how close: within `1e-14` of the largest value.

### python examples/l2p.py

```console
$ uv run python examples/l2p.py
order 10: 16 boxes, 97 targets, 21 coefficients stored per box, 66 on its triangle
largest error against the charges' potential: 4.1e-13

refused: interchange(reconstruct.b, reconstruct.a) illegal: R[box, a, b] = C[box, a, b] (l2p.py:276) at box=0, a=1, b=9 writes R[0, 1, 9], read by R[box, a, b] = -R[box, a - 2, b + 2] (l2p.py:278) at box=0, a=3, b=7, scheduled earlier (at nb=1, isl's choice)

refused: contract(l2p.R, lambda box, a, b: b) illegal: R[box, a, b] = -R[box, a - 2, b + 2] (l2p.py:278) at box=0, a=9, b=0 reads R[0, 7, 2], which the same statement at box=0, a=7, b=2 stored in slot 2, after the same statement at box=0, a=8, b=2 stored R[0, 8, 2] there (at nb=1, isl's choice); contract(l2p.R) finds the smallest window that keeps every read, lambda box, a, b: (a, b)

schedule: Schedule(l2p, target='c').fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box).contract(l2p.R, lambda box, a, b: (a, b)).contract(l2p.C, lambda box, a, b: (a, b))
  decided  isl  fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box) renames the instances of l2p one for one
  decided  isl  the order after fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box) runs every dependence of l2p forward
  decided  isl  for contract(l2p.R), every cell of l2p.R that l2p reads, l2p.reconstruct has stored by the time it is read
  decided  isl  no read of l2p.R sees its slot overwritten after contract(l2p.R, lambda box, a, b: (a, b)): l2p.R kept in 121 cells, R[box, a, b] in slot (a, b)
  decided  isl  for contract(l2p.C), every cell of l2p.C that l2p reads, l2p.unpack has stored by the time it is read
  decided  isl  no read of l2p.C sees its slot overwritten after contract(l2p.C, lambda box, a, b: (a, b)): l2p.C kept in 22 cells, C[box, a, b] in slot (a, b)
  pot: difference 0 within 1.1e-06 (approx) -> tested

on a device, each box a work group keeping its windows: on_groups(l2p.unpack.box): 2 facts, all decided by isl

on a GPU, each box a work group, as sumpy's: Schedule(l2p, target='opencl').fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box).on_groups(l2p.unpack.box)
  decided  isl  on_groups(l2p.unpack.box) renames the instances of l2p one for one
  decided  isl  the order after on_groups(l2p.unpack.box) runs every dependence of l2p forward, within one work item
```

### lanky check examples/l2p.py

Without Lean, the two theorems are `assumed` and the application is
`decided` by `theorem`, worth what the theorem is. With Lean and Mathlib
(`LANKY_LEAN_MATHLIB`, and lanky's `cas` extra, whose sympy finds the
induction's steps), every row isl decided is `proved` by Lean, and so are
both theorems and the application: `56 facts: 3 decided, 46 proved, 7
tested`, the three left decided being the class of the Taylor sum, by type,
and the two postconditions restated after their calls, by the call. That
takes Lean about two minutes from an empty proof cache. `lanky check
examples/l2p.py --show-lean compressed_taylor` prints the induction Lean
checked: on `a`, two at a time, each step a `linear_combination` of the
recurrence, the PDE and the hypothesis at `(a, b + 2)`, whose multipliers
Python found. Each statement is printed whole, the theorems a part to a
line, with each hypothesis under its name. Under the table, the `WITHOUT`
lines say what this machine lacks for Lean to prove the rows it did not,
and the command that installs it.

```console
$ uv run lanky check examples/l2p.py
STATUS                           EFFECTIVE  BY           WHERE       OWNER              STATEMENT
-------------------------------  ---------  -----------  ----------  -----------------  ----------------------------------------------------------------------------------------------------
assumed                          assumed    -            l2p.py:130  compressed_taylor  D : Fn[Nat, Nat, Real], R : Fn[Nat, Nat, Real]
                                                                                        | pde: forall a in Nat, b in Nat. D[a + 2, b] + D[a, b + 2] == 0
                                                                                        | stored: forall a in Nat, b in Nat where a < 2. R[a, b] == D[a, b]
                                                                                        | recurrence: forall a in Nat, b in Nat. R[a + 2, b] == -R[a, b + 2]
                                                                                        |- forall a in Nat, b in Nat. R[a, b] == D[a, b]
tested                           tested     python       l2p.py:130  compressed_taylor  the term computes what the annotations compute
assumed                          assumed    -            l2p.py:148  on_every_triangle  n : Nat, p : Nat, c : Fn[Nat, Nat, Nat, Real], D : Fn[Nat, Nat, Nat, Real],
                                                                                          R : Fn[Nat, Nat, Nat, Real]
                                                                                        | pde: forall i in Fin(n), a in Nat, b in Nat. D[i, a + 2, b] + D[i, a, b + 2] == 0
                                                                                        | stored: forall i in Fin(n), a in Nat, b in Nat where a < 2 and a + b <= p.
                                                                                            c[i, a, b] == D[i, a, b]
                                                                                        | copies: forall i in Fin(n), a in Nat, b in Nat where a < 2 and a + b <= p.
                                                                                            R[i, a, b] == c[i, a, b]
                                                                                        | recurrence: forall i in Fin(n), a in Nat, b in Nat where a + b + 2 <= p.
                                                                                            R[i, a + 2, b] == -R[i, a, b + 2]
                                                                                        |- forall i in Fin(n), a in Nat, b in Nat where a + b <= p. R[i, a, b] == D[i, a, b]
tested                           tested     python       l2p.py:148  on_every_triangle  the term computes what the annotations compute
decided                          decided    isl          l2p.py:199  storage_index      storage_index stores every cell at an address of Fin(21)
decided                          decided    isl          l2p.py:199  storage_index      storage_index stores no two cells at one address
decided                          decided    isl          l2p.py:199  storage_index      storage_index stores a cell at every address of Fin(21)
decided                          decided    isl          l2p.py:244  unpack             C[box, a, b] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:244  unpack             coeffs[box, 0] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:246  unpack             C[box, a, b] is in bounds for every instance of S1
decided                          decided    isl          l2p.py:246  unpack             coeffs[box, 2*(a + b) - 1 + a] is in bounds for every instance of S1
decided                          decided    isl          l2p.py:244  unpack             distinct instances of S0 write distinct cells of C
decided                          decided    isl          l2p.py:246  unpack             distinct instances of S1 write distinct cells of C
decided                          decided    isl          l2p.py:222  unpack             the source order runs every dependence forward in time
decided                          decided    isl          l2p.py:244  unpack             no cell of C is stored twice, or read before its store
decided                          decided    isl          l2p.py:222  unpack             (forall i in Fin(nb). C[i, 0, 0] == coeffs[i, 0])
                                                                                          and (forall i in Fin(nb), a in Nat, b in Nat where a < 2 and a + b >= 1 and a + b <= 10.
                                                                                            C[i, a, b] == coeffs[i, 2*(a + b) - 1 + a])
tested                           tested     interpreter  l2p.py:222  unpack             the traced term computes what the body computes
decided                          decided    isl          l2p.py:276  reconstruct        R[box, a, b] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:276  reconstruct        C[box, a, b] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:278  reconstruct        R[box, a, b] is in bounds for every instance of S1
decided                          decided    isl          l2p.py:278  reconstruct        R[box, a - 2, b + 2] is in bounds for every instance of S1
decided                          decided    isl          l2p.py:276  reconstruct        distinct instances of S0 write distinct cells of R
decided                          decided    isl          l2p.py:278  reconstruct        distinct instances of S1 write distinct cells of R
decided                          decided    isl          l2p.py:278  reconstruct        every cell of R that S1 reads, S0 or S1 wrote before the read
decided                          decided    isl          l2p.py:249  reconstruct        the source order runs every dependence forward in time
decided                          decided    isl          l2p.py:276  reconstruct        no cell of R is stored twice, or read before its store
decided                          decided    isl          l2p.py:249  reconstruct        (forall i in Fin(nb), a in Nat, b in Nat where a < 2 and a + b <= 10. R[i, a, b] == C[i, a, b])
                                                                                          and (forall i in Fin(nb), a in Nat, b in Nat where a + b + 2 <= 10.
                                                                                            R[i, a + 2, b] == -R[i, a, b + 2])
decided under on_every_triangle  assumed    theorem      l2p.py:249  reconstruct        by on_every_triangle, for every D with pde and stored:
                                                                                          forall i in Fin(nb), a in Nat, b in Nat where a + b <= 10. R[i, a, b] == D[i, a, b]
tested                           tested     interpreter  l2p.py:249  reconstruct        the traced term computes what the body computes
decided                          decided    isl          l2p.py:309  evaluate           pot[box, target] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:309  evaluate           R[box, a, b] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:309  evaluate           tx[box, target] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:309  evaluate           cx[box] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:309  evaluate           ty[box, target] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:309  evaluate           cy[box] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:309  evaluate           INV_FACT[a] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:309  evaluate           INV_FACT[b] is in bounds for every instance of S0
decided                          decided    isl          l2p.py:309  evaluate           cnt[box], the length of row box that bounds the loop over target, is in bounds
                                                                                          for every instance of S0
decided                          decided    isl          l2p.py:309  evaluate           distinct instances of S0 write distinct cells of pot
decided                          decided    isl          l2p.py:288  evaluate           the source order runs every dependence forward in time
decided                          decided    type         l2p.py:309  evaluate           the accumulation into pot[box, target] over a, b is approx
tested                           tested     interpreter  l2p.py:288  evaluate           the traced term computes what the body computes
decided                          decided    call         l2p.py:316  l2p                after unpack(...) in l2p: (forall i in Fin(nb). C[i, 0, 0] == coeffs[i, 0])
                                                                                          and (forall i in Fin(nb), a in Nat, b in Nat where a < 2 and a + b >= 1 and a + b <= 10.
                                                                                            C[i, a, b] == coeffs[i, 2*(a + b) - 1 + a])
decided                          decided    call         l2p.py:316  l2p                after reconstruct(...) in l2p:
                                                                                          (forall i in Fin(nb), a in Nat, b in Nat where a < 2 and a + b <= 10. R[i, a, b] == C[i, a, b])
                                                                                          and (forall i in Fin(nb), a in Nat, b in Nat where a + b + 2 <= 10.
                                                                                            R[i, a + 2, b] == -R[i, a, b + 2])
decided                          decided    isl          l2p.py:322  l2p                every cell of C that reconstruct reads, unpack stored before it
decided                          decided    isl          l2p.py:323  l2p                every cell of R that evaluate reads, reconstruct stored before it
tested                           tested     interpreter  l2p.py:316  l2p                the traced term computes what the body computes
decided                          decided    isl          l2p.py:334  l2p                fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box) renames the instances of l2p one for one
decided                          decided    isl          l2p.py:334  l2p                the order after fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box) runs every
                                                                                          dependence of l2p forward
decided                          decided    isl          l2p.py:340  l2p                for contract(l2p.R), every cell of l2p.R that l2p reads,
                                                                                          l2p.reconstruct has stored by the time it is read
decided                          decided    isl          l2p.py:340  l2p                no read of l2p.R sees its slot overwritten after contract(l2p.R, lambda box, a, b: (a, b)):
                                                                                          l2p.R kept in 121 cells, R[box, a, b] in slot (a, b)
decided                          decided    isl          l2p.py:340  l2p                for contract(l2p.C), every cell of l2p.C that l2p reads,
                                                                                          l2p.unpack has stored by the time it is read
decided                          decided    isl          l2p.py:340  l2p                no read of l2p.C sees its slot overwritten after contract(l2p.C, lambda box, a, b: (a, b)):
                                                                                          l2p.C kept in 22 cells, C[box, a, b] in slot (a, b)
decided                          decided    isl          l2p.py:347  l2p                on_groups(l2p.unpack.box) renames the instances of l2p one for one
decided                          decided    isl          l2p.py:347  l2p                the order after on_groups(l2p.unpack.box) runs every dependence of l2p forward, within one work item
tested                           tested     upstream     l2p.py:469  l2p                the native run of l2p agrees with sumpy_l2p in pot within 1e-14 of the largest value,
                                                                                          on its example inputs

56 facts: 2 assumed, 47 decided, 7 tested

WITHOUT LEAN: compressed_taylor, on_every_triangle and 44 more rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
WITHOUT MATHLIB: compressed_taylor, on_every_triangle and 3 more rows are not proved, since only Lean with Mathlib can state them. To have Lean prove them: python -m lanky.mathlib ~/mathlib (about 7 GB), then export LANKY_LEAN_MATHLIB=~/mathlib.
```

### loopty run examples/l2p.py

```console
$ uv run loopty run examples/l2p.py
l2p: Schedule(l2p, target='c').fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box)
  pot: difference 0 within 1.1e-06 (approx) -> tested
l2p: Schedule(l2p, target='c').fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box).contract(l2p.R, lambda box, a, b: (a, b)).contract(l2p.C, lambda box, a, b: (a, b))
  pot: difference 0 within 1.1e-06 (approx) -> tested
l2p: Schedule(l2p, target='opencl').fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box).on_groups(l2p.unpack.box)
  not run here: running on a device needs pyopencl, which is not installed; its facts are checked all the same
unpack: Schedule(unpack, target='c')
  C: difference 0 within 1e-06 (approx) -> tested
reconstruct: Schedule(reconstruct, target='c')
  R: difference 0 within 1e-06 (approx) -> tested
evaluate: Schedule(evaluate, target='c')
  pot: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE       OWNER        STATEMENT
-------  -----  ----------  -----------  ----------------------------------------------------------------------------------------------------
decided  isl    l2p.py:334  l2p          fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box) renames the instances of l2p one for one
decided  isl    l2p.py:334  l2p          the order after fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box) runs every
                                           dependence of l2p forward
tested   loopy  l2p.py:334  l2p          the scheduled run of l2p agrees with the native run to the accuracy its types state
decided  isl    l2p.py:340  l2p          for contract(l2p.R), every cell of l2p.R that l2p reads,
                                           l2p.reconstruct has stored by the time it is read
decided  isl    l2p.py:340  l2p          no read of l2p.R sees its slot overwritten after contract(l2p.R, lambda box, a, b: (a, b)):
                                           l2p.R kept in 121 cells, R[box, a, b] in slot (a, b)
decided  isl    l2p.py:340  l2p          for contract(l2p.C), every cell of l2p.C that l2p reads,
                                           l2p.unpack has stored by the time it is read
decided  isl    l2p.py:340  l2p          no read of l2p.C sees its slot overwritten after contract(l2p.C, lambda box, a, b: (a, b)):
                                           l2p.C kept in 22 cells, C[box, a, b] in slot (a, b)
tested   loopy  l2p.py:340  l2p          the scheduled run of l2p agrees with the native run to the accuracy its types state
decided  isl    l2p.py:347  l2p          on_groups(l2p.unpack.box) renames the instances of l2p one for one
decided  isl    l2p.py:347  l2p          the order after on_groups(l2p.unpack.box) runs every dependence of l2p forward, within one work item
tested   loopy  l2p.py:222  unpack       the scheduled run of unpack agrees with the native run to the accuracy its types state
tested   loopy  l2p.py:249  reconstruct  the scheduled run of reconstruct agrees with the native run to the accuracy its types state
tested   loopy  l2p.py:288  evaluate     the scheduled run of evaluate agrees with the native run to the accuracy its types state

13 facts: 8 decided, 5 tested

WITHOUT LEAN: 8 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```

## taylor_reconstruction.py

The seed of the demo's flagship, `l2p.py`: sumpy's compressed Taylor
reconstruction for the 2-D Laplace kernel, on one triangle, in one file and
one ledger, from the theorem about the
expansion to the kernel that computes it to the schedule that runs it.

- `on_the_triangle` is a lanky theorem: a table `R` that copies the table `c`
  where `a < 2` and follows the recurrence `R(a + 2)(b) == -R(a)(b + 2)`
  inside the triangle `a + b <= p` is `D` there, for any `D` that satisfies
  Laplace's PDE and whose entries `c` holds. Lean proves it for every `p`,
  with Mathlib; without it, the row is `assumed`.
- `reconstruct` is the loopty kernel that computes `R` from the `2p + 1`
  coefficients sumpy stores. Its postcondition, the copies and the
  recurrence, is what its statements store, so it is decided by `isl` from
  the term, each equation of reals an atom, resting on the fact that no cell
  of `R` is stored twice or read before its store.
- `@kernel(uses=[on_the_triangle])` applies the theorem to the postcondition:
  the copies and the recurrence are two of the theorem's hypotheses, so its
  conclusion, `R[a, b] == D(a)(b)` on the triangle, holds of the kernel's
  `R` for every `D` the other two hold of. The row reads `by
  on_the_triangle, for every D with pde and stored: ...`, rests on the
  theorem's proof and on the postcondition, and with Lean, Lean applies the
  theorem as it was proved and checks that it gives the claim. A kernel
  that computed something else, a recurrence off by one, would get no such
  row: every hypothesis about what the kernel writes has to be a conjunct
  of its postcondition.
- `cpu`, a schedule of the kernel, splits its rows; its casts are rows of
  the same ledger. Running `b` outside `a` is refused with the pair of
  instances it would reorder.

### python examples/taylor_reconstruction.py

```console
$ uv run python examples/taylor_reconstruction.py
order 6: 28 coefficients on the triangle from the 13 stored
largest difference from the derivative: 0.0e+00
at z = (0.6+0.8j): d^6 log r / dx^6 = -90.263040, reconstructed -90.263040

schedule: Schedule(reconstruct, target='c').split(reconstruct.a, 2)
  decided  isl  split(reconstruct.a, 2) renames the instances of reconstruct one for one
  decided  isl  the order after split(reconstruct.a, 2) runs every dependence of reconstruct forward
interchange(b, a) -> IllegalCast: interchange(reconstruct.b, reconstruct.a) illegal: R[a, b] = c[a, b] (taylor_reconstruction.py:136) at a=1, b=2 writes R[1, 2], read by R[a, b] = -R[a - 2, b + 2] (taylor_reconstruction.py:138) at a=3, b=0, scheduled earlier (at p=3, isl's choice)

  R: difference 0 within 1e-06 (approx) -> tested
```

### lanky check examples/taylor_reconstruction.py

Without Lean, the theorem is `assumed` (lanky's tester cannot draw a table
over all of `Nat`), the kernel's facts `decided` by `isl`, and the
application `decided` by `theorem`, worth what the theorem is. With Lean and
Mathlib (`LANKY_LEAN_MATHLIB`), every row but the readings and the trace's is
`proved`: `16 facts: 14 proved, 2 tested`. That takes Lean about a minute and
a half the first time. Checked again with nothing changed, every answer is
read from the proof cache, no Lean is started, and the check takes about two
seconds; the first check after a cold one may still ask Lean once, for an
attempt the cold one ran out of time for (lanky#134).

```console
$ uv run lanky check examples/taylor_reconstruction.py
STATUS                         EFFECTIVE  BY           WHERE                         OWNER            STATEMENT
-----------------------------  ---------  -----------  ----------------------------  ---------------  -------------------------------------------------------------------------------------------------
assumed                        assumed    -            taylor_reconstruction.py:73   on_the_triangle  p : Nat, c : Fn[Nat, Nat, Real], D : Fn[Nat, Nat, Real], R : Fn[Nat, Nat, Real]
                                                                                                      | pde: forall a in Nat, b in Nat. D(a + 2)(b) + D(a)(b + 2) == 0
                                                                                                      | stored: forall a in Fin(p + 1), b in Fin(p + 1) where a < 2 and a + b <= p. c(a)(b) == D(a)(b)
                                                                                                      | copies: forall a in Fin(p + 1), b in Fin(p + 1) where a < 2 and a + b <= p. R(a)(b) == c(a)(b)
                                                                                                      | recurrence: forall a in Fin(p + 1), b in Fin(p + 1) where a + b + 2 <= p.
                                                                                                          R(a + 2)(b) == -R(a)(b + 2)
                                                                                                      |- forall a in Fin(p + 1), b in Fin(p + 1) where a + b <= p. R(a)(b) == D(a)(b)
tested                         tested     python       taylor_reconstruction.py:73   on_the_triangle  the term computes what the annotations compute
decided                        decided    isl          taylor_reconstruction.py:136  reconstruct      R[a, b] is in bounds for every instance of S0
decided                        decided    isl          taylor_reconstruction.py:136  reconstruct      c[a, b] is in bounds for every instance of S0
decided                        decided    isl          taylor_reconstruction.py:138  reconstruct      R[a, b] is in bounds for every instance of S1
decided                        decided    isl          taylor_reconstruction.py:138  reconstruct      R[a - 2, b + 2] is in bounds for every instance of S1
decided                        decided    isl          taylor_reconstruction.py:136  reconstruct      distinct instances of S0 write distinct cells of R
decided                        decided    isl          taylor_reconstruction.py:138  reconstruct      distinct instances of S1 write distinct cells of R
decided                        decided    isl          taylor_reconstruction.py:138  reconstruct      every cell of R that S1 reads, S0 or S1 wrote before the read
decided                        decided    isl          taylor_reconstruction.py:116  reconstruct      the source order runs every dependence forward in time
decided                        decided    isl          taylor_reconstruction.py:136  reconstruct      no cell of R is stored twice, or read before its store
decided                        decided    isl          taylor_reconstruction.py:116  reconstruct      (forall a in Fin(p + 1), b in Fin(p + 1) where a < 2 and a + b <= p. R[a, b] == c[a, b])
                                                                                                        and (forall a in Fin(p + 1), b in Fin(p + 1) where a + b + 2 <= p. R[a + 2, b] == -R[a, b + 2])
decided under on_the_triangle  assumed    theorem      taylor_reconstruction.py:116  reconstruct      by on_the_triangle, for every D with pde and stored:
                                                                                                        forall a in Fin(p + 1), b in Fin(p + 1) where a + b <= p. R[a, b] == D(a)(b)
tested                         tested     interpreter  taylor_reconstruction.py:116  reconstruct      the traced term computes what the body computes
decided                        decided    isl          taylor_reconstruction.py:185  reconstruct      split(reconstruct.a, 2) renames the instances of reconstruct one for one
decided                        decided    isl          taylor_reconstruction.py:185  reconstruct      the order after split(reconstruct.a, 2) runs every dependence of reconstruct forward

16 facts: 1 assumed, 13 decided, 2 tested

WITHOUT LEAN: on_the_triangle and 13 more rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
WITHOUT MATHLIB: on_the_triangle and 2 more rows are not proved, since only Lean with Mathlib can state them. To have Lean prove them: python -m lanky.mathlib ~/mathlib (about 7 GB), then export LANKY_LEAN_MATHLIB=~/mathlib.
```

### loopty run examples/taylor_reconstruction.py

```console
$ uv run loopty run examples/taylor_reconstruction.py
reconstruct: Schedule(reconstruct, target='c').split(reconstruct.a, 2)
  R: difference 0 within 1e-06 (approx) -> tested

STATUS   BY     WHERE                         OWNER        STATEMENT
-------  -----  ----------------------------  -----------  -------------------------------------------------------------------------------------------
decided  isl    taylor_reconstruction.py:185  reconstruct  split(reconstruct.a, 2) renames the instances of reconstruct one for one
decided  isl    taylor_reconstruction.py:185  reconstruct  the order after split(reconstruct.a, 2) runs every dependence of reconstruct forward
tested   loopy  taylor_reconstruction.py:185  reconstruct  the scheduled run of reconstruct agrees with the native run to the accuracy its types state

3 facts: 2 decided, 1 tested

WITHOUT LEAN: 2 rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
```
