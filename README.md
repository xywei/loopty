# loopty

loopy, with types: the algorithm's theorem, the kernel and its transforms in
one Python file, and one ledger that says how each of their claims is known.

If you write loopy kernels, this is what changes. A kernel is a typed Python
function, which runs on numpy as it stands. A schedule is a chain of the
transforms you already use, `split`, `tag`, `prioritize`, `fuse`, with
`contract` to keep what passes between fused loops in a window. The theorem the algorithm
rests on is a [lanky](https://github.com/xywei/lanky) claim in the same file,
written as a Python annotation. Then one command prints every claim the file
makes and what settled it:

- **isl decides** that every access is in bounds, that no two instances write
  one cell, and that every transform keeps every dependence, and refuses an
  illegal transform with the two instances it would reorder;
- **Lean checks**: each fact isl decided is stated again as a Lean theorem,
  from loopty's own description of the kernel and not from what isl
  computed, and proved there, with isl's hints checked rather than trusted;
  with Mathlib, Lean also proves the algorithm's theorems, and checks each
  where a kernel cites it, against what the kernel stores;
- **the runs are compared**: the compiled run with the native one, and the
  native one with the upstream code the kernel was written after.

You write no Lean, no isl and no loopy kernel. `--show-lean` prints the
Lean, and `loopty run --emit-code` the generated code, for whoever wants to
read them.

## One file: sumpy's L2P

`examples/l2p.py` is sumpy's L2P for the 2-D Laplace kernel
(`E2PFromSingleBox`), abridged here to its three layers: the theorem sumpy's
compressed Taylor expansion rests on (lanky#86), the kernel that rebuilds
each box's triangle of derivatives and cites it, and the schedules.

```python
@theorem
def compressed_taylor(
    D: Fn[Nat, Nat, Real],
    R: Fn[Nat, Nat, Real],
    pde: all(D[a + 2, b] + D[a, b + 2] == 0 for a in Nat for b in Nat),
    stored: all(R[a, b] == D[a, b] for a in Nat for b in Nat if a < 2),
    recurrence: all(R[a + 2, b] == -R[a, b + 2] for a in Nat for b in Nat),
) -> all(R[a, b] == D[a, b] for a in Nat for b in Nat):
    """What sumpy stores of a harmonic table, reconstructed, is the table."""


@kernel(uses=[on_every_triangle])  # compressed_taylor, box by box on a + b <= P
def reconstruct(
    C: Arr[Where[box: Fin[nb], a: Fin[2], b: Fin[Q], a + b <= P], Real],
    R: Arr[Where[box: Fin[nb], a: Fin[Q], b: Fin[Q], a + b <= P], Real],
) -> ...:  # the copies and the recurrence: two of the theorem's hypotheses
    for box, a, b in R.dom:
        if a < 2:
            R[box, a, b] = C[box, a, b]
        else:
            R[box, a, b] = -R[box, a - 2, b + 2]


# ... unpack, which reads sumpy's coefficients through its storage index,
# evaluate, the Taylor sum at each target, and the program l2p that runs the three

fused = Schedule(l2p, target="c").fuse(  # the three calls' loops over boxes
    l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box
)
contracted = fused.contract(l2p.R).contract(l2p.C)  # one box's triangle at a time
on_a_gpu = fused.retarget("opencl").on_groups(l2p.unpack.box)  # sumpy's own
```

A schedule names what it moves by where the file defines it: `l2p.R` is the
table the program `l2p` makes, `l2p.unpack.box` the loop over boxes of the
kernel `unpack` where `l2p` calls it, and `reconstruct.b` the loop over `b`
in `reconstruct` itself. A name is checked when it is written, against the
traced kernel, and every message, schedule and fact prints the same paths.

`python examples/l2p.py` runs it on numpy, refuses the interchange of the
triangle's loops with the pair of instances it would reorder, refuses a
window of one row with the read it would break, finds the window that keeps
one box's triangle, and runs the schedule compiled:

```console
$ python examples/l2p.py
...
refused: interchange(reconstruct.b, reconstruct.a) illegal: R[box, a, b] = C[box, a, b] (l2p.py:271) at box=0, a=1, b=9 writes R[0, 1, 9], read by R[box, a, b] = -R[box, a - 2, b + 2] (l2p.py:273) at box=0, a=3, b=7, scheduled earlier (at nb=1, isl's choice)

refused: contract(l2p.R, lambda box, a, b: b) illegal: R[box, a, b] = -R[box, a - 2, b + 2] (l2p.py:273) at box=0, a=9, b=0 reads R[0, 7, 2], which the same statement at box=0, a=7, b=2 stored in slot 2, after the same statement at box=0, a=8, b=2 stored R[0, 8, 2] there (at nb=1, isl's choice); contract(l2p.R) finds the smallest window that keeps every read, lambda box, a, b: (a, b)

schedule: Schedule(l2p, target='c').fuse(l2p.unpack.box, l2p.reconstruct.box, l2p.evaluate.box).contract(l2p.R, lambda box, a, b: (a, b)).contract(l2p.C, lambda box, a, b: (a, b))
...
  decided  isl  no read of l2p.R sees its slot overwritten after contract(l2p.R, lambda box, a, b: (a, b)): l2p.R kept in 121 cells, R[box, a, b] in slot (a, b)
...
  pot: difference 0 within 1.1e-06 (approx) -> tested
...
```

`lanky check examples/l2p.py`, on a machine with Lean and Mathlib, says
how each claim is known and who settled it. Six of its 56 rows:

```text
$ lanky check examples/l2p.py
STATUS   EFFECTIVE  BY           WHERE       OWNER              STATEMENT
-------  ---------  -----------  ----------  -----------------  ----------------------------------------------------------------------------------------------------
proved   tested     lean         l2p.py:129  compressed_taylor  D : Fn[Nat, Nat, Real], R : Fn[Nat, Nat, Real]
                                                                | pde: forall a in Nat, b in Nat. D[a + 2, b] + D[a, b + 2] == 0
                                                                | stored: forall a in Nat, b in Nat where a < 2. R[a, b] == D[a, b]
                                                                | recurrence: forall a in Nat, b in Nat. R[a + 2, b] == -R[a, b + 2]
                                                                |- forall a in Nat, b in Nat. R[a, b] == D[a, b]
...
proved   proved     lean         l2p.py:198  storage_index      storage_index stores a cell at every address of Fin(21)
...
proved   tested     lean         l2p.py:246  reconstruct        by on_every_triangle, for every D with pde and stored:
                                                                  forall i in Fin(nb), a in Nat, b in Nat where a + b <= 10. R[i, a, b] == D[i, a, b]
...
proved   proved     lean         l2p.py:334  l2p                no read of l2p.R sees its slot overwritten after contract(l2p.R, lambda box, a, b: (a, b)):
                                                                  l2p.R kept in 121 cells, R[box, a, b] in slot (a, b)
...
proved   proved     lean         l2p.py:342  l2p                the order after on_groups(l2p.unpack.box) runs every dependence of l2p forward, within one work item
tested   tested     upstream     l2p.py:464  l2p                the native run of l2p agrees with sumpy_l2p in pot within 1e-14 of the largest value,
                                                                  on its example inputs

56 facts: 3 decided, 46 proved, 7 tested

LEAN proved all 44 facts isl decided. Lean's statements are loopty's, from its own description, not isl's results, and isl's hints were checked, not trusted. --show-lean prints the Lean.
```

`STATUS` is how a claim was settled, and `EFFECTIVE` what it is worth given
what it rests on. The theorem is proved for every order, by an induction
whose steps sympy found and Lean checked, and is worth `tested` because
lanky reads its annotations by running them. The application rests on the
theorem and on `reconstruct`'s postcondition, which isl decided from what
the kernel stores and Lean proved again. The window's fact and the GPU
schedule's cast rest on nothing weaker, and are worth `proved`. The last row
is the agreement with sumpy's own kernel. The three rows left `decided` are
the class of the Taylor sum, by type, and two postconditions restated after
their calls, by the call.

Without Lean the same file reads `decided isl` where this reads `proved
lean`, and the theorems read `assumed`, with a line under the table for
each piece that is missing and the command that installs it. That ledger,
which CI checks, is in
[examples/README.md](https://github.com/xywei/loopty/blob/main/examples/README.md#l2ppy).

## The demos

Five files, each written after a kernel of sumpy, pytential, grudge or
firedrake, codes whose kernels loopy generates. Each runs three ways,
`python FILE`, `lanky check FILE` and `loopty run FILE`, and
[examples/README.md](https://github.com/xywei/loopty/blob/main/examples/README.md)
has what each prints. "Proved" below is with Lean and Mathlib.

| file | written after | what its ledger has | how it compares |
|---|---|---|---|
| `l2p.py` | sumpy's L2P for the 2-D Laplace kernel, `E2PFromSingleBox`, and its schedule for a GPU | the compressed Taylor recurrence proved for every order and applied to the kernel's postcondition; sumpy's storage index one for one; the interchange refused with its pair; each box's triangle in a window that no read sees overwritten | the potential sumpy's own kernel computes, to round-off (an `upstream` row); unrolled, the checked schedule times as plain loopy's, and looped, as plain loopy given the same expression ([#270](https://github.com/xywei/loopty/issues/270)) |
| `hexahedra.py` | grudge PR #418's sum-factorized weak gradient and divergence on hexahedra | the programs' postconditions, the dense Kronecker operators, proved for every order from what the passes store and `sum_factorization`; the DOFArray the same layout as #418's `dim_tags`, for every order; a fusion one loop too deep refused with its pair; the intermediates kept a row and a slab at a time | #418's own `weak_local_grad` and `weak_local_div`, to 5e-15 of the largest value; each device launch times as the same launch in plain loopy, within 1% at the median and 8% at worst, where loopty keeps the program's rounding order |
| `advection.py` | grudge #418's weak advection right-hand side, upwind flux, lift and inverse mass, and libParanumal's split of it | the inverse mass pushed inward, and the lift the face gather's transpose, proved for every order; five device splits checked before they run; libParanumal's lift onto the face nodes, without its barriers, refused with the edge node two faces update | grudge's `WeakAdvectionOperator`, to 7e-15; the launches time as plain loopy's, within 1% at the median and 13% at worst, for the same reason |
| `layer_potential.py` | pytential's direct QBX: sumpy's `LayerPotential` with a line-Taylor expansion, and its CPU tile | the line-Taylor coefficients from their recurrence, proved for every order (that a logarithm's coefficients satisfy it is cited), and tied by a row to the terms `single_layer` adds; sumpy's tile checked cast by cast; the write race sumpy silences, a proved fact | sumpy's kernel as pytential calls it, within 1e-12 of the largest value (`upstream` rows); pytential's Green identity and jump relations converge at order 3.5 to 3.9, above the 3.4 its test asks; times as plain loopy given the same expression |
| `assembly.py` | PyOP2's assembly loop around FFC's P1 mass kernel, from firedrake's tests | the element mass `abs(J) (1 + delta_jk) / 24` as `mass_ffc`'s postcondition, proved over the reals; the rule exact on quadratics; the scatter keeps the total; PyOP2's GPU split refused with the node its two triangles share, and accepted once the updates are atomic | PyOP2's own `par_loop` and firedrake's `assemble`, to 5e-15; times as plain loopy given the same transforms and loop nesting, within a few percent |

The timings are `benchmarks/`'s drivers, run on a GPU and on CPUs beside
the upstream code; their numbers describe the machines, so they are not
kept here ([benchmarks/README.md](https://github.com/xywei/loopty/blob/main/benchmarks/README.md)).

## Run it

The demos run on the 0.3.0 releases of loopty and lanky on PyPI. The
examples are in the repository and not in the package, so clone the
release's tag for them. On Python 3.12 or 3.13 with a C compiler:

```sh
python3.13 -m venv .venv && . .venv/bin/activate
pip install loopty "lanky[lean,cas]"
git clone --branch v0.3.0 https://github.com/xywei/loopty.git && cd loopty
python examples/l2p.py
lanky check examples/l2p.py
loopty run examples/l2p.py
```

That much runs without Lean, and the ledger says what is missing, with the
command for each piece. Lean comes from [elan](https://github.com/leanprover/elan#installation),
at the version the Lean REPL supports, and Mathlib, for the theorems about
real numbers, is one command that fetches its compiled files:

```sh
elan default leanprover/lean4:v4.29.1
python -m lanky.mathlib ~/mathlib    # about 7 GB
export LANKY_LEAN_MATHLIB=~/mathlib
```

The first check with Lean builds Lean's REPL, which takes about a minute and
the network. A check after it reads Lean's answers from a cache and starts no
Lean. [Install](#install) has the rest.

## Where the limits are

- **loopty checks the kernels it traces.** A loopy kernel built elsewhere,
  sumpy's or TSFC's, is not checked, nor a transform applied to one. What
  ties a demo to the upstream code is a tested agreement row, not a proof.
- **Some steps stay trusted**: the printer from loopty's description to
  Lean, the Lean kernel and `omega`, and the step from the checked model to
  the code loopy and the compiler generate, which the comparison of the
  compiled run with the native one covers by sampling
  ([docs/lean-checks.md](https://github.com/xywei/loopty/blob/main/docs/lean-checks.md#what-stays-trusted)).
  lanky checks its reading of a theorem's annotations by running them at
  drawn points, so a proved theorem is worth `tested` in the `EFFECTIVE`
  column.
- **A proof about reals is about the reals.** A theorem, or a postcondition
  Lean proves with Mathlib, is about the kernel's term read over the real or
  complex numbers, not about the floats it computes, and the row says so
  ([#282](https://github.com/xywei/loopty/issues/282)). Without Mathlib,
  about 7 GB, those rows read `assumed` or `tested`.
- **On a device, no local memory or barriers yet**
  ([#256](https://github.com/xywei/loopty/issues/256)), so a kernel that
  stages passes in a work group's local memory, as the fastest
  sum-factorized and libParanumal kernels do, cannot be written. A window
  each work item keeps is decided by isl and not yet proved by Lean
  ([#295](https://github.com/xywei/loopty/issues/295)). CI runs no device.
- **loopy 2025.2.** loopty is held to the loopy release it is tested with.
  Its port to loopy `main`, on namedisl, is a draft
  ([#180](https://github.com/xywei/loopty/pull/180)) until loopy releases it.
- [Known limits](#known-limits) lists the open issues you are most likely to
  meet, each with what to do about it.

## A first kernel, row by row

The ledgers from here on are what CI prints, without Lean. Where Lean is
installed, the rows below that read `decided isl` read `proved lean`, but
for the few facts that stay isl's alone (Known limits).

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

Its three commands:

```console
$ python examples/spmv.py            # runs natively on numpy; the theorem is tested
$ lanky check examples/spmv.py       # the ledger: every obligation and who decided it
$ loopty run examples/spmv.py        # lowers through loopy, compiles, runs, compares
```

and the ledger `lanky check` prints, abridged to the rows discussed here (each
row verbatim, and checked by `scripts/refresh_example_outputs.py`):

```console
$ lanky check examples/spmv.py
STATUS   BY             WHERE        OWNER          STATEMENT
-------  -------------  -----------  -------------  ---------------------------------------------------------------------------------------------------
decided  isl            spmv.py:81   scan           off[0] is in bounds for every instance of off[0] = 0 (spmv.py:81)
decided  isl            spmv.py:83   scan           off[r + 1] is in bounds for every instance of off[r + 1] = off[r] + cnt[r] (spmv.py:83)
decided  isl            spmv.py:81   scan           distinct instances of off[0] = 0 (spmv.py:81) write distinct cells of off
decided  isl            spmv.py:81   scan           no cell of off is stored twice, or read before its store
decided  isl            spmv.py:71   scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
tested   property-test  spmv.py:86   scan_monotone  n : Nat, cnt : Fn[Fin(n), Nat], off : Fn[Fin(n + 1), Nat]
                                                    | h0: off(0) == 0
                                                    | hs: forall r in Fin(n). off(r + 1) == off(r) + cnt(r)
                                                    |- forall a in Fin(n + 1), b in Fin(n + 1) where a <= b. off(a) <= off(b)
decided  isl            spmv.py:114  spmv           y[r] is in bounds for every instance of y[r] = reduce_sum(...) (spmv.py:114)
decided  type           spmv.py:114  spmv           x[col[r, j]] is in bounds by type (col[r, j] : Fin(m))
decided  isl            spmv.py:114  spmv           distinct instances of y[r] = reduce_sum(...) (spmv.py:114) write distinct cells of y
decided  type           spmv.py:114  spmv           the accumulation into y[r] over j is approx
tested   interpreter    spmv.py:104  spmv           the traced term computes what the body computes
decided  isl            spmv.py:195  spmv           the order after split(spmv.j, 2) runs every dependence of spmv forward
decided  call           spmv.py:117  solve          after scan(...) in solve: off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
...
27 facts: 22 decided, 5 tested

WITHOUT LEAN: scan_monotone and 18 more rows are not proved. To have Lean prove what it can: pip install "lanky[lean]", and Lean v4.29.1 from elan (elan default leanprover/lean4:v4.29.1).
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

`scan`'s postcondition is `decided` by `isl`, from what the kernel's
statements store. No cell of `off` is stored twice, or read before its store,
which is the row above it, so what `off[r + 1] = off[r] + cnt[r]` stores at
`r` is what `off[r + 1]` holds when `scan` returns, and the claim is that equation
at every `r`. It is also evaluated at what every native run of `scan` left in
its arguments, on the file's example inputs and on drawn ones, and held after
each, which its provenance keeps. The `split(j, 2)` row is a cast of the
schedule the file holds, `rows_split`, decided as the step was taken: `lanky
check` lists the casts of a file's schedules beside the facts of the kernel
they schedule. The last row belongs to `solve`, the
`@program` that runs `scan` and then `spmv`. It restates `scan`'s postcondition
in the program's scope, `decided` by the call, and rests on `scan`'s own
postcondition fact, `postcondition:spmv.scan@69`, so it is worth what that fact
is worth. The id is the kind, then
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
Schedule(jacobi).tile(jacobi.t, jacobi.i, 8, 8) ->
  IllegalCast: tile(jacobi.t, jacobi.i, 8, 8) illegal: u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2 (stencil_skew.py:61) at t=0, i=8 writes u[1, 8], read by the same statement at t=1, i=7, scheduled earlier (at nt=16, nx=16, as hinted)
...
accepted: Schedule(jacobi, target='c').skew(jacobi.i, by=jacobi.t).tile(jacobi.t, jacobi.i, 8, 8)
  loop nest: jacobi.t.outer jacobi.i.outer jacobi.t.inner jacobi.i.inner
  decided  isl  skew(jacobi.i, by=jacobi.t) renames the instances of jacobi one for one
  decided  isl  the order after skew(jacobi.i, by=jacobi.t) runs every dependence of jacobi forward
  decided  isl  tile(jacobi.t, jacobi.i, 8, 8) renames the instances of jacobi one for one
  decided  isl  the order after tile(jacobi.t, jacobi.i, 8, 8) runs every dependence of jacobi forward
...
```

The rejection names two real instances of the kernel, not an empty set or a
failed pattern match: the statement as the file writes it, its line, and each
instance's loop variables. It is the pair a person would find by hand, at sizes
the message states, because which violating pair isl picks depends on them.

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
    for i, j in f.dom:
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
-------  -----------  -----------  -----  ----------------------------------------------------------------------------------------------------
decided  isl          pairs.py:77  pairs  f[i, j] is in bounds for every instance of f[i, j] = q[i] * q[j] /
                                            (1.0 + dx * dx + dy * dy) (pairs.py:77)
...
decided  isl          pairs.py:79  pairs  f[p, j] is in bounds for every instance of e[p] = reduce_sum(...) + reduce_sum(...) (pairs.py:79)
decided  isl          pairs.py:79  pairs  f[k, p] is in bounds for every instance of e[p] = reduce_sum(...) + reduce_sum(...) (pairs.py:79)
...
17 facts: 16 decided, 1 tested
```

`f[k, p]` is read under `k > p`, which makes `(k, p)` a point of the triangle,
while `f[p, p]` would be refused although the `n x n` box around the triangle
has the cell. How `f` is stored is a separate choice that changes no fact:
`Schedule(pairs)` keeps it in that box, and `Schedule(pairs).pack(pairs.f)` keeps
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
  before any code is generated. Fusing two kernels a program calls is one of
  them, a map per statement that runs both in one loop, and so the storage of
  what passes between them is a question of the schedule and not of the
  kernels: computed where it is read, kept in full, or kept in the cells live
  at once, with a checked fact that no read sees its slot overwritten.
- **Reassociation is visible in the type.** Splitting an accumulation and summing
  the pieces is not free on floating point. A trace reads the accumulation's
  class off what it sums, `realize(spmv.y, tree=True)` is what lowers it to
  `reassoc`, and the differential comparison against the native run widens its
  tolerance because of that fact and not because someone chose a number. The
  tolerance is per element and local: `exact` is bitwise, and `reassoc` and
  `approx` ask that `|got - want| <= eps_class * (|want| + 1)` at every cell, so
  it is the accuracy claimed for that cell and does not grow with the size of
  the output. Over an `exact` accumulation the same cast is refused, and a
  kernel with an `exact` output is compiled with floating-point contraction
  off, so a fused multiply-add cannot change its last bit.
- **An assembly's race is named by the mesh.** PyOP2 adds each cell's values
  into the cell's nodes, `y[cell[c, k]] += yl[c, k]`, and two cells that share
  a node add into one cell of `y`. That is an update: either order gives the
  sum, so the kernel owes the class of the sum and not distinct writes, and a
  schedule may run two updates of one cell in another order but not at the
  same time. A split of the cells onto work items is refused with the two
  updates that collide on the example inputs, the node PyOP2's two triangles
  share, not with a pair isl picked out of every cell the map could reach,
  and accepted once the updates are atomic:

  ```python
  cells = mass_action.c
  Schedule(mass_action, target="opencl").atomic(mass_action.y).split(
      cells, 32
  ).tag({cells.outer: "g.0", cells.inner: "l.0"})
  ```
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
- **A layout is a checked map.** Where an array's cells are stored is a Python
  function from a cell's indices to its address, and that it stores every
  cell inside its buffer, no two at one address and a cell at every address
  is a fact isl decides and Lean proves, as is that two layouts store every
  cell alike: sumpy's compressed-Taylor storage index, and grudge's
  `dim_tags` `N3,N0,N1,N2` against modepy's reshape of the DOFArray
  (`examples/storage_layouts.py`), at every order in
  `examples/hexahedra.py`, where isl tests it at a few orders and Lean proves
  it for every one. loopy's shaped arrays (loopy #1053) take a layout's
  injectivity on trust.
- **What isl decides, Lean proves**, one dependence class or one place at a
  time, from loopty's description of the kernel and never from a set isl
  computed, so the two answers are independent (the top of this page);
  [docs/lean-checks.md](https://github.com/xywei/loopty/blob/main/docs/lean-checks.md)
  has the details and the coverage.
- **A kernel is a rule with a domain of its own.** A kernel body calls a
  kernel as Python calls a function, and the call is inlined with binders of
  its own, which is loopy's substitution rule with a loop of its own (loopy
  #1044) expanded where it is used. A rule returns its value:

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

  What `dot`'s contract checks at each call natively, the row `A[i]` in
  bounds, an index argument in its sort, two rows apart where it writes
  one, `matvec` owes at every point it calls `dot`, as facts isl decides
  and Lean proves, and a callee's postcondition holds after each call,
  restated of what it was passed. An argument is evaluated once, when the
  call is made, as in Python, and a scalar is converted into the sort the
  callee declares, as its contract converts it natively: `same(x[i])` of a
  rule taking `s: np.float32` is `np.float32(x[i])` in `matvec`'s term,
  and a value the contract refuses (a real for `Int`) is refused where it
  is passed.
- **It plugs into a proof host.** loopty registers a theory, an isl oracle, an
  executor and a `run` verb with [lanky](https://github.com/xywei/lanky), so a
  residual obligation an oracle cannot decide is an ordinary theorem a person can
  prove, in the same ledger.

## Status

This is 0.3.0. The core works: kernels and programs run natively, trace, state
their facts, lower and run compiled on the C target, and every schedule step
is checked; the edges are sharp. New in 0.3.0 are the five demos at the top of
this page, and Lean proves again each fact isl decides. Below is what works,
what works in part and what does not work yet. After it, Known limits lists
the open issues you are most likely to meet, each with what to do about it.
None of them makes the ledger call a false fact `decided`, but in several the
compiled run computes something other than the native run, and the ledger
catches that only on the inputs it tries: read that section before you rely on
a compiled run. CI runs the suite on every change, on Python 3.12 and 3.13
with lanky's `main` and on 3.13 with the released lanky, and reruns every
command whose output this README, the quickstart and the examples' README
show, holding them to what it prints. It runs no OpenCL device. On the branch
`port/loopy-main`, which runs on loopy's main branch
([docs/loopy-main.md](https://github.com/xywei/loopty/blob/port/loopy-main/docs/loopy-main.md)),
CI also runs every example three ways, on 3.14 too, and against loopy main's
current head. Until 1.0 an interface can change in a minor release; the
CHANGELOG says when one does.

**Works.**

- `@kernel` and `@program`: inert, registering, running natively on numpy.
- A program's term, and one kernel for it. `Program.term` is what the body
  does run once against placeholders: every kernel call is recorded instead
  of run, and the callees' terms follow one another in call order, in the
  program's names. Sizes are unified through the arrays passed (`scale`'s `n`
  is `scan`'s `n + 1` when both are handed `off`), each call's loops get names
  of their own, and its statements' ids are named after it (`scan.S1`,
  `step@2.S0`), which a map per statement names them by; a message names a
  statement by its text and line. An array the body makes with `Arr.zeros_like(u)` is a
  temporary of the term, zeroed where it was made, and a loopy temporary of
  the lowered kernel, so the intermediate between two kernels is nobody's
  argument. The edges between the calls are not declared: an array one call
  writes and a later one reads is one array of the term, so the dependence is
  in the footprints, and it orders the instructions of the one loopy kernel
  the program lowers to. `LoopyExecutor().run(program, ...)` runs it,
  `Schedule(program)` schedules it, and `loopty run` compares every program in
  a file with its native run, as it does a kernel. A program may also make
  an array of a declared shape, whose sizes it reads off its arguments as it
  does natively, `Arr.zeros(TRIANGLE, n=x.dom.size)` over a domain or
  `Arr.zeros((x.dom.size, 3))`, or `Arr.zeros((x.dom[0].size,))` where `x`
  has a row 0 (the compiled run is refused where it has none, as natively);
  a call may be given one array for two parameters it only reads,
  `outer_product(u, u, z)`; and a callee may take an array over a
  `Where`, `Sigma` or union domain, whose sizes are unified across the calls
  and whose points two calls given it have to agree on, compared as isl sets
  (#74); an array the program makes over a domain is stored boxed, and
  `Schedule.pack` packs an argument over one. A body that does anything to
  an argument but pass it to a kernel, make an array like it, or read its
  sizes, is refused
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
- Fusion, and the storage of what passes between calls. An array a
  program makes and one call writes and a later call reads is an internal
  edge, and the program has a `definedness` fact for it, decided by isl:
  every cell the reader reads is one a call before it stored, or one of the
  zeros the array was made with, which the fact's reason then lists
  (`y[i] = f[i] + 1.0 (padding.py:19) reads the zeros at f[0] and f[n - 1]`),
  so zero padding at a boundary is read as written and `lanky check` passes.
  Where no read sees the zeros, every schedule of the program runs without
  them: isl decides that when the schedule is made (`every cell of
  burgers_rhs.f that burgers_rhs reads, burgers_rhs.flux has stored by the
  time it is read, so burgers_rhs runs without the zeros burgers_rhs.f is
  made with`), the fact is the schedule's first, and a window or a
  substitution rests on it.
  `Schedule(burgers_rhs).fuse(burgers_rhs.flux.j, burgers_rhs.divergence.i,
  shift=1)` runs the reader's loop inside the writer's, one step behind, and
  is `affine` with the map per statement it builds, the divergence's `i` at
  the flux's step `j = i + 1`, checked on the dependences between the calls;
  without the shift it is refused with the pair of instances, and the
  message names the fusion that is accepted. `fuse` takes one loop of each
  call, as many calls as are fused, each loop at the same depth of its own
  call, and fuses those loops and every loop outside them, leaving the
  loops inside one after the other: two passes over elements, the second
  of which reads every cell of the element's row the first stores, fuse on
  the element and the row (`fuse(weak_derivative.along_r.a,
  weak_derivative.along_s.a)`) and no deeper, and the refusal of the deeper
  fusion names the shallower one (`examples/sum_factorization.py`).
  `.substitute(burgers_rhs.f)` then stores none of `f`: the one statement that writes
  it becomes a substitution rule (loopy's `assignment_to_subst`) computed
  again at every read, legal when every read is of a cell it stored before
  the read, since no zeros are left to read, and nothing writes what it
  read in between, and every later step is checked against the
  dependences of the program as it then runs.
  `.contract(burgers_rhs.f)` keeps `f` in the cells live at once instead: fused one
  step behind, the divergence reads `f[j]` and `f[j - 2]` at step `j`, so a
  window of three cells is found, each flux computed once:

  ```python
  flux, divergence = burgers_rhs.flux, burgers_rhs.divergence
  fused = Schedule(burgers_rhs).fuse(flux.j, divergence.i, shift=1)
  windowed = fused.contract(burgers_rhs.f)          # lambda j: j % 3
  fused.contract(burgers_rhs.f, lambda j: j % 2)    # refused: a read sees its slot overwritten
  ```

  A placement is a Python function from a cell's indices to its slot, given
  or found axis by axis along the order as it stands (one row of the
  sum-factorized passes' `v`, `lambda e, a, b: b`); an array over a domain
  is placed by the domain's binders (one box's triangle of sumpy's L2P,
  `lambda box, a, b: (a, b)`, in `examples/l2p.py`). The array may be stored
  by several statements, and read by one of them, a recurrence. The step
  is legal when every read is of a cell stored before it, the
  `definedness` fact, and no read sees its slot overwritten between that
  store and the read, the `no-clobber` fact, decided by isl over the
  schedule's order and proved by Lean, and refuted with the read, the cell
  and the instance that overwrote its slot. From then on every pair of
  accesses to one slot is a dependence of the schedule, so a later step
  that breaks the window is refused with its pair. A window of a number of
  cells is each work item's own, so the boxes of the L2P on `g.0` each keep
  their triangle, and a read on a work item its cell's store is not on is
  refused; a window of a size, `f_window[n]`, is one for every work item
  (in global memory on OpenCL). It lowers to `double f_window[3]` with
  modulo addressing, or, with `registers=True` (the default on OpenCL), to
  a shift register whose every access has a constant index, `f_window[0] =
  f_window[1]` at each step. `examples/fusion.py` is the Burgers flux and divergence of
  `examples/composition.py`, fused, then substituted, or kept in a window,
  each compiled run compared with the native one.
- Declared layouts, checked (`loopty.layout`). A layout is a function from a
  cell's indices to its address, its cells and its buffer in its
  annotations, and a layout of pieces is `layout(first, second)`:

  ```python
  def centre(a: Fin[1], b: Fin[1]) -> Fin[2 * P + 1]:
      return 0

  def derivatives(
      cell: Where[a: Fin[2], b: Fin[Q], (a + b >= 1) & (a + b < Q)],
  ) -> Fin[2 * P + 1]:
      a, b = cell
      return 2 * (a + b) - 1 + a

  storage_index = layout(centre, derivatives)        # sumpy's get_storage_index
  as_dim_tags = dof_array.same_as("N3,N0,N1,N2")     # grudge's output layout
  ```

  `lanky check` asks three facts of each, every cell at an address of the
  buffer, no two at one address, a cell at every address (whose witness for
  Lean is isl's inverse, with `//` and `%`), and `same_as` one more, and a
  kernel reads through the layout's own function. A wrong nesting,
  `N3,N2,N1,N0`, is refuted with the cell it moves
  (`examples/storage_layouts.py`). A layout's body may compute with a size,
  `n**3 * e + (i + n * j + n**2 * k)` over `Fin[n]` for every order: isl
  cannot state `n * j`, so it decides each fact at `n` from 1 to 6, which
  reads `tested`, and Lean proves it for every `n`, given the steps `omega`
  lacks (the products that bound each digit, the digits of an address by
  division); two such layouts the same is a polynomial identity Lean
  multiplies out (`examples/hexahedra.py`). A call gives the size by name,
  `dof_array(e, i, j, k, n=4)`.
- Tracing a body to a typed term: accesses, statements, reductions, ragged
  fibers, branches, source locations, and a `TraceError` that names the fix
  when a `while` loops on a computed value, when a body has more paths
  through its branches than `loopty.trace.MAX_RUNS`, when a Python name, a
  global, or a list, dict or set carries state from one loop iteration to the
  next, when an array is used whole (`y[:] = ...`, `x * 2`, a numpy function of
  it), when a reduction's `if` clause is not a bound isl can state, when the
  trace changes Python state outside the arrays (a global, a closure variable,
  an object's attribute or slot, a list, dict, set or numpy array they hold, a
  ragged array's offsets, or the same in a helper the body calls), and when the
  body prints, reads input, opens a file or draws a random number. The
  kernel's own module and package count as its code wherever they are
  installed, site-packages included.
- Branches: `if`, `elif` and `else`, `and`, `or` and `not`, and conditional
  expressions `a if c else b`, on loop variables, sizes, scalars and data. The
  trace runs the body once per path through them (#266 for the select), each
  condition answered from the path, and joins the runs into one term: a
  statement after an `if`/`else` is one statement, a value two paths choose
  between on data, `vn[i] * (a[i] if vn[i] > 0 else b[i])`, is one statement
  computing a select (`?:` in C), and a path isl shows to be empty records
  nothing. The native run is Python's own: the branch not taken is not run.
  A path's conditions narrow a statement's isl domain where isl can state
  them: an affine comparison of loop variables, sizes and scalars of an
  integral sort. A condition that reads an array, compares with `!=`, or
  compares with a `Real` scalar is evaluated at run time only, the statement lists it in
  `Stmt.unnarrowed`, and the facts stated over its domain say so in their
  provenance. loopy's own bounds check reads a guard that names no array
  into isl only when it is affine in integers; under any other, a product of
  loop variables, a float literal or a `Real` scalar (`if i * 0.5 < 2:`,
  `if i * 2.0 < n:`), it checks an access at every point of the loop, and
  refuses one it cannot show in bounds there, one the guard keeps in bounds
  included. The refusal (`loopty.refusal.GuardNotRead`, a `LoopyIndexError`)
  names the statement, the guard, and a guard over integers that keeps the
  access in bounds, as the body writes it, `if i <= y.dom.size - 5:` for
  `x[i + 4]` (#148; note 23 in
  `docs/loopy-notes.md`). A condition has to be a truth value: natively
  `~(i > 0)` is `-2` or `-1` (`~` on a Python bool is bitwise), which
  Python's `if` takes as true, while the trace reads `not (i > 0)`, so the
  trace refuses it, naming `i <= 0` and `not`. What is stored into
  an array of `Bool` has to be a truth value too, a comparison, a connective
  of truth values or a `Bool` read, since natively the array is a bool and
  compiled a byte, into which C converts `0.5` as `0`: `b[i] = u[i]` is a
  `TraceError` naming `b[i] = u[i] != 0`, and an integer arriving at a bool
  array natively, `~(i > 0)` again, is refused there.
  Every operand of `&`, `|` and `~` has to be a truth value, wherever the
  connective is, the index of the cell written included, since the trace and
  the compiled kernel read them as `and`, `or` and `not` and natively they are
  bitwise on an integer: `(k[i] & 1) * x[i]` is a `TraceError` naming
  `k[i] != 0`, and `k % 2` for `k & 1`. And `~(i > 0)` used as a number
  (`x[i] * ~(i > 0)`, in a sum's body or an index) is a `TraceError` naming
  `i <= 0`, since natively it is the `-2` or `-1` the native run refuses only
  where it is stored into a `Bool` array. More operations are refused, naming what
  to write: `b[i] + c[i]` of two truth values, which is `or` natively and `2`
  compiled (`b[i] | c[i]` for `or`, `1 * b[i] + c[i]` for a count), and
  `b[i] - c[i]`, which numpy refuses; an integer element or scalar to a
  negative integer power, which numpy refuses and loopy computed as an
  integer (`1 / k[i] ** 2`, a real); `^`, `<<` or `>>` of a real, and `//` or
  `%` of a complex value, which numpy refuses and C cannot compile; a
  loop variable divided by a literal zero or shifted by a negative literal,
  which Python refuses natively, or shifted left by a literal of 64 or more,
  which Python computes exactly and the compiled run to `0`
  (`i * 2.0 ** 64`); `-b[i]` of a truth value, which numpy refuses (`~b[i]`
  for `not`, `-(1 * b[i])` for the integer); an integer literal beside a
  numpy integer whose type does not hold it, `u[i] // -1` of a `uint64` or
  `(b[i] // c[i]) * 200` of two truth values (an `int8` natively), which
  numpy refuses (`np.int64(-1)`); `^`, `<<` or `>>` of a `uint64` and a
  signed integer, which numpy refuses; and an integer literal past 64 bits
  where numpy does not compute it in a real or a `uint64` that holds it,
  `1.0 * (i + 2**64)`, which the compiled kernel has no integer type for
  (`2.0 ** 64`).
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
- isl decides, Lean checks (`loopty.description`, `loopty.obligations`;
  [docs/lean-checks.md](https://github.com/xywei/loopty/blob/main/docs/lean-checks.md)).
  Every set loopty builds keeps the text it was written from, on the set,
  with each call's renaming, and every schedule step a record of its maps as
  written; from those, and from no set isl computed, every fact isl decides
  is stated for lanky's Lean oracle in parts (one per place an access is
  made over, one per dependence class of a cast), which proves them with
  `omega` where Lean is installed. isl's work comes in as hints Lean checks:
  its functional form of a step's map as the witness that the step is total,
  its `lexmax` pieces as the writer of each cell a read reads, the instances
  it took of a program's universal hypotheses, and its counterexample to a
  refuted fact, which Lean proves false. `loopty run` asks Lean about the
  casts and layout facts it reports, and `--show-lean` prints the Lean. On
  the examples and the test kernels Lean proves every fact isl decides but
  those listed under Known limits; `scripts/lean_coverage.py` reports it.
- Definedness inside one kernel. An array a kernel reads only at cells an
  earlier instance of its own wrote (`scan` reads `off[r]` where `off[0] =
  0` or the row before stored it) gets a `definedness` fact where isl shows
  it, stated for Lean as a ∀∃ with isl's last writer as the witness; a
  kernel that reads what an array held when it was called owes nothing and
  gets none.
- Postconditions decided from the term (`loopty.dataflow`). Where no cell of
  an array a kernel writes is stored twice, or read before its store, a
  `single-assignment` fact says so, decided by isl and proved by Lean, and
  what each statement stores is what the kernel leaves: its value equation,
  `off[r + 1] == off[r] + cnt[r]` at every `r` of its statement's loop. The
  postcondition is decided under those equations, each taken at the cells
  the claim reads, and rests on the `single-assignment` facts. A claim about
  reals is decided the same way, each equation isl cannot read an atom, so
  `R[a + 2, b] == -R[a, b + 2]` is what the recurrence stores at `(a + 2,
  b)`; Lean states it with Mathlib's reals and proves it by `linarith`. The
  native runs still test it, and a run that refutes it refutes it.
- A kernel cites theorems, `@kernel(uses=[on_the_triangle])`. A theorem
  whose hypotheses are conjuncts of the postcondition, up to the names of
  their binders (a family `R(a)(b)` stands for the array `R[a, b]`), gives
  its conclusion at the kernel's arrays, a `by-theorem` fact resting on the
  theorem and the postcondition: what is left unbound stays universal, and
  the hypotheses the postcondition does not give are its hypotheses. Lean
  checks the application, the theorem as it was proved applied with the
  arrays loopty matched: `examples/taylor_reconstruction.py`,

  ```python
  @kernel(uses=[on_the_triangle])
  def reconstruct(
      c: Arr[Where[a : Fin[2], b : Fin[p + 1], a + b <= p], Real],
      R: Arr[Where[a : Fin[p + 1], b : Fin[p + 1], a + b <= p], Real],
  ) -> all(
      R[a, b] == c[a, b] for a in Fin[p + 1] for b in Fin[p + 1] if a < 2 if a + b <= p
  ) & all(
      R[a + 2, b] == -R[a, b + 2] for a in Fin[p + 1] for b in Fin[p + 1] if a + b + 2 <= p
  ):
      ...
  ```

  reads `by on_the_triangle, for every D with pde and stored: ... R[a, b] ==
  D(a)(b)`: whatever table `D` satisfies the PDE (`pde`) and has its stored
  entries in `c` (`stored`), `R` is `D` on the triangle. A theorem whose
  conclusion is a conjunct of the postcondition decides that conjunct
  instead, as a program's cited theorem decides a requirement. The
  application says something of what the kernel computes, or it is not
  made: every hypothesis about an array the kernel writes has to be a
  conjunct of the postcondition, so a kernel whose recurrence is off by one
  gets nothing from the theorem; and a conjunct about reals is a
  hypothesis only where nothing in it rounds (copies and sign changes),
  since the theorem is about the reals, and `s[i] == a[i] + b[i]` holds of
  the floats and not of them read as reals. The theorem is taken at the
  kernel's sizes in every sort as well: a table `D: Fn[Fin[n], Real]` that
  nothing binds, with `n` matched to the kernel's `p`, ranges over `Fin[p]`.
- A cited theorem about a series, applied to the terms a kernel adds
  (`loopty.summands`). A theorem that concludes `all(c(n) == E for n in
  Nat if ...)` is about the `n`-th term of a series; where the kernel's term
  adds `E` at whole numbers `n`, with the theorem's variables bound to the
  kernel's expressions, a `by-theorem` row says so, resting on the theorem.
  An identity the kernel cites, a theorem that concludes `L == R`, rewrites
  `R` in `E` into `L` first, the kernel's way of computing it.
  `examples/layer_potential.py`'s `single_layer` cites `line_taylor` and
  `by_the_conjugate`, `ratio(alpha, beta) == -beta / alpha`, and reads
  `single_layer adds line_taylor's c(n) for n in 1..5, for every c with
  away, first and ode, read over the complex numbers`: about the term it
  evaluates, read over the complex numbers, and not about the floats it
  produces. With Mathlib, Lean checks it a term at a time, the identity
  rewritten and the theorem applied.
- Equations of reals at their class (`loopty.reals`). An equation of a
  postcondition that rounds, a closed form the kernel computes another way,
  is compared after each native run as the compiled run is with the native
  one, at the class of what it reads: `|a - b| <= 1e-6 (|b| + 1)` for
  `approx`. One that rounds nothing (a copy, a sign change), one of
  integers and one whose class is `exact` are compared bit for bit, and the
  statement says which: `..., to the approx tolerance`. Where the cells it
  reads are stored once, by statements whose sums run over concrete bounds,
  Lean proves it with Mathlib over the reals, the tables at the values they
  hold; that is a proof of the term read over the reals, and the fact rests
  on the kernel's `trace-faithful` fact. `examples/assembly.py`'s `mass_ffc`
  promises the element's mass, `|J| (1 + delta_jk) / 24`, from FFC's
  15-digit tables.
- `lanky check` lists a file's schedules (#191). The casts of every schedule
  a file holds at module level are rows of the ledger beside the facts of
  the kernel or program it schedules, decided by isl as each step was taken
  and proved by Lean where it is installed, each id once.
- Mixed subscripts. `X[cell[c, k], d]` is in bounds by type on its first axis,
  `cell[c, k] : Fin(nv)`, and by isl on the second, where it was assumed.
- `Schedule.assume(k.n % 4 == 0)`, loopy's `assume`, checked: loopy is told,
  every later cast is decided and stated where the condition holds and rests
  on an `assumption` fact, and a call whose sizes break it is refused.
- `Schedule`: `tag`, `split`, `interchange`, `prioritize`, `tile`, `skew`,
  `affine`, `fuse`, each checked as a cast, each emitting its fact; `realize`
  and `atomic`, which move no instance and record what they make of a sum;
  and `substitute` and `contract`, checked as storage decisions;
  `retarget`, which replays every step against another loopy target and
  re-checks it. `prioritize` and `interchange` also take the loop of a sum,
  named before loops of its statement, which it then runs outside, as
  sumpy's CPU tile of a layer potential runs 256 sources at a time outside
  16 targets (`.split(single_layer.s, 256).prioritize(single_layer.s.outer,
  single_layer.t.inner)`): each
  sum still adds its terms in order, the loops it runs outside of order
  none of their instances, and loopy keeps the order where it keeps an
  accumulator per instance, on a hardware axis or an `ilp` loop, which the
  `buildable` fact asks (`examples/layer_potential.py` runs sumpy's tile on
  C with an `ilp` loop for its work items). A split sum adds
  its terms in order, outer loop first. A tag belongs to a loop, so
  `split`, `tile` and `affine` refuse a loop that carries one, and the loops
  they make are tagged after them. `affine(map)` takes an isl map from loops to the loops that
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
  as the diamond's `b` (`acoustic.loops.b`), counts its steps (`b = 2*b_step - a`,
  `Schedule.strides`) instead of testing a parity at every `b`. A loop on a
  hardware axis (`g.*`, `l.*`) is the launch grid, outside every other loop,
  and nothing in a kernel orders two of its work items through global memory,
  so the `monotone` cast also refuses any dependence between instances on two
  work items, with the dependence and the two work items as the witness (the
  stencil's `jacobi` with `i` on `g.0`, the acoustic pair likewise). A
  statement with no loop on the axis runs on every work item of it. A
  schedule runs without the zeros a program makes an array with where no
  read sees them, a `definedness` fact isl decides when the schedule is
  made, so `examples/l2p.py`'s `fused.on_groups(l2p.unpack.box)`, sumpy's
  schedule for a GPU, names only the file's loops and zeroes nothing. Zeros
  a read sees, padding at a boundary, run where the array is stored: their
  loop along an axis of the array goes on the axis of the loop that every
  statement storing the array stores it along. A sum on
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
  Any power compiles on the C target, of a complex base too. Integer
  arithmetic is 64 bits wide in both runs: `Nat` and `Int` are stored in 64
  bits, and a product of a `Fin[m]` array's entries or of loop variables is
  computed in 64, a subscript's too unless loopy reads the subscript as
  affine with no division (note 23). `//`,
  `%`, `<<` and `>>` are computed as numpy computes them, by functions
  loopty's targets define (`loopty.operations`): of reals too, by zero (`0`
  for integers) and past the width of a shift. An operation numpy computes in
  a numpy integer type is computed in it compiled too: one narrower than
  `int` (`np.int8`, `np.uint16`), which C computes as an `int`, is converted
  back into it, and an unsigned integer beside a signed one is computed in
  the signed type numpy computes in (`int64` for a `uint32` and an `int32`),
  where C's would be unsigned; integers compare exactly, as numpy's do, a
  negative one with an unsigned one included. A sum is computed as the body
  wrote it: `c[i] - a[i]` as the difference, and `c[i] + -a[i]` as the sum of
  `a[i]` negated in its own type, which differ for an `int8` `a` at `-128` and
  an unsigned one at every value but zero; `u[i] + -1` of an unsigned `u` and
  `k[i] - 2**31` of an `int32` `k`, which numpy refuses at every point, are
  refused by the trace (#151). A real less an integer keeps the sign of zero,
  `x[i] - 0` and `x[i] - i` at `x[i] = -0.0`, and a truth value subtracted,
  `k[i] - True`, which pymbolic takes no operand of, is refused naming the
  integer. Loop arithmetic beside a numpy integer whose
  type does not hold it at some point, `u[i] + (i - 3)` of a `uint32` `u` at
  `i = 0`, is refused by numpy natively there, and the compiled run is refused
  before it starts where the call's sizes reach such a point, naming it
  (`loopty.refusal.RefusedNatively`, #153). `abs` of an integer is numpy's,
  the smallest value of its type included, and `abs` of a real in a guard is
  written without a call, since loopy resolves no function only a guard calls
  (#154); another function only a guard calls is refused, naming the fix.
  `abs` of a term compares on either side, `if abs(k[i]) > 2:` and `127 ==
  abs(c[i])`, since lanky 0.2.0 (xywei/lanky#94). An integer literal past 64
  bits is written as a real where numpy computes in one (`x[i] * 2**70`), or as a
  `uint64` beside one (`u[i] + 2**63`), and refused elsewhere. A kernel named
  like a function, a macro or a type loopy, the C headers or OpenCL C know
  (`floor`, `pow`, `cpow`, `get_global_id`, `clamp`, `NULL`, `size_t`) is
  renamed in the generated code, and a parameter, size or loop variable
  named like a macro the headers define (`I` with complex values, `NAN`,
  `INT32_MAX`, `M_PI`, `NULL`) or a function the kernel calls (`pow` with a
  power) is refused, naming what the code means by it. So is a size, an
  integral scalar or a reduction binder named like a keyword of isl's (`max`,
  `floor`, `mod`), and a loop variable of such a name is renamed. See notes
  19, 20 and 23 in `docs/loopy-notes.md`.
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
- Elementary functions and complex parts, `loopty.math`: `sqrt`, `exp`,
  `log`, `sin`, `cos`, `real`, `imag` and `conj`, numpy's natively and the
  target library's compiled (`csqrt`, `clog`, `creal` on the C target,
  pyopencl's `cdouble_log` on OpenCL), each computed in the type numpy
  computes it in: `sqrt(k[i])` of an integer is a double, and `real`,
  `imag` and `conj` of a real are the number, `0` and the number. One numpy
  computes in half precision, `sqrt` of an `int8`, is a `TraceError` naming
  `sqrt(1.0 * k[i])`. A value of one compares as any value does,
  `if sqrt(x[i]) > 1.5:`, and numpy's own `np.sqrt`, Python's
  `math.sqrt` and `z[i].real` are refused when traced, naming the function
  of `loopty.math` to write. `examples/layer_potential.py` is pytential's QBX
  single and double layer over complex points. `ipow(x, n)` is `x` to an
  integer power multiplied out as loopy multiplies one, by squaring, in both
  runs, so they agree bit for bit and the compiled kernel calls loopy's own
  `loopy_pow_float64_int32`; `x ** n` of a real `x` is numpy's power, the C
  library's `pow`, in both. In `examples/l2p.py`'s Taylor sum that call was
  the whole cost on the C target.
- Agreement with the code a kernel mirrors, `@upstream(kernel)`
  (`loopty.upstream`): a function that computes, on the file's example
  inputs, what the upstream code computes there, and a ledger row under the
  kernel that compares it with the native run at the accuracy the kernel's
  types claim, `tested` by `upstream`, `refuted` with the cell, or `assumed`
  where the function cannot run (its package or its recording is not
  there). `examples/l2p.py` compares its program with sumpy's own
  `E2PFromSingleBox`, recorded on its inputs. `@upstream(kernel,
  within=1e-12)` compares each output computed in floating point within
  `1e-12` of the largest value the function gave for it, and the row says
  so (`... agrees with sumpy_single_layer in out within 1e-12 of the
  largest value, on its example inputs`): round-off between two orders of
  summation is far inside the class's allowance, and the row states how far.
- Constant tables, `loopty.table`: `W = table([1 / 6, 1 / 6, 1 / 6])`
  next to the kernel, read by the body as an array is (`W[q]`, `W.dom`). A
  table is a read-only copy of its values; its reads are in bounds over its
  shape, decided by isl, and the lowering writes it into the kernel as a
  read-only loopy temporary with the values as its initializer, `static
  double W[3] = {...}` on the C target and `__constant` on OpenCL, with every
  digit. A program holds one table for every call that reads it. Natively
  a table is read as an array is, and it is no argument: an array of its
  values is passed instead, `Arr.from_numpy(W.values.copy())`.
  `examples/quadrature.py` is firedrake's P1 mass kernel.
- Claims settled by running them, `loopty.tested`: a function that takes
  nothing and returns whether its claim held, alone or with what it
  measured, decorated `@tested`, is a row of the ledger, `tested` by `run`
  with its docstring's first line for statement, or `refuted` with the
  reason. It is for what no oracle reads, an accuracy that converges at its
  order say, and it never reads more than `tested`; the agreement with the
  code a kernel mirrors is `@upstream`'s row. `examples/layer_potential.py`
  measures pytential's Green identity and the jump relations this way, and
  compares both its kernels with sumpy's own through `@upstream`.
- Updates and atomics. A statement that adds to the cell it writes a value
  that reads nothing else of the array, `y[cell[c, k]] += yl[c, k]` or `y[r]
  += v[r, j]`, is an update. Its sum has the class of what it adds, as a
  reduction's has (`approx` for `Real`). An update whose instances can add to
  one cell owes that class, `the accumulation into y[cell[c, k]] over c, k is
  approx`, and no distinct writes; one that updates a cell of its own at
  every instance, `y[i] += a * x[i]`, owes its distinct writes. Two updates
  of one cell are no dependence, so a schedule may reorder them, which it
  records as a reassociation. A step that runs two of them at the same time,
  on two work items or in no order (`ilp`, `vec`), is refused with the pair
  and the fix until `Schedule.atomic(mass_action.y)`. Then each update is loopy's
  `AtomicUpdate` on OpenCL, a compare-and-swap, and the C target, which runs
  in one thread, keeps its code. A refused cast through a cell read from an
  array names two instances that touch one cell on the example inputs, with
  what the subscripts read, `where cell[0, 2] = 3 and cell[1, 1] = 3`. A
  write of that kind, `y[perm[i]] = x[i]`, has its distinct writes decided
  on the native runs. `examples/assembly.py` is PyOP2's P1 mass action,
  assembled.
- The argument contract, enforced at every entry point that runs a kernel
  (compiled, differential and native): two distinct array parameters may not
  share storage where the kernel writes either, a ragged argument has to
  agree with the counts array its type names and with any offsets passed
  alongside it, an argument over a domain has
  to have the declared domain's points at the sizes of the call, and a value
  of a refined sort such as `Fin[m]` has to be one — an array element and a
  scalar argument alike, and being one means being a finite whole number in
  range, not merely passing two comparisons. A value of `Nat` or `Int` is
  also inside the 64 bits both runs store it in, and one of `Fin[m]` inside
  the 32 the compiled run stores it in, since `2**32 + 5` would run natively
  as it was and be `5` compiled. An array the kernel writes has to be stored
  as its element sort is natively (`float64` for `Real`, `bool` for `Bool`,
  `int64` for `Nat` and `Int`, a signed integer of 32 bits or more for
  `Fin[m]`, a numpy sort as itself), since an integer `x` for a
  `Real` parameter truncates every write the compiled run keeps; an array it
  only reads is read by the native run in that dtype, as the compiled run
  converts it, so an integer `x` no longer overflows natively where the
  compiled double does not. A complex
  entry of a sort that is not complex has no imaginary part, and an entry of
  `Bool` stored as a number is `0` or `1`. A scalar is asked the same, and is
  converted into the dtype of its sort in both runs, since it is passed by
  value: `np.int64(2**32)` for a `Real` is a double natively too, a `Bool` a
  numpy bool, on which `~` is `not`, and a Python number a numpy scalar, so
  `a=0.7` and `a=np.float64(0.7)` compute alike beside a `float32`. These are the assumptions the typing
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
  schedule of a schedule, `Schedule(Schedule(k).split(k.i, 2))`, is that
  schedule built again from `k`: its steps replayed and checked again, on
  its target and with its sizes unless others are given. Its run is compared
  with `k`'s body, and `k` is not run again through the identity.

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
- A kernel's postcondition is decided from what its statements store where
  every cell of the arrays it claims about is stored once at most, and read
  only after its store (`loopty.dataflow`, below); elsewhere it is `tested`
  against its native runs, on the file's example inputs and on drawn ones,
  and its provenance says why it is not decided (`undecided`). A cell stored
  twice, a running value a loop overwrites, needs the last store before
  each read, which is not done; nor is an update, `y[i] += v` or a
  scatter-add `y[cell[c, k]] += v`, whose cell holds what it held on entry
  plus every update.
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
  A program's own return annotation is its postcondition, a claim about what
  it leaves in its parameters. It is `tested` on the program's native runs,
  an equation of an output compared at that output's class (`approx` for
  reals), and `decided` by isl from what each call stores where that decides
  it; otherwise Lean is asked to prove it from the same equations and the
  theorems the program cites, unfolding a call at a time (`loopty.unfolding`).
  `examples/hexahedra.py`'s `weak_grad` claims the dense Kronecker operators
  grudge's escape hatch applies, and Lean proves it with Mathlib from what
  the three passes store and `sum_factorization`.
- A program lowers sequentially: its calls' loops run one after another, as
  the program runs them, until a schedule fuses them. A fusion moves the
  outer loops of each side, as many as both have or as `loops` says, and a
  nest the lowering wrote as one domain is cut after them; loops a map
  takes that are not the outer loops of their domain leave the casts
  decided and the kernel unbuildable. Only a pointwise
  producer is substituted (one cell per instance, at its loop variables, no
  sum, no guard isl cannot state). A window is taken of an array its
  statements store a value in, not update, which they read in values,
  before any of their loops is tagged (#213).
  Every read gets the value as the store converted it, a read in a
  subscript included (`y[(int16_t) (t[i])]`), and a conversion inside the
  value stays (`(t[j] + 1) % n` of a 32-bit `Fin[m]` entry, computed in 64
  bits); a store's widening keeps the value, and a read in a subscript reads
  the value inside it (`x[t[i]]`, and `x[n - 1 - i]`, which loopy then reads
  as affine and checks). The
  compiled program is one call, so the contract checks its arguments when it
  starts and not at every call. What a callee's contract checks of the cells
  of an array an earlier call wrote, or the program made (an element sort
  `Fin[m]` or `Nat`, the offsets a ragged family is read through), is a
  `requirement` of the program: decided by isl under the hypotheses that held
  at the call, or, where they do not decide it, checked by the compiled
  program between the two calls, which then stops with the requirement's
  message where the native callee is refused (`examples/travel.py`). An
  integral array such a check reads is stored in 64 bits, as the native run
  holds it, a `Fin[m]` one included, so that a value written outside 32 bits
  is read as written and not as the store narrowed it; the compiled run
  refuses one passed in fewer bits, which the native run would narrow. Offsets a scan computes and a later call
  declares `Nat` without reading rows through them are checked: that they
  are naturals follows from the scan's recurrence only by induction. The
  counts of a ragged family an earlier call wrote are still refused: its rows
  are laid out in a buffer the program is given, which no hypothesis about
  the program's arrays can speak of. The kernels an array is passed to have to declare the same
  element sort for it, and every call has to read a ragged family's rows
  through the same offsets; a loop in the body whose trip count is an
  argument (a host loop) is refused, and so are an array made like a ragged
  one, an array made over a union of pieces, which is stored as one box
  (#208), and an array over a domain of a `Fin[m]` or `Nat` element sort
  that an earlier call wrote and a later one is given, whose cells no
  checked point reads yet (#209). A
  temporary made like a parameter, `Arr.zeros_like(u)`, has `u`'s dtype
  natively, so the compiled program refuses a `u` whose dtype does not hold
  what the compiled temporary holds: `float64` for `Real`, the dtype itself
  for a numpy one such as `np.complex128`, `bool` for `Bool`, `int64` for
  `Nat` and `Int`, and a signed integer of 32 bits or more for `Fin[m]`
  (`int64` for an array a checked point reads). On the C target the executor
  allocates a program's array on the heap and passes it after the program's
  own arguments, `double *__restrict__ f`, where loopy would declare it on
  the stack of the call, whose size bounded it (#62; note 16 in
  `docs/loopy-notes.md`); on OpenCL it is a global temporary, which the
  PyOpenCL host code allocates.
- Only a two-axis (row, fiber) ragged array, its rows counted by an array of
  one axis, is built and lowered; tracing refuses any other ragged type, a
  fiber after two dense axes, a dense axis after the fiber, or counts of two
  axes, before any fact is stated about it.
- Integers are 64 bits wide in both runs, and a result outside 64 bits wraps
  round compiled where numpy wraps or refuses, which is numpy's limit too.
  Arithmetic of loop variables, sizes and literals alone is Python's natively,
  which is exact: `1.0 * i ** 5` past `i = 6208` is a wider number natively
  and wraps compiled, which the differential fact reports (#139).
  A loop bound, and a subscript loopy reads as affine with no division, are
  loopy's index arithmetic, 32 bits wide; under `-fwrapv` such a subscript's
  sums and products give the cell it names. Any other subscript is computed
  in 64 bits, as the rest is, its sums of loop variables and sizes included:
  `x[(i * i) % n]` (#129), `x[(i * 499999) // 1000000]`, which loopy wrote
  with a division of the wrapped `499999 * i`, and `x[(i + n) // 2]`, whose
  sum wrapped round past `2**30` (#149). A sum of loop variables, sizes and
  small literals in a value or a guard on the loops is still 32 bits wide,
  `y[i] = i + n` past `2**31` (#162). A subscript over a narrow numpy integer
  type is computed in it, as numpy computes it: `x[s + i]` of a `uint8` `s`
  wraps round at 256 in both runs (#157). One of a signed type that wraps
  round below zero is refused natively, as any negative index is, and read
  before the array compiled; its in-bounds fact is not decided, since isl
  reads a narrow scalar or element as any integer, and is refuted or
  assumed. A `Fin[m]` array the kernel writes may be an `int32` one
  natively, whose entries the native run reads as `int64`, as the compiled
  run computes with them, unless a program's checked point reads it, which
  takes an `int64` one. An integer to a negative power whose exponent is not
  a literal is left to numpy's refusal natively, and computed as an integer
  compiled. Loop arithmetic beside a numpy integer that does not hold it is
  refused compiled where isl reads it and the call's sizes fix its loops
  (#153); a product of loop variables, a ragged row's loop, a store under a
  guard isl does not read, and a sum's body under both a guard and the
  generator's `if` clause where the guard is false, are left to the native
  refusal and computed compiled (#166). So is a value stored into a signed
  integer array that does not hold it, a `uint64` past `2**63` into `Int`,
  an `Int` into an `np.int8` array, or a real past `2**63`, or a NaN, into
  `Int`, whose value only the compiled kernel knows (#159). The written `-1 *
  u[i]` of an unsigned `u`, which numpy refuses, is read as `-u[i]`, which
  it computes (#168). A numpy scalar on the left of a term is traced as the
  Python number it equals: `np.float64(0.1) * x[i]` as `0.1 * x[i]`, single
  precision beside a `float32`, and `x[np.uint8(200) + i]` as `x[200 + i]`,
  not taken round (#167). Two statements of one kernel whose lowered
  arithmetic differs only by the dtype of an equal literal are generated
  with the first one's literal, since loopy maps a subexpression to the
  result of an equal one, and pymbolic finds a numpy scalar equal to any
  number of its value (#170). See notes 19, 20, 23 and 25 in
  `docs/loopy-notes.md`.
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
- A constant table's values are read in its dtype and have no sort beyond
  it: an index read from a table of integers, `x[perm[i]]` with `perm =
  table([2, 0, 1])`, is not in bounds by type, as one read from an array of
  `Fin[m]` is, and its in-bounds fact is `assumed` (#182). A table is a
  kernel's or a program's own; a table the body reads through a container,
  which no frame names, is named `table`. A table holds finite numbers
  (#193), and a loop bounded by a value read from one is refused when
  traced (#194).
- A kernel called inside another's loops (`loopty.calls`) is given whole
  arrays, rows of dense, ragged and table arrays, the cells an index row
  picks, scalars and functions; a row of an array over a domain is not
  passed (a whole one is), and the cells a gather picks are a copy natively,
  so a called kernel does not write them (#205). Two arguments over one array
  the callee writes have to be two rows of it, decided at every call; a
  whole array and a part of it are refused, and so is an argument that
  reads an array the callee writes, since natively it is read once, when
  the call is made, a value a function given captured included. A rule's claim about its value (`-> Real & (...)`)
  is refused: state it as a theorem about the rule (#206). A call made
  inside a sum over `Fin[...]` of a size, rather than over an array's `.dom`,
  has points the trace does not see, and its facts are `assumed` (#207). A rule that
  takes a function is compiled where it is called, not on its own. A real
  given for a numpy integer sort (`np.int8`) is truncated in both runs, and
  one outside the type converts as C leaves undefined and numpy warns; a
  `uint64` value given for `Fin[m]`, `Nat` or `Int` is refused, since the
  contract refuses one past `int64` (#232). A call that reads one cell of a
  row, `first(A[i])`, leaves `A`'s second size in no loop, which fails to
  compile (#220). A domain
  declared outside an annotation, for `Arr.zeros(TRIANGLE, ...)` in a
  program, names its binders and sizes as lanky variables, `i, j, n =
  Var("i"), Var("j"), Var("n")` (#210). A program's array over a domain
  kept in a window is kept in the box its placement's slots span, `11 x 11`
  for a triangle of 66 cells, as plain loopy keeps one in a temporary.
- `loopty.math` has eight functions (#184), and numpy's own functions on a
  traced value are refused rather than read as them (#195). A function
  only a guard calls is refused when lowered: loopy resolves a function by
  the values that call it (note 20 in `docs/loopy-notes.md`).
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
  read of the row inside such a loop is checked natively against the length
  the write left, which isl cannot read, so its in-bounds fact is
  `assumed` itself, with the reason (`cnt[r] = 0` inside `for j in
  val.dom[r]` makes the native run refuse `val[r, 1]`); a sum or an inner
  loop over the row reads the length again where it starts, and its reads
  stay decided. A program's layout facts are not tried on runs.
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
- An update's cell read from an array is any cell to isl, so a step that runs
  two updates through it at the same time is refused even where no input can
  make them collide (`y[perm[i]] += x[i]` of a permutation `perm`), with a pair
  from the example inputs where they have one and isl's otherwise, which the
  message says: make the updates atomic (#202). An `exact` update,
  `m[cell[c, k]] += 1` of integers, keeps the order of its updates and is
  refused `atomic`, as an `exact` sum is refused a tree, although integer
  addition gives the same bits in any order (#200).
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
- Zeros a read sees follow a loop that stores the array, and not a split
  of one ([#289](https://github.com/xywei/loopty/issues/289)): after
  `split(k.i, 4).tag({k.i.outer: "g.0", k.i.inner: "l.0"})`, a statement
  stores `f[4*k.i.outer + k.i.inner]`, along no loop of its own, so zeros
  a later read sees (padding at a boundary) stay on every work item and
  the tag is refused, saying which read sees them. Zeros no read sees are
  not run at all ([#290](https://github.com/xywei/loopty/issues/290)), so
  the split is legal for every array a program stores before it reads.
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
  term is unchanged, and so is every message, and a `Schedule` step names such
  a reduction by its handle, `split(k.j, 2)`, which reaches the iname
  `Lowering.reduction_inames` lists; the generated code has the fresh one.
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
  [docs/device-runs/](https://github.com/xywei/loopty/tree/main/docs/device-runs).
- CUDA, and targets beyond C and OpenCL.
- The two-level FMM point-to-point shape, which is a dependent sum of dependent
  sums and needs more than the two ragged axes lowering handles.

## Known limits

The open issues you are most likely to meet, and what to do about each. Most
show in `lanky check` or `loopty run` as a `refuted` or `assumed` fact or as
an error, but `loopty run` compares the compiled run with the native one only
on the inputs it runs.

- **A compiled run does not check indices**
  ([#171](https://github.com/xywei/loopty/issues/171)). `loopty run` and
  `LoopyExecutor.run` compile a kernel whatever its in-bounds facts say. Where
  an index leaves its array, the native run raises `IndexError` and the
  compiled run reads or writes outside the buffer, with no error. `lanky
  check` shows where this can happen: the access's row, `x[i + 4] is in bounds
  for every instance of y[i] = x[i + 4] (shift.py:9)`, reads `refuted` or
  `assumed`. Put the access under a condition isl can read, which decides
  the fact: `if i + 4 < x.dom.size:` before `y[i] = x[i + 4]`.
- **A numpy scalar on the left of an operation loses its type**
  ([#167](https://github.com/xywei/loopty/issues/167)). `np.float64(0.1) *
  x[i]` is traced as `0.1 * x[i]`, so beside a `float32` `x` the compiled run
  multiplies in single precision where numpy uses double. The difference is
  within the tolerance `loopty run` allows a `float32` input, so it reads
  `tested`. Put the scalar on the right: `x[i] * np.float64(0.1)`.
- **Loop arithmetic outside a subscript is 32 bits wide compiled**
  ([#162](https://github.com/xywei/loopty/issues/162)). A sum of loop
  variables and sizes used as a value, `y[i] = i + n`, `(i + n) // 2` or
  `1.0 * (i + n)`, wraps round compiled once it passes `2**31`, which takes a
  size past about `2**30`; natively it is exact. `loopty run` reports it only
  on an input that large. Make the first operand a 64-bit integer:
  `y[i] = i * np.int64(1) + n`. A loop bound and a condition on the loops,
  `if i + n > 5:`, stay 32 bits wide: keep their sums below `2**31`.
- **Integers past 64 bits**
  ([#139](https://github.com/xywei/loopty/issues/139)). Arithmetic of loop
  variables, sizes and literals alone is exact natively, as Python's integers
  are, and 64 bits wide compiled: `1.0 * i ** 5` wraps round compiled past
  `i = 6208`, which `loopty run` reports on an input that reaches it. Where
  you want a real, compute in one from the start: `(1.0 * i) ** 5`.
- **Values too large for a narrow numpy integer**
  ([#166](https://github.com/xywei/loopty/issues/166),
  [#159](https://github.com/xywei/loopty/issues/159),
  [#177](https://github.com/xywei/loopty/issues/177)). numpy raises when a
  value does not fit an integer type such as `np.int16`, and loopty refuses a
  compiled run before it starts wherever isl finds such a value at the call's
  sizes. The check misses a product of loop variables, `a[i] + i * i` of an
  `np.int16` `a`, loop arithmetic in a ragged row's loop, and a `uint64` past
  `2**63` stored into an `Int` array: the compiled run wraps these round, and
  `loopty run` raises the native run's error only on an input that reaches
  one. It also refuses one case numpy completes, a sum whose `if` clause
  repeats the statement's guard. In each case, widen the narrow value first,
  `a[i] * np.int64(1) + i * i`, which both runs compute alike, and store a
  `uint64` into an `np.uint64` array.
- **A few spellings fail without naming the fix**
  ([#172](https://github.com/xywei/loopty/issues/172),
  [#173](https://github.com/xywei/loopty/issues/173),
  [#175](https://github.com/xywei/loopty/issues/175)). A sum in a guard,
  `if reduce_sum(x[j] for j in x.dom) > 1:`, fails inside loopy with `name
  already used as iname`: compute the sum into an array in a statement of its
  own and read that array in the condition. A truth value first in arithmetic,
  `True - k[i]`, fails while tracing with Python's `TypeError`: write the
  integer, `1 - k[i]`. `k[i] - -(2**63)` of an `Int` is refused as a literal
  past 64 bits, though numpy computes it: write `k[i] + -(2**63)`, the same
  value in `int64`.
- **A refused program has written its outputs**
  ([#176](https://github.com/xywei/loopty/issues/176)). When a compiled
  program raises `RefusedNatively` for a call that follows one of its checks
  between calls, the whole program has already run: that call and the calls
  after it have written their outputs, where natively the program stopped
  inside that call. After the error, do not trust the arrays, and pass copies
  if you need them as they were.
- **A window keeps values stored, not updated**
  ([#213](https://github.com/xywei/loopty/issues/213),
  [#214](https://github.com/xywei/loopty/issues/214),
  [#229](https://github.com/xywei/loopty/issues/229),
  [#133](https://github.com/xywei/loopty/issues/133)). `Schedule.contract`
  refuses an array a statement updates (`f[j] += v` adds to the zeros a
  window drops: store each cell once, or keep the array whole), and a loop
  tagged with no order before it; an update of another array may read a
  window, and `atomic` does not lift the order of its slots. A cell stored
  twice is asked about at both stores, not only at the last before the
  read, so a window that is safe can be refused there. A window of a size, `f_window[n]`, is one
  for every work item (in global memory on OpenCL), so a loop it drops
  cannot go on a hardware axis after it; a window of a number of cells is
  each work item's own. An array
  kept in a window, and another computed from it where it is read, are
  refused together, in either order: `substitute` moves the reads of the
  window to where the other is read. A substitution through two
  intermediates, one computed from the other, is refused, unless the first
  is computed from arrays nothing in the program writes (grudge's `v.n`
  from the normals, in `examples/advection.py`): substitute one of them.
- **Some facts are isl's alone** ([docs/lean-checks.md](https://github.com/xywei/loopty/blob/main/docs/lean-checks.md#what-stays-isls)).
  Where Lean is installed, a cast through the part of a tile isl computes
  stays `decided isl`: loopty states it for isl only, and `lanky check
  --verbose` or `loopty run --verbose` says so. A cast refused through a substitution's carried read,
  or a window refused at a read in a sum, is refuted by isl with no
  counterexample stated for Lean
  ([#215](https://github.com/xywei/loopty/issues/215)).
- **A declared layout stands on its own**
  ([#216](https://github.com/xywei/loopty/issues/216)). Its buffer is one
  axis, `-> Fin[k]`, and an argument of a kernel does not name the layout it
  is stored through: a kernel reads through the layout's function, and its
  facts do not rest on the layout's.
- **The OpenCL target is not run in CI**
  ([#174](https://github.com/xywei/loopty/issues/174)). `--target opencl`,
  with the `opencl` extra, generates OpenCL and runs it through pyopencl, and
  nothing in CI runs a device. A schedule for `"opencl"` is checked anywhere:
  where pyopencl is not installed it is built on loopy's plain OpenCL target,
  its facts are in the ledger, `loopty run --emit-code` prints the OpenCL C a
  device runs, and `loopty run` says it ran none
  (`examples/l2p.py`'s `on_a_gpu`). The demos were run on two device classes
  by hand, and
  [docs/device-runs.md](https://github.com/xywei/loopty/blob/main/docs/device-runs.md)
  has the commands and the transcripts. Signed integer overflow is undefined
  in OpenCL C, which has no `-fwrapv`, so a signed integer sum that
  overflows, which numpy wraps round, can come out otherwise there: run such
  a kernel on the C target.
- **A theorem's or a postcondition's statement is lanky's reading of it.**
  loopty traces a kernel's body itself and checks that trace, the
  `trace-faithful` fact. A theorem, and the claim after a kernel's `->`, are
  annotations lanky reads. lanky 0.2.0 checks a theorem's reading at drawn
  values, the row `the term computes what the annotations compute` under it,
  and its [Known limits](https://github.com/xywei/lanky#known-limits) say what
  that check can miss. A postcondition has no such row: read its `STATEMENT`
  column in the ledger.

The Partial and Not yet lists under Status have the rest, and the
[open issues](https://github.com/xywei/loopty/issues) have every one.

## Install

This is the `port/loopy-main` branch: loopty on loopy's main branch, whose
domains are namedisl sets, with loopy, islpy and namedisl taken from Git and
`islpy<2026` dropped. It merges into `main` once a loopy release carries
namedisl. Clone it beside lanky and `uv sync --group dev`;
[docs/loopy-main.md](https://github.com/xywei/loopty/blob/port/loopy-main/docs/loopy-main.md)
has the install, what changes and why. What follows describes the release on
PyPI, which runs on loopy 2025.2.

```sh
pip install loopty
```

That is the 0.3.0 release, which has the five demos at the top of this page
([Run it](#run-it)). loopty runs on Python 3.12 and 3.13: the islpy it is held
to has no wheels for 3.14. CI, and every run this README reports, is on Linux.
In a uv project, `uv add loopty`, and the same for the extra below. The
install pulls in:

- [lanky](https://github.com/xywei/lanky) 0.3.0 or later, the proof host:
  `lanky check` prints the ledger, and loopty plugs into it;
- [loopy](https://github.com/inducer/loopy) 2025.2, which generates the code.
  loopty builds on loopy's internals, so a release of loopty is held to the
  loopy release it is tested with;
- [islpy](https://github.com/inducer/islpy) from 2025.2.5 and below 2026,
  since islpy 2026 removed a method loopy 2025.2 calls (note 4 in
  [docs/loopy-notes.md](https://github.com/xywei/loopty/blob/main/docs/loopy-notes.md));
- numpy 2 or later, whose promotion rules (NEP 50) the compiled run follows,
  and pymbolic.

A compiled run, `loopty run` or `LoopyExecutor`, also needs a C compiler,
whatever `gcc` is on the `PATH`: GCC on Linux, and on macOS the `gcc` of
Apple's command-line tools, which is clang. `lanky check` needs none, and
neither does calling a kernel natively; the demos under `python` compile a
schedule as well, so they do.

Device execution is an extra, which installs pyopencl, for
`loopty run FILE --target opencl` and `LoopyExecutor(target="opencl")`. The
device and its OpenCL driver are yours to provide, and CI never runs this
path:

```sh
pip install "loopty[opencl]"
```

lanky's Lean oracle is an extra of *lanky*, so a ledger that says `proved lean`
needs `pip install "lanky[lean]"` and a Lean toolchain in the same environment,
and a theorem about real numbers needs Mathlib too (lanky's README has the
details). The induction steps of a theorem for every order are found by
sympy, lanky's `cas` extra. Without them a fact isl decided reads `decided
isl`, a theorem `tested` or `assumed`, and a `WITHOUT LEAN`, `WITHOUT
MATHLIB` or `WITHOUT SYMPY` line under the table names the rows each piece
would prove and the command that installs it.

The examples and the documentation are in the repository and not in the
package; clone it to follow them.

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
dependence relation. Failure is an `IllegalCast` carrying the witness (by the
statements' ids), its `instances` as the message names them (by the
statements' text and line), and the refuted fact, whose reason is the
exception's message. Casts that change
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

- [docs/quickstart.md](https://github.com/xywei/loopty/blob/main/docs/quickstart.md): the two demos end to end,
  with the output the commands actually print.
- [examples/README.md](https://github.com/xywei/loopty/blob/main/examples/README.md): all eighteen demos, with every
  console block regenerated by `scripts/refresh_example_outputs.py`.
- [docs/device-runs.md](https://github.com/xywei/loopty/blob/main/docs/device-runs.md) and
  [docs/device-runs/](https://github.com/xywei/loopty/tree/main/docs/device-runs): the demos run on real OpenCL
  devices, with the commands and the transcripts.
- [docs/lean-checks.md](https://github.com/xywei/loopty/blob/main/docs/lean-checks.md): how a fact isl
  decides is stated for Lean and proved, what isl hands Lean as hints, what
  stays trusted, and the coverage over the examples.
- [docs/loopy-notes.md](https://github.com/xywei/loopty/blob/main/docs/loopy-notes.md): the loopy and islpy
  interactions that cost debugging time, each with its local workaround and
  why it is local.
- [docs/loopy-main.md](https://github.com/xywei/loopty/blob/port/loopy-main/docs/loopy-main.md): running on loopy's
  main branch, whose domains are namedisl sets: the install, where loopty
  meets namedisl, and every difference from loopy 2025.2.
- [CHANGELOG.md](https://github.com/xywei/loopty/blob/main/CHANGELOG.md).
- [lanky](https://github.com/xywei/lanky): the proof host loopty plugs into, and
  where the ledger, the statuses and the oracle protocol are defined.

## AI disclosure

This project is developed with substantial assistance from AI coding agents
(Anthropic's Claude, via Claude Code). Design, direction, and review are by
Xiaoyu Wei. Generated text and code are reviewed before release, but readers
should assume AI involvement throughout.

## License

MIT. See [LICENSE](https://github.com/xywei/loopty/blob/main/LICENSE).
