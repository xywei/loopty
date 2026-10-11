# isl decides, Lean checks

isl decides every fact loopty states about a kernel and every cast of a
schedule, in milliseconds. isl produces verdicts, not proofs, so a fact it
decided reads `decided`. Where lanky's Lean oracle is installed (`pip install
"lanky[lean]"` and a Lean toolchain, see lanky's README), each of those facts
is also stated as a Lean theorem and proved there, by `omega`, and then reads
`proved`, by `lean`.

That is worth something only because the theorem does not come from isl. It is
built from loopty's own description of the kernel: the loops' names, bounds
and guards as loopty wrote them for isl, the subscripts as the trace recorded
them, the source order as the 2d+1 vectors `loopty.flow` defines, and each
schedule step's maps as the step wrote them. A set isl computed (a dependence
relation, an image, a coalesced or simplified set) is never part of a
statement. So a set isl simplified wrong, or a relation it composed wrong, is
not what Lean proves, and isl's verdict and Lean's proof are two answers to
one question.

## What it looks like

Nothing to turn on. With Lean installed, the stencil's ledger reads:

```console
$ lanky check examples/stencil_skew.py
STATUS  BY           WHERE               OWNER   STATEMENT
------  -----------  ------------------  ------  ------------------------------------------------------
proved  lean         stencil_skew.py:62  jacobi  u[t + 1, i] is in bounds for every instance of S0
proved  lean         stencil_skew.py:62  jacobi  u[t, i - 1] is in bounds for every instance of S0
proved  lean         stencil_skew.py:62  jacobi  u[t, i + 1] is in bounds for every instance of S0
proved  lean         stencil_skew.py:62  jacobi  distinct instances of S0 write distinct cells of u
proved  lean         stencil_skew.py:54  jacobi  the source order runs every dependence forward in time
tested  interpreter  stencil_skew.py:54  jacobi  the traced term computes what the body computes

6 facts: 5 proved, 1 tested

LEAN proved all 5 facts isl decided. Lean's statements are loopty's, from its own description, not isl's results, and isl's hints were checked, not trusted. --show-lean prints the Lean.
```

and `loopty run` proves the casts of every schedule it runs:

```console
$ loopty run examples/stencil_skew.py
...
proved  lean   stencil_skew.py:62  jacobi  skew(i, by='t') renames the instances of jacobi one for one
proved  lean   stencil_skew.py:62  jacobi  the order after skew(i, by='t') runs every dependence of jacobi forward
proved  lean   stencil_skew.py:62  jacobi  tile(t,i,8,8) renames the instances of jacobi one for one
proved  lean   stencil_skew.py:62  jacobi  the order after tile(t,i,8,8) runs every dependence of jacobi forward
tested  loopy  stencil_skew.py:62  jacobi  the scheduled run of jacobi agrees with the native run to the accurac...

5 facts: 4 proved, 1 tested

LEAN proved all 4 facts isl decided. Lean's statements are loopty's, from its own description, not isl's results, and isl's hints were checked, not trusted. --show-lean prints the Lean.
```

(These transcripts are from a machine with Lean; CI runs without it, so the
documents the refresh script checks show `decided isl`.)

The Lean is one flag away, and nowhere else. `--show-lean` prints it for every
row that has some, or for the rows whose owner, id or statement contains the
text it is given:

```console
$ lanky check examples/stencil_skew.py --show-lean "u[t, i - 1]"
...
LEAN jacobi at stencil_skew.py:62: u[t, i - 1] is in bounds for every instance of S0
  proved by Lean (omega)
    set_option autoImplicit false in
    theorem Lanky.jacobi_in_bounds
        (nt : Int)
        (nx : Int)
        (t : Int)
        (i : Int)
        (h0 : nt ≥ 0)
        (h1 : nx ≥ 0)
        (h2 : t + 1 < nt)
        (h3 : i > 0)
        (h4 : i + 1 < nx)
        (h5 : 0 ≤ t)
        (h6 : t < nt)
        (h7 : 0 ≤ i)
        (h8 : i < nx)
        : 0 ≤ t ∧ t < nt ∧ 0 ≤ i - 1 ∧ i - 1 < nx := by
      omega
```

`h2` to `h4` are the guard the kernel wrote, `with when((t + 1 < steps.size) &
(i > 0) & (i + 1 < row.size))`, and `h5` to `h8` its two loops. The file the
source is part of replays with `lean` alone. `loopty run --show-lean` does the
same for the casts.

A refuted fact is refuted by isl, with isl's counterexample, and Lean then
proves the claim false there:

```console
$ lanky check tests/kernels/out_of_bounds.py
...
REFUTED shift at out_of_bounds.py:24: u[i + 1] is in bounds for every instance of S0
  witness: (1,)
  cells u[i + 1] reaches are cells u has, except [a0=1] at [n=1]
  Lean confirmed the counterexample: the claim is false at n=1, i=0
```

A refuted fact is never turned into a proof. Lean checks isl's
counterexample, and a fact isl refuted with no counterexample Lean can check
(a refused bijection, say) is not offered to Lean at all.

A fact isl decided and Lean did not prove keeps `decided`, and a `NOT PROVED`
line under the table names it, the parts Lean did not close, and whether Lean
ran out of time or found no proof; `--show-lean` shows what Lean said. With
`--verbose` (`lanky check`'s or `loopty run`'s), so does a fact loopty states
for isl alone. Lean's own words (`omega`, `decide`, its messages) appear only
under `--show-lean`.

## What each fact is, for Lean

Every fact is a universal over the sizes, the arrays a bound or a subscript
reads (a function of its indices, so `cnt[r]` is `cnt r`, an atom for `omega`
as its isl parameter is one for isl), and the loops; its hypotheses are the
domains; and its goal is the claim. A fact is stated in parts, each proved on
its own, in parallel.

| fact | one part per | the goal |
|---|---|---|
| in bounds | place the access is made over (a statement's loops, a guard's, a sum's) | the cell is a cell of the array, of its box, its ragged row or its polyhedral domain |
| in bounds, mixed subscripts | access | as above, with each index read from an array of a `Fin` sort assumed in that sort, which the type says |
| distinct writes | statement | two instances that differ write cells that differ |
| element sort | statement | the value written is a point of the sort |
| definedness | read | some statement wrote the cell read, before the read: a disjunction of existentials, one per writer; a write under a guard isl cannot state is no writer, since its domain is wider than where it stores |
| the source order | pair of accesses | the source runs the pair in the order the dependence says, which is its definition |
| a cast is bijective | statement the step moves | every instance has an image (an existential), one only, and no two share one |
| a cast is monotone | dependence class, a pair of accesses | two instances that touch one cell, in the source's order, keep it after every step: their time vectors compare as the source's do |
| what a program's call may assume | requirement, or flat access | the claim at every point of its domain, under the hypotheses isl used (a callee's postcondition, a theorem the program cites, a type the contract checks) |
| an assumption of a schedule | (a hypothesis of every later cast) | |

A class two statements never form, `S1` before `S0` in a program whose `S0`
runs first, has a hypothesis that is `False` as written, and is closed by
`contradiction`: that a class is empty is isl's verdict too, and Lean proves
it again.

## What isl gives Lean, which Lean checks

`omega` decides linear integer arithmetic with floor division and remainders
by numbers. What it does not do, isl does, and hands over as a hint. A hint
carries no trust: a wrong one leaves a goal `omega` cannot close, and the fact
stays `decided`.

- **The witness of an existential.** A step's map, isl's functional form of it
  (`PwMultiAff`), proves the step total: the tile's `t_outer = t // 8` and
  `t_inner = t % 8`. A read's writer, isl's `lexmax` of the instances that
  wrote its cell before it, in pieces, proves it defined: the scan's `off[r]`
  is `off[0]`'s write where `r = 0` and the row before's elsewhere, a case
  split Lean checks piece by piece.
- **Instances of a universal hypothesis.** `omega` never instantiates `∀ r,
  off (r + 1) = off r + cnt r`; the points isl's discharge took it at
  (`loopty.hypotheses`) are given as `have` lines, with an equation between
  any cell they read and the claim's that is the same cell written otherwise
  (`off (q - 1 + 1) = off q`).
- **Case splits,** for what `omega` lacks: it has no dark or grey shadows, and
  a split by the rational bounds isl knows rescues it. None of the examples
  needs one.
- **The counterexample of a refuted fact,** which Lean proves false by
  `decide`, evaluating the statement at the point in the kernel.

## What stays trusted

- The Lean kernel and `omega`'s soundness.
- The printer: loopty's description read back as lanky terms, and lanky's
  terms printed as Lean. A bug there could weaken a statement, so it is tested
  against three safeguards (`tests/test_lean_checks.py`):
  - *non-vacuity*: for every part of every fact of every example and test
    kernel, at a point isl samples, Lean proves every hypothesis of the part,
    an array given as the table of the values isl sampled, so the hypotheses
    as Lean reads them hold somewhere;
  - *mutations*: six variants of the stencil (subscripts moved, a guard
    changed, a write in place) under seven schedules (tiles, a skew, an
    interchange, a split, reversals, a wavefront), each cast stated as if isl
    had found it legal: Lean fails on exactly the dependence classes isl finds
    reversed, and on none of a legal one. Every cast the examples refuse fails
    in Lean on the class isl's pair is in, and a kernel's subscripts
    perturbed, its guard dropped or an index moved, are proved where isl
    decides them and not where isl refutes them;
  - *a differential*: every domain of every example and test kernel, and the
    cells every in-bounds fact's goal is stated over, read back from loopty's
    text, hold at exactly the points of a box where isl's set does, and
    Lean's `decide` of the printed predicate agrees with isl at points in and
    out of it.
- loopty's collectors, which isl and Lean share: the accesses a statement
  makes, the trace (the `trace-faithful` fact checks it), the source order.
- The gap between the model and the code: lowering, loopy and the compiler,
  which the native and compiled differential covers, by sampling.

## Cost

Each part is a declaration Lean elaborates, a few milliseconds to a second.
The parts of a fact are checked at once (`LANKY_LEAN_JOBS`, up to four
sessions), within the fact's budget (`LANKY_LEAN_BUDGET`), and a part that
runs out of time leaves the fact `decided`, saying so. Every answer of Lean's
is kept, by the hash of the Lean version and the declaration
(`LANKY_LEAN_PROOF_CACHE`, `off` for none), so a file checked again reads them
back: the stencil's ledger takes 2.3 s the first time on the machine the
coverage report ran on, and 1.4 s the next, against 0.6 s without Lean.
`LANKY_LEAN_DISABLE=1` turns Lean off, and the ledger reads `decided` as before.

## Coverage

`scripts/lean_coverage.py` offers every fact isl decides about every example
and test kernel to Lean, the casts each file's `rejected*` functions refuse
included, from an empty proof cache. Its report, with Lean v4.29.1 (a
file's time is Lean's on all its facts, a part's on one declaration):

Of 523 facts isl decides in 16 files, Lean proved 499; 9 stay decided; 15 refuted facts have their negation proved at isl's counterexample. 260 more facts are not isl's to decide (by type, tested, assumed).

Not all of them are equally hard. The 81 `ordering` facts, that the source
order runs every dependence forward, hold by the definition of a dependence
(two instances that touch one cell, which the source runs in that order), and
Lean's proof of them checks only that the printer states the source order the
same way twice. The others are claims isl could get wrong.

| file | isl facts | proved | decided, not proved | refuted, negation proved | Lean time |
|---|---:|---:|---:|---:|---:|
| `examples/composition.py` | 10 | 10 | 0 | 0 | 1.6 s |
| `examples/fusion.py` | 16 | 12 | 3 | 1 | 0.6 s |
| `examples/p2p.py` | 19 | 19 | 0 | 0 | 2.2 s |
| `examples/pairs.py` | 14 | 14 | 0 | 0 | 0.9 s |
| `examples/reshape_layouts.py` | 16 | 16 | 0 | 0 | 1.7 s |
| `examples/spmv.py` | 18 | 18 | 0 | 0 | 0.9 s |
| `examples/stencil_skew.py` | 10 | 9 | 0 | 1 | 2.4 s |
| `examples/travel.py` | 47 | 45 | 2 | 0 | 3.7 s |
| `examples/wavefront_acoustic.py` | 24 | 21 | 0 | 3 | 9.8 s |
| `tests/kernels/axpy.py` | 4 | 4 | 0 | 0 | 0.1 s |
| `tests/kernels/domains.py` | 90 | 89 | 0 | 1 | 6.0 s |
| `tests/kernels/out_of_bounds.py` | 4 | 3 | 0 | 1 | 0.2 s |
| `tests/kernels/spmv_min.py` | 14 | 14 | 0 | 0 | 0.0 s |
| `tests/kernels/stencil.py` | 5 | 5 | 0 | 0 | 0.0 s |
| `tests/kernels/travel.py` | 64 | 61 | 2 | 1 | 2.5 s |
| `tests/kernels/travel_wrong.py` | 168 | 159 | 2 | 7 | 9.6 s |

By kind:

| kind | isl facts | proved | decided, not proved | refuted, negation proved | parts | median s per part | slowest part |
|---|---:|---:|---:|---:|---:|---:|---:|
| bijective | 15 | 14 | 1 | 0 | 58 | 0.129 | 0.49 s |
| definedness | 19 | 18 | 1 | 0 | 22 | 0.086 | 0.18 s |
| disjoint-writes | 108 | 108 | 0 | 0 | 108 | 0.023 | 0.15 s |
| element-sort | 28 | 20 | 0 | 8 | 20 | 0.058 | 0.09 s |
| in-bounds | 230 | 228 | 0 | 2 | 245 | 0.053 | 1.12 s |
| layout | 6 | 0 | 6 | 0 | 0 | - | - |
| monotone | 20 | 14 | 1 | 5 | 115 | 0.199 | 0.61 s |
| ordering | 81 | 81 | 0 | 0 | 226 | 0.023 | 0.14 s |
| requirement | 16 | 16 | 0 | 0 | 16 | 0.146 | 0.43 s |

Every fact Lean did not prove, and why:

- `examples/fusion.py`, burgers_rhs: every cell of f that burgers_rhs reads, flux.S0 has stored by the time it is read. loopty states this kind of fact for isl only, from sets isl computed, and not yet from its own description of the kernel
- `examples/fusion.py`, burgers_rhs: substitute('f') renames the instances of burgers_rhs one for one. a substitution before this step changed what the instances read, and the dependences the step is checked against are isl's, not loopty's description
- `examples/fusion.py`, burgers_rhs: the order after substitute('f') runs every dependence of burgers_rhs forward. a substitution before this step changed what the instances read, and the dependences the step is checked against are isl's, not loopty's description
- `examples/travel.py`, through: the rows of val stay inside their buffers and apart while scan.S0 and scan.S1 write off. loopty states this kind of fact for isl only, from sets isl computed, and not yet from its own description of the kernel
- `examples/travel.py`, flat: the rows of wt stay inside their buffers and apart while scan_flat.S0 and scan_flat.S1 write off. loopty states this kind of fact for isl only, from sets isl computed, and not yet from its own description of the kernel
- `tests/kernels/travel.py`, through: the rows of val stay inside their buffers and apart while scan.S0 and scan.S1 write off. loopty states this kind of fact for isl only, from sets isl computed, and not yet from its own description of the kernel
- `tests/kernels/travel.py`, flat: the rows of wt stay inside their buffers and apart while scan_csr.S0 and scan_csr.S1 write off. loopty states this kind of fact for isl only, from sets isl computed, and not yet from its own description of the kernel
- `tests/kernels/travel_wrong.py`, two_buffers: the rows of wt stay inside their buffers and apart while scan_flat.S0 and scan_flat.S1 write off. loopty states this kind of fact for isl only, from sets isl computed, and not yet from its own description of the kernel
- `tests/kernels/travel_wrong.py`, two_buffers: the rows of wt2 stay inside their buffers and apart while scan_flat@2.S0 and scan_flat@2.S1 write off2. loopty states this kind of fact for isl only, from sets isl computed, and not yet from its own description of the kernel

## What stays isl's

Facts loopty states for isl alone, from sets isl computes, which Lean is not
asked about; each says so with `--verbose`, and the coverage report lists them.

- **A layout a kernel rewrites** (`the rows of val stay inside their buffers
  and apart`): an induction over the run, decided by isl over the cells a
  write reads; issue [#186](https://github.com/xywei/loopty/issues/186).
- **A substitution's definedness, and the casts after it:** a substitution
  changes what the instances read, and the dependences its definedness and the
  later steps are checked against are isl's; issue
  [#187](https://github.com/xywei/loopty/issues/187).
- **Work items:** a cast that puts a loop on a hardware axis also claims that
  no dependence joins two work items, which is stated for isl only; the cast
  is then left `decided` as a whole; none of the examples has one; issue
  [#188](https://github.com/xywei/loopty/issues/188).
- **A part of a tile isl computed:** a statement in one of two tiled loops is
  split by the part of the tile's map over its loop, which isl computes; a
  cast with such a step is left `decided`; none of the examples has one;
  issue [#189](https://github.com/xywei/loopty/issues/189).
