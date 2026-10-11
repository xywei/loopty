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
(a refused bijection, say) is not offered to Lean at all. A write through a
cell read from an array, `y[perm[i]] = x[i]`, is a write of any cell to isl,
so its distinct writes are not isl's to decide: the kernel's native runs
decide them, refuting the fact with two instances that write one cell on an
input, and Lean is not asked.

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
| definedness of a substitution or a window | read of the array | the producer, and only it, stored the cell read before the read, in the program's order |
| no read sees its slot overwritten (a window) | read of the array | three instances, the read, the producer's that stored the cell it reads, and another of the producer's that stored a different cell into the same slot after it: that one does not run before the read, in the order the steps leave; the slot is the placement, `j % 3` |
| a rewritten layout, by the row starts | statement that writes the offsets | where the offsets start at 0 and step by the counts, as the contract checks on entry, the value written is the value the cell holds: the induction step, its base case the contract's check |
| a rewritten layout, by induction | statement that writes the layout | the rows in order before the write imply them in order after it, at the cells the write reads |
| a declared layout | piece, or pair of pieces | every cell's address is in the buffer; two cells at one address are one cell; and every address of the buffer has a cell (an existential) |
| two layouts the same | pair of pieces | a cell of both is stored at one address by each, and a cell of one is a cell of the other |
| the source order | pair of accesses | the source runs the pair in the order the dependence says, which is its definition |
| a cast is bijective | statement the step moves | every instance has an image (an existential), one only, and no two share one |
| a cast is monotone | dependence class, a pair of accesses (two updates of one cell, `y[cell[c, k]] += v`, are none: either order gives the sum) | two instances that touch one cell, in the source's order, keep it after every step: their time vectors compare as the source's do |
| a cast after a substitution | class of a read it computes again and a write of what it reads | the instance of the producer that stored the cell read, before or after the write in the source's order, puts the read on that side of the write after every step |
| a cast after a window | pair of accesses to one slot | two instances that touch one slot, in the order the window was taken in, keep it after every step |
| what a program's call may assume | requirement, or flat access | the claim at every point of its domain, under the hypotheses isl used (a callee's postcondition, a theorem the program cites, a type the contract checks) |
| a row passed to a kernel called in a loop, or a cell read for a scalar it is passed, is in bounds | row, or cell | the row's indices are a point of the array's leading axes, at every point of the call |
| an integral argument of such a call is in its sort | argument | `0 ≤ v`, and `v < m` for `Fin[m]`, at every point of the call |
| two rows of one array passed to one call, one of them written, are apart | pair of rows | the rows' indices differ on an axis both fix, at every point of the call |
| single assignment | pair of statements that store into the array; read of the array and statement that stores into it | two instances, distinct where they are of one statement, store into cells that differ; an instance that stores the cell an instance reads runs before it, in the source order, and so is not the instance that reads it |
| a kernel's postcondition | conjunct | the conjunct at every point of its binders' domain, under what each statement stores, its value equation, with the instances isl took of the equations as hints; one about reals is stated with Mathlib's `ℝ` and closed by `linarith` |
| a cited theorem, applied | (one part) | the theorem as it was proved, and the conjuncts of the postcondition it was matched with, imply the claim: Lean applies the theorem with the arrays and sizes loopty bound given by name, and each premise left is a hypothesis |
| an assumption of a schedule | (a hypothesis of every later cast) | |

The points of a call of a kernel inside another kernel's loop
(`loopty.calls`) are the caller's loops around it and the binders of the
sums it is made inside, cut by the guards around it, as loopty wrote them
for isl; the facts of the callee's body are the caller's own, inlined, and
are stated as any statement's are. `matvec` calls `dot(A[i], x)` at every
`i`, and owes the row it passes:

```console
$ lanky check examples/calls.py --show-lean "the row passed to dot as a"
...
LEAN matvec at calls.py:100: A[i], the row passed to dot as a, is in bounds wherever dot is called at calls.py:100
  proved by Lean (omega)
    set_option autoImplicit false in
    theorem Lanky.matvec_in_bounds
        (m : Int)
        (i : Int)
        (h0 : m ≥ 0)
        (h1 : 0 ≤ i)
        (h2 : i < m)
        : 0 ≤ i ∧ i < m := by
      omega
```

A class two statements never form, `S1` before `S0` in a program whose `S0`
runs first, has a hypothesis that is `False` as written, and is closed by
`contradiction`: that a class is empty is isl's verdict too, and Lean proves
it again.

A kernel's postcondition is decided from what its statements store
(`loopty.dataflow`): where no cell of an array is stored twice or read before
its store, what a statement stores is what the cell holds when the kernel
returns, so each statement is an equation about the arrays the kernel leaves,
`∀ r, 0 ≤ r → r < n → off (r + 1) = off r + cnt r`. Lean proves the
postcondition from those equations, and the `single-assignment` fact that
licenses them is a fact of its own, which Lean proves too:

```console
$ lanky check examples/spmv.py --show-lean "off[r + 1] == off[r] + cnt[r])"
...
proved   proved     lean         spmv.py:79   scan           no cell of off is stored twice, or read before its store
proved   proved     lean         spmv.py:69   scan           off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
...
LEAN scan at spmv.py:69: off[0] == 0 and (forall r in Fin(n). off[r + 1] == off[r] + cnt[r])
  proved by Lean (omega; omega, with the hypotheses taken at 1 point)
    set_option autoImplicit false in
    theorem Lanky.scan_conjunct_0 (off : Int → Int) (h0 : off 0 = 0) : off 0 = 0 := by
      omega

    set_option autoImplicit false in
    theorem Lanky.scan_conjunct_1
        (n : Int)
        (cnt : Int → Int)
        (off : Int → Int)
        (r : Int)
        (h0 : n ≥ 0)
        (h1 : 0 ≤ r)
        (h2 : r < n)
        (h3 : ∀ r_0 : Int, 0 ≤ r_0 → r_0 < n → n ≥ 0 → off (r_0 + 1) = off r_0 + cnt r_0)
        : off (r + 1) = off r + cnt r := by
      have hi0 := h3 r
      omega
```

`h3` is `S1`'s value equation: what it stores at `r` is `off[r] + cnt[r]`,
and isl's discharge took it at `r`, the hint `have hi0 := h3 r`.

That the equations are what the kernel does is the reading of a kernel's
source order loopty and isl share (below), and the native runs test the
postcondition besides. A claim about reals is decided by isl with each
equation it cannot read an atom, the same equation at the same point being
one atom, and stated for Lean with Mathlib's reals (`LANKY_LEAN_MATHLIB`),
where `linarith` closes it; core Lean has no reals, and says so.

A theorem a kernel cites, `@kernel(uses=[on_the_triangle])`, is applied to its
postcondition: Lean is handed the theorem as it was proved, as a hypothesis,
and applies it to the claim with the arrays loopty matched to its families,
`apply h1 (p := p) (c := c) (R := R)`, so that a wrong match is a part Lean
does not prove.

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
- **The inverse of a layout,** isl's `lexmin` of it in pieces, with `//` and
  `%` by numbers, as the witness that every address of the buffer has a
  cell: for sumpy's storage index, the centre at address 0, and at an
  address `d >= 1` the cell `a = d + 1 - 2 * ((d + 1) // 2)`, `b = 3 * ((d +
  1) // 2) - d - 1`, which is `(d + 1) % 2` and `(d + 1) // 2 - a`.
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
  makes, the trace (the `trace-faithful` fact checks it), the source order,
  and the value equation of a statement whose array is stored once per cell,
  that what it stores is what the kernel leaves, which follows from the
  source order and the `single-assignment` fact Lean proves.
- An equation of reals is decided as the kernel computes it, in floats, each
  equation an atom, and stated for Lean over `ℝ`, where the same equation at
  the same point is the same proposition too. The two readings part where a
  value is not equal to itself: a `NaN` a kernel stores, or is given, so a
  postcondition decided `R[a, b] == c[a, b]` is false natively at a `NaN` of
  `c` (#227). A theorem a kernel cites is about the reals, and is given as
  hypotheses only the conjuncts in which nothing rounds, copies and sign
  changes, whose two readings agree: `s[i] == a[i] + b[i]` holds of the
  floats a kernel stores, and not of them read as reals.
- The gap between the model and the code: lowering, loopy and the compiler,
  which the native and compiled differential covers, by sampling.

## Cost

Each part is a declaration Lean elaborates, a few milliseconds to a second.
The parts of a fact are checked at once (`LANKY_LEAN_JOBS`, up to four
sessions), within the fact's budget (`LANKY_LEAN_BUDGET`), and a part that
runs out of time leaves the fact `decided`, saying so. Every answer of Lean's
is kept, by the hash of the Lean version and the declaration
(`LANKY_LEAN_PROOF_CACHE`, `off` for none), so a file checked again reads them
back: the stencil's ledger takes 2.5 s the first time on the machine the
coverage report ran on, and 1.3 s the next, against 0.6 s without Lean; the
slowest example, `travel.py`, 8.5 s, 4.4 s and 2.1 s.
A check whose every answer is in the cache starts no Lean at all (lanky
#118), a lanky theorem's attempts and the parts about reals included: the
cache is keyed by the version a session would run, and with Mathlib by the
project's toolchain and Mathlib revision, which lanky reads without starting
one, so a warm check costs what a check without Lean does. `LANKY_LEAN_DISABLE=1` turns Lean off, and the ledger reads `decided` as
before.

## Coverage

`scripts/lean_coverage.py` offers every fact isl decides about every example
and test kernel to Lean, the casts each file's `rejected*` functions refuse
included, from an empty proof cache. Its report, with Lean v4.29.1 (a
file's time is Lean's on all its facts, a part's on one declaration):

Of 1183 facts isl decides in 27 files (4 of them a theorem a kernel cites, applied to what isl decided), Lean proved 1142; 12 stay decided; 29 refuted facts have their negation proved at isl's counterexample. 421 more facts are not isl's to decide (by type, tested, assumed).

Not all of them are equally hard. The 161 `ordering` facts, that the source
order runs every dependence forward, hold by the definition of a dependence
(two instances that touch one cell, which the source runs in that order), and
Lean's proof of them checks only that the printer states the source order the
same way twice. The others are claims isl could get wrong. The first file's
time includes starting Lean.

| file | isl facts | proved | decided, not proved | refuted, negation proved | Lean time |
|---|---:|---:|---:|---:|---:|
| `examples/assembly.py` | 37 | 37 | 0 | 0 | 4.0 s |
| `examples/calls.py` | 44 | 44 | 0 | 0 | 1.2 s |
| `examples/composition.py` | 10 | 10 | 0 | 0 | 0.3 s |
| `examples/fusion.py` | 19 | 17 | 0 | 2 | 0.6 s |
| `examples/layer_potential.py` | 15 | 15 | 0 | 0 | 1.0 s |
| `examples/p2p.py` | 19 | 19 | 0 | 0 | 1.5 s |
| `examples/pairs.py` | 14 | 14 | 0 | 0 | 0.5 s |
| `examples/quadrature.py` | 12 | 12 | 0 | 0 | 0.7 s |
| `examples/reshape_layouts.py` | 16 | 16 | 0 | 0 | 0.9 s |
| `examples/spmv.py` | 20 | 20 | 0 | 0 | 0.5 s |
| `examples/stencil_skew.py` | 10 | 9 | 0 | 1 | 1.7 s |
| `examples/storage_layouts.py` | 18 | 18 | 0 | 0 | 1.9 s |
| `examples/sum_factorization.py` | 16 | 15 | 0 | 1 | 1.4 s |
| `examples/taylor_reconstruction.py` | 14 | 11 | 2 | 1 | 1.6 s |
| `examples/travel.py` | 53 | 53 | 0 | 0 | 2.8 s |
| `examples/wavefront_acoustic.py` | 24 | 21 | 0 | 3 | 7.7 s |
| `tests/kernels/axpy.py` | 4 | 4 | 0 | 0 | 0.0 s |
| `tests/kernels/calls.py` | 151 | 144 | 0 | 7 | 4.3 s |
| `tests/kernels/dataflow.py` | 219 | 209 | 10 | 0 | 7.8 s |
| `tests/kernels/domains.py` | 90 | 89 | 0 | 1 | 2.7 s |
| `tests/kernels/out_of_bounds.py` | 4 | 3 | 0 | 1 | 0.1 s |
| `tests/kernels/program_domains.py` | 15 | 15 | 0 | 0 | 0.5 s |
| `tests/kernels/spmv_min.py` | 16 | 16 | 0 | 0 | 0.1 s |
| `tests/kernels/stencil.py` | 5 | 5 | 0 | 0 | 0.0 s |
| `tests/kernels/travel.py` | 70 | 69 | 0 | 1 | 1.4 s |
| `tests/kernels/travel_wrong.py` | 194 | 187 | 0 | 7 | 19.4 s |
| `tests/kernels/windows.py` | 74 | 70 | 0 | 4 | 6.4 s |

By kind:

| kind | isl facts | proved | decided, not proved | refuted, negation proved | parts | median s per part | slowest part |
|---|---:|---:|---:|---:|---:|---:|---:|
| argument | 2 | 1 | 0 | 1 | 1 | 0.024 | 0.02 s |
| bijective | 40 | 40 | 0 | 0 | 164 | 0.068 | 0.49 s |
| by-theorem | 4 | 1 | 3 | 0 | 1 | 0.111 | 0.11 s |
| definedness | 53 | 53 | 0 | 0 | 79 | 0.046 | 0.25 s |
| disjoint-writes | 196 | 195 | 0 | 1 | 195 | 0.011 | 0.09 s |
| distinct-arguments | 2 | 1 | 0 | 1 | 1 | 0.016 | 0.02 s |
| element-sort | 28 | 20 | 0 | 8 | 20 | 0.026 | 0.08 s |
| in-bounds | 521 | 516 | 0 | 5 | 577 | 0.027 | 1.07 s |
| layout | 6 | 6 | 0 | 0 | 12 | 0.086 | 11.91 s |
| layout-in-bounds | 4 | 4 | 0 | 0 | 5 | 0.046 | 0.11 s |
| layout-one-for-one | 4 | 4 | 0 | 0 | 6 | 0.075 | 0.24 s |
| layout-onto | 4 | 4 | 0 | 0 | 4 | 0.126 | 0.27 s |
| layout-same | 1 | 1 | 0 | 0 | 1 | 0.071 | 0.07 s |
| monotone | 49 | 40 | 0 | 9 | 311 | 0.067 | 0.63 s |
| no-clobber | 14 | 10 | 0 | 4 | 21 | 0.129 | 0.21 s |
| ordering | 161 | 161 | 0 | 0 | 416 | 0.013 | 0.08 s |
| postcondition | 39 | 30 | 9 | 0 | 48 | 0.057 | 0.13 s |
| requirement | 16 | 16 | 0 | 0 | 16 | 0.079 | 0.25 s |
| single-assignment | 39 | 39 | 0 | 0 | 111 | 0.010 | 0.06 s |

Every fact Lean did not prove, and why:

- `examples/taylor_reconstruction.py`, reconstruct: (forall a in Fin(p + 1), b in Fin(p + 1) where a < 2 and a + b <= p. R[a, b] == c[a, b]) and (forall a in Fin(p + 1), b in Fin(p + 1) where a + b + 2 <= p. R[a + 2, b] == -R[a, b + 2]). the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `examples/taylor_reconstruction.py`, reconstruct: by on_the_triangle, for every D with pde and stored: forall a in Fin(p + 1), b in Fin(p + 1) where a + b <= p. R[a, b] == D(a)(b). the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `tests/kernels/dataflow.py`, triangle: (forall a in Fin(p + 1), b in Fin(p + 1) where a < 2 and a + b <= p. R[a, b] == c[a, b]) and (forall a in Fin(p + 1), b in Fin(p + 1) where a + b + 2 <= p. R[a + 2, b] == -R[a, b + 2]). the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `tests/kernels/dataflow.py`, reconstruct: (forall a in Fin(p + 1), b in Fin(p + 1) where a < 2 and a + b <= p. R[a, b] == c[a, b]) and (forall a in Fin(p + 1), b in Fin(p + 1) where a + b + 2 <= p. R[a + 2, b] == -R[a, b + 2]). the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `tests/kernels/dataflow.py`, reconstruct: by on_the_triangle, for every D with pde and stored: forall a in Fin(p + 1), b in Fin(p + 1) where a + b <= p. R[a, b] == D(a)(b). the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `tests/kernels/dataflow.py`, off_by_one: (forall a in Fin(p + 1), b in Fin(p + 1) where a < 2 and a + b <= p. R[a, b] == c[a, b]) and (forall a in Fin(p + 1), b in Fin(p + 1) where a + b + 2 <= p. R[a + 2, b] == -R[a, b + 1]). the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `tests/kernels/dataflow.py`, scaled: forall i in Fin(n). R[i] == c[i]*0.1. the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `tests/kernels/dataflow.py`, summed: forall i in Fin(n). s[i] == a[i] + b[i]. the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `tests/kernels/dataflow.py`, fill: forall i in Fin(n). R[i] == s. the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `tests/kernels/dataflow.py`, copy64: forall i in Fin(n). R[i] == c[i]. the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `tests/kernels/dataflow.py`, copy64: by copied: forall i in Fin(n). -R[i] == -c[i]. the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it
- `tests/kernels/dataflow.py`, ones: forall i in Fin(n). R[i] == c[i]. the claim is about real numbers, which core Lean does not have; where Mathlib is (LANKY_LEAN_MATHLIB), Lean checks it

Lean proved every other fact isl decides: every `single-assignment` fact and
every postcondition over integers, a window's `no-clobber`, a substitution's
`definedness` and the casts after it, a rewritten layout's facts and a
declared layout's, and the casts of an assembly whose updates of one node
need no order, among them. The 12 it did not are the claims about reals
above, which core Lean has no words for. The 29 refuted facts are the casts
and windows the files refuse on purpose (a window of two cells of the Burgers
flux, one of three for a reader three cells back, a step after the window
past the slot it reads, the loop run backwards after it, the reconstruction's
interchange) and the claims the test kernels make false on purpose, and Lean
proved each false at isl's counterexample.

With Mathlib (`LANKY_LEAN_MATHLIB`), the twelve claims about reals,
postconditions and the theorems applied to them, are proved too: 1154 proved,
none left decided, and the same 29 refuted facts proved false. The slowest
part about reals is a theorem's application, 0.4 s, and a postcondition's,
0.4 s.

## What stays isl's

Facts loopty states for isl alone, from sets isl computes, which Lean is not
asked about, and claims core Lean has no words for; each says so with
`--verbose`, and the coverage report lists them.

- **Work items:** a cast that puts a loop on a hardware axis also claims that
  no dependence joins two work items, which is stated for isl only; the cast
  is then left `decided` as a whole; none of the examples has one; issue
  [#188](https://github.com/xywei/loopty/issues/188).
- **A claim about reals, in core Lean:** a kernel's postcondition over a
  `Real` array, and a theorem applied to it, are stated with Mathlib's `ℝ`;
  without Mathlib they stay `decided`, saying why, and with it Lean proves
  them. Issue [#227](https://github.com/xywei/loopty/issues/227) is what an
  equation of floats means at a `NaN`.
- **Two updates of one cell at the same time:** a step that runs two updates
  of one cell of `y` on two work items or in no order, without
  `atomic("y")`, is refused with a `disjoint-writes` fact of the step, which
  is read off the step's order and its work items, as a crossing is, and is
  stated for isl only; where the cell is read from an array the pair is read
  off an example input, and the fact is the native run's, not isl's; issue
  [#201](https://github.com/xywei/loopty/issues/201). The casts of a schedule
  with updates are stated for Lean without the classes of two updates of one
  cell, which need no order; `examples/assembly.py` has them all proved, and
  so has `tests/kernels/windows.py` a split after a window whose reader is a
  scatter-add, `y[node[i]] += ...`: the window's classes, and none of two
  updates.
- **A part of a tile isl computed:** a statement in one of two tiled loops is
  split by the part of the tile's map over its loop, which isl computes; a
  cast with such a step is left `decided`; none of the examples has one;
  issue [#189](https://github.com/xywei/loopty/issues/189).
- **Some counterexamples:** a cast refused on a class a substitution carries
  over, or a window refused at a read in a sum, is refuted by isl with
  instances that do not give every variable of the part (the producer's
  instance, a sum's binders), so Lean is not asked; issue
  [#215](https://github.com/xywei/loopty/issues/215).
