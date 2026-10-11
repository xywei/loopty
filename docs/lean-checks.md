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
  Lean proved the claim false at u[i + 1] at n=1, i=0, by decide
```

A fact isl decided and Lean did not prove keeps `decided`, and a `NOT PROVED`
line under the table names it, the part Lean did not close, and why. With
`--verbose`, so does a fact loopty states for isl alone.

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
| definedness | read | some statement wrote the cell read, before the read: a disjunction of existentials, one per writer |
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
  - *non-vacuity*: at a point isl samples, Lean proves every hypothesis of a
    part, so the hypotheses as Lean reads them hold somewhere;
  - *mutations*: every cast the examples refuse, stated as if isl had found it
    legal, fails in Lean on the class isl's pair is in and no other, and a
    kernel's subscripts perturbed, its guard dropped or an index moved, are
    proved where isl decides them and not where isl refutes them;
  - *a differential*: every domain of every example and test kernel, read
    back from loopty's text, holds at exactly the points of a box where isl's
    set does, and Lean's `decide` of the printed predicate agrees with isl at
    points in and out of it.
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
included, from an empty proof cache. Its report, run on loopty's branch of
this change with Lean v4.29.1:

COVERAGE

## What stays isl's

Facts loopty states for isl alone, from sets isl computes, which Lean is not
asked about; each says so with `--verbose`, and the coverage report lists them.

- **A layout a kernel rewrites** (`the rows of val stay inside their buffers
  and apart`): an induction over the run, decided by isl over the cells a
  write reads; issue [#186](https://github.com/xywei/loopty/issues/186).
- **The casts after a substitution:** a substitution changes what the
  instances read, and the dependences the later steps are checked against are
  isl's; issue [#187](https://github.com/xywei/loopty/issues/187).
- **Work items:** a cast that puts a loop on a hardware axis also claims that
  no dependence joins two work items, which is stated for isl only; the cast
  is then left `decided` as a whole; issue [#188](https://github.com/xywei/loopty/issues/188).
- **A part of a tile isl computed:** a statement in one of two tiled loops is
  split by the part of the tile's map over its loop, which isl computes; a
  cast with such a step is left `decided`; issue [#189](https://github.com/xywei/loopty/issues/189).
