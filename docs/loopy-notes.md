# Notes on loopy and islpy

Thirteen interactions with loopty's dependencies that cost real debugging
time, each with the local workaround and the reason it is local. No upstream
issues were filed: these are notes so that the next person meets the answer
instead of the symptom.

Versions these were observed against: loopy 2025.2, islpy 2025.2.5, codepy as
pinned by loopy, on CPython 3.13.

## 1. The C target's host wrapper and device signature disagree about arguments

**Symptom.** A kernel with a value argument that occurs only in another
argument's *shape* segfaults when it is called. No diagnostic, no Python
traceback: the process dies inside the compiled code.

**Cause.** loopy's C target generates the device function's signature from the
names the kernel body actually needs, and the host wrapper's call from every
argument the kernel has. A `ValueArg` that appears only in a shape is in the
second list and not in the first, so the wrapper passes one more argument than
the function takes and every subsequent argument lands in the wrong register.

**Local fix.** `lower._used_names` collects the names the generated code
mentions (domain parameters, and variables in instructions), and `_arguments`
declares only those as value arguments. Arrays whose shape mentions an
undeclared name are then declared with `shape=None`.

**What that costs.** loopy no longer checks the shape of those arrays, and
loopty does not check it either, so a wrongly sized array is undefined
behaviour rather than an error. The ragged flat buffers and `x` in the spmv demo
are in that position. Checking shapes at the executor boundary would close it.

**The other instance (2026-09-19).** An *array* parameter the body never reads
or writes has the same effect: it is in the host wrapper's call and not in the
device function's signature, so every argument after it lands one register
early. Observed as `y[i] = 1.0` with an unused `x` returning all zeros and the
process dying with `double free or corruption` at exit. Value arguments could be
dodged by declaring only the used ones; an unused array cannot be dropped the
same way without also changing how the sizes it determines reach the kernel, so
`lower.py` refuses such a term with a `LoweringError` naming the parameter. A
parameter that only exists to determine a size has to be read somewhere, or
the size has to come from an array that is.

## 2. `pow` without `math.h` on the C target

**Symptom.** `r2 ** -0.5` in a kernel body produces C that calls `pow` without
including `math.h`, and gcc refuses the generated file.

**Cause.** loopy emits the call but does not register `pow` as a target
callable, so the header that declares it is never requested.

**Local fix.** Do not write `x ** -0.5`. Build the call to the target's library
function explicitly: `pymbolic.primitives.Call(Variable("sqrt"), (r2,))`, which
loopy resolves against the target and which does pull in the header.
`examples/p2p.py` has the dual-mode pattern (numpy on numbers, a `Call` on a
term), and it is the pattern any elementary function needs until loopty grows a
surface of its own for them.

**Related.** The callee of a `prim.Call` must not be collected as a *used name*,
or `sqrt` is declared as a value argument and loopy refuses the kernel with
"value argument 'sqrt' was not given". `lower._used_names` subtracts callees for
that reason.

## 3. A zero-length array cannot be passed to the C target

**Symptom.** A CSR matrix all of whose rows are empty fails with
`TypeError: expected c_double instead of float`, raised from
`loopy/target/c/c_execution.py`.

**Cause.** The C invoker passes an array argument as a pointer and, for an empty
one, tries to produce a null pointer by calling the pointer type on `0.0`:
`arg_t(0.0) if arg.size == 0 else ...`. `POINTER(c_double)(0.0)` is not a thing.

**Local fix.** `executor._pad_empty_arrays` gives a zero-length array argument
one cell before the call and restores the original afterwards, so the pad is
never mistaken for a result. It is sound exactly because the array is empty: no
index into it is in bounds, so no generated loop can touch the added cell.

**The limit of the fix.** Only arguments loopy declares *without* a shape are
padded, which for a lowered term means the flat buffer of a ragged axis and
nothing else. An argument that has a declared shape is how loopy infers the size
parameters, and lengthening one would make it infer the wrong size (padding an
empty `cnt` makes loopy believe `n == 1` and then complain that `off_cnt` should
have two entries). So a CSR matrix whose rows are all empty runs, and a kernel
called with *no rows at all* does not: every argument is empty then, including
the shape-bearing ones, and the call reaches loopy's `TypeError`.
`tests/test_adversarial.py` covers both, the second as a pinned failure.

## 4. Why `islpy<2026`, and why CI stays on 3.12 and 3.13

**The pin.** loopy 2025.2 calls two islpy methods that islpy 2026.2.2 removed:
`Aff.is_equal`, in `simplify_pw_aff` during code generation for a tiled loop
nest, and `BasicMap.is_bijective`, in `map_domain`. With islpy 2026 installed
the whole stencil demo fails with an `AttributeError` raised from inside loopy.
`pyproject.toml` therefore carries `islpy<2026` with that reason beside it.
Drop the ceiling when a loopy release supports islpy 2026, not before. The skew
used to go through `map_domain`; since it became an affine map rewritten by
loopty itself (note 13), nothing in loopty calls `map_domain`, and only the
first method holds the pin.

**The second consequence, which is easy to miss.** islpy 2025.x publishes no
cp314 wheels, so the pin also pins the interpreter: a machine whose only Python
is 3.14 cannot install loopty at all. `.github/workflows/ci.yml` runs 3.12 and
3.13 for that reason, `requires-python` is `>=3.12`, and the device runs needed
a uv-provisioned CPython 3.13 rather than the host's 3.14. If the pin moves, the
interpreter matrix moves with it.

## 5. Two deprecation warnings that are loopy's, not loopty's

The test suite turns `DeprecationWarning` into an error so that one of loopty's
own cannot hide in the noise of a run that compiles C. Two exemptions are listed
in `pyproject.toml` and again in `tests/conftest.py` (the second because a `-W`
on the command line overrides the ini file):

* `'GCCToolchain.copy' is deprecated`. loopy builds its C toolchain with
  codepy's deprecated `Toolchain.copy` inside `ExecutableCTarget.__init__`,
  before loopty is handed anything. Unreachable from here.
* `Aff.is_equal with implicit conversion of self to PwAff is deprecated`.
  Raised from `simplify_pw_aff` while loopy generates code for a loop whose
  bound is a piecewise affine expression, which a tiled or split loop always
  has. The method from note 4. It is the one that hides: loopy keeps a
  persistent code-generation cache under the user cache directory, and on a
  machine that has generated the kernel before, code generation is skipped and
  the warning never fires. A fresh CI runner has no cache and fails eight tests
  on it. To see what CI sees, run the suite with `XDG_CACHE_HOME` pointed at an
  empty directory.

There used to be a third, `BasicMap.is_bijective with implicit conversion of
self to Map is deprecated`, raised by `lp.map_domain`: it requires an
`isl.BasicMap` and then asks it whether it is bijective. It went with the call
(note 13), and the one test that still calls `map_domain`, to pin that loopy
refuses the diamond, silences it locally.

## 6. loopy's own loop-nest choice is not the term's

Not a bug, but the reason `lower_generic` ends by calling `prioritize_loops`.
Given the Jacobi stencil, loopy's scheduler puts the space loop outside the time
loop, which reverses a dependence relative to the body as written. The order the
body was written in is the order the term means, so lowering pins it; any
departure is a schedule, hence a cast, hence checked. `schedule._with_priority`
then *replaces* the priority at each accepted step rather than adding to it,
because `lp.prioritize_loops` accumulates and an interchange would otherwise
contradict the priority set before it.

**A priority is a preference.** loopy nests a kernel's domains as a tree: a
domain that names a loop as a parameter is defined inside that loop, and every
loop it defines runs inside it. A ragged fiber names its row, because the row's
length is read there (`[r, nl_cnt_r] -> { [j] : 0 <= j < nl_cnt_r }`), and a
statement that another statement leaves has its domain cut after the loops
around it (note 10), which then names them too. That nesting is a constraint,
and when the priority disagrees with it loopy warns "Cannot satisfy constraint
that iname 'r_inner' must be nested within 'j_outer'", drops the priority, and
generates a nest of its own choosing. `tile("r", "j", 2, 2)` on a ragged
recurrence `w[r + 1, j] = w[r, j] + val[r, j]` asks for `r_outer j_outer
r_inner j_inner`, which runs the dependence forward, and loopy generated
`r_inner r_outer j_inner j_outer`, which does not: the compiled `w` disagreed
with the body's while every cast fact was `decided`. `interchange("j", "r")` on
the same kernel reaches the same place.

The domains are not the whole of it. loopy also nests a loop inside another
when every instruction in the first is in the second and the second has more
(`find_loop_nest_around_map` in `loopy.schedule`, which both of its schedulers
keep to). A statement loop `k` over a row's cells, with a ragged reduction in
its body, `w[r + 1, k] = w[r, k] + reduce_sum(val[r, j] for j in val.dom[r])`,
shares the domain `{ [r, k] }` with the row, and is still nested in `r`,
because the row length is assigned in `r` and outside `k`. There
`tile("r", "k", 2, 2)` warned the same way, loopy generated `r_inner r_outer
k_inner k_outer`, and the compiled `w` disagreed with the body's.

**Local fix.** `schedule._nest_reason` reads the nesting after each step with
loopy's own `find_loop_nest_around_map` (and the loops those loops are nested
in, in turn) and refuses as unbuildable an order that puts a loop outside a
loop loopy nests it inside, naming the two, why loopy nests one in the other,
and the interchange that would put them right. Buildability is asked of the
schedule as it stands, so a later `interchange("r_inner", "j_outer")` makes the
tiled schedule buildable again, and it runs the tiles in the order that was
checked. Checking loopy's own linearized nest against the order would be the
complete answer, and would cost a scheduling pass per step.

## 7. The single-writer heuristic draws an edge against the body's order

**Symptom.** Two statements that feed each other, one of them across an
iteration of an enclosing loop, lower without complaint and then fail at the
first run with `DependencyCycleFound: S0, S1`. The acoustic update in
`examples/wavefront_acoustic.py` is the case: `S0` reads the pressure `S1` wrote
at the previous time level, and `S1` reads the velocity `S0` wrote at this one.

**Cause.** `lp.make_kernel` applies a heuristic to every instruction whose
`depends_on` is not marked final: it adds a dependence on the only writer of
each variable the instruction reads, wherever in the body that writer is.
`lower_generic` draws `S1 -> S0` from the data, and the heuristic adds
`S0 -> S1` because `S1` is the only writer of the pressure `S0` reads. An
instruction dependence orders two statements within one iteration; the order
across iterations is the loop's, so the added edge is not a dependence at all.

**Local fix.** Each statement's instruction is created with
`depends_on_is_final=True`. `lower_generic` already orders a statement after
every earlier one it could read from, write over, or overwrite the input of,
which is the whole of the order within an iteration, so there is nothing left
for the heuristic to add. One read is the layout's rather than the body's: the
flat index of a ragged access goes through the offsets argument. A kernel that
writes those offsets and reads through them (a scan fused with the product that
uses it) relied on the heuristic for that edge, and without it loopy refuses
the kernel with `VariableAccessNotOrdered`. The access collector every rule
reads, `flow.statement_accesses`, lists `off[r]` after every ragged access,
read or written, whenever the kernel declares the offsets, so the edge is
drawn, in the direction the body gives it, and the schedule checker and the
typing rules see the same read. Offsets the kernel does not declare are the
argument lowering adds, which nothing in the body can write, so there is no
edge to draw for them. The instructions that assign ragged bounds
(`cnt_r_init`) are final too. One reads the counts, or `off[r]` and
`off[r + 1]` when the counts are not a parameter, and it is ordered where the
first statement that needs it is: after every earlier writer of that array and
before every later one. Left to the heuristic, it waited for a later writer as
well, and a ragged loop followed by a statement that rewrites its offsets
became a cycle through the loop, the bound and the rewrite. Because the bound
is computed once, a statement that needs it after that array has been
rewritten would see the old row length, so `lower_generic` refuses that order
with a `LoweringError`. It refuses the same stale length across the iterations
of a loop inside the row that holds both a rewrite of the row's cell and a
statement bounded by it, since the body reads the length where the loop over
the fiber starts, once per iteration of that loop, and the kernel once per
row. The collector lists the bound's read too, on every
statement the bound bounds and over the loops up to its row
(`flow.layout_reads`), so the statement's own instruction is ordered against
writers of the counts as the bound's is, and a cast is checked against it.
`off[r + 1]` is listed only there, where it is read.

## 8. Two instructions cannot share a reduction iname

**Symptom.** Two statements that sum over the same binder with the same domain,
`s[0] = reduce_sum(a[j] for j in a.dom)` and then `s[1] = reduce_sum(b[j] ...)`,
lower without complaint and fail at the first run with
`pytools.graph.CycleError: EnterLoop(iname='j')`. So does a sum over `j`
followed by a loop over `j` that reads it, and two statements whose nested sums
bind `i` and `j` at the same two levels. One statement with two sums over `j`
is fine.

**Cause.** loopy realizes a reduction as an accumulator loop inside its
instruction, and an iname is one loop however many instructions use it. When
the second statement depends on the first (it writes the same array, or reads
what the first wrote), its reduction has to run inside a loop that has to
finish before the second statement may start. When the statements are
independent loopy fuses the two loops, which is why the collision stayed
hidden.

**Local fix.** `_Builder.plan_reductions` lets a reduction keep its binders
only when no other statement has them, as loop variables anywhere in the kernel
or as the binders of an earlier reduction, and gives it fresh inames (`j_0`)
otherwise; within one statement a name is shared only by reductions over the
same domain. A nested reduction's domain
names its enclosing binders as parameters, so a renamed outer binder is renamed
there too, or the inner loop would hang from the other statement's outer one
("Loop 'i' cannot be nested outside 'j_0'"). `Lowering.reduction_inames`
records the names each reduction ends up with, and `Schedule` addresses a
reduction by them.

## 9. Which flags the C target compiles with, and FMA contraction

**What loopy does.** `lp.ExecutableCTarget()` builds a `CCompiler`, which
guesses a codepy toolchain from Python's build configuration and then replaces
its compiler and flags with its own defaults: `gcc -std=c99 -O3 -fPIC`, plus the
kernel's `options.build_options`, appended in that order. So the compiler is
whatever `gcc` is on the path: GCC on Linux, clang on macOS.

**Whether `a * b + c` becomes one fused multiply-add.** GCC's default is
`-ffp-contract=fast` in the GNU dialects and `off` in a standard one such as
`-std=c99`, and on x86-64 it can contract only when told the instruction exists
(`-march` or `-mfma`), which loopy's flags do not. So a stock build on Linux
x86-64 contracts nothing. clang contracts within an expression by default (`-ffp-contract=on`),
and arm64 has FMA in its baseline, so the same kernel built there does fuse.
OpenCL C permits contraction by default and has no build option to forbid it;
`#pragma OPENCL FP_CONTRACT OFF` in the source is the way.

**Why it matters.** A fused multiply-add rounds once, and the native run of a
kernel body rounds after the multiplication and again after the addition, so
the two can differ in the last bit. An `exact` output is compared bit for bit.
With `a = 1 + 2**-30`, `b = 1 - 2**-30`, `c = -1` the native value is 0.0 and
the fused one `-2**-60`, which is what `tests/test_contraction.py` shows on a
machine with FMA by building the kernel with `-march=native -ffp-contract=fast`.

**Local fix.** A kernel with an `exact` output (`lower.allows_contraction`) is
lowered with `-ffp-contract=off` in its build options on the C target, which
both GCC and clang honour and which comes after any flag of the toolchain's, and
with the target's pragma in the source: `#pragma STDC FP_CONTRACT OFF` for C,
which clang honours and GCC ignores, and the OpenCL one for OpenCL. A kernel
whose outputs are all `approx` or `reassoc` is left to the compiler.

The flag pins the build loopty runs, and not the source `loopty run
--emit-code` prints, which someone may compile by hand. GCC in a GNU dialect
contracts by default whenever `-march` or `-mfma` gives it the instruction, and
it ignores the standard pragma, so the C for an exact kernel also carries GCC's
own spelling of the flag, behind a guard that keeps it from other compilers:

```c
#pragma STDC FP_CONTRACT OFF
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize ("fp-contract=off")
#endif
```

GCC documents the `optimize` pragma as meant for debugging. Here it asks for
less optimization rather than more, and for exactly what the flag asks.
`tests/test_contraction.py` compiles the emitted source with `gcc -std=gnu99
-O2 -mfma`: the approx kernel's assembly has a fused multiply-add and the exact
one's has none. On hardware with FMA it also builds both with `-march=native`
and calls them, and the exact one keeps the native run's bits.

## 10. One domain per loop, and a statement after an inner loop

**Symptom.** A dense kernel with statements at two depths of one loop,

```python
for r in y.dom:
    for j in a.dom[r]:
        y[r] = y[r] + a[r, j]
    z[r] = 1.0
```

failed to lower with `RuntimeError: domain '[n] -> { [r] : ... }' redefines
iname 'r' that is part of a previous domain`, from `lp.make_kernel`, so the
executor, `Schedule` and `loopty run` all failed with it. So did two inner loops
side by side in one outer loop.

**Cause.** loopy defines an iname in exactly one domain. Each statement
contributed its domain over every loop around it, `{ [r, j] }` for the first
statement and `{ [r] }` for the second, and `_merge_domains` merges only domains
over the same inames. A ragged inner loop already had its domain split into an
outer `{ [r] }` and an inner `{ [j] }` (the row length is assigned inside the
`r` loop, see `_statement_domains`), which is why the same shape with a ragged
inner loop worked.

**Local fix.** `lower._depth_cuts` finds, for each statement, the loops at which
another statement leaves its nest, and the statement's domain is cut there
into a domain per stretch of loops: `{ [r] }` and `[r] -> { [j] }`, the first
merging with the other statement's. An outer stretch drops the constraints that
mention an inner loop instead of projecting the inner loop out
(`_outer_part`): projecting `j` out of `0 <= j < m` leaves `m >= 1`, and two
inner loops side by side, over `m` and over `p`, would give the loop over `r`
the union of `m >= 1` and `p >= 1`, which is not convex and cannot be one loop.
Nothing is lost by dropping, because the innermost stretch keeps every
constraint. Constraints on the sizes alone go too, so the loop over `r` is the
same set in both statements and carries no predicate. A statement no other one
leaves keeps its single domain, and the code generated for every example is
what it was. What is left, one name for two different loops in a term built by
hand, is refused with a `LoweringError` naming the loop.

**Cut alike (2026-09-26).** A statement bounded by a ragged row's length is cut
after the row loop too, because the length is assigned there, and that cut has
to be shared. In

```python
for r in y.dom:
    for i in x.dom:
        for j in val.dom[r]:
            y[r] = y[r] + x[i] * val[r, j]
        z[r, i] = 1.0
```

the statement in the fiber was cut after `r` and after `i`, and the one beside
the fiber, which leaves no nest, not at all: `r` came out in `{ [r] }` and in
`{ [r, i] }`, which do not merge, and the kernel was refused for using `r` for
two loops, which it does not. `_depth_cuts` now counts the row's cut among the
others and passes every cut on to each statement that has the loops up to it
and more beyond, until nothing changes, so the second statement is cut after
`r` as well. The refusal that is left reads the term before it blames a name:
when every statement that has the loop has the same loops around it, the name
is one loop, and the message says the lowering could not give it one domain
instead of asking for a rename.

A statement beside an inner loop also has to stay out of it, which is note 12.

## 11. Hardware axes on reductions

What loopy 2025.2 generates code for, measured with its plain OpenCL target on
a double sum `reduce_sum(reduce_sum(a[i, j] for j in Fin[i + 1]) for i in
a.dom)`, on a single sum, and on a sum inside a statement loop:

| schedule | loopy | loopty's `buildable` |
|---|---|---|
| inner reduction `j` on `l.0`, outer `i` sequential | "instruction 'S0_i_init' does not use all local hw axes" | refused: a hardware axis on a nested reduction |
| `i` on `l.0`, `j` on `l.1` | the same | refused, the same |
| outer reduction `i` on `l.0`, `j` sequential | builds | buildable |
| `j` split, the inner half on `l.0` | "contains both parallel and sequential inames" | refused |
| `j` split, the inner half on `l.0`, the outer on `ilp` | the same | refused, the same |
| `j` split, the inner half on `ilp` (or `j` on `ilp`) | builds under some string hash seeds; under others "touched variable that (for privatization, e.g. as performed for ILP) required iname(s) ..." | refused |
| `j` split, the inner half on `unr` | builds | buildable |
| a reduction on `g.0`, `ilp.seq` or `vec` | "the only form of parallelism supported by reductions is 'local'" | refused |
| a reduction split, both halves on `l.*` | "contains more than one parallel iname" | refused |
| a reduction on `l.0` over a symbolic extent (`Fin[i + 1]`, `i < n`) | "a numeric maximum was not found" | refused |
| the same over `Fin[8]` | builds | buildable |
| a reduction on `l.0` in a statement loop on `l.1` over `Fin[n]` | "a numeric maximum was not found" | refused |
| the same with the statement loop on `g.0` | builds | buildable |

The first row is the nested case: loopy sets and updates the enclosing
reduction's accumulator outside the inner reduction's loop, in instructions
that do not run on its axis, and generates code only when every instruction
uses every local axis. `schedule._unbuildable_reason` refuses a parallel tag on
a nested reduction's iname with that reason. It used to be refused only by
accident, as a ragged fiber: the inner domain names the outer binder `i` as a
parameter, and every parameter that was not a size counted as data read out of
an array. An enclosing binder, or a loop of the statement, is not data now, so
a bound affine in one is a triangle, and a reduction over it is not a ragged
fiber.

The rows from the fourth on are how `map_reduction` in
`loopy.transform.realize_reduction` classifies a reduction's loops, and
`schedule._reduction_reason` asks them the way it does, with loopy's own tag
classes: an untagged loop and one loopy unrolls (`unr`, `ilp`) are summed in
sequence, a local axis in a tree across a group, and any other concurrent axis
not at all. So `ilp` is a sequence here, although the checker counts it as an
order-free loop and asks an accumulation's permission before a reduction loop
is tagged with it. A reduction is generated when all of its loops are
sequential, or when it is one loop on a local axis.

An `ilp` loop meets one more stage afterwards: loopy privatizes the
temporaries written in it along the loop, a reduction's accumulator among
them, and then refuses the instruction that initializes the accumulator outside
the loop. Or does not: the same kernel was refused under some values of
`PYTHONHASHSEED` and built under others, which points at an order loopy takes
from a set. A `buildable` fact that holds on some runs is not one, so a
reduction over an `ilp` loop is refused; `unr` unrolls the sum in order, and
builds under every seed tried. A local reduction then needs its
extent, and the extent of every local axis of the statement around it, to have
a numeric maximum, because loopy keeps the partial sums in an array in local
memory whose shape is fixed when the code is generated (`_get_int_iname_size`);
`schedule._extent_reason` asks `static_max_of_pw_aff(...,
constants_only=True)` of the loop's bounds, as loopy does.

## 12. loopy adds loops to an instruction whose loops are not final

**Symptom.** A statement that reads what an inner loop writes, beside that loop,

```python
for r in y.dom:
    for j in a.dom[r]:
        y[r] = y[r] + a[r, j]
    z[r] = z[r] + y[r]
```

lowers and runs, with a `LoopyWarning` that "the iname(s) 'j' on instruction
'S1' was/were automatically added", and computes the wrong `z`: the sum of the
partial sums of each row rather than the row's total. A copy `z[r] = y[r]`
before the inner loop gets `y[r]` after all but the last update. A later loop of
its own (`for q in z.dom: z[q] = z[q] + y[q]`) and a ragged inner loop went
wrong the same way before statements at two depths lowered at all.

**Cause.** `lp.make_kernel` adds loops to every instruction whose
`within_inames` are not marked final: for each variable the instruction reads,
the loops of the instructions that write it, less the loops those writers'
subscripts name. `y[r] = y[r] + a[r, j]` runs in `r` and `j` and names `r`, so
every reader of `y` is put inside the loop over `j`. It then runs once per `j`,
and not at all in a row whose loop over `j` is empty.

**Local fix.** Each statement's instruction is created with
`within_inames_is_final=True`: its loops are the loops around it in the source,
which `lower_generic` knows and loopy has nothing to add to. So are those of the
instructions that assign ragged bounds (`cnt_r_init`), which sit in the row loop
and the loops around it.

## 13. loopy's affine transforms refuse a map that is not unimodular

**Symptom.** `lp.map_domain(kernel, isl.BasicMap("{ [t, i] -> [a, b] : a = t +
i and b = t - i }"))` raises `LoopyError: No suitable equation for 't' found`
(or for `'i'`: which old iname it tries first follows the order of a Python
set, so it changes with the hash seed), and `lp.affine_map_inames(kernel, "t, i", "a, b", ["a = t + i", "b = t - i"])`
raises `RuntimeError: division with remainder in linear solve for 't'`. The
map is a bijection of the integer points onto its image, and it is the diamond
of diamond tiling.

**Cause.** Both transforms rewrite the instructions by solving the map for each
old iname as an affine function of the new ones, and both accept only a solution
with integer coefficients (`map_domain` looks for an equality in which the old
iname has coefficient 1 or -1). The diamond has determinant -2: its image is
only the points whose two coordinates have the same parity, and on that image
`t = (a + b) / 2`, which is exact there and not an integer affine expression.
`map_domain` can also refuse a map whose inverse is integer affine, depending on
the order in which isl eliminates the other variables: the embedding
`(t, i) -> (t + i, t - i, t)` fails the same way, for `t` or for `i`, although
`t` is its third coordinate and `i` its first minus its third.

**Local fix.** `schedule._affine_kernel` does the rewrite from the same isl map
without solving anything. The domain that defines the mapped loops becomes its
image under the map, which isl states exactly, with the parity as an
existentially quantified constraint: `[nt, nx] -> { [a, b] : (a + b) mod 2 = 0
and ... }`. A domain nested in those loops, which names them as parameters (a
ragged fiber, or the domain of a reduction), becomes its image too, with the
new loops as its parameters. Each old loop variable is replaced in every
instruction by the quasi-affine expression isl gives for the inverse on that
domain, `t = floor((a + b)/2)` and `i = floor((a + b)/2) - b`, which is exact on
the image. `Schedule.affine` goes through it, and so does `Schedule.skew`,
which is the affine map `(t, i) -> (t, i + t)` with both names kept; the tiling
after a skew or a diamond is still loopy's own `split_iname`.

**What loopy makes of it.** loopy 2025.2 generates correct code for the image:
the stencil, the coupled acoustic pair, and the stencil tiled in `(a, b)` all
agree with the native run bit for bit, at sizes of both parities. What it
generates is the image's bounding loops with the parity tested by an `if`
inside the innermost one, `if (-b - a + 2 * ((b + a) / 2) == 0)`, not a loop
that steps by two, so half the iterations of that loop do nothing. A kernel
that has to be fast in diamond coordinates would want the stride, which neither
loopy nor this rewrite produces.

**The limits of the fix.** A map applies to every statement in its loops,
because loopy gives the statements of a loop one domain, so a map per statement
(a time offset between two statements, which diamond tiling of the acoustic pair
needs) is refused. isl decides the casts of any map whatever the kernel looks
like; when the rewrite cannot write one for loopy (loops that no one domain
defines, such as a row and the ragged fiber inside it, an image that is not one
basic set, or a piecewise inverse), the schedule carries a `refuted`
`buildable` fact with the reason, and no kernel, rather than an error from
loopy.
