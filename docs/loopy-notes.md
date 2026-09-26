# Notes on loopy and islpy

Eleven interactions with loopty's dependencies that cost real debugging time, each
with the local workaround and the reason it is local. No upstream issues were
filed: these are notes so that the next person meets the answer instead of the
symptom.

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
nest, and `BasicMap.is_bijective`, in `map_domain`, which is what the skew uses.
With islpy 2026 installed the whole stencil demo fails with an `AttributeError`
raised from inside loopy. `pyproject.toml` therefore carries `islpy<2026` with
that reason beside it. Drop the ceiling when a loopy release supports islpy
2026, not before.

**The second consequence, which is easy to miss.** islpy 2025.x publishes no
cp314 wheels, so the pin also pins the interpreter: a machine whose only Python
is 3.14 cannot install loopty at all. `.github/workflows/ci.yml` runs 3.12 and
3.13 for that reason, `requires-python` is `>=3.12`, and the device runs needed
a uv-provisioned CPython 3.13 rather than the host's 3.14. If the pin moves, the
interpreter matrix moves with it.

## 5. Three deprecation warnings that are loopy's, not loopty's

The test suite turns `DeprecationWarning` into an error so that one of loopty's
own cannot hide in the noise of a run that compiles C. Three exemptions are listed
in `pyproject.toml` and again in `tests/conftest.py` (the second because a `-W`
on the command line overrides the ini file):

* `'GCCToolchain.copy' is deprecated`. loopy builds its C toolchain with
  codepy's deprecated `Toolchain.copy` inside `ExecutableCTarget.__init__`,
  before loopty is handed anything. Unreachable from here.
* `BasicMap.is_bijective with implicit conversion of self to Map is
  deprecated`. `lp.map_domain`, which the skew uses, requires an
  `isl.BasicMap` (its `_find_aff_subst_from_map` raises `RuntimeError` for
  anything else) and then asks that BasicMap whether it is bijective. There is
  no spelling of the call from loopty that avoids the warning. This is the same
  method as in note 4, so it disappears when the pin does.
* `Aff.is_equal with implicit conversion of self to PwAff is deprecated`.
  Raised from `simplify_pw_aff` while loopy generates code for a loop whose
  bound is a piecewise affine expression, which a tiled or split loop always
  has. The other method from note 4. It is the one that hides: loopy keeps a
  persistent code-generation cache under the user cache directory, and on a
  machine that has generated the kernel before, code generation is skipped and
  the warning never fires. A fresh CI runner has no cache and fails eight tests
  on it. To see what CI sees, run the suite with `XDG_CACHE_HOME` pointed at an
  empty directory.

## 6. loopy's own loop-nest choice is not the term's

Not a bug, but the reason `lower_generic` ends by calling `prioritize_loops`.
Given the Jacobi stencil, loopy's scheduler puts the space loop outside the time
loop, which reverses a dependence relative to the body as written. The order the
body was written in is the order the term means, so lowering pins it; any
departure is a schedule, hence a cast, hence checked. `schedule._with_priority`
then *replaces* the priority at each accepted step rather than adding to it,
because `lp.prioritize_loops` accumulates and an interchange would otherwise
contradict the priority set before it.

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
reads, `flow.statement_accesses`, lists `off[r]` and `off[r + 1]` after every
ragged access, read or written, whenever the kernel declares the offsets, so the
edge is drawn, in the direction the body gives it, and the schedule checker and
the typing rules see the same read. Offsets the kernel does not declare are the
argument lowering adds, which nothing in the body can write, so there is no
edge to draw for them. The instructions that assign ragged bounds
(`cnt_r_init`) are final too. One reads the counts, or the offsets when the
counts are not a parameter, and it is ordered where the first statement that
needs it is: after every earlier writer of that array and before every later
one. Left to the heuristic, it waited for a later writer as well, and a ragged
loop followed by a statement that rewrites its offsets became a cycle through
the loop, the bound and the rewrite. Because the bound is computed once, a
statement that needs it after that array has been rewritten would see the old
row length, so `lower_generic` refuses that order with a `LoweringError`.

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
(`_outer_part`): projecting `j` out of `0 <= j < m` leaves `m >= 1`, and a loop
over `r` that ran only when the inner loop has an iteration would skip every
statement after that inner loop. Nothing is lost by dropping, because the
innermost stretch keeps every constraint. Constraints on the sizes alone go
too, so the loop over `r` is the same set in both statements and carries no
predicate. A statement no other one leaves keeps its single domain, so the code
generated for every kernel that lowered before is what it was. What is left, one
name for two different loops in a term built by hand, is refused with a
`LoweringError` naming the loop.

## 11. Hardware axes on reductions

What loopy 2025.2 generates code for, measured with its plain OpenCL target on
a double sum `reduce_sum(reduce_sum(a[i, j] for j in Fin[i + 1]) for i in
a.dom)` and on a sum inside a statement loop:

| schedule | loopy | loopty's `buildable` |
|---|---|---|
| inner reduction `j` on `l.0`, outer `i` sequential | "instruction 'S0_i_init' does not use all local hw axes" | refused: a hardware axis on a nested reduction |
| `i` on `l.0`, `j` on `l.1` | the same | refused, the same |
| outer reduction `i` on `l.0`, `j` sequential | builds | buildable |
| `j` split, the inner half on `l.0` | "contains both parallel and sequential inames" | refused |
| a reduction on `g.0` | "the only form of parallelism supported by reductions is 'local'" | not checked |
| a reduction split, both halves on `l.*` | "contains more than one parallel iname" | not checked |
| a local axis over a symbolic extent | "a numeric maximum was not found" | not checked |

The first row is the nested case: loopy sets and updates the enclosing
reduction's accumulator outside the inner reduction's loop, in instructions
that do not run on its axis, and generates code only when every instruction
uses every local axis. `schedule._unbuildable_reason` refuses a parallel tag on
a nested reduction's iname with that reason. It used to be refused only by
accident, as a ragged fiber: the inner domain names the outer binder `i` as a
parameter, and every parameter that was not a size counted as data read out of
an array. An enclosing binder, or a loop of the statement, is not data now, so
a bound affine in one is a triangle, and a reduction over it is not a ragged
fiber. The rows marked "not checked" are limits the check does not know yet
(issue #35); a schedule that hits one passes `buildable` and fails in code
generation.

