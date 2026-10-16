# Running on loopy main

loopy's main branch keeps a kernel's domains as
[namedisl](https://github.com/inducer/namedisl) sets (loopy #1023): isl
objects whose dimensions are known by name, which loopy's maintainers and
sumpy, pytato and arraycontext already run on. No release of loopy carries it
yet. loopty's `main` stays on loopy 2025.2, which is what `pip install loopty`
gets; the branch `port/loopy-main` runs on loopy main, and is merged into
`main` once a loopy release carries namedisl.

Nothing a user writes changes. Every example prints the same ledger on both,
row for row, and the compiled runs agree with the native ones as before; the
code loopy generates differs in the four ways listed below.

## Install

```sh
git clone https://github.com/xywei/lanky.git
git clone --branch port/loopy-main https://github.com/xywei/loopty.git
cd loopty
uv sync --group dev
uv run pytest -q
```

`[tool.uv.sources]` takes loopy, islpy and namedisl from their main branches,
as loopy main's own `requirements.txt` does, and `uv.lock` pins the commits
the branch was last tested against. islpy is built from source, so the first
sync needs a C++ compiler and takes a few minutes. To move to the current
heads:

```sh
uv lock --upgrade-package loopy --upgrade-package islpy --upgrade-package namedisl
uv sync --group dev
```

The branch drops main's `islpy<2026`: islpy 2026 is what loopy main is built
against, and it builds for Python 3.14 too, which main cannot install on
(note 4 of [loopy-notes.md](loopy-notes.md)). The suite passes on 3.12, 3.13
and 3.14, and also with namedisl's release, 2026.1.1, in place of its main
branch.

CI: `.github/workflows/loopy-main.yml` runs the suite, the three commands of
every example and the transcripts against the pinned commits, on 3.13 and
3.14, and against the current heads on 3.13. `ci.yml` runs as on `main`, and
its job that takes lanky from PyPI names loopy, islpy and namedisl by their
Git references.

## Where loopty meets namedisl

`loopty.named` is the one module that does. loopty's own analysis (`flow`,
`typing`, `hypotheses`, `oracle`, `domain`, `dataflow`) stays on islpy: it builds isl
objects from text, reasons by position where it means to (the padded instance
space, with the statement's index in it), and wants a mismatch of dimensions
to fail, where namedisl aligns two objects by name and gives a name one side
lacks an unconstrained dimension. Its objects already keep namedisl's
invariants, every dimension named and no two alike (`loopty.names`), so they
cross without loss.

| | what it does |
|---|---|
| `to_named(obj)`, `to_named_set(obj)` | an isl object loopty built, given to loopy: a domain, the assumptions, a set loopy writes as a condition |
| `as_isl(obj)` | an object loopy keeps or computes, a domain or a loop's bounds, read by loopty |
| `domains_of(entry)`, `with_domains(entry, domains)` | a kernel's domains, read and written as islpy sets |
| `aff_from_expr(space, expr)` | loopy's reader of an affine expression, asked over an islpy space |

Each takes an object already on its side as it is. A domain comes back as an
`isl.Set`, where loopy 2025.2 kept an `isl.BasicSet`.

## What changes on loopy main, and what loopty does

1. **Domains and assumptions are namedisl sets**, and `lp.assume` deprecates
   a raw islpy set (to stop working in 2028). loopty crosses at
   `loopty.named`, in `lower.lower_generic` (the kernel's domains, its
   assumptions, a narrowed statement's predicate), in the affine rewrite and
   the counted loops of `schedule`, in the buildability checks that read a
   kernel's domains, and in the hoisting of a sum's invariant operands
   (`loopty.hoist`, item 10).
2. **loopy's reader takes a mapping of names**, `aff_from_expr(vars_to_aff,
   expr)` where 2025.2 took a space, and its guard,
   `with_aff_conversion_guard`, no longer catches a `TypeError`. The readings
   of `loopty.isl_reading` (note 23) decline with loopy's own
   `ExpressionToAffineConversionError`, as loopy declines a call, and a fourth
   reading turns the `TypeError` an operator isl has none of raises back into
   a decline: `if i < i ^ 1:` failed in loopy's bounds check without it.
   The guard is replaced only inside `declining()`, like the other three.
3. **The bounds check sees every size.** loopy main checks an access over its
   instruction's domain intersected with the kernel's assumptions, aligned by
   name, which brings in every parameter; 2025.2 left the assumptions out
   when they said nothing. So `off[0] = 0` in a scan, outside every loop, is
   checked against `off: Arr[Fin[n + 1], Nat]`, and holds only for `n >= 0`.
   loopty now tells loopy that every size is non-negative, always, which the
   contract enforces (`contract.sizes_not_negative`) and every fact assumes;
   main tells it only beside what a scalar's sort says. A kernel whose own
   ledger refutes such an access, `y[0]` of `y: Arr[Fin[n], ...]`, is refused
   by loopy main too.
4. **`lp.map_domain` maps the diamond.** It takes a map that is not
   unimodular, and loops over every `b` with a parity test in the innermost
   loop; `lp.affine_map_inames` is gone. loopty keeps its own rewrite, whose
   map is the one the checker decided and whose counted loops have no parity
   test (note 13).
5. **Two vectorized schedules generate code.** A ragged row's length read in
   a `vec` loop, and a sum on a local axis inside one, made loopy 2025.2 raise
   from code generation; loopy main writes OpenCL for them that does not
   compile (a vector as a loop's bound; the vectorized loop's variable never
   declared). `buildable` refuses both, as before, and says "loopy cannot
   generate working code for it" (note 14).
6. **Messages.** loopy main names the index of an access `_lpy_s0` in its
   bounds check's message, where 2025.2 wrote `i0`.
7. **The order of a domain's loops** is namedisl's, not the nest's: a tile
   of `r` and `c` gives `[r_inner, r_outer, c_outer, c_inner]`. loopty reads
   a domain's loops by name:
   - a stride's offset (`Schedule.strides`) names the loops outermost first,
     `2*b_inner_step - a_outer - b_outer - a_inner` for the tiled diamond,
     where `main` wrote them in loopy 2025.2's order;
   - `fuse` cuts the producer's domain after the fused loops wherever they
     sit in it, as `main` does too since #192 (loopy 2025.2 puts the split
     loops last, `[j, i_outer, i_inner]`). So a fusion after a `split` or a
     `tile` of the producer's outer loop builds on both.
8. **Still one convex piece per domain.** loopy main keeps a domain as a set
   that may have several pieces, but its code generation still asks for one
   (`isl_helpers.convexify`), so loopty's one-basic-set restriction on the
   images of its maps stays.
9. **A loop over a product of sizes fails earlier.** loopy main's bounds
   check adds a size that only an array's shape names to the domain it
   checks an access over, where 2025.2 skipped the access. So `y[i]` in a
   loop over `Fin[n * m]` is now checked against the shape `n * m`, which isl
   cannot read, and code generation stops with isl's error
   (`isl_pw_aff_mul`). On `main` the code is generated and the run fails
   when it is called (#197). An access at an affine index inside loops that
   bring both sizes in fails in the same place on both (#245). loopty's own
   in-bounds fact for such an access is decided as before.
10. **`loopy.isl_helpers.make_slab` is gone**, and `get_access_map` takes
    and gives namedisl. The hoisting of a sum's invariant operands asks
    loopy's own question of its bounds check, whether every cell an operand
    reads is in its array at every instance of the statement (note 33 of
    [loopy-notes.md](loopy-notes.md)). It reads the instruction's domain,
    the kernel's assumptions and the loops' domains with `as_isl`, gives
    the instances to `get_access_map` with `to_named_set`, and builds the
    array's box itself, `0 <= cell < extent` on each axis, reading the
    extent with `loopty.named.aff_from_expr`, which declines `n * m` as
    `make_slab`'s reading did. The same operands move on both loopys.

## What differs in the generated code

`loopty run FILE --emit-code` for each example, against `main`:

- `composition`, `fusion`, `hexahedra`, `pairs`, `spmv`,
  `sum_factorization`, `travel`: the same text.
- `advection`, `assembly`, `calls`, `l2p`, `layer_potential`, `p2p`,
  `quadrature`, `reshape_layouts`, `stencil_skew`, `storage_layouts`,
  `taylor_reconstruction`, `wavefront_acoustic`:
  - the terms of an affine expression in another order, `x[-1 + n + -1 * i]`
    for `x[-1 + -1 * i + n]`, `cnt[t_in + 4 * t_out]` for
    `cnt[4 * t_out + t_in]` and `R[11 * a + b + 121 * i]` for
    `R[121 * i + 11 * a + b]`, which is how loopy main prints an affine
    expression over namedisl's order of names. It is the only difference in
    `advection` (`L[n + k_0]` for `L[k_0 + n]`) and `storage_layouts`;
  - and, where a split or a tile divides a size, plain division in place of
    loopy's floor-division helper, `(3 + n) / 4` for
    `loopy_floor_div_pos_b_int32(3 + n, 4)`, with the helper's definition
    gone: loopy simplifies with `n >= 0` (item 3). With no assumption on the
    sizes, loopy main writes the helper as 2025.2 does. Eight of the twelve,
    all but `advection`, `assembly`, `l2p` and `storage_layouts`, which
    divide no size;
  - and, in `wavefront_acoustic`'s tiled diamond only, the bounds of the two
    inner loops in other pieces, without the conditions the loops around
    them already imply (`-3 + nx >= 0` inside `if (-2 + nt >= 0 && -3 + nx
    >= 0)`). This is loopy main's code generation, with or without the
    sizes' assumption. Compiled and run, the two loop nests visit the same
    iterations in the same order for every `nt` and `nx` up to 40;
  - and, in `assembly` and `l2p`, a subscript simplified before a loop's
    value is put in: loopy main's C code generator simplifies a subscript
    and then puts in the value of an unrolled loop or the name of a
    hardware axis, where 2025.2 put it in first and then simplified. In
    `assembly`, constants are left unfolded in the subscripts of the
    unrolled loop, `cell[1 + 3 + 12 * c_outer]` and `cell[3 * 2 + 12 *
    c_outer]` for `cell[4 + 12 * c_outer]` and `cell[6 + 12 * c_outer]`. In
    `l2p`'s schedule for a GPU, whose loop over boxes is the group axis,
    loopy main folds a subscript 2025.2 left as written, `R[-20 + 11 * a_0
    + b_0 + 121 * gid(0)]` for `R[121 * gid(0) + 11 * (-2 + a_0) + 2 +
    b_0]`: 2025.2's reader declines `gid(0)`, a call, and so the whole
    subscript. The values are the same.

The ledgers (`lanky check --json`, `loopty run --json`) are the same for all
nineteen examples, and so is everything `python FILE`, `lanky check FILE` and
`loopty run FILE` print, on standard output and on standard error, and what
`loopty run FILE --target opencl` prints on PoCL's CPU device. Every
transcript in the documentation is the same on both. The hoisting moves the
same operands out of the same sums on both: `assembly`, `l2p` and
`layer_potential` assign their `invariant_` temporaries in 5, 8 and 32
statements, on each loopy.

## Keeping the branch

Merge `main` into `port/loopy-main` as it moves. What `main` adds that reads a
loopy kernel's domains or bounds, or calls loopy's reader, goes through
`loopty.named`. The windows (`Schedule.contract`, `loopty.window`), the
declared layouts (`loopty.layout`), the constant tables, and the updates
with `Schedule.atomic` crossed nothing new when they were merged: they
rewrite a kernel's temporaries, instructions and atomicity by name, a layout
is checked on loopty's side and reaches loopy as `dim_tags`, and
`Schedule.assume` gives `lp.assume` text, which loopy main reads into a
namedisl set itself.

Nor did the postconditions decided from what a kernel's statements store, the
theorems a kernel cites, or a kernel called inside another kernel's loop.
`loopty.dataflow` is analysis, and stays on islpy with the rest. A call
(`loopty.calls`) is inlined when the body is traced, so loopy is given the
caller's statements, with binders of their own, as before. A program's own
array over a domain is a loopy temporary whose shape is the domain's box.
Before a compiled run, `contract.made_extents` refuses sizes that would give
an array the program makes a negative length, as numpy does natively.

Nor, in `src/`, did the examples `l2p`, `hexahedra`, `advection`,
`layer_potential` and `assembly` and what came with them. A sum's domain
over only the loops its bounds read (`lower._without_loops_ignored`, note 32
of [loopy-notes.md](loopy-notes.md)) is loopty's own islpy set, and reaches
`lp.make_kernel` through `to_named_set` with the others. The zeros a device
program distributes over its work items are tagged from the kernel's
instructions, their assignees and loops. `CheckedOpenCLTarget` is loopy's
plain `OpenCLTarget` with loopty's readings, as the other targets are, and
the first scheduler's warning is silenced through `silenced_warnings`. The
work-item check's new relations are loopty's analysis. What did read loopy's
domains came with the tests and a benchmark: the tests of #264 read a lowered
kernel's domains through `domains_of`, as the other tests do, and
`benchmarks/assembly.py`'s `nested` takes them as islpy and gives them back
as namedisl sets itself, since `--baseline-only` imports no loopty.

Of what 0.4 brought, the hoisting of a sum's invariant operands
(`loopty.hoist`, #309) read loopy's domains, and was ported (item 10). The
rest crossed nothing new:

- the handles (#304) name loops, arrays, sizes and statements, and reach
  loopy as the inames and variables they resolve to;
- the branches (#319) are traced into loopty's own islpy sets: a path's
  conditions narrow its statement's domain, which reaches `lp.make_kernel`
  through `to_named_set` with the others; two branches whose union is not
  convex run over its hull, still loopty's set; and the predicate that cuts
  a statement back to its own points reaches loopy through
  `set_to_cond_expr` and `to_named`, as before;
- a select is pymbolic's `If` in the instruction, whose condition loopy
  main's bounds check reads itself;
- the statements' names (#316) are instruction ids;
- a schedule runs without the zeros no read sees (#318) by removing their
  instructions and loops with `lp.remove_instructions` and
  `lp.remove_unused_inames`;
- `CheckedOpenCLTarget`'s types (#291) are loopy's OpenCL registry, whose
  names loopy main keeps;
- the paths of a call (#326) are loopty's own sets.

Outside `src/`, `test_loopy_main`'s kernel with a guarded conversion branches
with `if`, since `with when(...)` is gone, and splits by a handle; and #318's
test of a device schedule's code reads its subscripts in namedisl's order,
`f[lid(0) + 4 * gid(0)]`.

This branch is where loopty runs on Python 3.14, and 0.4's reading of a
body's bytecode (`loopty.bytecode`, for `for box, a, b in R.dom:` and for a
`while` on a traced condition) did not know two of 3.14's layouts: a
`FOR_ITER` behind an `EXTENDED_ARG`, which `advection`'s long lift has on
3.14, and a `while` tested at the top of its loop. #332 fixes both on `main`,
and is merged here ahead of it.

When loopy releases with namedisl, the branch drops the Git sources, sets the
floors to the releases, and merges.
