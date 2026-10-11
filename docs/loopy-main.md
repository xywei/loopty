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
code loopy generates differs in the two ways listed below.

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
`typing`, `hypotheses`, `oracle`, `domain`) stays on islpy: it builds isl
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
   the counted loops of `schedule`, and in the buildability checks that read a
   kernel's domains.
2. **loopy's reader takes a mapping of names**, `aff_from_expr(vars_to_aff,
   expr)` where 2025.2 took a space, and its guard,
   `with_aff_conversion_guard`, no longer catches a `TypeError`. The readings
   of `loopty.isl_reading` (note 23) decline with loopy's own
   `ExpressionToAffineConversionError`, as loopy declines a call, and a fourth
   reading turns the `TypeError` an operator isl has none of raises back into
   a decline: `when(i < i ^ 1)` failed in loopy's bounds check without it.
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
     sit in it. So a fusion after a `tile` builds as on `main`, and one after
     a `split` of the producer's outer loop builds too, which `main` refuses
     (#192: loopy 2025.2 puts the split loops last, `[j, i_outer, i_inner]`).
8. **Still one convex piece per domain.** loopy main keeps a domain as a set
   that may have several pieces, but its code generation still asks for one
   (`isl_helpers.convexify`), so loopty's one-basic-set restriction on the
   images of its maps stays.

## What differs in the generated code

`loopty run FILE --emit-code` for each example, against `main`:

- `composition`, `fusion`, `pairs`, `travel`: the same text.
- `spmv`, `p2p`, `reshape_layouts`, `stencil_skew`, `wavefront_acoustic`:
  - the terms of an affine expression in another order, `x[-1 + n + -1 * i]`
    for `x[-1 + -1 * i + n]` and `cnt[t_in + 4 * t_out]` for
    `cnt[4 * t_out + t_in]`, which is how loopy main prints an affine
    expression over namedisl's order of names;
  - and, where a split or a tile divides a size, plain division in place of
    loopy's floor-division helper, `(3 + n) / 4` for
    `loopy_floor_div_pos_b_int32(3 + n, 4)`, with the helper's definition
    gone and some conditions on the sizes dropped from the bounds: loopy
    simplifies with `n >= 0` (item 3). With no assumption on the sizes, loopy
    main writes the helper as 2025.2 does.

The ledgers (`lanky check --json`, `loopty run --json`) are the same, and so
is everything `python FILE`, `lanky check FILE` and `loopty run FILE` print.
One transcript excerpt in `quickstart.md` changed with the term order.

## Keeping the branch

Merge `main` into `port/loopy-main` as it moves. What `main` adds that reads a
loopy kernel's domains or bounds, or calls loopy's reader, goes through
`loopty.named`. When loopy releases with namedisl, the branch drops the Git
sources, sets the floors to the releases, and merges.
