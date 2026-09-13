# loopty

loopy, with types.

loopty is loop + ty, for types: a typed polyhedral layer over
[loopy](https://github.com/inducer/loopy). Kernels are decorated Python
functions whose bodies run natively under plain `python` as the reference
implementation and are traced under `loopty` to build a typed term. Types are
isl objects: a statement's type is its iteration domain (an isl set) and its
read, write, and accumulation footprints (isl maps); dependences are derived by
isl flow analysis, not declared; index types include ragged, dependent shapes
(CSR-style data as dependent sums) so disjointness and in-bounds facts come from
the shape rather than from offset arithmetic; loop transformations are checked
as casts along bijections, with a concrete witness on failure; loopy generates
the code (OpenCL, CUDA, C). loopty is the first plugin for its sister project
[lanky](https://github.com/xywei/lanky) (a Python-hosted proof language over
Lean 4): loopty's typing rules emit facts into lanky's ledger, its isl oracle
decides the Presburger ones, and residual obligations become lanky theorems.

## Status

Work in progress. This is a placeholder release to reserve the name; nothing
works yet. loopty is designed as a plugin for
[lanky](https://github.com/xywei/lanky), the host it plugs into, and will land
as the interfaces on both sides settle.

## Name

loopty is loop + ty, for types: loops, typed. It follows `loopy`, `sumpy`, and
`pytato` in the naming tradition of the loopy ecosystem.

## Planned architecture

- Kernel bodies are traced: the decorated function runs natively under plain
  `python`, and the same body is traced under `loopty` to build a typed term.
- Index types are isl sets, with dependent sums for ragged data, so CSR-style
  shapes carry their own disjointness and in-bounds facts.
- Each statement carries read, write, and accumulation footprints; dependences
  are derived by isl flow analysis rather than declared.
- Loop transformations are checked as casts along bijections, with a concrete
  witness produced when a cast fails.
- Schedules and tags are maps into a target's execution and memory types.
- loopy is the code generator (OpenCL, CUDA, C).
- The plain `python` run is the reference implementation and the
  differential-test oracle.
- With lanky, loopty registers a theory, an isl oracle, an executor, and a `run`
  verb.

## Install

```sh
uv add loopty
```

```sh
pip install loopty
```

Nothing works yet: installing loopty today gets you a placeholder that prints
its own status.

## AI disclosure

This project is developed with substantial assistance from AI coding agents
(Anthropic's Claude, via Claude Code). Design, direction, and review are by
Xiaoyu Wei. Generated text and code are reviewed before release, but readers
should assume AI involvement throughout.

## License

MIT. See [LICENSE](LICENSE).
