# Device runs

loopty's C target compiles and runs locally, which is what the test suite uses.
This note records what happened when the same demos were compiled for
`lp.PyOpenCLTarget` and run on real OpenCL devices on 2026-09-18, and what the
compiled device runs agreed with.

The runs happened on a private remote compute host. Nothing here names it: the
devices are described by vendor and model class, which is what a reader needs to
judge the evidence, and the logs under `docs/device-runs/` have had the run
directory replaced with `<run>`.

## Devices

Two device classes, one physical machine, selected explicitly through
`PYOPENCL_CTX` rather than letting pyopencl choose. A run whose device nobody
wrote down is not evidence of anything.

| tag | platform | device | driver / build |
|---|---|---|---|
| `h200-cuda` | NVIDIA CUDA, `OpenCL 3.0 CUDA 13.3.44` | NVIDIA H200 NVL, GPU, 132 compute units, `cl_khr_fp64` | driver 610.43.02 |
| `pocl-epyc-cpu` | Portable Computing Language, `OpenCL 3.0 PoCL 7.0 ... LLVM 19.1.7, SLEEF` | AMD EPYC 9335, CPU, 128 compute units, `cl_khr_fp64` | PoCL 7.0 |

Software: Python 3.13.15, numpy 2.5.3, pymbolic 2025.1, islpy 2025.2.5,
loopy 2025.2, pyopencl 2026.1.4, lanky 0.1.0.dev0, loopty 0.1.0.dev0, in a fresh
virtual environment holding nothing but those. The islpy ceiling matters: loopy
2025.2 calls `Aff.is_equal` and `BasicMap.is_bijective`, which islpy 2026 removed,
so `islpy<2026` in `pyproject.toml` is what lets the skewed stencil compile at
all.

Every result below is identical on the two device classes. That is worth saying
plainly: nothing loopty did depended on the device, and the one refusal below is
loopy's, not the hardware's.

## The three commands, per demo

`python FILE`, `lanky check FILE` and `loopty run FILE --target opencl` were run
for each of `examples/reshape_layouts.py`, `examples/spmv.py` and
`examples/stencil_skew.py`. All nine exited 0 on both devices
(`docs/device-runs/h200-cuda-commands.log`,
`docs/device-runs/pocl-epyc-cpu-commands.log`).

One thing about that third command is easy to misread, so it is recorded here
rather than left to be rediscovered. `--target opencl` only reaches kernels that
no schedule in the file mentions. The demo files build their schedules with
`target="c"`, because the C target is what the local suite can compile, and
`loopty run` honours the target a schedule was built for. So
`loopty run examples/spmv.py --target opencl` put `scan` on the device and ran
`spmv` on the C target, and `loopty run examples/stencil_skew.py --target opencl`
ran `jacobi` on the C target. The exit code is 0 and the ledger is right, but the
flag did less than its name suggests.

That is why the device evidence below comes from
`docs/device-runs/device_demos.py`, which rebuilds the demos' own schedules with
`target="opencl"` and runs them. It is a driver, not a test: it takes the demo
modules as they are and only changes the target.

## What ran on a device, and what it agreed with

Each case compares the compiled device run against the demo's own Python body on
the same inputs, at the tolerance the exactness class states. Seven of eight
cases pass on both devices; the numbers below are byte-identical between the two.

| case | schedule | output | difference | tolerance | verdict |
|---|---|---|---|---|---|
| spmv, split and realize | `split("j", 2).realize("y", tree=True)` | `y` | 5.55e-17 | 2.46e-06 (`approx`) | agrees |
| spmv, rows across groups | `tag(r="g.0")` | `y` | 2.78e-17 | 2.46e-06 (`approx`) | agrees |
| scan | identity | `off` | 0 | 0 (`exact`) | agrees, bitwise |
| stencil, skewed and tiled | `skew("i", by="t").tile("t", "i", 8, 8)` | `u` | 0 | 1.55e-05 (`approx`) | agrees, bitwise |
| reshape, blocked transpose | `split("i", 2).interchange("i_out", "j", "i_in")` | `b` | 0 | 6.6e-05 (`approx`) | agrees, bitwise |
| reshape, row-major view | identity | `mat` | 0 | 6.6e-05 (`approx`) | agrees, bitwise |
| reshape, column-major view | identity | `mat` | 0 | 6.6e-05 (`approx`) | agrees, bitwise |
| spmv, the design note's device schedule | `tag(r="g.0").split("j", 32).tag(j_in="l.0").realize("y", tree=True)` | | | | **refused by loopy** |

The spmv product was also checked against the dense matrix written out by hand,
not only against the traced body: on the device, `max |y - A x| = 5.551e-17` over
a six-row random CSR matrix, which is one unit in the last place of the largest
entry. The blocked transpose was checked as `b == a.T` exactly, and came back
`True`.

Two things in that table are the point of the exercise. The stencil's skewed and
tiled schedule reproduces the Jacobi sweep *bitwise* on a GPU, which is what a
cast that only permutes instances should do; and the two spmv schedules that
reassociate the accumulation land at 1e-17, comfortably inside the reassociation
tolerance they asked for rather than at it. The `reassoc` allowance is not being
consumed.

The illegal cast is still refused with its witness when the target is a device,
which is the other half of the claim. Verbatim, from the device run:

```
tile(t,i,8,8) illegal: instance S0[t=6, i=8] writes u[7, 8] read by
S0[t=7, i=7] scheduled earlier
witness: (('S0', {'t': 6, 'i': 8}), ('S0', {'t': 7, 'i': 7}), {'nt': 16, 'nx': 16})
```

## The one failure: the design note's spmv device schedule

The design note specifies

```python
Schedule(spmv).tag(r="g.0").split("j", 32, inner="j_in").tag(j_in="l.0").realize("y", tree=True)
```

one work group per row, the entries of a row across the lanes of that group.
loopty accepts it. All nine cast facts come back `decided` by isl: each step is a
bijection on statement instances, each new order runs every dependence forward,
and the reassociation is licensed because the accumulation is `reassoc` rather
than `exact`. The schedule then does not reach a device, because loopy will not
generate code for it. Verbatim:

```
FAILED: LoopyError
Reduction over 'j_out, j_in' contains both parallel and sequential inames.
It must be split (using split_reduction_{in,out}ward) before code generation.
```

`docs/device-runs/reduction_probe.py` follows that up, and the answer is that
loopy's own remedy does not rescue it either. Five attempts, same on both devices
(`docs/device-runs/h200-cuda-reduction-probe.log`):

1. `tag(r="g.0")` alone: **runs**, agrees at 2.78e-17.
2. `tag(r="g.0").tag(j="l.0")`, the whole ragged reduction across lanes, without
   splitting: refused by `StaticValueFindingError`, "a numeric maximum was not
   found for PwAff `[n, nl_cnt_r] -> { [(nl_cnt_r)] : n > 0 and nl_cnt_r > 0 }`".
   A work-group axis needs a static extent and a row's length is data.
3. the design note's schedule unmodified: the `LoopyError` above.
4. the same, plus `lp.split_reduction_outward(knl, "j_out")`: loopy now builds
   exactly the tree that was asked for, a 32-wide local accumulator reduced in
   five stages, and then refuses it: "Domain number 1 has a data-dependent
   parameter `nl_cnt_r` and contains parallel inames `j_in`. This is not allowed
   (for now)."
5. the same, plus `lp.split_reduction_inward(knl, "j_in")`: the same refusal.

So the obstacle is not the mixed-tag reduction, which is a fixable shape; it is
that loopy 2025.2 does not put a hardware axis inside a domain whose bound comes
from an array. The ragged inner loop of a CSR product is exactly such a domain:
its bound is `cnt[r]`, reflected as the parameter `nl_cnt_r`. The parenthesis in
loopy's own message, "for now", says this is a restriction rather than a
principle.

What this does and does not mean:

- It is **not** a loopty bug in the checker. loopty's answer, that the schedule is
  a legal cast, is the right answer about the transformation; the refusal is a
  code-generation limit downstream of it.
- It **is** a gap between what loopty accepts and what it can run, and it is now
  detected. `schedule.py` asks a third question of every accepted step, about
  the target rather than about meaning: a parallel tag inside a domain whose
  bound comes from an array, or a reduction split across parallel and sequential
  inames, produces a `refuted` fact of kind `buildable` decided by
  `loopy-target` with the reason in words, and `UnbuildableSchedule` is raised
  as soon as anything asks the schedule for code. The measurement in this
  document is where both limits come from. What that check does *not* do is
  model the backend: it knows these two limits and no others, so a schedule it
  passes can still fail in code generation for a reason nobody has met.
- `examples/spmv.py` keeps this schedule behind `device_schedule()`, whose
  docstring now says what was measured here: it does not run on a device either.
  The demo prints its decided casts and the refused `buildable` fact side by
  side. `spmv.rows_parallel()` is the schedule for this shape that does build.

What does work on a device for a ragged product is one row per work group,
`tag(r="g.0")`, with the row's entries summed sequentially inside the group. That
ran, and it agreed.

## Reproducing

```bash
uv venv --python 3.13 .venv          # islpy<2026 has no cp314 wheels
uv pip install -e ../lanky -e . pyopencl
python -c "import pyopencl as cl; [print(i, p.name, [d.name for d in p.get_devices()]) for i, p in enumerate(cl.get_platforms())]"
PYOPENCL_CTX=<platform>:<device> python docs/device-runs/device_demos.py
PYOPENCL_CTX=<platform>:<device> python docs/device-runs/reduction_probe.py
```

Both drivers find the checkout by walking up from their own directory, or take
`LOOPTY_REPO`. Neither one needs the test suite, and neither imports pyopencl
until a device is actually asked for.

## Files

| file | what it holds |
|---|---|
| `device-runs/device_demos.py` | the driver: the demos' schedules rebuilt for `target="opencl"` |
| `device-runs/reduction_probe.py` | the five attempts at the design note's spmv schedule |
| `device-runs/h200-cuda-commands.log` | the nine `python` / `lanky check` / `loopty run` runs, GPU |
| `device-runs/h200-cuda-device-demos.log` | the device driver, GPU |
| `device-runs/h200-cuda-reduction-probe.log` | the reduction investigation, GPU |
| `device-runs/pocl-epyc-cpu-*.log` | the same three, PoCL on the CPU |

The device-demos and reduction-probe logs each open with a `NoSuchEntryError`
traceback from `pytools.persistent_dict`. That is loopy's transformation cache
missing on a cold run, not a failure; it reaches the log because stderr and
stdout are captured together. The real refusals are the `RESULT:` and `FAILED:`
lines.
