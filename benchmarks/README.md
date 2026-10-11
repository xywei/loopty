# Benchmarks

Drivers that time a demo's kernels against the code they are written after,
and check that the two compute the same thing. They are not tests: their
numbers describe the machine that ran them, which is why no transcript of one
is kept in the repository, and CI runs none of them.

## l2p.py

sumpy's L2P for the 2-D Laplace kernel, `examples/l2p.py`, three ways on one
tree, one set of coefficients and one device:

- sumpy's own `E2PFromSingleBox(...).get_optimized_kernel()`, which needs
  sumpy and the loopy it is written for (loopy main, on namedisl), so it runs
  in an environment of its own;
- the same algorithm written directly in plain loopy, as a kernel author
  would (loopy #970's L2P microkernel with the Laplace step put back), with
  variants that make loopty's choices, so that each cost can be told apart.
  On the C target it is built with the flags loopty builds its kernels with,
  `-fwrapv` beside loopy's own, which changes the code GCC makes; `--no-wrapv`
  builds it with loopy's flags alone;
- loopty's schedules of the example, written out at the order asked for,
  lowered by loopty and called as the loopy kernel they are.

```console
$ python benchmarks/l2p.py inputs --order 10 --level 6 --out in.npz
$ python benchmarks/l2p.py loopty --inputs in.npz --target c --variant unrolled --save loopty.npz
$ python benchmarks/l2p.py loopy --inputs in.npz --target c --variant unrolled --save loopy.npz
$ PYOPENCL_CTX=0 python benchmarks/l2p.py sumpy --inputs in.npz --save sumpy.npz
$ python benchmarks/l2p.py compare in.npz loopty.npz loopy.npz sumpy.npz
```

Each run prints one JSON record: the device (platform, device, driver), the
versions that compiled it, the flags the C kernel was built with, the time of the
first call (code generation and compilation, in fresh caches with `--cold`),
and the median, minimum and maximum of the timed calls after the untimed ones.
`compare` gives the largest difference between every two potentials and to
the direct sum. `example` and `record` write what `examples/l2p.py`'s
`upstream` row compares with: sumpy's own potential on the example's inputs,
`examples/l2p_sumpy.json`. The module docstring has the details.
