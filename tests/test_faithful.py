"""The faithfulness fact: whether the traced term computes what the body computes.

Tracing refuses the hidden state it knows how to look for, and each of those
checks stops somewhere: one level into a container, the frame that runs the
``for``, the builtins it recognizes. The kernels below keep their state just
past those edges (issue #14's remaining cases), so each one traces, to a term
that is one iteration of what the body computes, and each one has to come out
``refuted`` by the fact, with the input and the first cell that differs.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from lanky.ledger import Status
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, reduce_sum, when
from loopty.faithful import KIND, SIZES, sample_arguments
from loopty.kernel import Kernel
from loopty.tolerance import disagreement


def faithful(fn):
    """The ``trace-faithful`` fact of ``fn`` as a kernel, which is its last fact."""
    fact = Kernel(fn).facts()[-1]
    assert fact.kind == KIND
    return fact


def assert_refuted(fact, cell: str = "y[0]") -> dict:
    """The fact is refuted at ``cell``, and says on which input."""
    assert fact.status is Status.REFUTED, fact.provenance
    assert fact.decided_by == "interpreter"
    counterexample = fact.provenance["counterexample"]
    assert counterexample["cell"] == cell
    assert counterexample["input"].startswith("sample 1 (")
    assert counterexample["body"] != counterexample["term"]
    assert "does not compute what the body computes" in fact.provenance["reason"]
    assert fact.provenance["inputs"] == [
        {"input": counterexample["input"], "outcome": "differed"}
    ]
    # A sampled input is small enough to write down, so it can be read back.
    assert set(fact.provenance["arguments"]) == {"x", "y"}
    return counterexample


# {{{ a faithful trace


def axpy(a: Real, x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in y.dom:
        y[i] = a * x[i] + y[i]


def test_a_faithful_kernel_is_tested_on_every_sample() -> None:
    fact = faithful(axpy)
    assert fact.status is Status.TESTED
    assert fact.decided_by == "interpreter"
    assert fact.statement == "the traced term computes what the body computes"
    assert fact.provenance["compared"] == 3
    assert [entry["outcome"] for entry in fact.provenance["inputs"]] == ["agreed"] * 3
    assert "counterexample" not in fact.provenance


def test_the_fact_is_the_last_of_a_kernels_facts() -> None:
    facts = Kernel(axpy).facts()
    assert [fact.kind for fact in facts].count(KIND) == 1
    assert facts[-1].id.endswith(":trace-faithful")


# }}}


# {{{ state the tracer does not see, refuted by the fact


def test_state_in_an_attribute_of_an_attribute_is_refuted() -> None:
    # The trace-wide snapshot compares the attributes of an object a name
    # holds, one level down; ``holder.inner`` is still the same object.
    holder = SimpleNamespace(inner=SimpleNamespace(s=0.0))

    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        for _i in x.dom:
            holder.inner.s = holder.inner.s + 1.0
        y[0] = holder.inner.s

    counterexample = assert_refuted(faithful(counted))
    assert counterexample["term"] == 1.0


def test_state_in_a_nested_container_is_refuted() -> None:
    # The loop snapshot copies ``acc`` one level deep, and ``acc[0]`` is the
    # same list after the iteration: the trace recorded ``y[0] = 1.0``.
    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        acc = [[0.0]]
        for _i in x.dom:
            acc[0][0] = acc[0][0] + 1.0
        y[0] = acc[0][0]

    counterexample = assert_refuted(faithful(counted))
    assert counterexample["term"] == 1.0
    n = int(counterexample["input"].split("n=")[1].rstrip(")"))
    assert counterexample["body"] == float(n)


def test_a_name_first_bound_behind_a_dir_probe_is_refused_by_the_fact() -> None:
    # ``dir()`` asks which names are bound, as ``locals()`` does, and is not
    # one of the probes the loop refuses: only the first iteration binds ``s``.
    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        for _i in x.dom:
            if "s" not in dir():
                s = 0.0
            s = s + 1.0
        y[0] = s

    assert_refuted(faithful(counted))


def test_a_loop_over_a_wrapping_generator_is_refuted() -> None:
    # The loop's snapshot is taken in the generator's frame, which carries
    # nothing; ``s`` lives in the body's frame and is never compared.
    def points(dom):
        yield from dom

    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        s = 0.0
        for _i in points(x.dom):
            s = s + 1.0
        y[0] = s

    assert_refuted(faithful(counted))


def test_state_in_a_deque_is_refuted() -> None:
    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        acc = deque([0.0])
        for _i in x.dom:
            acc.append(acc.pop() + 1.0)
        y[0] = acc[0]

    assert_refuted(faithful(counted))


def test_a_frame_probe_is_refuted() -> None:
    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        for _i in x.dom:
            if "s" not in sys._getframe().f_locals:
                s = 0.0
            s = s + 1.0
        y[0] = s

    assert_refuted(faithful(counted))


# }}}


# {{{ guards


def test_a_guard_against_a_real_scalar_is_tested() -> None:
    # The comparison used to be a constraint of the domain, with a an integer
    # parameter, and the interpreter would not fix a parameter at a drawn
    # value that is not an integer, so the fact was left assumed.
    def below(a: Real, y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(i < a):
                y[i] = 1.0

    fact = faithful(below)
    assert fact.status is Status.TESTED, fact.provenance
    assert fact.provenance["compared"] == 3


def test_a_guard_whose_native_value_is_an_integer_is_refuted() -> None:
    # ~ on the Python bool i > 0 is -2 or -1 natively, and 'not' in the trace.
    # The native run refuses the spelling, whatever the input, so the fact is
    # refuted with the refusal, which names the fix, and not skipped.
    def flipped(y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(~(i > 0)):
                y[i] = 1.0

    fact = faithful(flipped)
    assert fact.status is Status.REFUTED, fact.provenance
    assert fact.decided_by == "interpreter"
    counterexample = fact.provenance["counterexample"]
    assert counterexample["input"].startswith("sample 1 (")
    assert counterexample["body raised"].startswith("TraceError: the guard of")
    reason = fact.provenance["reason"]
    assert "the body, run natively, is refused" in reason
    assert "'i <= 0' for '~(i > 0)'" in reason
    assert fact.provenance["inputs"] == [
        {"input": counterexample["input"], "outcome": "differed"}
    ]


def test_a_native_refusal_refutes_where_the_term_raised_too() -> None:
    # The interpreted term reads x[n] in the second loop, and the body never
    # gets there: its first guard is refused at i = 0. The refusal is about
    # the spelling, so the fact is refuted, and the reason does not claim that
    # the term ran.
    def flipped_then_shifted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(~(i > 0)):
                y[i] = 1.0
        for i in y.dom:
            y[i] = x[i + 1]

    fact = faithful(flipped_then_shifted)
    assert fact.status is Status.REFUTED, fact.provenance
    counterexample = fact.provenance["counterexample"]
    assert counterexample["body raised"].startswith("TraceError: the guard of")
    assert counterexample["term raised"].startswith("IndexError")
    reason = fact.provenance["reason"]
    assert "the body, run natively, is refused" in reason
    assert "interpreted, runs" not in reason


# }}}


# {{{ when nothing can be compared


def test_a_kernel_whose_body_never_runs_is_assumed() -> None:
    # Every sampled input reads past the end natively, which says nothing
    # about whether the term is faithful.
    def shift(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            v[i] = u[i + 1]

    fact = faithful(shift)
    assert fact.status is Status.ASSUMED
    assert fact.decided_by is None
    assert fact.provenance["reason"].startswith("no input ran natively")
    assert all(
        entry["outcome"].startswith("skipped: the body raised IndexError")
        for entry in fact.provenance["inputs"]
    )


def _mystery(value):
    """A call the interpreter has no numpy counterpart for."""
    import pymbolic.primitives as prim

    if isinstance(value, prim.ExpressionNode):
        return prim.Call(prim.Variable("mystery"), (value,))
    return value


def test_a_term_the_interpreter_cannot_read_is_assumed() -> None:
    def mysterious(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = _mystery(x[i])

    fact = faithful(mysterious)
    assert fact.status is Status.ASSUMED
    assert "no numpy counterpart for the call mystery(x[i])" in (
        fact.provenance["reason"]
    )


# }}}


# {{{ the inputs


def spmv(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    col: Arr[Fin[n], Fin[cnt], Fin[m]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])


def test_samples_follow_the_declared_types() -> None:
    term = Kernel(spmv).term
    for index in range(5):
        arguments, sizes = sample_arguments(term, (0, index))
        assert set(sizes) == {"m", "n"}
        assert all(SIZES[0] <= value < SIZES[1] for value in sizes.values())
        cnt = arguments["cnt"].numpy()
        assert cnt.shape == (sizes["n"],)
        assert cnt.dtype == np.int64 and (cnt >= 0).all()
        # Both ragged arguments are laid out by the counts they name.
        for name in ("col", "val"):
            assert list(arguments[name].counts) == list(cnt)
        col = arguments["col"].numpy()
        assert ((0 <= col) & (col < sizes["m"])).all()
        assert arguments["x"].numpy().shape == (sizes["m"],)
    again, _ = sample_arguments(term, (0, 3))
    first, _ = sample_arguments(term, (0, 3))
    assert again["val"].numpy().tobytes() == first["val"].numpy().tobytes()


def test_a_scalar_named_like_a_size_is_that_size() -> None:
    def window(k: Nat, x: Arr[Fin[k], Real], y: Arr[Fin[k], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = x[i]

    arguments, sizes = sample_arguments(Kernel(window).term, (0, 0))
    assert arguments["k"] == sizes["k"] == len(arguments["x"].numpy())


GAP = '''
from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, kernel


@kernel
def counted(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):
    acc = [[0.0]]
    for _i in x.dom:
        acc[0][0] = acc[0][0] + 1.0
    y[0] = acc[0][0]


def example_inputs():
    return {"x": Arr.from_numpy(np.arange(3.0)), "y": Arr.zeros(1)}
'''


def test_check_refutes_the_trace_on_the_example_inputs(tmp_path, capsys) -> None:
    from lanky.cli import main as lanky_main

    path = tmp_path / "gap.py"
    path.write_text(GAP, encoding="utf-8")
    out_path = tmp_path / "ledger.json"
    code = lanky_main(["check", str(path), "--json", str(out_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "REFUTED counted at gap.py:" in out
    assert "the traced term computes what the body computes" in out
    assert (
        "{'input': 'example_inputs()', 'cell': 'y[0]', 'body': 3.0, 'term': 1.0}"
    ) in out
    facts = json.loads(out_path.read_text(encoding="utf-8"))
    (fact,) = [fact for fact in facts if fact["kind"] == KIND]
    assert fact["status"] == "refuted"
    assert fact["decided_by"] == "interpreter"
    assert fact["provenance"]["inputs"] == [
        {"input": "example_inputs()", "outcome": "differed"}
    ]
    # Everything else about the kernel is still decided: it is the term that
    # is wrong, and every other fact is about the term.
    others = {fact["status"] for fact in facts if fact["kind"] != KIND}
    assert others == {"decided"}


RUNS_ON_SAMPLES = '''
from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, kernel


@kernel
def doubled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
    for i in y.dom:
        y[i] = 2.0 * x[i]


def example_inputs():
    # One cell short: the native run reads past the end of ``x``.
    return {"x": Arr.from_numpy(np.arange(2.0)), "y": Arr.zeros(3)}
'''


def test_an_example_the_body_cannot_run_is_skipped(tmp_path) -> None:
    from lanky.check import check_path

    path = tmp_path / "short.py"
    path.write_text(RUNS_ON_SAMPLES, encoding="utf-8")
    (fact,) = [fact for fact in check_path(str(path)) if fact.kind == KIND]
    assert fact.status is Status.TESTED
    first, *samples = fact.provenance["inputs"]
    assert first["input"] == "example_inputs()"
    assert first["outcome"].startswith("skipped: the body raised")
    assert [entry["outcome"] for entry in samples] == ["agreed"] * 3


def total(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
    y[0] = reduce_sum(x[j] for j in x.dom)


def test_an_input_too_large_to_interpret_is_not_run_natively_either(
    monkeypatch,
) -> None:
    # The interpreter's work is bounded and the body's is not, so the bound is
    # asked first: a benchmark-sized example costs neither run.
    import loopty.faithful as faithful_module

    monkeypatch.setattr(faithful_module, "MAX_INSTANCES", 5)
    calls: list[dict] = []

    def body(**arguments) -> None:
        calls.append(arguments)

    arguments = {"x": Arr.from_numpy(np.ones(10)), "y": Arr.zeros(1)}
    outcome = faithful_module._compare(
        body, Kernel(total).term, "example_inputs()", arguments
    )
    assert outcome == (
        "skipped",
        "too large to interpret: more than 5 statement instances and "
        "reduction terms",
    )
    assert calls == []


def test_an_input_the_body_refuses_is_skipped_whatever_the_term_raised() -> None:
    # The term reads past the end of ``x`` on this input and so does the body:
    # the input says nothing about whether the term is the body.
    import loopty.faithful as faithful_module

    def shift(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            v[i] = u[i + 1]

    kernel = Kernel(shift)
    outcome = faithful_module._compare(
        kernel, kernel.term, "sample 1", {"u": Arr.zeros(3), "v": Arr.zeros(3)}
    )
    assert outcome is not None
    assert outcome[0] == "skipped"
    assert outcome[1].startswith("the body raised IndexError")


def test_example_inputs_that_are_not_a_dictionary_are_reported(tmp_path) -> None:
    from lanky.check import check_path

    path = tmp_path / "listed.py"
    path.write_text(
        RUNS_ON_SAMPLES.replace(
            'return {"x": Arr.from_numpy(np.arange(2.0)), "y": Arr.zeros(3)}',
            "return [Arr.zeros(3), Arr.zeros(3)]",
        ),
        encoding="utf-8",
    )
    (fact,) = [fact for fact in check_path(str(path)) if fact.kind == KIND]
    assert fact.status is Status.TESTED
    assert fact.provenance["inputs"][0] == {
        "input": "example_inputs()",
        "outcome": "skipped: example_inputs() did not return a dictionary",
    }


# }}}


# {{{ what agreeing means


def test_an_exact_output_is_compared_bit_for_bit() -> None:
    zeros = np.array([0.0, np.nan, 1.0])
    signed = np.array([-0.0, np.nan, 1.0])
    assert list(disagreement(signed, zeros, "exact")) == [True, False, False]
    assert list(disagreement(signed, zeros, "approx")) == [False, False, False]
    assert list(
        disagreement(np.array([3, 4]), np.array([3, 5]), "exact")
    ) == [False, True]
    # The last bit of a double is a difference.
    one = np.array([1.0])
    assert disagreement(np.nextafter(one, 2.0), one, "exact").all()


def test_two_nans_agree_whatever_their_sign_and_payload() -> None:
    # IEEE 754 leaves a NaN result's sign and payload open, and pymbolic's
    # ``-1*x`` for ``-x`` keeps the sign of a NaN that negation flips.
    nan = np.array([np.nan, np.nan, np.nan])
    other = np.array([-np.nan, np.nan, 1.0])
    other[1] = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), np.float64)[0]
    assert np.isnan(other[1])
    assert list(disagreement(other, nan, "exact")) == [False, False, True]
    # A complex cell is compared part by part, NaN included.
    want = np.array([complex(np.nan, 1.0), complex(np.nan, 1.0), complex(0.0, 1.0)])
    got = np.array([complex(-np.nan, 1.0), complex(np.nan, -1.0), complex(-0.0, 1.0)])
    assert list(disagreement(got, want, "exact")) == [False, True, True]


NEGATED = '''
from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, kernel


@kernel
def negated(x: Arr[Fin[n], Real.exact], y: Arr[Fin[n], Real.exact]):
    for i in y.dom:
        y[i] = -x[i]


def example_inputs():
    x = np.array([np.nan, np.inf, -0.0, 1.0])
    return {"x": Arr.from_numpy(x), "y": Arr.zeros(4)}
'''


def test_a_nan_an_exact_kernel_computes_is_not_a_difference(tmp_path) -> None:
    # The body negates, the term multiplies by -1, and the two NaNs differ in
    # their sign bit; every other cell, the infinity and the signed zero among
    # them, has the same bits in both runs.
    from lanky.check import check_path

    path = tmp_path / "negated.py"
    path.write_text(NEGATED, encoding="utf-8")
    (fact,) = [fact for fact in check_path(str(path)) if fact.kind == KIND]
    assert fact.status is Status.TESTED, fact.provenance
    assert fact.provenance["inputs"][0] == {
        "input": "example_inputs()",
        "outcome": "agreed",
    }


def test_an_approx_output_is_compared_at_its_tolerance() -> None:
    want = np.array([1.0, 1.0, np.inf, 0.0])
    got = np.array([1.0 + 1e-9, 1.1, np.inf, 1e-7])
    assert list(disagreement(got, want, "approx")) == [False, True, False, False]
    assert list(disagreement(got, want, "reassoc")) == [True, True, False, True]
    assert disagreement(np.zeros(2), np.zeros(3), "approx").all()


def test_a_complex_cell_with_a_nan_part_matches_part_by_part() -> None:
    # np.isnan of a complex is true when either part is, which used to let a
    # NaN real part excuse any imaginary part.
    want = np.array([complex(np.nan, 1.0), complex(np.nan, 1.0), complex(1.0, 2.0)])
    got = np.array([complex(np.nan, 1.0), complex(np.nan, 2.0), complex(1.0, 2.0)])
    for exactness in ("approx", "reassoc"):
        assert list(disagreement(got, want, exactness)) == [False, True, False]


# }}}


@pytest.mark.parametrize("sample", range(3))
def test_the_samples_of_a_ragged_kernel_run_natively(sample: int) -> None:
    # A sample the body cannot run is skipped, so a sampler that broke the
    # contract (a row length disagreeing with its counts) would leave the fact
    # untested rather than failing it; every sample has to run.
    kernel = Kernel(spmv)
    arguments, _sizes = sample_arguments(kernel.term, (0, sample))
    kernel(**arguments)
