"""The theorem matcher's higher-order rule, which loopty shares with lanky's proofs.

A theorem's family applied to the variables its own binders bind matches a
term of the program as a function of the program's variables lined up with
them (``lanky.matching.higher_order_value``): ``sum(f[k] for k in Fin[n])``
matches ``sum(x[k] * y[k] for k in Fin[m])`` with ``f`` the function ``k ↦
x[k] * y[k]``. The rule is off unless asked for, a family is bound to an
array where it can be, and a theorem's statement put in the program's terms
applies the function where it applied the family. None of this needs Lean.
"""

from __future__ import annotations

from lanky.matching import FamilyValue
from lanky.prelude import Fin
from lanky.terms import Sum, Var, render

from loopty.hypotheses import _apply, _Matcher

f, g, n, k, j = Var("f"), Var("g"), Var("n"), Var("k"), Var("j")
x, y, m, s = Var("x"), Var("y"), Var("m"), Var("s")


def theorem_sum(family=f, bound=n):
    """``sum(f[k] for k in Fin[n])``, a theorem's side."""
    return Sum(((k, Fin[bound]),), family[k])


def test_a_family_matches_a_product_as_a_function_of_its_index() -> None:
    """``f`` is ``k ↦ x[k] * y[k]`` and ``n`` is ``m``."""
    target = Sum(((j, Fin[m]),), x[j] * y[j])
    matcher = _Matcher({"f"}, {"n"}, higher_order=True)
    sigma: dict = {}
    assert matcher.match(theorem_sum(), target, sigma, {})
    assert isinstance(sigma["f"], FamilyValue)
    assert sigma["f"].shown() == "lambda j: x[j]*y[j]"
    assert render(sigma["n"]) == "m"
    # the statement in the program's terms applies the function
    claim = theorem_sum() == s
    assert render(_apply(claim, sigma, {"f"})) == "sum(x[k]*y[k] for k in Fin(m)) == s"


def test_the_rule_is_off_unless_asked_for() -> None:
    """Without ``higher_order``, a family matches an array read alone, as before."""
    target = Sum(((j, Fin[m]),), x[j] * y[j])
    sigma: dict = {}
    assert not _Matcher({"f"}, {"n"}).match(theorem_sum(), target, sigma, {})


def test_an_array_is_bound_where_it_can_be() -> None:
    """``f[k]`` against ``x[j]`` binds ``f`` to the array ``x``, the rule on or off."""
    target = Sum(((j, Fin[m]),), x[j])
    for higher_order in (False, True):
        sigma: dict = {}
        assert _Matcher({"f"}, {"n"}, higher_order=higher_order).match(
            theorem_sum(), target, sigma, {}
        )
        assert sigma["f"] == "x"


def test_a_function_is_one_function_for_the_whole_side() -> None:
    """A family used twice binds once; its second use is the same function there."""
    pattern = Sum(((k, Fin[n]),), f[k] + f[k])
    matcher = _Matcher({"f"}, {"n"}, higher_order=True)
    same = Sum(((j, Fin[m]),), x[j] * y[j] + x[j] * y[j])
    sigma: dict = {}
    assert matcher.match(pattern, same, sigma, {})
    differs = Sum(((j, Fin[m]),), x[j] * y[j] + x[j])
    assert not matcher.match(pattern, differs, {}, {})


def test_a_variable_of_another_binder_does_not_escape() -> None:
    """A value mentioning a variable no argument of the family stands for is refused."""
    pattern = Sum(((k, Fin[n]), (Var("i"), Fin[n])), f[k])
    target = Sum(((j, Fin[m]), (Var("l"), Fin[m])), x[j] * y[Var("l")])
    assert not _Matcher({"f"}, {"n"}, higher_order=True).match(pattern, target, {}, {})
