"""Scoped storage requests preserve contraction and refuse unsupported controls."""

import pytest

from loopty.schedule import IllegalCast
from loopty.storage import UnsupportedStoragePlan
from test_window import WINDOWS


def test_single_window_matches_contract_and_keeps_the_call_site():
    schedule = WINDOWS.fused
    plan = schedule.storage(WINDOWS.burgers.f)
    plan.stage(name="window", placement=lambda j: j % 3, registers=False)
    result = schedule.materialize(plan)
    expected = schedule.contract(WINDOWS.burgers.f, lambda j: j % 3, registers=False)
    assert result.key == expected.key
    assert result._steps_at[-1].startswith("test_storage.py:")
    assert schedule.windows == {}
    assert plan.inspect()["lowering_limit"] is None


def test_illegal_window_leaves_source_and_plan_usable():
    schedule = WINDOWS.fused
    plan = schedule.storage(WINDOWS.burgers.f)
    plan.stage(name="short", placement=lambda j: j % 2)
    with pytest.raises(IllegalCast, match="stored in slot"):
        schedule.materialize(plan)
    assert schedule.windows == {}
    assert len(plan.stages) == 1
    assert schedule.contract(WINDOWS.burgers.f).windows["f"].extents == (3,)


def test_stage_routing_is_explicit_and_cross_plan_handles_are_rejected():
    schedule = WINDOWS.fused
    plan = schedule.storage(WINDOWS.burgers.f)
    use = WINDOWS.burgers.divergence
    tile = plan.stage(name="tile", uses=use, memory="local")
    fragment = plan.stage(name="fragment", source=tile, uses=use, memory="private")
    plan.serve(use, from_=fragment)
    assert plan.inspect()["routes"] == {repr(use): "fragment"}
    with pytest.raises(ValueError, match="different supplying stage"):
        plan.serve(use, from_=tile)
    other = schedule.storage(WINDOWS.burgers.f)
    with pytest.raises(ValueError, match="another plan"):
        other.stage(name="foreign", source=tile)
    assert other.stages == ()
    with pytest.raises(UnsupportedStoragePlan, match="exactly one"):
        schedule.materialize(plan)
    assert schedule.windows == {}


def test_stale_schedule_and_unsupported_memory_do_not_fall_back():
    schedule = WINDOWS.fused
    plan = schedule.storage(WINDOWS.burgers.f)
    plan.stage(name="window", memory="local")
    with pytest.raises(ValueError, match="different schedule snapshot"):
        schedule.retarget("opencl").materialize(plan)
    with pytest.raises(UnsupportedStoragePlan, match="explicit memory"):
        schedule.materialize(plan)


def test_owner_and_reuse_cannot_name_the_same_coordinate():
    plan = WINDOWS.fused.storage(WINDOWS.burgers.f)
    loop = WINDOWS.burgers.flux.j
    with pytest.raises(ValueError, match="both distinguish owners"):
        plan.stage(name="window", at=(loop,), per=(loop,), reuse_over=(loop,))
    assert plan.stages == ()
