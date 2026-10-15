import math

from speculative_sampling.models.h3 import H3SpeculativeSamplingLiveSession


def finish(session, seconds=1.0):
    task = session.plan()
    assert task is not None
    session.publish(task, task["proposal_frames"])
    assert session.complete(task, seconds)
    return task


def consume(session, target):
    cursor = session.committed
    while cursor < target:
        units = session.deliver(cursor)
        assert units
        cursor = units[-1].end
        session.heartbeat(cursor)


def test_bootstrap_refill_uses_boundary_image_and_drops_duplicate_frame():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=2,
    )
    first = finish(session)
    assert first["anchor"] is None
    assert session.horizon == 124
    assert len(session.units) == 124

    consume(session, 100)
    refill = session.plan()
    assert refill is not None
    assert refill["boundary"] == 124
    assert refill["anchor"]["source_frame"] == 123
    assert refill["trim_frames"] == 1
    session.publish(refill, 124)
    session.complete(refill, 1)
    assert session.horizon == 247
    assert len([unit for unit in session.units if unit.proposal == refill["proposal"]]) == 123


def test_prompt_update_branches_only_from_ready_future_image():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
    )
    finish(session)
    consume(session, 17)
    session.update_prompt("second")
    replacement = session.plan()
    assert replacement is not None
    assert replacement["action"] == "update"
    projected = session.committed + math.ceil(
        session.condition_ready_to_first_commit_seconds * 24
    )
    assert replacement["boundary"] >= projected
    assert replacement["boundary"] <= projected + 8
    assert replacement["anchor"] is not None
    old_horizon = 124
    assert session.horizon == replacement["boundary"] < old_horizon
    session.publish(replacement, 124)
    session.complete(replacement, 1)
    assert session.horizon == replacement["boundary"] + 123


def test_next_prompts_are_consumed_only_by_normal_continuations():
    session = H3SpeculativeSamplingLiveSession(
        "10A", "reference.png", initial_budget=1,
    )
    session.update_prompt("10A", next_prompts=["10B", "post battle"])
    first = finish(session)
    assert first["prompt"] == "10A"
    assert session.prompt == "10A"

    consume(session, 100)
    second = session.plan()
    assert second["action"] == "continuation"
    assert second["boundary"] == 124
    assert second["prompt"] == "10B"
    assert session.prompt == "10B"
    session.publish(second, second["proposal_frames"])
    assert session.complete(second, 1)

    consume(session, 223)
    third = session.plan()
    assert third["action"] == "continuation"
    assert third["boundary"] == 247
    assert third["prompt"] == "post battle"


def test_next_prompt_defaults_to_current_prompt():
    session = H3SpeculativeSamplingLiveSession(
        "current", "reference.png", initial_budget=1,
    )
    finish(session)
    consume(session, 100)
    continuation = session.plan()
    assert continuation["action"] == "continuation"
    assert continuation["prompt"] == "current"
    assert continuation["prompt_version"] == 1


def test_legacy_followup_alias_advances_on_continuation():
    session = H3SpeculativeSamplingLiveSession(
        "action", "reference.png", initial_budget=1,
    )
    session.set_followup_prompt(1, "idle")
    finish(session)
    assert session.prompt == "action"
    consume(session, 100)
    continuation = session.plan()
    assert continuation["prompt"] == "idle"


def test_h3_uses_independent_estimates_for_updates_and_continuations():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png",
        condition_ready_to_first_commit_seconds=1.4,
    )
    assert session.condition_estimate is not session.continuation_estimate
    session.condition_estimate.observe(0.8)
    ema = session.snapshot()["timeline_extension"]["ema"]
    assert ema["mode"] == "separate"
    assert ema["condition_observations"] == 1
    assert ema["continuation_observations"] == 0
    assert ema["condition_value_seconds"] != ema["continuation_value_seconds"]


def test_hot_scheduling_lead_resets_shared_ema_for_next_decision():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        condition_ready_to_first_commit_seconds=1.4,
        ema_margin_seconds=0.2,
    )
    finish(session)
    consume(session, 17)
    session.condition_estimate.observe(1.0)
    session.continuation_estimate.observe(1.1)
    assert session.condition_estimate.count == 1
    assert session.continuation_estimate.count == 1

    result = session.update_scheduling_lead(2.6)

    assert math.isclose(result["lead_seconds"], 2.6)
    assert result["condition_observations"] == 0
    assert result["continuation_observations"] == 0
    session.update_prompt("second")
    assert session.rewrite_target_frame == session.committed + math.ceil(2.6 * 24)


def test_hot_scheduling_lead_rejects_unsafe_values():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", ema_margin_seconds=0.2,
    )
    for value in (None, float("nan"), 0.1, 10.1):
        try:
            session.update_scheduling_lead(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid lead {value!r}")


def test_hot_split_scheduling_leads_update_independently():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png",
        condition_ready_to_first_commit_seconds=1.4,
        continuation_to_first_commit_seconds=1.8,
    )

    result = session.update_scheduling_lead(
        condition_seconds=1.2, continuation_seconds=1.7,
    )

    assert result["lead_seconds"] is None
    assert math.isclose(result["condition_lead_seconds"], 1.2)
    assert math.isclose(result["continuation_lead_seconds"], 1.7)
    assert math.isclose(session.condition_estimate.lead_seconds, 1.2)
    assert math.isclose(session.continuation_estimate.lead_seconds, 1.7)


def test_slow_outlier_only_inflates_refill_lead_not_prompt_lead():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
    )
    finish(session, seconds=4)
    consume(session, 17)
    session.update_prompt("second")

    replacement = session.plan()

    assert session.budget >= 5
    assert replacement is not None
    assert replacement["action"] == "update"
    assert session.controller.config.compute_lead == session._groups_for_seconds(
        session.condition_ready_to_first_commit_seconds
    )


def test_replaced_suffix_keeps_frame_identity_across_branch():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        condition_ready_to_first_commit_seconds=1.85,
    )
    finish(session)
    consume(session, 17)
    session.update_prompt("second")
    replacement = session.plan()
    session.publish(replacement, 124)
    session.complete(replacement, 1)

    replacement_groups = [
        group for group in session.controller.predictions
        if group.job_id == replacement["proposal"]
    ]
    live_group = replacement_groups[0]
    logical_end = live_group.end
    current_frame_end = live_group.payload.end
    assert session._frame_at(logical_end) == current_frame_end
    assert logical_end == current_frame_end
    assert session._logical_at_or_before(current_frame_end - 1) < logical_end
    assert session._logical_at_or_before(current_frame_end) == logical_end

    cursor = session.committed
    while cursor < current_frame_end:
        units = session.deliver(cursor)
        assert units
        cursor = units[-1].end
        assert cursor == session.committed
        session.heartbeat(cursor)


def test_partial_media_is_deliverable_without_finishing_job():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
    )
    task = session.plan()
    session.publish(task, 17)
    assert session.busy == task
    assert session.horizon == 17
    units = session.deliver(0)
    assert units
    assert units[-1].end <= 17


def test_short_native_proposal_is_a_complete_22_frame_arm():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=0.5, proposal_frames=22,
    )
    first = finish(session)
    assert first["proposal_frames"] == 22
    assert session.horizon == 22
    assert len(session.units) == 22

    consume(session, 22)
    continuation = session.plan()
    assert continuation is not None
    assert continuation["proposal_frames"] == 22
    assert continuation["trim_frames"] == 1
    session.publish(continuation, 22)
    session.complete(continuation, 0.5)
    assert session.horizon == 43


def test_jit_refill_uses_playback_cover_not_transport_commit():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
    )
    finish(session)

    cursor = 0
    while cursor < session.horizon:
        units = session.deliver(cursor)
        assert units
        cursor = units[-1].end

    assert session.committed == session.horizon
    assert session.played == 0
    assert session.plan() is None

    session.heartbeat(100)
    continuation = session.plan()
    assert continuation is not None
    assert continuation["boundary"] == session.horizon


def test_extrapolated_playhead_removes_heartbeat_cadence_from_jit_refill():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        condition_ready_to_first_commit_seconds=1.72,
        playhead_clock_mode="extrapolated",
        playhead_feedback_timeout_seconds=0.75,
    )
    finish(session)
    cursor = 0
    while cursor < 90:
        units = session.deliver(cursor, max_frames=1)
        assert units
        cursor = units[-1].end
    session.heartbeat(80)

    # At the acknowledged frame 44 frames remain, just outside the 42-frame
    # lead. Two tenths of continuous server-clock progress makes refill due.
    assert session.plan(now=session.playhead_feedback_at) is None
    continuation = session.plan(now=session.playhead_feedback_at + 0.2)

    assert continuation is not None
    assert continuation["action"] == "continuation"
    assert continuation["boundary"] == session.horizon


def test_extrapolated_playhead_stops_when_feedback_is_stale():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        condition_ready_to_first_commit_seconds=1.72,
        playhead_clock_mode="extrapolated",
        playhead_feedback_timeout_seconds=0.25,
    )
    finish(session)
    cursor = 0
    while cursor < 90:
        units = session.deliver(cursor, max_frames=1)
        cursor = units[-1].end
    session.heartbeat(80)

    assert session.plan(now=session.playhead_feedback_at + 0.3) is None


def test_extrapolated_playhead_never_passes_committed_media():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        playhead_clock_mode="extrapolated",
    )
    finish(session)
    units = session.deliver(0, max_frames=10)
    session.heartbeat(units[-1].end)

    predicted = session._scheduling_playhead(session.playhead_feedback_at + 0.5)

    assert predicted == session.committed == units[-1].end


def test_committed_clock_makes_rtc_continuation_symmetric_with_rewrite():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        condition_ready_to_first_commit_seconds=1.35,
        continuation_to_first_commit_seconds=1.35,
        playhead_clock_mode="committed",
    )
    finish(session)
    cursor = 0
    due_at = session.horizon - math.ceil(1.35 * 24)
    while cursor < due_at:
        units = session.deliver(cursor, max_frames=1)
        cursor = units[-1].end

    continuation = session.plan()

    assert continuation is not None
    assert continuation["action"] == "continuation"
    assert continuation["boundary"] == session.horizon
    session.cancel(continuation)
    session.update_prompt("second")
    assert session.rewrite_target_frame == session.committed + math.ceil(1.35 * 24)


def test_prompt_one_frame_past_horizon_uses_ready_horizon_not_full_bridge():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        condition_ready_to_first_commit_seconds=1.08,
        continuation_to_first_commit_seconds=1.08,
        playhead_clock_mode="committed",
        rewrite_horizon_fallback=True,
    )
    finish(session)
    session.proposal_metrics[99] = {
        "proposal": 99, "action": "update", "first_ready_s": 0.81,
    }
    cursor = 0
    while cursor < 99:
        units = session.deliver(cursor, max_frames=1)
        cursor = units[-1].end
        session.heartbeat(cursor)

    session.update_prompt("second")
    replacement = session.plan()

    assert session.committed == 99
    assert session.horizon == 124
    assert 124 < 99 + math.ceil(1.08 * 24)
    assert session.rewrite_target_frame == 124
    assert replacement["action"] == "update"
    assert replacement["boundary"] == 124


def test_prompt_keeps_bridge_fallback_when_ready_horizon_is_truly_too_close():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        condition_ready_to_first_commit_seconds=1.08,
        continuation_to_first_commit_seconds=1.08,
        playhead_clock_mode="committed",
        rewrite_horizon_fallback=True,
    )
    finish(session)
    session.proposal_metrics[99] = {
        "proposal": 99, "action": "update", "first_ready_s": 0.81,
    }
    cursor = 0
    while cursor < 110:
        units = session.deliver(cursor, max_frames=1)
        cursor = units[-1].end
        session.heartbeat(cursor)

    session.update_prompt("second")
    fallback = session.plan()

    assert session.rewrite_target_frame == 110 + math.ceil(1.08 * 24)
    assert fallback["action"] == "bridge"
    assert fallback["boundary"] == 124


def test_browser_snapshot_keeps_controller_history_compact():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
    )
    finish(session)
    snapshot = session.snapshot()
    assert "predictions" not in snapshot["controller"]
    assert snapshot["controller"]["prediction_groups"] == 124


def test_heartbeat_compacts_played_history_and_bounds_session_caches():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
    )
    for _ in range(6):
        finish(session)
        target = max(session.committed, session.horizon - 20)
        consume(session, target)

    assert session.controller.archived_until > 0
    assert len(session.controller.proposals) <= 2
    assert len(session.units) <= session.proposal_frames + 24
    assert len(session.controller.events) <= 512
    assert session.delivery_resume_cursor <= session.played


def test_prompt_keeps_useful_continuation_then_dispatches_latest_update():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
    )
    finish(session)
    consume(session, 100)
    continuation = session.plan()
    assert continuation["action"] == "continuation"
    # The old continuation is useful cover for the projected safe point and
    # must finish instead of being cancelled into a two-generation bridge.
    session.update_prompt("second")
    assert session.rewrite_target_frame > continuation["boundary"]
    assert not session.should_cancel_inflight(continuation)
    session.publish(continuation, continuation["proposal_frames"])
    session.complete(continuation, 1)
    replacement = session.plan()
    assert replacement["action"] == "update"
    assert replacement["boundary"] == session.rewrite_target_frame


def test_prompt_cancels_inflight_continuation_when_ready_future_covers_cut():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        condition_ready_to_first_commit_seconds=0.5,
    )
    finish(session)
    cursor = session.committed
    while cursor < 112:
        units = session.deliver(cursor, max_frames=1)
        cursor = units[-1].end
        session.heartbeat(cursor)
    continuation = session.plan()
    assert continuation["action"] == "continuation"
    session.update_prompt("second")
    assert session.rewrite_target_frame == 124
    assert continuation["boundary"] == 124
    assert session.should_cancel_inflight(continuation)
    assert session.cancel(continuation)
    replacement = session.plan()
    assert replacement["action"] == "update"
    assert replacement["boundary"] == 124


def test_prompt_update_target_is_frozen_while_continuation_finishes():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        condition_ready_to_first_commit_seconds=1.0,
    )
    finish(session)
    consume(session, 100)
    continuation = session.plan()
    session.update_prompt("second")
    target = session.rewrite_target_frame
    session.publish(continuation, continuation["proposal_frames"])
    session.complete(continuation, 1)
    consume(session, target)

    replacement = session.plan()

    assert replacement["boundary"] == target


def test_base_policy_waits_and_appends_only_at_proposal_horizon():
    session = H3SpeculativeSamplingLiveSession(
        "first", "reference.png", initial_budget=1,
        serving_policy="base",
    )
    finish(session)
    session.update_prompt("second")
    assert session.plan() is None
    consume(session, 100)
    replacement = session.plan()
    assert replacement is not None
    assert replacement["boundary"] == 124
