import unittest

from speculative_sampling import (
    SpeculativeSamplingController,
    ControllerConfig,
    GenerationReason,
    PredictionGroup,
)


def complete_chunk(controller, command):
    groups = [
        PredictionGroup(
            start=command.retain_until + index,
            end=command.retain_until + index + 1,
            job_id=command.job_id,
            condition_version=command.condition_version,
            payload=f"job-{command.job_id}-group-{index}",
        )
        for index in range(command.chunk_size)
    ]
    assert controller.publish(command.job_id, groups)
    assert controller.finish(command.job_id)


class SpeculativeSamplingControllerTests(unittest.TestCase):
    def make_controller(self):
        return SpeculativeSamplingController(
            "prompt A",
            ControllerConfig(chunk_size=5, delivery_size=1, compute_lead=2, refill_lead=1),
        )

    def bootstrapped(self):
        controller = self.make_controller()
        command = controller.dispatch()
        self.assertEqual(command.reason, GenerationReason.BOOTSTRAP)
        self.assertEqual(command.retain_until, 0)
        complete_chunk(controller, command)
        return controller

    def test_bootstrap_samples_one_complete_native_chunk(self):
        controller = self.bootstrapped()
        self.assertEqual(controller.horizon, 5)
        self.assertEqual(controller.applied_condition_version, 1)
        self.assertIsNone(controller.dispatch())

    def test_delivery_is_one_group_not_one_generation(self):
        controller = self.bootstrapped()
        first = controller.deliver()
        self.assertEqual([(group.start, group.end) for group in first], [(0, 1)])
        self.assertEqual(controller.committed_until, 1)
        self.assertEqual(controller.horizon, 5)

    def test_update_uses_same_cutoff_for_predictions_and_generator(self):
        controller = self.bootstrapped()
        controller.deliver()
        controller.advance_playback(1)
        controller.update_condition("prompt B")
        command = controller.dispatch()
        self.assertEqual(command.reason, GenerationReason.UPDATE)
        self.assertEqual(command.retain_until, 3)
        self.assertEqual(controller.horizon, 3)
        self.assertEqual(controller.snapshot().discarded_groups, 2)
        self.assertEqual(command.condition, "prompt B")

    def test_busy_updates_coalesce_without_mutating_active_job(self):
        controller = self.bootstrapped()
        controller.update_condition("prompt B")
        active = controller.dispatch()
        controller.update_condition("prompt C")
        controller.update_condition("prompt D")
        self.assertIsNone(controller.dispatch())
        self.assertEqual(active.condition, "prompt B")
        complete_chunk(controller, active)
        next_command = controller.dispatch()
        self.assertEqual(next_command.condition, "prompt D")
        self.assertEqual(next_command.condition_version, 4)

    def test_short_future_bridges_old_condition_until_update_is_safe(self):
        controller = SpeculativeSamplingController(
            "prompt A",
            ControllerConfig(chunk_size=3, delivery_size=1, compute_lead=5, refill_lead=3),
        )
        first = controller.dispatch()
        complete_chunk(controller, first)
        controller.update_condition("prompt B")

        bridge = controller.dispatch()
        self.assertEqual(bridge.reason, GenerationReason.BRIDGE)
        self.assertEqual(bridge.condition_version, 1)
        self.assertEqual(bridge.retain_until, 3)
        self.assertEqual(controller.pending_condition_version, 2)
        complete_chunk(controller, bridge)

        update = controller.dispatch()
        self.assertEqual(update.reason, GenerationReason.UPDATE)
        self.assertEqual(update.condition_version, 2)
        self.assertEqual(update.retain_until, 5)

    def test_bridge_that_cannot_gain_on_playback_falls_back_to_update_recovery(self):
        controller = SpeculativeSamplingController(
            "prompt A",
            ControllerConfig(chunk_size=3, delivery_size=1, compute_lead=5, refill_lead=3),
        )
        first = controller.dispatch()
        complete_chunk(controller, first)
        controller.update_condition("prompt B")

        bridge = controller.dispatch()
        self.assertEqual(bridge.reason, GenerationReason.BRIDGE)
        complete_chunk(controller, bridge)
        for _ in range(3):
            controller.deliver()

        update = controller.dispatch()

        self.assertEqual(update.reason, GenerationReason.RECOVERY)
        self.assertEqual(update.condition_version, 2)
        self.assertEqual(update.retain_until, 6)
        self.assertIsNone(controller.pending_condition_version)

    def test_stale_callback_cannot_reinstall_invalidated_suffix(self):
        controller = self.bootstrapped()
        controller.update_condition("prompt B")
        active = controller.dispatch()
        stale = PredictionGroup(
            start=active.retain_until,
            end=active.retain_until + 1,
            job_id=active.job_id + 100,
            condition_version=active.condition_version,
        )
        self.assertFalse(controller.publish(stale.job_id, [stale]))
        self.assertEqual(controller.horizon, active.retain_until)

    def test_committed_prefix_cannot_be_rewritten(self):
        controller = self.bootstrapped()
        for position in range(1, 4):
            controller.deliver()
            controller.advance_playback(position)
        controller.update_condition("prompt B")
        command = controller.dispatch()
        self.assertGreaterEqual(command.retain_until, controller.committed_until)

    def test_refill_is_separate_from_delivery(self):
        controller = self.bootstrapped()
        self.assertIsNone(controller.dispatch())
        for position in range(1, 5):
            controller.deliver()
            controller.advance_playback(position)
        command = controller.dispatch()
        self.assertEqual(command.reason, GenerationReason.REFILL)
        self.assertEqual(command.retain_until, 5)

    def test_backend_commit_triggers_refill_without_browser_playback(self):
        controller = self.bootstrapped()
        for _ in range(4):
            controller.deliver()
        self.assertEqual(controller.played_until, 0)
        command = controller.dispatch()
        self.assertEqual(command.reason, GenerationReason.REFILL)
        self.assertEqual(command.retain_until, 5)

    def test_finish_requires_exactly_one_native_chunk(self):
        controller = self.make_controller()
        command = controller.dispatch()
        partial = [
            PredictionGroup(i, i + 1, command.job_id, command.condition_version)
            for i in range(4)
        ]
        controller.publish(command.job_id, partial)
        with self.assertRaises(ValueError):
            controller.finish(command.job_id)

    def test_generator_kv_frontier_never_moves_back_after_finish(self):
        controller = self.bootstrapped()
        controller.update_condition("prompt B")
        command = controller.dispatch()
        self.assertEqual(command.retain_until, 2)
        complete_chunk(controller, command)
        self.assertEqual(controller.locked_until, 2)
        controller.update_condition("prompt C")
        next_command = controller.dispatch()
        self.assertGreaterEqual(next_command.retain_until, 2)

    def test_slow_compute_bridges_until_explicit_recovery_if_sender_exhausts(self):
        controller = SpeculativeSamplingController(
            "prompt A",
            ControllerConfig(chunk_size=5, delivery_size=1, compute_lead=99, refill_lead=1),
        )
        first = controller.dispatch()
        complete_chunk(controller, first)
        controller.update_condition("prompt B")
        update = controller.dispatch()
        self.assertIsNone(update)
        self.assertEqual(controller.pending_condition_version, 2)
        self.assertEqual(controller.horizon, 5)
        for position in range(1, 5):
            controller.deliver()
            controller.advance_playback(position)
        bridge = controller.dispatch()
        self.assertEqual(bridge.reason, GenerationReason.BRIDGE)
        self.assertEqual(bridge.condition_version, 1)
        controller.cancel_active(bridge.job_id)
        controller.pending_condition_version = 2
        controller.deliver()
        controller.advance_playback(5)
        recovery = controller.dispatch()
        self.assertEqual(recovery.reason, GenerationReason.RECOVERY)
        self.assertEqual(recovery.retain_until, 5)


if __name__ == "__main__":
    unittest.main()
