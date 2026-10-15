import unittest
from speculative_sampling import (
    SpeculativeSamplingController, ControllerConfig, PredictionGroup, SourceDependency,
)


def complete(controller, job):
    groups = [PredictionGroup(
        i, i + 1, job.job_id, job.condition_version, payload=str(i))
        for i in range(job.retain_until, job.retain_until + job.chunk_size)]
    controller.publish(job.job_id, groups)
    controller.finish(job.job_id)
    return groups


class CompactionTests(unittest.TestCase):
    def test_hundred_chunks_same_commands_media_and_horizons_bounded_payloads(self):
        baseline, compact = [
            SpeculativeSamplingController("P1", ControllerConfig(5, compute_lead=1, refill_lead=1))
            for _ in range(2)]
        for _ in range(100):
            left, right = baseline.dispatch(), compact.dispatch()
            self.assertEqual(left, right)
            complete(baseline, left)
            complete(compact, right)
            for _ in range(5):
                self.assertEqual(baseline.deliver(), compact.deliver())
                for controller in (baseline, compact):
                    controller.advance_playback(controller.committed_until)
                compact.compact_played(max(0, compact.played_until - 2))
                for name in ("horizon", "played_until", "committed_until", "locked_until"):
                    self.assertEqual(getattr(baseline, name), getattr(compact, name))
                self.assertEqual(baseline.snapshot().media_ready, compact.snapshot().media_ready)
            self.assertLessEqual(len(compact.proposals), 2)
            self.assertLessEqual(len(compact.predictions), 2)

    def test_all_archived_still_refills_not_bootstrap_and_rejects_stale(self):
        c = SpeculativeSamplingController("P1", ControllerConfig(2))
        first = c.dispatch()
        old = complete(c, first)
        for _ in range(2):
            c.deliver()
        c.advance_playback(2)
        c.compact_played(2)
        self.assertFalse(c.media_ready(first.job_id, old))
        second = c.dispatch()
        self.assertEqual(second.retain_until, 2)
        self.assertNotEqual(second.reason.value, "bootstrap")

    def test_compaction_preserves_source_provenance_of_unplayed_media(self):
        c = SpeculativeSamplingController("P1", ControllerConfig(2))
        first = c.dispatch()
        complete(c, first)
        c.deliver()
        c.deliver()
        c.advance_playback(2)
        second = c.dispatch()
        groups = complete(c, second)
        from dataclasses import replace
        dependency = SourceDependency(first.job_id, 0, 1)
        c.proposals[second.job_id].media[2]["media"] = replace(
            groups[0], source_dependencies=(dependency,))
        c.compact_played(2)
        self.assertIn(first.job_id, c.proposals)
        self.assertTrue(c._dependencies_valid(
            c.proposals[second.job_id].media[2]["media"]))
        for _ in range(2):
            c.deliver()
        c.advance_playback(4)
        c.compact_played(4)
        self.assertEqual(c.proposals, {})

    def test_unplayed_cannot_be_compacted(self):
        c = SpeculativeSamplingController("P1", ControllerConfig(2))
        complete(c, c.dispatch())
        c.deliver()
        with self.assertRaises(ValueError):
            c.compact_played(1)

    def test_partial_updates_match_uncompacted_controller(self):
        baseline, compact = [
            SpeculativeSamplingController("P1", ControllerConfig(5, compute_lead=1, refill_lead=1))
            for _ in range(2)]
        for controller in (baseline, compact):
            complete(controller, controller.dispatch())
        for version in range(2, 102):
            for _ in range(2):
                self.assertEqual(baseline.deliver(), compact.deliver())
                for controller in (baseline, compact):
                    controller.advance_playback(controller.committed_until)
                compact.compact_played(max(0, compact.played_until - 2))
            baseline.update_condition(f"P{version}")
            compact.update_condition(f"P{version}")
            a, b = baseline.dispatch(), compact.dispatch()
            self.assertEqual(a, b)
            complete(baseline, a)
            complete(compact, b)
            self.assertEqual(baseline.snapshot().media_ready, compact.snapshot().media_ready)
            self.assertLessEqual(len(compact.proposals), 4)
            self.assertLessEqual(len(compact.conditions), 5)

    def test_event_sink_retains_audit_with_bounded_local_history(self):
        persisted = []
        c = SpeculativeSamplingController("P1", ControllerConfig(2),
                                event_sink=persisted.append, event_history_limit=3)
        complete(c, c.dispatch())
        for _ in range(2):
            c.deliver()
        self.assertGreater(len(persisted), 3)
        self.assertEqual(c.events, persisted[-3:])
