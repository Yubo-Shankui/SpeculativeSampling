import unittest

from speculative_sampling import (
    BoundaryReadiness, SpeculativeSamplingController, ControllerConfig,
    ExactBoundaryAdapter, OutputGroup, PredictionGroup,
)


class ReadyAdapter(ExactBoundaryAdapter):
    def boundary_readiness(self, boundary):
        return BoundaryReadiness(True, True, True, (0.5, 0.8))


def mapping(start=0, count=5, *, audio=False, dependency=None):
    return [OutputGroup(i, i + 1, i, i + 1, i / 3, (i + 1) / 3,
                        audio, dependency) for i in range(start, start + count)]


def ready(command, start, end=None, **kw):
    return PredictionGroup(start, start + 1 if end is None else end,
                           command.job_id, command.condition_version, **kw)


class SplitProtocolTests(unittest.TestCase):
    def make(self, **kwargs):
        return SpeculativeSamplingController("P1", ControllerConfig(5, compute_lead=2), **kwargs)

    def test_finish_before_media_and_late_retained_callbacks(self):
        c = self.make(boundary_adapter=ReadyAdapter())
        first = c.dispatch()
        c.latents_ready(first.job_id, mapping(), native_count=5)
        c.finish(first.job_id)
        self.assertEqual(c.snapshot().generated_horizon, 5)
        self.assertEqual(c.snapshot().media_ready, 0)
        c.update_condition("P2")
        second = c.dispatch(deadline_plan=(0.6, 1))
        self.assertEqual(second.retain_until, 2)
        self.assertTrue(c.media_ready(first.job_id, [ready(first, 0), ready(first, 1)]))
        self.assertFalse(c.media_ready(first.job_id, [ready(first, 2)]))
        self.assertEqual(c.deliver()[0].job_id, first.job_id)
        self.assertEqual(c.active_job, second)

    def test_av_holes_and_duplicates(self):
        c = self.make()
        job = c.dispatch()
        c.latents_ready(job.job_id, mapping(audio=True), native_count=5)
        c.finish(job.job_id)
        c.media_ready(job.job_id, [ready(job, 1)], kind="media")
        c.media_ready(job.job_id, [ready(job, 0, payload="original")], kind="video")
        self.assertEqual(c.deliver(), ())
        self.assertEqual(c.snapshot().media_ready, 0)
        c.media_ready(job.job_id, [ready(job, 0)], kind="audio")
        c.media_ready(job.job_id, [ready(job, 0, payload="duplicate")], kind="video")
        self.assertEqual(c.deliver()[0].payload, "original")
        self.assertEqual(c.snapshot().media_ready, 2)
        ready_events = [event for event in c.events if event["event"] == "MediaReady"]
        self.assertEqual([event["start"] for event in ready_events], [1, 0])

    def test_trimmed_output_M_not_K(self):
        c = self.make()
        job = c.dispatch()
        c.latents_ready(job.job_id, mapping(count=4), native_count=5)
        c.finish(job.job_id)
        self.assertEqual(c.horizon, 4)
        c.media_ready(job.job_id, [ready(job, 0)])
        self.assertEqual(c.deliver()[0].end, 1)

    def test_deadlines_reject_without_losing_old_predictions(self):
        c = self.make(boundary_adapter=ReadyAdapter())
        job = c.dispatch()
        c.latents_ready(job.job_id, mapping(), native_count=5)
        c.finish(job.job_id)
        c.update_condition("P2")
        self.assertIsNone(c.dispatch(deadline_plan=(0.4, 1)))
        self.assertEqual(c.horizon, 5)
        self.assertEqual(c.pending_condition_version, 2)

    def test_decoder_dependency_is_protected(self):
        c = self.make(boundary_adapter=ReadyAdapter())
        job = c.dispatch()
        groups = mapping()
        groups[0] = mapping(count=1, dependency=3)[0]
        c.latents_ready(job.job_id, groups, native_count=5)
        c.finish(job.job_id)
        c.media_ready(job.job_id, [ready(job, 0)])
        c.deliver()
        c.update_condition("P2")
        self.assertEqual(c.dispatch().retain_until, 3)

    def test_unencoded_pending_and_busy_coalescing(self):
        c = self.make(boundary_adapter=ReadyAdapter())
        job = c.dispatch()
        c.update_condition({"handle": "P2"}, ready=False)
        version = c.update_condition({"handle": "P3"}, ready=False)
        self.assertEqual(job.condition, "P1")
        c.latents_ready(job.job_id, mapping(), native_count=5)
        c.finish(job.job_id)
        self.assertIsNone(c.dispatch())
        c.mark_condition_ready(version, {"handle": "encoded-P3"})
        self.assertEqual(c.dispatch().condition, {"handle": "encoded-P3"})

    def test_failure_holds_slot_until_all_workers_terminate(self):
        c = self.make()
        job = c.dispatch()
        c.job_failed(job.job_id, "rank failure")
        self.assertEqual(c.state, "Terminating")
        self.assertIsNone(c.dispatch())
        self.assertFalse(c.finish(job.job_id))
        c.worker_terminated(job.job_id)
        self.assertIsNone(c.active_job)
        self.assertTrue(c.snapshot().recovery)
        self.assertIsNotNone(c.dispatch())

    def test_native_chunk_must_be_complete(self):
        c = self.make()
        job = c.dispatch()
        with self.assertRaises(ValueError):
            c.latents_ready(job.job_id, mapping(), native_count=1)
        self.assertEqual(c.horizon, 0)

    def test_audio_dependency_is_protected_even_when_video_ready_first(self):
        c = self.make(boundary_adapter=ReadyAdapter())
        job = c.dispatch()
        c.latents_ready(job.job_id, mapping(audio=True), native_count=5)
        c.finish(job.job_id)
        c.media_ready(job.job_id, [ready(job, 0, dependency_end=1)], kind="video")
        c.media_ready(job.job_id, [ready(job, 0, dependency_end=4)], kind="audio")
        c.deliver()
        c.update_condition("P2")
        self.assertEqual(c.dispatch().retain_until, 4)

    def test_same_batch_duplicates_cannot_overwrite_first_payload(self):
        c = self.make()
        job = c.dispatch()
        c.latents_ready(job.job_id, mapping(), native_count=5)
        c.finish(job.job_id)
        c.media_ready(job.job_id, [ready(job, 0, payload="first"),
                                   ready(job, 0, payload="overwrite")])
        self.assertEqual(c.proposals[job.job_id].media[0]["media"].payload, "first")
        self.assertEqual(c.deliver()[0].payload, "first")

    def test_deadlines_recomputed_for_each_candidate_and_fixed_on_command(self):
        c = self.make(boundary_adapter=ReadyAdapter())
        job = c.dispatch()
        c.latents_ready(job.job_id, mapping(), native_count=5)
        c.finish(job.job_id)
        c.update_condition("P2")
        visited = []

        def deadlines(boundary):
            visited.append(boundary)
            return (boundary / 5, boundary / 5 + .2)

        command = c.dispatch(deadline_plan=deadlines)
        self.assertEqual(command.retain_until, 3)
        self.assertEqual(command.deadline_plan, (.6, .8))
        self.assertIn(2, visited)
        self.assertIn(3, visited)

    def test_invalid_deadline_does_not_mutate_existing_prediction(self):
        c = self.make(boundary_adapter=ReadyAdapter())
        job = c.dispatch()
        c.latents_ready(job.job_id, mapping(), native_count=5)
        c.finish(job.job_id)
        c.update_condition("P2")
        with self.assertRaises(ValueError):
            c.dispatch(deadline_plan=lambda _: (float("nan"), 1))
        self.assertEqual(c.horizon, 5)
        self.assertEqual(c.pending_condition_version, 2)
        self.assertIsNone(c.active_job)


if __name__ == "__main__":
    unittest.main()
