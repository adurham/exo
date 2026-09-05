#!/usr/bin/env python3
"""Synthetic self-test for parse_worker_marks.py.

STDLIB unittest, no pytest needed. Run as:
    /usr/bin/python3 test_parse_worker_marks.py -v

All input lines below are hand-constructed in the emitter's EXACT format
(see the quoted format string at the top of parse_worker_marks.py):

    "PHASE_MARK state_applied event_idx=<int> t=<float %.6f>"
    "PHASE_MARK plan_step_observed event_idx=<int> t=<float %.6f> wake_kind=<kind>"

A loguru-style preamble is prepended to some lines to prove the parser
locates the PHASE_MARK payload regardless of what precedes it.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parse_worker_marks as pwm  # noqa: E402


def sa(event_idx, t, preamble="2026-09-04 12:00:00.000 | INFO | worker:run:1 - "):
    return "%sPHASE_MARK state_applied event_idx=%d t=%.6f" % (preamble, event_idx, t)


def pso(event_idx, t, wake_kind, task=None, preamble="2026-09-04 12:00:00.000 | INFO | worker:run:1 - "):
    if task is None:
        # Old-format line (pre-round-13-A2 instrumentation): no task= field.
        return "%sPHASE_MARK plan_step_observed event_idx=%d t=%.6f wake_kind=%s" % (
            preamble,
            event_idx,
            t,
            wake_kind,
        )
    return "%sPHASE_MARK plan_step_observed event_idx=%d t=%.6f wake_kind=%s task=%s" % (
        preamble,
        event_idx,
        t,
        wake_kind,
        task,
    )


class TestParsing(unittest.TestCase):
    def test_ignores_noise_lines(self):
        lines = [
            "some unrelated log line about zenoh peers",
            sa(1, 100.000000),
            "2026-09-04 12:00:00 | DEBUG | some other module - nothing to see",
            pso(1, 100.010000, "event"),
            "",
            "   ",
        ]
        marks = pwm.parse_lines(lines)
        self.assertEqual(len(marks), 2)
        self.assertEqual(marks[0].kind, "state_applied")
        self.assertEqual(marks[1].kind, "plan_step_observed")


class TestPairingAndStats(unittest.TestCase):
    def test_simple_1to1_pairing(self):
        """(1) simple 1:1 pairing."""
        lines = [
            sa(1, 100.000000),
            pso(1, 100.005000, "event"),
            sa(2, 100.100000),
            pso(2, 100.108000, "event"),
        ]
        marks = pwm.parse_lines(lines)
        result = pwm.pair_marks(marks)
        self.assertEqual(len(result.pairs), 2)
        self.assertEqual(len(result.unpaired_observed), 0)
        self.assertEqual(len(result.unclaimed_state_applied), 0)

        d0 = result.pairs[0][2]
        d1 = result.pairs[1][2]
        self.assertAlmostEqual(d0, 5.0, places=3)
        self.assertAlmostEqual(d1, 8.0, places=3)

    def test_coalescing_earliest_unpaired_rule(self):
        """(2) coalescing: 3 state_applied between 2 wakes; the wake must
        pair to the EARLIEST unpaired state_applied, not the latest, and
        not by counting."""
        lines = [
            # Wake #1 drains nothing yet (queue empty at t=... not applicable
            # here -- start fresh): first, 3 state_applied land before ANY
            # wake, simulating coalescing.
            sa(10, 200.000000),  # earliest unpaired -> must be picked first
            sa(11, 200.020000),
            sa(12, 200.040000),
            # First wake fires: must pair to event_idx=10 (the earliest),
            # NOT event_idx=12 (the latest/most recent).
            pso(12, 200.050000, "event"),
            # Second wake fires: must pair to the next-earliest remaining
            # unclaimed entry, event_idx=11.
            pso(12, 200.060000, "event"),
        ]
        marks = pwm.parse_lines(lines)
        result = pwm.pair_marks(marks)
        self.assertEqual(len(result.pairs), 2)

        first_pair_sa, first_pair_obs, first_delta = result.pairs[0]
        second_pair_sa, second_pair_obs, second_delta = result.pairs[1]

        # Earliest-unpaired rule: first wake claims event_idx=10 (t=200.000000),
        # not event_idx=12 (t=200.040000, the latest).
        self.assertEqual(first_pair_sa.event_idx, 10)
        self.assertAlmostEqual(first_delta, 50.0, places=3)  # 200.050 - 200.000 = 0.050s = 50ms

        # Second wake claims the next-earliest remaining: event_idx=11.
        self.assertEqual(second_pair_sa.event_idx, 11)
        self.assertAlmostEqual(second_delta, 40.0, places=3)  # 200.060 - 200.020 = 0.040s = 40ms

        # event_idx=12's state_applied is left unclaimed (correct coalescing,
        # not a lost wake) -- diagnostic count must reflect it.
        self.assertEqual(len(result.unclaimed_state_applied), 1)
        self.assertEqual(result.unclaimed_state_applied[0].event_idx, 12)

    def test_noncontiguous_event_idx_not_paired_by_counting(self):
        """(3) non-contiguous event_idx values, proving pairing is by
        arrival-order queue position, NOT by counting or by event_idx
        arithmetic (e.g. NOT "pair idx N with idx N+1")."""
        lines = [
            sa(1000, 300.000000),
            sa(1007, 300.010000),  # gap: 1001-1006 never emitted (unrelated events)
            sa(1050, 300.020000),  # gap: big jump
            pso(1050, 300.025000, "event"),  # must pair to event_idx=1000 (earliest), not 1050
            pso(1050, 300.035000, "event"),  # must pair to event_idx=1007 (next earliest)
        ]
        marks = pwm.parse_lines(lines)
        result = pwm.pair_marks(marks)
        self.assertEqual(len(result.pairs), 2)
        self.assertEqual(result.pairs[0][0].event_idx, 1000)
        self.assertEqual(result.pairs[1][0].event_idx, 1007)
        self.assertEqual(len(result.unclaimed_state_applied), 1)
        self.assertEqual(result.unclaimed_state_applied[0].event_idx, 1050)

    def test_percentile_and_median_sane(self):
        deltas = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 100.0]
        stats = pwm.compute_stats(deltas)
        self.assertEqual(stats["n"], 10)
        self.assertEqual(stats["min_ms"], 5.0)
        self.assertEqual(stats["max_ms"], 100.0)
        self.assertAlmostEqual(stats["median_ms"], 9.5, places=3)


class TestWakeClassification(unittest.TestCase):
    def test_event_raced_timeout_not_counted_against_zero(self):
        """(4) event_raced_timeout IS event-driven and must NOT be counted
        as a true timeout."""
        lines = [
            sa(1, 400.000000),
            pso(1, 400.005000, "event_raced_timeout"),
        ]
        marks = pwm.parse_lines(lines)
        result = pwm.pair_marks(marks)
        wake_counts = pwm.classify_wakes(result.pairs)
        self.assertEqual(wake_counts["event_raced_timeout"], 1)
        self.assertEqual(wake_counts["timeout"], 0)
        self.assertEqual(wake_counts["event"], 0)

    def test_true_timeout_is_counted(self):
        """(5) a true timeout IS counted."""
        lines = [
            sa(1, 500.000000),
            pso(1, 500.100000, "timeout"),
        ]
        marks = pwm.parse_lines(lines)
        result = pwm.pair_marks(marks)
        wake_counts = pwm.classify_wakes(result.pairs)
        self.assertEqual(wake_counts["timeout"], 1)
        self.assertEqual(wake_counts["event_raced_timeout"], 0)
        self.assertEqual(wake_counts["event"], 0)

    def test_mixed_3way_classification_counts(self):
        lines = [
            sa(1, 600.000000),
            pso(1, 600.005000, "event"),
            sa(2, 600.100000),
            pso(2, 600.200000, "timeout"),
            sa(3, 600.300000),
            pso(3, 600.305000, "event_raced_timeout"),
        ]
        marks = pwm.parse_lines(lines)
        result = pwm.pair_marks(marks)
        wake_counts = pwm.classify_wakes(result.pairs)
        self.assertEqual(wake_counts, {"event": 1, "event_raced_timeout": 1, "timeout": 1})


class TestGateAVerdict(unittest.TestCase):
    def test_gate_a_pass_case(self):
        deltas = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0]
        stats = pwm.compute_stats(deltas)
        verdict = pwm.gate_a_verdict(stats, request_path_timeouts=0)
        self.assertTrue(verdict["median_pass"])
        self.assertTrue(verdict["p99_pass"])
        self.assertTrue(verdict["request_path_timeout_pass"])
        self.assertTrue(verdict["overall_pass"])

    def test_gate_a_fail_on_median(self):
        deltas = [50.0] * 10
        stats = pwm.compute_stats(deltas)
        verdict = pwm.gate_a_verdict(stats, request_path_timeouts=0)
        self.assertFalse(verdict["median_pass"])
        self.assertFalse(verdict["overall_pass"])

    def test_gate_a_fail_on_timeout_count(self):
        deltas = [1.0] * 10
        stats = pwm.compute_stats(deltas)
        verdict = pwm.gate_a_verdict(stats, request_path_timeouts=1)
        self.assertTrue(verdict["median_pass"])
        self.assertTrue(verdict["p99_pass"])
        self.assertFalse(verdict["request_path_timeout_pass"])
        self.assertFalse(verdict["overall_pass"])


class TestA2TaskTypeScoping(unittest.TestCase):
    """Amendment A2: CreateRunner/DownloadModel backoff-gated true-timeout
    dispatches must be excluded from Gate A's zero; all other task types
    are request-path and must count. Old-format lines with no task= field
    must never silently resolve to a clean pass."""

    def test_createrunner_true_timeout_excluded_gate_a_can_pass(self):
        lines = [
            sa(1, 800.000000),
            pso(1, 800.100000, "timeout", task="CreateRunner"),
        ]
        report_text, json_summary = pwm.analyze_source("createrunner_timeout", lines)
        scoping = json_summary["timeout_scoping"]
        self.assertEqual(scoping["total_true_timeout_wakes"], 1)
        self.assertEqual(scoping["backoff_gated_true_timeout_wakes"], 1)
        self.assertEqual(scoping["request_path_true_timeout_wakes"], 0)
        self.assertEqual(scoping["undetermined_true_timeout_wakes"], 0)
        self.assertEqual(scoping["gate_a_relevant_true_timeout_wakes"], 0)
        self.assertFalse(scoping["undetermined"])
        self.assertTrue(json_summary["gate_a"]["request_path_timeout_pass"])
        self.assertIn("CreateRunner", scoping["by_task_type"])
        self.assertEqual(scoping["by_task_type"]["CreateRunner"], 1)

    def test_downloadmodel_true_timeout_also_excluded(self):
        lines = [
            sa(1, 810.000000),
            pso(1, 810.100000, "timeout", task="DownloadModel"),
        ]
        report_text, json_summary = pwm.analyze_source("downloadmodel_timeout", lines)
        scoping = json_summary["timeout_scoping"]
        self.assertEqual(scoping["backoff_gated_true_timeout_wakes"], 1)
        self.assertEqual(scoping["request_path_true_timeout_wakes"], 0)
        self.assertEqual(scoping["gate_a_relevant_true_timeout_wakes"], 0)
        self.assertTrue(json_summary["gate_a"]["request_path_timeout_pass"])
        self.assertEqual(scoping["by_task_type"]["DownloadModel"], 1)

    def test_request_path_task_true_timeout_counted_gate_a_fails(self):
        """A normal inference/chat dispatch (TextGeneration) timing out on
        a true timeout is REQUEST-PATH and must count against Gate A's
        zero -- Gate A must FAIL."""
        lines = [
            sa(1, 820.000000),
            pso(1, 820.100000, "timeout", task="TextGeneration"),
        ]
        report_text, json_summary = pwm.analyze_source("request_path_timeout", lines)
        scoping = json_summary["timeout_scoping"]
        self.assertEqual(scoping["request_path_true_timeout_wakes"], 1)
        self.assertEqual(scoping["backoff_gated_true_timeout_wakes"], 0)
        self.assertEqual(scoping["gate_a_relevant_true_timeout_wakes"], 1)
        self.assertFalse(json_summary["gate_a"]["request_path_timeout_pass"])
        self.assertFalse(json_summary["gate_a"]["overall_pass"])
        self.assertEqual(scoping["by_task_type"]["TextGeneration"], 1)

    def test_unanticipated_task_type_shows_up_as_itself(self):
        """A task type nobody anticipated must appear in the data as
        itself, not be silently bucketed into either classification."""
        lines = [
            sa(1, 825.000000),
            pso(1, 825.100000, "timeout", task="SomeFutureTaskType"),
        ]
        _, json_summary = pwm.analyze_source("unanticipated_task_type", lines)
        scoping = json_summary["timeout_scoping"]
        self.assertEqual(scoping["by_task_type"]["SomeFutureTaskType"], 1)
        # Not in the backoff-gated set -> treated as request-path.
        self.assertEqual(scoping["request_path_true_timeout_wakes"], 1)
        self.assertEqual(scoping["backoff_gated_true_timeout_wakes"], 0)

    def test_mixed_backoff_gated_and_request_path_counts_differ(self):
        """A run containing BOTH a backoff-gated true timeout and a
        request-path true timeout: the two counts must differ and both
        must be reported correctly, and Gate A must FAIL on the
        request-path one alone."""
        lines = [
            sa(1, 830.000000),
            pso(1, 830.100000, "timeout", task="CreateRunner"),
            sa(2, 830.200000),
            pso(2, 830.300000, "timeout", task="DownloadModel"),
            sa(3, 830.400000),
            pso(3, 830.500000, "timeout", task="TextGeneration"),
            sa(4, 830.600000),
            pso(4, 830.605000, "event"),
        ]
        _, json_summary = pwm.analyze_source("mixed_run", lines)
        scoping = json_summary["timeout_scoping"]
        self.assertEqual(scoping["total_true_timeout_wakes"], 3)
        self.assertEqual(scoping["backoff_gated_true_timeout_wakes"], 2)
        self.assertEqual(scoping["request_path_true_timeout_wakes"], 1)
        self.assertEqual(scoping["undetermined_true_timeout_wakes"], 0)
        self.assertEqual(scoping["gate_a_relevant_true_timeout_wakes"], 1)
        self.assertNotEqual(
            scoping["backoff_gated_true_timeout_wakes"],
            scoping["request_path_true_timeout_wakes"],
        )
        self.assertEqual(scoping["by_task_type"]["CreateRunner"], 1)
        self.assertEqual(scoping["by_task_type"]["DownloadModel"], 1)
        self.assertEqual(scoping["by_task_type"]["TextGeneration"], 1)
        self.assertFalse(json_summary["gate_a"]["request_path_timeout_pass"])
        self.assertFalse(json_summary["gate_a"]["overall_pass"])

    def test_old_format_no_task_field_true_timeout_is_undetermined(self):
        """A true timeout on an old-format line (no task= field) must fire
        the UNDETERMINED warning and must NOT silently produce a clean
        Gate-A pass."""
        lines = [
            sa(1, 840.000000),
            pso(1, 840.100000, "timeout"),  # no task= -> old format
        ]
        report_text, json_summary = pwm.analyze_source("old_format_timeout", lines)
        scoping = json_summary["timeout_scoping"]
        self.assertEqual(scoping["undetermined_true_timeout_wakes"], 1)
        self.assertTrue(scoping["undetermined"])
        self.assertIn("<UNDETERMINED>", scoping["by_task_type"])
        # Conservatively counted against Gate A's zero -- never a silent pass.
        self.assertEqual(scoping["gate_a_relevant_true_timeout_wakes"], 1)
        self.assertFalse(json_summary["gate_a"]["request_path_timeout_pass"])
        self.assertFalse(json_summary["gate_a"]["overall_pass"])
        self.assertIn("UNDETERMINED", report_text)
        self.assertIn(_TASK_TYPE_FIELD_MISSING_TEXT, report_text)

    def test_old_format_no_true_timeout_not_undetermined(self):
        """Old-format lines (no task=) with NO true timeout present are
        fine -- undetermined only fires when there is an actual true
        timeout to scope."""
        lines = [
            sa(1, 850.000000),
            pso(1, 850.005000, "event"),  # old format, non-timeout wake_kind
        ]
        _, json_summary = pwm.analyze_source("old_format_no_timeout", lines)
        scoping = json_summary["timeout_scoping"]
        self.assertEqual(scoping["undetermined_true_timeout_wakes"], 0)
        self.assertFalse(scoping["undetermined"])
        self.assertTrue(json_summary["gate_a"]["request_path_timeout_pass"])


_TASK_TYPE_FIELD_MISSING_TEXT = pwm._TASK_TYPE_FIELD_MISSING


class TestBaselineFingerprint(unittest.TestCase):
    def test_baseline_matches_expected_band(self):
        deltas = [40.0, 45.0, 50.0, 55.0, 60.0] * 4  # median ~50, within band
        stats = pwm.compute_stats(deltas)
        fp = pwm.baseline_fingerprint_check(stats)
        self.assertTrue(fp["median_match"])

    def test_postfix_data_does_not_match_baseline_band(self):
        deltas = [2.0, 3.0, 4.0, 5.0] * 4
        stats = pwm.compute_stats(deltas)
        fp = pwm.baseline_fingerprint_check(stats)
        self.assertFalse(fp["median_match"])
        self.assertFalse(fp["fingerprint_matches_baseline"])


class TestEmptyAndGarbageInput(unittest.TestCase):
    def test_empty_input_raises(self):
        """(6a) fully empty input must raise, not silently produce a verdict."""
        with self.assertRaises(ValueError):
            pwm.analyze_source("empty_test", [])

    def test_garbage_only_input_raises(self):
        """(6b) garbage/unrelated lines with zero PHASE_MARK content must raise."""
        lines = [
            "totally unrelated log line",
            "another line with no PHASE_MARK anywhere",
            "2026-09-04 zenoh peer connected 192.168.86.201",
        ]
        with self.assertRaises(ValueError):
            pwm.analyze_source("garbage_test", lines)

    def test_cli_main_exits_nonzero_on_garbage_file(self):
        """End-to-end: running main() against a garbage-only file must
        return a non-zero exit code (never a silent 0/pass)."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            f.write("nothing but noise\nno marks here\n")
            tmp_path = f.name
        try:
            exit_code = pwm.main(["parse_worker_marks.py", tmp_path])
            self.assertNotEqual(exit_code, 0)
        finally:
            os.unlink(tmp_path)

    def test_unpaired_only_input_also_raises(self):
        """Marks parse fine individually but zero pairs form (e.g. only
        plan_step_observed lines, no state_applied at all) -- must also be
        treated as invalid for a Gate-A verdict, not silently pass."""
        lines = [pso(1, 100.0, "event"), pso(2, 100.1, "event")]
        with self.assertRaises(ValueError):
            pwm.analyze_source("unpaired_only_test", lines)


class TestEndToEndAnalyze(unittest.TestCase):
    def test_full_analyze_source_smoke(self):
        lines = [
            sa(1, 700.000000),
            pso(1, 700.003000, "event"),
            sa(2, 700.100000),
            sa(3, 700.110000),
            pso(3, 700.115000, "event"),  # pairs to event_idx=2 (earliest unpaired)
            sa(4, 700.300000),
            pso(4, 700.400000, "timeout"),
        ]
        report_text, json_summary = pwm.analyze_source("smoke_test", lines)
        self.assertIn("GATE A OVERALL", report_text)
        self.assertEqual(json_summary["counts"]["paired_n"], 3)
        # sa(3) is left unclaimed: pso(3) paired to the earlier sa(2), and
        # sa(3) itself is never claimed by any later observed mark in this
        # synthetic stream -- correct coalescing diagnostic, not a bug.
        self.assertEqual(json_summary["counts"]["state_applied_never_observed"], 1)
        self.assertEqual(json_summary["wake_classification"]["timeout"], 1)
        # A2 undetermined warning must fire since there's a true timeout present.
        self.assertTrue(json_summary["timeout_scoping"]["undetermined"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
