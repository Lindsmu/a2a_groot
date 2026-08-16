# SPDX-License-Identifier: Apache-2.0

"""Regression tests for one-record-per-step A2A component logging."""

from gr00t.experiment.trainer import _should_log_step_metrics


def test_a2a_component_logging_skips_step_zero_and_non_intervals():
    assert not _should_log_step_metrics(0, 5, None)
    assert not _should_log_step_metrics(4, 5, None)
    assert not _should_log_step_metrics(5, 0, None)


def test_a2a_component_logging_emits_once_per_global_step():
    assert _should_log_step_metrics(10, 5, None)
    assert not _should_log_step_metrics(10, 5, 10)
    assert _should_log_step_metrics(15, 5, 10)
