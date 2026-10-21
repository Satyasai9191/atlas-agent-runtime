
"""Unit tests for ATLAS run-state transitions."""

from datetime import datetime, timedelta, timezone

import pytest

from atlas.runtime.models import AgentRun, RunStatus
from atlas.runtime.state import (
    InvalidRunTransition,
    cancel_run,
    fail_run,
    resume_after_approval,
    start_run,
    succeed_run,
    timeout_run,
    transition_run,
    wait_for_approval,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = T0 + timedelta(seconds=1)
T2 = T0 + timedelta(seconds=2)
T3 = T0 + timedelta(seconds=3)


def make_run() -> AgentRun:
    """Create a predictable run for lifecycle tests."""
    return AgentRun(
        agent_id="research-agent",
        input={"question": "What is ATLAS?"},
        created_at=T0,
    )


def test_start_run_sets_status_and_start_time() -> None:
    original = make_run()

    updated = start_run(original, at=T1)

    assert updated.status is RunStatus.RUNNING
    assert updated.started_at == T1
    assert updated.finished_at is None

    # Transitions return a new snapshot instead of mutating the original.
    assert original.status is RunStatus.PENDING
    assert original.started_at is None


def test_pending_run_cannot_skip_directly_to_success() -> None:
    with pytest.raises(InvalidRunTransition):
        transition_run(
            make_run(),
            RunStatus.SUCCEEDED,
            at=T1,
            output={"answer": "done"},
        )


def test_success_requires_an_output_dictionary() -> None:
    running = start_run(make_run(), at=T1)

    with pytest.raises(ValueError, match="output dictionary"):
        transition_run(running, RunStatus.SUCCEEDED, at=T2)


def test_success_records_output_and_finish_time() -> None:
    running = start_run(make_run(), at=T1)

    completed = succeed_run(
        running,
        output={"answer": "ATLAS is an agent runtime."},
        at=T2,
    )

    assert completed.status is RunStatus.SUCCEEDED
    assert completed.output == {"answer": "ATLAS is an agent runtime."}
    assert completed.started_at == T1
    assert completed.finished_at == T2
    assert completed.error is None


def test_approval_pause_and_resume_preserve_original_start_time() -> None:
    running = start_run(make_run(), at=T1)
    waiting = wait_for_approval(running, at=T2)
    resumed = resume_after_approval(waiting, at=T3)

    assert waiting.status is RunStatus.WAITING_FOR_APPROVAL
    assert waiting.started_at == T1
    assert waiting.finished_at is None

    assert resumed.status is RunStatus.RUNNING
    assert resumed.started_at == T1
    assert resumed.finished_at is None


def test_failed_run_requires_a_nonblank_error() -> None:
    running = start_run(make_run(), at=T1)

    with pytest.raises(ValueError, match="cannot be blank"):
        fail_run(running, "   ", at=T2)

    with pytest.raises(ValueError, match="must include an error"):
        transition_run(running, RunStatus.FAILED, at=T2)


def test_failure_records_error_and_finish_time() -> None:
    running = start_run(make_run(), at=T1)

    failed = fail_run(running, "Model provider unavailable.", at=T2)

    assert failed.status is RunStatus.FAILED
    assert failed.error == "Model provider unavailable."
    assert failed.finished_at == T2
    assert failed.output is None


def test_pending_run_can_be_cancelled_without_starting() -> None:
    cancelled = cancel_run(make_run(), reason="Request withdrawn.", at=T1)

    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.started_at is None
    assert cancelled.finished_at == T1
    assert cancelled.error == "Request withdrawn."


def test_timeout_records_reason_and_finish_time() -> None:
    running = start_run(make_run(), at=T1)
    timed_out = timeout_run(running, at=T2)

    assert timed_out.status is RunStatus.TIMED_OUT
    assert timed_out.error == "Execution timed out."
    assert timed_out.finished_at == T2


def test_transition_rejects_naive_timestamps() -> None:
    naive_timestamp = datetime(2026, 1, 1, 0, 0, 1)

    with pytest.raises(ValueError, match="timezone-aware"):
        start_run(make_run(), at=naive_timestamp)


def test_transition_cannot_precede_creation_or_start_time() -> None:
    with pytest.raises(ValueError, match="created_at"):
        start_run(make_run(), at=T0 - timedelta(seconds=1))

    running = start_run(make_run(), at=T2)

    with pytest.raises(ValueError, match="started_at"):
        succeed_run(running, output={}, at=T1)


def test_terminal_run_cannot_transition_again() -> None:
    running = start_run(make_run(), at=T1)
    completed = succeed_run(running, output={}, at=T2)

    with pytest.raises(InvalidRunTransition):
        start_run(completed, at=T3)


def test_running_run_cannot_succeed_with_an_error() -> None:
    running = start_run(make_run(), at=T1)

    with pytest.raises(ValueError, match="cannot have an error"):
        transition_run(
            running,
            RunStatus.SUCCEEDED,
            at=T2,
            output={},
            error="Unexpected error",
        )
