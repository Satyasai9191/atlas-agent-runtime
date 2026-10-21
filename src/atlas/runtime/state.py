
"""State-transition operations for ATLAS agent runs.

This module owns lifecycle transitions. Each operation returns a new,
validated AgentRun snapshot rather than mutating the supplied snapshot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from atlas.runtime.models import AgentRun, RunStatus


class InvalidRunTransition(ValueError):
    """Raised when a run is asked to enter an invalid lifecycle state."""


_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
        }
    ),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.WAITING_FOR_APPROVAL,
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
        }
    ),
    RunStatus.WAITING_FOR_APPROVAL: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
        }
    ),
}


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def transition_run(
    run: AgentRun,
    target_status: RunStatus,
    *,
    at: datetime | None = None,
    output: dict[str, Any] | None = None,
    error: str | None = None,
) -> AgentRun:
    """Validate and apply one lifecycle transition to a run snapshot.

    The supplied run is not mutated. A new, validated snapshot is returned.
    """
    timestamp = at if at is not None else _utc_now()

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Transition timestamp must be timezone-aware.")

    if timestamp < run.created_at:
        raise ValueError("Transition timestamp cannot precede created_at.")

    if run.started_at is not None and timestamp < run.started_at:
        raise ValueError("Transition timestamp cannot precede started_at.")

    if run.finished_at is not None:
        raise InvalidRunTransition(
            "A run with a finished_at timestamp cannot transition again."
        )

    allowed = _ALLOWED_TRANSITIONS.get(run.status, frozenset())

    if target_status not in allowed:
        allowed_values = ", ".join(
            sorted(status.value for status in allowed)
        ) or "none"

        raise InvalidRunTransition(
            f"Cannot transition from {run.status.value!r} "
            f"to {target_status.value!r}. Allowed next states: "
            f"{allowed_values}."
        )

    if target_status is RunStatus.SUCCEEDED:
        if output is None:
            raise ValueError(
                "A successful run must provide an output dictionary. "
                "Use {} when the result is intentionally empty."
            )
        if error is not None:
            raise ValueError("A successful run cannot have an error.")

    elif output is not None:
        raise ValueError("Output can only be supplied when a run succeeds.")

    normalized_error: str | None = None

    if error is not None:
        normalized_error = error.strip()

        if not normalized_error:
            raise ValueError("Error or reason cannot be blank.")

        if len(normalized_error) > 4_000:
            raise ValueError("Error or reason cannot exceed 4000 characters.")

    if target_status is RunStatus.FAILED and normalized_error is None:
        raise ValueError("A failed run must include an error message.")

    if (
        normalized_error is not None
        and target_status
        not in {
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
        }
    ):
        raise ValueError(
            "An error or reason is only accepted for failure, "
            "cancellation, or timeout."
        )

    # Work on a copy so failed validation cannot partially update the input.
    data = run.model_dump(mode="python")
    data["status"] = target_status

    if target_status is RunStatus.RUNNING:
        if data["started_at"] is None:
            data["started_at"] = timestamp

        # Clear any previous transient error when execution resumes.
        data["error"] = None

    elif target_status is RunStatus.SUCCEEDED:
        data["output"] = output
        data["error"] = None

    elif target_status is RunStatus.FAILED:
        data["error"] = normalized_error

    elif target_status in {
        RunStatus.CANCELLED,
        RunStatus.TIMED_OUT,
    }:
        if normalized_error is not None:
            data["error"] = normalized_error

    if target_status.is_terminal:
        data["finished_at"] = timestamp

    # Revalidate the complete snapshot before returning it.
    return AgentRun.model_validate(data)


def start_run(
    run: AgentRun,
    *,
    at: datetime | None = None,
) -> AgentRun:
    """Start a pending run."""
    return transition_run(run, RunStatus.RUNNING, at=at)


def wait_for_approval(
    run: AgentRun,
    *,
    at: datetime | None = None,
) -> AgentRun:
    """Pause a running run while awaiting human approval."""
    return transition_run(run, RunStatus.WAITING_FOR_APPROVAL, at=at)


def resume_after_approval(
    run: AgentRun,
    *,
    at: datetime | None = None,
) -> AgentRun:
    """Resume a run that was waiting for approval."""
    return transition_run(run, RunStatus.RUNNING, at=at)


def succeed_run(
    run: AgentRun,
    output: dict[str, Any],
    *,
    at: datetime | None = None,
) -> AgentRun:
    """Finish a running run successfully with its result."""
    return transition_run(
        run,
        RunStatus.SUCCEEDED,
        at=at,
        output=output,
    )


def fail_run(
    run: AgentRun,
    error: str,
    *,
    at: datetime | None = None,
) -> AgentRun:
    """Finish a running run with an error."""
    return transition_run(
        run,
        RunStatus.FAILED,
        at=at,
        error=error,
    )


def cancel_run(
    run: AgentRun,
    reason: str | None = None,
    *,
    at: datetime | None = None,
) -> AgentRun:
    """Cancel a pending, running, or approval-waiting run."""
    return transition_run(
        run,
        RunStatus.CANCELLED,
        at=at,
        error=reason,
    )


def timeout_run(
    run: AgentRun,
    reason: str = "Execution timed out.",
    *,
    at: datetime | None = None,
) -> AgentRun:
    """Mark a pending, running, or approval-waiting run as timed out."""
    return transition_run(
        run,
        RunStatus.TIMED_OUT,
        at=at,
        error=reason,
    )
