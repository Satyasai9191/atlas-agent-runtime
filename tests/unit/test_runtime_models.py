"""Unit tests for ATLAS runtime data contracts."""

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from atlas.runtime.models import AgentRun, AgentRunRequest, ExecutionBudget, RunStatus


def test_run_request_has_safe_defaults() -> None:
    request = AgentRunRequest(agent_id="research-agent")

    assert request.input == {}
    assert request.metadata == {}
    assert request.budget == ExecutionBudget()


def test_agent_id_is_trimmed_and_cannot_be_blank() -> None:
    request = AgentRunRequest(agent_id="  research-agent  ")
    assert request.agent_id == "research-agent"

    with pytest.raises(ValidationError):
        AgentRunRequest(agent_id="   ")


def test_unknown_request_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AgentRunRequest(agent_id="research-agent", agnet_id="typo")  # type: ignore[call-arg]


def test_budget_rejects_invalid_limits() -> None:
    with pytest.raises(ValidationError):
        ExecutionBudget(max_steps=0)

    with pytest.raises(ValidationError):
        ExecutionBudget(timeout_seconds=0)

    with pytest.raises(ValidationError):
        ExecutionBudget(max_total_tokens=0)


def test_zero_tool_calls_is_a_valid_way_to_disable_tools() -> None:
    budget = ExecutionBudget(max_tool_calls=0)
    assert budget.max_tool_calls == 0


def test_budget_is_immutable() -> None:
    budget = ExecutionBudget()

    with pytest.raises(ValidationError):
        budget.max_steps = 100  # type: ignore[misc]


def test_optional_usage_limits_and_decimal_cost_are_supported() -> None:
    budget = ExecutionBudget(max_total_tokens=10_000, max_cost_usd=Decimal("1.25"))

    assert budget.max_total_tokens == 10_000
    assert budget.max_cost_usd == Decimal("1.25")


def test_run_defaults_to_pending_with_aware_utc_timestamp() -> None:
    run = AgentRun(agent_id="research-agent", input={"question": "What is ATLAS?"})

    assert run.status is RunStatus.PENDING
    assert run.run_id is not None
    assert run.created_at.tzinfo is not None
    assert run.started_at is None
    assert run.finished_at is None
    assert run.output is None
    assert run.steps_executed == 0
    assert run.tool_calls_executed == 0
    assert run.tokens_used == 0
    assert run.cost_usd == Decimal("0")


def test_run_rejects_naive_timestamps() -> None:
    with pytest.raises(ValidationError):
        AgentRun(agent_id="research-agent", created_at=datetime(2026, 1, 1))


def test_run_counters_and_cost_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        AgentRun(agent_id="research-agent", steps_executed=-1)

    with pytest.raises(ValidationError):
        AgentRun(agent_id="research-agent", cost_usd=Decimal("-0.01"))


def test_terminal_statuses_are_identified_explicitly() -> None:
    assert RunStatus.SUCCEEDED.is_terminal
    assert RunStatus.FAILED.is_terminal
    assert RunStatus.CANCELLED.is_terminal
    assert RunStatus.TIMED_OUT.is_terminal
    assert not RunStatus.PENDING.is_terminal
    assert not RunStatus.RUNNING.is_terminal
    assert not RunStatus.WAITING_FOR_APPROVAL.is_terminal
