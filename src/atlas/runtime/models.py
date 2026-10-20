"""Validated data contracts used by the ATLAS runtime.

These models describe requests, execution limits, and run snapshots. They deliberately
contain no orchestration logic; the runtime engine and state layer will own transitions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    """Return an explicitly timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class RunStatus(StrEnum):
    """Lifecycle states an agent run can occupy."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        """Whether the run has reached a state from which it should not resume."""
        return self in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
        }


class AtlasModel(BaseModel):
    """Shared validation policy for ATLAS runtime data models."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class ExecutionBudget(AtlasModel):
    """Resource limits that bound a single agent run.

    The budget is immutable so a running task cannot silently loosen its own limits.
    A future policy layer may decide whether a caller is allowed to request a given budget.
    """

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=25, ge=1, description="Maximum runtime steps allowed.")
    max_tool_calls: int = Field(
        default=50,
        ge=0,
        description="Maximum tool invocations allowed; zero disables tool use.",
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Wall-clock time limit for the run, in seconds.",
    )
    max_total_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional total token limit across model calls.",
    )
    max_cost_usd: Decimal | None = Field(
        default=None,
        ge=Decimal("0"),
        description="Optional cost ceiling in USD; Decimal avoids binary-float rounding.",
    )


class AgentRunRequest(AtlasModel):
    """Validated input supplied when a caller asks ATLAS to start a run."""

    agent_id: str = Field(
        min_length=1,
        max_length=200,
        description="Identifier of the agent configuration to execute.",
    )
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-object payload passed to the agent.",
    )
    budget: ExecutionBudget = Field(
        default_factory=ExecutionBudget,
        description="Limits requested for this run.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Caller-supplied labels for correlation and observability.",
    )


class AgentRun(AtlasModel):
    """Snapshot of a run's identity, lifecycle, result, and resource usage.

    The engine and state layer will be responsible for valid state transitions and for
    updating timestamps and counters as work proceeds.
    """

    run_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this run.")
    agent_id: str = Field(min_length=1, max_length=200)
    status: RunStatus = RunStatus.PENDING
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | None = None
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    metadata: dict[str, str] = Field(default_factory=dict)

    created_at: AwareDatetime = Field(default_factory=_utc_now)
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None

    steps_executed: int = Field(default=0, ge=0)
    tool_calls_executed: int = Field(default=0, ge=0)
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    error: str | None = Field(default=None, min_length=1, max_length=4_000)
