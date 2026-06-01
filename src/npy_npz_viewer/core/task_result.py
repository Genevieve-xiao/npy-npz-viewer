"""
Shared task result types.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TaskResult:
    """Uniform result for background and core operations."""

    success: bool
    data: Any = None
    warning: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    sampled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: Any = None, **kwargs) -> "TaskResult":
        return cls(success=True, data=data, **kwargs)

    @classmethod
    def fail(cls, error: str, **kwargs) -> "TaskResult":
        return cls(success=False, error=error, **kwargs)
