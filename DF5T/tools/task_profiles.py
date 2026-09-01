"""
Deprecated compatibility shim. Use `tools.em_task_spec` (SPECS, TaskPhysicsSpec, TASKS).
"""
from __future__ import annotations

from tools.em_task_spec import SPECS, TASKS, TaskPhysicsSpec

# Legacy aliases — old TaskProfile had extra unused fields; callers should migrate to TaskPhysicsSpec.
TaskProfile = TaskPhysicsSpec
PROFILES = SPECS

__all__ = ["TASKS", "PROFILES", "TaskProfile", "TaskPhysicsSpec", "SPECS"]
