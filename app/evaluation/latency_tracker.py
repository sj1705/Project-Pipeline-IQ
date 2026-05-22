import time
from typing import Dict


class LatencyTracker:
    """Track latency for each stage of the RAG pipeline."""

    def __init__(self):
        self.stages: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, stage: str):
        """Mark the start of a pipeline stage."""
        self._start_times[stage] = time.time()

    def end(self, stage: str):
        """Mark the end of a pipeline stage. Records duration in ms."""
        if stage in self._start_times:
            duration_ms = (time.time() - self._start_times[stage]) * 1000
            self.stages[stage] = round(duration_ms, 2)
            del self._start_times[stage]

    def get_total(self) -> float:
        """Get total latency across all stages."""
        return round(sum(self.stages.values()), 2)

    def get_report(self) -> Dict:
        """Get full latency report."""
        return {
            "stages": self.stages,
            "total_ms": self.get_total(),
        }

    def reset(self):
        """Reset all timings for a new request."""
        self.stages = {}
        self._start_times = {}