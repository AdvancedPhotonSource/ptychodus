from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class MemoryStatistics:
    total_physical_memory_bytes: int
    available_memory_bytes: int
    percent_usage: float


class MemoryPresenter:
    def get_statistics(self) -> MemoryStatistics:
        mem = psutil.virtual_memory()
        return MemoryStatistics(
            total_physical_memory_bytes=mem.total,
            available_memory_bytes=mem.available,
            percent_usage=mem.percent,
        )
