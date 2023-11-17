from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class MemoryStatistics:
    totalMemoryInBytes: int
    availableMemoryInBytes: int
    memoryUsagePercent: float


class MemoryPresenter:

    def getStatistics(self) -> MemoryStatistics:
        mem = psutil.virtual_memory()
        stats = MemoryStatistics(
            totalMemoryInBytes=mem.total,
            availableMemoryInBytes=mem.available,
            memoryUsagePercent=mem.percent,
        )
        return stats
