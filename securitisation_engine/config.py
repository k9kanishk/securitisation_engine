from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import DealConfig, TrancheConfig


@dataclass(frozen=True)
class EngineConfig:
    deal: DealConfig
    tranches: List[TrancheConfig]

    def sorted_tranches(self) -> List[TrancheConfig]:
        return sorted(self.tranches, key=lambda t: t.rank)

    def non_residual_tranches(self) -> List[TrancheConfig]:
        return [t for t in self.sorted_tranches() if not t.is_residual]

    def tranche_names(self) -> List[str]:
        return [t.name for t in self.sorted_tranches()]
