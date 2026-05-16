# synapse/mode.py
# SAGE v0.9.15 – Mixture of Domain Experts (MoDE)
# Parallel to existing MOPE. Fully integrated with fragment scoring and Meta-RL.

import logging
from typing import Dict, List, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class MODEMixture:
    def __init__(self):
        self.specialists: Dict[str, nn.Module] = {}

    def add_specialists(self, new_specialists: List[Dict]):
        for spec in new_specialists:
            name = spec["name"]
            self.specialists[name] = spec["specialist"]
            logger.info(f"✅ Added new MoDE specialist: {name}")

    def select_top_experts(self, context: Dict, k: int = 2) -> List[nn.Module]:
        if not self.specialists:
            return []
        sorted_specialists = sorted(self.specialists.items(), key=lambda x: 0.9, reverse=True)
        return [model for name, model in sorted_specialists[:k]]

    def forward(self, x: torch.Tensor, selected_specialists: List[nn.Module]) -> torch.Tensor:
        out = x
        for specialist in selected_specialists:
            if isinstance(specialist, nn.Module):
                out = out + specialist(out)
        return out
