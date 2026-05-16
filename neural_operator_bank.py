# synapse/neural_operator_bank.py
# SAGE v0.9.15 – Neural Operator (PINO) Bank
# Lazy-loaded registry of base engines + distilled custom surrogates.

import logging
from typing import Dict, List, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class NeuralOperatorBank:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.bank: Dict[str, Any] = {}
        self._load_default_engines()

    def _load_default_engines(self):
        default_engines = ["FNO", "DeepONet", "PINO", "PINTO", "MF-FNO", "FC-PINO"]
        for name in default_engines:
            self.bank[name] = {
                "model": None,
                "name": name,
                "output_dim": 128,
                "metadata": {"supported_regimes": ["general", "turbulent", "quantum", "thermal"]}
            }
        logger.info(f"✅ NeuralOperatorBank initialized with {len(default_engines)} base engines")

    def get_best_backbone_for_step(self, step_name: str):
        if "turbulent" in step_name.lower() or "flow" in step_name.lower():
            return self.bank.get("FNO") or self.bank.get("PINO") or list(self.bank.values())[0]
        return self.bank.get("PINO") or list(self.bank.values())[0]

    def create_blended_model(self, engine_names: List[str], mode_specialists: List[Any]) -> nn.Module:
        class BlendedSurrogate(nn.Module):
            def __init__(self, bank, engine_names, mode_specialists):
                super().__init__()
                self.engines = [bank[name]["model"] for name in engine_names if name in bank and bank[name]["model"] is not None]
                self.specialists = mode_specialists

            def forward(self, x):
                if not self.engines:
                    return torch.zeros_like(x)
                outputs = [eng(x) for eng in self.engines]
                blended = torch.mean(torch.stack(outputs), dim=0)
                for spec in self.specialists:
                    if isinstance(spec, nn.Module):
                        blended = blended + spec(blended)
                return blended

        return BlendedSurrogate(self.bank, engine_names, mode_specialists).to(self.device)

    def register_new_entries(self, new_entries: List[Dict]):
        for entry in new_entries:
            name = entry["name"]
            self.bank[name] = {
                "model": entry["model"],
                "name": name,
                "output_dim": getattr(entry["model"], "output_dim", 128),
                "metadata": {"is_custom": True, "performance": entry["performance"]}
            }
            logger.info(f"✅ Registered new custom surrogate: {name}")
