import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SynapseConfig:
    """Central configuration for Synapse Intelligence Layer."""
    shared_vault_path: str = "shared_vaults"          # Shared with EM instances (local or IPFS gateway)
    encryption_key: str = os.getenv("SYNAPSE_ENCRYPTION_KEY", "default_sage_key_change_me")
    tiered_access_enabled: bool = True
    daily_loop_interval_seconds: int = 86400          # 24 hours
    red_team_enabled: bool = True
    model_distillation_enabled: bool = False          # Future flag
    max_model_size_gb: float = 8.0

    def __post_init__(self):
        Path(self.shared_vault_path).mkdir(parents=True, exist_ok=True)
