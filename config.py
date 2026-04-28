import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SynapseConfig:
    """Central configuration for Synapse Intelligence Layer with full IPFS support."""
    shared_vault_path: str = "shared_vaults"
    encryption_key: str = os.getenv("SYNAPSE_ENCRYPTION_KEY", "default_sage_key_change_me_in_production")
    tiered_access_enabled: bool = True
    daily_loop_interval_seconds: int = 86400
    red_team_enabled: bool = True
    model_distillation_enabled: bool = False
    max_model_size_gb: float = 8.0

    # IPFS Distributed Storage
    ipfs_enabled: bool = True
    ipfs_gateway: str = os.getenv("IPFS_GATEWAY", "http://127.0.0.1:8080")  # local daemon or public gateway
    ipfs_api: str = os.getenv("IPFS_API", "http://127.0.0.1:5001")          # for add/pin (daemon only)
    ipfs_pin: bool = True
    ipfs_public_gateway: str = "https://ipfs.io/ipfs/"                     # fallback for loading

    def __post_init__(self):
        Path(self.shared_vault_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ SynapseConfig loaded — IPFS {'ENABLED' if self.ipfs_enabled else 'disabled (local-only)'}")
