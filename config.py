import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SynapseConfig:
    """Central configuration for Synapse Intelligence Layer with full IPFS + LLM + training data support."""

    # Core paths
    shared_vault_path: str = "shared_vaults"

    # Security
    encryption_key: str = os.getenv("SYNAPSE_ENCRYPTION_KEY", "default_sage_key_change_me_in_production")

    # Runtime
    tiered_access_enabled: bool = True
    daily_loop_interval_seconds: int = 86400
    red_team_enabled: bool = True
    model_distillation_enabled: bool = False
    max_model_size_gb: float = 8.0

    # IPFS Distributed Storage
    ipfs_enabled: bool = True
    ipfs_gateway: str = os.getenv("IPFS_GATEWAY", "http://127.0.0.1:8080")
    ipfs_api: str = os.getenv("IPFS_API", "http://127.0.0.1:5001")
    ipfs_pin: bool = True
    ipfs_public_gateway: str = "https://ipfs.io/ipfs/"

    # Grounded LLM for Chat Interface
    llm_backend: str = os.getenv("LLM_BACKEND", "openai")          # "openai", "ollama", "groq", "local"
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_api_base: str = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")

    # Training Data Vault (nightly intelligent cleaning)
    training_data_vault_path: str = "synapse/data/training_data"
    training_data_min_score: float = 0.78
    training_data_max_red_team_risk: float = 0.45

    def __post_init__(self):
        Path(self.shared_vault_path).mkdir(parents=True, exist_ok=True)
        Path(self.training_data_vault_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ SynapseConfig loaded — IPFS {'ENABLED' if self.ipfs_enabled else 'disabled'} | LLM backend: {self.llm_backend}")
