import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from synapse.config import SynapseConfig
from synapse.graph_mining import GraphMiner
from synapse.meta_rl_loop import MetaRLLoop
from synapse.neural_net_head import NeuralNetHead
from synapse.model_distillation import ModelDistiller
from synapse.chat_interface import SynapseChatInterface
from synapse.defense_red_team import DefenseRedTeam
from synapse.kas import RecursiveKAS
from synapse.economic_layer import EconomicLayer
from synapse.utils import load_shared_vaults, save_to_vaults

logger = logging.getLogger(__name__)

class SynapseMetaAgent:
    """Central Synapse Meta-Agent — orchestrates all intelligence subsystems for SAGE."""

    def __init__(self, config: SynapseConfig = None):
        self.config = config or SynapseConfig()
        self.graph_miner = GraphMiner()
        self.meta_rl = MetaRLLoop()
        self.neural_head = NeuralNetHead()
        self.distiller = ModelDistiller()
        self.chat = SynapseChatInterface()
        self.defense = DefenseRedTeam()
        self.kas = RecursiveKAS()
        self.economic = EconomicLayer()
        
        self.last_loop = datetime.now()
        logger.info("🚀 SynapseMetaAgent v0.9.12 MAX SOTA initialized — full integration with EM/SAGE repository active")

    def run_daily_intelligence_cycle(self):
        """Main daily self-improvement loop — the heart of SAGE compounding."""
        logger.info("🔄 Starting Synapse daily intelligence cycle")
        
        # Load latest data from shared Solve/Strategy Layer (from all EM instances)
        vaults = load_shared_vaults(self.config.shared_vault_path)
        
        # Graph mining + pattern discovery
        mined_patterns = self.graph_miner.mine(vaults)
        
        # Meta-RL self-audit and improvement proposals
        rl_results = self.meta_rl.run_audit_and_improve(mined_patterns)
        
        # Neural Net Head scoring and calibration
        scored_insights = self.neural_head.score_and_calibrate(rl_results)
        
        # Defense red-teaming
        hardened_insights = self.defense.red_team_and_harden(scored_insights)
        
        # Recursive KAS for new knowledge
        kas_results = self.kas.recursive_hunt(hardened_insights)
        
        # Polishing + Economic Layer synthesis (proposals, toolkits, marketplace)
        polished_products = self.economic.polish_and_synthesize(kas_results)
        
        # Save improved intelligence back to shared vaults
        save_to_vaults(polished_products, self.config.shared_vault_path)
        
        # Distillation readiness check for future Enigma models
        distillation_ready = self.distiller.check_readiness(polished_products)
        
        self.last_loop = datetime.now()
        logger.info(f"✅ Synapse daily cycle complete — {len(polished_products)} new/improved artifacts | Distillation ready: {distillation_ready}")
        
        return {
            "status": "success",
            "artifacts_generated": len(polished_products),
            "distillation_ready": distillation_ready,
            "timestamp": datetime.now().isoformat()
        }

    def get_synapse_chat_response(self, user_query: str, user_tier: str = "standard") -> Dict[str, Any]:
        """Synapse Chat / Co-pilot interface — tiered access to intelligence."""
        return self.chat.handle_query(user_query, user_tier)

    def start_background_loop(self):
        """Start the continuous daily intelligence loop in background."""
        def loop():
            while True:
                self.run_daily_intelligence_cycle()
                time.sleep(self.config.daily_loop_interval_seconds)
        threading.Thread(target=loop, daemon=True).start()
        logger.info("🌍 Synapse background intelligence loop started")

# Global instance (imported by local EM instances for push/pull)
synapse = SynapseMetaAgent()
