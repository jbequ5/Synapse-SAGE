"""
Synapse Core — v0.9.12 10/10 MAXIMUM SOTA
Central orchestrator with full nightly intelligence cycle + lightweight additions
(includes vault pruning, training data cleaning, market summary, health report).
"""

import time
import logging
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from synapse.config import SynapseConfig
from synapse.graph_mining import GraphMiner
from synapse.meta_rl_loop import MetaRLLoop
from synapse.neural_net_head import NeuralNetHead
from synapse.model_distillation import ModelDistiller
from synapse.chat_interface import SynapseChatInterface
from synapse.defense_red_team import DefenseRedTeam
from synapse.kas import RecursiveKAS
from synapse.economic_layer import EconomicLayer
from synapse.utils import load_shared_vaults, save_to_vaults, prune_old_vaults

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
        logger.info("🚀 SynapseMetaAgent v0.9.12 MAX SOTA initialized — full nightly cycle + training data vault active")

    def run_daily_intelligence_cycle(self):
        """Main daily self-improvement loop — the heart of SAGE compounding."""
        logger.info("🔄 Starting Synapse daily intelligence cycle")
        
        # Load latest data from shared Solve/Strategy Layer
        vaults = load_shared_vaults(self.config.shared_vault_path)
        
        # Graph mining + pattern discovery
        mined_patterns = self.graph_miner.mine(vaults)
        
        # Meta-RL self-audit and improvement proposals (includes nightly red-teaming)
        rl_results = self.meta_rl.run_audit_and_improve(mined_patterns)
        
        # Neural Net Head scoring and calibration
        scored_insights = self.neural_head.calibrate_from_history()
        
        # Defense red-teaming
        hardened_insights = self.defense.red_team_and_harden(scored_insights)
        
        # Recursive KAS for new knowledge
        kas_results = self.kas.recursive_hunt(hardened_insights)
        
        # Polishing + Economic Layer synthesis
        polished_products = self.economic.polish_and_synthesize(kas_results)
        
        # Lightweight nightly maintenance tasks
        self._run_lightweight_nightly_tasks(vaults, polished_products)
        
        # Save improved intelligence back to shared vaults
        save_to_vaults(polished_products, self.config.shared_vault_path)
        
        # Distillation readiness check
        distillation_ready = self.distiller.check_readiness(polished_products)
        
        self.last_loop = datetime.now()
        logger.info(f"✅ Synapse daily cycle complete — {len(polished_products)} new/improved artifacts | Distillation ready: {distillation_ready}")

        return {
            "status": "success",
            "artifacts_generated": len(polished_products),
            "distillation_ready": distillation_ready,
            "timestamp": datetime.now().isoformat()
        }

    def _run_lightweight_nightly_tasks(self, vaults: Dict, polished_products: List[Dict]):
        """Lightweight nightly tasks: pruning, training data cleaning, market summary, health report."""
        # Vault pruning
        prune_old_vaults(self.config.shared_vault_path, max_age_days=14)

        # Training Data Vault — intelligent nightly cleaning for Enigma model training
        self._clean_and_prepare_training_data(vaults, polished_products)

        # Market feedback aggregation + summary
        summary = self.economic.get_market_summary()
        logger.info(f"💰 Nightly market summary — Total value created: {summary.get('total_value_created', 0):.2f}")

        # Swarm-wide health / provenance report
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "red_team_summary": self.defense.get_red_team_summary(),
            "kas_fragments_added": len(polished_products),
            "market_summary": summary
        }
        save_to_vaults([health_report], self.config.shared_vault_path, vault_name="nightly_reports")

    def _clean_and_prepare_training_data(self, vaults: Dict, polished_products: List[Dict]):
        """Nightly intelligent cleaning for Enigma model training data vault."""
        training_dir = Path(self.config.training_data_vault_path)
        training_dir.mkdir(parents=True, exist_ok=True)

        clean_data = []
        for vault_name, fragments in vaults.items():
            for frag in fragments:
                score = frag.get("combined_score", 0.0)
                risk = frag.get("red_team_risk", 1.0)
                freshness = self.kas.assess_freshness(frag)

                if (score >= self.config.training_data_min_score and
                    risk <= self.config.training_data_max_red_team_risk and
                    freshness > 0.6):
                    clean_data.append({
                        "input": frag.get("content", ""),
                        "target_score": score,
                        "efs": frag.get("efs", 0.0),
                        "verifier_quality": frag.get("verifier_quality", 0.0),
                        "provenance": frag.get("provenance", {}),
                        "timestamp": datetime.now().isoformat()
                    })

        if clean_data:
            timestamp = datetime.now().isoformat().replace(":", "-")
            (training_dir / f"training_batch_{timestamp}.json").write_text(
                json.dumps(clean_data, indent=2), encoding="utf-8"
            )
            logger.info(f"📦 Training data vault cleaned — {len(clean_data)} high-quality samples prepared")

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
        logger.info("🌍 Synapse background intelligence loop started (daily cycle + lightweight nightly tasks)")

# Global instance (imported by local EM instances for push/pull)
synapse = SynapseMetaAgent()
