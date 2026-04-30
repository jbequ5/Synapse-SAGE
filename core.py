"""
Synapse Meta-Agent Core — v0.9.13 MAXIMUM SOTA
Central orchestrator for the entire Synapse intelligence layer.
Fully vector-first 5-objective design. Coordinates Graph Mining → NeuralNetHead → Meta-RL → KAS → Defense → Economic Layer → Distillation.
Enforces private-gatekeeper handoff and internal ranked vaults only.
"""

import time
import logging
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from synapse.config import SynapseConfig
from synapse.graph_mining import graph_miner
from synapse.meta_rl_loop import meta_rl_loop
from synapse.neural_net_head import neural_net_head
from synapse.model_distillation import model_distiller
from synapse.chat_interface import synapse_chat
from synapse.defense_red_team import defense_red_team
from synapse.kas import recursive_kas
from synapse.economic_layer import economic_layer
from synapse.utils import load_shared_vaults, save_to_vaults, prune_old_vaults

logger = logging.getLogger(__name__)

class SynapseMetaAgent:
    """Central Synapse Meta-Agent — orchestrates all intelligence subsystems for SAGE."""

    def __init__(self, config: SynapseConfig = None):
        self.config = config or SynapseConfig()
        self.graph_miner = graph_miner
        self.meta_rl = meta_rl_loop
        self.neural_head = neural_net_head
        self.distiller = model_distiller
        self.chat = synapse_chat
        self.defense = defense_red_team
        self.kas = recursive_kas
        self.economic = economic_layer
        
        self.last_loop = datetime.now()
        logger.info("🚀 SynapseMetaAgent v0.9.13 MAX SOTA initialized — full vector-first integration + private-gatekeeper model complete")

    def run_daily_intelligence_cycle(self):
        """Full nightly intelligence cycle — the heart of the self-improving flywheel."""
        logger.info("🔄 Starting Synapse daily intelligence cycle")
        
        try:
            # Load only internal ranked vaults (gate already happened)
            vaults = load_shared_vaults(self.config.shared_vault_path, vault_type="internal")
            
            # 1. Graph Mining on ranked internal vaults
            mined_patterns = self.graph_miner.mine(vaults)
            
            # 2. Meta-RL audit & improvement (targets weakest objectives)
            rl_results = self.meta_rl.run_audit_and_improve(mined_patterns)
            
            # 3. NeuralNetHead vector scoring & calibration
            scored_insights = self.neural_head.score_and_calibrate(rl_results)
            
            # 4. Defense Red Team hardening
            hardened_insights = self.defense.run_ahe_cycle(scored_insights)
            
            # 5. KAS recursive knowledge acquisition
            kas_results = self.kas.recursive_hunt(hardened_insights)
            
            # 6. Economic polishing & synthesis
            polished_products = self.economic.polish_and_synthesize(kas_results)
            
            # 7. Lightweight nightly tasks + training data preparation
            self._run_lightweight_nightly_tasks(vaults, polished_products)
            
            # 8. Save polished artifacts
            save_to_vaults(polished_products, self.config.shared_vault_path, vault_name="internal")
            
            # 9. Check readiness for model distillation
            distillation_ready = self.distiller.check_readiness(polished_products)
            
            self.last_loop = datetime.now()
            logger.info(f"✅ Synapse daily cycle complete — {len(polished_products)} new/improved artifacts | Distillation ready: {distillation_ready}")
            
            return {
                "status": "success",
                "artifacts_generated": len(polished_products),
                "distillation_ready": distillation_ready,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Synapse daily cycle failed: {e}")
            return {"status": "error", "error": str(e)}

    def _run_lightweight_nightly_tasks(self, vaults: Dict, polished_products: List[Dict]):
        """Pruning, training data cleaning, market summary, health reporting."""
        prune_old_vaults(self.config.shared_vault_path, max_age_days=14)
        self._clean_and_prepare_training_data(vaults, polished_products)
        
        summary = self.economic.get_market_summary()
        logger.info(f"💰 Nightly market summary — Total value created: {summary.get('total_value_created', 0):.2f}")

        health_report = {
            "timestamp": datetime.now().isoformat(),
            "red_team_summary": self.defense.get_red_team_summary(),
            "kas_fragments_added": len(polished_products),
            "market_summary": summary
        }
        save_to_vaults([health_report], self.config.shared_vault_path, vault_name="nightly_reports")

    def _clean_and_prepare_training_data(self, vaults: Dict, polished_products: List[Dict]):
        """Prepares high-quality data for model distillation using all 5 objectives."""
        training_dir = Path(self.config.training_data_vault_path)
        training_dir.mkdir(parents=True, exist_ok=True)

        clean_data = []
        for vault_name, fragments in vaults.items():
            for frag in fragments:
                score = frag.get("hybrid_rank_score", frag.get("combined_score", 0.0))
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
                        "timestamp": datetime.now().isoformat(),
                        "objective_vector": frag.get("objective_vector", {})
                    })

        if clean_data:
            timestamp = datetime.now().isoformat().replace(":", "-")
            (training_dir / f"training_batch_{timestamp}.json").write_text(
                json.dumps(clean_data, indent=2), encoding="utf-8"
            )
            logger.info(f"📦 Training data vault cleaned — {len(clean_data)} high-quality samples prepared")

    def get_synapse_chat_response(self, user_query: str, user_tier: str = "standard") -> Dict[str, Any]:
        """Public entry point for Synapse Chat / Co-Pilot."""
        return self.chat.handle_query(user_query, user_tier)

    def start_background_loop(self):
        """Starts the daily intelligence cycle in background thread."""
        def loop():
            while True:
                self.run_daily_intelligence_cycle()
                time.sleep(self.config.daily_loop_interval_seconds)
        threading.Thread(target=loop, daemon=True).start()
        logger.info("🌍 Synapse background intelligence loop started (daily cycle + lightweight nightly tasks)")

# Global instance
synapse = SynapseMetaAgent()
