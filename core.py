"""
Synapse Meta-Agent Core — v0.9.13 MAXIMUM SOTA
Central orchestrator for the entire Synapse intelligence layer.
Fully vector-first 5-objective design. Coordinates Graph Mining → NeuralNetHead → Meta-RL → KAS → Defense → Economic Layer → Distillation.
Enforces private-gatekeeper handoff and internal ranked vaults only.
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List

from config import SynapseConfig
from graph_mining import graph_miner
from meta_rl_loop import meta_rl_loop
from neural_net_head import neural_net_head
from model_distillation import model_distiller
from chat_interface import synapse_chat
from defense_red_team import defense_red_team
from kas import recursive_kas  # assuming this exists
from economic_layer import economic_layer
from utils import load_shared_vaults, save_to_vaults, prune_old_vaults

logger = logging.getLogger(__name__)

class SynapseMetaAgent:
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
        logger.info("🚀 SynapseMetaAgent v0.9.13 MAXIMUM SOTA initialized — full vector-first integration")

    def run_daily_intelligence_cycle(self):
        """Full nightly intelligence cycle — the heart of the self-improving flywheel."""
        logger.info("🔄 Starting Synapse daily intelligence cycle")

        try:
            vaults = load_shared_vaults(self.config.shared_vault_path, vault_type="internal")
            mined_patterns = self.graph_miner.mine(vaults)

            # Meta-RL is now the true final pass (no-arg call)
            rl_results = self.meta_rl.run_audit_and_improve()

            # NeuralNetHead scoring (correct usage)
            scored_insights = self.neural_head.score_advice(mined_patterns[0] if mined_patterns else {"content": "cycle_summary"})

            # Defense, KAS, Economic (in correct order)
            hardened = self.defense.run_ahe_cycle()
            kas_results = self.kas.recursive_hunt(hardened)
            polished_products = self.economic.polish_and_synthesize(kas_results)

            self._run_lightweight_nightly_tasks(vaults, polished_products)

            save_to_vaults(polished_products, self.config.shared_vault_path, vault_name="internal")

            distillation_ready = self.distiller.check_readiness(polished_products)

            self.last_loop = datetime.now()
            logger.info(f"✅ Synapse daily cycle complete — {len(polished_products)} artifacts | Distillation ready: {distillation_ready}")
            return {"status": "success", "artifacts_generated": len(polished_products), "distillation_ready": distillation_ready}

        except Exception as e:
            logger.error(f"❌ Synapse daily cycle failed: {e}")
            return {"status": "error", "error": str(e)}

    def _run_lightweight_nightly_tasks(self, vaults: Dict, polished_products: List[Dict]):
        prune_old_vaults(self.config.shared_vault_path, max_age_days=14)
        # Training data prep is now handled inside Meta-RL Distillation Prep Vault — no duplication
        summary = self.economic.get_market_summary()
        logger.info(f"💰 Nightly market summary — Total value created: {summary.get('total_value_created', 0):.2f}")
        self._check_distil_sn97_updates()

    def _check_distil_sn97_updates(self):
        """Nightly check for updates to unarbos/distil (Distil SN97) — logs latest techniques."""
        try:
            r = requests.get("https://api.github.com/repos/unarbos/distil/commits/main", timeout=10)
            if r.status_code == 200:
                latest = r.json()[0]
                commit_date = latest["commit"]["author"]["date"]
                message = latest["commit"]["message"][:150]
                logger.info(f"📡 Distil SN97 latest commit: {commit_date} — {message}")
        except Exception as e:
            logger.debug(f"Distil SN97 update check failed: {e}")

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
        logger.info("🌍 Synapse background intelligence loop started (daily cycle + lightweight nightly tasks + Distil SN97 sync)")

# Global instance
synapse = SynapseMetaAgent()
