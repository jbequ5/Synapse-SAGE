"""
Synapse Meta-RL Loop — v0.9.13 MAXIMUM SOTA
Fully vector-first: consumes the complete 5-objective vector from NeuralNetHead.
Dynamic, context-aware targeting of weakest objectives + GraphMiner synergy.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

from synapse.config import SynapseConfig
from synapse.utils import load_shared_vaults, save_to_vaults
from synapse.neural_net_head import neural_net_head
from synapse.defense_red_team import defense_red_team
from synapse.graph_mining import graph_miner   # ← New integration

logger = logging.getLogger(__name__)

class MetaRLLoop:
    def __init__(self, config: SynapseConfig = None):
        self.config = config or SynapseConfig()
        self.audit_history_path = Path("synapse/data/audit_history.json")
        self.audit_history = self._load_audit_history()
        logger.info("🔄 MetaRLLoop v0.9.13 MAX SOTA initialized — full 5-objective vector-driven design")

    def _load_audit_history(self) -> List[Dict]:
        if self.audit_history_path.exists():
            try:
                return json.loads(self.audit_history_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    def _save_audit_history(self):
        self.audit_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_history_path.write_text(json.dumps(self.audit_history, indent=2), encoding="utf-8")

    def run_audit_and_improve(self, mined_patterns: List[Dict] = None) -> Dict[str, Any]:
        """Main meta-RL cycle — driven by full 5-objective vector from telemetry and GraphMiner."""
        logger.info("🔄 Running Meta-RL self-audit cycle")

        # Get system-wide strongest/weakest objectives from GraphMiner
        strongest = graph_miner._get_strongest_objectives()

        # Full 5-objective vector from NeuralNetHead (primary signal)
        neural_score = neural_net_head.score_advice(
            {"mined_patterns": len(mined_patterns or []), "strongest_objectives": strongest},
            self._perform_audit()
        )

        # Red-team risk weighting on the full vector
        red_team_risk = self._perform_audit().get("red_team_risk", 0.0)
        final_success_score = neural_score["combined_score"] * (1.0 - red_team_risk * 0.3)

        # Generate targeted proposals based on weakest objectives in the vector
        proposals = self._generate_vector_targeted_proposals(neural_score, strongest)

        calibration_result = neural_net_head.calibrate_from_history()

        self.audit_history.append({
            "timestamp": datetime.now().isoformat(),
            "success_score": final_success_score,
            "neural_scores": neural_score,
            "strongest_objectives": strongest,
            "red_team_risk": red_team_risk,
            "proposals_count": len(proposals),
            "calibration_delta": calibration_result.get("calibration_delta", 0.0)
        })
        self._save_audit_history()

        refined_strategies = self._prepare_refined_strategies(proposals)
        save_to_vaults(refined_strategies, self.config.shared_vault_path, vault_name="strategy")

        logger.info(f"✅ Meta-RL cycle complete — Success Score: {final_success_score:.3f} | Red-team risk: {red_team_risk:.3f} | Weakest objective targeted: {proposals[0]['target'] if proposals else 'none'}")

        return {
            "status": "success",
            "success_score": final_success_score,
            "neural_scores": neural_score,
            "strongest_objectives": strongest,
            "proposals": proposals,
            "calibration_delta": calibration_result.get("calibration_delta", 0.0)
        }

    def _perform_audit(self) -> Dict:
        """Collect recent audit data for risk and context."""
        recent = self.audit_history[-50:] if len(self.audit_history) > 50 else self.audit_history
        return {
            "recent_advice_count": len(recent),
            "avg_past_success": np.mean([a.get("success_score", 0.0) for a in recent]) if recent else 0.65,
            "red_team_risk": np.mean([a.get("red_team_risk", 0.0) for a in recent]) if recent else 0.0
        }

    def _generate_vector_targeted_proposals(self, neural_score: Dict, strongest: Dict) -> List[Dict]:
        """Generate targeted proposals based on the weakest objectives in the full vector."""
        proposals = []
        objective_scores = {k: v for k, v in neural_score.items() 
                           if k in ["implementation_quality", "prediction_accuracy", "value_creation", "learning_to_learn", "robustness"]}

        # Target the single weakest objective for high-signal, focused improvement
        for obj, score in sorted(objective_scores.items(), key=lambda x: x[1]):
            if score < 0.78:  # tunable threshold
                proposals.append({
                    "target": obj,
                    "change": f"Improve {obj} (current: {score:.3f}) — system weakest objective",
                    "expected_impact": f"+{round(0.12 + (1.0 - score) * 0.45, 2)} overall success",
                    "priority": "high",
                    "system_strongest": list(strongest.keys())[:2]
                })
                break  # focus on the single most important gap
        return proposals

    def _prepare_refined_strategies(self, proposals: List[Dict]) -> List[Dict]:
        """Prepare refined strategies for the shared strategy vault."""
        strategies = []
        for p in proposals:
            strategies.append({
                "type": "refined_strategy",
                "target_subsystem": p["target"],
                "description": p["change"],
                "expected_impact": p["expected_impact"],
                "timestamp": datetime.now().isoformat(),
                "source": "meta_rl_loop"
            })
        return strategies

# Global instance
meta_rl_loop = MetaRLLoop()
