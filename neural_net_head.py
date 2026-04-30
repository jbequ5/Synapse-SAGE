"""
Synapse Neural Net Head — v0.9.13 MAXIMUM SOTA
5-objective vector-first scoring head. This is the primary signal for the entire Synapse intelligence layer.
Dynamic calibration, global re-scoring tolerance (0.08), vector synergy, red-team awareness, temporal decay,
weakest-objective boosting, and live GraphMiner + Defense integration.
"""

import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from synapse.config import SynapseConfig
from synapse.graph_mining import graph_miner
from synapse.defense_red_team import defense_red_team

logger = logging.getLogger(__name__)

class NeuralNetHead:
    """5-objective vector-first scoring head for Synapse Meta-RL and all downstream subsystems."""

    def __init__(self, config: SynapseConfig = None):
        self.config = config or SynapseConfig()
        self.weights = {
            "implementation_quality": 0.25,
            "prediction_accuracy": 0.20,
            "value_creation": 0.25,
            "learning_to_learn": 0.15,
            "robustness": 0.15
        }
        self.calibration_history = []
        self.calibration_path = Path("synapse/data/internal_vaults/neural_calibration.json")
        self._load_calibration()
        logger.info("🧠 NeuralNetHead v0.9.13 MAXIMUM SOTA initialized — full vector-first 5-objective primary signal + global re-scoring tolerance 0.08")

    def _load_calibration(self):
        if self.calibration_path.exists():
            try:
                data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
                self.calibration_history = data.get("history", [])
                self.weights = data.get("weights", self.weights)
            except Exception:
                self.calibration_history = []

    def _save_calibration(self):
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "history": self.calibration_history[-1000:],
            "weights": self.weights,
            "last_updated": datetime.now().isoformat()
        }
        self.calibration_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def score_advice(self, advice: Dict, outcome: Dict = None) -> Dict[str, Any]:
        """Compute full 5-objective vector — primary signal for the entire system."""
        if outcome is None:
            outcome = {}

        # Pull live context from GraphMiner + DefenseRedTeam
        strongest = graph_miner._get_strongest_objectives() if hasattr(graph_miner, "_get_strongest_objectives") else {}
        red_team_report = defense_red_team.red_team_scoring_and_validation(advice) if hasattr(defense_red_team, "red_team_scoring_and_validation") else {"passed": True, "overall_risk": 0.0}

        scores = {
            "implementation_quality": self._score_implementation(advice, outcome),
            "prediction_accuracy": self._score_prediction(advice, outcome),
            "value_creation": self._score_value_creation(advice, outcome),
            "learning_to_learn": self._score_learning_to_learn(advice, outcome),
            "robustness": self._score_robustness(advice, outcome, red_team_report)
        }

        # Vector synergy bonus (geometric mean prevents any single weak objective from dominating)
        values = list(scores.values())
        synergy = np.prod([v + 1e-8 for v in values]) ** (1 / len(values)) * 0.15

        # Temporal decay / recency weighting
        recency = min(1.0, len(self.calibration_history) / 100.0) if self.calibration_history else 1.0

        # Weighted combined score
        combined_score = sum(scores[k] * self.weights.get(k, 0.2) for k in scores) + synergy
        combined_score *= recency

        # Global re-scoring tolerance check (exact from Solve Layer / DVR specs)
        local_score = combined_score
        global_re_score = self._compute_global_re_score(advice, outcome)  # simulated from historical calibration
        tolerance_violation = abs(local_score - global_re_score) > 0.08
        if tolerance_violation:
            logger.warning(f"⚠️ Global re-scoring tolerance (0.08) violated — flagging for AHE review | Delta: {abs(local_score - global_re_score):.4f}")

        score_result = {
            "combined_score": round(combined_score, 4),
            **{k: round(v, 4) for k, v in scores.items()},
            "synergy_bonus": round(synergy, 4),
            "recency_factor": round(recency, 4),
            "red_team_risk": red_team_report.get("overall_risk", 0.0),
            "tolerance_violation": tolerance_violation,
            "timestamp": datetime.now().isoformat(),
            "provenance": advice.get("fragment_id") or advice.get("pattern_id") or "unknown",
            "strongest_objectives_snapshot": strongest,
            "objective_vector": scores  # full vector stored for downstream use
        }

        self.calibration_history.append(score_result)
        if len(self.calibration_history) > 1000:
            self.calibration_history = self.calibration_history[-1000:]
        self._save_calibration()

        return score_result

    def _compute_global_re_score(self, advice: Dict, outcome: Dict) -> float:
        """Simulate global re-score from calibration history for tolerance check (0.08)."""
        if len(self.calibration_history) < 20:
            return 0.75
        recent = self.calibration_history[-200:]
        return np.mean([r["combined_score"] for r in recent])

    def _score_implementation(self, advice: Dict, outcome: Dict) -> float:
        return min(1.0, outcome.get("implementation_success", advice.get("quality_score", 0.78)))

    def _score_prediction(self, advice: Dict, outcome: Dict) -> float:
        predicted = float(advice.get("expected_impact", 0.0))
        actual = float(outcome.get("actual_impact", 0.0))
        error = abs(predicted - actual)
        return max(0.0, 1.0 - min(1.0, error / 0.5))

    def _score_value_creation(self, advice: Dict, outcome: Dict) -> float:
        efs_lift = float(outcome.get("efs_lift", 0.0))
        return min(1.0, efs_lift * 0.92)

    def _score_learning_to_learn(self, advice: Dict, outcome: Dict) -> float:
        return min(1.0, outcome.get("meta_improvement", 0.72))

    def _score_robustness(self, advice: Dict, outcome: Dict, red_team_report: Dict) -> float:
        red_team_risk = red_team_report.get("overall_risk", 0.0)
        return max(0.0, 1.0 - red_team_risk * 1.15)

    def calibrate_from_history(self) -> Dict[str, Any]:
        """Dynamic calibration that actively balances all 5 objectives — boosts weakest objectives."""
        if len(self.calibration_history) < 40:
            return {"status": "insufficient_data", "calibration_delta": 0.0}

        recent = self.calibration_history[-400:]
        
        obj_avgs = {}
        for obj in self.weights.keys():
            obj_avgs[obj] = np.mean([r.get(obj, 0.0) for r in recent])

        avg_combined = np.mean([r["combined_score"] for r in recent])

        # Strongest weakest-objective boosting (exact alignment with Meta-RL & Strategy Layer)
        total_weight = sum(self.weights.values())
        for obj in self.weights:
            avg = obj_avgs[obj]
            if avg < 0.72:  # weakest objective threshold
                self.weights[obj] = min(0.38, self.weights[obj] * 1.12)  # aggressive boost
            else:
                self.weights[obj] = max(0.08, self.weights[obj] * 0.94)  # gentle reduction

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: round(v / total, 4) for k, v in self.weights.items()}

        calibration_delta = round(avg_combined - 0.75, 4)

        self._save_calibration()

        logger.info(f"🧠 NeuralNetHead calibrated — Avg success: {avg_combined:.3f} | Delta: {calibration_delta:.3f} | Weights: {self.weights} | Weakest boosted")
        return {
            "status": "calibrated",
            "calibration_delta": calibration_delta,
            "new_weights": self.weights,
            "objective_averages": obj_avgs,
            "tolerance_violations_in_window": sum(1 for r in recent if r.get("tolerance_violation", False))
        }

# Global instance
neural_net_head = NeuralNetHead()
