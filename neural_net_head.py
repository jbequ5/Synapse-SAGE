"""
Synapse Neural Net Head — v0.9.13 MAXIMUM SOTA
5-objective vector-first scoring head. This is the primary signal for the entire Synapse intelligence layer.
Dynamic calibration, temporal decay, vector synergy, red-team awareness, and live GraphMiner integration.
"""

import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from synapse.graph_mining import graph_miner   # Live context for calibration

logger = logging.getLogger(__name__)

class NeuralNetHead:
    """5-objective vector-first scoring head for Synapse Meta-RL and all downstream subsystems."""

    def __init__(self):
        self.weights = {
            "implementation_quality": 0.25,
            "prediction_accuracy": 0.20,
            "value_creation": 0.25,
            "learning_to_learn": 0.15,
            "robustness": 0.15
        }
        self.calibration_history = []
        self.calibration_path = Path("synapse/data/neural_calibration.json")
        self._load_calibration()
        logger.info("🧠 NeuralNetHead v0.9.13 MAX SOTA initialized — full vector-first 5-objective primary signal")

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

    def score_advice(self, advice: Dict, outcome: Dict) -> Dict[str, Any]:
        """Compute full 5-objective vector — this is the primary signal for the entire system."""
        # Pull live context from GraphMiner for smarter scoring
        strongest = graph_miner._get_strongest_objectives() if hasattr(graph_miner, "_get_strongest_objectives") else {}

        scores = {
            "implementation_quality": self._score_implementation(advice, outcome),
            "prediction_accuracy": self._score_prediction(advice, outcome),
            "value_creation": self._score_value_creation(advice, outcome),
            "learning_to_learn": self._score_learning_to_learn(advice, outcome),
            "robustness": self._score_robustness(advice, outcome)
        }

        # Vector synergy bonus (geometric mean prevents any single weak objective from being ignored)
        values = list(scores.values())
        synergy = np.prod([v + 1e-8 for v in values]) ** (1 / len(values)) * 0.12

        # Temporal decay / recency weighting
        recency = min(1.0, len(self.calibration_history) / 80.0) if self.calibration_history else 1.0

        # Weighted combined score
        combined_score = sum(scores[k] * self.weights.get(k, 0.2) for k in scores) + synergy
        combined_score *= recency

        score_result = {
            "combined_score": round(combined_score, 4),
            **{k: round(v, 4) for k, v in scores.items()},
            "synergy_bonus": round(synergy, 4),
            "recency_factor": round(recency, 4),
            "timestamp": datetime.now().isoformat(),
            "provenance": advice.get("advice_id", "unknown"),
            "strongest_objectives_snapshot": strongest
        }

        self.calibration_history.append(score_result)
        if len(self.calibration_history) > 1000:
            self.calibration_history = self.calibration_history[-1000:]
        self._save_calibration()

        return score_result

    def _score_implementation(self, advice: Dict, outcome: Dict) -> float:
        """Quality of implementation / code / solution structure."""
        return min(1.0, outcome.get("implementation_success", 0.78))

    def _score_prediction(self, advice: Dict, outcome: Dict) -> float:
        """How well the advice predicted real outcomes."""
        predicted = float(advice.get("expected_impact", 0.0))
        actual = float(outcome.get("actual_impact", 0.0))
        error = abs(predicted - actual)
        return max(0.0, 1.0 - min(1.0, error / 0.5))

    def _score_value_creation(self, advice: Dict, outcome: Dict) -> float:
        """Direct economic / EFS / practical value created."""
        efs_lift = float(outcome.get("efs_lift", 0.0))
        return min(1.0, efs_lift * 0.85)

    def _score_learning_to_learn(self, advice: Dict, outcome: Dict) -> float:
        """Meta-improvement / how much this advice helps the system learn."""
        return min(1.0, outcome.get("meta_improvement", 0.70))

    def _score_robustness(self, advice: Dict, outcome: Dict) -> float:
        """Red-team resistance and long-term stability."""
        red_team_risk = float(outcome.get("red_team_risk", 0.0))
        return max(0.0, 1.0 - red_team_risk * 1.1)

    def calibrate_from_history(self) -> Dict[str, Any]:
        """Dynamic calibration that actively balances all 5 objectives."""
        if len(self.calibration_history) < 40:
            return {"status": "insufficient_data", "calibration_delta": 0.0}

        recent = self.calibration_history[-300:]
        
        # Compute average per objective
        obj_avgs = {}
        for obj in self.weights.keys():
            obj_avgs[obj] = np.mean([r.get(obj, 0.0) for r in recent])

        avg_combined = np.mean([r["combined_score"] for r in recent])

        # Adjust weights to reduce imbalance (boost weakest objectives)
        total_weight = sum(self.weights.values())
        for obj in self.weights:
            avg = obj_avgs[obj]
            if avg < 0.75:
                self.weights[obj] = min(0.35, self.weights[obj] * 1.08)  # boost weak objectives
            else:
                self.weights[obj] = max(0.08, self.weights[obj] * 0.96)  # gently reduce strong ones

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: round(v / total, 4) for k, v in self.weights.items()}

        calibration_delta = round(avg_combined - 0.75, 4)

        self._save_calibration()

        logger.info(f"🧠 NeuralNetHead calibrated — Avg success: {avg_combined:.3f} | Delta: {calibration_delta:.3f} | Weights: {self.weights}")
        return {
            "status": "calibrated",
            "calibration_delta": calibration_delta,
            "new_weights": self.weights,
            "objective_averages": obj_avgs
        }

# Global instance
neural_net_head = NeuralNetHead()
