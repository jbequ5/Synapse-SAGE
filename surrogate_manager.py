# surrogate_manager.py
"""
SAGE v0.9.14 – SOTA Surrogate Manager for Expensive Simulators
Handles fast inner-loop predictions with uncertainty, adaptive full-simulation triggers,
and self-improvement signals for Meta-RL + dynamic specialist discovery.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class SurrogateManager:
    """SOTA surrogate manager for recursive optimization loops with heavy simulators."""

    def __init__(self, n_initial_full_runs: int = 40, uncertainty_threshold: float = 0.12,
                 full_run_interval: int = 25):
        self.n_initial_full_runs = n_initial_full_runs
        self.uncertainty_threshold = uncertainty_threshold
        self.full_run_interval = full_run_interval
        self.full_run_history: List[Tuple[np.ndarray, float]] = []  # (input_vector, true_X_score)
        self.surrogate = GaussianProcessRegressor(
            kernel=C(1.0) * Matern(length_scale=1.0, nu=2.5),
            n_restarts_optimizer=15,
            random_state=42,
            normalize_y=True
        )
        self.trained = False
        self.iteration = 0
        logger.info("🔬 SurrogateManager initialized — SOTA GP with uncertainty + adaptive triggering")

    def add_full_run(self, input_vector: np.ndarray, true_X_score: float) -> None:
        """Record a ground-truth full simulation run and retrain if enough data."""
        self.full_run_history.append((input_vector.copy(), float(true_X_score)))
        if len(self.full_run_history) >= self.n_initial_full_runs:
            self._train_surrogate()
        logger.debug(f"Full simulation recorded — total full runs: {len(self.full_run_history)}")

    def predict_with_uncertainty(self, input_vector: np.ndarray) -> Tuple[float, float]:
        """Fast surrogate prediction + uncertainty. Used in the inner recursive loop."""
        self.iteration += 1
        if not self.trained or len(self.full_run_history) < 10:
            return 0.0, 1.0  # high uncertainty until trained
        pred, std = self.surrogate.predict(input_vector.reshape(1, -1), return_std=True)
        return float(pred[0]), float(std[0])

    def should_trigger_full_simulation(self, uncertainty: float) -> bool:
        """SOTA adaptive triggering: uncertainty-based + periodic."""
        if not self.trained:
            return True
        if uncertainty > self.uncertainty_threshold:
            return True
        # Periodic ground-truth to keep surrogate honest
        return self.iteration % self.full_run_interval == 0

    def _train_surrogate(self) -> None:
        """Train on accumulated full runs."""
        if len(self.full_run_history) < self.n_initial_full_runs:
            return
        X = np.array([row[0] for row in self.full_run_history])
        y = np.array([row[1] for row in self.full_run_history])
        self.surrogate.fit(X, y)
        self.trained = True
        logger.info(f"✅ Surrogate retrained on {len(self.full_run_history)} full runs")

    def get_surrogate_error_signal(self) -> float:
        """Error signal for Meta-RL, dynamic specialist discovery, and KAS."""
        if not self.trained or len(self.full_run_history) < 8:
            return 0.0
        recent_X = np.array([row[0] for row in self.full_run_history[-8:]])
        recent_true = np.array([row[1] for row in self.full_run_history[-8:]])
        pred = self.surrogate.predict(recent_X)
        errors = np.abs(pred - recent_true)
        return float(np.mean(errors))

    def get_surrogate_stats(self) -> Dict[str, float]:
        """Observability for KAS success metrics and economic subsystem."""
        if not self.trained:
            return {"trained": False, "avg_uncertainty": 1.0}
        return {
            "trained": True,
            "avg_uncertainty": float(np.mean([self.predict_with_uncertainty(v)[1] for v, _ in self.full_run_history[-10:]])),
            "error_signal": self.get_surrogate_error_signal()
        }

# Global instance (imported where needed)
surrogate_manager = SurrogateManager()
