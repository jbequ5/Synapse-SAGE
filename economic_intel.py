"""
Synapse Economic Layer — v0.9.13 MAXIMUM SOTA
Marketplace, proposal engine, and product polishing subsystem optimized for MAXIMUM VALUE CREATION.
Fully vector-first 5-objective design with dynamic ROI, red-team gating, marketplace feedback loop,
and compounding back into KAS/Meta-RL. Internal-vault-only persistence.
"""

import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from synapse.config import SynapseConfig
from synapse.graph_mining import graph_miner
from synapse.meta_rl_loop import meta_rl_loop
from synapse.neural_net_head import neural_net_head
from synapse.defense_red_team import defense_red_team
from synapse.kas import recursive_kas
from synapse.utils import load_shared_vaults, save_to_vaults

logger = logging.getLogger(__name__)

class EconomicLayer:
    """Economic Layer — maximum value creation engine for SAGE."""

    def __init__(self, config: SynapseConfig = None):
        self.config = config or SynapseConfig()
        self.products_dir = Path("synapse/data/internal_vaults/economic_products")
        self.products_dir.mkdir(parents=True, exist_ok=True)
        self.market_feedback = []  # tracks uptake, conversion, EFS lift, vector impact
        logger.info("💰 EconomicLayer v0.9.13 MAXIMUM SOTA initialized — full vector-first 5-objective value creation + flywheel compounding")

    def polish_and_synthesize(self, fragments: List[Dict]) -> List[Dict]:
        """Main synthesis pipeline optimized for maximum value creation."""
        logger.info(f"💰 Starting product synthesis — {len(fragments)} input fragments from internal vaults")

        # 1. Aggressive red-team gating (Defense integration)
        hardened = defense_red_team.red_team_and_harden(fragments)

        # Pull live strongest objectives from GraphMiner for intelligent prioritization
        strongest = graph_miner._get_strongest_objectives()

        polished_products = []
        for frag in hardened:
            product = self._synthesize_product(frag, strongest)
            if product and product.get("projected_value_creation", 0) > 0.68:  # higher bar for economic value
                polished_products.append(product)

        # 2. Final red-team on polished products
        final_products = defense_red_team.red_team_and_harden(polished_products)

        # 3. Persist ONLY to internal ranked vaults (private-gatekeeper rule)
        save_to_vaults(final_products, self.config.shared_vault_path, vault_name="internal/economic_products")

        # 4. Close the flywheel: compound back into KAS + Meta-RL
        meta_rl_loop.run_audit_and_improve(final_products)
        recursive_kas.recursive_hunt(final_products[:10])  # recursive improvement on high-value products

        logger.info(f"✅ EconomicLayer synthesis complete — {len(final_products)} high-value products published to internal vaults")
        return final_products

    def _synthesize_product(self, fragment: Dict, strongest_objectives: Dict = None) -> Dict:
        """Deep polishing with dynamic value creation estimation using full 5-objective vector."""
        neural_score = neural_net_head.score_advice(fragment, {"actual_impact": 0.92})
        red_team_risk = fragment.get("red_team_risk", 0.0) if isinstance(fragment.get("red_team_risk"), (int, float)) else 0.0
        kas_freshness = recursive_kas.assess_freshness(fragment)

        vec = neural_score.get("objective_vector", {}) if isinstance(neural_score, dict) else neural_score

        # Fully vector-driven projected value creation (exact weighting from Economic specs)
        projected_value = (
            vec.get("value_creation", 0.0) * 0.40 +           # primary economic driver
            vec.get("implementation_quality", 0.0) * 0.20 +
            vec.get("robustness", 0.0) * 0.20 +               # long-term sustainability
            (1.0 - red_team_risk) * 0.10 +
            kas_freshness * 0.05 +
            vec.get("learning_to_learn", 0.0) * 0.05          # meta-improvement bonus
        )

        product = {
            "type": "synthesized_product",
            "title": f"High-Value Product — {fragment.get('source', 'KAS')} [{fragment.get('pattern_id', fragment.get('node_id', 'unknown'))[:30]}]",
            "description": fragment.get("content", "")[:1500],
            "objective_vector": vec,
            "combined_score": round(neural_score.get("combined_score", 0.0), 4),
            "projected_value_creation": round(projected_value, 4),
            "monetization_strategy": self._generate_dynamic_monetization(projected_value, vec),
            "target_audience": ["EM_miners", "synapse_users", "product_dev", "alpha_contributors", "sponsors"],
            "timestamp": datetime.now().isoformat(),
            "provenance": {
                "source_fragment_id": fragment.get("pattern_id") or fragment.get("node_id"),
                "kas_recursion_depth": getattr(recursive_kas, "recursion_depth", 0),
                "red_team_risk": red_team_risk,
                "strongest_objectives": strongest_objectives,
                "freshness": kas_freshness
            }
        }

        product["proposal"] = self._generate_proposal(product)
        return product

    def _generate_dynamic_monetization(self, projected_value: float, neural_score: Dict) -> Dict:
        """Adaptive monetization fully driven by the 5-objective vector."""
        robustness = neural_score.get("robustness", 0.5)
        learning_to_learn = neural_score.get("learning_to_learn", 0.5)
        value_creation = neural_score.get("value_creation", 0.5)

        if projected_value > 0.85 and robustness > 0.78:
            return {
                "model": "premium_license + revenue_share",
                "pricing": "tiered (free core + $49–$299/mo)",
                "expected_roi": round(projected_value * 1.95, 2),
                "vector_justification": "high value_creation + robustness"
            }
        elif projected_value > 0.72 or learning_to_learn > 0.78:
            return {
                "model": "bundle_with_distilled_enigma_model",
                "pricing": "$99 one-time + usage credits",
                "expected_roi": round(projected_value * 1.55, 2),
                "vector_justification": "strong learning_to_learn + value_creation"
            }
        else:
            return {
                "model": "open_source_with_commercial_support",
                "pricing": "community + paid enterprise support",
                "expected_roi": round(projected_value * 1.25, 2),
                "vector_justification": "balanced vector with community growth potential"
            }

    def _generate_proposal(self, product: Dict) -> Dict:
        """High-quality, data-grounded business proposal for marketplace/toolkits."""
        return {
            "type": "marketplace_proposal",
            "product_id": product.get("title"),
            "monetization_strategy": product["monetization_strategy"],
            "projected_efs_lift": round(product.get("projected_value_creation", 0) * 0.35, 3),
            "target_price": round(product.get("projected_value_creation", 0) * 1250, 2),
            "timestamp": datetime.now().isoformat(),
            "vector_summary": {k: round(v, 3) for k, v in product.get("objective_vector", {}).items()}
        }

    def record_market_feedback(self, product_id: str, uptake: float, efs_lift: float, conversion: float, vector_impact: Dict = None):
        """Marketplace feedback loop — critical for maximum value creation and Meta-RL tuning."""
        entry = {
            "product_id": product_id,
            "uptake": uptake,
            "efs_lift": efs_lift,
            "conversion": conversion,
            "vector_impact": vector_impact or {},
            "timestamp": datetime.now().isoformat()
        }
        self.market_feedback.append(entry)
        if len(self.market_feedback) > 500:
            self.market_feedback = self.market_feedback[-500:]
        logger.info(f"💰 Marketplace feedback recorded for {product_id} — uptake: {uptake:.2f}, EFS lift: {efs_lift:.3f}")

    def get_market_summary(self) -> Dict:
        """Rich summary for Synapse Chat / dashboard / nightly cycle with real value metrics."""
        if not self.market_feedback:
            return {"total_products": 0, "total_value_created": 0.0, "avg_uptake": 0.0, "avg_efs_lift": 0.0}
        total_uptake = np.mean([f["uptake"] for f in self.market_feedback])
        total_efs_lift = np.mean([f["efs_lift"] for f in self.market_feedback])
        return {
            "total_products": len(self.market_feedback),
            "avg_uptake": round(total_uptake, 3),
            "avg_efs_lift": round(total_efs_lift, 3),
            "total_value_created": round(total_uptake * total_efs_lift * 1250, 2),  # calibrated to projected pricing
            "timestamp": datetime.now().isoformat()
        }

# Global instance
economic_layer = EconomicLayer()
