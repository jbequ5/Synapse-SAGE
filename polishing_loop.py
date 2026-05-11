# synapse_polishing_loop.py
# SAGE v0.9.14 – Synapse Nightly Polishing Loop
# Central orchestration for re-scoring, vault promotion, EconomicLayer, and flywheel closure

import logging
from datetime import datetime
from synapse.fragment_rescoring import FragmentRescoringEngine
from synapse.economic_layer import economic_layer
from synapse.utils import load_shared_vaults, save_to_vaults
from synapse.defense_red_team import defense_red_team   # for stability integration

logger = logging.getLogger(__name__)

def run_synapse_polishing_loop(days_running: int = 30):
    """Main nightly polishing loop for Synapse — calls re-scoring + EconomicLayer."""
    logger.info("🔄 Starting Synapse nightly polishing loop")

    # 1. Load all fragments from internal vaults
    fragments = load_shared_vaults(vault_name="internal")

    # 2. Re-score everything with latest global context
    rescoring = FragmentRescoringEngine()
    updated_fragments = rescoring.run_nightly_rescoring(
        fragments,
        current_fragment_count=len(fragments),
        days_running=days_running
    )

    # 3. Run full EconomicLayer synthesis (hardened scoring + polishing + monetization)
    polished_products = economic_layer.polish_and_synthesize(updated_fragments)

    # 4. Optimal upgrade: Defense stability check + circuit breaker
    # (wires directly into the existing polishing loop using the upgraded defense_red_team)
    defense_report = defense_red_team.run_ahe_cycle()
    stability_score = defense_report.get("stability_score", 1.0)
    negative_examples = defense_report.get("negative_examples", [])

    if stability_score < 0.75:
        logger.warning(f"🚨 FLYWHEEL STABILITY ALERT — score {stability_score:.4f}. Negative examples detected. Pausing high-priority escalation.")
        # Circuit breaker: skip high-priority escalation on instability
        high_priority = []
    else:
        # 4. Optional high-priority gap handling (KAS escalation)
        high_priority = [f for f in polished_products if f.get("final_impact_score", 0) >= 0.85]

    if high_priority:
        logger.info(f"🚀 {len(high_priority)} high-priority gaps escalated for deeper KAS hunts")

    logger.info(f"✅ Synapse polishing loop completed — {len(polished_products)} products in internal vaults | Stability: {stability_score:.4f}")
    return polished_products
