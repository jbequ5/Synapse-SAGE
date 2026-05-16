# synapse_polishing_loop.py
# SAGE v0.9.15 – Synapse Nightly Polishing Loop
# Central orchestration for re-scoring, vault promotion, EconomicLayer, PINO distillation, and flywheel closure

import logging
from datetime import datetime
from synapse.fragment_rescoring import FragmentRescoringEngine
from synapse.economic_layer import economic_layer
from synapse.utils import load_shared_vaults, save_to_vaults
from synapse.defense_red_team import defense_red_team
from surrogate_manager import surrogate_manager

# PINO distillation engine (new custom surrogates + MoDE specialists)
from synapse.pino_distillation import PINODistillationEngine

logger = logging.getLogger(__name__)

def run_synapse_polishing_loop(days_running: int = 30):
    """Main nightly polishing loop for Synapse — calls re-scoring + EconomicLayer + PINO distillation."""
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

    # 4. PINO distillation — creates new MoDE specialists and custom surrogate entries in the bank
    #    This is the core of the surrogate evolution layer and runs on high-yield fragments only
    distillation_engine = PINODistillationEngine(
        mope_mixture=global_mope_mixture,
        mode_mixture=global_mode_mixture,
        pino_bank=global_pino_bank,
        meta_rl_trainer=global_meta_rl_trainer,
        scoring_module=SolveFragmentScoringModule()
    )
    distillation_stats = distillation_engine.run_distillation(days_running=days_running)
    logger.info(f"📈 PINO distillation stats: {distillation_stats}")

    # 5. Optimal upgrade: Defense stability check + circuit breaker
    defense_report = defense_red_team.run_ahe_cycle()
    stability_score = defense_report.get("stability_score", 1.0)
    negative_examples = defense_report.get("negative_examples", [])

    # New: Surrogate error signal for polishing stability
    surrogate_error = surrogate_manager.get_surrogate_error_signal()
    if surrogate_error > 0.15:
        logger.warning(f"🚨 SURROGATE ERROR ALERT — {surrogate_error:.4f}. High uncertainty in simulation loop.")

    if stability_score < 0.75:
        logger.warning(f"🚨 FLYWHEEL STABILITY ALERT — score {stability_score:.4f}. Negative examples detected. Pausing high-priority escalation.")
        high_priority = []
    else:
        # 6. Optional high-priority gap handling (KAS escalation)
        high_priority = [f for f in polished_products if f.get("final_impact_score", 0) >= 0.85]

    if high_priority:
        logger.info(f"🚀 {len(high_priority)} high-priority gaps escalated for deeper KAS hunts")

    logger.info(f"✅ Synapse polishing loop completed — {len(polished_products)} products in internal vaults | Stability: {stability_score:.4f}")
    return polished_productsq
