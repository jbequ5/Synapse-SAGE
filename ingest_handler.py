# PRIVATE Synapse backend — ingest_handler.py (or add to core.py)
# Maximum intelligence ingestion + strict order enforcement
# Fragments → Rescore + Mine first → Meta-RL + NN → Distillation last

from fastapi import APIRouter, Request
from datetime import datetime
import logging
from typing import Dict, Any

from fragment_rescoring import FragmentRescoringEngine
from graph_mining import graph_miner
from meta_rl_loop import MetaRLLoop
from neural_net_head import NeuralNetHead
from model_distillation import model_distiller
from knowledge_acquisition import recursive_kas
from polishing_loop import run_synapse_polishing_loop
from vault_promotion_gate import vault_promotion_gate
from surrogate_manager import surrogate_manager
from defense_red_team import defense_red_team
from economic_layer import economic_layer
from utils import load_shared_vaults, save_to_vaults

logger = logging.getLogger(__name__)
router = APIRouter()

rescoring_engine = FragmentRescoringEngine()
meta_rl = MetaRLLoop()
neural_head = NeuralNetHead()

@router.post("/ingest")
async def ingest(request: Request):
    payload = await request.json()
    fragments = payload.get("fragments", [])
    telemetry = payload.get("telemetry", {})
    provenance = payload.get("provenance", {"source": "unknown"})

    logger.info(f"📥 Ingest received — {len(fragments)} fragments | gaps: {len(telemetry.get('gaps', []))}")

    # ── 1. RESCORING + MINING FIRST (mandatory step)
    if fragments:
        logger.info("🔄 Step 1: Rescoring + Graph Mining")
        updated_fragments = rescoring_engine.run_nightly_rescoring(
            fragments,
            current_fragment_count=len(fragments),
            days_running=30
        )
        mined_insights = graph_miner.mine({"internal": updated_fragments})
        save_to_vaults(mined_insights, "shared_vaults", vault_name="internal/mined")

    # ── 2. META-RL + NN OBJECTIVES
    logger.info("🔄 Step 2: Meta-RL Loop + NeuralNetHead")
    rl_results = meta_rl.run_audit_and_improve()
    scored = neural_head.score_advice(
        advice=telemetry.get("gaps", [{}])[0] if telemetry.get("gaps") else {"content": "ingest_cycle"},
        outcome=telemetry
    )

    # Route granular gaps to the right subsystems
    for gap in telemetry.get("gaps", []):
        action = gap.get("suggested_action")
        if action == "new_mope_model":
            surrogate_manager.trigger_new_mope_model(gap)
        elif action == "new_nn_objective":
            neural_head.discover_and_test_new_objective(gap)
        elif action == "new_kas_training_signal":
            recursive_kas._targeted_hunt_for_step("kas_gap", gap, domain_tag="general")
        elif action == "new_verifier_model":
            defense_red_team.create_new_verifier(gap)
        elif action in ["new_synthesis_objective", "new_novelty_objective"]:
            polishing_loop.run_synapse_polishing_loop()  # triggers polishing

    # ── 3. DISTILLATION PREP + DISTILLATION LAST
    logger.info("🔄 Step 3: Distillation Prep + Final Distillation")
    distillation_ready = model_distiller.check_readiness(scored)
    if distillation_ready:
        model_distiller.run_distillation(prepared_batch=rl_results)

    # Final vault promotion & economic synthesis
    vault_promotion_gate.evaluate_all(updated_fragments if 'updated_fragments' in locals() else fragments)
    economic_layer.polish_and_synthesize(updated_fragments if 'updated_fragments' in locals() else [])

    logger.info("✅ Ingest complete — full ordered pipeline executed")
    return {
        "status": "ingested",
        "fragments_processed": len(fragments),
        "gaps_processed": len(telemetry.get("gaps", [])),
        "meta_rl_complete": True,
        "distillation_ready": distillation_ready,
        "timestamp": datetime.now().isoformat()
    }
