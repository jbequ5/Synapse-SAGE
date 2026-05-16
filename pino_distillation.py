# synapse/pino_distillation.py
# SAGE v0.9.15 – PINO Distillation Engine
# Creates new MoDE specialists and new named entries in the PINO bank from high-yield fragments.

import logging
from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from synapse.fragment import SolveFragment, load_shared_vaults, save_to_vaults
from synapse.mope import MOPEMixture
from synapse.mode import MODEMixture
from synapse.neural_operator_bank import NeuralOperatorBank
from synapse.meta_rl import MetaRLTrainer
from synapse.solve_fragment_scoring import SolveFragmentScoringModule

logger = logging.getLogger(__name__)

class PINODistillationEngine:
    def __init__(self,
                 mope_mixture: MOPEMixture,
                 mode_mixture: MODEMixture,
                 pino_bank: NeuralOperatorBank,
                 meta_rl_trainer: MetaRLTrainer,
                 scoring_module: SolveFragmentScoringModule,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.mope_mixture = mope_mixture
        self.mode_mixture = mode_mixture
        self.pino_bank = pino_bank
        self.meta_rl = meta_rl_trainer
        self.scoring_module = scoring_module
        self.device = device
        self.config = {
            "min_impact_score": 0.85,
            "num_epochs": 8,
            "lr": 3e-4,
            "batch_size": 64
        }

    def run_distillation(self, days_running: int = 30) -> Dict[str, Any]:
        logger.info("🔬 Starting full PINO distillation pipeline")

        fragments: List[SolveFragment] = load_shared_vaults(vault_name="internal")
        high_yield_fragments = [f for f in fragments if f.final_impact_score >= self.config["min_impact_score"]]

        if len(high_yield_fragments) < 50:
            logger.info("⚠️ Insufficient high-yield fragments for distillation this cycle")
            return {"new_mode_specialists": 0, "new_bank_entries": 0, "fragments_used": 0}

        logger.info(f"📦 Using {len(high_yield_fragments)} high-yield fragments for distillation")

        grouped = self._group_fragments_by_step(high_yield_fragments)

        new_mode_specialists = self._distill_new_mode_specialists(grouped)
        new_bank_entries = self._distill_new_bank_entries(grouped)

        self.mode_mixture.add_specialists(new_mode_specialists)
        self.pino_bank.register_new_entries(new_bank_entries)

        self._save_distilled_artifacts(new_mode_specialists, new_bank_entries)

        logger.info(f"✅ PINO distillation complete — {len(new_mode_specialists)} new MoDE specialists + "
                    f"{len(new_bank_entries)} new custom surrogate entries added to bank")

        return {
            "new_mode_specialists": len(new_mode_specialists),
            "new_bank_entries": len(new_bank_entries),
            "fragments_used": len(high_yield_fragments)
        }

    def _group_fragments_by_step(self, fragments: List[SolveFragment]) -> Dict[str, List[SolveFragment]]:
        groups: Dict[str, List[SolveFragment]] = {}
        for f in fragments:
            step = f.metadata.get("step_type", "general")
            groups.setdefault(step, []).append(f)
        return groups

    def _distill_new_mode_specialists(self, grouped: Dict[str, List[SolveFragment]]) -> List[Dict]:
        new_specialists = []
        for step_name, frag_list in grouped.items():
            inputs, targets = self._prepare_distillation_data(frag_list)
            base_backbone = self.pino_bank.get_best_backbone_for_step(step_name)
            adapter = self._create_conditional_adapter(base_backbone)

            self.meta_rl.supervised_fine_tune(
                model=adapter,
                inputs=inputs,
                targets=targets,
                epochs=self.config["num_epochs"],
                lr=self.config["lr"]
            )

            new_specialists.append({
                "name": f"MoDE-{step_name}-v{len(new_specialists)+1}",
                "specialist": adapter,
                "base_backbone": base_backbone.name,
                "trained_on": len(frag_list),
                "final_impact_score": self.scoring_module.score_fragment_batch(frag_list).mean().item()
            })
        return new_specialists

    def _distill_new_bank_entries(self, grouped: Dict[str, List[SolveFragment]]) -> List[Dict]:
        new_entries = []
        for step_name, frag_list in grouped.items():
            best_fragment = max(frag_list, key=lambda f: f.final_impact_score)
            team_recipe = best_fragment.metadata.get("team_composition", {})
            bank_engines = best_fragment.metadata.get("bank_engines_used", [])

            blended_model = self.pino_bank.create_blended_model(
                engine_names=bank_engines,
                mode_specialists=team_recipe.get("mode_experts", [])
            )

            new_entries.append({
                "name": f"{step_name.capitalize()}-Mix-v{len(new_entries)+1}",
                "model": blended_model,
                "recipe": team_recipe,
                "performance": best_fragment.final_impact_score,
                "description": f"Custom surrogate distilled from {len(frag_list)} high-yield fragments"
            })
        return new_entries

    def _prepare_distillation_data(self, fragments: List[SolveFragment]):
        inputs = []
        targets = []
        for f in fragments:
            input_vec = torch.tensor(f.metadata.get("input_features", [0.0] * 32), dtype=torch.float32)
            target_vec = torch.tensor([
                f.value_creation,
                f.implementation_quality,
                f.robustness,
                f.learning_to_learn,
                f.prediction_accuracy
            ], dtype=torch.float32)
            inputs.append(input_vec)
            targets.append(target_vec)
        return torch.stack(inputs).to(self.device), torch.stack(targets).to(self.device)

    def _create_conditional_adapter(self, base_backbone):
        class ConditionalAdapter(nn.Module):
            def __init__(self, base_dim):
                super().__init__()
                self.adapter = nn.Sequential(
                    nn.Linear(base_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, base_dim)
                )
            def forward(self, x):
                return x + self.adapter(x)
        return ConditionalAdapter(base_backbone.output_dim).to(self.device)

    def _save_distilled_artifacts(self, new_mode_specialists, new_bank_entries):
        artifacts = {
            "new_mode_specialists": new_mode_specialists,
            "new_pino_bank_entries": new_bank_entries,
            "timestamp": str(torch.datetime.now())
        }
        save_to_vaults(artifacts, vault_name="training")
        logger.info("💾 Distilled MoDE specialists and custom surrogates saved to training vault")
