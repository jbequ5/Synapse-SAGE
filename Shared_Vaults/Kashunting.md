# KAS Depth-Aware Hunting Instructions (Vector-First)

## Depth 0 – Broad Exploration
Focus on the current weakest objective (from GraphMiner _get_strongest_objectives()). Gather high-signal fragments that could plausibly improve it. Emphasize novelty and cross-vault synergy.

## Depth 1–2 – Targeted Acquisition
Hunt for fragments that specifically boost the weakest objective while maintaining:
- robustness ≥ 0.78
- value_creation ≥ 0.70
Prioritize real ToolHunter results (GitHub, arXiv, HF) that contain concrete implementation patterns or verifiable claims.

## Depth 3+ – Meta-Hunt
Seek knowledge that improves learning_to_learn itself:
- Better calibration techniques for NeuralNetHead
- More effective red-team strategies
- Ways to make the distillation process more efficient
- Meta-patterns from reflection_log.md and past_proposals

## Universal Constraints (apply at every depth)
- Red-team every candidate fragment before acquisition.
- Only accept fragments that pass DefenseRedTeam with overall_risk < 0.55.
- Prefer fragments with strong objective_vector balance over single-objective spikes.

These instructions are dynamically injected by KAS at each recursion depth.
