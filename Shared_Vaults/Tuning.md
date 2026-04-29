# Synapse Tuning & Calibration Guide (Living Document)

## Objective Weighting Principles
- value_creation and robustness are primary long-term drivers — never let either drop below 0.75.
- learning_to_learn is the meta-objective: when it is the weakest, Meta-RL and KAS must prioritize improvements to the system’s own learning mechanisms.
- implementation_quality and prediction_accuracy are tactical — they support the above but must not dominate.

## Known Failure Modes & Mitigations
- Short-term EFS hacking → always weight robustness heavily in EconomicLayer and Defense.
- Objective imbalance → NeuralNetHead automatically boosts the weakest objective during calibration.
- Stale knowledge in vaults → KAS freshness gating + GraphMiner temporal decay on edges.
- Self-congratulatory drift → nightly reflection_log.md + external human review of key proposals.

## Calibration Heuristics for NeuralNetHead
- If any objective < 0.75 for > 3 consecutive cycles → increase its weight by 8% and re-normalize.
- Synergy bonus only applies when geometric mean of the five objectives > 0.82.
- Use reflection_log.md excerpts in every calibration step to incorporate historical lessons.

## KAS & Meta-RL Tuning Rules
- Prioritize fragments that improve the current weakest objective while maintaining robustness > 0.78.
- Depth-aware hunting instructions are defined in kashunting.md.

Last updated: 2026-04-29
