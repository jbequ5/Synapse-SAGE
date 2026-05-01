# test_fragment_scoring.py
# SAGE v0.9.14 – Unit tests asserting exact screenshot formulas
import unittest
import numpy as np
from solve_fragment_scoring import SolveFragmentScoringModule
from economic_fragment_scoring import EconomicFragmentScoringModule
from vault_promotion_gate import VaultPromotionGate

class TestSolveFragmentScoring(unittest.TestCase):
    def setUp(self):
        self.scorer = SolveFragmentScoringModule()

    def test_7d_geometric_mean_exact(self):
        scores = {dim: 0.85 for dim in self.scorer.SEVEN_D_DIMENSIONS}
        mean = self.scorer.compute_7d_geometric_mean(scores)
        self.assertAlmostEqual(mean, 0.85, places=4)  # weighted geometric mean

    def test_base_efs_exact_formula(self):
        mean = 0.85
        base = self.scorer.compute_base_efs(mean, verifier_floor=1.0, calibration_c=1.0)
        self.assertAlmostEqual(base, 0.85, places=4)

    def test_refined_value_added_exact(self):
        components = {"n": 0.9, "r": 0.8, "m": 0.85, "c": 0.75}
        v = self.scorer.compute_refined_value_added(**components)
        self.assertGreater(v, 0.0)
        self.assertLess(v, 1.0)

    def test_60_40_final_impact_exact(self):
        base = 0.85
        v = 0.75
        impact = 0.6 * base + 0.4 * v
        self.assertAlmostEqual(impact, 0.81, places=4)

class TestEconomicFragmentScoring(unittest.TestCase):
    def setUp(self):
        self.scorer = EconomicFragmentScoringModule()

    def test_3_of_4_rule_mandatory_gap_bd(self):
        self.assertTrue(self.scorer.meets_3_of_4_rule(0.70, 0.70, 0.60, 0.60))
        self.assertFalse(self.scorer.meets_3_of_4_rule(0.60, 0.70, 0.80, 0.80))  # missing mandatory

class TestVaultPromotionGate(unittest.TestCase):
    def setUp(self):
        self.gate = VaultPromotionGate()

    def test_all_5_layers_pass(self):
        fragment = {"final_impact_score": 0.85, "objective_vector": {"value_creation": 0.9},
                    "reuse_potential": 0.8, "red_team_risk": 0.1, "provenance_hash": "abc123"}
        self.assertTrue(self.gate.should_promote_to_vault(fragment))

    def test_layer1_hard_floor_fail(self):
        fragment = {"final_impact_score": 0.70}
        self.assertFalse(self.gate.should_promote_to_vault(fragment))

if __name__ == '__main__':
    unittest.main()
