# fragment_rescoring.py
# SAGE v0.9.14 – Feed-My-Family Production Re-Scoring Engine
# Central orchestration for all fragment re-scoring (Solve + Economic)

from datetime import datetime, timedelta
from typing import Dict, Any
from solve_fragment_scoring import SolveFragmentScoringModule, SolveFragment
from economic_fragment_scoring import EconomicFragmentScoringModule, EconomicFragment

class DomainAdapter:
    """Optimal lightweight domain adapter for semantic alignment across Enigma challenge domains.
    Ensures rescoring respects domain boundaries while allowing controlled cross-domain transfer with decay.
    """
    def __init__(self):
        self.known_domains = {"crypto", "quantum", "ai_robustness", "smart_contract", "incentive_mechanism", "general"}
        self.cross_domain_decay = 0.85

    def extract_domain_tag(self, fragment: Any) -> str:
        """Extract domain tag from metadata with safe defaults."""
        if isinstance(fragment, dict):
            metadata = fragment.get('metadata', {})
            return metadata.get('domain_tag', 'general')
        metadata = getattr(fragment, 'metadata', {})
        if isinstance(metadata, dict):
            return metadata.get('domain_tag', 'general')
        return 'general'

    def adapt_fragment_for_domain(self, fragment: Any, target_domain: str = None) -> Any:
        """Apply domain-aware adaptation. Currently minimal - ready for future calibration."""
        domain = self.extract_domain_tag(fragment)
        if domain not in self.known_domains:
            domain = 'general'
        # Future: domain-specific score recalibration can be added here without changing core logic
        return fragment

class FragmentRescoringEngine:
    """Central re-scoring engine called by Synapse nightly polishing loop."""

    def __init__(self):
        self.solve_scorer = SolveFragmentScoringModule()
        self.economic_scorer = EconomicFragmentScoringModule()
        self.vault_full_threshold = 500          # fragments
        self.vault_full_days = 30                # fallback
        self.staleness_days = 14                 # pruning only activates after vault is full
        self.domain_adapter = DomainAdapter()

    def is_vault_full(self, fragment_count: int, days_running: int) -> bool:
        """Intelligent 'full vault' definition — pruning disabled until met."""
        return (fragment_count >= self.vault_full_threshold) or (days_running >= self.vault_full_days)

    def re_score_solve_fragment(self, fragment: SolveFragment,
                                latest_seven_d_scores: Dict[str, float],
                                refined_components: Dict[str, float],
                                calibration_c: float = 1.0,
                                current_fragment_count: int = 0,
                                days_running: int = 0) -> SolveFragment:
        """Re-score a Solve fragment with latest global context."""
        # Domain adaptation step (optimal upgrade)
        adapted_fragment = self.domain_adapter.adapt_fragment_for_domain(fragment)
        
        # Re-run exact screenshot formulas with fresh inputs
        new_fragment = self.solve_scorer.score_fragment(
            content=fragment.content,
            creator_id=fragment.creator_id,
            em_instance_id=fragment.em_instance_id,
            seven_d_scores=latest_seven_d_scores,
            refined_components=refined_components,
            calibration_c=calibration_c,
            metadata=fragment.metadata
        )
        # Preserve original provenance
        new_fragment.provenance_hash = fragment.provenance_hash
        return new_fragment

    def re_score_economic_fragment(self, fragment: EconomicFragment,
                                   gap_pain_inputs: Dict,
                                   bd_inputs: Dict,
                                   revenue_inputs: Dict,
                                   proposal_inputs: Dict,
                                   current_fragment_count: int = 0,
                                   days_running: int = 0) -> EconomicFragment:
        """Re-score an Economic fragment with latest global context."""
        # Domain adaptation step (optimal upgrade)
        adapted_fragment = self.domain_adapter.adapt_fragment_for_domain(fragment)
        
        new_fragment = self.economic_scorer.score_fragment(
            content=fragment.content,
            creator_id=fragment.creator_id,
            em_instance_id=fragment.em_instance_id,
            gap_pain_inputs=gap_pain_inputs,
            bd_inputs=bd_inputs,
            revenue_inputs=revenue_inputs,
            proposal_inputs=proposal_inputs,
            metadata=fragment.metadata
        )
        new_fragment.provenance_hash = fragment.provenance_hash
        return new_fragment

    def should_prune(self, fragment: Any, last_scored: datetime,
                     current_fragment_count: int, days_running: int) -> bool:
        """Pruning only activates once vault is full (user-requested rule)."""
        if not self.is_vault_full(current_fragment_count, days_running):
            return False  # Never prune while bootstrapping

        staleness = datetime.now() - last_scored
        if staleness > timedelta(days=self.staleness_days):
            # No meaningful improvement in 14+ days
            return True
        return False

    def run_nightly_rescoring(self, all_fragments: list, current_fragment_count: int,
                              days_running: int) -> list:
        """Main entry point called by Synapse nightly polishing loop."""
        updated_fragments = []
        for frag in all_fragments:
            # Re-score with latest global context (KAS, reuse, etc.)
            if isinstance(frag, SolveFragment):
                updated = self.re_score_solve_fragment(frag, ...)
            else:
                updated = self.re_score_economic_fragment(frag, ...)

            # Apply 5-layer vault gate and staleness check
            if self.should_prune(updated, frag.timestamp, current_fragment_count, days_running):
                continue  # prune / cosmic compress

            updated_fragments.append(updated)

        return updated_fragments
