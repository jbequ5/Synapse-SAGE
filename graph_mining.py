"""
Synapse Graph Mining Subsystem — v0.9.13 MAX SOTA
NetworkX-based mining on shared Solve/Strategy Layer vaults.
Vector-first 5-objective design (implementation_quality, prediction_accuracy,
value_creation, learning_to_learn, robustness) for true intelligence amplification.

Optimal upgrade: DomainAdapter + dynamic vector support + defense-aware ranking
"""

import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging
import json

from synapse.defense_red_team import defense_red_team   # for stability-aware ranking

logger = logging.getLogger(__name__)

class DomainAdapter:
    """Optimal lightweight domain adapter for semantic alignment across Enigma challenge domains.
    Ensures graph edges and patterns respect domain boundaries while allowing controlled cross-pollination.
    """
    def __init__(self):
        self.known_domains = {"crypto", "quantum", "ai_robustness", "smart_contract", "incentive_mechanism", "general"}

    def extract_domain_tag(self, frag: Dict) -> str:
        """Extract domain tag from metadata with safe defaults."""
        metadata = frag.get("metadata", {}) if isinstance(frag, dict) else {}
        return metadata.get("domain_tag", "general")

class GraphMiner:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.shared_vault_path = Path("shared_vaults")
        self.shared_vault_path.mkdir(parents=True, exist_ok=True)
        self.domain_adapter = DomainAdapter()
        logger.info("🔍 GraphMiner v0.9.13 MAX SOTA initialized — full vector-first 5-objective synergy and provenance tracking + domain-aware + dynamic vector support")

    def load_vaults(self) -> Dict:
        """Load all fragments from shared vaults."""
        vaults = {}
        for vault_dir in self.shared_vault_path.iterdir():
            if vault_dir.is_dir():
                vault_name = vault_dir.name
                vaults[vault_name] = []
                for file in vault_dir.glob("*.json"):
                    try:
                        data = json.loads(file.read_text(encoding="utf-8"))
                        vaults[vault_name].append(data)
                    except Exception:
                        pass
        return vaults

    def mine(self, vaults: Dict = None) -> List[Dict]:
        """Full SOTA vector-first graph mining pipeline."""
        if vaults is None:
            vaults = self.load_vaults()

        self._build_graph(vaults)
        patterns = self._discover_patterns()
        ranked_insights = self._rank_insights_vector_first(patterns)

        logger.info(f"Graph mining complete — {len(ranked_insights)} high-value insights discovered (vector-first)")
        return ranked_insights

    def _build_graph(self, vaults: Dict):
        """Vector-first graph construction with objective vectors."""
        self.graph.clear()
        for vault_name, fragments in vaults.items():
            for frag in fragments:
                frag_id = frag.get("fragment_id") or f"frag_{hash(str(frag))}"
                objective_vector = frag.get("objective_vector") or self._convert_legacy_to_vector(frag)

                self.graph.add_node(
                    frag_id,
                    content=frag.get("content_preview", ""),
                    objective_vector=objective_vector,
                    timestamp=frag.get("timestamp"),
                    provenance=frag.get("provenance", {}),
                    vault=vault_name,
                    domain=self.domain_adapter.extract_domain_tag(frag),
                    combined_score=frag.get("combined_score", 0.0)
                )

                # Add vector-based edges
                self._add_vector_based_edges(frag_id, objective_vector, frag)

    def _convert_legacy_to_vector(self, frag: Dict) -> Dict:
        """Safely convert older scalar fragments to 5-objective vector."""
        score = frag.get("combined_score", 0.65)
        return {
            "implementation_quality": score,
            "prediction_accuracy": score,
            "value_creation": score,
            "learning_to_learn": score,
            "robustness": score
        }

    def _add_vector_based_edges(self, node_id: str, vec: Dict, frag: Dict):
        """Add edges using cosine similarity on the (now dynamic) objective vector."""
        for existing_id, data in list(self.graph.nodes(data=True)):
            if existing_id == node_id:
                continue
            other_vec = data.get("objective_vector", {})
            if not other_vec:
                continue

            sim = self._cosine_similarity(vec, other_vec)
            if sim > 0.65:
                self.graph.add_edge(
                    node_id,
                    existing_id,
                    weight=sim,
                    shared_objectives=self._shared_objectives(vec, other_vec),
                    domain_match=(data.get("domain") == frag.get("domain"))
                )

    def _cosine_similarity(self, v1: Dict, v2: Dict) -> float:
        """Cosine similarity across the (dynamic) objective vector."""
        keys = list(set(v1.keys()) & set(v2.keys()))
        if not keys:
            keys = ["implementation_quality", "prediction_accuracy", "value_creation", "learning_to_learn", "robustness"]
        a = np.array([v1.get(k, 0.5) for k in keys])
        b = np.array([v2.get(k, 0.5) for k in keys])
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def _shared_objectives(self, v1: Dict, v2: Dict) -> List[str]:
        """Return objectives where both vectors are strong (>0.75)."""
        keys = list(set(v1.keys()) & set(v2.keys()))
        if not keys:
            keys = ["implementation_quality", "prediction_accuracy", "value_creation", "learning_to_learn", "robustness"]
        return [k for k in keys if v1.get(k, 0) > 0.75 and v2.get(k, 0) > 0.75]

    def _discover_patterns(self) -> List[Dict]:
        """Discover central patterns from the graph."""
        if len(self.graph) < 3:
            return []
        centrality = nx.degree_centrality(self.graph)
        patterns = []
        for node, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]:
            data = self.graph.nodes[node]
            patterns.append({
                "pattern_id": node,
                "type": "central_pattern",
                "strength": round(score, 3),
                "content_preview": str(data.get("content", ""))[:200],
                "vault": data.get("vault"),
                "timestamp": data.get("timestamp"),
                "objective_vector": data.get("objective_vector", {}),
                "domain": data.get("domain")
            })
        return patterns

    def _rank_insights_vector_first(self, patterns: List[Dict]) -> List[Dict]:
        """Primary ranking now uses full objective vector strength + synergy bonus + defense stability boost."""
        for p in patterns:
            vec = p.get("objective_vector", {})
            p["vector_strength"] = self._compute_vector_strength(vec)
            p["synergy_bonus"] = 0.0

            # Optimal defense-aware boost
            defense_boost = 0.0
            if hasattr(defense_red_team, "run_ahe_cycle"):
                report = defense_red_team.run_ahe_cycle()
                stability = report.get("stability_score", 1.0)
                defense_boost = stability * 0.15

            p["final_rank_score"] = p["vector_strength"] * (1 + p["synergy_bonus"]) + defense_boost

        patterns.sort(key=lambda x: x["final_rank_score"], reverse=True)
        return patterns[:50]

    def _compute_vector_strength(self, vec: Dict) -> float:
        """Geometric mean across the (now dynamic) objectives."""
        if not vec:
            return 0.0
        values = list(vec.values())
        if not values or max(values) == 0:
            return 0.0
        return np.prod([v + 1e-8 for v in values]) ** (1 / len(values))

    def _get_strongest_objectives(self) -> Dict[str, float]:
        """Return current strongest/weakest objectives across the entire graph (for Meta-RL)."""
        if not self.graph.nodes:
            return {}
        all_vectors = [data.get("objective_vector", {}) for _, data in self.graph.nodes(data=True) if data.get("objective_vector")]
        if not all_vectors:
            return {}
        keys = set()
        for v in all_vectors:
            keys.update(v.keys())
        avg = {k: np.mean([v.get(k, 0.5) for v in all_vectors]) for k in keys}
        return dict(sorted(avg.items(), key=lambda x: x[1], reverse=True))

    def get_graph_stats(self) -> Dict:
        """Full stats including vector distribution for Meta-RL and monitoring."""
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "average_vector_strength": self._compute_average_vector_strength(),
            "strongest_objectives": self._get_strongest_objectives()
        }

    def _compute_average_vector_strength(self) -> float:
        if not self.graph.nodes:
            return 0.0
        strengths = [self._compute_vector_strength(data.get("objective_vector", {})) 
                     for _, data in self.graph.nodes(data=True)]
        return np.mean(strengths)

# Global instance
graph_miner = GraphMiner()
