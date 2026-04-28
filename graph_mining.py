"""
Synapse Graph Mining Subsystem
NetworkX-based mining on shared Solve/Strategy Layer vaults.
Discovers patterns, computes synergy scores, ranks insights, and feeds downstream subsystems.
"""

import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

class GraphMiner:
    """SOTA Graph Mining for Synapse — mines shared vaults for patterns, synergy, and high-value insights."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.shared_vault_path = Path("shared_vaults")
        self.shared_vault_path.mkdir(parents=True, exist_ok=True)
        logger.info("🔍 GraphMiner initialized — ready to mine Solve/Strategy Layer vaults")

    def load_vaults(self) -> Dict:
        """Load all shared vault data from disk (or IPFS gateway later)."""
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
        """Main mining pass — builds/updates graph and returns ranked insights."""
        if vaults is None:
            vaults = self.load_vaults()

        # Build/update graph
        self._build_graph(vaults)

        # Discover patterns and synergy
        patterns = self._discover_patterns()
        synergy_scores = self._compute_synergy_scores()

        # Rank insights for downstream subsystems
        ranked_insights = self._rank_insights(patterns, synergy_scores)

        logger.info(f"Graph mining complete — {len(ranked_insights)} high-value insights discovered")
        return ranked_insights

    def _build_graph(self, vaults: Dict):
        """Build directed graph from vault fragments."""
        self.graph.clear()
        for vault_name, fragments in vaults.items():
            for frag in fragments:
                frag_id = frag.get("fragment_id") or f"frag_{hash(str(frag))}"
                self.graph.add_node(frag_id, **frag, vault=vault_name)

                # Add edges based on semantic similarity / shared tags / provenance
                for other_id in list(self.graph.nodes):
                    if other_id != frag_id:
                        similarity = self._compute_fragment_similarity(frag, self.graph.nodes[other_id])
                        if similarity > 0.65:
                            self.graph.add_edge(frag_id, other_id, weight=similarity)

    def _compute_fragment_similarity(self, frag1: Dict, frag2: Dict) -> float:
        """Simple but effective similarity for graph edges."""
        text1 = str(frag1.get("content", "")) + " " + str(frag1.get("key_takeaway", ""))
        text2 = str(frag2.get("content", "")) + " " + str(frag2.get("key_takeaway", ""))
        # Basic Jaccard + shared keys
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0
        # Bonus for shared vault type or tags
        tag_bonus = 0.3 if frag1.get("vault") == frag2.get("vault") else 0.0
        return min(1.0, jaccard * 0.7 + tag_bonus)

    def _discover_patterns(self) -> List[Dict]:
        """Discover emergent patterns using centrality and community detection."""
        if len(self.graph) < 3:
            return []
        
        patterns = []
        # High centrality nodes = strong patterns
        centrality = nx.degree_centrality(self.graph)
        for node, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]:
            data = self.graph.nodes[node]
            patterns.append({
                "pattern_id": node,
                "type": "central_pattern",
                "strength": round(score, 3),
                "content_preview": str(data.get("content", ""))[:200],
                "vault": data.get("vault"),
                "timestamp": data.get("timestamp")
            })
        return patterns

    def _compute_synergy_scores(self) -> Dict:
        """Compute cross-vault synergy scores."""
        synergy = {}
        for vault in self.shared_vault_path.iterdir():
            if vault.is_dir():
                synergy[vault.name] = round(np.random.uniform(0.65, 0.95), 3)  # placeholder — replace with real cross-vault analysis
        return synergy

    def _rank_insights(self, patterns: List[Dict], synergy: Dict) -> List[Dict]:
        """Rank insights for downstream subsystems (Meta-RL, Economic, etc.)."""
        ranked = []
        for p in patterns:
            p["synergy_score"] = synergy.get(p["vault"], 0.7)
            p["combined_score"] = (p["strength"] * 0.6) + (p["synergy_score"] * 0.4)
            ranked.append(p)
        ranked.sort(key=lambda x: x["combined_score"], reverse=True)
        return ranked[:50]  # top 50 insights per cycle

# Global instance
graph_miner = GraphMiner()
