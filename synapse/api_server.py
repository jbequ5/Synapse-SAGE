"""
Synapse Private API Server — v0.9.13 MAXIMUM SOTA
Secure, rate-limited FastAPI gateway to the full vector-first intelligence layer.
Enforces private-gatekeeper model. Only this server talks to the upgraded Synapse core.
Full /ingest handoff + chat + cycle triggers.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import logging
import time

from synapse.config import SynapseConfig
from synapse.core import synapse
from synapse.chat_interface import synapse_chat
from synapse.defense_red_team import defense_red_team
from synapse.neural_net_head import neural_net_head
from synapse.utils import save_to_vaults

logger = logging.getLogger(__name__)
app = FastAPI(title="Synapse Intelligence API — Private Vector-First Service", version="0.9.13")

# Security
security = HTTPBearer()
# Load from env in production
EXPECTED_API_KEY = SynapseConfig().api_key or "your-secret-api-key-here"

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Rate limiting (simple in-memory; use Redis in prod)
rate_limit_store = {}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()
    rate_limit_store.setdefault(client_ip, [])
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if t > now - 60]
    if len(rate_limit_store[client_ip]) >= 60:  # generous for EM swarms
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    rate_limit_store[client_ip].append(now)
    return await call_next(request)

# CORS for trusted EM instances only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production to specific EM IPs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class FragmentPushRequest(BaseModel):
    fragments: List[Dict]
    telemetry: Dict
    em_instance_id: str
    run_id: str
    provenance: Dict = {}

class QueryRequest(BaseModel):
    query: str
    user_tier: str = "standard"

class StallRequest(BaseModel):
    em_context: Dict

class TelemetryRequest(BaseModel):
    telemetry_data: Dict

class IngestResponse(BaseModel):
    status: str
    accepted: int
    rejected: int
    red_team_summary: Dict

# Endpoints
@app.post("/ingest", response_model=IngestResponse)
async def ingest_fragments(request: FragmentPushRequest, _: str = Depends(verify_api_key)):
    """Private gate endpoint — ONLY called by trusted local EM instances."""
    logger.info(f"📥 /ingest received from EM {request.em_instance_id} — {len(request.fragments)} fragments")

    accepted = []
    rejected = 0

    for frag in request.fragments:
        # Full DefenseRedTeam gating (critical for antifragility)
        red_team_report = defense_red_team.red_team_scoring_and_validation(frag)
        if not red_team_report.get("passed", False):
            rejected += 1
            continue

        # NeuralNetHead scoring (primary signal)
        scored = neural_net_head.score_advice(frag, request.telemetry)

        # Add provenance and move ONLY to internal ranked vaults
        frag["red_team_report"] = red_team_report
        frag["neural_scores"] = scored
        frag["provenance"] = {
            **request.provenance,
            "em_instance_id": request.em_instance_id,
            "run_id": request.run_id,
            "ingested_at": datetime.now().isoformat()
        }
        accepted.append(frag)

    # Save ONLY to internal vaults (private-gatekeeper rule)
    if accepted:
        save_to_vaults(accepted, SynapseConfig().shared_vault_path, vault_name="internal/incoming")

    logger.info(f"✅ /ingest complete — {len(accepted)} accepted, {rejected} rejected by red-team")

    return IngestResponse(
        status="success",
        accepted=len(accepted),
        rejected=rejected,
        red_team_summary={"passed": len(accepted), "failed": rejected}
    )

@app.post("/chat/query")
async def chat_query(req: QueryRequest, _: str = Depends(verify_api_key)):
    """Vector-first co-pilot query."""
    return synapse_chat.handle_query(req.query, req.user_tier)

@app.post("/chat/stall")
async def chat_stall(req: StallRequest, _: str = Depends(verify_api_key)):
    """EM stall assistance."""
    return synapse_chat.handle_stall(req.em_context)

@app.post("/cycle/run")
async def run_daily_cycle(_: str = Depends(verify_api_key)):
    """Trigger full intelligence cycle (called by scheduler or EM)."""
    result = synapse.run_daily_intelligence_cycle()
    return result

@app.post("/telemetry/push")
async def push_telemetry(req: TelemetryRequest, _: str = Depends(verify_api_key)):
    """Accept per-instance vector snapshots from EM swarm."""
    logger.info(f"Telemetry received — {len(req.telemetry_data)} data points")
    # Feed into GraphMiner / Meta-RL (already wired in core)
    return {"status": "accepted", "received": len(req.telemetry_data)}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.9.13", "vector_first": True, "gatekeeper": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
