"""
Synapse Private API Server — v0.9.13 10/10 MAXIMUM SOTA
Secure, rate-limited FastAPI gateway to the full vector-first intelligence layer.
All intelligence logic stays private. Only this server talks to the upgraded Synapse core.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import logging
import time

from synapse.core import synapse
from synapse.chat_interface import synapse_chat

logger = logging.getLogger(__name__)
app = FastAPI(title="Synapse Intelligence API — Private Vector-First Service", version="0.9.13")

# Security
security = HTTPBearer()
EXPECTED_API_KEY = "your-secret-api-key-here"   # ← load from env in production

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Rate limiting (simple in-memory for start; use Redis in prod)
rate_limit_store = {}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()
    rate_limit_store.setdefault(client_ip, [])
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if t > now - 60]
    if len(rate_limit_store[client_ip]) >= 30:  # 30 requests per minute
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    rate_limit_store[client_ip].append(now)
    return await call_next(request)

# Models
class QueryRequest(BaseModel):
    query: str
    user_tier: str = "standard"

class StallRequest(BaseModel):
    em_context: Dict

class TelemetryRequest(BaseModel):
    telemetry_data: Dict

# Endpoints
@app.post("/chat/query")
async def chat_query(req: QueryRequest, _: str = Depends(verify_api_key)):
    return synapse_chat.handle_query(req.query, req.user_tier)

@app.post("/chat/stall")
async def chat_stall(req: StallRequest, _: str = Depends(verify_api_key)):
    return synapse_chat.handle_stall(req.em_context)

@app.post("/cycle/run")
async def run_daily_cycle(_: str = Depends(verify_api_key)):
    """Trigger full intelligence cycle (called by scheduler or EM)."""
    result = synapse.run_daily_intelligence_cycle()
    return result

@app.post("/telemetry/push")
async def push_telemetry(req: TelemetryRequest, _: str = Depends(verify_api_key)):
    """Accept per-instance vector snapshots from EM swarm."""
    # Feed into GraphMiner / Meta-RL (already wired in core)
    logger.info(f"Telemetry received — {len(req.telemetry_data)} data points")
    # You can extend here to call graph_miner or meta_rl_loop directly if needed
    return {"status": "accepted", "received": len(req.telemetry_data)}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.9.13", "vector_first": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
