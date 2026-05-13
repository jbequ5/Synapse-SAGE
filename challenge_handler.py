# PRIVATE Synapse backend — challenge_handler.py
from fastapi import APIRouter, HTTPException
from pathlib import Path
import re
import logging
from typing import Dict, List

router = APIRouter()
logger = logging.getLogger(__name__)

CHALLENGE_MD_PATH = Path("challenges/challenge.md")   # ← Put your challenge.md here in the private repo

def parse_challenge_md() -> List[Dict]:
    if not CHALLENGE_MD_PATH.exists():
        raise FileNotFoundError(f"challenge.md not found at {CHALLENGE_MD_PATH}")
    text = CHALLENGE_MD_PATH.read_text(encoding="utf-8")
    challenges = []
    sections = re.split(r'(?m)^## Challenge \d+: ', text)[1:]
    for i, section in enumerate(sections, 1):
        title_match = re.match(r'(.+?)(?:\n|$)', section)
        title = title_match.group(1).strip() if title_match else f"Challenge {i}"
        id_match = re.search(r'\*\*ID:\*\* (.+?)(?:\n|$)', section)
        challenge_id = id_match.group(1).strip() if id_match else f"challenge-{i:03d}"
        desc_match = re.search(r'(Description:.*?(?=\*\*Verification Spec:\*\*|$))', section, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else section[:800]
        spec_match = re.search(r'\*\*Verification Spec:\*\*\s*(.+?)(?=\n## Challenge|\Z)', section, re.DOTALL)
        verification_spec = spec_match.group(1).strip() if spec_match else ""
        challenges.append({
            "id": challenge_id,
            "title": title,
            "description": description,
            "verification_spec": verification_spec,
            "full_text": section
        })
    return challenges

@router.get("/get_challenges")
async def get_challenges():
    """Return all active challenges with dense verification specs."""
    return {"challenges": parse_challenge_md()}

@router.post("/get_challenge")
async def get_challenge(payload: Dict):
    """Return single challenge by ID (with full dense verification spec)."""
    challenge_id = payload.get("challenge_id")
    challenges = parse_challenge_md()
    for chal in challenges:
        if chal["id"] == challenge_id:
            return chal
    raise HTTPException(status_code=404, detail=f"Challenge {challenge_id} not found")
