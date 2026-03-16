"""NEXUS API — FastAPI backend for the NEXUS web interface."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.house_b import HouseB
from nexus.core.house_c import HouseC
from nexus.core.house_d import HouseD
from nexus.core.house_omega import HouseOmega
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.model_router import ModelRouter

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="NEXUS", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directories exist on startup (for Railway ephemeral filesystem)
os.makedirs("data/knowledge_store", exist_ok=True)
os.makedirs("data/builds", exist_ok=True)

graph = KnowledgeGraph()
router = ModelRouter()
house_b = HouseB(knowledge_graph=graph, router=router)
house_c = HouseC(knowledge_graph=graph, router=router)
house_d = HouseD(knowledge_graph=graph, router=router, min_cycles=1)

omega = HouseOmega(
    knowledge_graph=graph,
    house_b=house_b,
    house_c=house_c,
    house_d=house_d,
    sleep_cycle_interval=50,
    max_refinements=3,
)

starter_beliefs = [
    BeliefCertificate(
        claim="Clean code is better than clever code",
        source="NEXUS founding axiom",
        confidence=0.9,
        domain="Software Engineering",
        executable_proof="print('clean')",
        decay_rate=0.05,
    ),
    BeliefCertificate(
        claim="Tests must run before code is trusted",
        source="NEXUS founding axiom",
        confidence=0.95,
        domain="Software Engineering",
        executable_proof="assert True",
        decay_rate=0.05,
    ),
    BeliefCertificate(
        claim="Every system needs a kill switch",
        source="NEXUS founding axiom",
        confidence=0.99,
        domain="System Architecture",
        executable_proof="assert True",
        decay_rate=0.05,
    ),
]
graph.inject_external_signal(starter_beliefs)


class RunRequest(BaseModel):
    user_input: str


@app.post("/api/run")
async def run_cycle(req: RunRequest) -> dict[str, Any]:
    """Execute a full NEXUS cycle."""
    result = omega.run(req.user_input)
    return result.to_dict()


@app.get("/api/health")
async def get_health() -> dict[str, Any]:
    """Return system health snapshot."""
    return omega.get_health().to_dict()


@app.get("/api/beliefs")
async def get_beliefs() -> list[dict[str, Any]]:
    """Return all beliefs in the knowledge graph."""
    return [b.to_dict() for b in graph.beliefs.values()]


@app.get("/api/cycles")
async def get_cycles() -> list[dict[str, Any]]:
    """Return cycle history."""
    return [c.to_dict() for c in omega.get_cycle_history(last_n=50)]


@app.get("/")
async def serve_index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/dashboard")
async def serve_dashboard() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "dashboard.html").read_text(encoding="utf-8"))
