"""
api/main.py
────────────
Point d'entrée de l'application FastAPI.

Lance avec : uvicorn api.main:app --reload --port 8000
Swagger UI  : http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import query_router, ingest_router
from api.schemas import HealthResponse
from database import neo4j_client, vector_store
from config import settings

# ── Configuration du logging ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Cycle de vie de l'application ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code exécuté au démarrage et à l'arrêt de l'application.
    (Remplace les anciens @app.on_event deprecated)
    """
    # Démarrage
    logger.info("Démarrage du système Multi-Agent RAG...")
    neo4j_client.create_constraints()   # crée les contraintes Neo4j si absentes
    logger.info("Système prêt ✓")

    yield  # L'app est en cours d'exécution

    # Arrêt
    logger.info("Arrêt du système — fermeture des connexions...")
    neo4j_client.close()
    logger.info("Connexions fermées ✓")


# ── Application FastAPI ────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Agent RAG System",
    description=(
        "Système de Retrieval-Augmented Generation multi-agents.\n\n"
        "**Pipeline :** SupervisorAgent → RetrievalAgent → ReasoningAgent → AnswerAgent\n\n"
        "**Technologies :** LangGraph · LangChain · Neo4j · ChromaDB · FastAPI"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware CORS (pour les frontends web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Enregistrement des routes ──────────────────────────────────────────────
app.include_router(query_router)
app.include_router(ingest_router)


# ── Endpoints utilitaires ──────────────────────────────────────────────────

@app.get("/", summary="Accueil", tags=["Système"])
async def root():
    return {
        "message": "Multi-Agent RAG System",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérification de santé",
    tags=["Système"],
)
async def health_check() -> HealthResponse:
    """Vérifie l'état de tous les composants du système."""
    neo4j_ok = neo4j_client.verify_connection()
    doc_count = vector_store.count()

    return HealthResponse(
        status="healthy" if neo4j_ok else "degraded",
        neo4j_connected=neo4j_ok,
        vector_store_documents=doc_count,
    )
