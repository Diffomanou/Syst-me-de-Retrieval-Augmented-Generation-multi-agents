"""
api/schemas.py
───────────────
Modèles Pydantic pour la validation des données entrantes et sortantes de l'API.
"""

from pydantic import BaseModel, Field


# ── Requêtes (entrée) ──────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Corps de la requête POST /query"""
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Question posée au système RAG",
        examples=["Qu'est-ce que l'apprentissage automatique ?"],
    )
    include_reasoning: bool = Field(
        default=False,
        description="Inclure le raisonnement intermédiaire dans la réponse",
    )
    include_graph_context: bool = Field(
        default=False,
        description="Inclure le contexte du graphe Neo4j dans la réponse",
    )


class IngestRequest(BaseModel):
    """Corps de la requête POST /ingest"""
    title: str = Field(..., description="Titre du document")
    content: str = Field(
        ...,
        min_length=10,
        description="Contenu textuel du document à ingérer",
    )
    source: str = Field(
        default="manual",
        description="Source ou origine du document",
    )


# ── Réponses (sortie) ──────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    """Réponse du endpoint POST /query"""
    answer: str
    sources: list[str]
    iterations: int
    reasoning: str | None = None
    graph_context: str | None = None
    error: str | None = None


class IngestResponse(BaseModel):
    """Réponse du endpoint POST /ingest"""
    success: bool
    chunk_ids: list[str]
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """Réponse du endpoint GET /health"""
    status: str
    neo4j_connected: bool
    vector_store_documents: int
    version: str = "1.0.0"
