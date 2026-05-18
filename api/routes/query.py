"""
api/routes/query.py
────────────────────
Endpoint POST /query — Point d'entrée principal pour les questions au système RAG.
"""

import logging
from fastapi import APIRouter, HTTPException

from api.schemas import QueryRequest, QueryResponse
from graph import run_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Interroger le système RAG",
    description=(
        "Soumet une question au pipeline multi-agents. "
        "Le système récupère les documents pertinents, les analyse "
        "et génère une réponse sourcée."
    ),
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Pipeline :
      1. SupervisorAgent  — valide la requête
      2. RetrievalAgent   — récupère les documents (ChromaDB + Neo4j)
      3. ReasoningAgent   — analyse et synthétise le contexte
      4. AnswerAgent      — génère la réponse finale
    """
    logger.info("Nouvelle requête : %s", request.question)

    try:
        result = run_query(request.question)
    except Exception as exc:
        logger.error("Erreur du pipeline : %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        iterations=result["iterations"],
        reasoning=result["reasoning"] if request.include_reasoning else None,
        graph_context=result["graph_context"] if request.include_graph_context else None,
        error=result.get("error"),
    )
