"""
agents/retrieval_agent.py
──────────────────────────
Agent de récupération (Retrieval Agent).

Responsabilité :
  1. Effectue une recherche vectorielle dans ChromaDB (similarité sémantique)
  2. Extrait des mots-clés de la requête pour interroger Neo4j
  3. Fusionne et déduplique les résultats des deux sources
  4. Met à jour l'état avec les documents et le contexte graphe trouvés
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from database import neo4j_client, vector_store

logger = logging.getLogger(__name__)

# Prompt pour extraire les concepts-clés d'une requête
KEYWORD_EXTRACTION_PROMPT = """Tu es un expert en extraction de mots-clés.
Extrais les 3 à 5 concepts principaux de la requête suivante.
Réponds UNIQUEMENT avec les mots-clés séparés par des virgules, sans phrase.

Exemple :
Requête : "Comment fonctionne l'apprentissage automatique pour la vision par ordinateur ?"
Réponse : apprentissage automatique, vision par ordinateur, deep learning, réseaux de neurones

Maintenant extrais les concepts de :"""


class RetrievalAgent:
    """
    Agent qui récupère les documents pertinents depuis les deux sources :
    - ChromaDB : recherche sémantique par vecteurs
    - Neo4j    : traversée du graphe de connaissances
    """

    def __init__(self) -> None:
        self._llm = ChatGroq(
            model=settings.llm_model,
            temperature=0,          # 0 = déterministe pour l'extraction
            api_key=settings.groq_api_key,
        )

    def _extract_keywords(self, query: str) -> list[str]:
        """Utilise le LLM pour extraire les mots-clés de la requête."""
        response = self._llm.invoke(
            [
                SystemMessage(content=KEYWORD_EXTRACTION_PROMPT),
                HumanMessage(content=query),
            ]
        )
        raw = response.content.strip()
        keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]
        logger.debug("Mots-clés extraits : %s", keywords)
        return keywords

    def run(self, state: dict) -> dict:
        """
        Nœud LangGraph : exécute la récupération et met à jour l'état.

        Paramètre : state (dict) — état courant du graphe
        Retour    : état mis à jour avec retrieved_documents et graph_context
        """
        query = state["query"]
        logger.info("[RetrievalAgent] Requête reçue : %s", query)

        # ── Étape 1 : Recherche vectorielle (ChromaDB) ─────────────
        vector_results = []
        try:
            vector_results = vector_store.similarity_search(query, k=5)
            logger.info(
                "[RetrievalAgent] %d documents trouvés dans ChromaDB",
                len(vector_results),
            )
        except Exception as exc:
            logger.warning("[RetrievalAgent] Erreur ChromaDB : %s", exc)

        # ── Étape 2 : Extraction des mots-clés pour Neo4j ──────────
        keywords = []
        try:
            keywords = self._extract_keywords(query)
        except Exception as exc:
            logger.warning("[RetrievalAgent] Erreur extraction mots-clés : %s", exc)

        # ── Étape 3 : Contexte graphe (Neo4j) ──────────────────────
        graph_context = ""
        try:
            graph_context = neo4j_client.get_graph_context(keywords)
            logger.info("[RetrievalAgent] Contexte graphe récupéré")
        except Exception as exc:
            logger.warning("[RetrievalAgent] Erreur Neo4j : %s", exc)
            graph_context = "Graphe de connaissances non disponible."

        # ── Étape 4 : Construction de la liste des sources ─────────
        sources = list(
            {
                doc["metadata"].get("source", "source inconnue")
                for doc in vector_results
            }
        )

        return {
            **state,
            "retrieved_documents": vector_results,
            "graph_context": graph_context,
            "sources": sources,
            "next_agent": "reasoning",
            "iteration": state.get("iteration", 0) + 1,
        }


# Instance unique
retrieval_agent = RetrievalAgent()
