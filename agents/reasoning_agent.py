"""
agents/reasoning_agent.py
──────────────────────────
Agent de raisonnement (Reasoning Agent).

Responsabilité :
  - Reçoit les documents récupérés et le contexte du graphe
  - Analyse, synthétise et évalue leur pertinence par rapport à la requête
  - Produit un "raisonnement intermédiaire" qui guidera l'AnswerAgent
  - Décide si davantage de contexte est nécessaire (ou si on peut répondre)
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings

logger = logging.getLogger(__name__)

REASONING_SYSTEM_PROMPT = """Tu es un expert en analyse et synthèse de documents.
Tu reçois :
1. Une question posée par un utilisateur
2. Des extraits de documents récupérés (avec leur score de pertinence)
3. Un contexte issu d'un graphe de connaissances

Ton rôle :
- Analyser la pertinence de chaque document par rapport à la question
- Identifier les informations clés qui permettent de répondre
- Signaler les contradictions ou lacunes d'information
- Produire une synthèse structurée et raisonnée qui servira de base à la réponse finale

Si les documents ne contiennent pas assez d'information, indique-le clairement.
Réponds en français, de manière structurée (points clés, synthèse)."""


class ReasoningAgent:
    """
    Agent qui analyse le contexte récupéré et produit un raisonnement structuré.
    """

    def __init__(self) -> None:
        self._llm = ChatGroq(
            model=settings.llm_model,
            temperature=0.2,        # légère créativité pour la synthèse
            api_key=settings.groq_api_key,
        )

    def _format_documents(self, documents: list[dict]) -> str:
        """Formate les documents récupérés pour le prompt."""
        if not documents:
            return "Aucun document récupéré."

        parts = []
        for i, doc in enumerate(documents, 1):
            score = 1 - doc.get("distance", 1)   # conversion distance → score
            source = doc.get("metadata", {}).get("source", "?")
            parts.append(
                f"[Document {i}] (pertinence: {score:.2f}, source: {source})\n"
                f"{doc['content']}\n"
            )
        return "\n---\n".join(parts)

    def run(self, state: dict) -> dict:
        """
        Nœud LangGraph : analyse le contexte et produit un raisonnement.

        Paramètre : state (dict) — état courant
        Retour    : état mis à jour avec le champ 'reasoning'
        """
        query = state["query"]
        documents = state.get("retrieved_documents", [])
        graph_context = state.get("graph_context", "")

        logger.info(
            "[ReasoningAgent] Analyse de %d documents pour : %s",
            len(documents),
            query,
        )

        # Construction du prompt utilisateur
        user_prompt = f"""QUESTION : {query}

DOCUMENTS RÉCUPÉRÉS :
{self._format_documents(documents)}

CONTEXTE DU GRAPHE DE CONNAISSANCES :
{graph_context}

Produis ton analyse et ta synthèse :"""

        response = self._llm.invoke(
            [
                SystemMessage(content=REASONING_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )

        reasoning = response.content.strip()
        logger.info("[ReasoningAgent] Raisonnement produit (%d chars)", len(reasoning))

        return {
            **state,
            "reasoning": reasoning,
            "next_agent": "answer",
        }


# Instance unique
reasoning_agent = ReasoningAgent()
