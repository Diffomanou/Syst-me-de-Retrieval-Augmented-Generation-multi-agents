"""
agents/answer_agent.py
───────────────────────
Agent de réponse (Answer Agent).

Responsabilité :
  - Reçoit la question, les documents et le raisonnement intermédiaire
  - Génère une réponse finale claire, précise et sourcée
  - Cite les sources utilisées
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = """Tu es un assistant expert en réponse à des questions.
Tu reçois :
- Une question posée par l'utilisateur
- Une analyse raisonnée produite par un agent spécialisé
- Les sources disponibles

Ton rôle :
- Générer une réponse finale complète, claire et bien structurée
- Baser ta réponse UNIQUEMENT sur les informations fournies dans l'analyse
- Citer les sources pertinentes dans ta réponse (ex. : [Source : nom_fichier])
- Si l'information est insuffisante, l'indiquer honnêtement

Règles :
- Réponds en français (sauf si la question est posée dans une autre langue)
- Utilise des listes à puces ou des paragraphes selon la nature de la réponse
- Sois concis mais complet
- Ne fabrique jamais d'information qui n'est pas dans le contexte fourni"""


class AnswerAgent:
    """
    Agent qui génère la réponse finale à partir du raisonnement produit.
    """

    def __init__(self) -> None:
        self._llm = ChatGroq(
            model=settings.llm_model,
            temperature=0.3,
            api_key=settings.groq_api_key,
        )

    def run(self, state: dict) -> dict:
        """
        Nœud LangGraph : génère la réponse finale.

        Paramètre : state (dict) — état courant
        Retour    : état mis à jour avec le champ 'answer'
        """
        query = state["query"]
        reasoning = state.get("reasoning", "Aucune analyse disponible.")
        sources = state.get("sources", [])

        logger.info("[AnswerAgent] Génération de la réponse pour : %s", query)

        sources_text = (
            "\n".join(f"- {s}" for s in sources) if sources else "Aucune source identifiée."
        )

        user_prompt = f"""QUESTION : {query}

ANALYSE DE L'AGENT DE RAISONNEMENT :
{reasoning}

SOURCES DISPONIBLES :
{sources_text}

Génère maintenant la réponse finale :"""

        response = self._llm.invoke(
            [
                SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )

        answer = response.content.strip()
        logger.info("[AnswerAgent] Réponse générée (%d chars)", len(answer))

        return {
            **state,
            "answer": answer,
            "next_agent": "end",
        }


# Instance unique
answer_agent = AnswerAgent()
