"""
agents/supervisor_agent.py
───────────────────────────
Agent superviseur (Supervisor Agent).

Responsabilité :
  - C'est le point d'entrée du graphe LangGraph
  - Valide et nettoie la requête
  - Décide du premier agent à appeler (toujours 'retrieval' pour une nouvelle requête)
  - Gère la sécurité anti-boucle (max_iterations)
  - Peut court-circuiter vers 'end' si la requête est invalide
"""

import logging

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5   # sécurité : max 5 itérations avant arrêt forcé


class SupervisorAgent:
    """
    Agent superviseur : orchestre le flux entre les autres agents.
    """

    def run(self, state: dict) -> dict:
        """
        Nœud LangGraph : valide la requête et initialise le flux.

        Paramètre : state (dict) — état courant
        Retour    : état mis à jour avec next_agent
        """
        query = state.get("query", "").strip()
        iteration = state.get("iteration", 0)

        logger.info("[SupervisorAgent] Supervision de la requête (iter=%d)", iteration)

        # ── Sécurité anti-boucle ───────────────────────────────────
        if iteration >= MAX_ITERATIONS:
            logger.warning("[SupervisorAgent] Limite d'itérations atteinte")
            return {
                **state,
                "answer": "Le système a atteint la limite d'itérations. Veuillez reformuler votre question.",
                "next_agent": "end",
                "error": "max_iterations_reached",
            }

        # ── Validation de la requête ───────────────────────────────
        if not query:
            return {
                **state,
                "answer": "Veuillez fournir une question.",
                "next_agent": "end",
                "error": "empty_query",
            }

        if len(query) < 5:
            return {
                **state,
                "answer": "Votre question est trop courte. Veuillez la détailler.",
                "next_agent": "end",
                "error": "query_too_short",
            }

        # ── Flux normal : on démarre par la récupération ───────────
        logger.info("[SupervisorAgent] Requête valide → RetrievalAgent")
        return {
            **state,
            "query": query,
            "next_agent": "retrieval",
        }

    def route(self, state: dict) -> str:
        """
        Fonction de routage appelée par LangGraph pour décider du nœud suivant.
        Retourne le nom du prochain nœud.
        """
        return state.get("next_agent", "end")


# Instance unique
supervisor_agent = SupervisorAgent()
