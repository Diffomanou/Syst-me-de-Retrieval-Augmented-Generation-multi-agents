"""
graph/workflow.py
──────────────────
Construction et compilation du graphe LangGraph.

Le graphe définit :
  - Les nœuds (chaque agent = un nœud)
  - Les arêtes (les transitions entre agents)
  - Les arêtes conditionnelles (décisions de routage)

Flux complet :
  START → supervisor → retrieval → reasoning → answer → END
               │
               └──────────────────────────────────────→ END
                         (si requête invalide)
"""

import logging
from langgraph.graph import StateGraph, END

from agents import (
    initial_state,
    supervisor_agent,
    retrieval_agent,
    reasoning_agent,
    answer_agent,
)

logger = logging.getLogger(__name__)


def build_workflow():
    """
    Construit et compile le graphe multi-agents.

    Retour
    ------
    Un graphe LangGraph compilé, prêt à recevoir des requêtes.
    """

    # ── 1. Création du graphe avec le type d'état ──────────────────
    graph = StateGraph(dict)

    # ── 2. Ajout des nœuds (un nœud = une fonction agent.run) ──────
    graph.add_node("supervisor", supervisor_agent.run)
    graph.add_node("retrieval", retrieval_agent.run)
    graph.add_node("reasoning", reasoning_agent.run)
    graph.add_node("answer", answer_agent.run)

    # ── 3. Point d'entrée du graphe ────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── 4. Arêtes conditionnelles depuis le Supervisor ─────────────
    # Le Supervisor décide : aller vers 'retrieval' ou vers END
    graph.add_conditional_edges(
        source="supervisor",
        path=supervisor_agent.route,
        path_map={
            "retrieval": "retrieval",
            "end": END,
        },
    )

    # ── 5. Arêtes fixes après récupération et raisonnement ─────────
    # Après retrieval → toujours vers reasoning
    graph.add_edge("retrieval", "reasoning")

    # Après reasoning → toujours vers answer
    graph.add_edge("reasoning", "answer")

    # Après answer → fin du graphe
    graph.add_edge("answer", END)

    # ── 6. Compilation ─────────────────────────────────────────────
    compiled = graph.compile()
    logger.info("Graphe LangGraph compilé avec succès")
    return compiled


def run_query(query: str) -> dict:
    """
    Interface publique pour exécuter une requête dans le pipeline multi-agents.

    Paramètre
    ---------
    query : str — question posée par l'utilisateur

    Retour
    ------
    dict contenant au minimum :
      - answer  : réponse finale
      - sources : liste des sources utilisées
      - error   : message d'erreur si applicable (None sinon)
    """
    workflow = build_workflow()
    state = initial_state(query)

    logger.info("Exécution du pipeline pour : %s", query)
    final_state = workflow.invoke(state)

    return {
        "answer": final_state.get("answer", "Aucune réponse générée."),
        "sources": final_state.get("sources", []),
        "reasoning": final_state.get("reasoning", ""),
        "graph_context": final_state.get("graph_context", ""),
        "error": final_state.get("error"),
        "iterations": final_state.get("iteration", 0),
    }
