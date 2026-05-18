"""
agents/state.py
────────────────
Définit l'état partagé entre tous les agents du graphe LangGraph.

L'état est un dictionnaire typé qui circule à travers les nœuds du graphe.
Chaque agent peut lire et modifier cet état.

Flux de l'état :
  START
    │
    ▼
  SupervisorAgent  ──── décide ──→  RetrievalAgent
                                         │
                                         ▼
                                    ReasoningAgent
                                         │
                                         ▼
                                    AnswerAgent
                                         │
                                         ▼
                                       END
"""

from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RAGState(dict):
    """
    État global partagé entre tous les agents.

    Champs
    ------
    query               : question initiale de l'utilisateur
    messages            : historique de conversation (géré par LangGraph)
    retrieved_documents : chunks récupérés depuis ChromaDB
    graph_context       : contexte extrait du graphe Neo4j
    reasoning           : analyse produite par le ReasoningAgent
    answer              : réponse finale générée par l'AnswerAgent
    sources             : liste des sources utilisées
    next_agent          : prochain agent à appeler (décidé par le Supervisor)
    iteration           : compteur d'itérations (sécurité anti-boucle infinie)
    error               : message d'erreur si un agent échoue
    """

    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_documents: list[dict]
    graph_context: str
    reasoning: str
    answer: str
    sources: list[str]
    next_agent: str
    iteration: int
    error: str | None


def initial_state(query: str) -> dict:
    """
    Crée un état initial à partir d'une requête utilisateur.
    Tous les champs sont initialisés à des valeurs vides/nulles.
    """
    return {
        "query": query,
        "messages": [],
        "retrieved_documents": [],
        "graph_context": "",
        "reasoning": "",
        "answer": "",
        "sources": [],
        "next_agent": "retrieval",
        "iteration": 0,
        "error": None,
    }
