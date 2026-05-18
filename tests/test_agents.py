"""
tests/test_agents.py
─────────────────────
Tests unitaires pour chaque agent du pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.state import initial_state
from agents.supervisor_agent import SupervisorAgent
from agents.retrieval_agent import RetrievalAgent
from agents.reasoning_agent import ReasoningAgent
from agents.answer_agent import AnswerAgent


# ══════════════════════════════════════════════════════════════
# SupervisorAgent
# ══════════════════════════════════════════════════════════════

class TestSupervisorAgent:

    def setup_method(self):
        self.agent = SupervisorAgent()

    def test_valid_query_routes_to_retrieval(self):
        state = initial_state("Qu'est-ce que le machine learning ?")
        result = self.agent.run(state)
        assert result["next_agent"] == "retrieval"

    def test_empty_query_routes_to_end(self):
        state = initial_state("")
        result = self.agent.run(state)
        assert result["next_agent"] == "end"
        assert result["error"] == "empty_query"

    def test_short_query_routes_to_end(self):
        state = initial_state("AI")
        result = self.agent.run(state)
        assert result["next_agent"] == "end"
        assert result["error"] == "query_too_short"

    def test_max_iterations_stops_loop(self):
        state = {**initial_state("Test"), "iteration": 10}
        result = self.agent.run(state)
        assert result["next_agent"] == "end"
        assert result["error"] == "max_iterations_reached"

    def test_route_function(self):
        assert self.agent.route({"next_agent": "retrieval"}) == "retrieval"
        assert self.agent.route({"next_agent": "end"}) == "end"


# ══════════════════════════════════════════════════════════════
# RetrievalAgent
# ══════════════════════════════════════════════════════════════

class TestRetrievalAgent:

    @patch("agents.retrieval_agent.vector_store")
    @patch("agents.retrieval_agent.neo4j_client")
    def test_retrieval_updates_state(self, mock_neo4j, mock_vector):
        # Configuration des mocks
        mock_vector.similarity_search.return_value = [
            {
                "id": "doc1",
                "content": "Contenu test",
                "metadata": {"source": "test.pdf"},
                "distance": 0.2,
            }
        ]
        mock_neo4j.get_graph_context.return_value = "Contexte graphe test"

        with patch("agents.retrieval_agent.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content="machine learning, intelligence artificielle"
            )
            mock_llm_class.return_value = mock_llm

            agent = RetrievalAgent()
            state = initial_state("Qu'est-ce que le machine learning ?")
            result = agent.run(state)

        assert len(result["retrieved_documents"]) == 1
        assert result["graph_context"] == "Contexte graphe test"
        assert result["next_agent"] == "reasoning"
        assert result["iteration"] == 1

    @patch("agents.retrieval_agent.vector_store")
    @patch("agents.retrieval_agent.neo4j_client")
    def test_handles_empty_results(self, mock_neo4j, mock_vector):
        mock_vector.similarity_search.return_value = []
        mock_neo4j.get_graph_context.return_value = "Aucun contexte"

        with patch("agents.retrieval_agent.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="concept")
            mock_llm_class.return_value = mock_llm

            agent = RetrievalAgent()
            state = initial_state("Question test ?????")
            result = agent.run(state)

        assert result["retrieved_documents"] == []
        assert result["sources"] == []


# ══════════════════════════════════════════════════════════════
# ReasoningAgent
# ══════════════════════════════════════════════════════════════

class TestReasoningAgent:

    @patch("agents.reasoning_agent.ChatOpenAI")
    def test_reasoning_produces_analysis(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Analyse : le document traite du machine learning de manière claire."
        )
        mock_llm_class.return_value = mock_llm

        agent = ReasoningAgent()
        state = {
            **initial_state("Qu'est-ce que le ML ?"),
            "retrieved_documents": [
                {"content": "ML est...", "metadata": {"source": "doc.pdf"}, "distance": 0.1}
            ],
            "graph_context": "Concepts : IA, données",
        }
        result = agent.run(state)

        assert "reasoning" in result
        assert len(result["reasoning"]) > 0
        assert result["next_agent"] == "answer"


# ══════════════════════════════════════════════════════════════
# AnswerAgent
# ══════════════════════════════════════════════════════════════

class TestAnswerAgent:

    @patch("agents.answer_agent.ChatOpenAI")
    def test_answer_generates_response(self, mock_llm_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Le machine learning est une technique d'IA qui permet aux systèmes d'apprendre."
        )
        mock_llm_class.return_value = mock_llm

        agent = AnswerAgent()
        state = {
            **initial_state("Qu'est-ce que le ML ?"),
            "reasoning": "Synthèse : le ML utilise des données pour apprendre.",
            "sources": ["source1.pdf"],
        }
        result = agent.run(state)

        assert "answer" in result
        assert len(result["answer"]) > 0
        assert result["next_agent"] == "end"
