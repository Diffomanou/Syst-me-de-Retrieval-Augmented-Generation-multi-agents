"""
tests/test_api.py
──────────────────
Tests d'intégration pour les endpoints FastAPI.
"""

import pytest
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════
# Endpoint /health
# ══════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        with patch("api.main.neo4j_client") as mock_neo4j, \
             patch("api.main.vector_store") as mock_vs:
            mock_neo4j.verify_connection.return_value = True
            mock_vs.count.return_value = 42

            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "neo4j_connected" in data
        assert "vector_store_documents" in data


# ══════════════════════════════════════════════════════════════
# Endpoint /query
# ══════════════════════════════════════════════════════════════

class TestQueryEndpoint:

    def test_valid_query_returns_200(self, client, sample_query):
        with patch("api.routes.query.run_query") as mock_run:
            mock_run.return_value = {
                "answer": "Le machine learning est une technique d'IA.",
                "sources": ["doc_test.pdf"],
                "reasoning": "Synthèse basée sur les documents.",
                "graph_context": "Concepts : IA",
                "error": None,
                "iterations": 1,
            }
            response = client.post("/query", json=sample_query)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "sources" in data

    def test_short_question_returns_422(self, client):
        """FastAPI doit rejeter les questions trop courtes (validation Pydantic)."""
        response = client.post("/query", json={"question": "AI"})
        assert response.status_code == 422

    def test_missing_question_returns_422(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_include_reasoning_flag(self, client):
        with patch("api.routes.query.run_query") as mock_run:
            mock_run.return_value = {
                "answer": "Réponse.",
                "sources": [],
                "reasoning": "Raisonnement détaillé.",
                "graph_context": "",
                "error": None,
                "iterations": 1,
            }
            response = client.post(
                "/query",
                json={
                    "question": "Qu'est-ce que le deep learning ?",
                    "include_reasoning": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["reasoning"] is not None


# ══════════════════════════════════════════════════════════════
# Endpoint /ingest
# ══════════════════════════════════════════════════════════════

class TestIngestEndpoint:

    def test_valid_ingest_returns_200(self, client, sample_ingest):
        with patch("api.routes.ingest.vector_store") as mock_vs, \
             patch("api.routes.ingest.neo4j_client") as mock_neo4j, \
             patch("api.routes.ingest.ChatOpenAI") as mock_llm_class:

            mock_llm = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
            mock_llm.invoke.return_value.__class__.__name__ = "AIMessage"
            mock_llm.invoke.return_value.content = "machine learning, IA, données"
            mock_llm_class.return_value = mock_llm

            mock_vs.add_documents.return_value = ["id1", "id2"]
            mock_neo4j.add_document.return_value = None

            response = client.post("/ingest", json=sample_ingest)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["chunks_created"] > 0

    def test_short_content_returns_422(self, client):
        response = client.post(
            "/ingest",
            json={"title": "Test", "content": "Court", "source": "test"},
        )
        assert response.status_code == 422
