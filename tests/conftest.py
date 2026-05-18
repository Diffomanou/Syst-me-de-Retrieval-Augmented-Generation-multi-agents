"""
tests/conftest.py
──────────────────
Configuration globale des tests pytest.
Définit les fixtures partagées entre tous les fichiers de test.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client():
    """Client HTTP de test pour FastAPI (ne lance pas de vrai serveur)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_openai():
    """Mock du LLM OpenAI pour éviter les appels API pendant les tests."""
    with patch("langchain_openai.ChatOpenAI") as mock:
        instance = MagicMock()
        instance.invoke.return_value = MagicMock(
            content="Réponse de test générée par le mock."
        )
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_vector_store():
    """Mock du vector store ChromaDB."""
    with patch("database.vector_store.similarity_search") as mock:
        mock.return_value = [
            {
                "id": "test-doc-1",
                "content": "Contenu de test pour le document 1.",
                "metadata": {"source": "test_source.pdf", "title": "Doc Test"},
                "distance": 0.15,
            }
        ]
        yield mock


@pytest.fixture
def mock_neo4j():
    """Mock du client Neo4j."""
    with patch("database.neo4j_client.get_graph_context") as mock:
        mock.return_value = "Contexte graphe de test : concepts liés trouvés."
        yield mock


@pytest.fixture
def sample_query() -> dict:
    """Requête de test standard."""
    return {"question": "Qu'est-ce que le machine learning ?"}


@pytest.fixture
def sample_ingest() -> dict:
    """Document de test pour l'ingestion."""
    return {
        "title": "Introduction au Machine Learning",
        "content": (
            "Le machine learning est une branche de l'intelligence artificielle. "
            "Il permet aux systèmes d'apprendre automatiquement à partir des données. "
            "Les algorithmes de ML incluent la régression, les arbres de décision "
            "et les réseaux de neurones."
        ),
        "source": "test_document.txt",
    }
