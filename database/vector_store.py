"""
database/vector_store.py
─────────────────────────
Gère la base vectorielle ChromaDB.
Embeddings : HuggingFace sentence-transformers (100% local et gratuit).
"""

import logging
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Encapsule ChromaDB pour le stockage et la recherche vectorielle."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Embeddings HuggingFace — téléchargés une seule fois, tournent en local
        self._embedder = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},   # utilise le CPU (pas besoin de GPU)
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(
            "VectorStore initialisé (HuggingFace '%s') — %d documents",
            settings.embedding_model,
            self._collection.count(),
        )

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Encode et stocke les textes dans ChromaDB."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        embeddings = self._embedder.embed_documents(texts)
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("%d documents ajoutés au vector store", len(texts))
        return ids

    def similarity_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retourne les k chunks les plus proches sémantiquement."""
        query_embedding = self._embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        documents = []
        for i, doc_id in enumerate(results["ids"][0]):
            documents.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return documents

    def count(self) -> int:
        return self._collection.count()


vector_store = VectorStore()
