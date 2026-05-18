"""
database/neo4j_client.py
─────────────────────────
Client Neo4j — gère la connexion et toutes les opérations sur le graphe.

Schéma du graphe :
  (:Document {id, title, source, chunk_index})
  (:Concept   {name})
  (:Document)-[:MENTIONS]->(:Concept)
  (:Concept)-[:RELATED_TO]->(:Concept)
"""

import logging
from typing import Any

from neo4j import GraphDatabase, Driver
from config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Encapsule toutes les interactions avec la base Neo4j."""

    def __init__(self) -> None:
        self._driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Neo4j connecté à %s", settings.neo4j_uri)

    # ──────────────────────────────────────────────────────────
    # Cycle de vie
    # ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Ferme le driver (appeler à l'arrêt de l'app)."""
        self._driver.close()

    def verify_connection(self) -> bool:
        """Renvoie True si la connexion est opérationnelle."""
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.error("Neo4j non joignable : %s", exc)
            return False

    # ──────────────────────────────────────────────────────────
    # Initialisation du schéma
    # ──────────────────────────────────────────────────────────

    def create_constraints(self) -> None:
        """Crée les contraintes d'unicité (à appeler au démarrage)."""
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT doc_id IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT concept_name IF NOT EXISTS "
                "FOR (c:Concept) REQUIRE c.name IS UNIQUE"
            )
        logger.info("Contraintes Neo4j créées / vérifiées")

    # ──────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────

    def add_document(
        self,
        doc_id: str,
        title: str,
        source: str,
        chunk_index: int,
        concepts: list[str],
    ) -> None:
        """
        Insère un nœud Document et ses concepts dans le graphe.

        Paramètres
        ----------
        doc_id      : identifiant unique du chunk
        title       : titre du document parent
        source      : origine (nom de fichier, URL, etc.)
        chunk_index : position du chunk dans le document
        concepts    : mots-clés / entités extraites du contenu
        """
        with self._driver.session() as session:
            # 1. Crée ou met à jour le nœud Document
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.title = $title,
                    d.source = $source,
                    d.chunk_index = $chunk_index
                """,
                id=doc_id,
                title=title,
                source=source,
                chunk_index=chunk_index,
            )

            # 2. Pour chaque concept, crée le nœud et la relation
            for concept in concepts:
                session.run(
                    """
                    MERGE (c:Concept {name: $name})
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:MENTIONS]->(c)
                    """,
                    name=concept.lower().strip(),
                    doc_id=doc_id,
                )

    # ──────────────────────────────────────────────────────────
    # Recherche
    # ──────────────────────────────────────────────────────────

    def find_related_concepts(self, concepts: list[str], depth: int = 2) -> list[str]:
        """
        Parcourt le graphe pour trouver les concepts liés (jusqu'à `depth` sauts).

        Retourne une liste de noms de concepts voisins.
        """
        if not concepts:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Concept)
                WHERE c.name IN $concepts
                MATCH (c)-[:RELATED_TO*1..$depth]-(related:Concept)
                RETURN DISTINCT related.name AS name
                LIMIT 20
                """,
                concepts=[c.lower().strip() for c in concepts],
                depth=depth,
            )
            return [record["name"] for record in result]

    def find_documents_by_concepts(self, concepts: list[str]) -> list[dict[str, Any]]:
        """
        Retourne les documents qui mentionnent au moins un des concepts donnés.
        """
        if not concepts:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document)-[:MENTIONS]->(c:Concept)
                WHERE c.name IN $concepts
                WITH d, COUNT(c) AS relevance
                ORDER BY relevance DESC
                LIMIT 10
                RETURN d.id AS id, d.title AS title,
                       d.source AS source, relevance
                """,
                concepts=[c.lower().strip() for c in concepts],
            )
            return [dict(record) for record in result]

    def get_graph_context(self, query_concepts: list[str]) -> str:
        """
        Construit un texte de contexte à partir du graphe de connaissances.
        Utilisé par le RetrievalAgent pour enrichir la réponse.
        """
        related = self.find_related_concepts(query_concepts)
        docs = self.find_documents_by_concepts(query_concepts + related)

        if not docs:
            return "Aucun contexte trouvé dans le graphe de connaissances."

        lines = ["Contexte extrait du graphe de connaissances :"]
        for doc in docs[:5]:
            lines.append(
                f"- Document '{doc['title']}' (source: {doc['source']}) "
                f"— pertinence: {doc['relevance']}"
            )
        if related:
            lines.append(f"\nConcepts associés découverts : {', '.join(related[:10])}")

        return "\n".join(lines)


# Instance unique (pattern Singleton léger)
neo4j_client = Neo4jClient()
