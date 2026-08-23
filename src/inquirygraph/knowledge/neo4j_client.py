import re
import time
import uuid
from dataclasses import dataclass

from neo4j import GraphDatabase

from inquirygraph.config.settings import settings


@dataclass
class GraphClaim:
    claim_id: str
    text: str
    source_url: str
    theme: str


class KnowledgeGraph:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=10.0,
            connection_acquisition_timeout=10.0,
        )
        try:
            self._wait_until_ready()
            self._ensure_schema()
        except Exception:
            self.driver.close()
            raise

    def __enter__(self) -> "KnowledgeGraph":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.driver.close()

    def _wait_until_ready(self, attempts: int = 6, delay_seconds: float = 5.0) -> None:
        """Wait for Neo4j Bolt after a fresh container startup."""
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                self.driver.verify_connectivity()
                return
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(delay_seconds)
        raise RuntimeError(
            f"Neo4j was not ready after {attempts * delay_seconds:.0f}s"
        ) from last_error

    def _ensure_schema(self) -> None:
        with self.driver.session() as session:
            session.run(
                """
                CREATE CONSTRAINT investigation_id IF NOT EXISTS
                FOR (i:Investigation) REQUIRE i.id IS UNIQUE
                """
            )
            session.run(
                """
                CREATE CONSTRAINT entity_key IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.key IS UNIQUE
                """
            )

    def init_investigation(self, investigation_id: str, query: str) -> None:
        with self.driver.session() as session:
            # Use a query-prefixed parameter name to avoid colliding with the
            # Neo4j driver's own `query` keyword argument on Session.run().
            session.run(
                """
                MERGE (i:Investigation {id: $id})
                SET i.query = $investigation_query, i.updated_at = datetime()
                """,
                id=investigation_id,
                investigation_query=query,
            )

    def upsert_knowledge(
        self,
        investigation_id: str,
        entities: list[str],
        claims: list[dict],
        source_url: str,
        source_title: str,
    ) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MATCH (i:Investigation {id: $investigation_id})
                MERGE (s:Source {url: $source_url})
                SET s.title = $source_title
                MERGE (i)-[:HAS_SOURCE]->(s)
                """,
                investigation_id=investigation_id,
                source_url=source_url,
                source_title=source_title,
            )

            for name in entities:
                key = _entity_key(investigation_id, name)
                session.run(
                    """
                    MATCH (i:Investigation {id: $investigation_id})
                    MERGE (e:Entity {key: $key})
                    SET e.name = $name, e.normalized = toLower($name)
                    MERGE (i)-[:MENTIONS]->(e)
                    """,
                    investigation_id=investigation_id,
                    key=key,
                    name=name.strip(),
                )

            for claim in claims:
                claim_id = str(uuid.uuid4())
                session.run(
                    """
                    MATCH (i:Investigation {id: $investigation_id})
                    MATCH (s:Source {url: $source_url})
                    CREATE (c:Claim {id: $claim_id, text: $text, theme: $theme})
                    MERGE (i)-[:HAS_CLAIM]->(c)
                    MERGE (c)-[:FROM_SOURCE]->(s)
                    """,
                    investigation_id=investigation_id,
                    source_url=source_url,
                    claim_id=claim_id,
                    text=claim["text"],
                    theme=claim.get("theme", "general"),
                )
                session.run(
                    """
                    MATCH (c:Claim {id: $claim_id}), (s:Source {url: $source_url})
                    MERGE (c)-[:SUPPORTED_BY]->(s)
                    """,
                    claim_id=claim_id,
                    source_url=source_url,
                )
                for entity_name in claim.get("entities", []):
                    key = _entity_key(investigation_id, entity_name)
                    session.run(
                        """
                        MERGE (e:Entity {key: $key})
                        SET e.name = $name
                        WITH e
                        MATCH (c:Claim {id: $claim_id})
                        MERGE (c)-[:ABOUT]->(e)
                        """,
                        key=key,
                        name=entity_name.strip(),
                        claim_id=claim_id,
                    )

    def add_contradictions(self, contradictions: list[dict]) -> None:
        with self.driver.session() as session:
            for contradiction in contradictions:
                session.run(
                    """
                    MATCH (a:Claim), (b:Claim)
                    WHERE a.text = $claim_a AND b.text = $claim_b
                    MERGE (a)-[r:CONTRADICTS]->(b)
                    SET r.explanation = $explanation
                    """,
                    claim_a=contradiction["claim_a"],
                    claim_b=contradiction["claim_b"],
                    explanation=contradiction.get("explanation", ""),
                )

    def get_claims(self, investigation_id: str, limit: int = 20) -> list[GraphClaim]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (i:Investigation {id: $investigation_id})-[:HAS_CLAIM]->(c:Claim)
                OPTIONAL MATCH (c)-[:FROM_SOURCE]->(s:Source)
                RETURN c.id AS claim_id, c.text AS text, c.theme AS theme,
                       coalesce(s.url, '') AS source_url
                LIMIT $limit
                """,
                investigation_id=investigation_id,
                limit=limit,
            )
            return [
                GraphClaim(
                    claim_id=row["claim_id"],
                    text=row["text"],
                    source_url=row["source_url"],
                    theme=row["theme"] or "general",
                )
                for row in result
            ]

    def get_claims_for_entities(self, investigation_id: str, entities: list[str]) -> list[GraphClaim]:
        if not entities:
            return self.get_claims(investigation_id)

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (i:Investigation {id: $investigation_id})-[:HAS_CLAIM]->(c:Claim)-[:ABOUT]->(e:Entity)
                WHERE toLower(e.name) IN $entities
                OPTIONAL MATCH (c)-[:FROM_SOURCE]->(s:Source)
                RETURN DISTINCT c.id AS claim_id, c.text AS text, c.theme AS theme,
                       coalesce(s.url, '') AS source_url
                LIMIT 15
                """,
                investigation_id=investigation_id,
                entities=[e.lower() for e in entities],
            )
            return [
                GraphClaim(
                    claim_id=row["claim_id"],
                    text=row["text"],
                    source_url=row["source_url"],
                    theme=row["theme"] or "general",
                )
                for row in result
            ]


def _entity_key(investigation_id: str, name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    return f"{investigation_id}::{normalized}"
