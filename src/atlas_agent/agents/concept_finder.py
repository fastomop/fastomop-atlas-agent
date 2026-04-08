"""Concept Finder Agent - Finds OMOP concepts for clinical entities."""
from typing import List

from ..models import ClinicalEntity, ConceptMatch
from ..tools import MilvusSearchTool


class ConceptFinderAgent:
    """Agent that finds OMOP concepts matching clinical entities."""

    def __init__(self):
        """Initialize concept finder agent with Milvus vector search."""
        # Use Milvus for semantic search with embeddings
        # (Standard concepts with domain filtering and relationship data)
        self.search_tool = MilvusSearchTool()
        print("✓ Using Milvus vector search with semantic embeddings (4.1M OMOP concepts)")

    def find_concepts(
        self,
        entity: ClinicalEntity,
        top_k: int = 10,
        min_similarity: float = 0.6
    ) -> List[ConceptMatch]:
        """
        Find OMOP concepts matching a clinical entity using WebAPI with vocabulary filters.

        Args:
            entity: Clinical entity to find concepts for
            top_k: Maximum number of candidates to consider
            min_similarity: Minimum similarity threshold (not used for keyword search)

        Returns:
            List of ConceptMatch objects ranked by relevance
        """
        # Use Milvus semantic search with domain filtering
        # (Post-filters by domain after semantic search to avoid DiskANN graph issues)
        candidates = self.search_tool.search_concepts(
            query_text=entity.text,
            domain_filter=entity.domain,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        if not candidates:
            # Try without domain filter if no matches
            candidates = self.search_tool.search_concepts(
                query_text=entity.text,
                top_k=top_k,
                min_similarity=min_similarity - 0.1,  # Slightly lower threshold
            )

        # Return ALL candidates for relationship reasoning
        # The RelationshipReasoner will use OMOP relationship data to select the best match(es)
        # This ensures relationship evidence is used for disambiguation, not just similarity scores
        return candidates
