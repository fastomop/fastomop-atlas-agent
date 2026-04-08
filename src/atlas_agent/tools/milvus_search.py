"""Milvus search tool for semantic concept retrieval."""
import os
import json
import numpy as np
from typing import List, Optional
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

from ..config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, EMBEDDING_MODEL
from ..models import ConceptMatch


class MilvusSearchTool:
    """Tool for searching OMOP concepts in Milvus vector database."""

    def __init__(self):
        self.host = MILVUS_HOST
        self.port = MILVUS_PORT
        self.collection_name = COLLECTION_NAME
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to Milvus
        connections.connect(host=self.host, port=self.port)
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def search_concepts(
        self,
        query_text: str,
        domain_filter: Optional[str] = None,
        top_k: int = 10,
        min_similarity: float = 0.5,
        include_relationships: bool = True,
        use_hybrid_search: bool = True,
    ) -> List[ConceptMatch]:
        """
        Search for concepts using hybrid semantic + exact text matching.

        Args:
            query_text: Clinical term to search for
            domain_filter: Optional OMOP domain (Condition, Procedure, Drug, Measurement, Observation)
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity score (0-1)
            include_relationships: Whether to include relationship data from Milvus (default: True)
            use_hybrid_search: If True, combines semantic search with exact text matching (default: True)

        Returns:
            List of ConceptMatch objects sorted by similarity (highest first)
        """
        # Generate query embedding
        # Use the EXACT method that works in omop-vectordb environment
        query_embedding = self.model.encode([query_text])[0]

        # IMPORTANT: Do NOT apply domain filter during vector search
        # DiskANN index with scalar filtering can return poor results because the filtered
        # graph traversal may not reach the best semantic matches.
        # Instead, we search without filter and apply domain filtering post-search.
        expr = None  # No pre-filtering

        # Search parameters - use EXACT params from working manual search
        search_params = {"metric_type": "COSINE", "params": {}}

        # Build output fields list
        output_fields = [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "standard_concept",
            "concept_class_id",
            "concept_code",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
        ]

        # Add relationship fields if requested
        if include_relationships:
            output_fields.extend([
                "parent_concept_id",
                "relationship_types",
            ])

        # Execute search
        # Fetch more results than needed since we'll filter by domain post-search
        search_limit = top_k * 3 if domain_filter else top_k
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=search_limit,
            expr=expr,
            output_fields=output_fields,
        )

        # Parse results into ConceptMatch objects
        # Note: With flat synonym storage, we may get multiple hits for same concept_id
        # (one per synonym). We deduplicate by keeping the highest scoring match.
        concept_matches = {}  # concept_id -> ConceptMatch

        for hit in results[0]:
            # Get entity dict
            entity = hit.entity.to_dict()

            # Only include results above similarity threshold
            similarity = float(hit.distance)
            if similarity < min_similarity:
                continue

            # POST-SEARCH DOMAIN FILTERING
            # Apply domain filter AFTER semantic search to avoid DiskANN graph traversal issues
            if domain_filter and entity["domain_id"] != domain_filter:
                continue  # Skip concepts not in target domain

            concept_id = int(entity["concept_id"])

            # If we've already seen this concept_id, only keep higher scoring match
            if concept_id in concept_matches:
                if similarity <= concept_matches[concept_id].similarity_score:
                    continue  # Skip lower scoring duplicate

            # Parse relationship data if included
            parent_concept_id = None
            relationship_types = []

            if include_relationships:
                parent_concept_id = entity.get("parent_concept_id")
                rel_types_json = entity.get("relationship_types", "[]")
                try:
                    relationship_types = json.loads(rel_types_json) if rel_types_json else []
                except json.JSONDecodeError:
                    relationship_types = []

            # Create ConceptMatch
            # Clamp similarity to [0, 1] to handle floating point rounding errors
            clamped_similarity = min(1.0, max(0.0, similarity))

            match = ConceptMatch(
                concept_id=concept_id,
                concept_name=entity["concept_name"],
                domain_id=entity["domain_id"],
                vocabulary_id=entity["vocabulary_id"],
                standard_concept=entity.get("standard_concept", ""),
                concept_class_id=entity.get("concept_class_id", ""),
                concept_code=entity.get("concept_code", ""),
                valid_start_date=entity.get("valid_start_date", ""),
                valid_end_date=entity.get("valid_end_date", ""),
                invalid_reason=entity.get("invalid_reason"),
                similarity_score=clamped_similarity,
                matched_entity=query_text,
                parent_concept_id=parent_concept_id,
                relationship_types=relationship_types,
            )
            concept_matches[concept_id] = match

        # HYBRID SEARCH: Add exact/fuzzy text matching as fallback
        # This helps when embedding model fails (e.g., rare medical terms)
        if use_hybrid_search and len(concept_matches) < top_k:
            text_matches = self._exact_text_search(
                query_text=query_text,
                domain_filter=domain_filter,
                limit=top_k,
                include_relationships=include_relationships,
            )

            # Merge text matches with semantic matches (avoid duplicates)
            for match in text_matches:
                if match.concept_id not in concept_matches:
                    # Assign high similarity for exact text matches
                    match.similarity_score = 0.95  # Boost exact matches
                    concept_matches[match.concept_id] = match

        # Return deduplicated matches sorted by similarity
        matches = sorted(concept_matches.values(), key=lambda x: x.similarity_score, reverse=True)
        return matches[:top_k]  # Ensure we don't exceed top_k after merging

    def get_concept_by_id(
        self,
        concept_id: int,
        include_relationships: bool = True,
    ) -> Optional[ConceptMatch]:
        """
        Retrieve a specific concept by its ID.

        Args:
            concept_id: OMOP concept ID
            include_relationships: Whether to include relationship data

        Returns:
            ConceptMatch object or None if not found
        """
        # Build output fields
        output_fields = [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "standard_concept",
            "concept_class_id",
            "concept_code",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
        ]

        if include_relationships:
            output_fields.extend([
                "parent_concept_id",
                "relationship_types",
            ])

        # Query by concept_id
        expr = f"concept_id == {concept_id}"
        results = self.collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=1,
        )

        if not results:
            return None

        # Parse result
        entity = results[0]

        # Parse relationship data if included
        parent_concept_id = None
        relationship_types = []

        if include_relationships:
            parent_concept_id = entity.get("parent_concept_id")
            rel_types_json = entity.get("relationship_types", "[]")
            try:
                relationship_types = json.loads(rel_types_json) if rel_types_json else []
            except json.JSONDecodeError:
                relationship_types = []

        return ConceptMatch(
            concept_id=int(entity["concept_id"]),
            concept_name=entity["concept_name"],
            domain_id=entity["domain_id"],
            vocabulary_id=entity["vocabulary_id"],
            standard_concept=entity.get("standard_concept", ""),
            concept_class_id=entity.get("concept_class_id", ""),
            concept_code=entity.get("concept_code", ""),
            valid_start_date=entity.get("valid_start_date", ""),
            valid_end_date=entity.get("valid_end_date", ""),
            invalid_reason=entity.get("invalid_reason"),
            similarity_score=1.0,  # Direct lookup, perfect match
            matched_entity=entity["concept_name"],
            parent_concept_id=parent_concept_id,
            relationship_types=relationship_types,
        )

    def find_concepts_by_relationship(
        self,
        relationship_types: List[str],
        domain_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[ConceptMatch]:
        """
        Find concepts that have specific relationship types.

        Args:
            relationship_types: List of relationship types to filter by
                               (e.g., ['May treat', 'FDA indication of'])
            domain_filter: Optional OMOP domain filter
            limit: Maximum number of results

        Returns:
            List of ConceptMatch objects with these relationships
        """
        # Build output fields
        output_fields = [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "standard_concept",
            "concept_class_id",
            "concept_code",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
            "parent_concept_id",
            "relationship_types",
        ]

        # Build filter expression
        # Note: This is a simplified implementation
        # Full relationship graph traversal would require more complex queries
        expr_parts = []

        if domain_filter:
            expr_parts.append(f"domain_id == '{domain_filter}'")

        # Combine filters
        expr = " and ".join(expr_parts) if expr_parts else None

        # Query Milvus
        # Note: We can't directly filter by JSON array contents in Milvus easily
        # So we fetch a larger set and filter in Python
        results = self.collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=limit * 3,  # Fetch more, filter in Python
        )

        # Filter by relationship types in Python
        matches = []
        for entity in results:
            rel_types_json = entity.get("relationship_types", "[]")
            try:
                entity_rels = json.loads(rel_types_json) if rel_types_json else []
            except json.JSONDecodeError:
                entity_rels = []

            # Check if any requested relationship type is present
            if any(rel in entity_rels for rel in relationship_types):
                parent_concept_id = entity.get("parent_concept_id")

                match = ConceptMatch(
                    concept_id=int(entity["concept_id"]),
                    concept_name=entity["concept_name"],
                    domain_id=entity["domain_id"],
                    vocabulary_id=entity["vocabulary_id"],
                    standard_concept=entity.get("standard_concept", ""),
                    concept_class_id=entity.get("concept_class_id", ""),
                    concept_code=entity.get("concept_code", ""),
                    valid_start_date=entity.get("valid_start_date", ""),
                    valid_end_date=entity.get("valid_end_date", ""),
                    invalid_reason=entity.get("invalid_reason"),
                    similarity_score=1.0,  # Exact relationship match
                    matched_entity="relationship query",
                    parent_concept_id=parent_concept_id,
                    relationship_types=entity_rels,
                )
                matches.append(match)

                if len(matches) >= limit:
                    break

        return matches

    def _exact_text_search(
        self,
        query_text: str,
        domain_filter: Optional[str] = None,
        limit: int = 10,
        include_relationships: bool = True,
    ) -> List[ConceptMatch]:
        """
        Exact/fuzzy text search fallback when semantic search fails.

        Searches concept_name field using exact match and case-insensitive partial match.

        Args:
            query_text: Text to search for
            domain_filter: Optional domain filter
            limit: Maximum results
            include_relationships: Include relationship data

        Returns:
            List of ConceptMatch objects from text matching
        """
        # Build output fields
        output_fields = [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "standard_concept",
            "concept_class_id",
            "concept_code",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
        ]

        if include_relationships:
            output_fields.extend([
                "parent_concept_id",
                "relationship_types",
            ])

        # Try exact match first (case-sensitive)
        expr_parts = [f'concept_name == "{query_text}"']
        if domain_filter:
            expr_parts.append(f'domain_id == "{domain_filter}"')

        expr = " and ".join(expr_parts)
        results = self.collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=limit,
        )

        # If no exact match, try case-insensitive variations
        if not results:
            variations = [
                query_text.lower(),
                query_text.upper(),
                query_text.title(),
                query_text.capitalize(),
            ]

            for variation in variations:
                expr_parts = [f'concept_name == "{variation}"']
                if domain_filter:
                    expr_parts.append(f'domain_id == "{domain_filter}"')

                expr = " and ".join(expr_parts)
                results = self.collection.query(
                    expr=expr,
                    output_fields=output_fields,
                    limit=limit,
                )
                if results:
                    break

        # Convert to ConceptMatch objects
        matches = []
        for entity in results:
            # Parse relationship data if included
            parent_concept_id = None
            relationship_types = []

            if include_relationships:
                parent_concept_id = entity.get("parent_concept_id")
                rel_types_json = entity.get("relationship_types", "[]")
                try:
                    relationship_types = json.loads(rel_types_json) if rel_types_json else []
                except json.JSONDecodeError:
                    relationship_types = []

            match = ConceptMatch(
                concept_id=int(entity["concept_id"]),
                concept_name=entity["concept_name"],
                domain_id=entity["domain_id"],
                vocabulary_id=entity["vocabulary_id"],
                standard_concept=entity.get("standard_concept", ""),
                concept_class_id=entity.get("concept_class_id", ""),
                concept_code=entity.get("concept_code", ""),
                valid_start_date=entity.get("valid_start_date", ""),
                valid_end_date=entity.get("valid_end_date", ""),
                invalid_reason=entity.get("invalid_reason"),
                similarity_score=0.95,  # High score for exact text match
                matched_entity=query_text,
                parent_concept_id=parent_concept_id,
                relationship_types=relationship_types,
            )
            matches.append(match)

        return matches
