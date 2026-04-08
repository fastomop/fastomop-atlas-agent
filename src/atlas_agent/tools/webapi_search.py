"""Direct WebAPI vocabulary search (no MCP, no Milvus)."""
import httpx
from typing import List, Optional

from ..models import ConceptMatch


class WebAPISearchTool:
    """
    Search OMOP vocabulary directly via OHDSI WebAPI.

    Lightweight alternative to both Milvus and MCP - just direct HTTP calls.
    """

    def __init__(
        self,
        webapi_url: str = "http://localhost:8080/WebAPI",
        cdm_source: str = "MIMIC"
    ):
        """
        Initialize WebAPI search tool.

        Args:
            webapi_url: Base URL for OHDSI WebAPI
            cdm_source: CDM source key
        """
        self.base_url = webapi_url.rstrip("/")
        self.cdm_source = cdm_source

    def search_concepts(
        self,
        query_text: str,
        domain_filter: Optional[str] = None,
        top_k: int = 20,
        min_similarity: float = 0.5,
        matched_entity: Optional[str] = None,
        **kwargs
    ) -> List[ConceptMatch]:
        """
        Search for concepts using WebAPI.

        Args:
            query_text: Text to search for
            domain_filter: Optional domain filter (Condition, Drug, Measurement, Procedure)
            top_k: Maximum number of results
            min_similarity: Not used (keyword search)

        Returns:
            List of ConceptMatch objects
        """
        # Build search URL
        url = f"{self.base_url}/vocabulary/{self.cdm_source}/search"

        # Build query params with strict filters (following omcp_vocab pattern)
        params = {
            "query": query_text,
            "pageSize": 100,  # Get more results to filter and sort
            "standardConcept": "S",  # Only standard concepts
        }

        # Add domain filter if specified
        if domain_filter:
            # WebAPI uses domain IDs like "Condition", "Drug"
            params["domainId"] = [domain_filter]

        try:
            # Make HTTP request
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                results = response.json()

            # WebAPI returns a list directly, not {"content": [...]}
            all_concepts = results if isinstance(results, list) else results.get("content", [])

            # Apply strict post-filters (following omcp_vocab pattern)
            filtered_concepts = []
            for concept in all_concepts:
                # Only standard concepts
                if concept.get("STANDARD_CONCEPT") != "S":
                    continue

                # Only valid concepts (not deprecated/invalid)
                if concept.get("INVALID_REASON") and concept.get("INVALID_REASON") != "V":
                    continue

                # Domain-specific vocabulary filtering
                if domain_filter == "Condition":
                    # Only SNOMED for conditions
                    if concept.get("VOCABULARY_ID") != "SNOMED":
                        continue
                    if concept.get("DOMAIN_ID") != "Condition":
                        continue
                elif domain_filter == "Drug":
                    # Only RxNorm for drugs
                    if concept.get("VOCABULARY_ID") not in ["RxNorm", "RxNorm Extension"]:
                        continue
                    if concept.get("DOMAIN_ID") != "Drug":
                        continue
                elif domain_filter:
                    # For other domains, just check domain matches
                    if concept.get("DOMAIN_ID") != domain_filter:
                        continue

                filtered_concepts.append(concept)

            # Sort by relevance (following omcp_vocab pattern)
            query_lower = query_text.lower()
            def sort_key(c):
                name_lower = c.get("CONCEPT_NAME", "").lower()
                # Exact match = 0, starts with = 1, contains = 2
                if name_lower == query_lower:
                    match_score = 0
                elif name_lower.startswith(query_lower):
                    match_score = 1
                else:
                    match_score = 2
                # Secondary sort: shorter names first (more general)
                # Tertiary sort: alphabetically
                return (match_score, len(name_lower), name_lower)

            filtered_concepts.sort(key=sort_key)

            # Limit to requested page size
            concepts = filtered_concepts[:top_k]

            # Convert to ConceptMatch objects
            matches = []
            for idx, concept in enumerate(concepts):
                # Rank-based pseudo-similarity score (clamped to [0.1, 1.0])
                similarity = max(0.1, 1.0 - (idx * 0.05))

                # Convert timestamps to ISO date strings
                from datetime import datetime
                valid_start = datetime.fromtimestamp(concept.get("VALID_START_DATE", 0) / 1000).strftime("%Y-%m-%d") if concept.get("VALID_START_DATE") else "1970-01-01"
                valid_end = datetime.fromtimestamp(concept.get("VALID_END_DATE", 0) / 1000).strftime("%Y-%m-%d") if concept.get("VALID_END_DATE") else "2099-12-31"

                match = ConceptMatch(
                    concept_id=int(concept["CONCEPT_ID"]),
                    concept_name=concept["CONCEPT_NAME"],
                    concept_code=concept.get("CONCEPT_CODE", ""),
                    vocabulary_id=concept.get("VOCABULARY_ID", ""),
                    domain_id=concept.get("DOMAIN_ID", domain_filter or ""),
                    concept_class_id=concept.get("CONCEPT_CLASS_ID", ""),
                    standard_concept=concept.get("STANDARD_CONCEPT", ""),
                    valid_start_date=valid_start,
                    valid_end_date=valid_end,
                    invalid_reason=concept.get("INVALID_REASON"),
                    similarity_score=similarity,
                    matched_entity=matched_entity or query_text
                )
                matches.append(match)

            return matches

        except httpx.HTTPError as e:
            print(f"WebAPI search failed: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def get_concept_by_id(self, concept_id: int) -> Optional[ConceptMatch]:
        """
        Get a concept by its ID.

        Args:
            concept_id: OMOP concept ID

        Returns:
            ConceptMatch if found, None otherwise
        """
        url = f"{self.base_url}/vocabulary/{self.cdm_source}/concept/{concept_id}"

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
                response.raise_for_status()
                concept = response.json()

            # Convert to ConceptMatch
            from datetime import datetime
            valid_start = datetime.fromtimestamp(concept.get("VALID_START_DATE", 0) / 1000).strftime("%Y-%m-%d") if concept.get("VALID_START_DATE") else "1970-01-01"
            valid_end = datetime.fromtimestamp(concept.get("VALID_END_DATE", 0) / 1000).strftime("%Y-%m-%d") if concept.get("VALID_END_DATE") else "2099-12-31"

            return ConceptMatch(
                concept_id=int(concept["CONCEPT_ID"]),
                concept_name=concept["CONCEPT_NAME"],
                concept_code=concept.get("CONCEPT_CODE", ""),
                vocabulary_id=concept.get("VOCABULARY_ID", ""),
                domain_id=concept.get("DOMAIN_ID", ""),
                concept_class_id=concept.get("CONCEPT_CLASS_ID", ""),
                standard_concept=concept.get("STANDARD_CONCEPT", ""),
                valid_start_date=valid_start,
                valid_end_date=valid_end,
                invalid_reason=concept.get("INVALID_REASON"),
                similarity_score=1.0,  # Direct ID lookup
                matched_entity=f"concept_id:{concept_id}"
            )

        except Exception as e:
            print(f"⚠️  Failed to get concept {concept_id}: {e}")
            return None

    def search_by_code(self, code: str, vocabulary_id: str) -> Optional[ConceptMatch]:
        """
        Search for a concept by its code.

        Args:
            code: Concept code
            vocabulary_id: Vocabulary ID

        Returns:
            ConceptMatch if found, None otherwise
        """
        # Search and filter for exact match
        results = self.search_concepts(query_text=code, top_k=10)

        for result in results:
            if result.concept_code == code and result.vocabulary_id == vocabulary_id:
                return result

        return None
