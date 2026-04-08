"""Concept set models."""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class RelatedConcept(BaseModel):
    """A concept discovered through relationship traversal."""

    concept_id: int
    concept_name: str
    domain_id: str
    relationship_type: str = Field(
        description="Type of relationship (e.g., 'May treat', 'Has manifestation', 'Is a')"
    )
    relationship_direction: Literal["outgoing", "incoming"] = Field(
        description="Direction of relationship from source concept"
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How relevant this related concept is (0-1)"
    )
    rationale: str = Field(
        description="Why this relationship matters for the concept set"
    )

class ConceptMatch(BaseModel):
    """A matched OMOP concept from Milvus."""

    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    standard_concept: str
    concept_class_id: str
    concept_code: str
    valid_start_date: str
    valid_end_date: str
    invalid_reason: Optional[str] = None
    similarity_score: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score"
    )
    matched_entity: Optional[str] = Field(
        default="",
        description="Which clinical entity this concept matches"
    )

    # NEW: Relationship data from Milvus
    parent_concept_id: Optional[str] = Field(
        default=None,
        description="Direct parent concept via 'Is a' or 'Subsumes' relationship"
    )
    relationship_types: List[str] = Field(
        default_factory=list,
        description="All relationship types this concept participates in"
    )

    # NEW: Related concepts discovered during reasoning
    related_concepts: Optional[List[RelatedConcept]] = Field(
        default=None,
        description="Concepts discovered through relationship traversal"
    )

class ConceptSetItem(BaseModel):
    """Single item in concept set (internal representation)."""

    concept: ConceptMatch
    include_descendants: bool = True
    is_excluded: bool = False
    include_mapped: bool = False
    rationale: str = Field(
        description="Clinical reasoning for including this concept"
    )

class ConceptSet(BaseModel):
    """Complete concept set (internal representation)."""

    name: str = Field(description="Name of the concept set")
    description: str = Field(description="Clinical description of what this set represents")
    items: list[ConceptSetItem]
    validation_notes: list[str] = Field(
        default_factory=list,
        description="Clinical validation notes and warnings"
    )
    coverage_summary: str = Field(
        default="",
        description="Summary of domain coverage"
    )
