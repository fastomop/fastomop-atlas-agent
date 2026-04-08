"""ATLAS JSON export format models."""
from pydantic import BaseModel
from typing import Optional

class AtlasConcept(BaseModel):
    """ATLAS concept format (exact match to ATLAS export)."""

    CONCEPT_ID: int
    CONCEPT_NAME: str
    DOMAIN_ID: str
    VOCABULARY_ID: str
    CONCEPT_CLASS_ID: str
    STANDARD_CONCEPT: str
    CONCEPT_CODE: str
    VALID_START_DATE: str
    VALID_END_DATE: str
    INVALID_REASON: str
    INVALID_REASON_CAPTION: str
    STANDARD_CONCEPT_CAPTION: str

class AtlasConceptSetItem(BaseModel):
    """ATLAS concept set item (exact match to ATLAS export)."""

    concept: AtlasConcept
    isExcluded: bool
    includeDescendants: bool
    includeMapped: bool

class AtlasConceptSet(BaseModel):
    """ATLAS concept set export format."""

    items: list[AtlasConceptSetItem]

    def to_json(self) -> str:
        """Export as ATLAS-compatible JSON."""
        return self.model_dump_json(indent=2, by_alias=True)
