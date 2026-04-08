"""Clinical entity models."""
from pydantic import BaseModel, Field
from typing import Literal, Optional

class ClinicalEntity(BaseModel):
    """Extracted entity from clinical description."""

    text: str = Field(description="The entity text as it appears in the description")

    entity_type: Literal[
        "condition",      # Diseases, disorders, diagnoses
        "symptom",        # Patient-reported symptoms, clinical signs/findings
        "procedure",      # Surgical or clinical procedures
        "measurement",    # Lab tests, vitals, clinical measurements
        "drug",          # Medications or drug classes
        "device",        # Medical devices, implants
        "observation",   # Clinical observations (distinct from symptoms)
        "specimen",      # Biological specimens
        "visit",         # Care setting, encounter type
        "demographic",   # Age, race, ethnicity, gender
        "modifier",      # Severity, temporal, anatomical qualifiers
    ] = Field(
        description="Type of clinical entity"
    )

    domain: Literal[
        "Condition",     # Diseases, disorders
        "Procedure",     # Procedures
        "Measurement",   # Labs, vitals
        "Drug",         # Medications
        "Device",       # Medical devices
        "Observation",  # Symptoms, findings, observations, modifiers, age groups
        "Specimen",     # Biological samples
        "Visit",        # Encounter/care setting
        "Gender",       # Gender
        "Race",         # Race
        "Ethnicity",    # Ethnicity
    ] = Field(
        description="OMOP domain for searching"
    )

    is_required: bool = Field(
        default=True,
        description="Whether this entity MUST be in the concept set"
    )

    requires_descendants: bool = Field(
        default=True,
        description="Whether to include descendant concepts (ATLAS includeDescendants)"
    )

    is_exclusion: bool = Field(
        default=False,
        description="Whether this should be excluded from the concept set"
    )

    temporal_constraint: Optional[Literal[
        "acute",           # Acute onset/current episode
        "chronic",         # Chronic/long-standing condition
        "active",          # Currently active disease (not remission)
        "incident",        # New onset/first occurrence
        "prevalent",       # Existing cases
        "historical",      # Past history only
    ]] = Field(
        default=None,
        description="Temporal constraint for this entity (metadata only - helps with better concept selection)"
    )

    relationship_to_primary: Optional[str] = Field(
        default=None,
        description="How this entity relates to the primary condition (metadata only - helps with entity prioritization)"
    )

    rationale: str = Field(
        description="Clinical reasoning for why this entity was extracted"
    )

class ParsedClinicalDescription(BaseModel):
    """Result from clinical parser agent."""

    original_text: str = Field(description="The input clinical description")

    entities: list[ClinicalEntity] = Field(
        description="Extracted clinical entities"
    )

    interpretation: str = Field(
        description="Overall clinical interpretation of the description"
    )

    concept_set_strategy: str = Field(
        description="High-level strategy for building the concept set"
    )
