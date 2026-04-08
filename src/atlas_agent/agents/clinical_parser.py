"""Clinical Parser Agent - Extracts structured entities from clinical descriptions."""
from agno.agent import Agent

from ..config import get_agent_config
from ..model_factory import create_model
from ..models import ParsedClinicalDescription


class ClinicalParserAgent:
    """Agent that parses clinical descriptions into structured entities."""

    def __init__(self):
        # Get agent configuration
        agent_config = get_agent_config("clinical_parser")

        # Create model
        model = create_model(agent_config)

        # Disable native structured outputs for Ollama models
        if agent_config.get("MODEL_TYPE") == "ollama":
            model.supports_native_structured_outputs = False

        self.agent = Agent(
            name=agent_config.get("name", "Clinical Parser"),
            model=model,
            description=agent_config.get("description", "Extract clinical entities for OMOP concept sets"),
            instructions=[
                "Extract clinical entities from vignettes for OMOP concept set creation.",
                "",
                "ENTITY TYPES: condition, symptom, procedure, measurement, drug, device, visit, demographic, observation",
                "DOMAINS: Condition, Observation, Procedure, Measurement, Drug, Device, Visit, Specimen",
                "",
                "ENTITY FIELDS:",
                "- entity_type: condition/symptom/procedure/measurement/drug/device/visit/demographic/observation",
                "- domain: OMOP domain from list above",
                "- is_required: true for core entities, false for optional",
                "- requires_descendants: true for conditions/drugs (includes subtypes), false for specific entities",
                "- is_exclusion: true if entity should be EXCLUDED from concept set",
                "- temporal_constraint: active/chronic/acute/incident/prevalent (if specified)",
                "- relationship_to_primary: must_coexist/diagnostic_confirmation/treatment_context (if applicable)",
                "- rationale: brief reason for extraction",
                "",
                "CRITICAL RULES:",
                "",
                "1. DOMAIN DISTINCTION:",
                "   - Measurement: test names ('ANA test', 'blood pressure')",
                "   - Observation: test results/findings ('ANA positive', 'proteinuria', 'hypertension')",
                "   - Condition: diseases (always requires_descendants=true)",
                "",
                "2. SYNONYMS: Extract ONCE using full medical term",
                "   - 'Synonyms: SLE, Systemic lupus' → Extract 'Systemic lupus erythematosus' only",
                "   - Use full terms, avoid abbreviations",
                "",
                "3. EXCLUSIONS:",
                "   - Extract from sections: 'Related, differential, not sufficient for inclusion'",
                "   - Parent-child hierarchies: Extract ONLY parent with requires_descendants=true",
                "   - 'Cutaneous lupus (discoid, subacute)' → Extract 'Cutaneous lupus erythematosus' once",
                "",
                "4. SKIP CONTEXT:",
                "   - Sections marked 'for context; not encoded in this concept set' → DO NOT extract",
                "   - Diagnostic criteria, treatments, age constraints → DO NOT extract",
                "",
                "5. MANIFESTATIONS:",
                "   - Only extract if marked 'isolated' or in exclusion sections",
                "   - Don't extract disease manifestations as separate entities",
            ],
            # Use output_schema - with supports_native_structured_outputs=False, agno will parse JSON from text
            output_schema=ParsedClinicalDescription,
            markdown=False,
        )

    def _restructure_vignette(self, clinical_description: str) -> str:
        """
        Preprocess: Convert narrative vignette into explicit structured format.

        This helps the model understand inclusion/exclusion intent more clearly.
        """
        # Check if already structured (has explicit INCLUDE/EXCLUDE sections)
        if "INCLUDE:" in clinical_description or "EXCLUDE:" in clinical_description:
            return clinical_description

        # OPTIMIZATION: Skip restructuring for Mind Meets Machines vignettes
        # These are already well-structured with clear sections
        if any(marker in clinical_description for marker in [
            "# [C0", "## [C0",  # Challenge ID markers with bracket
            "# C0", "## C0",  # Challenge ID markers without bracket
            "Clinical Scope and Granularity",  # Standard section
            "Related, differential or comorbid conditions",  # Standard section
            "Synonyms:",  # Synonym section
        ]):
            #print("   ⚡ Skipping restructuring (Mind Meets Machines vignette detected)")
            return clinical_description

        # Restructure narrative vignette into explicit format
        restructure_prompt = f"""
Convert this narrative clinical vignette into a clear, structured format that explicitly states what should be INCLUDED and EXCLUDED in the concept set.

ORIGINAL VIGNETTE:
{clinical_description}

YOUR TASK:
Analyze the vignette and create a structured version with these sections:

1. CORE DEFINITION: One sentence summary of what we're looking for
2. INCLUDE: List all conditions/terms that SHOULD be in the concept set
3. EXCLUDE: List all conditions/terms that should NOT be in the concept set
4. CONTEXT: Clinical context (population, temporality, treatments mentioned)

CRITICAL INSTRUCTIONS:
- Section "Related, differential or comorbid conditions that are not sufficient for inclusion" → these go in EXCLUDE
- Section "Synonyms" → these go in INCLUDE
- "Clinical Scope and Granularity" details → extract temporal constraints, population, etiology filters
- Treatments/serology mentioned in "Diagnostic Criteria" or "Common Treatments" → these are CONTEXT, not entities to include
- Section "Diagnostic Criteria (for context; not encoded in this concept set)" → these are CONTEXT ONLY, do NOT extract as entities
- Phrases like "for context", "support phenotype confirmation but are not part of", "not encoded in this concept set" → these are CONTEXT ONLY
- Be explicit: if something is "not sufficient for inclusion", list it under EXCLUDE

CRITICAL: CONTEXT vs CONCEPTS
- If a section says "for context; not encoded in this concept set" → DO NOT extract those as entities
- If a section says "support phenotype confirmation but are not part of this diagnosis concept set" → DO NOT extract those as entities
- Serology/labs mentioned in "Diagnostic Criteria" → CONTEXT ONLY (not entities)
- Treatments mentioned in "Common Treatments" → CONTEXT ONLY (not entities)
- Population age constraints like "adults 18+" → CONTEXT ONLY, do NOT extract as demographic concept (should be cohort entry criteria)

Format your response as:

CORE DEFINITION:
[One sentence]

INCLUDE:
- [term 1]
- [term 2]
...

EXCLUDE:
- [term 1]
- [term 2]
...

CONTEXT:
- Population: [e.g., adults 18+]
- Temporality: [e.g., currently active]
- Etiology: [e.g., primary only, exclude drug-induced]
- Serology: [if mentioned for context]
- Treatments: [if mentioned for context]
"""

        response = self.agent.run(restructure_prompt)
        restructured = response.content if hasattr(response, 'content') else str(response)

        # If the agent returned a ParsedClinicalDescription object (because output_schema is set),
        # convert it to JSON string so it can be used as input for the next step
        if isinstance(restructured, ParsedClinicalDescription):
            import json
            restructured = json.dumps({
                "original_text": restructured.original_text,
                "entities": [e.model_dump() for e in restructured.entities],
                "interpretation": restructured.interpretation,
                "concept_set_strategy": restructured.concept_set_strategy
            }, indent=2)

        print(f"\n📋 Restructured vignette:")
        print("=" * 80)
        print(restructured[:1000] if len(str(restructured)) > 1000 else restructured)
        print("=" * 80 + "\n")

        return restructured

    def _validate_entities(self, parsed: ParsedClinicalDescription) -> ParsedClinicalDescription:
        """
        Post-extraction validation to filter out inappropriate entities.

        Args:
            parsed: Parsed description with entities

        Returns:
            ParsedClinicalDescription with validated entities
        """
        validated_entities = []
        skipped_entities = []

        for entity in parsed.entities:
            # Skip category descriptions (too generic for search)
            category_patterns = [
                "organ-specific diagnoses",
                "organ involvement",
                "system involvement",
                "manifestations",
                "specific body systems",
                "inflammatory conditions",
            ]

            entity_text_lower = entity.text.lower()
            if any(pattern in entity_text_lower for pattern in category_patterns):
                skipped_entities.append((entity.text, "Too generic - category description"))
                continue

            # Skip if entity is just an adjective/modifier without substance
            if len(entity.text.split()) <= 2 and entity.entity_type == "modifier":
                # Allow anatomical modifiers but skip generic ones
                generic_modifiers = ["specific", "various", "multiple", "all", "any", "certain"]
                if any(mod in entity_text_lower for mod in generic_modifiers):
                    skipped_entities.append((entity.text, "Too generic - vague modifier"))
                    continue

            validated_entities.append(entity)

        if skipped_entities:
            print(f"\n⚠️  Validation: Skipped {len(skipped_entities)} generic entities:")
            for text, reason in skipped_entities[:5]:
                print(f"   - '{text}': {reason}")
            if len(skipped_entities) > 5:
                print(f"   ... and {len(skipped_entities) - 5} more")

        return ParsedClinicalDescription(
            original_text=parsed.original_text,
            entities=validated_entities,
            interpretation=parsed.interpretation,
            concept_set_strategy=parsed.concept_set_strategy,
        )

    def parse(self, clinical_description: str) -> ParsedClinicalDescription:
        """
        Parse a clinical description into structured entities.

        Args:
            clinical_description: Natural language clinical description (narrative or structured)

        Returns:
            ParsedClinicalDescription with extracted entities
        """
        # Step 1: Restructure narrative vignette into explicit format
        structured_description = self._restructure_vignette(clinical_description)

        # Step 2: Extract entities from structured description
        prompt = f"""
Extract entities for an OMOP concept set from this clinical vignette:

{structured_description}

Apply the extraction rules from your instructions. Extract:
1. All conditions, procedures, drugs, and measurements mentioned in INCLUDE sections
2. All exclusions from "Related, differential, not sufficient for inclusion" sections
3. Risk factors and comorbidities if explicitly listed
4. Measurements and biomarkers (BNP, troponin, etc.)

Skip:
- Context sections marked "for context; not encoded"
- Treatments mentioned only for background
- Age constraints and demographic filters (unless part of the condition definition)
- Synonym duplicates (extract once using full medical term)
- Hierarchy duplicates (extract parent only with requires_descendants=true)
"""
        print("   ⏳ Calling LLM for entity extraction (this may take 1-2 minutes)...")
        import time
        start = time.time()
        response = self.agent.run(prompt)
        elapsed = time.time() - start
        print(f"   ✓ LLM responded in {elapsed:.1f}s")

        # Handle both ParsedClinicalDescription objects and string responses
        import json
        import re

        # Try to get the actual response content
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        # If content is already a ParsedClinicalDescription, use it directly
        if isinstance(content, ParsedClinicalDescription):
            parsed = content
        elif isinstance(content, str):
            # Try to parse JSON from string
            try:
                # First try direct JSON parse
                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    parsed_json = json.loads(json_match.group(1))
                else:
                    # Last resort: try to find any JSON object in the response
                    json_match = re.search(r'\{.*?"entities".*?\}', content, re.DOTALL)
                    if json_match:
                        parsed_json = json.loads(json_match.group(0))
                    else:
                        # If all parsing fails, create empty result with required fields
                        print(f"⚠️  Failed to parse LLM response, using empty entity list")
                        print(f"Response preview: {content[:500]}")
                        parsed_json = {
                            "original_text": structured_description,
                            "entities": [],
                            "interpretation": "Failed to parse LLM response",
                            "concept_set_strategy": "No strategy available due to parsing failure"
                        }

            # Fix entity field name variations
            if 'entities' in parsed_json and len(parsed_json['entities']) > 0:
                for entity in parsed_json['entities']:
                    if 'name' in entity and 'text' not in entity:
                        entity['text'] = entity.pop('name')
                    elif 'entity_name' in entity and 'text' not in entity:
                        entity['text'] = entity.pop('entity_name')
                    elif 'entity_text' in entity and 'text' not in entity:
                        entity['text'] = entity.pop('entity_text')

            parsed = ParsedClinicalDescription(**parsed_json)
        else:
            raise ValueError(f"Unexpected response type: {type(content)}")

        # Step 3: Validate entities to filter out inappropriate ones
        return self._validate_entities(parsed)
