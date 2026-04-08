"""Relationship Reasoner Agent - Uses OMOP relationships for clinical reasoning."""
from typing import List
from agno.agent import Agent

from ..config import get_agent_config
from ..model_factory import create_model
from ..models import ClinicalEntity, ConceptMatch, RelatedConcept
from ..tools import MilvusSearchTool


class RelationshipReasonerAgent:
    """
    Agent that uses OMOP relationships to:
    1. Disambiguate concepts (choose most clinically appropriate)
    2. Validate coherence (check if concepts make sense together)
    3. Enrich with clinical rationale using relationship evidence
    """

    def __init__(self):
        # Use Milvus for concept lookups with relationship data
        self.search_tool = MilvusSearchTool()

        # Get agent configuration
        agent_config = get_agent_config("relationship_reasoner")

        self.agent = Agent(
            name=agent_config.get("name", "Relationship Reasoner"),
            model=create_model(agent_config),
            description=agent_config.get("description", "Clinical reasoning expert using OMOP relationships"),
            instructions=[
                "You are a clinical reasoning expert who validates and enriches concept selections",
                "for ATLAS concept sets using clinical knowledge and OMOP vocabulary principles.",
                "",
                "AVAILABLE RELATIONSHIP TYPES (52 total):",
                "",
                "1. HIERARCHY:",
                "   - 'Is a', 'Subsumes' → Navigate concept hierarchy",
                "",
                "2. TREATMENT & REGULATORY:",
                "   - 'May treat', 'May be treated by' → Drug-condition treatment",
                "   - 'May prevent', 'May be prevented by' → Prevention",
                "   - 'Has FDA indication', 'FDA indication of' → FDA-approved indications",
                "   - 'Has EMA indication', 'EMA indication of' → EMA-approved indications",
                "   - 'Has PMDA indication', 'PMDA indication of' → PMDA (Japan)",
                "   - 'Has NMPA indication', 'NMPA indication of' → NMPA (China)",
                "   - 'Has HC indication', 'HC indication of' → Health Canada",
                "",
                "3. CLINICAL FINDINGS & MANIFESTATIONS:",
                "   - 'Has complication' → Complications of a disease",
                "   - 'Has manifestation', 'Manifestation of' → Clinical manifestations",
                "   - 'Has asso finding', 'Asso finding of' → Associated findings",
                "   - 'Finding asso with', 'Asso with finding' → Findings associated with entity",
                "   - 'Has temp finding' → Temporal findings",
                "",
                "4. CAUSALITY & ETIOLOGY:",
                "   - 'Has causative agent', 'Causative agent of' → Causal relationships",
                "   - 'Has due to', 'Due to of' → Etiological relationships",
                "",
                "5. TEMPORAL:",
                "   - 'Occurs before', 'Occurs after' → Temporal sequence",
                "   - 'Has occurrence', 'Occurrence of' → Event occurrence",
                "",
                "6. ANATOMICAL & MORPHOLOGICAL:",
                "   - 'Has finding site', 'Finding site of' → Anatomical location",
                "   - 'Has asso morph', 'Asso morph of' → Associated morphology",
                "",
                "7. DIAGNOSTIC & INTERPRETIVE:",
                "   - 'Has interprets', 'Interprets of' → Lab test interprets finding",
                "",
                "8. CONTEXTUAL:",
                "   - 'Has finding context', 'Finding context of' → Clinical context",
                "   - 'Using finding method', 'Finding method of' → Diagnostic method",
                "   - 'Using finding inform', 'Finding inform of' → Finding informant",
                "   - 'Has asso proc', 'Asso proc of' → Associated procedure",
                "   - 'Has asso visit', 'Asso visit of' → Associated visit type",
                "",
                "YOUR TASKS:",
                "",
                "1. DISAMBIGUATION:",
                "   - When multiple concepts match an entity, use relationships to choose the best one",
                "   - Example: 'diabetic complication' → prefer concepts with 'Has complication' relationships",
                "   - Example: 'diabetic retinopathy' vs 'diabetic foot' → check 'Has finding site' (eye vs foot)",
                "",
                "2. COHERENCE VALIDATION:",
                "   - Check if concepts make clinical sense together",
                "   - Example: If drug + condition both present, verify drug has treatment relationship",
                "   - Example: If symptom + condition, verify symptom is manifestation of condition",
                "",
                "3. CLINICAL RATIONALE:",
                "   - Provide evidence-based reasoning using relationship data",
                "   - Example: 'Selected because it has FDA indication for [condition]'",
                "   - Example: 'Included because it is a manifestation of [primary condition]'",
                "",
                "4. QUALITY SCORING:",
                "   - Score each concept based on relationship evidence (0-1)",
                "   - Higher scores for concepts with strong clinical relationships to context",
                "   - Lower scores for concepts with weak or no relationships",
                "",
                "OUTPUT FORMAT:",
                "For each concept, provide:",
                "- Selected: Yes/No",
                "- Quality Score: 0-1 (based on relationship evidence)",
                "- Rationale: Evidence-based clinical reasoning",
                "- Coherence Warnings: Any issues detected (if applicable)",
            ],
            markdown=False,
        )

    def _compare_candidates_for_issues(
        self,
        entity: ClinicalEntity,
        candidate_concepts: List[ConceptMatch],
    ) -> str:
        """
        Compare top candidates side-by-side to detect common issues.

        Flags:
        - Laterality constraints (right/left when bilateral needed)
        - Parent concepts among candidates
        - Generic vs specific trade-offs

        Returns:
            Warning message if issues found, empty string otherwise
        """
        if len(candidate_concepts) < 2:
            return ""

        warnings = []

        # Check for laterality constraints (right/left/bilateral)
        laterality_terms = {
            'right': [],
            'left': [],
            'bilateral': [],
            'unilateral': [],
        }

        for concept in candidate_concepts[:10]:
            name_lower = concept.concept_name.lower()
            for term in laterality_terms:
                if term in name_lower:
                    laterality_terms[term].append(concept)

        # Warning: If we have right OR left but not bilateral option
        if (laterality_terms['right'] or laterality_terms['left']) and not laterality_terms['bilateral']:
            if laterality_terms['right'] and not laterality_terms['left']:
                warnings.append(
                    f"⚠️ LATERALITY: Only RIGHT-sided concepts found. "
                    f"Entity '{entity.text}' may need bilateral/non-lateralized concept. "
                    f"Right-sided concepts: {', '.join([f'[{c.concept_id}]' for c in laterality_terms['right'][:3]])}"
                )
            elif laterality_terms['left'] and not laterality_terms['right']:
                warnings.append(
                    f"⚠️ LATERALITY: Only LEFT-sided concepts found. "
                    f"Entity '{entity.text}' may need bilateral/non-lateralized concept. "
                    f"Left-sided concepts: {', '.join([f'[{c.concept_id}]' for c in laterality_terms['left'][:3]])}"
                )

        # Check for parent-child relationships among candidates
        concept_ids = {c.concept_id for c in candidate_concepts}
        parent_child_pairs = []

        for concept in candidate_concepts[:10]:
            if concept.parent_concept_id and concept.parent_concept_id in concept_ids:
                parent = next((c for c in candidate_concepts if c.concept_id == concept.parent_concept_id), None)
                if parent:
                    parent_child_pairs.append((parent, concept))

        if parent_child_pairs:
            warnings.append(
                f"ℹ️ HIERARCHY: Parent-child pairs detected in candidates. "
                f"Consider if broader or more specific concept is appropriate: "
                + ", ".join([f"[{p.concept_id}] {p.concept_name} → [{c.concept_id}] {c.concept_name}"
                           for p, c in parent_child_pairs[:2]])
            )

        return "\n".join(warnings) if warnings else ""

    def _apply_mandatory_filters(
        self,
        entity: ClinicalEntity,
        candidate_concepts: List[ConceptMatch],
    ) -> List[ConceptMatch]:
        """
        Apply mandatory rejection rules before LLM reasoning.

        This ensures inappropriate concepts are filtered out regardless of LLM decision.

        Args:
            entity: The entity we're finding concepts for
            candidate_concepts: Candidate concepts to filter

        Returns:
            Filtered list of concepts
        """
        filtered_concepts = []
        rejected_concepts = []

        # Define mandatory rejection rules based on entity requirements
        for concept in candidate_concepts:
            # Rule 1: Domain mismatch for non-exclusion inclusions
            if not entity.is_exclusion:
                if entity.domain == "Condition" and concept.domain_id not in {"Condition", "Observation"}:
                    rejected_concepts.append((concept, f"Domain mismatch: entity requires {entity.domain}, concept is {concept.domain_id}"))
                    continue

                # Rule 2: Forbidden concept classes for diagnosis sets
                forbidden_classes = {"Procedure", "Substance", "Answer", "Context-dependent"}
                if entity.entity_type in {"condition", "symptom", "observation"} and concept.concept_class_id in forbidden_classes:
                    rejected_concepts.append((concept, f"Forbidden concept class '{concept.concept_class_id}' for {entity.entity_type}"))
                    continue

                # Rule 3: Measurement domain concepts should not be in diagnosis sets (unless entity is measurement)
                if entity.entity_type in {"condition", "symptom"} and concept.domain_id == "Measurement":
                    rejected_concepts.append((concept, f"Measurement domain concept for {entity.entity_type} entity - should be in separate measurement phenotype"))
                    continue

            # Rule 4: UK Biobank survey answers are never appropriate
            if "biobank" in concept.vocabulary_id.lower() and concept.concept_class_id == "Answer":
                rejected_concepts.append((concept, "Survey answer value from UK Biobank - not appropriate for concept sets"))
                continue

            # Passed all mandatory filters
            filtered_concepts.append(concept)

        # Log rejections
        if rejected_concepts:
            print(f"\n🚫 Mandatory filters: Rejected {len(rejected_concepts)} concept(s) for '{entity.text}':")
            for concept, reason in rejected_concepts[:3]:
                print(f"   - [{concept.concept_id}] {concept.concept_name}: {reason}")
            if len(rejected_concepts) > 3:
                print(f"   ... and {len(rejected_concepts) - 3} more")

        return filtered_concepts

    def _select_best_candidates(
        self,
        entity: ClinicalEntity,
        candidate_concepts: List[ConceptMatch],
    ) -> List[ConceptMatch]:
        """
        Compare all candidates side-by-side and select best matches based on granularity/laterality.

        This is a pre-filtering step before relationship reasoning.

        Args:
            entity: The entity we're finding concepts for
            candidate_concepts: All candidates from vector search

        Returns:
            Top 2-3 best matching concepts
        """
        if len(candidate_concepts) <= 3:
            return candidate_concepts  # Already small, skip comparison

        # Format candidates for comparison
        candidates_text = "\n".join([
            f"{i+1}. [{c.concept_id}] {c.concept_name}\n"
            f"   Domain: {c.domain_id}, Class: {c.concept_class_id}, Similarity: {c.similarity_score:.3f}"
            for i, c in enumerate(candidate_concepts)
        ])

        prompt = f"""
Compare these {len(candidate_concepts)} concept candidates and select the 2-3 BEST matches for the entity.

TARGET ENTITY: "{entity.text}"
Entity Type: {entity.entity_type}
Is Exclusion: {entity.is_exclusion}

CANDIDATES:
{candidates_text}

SELECTION CRITERIA:

1. COMPOSITE ENTITIES: If entity lists multiple options with "or", "and", comma-separated:
   - FIRST: Check if a PARENT concept covers all mentioned parts
   - If parent exists → select parent only (with descendants=true it will cover all)
   - If no parent → select concepts for EACH mentioned entity
   - Example: "kidney or liver disease" → if no parent, select BOTH kidney AND liver
   - Example: "A, B, or C cancer" → if "Malignant neoplasm of A/B/C system" exists as parent, select parent

2. LATERALITY: If entity doesn't specify side (right/left), REJECT lateralized concepts

3. GRANULARITY: Match the specificity level of the entity
   - Generic entity → generic concept
   - Specific entity → specific concept

4. TEMPORAL: If entity doesn't specify timing (acute/chronic), prefer non-temporal concepts

5. SEMANTIC: Prioritize similarity scores - higher scores indicate better matches

OUTPUT FORMAT:
List 2-5 concept IDs that best match the entity. One ID per line.

EXAMPLE 1 (Simple):
Entity: "heart failure"
Candidates: "Heart failure", "Acute heart failure", "CHF of right ventricle"
Selection:
316139
319835
(Reasoning: Selected generic and moderately specific. Rejected overly constrained concepts)

EXAMPLE 2 (Composite - select parent if available):
Entity: "kidney or liver disease"
Candidates: "Kidney disease", "Liver disease", "Organ dysfunction", "Acute kidney injury"
Selection:
11111
(Reasoning: "Organ dysfunction" is parent covering both kidney and liver. Selected parent only since descendants=true covers both)

EXAMPLE 3 (Composite - select multiple if no parent):
Entity: "kidney or liver disease"
Candidates: "Kidney disease", "Liver disease", "Acute kidney injury", "Kidney of left side"
Selection:
12345
67890
(Reasoning: No parent concept available. Entity mentions both organs, so selected both. Rejected overly specific)

YOUR SELECTION (just IDs, one per line):
"""

        response = self.agent.run(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse selected IDs
        import re
        selected_ids = [int(id) for id in re.findall(r'\b(\d{6,})\b', response_text)]

        # Return matching concepts (fallback to top 3 if parsing fails)
        if selected_ids:
            selected = [c for c in candidate_concepts if c.concept_id in selected_ids[:3]]
            if selected:
                print(f"   🎯 Pre-selected {len(selected)} candidates: {', '.join([f'[{c.concept_id}]' for c in selected])}")
                return selected

        # Fallback: return top 3 by similarity
        return candidate_concepts[:3]

    def reason_about_concepts(
        self,
        entity: ClinicalEntity,
        candidate_concepts: List[ConceptMatch],
        all_entities: List[ClinicalEntity],
        max_refinement_iterations: int = 2,
    ) -> List[ConceptMatch]:
        """
        Use relationships to enrich and disambiguate concepts.

        Args:
            entity: The entity we're finding concepts for
            candidate_concepts: Initial concept matches from vector search
            all_entities: All entities from clinical description (for context)
            max_refinement_iterations: Maximum number of search refinements

        Returns:
            Enriched concepts with relationship reasoning and quality scores
        """
        if not candidate_concepts:
            return []

        # Apply mandatory filters first (hard rejections)
        filtered_candidates = self._apply_mandatory_filters(entity, candidate_concepts)

        if not filtered_candidates:
            print(f"⚠️  All candidates rejected by mandatory filters for '{entity.text}'")
            return []

        # NEW: Pre-select best candidates based on granularity/laterality
        if len(filtered_candidates) > 3:
            filtered_candidates = self._select_best_candidates(entity, filtered_candidates)

        iteration = 0
        current_candidates = filtered_candidates

        while iteration < max_refinement_iterations:
            # Compare candidates side-by-side for common issues
            comparison_warnings = self._compare_candidates_for_issues(entity, current_candidates)

            # Analyze relationship patterns including hierarchy
            relationship_summary = self._analyze_relationships(current_candidates)
            hierarchy_analysis = self._analyze_hierarchy(current_candidates)

            # Check cross-entity coherence
            coherence_check = self._check_coherence(entity, current_candidates, all_entities)

            # Format for LLM review
            concepts_text = self._format_concepts_with_relationships(current_candidates)
            entities_text = ", ".join([f"'{e.text}' ({e.entity_type})" for e in all_entities])

            # LLM reasoning prompt
            prompt = f"""
Analyze these concept candidates using OMOP relationship data to select the best match(es).

TARGET ENTITY: "{entity.text}"
Entity Type: {entity.entity_type}
Domain: {entity.domain}
Requires Descendants: {entity.requires_descendants}
Is Exclusion: {entity.is_exclusion}
Temporal Constraint: {entity.temporal_constraint or 'None'}
Relationship to Primary: {entity.relationship_to_primary or 'None'}

CONTEXT FROM CLINICAL DESCRIPTION:
All entities: {entities_text}

CANDIDATE CONCEPTS WITH RELATIONSHIPS:
{concepts_text}

{"⚠️ COMPARISON WARNINGS:\n" + comparison_warnings + "\n" if comparison_warnings else ""}
RELATIONSHIP ANALYSIS:
{relationship_summary}

HIERARCHY ANALYSIS:
{hierarchy_analysis}

COHERENCE CHECK:
{coherence_check}

INSTRUCTIONS:

1. EVALUATE SPECIFICITY & CONTEXT using hierarchical relationships:
   - Check "Is a" relationships to understand what category each concept belongs to
   - Check "Subsumes" relationships to see if there are more specific child concepts
   - Reject overly generic concepts (e.g., "True positive" for "ANA positive")
   - Reject overly specific concepts if a broader term is more appropriate

2. EVALUATE GRANULARITY:
   - Is the concept at the right level of detail for "{entity.text}"?
   - If too broad: Should we search for more specific child concepts?
   - If too narrow: Should we search for broader parent concepts?
   - If wrong category: Should we reject and search for different concepts?

3. CHECK SPECIFICITY CONSTRAINTS (if warnings present):
   - Review comparison warnings about laterality, temporality, or other constraints
   - If entity is general but concept is overly constrained: REJECT or REFINE
   - If warnings suggest missing better options: Consider requesting REFINE

4. EVALUATE CLINICAL RELEVANCE:
   - Does the concept make clinical sense in the context?
   - Use treatment, manifestation, causality relationships as evidence
   - Prefer concepts with rich relationship data over isolated concepts

5. MAKE A DECISION:
   You have THREE options:

   A) ACCEPT: Select one or more concepts that match well
   B) REFINE: Request a new search with better query terms or for specific ancestor/child concepts
   C) REJECT: Reject all candidates if none are appropriate

For each candidate, provide:
- Concept ID: [concept_id]
- Decision: Accept/Reject
- Quality Score: 0.X (0.4 semantic + 0.3 relationships + 0.3 specificity)
- Rationale: [Clinical reasoning with hierarchy and relationship evidence]
- Warnings: [Issues or 'None']

Then provide an OVERALL DECISION:
OVERALL: [ACCEPT/REFINE/REJECT]
If REFINE: Suggest new search query or specific concept IDs to explore
If ACCEPT: List the accepted concept IDs

EXAMPLES:

Example 1 (Generic concept rejection):
- Concept ID: 4189525
- Decision: Reject
- Quality Score: 0.30
- Rationale: "True positive" is too generic for "ANA positive". Parent concepts include "Clinical finding" which is extremely broad. Lacks specificity.
- Warnings: Overly generic

- Concept ID: 4163958
- Decision: Accept
- Quality Score: 0.85
- Rationale: "Antinuclear antibody" is specific and clinically relevant. Parent concepts include "Laboratory test finding". Good semantic match.
- Warnings: None

OVERALL: ACCEPT
Accepted IDs: 4163958

Example 2 (Wrong granularity - too broad):
- Concept ID: 201820
- Decision: Reject
- Quality Score: 0.40
- Rationale: "Diabetes mellitus" is too broad for "type 2 diabetes". Should explore child concepts using "Subsumes" relationships.
- Warnings: Too broad

OVERALL: REFINE
Suggested action: Search for child concepts of 201820, specifically looking for "type 2 diabetes mellitus"

Example 3 (Wrong granularity - too specific):
- Concept ID: 443238
- Decision: Reject
- Quality Score: 0.45
- Rationale: "Diabetic retinopathy due to type 2 diabetes mellitus" is too specific when query is just "diabetic retinopathy". Parent concept "Diabetic retinopathy" is more appropriate.
- Warnings: Too specific

OVERALL: REFINE
Suggested action: Search for parent concepts or use query "diabetic retinopathy" without type specification

Example 4 (Wrong category):
- Concept ID: 40213154
- Decision: Reject
- Quality Score: 0.20
- Rationale: This is a drug concept but entity is a condition. Parent relationships show pharmaceutical category. Wrong domain.
- Warnings: Wrong category/domain

OVERALL: REJECT
Reason: All candidates are in wrong domain

Example 5 (Multiple good matches):
- Concept ID: 316139
- Decision: Accept
- Quality Score: 0.90
- Rationale: "Heart failure" is a good match. Has rich relationships including "May be treated by" for relevant medications.
- Warnings: None

- Concept ID: 319835
- Decision: Accept
- Quality Score: 0.88
- Rationale: "Congestive heart failure" is also appropriate. Slightly more specific. "Is a" relationship to Heart failure.
- Warnings: Consider if specificity is needed

OVERALL: ACCEPT
Accepted IDs: 316139, 319835
Note: Both are valid; orchestrator can decide which or both to include based on descendant needs.

Example 6 (Laterality specificity issue):
Entity: "pneumonia" (no laterality specified)
- Concept ID: 12345
- Decision: Reject
- Quality Score: 0.45
- Rationale: "Pneumonia of right lung" is overly constrained with laterality. Entity doesn't specify side, so bilateral or non-lateralized concept is more appropriate.
- Warnings: Laterality constraint detected

- Concept ID: 67890
- Decision: Accept
- Quality Score: 0.90
- Rationale: "Pneumonia" without laterality constraint. Captures all cases regardless of side. More appropriate for general entity.
- Warnings: None

OVERALL: ACCEPT
Accepted IDs: 67890
"""

            # Get LLM reasoning
            response = self.agent.run(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse the decision
            decision = self._parse_reasoning_decision(response_text, current_candidates)

            if decision['action'] == 'ACCEPT':
                return decision['selected_concepts']
            elif decision['action'] == 'REFINE' and iteration < max_refinement_iterations - 1:
                # Perform refinement search
                print(f"  🔄 Refining search: {decision['refinement_reason']}")
                refined_candidates = self._perform_refinement_search(
                    entity=entity,
                    refinement_suggestion=decision['refinement_suggestion'],
                    original_candidates=current_candidates,
                )
                if refined_candidates:
                    current_candidates = refined_candidates
                    iteration += 1
                    continue
                else:
                    # Refinement failed, return best from original
                    print(f"  ⚠️  Refinement search found no results, using original candidates")
                    return self._parse_llm_selections(response_text, current_candidates)
            else:
                # REJECT or max iterations reached
                return self._parse_llm_selections(response_text, current_candidates)

        # Max iterations reached
        return self._parse_llm_selections(response_text, current_candidates)

    def _analyze_hierarchy(self, concepts: List[ConceptMatch]) -> str:
        """Analyze hierarchical relationships (Is a, Subsumes) to understand concept context."""
        hierarchy_notes = []

        for c in concepts:
            if not c.relationship_types:
                continue

            # Check for hierarchical relationships
            is_a_rels = [r for r in c.relationship_types if r == 'Is a']
            subsumes_rels = [r for r in c.relationship_types if r == 'Subsumes']

            if is_a_rels:
                hierarchy_notes.append(
                    f"[{c.concept_id}] {c.concept_name}: Has {len(is_a_rels)} parent concept(s) via 'Is a' - "
                    f"belongs to broader category"
                )

            if subsumes_rels:
                hierarchy_notes.append(
                    f"[{c.concept_id}] {c.concept_name}: Has {len(subsumes_rels)} child concept(s) via 'Subsumes' - "
                    f"more specific variants exist"
                )

            # Check if concept seems too generic (has many children but few parents)
            if len(subsumes_rels) > 10 and len(is_a_rels) < 3:
                hierarchy_notes.append(
                    f"  ⚠️  [{c.concept_id}] may be too generic (many children, few parents)"
                )

            # Check if concept seems too specific (has many parents but no children)
            if len(is_a_rels) > 5 and len(subsumes_rels) == 0:
                hierarchy_notes.append(
                    f"  ℹ️  [{c.concept_id}] is highly specific (many parents, no children)"
                )

        if not hierarchy_notes:
            return "No significant hierarchical relationships found in candidates."

        return "\n".join(hierarchy_notes)

    def _parse_reasoning_decision(
        self,
        response_text: str,
        candidate_concepts: List[ConceptMatch],
    ) -> dict:
        """
        Parse LLM decision: ACCEPT, REFINE, or REJECT.

        Returns:
            dict with keys: action, selected_concepts, refinement_suggestion, refinement_reason
        """
        import re

        # Look for OVERALL decision
        overall_match = re.search(r'OVERALL:\s*(ACCEPT|REFINE|REJECT)', response_text, re.IGNORECASE)

        if not overall_match:
            # Default to parsing selections if no clear decision
            selected = self._parse_llm_selections(response_text, candidate_concepts)
            return {
                'action': 'ACCEPT',
                'selected_concepts': selected,
                'refinement_suggestion': None,
                'refinement_reason': None,
            }

        action = overall_match.group(1).upper()

        if action == 'ACCEPT':
            # Parse accepted IDs
            accepted_ids_match = re.search(r'Accepted IDs?:\s*([\d,\s]+)', response_text)
            if accepted_ids_match:
                id_str = accepted_ids_match.group(1)
                accepted_ids = [int(id.strip()) for id in re.findall(r'\d{6,}', id_str)]
                selected = [c for c in candidate_concepts if c.concept_id in accepted_ids]
            else:
                # Fall back to parsing selections
                selected = self._parse_llm_selections(response_text, candidate_concepts)

            return {
                'action': 'ACCEPT',
                'selected_concepts': selected if selected else [candidate_concepts[0]],
                'refinement_suggestion': None,
                'refinement_reason': None,
            }

        elif action == 'REFINE':
            # Extract refinement suggestion
            suggestion_match = re.search(r'Suggested action:\s*(.+?)(?:\n|$)', response_text)
            suggestion = suggestion_match.group(1).strip() if suggestion_match else response_text

            # Extract reason
            reason_match = re.search(r'REFINE\s*\n(.+?)(?:Suggested action|$)', response_text, re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "Refinement needed"

            return {
                'action': 'REFINE',
                'selected_concepts': [],
                'refinement_suggestion': suggestion,
                'refinement_reason': reason,
            }

        else:  # REJECT
            return {
                'action': 'REJECT',
                'selected_concepts': [],
                'refinement_suggestion': None,
                'refinement_reason': 'All candidates rejected',
            }

    def _perform_refinement_search(
        self,
        entity: ClinicalEntity,
        refinement_suggestion: str,
        original_candidates: List[ConceptMatch],
    ) -> List[ConceptMatch]:
        """
        Perform a refined search based on LLM suggestion.

        The suggestion might be:
        - A new search query
        - A request to explore child/parent concepts
        - A specific concept ID to explore
        """
        import re

        # Check if suggestion mentions specific concept IDs
        concept_id_matches = re.findall(r'\b(\d{6,})\b', refinement_suggestion)

        if concept_id_matches:
            # LLM wants to explore specific concepts or their children/parents
            concept_id = int(concept_id_matches[0])

            # Check if we should look for children or parents
            if 'child' in refinement_suggestion.lower() or 'more specific' in refinement_suggestion.lower():
                # Search for more specific terms based on the concept name
                parent_concept = next((c for c in original_candidates if c.concept_id == concept_id), None)
                if parent_concept:
                    # Search with more specific query
                    refined_query = f"{entity.text} specific"
                    return self.search_tool.search_concepts(
                        query_text=refined_query,
                        domain_filter=entity.domain,
                        top_k=10,
                        min_similarity=0.5,
                    )

            elif 'parent' in refinement_suggestion.lower() or 'broader' in refinement_suggestion.lower():
                # Search for broader terms
                child_concept = next((c for c in original_candidates if c.concept_id == concept_id), None)
                if child_concept:
                    # Extract the core term without qualifiers
                    simplified_query = re.sub(r'\s+(due to|with|of|in)\s+.+', '', entity.text)
                    return self.search_tool.search_concepts(
                        query_text=simplified_query,
                        domain_filter=entity.domain,
                        top_k=10,
                        min_similarity=0.5,
                    )

        # Check if suggestion contains a new search query in quotes
        query_match = re.search(r'"([^"]+)"', refinement_suggestion)
        if query_match:
            new_query = query_match.group(1)
            return self.search_tool.search_concepts(
                query_text=new_query,
                domain_filter=entity.domain,
                top_k=10,
                min_similarity=0.5,
            )

        # Default: try the suggestion text itself as a query
        return self.search_tool.search_concepts(
            query_text=refinement_suggestion[:100],  # Limit length
            domain_filter=entity.domain,
            top_k=10,
            min_similarity=0.5,
        )

    def _analyze_relationships(self, concepts: List[ConceptMatch]) -> str:
        """Summarize relationship patterns in candidates."""
        all_rels = set()
        for c in concepts:
            all_rels.update(c.relationship_types)

        if not all_rels:
            return "No relationship data available for candidates."

        summary = f"Found {len(all_rels)} relationship types across {len(concepts)} concepts:\n"

        # Categorize relationships
        treatment_rels = [r for r in all_rels if any(
            keyword in r.lower() for keyword in ['treat', 'indication', 'prevent']
        )]
        clinical_rels = [r for r in all_rels if any(
            keyword in r.lower() for keyword in ['manifestation', 'complication', 'finding', 'asso']
        )]
        causal_rels = [r for r in all_rels if any(
            keyword in r.lower() for keyword in ['causative', 'due to']
        )]
        anatomical_rels = [r for r in all_rels if any(
            keyword in r.lower() for keyword in ['site', 'morph']
        )]
        hierarchy_rels = [r for r in all_rels if r in ['Is a', 'Subsumes']]

        if treatment_rels:
            summary += f"  Treatment/Prevention: {', '.join(treatment_rels[:5])}\n"
        if clinical_rels:
            summary += f"  Clinical Findings: {', '.join(clinical_rels[:5])}\n"
        if causal_rels:
            summary += f"  Causality: {', '.join(causal_rels[:3])}\n"
        if anatomical_rels:
            summary += f"  Anatomical: {', '.join(anatomical_rels[:3])}\n"
        if hierarchy_rels:
            summary += f"  Hierarchy: {', '.join(hierarchy_rels)}\n"

        return summary

    def _check_coherence(
        self,
        entity: ClinicalEntity,
        concepts: List[ConceptMatch],
        all_entities: List[ClinicalEntity],
    ) -> str:
        """Check if concepts are coherent with other entities in description."""
        coherence_notes = []

        # Extract entity types from description
        condition_entities = [e for e in all_entities if e.entity_type == "condition"]
        drug_entities = [e for e in all_entities if e.entity_type == "drug"]
        symptom_entities = [e for e in all_entities if e.entity_type == "symptom"]

        # Check drug-condition coherence
        if entity.entity_type == "drug" and condition_entities:
            condition_names = [c.text for c in condition_entities]
            coherence_notes.append(
                f"Drug entity should have treatment relationship (May treat, FDA indication, etc.) "
                f"to conditions: {', '.join(condition_names)}"
            )

            # Check if any candidate has treatment relationships
            has_treatment_rels = any(
                any(rel in c.relationship_types for rel in [
                    'May treat', 'May be treated by', 'Has FDA indication', 'FDA indication of',
                    'Has EMA indication', 'EMA indication of'
                ])
                for c in concepts
            )
            if has_treatment_rels:
                coherence_notes.append("✓ Treatment relationships found in candidates")

        # Check symptom-condition coherence
        if entity.entity_type == "symptom" and condition_entities:
            condition_names = [c.text for c in condition_entities]
            coherence_notes.append(
                f"Symptom should have manifestation relationship to conditions: {', '.join(condition_names)}"
            )

            # Check for manifestation relationships
            has_manifestation = any(
                any(rel in c.relationship_types for rel in [
                    'Manifestation of', 'Has manifestation', 'Asso finding of', 'Finding asso with'
                ])
                for c in concepts
            )
            if has_manifestation:
                coherence_notes.append("✓ Manifestation relationships found in candidates")

        # Check procedure-condition coherence
        if entity.entity_type == "procedure" and condition_entities:
            coherence_notes.append(
                f"Procedure may have diagnostic/therapeutic relationship to conditions"
            )

        if not coherence_notes:
            return "No specific coherence checks needed for this entity type."

        return "\n".join(coherence_notes)

    def _format_concepts_with_relationships(self, concepts: List[ConceptMatch]) -> str:
        """Format concepts showing their relationships."""
        lines = []
        for i, c in enumerate(concepts, 1):
            lines.append(
                f"{i}. [{c.concept_id}] {c.concept_name}"
            )
            lines.append(f"   Domain: {c.domain_id}, Standard: {c.standard_concept}, Similarity: {c.similarity_score:.3f}")

            if c.parent_concept_id:
                lines.append(f"   Parent Concept: {c.parent_concept_id}")

            if c.relationship_types:
                # Show up to 10 relationships
                rel_display = c.relationship_types[:10]
                if len(c.relationship_types) > 10:
                    rel_display.append(f"... and {len(c.relationship_types) - 10} more")
                lines.append(f"   Relationships ({len(c.relationship_types)}): {', '.join(rel_display)}")
            else:
                lines.append("   Relationships: None")

            lines.append("")  # Blank line

        return "\n".join(lines)

    def _parse_llm_selections(
        self,
        response_text: str,
        candidate_concepts: List[ConceptMatch],
    ) -> List[ConceptMatch]:
        """
        Parse LLM response and enrich selected concepts.

        Extracts concept IDs marked as "Selected: Yes" and updates their rationale.
        """
        import re

        # Simple parsing - look for concept IDs marked as selected
        selected_concepts = []

        # Split response into blocks (one per concept)
        blocks = response_text.split("Concept ID:")

        for block in blocks[1:]:  # Skip first empty block
            lines = block.strip().split('\n')

            # Extract concept ID
            concept_id_match = re.search(r'(\d{6,})', lines[0])
            if not concept_id_match:
                continue

            concept_id = int(concept_id_match.group(1))

            # Check if selected
            selected = any('selected: yes' in line.lower() for line in lines)
            if not selected:
                continue

            # Extract quality score
            quality_score = None
            for line in lines:
                score_match = re.search(r'quality score:\s*([0-9.]+)', line.lower())
                if score_match:
                    quality_score = float(score_match.group(1))
                    break

            # Extract rationale
            rationale = None
            for line in lines:
                if 'rationale:' in line.lower():
                    rationale = line.split(':', 1)[1].strip()
                    break

            # Find matching concept
            matching_concept = next(
                (c for c in candidate_concepts if c.concept_id == concept_id),
                None
            )

            if matching_concept:
                # Update concept with enriched rationale
                if rationale:
                    matching_concept.matched_entity = f"{matching_concept.matched_entity} (relationship-validated)"

                selected_concepts.append(matching_concept)

        # If LLM didn't select anything or parsing failed, return top candidate
        if not selected_concepts and candidate_concepts:
            return [candidate_concepts[0]]

        return selected_concepts
