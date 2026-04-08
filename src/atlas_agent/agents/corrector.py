"""Corrector Agent - Attempts to fix a failed concept set based on validation notes."""
from agno.agent import Agent

from ..config import get_agent_config
from ..model_factory import create_model
from ..models import ConceptSet, ParsedClinicalDescription, ConceptSetItem
import json

class CorrectorAgent:
    """
    Agent that attempts to correct a concept set based on validation feedback.
    """

    def __init__(self):
        # Get agent configuration
        agent_config = get_agent_config("corrector")

        self.agent = Agent(
            name=agent_config.get("name", "Concept Set Corrector"),
            model=create_model(agent_config),
            description=agent_config.get("description", "Clinical informatics expert who corrects OMOP concept sets based on validation feedback."),
            instructions=[
                "You are a clinical informatics expert tasked with correcting an OMOP concept set that has failed validation.",
                "You will be given the original clinical description, the list of concepts, and the validation notes.",
                "Your goal is to fix the issues while preserving the original clinical intent.",
                "You can add, remove, or modify items in the concept set.",
                "Focus on addressing the specific validation notes.",
                "Output ONLY the corrected concept set items as a valid JSON list of objects. Do not include any other text or explanation.",
            ],
            markdown=True,  # Expecting a JSON code block
        )

    def correct_concept_set(self, concept_set: ConceptSet, parsed_description: ParsedClinicalDescription) -> ConceptSet:
        """
        Attempts to correct a concept set.

        Args:
            concept_set: The failed ConceptSet with validation_notes.
            parsed_description: The original parsed clinical description.

        Returns:
            A new, corrected ConceptSet.
        """
        print("   correctional logic...")

        # Convert items to proper JSON array
        import json
        items_list = [item.model_dump() for item in concept_set.items]
        items_json = json.dumps(items_list, indent=2)
        validation_notes_str = "\n".join(concept_set.validation_notes)

        prompt = f"""
The following concept set for the description "{concept_set.description}" failed validation.
Validation Notes:
{validation_notes_str}

Here is the original list of concept set items (JSON format):
{items_json}

Please correct the list of items to address the validation notes and output the full, corrected list as a JSON array.
Your output should be a single JSON code block containing the list of corrected items.
Each item must have this exact structure:
{{
  "concept": {{ ConceptMatch object with all required fields }},
  "include_descendants": true/false,
  "is_excluded": false,
  "include_mapped": false,
  "rationale": "Clinical reasoning..."
}}
"""

        response = self.agent.run(prompt)

        try:
            # Extract JSON from the response
            json_str = response.content.strip().lstrip("```json").rstrip("```").strip()
            corrected_items_dict = json.loads(json_str)

            # Re-create ConceptSetItems from the corrected dictionary
            corrected_items = [ConceptSetItem(**item_dict) for item_dict in corrected_items_dict]
            
            # Create a new concept set with the corrected items
            new_concept_set = ConceptSet(
                name=concept_set.name + " (Corrected)",
                description=concept_set.description,
                items=corrected_items,
            )
            print(f"  ✓ Corrector Agent proposed {len(corrected_items)} corrected items.")
            return new_concept_set

        except (json.JSONDecodeError, TypeError, Exception) as e:
            print(f"  ⚠️ Corrector Agent failed to produce valid JSON: {e}")
            print(f"  Response preview: {str(response.content)[:500]}")
            # Return the original concept set if correction fails
            return concept_set
