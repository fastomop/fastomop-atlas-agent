"""Test ATLAS agent with SLE challenge case from Mind Meets Machines."""
from src.atlas_agent.agents import OrchestratorAgent
import json

# Initialize orchestrator
print("🚀 Initializing ATLAS agent for SLE challenge...")
orchestrator = OrchestratorAgent()

# SLE Challenge Case from Mind Meets Machines C01
# Source: https://raw.githubusercontent.com/ohdsi-studies/MindMeetsMachines/main/C01/Systemic%20Lupus%20Erythematous%20(SLE).md
clinical_description = """
# C01 Systemic Lupus Erythematosus

Build a concept set for Systemic Lupus Erythematosus — systemic form that captures current, clinically active disease for use in a phenotype for clinically active, prevalent SLE supporting comparative-effectiveness work. The concept set must reflect explicit systemic SLE diagnoses (including "SLE with organ involvement"). The motivating question is generalized as: among patients with active systemic SLE, what are 12-month outcomes (e.g., steroid burden, inadequate response) after initiating Drug A versus Drug B?

### Clinical Case Definition

A chronic, systemic autoimmune disease characterized by multisystem involvement and immunologic abnormalities consistent with SLE (e.g., anti-dsDNA and/or anti-Sm antibodies, hypocomplementemia). In routine care, SLE is managed primarily by rheumatology and treated with antimalarials, immunosuppressants, biologics, and judicious glucocorticoids.

### Diagnostic Criteria (for context; not encoded in this concept set)

Serology: ANA (entry), anti-dsDNA and/or anti-Sm positivity; low C3/C4. Organ involvement: renal (proteinuria, biopsy-proven LN), hematologic, mucocutaneous, musculoskeletal, neuropsychiatric, serositis. Classification frameworks (e.g., 2019 EULAR/ACR) inform measurement concept sets but are not used to gate diagnosis codes here.

### Presentation & Course

Relapsing-remitting with flares and remissions; severity ranges from mild mucocutaneous/arthralgia to life-threatening organ disease (e.g., nephritis). Long-term morbidity is strongly influenced by cumulative glucocorticoid exposure.

### Common Treatments/Management

Hydroxychloroquine, azathioprine, mycophenolate, methotrexate, cyclophosphamide; calcineurin inhibitors (e.g., tacrolimus; voclosporin for LN); biologics (belimumab, anifrolumab); off-label rituximab; systemic glucocorticoids (oral and pulse). These support phenotype confirmation but are not part of this diagnosis concept set.

### Clinical Scope and Granularity

* **Disease entity:** Chronic, systemic autoimmune disease with a relapsing–remitting course; typical features include mucocutaneous, musculoskeletal, hematologic, renal, neuropsychiatric, and serosal involvement, with immunologic abnormalities consistent with SLE.
* **Temporality:** The phenotype should identify patients with currently active systemic SLE. Both newly diagnosed (incident) and existing (prevalent) cases are in scope, provided there is evidence of disease activity. Historical disease in remission is out of scope.
* **Severity & acuity:** All severities —from mild to life-threatening organ disease—are in scope when the diagnosis explicitly denotes active systemic SLE.
* **Manifestations:** Organ/system involvement that is linked to SLE.
* **Etiology:** Primary systemic SLE only; other etiologies (for example, drug-induced) are not within the scope.
* **Population:** Adult (18 and above).

### Related, differential or comorbid conditions that are not sufficient for inclusion:

* Cutaneous lupus erythematosus (discoid, subacute cutaneous) without systemic SLE.
* Drug-induced lupus; neonatal lupus.
* Antiphospholipid syndrome without SLE.
* Undifferentiated or mixed connective tissue disease; systemic sclerosis; dermatomyositis/polymyositis; Sjögren's syndrome; rheumatoid arthritis.
* Organ-specific diagnoses (e.g., nephritis, serositis, cytopenias, CNS vasculitis) without explicit SLE linkage.
* Non-SLE "lupus" such as lupus vulgaris.
* Antiphospholipid syndrome (APS) without SLE.
* Fibromyalgia.

### Synonyms

* Systemic lupus erythematosus
* SLE
* Systemic lupus
* Disseminated lupus erythematosus (historic)
* SLE with organ involvement (e.g., "SLE with nephritis")
"""

print(f"\n{'='*80}")
print("TESTING: Systemic Lupus Erythematosus Concept Set")
print(f"{'='*80}\n")
print("Challenge: Create concept set for clinically active systemic SLE")
print("- Must capture systemic form with organ involvement")
print("- Must exclude cutaneous-only lupus")
print("- Must exclude drug-induced and other non-systemic forms")
print("- Must exclude related autoimmune conditions")
print(f"\n{'='*80}\n")

try:
    # Create concept set
    print("⏳ Running orchestrator agent (this may take 2-3 minutes)...\n")

    concept_set, atlas_json = orchestrator.create_concept_set(
        clinical_description=clinical_description,
        validate=True,
        export_path="output/sle_concept_set.json",
    )

    # Print detailed explanation
    print("\n" + "="*80)
    print("CONCEPT SET EXPLANATION")
    print("="*80 + "\n")
    print(orchestrator.explain_concept_set(concept_set))

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total concepts included: {len(concept_set.items)}")

    # Group by domain
    domain_counts = {}
    for item in concept_set.items:
        domain = item.concept.domain_id
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print("\nConcepts by domain:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")

    # Show some example concepts
    print(f"\n{'='*80}")
    print("EXAMPLE INCLUDED CONCEPTS (first 10)")
    print(f"{'='*80}")
    for i, item in enumerate(concept_set.items[:10], 1):
        include_desc = "+" if not item.is_excluded else "-"
        descendants = f" [+descendants]" if item.include_descendants else ""
        print(f"{i:2d}. {include_desc} [{item.concept.concept_id}] {item.concept.concept_name}{descendants}")
        print(f"     Domain: {item.concept.domain_id}, Vocab: {item.concept.vocabulary_id}")

    if len(concept_set.items) > 10:
        print(f"\n... and {len(concept_set.items) - 10} more concepts")

    # Check for exclusions
    exclusions = [item for item in concept_set.items if item.is_excluded]
    if exclusions:
        print(f"\n{'='*80}")
        print(f"EXCLUDED CONCEPTS ({len(exclusions)} total)")
        print(f"{'='*80}")
        for i, item in enumerate(exclusions[:5], 1):
            print(f"{i}. [-] [{item.concept.concept_id}] {item.concept.concept_name}")
            print(f"     Reason: {item.rationale}")
        if len(exclusions) > 5:
            print(f"\n... and {len(exclusions) - 5} more exclusions")

    # Show ATLAS JSON structure
    print(f"\n{'='*80}")
    print("ATLAS JSON PREVIEW (first item)")
    print(f"{'='*80}")
    if atlas_json['items']:
        print(json.dumps(atlas_json['items'][0], indent=2))

    print(f"\n{'='*80}")
    print("✅ SLE CHALLENGE TEST COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"📁 Exported to: output/sle_concept_set.json")
    print(f"📊 Total concepts: {len(concept_set.concepts)}")
    print(f"🎯 Ready for ATLAS import and phenotype validation")

except Exception as e:
    print(f"\n{'='*80}")
    print("❌ TEST FAILED")
    print(f"{'='*80}")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
