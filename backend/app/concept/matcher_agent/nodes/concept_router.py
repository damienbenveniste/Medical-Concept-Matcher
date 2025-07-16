from pydantic import BaseModel, Field
from typing import Literal
from app.clients import async_openai_client
from app.indexing.indexer import Indexer, Datasets
from app.concept.matcher_agent.state import MatcherAgentState
import logging


SYSTEM_PROMPT = """
You are a clinical-coding routing assistant.

**Task**
Given a single free-text *concept name*, decide which one of the following
high-level code families it most likely belongs to, or return **"none"** if no
reasonable match exists.

  • diagnosis   → ICD-10/ICD-10-CM (diseases, conditions)
  • procedure   → CPT (medical, surgical, radiology, lab procedures)
  • measurement → LOINC (laboratory tests, survey items, vital signs)
  • drug        → RxNorm / ATC (medications, drug classes)
  • none        → concept does not map to any of the above

False-positives are worse than false-negatives: if you are uncertain,
choose **"none"**.

**Instructions**

1. Read the `concept_name` provided in the USER message.
2. Compare against the vocabulary styles shown below and your medical
   knowledge.
3. Choose the **single best** `category`.
4. Return **only** a JSON object that conforms to the schema below.
   No extra keys, no markdown, no comments.

**Grounding context (abbreviated)**
Below are small samples from each vocabulary to remind you of typical
terminology. *They are **examples only**—do not limit yourself to this subset.*
"""


class ConceptRouting(BaseModel):
    concept_type: Literal['diagnose', 'procedure', 'measurement', 'drug'] | None = Field(
        default=None,
        description=(
            "High-level vocabulary family assigned to the concept: "
            "'diagnose' → ICD-10 diagnosis; "
            "'procedure' → CPT procedure; "
            "'measurement' → LOINC observable; "
            "'drug' → RxNorm/ATC medication or drug class. "
            "Use None when the concept does not belong to any of these four groups."
        )
    )


class ConceptRouter:
    """Clinical concept routing agent that determines the appropriate vocabulary type for a given concept."""

    async def route(self, state: MatcherAgentState) -> ConceptRouting:
        """Determine the appropriate vocabulary type for a medical concept.
        
        Args:
            state: Current state containing the concept to route.
            
        Returns:
            ConceptRouting object containing the determined concept type.
            
        Raises:
            ConnectionError: If OpenAI API call fails.
        """
        icd_samples = await self.get_samples(state, Datasets.ICD)
        atc_samples = await self.get_samples(state, Datasets.ATC)
        loinc_samples = await self.get_samples(state, Datasets.LOINC)
        cpt_samples = await self.get_samples(state, Datasets.CPT)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  ICD-10 samples  ###\n\n{"\n".join(icd_samples)}"},
            {"role": "system", "content": f"###  ATC samples  ###\n\n{"\n".join(atc_samples)}"},
            {"role": "system", "content": f"###  CPT samples  ###\n\n{"\n".join(cpt_samples)}"},
            {"role": "system", "content": f"###  LOINC samples  ###\n\n{"\n".join(loinc_samples)}"},
            {"role": "user", "content": f"concept: {state.concept}"},
        ]

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1',
                input=messages,
                temperature=0.1,
                text_format=ConceptRouting,
            )
        except Exception as e:
            logging.error(str(e))
            raise ConnectionError(f"Something wrong with Openai: {str(e)}")

        return response.output_parsed
    
    async def get_samples(self, state: MatcherAgentState, collection: str) -> list[str]:
        """Retrieve sample documents from a specific vocabulary collection.
        
        Args:
            state: Current state containing the concept to search for.
            collection: Name of the vocabulary collection to search in.
            
        Returns:
            List of formatted sample strings from the collection.
        """
        indexer = Indexer()
        results = await indexer.search(state.concept, collection, max_results=3)
        formatted_results: list[str] = ['\t'.join(d.model_dump().values()) for d in results]
        return formatted_results
    
    async def __call__(self, state: MatcherAgentState) -> MatcherAgentState:
        """Process the routing step and update the state with the determined concept type.
        
        Args:
            state: Current state to process.
            
        Returns:
            Updated state with concept_type set.
        """
        concept_routing = await self.route(state)
        state.concept_type = concept_routing.concept_type
        return state
    

concept_router = ConceptRouter()