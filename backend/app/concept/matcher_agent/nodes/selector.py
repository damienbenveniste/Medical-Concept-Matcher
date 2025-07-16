from app.indexing.indexer import Data
from pydantic import BaseModel, Field
from app.concept.matcher_agent.state import MatcherAgentState, CodeItem
from app.clients import async_openai_client
import logging


SYSTEM_PROMPT = """
You are a clinical-coding selector.

Inputs you receive
──────────────────
1. concept_name        → free-text term from the user
2. concept_type        → one of {diagnosis | procedure | measurement | drug}
3. candidate_codes     → exactly 20 rows retrieved from the relevant
                          vocabulary.  Each row is formatted:
                            <CODE>  <OFFICIAL_DESCRIPTION>

Task
────
• Examine every candidate row.
• Choose ONLY those code(s) that unambiguously match the concept.
• “Unambiguous” means ≥ 90 % confidence that a trained clinical coder
  would pick the same code(s) without hesitation.
• If no candidate meets that bar, return an empty list.

Rules
─────
1. You may ONLY return codes that appear in the candidate list.
2. Prefer specificity: pick the most precise code that still matches.
3. False-positives are worse than false-negatives.  
   When in doubt → leave the list empty.
4. Output **only** the JSON that conforms to the schema below—no
   markdown, no extra keys, no comments.
"""



class CodeSelection(BaseModel):
    """Response model for code selection results.
    
    Attributes:
        selected_codes: List of selected code items that meet the confidence threshold.
                       Should be sorted by descending confidence.
    """
    selected_codes: list[CodeItem] = Field(
        ...,
        description=(
            "Zero or more Data entries, sorted by descending confidence. "
            "Return an empty list if no candidate achieves ≥90 confidence."
        )
    )


class CodeSelector:
    """Selects the most appropriate medical codes from retrieved candidates.
    
    This class uses an LLM to evaluate candidate codes and select only those
    that unambiguously match the given medical concept with high confidence.
    """

    async def select(self, state: MatcherAgentState) -> CodeSelection:
        """Select appropriate codes from retrieved candidates.
        
        Args:
            state: Current state containing concept information and retrieved documents.
            
        Returns:
            CodeSelection containing validated codes that meet confidence threshold.
            
        Raises:
            NotImplementedError: If concept_type is not supported.
            ConnectionError: If OpenAI API call fails.
        """

        if state.concept_type == 'diagnose':
            sample_type = "ICD-10 documents"
        elif state.concept_type == 'procedure':
            sample_type = "CPT documents"
        elif state.concept_type == 'measurement':
            sample_type = "LOINC documents"
        elif state.concept_type == 'drug':
            sample_type = "ATC documents"
        else:
            raise NotImplementedError(f"Unsupported concept type: {state.concept_type}")
        
        documents = "\n".join(['\t'.join(d.model_dump().values()) for d in state.retrieved_documents])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"### {sample_type} ###\n\n{documents}"},
            {"role": "user", "content": f"concept: {state.concept}\nconcept type: {state.concept_type}"},
        ]

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1',
                input=messages,
                temperature=0.1,
                text_format=CodeSelection,
            )
        except Exception as e:
            logging.error(str(e))
            raise ConnectionError(f"Something wrong with Openai: {str(e)}")
        
        valid_results = []
        valid_codes = set(d.code for d in state.retrieved_documents)
        for item in response.output_parsed.selected_codes:
            if item.code in valid_codes and item.confidence >= 90:
                valid_results.append(item)

        return CodeSelection(selected_codes=valid_results)
        
    async def __call__(self, state: MatcherAgentState) -> MatcherAgentState:
        """Process state through code selection.
        
        Args:
            state: Current matcher agent state.
            
        Returns:
            Updated state with selected codes populated.
        """
        selection = await self.select(state)
        state.selected_codes = selection.selected_codes
        return state
    

code_selector = CodeSelector()