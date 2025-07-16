
from typing import Union
from app.concept.matcher_agent.state import MatcherAgentState, CodeItem
from app.clients import async_openai_client
from pydantic import BaseModel, Field
import asyncio
import logging

SYSTEM_PROMPT = """
You are a clinical-coding QA agent.

Goal
────
For each candidate code proposed by the selector stage, judge whether it
is a *clinically valid* match for the input concept.  Reject any code that
is ambiguous, overly generic, obsolete, or clearly unrelated.

Inputs you receive
──────────────────
• concept_name   - the original free-text term the user wants coded  
• concept_type   - one of {diagnosis | procedure | measurement | drug}  
• code_items     - list of objects, each containing:
      · code          (e.g., "J98.11")  
      · system        ("ICD-10", "CPT", "LOINC", "RxNorm", "ATC")  
      · description   (official descriptor text)  
      · orig_conf     (confidence score from selector stage, 90-100)

Evaluation rules
────────────────
1. **accept** the code only if you are ≥ 90 % confident it is the correct,
   specific match for *concept_name*.
2. Otherwise **reject** it.  
   · Reject even slightly plausible codes if any uncertainty remains.  
   · Reject codes whose descriptor lacks the key clinical term(s).  
3. Consider laterality, dose-form, encounter type, panel vs. analyte,
   ingredient vs. combo, etc.  Specific beats generic.
"""

class CodeFilterResult(BaseModel):
    """Result of the post-processing validation step for a single candidate code.
    
    Attributes:
        is_valid: True if the candidate code is an unambiguous, ≥ 90% confidence match
                  for the input concept; False if it should be rejected.
    """
    is_valid: bool = Field(
        ...,
        description=(
            "True if the candidate code is an unambiguous, ≥ 90 %-confidence match "
            "for the input concept; False if it should be rejected."
        )
    )



class Validator:
    """Validates selected medical codes for clinical accuracy.
    
    This class performs post-processing validation on codes selected by the selector,
    ensuring they meet clinical standards and are appropriate matches for the concept.
    """

    async def filter_doc(self, state: MatcherAgentState, document: CodeItem) -> CodeFilterResult:
        """Filter a single code document through validation.
        
        Args:
            state: Current matcher agent state containing concept information.
            document: Code item to validate.
            
        Returns:
            CodeFilterResult indicating whether the code is valid.
            
        Raises:
            ConnectionError: If OpenAI API call fails.
        """

        formatted_code = '\t'.join(str(v) for v in document.model_dump().values())

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"concept: {state.concept}\nconcept type: {state.concept_type}\nselected code:{formatted_code}"},
        ]

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1',
                input=messages,
                temperature=0.1,
                text_format=CodeFilterResult,
            )
        except Exception as e:
            logging.error(str(e))
            raise ConnectionError(f"Something wrong with Openai: {str(e)}")

        return response.output_parsed
    
    async def filter_documents(self, state: MatcherAgentState) -> list[Union[CodeFilterResult, Exception]]:
        """Filter multiple code documents concurrently.
        
        Args:
            state: Current matcher agent state containing selected codes.
            
        Returns:
            List of validation results or exceptions from concurrent processing.
        """

        data = state.selected_codes
        tasks = [
            asyncio.create_task(self.filter_doc(state, doc)) 
            for doc in data
        ]
        filters = await asyncio.gather(*tasks, return_exceptions=True)
        return filters

    async def __call__(self, state: MatcherAgentState) -> MatcherAgentState:
        """Process state through validation.
        
        Args:
            state: Current matcher agent state.
            
        Returns:
            Updated state with only valid codes retained.
        """
        
        filters = await self.filter_documents(state)
        filtered_codes = []

        for doc, filter_result in zip(state.selected_codes, filters):
            # Check if the result is an exception
            if isinstance(filter_result, Exception):
                logging.error(f"Error filtering code {doc.code}: {filter_result}")
                continue
            
            # Check if the filter result is valid
            if filter_result.is_valid:
                filtered_codes.append(doc)

        state.selected_codes = filtered_codes
        return state
    

validator = Validator()