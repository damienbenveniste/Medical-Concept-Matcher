from fastapi import APIRouter, HTTPException
from app.concept.matcher_agent.agent import matcher_agent
from app.concept.matcher_agent.state import MatcherAgentState, CodeItem
from pydantic import BaseModel
from typing import Literal, Optional
import logging


router = APIRouter()


class ConceptRequest(BaseModel):
    """Request model for concept matching.
    
    Attributes:
        concept: The medical concept text to match against code vocabularies.
    """
    concept: str

class Result(CodeItem):
    """Result model extending CodeItem with concept type.
    
    Attributes:
        concept_type: The determined type of medical concept.
    """
    concept_type: Optional[Literal['diagnose', 'procedure', 'measurement', 'drug']] = None

class ConceptResponse(BaseModel):
    """Response model for concept matching results.
    
    Attributes:
        concept: List of matched medical codes with confidence scores.
    """
    concept: list[Result]



@router.post("/match", response_model=ConceptResponse)
async def chat(request: ConceptRequest) -> ConceptResponse:
    """Match a medical concept to appropriate codes.
    
    Args:
        request: The concept matching request containing the text to match.
        
    Returns:
        ConceptResponse containing matched codes with confidence scores.
        
    Raises:
        HTTPException: If the matching process fails.
    """

    try:
        
        # Initialize state
        initial_state = MatcherAgentState(
            concept=request.concept,
        )
        
        # Run the agent
        result: dict = await matcher_agent.ainvoke(initial_state, debug=True)
        
        # Cast result to MatcherAgentState
        final_state = MatcherAgentState(**result)

        response = ConceptResponse(
            concept=[
                Result(text=data.text, code=data.code, confidence=data.confidence, concept_type=final_state.concept_type)
                for data in final_state.selected_codes
            ]
        )

        return response
        
    except Exception as e:
        logging.error(f"Matcher agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))