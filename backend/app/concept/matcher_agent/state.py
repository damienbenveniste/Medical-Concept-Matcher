from pydantic import BaseModel, Field
from typing import Literal, Optional
from app.indexing.indexer import Data


class CodeItem(Data):
    """A medical code item with confidence score.
    
    Extends the base Data class with a confidence score for selected codes.
    
    Attributes:
        confidence: Model-estimated confidence score (90-100).
                   Codes below 90 should not be output.
    """
    confidence: int = Field(
        ...,
        description="Model-estimated confidence (90-100). "
                    "Do not output codes below 90.",
        gte=0,
        lte=100,
    )


class MatcherAgentState(BaseModel):
    """State object for the matcher agent workflow.
    
    Tracks the progression of concept matching through retrieval, selection,
    and validation phases.
    
    Attributes:
        concept: The original medical concept text to match.
        concept_type: The determined type of medical concept (diagnosis, procedure, etc.).
        retrieved_documents: Candidate codes retrieved from the knowledge base.
        selected_codes: Codes selected as best matches with confidence scores.
    """
    concept: Optional[str] = None
    concept_type: Optional[Literal['diagnose', 'procedure', 'measurement', 'drug']] = None
    retrieved_documents: list[Data] = []
    selected_codes: list[CodeItem] = []