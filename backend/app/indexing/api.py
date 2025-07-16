

from typing import Dict
from fastapi import APIRouter, HTTPException
from app.indexing.indexer import Indexer
import logging

router = APIRouter()


@router.post("/index")
async def index_documents() -> Dict[str, str]:
    """Index all medical documents for search.
    
    Processes and indexes all medical vocabulary documents to make them
    searchable by the concept matcher.
    
    Returns:
        Dictionary containing status and message about the indexing operation.
        
    Raises:
        HTTPException: If the indexing process fails.
    """

    try:
        # Initialize indexer
        indexer = Indexer()
        
        # Process and index the documents
        await indexer.index_all()
        
        return {
            "status": "success", 
            "message": "Documents have been successfully indexed and are ready for search"
        }

    except Exception as e:
        logging.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")