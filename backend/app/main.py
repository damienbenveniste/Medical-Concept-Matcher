"""Main FastAPI application for the Medical Concept Matcher.

This module sets up the FastAPI application with routers for indexing
and concept matching functionality.
"""
from fastapi import FastAPI
from app.indexing.api import router as indexing_router
from app.concept.api import router as concept_router


# Initialize FastAPI application
app = FastAPI(
    debug=True,
    title="Medical Concept Matcher App",
    docs_url='/docs'
)

# Register API routers
app.include_router(indexing_router, prefix="/indexing")  # Handles document indexing
app.include_router(concept_router, prefix="/concept")    # Handles concept matching