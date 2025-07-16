from app.concept.matcher_agent.state import MatcherAgentState
from app.indexing.indexer import Indexer, Data, Datasets


class Retriever:
    """Document retrieval agent that searches for relevant documents from the appropriate vocabulary collection."""

    async def retrieve(self, state: MatcherAgentState) -> list[Data]:
        """Retrieve relevant documents from the appropriate vocabulary collection.
        
        Args:
            state: Current state containing the concept and concept type.
            
        Returns:
            List of relevant Data objects from the vocabulary collection.
            
        Raises:
            NotImplementedError: If the concept type is not supported.
        """
        collection: str
        
        if state.concept_type == 'diagnose':
            collection = Datasets.ICD
        elif state.concept_type == 'procedure':
            collection = Datasets.CPT
        elif state.concept_type == 'measurement':
            collection = Datasets.LOINC
        elif state.concept_type == 'drug':
            collection = Datasets.ATC
        else:
            raise NotImplementedError(f"Concept type '{state.concept_type}' is not supported")
        
        indexer = Indexer()
        results: list[Data] = await indexer.search(state.concept, collection=collection, max_results=20)
        return results

    async def __call__(self, state: MatcherAgentState) -> MatcherAgentState:
        """Process the retrieval step and update the state with retrieved documents.
        
        Args:
            state: Current state to process.
            
        Returns:
            Updated state with retrieved_documents populated.
        """
        results = await self.retrieve(state)
        state.retrieved_documents = results
        return state
    

retriever = Retriever()
        