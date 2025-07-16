from dataclasses import dataclass
from typing import Union
from langgraph.graph import END, StateGraph, START
from app.concept.matcher_agent.state import MatcherAgentState
from app.concept.matcher_agent.nodes.concept_router import concept_router
from app.concept.matcher_agent.nodes.retrieval import retriever
from app.concept.matcher_agent.nodes.selector import code_selector
from app.concept.matcher_agent.nodes.validation import validator

@dataclass(frozen=True)
class Nodes:
    """Node name constants for the matcher agent graph."""
    CONCEPT_ROUTER = "concept_router"
    RETRIEVER = 'retriever'
    CODE_SELECTOR = 'code_selector'
    VALIDATOR = 'validator'


def concept_type_routing(state: MatcherAgentState) -> Union[str, type]:
    """Route based on concept type presence.
    
    Args:
        state: Current matcher agent state.
        
    Returns:
        END if no concept type is found, otherwise next node name.
    """
    # If concept type couldn't be determined, end the workflow
    if state.concept_type is None:
        return END
    else:
        # Continue to retrieval if concept type is valid
        return Nodes.RETRIEVER


# Create the state graph for the matcher agent workflow
builder = StateGraph(MatcherAgentState)

# Add all nodes to the graph
builder.add_node(Nodes.CONCEPT_ROUTER, concept_router)  # Determines concept type
builder.add_node(Nodes.RETRIEVER, retriever)           # Retrieves candidate codes
builder.add_node(Nodes.CODE_SELECTOR, code_selector)   # Selects best matching codes
builder.add_node(Nodes.VALIDATOR, validator)           # Validates selected codes

# Define the workflow edges
builder.add_edge(START, Nodes.CONCEPT_ROUTER)  # Start with concept routing

# Conditional routing after concept type determination
builder.add_conditional_edges(
    Nodes.CONCEPT_ROUTER, 
    concept_type_routing,
    {
        END: END,                        # End if no concept type found
        Nodes.RETRIEVER: Nodes.RETRIEVER,  # Continue to retrieval if valid
    }
)

# Linear workflow for successful matches
builder.add_edge(Nodes.RETRIEVER, Nodes.CODE_SELECTOR)    # Retrieval -> Selection
builder.add_edge(Nodes.CODE_SELECTOR, Nodes.VALIDATOR)    # Selection -> Validation
builder.add_edge(Nodes.VALIDATOR, END)                    # Validation -> End

# Compile the graph into an executable agent
matcher_agent = builder.compile()