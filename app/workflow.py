from typing import TypedDict
from langgraph.graph import StateGraph, END

from app.agents import (
    RetrieverAgent,
    ConsultationAgent,
    DiagnosisSupportAgent,
    ValidatorAgent
)

retriever = RetrieverAgent()
consultant = ConsultationAgent()
diagnosis = DiagnosisSupportAgent()
validator = ValidatorAgent()


class AgentState(TypedDict):
    query: str
    docs: list
    answer: str
    diagnosis: str


def retrieve_node(state):
    docs = retriever.run(state["query"])
    return {"docs": docs}


def consultation_node(state):
    answer = consultant.run(state["query"], state["docs"])
    return {"answer": answer}


def diagnosis_node(state):
    diagnosis_text = diagnosis.run(state["answer"])
    return {"diagnosis": diagnosis_text}


def validation_node(state):
    validated = validator.run(state["answer"])
    return {"answer": validated}


workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("consult", consultation_node)
workflow.add_node("diagnosis", diagnosis_node)
workflow.add_node("validate", validation_node)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "consult")
workflow.add_edge("consult", "diagnosis")
workflow.add_edge("diagnosis", "validate")
workflow.add_edge("validate", END)

graph = workflow.compile()