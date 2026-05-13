from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    retrieved_context: List[str]
    diagnosis: str


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int