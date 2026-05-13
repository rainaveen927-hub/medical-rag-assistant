from fastapi import FastAPI
from app.schemas import (
    QueryRequest,
    QueryResponse,
    FeedbackRequest
)

from app.workflow import graph
from app.feedback import save_feedback

app = FastAPI(
    title="Medical Multi-Agent RAG API"
)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    result = graph.invoke({
        "query": request.query
    })

    contexts = [
        doc.page_content
        for doc in result["docs"]
    ]

    return QueryResponse(
        answer=result["answer"],
        retrieved_context=contexts,
        diagnosis=result["diagnosis"]
    )


@app.post("/feedback")
def feedback(data: FeedbackRequest):

    save_feedback(
        data.query,
        data.response,
        data.rating
    )

    return {"message": "Feedback saved"}