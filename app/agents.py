from langchain.tools import tool
from app.rag import retrieve_docs, generate_answer


@tool
def rag_tool(query: str):
    """Retrieve medical context and generate answer."""
    docs = retrieve_docs(query)
    answer = generate_answer(query, docs)

    return {
        "answer": answer,
        "docs": [d.page_content for d in docs]
    }


class RetrieverAgent:
    def run(self, query):
        docs = retrieve_docs(query)
        return docs


class ConsultationAgent:
    def run(self, query, docs):
        return generate_answer(query, docs)


class DiagnosisSupportAgent:
    def run(self, answer):
        diagnosis_prompt = f"""
        Extract possible medical condition from:

        {answer}

        Return:
        - Possible Disease
        - Severity
        - Recommendation
        """

        return diagnosis_prompt


class ValidatorAgent:
    def run(self, answer):
        if "consult a doctor" not in answer.lower():
            answer += "\n\nPlease consult a qualified doctor."

        return answer