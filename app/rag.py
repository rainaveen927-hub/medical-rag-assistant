from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

VECTOR_DB_PATH = "app/vectorstore/faiss_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=200,
    temperature=0.3,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

def retrieve_docs(query):
    docs = retriever.invoke(query)
    return docs


def generate_answer(query, docs):

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    <|system|>
    You are a helpful medical AI assistant.

    Use the retrieved medical context to answer the question.

    Rules:
    - Do not repeat the context
    - Give short helpful answers
    - Use simple language
    - Mention seeing a doctor if symptoms worsen

    Retrieved Context:
    {context}

    </s>

    <|user|>
    {query}
    </s>

    <|assistant|>
    """

    response = llm.invoke(prompt)

    # Clean assistant output
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]

    return response.strip()