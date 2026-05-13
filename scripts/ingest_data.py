import json
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

documents = []

# =========================================================
# LOAD format_dataset.csv
# =========================================================

df = pd.read_csv("app/data/format_dataset_1.csv")

for _, row in df.iterrows():

    content = f"""
    Disease: {row.to_dict()}
    """

    documents.append(
        Document(
            page_content=content,
            metadata={"source": "format_dataset"}
        )
    )

# =========================================================
# LOAD chatdoctor5k.json
# =========================================================

with open("app/data/chatdoctor5k.json", "r") as f:
    doctor_data = json.load(f)

for item in doctor_data:

    content = f"""
    Instruction:
    {item.get("instruction", "")}

    Patient Query:
    {item.get("input", "")}

    Doctor Response:
    {item.get("output", "")}
    """

    documents.append(
        Document(
            page_content=content,
            metadata={"source": "chatdoctor5k"}
        )
    )

# =========================================================
# LOAD ragcare_qa.csv
# =========================================================

qa_df = pd.read_csv("app/data/ragcare_qa.csv")

for _, row in qa_df.iterrows():

    content = f"""
    Question:
    {row.get('question', '')}

    Answer:
    {row.get('answer', '')}
    """

    documents.append(
        Document(
            page_content=content,
            metadata={"source": "ragcare_qa"}
        )
    )

# =========================================================
# CHUNKING
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# =========================================================
# EMBEDDINGS
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# VECTOR STORE
# =========================================================

vectorstore = FAISS.from_documents(
    docs,
    embedding_model
)

vectorstore.save_local(
    "app/vectorstore/faiss_index"
)

print(f"Indexed {len(docs)} chunks successfully.")