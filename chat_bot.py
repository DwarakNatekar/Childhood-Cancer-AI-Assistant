import streamlit as st
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv(override=True)


st.set_page_config(
    page_title="Childhood Cancer AI Assistant",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Childhood Cancer AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vector_db_path = os.path.join(BASE_DIR, "vector_db")
collection_name  = "document_collection"

# loading the embedding model - default model
embedding = HuggingFaceEmbeddings()

# initialize openaillm
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    streaming=True
)

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding,
    persist_directory=vector_db_path
)

retriever = vector_store.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

user_prompt = st.chat_input("Ask about childhood cancer...")

if user_prompt:

    st.chat_message("user").markdown(user_prompt)

    st.session_state.chat_history.append(
        {
            "role":"user",
            "content":user_prompt
        }
    )

    response = qa_chain.invoke({"query": user_prompt})

    assistant_response = response["result"]

    with st.chat_message("assistant"):
        st.write_stream(word + " " for word in assistant_response.split())

        with st.expander("Sources Used"):
            for doc in response["source_documents"]:
                file_name = os.path.basename(doc.metadata.get("source", "Unknown File"))
                page = doc.metadata.get("page", "N/A")
                st.write(f"📄 **{file_name}** | **Page:** {page + 1 if isinstance(page, int) else page}")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": assistant_response
        }
    )


