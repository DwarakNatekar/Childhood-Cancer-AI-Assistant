import streamlit as st
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
# langchain library has been restructured recently. Use langchain_classic for older imports as mentioned below.
from langchain_classic.chains import RetrievalQA

from dotenv import load_dotenv

load_dotenv(override=True)
# print(os.getenv("OPENAI_API_KEY"))

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
vector_db_path = r"C:\AI Engineer\AI Projects\WHO Childhood Cancer Assistant\vector_db"
collection_name  = "document_collection"

# loading the embedding model - default model
embedding = HuggingFaceEmbeddings()

# initialize openaillm
llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.0
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

    response = qa_chain.invoke({"query":user_prompt})

    assistant_response = response["result"]

    st.session_state.chat_history.append(
        {
            "role":"assistant",
            "content":assistant_response
        }
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
        with st.expander("Sources Used"):
            for doc in response["source_documents"]:
                st.write(doc.metadata)


