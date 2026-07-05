from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import nltk
import ssl
   
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt_tab")

# configuration
docs_dir_path = r"C:\AI Engineer\AI Projects\WHO Childhood Cancer Assistant\docs_dir"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vector_db_path = os.path.join(BASE_DIR, "vector_db")
collection_name  = "document_collection"

# loading the embedding model
embedding = HuggingFaceEmbeddings()

# directory loader
loader = DirectoryLoader(
    path=docs_dir_path,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# single file can also be loaded with UnstructuredFileLoader

# load the documents
documents = loader.load()

print(type(documents))

len(documents)

# initializing the text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500
)

# splitting the text into smaller chunks
text_chunks = text_splitter.split_documents(documents)

len(text_chunks)

# creating the vector store
vector_store = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding,
    persist_directory=vector_db_path,
    collection_name=collection_name
)

