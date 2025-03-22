from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data/books"
CHROMA_PATH = "chroma_db"

# Load documents
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md", show_progress=True)
    documents = loader.load()
    return documents

documents = load_documents()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)

# Initialize embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Initialize ChromaDB
db = Chroma(
    collection_name="my_collection",
    embedding_function=embedding_function,
    persist_directory=CHROMA_PATH
)

# Add document chunks to the database
db.add_documents(chunks)
db.persist()  # Save to disk

# Verify if documents are stored correctly
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
print("Number of documents in DB:", db._collection.count())
