from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = "data/books"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

# Load documents first
documents = load_documents()

# Create text splitter to generate chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True,
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

# # Print the first chunk to verify
# print(chunks[0] if chunks else "No chunks generated.")



#code to use Hugging Face embeddings

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma_db"

# Use a local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a new DB from the documents
db = Chroma.from_documents(
    chunks, embedding_model, persist_directory=CHROMA_PATH
)
