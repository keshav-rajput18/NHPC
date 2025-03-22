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

# Print the first chunk to verify
print(chunks[0] if chunks else "No chunks generated.")
