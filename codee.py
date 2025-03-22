from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM  # Corrected to use OllamaLLM

# Define paths
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

# Verify if documents are stored correctly
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
print("Number of documents in DB:", db._collection.count())

# Function to process a query
def process_query(query_text):
    if not query_text:
        return "Error: Query text cannot be empty."
    
    # Perform search in the Chroma database
    results = db.similarity_search(query_text, k=5)  # Get top 5 results

    # Prepare context from search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    # Define the prompt template
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    # Create the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Use DeepSeek R1 as the LLM
    model = OllamaLLM(model="deepseek-r1")  # Updated to OllamaLLM
    response_text = model.invoke(prompt)

    return response_text

# Example usage
query_text = input("Enter your query: ")
response = process_query(query_text)
print("\nAI Response:\n", response)
