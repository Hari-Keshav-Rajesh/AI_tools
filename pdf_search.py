from langchain_openai import ChatOpenAI
import fitz  # PyMuPDF
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model = "llama3",
    temperature = 0.0,
    base_url = 'http://localhost:11434/v1',
    api_key='ollama',
)

def pdf_to_text(pdf_path):
    """
    Converts a PDF file to text.
    """
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

@tool
def pdf_search():
    """
    This tool searches a PDF document for a query and returns the most relevant documents.
    """
    # Convert PDF to text
    pdf_text = pdf_to_text("GDP.pdf")
    
    # Save the text to a temporary file
    with open("temp_text.txt", "w", encoding="utf-8") as f:
        f.write(pdf_text)
    
    # Load the text document
    loader = TextLoader("temp_text.txt")
    documents = loader.load()

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Use a pre-trained sentence transformer model for embeddings
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a Chroma vector store from the documents
    db = Chroma.from_documents(docs, embedding_function)

    # Query the vector store
    query = "Top 10 GDP countries"
    results = db.similarity_search(query)

    return results

tools = [pdf_search]

template = '''
    You are a Researcher. You will answer questions based on the document you have been given.
    You have access to the following tools: pdf_search
    Take the user question and answer appropriately.

    Question:
    {question}

    Answer:
    (required answer here)
    '''

prompt = ChatPromptTemplate.from_template(
    template
)

llm_with_tools = llm.bind_tools(tools,tool_choice="pdf_search")


'''

output = llm_with_tools.invoke("What is mitblr.club?")

print(output)

'''

print(pdf_search())




