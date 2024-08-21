import os

from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

# Load PDF and store it in ChromaDB
pdf_loader = PyPDFLoader("data.pdf")
documents = pdf_loader.load()

embeddings = OpenAIEmbeddings(model="gpt-4o")
chroma_store = Chroma.from_documents(documents, embeddings)

# Initialize the LLM with GPT-4o
api_key = os.getenv("OPENAI_API_KEY")
openai_llm = OpenAI(model="gpt-4o", api_key=api_key)

# Create a basic prompt template
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="You are a helpful assistant. User said: {user_input}"
)

# Create an LLMChain with the LLM and the prompt
llm_chain = LLMChain(
    llm=openai_llm,
    prompt=prompt_template
)