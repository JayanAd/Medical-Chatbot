from src.helper import load_pdf,text_split,download_huggingface_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Load pdf
extracted_data = load_pdf("data/")

# create chunks
text_chunks = text_split(extracted_data)

# download the embedding
embeddings = download_huggingface_embeddings()

# Initializing Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)

index_name = "chatbot"
index = pc.Index(index_name)

# Creating Embeddings for each of the text chunks and storing
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks],embeddings.embed_query,index_name=index_name)


