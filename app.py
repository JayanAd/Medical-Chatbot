from flask import Flask, render_template,jsonify,request
from src.helper import download_huggingface_embeddings
from langchain_community.vectorstores import Pinecone
# from langchain.vectorstores import Pinecone
import pinecone
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from src.prompt import *
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# download the embedding
embeddings = download_huggingface_embeddings()

# Initializing Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "chatbot"

# Loading the existing index from pinecone
docsearch = Pinecone.from_existing_index(index_name,embeddings)

# creating Prompt Template

PROMPT = PromptTemplate(template=prompt_template,input_variables=["context","question"])
chain_type_kwargs={"prompt":PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

# Intilize qa

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


# Default Route 

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query":input})
    print("Response: ",result["result"])
    return str(result["result"])

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)