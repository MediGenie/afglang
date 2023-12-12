from fastapi.responses import HTMLResponse
from fastapi import FastAPI, File, UploadFile, Depends
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain 
from langchain.vectorstores.pinecone import Pinecone
from functions import split_data, data_injestion, retrieval_QA
from pinecone_helper import get_documents, existing_index
from dotenv import load_dotenv
import pinecone
import os 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

llm = OpenAI()

docs_folder_path = "backend_docs"

if not os.path.exists(docs_folder_path):
    os.makedirs(docs_folder_path)

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    logging.info(f"Received file upload request: {file.filename}")

    if not file.filename.endswith('.txt'):
        logging.warning(f"File '{file.filename}' is not a text file.")
        return {"error": "Only text files are allowed"}

    file_path = os.path.join(docs_folder_path, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    logging.info(f"File '{file.filename}' saved to '{file_path}'.")

    try:
        docs = split_data(file_path)
        logging.info("File split into sections successfully.")
        data_injestion(docs)
        logging.info("Data ingested into Pinecone successfully.")
        return {"result": "File uploaded and processed successfully"}
    except Exception as e:
        logging.error(f"Error during file processing: {e}")
        return {"error": str(e)}


@app.get("/upload_file/")
def get_upload_file():
    return {"message": "Please upload a file using the POST method."}

@app.post("/insert_data_into_vector_database/")
def insert_data():
    if os.path.exists(docs_folder_path) and os.path.isdir(docs_folder_path):
        files_in_docs = os.listdir(docs_folder_path)
        if files_in_docs:
            file = files_in_docs[0]
            docs = split_data(f"{docs_folder_path}/{file}")
            data_injestion(docs)
            return {"file":"data_injested"}        
        else:
            return {"error":"The 'docs' folder is empty."}
    else:
        return {"error":"No docs folder exists"}

@app.post("/query/")
def query(text: str):
    vector_docs = existing_index()
    chain = retrieval_QA(vector_docs)
    answer = chain({"question":text}, return_only_outputs = True)['answer']
    return {"answer": answer}    


@app.get("/", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
        <head>
            <title>Upload Text File</title>
        </head>
        <body>
            <h2>Upload Text File to Pinecone</h2>
            <form action="/upload_file/" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept=".txt"><br><br>
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """


