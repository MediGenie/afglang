from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from pinecone_helper import create_index, data_injest
from dotenv import load_dotenv
import os 
from langchain.docstore.document import Document
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAI()

def split_data(file_name):
    logging.info(f"Loading data from file: {file_name}")

    # Check the file extension and use the appropriate loader
    if file_name.endswith('.pdf'):
        loader = PyPDFLoader(file_name)
        logging.info("Using PyPDFLoader for PDF file.")
    elif file_name.endswith('.txt'):
        loader = TextLoader(file_name)
        logging.info("Using TextLoader for text file.")
    else:
        logging.error("Unsupported file type.")
        raise ValueError("Unsupported file type")

    # Load the data
    data = loader.load()
    logging.info("Data loaded successfully.")

    # Split the data into chunks
    final_docs = split_data_into_chunks(data)
    logging.info("Data split into chunks successfully.")
    return final_docs


def split_data_into_chunks(dataset):
    logging.info("Starting the process to split dataset into chunks.")

    #Initialize text splitter with specified parameters
    text_splitter = CharacterTextSplitter(
        separator="**********",
        chunk_size=10,
        chunk_overlap=0,
    )

#     text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 1000,
#     chunk_overlap  = 200,
#     length_function = len,
#     is_separator_regex = False,
# )

    logging.info("Initialized CharacterTextSplitter with separator '**********', chunk_size 10, and chunk_overlap 0.")
    logging.info("Splitting dataset into chunks based on specified separator.")

    #Split the dataset into chunks
    #data = text_splitter.split_documents(dataset)

    logging.info(f"Dataset split into {len(chunks)} chunks. Showing first chunk for reference: {chunks[0] if chunks else 'No chunks available'}")

    # Prepare list of Document objects
    documents = []

    for i, doc in enumerate(chunks):
        # Assuming each chunk is a Document object, access its page_content
        content = doc.page_content

        # Check and extract 질문 and 답변
        if '질문:' in content and '답변:' in content:
            question_part, answer_part = content.split('답변:')
            question = question_part.split('질문:')[1].strip(' []')
            answer = answer_part.strip(' []')
            documents.append(Document(page_content=question, metadata={"answer": answer}))
            logging.info(f"Chunk {i+1} processed successfully. Question extracted: {question}")
        else:
            logging.warning(f"Chunk {i+1} does not contain a valid Q&A pair.")

    logging.info("Completed processing all chunks.")
    return data



def data_injestion(docs_split):
    logging.info("Starting data ingestion process")
    data_injest(docs_split)
    logging.info("Data ingestion completed")

def retrieval_QA(vector_array):
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm = llm, 
        retriever = vector_array.as_retriever()
    )
    return chain