from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

load_dotenv()

# Define the API Key
API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = API_KEY

# Vérification de la clé d'API
if not API_KEY:
    raise ValueError("La clé d'API n'est pas définie")

def load_and_process_data(file_path, model_name):
    """loading and processing the data"""
    """"args : file_path, model_name  """

    # Load the data
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()

    # Embeddings
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})

    # Create the DB
    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=documents, embedding=instructor_embeddings, persist_directory=persist_directory)

    # persist the db to disk and then load it again
    vectordb.persist()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=instructor_embeddings)
    return vectordb


def get_relevant_documents(vectordb, query, k):
    """ Make a retriever and get relevant documents"""
    
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs


def create_qa_chain(docs, embeddings):
    """ Split the documents and create the retriever and create the chain to answer questions                   """
    """ Possiblity to use tiktoken with to commented text_splitter commented coding line (but way longer to run)"""

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)     # to use tiktoken : text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    docsearch = Chroma.from_documents(texts, embeddings)

    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=docsearch.as_retriever())
    return qa_chain


def query(question: str):
    """Query the chain and return the anwser based on your question"""
    return {"Answer": qa_chain.run(question)}


# Use the functions
vectordb = load_and_process_data("./transcript/transcript.txt", "hkunlp/instructor-xl")
docs = get_relevant_documents(vectordb, "Où a lieu la discussion ?", 3)
embeddings = OpenAIEmbeddings()
qa_chain = create_qa_chain(docs, embeddings)
print(query("Dans quelle salle la discussion a lieu?"))