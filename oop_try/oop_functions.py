from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

class DataProcessor:
    def __init__(self, file_path, model_name):
        self.file_path = file_path
        self.model_name = model_name
        self.instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})

    def load_and_process_data(self):
        loader = UnstructuredFileLoader(self.file_path)
        documents = loader.load()
        persist_directory = 'db'
        vectordb = Chroma.from_documents(documents=documents, embedding=self.instructor_embeddings, persist_directory=persist_directory)
        vectordb.persist()
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.instructor_embeddings)
        return vectordb

class QueryProcessor:
    def __init__(self, vectordb, query, k):
        self.vectordb = vectordb
        self.query = query
        self.k = k

    def get_relevant_documents(self):
        retriever = self.vectordb.as_retriever(search_kwargs={"k": self.k})
        docs = retriever.get_relevant_documents(self.query)
        return docs

class QAChainCreator:
    def __init__(self, docs, embeddings):
        self.docs = docs
        self.embeddings = embeddings

    def create_qa_chain(self):
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(self.docs)
        docsearch = Chroma.from_documents(texts, self.embeddings)
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=docsearch.as_retriever())
        return qa_chain


# Use the classes
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = API_KEY

if not API_KEY:
    raise ValueError("La clé d'API n'est pas définie")

data_processor = DataProcessor("./transcript/transcript.txt", "hkunlp/instructor-xl")
vectordb = data_processor.load_and_process_data()

query_processor = QueryProcessor(vectordb, "Où a lieu la discussion ?", 3)
docs = query_processor.get_relevant_documents()

embeddings = OpenAIEmbeddings()

chain_creator = QAChainCreator(docs, embeddings)
qa_chain = chain_creator.create_qa_chain()

print({"Answer": qa_chain.run("Dans quelle salle la discussion a lieu?")})


def query(question: str):

    data_processor = DataProcessor("./transcript/transcript.txt", "hkunlp/instructor-xl")
    vectordb = data_processor.load_and_process_data()

    query_processor = QueryProcessor(vectordb, "Où a lieu la discussion ?", 3)
    docs = query_processor.get_relevant_documents()

    embeddings = OpenAIEmbeddings()

    chain_creator = QAChainCreator(docs, embeddings)
    qa_chain = chain_creator.create_qa_chain()

    return {"Answer": qa_chain.run(question)}