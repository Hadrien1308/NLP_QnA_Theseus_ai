# QnA on TextFile

## Context

### Goal :
- Make an API that answers a question based on a text file named transcript.txt

### Data:
- A Text File : transcript.txt is the investors's meeting script.

### Librairies :
- All libraries needed to do this project are present in requirements.text
- Use the following command to charge all the libraries needed with PIP : pip install -r requirements.txt or use `conda` to avoid dependencies conflicts. 

### API_KEY
- Use your own openai API_KEY by putting it as an environement variable in a .env file in same directory as functions.py and main.py.


## Code Documentation

### .gitignore file and .env file:
- create a .gitignore file and put .env and __pycache__ in it in order not to share your OPENAI_API_KEY by accident (security good practice)
- In .env write OPENAI_API_KEY = your_api_key (your_api_key shall not be written quoting like this 'your_api_key')

### Number of python files : 
- 2 files functions.py, main.py

### Files's functions:
- functions.py is performing a query function that asks a questions to our text file using OpenAI API, langchain functions, chromadb and tiktoken to tokenise transcript.txt
- main.py takes a question as an input and retrieves an answer using fastapi (enter in terminal : uvicorn main:app --reload)

### oop_try 
- 2 python files : oop_functions.py and oop_main.py
- oop_main.py launching fastapi (enter in terminal uvicorn oop_main:app --reload)
- oop_functions.py organizes the query function in three classes :

   1. **DataProcessor:** This class is responsible for loading and processing the data. 
   In its __init__, it takes the path to a file to load and the name of an embedding model to use.

   2. **QueryProcessor:** This class is responsible for retrieving the most relevant documents in relation to a given query. 
   In its __init__, it takes an embedding vector database, a query, and a number k. 

   3. **QAChainCreator:** This class is responsible for creating a question-answering (QA) chain. 
   In its __init__, it takes a list of documents and an embedding system.
   Its create_qa_chain method splits the documents into chunks.

These classes enable the creation of a question-answering system from an unstructured document.  