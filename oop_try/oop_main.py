from fastapi import FastAPI
from oop_functions import query

app = FastAPI()

@app.get("/query/{question}")
def read_query(question: str):
    return query(question)