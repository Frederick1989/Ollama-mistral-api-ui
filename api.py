# API stuff
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Python Niceities 
from pydantic import BaseModel

# My ollama app
from app.langchain_ollama_app import store_new_information, prompt_query, web_scraper

# Models for my API
class Question(BaseModel):
    question: str

class Inform(BaseModel):
    fact: str

class Article(BaseModel):
    url: str

# Set up the app
app=FastAPI()

@app.get("/")
async def home():
    return "Server Running"

@app.post("/scrape")
async def read_article(input: Article):
    web_scraper(input.url)
    return "Read article at '" + input.url + "'"

@app.post("/store")
async def learn_fact(input: Inform):
    store_new_information(input.fact)
    return "Learned '" + input.fact + "'"

@app.post("/ask")
async def ask_question(input: Question):
    answer = prompt_query(input.question)
    print('llm replies to api : \n',answer)
    json_answer = jsonable_encoder(answer)
    return JSONResponse(json_answer)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5173)