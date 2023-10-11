# A couple of magic strings
chatModel               = "llama2"
collection_name         = "absa_gpt"
collection_directory    = "data/chroma"
data_store              ="../data/unprocessed"
Ollama_host ="http://localhost:11434"

# Some general python niceities
from typing import List
from pydantic import BaseModel
from datetime import datetime
from PyPDF2 import PdfReader
# Support our embeddings, vectordatabase, and langchain needs
import chromadb
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain.embeddings import OllamaEmbeddings
import os
# Set up the vector database and client we will use
persistent_client = chromadb.PersistentClient(path=collection_directory)
collection = persistent_client.get_or_create_collection(collection_name)

vectorstore = Chroma(client=persistent_client,
                     collection_name=collection_name,
                     embedding_function=OllamaEmbeddings(), 
                     persist_directory=collection_directory)


##########################################
# LLM Needs
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate 
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader


from langchain.cache import RedisSemanticCache
from langchain.vectorstores.redis import Redis
import langchain
import redis



# use any embedding provider...

# redis_host = "redis-11474.c239.us-east-1-2.ec2.cloud.redislabs.com"
# redis_port=11474
# password='sXJehp3hDTpAHUp59sO93L1GmsGne5TZ'
# redis_username='default'
# redis_url=f'redis://{redis_username}:{password}@{redis_host}:{redis_port}'

# r = redis.Redis(host=redis_host, port=redis_port, password=password)

# # r = redis.Redis(host=redis_url, port=port, password=password)
# print(r.info() )

role ='support'
if not role=='support':
    role='sales'

#  Hierarchical Navigable Small World , vs KNN  HNSW is superior in terms of performance and our usecases  300X improvement
# vector_schema = {
#     "algorithm": "HNSW"
# }

# vectorstore= Redis.from_existing_index(index_name=role, embedding=OllamaEmbeddings(), redis_url=redis_url,schema=vector_schema)

# langchain.llm_cache = RedisSemanticCache(
#     embedding=OllamaEmbeddings(),
#     redis_url=redis_url
# )


#  prompt engineering
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""

# prompt template
CHAIN_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Set up out LLM (Ollama) for streaming callback
chatllm = Ollama(base_url=Ollama_host,
             model=chatModel,
             verbose=False,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# LLM Needs
##########################################


# Define out interface
# Read an article and store it , confluence, absa.co.za etc
def web_scraper(url) : 
    data = WebBaseLoader(url).load()
    text_splitter=RecursiveCharacterTextSplitter( chunk_size = 100, chunk_overlap  = 20, length_function = len, is_separator_regex = False).split_documents(data)
    ids = vectorstore.add_documents(text_splitter)
    vectorstore.persist()
    print(ids)
    return


def learn_from_existing_documents():
    # Walk through the files in the directory
    try:
        for filename in os.listdir(data_store):
            docs=[]
            if filename.endswith('.pdf'):
                file=f"{data_store}/{filename}"
                reader =PdfReader(file)
                i = 1
                for page in reader.pages:
                    docs.append(Document(page_content=page.extract_text(), metadata={'page':i,'source':filename}))
                    i += 1
                    print(f'processing page :{i}')
                text_splitter=RecursiveCharacterTextSplitter( chunk_size = 1000, chunk_overlap  = 0, length_function = len, is_separator_regex = False).split_documents(docs)
                print(f'text_splitter : {len(text_splitter)} last page : {text_splitter[-1]}')
                # TOD this is very slow ...
                vectorstore.add_documents(docs) # this should be the text_splitter  very slow at the moment
                # vectorstore
                print('vectorstore : ',vectorstore)
                vectorstore.persist()
                # pages = loader.load_and_split(text_splitter=all_split)
                # vectorstore= Redis.from_texts_return_keys(texts=text_splitter, embedding=OllamaEmbeddings(), redis_url=redis_url,schema=vector_schema)
    except Exception as e:
        print('something went wrong loading pdfs\n\n', e)
    return True

# def similarity_search_query():
#     pass
# Learn a new fact
def store_new_information(text) :
    current_datetime = datetime.now()
    source = "Informed by user on " + current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    document = Document(page_content=text, metadata={"source": source})
    vectorstore.add_documents([document])
    vectorstore.persist()
    return

class SourcedAnswer(BaseModel):
    answer: str
    sources: List

# reason based on existing documents from vector store 
def prompt_query(question) :
    qa_chain = RetrievalQA.from_chain_type(
        chatllm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": CHAIN_PROMPT_TEMPLATE},
        return_source_documents=True
    )

    answer =  qa_chain({"query": question})
    
    sourced_answer = SourcedAnswer(
        answer=answer["result"],
        sources = answer["source_documents"]
    )
    return sourced_answer


store_new_information('+27 11 797 4003 this is absa new number ')
# learn_from_existing_documents()
prompt_query('does absa have a new number and what is it?')