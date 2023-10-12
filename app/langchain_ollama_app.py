import os
from typing import List
from pydantic import BaseModel
from datetime import datetime
from PyPDF2 import PdfReader
import chromadb
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain.embeddings import OllamaEmbeddings
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

class Sources(BaseModel):
    answer: str
    sources: List

class largeLearningModel():
    def __init__(self,role='sales') -> None:
        self._roles_=['sales','support']
        self.role =self.set_role(role)
        self.data_stores= {'sales':'absa_gpt','support':'IT_support_gpt'}
        self.collection_name=self.data_stores[self.role]
        self.collection_directory    = "data/chroma"
        self.chatModel="llama2"
        self.unprocessed_data="../data/unprocessed"
        self.llm_host="http://localhost:11434"
        self.embeddings = OllamaEmbeddings()

        # Set up the vector database and client we will use
        self.persistent_client = chromadb.PersistentClient(path=self.collection_directory)
        self.collection = self.persistent_client.get_or_create_collection(self.collection_name)
        self.vectorstore = Chroma(client=self.persistent_client,
                            collection_name=self.collection_name,
                            embedding_function=self.embeddings, 
                            persist_directory=self.collection_directory)

        # Set up out LLM (Ollama) for streaming callback
        chatllm = Ollama(base_url=self.llm_host,
                    model=self.chatModel,
                    verbose=False,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        #  Hierarchical Navigable Small World , vs KNN  HNSW is superior in terms of performance and our usecases  300X improvement
        # vector_schema = {
        #     "algorithm": "HNSW"
        # }

    def get_role(self):
        return self.role
    def set_role(self, role='sales'):
        self.role =role if role in self._roles_ else self._roles_[0]
        return   self.role     

    def get_template(self):
        #  prompt engineering
        support_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum and keep the answer as concise as possible.
        {context}
        Question: {question}
        Helpful step-by-step Answer: 
        1. {step1}
        2. {step2}  
        3. {step3}
        """

        default_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum and keep the answer as concise as possible. 
        {context}
        Question: {question}
        Helpful Answer:"""

        templates ={'support':support_template,'sales':default_template}
        self.template = templates[self.role]
        # prompt template
        CHAIN_PROMPT_TEMPLATE = PromptTemplate(
            input_variables=["context", "question"],
            template=self.template,
        )
        return CHAIN_PROMPT_TEMPLATE

    # Read an article and store it , confluence, absa.co.za etc
    def web_article_scraper(self,url) : 
        data = WebBaseLoader(url).load()
        text_splitter=RecursiveCharacterTextSplitter( chunk_size = 100, chunk_overlap  = 20, length_function = len, is_separator_regex = False).split_documents(data)
        ids = self.vectorstore.add_documents(text_splitter)
        self.vectorstore.persist()
        print(ids)
        return

    def learn_from_existing_documents(self):
        # read all docs in a directory and store them in a vector db
        try:
            for filename in os.listdir(self.unprocessed_data):
                docs=[]
                if filename.endswith('.pdf'):
                    file=f"{self.unprocessed_data}/{filename}"
                    reader =PdfReader(file)
                    i = 1
                    for page in reader.pages:
                        docs.append(Document(page_content=page.extract_text(), metadata={'page':i,'source':filename}))
                        i += 1
                        print(f'processing page :{i}')
                    text_splitter=RecursiveCharacterTextSplitter( chunk_size = 1000, chunk_overlap  = 0, length_function = len, is_separator_regex = False).split_documents(docs)
                    print(f'text_splitter : {len(text_splitter)} last page : {text_splitter[-1]}')
                    self.vectorstore.add_documents(text_splitter) # this should be the text_splitter  very slow at the moment
                    print('vectorstore : ',self.vectorstore)
                    self.vectorstore.persist()
                    # pages = loader.load_and_split(text_splitter=all_split)
                    # vectorstore= Redis.from_texts_return_keys(texts=text_splitter, embedding=OllamaEmbeddings(), redis_url=redis_url,schema=vector_schema)
        except Exception as e:
            print('something went wrong loading pdfs\n\n', e)
        return True

    def store_new_information(self,text) :
        current_datetime = datetime.now()
        source = "Informed by user on " + current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        document = Document(page_content=text, metadata={"source": source})
        self.vectorstore.add_documents([document])
        self.vectorstore.persist()
        return True


    # reason based on existing documents from vector store 
    def prompt_query(self,question) :
        qa_chain = RetrievalQA.from_chain_type(
            self.chatllm,
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": self.get_template()},
            return_source_documents=True
        )
        answer =  qa_chain({"query": question})
        sourced_answer = Sources(
            answer=answer["result"],
            sources = answer["source_documents"]
        )
        return sourced_answer



llm_sales = largeLearningModel('sales')
llm_support = largeLearningModel('support')

# store_new_information('+27 11 797 4003 this is fnb bank's new number ')
# learn_from_existing_documents()
llm_sales.prompt_query('does loosey goosey offer in person interactions?')
llm_support.prompt_query('why is my flash drive not working on a work laptop?')