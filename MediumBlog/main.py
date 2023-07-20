from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA,HuggingFaceHub
import pinecone

pinecone.init(      
	api_key='44ad9fba-6b06-40f1-b938-e2adfd1387c2',      
	environment='us-west1-gcp-free'      
)      
index = pinecone.Index('medblogs')


import os

if __name__=='__main__':
    print('Hello from Medium Langchain.....')
    loader=TextLoader('/Volumes/TAPPS/LLM_UB/MediumBlog/mediumBlog1.txt')
    document=loader.load()
    # print(document)

    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts=text_splitter.split_documents(document)
    print(len(texts))

    repo_id='google/flan-t5-xxl'
    llm=HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature":0.5,}
    )


    embeddings=HuggingFaceHubEmbeddings()
    docsearch=Pinecone.from_documents(texts,embeddings,index_name="medblogs")
    qa=VectorDBQA.from_chain_type(llm=llm,chain_type='stuff',vectorstore=docsearch)

    query="Explain about vector Data Base?"
    result=qa({"query":query})
    print(result)