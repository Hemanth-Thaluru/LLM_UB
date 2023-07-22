
import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain import VectorDBQA,HuggingFaceHub


import pinecone

pinecone.init(      
	api_key='44ad9fba-6b06-40f1-b938-e2adfd1387c2',      
	environment='us-west1-gcp-free'      
)      
index = pinecone.Index('medblogs')

def ingest_doc():
    loader = ReadTheDocsLoader(path="/Volumes/TAPPS/LLM_UB/website_info_llm/website_info/langchain-docs/api.python.langchain.com/en/latest/agents")
    raw_data = loader.load()
    print(len(raw_data))
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50, separators=['\n\n','\n'," ",""])
    documents=text_splitter.split_documents(documents=raw_data)
    print(f'Length of all chunks {len(documents)}')

    for doc in documents:
        old_path=doc.metadata['source']
        v=old_path.split("/")[-1]
        new_path=f'https://api.python.langchain.com/en/latest/agents/{v}'
        doc.metadata['source']=new_path

    for doc in documents:
        old_path=doc.metadata['source']
        print(old_path)
        break

    repo_id='google/flan-t5-xxl'
    llm=HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature":0.5,}
    )


    embeddings=HuggingFaceHubEmbeddings()
    docsearch=Pinecone.from_documents(documents,embeddings,index_name="medblogs")
    qa=VectorDBQA.from_chain_type(llm=llm,chain_type='stuff',vectorstore=docsearch)

    query="What is the base class for parsing agent output into agent action/finish."
    result=qa({"query":query})
    print(result)



if __name__ =="__main__":
    ingest_doc()