from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain import VectorDBQA,HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
# from const import INDEX_NAME

import pinecone

pinecone.init(      
	api_key='44ad9fba-6b06-40f1-b938-e2adfd1387c2',      
	environment='us-west1-gcp-free'      
)   
INDEX_NAME="medblogs"

def run_llm(query,chat_history = []):
    embeddings=HuggingFaceHubEmbeddings()
    docsearch=Pinecone.from_existing_index(index_name=INDEX_NAME,embedding=embeddings)
    repo_id='google/flan-t5-xxl'
    llm=HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature":0.5,}
    )

    # qa=RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=docsearch.as_retriever(),return_source_documents=True)
    qa=ConversationalRetrievalChain.from_llm(llm=llm,retriever=docsearch.as_retriever(),return_source_documents=True)
    result=qa({"question":query,"chat_history":chat_history})
    return result

if __name__ == "__main__":
    query="Link for create_sql_agent ?"
    answer=run_llm(query=query)
    print("-"*80)
    print('Question:')
    print(query)
    print('Answer:')
    print(answer['result'])
    print("-"*80)
