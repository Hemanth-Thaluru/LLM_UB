from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain import VectorDBQA,HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA



if __name__=='__main__':
    print('Hello from Medium Langchain.....')
    pdf_path="/Volumes/TAPPS/LLM_UB/local_pdf_llm/1706.03762.pdf"
    loader=PyPDFLoader(file_path=pdf_path)
    documents=loader.load()
    # print(document)

    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator='\n')
    texts=text_splitter.split_documents(documents)
    print('Hello')
    print(len(texts))

    repo_id='bigscience/bloom'
    llm=HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature":0.5,}
    )


    embeddings=HuggingFaceHubEmbeddings()
    docsearch=FAISS.from_documents(texts,embeddings)
    docsearch.save_local('attention_llm_faiss')
    new_vector_store=FAISS.load_local("attention_llm_faiss",embeddings=embeddings)

    qa=RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=new_vector_store.as_retriever())
    result=qa.run("Authors ?")
    print(result)
    # query="Explain about vector Data Base?"
    # result=qa({"query":query})
    # print(result)