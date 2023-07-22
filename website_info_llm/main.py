
from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Chat Bot for Buffalo University Questions")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"]= []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"]= []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"]= []

def modify_source_urls(source_urls):
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

prompt=st.text_input("Prompt",placeholder="Enter Something here")
# st.text_area(pr)
if prompt or st.session_state["chat_history"]:
    with st.spinner('Loading for you'):
        generated_response=run_llm(query=prompt,chat_history= st.session_state["chat_history"])
        sources=set(
            [doc.metadata['source'] for doc in generated_response['source_documents']]
        )
        formatted_answer=(f'{generated_response["answer"]} \n\n {modify_source_urls(sources)}')
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_answer)
        st.session_state["chat_history"].append((prompt,generated_response['answer']))

    if st.session_state["user_prompt_history"]:
        for user_query,generated_response_each in reversed(list(zip(
            st.session_state["user_prompt_history"],
            st.session_state["chat_answer_history"]
        ))):
            message = st.chat_message("user")
            # message(user_query,is_user=True)
            message.write(user_query)
            message = st.chat_message("assistant")
            message.write(generated_response_each)
            # message(generated_response_each)
prompt=""



