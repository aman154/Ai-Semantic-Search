import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import Annoy  # Replace with your preferred vector store
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_extras.colored_header import colored_header
from streamlit_chat import message
import os

os.environ['GOOGLE_API_KEY'] = 'AIzaSyD3INqv4Ogur_JuAOFf44ScBWdAumveQI8'

# Define a password for document upload
DOCUMENT_UPLOAD_PASSWORD = "aman"  # Change this to your desired password

def authenticate_document_upload():
    password_input = st.sidebar.text_input("Enter Document Upload Password:", type="password")
    return password_input == DOCUMENT_UPLOAD_PASSWORD

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = Annoy.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, creater in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
           st.write("👤 Human:", creater.content)
            #message(creater.content,key=str(i),avatar_style="initials")
        else:
            st.write("🤖 Bot:", creater.content)
            #message(creater.content,avatar_style="icon", is_user=True
                #, key=str(i) + 'data_by_user',seed="user seed",icon_name="fa-robot")

def main():
    st.markdown(""" <style>
    MainMenu {visibility: hidden;} 
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)
    #st.set_page_config("Query_Chat")
    st.title(":orange[AI Semantic Search]")
    colored_header(label='', description='', color_name='gray-30')
    padding = 0
    st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

    # User input without password
    user_question = st.text_input("Ask a Question from the PDF Files")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload your Documents")

        # Password-protected document upload
        if authenticate_document_upload():
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Done")

if __name__ == "__main__":
    main()
