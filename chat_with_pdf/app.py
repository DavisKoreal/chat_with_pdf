import streamlit as st
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os

load_dotenv()

with st.sidebar:
    st.title("Random LLM CHAT APP")
    st.markdown('''## This is a demo of a chat with LLM app that can
                be used as a module to another subsequent project.
                
                enjoy
                
                ''')
    add_vertical_space(5)
    st.write('''Definitely made with love for tinkering,
              learning, and fun(Fooling around)''')
    

def main():
    st.header("Welcome to the LLM chat app")
    #we get a file from the user
    pdf = st.file_uploader("Upload a pdf file", type="pdf")
    if pdf is not None:
        st.write("pdf file uploaded successfully")
        pdf_reader = PdfReader(pdf)


        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
            )
        chunks = text_splitter.split_text(text)



        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
    
        #accept user input
        query = st.text_input("Enter your question to chat with the pdf")
        if query:
            docs = VectorStore.similarity_search(query, k=3)
            llm = OpenAI(model_name= 'gpt-3.5-turbo')
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(docs, query)
            st.write(response)
    


if __name__ == "__main__":
    main()