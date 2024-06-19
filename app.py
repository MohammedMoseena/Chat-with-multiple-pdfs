'''
Streamlit's capabilities to create a user-friendly interface where users can upload PDF files, ask questions, and receive answers based on the content of those PDFs using AI-powered question-answering mechanisms.
'''
import streamlit as st 

import os

#provides the essential functionality to read and extract text from PDF files, 
from PyPDF2 import PdfReader 

# Helps to convert large text to smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

'''
The GoogleGenerativeAIEmbeddings class encapsulates methods and functionalities to convert raw text inputs into dense vector representations (embeddings). These embeddings capture semantic meanings and relationships between words and sentences, enabling more sophisticated analysis and processing of text data.
'''
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# integrating with Google's Generative AI services
import google.generativeai as genai

#FAISS-Facebook AI Similarity Search
from langchain.vectorstores import FAISS

# integrating advanced conversational AI capabilities from Google's Generative AI models
from langchain_google_genai import ChatGoogleGenerativeAI

#integration of a robust question-answering pipeline.
from langchain.chains.question_answering import load_qa_chain

'''
Structured prompts help AI models interpret and process natural language inputs more effectively. They ensure that the model receives relevant information in a consistent format, which can lead to more accurate responses and better user experiences in applications such as chatbots or question-answering systems.

'''

from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

'''
 os module facilitates loading and accessing environment variables, which are crucial for configuring API keys securely without exposing them directly in the code and  it sets the API key required to authenticate and access Google's Generative AI services.
'''
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Read PDF and go through each page and get the text in each page
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") 

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}\n
    Question: {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform a similarity search on the user question
    docs = new_db.similarity_search(user_question)
    
    # If no similar document is found, try a more generic search
    if not docs:
        docs = new_db.similarity_search(user_question.split()[-1])
    
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs using Gemini 👩‍💻")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on the submit button", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs is not None:
                with st.spinner("Processing...."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
