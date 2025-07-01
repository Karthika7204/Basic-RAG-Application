import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

#api configuration
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

#Read multiple pdf using PyPDF2 reader and extract the text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#Convert the raw text data into managable chunks using RecursiveCharacterTextSplitter
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Captioning the image using gemini-modal-1.5-flash 
def get_image_summaries(image_files):
    model = genai.GenerativeModel("gemini-1.5-flash")
    summaries = []

    for image_file in image_files:
        image = Image.open(image_file)
        response = model.generate_content([
            image,
            "Describe the contents of this image. Focus on any text, tables, or visual elements that can be used for question answering."
        ])
        summaries.append(response.text)

    return summaries

#Converts chunks to vector embeddings using Gemini embeddings and Store it in the FAISS db
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

#Creates a QA chain using a Gemini chat model and a custom prompt.
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say "Answer is not available in the context." Don't make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

#Processing the user query and retrive the related context
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

#Sets up the Streamlit UI.
def main():
    st.set_page_config(page_title="Chat PDF + Image RAG")
    st.header("Multimodal RAG: PDF and Image Q&A", divider='rainbow')

    user_question = st.text_input("Ask a Question from PDF/Image Documents")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload Documents")
        pdf_docs = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
        image_files = st.file_uploader("Upload Image Files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):

                # Process PDF text
                raw_text = get_pdf_text(pdf_docs) if pdf_docs else ""

                # Process image text
                image_summaries = get_image_summaries(image_files) if image_files else []

                # Combine all text and image content
                full_text = raw_text + "\n".join(image_summaries)
                text_chunks = get_text_chunks(full_text)

                # Store in FAISS
                get_vector_store(text_chunks)
                st.success("Documents indexed successfully!")

if __name__ == "__main__":
    main()
