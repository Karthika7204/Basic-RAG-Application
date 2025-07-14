import pandas as pd
import streamlit as st
import os
import fitz  
import io
from PyPDF2 import PdfReader
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from faster_whisper import WhisperModel
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.docstore.document import Document


#api configuration
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#Read multiple pdf using PyPDF2 reader and extract the text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_images_from_pdfs(pdf_docs):
    images = []

    for pdf_file in pdf_docs:
        try:
            file_bytes = pdf_file.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            for page_index in range(len(doc)):
                try:
                    page = doc.load_page(page_index)  # safer than doc[page_index]
                    image_list = page.get_images(full=True)

                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        images.append(image)

                except Exception as e:
                    print(f"Skipping corrupted page {page_index}: {e}")
                    continue

        except Exception as e:
            print(f"Skipping invalid PDF: {e}")
            continue

    return images


#Captioning the image using gemini-modal-1.5-flash 
def get_image_summaries(image_objects):
    model = genai.GenerativeModel("gemini-1.5-flash")
    summaries = []

    for image in image_objects:
        response = model.generate_content([
            image,
            "Describe the contents of this image. Focus on any text, tables, or visual elements that can be used for question answering."
        ])
        summaries.append(response.text)

    return summaries


def get_audio_text(audio_files):
    model = WhisperModel("tiny", device="cpu")
    combined_text = ""

    # Make sure the folder exists
    audio_folder = "audio_uploads"
    os.makedirs(audio_folder, exist_ok=True)

    for audio_file in audio_files:
        file_path = os.path.join(audio_folder, audio_file.name)

        # Save the file to the folder
        with open(file_path, "wb") as f:
            f.write(audio_file.read())

        # Transcribe the saved file
        segments, _ = model.transcribe(file_path)
        audio_text = " ".join([segment.text for segment in segments])
        combined_text += audio_text + "\n"

        # Optionally delete after use:
        os.remove(file_path)

    return combined_text


#Convert the raw text data into managable chunks using RecursiveCharacterTextSplitter
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Converts chunks to vector embeddings using Gemini embeddings and Store it in the FAISS db
def get_vector_store(chunks, source_type):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docs = [Document(page_content=chunk, metadata={"source": source_type}) for chunk in chunks]
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    vector_store.save_local("faiss_index")

#Creates a QA chain using a Gemini chat model and a custom prompt.
def get_conversational_chain():
    prompt_template = """
    You will be given some context and possibly multiple questions. Answer each question as detailly,clearly and separately as possible based on the context. 
    If an answer is not available, say "Not available in the context."

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
    st.set_page_config(page_title="Multimodal Rag")
    st.header("Multimodal RAG: PDF,Image and Audio Q&A", divider='rainbow')

    user_question = st.text_input("Ask a Question from PDF/Image Documents")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload Documents")
        pdf_docs = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
        image_files = st.file_uploader("Upload Image Files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        audio_files = st.file_uploader("Upload Audio Files", type=["mp3"], accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):

                #Process PDF text
                raw_text = get_pdf_text(pdf_docs) if pdf_docs else ""

                #Process image text
                uploaded_images = [Image.open(img) for img in image_files] if image_files else []
                pdf_extracted_images = extract_images_from_pdfs(pdf_docs) if pdf_docs else []
                all_images = uploaded_images + pdf_extracted_images

                # Caption all images together
                image_summaries = get_image_summaries(all_images) if all_images else []


                #Process audio text
                audio_text = get_audio_text(audio_files) if audio_files else ""

                #Combine all text and image content
                full_text = raw_text + "\n".join(image_summaries) + audio_text
                text_chunks = get_text_chunks(full_text)

                #Store in FAISS
                get_vector_store(text_chunks, source_type="multimodal")
                st.success("Documents indexed successfully!")

if __name__ == "__main__":
    main()
