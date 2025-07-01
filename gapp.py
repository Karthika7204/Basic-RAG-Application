import gradio as gr
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

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# PDF Reader
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Text chunking
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Gemini Vision image summarization
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

# Create and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load Gemini QA chain
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

# Vector-based answer generation
def generate_answer(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

# Full processing function
def process_inputs(pdf_files, image_files):
    raw_text = get_pdf_text(pdf_files) if pdf_files else ""
    image_summaries = get_image_summaries(image_files) if image_files else []
    full_text = raw_text + "\n".join(image_summaries)
    chunks = get_text_chunks(full_text)
    get_vector_store(chunks)
    return "✅ Documents processed and indexed!"

with gr.Blocks(title="Multimodal RAG - Gemini Q&A", css="footer {display: none;}") as demo:
    gr.Markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h1 style="color: #4F46E5;">📚 Multimodal RAG Q&A 🤖</h1>
            <p style="font-size: 16px;">
                Upload <strong>PDFs</strong> and <strong>Images</strong>, then ask any question about them.<br>
                Powered by <span style="color:#10b981;"><strong>Gemini + FAISS</strong></span> for smart document understanding.
            </p>
        </div>
    """)

    with gr.Group():
        gr.Markdown("### 📤 Upload your Documents")
        with gr.Row():
            pdf_input = gr.File(label="📎 Upload PDF files", file_types=[".pdf"], file_count="multiple")
            img_input = gr.File(label="🖼️ Upload Image files", file_types=[".png", ".jpg", ".jpeg"], file_count="multiple")
        
        process_button = gr.Button("🚀 Submit & Process", variant="primary", size="lg")
        process_output = gr.Textbox(label="📢 Status", placeholder="Waiting for input...", interactive=False)

    gr.Markdown("---")

    with gr.Group():
        gr.Markdown("### 💬 Ask a Question")
        with gr.Row():
            question_input = gr.Textbox(label="Your Question", placeholder="e.g. What is the budget mentioned in the document?")
        
        answer_output = gr.Textbox(label="🔍 Answer", placeholder="Answer will appear here...", lines=6)

    process_button.click(fn=process_inputs, inputs=[pdf_input, img_input], outputs=process_output)
    question_input.submit(fn=generate_answer, inputs=question_input, outputs=answer_output)

# Run app
if __name__ == "__main__":
    demo.launch()

