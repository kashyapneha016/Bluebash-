import tempfile
import psycopg2
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

# Set up your Google API key
google_api_key = "your_api_key"
genai.configure(api_key=google_api_key)

# Configure model
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0.2, convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# Connect to PostgreSQL database
def get_db_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",  
        port="5432"
    )

# Create table in PostgreSQL if not exists
def create_table():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdf_data (
                id SERIAL PRIMARY KEY,
                pdf_text TEXT,
                embedding REAL[]
            )
        """)
        conn.commit()
    conn.close()

# Insert text and embeddings into PostgreSQL
def insert_pdf_data(text, embedding):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO pdf_data (pdf_text, embedding)
            VALUES (%s, %s)
        """, (text, embedding))
        conn.commit()
    conn.close()

# Function to process PDF and embeddings
def process_pdf(uploaded_file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load PDF and extract text using the temporary file path
    pdf_loader = PyPDFLoader(tmp_file_path)
    pages = pdf_loader.load_and_split()

    # Text Splitter and embeddings generation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    # Store in PostgreSQL
    create_table()
    for text in texts:
        embedding = embeddings.embed_documents([text])[0]
        insert_pdf_data(text, embedding)

    return texts

# Function for question answering
def answer_question(question, texts):
    # Set up vector search for Q&A retrieval
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(model, retriever=vector_index, return_source_documents=True)
    result = qa_chain({"query": question})

    return result

# Set up Streamlit UI
st.title("PDF Document Q&A System")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Process PDF and generate embeddings
    texts = process_pdf(uploaded_file)

    st.success("PDF text and embeddings saved to database.")

    # Show page content for user to review
    st.subheader("Review PDF Content")
    page_num = st.slider("Select page number", min_value=1, max_value=len(texts), value=1)
    st.write(texts[page_num - 1])

    # Question & Answer Section
    st.subheader("Ask a Question")

    question = st.text_input("Enter your question:")

    if question:
        # Get answer from the main logic
        result = answer_question(question, texts)

        st.write("Answer: ", result["result"])
        st.write("Sources: ", result["source_documents"])
else:
    st.write("Please upload a PDF to get started.")
