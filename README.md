# PDF Document Q&A System

This application allows users to upload a PDF document, extract text from it, generate embeddings for the text, and store the text and embeddings in a PostgreSQL database. Users can then ask questions about the document, and the system will retrieve relevant information from the document to provide answers.

## Tools and Frameworks Used

### 1. **Streamlit**
   - **Purpose**: Used for building the web interface of the application. Streamlit provides an easy way to create interactive web applications with minimal coding.
   - **Documentation**: [Streamlit](https://streamlit.io/)

### 2. **Langchain**
   - **Purpose**: Used for handling the document processing and natural language processing (NLP) workflows. Langchain is leveraged for splitting the text and creating embeddings.
   - **Key Components**:
     - `PyPDFLoader`: To load and extract text from PDF files.
     - `RecursiveCharacterTextSplitter`: To split the document text into manageable chunks for embeddings.
     - `Chroma`: To perform vector-based searches for question answering.
     - `RetrievalQA`: To perform question answering by retrieving relevant sections from the documents.

### 3. **Google Generative AI (Gemini)**
   - **Purpose**: Used for generating text embeddings and answering questions based on the document's content.
   - **Key Components**:
     - `GoogleGenerativeAIEmbeddings`: For generating embeddings of the text from the document.
     - `ChatGoogleGenerativeAI`: For interacting with the model to answer questions.

### 4. **PostgreSQL**
   - **Purpose**: Used to store the extracted text and its corresponding embeddings. PostgreSQL provides a reliable way to store the data for later use and retrieval.
   - **Library**: `psycopg2` for interacting with the PostgreSQL database.

### 5. **Python**
   - **Purpose**: The primary programming language used to build the application.
   - **Libraries**:
     - `tempfile`: To handle temporary files for storing uploaded PDFs.
     - `psycopg2`: PostgreSQL adapter to connect and interact with the database.
     - `langchain`: For document processing and embeddings.

## How to Set Up and Run the Application

### 1. **Install Dependencies**
   Clone this repository and install the required dependencies. You can create a virtual environment for better isolation:

   ```bash
   git clone <repository-url>
   cd <project-folder>
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   pip install -r requirements.txt
