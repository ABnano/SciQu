import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
import tempfile
import os
import random
import requests


import os
import streamlit as st
# Other imports...

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'bg.png.webp')

# Streamlit UI setup
st.set_page_config(page_title="BabuBot", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #fbc2eb, #a6c1ee);
        color: black;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stFileUploader {
        background-color: #ff9800;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Welcome to BabuBot")
st.image(image_path, use_column_width=True)  # Use the relative path
st.write("Upload a PDF file and ask questions to retrieve relevant document sections.")
# The rest of your script...

# Initialize components
local_model = "mistral"
llm = ChatOllama(model=local_model)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Streamlit UI setup
st.set_page_config(page_title="BabuBot", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #fbc2eb, #a6c1ee);
        color: black;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stFileUploader {
        background-color: #ff9800;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Welcome to BabuBot")
st.image("bg.png.webp", use_column_width=True)  # Replace with your image URL
st.write("Upload a PDF file and ask questions to retrieve relevant document sections.")

# Session state to store history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Colors for different questions and answers
colors = ["#ff4b4b", "#4bffa5", "#4bb3ff", "#f4ff4b", "#ffb84b"]

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.info("Uploading and processing PDF file...")
    progress_bar = st.progress(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name
    progress_bar.progress(10)
    
    try:
        # Process the PDF
        st.info("Processing the PDF...")
        loader = UnstructuredPDFLoader(file_path=temp_file_path)
        data = loader.load()
        progress_bar.progress(30)
        
        st.info("Splitting the document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        progress_bar.progress(50)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        progress_bar.progress(60)
        
        st.info("Creating vector database from document chunks...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
            collection_name="local-rag"
        )
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT
        )
        progress_bar.progress(80)
        
        st.success("PDF file processed successfully!")
        progress_bar.progress(100)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        progress_bar.progress(0)

    # Query input
    query = st.text_input("Ask a question about the document")

    if query:
        st.info("Retrieving information...")
        
        # Determine if the query is related to the Materials Project
        if "materials project" in query.lower():
            st.write("Retrieving additional information from the Materials Project...")
            api_key = "UOC1KWzQHn3ZtIYsU30ykzNBqXRWWn6X"  # Make sure to replace with your actual API key
            response = requests.get(
                f"https://materialsproject.org/rest/v2/materials/{query}/vasp?API_KEY={api_key}"
            )
            if response.status_code == 200:
                materials_data = response.json()
                st.write("Materials Project Data:")
                st.json(materials_data)
            else:
                st.write("Failed to retrieve data from the Materials Project API.")
        else:
            # Process the query with the local RAG model
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            result = chain.invoke({"question": query})
            st.session_state['history'].append((query, result))
            st.write("Answer:")
            st.write(result)

# Display history
if st.session_state['history']:
    st.write("## Question and Answer History")
    for q, a in st.session_state['history']:
        color = random.choice(colors)
        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 10px;'>"
                    f"<strong>Question:</strong> {q}<br>"
                    f"<strong>Answer:</strong> {a}</div>", unsafe_allow_html=True)
        st.write("---")
