import streamlit as st
import os
import datetime
from dotenv import load_dotenv

# Importing from your project modules
from data_loader import load_data
from vector_db import create_vector_db, add_documents
from chains import get_chain
from text_splliter import split_text_into_chunks
from retriever import retrieve_with_mmr

load_dotenv()

# --- Configuration & Setup ---
BOOKS_DIR = "uploaded_books"
RESPONSES_DIR = "saved_responses"

# Ensure directories exist for saving files
os.makedirs(BOOKS_DIR, exist_ok=True)
os.makedirs(RESPONSES_DIR, exist_ok=True)

st.set_page_config(page_title="Philosophical RAG Assistant", page_icon="📚")
st.title("Philosophical RAG Assistant 📚")

# --- Sidebar: Document Upload & Processing ---
with st.sidebar:
    st.header("1. Upload Books")
    st.write("Upload PDF documents to build the knowledge base.")
    
    # Accept multiple files
    uploaded_files = st.file_uploader("Select PDF books", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Save Books"):
        if uploaded_files:
            with st.spinner("Processing books and updating database..."):
                # Initialize the vector DB using your existing logic
                vector_db = create_vector_db()
                
                processed_any = False
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(BOOKS_DIR, uploaded_file.name)
                    
                    # --- NEW FEATURE: Check if book is already processed ---
                    if os.path.exists(file_path):
                        st.info(f"Skipped: `{uploaded_file.name}` (Already processed)")
                        continue # Skip the rest of the loop for this specific file
                    
                    # Save the new uploaded book to the local directory
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Execute your project's data processing pipeline
                    st.write(f"Embedding `{uploaded_file.name}`...")
                    data = load_data(file_path)
                    chunks = split_text_into_chunks(data)
                    add_documents(vector_db, chunks)
                    processed_any = True
                    
                if processed_any:
                    st.success("New books successfully saved and embedded!")
                else:
                    st.success("All selected books are already in the database.")
        else:
            st.warning("Please upload at least one PDF file first.")

    if st.button("Clear Knowledge Base"):
        if os.path.exists(BOOKS_DIR):
            for filename in os.listdir(BOOKS_DIR):
                file_path = os.path.join(BOOKS_DIR, filename)
                os.remove(file_path)
            try:
                vector_db.clear_collection()  # Clear the vector DB collection as well
                st.success("All uploaded books have been cleared.")
            except Exception as e:
                st.error(f"No books found in the vector database to clear")
        else:
            st.warning("No uploaded books found to clear.")

# --- Main Area: Query & Response ---
st.header("2. Ask a Question")
query = st.text_area("Enter your philosophical question based on the uploaded books:")

if st.button("Generate Answer"):
    if query:
        with st.spinner("Consulting the texts..."):
            try:
                # Initialize retrieval and the LangChain pipeline
                vector_db = create_vector_db()
                retriever = retrieve_with_mmr(vector_db)
                chain = get_chain()
                
                # Retrieve documents and generate the response
                retrieved_docs = retriever.invoke(query)
                response = chain.invoke({
                    "context": retrieved_docs,
                    "question": query
                })
                
                # Display the response
                st.markdown("### Answer")
                st.write(response)
                
                # --- NEW FEATURE: Display Retrieved Text ---
                st.markdown("### Sources")
                with st.expander("Click to view the exact text retrieved from the books"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i + 1}:**")
                        # The actual text is stored in the page_content attribute
                        st.write(doc.page_content) 
                        st.divider() # Adds a neat visual line between chunks
                        
                # Save the response to the directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                response_filename = f"response_{timestamp}.txt"
                response_filepath = os.path.join(RESPONSES_DIR, response_filename)
                if st.button("Save Response"):
                    with open(response_filepath, "w", encoding="utf-8") as f:
                        f.write(f"Date: {datetime.datetime.now()}\n")
                        f.write(f"Question: {query}\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Answer:\n{response}\n")
                        
                    st.success(f"Response successfully saved to `{response_filepath}`")
                
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
    else:
        st.warning("Please enter a question to get an answer.")