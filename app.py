import streamlit as st
import os
from PyPDF2 import PdfReader
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# Constants
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAINED_EMBEDDING_PATH = "trained_models/embedding"

# Initialize session state
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = None
if 'pipe' not in st.session_state:
    st.session_state.pipe = None

# Text processing functions
def extract_text(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,:;!?()\'\"\""\\\–\\\—\s]', '', text)
    return text.lower().strip()

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    return text_splitter.split_text(text)

def process_pdf(pdf_file):
    # Handle both file objects and bytes
    if hasattr(pdf_file, 'getvalue'):
        pdf_content = pdf_file.getvalue()
    else:
        pdf_content = pdf_file

    # Save content temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_content)
        tmp_path = tmp_file.name

    # Process the PDF
    raw_text = extract_text(tmp_path)
    if not raw_text:
        return None, None

    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    
    # Clean up temporary file
    os.unlink(tmp_path)
    
    return chunks, chunks

def create_faiss_index(text_chunks, embedding_model_name):
    if not text_chunks:
        st.error("No text chunks available to create FAISS index.")
        return None, None

    try:
        # Try to load trained model first
        if os.path.exists(TRAINED_EMBEDDING_PATH):
            embedding_model = SentenceTransformer(TRAINED_EMBEDDING_PATH)
            st.info("Using fine-tuned embedding model")
        else:
            embedding_model = SentenceTransformer(embedding_model_name)
            st.info("Using base embedding model")
            
        embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        return index, embedding_model
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None, None

def retrieve_context(query, faiss_index, embedding_model, text_chunks, top_k=3):
    if faiss_index is None or embedding_model is None:
        return ""
    
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding).astype('float32'), top_k)
    
    retrieved_chunks = []
    seen_texts = set()
    for idx in indices[0]:
        chunk = text_chunks[idx]
        if chunk not in seen_texts:
            retrieved_chunks.append(chunk)
            seen_texts.add(chunk)
    
    return "\n\n".join(retrieved_chunks)

# Streamlit UI
st.title("PDF Question Answering System")
st.write("Upload a PDF and ask questions about its content.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Process PDF when uploaded
    with st.spinner("Processing PDF..."):
        text_chunks, retrieval_chunks = process_pdf(uploaded_file)
        if text_chunks:
            st.session_state.text_chunks = retrieval_chunks
            st.session_state.faiss_index, st.session_state.embedding_model = create_faiss_index(
                retrieval_chunks, EMBEDDING_MODEL_NAME
            )
            st.success("PDF processed successfully!")

# Initialize the pipeline if not already done
if st.session_state.pipe is None:
    with st.spinner("Loading language model..."):
        try:
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_NAME,
                trust_remote_code=True
            )
            
            # Load model with specific configuration
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Create pipeline with loaded components
            st.session_state.pipe = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                trust_remote_code=True
            )
            st.info("Using TinyLlama model")
        except Exception as e:
            st.error(f"Error loading language model: {e}")
            st.session_state.pipe = None

# Question input
question = st.text_input("Ask a question about the PDF content:")

if question and st.session_state.faiss_index is not None:
    with st.spinner("Generating answer..."):
        # Retrieve context
        context = retrieve_context(
            question,
            st.session_state.faiss_index,
            st.session_state.embedding_model,
            st.session_state.text_chunks,
            top_k=3
        )

        # Create prompt
        messages = [
            {
                "role": "system",
                "content": "You are an expert assistant. Given the context and question below, give a concise, clear answer. If context is incomplete, use your general knowledge. Always answer in 1-2 sentences.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ]

        prompt = st.session_state.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate answer with simplified parameters
        outputs = st.session_state.pipe(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
        )

        # Extract and display only the answer
        generated_text = outputs[0]["generated_text"]
        # Find the last assistant response
        if "<|assistant|>" in generated_text:
            answer = generated_text.split("<|assistant|>")[-1].strip()
        else:
            answer = generated_text.strip()
            
        # Display answer in a nice format
        st.markdown("### Answer")
        st.markdown(answer) 