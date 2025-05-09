34r

import fitz
print(fitz.__version__)

import os
import re
import fitz  # PyMuPDF
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter





# --- Configuration ---
# PDF_DIRECTORY = "/home/ubuntu/upload/PDF files/"

PDF_DIRECTORY = "PDF files/Digital/"
# FAISS_INDEX_PATH = "/home/ubuntu/faiss_index.idx"
FAISS_INDEX_PATH = "faiss_index.idx"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = LLM_MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# pdf_files = [f for f in os.listdir() if f.endswith('.pdf')]
# print("PDF files found:", pdf_files)
pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
print("PDF files found:", pdf_files)

# --- Text Processing Functions (from user's notebook) ---
def extract_text(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
    return text.strip()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    # text = re.sub(r'[^a-zA-Z0-9.?!\s]', '', text)  # Keep spaces for sentence structure
    # text = re.sub(r'[^a-zA-Z0-9.,:;!?()\'\"\“\”-–—\s]', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,:;!?()\'\"\“\\\”\\\–\\\—\s]', '', text)

    return text.lower().strip()

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    return text_splitter.split_text(text)

"""Adjusted for a specific number of pdfs only"""

# --- Core RAG Pipeline Functions ---
def load_and_process_pdfs(pdf_dir, max_files=None):
    all_chunks = []
    if not os.path.exists(pdf_dir):
        print(f"Error: PDF directory not found at {pdf_dir}")
        print("Please ensure PDF files are uploaded to the correct location.")
        return [], []

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return [], []

    # Limit the number of PDFs if max_files is specified
    if max_files is not None:
        pdf_files = pdf_files[:max_files]

    print(f"Found PDF files: {pdf_files}")
    processed_chunks_for_retrieval = []  # Store chunks for FAISS

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Processing {pdf_path}...")
        raw_text = extract_text(pdf_path)
        if not raw_text:
            continue
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)
        processed_chunks_for_retrieval.extend(chunks)
        print(f"Extracted {len(chunks)} chunks from {pdf_file}.")

    print(f"Total chunks from all PDFs: {len(processed_chunks_for_retrieval)}")
    return processed_chunks_for_retrieval, processed_chunks_for_retrieval

def create_faiss_index(text_chunks, embedding_model_name, index_path):
    if not text_chunks:
        print("No text chunks available to create FAISS index.")
        return None, None
    print(f"Loading embedding model: {embedding_model_name}")
    try:
        embedding_model = SentenceTransformer(embedding_model_name)
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None, None

    print("Generating embeddings for text chunks...")
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    print("FAISS index created and saved successfully.")
    return index, embedding_model

def load_llm_and_tokenizer(model_name):
    print(f"Attempting to load tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer for {model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        return None, None

    print(f"Attempting to load model {model_name}...")
    try:
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            print(f"Model {model_name} loaded successfully on available CUDA device(s).")
        else:
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_name,
            #     torch_dtype=torch.float16,
            # )
            # model = AutoModelForCausalLM.from_pretrained
            #  (
            #   model_name,
            #   torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            #   )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

            print(f"Model {model_name} loaded successfully on CPU.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

# def create_prompt_template(question, context):
# #     template = f"""You are a helpful assistant. Use the following context to answer the question accurately and concisely. If you don't know, say "I don't know."

# # Context:
# # {context}

# # Question: {question}

# # Answer:"""



import torch
from transformers import pipeline

# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True, device_map="auto")

def create_prompt_template(question, context, tokenizer, max_tokens=2048):
    messages = [
        {
            "role": "system",
            "content": "You are an expert assistant. Given the context and question below, give a concise, clear answer. If context is incomplete, use your general knowledge. Always answer in 1-2 sentences.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ]


    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    # Truncate raw context before tokenizing if too long
    # prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    # return prompt, prompt_tokens
    return prompt

def retrieve_context(query, faiss_index, embedding_model, text_chunks, top_k=3):
    if faiss_index is None or embedding_model is None:
        print("FAISS index or embedding model not available for retrieval.")
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

    context = "\n\n".join(retrieved_chunks)
    print(f"Retrieved context (first 500 chars): {context[:500]}...")
    return context

def generate_answer(prompt, llm_model, llm_tokenizer, short=False):
    if llm_model is None or llm_tokenizer is None:
        return "Sorry, I am unable to generate an answer at this time."

    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_sequences = llm_model.generate(
        **inputs,
        max_new_tokens=150,
        num_return_sequences=1,
        pad_token_id=llm_tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=llm_tokenizer.eos_token_id
    )
    full_generated_text = llm_tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    # Extract answer after "Answer:"
    answer = ""
    if "Answer:" in full_generated_text:
        answer = full_generated_text.split("Answer:")[-1].strip()
    else:
        answer = full_generated_text.strip()

    # Shorten to 1-2 sentences
    if short:
        sentences = re.split(r'(?<=[.?!])\s+', answer)
        answer = ' '.join(sentences[:2]).strip()

    # outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    # print(outputs[0]["generated_text"])

    return outputs

# # --- Main Execution ---
# if __name__ == "__main__":
print("Starting RAG pipeline setup...")

if not os.path.exists(PDF_DIRECTORY):
    print(f"PDF directory {PDF_DIRECTORY} does not exist. Creating it.")
    try:
        os.makedirs(PDF_DIRECTORY, exist_ok=True)
        print(f"Created directory {PDF_DIRECTORY}. Please add PDF files there to proceed fully.")
    except Exception as e:
        print(f"Could not create directory {PDF_DIRECTORY}: {e}")
        exit()

all_text_chunks, retrieval_text_chunks = load_and_process_pdfs(PDF_DIRECTORY, max_files=1)

faiss_index = None
embedding_model = None

if retrieval_text_chunks:
    faiss_index, embedding_model = create_faiss_index(retrieval_text_chunks, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH)
    if faiss_index and embedding_model:
        print("FAISS index and embedding model are ready.")
    else:
        print("Failed to create FAISS index or load embedding model.")
else:
    print("No text chunks processed. Skipping FAISS index creation.")
    print(f"Please ensure PDF files are available in {PDF_DIRECTORY}.")

llm_model, llm_tokenizer = load_llm_and_tokenizer(LLM_MODEL_NAME)
if not (llm_model and llm_tokenizer):
    print("Failed to load LLM or tokenizer. Exiting.")
    exit()
print("LLM and tokenizer are ready.")

print("RAG pipeline setup with prompt engineering is complete.")
print("You can now test the pipeline with questions.")

# --- Test Section ---
test_questions = [
    # "ما هي عاصمة فرنسا؟",
    # "اشرح مفهوم الثقب الأسود ببساطة."
    # "Eleventh and Twelfth Dynasties are known as"
    "Who is the old man"
]

for test_question in test_questions:
    print(f"\n--- Testing with question: {test_question} ---")
    retrieved_ctx = ""
    if faiss_index and embedding_model and retrieval_text_chunks: # Only retrieve if index is built
        retrieved_ctx = retrieve_context(test_question, faiss_index, embedding_model, retrieval_text_chunks, top_k=10)

    if not retrieved_ctx:
        print("Could not retrieve context for the question or no PDFs processed. Answering without external context.")
        retrieved_ctx = "No relevant context"

    # final_prompt = create_prompt_template(test_question, retrieved_ctx)
    final_prompt = create_prompt_template(test_question, retrieved_ctx, llm_tokenizer, max_tokens=2048)
    print(f"Final prompt for LLM:\n{final_prompt}")
    # answer = generate_answer(final_prompt, llm_model, llm_tokenizer)
    # answer = generate_answer(final_prompt, llm_model, llm_tokenizer)

        # print("Testinggggggggggggggggg" , outputs[0]["generated_text"])

    # print(answer)

    # llm_model, llm_tokenizer
    answer = generate_answer(final_prompt, llm_model, llm_tokenizer, short=False)
    #  generate_answer(prompt, llm_model, llm_tokenizer, short=False)
    print(answer[0]["generated_text"])

print("\n--- All tests completed. ---")

import torch
from transformers import pipeline
import warnings

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")




test_questions = [
    "What a cmos?",
    "What is the difference between NMOS and PMOS?"
]


for question in test_questions:
    # question = "Who is the old man"

    warnings.filterwarnings("ignore", category=UserWarning, module="accelerate.big_modeling")

    context = ""
    if faiss_index and embedding_model and retrieval_text_chunks: # Only retrieve if index is built
        context = retrieve_context(question, faiss_index, embedding_model, retrieval_text_chunks, top_k=7)

    if not context:
        print("Could not retrieve context for the question or no PDFs processed. Answering without external context.")
        context = "No relevant context"





    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
            {
                "role": "system",
                "content": "You are an expert assistant. Given the context and question below, give a concise, clear answer. If context is incomplete, use your general knowledge. Always answer in 1-2 sentences.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ]
    # warnings.filterwarnings("ignore", category=UserWarning, module="accelerate.big_modeling")

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # warnings.filterwarnings("ignore", category=UserWarning, module="accelerate.big_modeling")

    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    warnings.filterwarnings("ignore", category=UserWarning, module="accelerate.big_modeling")

    print(outputs[0]["generated_text"])
print("Finished execution")


