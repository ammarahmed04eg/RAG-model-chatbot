# PDF Question Answering System

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content using a RAG (Retrieval-Augmented Generation) pipeline.

## Features

- PDF document upload and processing
- Text extraction and chunking
- Semantic search using FAISS
- Question answering using TinyLlama model
- Clean and intuitive user interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dejigner/RAG_Depi.git
cd RAG_Depi
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app locally:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload a PDF document and ask questions about its content

## Deployment

This app is deployed on Streamlit Community Cloud. You can access it at: [Your Streamlit App URL]

To deploy your own version:

1. Fork this repository
2. Sign up for a Streamlit Community Cloud account at https://streamlit.io/cloud
3. Connect your GitHub account
4. Click "New app" and select this repository
5. Select `app.py` as your main file
6. Click "Deploy"

## Requirements

- Python 3.8+
- PyPDF2
- Streamlit
- PyTorch
- Transformers
- Sentence Transformers
- FAISS
- Other dependencies listed in requirements.txt

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Project dependencies
- `packages.txt`: System dependencies
- `README.md`: Project documentation