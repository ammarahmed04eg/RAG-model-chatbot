import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses
)
import logging
from typing import List
from app import process_pdf, chunk_text, clean_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_embedding_training_data(text_chunks: List[str]) -> List[InputExample]:
    """Prepare data for embedding model training using contrastive learning."""
    examples = []
    for i in range(len(text_chunks)):
        # Create positive pairs (adjacent chunks)
        if i < len(text_chunks) - 1:
            examples.append(InputExample(
                texts=[text_chunks[i], text_chunks[i + 1]],
                label=1.0
            ))
        # Create negative pairs (random chunks)
        if i < len(text_chunks) - 2:
            examples.append(InputExample(
                texts=[text_chunks[i], text_chunks[i + 2]],
                label=0.0
            ))
    return examples

def train_embedding_model(
    model_name: str,
    text_chunks: List[str],
    output_dir: str,
    batch_size: int = 32,
    epochs: int = 3
):
    """Fine-tune the embedding model on the PDF content."""
    logger.info("Preparing embedding model training data...")
    train_examples = prepare_embedding_training_data(text_chunks)
    
    # Initialize model
    model = SentenceTransformer(model_name)
    
    # Prepare training data
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Training loss
    train_loss = losses.ContrastiveLoss(model)
    
    # Train the model
    logger.info("Starting embedding model training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    # Save the model
    model.save(output_dir)
    logger.info(f"Embedding model saved to {output_dir}")

def main():
    # Configuration
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    PDF_DIRECTORY = "PDF files/"
    
    # Create output directory
    os.makedirs("trained_models/embedding", exist_ok=True)
    
    # Load and process PDFs
    all_chunks = []
    for pdf_file in os.listdir(PDF_DIRECTORY):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
            try:
                # Read the PDF file content
                with open(pdf_path, 'rb') as f:
                    pdf_content = f.read()
                    chunks, _ = process_pdf(pdf_content)
                    if chunks:
                        all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                continue
    
    if not all_chunks:
        logger.error("No text chunks found in PDFs. Please ensure PDFs are properly loaded.")
        return
    
    # Train embedding model
    logger.info("Starting embedding model training...")
    train_embedding_model(
        model_name=EMBEDDING_MODEL_NAME,
        text_chunks=all_chunks,
        output_dir="trained_models/embedding"
    )
    
    logger.info("Embedding model training completed successfully!")

if __name__ == "__main__":
    main() 