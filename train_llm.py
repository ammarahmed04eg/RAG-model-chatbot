import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import logging
from typing import List
from app import process_pdf, chunk_text, clean_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    
    def __len__(self):
        return len(self.encodings.input_ids)

def prepare_llm_training_data(text_chunks: List[str], tokenizer) -> Dataset:
    """Prepare data for LLM fine-tuning."""
    # Create training examples with context and questions
    examples = []
    for i in range(len(text_chunks) - 1):
        context = text_chunks[i]
        next_chunk = text_chunks[i + 1]
        
        # Create a simple question-answer pair
        prompt = f"Context: {context}\nQuestion: What comes next?\nAnswer: {next_chunk}"
        examples.append({"text": prompt})
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(examples)
    
    # Tokenize the dataset with smaller max_length
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256  # Reduced max length
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset

def train_llm_model(
    model_name: str,
    text_chunks: List[str],
    output_dir: str,
    batch_size: int = 1,  # Reduced batch size
    epochs: int = 3,
    learning_rate: float = 2e-5
):
    """Fine-tune the LLM on the PDF content."""
    logger.info("Preparing LLM training data...")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder="offload",
        offload_state_dict=True,  # Enable state dict offloading
    )
    
    # Prepare dataset
    dataset = prepare_llm_training_data(text_chunks, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16,  # Increased gradient accumulation
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        learning_rate=learning_rate,
        max_grad_norm=1.0,
        optim="adamw_torch",
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        warmup_steps=100,
        # Memory optimization settings
        dataloader_num_workers=0,  # Disable multiprocessing
        dataloader_pin_memory=False,  # Disable pinned memory
        torch_compile=False,  # Disable torch compile
        # Optimize memory usage
        max_steps=-1,  # Run for all epochs
        remove_unused_columns=True,
        label_names=["labels"],
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # Optimize for GPU memory alignment
        )
    )
    
    # Train the model
    logger.info("Starting LLM training...")
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LLM model saved to {output_dir}")

def main():
    # Configuration
    LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    PDF_DIRECTORY = "PDF files/"
    
    # Create output directories
    os.makedirs("trained_models/llm", exist_ok=True)
    os.makedirs("offload", exist_ok=True)  # Create offload directory
    
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
    
    # Train LLM
    logger.info("Starting LLM training...")
    train_llm_model(
        model_name=LLM_MODEL_NAME,
        text_chunks=all_chunks,
        output_dir="trained_models/llm"
    )
    
    logger.info("LLM training completed successfully!")

if __name__ == "__main__":
    main() 