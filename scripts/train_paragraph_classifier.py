"""
BERT 문단 분류기 학습 스크립트 (Stage 1)
==========================================
GPT-OSS 합성 데이터로 문단 단위 AI/Human 분류기 학습

실행: python scripts/train_paragraph_classifier.py
Colab: notebooks/colab/bert_paragraph_classifier_colab.ipynb
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset

# ============================================
# Configuration
# ============================================
MODEL_NAME = "klue/bert-base"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
OUTPUT_DIR = "./outputs/paragraph_classifier"

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "gpt_oss_synthetic_pairs_cleaned.csv"

# ============================================
# Load Data
# ============================================
def load_paragraph_data(data_path: str):
    """Load GPT-OSS synthetic paragraph pairs"""
    df = pd.read_csv(data_path)
    
    # Create paragraph-level dataset
    # Human paragraphs: label 0
    # AI paragraphs: label 1
    paragraphs = []
    labels = []
    
    for _, row in df.iterrows():
        if 'human_text' in df.columns and 'ai_text' in df.columns:
            paragraphs.append(row['human_text'])
            labels.append(0)
            paragraphs.append(row['ai_text'])
            labels.append(1)
        else:
            # Fallback: assume 'text' and 'label' columns
            paragraphs.append(row['text'])
            labels.append(int(row['label']))
    
    return pd.DataFrame({'text': paragraphs, 'label': labels})


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding=False
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions),
        'auc': roc_auc_score(labels, probs)
    }


# ============================================
# Main Training
# ============================================
def main():
    print("=" * 60)
    print("🚀 Stage 1: BERT 문단 분류기 학습")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data from: {DATA_PATH}")
    if not DATA_PATH.exists():
        print(f"⚠️  Data file not found. Please prepare: {DATA_PATH}")
        print("   Expected format: CSV with 'human_text' and 'ai_text' columns")
        return
    
    df = load_paragraph_data(str(DATA_PATH))
    print(f"   Total samples: {len(df):,}")
    print(f"   Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Train/Val split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,}")
    
    # Tokenizer
    print(f"\n🔧 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=['text']
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=['text']
    )
    
    # Model
    print(f"\n🧠 Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        logging_steps=50
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\n🏋️ Starting training...")
    trainer.train()
    
    # Evaluate
    print("\n📊 Final evaluation:")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"   {k}: {v:.4f}")
    
    # Save
    save_path = Path(OUTPUT_DIR) / "best_model"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"\n✅ Model saved to: {save_path}")


if __name__ == "__main__":
    main()
