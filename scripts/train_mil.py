"""
MIL (Multiple Instance Learning) 학습 스크립트 (Stage 2)
=========================================================
문서 라벨을 이용해 문단 분류기를 Fine-tuning

핵심 가정: "문서 내에 단 하나의 AI 문단만 있어도 그 문서는 AI 라벨이다"
- Human 문서: 모든 문단 → 0 (Human)으로 학습
- AI 문서: 문단들 중 Max Pooling → 라벨 1과 대조

실행: python scripts/train_mil.py
Colab: notebooks/colab/paragraph_maxpool_colab.ipynb
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================
# Configuration
# ============================================
MODEL_NAME = "klue/bert-base"
CHECKPOINT_PATH = "./outputs/paragraph_classifier/best_model"  # Stage 1 결과
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
EPOCHS = 2
OUTPUT_DIR = "./outputs/mil_model"

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"


# ============================================
# MIL Model Wrapper
# ============================================
class MILWrapper(nn.Module):
    """Multiple Instance Learning wrapper for BERT"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward_paragraph(self, input_ids, attention_mask):
        """Get AI probability for a single paragraph"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs[:, 1]  # AI probability
    
    def forward_document(self, paragraphs, tokenizer, device):
        """
        Get document score via Max Pooling over paragraphs
        
        Args:
            paragraphs: list of paragraph texts
            tokenizer: tokenizer
            device: torch device
            
        Returns:
            max_prob: max AI probability among paragraphs
            para_probs: list of AI probabilities for each paragraph
        """
        para_probs = []
        
        for para in paragraphs:
            inputs = tokenizer(
                para,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                prob = self.forward_paragraph(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
            para_probs.append(prob.item())
        
        max_prob = max(para_probs)
        return max_prob, para_probs


# ============================================
# Data Processing
# ============================================
def split_document_to_paragraphs(text, min_length=50):
    """Split document into paragraphs"""
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > min_length]
    if not paragraphs:
        paragraphs = [text]
    return paragraphs


def prepare_mil_batch(df, tokenizer, device, max_paragraphs=10):
    """Prepare MIL training batch"""
    for _, row in df.iterrows():
        text = row['full_text']
        label = row['generated']
        
        paragraphs = split_document_to_paragraphs(text)[:max_paragraphs]
        
        yield {
            'paragraphs': paragraphs,
            'label': label,
            'title': row.get('title', '')
        }


# ============================================
# Training Loop
# ============================================
def train_epoch(model, optimizer, data_iter, tokenizer, device, is_training=True):
    """Train or evaluate one epoch"""
    if is_training:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    predictions = []
    labels = []
    
    criterion = nn.BCELoss()
    
    for batch in tqdm(data_iter, desc="Training" if is_training else "Evaluating"):
        paragraphs = batch['paragraphs']
        label = batch['label']
        
        # Forward pass for each paragraph
        para_probs = []
        for para in paragraphs:
            inputs = tokenizer(
                para,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(device)
            
            if is_training:
                outputs = model.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                probs = torch.softmax(outputs.logits, dim=-1)
                para_probs.append(probs[0, 1])
            else:
                with torch.no_grad():
                    outputs = model.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                    probs = torch.softmax(outputs.logits, dim=-1)
                    para_probs.append(probs[0, 1].item())
        
        # Max Pooling
        if is_training:
            max_prob = torch.max(torch.stack(para_probs))
            target = torch.tensor([float(label)]).to(device)
            
            loss = criterion(max_prob.unsqueeze(0), target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions.append(max_prob.item())
        else:
            max_prob = max(para_probs)
            predictions.append(max_prob)
        
        labels.append(label)
    
    avg_loss = total_loss / len(labels) if is_training else 0
    auc = roc_auc_score(labels, predictions)
    
    return avg_loss, auc, predictions, labels


# ============================================
# Main
# ============================================
def main():
    print("=" * 60)
    print("🚀 Stage 2: MIL Fine-tuning")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load Stage 1 model
    print(f"\n🔧 Loading Stage 1 model from: {CHECKPOINT_PATH}")
    if not Path(CHECKPOINT_PATH).exists():
        print(f"⚠️  Stage 1 model not found. Run train_paragraph_classifier.py first.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
    model = MILWrapper(base_model).to(device)
    
    # Load data
    print(f"\n📂 Loading data from: {DATA_PATH}")
    if not DATA_PATH.exists():
        print(f"⚠️  Data file not found: {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    print(f"   Total documents: {len(df):,}")
    print(f"   Class distribution: {df['generated'].value_counts().to_dict()}")
    
    # Sample for faster training (optional)
    df_sample = df.sample(n=min(5000, len(df)), random_state=42)
    train_df, val_df = train_test_split(df_sample, test_size=0.2, stratify=df_sample['generated'], random_state=42)
    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    print("\n🏋️ Starting MIL training...")
    best_auc = 0
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        # Train
        train_iter = list(prepare_mil_batch(train_df, tokenizer, device))
        train_loss, train_auc, _, _ = train_epoch(
            model, optimizer, train_iter, tokenizer, device, is_training=True
        )
        print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        
        # Validate
        val_iter = list(prepare_mil_batch(val_df, tokenizer, device))
        _, val_auc, _, _ = train_epoch(
            model, optimizer, val_iter, tokenizer, device, is_training=False
        )
        print(f"Val - AUC: {val_auc:.4f}")
        
        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = Path(OUTPUT_DIR) / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)
            model.model.save_pretrained(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            print(f"✅ New best model saved (AUC: {val_auc:.4f})")
    
    print(f"\n🎉 Training complete! Best AUC: {best_auc:.4f}")
    print(f"   Model saved to: {OUTPUT_DIR}/best_model")


if __name__ == "__main__":
    main()
