# 📜 Scripts

> 핵심 학습 및 추론 스크립트

## 🎯 학습 파이프라인

| 스크립트 | 용도 | 실행 방법 |
|----------|------|-----------|
| `train_paragraph_classifier.py` | Stage 1: BERT 문단 분류기 학습 | `python scripts/train_paragraph_classifier.py` |
| `train_mil.py` | Stage 2: MIL Fine-tuning | `python scripts/train_mil.py` |

## 📊 ML 베이스라인

| 스크립트 | 용도 |
|----------|------|
| `ml_submission_pipeline.py` | 메타 피처 기반 ML 모델 (HistGradientBoosting) |

## 🚀 실행 순서

```bash
cd /path/to/AI-Human-Distinction

# 1. Stage 1 - 문단 분류기 학습 (Colab 권장)
python scripts/train_paragraph_classifier.py

# 2. Stage 2 - MIL Fine-tuning
python scripts/train_mil.py
```

> ⚠️ GPU 학습은 Google Colab 사용을 권장합니다.
