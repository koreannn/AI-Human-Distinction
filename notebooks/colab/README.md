# 📓 Colab Notebooks

> AI Text Detector 프로젝트의 학습 및 분석 노트북 모음

## 🎯 추천 학습 순서

### Stage 1: 기초 분석
| 순서 | 노트북 | 설명 |
|------|--------|------|
| 1️⃣ | `bert_baseline.ipynb` | BERT 기본 구조 및 분류 실험 |
| 2️⃣ | `token_analysis.ipynb` | 토큰 분석 및 어텐션 시각화 |

### Stage 2: 핵심 학습 파이프라인
| 순서 | 노트북 | 설명 |
|------|--------|------|
| 3️⃣ | `bert_paragraph_classifier_colab.ipynb` | **문단 분류기 학습 (Stage 1)** |
| 4️⃣ | `paragraph_maxpool_colab.ipynb` | **MIL Fine-tuning (Stage 2)** |

### Stage 3: 설명 가능성 분석
| 순서 | 노트북 | 설명 |
|------|--------|------|
| 5️⃣ | `explainability_analysis_colab.ipynb` | LIME, Attention 분석 |
| 6️⃣ | `style_trajectory_analysis_colab.ipynb` | 문체 변화 분석 |

---

## 📂 실험 노트북 (참고용)

이전 실험에서 가져온 노트북들:

| 노트북 | 내용 |
|--------|------|
| `original_data_pipeline.ipynb` | 데이터 전처리 파이프라인 |
| `bert_paragraph_classifier.ipynb` | 초기 문단 분류기 실험 |
| `style_trajectory_analysis.ipynb` | 문체 분석 로컬 버전 |

---

## 🚀 Colab에서 실행

1. Google Drive에 업로드
2. Colab에서 열기
3. 런타임 → GPU로 변경
4. 셀 순서대로 실행

> ⚠️ 데이터 파일(`train.csv`, `test.csv`)은 Drive에 별도 업로드 필요
