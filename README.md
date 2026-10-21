# Vector DB & RAG 검색 성능 비교 실습

## 1. 프로젝트 개요

본 실습에서는 AI·반도체 관련 PDF 문서를 대상으로 Vector DB 기반 검색 환경을 구축하고, 검색 방식을 단계적으로 확장하면서 검색 결과와 RAG 품질이 어떻게 달라지는지 비교하였다.

실습은 크게 두 단계로 구성하였다.

- **실습 1**: PDF Loading → Chunking → BGE-M3 Embedding → FAISS → Qdrant → Metadata Filter
- **실습 2**: Dense Search → BM25 → RRF 기반 Hybrid Search → Reranker → RAGAS 평가

최종 비교 질의는 다음과 같이 통일하였다.

```text
HBM이란 무엇인가?
```

---

## 2. 프로젝트 구조

```text
vector_db/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── 01_load_chunk.py
│   ├── 02_embedding.py
│   ├── 03_query_embedding.py
│   ├── 04_faiss_search.py
│   ├── 05_qdrant_upsert.py
│   ├── 06_qdrant_search.py
│   ├── 07_query_hbm_embedding.py
│   ├── 08_dense_bm25_compare.py
│   ├── 09_rrf.py
│   ├── 10_reranker.py
│   ├── 11_compare_three_stages.py
│   └── 12_ragas_eval.py
├── results/
│   ├── practice1_faiss.txt
│   ├── practice1_qdrant_filter.txt
│   ├── practice2_dense_vs_bm25_final.txt
│   ├── practice2_rrf_final.txt
│   ├── practice2_reranker_final.txt
│   ├── practice2_three_stage_final.txt
│   ├── practice2_ragas_final.txt
│   ├── three_stage_comparison.csv
│   ├── ragas_results.csv
│   └── ragas_details.json
├── data/
└── artifacts/
```

> `.env`, `.venv`, `qdrant_storage`, 캐시 파일 및 모델 캐시는 제출 대상에서 제외한다.

---

## 3. 실습 환경 및 데이터

총 16개의 AI·반도체 관련 PDF 문서를 대상으로 실습을 진행하였다.

| 항목 | 결과 |
|---|---:|
| PDF 문서 수 | 16 |
| 추출 페이지 수 | 843 |
| 생성 Chunk 수 | 2,399 |
| Chunk Size | 500 |
| Chunk Overlap | 50 |
| Embedding Model | `BAAI/bge-m3` |
| Embedding Dimension | 1,024 |
| Embedding Shape | `(2399, 1024)` |
| 전체 임베딩 시간 | 약 26.82초 |

문서에서 텍스트를 추출한 뒤 `RecursiveCharacterTextSplitter`를 이용해 Chunk를 생성하고, BGE-M3로 1,024차원 임베딩을 생성하였다.

---

# 4. 실습 1 — Vector DB 구축 및 검색

## 4.1 PDF Loading & Chunking

PDF 문서를 페이지 단위로 읽고 다음 Metadata를 함께 저장하였다.

- `source`: 원본 PDF 파일명
- `page`: PDF 페이지 번호
- `text`: Chunk 본문

이후 `chunk_size=500`, `chunk_overlap=50`으로 총 2,399개의 Chunk를 생성하였다.

---

## 4.2 BGE-M3 Embedding

생성된 Chunk 전체에 대해 `BAAI/bge-m3` 모델로 임베딩을 생성하였다.

```text
Embedding Shape: (2399, 1024)
dtype: float32
```

생성된 임베딩은 이후 FAISS와 Qdrant에서 동일하게 활용하였다.

---

## 4.3 FAISS Search

FAISS HNSW 인덱스를 구축하고 다음 질의를 검색하였다.

```text
HBM이란 무엇인가?
```

검색 결과 HBM의 구조, TSV 기반 적층 방식, 메모리 병목, AI 반도체에서의 활용 등을 설명하는 문서가 상위에 검색되었다.

특히 다음과 같은 내용이 상위 결과에 포함되었다.

- HBM의 DRAM 적층 및 TSV 구조
- HBM의 고대역폭 특성
- AI 연산에서의 메모리 병목
- AI GPU/NPU에서 HBM의 중요성

상세 실행 결과는 다음 파일에 저장하였다.

- [`results/practice1_faiss.txt`](results/practice1_faiss.txt)

---

## 4.4 Qdrant Search & Metadata Filter

동일한 임베딩을 Qdrant의 `rag_docs` Collection에 저장하였다.

각 Point에는 Vector와 함께 다음 Payload를 저장하였다.

```text
text
source
page
```

일반 Vector Search 외에도 다음 Metadata Filter를 적용하였다.

```text
page >= 5
```

필터 적용 후 검색 결과가 모두 조건을 만족하는 것을 확인하였다.

FAISS와 Qdrant 모두 의미 기반 검색이 가능했지만, Qdrant에서는 문서 출처, 페이지 등 Metadata 조건을 함께 활용할 수 있었다.

실제 RAG 서비스에서는 작성일, 문서 유형, 부서, 접근 권한 등으로 검색 범위를 제한해야 할 수 있으므로 이러한 Metadata 관리가 중요하다고 판단하였다.

상세 실행 결과:

- [`results/practice1_qdrant_filter.txt`](results/practice1_qdrant_filter.txt)

---

# 5. 실습 2 — Dense → Hybrid(RRF) → Reranker

실습 2에서는 동일한 질문에 대해 검색 방식을 단계적으로 추가하면서 Top-K 결과의 변화를 비교하였다.

```text
HBM이란 무엇인가?
```

---

## 5.1 Dense Search

BGE-M3 임베딩 기반 Dense Search에서는 HBM과 의미적으로 관련된 문서가 안정적으로 검색되었다.

Dense Top-5에는 다음 내용이 포함되었다.

- HBM 구조 및 TSV 적층 방식
- HBM과 메모리 병목
- HBM의 대용량·고대역폭 특성
- AI 가속기에서의 HBM 중요성
- HBM 산업 및 시장 관련 내용

Dense Search는 단순히 `"HBM"` 문자열이 존재하는지를 확인하기보다 질의와 문서의 의미적 유사성을 기반으로 검색하였다.

---

## 5.2 BM25 Search

Dense Search와 별도로 BM25 기반 Sparse Search를 구성하였다.

BM25는 정확한 키워드가 포함된 문서를 직접 찾는 데 강점이 있어 Dense Search를 보완할 수 있다.

다만 이번 실험에서는 짧은 질의와 한국어 토큰화 특성의 영향으로 HBM과 직접 관련성이 낮은 문서도 일부 높은 순위로 검색되었다.

상세 비교 결과:

- [`results/practice2_dense_vs_bm25_final.txt`](results/practice2_dense_vs_bm25_final.txt)

---

## 5.3 Hybrid Search — Dense + BM25 + RRF

Dense와 BM25의 검색 순위를 RRF(Reciprocal Rank Fusion)로 결합하였다.

Hybrid Top-5 결과는 다음과 같았다.

| 순위 | 검색 결과 | 판단 |
|---:|---|---|
| 1 | HBM 구조 및 시장 관련 문서 | 관련 |
| 2 | 참고문헌 페이지 | 낮은 관련성 |
| 3 | UAE AI 데이터센터 문서 | 낮은 관련성 |
| 4 | HBM·PIM 및 메모리 병목 문서 | 관련 |
| 5 | UAE AI 데이터센터 문서 | 낮은 관련성 |

Hybrid Search를 통해 Dense와 Sparse 검색 결과를 함께 활용할 수 있었지만, 이번 질의에서는 BM25 쪽 Noise도 함께 유입되었다.

이를 통해 **Hybrid Search를 적용하는 것만으로 검색 결과가 항상 좋아지는 것은 아니며, Sparse 검색의 토큰화 방식과 후보 품질도 함께 확인해야 한다**는 점을 확인하였다.

상세 실행 결과:

- [`results/practice2_rrf_final.txt`](results/practice2_rrf_final.txt)

---

## 5.4 Reranker

Hybrid Search에서 생성한 후보 문서 10개를 `BAAI/bge-reranker-v2-m3`로 다시 평가하였다.

최종 Top-5는 다음과 같았다.

| 순위 | 주요 내용 | Reranker Score |
|---:|---|---:|
| 1 | HBM과 메모리 병목 | 0.9258 |
| 2 | HBM 구조 및 TSV | 0.8975 |
| 3 | HBM 정의 및 고대역폭 특성 | 0.8805 |
| 4 | HBM과 AI 가속기 경쟁력 | 0.8487 |
| 5 | HBM 시장 및 기술 경쟁 | 0.6525 |

Hybrid 단계에서 상위에 포함되었던 참고문헌이나 UAE 데이터센터 관련 문서는 Reranker Score가 거의 0으로 평가되어 하위로 이동하였다.

즉 이번 실험에서는 다음과 같은 흐름을 확인하였다.

```text
Dense / BM25
      ↓
RRF로 후보 결합
      ↓
일부 Noise 포함
      ↓
Reranker로 질문-문서 재평가
      ↓
HBM 관련 문서가 상위에 재배치
```

상세 실행 결과:

- [`results/practice2_reranker_final.txt`](results/practice2_reranker_final.txt)
- [`results/practice2_three_stage_final.txt`](results/practice2_three_stage_final.txt)
- [`results/three_stage_comparison.csv`](results/three_stage_comparison.csv)

---

# 6. Dense → Hybrid → Reranker 비교

최종 Top-5 결과를 비교하면 다음과 같다.

| Rank | Dense | Hybrid(RRF) | Reranker |
|---:|---|---|---|
| 1 | HBM 구조/시장 | HBM 구조/시장 | HBM·메모리 병목 |
| 2 | HBM·메모리 병목 | 참고문헌 | HBM 구조/TSV |
| 3 | HBM 정의/특징 | UAE AI 데이터센터 | HBM 정의/특징 |
| 4 | HBM 시장/기술 | HBM·메모리 병목 | HBM·AI 가속기 |
| 5 | HBM·AI 가속기 | UAE AI 데이터센터 | HBM 시장/기술 |

Dense Search 자체도 비교적 높은 품질의 결과를 보여주었다.

Hybrid Search에서는 검색 후보가 다양해졌지만 관련성이 낮은 문서도 상위로 올라왔다. 이후 Reranker가 질의와 후보 문서를 직접 비교하면서 HBM과 직접 관련된 문서가 다시 상위에 배치되었다.

따라서 이번 실험에서는 **Hybrid가 후보를 넓히고, Reranker가 최종 정밀도를 보완하는 구조**를 확인할 수 있었다.

---

# 7. RAGAS 평가

검색 방식별 결과를 기반으로 생성 답변을 만들고 RAGAS를 통해 정량 평가하였다.

사용한 지표는 다음과 같다.

- **Faithfulness**: 생성 답변이 검색 문서의 내용에 근거하는가
- **Answer Relevancy**: 답변이 질문의 의도와 관련되어 있는가
- **Context Precision**: 검색된 Context 중 실제로 유용한 내용이 적절하게 상위에 위치하는가

## 7.1 평가 결과

| 검색 방식 | Faithfulness | Answer Relevancy | Context Precision |
|---|---:|---:|---:|
| Dense | **1.0000** | 0.6751 | **1.0000** |
| Hybrid (RRF) | **1.0000** | **0.7441** | 0.7500 |
| Reranker | **1.0000** | 0.6556 | **1.0000** |

상세 결과:

- [`results/practice2_ragas_final.txt`](results/practice2_ragas_final.txt)
- [`results/ragas_results.csv`](results/ragas_results.csv)
- [`results/ragas_details.json`](results/ragas_details.json)

---

## 7.2 결과 분석

### Faithfulness

세 방식 모두 `1.0000`으로 나타났다.

따라서 Dense, Hybrid, Reranker 기반으로 생성한 답변 모두 검색 문서의 내용을 벗어나지 않고 근거에 충실하게 생성되었다고 볼 수 있다.

### Answer Relevancy

Hybrid Search가 `0.7441`로 가장 높은 점수를 기록하였다.

다만 이번 평가는 하나의 질의를 기준으로 수행했기 때문에 이 결과만으로 Hybrid 방식이 전체적으로 가장 우수하다고 판단하기는 어렵다.

### Context Precision

가장 뚜렷한 변화는 Context Precision에서 나타났다.

```text
Dense       1.00
Hybrid      0.75
Reranker    1.00
```

Hybrid 단계에서는 참고문헌 및 UAE 데이터센터 문서와 같이 질문과 직접 관련성이 낮은 Context가 상위에 포함되면서 Context Precision이 감소하였다.

Reranker 적용 후에는 이러한 문서가 하위로 이동하고 HBM 정의, TSV 구조, 메모리 병목과 직접 관련된 문서가 상위로 재배치되면서 Context Precision이 다시 `1.00`으로 회복되었다.

즉 **RAGAS의 Context Precision 변화와 실제 검색 결과의 변화가 일치하는 것을 확인할 수 있었다.**

---

# 8. RAG 설정 시 주요 고려사항

이번 실습을 통해 RAG 성능은 특정 Vector DB나 Embedding Model 하나로 결정되는 것이 아니라, **Chunking → Retrieval → Filtering → Reranking → Evaluation의 전체 파이프라인 설정에 영향을 받는다**고 판단하였다.

## 8.1 Chunking

Chunk가 너무 작으면 검색 위치는 정밀해질 수 있지만 필요한 문맥이 여러 Chunk로 나뉠 수 있다.

반대로 너무 큰 Chunk는 하나의 Chunk 안에 여러 주제가 섞여 검색 정확도를 낮출 수 있다.

따라서 문서 구조와 실제 사용자 질의의 특성을 기준으로 Chunk Size와 Overlap을 조정해야 하며, 긴 문맥이 필요한 경우 Parent-Child 방식도 고려할 수 있다.

## 8.2 Dense와 Sparse 검색

Dense Search는 표현이 달라도 의미적으로 유사한 문서를 찾는 데 강점이 있다.

반면 BM25는 제품명, 코드, 버전, 고유명사와 같이 정확한 문자열이 중요한 검색을 보완할 수 있다.

따라서 두 방식을 함께 사용하면 검색 범위를 확장할 수 있지만, 이번 실험처럼 Sparse Search의 Noise가 함께 유입될 가능성도 존재한다.

특히 한국어에서는 조사와 형태소에 따라 Tokenization 결과가 달라질 수 있으므로 BM25의 Tokenizer 설정 또한 검색 품질에 영향을 줄 수 있다고 생각한다.

## 8.3 Metadata Filter

실제 서비스에서는 전체 문서 중 의미가 비슷한 문서를 검색하는 것만으로 충분하지 않을 수 있다.

예를 들어 다음과 같은 조건이 필요할 수 있다.

- 최신 문서만 검색
- 특정 부서 문서만 검색
- 특정 문서 유형만 검색
- 사용자에게 접근 권한이 있는 문서만 검색

따라서 Vector Search와 Metadata Filter를 함께 설계하는 것이 중요하다.

## 8.4 Reranker

Reranker는 질문과 검색 후보를 직접 비교하기 때문에 최종 검색 정밀도를 높이는 데 효과적이었다.

하지만 추가적인 모델 추론이 필요하므로 검색 후보 수가 많을수록 응답 시간과 연산 비용이 증가한다.

따라서 모든 검색 결과를 Reranker에 넣기보다 1차 검색에서 Top-N 후보를 선정한 후 제한된 후보만 재평가하는 방식이 적절하다고 생각한다.

## 8.5 Evaluation

한두 개의 예시 질문만으로 RAG 구조의 우열을 판단하기는 어렵다.

이번 실험에서도 Dense → Hybrid → Reranker 순서로 기술을 추가했지만 모든 RAGAS 지표가 순차적으로 증가하지는 않았다.

따라서 실제 서비스에서는 다음과 같이 난이도와 성격이 다른 평가 질의를 구성할 필요가 있다.

- 단일 문서에서 답할 수 있는 질문
- 정확한 키워드가 중요한 질문
- 여러 문서를 종합해야 하는 질문
- Metadata Filter가 필요한 질문

또한 RAGAS와 같은 정량 평가뿐 아니라 실제 검색 결과를 직접 확인하는 정성 평가도 함께 수행해야 한다.

---

# 9. 결론

이번 실습에서는 16개의 PDF 문서를 2,399개의 Chunk로 구성하고 BGE-M3 기반 Vector Search 환경을 구축하였다.

FAISS와 Qdrant를 이용해 Vector Search를 수행하고, Qdrant에서는 Metadata Filter를 적용하여 Vector DB의 활용 방식을 확인하였다.

이후 Dense Search에 BM25와 RRF를 결합하고 BGE Reranker를 추가하여 검색 결과의 변화를 비교하였다.

Hybrid Search에서는 일부 Noise가 유입되었지만 Reranker 적용 후 HBM과 직접 관련된 문서가 다시 상위에 배치되었으며, RAGAS Context Precision 역시 `0.75 → 1.00`으로 회복되었다.

이를 통해 **검색 기술을 추가하는 것 자체가 성능 향상을 보장하는 것은 아니며, 실제 검색 결과와 평가 지표를 기반으로 각 단계의 효과를 검증해야 한다**는 점을 확인하였다.

결국 좋은 RAG 시스템은 가장 복잡한 구조를 적용한 시스템이 아니라,

> **실제 데이터와 질의 특성에 맞춰 검색 품질, 응답 속도, 비용, 운영 복잡도의 균형을 검증하고 적절한 구조를 선택한 시스템**

이라고 생각한다.

---

# 10. 실행 순서

가상환경 활성화 후 필요한 패키지를 설치한다.

```bash
pip install -r requirements.txt
```

환경 변수는 `.env.example`을 참고하여 `.env`에 설정한다.

```text
OPENAI_API_KEY=your_openai_api_key_here
RAGAS_MODEL=gpt-6-luna
```

주요 실습 실행 순서는 다음과 같다.

```bash
python src/01_load_chunk.py
python src/02_embedding.py
python src/03_query_embedding.py
python src/04_faiss_search.py
python src/05_qdrant_upsert.py
python src/06_qdrant_search.py

python src/07_query_hbm_embedding.py
python src/08_dense_bm25_compare.py
python src/09_rrf.py
python src/10_reranker.py
python src/11_compare_three_stages.py

RAGAS_MODEL=gpt-6-luna python src/12_ragas_eval.py
```

> Qdrant 실행 환경이 필요한 경우 Qdrant 서버를 먼저 실행한 뒤 `05_qdrant_upsert.py`, `06_qdrant_search.py`를 실행한다.
