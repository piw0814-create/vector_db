import asyncio
import csv
import json
import os

import faiss
import numpy as np

from openai import OpenAI, AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
)
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 설정
# ============================================================

TOP_K = 5

# 현재 API에서 지원되는 평가용 모델
MODEL = os.getenv(
    "RAGAS_MODEL",
    "gpt-5.4-mini"
)

EMBEDDING_MODEL = "text-embedding-3-small"


# 기존 실습 검색어와 동일하게 평가
USER_INPUT = "HBM이란 무엇인가?"

# Context Precision 평가에 사용할 기준 답안
REFERENCE = """
HBM(High Bandwidth Memory)은 여러 개의 DRAM 다이를 적층하고
TSV 등의 기술로 연결하여 높은 메모리 대역폭을 제공하는 메모리이다.
AI GPU와 NPU처럼 대규모 데이터를 빠르게 처리해야 하는 시스템에서
메모리 병목을 줄이고 연산 장치에 데이터를 빠르게 공급하기 때문에
AI 반도체의 핵심 메모리 기술로 중요하다.
""".strip()


# ============================================================
# 데이터 로드
# ============================================================

with open(
    "artifacts/chunks.jsonl",
    "r",
    encoding="utf-8"
) as f:
    chunks = [
        json.loads(line)
        for line in f
    ]


# ============================================================
# 1. Dense Top-10
# ============================================================

index = faiss.read_index(
    "artifacts/faiss_hnsw.index"
)

q_vec = np.load(
    "artifacts/query_hbm.npy"
).astype("float32")

_, dense_ids = index.search(
    q_vec,
    TOP_K
)

dense_ids = [
    int(x)
    for x in dense_ids[0]
]


# ============================================================
# 2. Hybrid RRF Top-10
# ============================================================

with open(
    "artifacts/rrf_candidates.json",
    "r",
    encoding="utf-8"
) as f:
    rrf_data = json.load(f)

hybrid_ids = [
    int(x)
    for x in rrf_data["candidate_ids"][:TOP_K]
]


# ============================================================
# 3. Reranker Top-10
# ============================================================

with open(
    "artifacts/reranked_results.json",
    "r",
    encoding="utf-8"
) as f:
    reranker_data = json.load(f)

reranker_ids = [
    int(x)
    for x in reranker_data["reranked_ids"][:TOP_K]
]


STAGES = {
    "Dense": dense_ids,
    "Hybrid_RRF": hybrid_ids,
    "Reranker": reranker_ids,
}


# ============================================================
# OpenAI / RAGAS 설정
# ============================================================

sync_client = OpenAI()
async_client = AsyncOpenAI()

evaluator_llm = llm_factory(
    MODEL,
    client=async_client,
    max_tokens=4096
)

evaluator_embeddings = OpenAIEmbeddings(
    client=async_client,
    model=EMBEDDING_MODEL
)


faithfulness_metric = Faithfulness(
    llm=evaluator_llm
)

answer_relevancy_metric = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

context_precision_metric = ContextPrecision(
    llm=evaluator_llm
)


# ============================================================
# 검색 문서를 이용해 실제 RAG 답변 생성
# ============================================================

def generate_answer(contexts):

    context_text = "\n\n---\n\n".join(
        contexts
    )

    prompt = f"""
아래 검색된 문서만 근거로 질문에 답하세요.

규칙:
- 검색 문서에 없는 내용을 임의로 추가하지 마세요.
- 질문이 짧은 키워드형이면 개념과 중요성을 중심으로 설명하세요.
- 한국어로 간결하게 답하세요.

[질문]
{USER_INPUT}

[검색 문서]
{context_text}
""".strip()

    response = sync_client.responses.create(
        model=MODEL,
        input=prompt
    )

    return response.output_text.strip()


# ============================================================
# RAGAS 평가
# ============================================================

async def evaluate_stage(
    stage_name,
    doc_ids
):

    contexts = [
        chunks[i]["text"]
        for i in doc_ids
    ]

    print(
        f"\n[{stage_name}] "
        f"답변 생성 중..."
    )

    answer = generate_answer(
        contexts
    )

    print("\n생성 답변:")
    print(answer)

    print("\nRAGAS 평가 중...")


    faithfulness = (
        await faithfulness_metric.ascore(
            user_input=USER_INPUT,
            response=answer,
            retrieved_contexts=contexts
        )
    )


    answer_relevancy = (
        await answer_relevancy_metric.ascore(
            user_input=USER_INPUT,
            response=answer
        )
    )


    context_precision = (
        await context_precision_metric.ascore(
            user_input=USER_INPUT,
            reference=REFERENCE,
            retrieved_contexts=contexts
        )
    )


    result = {
        "stage": stage_name,
        "faithfulness": float(
            faithfulness.value
        ),
        "answer_relevancy": float(
            answer_relevancy.value
        ),
        "context_precision": float(
            context_precision.value
        ),
        "answer": answer,
        "doc_ids": doc_ids
    }

    return result


# ============================================================
# Main
# ============================================================

async def main():

    print(
        f"Evaluator Model: {MODEL}"
    )

    print(
        f"Embedding Model: "
        f"{EMBEDDING_MODEL}"
    )

    results = []

    for stage_name, doc_ids in STAGES.items():

        result = await evaluate_stage(
            stage_name,
            doc_ids
        )

        results.append(
            result
        )


    # --------------------------------
    # 결과 출력
    # --------------------------------

    print(
        "\n"
        "============================================================"
    )

    print(
        "RAGAS RESULT"
    )

    print(
        "============================================================"
    )

    print(
        f"{'Stage':<15}"
        f"{'Faithfulness':>15}"
        f"{'Answer Rel.':>15}"
        f"{'Context Prec.':>17}"
    )

    for r in results:

        print(
            f"{r['stage']:<15}"
            f"{r['faithfulness']:>15.4f}"
            f"{r['answer_relevancy']:>15.4f}"
            f"{r['context_precision']:>17.4f}"
        )


    # --------------------------------
    # CSV 저장
    # --------------------------------

    with open(
        "results/ragas_results.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "stage",
            "faithfulness",
            "answer_relevancy",
            "context_precision"
        ])

        for r in results:

            writer.writerow([
                r["stage"],
                r["faithfulness"],
                r["answer_relevancy"],
                r["context_precision"]
            ])


    # --------------------------------
    # 상세 결과 JSON 저장
    # --------------------------------

    with open(
        "results/ragas_details.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        "\n저장 완료:"
    )

    print(
        "- results/ragas_results.csv"
    )

    print(
        "- results/ragas_details.json"
    )


if __name__ == "__main__":
    asyncio.run(main())
